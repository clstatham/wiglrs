use bevy::prelude::*;
use bevy_tasks::AsyncComputeTaskPool;

use std::{collections::BTreeMap, sync::atomic::AtomicUsize};
use tokio::sync::{
    mpsc::{self, Receiver, Sender},
    oneshot,
};

use self::{
    replay_buffer::PpoBuffer,
    thinkers::{ppo::PpoThinker, Status, Thinker},
};
use crate::{envs::Env, FrameStack, TbWriter, Timestamp};

pub mod replay_buffer;
pub mod thinkers;

pub struct Brain<E: Env, T: Thinker<E>> {
    pub name: String,
    pub timestamp: Timestamp,
    pub last_action: E::Action,
    pub thinker: T,
    pub writer: TbWriter,
    pub metadata: T::Metadata,
    last_trained_at: usize,
    learn_waker: Option<(oneshot::Sender<()>, usize, PpoBuffer<E>, E::Params)>,
    rx: Receiver<BrainControl<E>>,
}

impl<E: Env, T: Thinker<E>> Brain<E, T> {
    pub fn new(
        thinker: T,
        name: String,
        timestamp: Timestamp,
        rx: Receiver<BrainControl<E>>,
    ) -> Self {
        let mut writer = TbWriter::default();
        writer.init(Some(name.as_str()), &timestamp);
        Self {
            metadata: thinker.init_metadata(1),
            timestamp: timestamp.to_owned(),
            last_action: E::Action::default(),
            thinker,
            writer,
            name,
            rx,
            learn_waker: None,
            last_trained_at: 0,
        }
    }

    pub fn save(&self) -> Result<(), Box<dyn std::error::Error>> {
        let path = format!("training/{}/{}", self.timestamp, self.name);
        std::fs::create_dir_all(&path).ok();
        self.thinker.save(path)?;
        Ok(())
    }
}

impl<E: Env, T: Thinker<E>> Brain<E, T> {
    pub fn act(
        &mut self,
        obs: &FrameStack<E::Observation>,
        params: &E::Params,
    ) -> Option<E::Action> {
        let action = self.thinker.act(obs, &mut self.metadata, params);
        if let Some(ref action) = action {
            self.last_action = action.clone();
        }
        // self.writer
        //     .add_scalar("Entropy", self.thinker.recent_entropy, frame_count);
        action
    }

    pub fn learn(&mut self, frame_count: usize, rb: &PpoBuffer<E>, params: &E::Params) {
        self.thinker.learn(rb, params);
        let net_reward = rb.reward.iter().sum::<f32>();
        let status = self.thinker.status();
        status.log(&mut self.writer, frame_count);
        self.writer.add_scalar("Reward", net_reward, frame_count);
    }

    pub async fn poll(&mut self) -> BrainStatus<E, T> {
        if let Some((waker, frame_count, rb, params)) = self.learn_waker.take() {
            self.learn(frame_count, &rb, &params);
            self.last_trained_at = frame_count;
            waker.send(()).unwrap();
        }
        match self.rx.recv().await {
            Some(ctrl) => match ctrl {
                BrainControl::NewObs {
                    frame_count: _,
                    obs,
                    params,
                } => {
                    // if frame_count % N_FRAME_STACK == 0 {
                    let meta = self.metadata.clone();
                    let status = self.thinker.status().clone();
                    let action = self.act(&obs, &params);
                    BrainStatus::NewStatus(ThinkerStatus {
                        last_action: action,
                        status: Some(status),
                        meta: Some(meta),
                    })
                    // } else {
                    //     BrainStatus::Ready
                    // }
                }
                BrainControl::Learn {
                    frame_count,
                    rb,
                    params,
                } => {
                    let (tx, rx) = oneshot::channel();
                    self.learn_waker = Some((tx, frame_count, rb, params));
                    BrainStatus::Wait(rx)
                }
            },
            None => BrainStatus::Error,
        }
    }
}

pub type AgentThinker = PpoThinker;

#[derive(Debug)]
pub struct ThinkerStatus<E: Env, T: Thinker<E>> {
    pub last_action: Option<E::Action>,
    pub status: Option<T::Status>,
    pub meta: Option<T::Metadata>,
}

impl<E: Env, T: Thinker<E>> Default for ThinkerStatus<E, T>
where
    T::Status: Default,
{
    fn default() -> Self {
        Self {
            last_action: Some(E::Action::default()),
            status: Some(T::Status::default()),
            meta: None,
        }
    }
}

impl<E: Env, T: Thinker<E>> Clone for ThinkerStatus<E, T>
where
    T::Status: Clone,
{
    fn clone(&self) -> Self {
        Self {
            last_action: self.last_action.clone(),
            status: self.status.clone(),
            meta: self.meta.clone(),
        }
    }
}

pub enum BrainStatus<E: Env, T: Thinker<E>> {
    NewStatus(ThinkerStatus<E, T>),
    Ready,
    Wait(oneshot::Receiver<()>),
    Error,
}

pub enum BrainControl<E: Env> {
    NewObs {
        obs: FrameStack<<E as Env>::Observation>,
        frame_count: usize,
        params: <E as Env>::Params,
    },
    Learn {
        rb: PpoBuffer<E>,
        frame_count: usize,
        params: <E as Env>::Params,
    },
}

#[derive(Resource)]
pub struct BrainBank<E: Env, T: Thinker<E>> {
    rxs: BTreeMap<usize, Receiver<BrainStatus<E, T>>>,
    txs: BTreeMap<usize, Sender<BrainControl<E>>>,
    statuses: BTreeMap<usize, ThinkerStatus<E, T>>,
    n_brains: usize,
    pub entity_to_brain: BTreeMap<Entity, usize>,
}

impl<E: Env, T: Thinker<E>> Default for BrainBank<E, T> {
    fn default() -> Self {
        Self {
            rxs: BTreeMap::default(),
            txs: BTreeMap::default(),
            statuses: BTreeMap::default(),
            n_brains: 0,
            entity_to_brain: BTreeMap::default(),
        }
    }
}

impl<E: Env + 'static, T: Thinker<E>> BrainBank<E, T>
where
    T: Send + 'static,
    T::Metadata: Send + Sync,
    T::Status: Send + Sync,
{
    pub fn spawn(
        &mut self,
        cons: impl FnOnce(Receiver<BrainControl<E>>) -> Brain<E, T> + Send + 'static,
    ) -> usize {
        static BRAIN_IDS: AtomicUsize = AtomicUsize::new(0);
        let id = BRAIN_IDS.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        self.n_brains += 1;
        let (tx, rx) = mpsc::channel(4);
        let (c_tx, c_rx) = mpsc::channel(4);
        let mut brain = cons(c_rx);
        println!("Brain {id} constructed");
        // run the brain on its own thread, not bevy's
        std::thread::spawn(move || {
            futures_lite::future::block_on(async {
                loop {
                    let status = brain.poll().await;
                    match status {
                        BrainStatus::Error => panic!("Brain returned error status"),
                        BrainStatus::Ready => std::thread::yield_now(),
                        status => tx.send(status).await.unwrap(),
                    }
                }
            });
        });
        self.rxs.insert(id, rx);
        self.txs.insert(id, c_tx);
        id
    }

    pub fn send_obs(
        &self,
        brain: usize,
        obs: FrameStack<E::Observation>,
        frame_count: usize,
        params: E::Params,
    ) {
        AsyncComputeTaskPool::get().scope(|scope| {
            scope.spawn(async {
                self.txs
                    .get(&brain)
                    .unwrap()
                    .send(BrainControl::NewObs {
                        obs,
                        frame_count,
                        params,
                    })
                    .await
                    .unwrap();
            })
        });
    }

    pub async fn learn(
        &self,
        brain: usize,
        frame_count: usize,
        rb: PpoBuffer<E>,
        params: E::Params,
    ) {
        let tx = self.txs.get(&brain).unwrap();
        tx.send(BrainControl::Learn {
            frame_count,
            rb,
            params,
        })
        .await
        .unwrap();
    }

    pub fn get_status(&mut self, brain: usize) -> Option<ThinkerStatus<E, T>> {
        while let Ok(status) = self.rxs.get_mut(&brain).unwrap().try_recv() {
            match status {
                BrainStatus::NewStatus(status) => {
                    self.statuses.insert(brain, status);
                }
                BrainStatus::Wait(waker) => {
                    AsyncComputeTaskPool::get().scope(|scope| {
                        scope.spawn(async {
                            waker.await.unwrap();
                        })
                    });
                }
                _ => {}
            }
        }
        self.statuses.get(&brain).cloned()
    }

    pub fn assign_entity(&mut self, brain: usize, ent: Entity) {
        self.entity_to_brain.insert(ent, brain);
    }

    pub fn brain_iter(&self) -> impl Iterator<Item = usize> {
        0..self.n_brains
    }
}
