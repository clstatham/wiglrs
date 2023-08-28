use bevy::prelude::*;
use bevy_tasks::AsyncComputeTaskPool;
use burn_tch::TchBackend;
use std::{
    collections::{BTreeMap, VecDeque},
    fmt,
    sync::atomic::AtomicUsize,
};
use tokio::sync::{
    mpsc::{self, Receiver, Sender},
    oneshot,
};

use self::{
    replay_buffer::PpoBuffer,
    thinkers::{
        ppo::{HiddenStates, PpoThinker},
        SharedThinker, Status, Thinker,
    },
};
use crate::{hparams::N_FRAME_STACK, Action, Observation, TbWriter, Timestamp};
use serde::{Deserialize, Serialize};

pub mod replay_buffer;
pub mod thinkers;

#[derive(Clone, Serialize, Deserialize)]
pub struct FrameStack(VecDeque<Observation>);

impl Default for FrameStack {
    fn default() -> Self {
        Self(vec![Observation::default(); N_FRAME_STACK].into())
    }
}

impl fmt::Debug for FrameStack {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "FrameStack")
    }
}

impl FrameStack {
    pub fn push(&mut self, s: Observation) {
        if self.0.len() >= N_FRAME_STACK {
            self.0.pop_front();
        }
        self.0.push_back(s);
    }

    pub fn as_vec(&self) -> Vec<Observation> {
        self.0.clone().into()
    }
}

pub struct Brain<T: Thinker> {
    pub name: String,
    pub timestamp: Timestamp,
    pub fs: FrameStack,
    pub last_action: Action,
    pub thinker: T,
    pub writer: TbWriter,
    pub metadata: T::Metadata,
    last_trained_at: usize,
    learn_waker: Option<(oneshot::Sender<()>, usize, PpoBuffer)>,
    rx: Receiver<BrainControl>,
}

impl<T: Thinker> Brain<T> {
    pub fn new(thinker: T, name: String, timestamp: Timestamp, rx: Receiver<BrainControl>) -> Self {
        let mut writer = TbWriter::default();
        writer.init(Some(name.as_str()), &timestamp);
        Self {
            metadata: thinker.init_metadata(1),
            timestamp: timestamp.to_owned(),
            fs: FrameStack::default(),
            last_action: Action::default(),
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

impl<T: Thinker> Brain<T> {
    pub fn act(&mut self) -> Action {
        let action = self.thinker.act(self.fs.clone(), &mut self.metadata);
        self.last_action = action;
        // self.writer
        //     .add_scalar("Entropy", self.thinker.recent_entropy, frame_count);
        action
    }

    pub fn learn(&mut self, frame_count: usize, rb: &PpoBuffer) {
        self.thinker.learn(rb);
        let net_reward = rb.reward.iter().sum::<f32>();
        let status = self.thinker.status();
        status.log(&mut self.writer, frame_count);
        self.writer.add_scalar("Reward", net_reward, frame_count);
    }

    pub async fn poll(&mut self) -> BrainStatus<T> {
        if let Some((waker, frame_count, rb)) = self.learn_waker.take() {
            self.learn(frame_count, &rb);
            self.last_trained_at = frame_count;
            waker.send(()).unwrap();
        }
        match self.rx.recv().await {
            Some(ctrl) => match ctrl {
                BrainControl::NewObs { frame_count, obs } => {
                    self.fs.push(obs);
                    if frame_count % N_FRAME_STACK == 0 {
                        let meta = self.metadata.clone();
                        let status = self.thinker.status().clone();
                        self.act();
                        BrainStatus::NewStatus(ThinkerStatus {
                            last_action: self.last_action,
                            status: Some(status),
                            fs: self.fs.clone(),
                            meta: Some(meta),
                        })
                    } else {
                        BrainStatus::Ready
                    }
                }
                BrainControl::Learn { frame_count, rb } => {
                    let (tx, rx) = oneshot::channel();
                    self.learn_waker = Some((tx, frame_count, rb));
                    BrainStatus::Wait(rx)
                }
            },
            None => BrainStatus::Error,
        }
    }
}

pub type AgentThinker = SharedThinker<PpoThinker>;

#[derive(Debug)]
pub struct ThinkerStatus<T: Thinker> {
    pub last_action: Action,
    pub status: Option<T::Status>,
    pub fs: FrameStack,
    pub meta: Option<T::Metadata>,
}

impl<T: Thinker> Default for ThinkerStatus<T>
where
    T::Status: Default,
{
    fn default() -> Self {
        Self {
            last_action: Action::default(),
            status: Some(T::Status::default()),
            fs: FrameStack::default(),
            meta: None,
        }
    }
}

impl<T: Thinker> Clone for ThinkerStatus<T>
where
    T::Status: Clone,
{
    fn clone(&self) -> Self {
        Self {
            last_action: self.last_action,
            status: self.status.clone(),
            fs: self.fs.clone(),
            meta: self.meta.clone(),
        }
    }
}

pub enum BrainStatus<T: Thinker> {
    NewStatus(ThinkerStatus<T>),
    Ready,
    Wait(oneshot::Receiver<()>),
    Error,
}

pub enum BrainControl {
    NewObs {
        obs: Observation,
        frame_count: usize,
    },
    Learn {
        rb: PpoBuffer,
        frame_count: usize,
    },
}

#[derive(Default, Resource)]
pub struct BrainBank<T: Thinker> {
    rxs: BTreeMap<usize, Receiver<BrainStatus<T>>>,
    txs: BTreeMap<usize, Sender<BrainControl>>,
    statuses: BTreeMap<usize, ThinkerStatus<T>>,
    n_brains: usize,
    pub entity_to_brain: BTreeMap<Entity, usize>,
}

impl<T: Thinker> BrainBank<T>
where
    T: Send + 'static,
    T::Metadata: Send + Sync,
    T::Status: Send + Sync,
{
    pub fn spawn(
        &mut self,
        cons: impl FnOnce(Receiver<BrainControl>) -> Brain<T> + Send + 'static,
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

    pub fn send_obs(&self, ent: Entity, obs: Observation, frame_count: usize) {
        AsyncComputeTaskPool::get().scope(|scope| {
            scope.spawn(async {
                self.txs
                    .get(&self.entity_to_brain[&ent])
                    .unwrap()
                    .send(BrainControl::NewObs { obs, frame_count })
                    .await
                    .unwrap();
            })
        });
    }

    pub async fn learn(&self, brain: usize, frame_count: usize, rb: PpoBuffer) {
        let tx = self.txs.get(&brain).unwrap();
        tx.send(BrainControl::Learn { frame_count, rb })
            .await
            .unwrap();
    }

    pub fn get_status(&mut self, brain: usize) -> Option<ThinkerStatus<T>> {
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
