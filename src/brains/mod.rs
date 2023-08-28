use bevy::prelude::*;
use burn_tch::TchBackend;
use futures_lite::future;
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
        Thinker,
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

impl Brain<PpoThinker> {
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
        self.writer.add_scalar("Reward", net_reward, frame_count);
        self.writer
            .add_scalar("Loss/Policy", self.thinker.recent_policy_loss, frame_count);
        self.writer
            .add_scalar("Loss/Value", self.thinker.recent_value_loss, frame_count);
        self.writer.add_scalar(
            "Loss/EntropyPenalty",
            self.thinker.recent_entropy_loss,
            frame_count,
        );
        self.writer
            .add_scalar("PolicyClampRatio", self.thinker.recent_nclamp, frame_count);
    }

    pub async fn poll(&mut self) -> BrainStatus {
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
                        self.act();
                        BrainStatus::NewStatus(ThinkerStatus {
                            last_action: self.last_action,
                            recent_policy_loss: self.thinker.recent_policy_loss,
                            recent_value_loss: self.thinker.recent_value_loss,
                            recent_entropy_loss: self.thinker.recent_entropy_loss,
                            recent_nclamp: self.thinker.recent_nclamp,
                            recent_mu: self.thinker.recent_mu.clone(),
                            recent_std: self.thinker.recent_std.clone(),
                            recent_entropy: self.thinker.recent_entropy,
                            fs: self.fs.clone(),
                            hiddens: Some(self.metadata.clone()),
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

pub type AgentThinker = PpoThinker;

#[derive(Clone, Default, Debug)]
pub struct ThinkerStatus {
    pub last_action: Action,
    pub recent_mu: Vec<f32>,
    pub recent_std: Vec<f32>,
    pub recent_policy_loss: f32,
    pub recent_value_loss: f32,
    pub recent_entropy_loss: f32,
    pub recent_nclamp: f32,
    pub recent_entropy: f32,
    pub fs: FrameStack,
    pub hiddens: Option<HiddenStates<TchBackend<f32>>>,
}

#[derive(Debug)]
pub enum BrainStatus {
    NewStatus(ThinkerStatus),
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
pub struct BrainBank {
    rxs: BTreeMap<usize, Receiver<BrainStatus>>,
    txs: BTreeMap<usize, Sender<BrainControl>>,
    statuses: BTreeMap<usize, ThinkerStatus>,
    n_brains: usize,
    pub entity_to_brain: BTreeMap<Entity, usize>,
}

impl BrainBank {
    pub fn spawn(
        &mut self,
        cons: impl FnOnce(Receiver<BrainControl>) -> Brain<AgentThinker> + Send + 'static,
    ) -> usize {
        static BRAIN_IDS: AtomicUsize = AtomicUsize::new(0);
        let id = BRAIN_IDS.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        self.n_brains += 1;
        let (tx, rx) = mpsc::channel(16);
        let (c_tx, c_rx) = mpsc::channel(16);
        let mut brain = cons(c_rx);
        println!("Brain {id} constructed");
        std::thread::spawn(move || {
            future::block_on(async move {
                loop {
                    let status = brain.poll().await;
                    match status {
                        BrainStatus::Error => panic!("Brain returned error status"),
                        BrainStatus::Ready => {}
                        status => tx.send(status).await.unwrap(),
                    }
                    std::thread::yield_now();
                }
            })
        });
        self.rxs.insert(id, rx);
        self.txs.insert(id, c_tx);
        // });
        id
    }

    pub fn send_obs(&self, ent: Entity, obs: Observation, frame_count: usize) {
        // if let Some(tx) = self.txs.get(&self.entity_to_brain[&ent]) {
        future::block_on(async {
            self.txs
                .get(&self.entity_to_brain[&ent])
                .unwrap()
                .send(BrainControl::NewObs { obs, frame_count })
                .await
                .unwrap();
        });
        // }
    }

    pub fn learn(&self, brain: usize, frame_count: usize, rb: PpoBuffer) {
        let tx = self.txs.get(&brain).unwrap();
        future::block_on(async {
            tx.send(BrainControl::Learn { frame_count, rb })
                .await
                .unwrap();
        });
    }

    pub fn get_status(&mut self, brain: usize) -> Option<ThinkerStatus> {
        while let Ok(status) = self.rxs.get_mut(&brain).unwrap().try_recv() {
            match status {
                BrainStatus::NewStatus(status) => {
                    self.statuses.insert(brain, status);
                }
                BrainStatus::Wait(waker) => future::block_on(async {
                    waker.await.unwrap();
                }),
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
