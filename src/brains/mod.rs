use bevy::prelude::*;
use std::{collections::BTreeMap, sync::atomic::AtomicU64};

use crate::{hparams::N_FRAME_STACK, Action, Observation};

use self::{
    replay_buffer::ReplayBuffer,
    thinkers::{ppo::PpoThinker, Thinker},
};

pub mod replay_buffer;
pub mod thinkers;

#[derive(Clone, Copy)]
pub struct FrameStack([Observation; N_FRAME_STACK]);

impl Default for FrameStack {
    fn default() -> Self {
        Self([Observation::default(); N_FRAME_STACK])
    }
}

impl FrameStack {
    pub fn push(&mut self, s: Observation) {
        #[allow(clippy::reversed_empty_ranges)]
        for i in 0..N_FRAME_STACK - 1 {
            self.0[i] = self.0[i + 1];
        }
        self.0[N_FRAME_STACK - 1] = s;
    }

    pub fn as_vec(&self) -> Vec<Observation> {
        self.0.to_vec()
    }
}

pub struct Brain<T: Thinker> {
    pub name: String,
    pub timestamp: String,
    pub version: u64,
    pub kills: usize,
    pub color: Color,
    pub id: u64,
    pub rb: ReplayBuffer,
    pub fs: FrameStack,
    pub thinker: T,
}

impl<T: Thinker> Brain<T> {
    pub fn new(thinker: T, timestamp: String) -> Self {
        static BRAIN_IDS: AtomicU64 = AtomicU64::new(0);
        let id = BRAIN_IDS.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let name = crate::names::random_name();

        Self {
            name,
            timestamp,
            color: Color::rgb(rand::random(), rand::random(), rand::random()),
            id,
            kills: 0,
            version: 0,
            rb: ReplayBuffer::default(),
            fs: FrameStack::default(),
            thinker,
        }
    }

    pub fn act(&mut self, obs: Observation) -> Action {
        self.fs.push(obs);
        self.thinker.act(self.fs)
    }

    pub fn learn(&mut self) {
        self.thinker.learn(&mut self.rb)
    }

    pub fn save(&self) -> Result<(), Box<dyn std::error::Error>> {
        let path = format!(
            "training/{}/{}_{}_{}K_{}D",
            self.timestamp, self.id, self.name, self.kills, self.version
        );
        std::fs::create_dir_all(&path).ok();
        self.thinker.save(path)
    }
}

pub type BrainBank = BTreeMap<Entity, Brain<PpoThinker>>;
