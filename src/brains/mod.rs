use bevy::prelude::*;
use std::{
    collections::BTreeMap,
    sync::{atomic::AtomicU64, Arc, Mutex},
};

use crate::{Action, Observation};

use self::{replay_buffer::ReplayBuffer, thinkers::Thinker};

pub mod replay_buffer;
pub mod thinkers;

#[derive(Clone)]
pub struct FrameStack<const NSTACK: usize>([Observation; NSTACK]);

impl<const NSTACK: usize> Default for FrameStack<NSTACK> {
    fn default() -> Self {
        Self([Observation::default(); NSTACK])
    }
}

impl<const NSTACK: usize> FrameStack<NSTACK> {
    pub fn push(&mut self, s: Observation) {
        for i in 0..NSTACK - 1 {
            self.0[i] = self.0[i + 1];
        }
        self.0[NSTACK - 1] = s;
    }
}

pub struct Brain {
    pub name: String,
    pub version: u64,
    pub color: Color,
    pub id: u64,
    pub rb: ReplayBuffer,
    pub thinker: Arc<Mutex<dyn Thinker + 'static>>,
}

impl Brain {
    pub fn new(thinker: impl Thinker + 'static) -> Self {
        static BRAIN_IDS: AtomicU64 = AtomicU64::new(0);
        let id = BRAIN_IDS.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let name = crate::names::random_name();

        Self {
            name,
            color: Color::rgb(rand::random(), rand::random(), rand::random()),
            id,
            version: 0,
            rb: ReplayBuffer::default(),
            thinker: Arc::new(Mutex::new(thinker)),
        }
    }

    pub fn act(&mut self, obs: Observation) -> Action {
        self.thinker.lock().unwrap().act(obs)
    }

    pub fn learn(&mut self) {
        self.thinker.lock().unwrap().learn(&mut self.rb);
    }
}

pub type BrainBank = BTreeMap<Entity, Brain>;
