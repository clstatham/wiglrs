use bevy::prelude::*;
use std::{
    collections::{BTreeMap, VecDeque},
    fs::File,
    sync::atomic::AtomicU64,
};

use self::{
    replay_buffer::ReplayBuffer,
    thinkers::{
        ppo::{Be, PpoThinker},
        Thinker,
    },
};
use crate::{
    hparams::{AGENT_RB_MAX_LEN, N_FRAME_STACK},
    Action, Observation, TbWriter, Timestamp,
};
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

impl FrameStack {
    pub fn push(&mut self, s: Observation) {
        #[allow(clippy::reversed_empty_ranges)]
        for i in 0..N_FRAME_STACK - 1 {
            self.0[i] = self.0[i + 1].clone();
        }
        self.0[N_FRAME_STACK - 1] = s;
    }

    pub fn as_vec(&self) -> Vec<Observation> {
        self.0.clone().into()
    }
}

pub struct Brain<T: Thinker> {
    pub name: String,
    pub timestamp: Timestamp,
    pub deaths: u64,
    pub kills: usize,
    pub color: Color,
    pub id: u64,
    pub rb: ReplayBuffer<AGENT_RB_MAX_LEN>,
    pub fs: FrameStack,
    pub thinker: T,
    pub writer: TbWriter,
}

impl<T: Thinker> Brain<T> {
    pub fn new(thinker: T, timestamp: &Timestamp) -> Self {
        static BRAIN_IDS: AtomicU64 = AtomicU64::new(0);
        let id = BRAIN_IDS.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let name = crate::names::random_name();
        let mut writer = TbWriter::default();
        writer.init(Some(format!("{}_{}", id, &name).as_str()), timestamp);
        Self {
            name,
            timestamp: timestamp.to_owned(),
            color: Color::rgb(rand::random(), rand::random(), rand::random()),
            id,
            kills: 0,
            deaths: 0,
            rb: ReplayBuffer::default(),
            fs: FrameStack::default(),
            thinker,
            writer,
        }
    }

    pub fn save(&self) -> Result<(), Box<dyn std::error::Error>> {
        let path = format!(
            "training/{}/{}_{}_{}K_{}D",
            self.timestamp, self.id, self.name, self.kills, self.deaths
        );
        std::fs::create_dir_all(&path).ok();
        self.thinker.save(path)?;
        let rb = postcard::to_allocvec(&self.rb)?;
        let mut rb_f = File::create(format!(
            "training/{}/{}_{}_rb.postcard",
            self.timestamp, self.id, self.name
        ))?;
        use std::io::Write;
        rb_f.write_all(&rb)?;
        Ok(())
    }
}

impl Brain<PpoThinker> {
    pub fn act(&mut self, obs: Observation, frame_count: usize) -> Action {
        self.fs.push(obs);
        let action = self.thinker.act(self.fs.clone());
        self.writer
            .add_scalar("Entropy", self.thinker.recent_entropy, frame_count);
        action
    }

    pub fn learn(&mut self, frame_count: usize) {
        self.thinker.learn(&mut self.rb);
        let mean_reward =
            self.rb.buf.iter().map(|s| s.reward).sum::<f32>() / self.rb.buf.len() as f32;
        self.writer.add_scalar("Reward", mean_reward, frame_count);
        self.writer
            .add_scalar("Loss/Policy", self.thinker.recent_policy_loss, frame_count);
        self.writer
            .add_scalar("Loss/Value", self.thinker.recent_value_loss, frame_count);
        self.writer.add_scalar(
            "Loss/EntropyPenalty",
            self.thinker.recent_entropy_loss,
            frame_count,
        );
    }
}

pub type BrainBank = BTreeMap<Entity, Brain<PpoThinker>>;
