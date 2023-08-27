use bevy::prelude::*;
use std::{
    collections::{BTreeMap, VecDeque},
    sync::{atomic::AtomicUsize, Arc, Mutex},
};

use self::{
    replay_buffer::SartAdvBuffer,
    thinkers::{ppo::PpoThinker, SharedThinker, Thinker},
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
    pub deaths: usize,
    pub kills: usize,
    pub color: Color,
    pub id: usize,
    pub fs: FrameStack,
    pub last_action: Action,
    pub thinker: T,
    pub writer: TbWriter,
    pub metadata: T::Metadata,
}

impl<T: Thinker> Brain<T> {
    pub fn new(thinker: T, timestamp: &Timestamp) -> Self {
        static BRAIN_IDS: AtomicUsize = AtomicUsize::new(0);
        let name = crate::names::random_name();
        let mut writer = TbWriter::default();
        writer.init(Some(name.as_str()), timestamp);
        Self {
            metadata: thinker.init_metadata(1),
            name,
            timestamp: timestamp.to_owned(),
            color: Color::rgb(rand::random(), rand::random(), rand::random()),
            id: BRAIN_IDS.fetch_add(1, std::sync::atomic::Ordering::SeqCst),
            kills: 0,
            deaths: 0,
            fs: FrameStack::default(),
            last_action: Action::default(),
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
        Ok(())
    }
}

impl Brain<PpoThinker> {
    pub fn act(&mut self, _obs: Observation, frame_count: usize) -> Action {
        let action = self.thinker.act(self.fs.clone(), &mut self.metadata);
        self.last_action = action;
        self.writer
            .add_scalar("Entropy", self.thinker.recent_entropy, frame_count);
        action
    }

    pub fn learn(&mut self, frame_count: usize, rbs: &BTreeMap<usize, SartAdvBuffer>) {
        self.thinker.learn(&rbs[&self.id]);
        let net_reward = rbs[&self.id].reward.iter().sum::<f32>();
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
}

impl Brain<SharedThinker<PpoThinker>> {
    pub fn act(&mut self, _obs: Observation, frame_count: usize) -> Action {
        let action = self.thinker.act(self.fs.clone(), &mut self.metadata);
        self.last_action = action;
        self.writer
            .add_scalar("Entropy", self.thinker.lock().recent_entropy, frame_count);
        action
    }

    pub fn learn(&mut self, frame_count: usize, rbs: &BTreeMap<usize, SartAdvBuffer>) {
        self.thinker.learn(&rbs[&self.id]);
        let net_reward = rbs[&self.id].reward.iter().sum::<f32>();
        self.writer.add_scalar("Reward", net_reward, frame_count);
        self.writer.add_scalar(
            "Loss/Policy",
            self.thinker.lock().recent_policy_loss,
            frame_count,
        );
        self.writer.add_scalar(
            "Loss/Value",
            self.thinker.lock().recent_value_loss,
            frame_count,
        );
        self.writer.add_scalar(
            "Loss/EntropyPenalty",
            self.thinker.lock().recent_entropy_loss,
            frame_count,
        );
        self.writer.add_scalar(
            "PolicyClampRatio",
            self.thinker.lock().recent_nclamp,
            frame_count,
        );
    }
}

pub type BrainBank = BTreeMap<Entity, Brain<SharedThinker<PpoThinker>>>;
