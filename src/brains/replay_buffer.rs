use std::collections::VecDeque;

use itertools::Itertools;
use serde::{Deserialize, Serialize};

use crate::Action;

use super::FrameStack;

#[derive(Clone, Default, Serialize, Deserialize)]
pub struct SavedStep {
    pub obs: FrameStack,
    pub action: Action,
    pub reward: f32,
    pub terminal: bool,
}

impl SavedStep {
    pub fn unzip(self) -> (FrameStack, Action, f32, bool) {
        (self.obs, self.action, self.reward, self.terminal)
    }
}

#[derive(Clone, Default, Serialize, Deserialize)]
pub struct ReplayBuffer<const MAX_LEN: usize> {
    pub buf: VecDeque<SavedStep>,
}

impl<const MAX_LEN: usize> ReplayBuffer<MAX_LEN> {
    pub fn remember(&mut self, step: SavedStep) {
        if self.buf.len() >= MAX_LEN {
            self.buf.pop_front();
        }
        self.buf.push_back(step);
    }

    pub fn sample_batch(
        &self,
        batch_size: usize,
    ) -> (Vec<FrameStack>, Vec<Action>, Vec<f32>, Vec<bool>) {
        use rand::prelude::*;
        let mut batch = vec![SavedStep::default(); batch_size];
        self.buf
            .iter()
            .cloned()
            .choose_multiple_fill(&mut thread_rng(), &mut batch);
        let (s, a, r, t) = batch.into_iter().map(|b| b.to_owned().unzip()).multiunzip();
        (s, a, r, t)
    }

    pub fn unzip(&self) -> (Vec<FrameStack>, Vec<Action>, Vec<f32>, Vec<bool>) {
        self.buf.iter().map(|b| b.to_owned().unzip()).multiunzip()
    }
}
