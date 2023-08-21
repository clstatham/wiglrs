use std::collections::VecDeque;

use itertools::Itertools;

use crate::{Action, Observation};

#[derive(Clone, Default)]
pub struct SavedStep {
    pub obs: Observation,
    pub action: Action,
    pub reward: f32,
    pub terminal: bool,
}

impl SavedStep {
    pub fn unzip(self) -> (Observation, Action, f32, bool) {
        (self.obs, self.action, self.reward, self.terminal)
    }
}

#[derive(Clone, Default)]
pub struct ReplayBuffer<const MAX_LEN: usize> {
    pub buf: VecDeque<SavedStep>,
}

impl<const MAX_LEN: usize> ReplayBuffer<MAX_LEN> {
    pub fn remember(&mut self, step: SavedStep) {
        self.buf.push_back(step);
        if self.buf.len() >= MAX_LEN {
            self.buf.pop_front();
        }
    }

    pub fn sample_batch(
        &self,
        batch_size: usize,
    ) -> (Vec<Observation>, Vec<Action>, Vec<f32>, Vec<bool>) {
        use rand::prelude::*;
        let mut batch = vec![SavedStep::default(); batch_size];
        self.buf
            .iter()
            .cloned()
            .choose_multiple_fill(&mut thread_rng(), &mut batch);
        let (s, a, r, t) = batch.into_iter().map(|b| b.to_owned().unzip()).multiunzip();
        (s, a, r, t)
    }
}
