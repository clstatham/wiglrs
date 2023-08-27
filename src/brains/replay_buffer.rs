use std::collections::{BTreeMap, VecDeque};

use rand::{seq::IteratorRandom, thread_rng};

use serde::{Deserialize, Serialize};

use crate::Action;

use super::FrameStack;

#[derive(Clone, Default, Serialize, Deserialize)]
pub struct Sart {
    pub obs: FrameStack,
    pub action: Action,
    pub reward: f32,
    pub terminal: bool,
}

impl Sart {
    pub fn unzip(self) -> (FrameStack, Action, f32, bool) {
        (self.obs, self.action, self.reward, self.terminal)
    }
}

#[derive(Clone, Default, Serialize, Deserialize)]
pub struct SartAdv {
    pub obs: FrameStack,
    pub action: Action,
    pub reward: f32,
    pub advantage: Option<f32>,
    pub returns: Option<f32>,
    pub terminal: bool,
}

impl SartAdv {
    pub fn unzip(self) -> (FrameStack, Action, f32, Option<f32>, Option<f32>, bool) {
        (
            self.obs,
            self.action,
            self.reward,
            self.advantage,
            self.returns,
            self.terminal,
        )
    }
}

#[derive(Clone, Default, Serialize, Deserialize)]
pub struct SartAdvBuffer {
    pub obs: VecDeque<FrameStack>,
    pub action: VecDeque<Action>,
    pub reward: VecDeque<f32>,
    pub advantage: VecDeque<Option<f32>>,
    pub returns: VecDeque<Option<f32>>,
    pub terminal: VecDeque<bool>,
    current_trajectory_start: usize,
}

impl SartAdvBuffer {
    pub fn remember_sart(&mut self, step: Sart, max_len: Option<usize>) {
        if let Some(max_len) = max_len {
            while self.obs.len() >= max_len {
                self.obs.pop_front();
            }
            while self.action.len() >= max_len {
                self.action.pop_front();
            }
            while self.reward.len() >= max_len {
                self.reward.pop_front();
            }
            while self.advantage.len() >= max_len {
                self.advantage.pop_front();
            }
            while self.returns.len() >= max_len {
                self.returns.pop_front();
            }
            while self.terminal.len() >= max_len {
                self.terminal.pop_front();
            }
        }

        let Sart {
            obs,
            action,
            reward,
            terminal,
        } = step;

        self.obs.push_back(obs);
        self.action.push_back(action);
        self.reward.push_back(reward);
        self.terminal.push_back(terminal);
        self.advantage.push_back(None);
        self.returns.push_back(None);
        self.current_trajectory_start += 1;
    }

    pub fn finish_trajectory(&mut self) {
        let endpoint = self.obs.len();
        let startpoint = endpoint - self.current_trajectory_start;
        // push a temporary value of 0 so we can backprop through time
        self.action.push_back(Action {
            metadata: Some(crate::ActionMetadata {
                val: 0.0,
                ..Default::default()
            }),
            ..Default::default()
        });
        let mut gae = 0.0;
        let mut ret = 0.0;

        for i in (startpoint..endpoint).rev() {
            let mask = if self.terminal[i] { 0.0 } else { 1.0 };
            let delta = self.reward[i] + 0.99 * self.action[i + 1].metadata.unwrap().val * mask
                - self.action[i].metadata.unwrap().val;
            gae = delta + 0.99 * 0.95 * mask * gae;
            self.advantage[i] = Some(gae);
            ret = self.reward[i] + 0.99 * mask * ret;
            self.returns[i] = Some(ret);
        }
        // remove the temporary value so we don't sample from it
        self.action.pop_back();
        self.current_trajectory_start = 0;
    }

    pub fn sample_batch(&self, batch_size: usize) -> Option<SartAdvBuffer> {
        use rand::prelude::*;

        let end_of_last_traj = self.obs.len() - self.current_trajectory_start;
        let mut idxs = vec![0; batch_size];
        (0..end_of_last_traj).choose_multiple_fill(&mut thread_rng(), &mut idxs);
        let mut batch = SartAdvBuffer::default();
        for i in idxs {
            batch.obs.push_back(self.obs[i].to_owned());
            batch.action.push_back(self.action[i]);
            batch.reward.push_back(self.reward[i]);
            batch.terminal.push_back(self.terminal[i]);
            batch.advantage.push_back(self.advantage[i]);
            batch.returns.push_back(self.returns[i]);
        }
        Some(batch)
    }

    pub fn sample_many(
        rbs: &BTreeMap<usize, SartAdvBuffer>,
        batch_size: usize,
    ) -> Option<SartAdvBuffer> {
        let mut batch = SartAdvBuffer::default();
        for _ in 0..batch_size {
            let r = rbs.values().choose(&mut thread_rng())?;
            let b = r.sample_batch(1)?;
            batch.obs.push_back(b.obs[0].to_owned());
            batch.action.push_back(b.action[0].to_owned());
            batch.reward.push_back(b.reward[0].to_owned());
            batch.terminal.push_back(b.terminal[0].to_owned());
            batch.advantage.push_back(b.advantage[0].to_owned());
            batch.returns.push_back(b.returns[0].to_owned());
        }
        Some(batch)
    }
}
