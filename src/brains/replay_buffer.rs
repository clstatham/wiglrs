use std::collections::VecDeque;

use bevy::prelude::Component;
use burn_tch::TchBackend;

use crate::{
    envs::{Action, Env},
    FrameStack,
};

use super::thinkers::ppo::HiddenStates;

#[derive(Clone)]
pub struct Sart<E: Env> {
    pub obs: FrameStack<E::Observation>,
    pub action: E::Action,
    pub reward: f32,
    pub terminal: bool,
}

impl<E: Env> Sart<E> {
    pub fn unzip(self) -> (FrameStack<E::Observation>, E::Action, f32, bool) {
        (self.obs, self.action, self.reward, self.terminal)
    }
}

#[derive(Clone, Default)]
pub struct SartAdv<E: Env> {
    pub obs: E::Observation,
    pub action: E::Action,
    pub reward: f32,
    pub advantage: Option<f32>,
    pub returns: Option<f32>,
    pub terminal: bool,
}

impl<E: Env> SartAdv<E> {
    pub fn unzip(
        self,
    ) -> (
        E::Observation,
        E::Action,
        f32,
        Option<f32>,
        Option<f32>,
        bool,
    ) {
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

#[derive(Debug, Clone, Default)]
pub struct PpoMetadata {
    pub val: f32,
    pub logp: f32,
    pub hiddens: Option<HiddenStates<TchBackend<f32>>>,
}

#[derive(Component)]
pub struct PpoBuffer<E: Env> {
    pub obs: VecDeque<FrameStack<E::Observation>>,
    pub action: VecDeque<E::Action>,
    pub reward: VecDeque<f32>,
    pub advantage: VecDeque<Option<f32>>,
    pub returns: VecDeque<Option<f32>>,
    pub terminal: VecDeque<bool>,
    current_trajectory_start: usize,
}

impl<E: Env> Clone for PpoBuffer<E> {
    fn clone(&self) -> Self {
        Self {
            obs: self.obs.clone(),
            action: self.action.clone(),
            returns: self.returns.clone(),
            reward: self.reward.clone(),
            advantage: self.advantage.clone(),
            terminal: self.terminal.clone(),
            current_trajectory_start: self.current_trajectory_start,
        }
    }
}

impl<E: Env> Default for PpoBuffer<E> {
    fn default() -> Self {
        Self {
            obs: VecDeque::default(),
            action: VecDeque::default(),
            reward: VecDeque::default(),
            advantage: VecDeque::default(),
            returns: VecDeque::default(),
            terminal: VecDeque::default(),
            current_trajectory_start: 0,
        }
    }
}

impl<E: Env> PpoBuffer<E>
where
    E::Action: Action<E, Metadata = PpoMetadata>,
{
    pub fn remember_sart(&mut self, step: Sart<E>, max_len: Option<usize>) {
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

        if action.metadata().hiddens.is_some() {
            self.obs.push_back(obs);
            self.action.push_back(action);
            self.reward.push_back(reward);
            self.terminal.push_back(terminal);
            self.advantage.push_back(None);
            self.returns.push_back(None);

            self.current_trajectory_start += 1;
            if let Some(max_len) = max_len {
                if self.current_trajectory_start >= max_len {
                    self.current_trajectory_start = max_len;
                    self.finish_trajectory(); // in case one of them is an ABSOLUTE GAMER and doesn't die for like 100_000 frames
                }
            }
        }
    }

    pub fn finish_trajectory(&mut self) {
        let endpoint = self.obs.len();
        let startpoint = endpoint - self.current_trajectory_start;
        // push a temporary value of 0
        self.action.push_back(E::Action::default());
        let mut gae = 0.0;
        let mut ret = 0.0;

        for i in (startpoint..endpoint).rev() {
            let mask = if self.terminal[i] { 0.0 } else { 1.0 };
            let delta = self.reward[i] + 0.99 * self.action[i + 1].metadata().val * mask
                - self.action[i].metadata().val;
            gae = delta + 0.99 * 0.95 * mask * gae;
            self.advantage[i] = Some(gae);
            ret = self.reward[i] + 0.99 * mask * ret;
            self.returns[i] = Some(ret);
        }
        // remove the temporary value so we don't sample from it
        self.action.pop_back();
        self.current_trajectory_start = 0;
    }

    pub fn sample_batch(&self, batch_size: usize) -> Option<PpoBuffer<E>> {
        use rand::prelude::*;

        let end_of_last_traj = self.obs.len() - self.current_trajectory_start;
        let mut idxs = vec![0; batch_size];
        (0..end_of_last_traj).choose_multiple_fill(&mut thread_rng(), &mut idxs);
        let mut batch = PpoBuffer::default();
        for i in idxs {
            batch.obs.push_back(self.obs[i].to_owned());
            batch.action.push_back(self.action[i].clone());
            batch.reward.push_back(self.reward[i]);
            batch.terminal.push_back(self.terminal[i]);
            batch.advantage.push_back(self.advantage[i]);
            batch.returns.push_back(self.returns[i]);
        }
        Some(batch)
    }
}
