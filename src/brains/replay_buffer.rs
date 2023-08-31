use std::collections::VecDeque;

use bevy::{core::FrameCount, prelude::*};
use bevy_prng::ChaCha8Rng;
use bevy_rand::prelude::EntropyComponent;
use burn_tch::TchBackend;
use itertools::Itertools;
use rand::seq::IteratorRandom;

use crate::{
    envs::{
        ffa::{Agent, Reward, Terminal},
        Action, Env, Params,
    },
    FrameStack,
};

use super::thinkers::ppo::HiddenStates;

#[derive(Clone)]
pub struct Sart<E: Env> {
    pub obs: FrameStack<Box<[f32]>>,
    pub action: E::Action,
    pub reward: f32,
    pub terminal: bool,
}

impl<E: Env> Sart<E> {
    pub fn unzip(self) -> (FrameStack<Box<[f32]>>, E::Action, f32, bool) {
        (self.obs, self.action, self.reward, self.terminal)
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
    pub max_len: Option<usize>,
    pub obs: VecDeque<FrameStack<Box<[f32]>>>,
    pub action: VecDeque<E::Action>,
    pub reward: VecDeque<f32>,
    pub advantage: VecDeque<Option<f32>>,
    // pub returns: VecDeque<Option<f32>>,
    pub terminal: VecDeque<bool>,
    current_trajectory_start: usize,
    // returns_norm: RmsNormalize<TchBackend<f32>, 2>,
}

impl<E: Env> Clone for PpoBuffer<E> {
    fn clone(&self) -> Self {
        Self {
            max_len: self.max_len,
            obs: self.obs.clone(),
            action: self.action.clone(),
            // returns: self.returns.clone(),
            reward: self.reward.clone(),
            advantage: self.advantage.clone(),
            terminal: self.terminal.clone(),
            current_trajectory_start: self.current_trajectory_start,
        }
    }
}

impl<E: Env> PpoBuffer<E>
where
    E::Action: Action<E, Metadata = PpoMetadata>,
{
    pub fn new(max_len: Option<usize>) -> Self {
        Self {
            max_len,
            obs: VecDeque::default(),
            action: VecDeque::default(),
            reward: VecDeque::default(),
            advantage: VecDeque::default(),
            // returns: VecDeque::default(),
            terminal: VecDeque::default(),
            current_trajectory_start: 0,
            // returns_norm: RmsNormalize::new([1, 1].into()),
        }
    }

    pub fn remember_sart(&mut self, step: Sart<E>) {
        if let Some(max_len) = self.max_len {
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
            // while self.returns.len() >= max_len {
            //     self.returns.pop_front();
            // }
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
            self.action.push_back(action.clone());
            self.reward.push_back(reward);
            self.terminal.push_back(terminal);
            self.advantage.push_back(None);
            // self.returns.push_back(None);

            self.current_trajectory_start += 1;
            if let Some(max_len) = self.max_len {
                if self.current_trajectory_start >= max_len {
                    self.current_trajectory_start = max_len;
                    self.finish_trajectory(Some(action.metadata().val)); // in case one of them is an ABSOLUTE GAMER and doesn't die for like 100_000 frames
                }
            }
        }
    }

    pub fn finish_trajectory(&mut self, final_val: Option<f32>) {
        let endpoint = self.obs.len();
        let startpoint = endpoint - self.current_trajectory_start;
        let mut vals = self.action.iter().map(|a| a.metadata().val).collect_vec();
        vals.push(final_val.unwrap_or(0.0));
        let mut gae = 0.0;
        for i in (startpoint..endpoint).rev() {
            let mask = if self.terminal[i] { 0.0 } else { 1.0 };
            let delta = self.reward[i] + 0.99 * vals[i + 1] * mask - vals[i];
            gae = delta + 0.99 * 0.95 * mask * gae;
            self.advantage[i] = Some(gae);
            // ret = self.reward[i] + 0.99 * mask * ret;
            // self.returns[i] = Some(ret);
        }
        self.current_trajectory_start = 0;
    }

    pub fn sample_batch(
        &mut self,
        batch_size: usize,
        rng: &mut EntropyComponent<ChaCha8Rng>,
    ) -> Option<PpoBuffer<E>> {
        let end_of_last_traj = self.obs.len() - self.current_trajectory_start;
        // dbg!(end_of_last_traj);
        let idxs = (0..end_of_last_traj).choose_multiple(rng, batch_size);
        let mut batch = PpoBuffer::new(None);
        for i in idxs {
            batch.obs.push_back(self.obs[i].to_owned());
            batch.action.push_back(self.action[i].clone());
            batch.reward.push_back(self.reward[i]);
            batch.terminal.push_back(self.terminal[i]);
            batch.advantage.push_back(self.advantage[i]);
            // batch.returns.push_back(self.returns[i]);
        }
        Some(batch)
    }
}

pub fn store_sarts<E: Env>(
    params: Res<E::Params>,
    observations: Query<&FrameStack<Box<[f32]>>, With<Agent>>,
    actions: Query<&E::Action, With<Agent>>,
    mut rewards: Query<&mut Reward, With<Agent>>,
    mut rbs: Query<&mut PpoBuffer<E>, With<Agent>>,
    terminals: Query<&Terminal, With<Agent>>,
    agents: Query<Entity, With<Agent>>,
    frame_count: Res<FrameCount>,
) where
    E::Action: Action<E, Metadata = PpoMetadata>,
{
    if frame_count.0 as usize % params.agent_frame_stack_len() == 0 {
        for agent_ent in agents.iter() {
            let (action, reward, terminal) = (
                actions.get(agent_ent).unwrap().clone(),
                rewards.get(agent_ent).unwrap().0,
                terminals.get(agent_ent).unwrap().0,
            );
            let obs = observations.get(agent_ent).unwrap().clone();
            rbs.get_mut(agent_ent).unwrap().remember_sart(Sart {
                obs,
                action: action.to_owned(),
                reward,
                terminal,
            });
        }
        for mut reward in rewards.iter_mut() {
            reward.0 = 0.0;
        }
    }
}
