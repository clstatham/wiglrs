use std::collections::VecDeque;

use bevy::{core::FrameCount, prelude::*};
use bevy_prng::ChaCha8Rng;
use bevy_rand::prelude::EntropyComponent;
use itertools::Itertools;
use rand::seq::SliceRandom;

use crate::{
    brains::{
        learners::{Buffer, Sart},
        models::{Policy, ValueEstimator},
    },
    envs::{Action, Agent, Env, Params, Reward, StepMetadata, Terminal},
    FrameStack,
};

#[derive(Debug, Clone, Default, Component)]
pub struct PpoMetadata {
    pub val: f32,
    pub logp: f32,
}

impl StepMetadata for PpoMetadata {
    fn calculate<E: Env, P: Policy, V: ValueEstimator>(
        obs: &FrameStack<Box<[f32]>>,
        action: &E::Action,
        policy: &P,
        value: &V,
    ) -> Self
    where
        E::Action: Action<E, Logits = P::Logits>,
    {
        let obs = obs.as_tensor();
        Self {
            val: value
                .estimate_value(&obs)
                .unwrap()
                .reshape(())
                .unwrap()
                .to_scalar()
                .unwrap(),
            logp: policy
                .log_prob(
                    action.logits().unwrap(),
                    &action.as_tensor().unsqueeze(0).unwrap(),
                )
                .unwrap()
                .reshape(())
                .unwrap()
                .to_scalar()
                .unwrap(),
        }
    }
}

#[derive(Component)]
pub struct PpoBuffer<E: Env>
where
    Self: Buffer<E>,
{
    pub max_len: Option<usize>,
    pub obs: VecDeque<FrameStack<Box<[f32]>>>,
    pub action: VecDeque<E::Action>,
    pub reward: VecDeque<f32>,
    pub advantage: VecDeque<Option<f32>>,
    pub metadata: VecDeque<<Self as Buffer<E>>::Metadata>,
    pub terminal: VecDeque<bool>,
    current_trajectory_start: usize,
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
            metadata: self.metadata.clone(),
            current_trajectory_start: self.current_trajectory_start,
        }
    }
}

impl<E: Env> PpoBuffer<E> {
    pub fn new(max_len: Option<usize>) -> Self {
        Self {
            max_len,
            obs: VecDeque::default(),
            action: VecDeque::default(),
            reward: VecDeque::default(),
            advantage: VecDeque::default(),
            terminal: VecDeque::default(),
            metadata: VecDeque::default(),
            current_trajectory_start: 0,
        }
    }
}
impl<E: Env> Buffer<E> for PpoBuffer<E> {
    type Metadata = PpoMetadata;

    fn remember_sart(&mut self, step: Sart<E, PpoMetadata>) {
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
            while self.terminal.len() >= max_len {
                self.terminal.pop_front();
            }
            while self.metadata.len() >= max_len {
                self.metadata.pop_front();
            }
        }

        let Sart {
            obs,
            action,
            reward,
            terminal,
            metadata,
        } = step;

        self.obs.push_back(obs);
        self.action.push_back(action.clone());
        self.reward.push_back(reward);
        self.terminal.push_back(terminal);
        self.advantage.push_back(None);
        self.metadata.push_back(metadata);
    }

    fn finish_trajectory(&mut self, final_val: Option<f32>) {
        let endpoint = self.obs.len();
        let mut vals = self.metadata.iter().map(|m| m.val).collect_vec();
        vals.push(final_val.unwrap_or(0.0));
        let mut gae = 0.0;
        for i in (0..endpoint).rev() {
            let mask = if self.terminal[i] { 0.0 } else { 1.0 };
            let delta = self.reward[i] + 0.99 * vals[i + 1] * mask - vals[i];
            gae = delta + 0.99 * 1.0 * mask * gae;
            self.advantage[i] = Some(gae);
        }
    }

    fn shuffled_and_batched(
        &self,
        batch_size: usize,
        rng: &mut EntropyComponent<ChaCha8Rng>,
    ) -> Vec<PpoBuffer<E>> {
        let mut idxs = (0..self.obs.len()).collect_vec();
        idxs.shuffle(rng);
        let mut counter = 0;
        let mut batches = vec![];
        while counter + batch_size < idxs.len() {
            let mut batch = PpoBuffer::new(None);
            let end = counter + batch_size;
            let batch_idxs = &idxs[counter..end];
            for i in batch_idxs {
                batch.obs.push_back(self.obs[*i].to_owned());
                batch.action.push_back(self.action[*i].clone());
                batch.reward.push_back(self.reward[*i]);
                batch.terminal.push_back(self.terminal[*i]);
                batch.advantage.push_back(self.advantage[*i]);
                batch.metadata.push_back(self.metadata[*i].clone());
            }
            counter = end;
            batches.push(batch);
        }
        batches
    }
}

pub fn store_sarts<E: Env, P: Policy, V: ValueEstimator>(
    params: Res<E::Params>,
    observations: Query<&FrameStack<Box<[f32]>>, With<Agent>>,
    actions: Query<&E::Action, With<Agent>>,
    mut rewards: Query<&mut Reward, With<Agent>>,
    mut rbs: Query<&mut PpoBuffer<E>, With<Agent>>,
    terminals: Query<&Terminal, With<Agent>>,
    agents: Query<Entity, With<Agent>>,
    frame_count: Res<FrameCount>,
    pv: Query<(&P, &V), With<Agent>>,
) where
    E::Action: Action<E, Logits = P::Logits>,
{
    if frame_count.0 as usize % params.agent_frame_stack_len() == 0 {
        for agent_ent in agents.iter() {
            let (action, reward, terminal) = (
                actions.get(agent_ent).unwrap().clone(),
                rewards.get(agent_ent).unwrap().0,
                terminals.get(agent_ent).unwrap().0,
            );
            let obs = observations.get(agent_ent).unwrap().clone();
            let policy = pv.get_component::<P>(agent_ent).unwrap();
            let value = pv.get_component::<V>(agent_ent).unwrap();
            let metadata = PpoMetadata::calculate::<E, P, V>(&obs, &action, policy, value);
            rbs.get_mut(agent_ent).unwrap().remember_sart(Sart {
                obs,
                action: action.to_owned(),
                reward,
                terminal,
                metadata,
            });
        }
        for mut reward in rewards.iter_mut() {
            reward.0 = 0.0;
        }
    }
}
