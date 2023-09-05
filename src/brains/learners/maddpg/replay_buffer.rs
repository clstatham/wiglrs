use bevy::{core::FrameCount, prelude::*};
use itertools::Itertools;
use rand_distr::{Distribution, Uniform};

use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
};

use crate::{
    brains::{
        learners::{
            utils::{RmsNormalize, SumTree},
            Buffer, OffPolicyBuffer, Sart,
        },
        models::{Policy, ValueEstimator},
        AgentPolicy, AgentValue, Policies, ValueEstimators,
    },
    envs::{
        Action, Agent, AgentId, CurrentObservation, Env, NextObservation, Params, Reward,
        StepMetadata, Terminal,
    },
    FrameStack,
};

use super::Maddpg;

pub struct Experience<E: Env> {
    pub agent_idx: usize,
    pub obs: FrameStack<Box<[f32]>>,
    pub action: E::Action,
    pub next_obs: FrameStack<Box<[f32]>>,
    pub terminal: bool,
    pub reward: f32,
    pub value: f32,
    pub target_value: f32,
}

impl<E: Env> Clone for Experience<E>
where
    E::Action: Clone,
{
    fn clone(&self) -> Self {
        Self {
            agent_idx: self.agent_idx,
            obs: self.obs.clone(),
            action: self.action.clone(),
            next_obs: self.next_obs.clone(),
            terminal: self.terminal,
            reward: self.reward,
            value: self.value,
            target_value: self.target_value,
        }
    }
}

#[derive(Resource)]
pub struct MaddpgBufferInner<E: Env>
where
    Self: Buffer<E>,
{
    pub max_len: usize,
    pub tree: SumTree<Experience<E>>,
    pub alpha: f64,
    pub beta: f64,
    pub epsilon: f64,
}

impl<E: Env> MaddpgBufferInner<E> {
    pub fn new(max_len: usize) -> Self {
        Self {
            max_len,
            alpha: 0.6,
            beta: 0.4,
            epsilon: 0.01,
            tree: SumTree::new(max_len),
        }
    }

    fn get_priority(&self, error: f64) -> f64 {
        (error.abs() + self.epsilon).powf(self.alpha)
    }
}

impl<E: Env> Clone for MaddpgBufferInner<E> {
    fn clone(&self) -> Self {
        Self {
            alpha: self.alpha,
            beta: self.beta,
            epsilon: self.epsilon,
            max_len: self.max_len,
            tree: self.tree.clone(),
        }
    }
}

impl<E: Env> Buffer<E> for MaddpgBufferInner<E> {
    type Experience = Experience<E>;
    fn remember_sart(&mut self, step: Self::Experience) {
        let error = if step.terminal {
            (step.reward - step.value).abs()
        } else {
            ((step.reward + 0.99 * step.target_value) - step.value).abs()
        };
        self.tree.add(self.get_priority(error as f64), step);
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct MaddpgMetadata {
    pub idx: usize,
}

impl StepMetadata for MaddpgMetadata {
    fn calculate<A, P: Policy, V: ValueEstimator>(
        _obs: &FrameStack<Box<[f32]>>,
        _action: &A,
        _policy: &P,
        _value: &V,
    ) -> Self
    where
        A: Action<Logits = P::Logits>,
    {
        unimplemented!()
    }
}

#[derive(Resource)]
pub struct MaddpgBuffer<E: Env> {
    pub bufs: Vec<Arc<Mutex<MaddpgBufferInner<E>>>>,
}

impl<E: Env> MaddpgBuffer<E> {
    pub fn new(num_agents: usize, max_len: usize) -> Self {
        let mut bufs = vec![];
        for _ in 0..num_agents {
            bufs.push(Arc::new(Mutex::new(MaddpgBufferInner::new(max_len))));
        }
        Self {
            bufs,
            ..Default::default()
        }
    }
}

impl<E: Env> Clone for MaddpgBuffer<E> {
    fn clone(&self) -> Self {
        Self {
            bufs: self.bufs.clone(),
        }
    }
}

impl<E: Env> Default for MaddpgBuffer<E> {
    fn default() -> Self {
        Self {
            bufs: Vec::default(),
        }
    }
}

impl<E: Env> Buffer<E> for MaddpgBuffer<E> {
    type Experience = Vec<Experience<E>>;
    fn remember_sart(&mut self, step: Vec<Experience<E>>) {
        for (buf, step) in self.bufs.iter().zip(step.into_iter()) {
            buf.lock().unwrap().remember_sart(step);
        }
    }
}

impl<E: Env> OffPolicyBuffer<E> for MaddpgBuffer<E> {
    fn sample(&self, batch_size: usize) -> Vec<Vec<Experience<E>>> {
        let mut out = Vec::new();

        for i in 0..batch_size {
            let mut temp = Vec::new();
            for buf in self.bufs.iter() {
                let buf = buf.lock().unwrap();
                let segment = buf.tree.total() / batch_size as f64;
                let a = segment * i as f64;
                let b = segment * (i + 1) as f64;
                let dist = Uniform::new(a, b);
                let (idx, p, data) = buf.tree.get(dist.sample(&mut rand::thread_rng()));
                // dbg!(p);
                temp.push(data.unwrap());
            }
            out.push(temp);
        }
        out
    }
}

pub fn store_sarts<E: Env>(
    params: Res<Params>,
    observations: Query<(&AgentId, &NextObservation, &CurrentObservation), With<Agent>>,
    actions: Query<&E::Action, With<Agent>>,
    mut rewards: Query<(&mut Reward, &mut RmsNormalize), With<Agent>>,
    learner: Res<Maddpg<E>>,
    terminals: Query<&Terminal, With<Agent>>,
    agents: Query<Entity, With<Agent>>,
    frame_count: Res<FrameCount>,
    policies: Res<Policies<AgentPolicy>>,
    values: Res<ValueEstimators<AgentValue>>,
) where
    E::Action: Action<Logits = candle_core::Tensor>,
{
    if frame_count.0 as usize % params.get_int("agent_frame_stack_len").unwrap() as usize == 0 {
        for (mut reward, mut norm) in rewards.iter_mut() {
            reward.0 = norm.forward_ret(reward.0);
        }
        if frame_count.0 as usize > params.get_int("agent_warmup").unwrap() as usize {
            let mut current_actions = vec![None; policies.0.len()];
            let mut target_actions = vec![None; policies.0.len()];
            let mut all_current_obs = vec![None; policies.0.len()];
            let mut all_next_obs = vec![None; policies.0.len()];
            for agent_ent in agents.iter() {
                let (agent_idx, next_obs, current_obs) = observations.get(agent_ent).unwrap();
                let current_obs = current_obs.0.as_tensor().unsqueeze(0).unwrap();
                let next_obs = next_obs
                    .0
                    .as_ref()
                    .unwrap()
                    .as_tensor()
                    .unsqueeze(0)
                    .unwrap();
                all_current_obs[agent_idx.0] = Some(current_obs.clone());
                all_next_obs[agent_idx.0] = Some(next_obs.clone());
                let policy = &policies.0[agent_idx.0];
                let target_action = policy.target_policy.action_logits(&next_obs).unwrap();
                target_actions[agent_idx.0] = Some(target_action);
                current_actions[agent_idx.0] = Some(
                    actions
                        .get(agent_ent)
                        .unwrap()
                        .as_tensor()
                        .unsqueeze(0)
                        .unwrap(),
                );
            }
            let (value, target_value) = {
                let current_actions = candle_core::Tensor::cat(
                    &current_actions
                        .into_iter()
                        .map(Option::unwrap)
                        .collect_vec(),
                    1,
                )
                .unwrap();
                let target_actions = candle_core::Tensor::cat(
                    &target_actions.into_iter().map(Option::unwrap).collect_vec(),
                    1,
                )
                .unwrap();
                let current_obs = candle_core::Tensor::cat(
                    &all_current_obs
                        .into_iter()
                        .map(Option::unwrap)
                        .collect_vec(),
                    1,
                )
                .unwrap();
                let next_obs = candle_core::Tensor::cat(
                    &all_next_obs.into_iter().map(Option::unwrap).collect_vec(),
                    1,
                )
                .unwrap();
                let value: f32 = values.0[0]
                    .critic
                    .estimate_value(&current_obs, Some(&current_actions))
                    .unwrap()
                    .reshape(())
                    .unwrap()
                    .to_scalar()
                    .unwrap();
                let target_value: f32 = values.0[0]
                    .target_critic
                    .estimate_value(&next_obs, Some(&target_actions))
                    .unwrap()
                    .reshape(())
                    .unwrap()
                    .to_scalar()
                    .unwrap();
                (value, target_value)
            };
            for agent_ent in agents.iter() {
                let (action, reward, terminal) = (
                    actions.get(agent_ent).unwrap().clone(),
                    rewards.get(agent_ent).unwrap().0 .0,
                    terminals.get(agent_ent).unwrap().0,
                );
                // let agent_idx = agent_ids.get(agent_ent).unwrap().0;
                let (agent_idx, next_obs, current_obs) = observations.get(agent_ent).unwrap();

                let exp = Experience {
                    agent_idx: agent_idx.0,
                    obs: current_obs.0.clone(),
                    action,
                    next_obs: next_obs.0.as_ref().unwrap().clone(),
                    terminal,
                    reward,
                    value,
                    target_value,
                };

                learner.buffer.bufs[agent_idx.0]
                    .lock()
                    .unwrap()
                    .remember_sart(exp);
            }
        }
        for (mut reward, _) in rewards.iter_mut() {
            reward.0 = 0.0;
        }
    }
}
