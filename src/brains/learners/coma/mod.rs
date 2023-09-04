use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
};

use candle_core::Tensor;

use bevy::prelude::*;
use bevy_prng::ChaCha8Rng;
use bevy_rand::prelude::EntropyComponent;
use itertools::Itertools;

use crate::{
    brains::{
        learners::DEVICE,
        models::{CentralizedCritic, CompoundPolicy, Policy, ValueEstimator},
    },
    envs::{Action, Env, StepMetadata},
    FrameStack,
};

use super::{Buffer, Learner, Sart, Status};

#[derive(Default)]
pub struct ComaMetadata {
    pub id: usize,
}

impl StepMetadata for ComaMetadata {
    fn calculate<E: Env, P: Policy, V: ValueEstimator>(
        obs: &FrameStack<Box<[f32]>>,
        action: &E::Action,
        policy: &P,
        value: &V,
    ) -> Self
    where
        E::Action: crate::envs::Action<E, Logits = P::Logits>,
    {
        todo!()
    }
}

#[derive(Component)]
pub struct ComaBufferInner<E: Env>
where
    Self: Buffer<E>,
{
    pub max_len: Option<usize>,
    pub obs: VecDeque<FrameStack<Box<[f32]>>>,
    pub action: VecDeque<E::Action>,
    pub reward: VecDeque<f32>,
}

impl<E: Env> ComaBufferInner<E> {
    pub fn unzip(&self) -> (Vec<FrameStack<Box<[f32]>>>, Vec<E::Action>, Vec<f32>) {
        (
            Vec::from(self.obs.clone()),
            Vec::from(self.action.clone()),
            Vec::from(self.reward.clone()),
        )
    }
}

impl<E: Env> Clone for ComaBufferInner<E> {
    fn clone(&self) -> Self {
        Self {
            max_len: self.max_len,
            obs: self.obs.clone(),
            action: self.action.clone(),
            reward: self.reward.clone(),
        }
    }
}

impl<E: Env> Default for ComaBufferInner<E> {
    fn default() -> Self {
        Self {
            max_len: None,
            obs: VecDeque::default(),
            action: VecDeque::default(),
            reward: VecDeque::default(),
        }
    }
}

impl<E: Env> Buffer<E> for ComaBufferInner<E> {
    type Metadata = ();

    fn remember_sart(&mut self, step: super::Sart<E, ()>) {
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
        }

        let Sart {
            obs,
            action,
            reward,
            ..
        } = step;

        self.obs.push_back(obs);
        self.action.push_back(action.clone());
        self.reward.push_back(reward);
    }

    fn finish_trajectory(&mut self, final_val: Option<f32>) {
        todo!()
    }

    fn shuffled_and_batched(
        &self,
        batch_size: usize,
        rng: &mut EntropyComponent<ChaCha8Rng>,
    ) -> Vec<Self> {
        unimplemented!()
    }
}

#[derive(Component)]
pub struct ComaBuffer<E: Env> {
    bufs: Vec<Arc<Mutex<ComaBufferInner<E>>>>,
}

impl<E: Env> Clone for ComaBuffer<E> {
    fn clone(&self) -> Self {
        Self {
            bufs: self.bufs.clone(),
        }
    }
}

impl<E: Env> Default for ComaBuffer<E> {
    fn default() -> Self {
        Self {
            bufs: Vec::default(),
        }
    }
}

impl<E: Env> Buffer<E> for ComaBuffer<E> {
    type Metadata = ComaMetadata;

    fn remember_sart(&mut self, step: super::Sart<E, Self::Metadata>) {
        let Sart {
            obs,
            action,
            reward,
            terminal,
            metadata,
        } = step;
        self.bufs[metadata.id].lock().unwrap().remember_sart(Sart {
            obs,
            action,
            reward,
            terminal,
            metadata: (),
        });
    }

    fn finish_trajectory(&mut self, final_val: Option<f32>) {
        todo!()
    }

    fn shuffled_and_batched(
        &self,
        batch_size: usize,
        rng: &mut EntropyComponent<ChaCha8Rng>,
    ) -> Vec<Self> {
        todo!()
    }
}

#[derive(Component, Default, Clone, Copy)]
pub struct ComaStatus;

impl Status for ComaStatus {
    fn log(&self, writer: &mut crate::TbWriter, step: usize) {}
}

#[derive(Component)]
pub struct Coma {
    pub action_len: usize,
    pub gamma: f32,
}

impl<E: Env, P: Policy, V: ValueEstimator> Learner<E, CompoundPolicy<P>, CentralizedCritic<V>>
    for Coma
{
    type Buffer = ComaBuffer<E>;

    type Status = ComaStatus;

    fn learn(
        &mut self,
        policy: &CompoundPolicy<P>,
        value: &CentralizedCritic<V>,
        buffer: &Self::Buffer,
        rng: &mut EntropyComponent<ChaCha8Rng>,
    ) {
        let num_agents = policy.len();
        assert_eq!(buffer.bufs.len(), num_agents);
        let (all_obs, all_actions, all_rewards): (Vec<_>, Vec<_>, Vec<_>) = buffer
            .bufs
            .iter()
            .map(|buf| buf.lock().unwrap().unzip())
            .multiunzip();
        let mut agent_obs = vec![];
        for obs in all_obs {
            let temp =
                Tensor::stack(&obs.into_iter().map(|o| o.as_tensor()).collect_vec(), 0).unwrap();
            agent_obs.push(temp);
        }
        let mut agent_actions = vec![];
        for action in all_actions {
            let temp =
                Tensor::stack(&action.into_iter().map(|a| a.as_tensor()).collect_vec(), 0).unwrap();
            agent_actions.push(temp);
        }
        let mut agent_rewards = vec![];
        for reward in all_rewards {
            agent_rewards.push(Tensor::from_iter(reward, &DEVICE).unwrap());
        }

        let q = value
            .estimate_value(&Tensor::cat(&agent_obs, 1).unwrap())
            .unwrap();
    }

    fn status(&self) -> Self::Status {
        todo!()
    }
}
