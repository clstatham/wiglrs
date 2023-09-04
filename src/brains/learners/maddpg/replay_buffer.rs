use bevy::{core::FrameCount, prelude::*};
use bevy_prng::ChaCha8Rng;
use bevy_rand::prelude::EntropyComponent;
use itertools::Itertools;
use rand_distr::Distribution;

use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
};

use crate::{
    brains::{
        learners::{utils::RmsNormalize, Buffer, OffPolicyBuffer, Sart},
        models::{Policy, ValueEstimator},
    },
    envs::{Action, Agent, AgentId, Env, Params, Reward, StepMetadata, Terminal},
    FrameStack,
};

use super::Maddpg;

#[derive(Resource)]
pub struct MaddpgBufferInner<E: Env>
where
    Self: OffPolicyBuffer<E>,
{
    pub max_len: Option<usize>,
    pub obs: VecDeque<FrameStack<Box<[f32]>>>,
    pub actions: VecDeque<E::Action>,
    // todo: fixme: this is hacky and lazy - we only populate next_obs when sampling batches
    pub next_obs: VecDeque<Option<FrameStack<Box<[f32]>>>>,
    pub terminals: VecDeque<bool>,
    pub rewards: VecDeque<f32>,
}

impl<E: Env> MaddpgBufferInner<E> {
    pub fn new(max_len: Option<usize>) -> Self {
        Self {
            max_len,
            ..Default::default()
        }
    }

    pub fn unzip(
        &self,
    ) -> (
        Vec<FrameStack<Box<[f32]>>>,
        Vec<E::Action>,
        Vec<f32>,
        Vec<Option<FrameStack<Box<[f32]>>>>,
        Vec<bool>,
    ) {
        (
            Vec::from(self.obs.clone()),
            Vec::from(self.actions.clone()),
            Vec::from(self.rewards.clone()),
            Vec::from(self.next_obs.clone()),
            Vec::from(self.terminals.clone()),
        )
    }
}

impl<E: Env> Default for MaddpgBufferInner<E> {
    fn default() -> Self {
        Self {
            max_len: None,
            obs: VecDeque::default(),
            actions: VecDeque::default(),
            next_obs: VecDeque::default(),
            terminals: VecDeque::default(),
            rewards: VecDeque::default(),
        }
    }
}

impl<E: Env> Clone for MaddpgBufferInner<E> {
    fn clone(&self) -> Self {
        Self {
            max_len: self.max_len,
            obs: self.obs.clone(),
            actions: self.actions.clone(),
            next_obs: self.next_obs.clone(),
            terminals: self.terminals.clone(),
            rewards: self.rewards.clone(),
        }
    }
}

impl<E: Env> Buffer<E> for MaddpgBufferInner<E> {
    type Metadata = ();
    fn remember_sart(&mut self, step: Sart<E, Self::Metadata>) {
        if let Some(max_len) = self.max_len {
            while self.obs.len() >= max_len {
                self.obs.pop_front();
            }
            while self.actions.len() >= max_len {
                self.actions.pop_front();
            }
            while self.rewards.len() >= max_len {
                self.rewards.pop_front();
            }
            while self.terminals.len() >= max_len {
                self.terminals.pop_front();
            }
            while self.next_obs.len() >= max_len {
                self.next_obs.pop_front();
            }
        }

        let Sart {
            obs,
            action,
            reward,
            terminal,
            ..
        } = step;

        self.obs.push_back(obs);
        self.actions.push_back(action);
        self.rewards.push_back(reward);
        self.terminals.push_back(terminal);
        self.next_obs.push_back(None);
    }
}

impl<E: Env> OffPolicyBuffer<E> for MaddpgBufferInner<E> {
    fn sample(&self, batch_size: usize) -> Self {
        use rand::seq::SliceRandom;
        let mut idxs = (0..self.obs.len() - 1).collect_vec();
        idxs.shuffle(&mut rand::thread_rng());
        let mut out = Self::default();

        for i in &idxs[..batch_size] {
            out.obs.push_back(self.obs[*i].clone());
            out.actions.push_back(self.actions[*i].clone());
            out.next_obs.push_back(Some(self.obs[*i + 1].clone()));
            out.rewards.push_back(self.rewards[*i]);
        }

        out
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct MaddpgMetadata {
    pub idx: usize,
}

impl StepMetadata for MaddpgMetadata {
    fn calculate<E: Env, P: Policy, V: ValueEstimator>(
        obs: &FrameStack<Box<[f32]>>,
        action: &E::Action,
        policy: &P,
        value: &V,
    ) -> Self
    where
        E::Action: Action<E, Logits = P::Logits>,
    {
        unimplemented!()
    }
}

#[derive(Resource)]
pub struct MaddpgBuffer<E: Env> {
    pub bufs: Vec<Arc<Mutex<MaddpgBufferInner<E>>>>,
}

impl<E: Env> MaddpgBuffer<E> {
    pub fn new(num_agents: usize, max_len: Option<usize>) -> Self {
        let mut bufs = vec![];
        for _ in 0..num_agents {
            bufs.push(Arc::new(Mutex::new(MaddpgBufferInner::new(max_len))));
        }
        Self { bufs }
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
    type Metadata = MaddpgMetadata;
    fn remember_sart(&mut self, step: Sart<E, Self::Metadata>) {
        self.bufs[step.metadata.idx]
            .lock()
            .unwrap()
            .remember_sart(Sart {
                obs: step.obs,
                action: step.action,
                reward: step.reward,
                terminal: step.terminal,
                metadata: (),
            });
    }
}

impl<E: Env> OffPolicyBuffer<E> for MaddpgBuffer<E> {
    fn sample(&self, batch_size: usize) -> Self {
        let buf_len = self.bufs[0].lock().unwrap().obs.len();
        let mut out = Self::default();
        let dist = rand::distributions::Uniform::new(0, buf_len - 2);

        // this is a little weird: we actually use the `inner` buffers to hold one sample for each agent, and return `batch_size` of them
        // this is a hack and i am ashamed
        for _ in 0..batch_size {
            let i = dist.sample(&mut rand::thread_rng());
            let mut temp = MaddpgBufferInner::default();
            for buf in self.bufs.iter() {
                let buf = buf.lock().unwrap();
                temp.obs.push_back(buf.obs[i].clone());
                temp.actions.push_back(buf.actions[i].clone());
                temp.next_obs.push_back(Some(buf.obs[i + 1].clone()));
                temp.rewards.push_back(buf.rewards[i]);
                temp.terminals.push_back(buf.terminals[i]);
            }
            out.bufs.push(Arc::new(Mutex::new(temp)));
        }
        out
    }
}

pub fn store_sarts<E: Env, P: Policy, V: ValueEstimator>(
    params: Res<E::Params>,
    observations: Query<&FrameStack<Box<[f32]>>, With<Agent>>,
    actions: Query<&E::Action, With<Agent>>,
    mut rewards: Query<(&mut Reward, &mut RmsNormalize), With<Agent>>,
    mut learner: ResMut<Maddpg<E>>,
    terminals: Query<&Terminal, With<Agent>>,
    agents: Query<Entity, With<Agent>>,
    agent_ids: Query<&AgentId, With<Agent>>,
    frame_count: Res<FrameCount>,
) where
    E::Action: Action<E, Logits = P::Logits>,
{
    if frame_count.0 as usize % params.agent_frame_stack_len() == 0 {
        if frame_count.0 as usize > params.agent_warmup() {
            for agent_ent in agents.iter() {
                let (action, reward, terminal) = (
                    actions.get(agent_ent).unwrap().clone(),
                    rewards.get(agent_ent).unwrap().0 .0,
                    terminals.get(agent_ent).unwrap().0,
                );
                let obs = observations.get(agent_ent).unwrap().clone();
                let metadata = MaddpgMetadata {
                    idx: agent_ids.get(agent_ent).unwrap().0,
                };
                learner.buffer.remember_sart(Sart {
                    obs,
                    action: action.to_owned(),
                    reward,
                    terminal,
                    metadata,
                });
            }
        }
        for (mut reward, _) in rewards.iter_mut() {
            reward.0 = 0.0;
        }
    }
}
