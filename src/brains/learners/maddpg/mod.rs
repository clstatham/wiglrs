use bevy::prelude::*;
use bevy_prng::ChaCha8Rng;
use bevy_rand::prelude::EntropyComponent;
use itertools::Itertools;

use crate::{
    brains::models::{
        CentralizedCritic, CompoundPolicy, CopyWeights, CriticWithTarget, Policy, PolicyWithTarget,
        ValueEstimator,
    },
    envs::{Action, Env},
};

use candle_core::{IndexOp, Tensor};

use self::replay_buffer::{MaddpgBuffer, MaddpgBufferInner};

use super::{Learner, OffPolicyBuffer, Status, DEVICE};

pub mod replay_buffer;

#[derive(Debug, Clone, Copy, Default, Component)]
pub struct MaddpgStatus {
    pub policy_loss: f32,
    pub value_loss: f32,
}

impl Status for MaddpgStatus {
    fn log(&self, writer: &mut crate::TbWriter, step: usize) {
        writer.add_scalar("Policy/Loss", self.policy_loss, step);
        writer.add_scalar("Value/Loss", self.value_loss, step);
    }
}

#[derive(Resource)]
pub struct Maddpg<E: Env> {
    pub gamma: f32,
    pub tau: f32,
    // pub soft_update_interval: usize,
    pub steps_done: usize,
    pub batch_size: usize,
    pub status: MaddpgStatus,
    pub buffer: MaddpgBuffer<E>,
}

impl<E: Env, P: Policy + CopyWeights, V: ValueEstimator + CopyWeights>
    Learner<E, PolicyWithTarget<P>, CriticWithTarget<V>> for Maddpg<E>
where
    P: Policy<Logits = Tensor>,
{
    type Buffer = MaddpgBuffer<E>;
    type Status = MaddpgStatus;

    fn learn(&mut self, actors: &[PolicyWithTarget<P>], critics: &[CriticWithTarget<V>]) {
        let num_agents = actors.len();
        assert_eq!(self.buffer.bufs.len(), num_agents);
        assert_eq!(critics.len(), 1);
        let value = &critics[0];

        let mut total_al = 0.0f32;
        let mut total_vl = 0.0f32;
        for agent_idx in 0..num_agents {
            let batch = self.buffer.sample(self.batch_size);
            let (all_obs, all_actions, all_rewards, all_next_obs, all_terminals): (
                Vec<_>,
                Vec<_>,
                Vec<_>,
                Vec<_>,
                Vec<_>,
            ) = batch
                .bufs
                .iter()
                .map(|buf| buf.lock().unwrap().unzip())
                .multiunzip();

            let mut batch_obs = vec![];
            for obs in all_obs {
                let temp = Tensor::stack(
                    &obs.into_iter().map(|o| o.as_flat_tensor()).collect_vec(),
                    0,
                )
                .unwrap();
                batch_obs.push(temp);
            }
            let mut batch_actions = vec![];
            for action in all_actions {
                let temp =
                    Tensor::stack(&action.into_iter().map(|a| a.as_tensor()).collect_vec(), 0)
                        .unwrap();
                batch_actions.push(temp);
            }
            let mut batch_rewards = vec![];
            for reward in all_rewards {
                batch_rewards.push(Tensor::from_vec(reward, num_agents, &DEVICE).unwrap());
            }
            let mut batch_next_obs = vec![];
            for next_obs in all_next_obs {
                let temp = Tensor::stack(
                    &next_obs
                        .into_iter()
                        .map(|o| o.as_ref().unwrap().as_flat_tensor())
                        .collect_vec(),
                    0,
                )
                .unwrap();
                batch_next_obs.push(temp);
            }
            let mut batch_non_final_masks = vec![];
            for terminal in all_terminals {
                batch_non_final_masks.push(
                    Tensor::from_vec(
                        terminal
                            .iter()
                            .map(|t| if *t { 0.0f32 } else { 1.0f32 })
                            .collect_vec(),
                        num_agents,
                        &DEVICE,
                    )
                    .unwrap(),
                );
            }

            assert_eq!(batch_obs.len(), self.batch_size); // sanity check

            let whole_state = Tensor::stack(&batch_obs, 0).unwrap();
            let whole_state_flat = whole_state.flatten_from(1).unwrap();
            let whole_action = Tensor::stack(&batch_actions, 0).unwrap();
            let whole_action_flat = whole_action.flatten_from(1).unwrap();
            let current_q = value
                .estimate_value(&whole_state_flat, Some(&whole_action_flat))
                .unwrap();

            let mut next_actions = vec![];
            for i in 0..num_agents {
                next_actions.push(
                    actors
                        .get(i)
                        .unwrap()
                        .action_logits(&whole_state.i((.., i)).unwrap())
                        .unwrap(),
                );
            }
            let next_actions = Tensor::stack(&next_actions, 0)
                .unwrap()
                .transpose(0, 1)
                .unwrap()
                .contiguous()
                .unwrap();

            let next_states = Tensor::stack(&batch_next_obs, 0)
                .unwrap()
                .transpose(0, 1)
                .unwrap()
                .contiguous()
                .unwrap();
            let target_q = value
                .target_critic
                .estimate_value(
                    &next_states
                        .reshape((self.batch_size, next_states.elem_count() / self.batch_size))
                        .unwrap(),
                    Some(
                        &next_actions
                            .reshape((self.batch_size, next_actions.elem_count() / self.batch_size))
                            .unwrap(),
                    ),
                )
                .unwrap();

            let batch_rewards = Tensor::stack(&batch_rewards, 0).unwrap();
            let batch_non_final_masks = Tensor::stack(&batch_non_final_masks, 0).unwrap();
            let target_q = target_q
                .affine(self.gamma as f64, 0.0)
                .unwrap()
                .mul(&batch_non_final_masks.i((.., agent_idx)).unwrap())
                .unwrap()
                .add(&batch_rewards.i((.., agent_idx)).unwrap())
                .unwrap();
            let loss_q = (&current_q - target_q.detach())
                .unwrap()
                .sqr()
                .unwrap()
                .mean(0)
                .unwrap();
            let vl: f32 = loss_q.to_scalar().unwrap();
            let q_grads = loss_q.backward().unwrap();
            value.apply_gradients(&q_grads).unwrap();

            let whole_state = Tensor::stack(&batch_obs, 0).unwrap();
            let state_i = whole_state.i((.., agent_idx)).unwrap();
            let action_i = actors
                .get(agent_idx)
                .unwrap()
                .policy
                .action_logits(&state_i)
                .unwrap();
            let mut ac = batch_actions.clone();
            for (i, actions) in ac.iter_mut().enumerate() {
                let before = actions.i(0..agent_idx).unwrap();
                let after = actions.i(agent_idx + 1..).unwrap();
                *actions = Tensor::cat(
                    &[before, action_i.i(i).unwrap().unsqueeze(0).unwrap(), after],
                    0,
                )
                .unwrap();
            }
            let whole_action = Tensor::stack(&ac, 0).unwrap().flatten_from(1).unwrap();
            let actor_loss = value
                .estimate_value(&whole_state, Some(&whole_action))
                .unwrap()
                .neg()
                .unwrap()
                .mean(0)
                .unwrap();
            let al: f32 = actor_loss.to_scalar().unwrap();
            let actor_grads = actor_loss.backward().unwrap();
            actors
                .get(agent_idx)
                .unwrap()
                .apply_gradients(&actor_grads)
                .unwrap();

            total_al += al;
            total_vl += vl;
        }

        // if self.steps_done % self.soft_update_interval == 0 && self.steps_done > 0 {
        // println!("updating");
        for i in 0..num_agents {
            actors.get(i).unwrap().soft_update(self.tau);
        }
        value.soft_update(self.tau);
        // }
        self.steps_done += 1;

        self.status.policy_loss = total_al / num_agents as f32;
        self.status.value_loss = total_vl / num_agents as f32;

        println!("Policy loss: {}", self.status.policy_loss);
        println!("Value loss: {}", self.status.value_loss);
    }

    fn status(&self) -> Self::Status {
        self.status
    }
}
