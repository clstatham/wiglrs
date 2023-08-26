use std::collections::VecDeque;

use itertools::Itertools;
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
pub struct SartAdvBatch {
    pub obs: Vec<FrameStack>,
    pub action: Vec<Action>,
    pub reward: Vec<f32>,
    pub advantage: Vec<f32>,
    pub returns: Vec<f32>,
    pub terminal: Vec<bool>,
}

#[derive(Clone, Default, Serialize, Deserialize)]
pub struct SartAdvBuffer<const MAX_LEN: usize> {
    pub buf: VecDeque<SartAdv>,
    current_trajectory_start: usize,
}

impl<const MAX_LEN: usize> SartAdvBuffer<MAX_LEN> {
    pub fn remember_sart(&mut self, step: Sart) {
        if self.buf.len() >= MAX_LEN {
            self.buf.pop_front();
        }
        self.buf.push_back(SartAdv {
            obs: step.obs,
            action: step.action,
            reward: step.reward,
            advantage: None,
            returns: None,
            terminal: step.terminal,
        });
        self.current_trajectory_start += 1;
    }

    pub fn finish_trajectory(&mut self) {
        let endpoint = self.buf.len();
        let startpoint = endpoint - self.current_trajectory_start;
        // push a temporary value of 0 so we can backprop through time
        self.buf.push_back(SartAdv {
            action: Action {
                metadata: Some(crate::ActionMetadata {
                    val: 0.0,
                    ..Default::default()
                }),
                ..Default::default()
            },
            ..Default::default()
        });
        let mut gae = 0.0;
        let mut ret = 0.0;

        for i in (startpoint..endpoint).rev() {
            let mask = if self.buf[i].terminal { 0.0 } else { 1.0 };
            let delta = self.buf[i].reward
                + 0.99 * self.buf[i + 1].action.metadata.unwrap().val * mask
                - self.buf[i].action.metadata.unwrap().val;
            gae = delta + 0.99 * 0.95 * mask * gae;
            self.buf[i].advantage = Some(gae);
        }
        for i in (startpoint..endpoint).rev() {
            let mask = if self.buf[i].terminal { 0.0 } else { 1.0 };

            ret = self.buf[i].reward + 0.99 * mask * ret;
            self.buf[i].returns = Some(ret);
        }
        // remove the temporary value so we don't sample from it
        self.buf.pop_back();
        self.current_trajectory_start = 0;
    }

    pub fn sample_batch(&mut self, batch_size: usize) -> SartAdvBatch {
        use rand::prelude::*;

        let end_of_last_traj = self.buf.len() - self.current_trajectory_start;
        assert!(
            end_of_last_traj > batch_size,
            "Not enough samples to collect a batch yet!"
        );
        let mut batch = vec![SartAdv::default(); batch_size];
        self.buf
            .iter()
            .take(end_of_last_traj)
            .cloned()
            .choose_multiple_fill(&mut thread_rng(), &mut batch);
        let (s, a, r, adv, ret, t): (Vec<_>, Vec<_>, Vec<_>, Vec<_>, Vec<_>, Vec<_>) =
            batch.into_iter().map(|b| b.to_owned().unzip()).multiunzip();
        let adv = adv.into_iter().map(|a| a.unwrap()).collect_vec();
        let ret = ret.into_iter().map(|a| a.unwrap()).collect_vec();
        SartAdvBatch {
            obs: s,
            action: a,
            reward: r,
            advantage: adv,
            returns: ret,
            terminal: t,
        }
    }

    pub fn unzip(&mut self) -> SartAdvBatch {
        assert!(!self.buf.is_empty(), "Replay buffer is empty");
        assert_eq!(
            self.current_trajectory_start, 0,
            "You MUST call `finish_trajectory` first!"
        );
        let (s, a, r, adv, ret, t): (Vec<_>, Vec<_>, Vec<_>, Vec<_>, Vec<_>, Vec<_>) = self
            .buf
            .iter()
            .cloned()
            .map(|b| b.to_owned().unzip())
            .multiunzip();
        let adv = adv.into_iter().map(|a| a.unwrap()).collect_vec();
        let ret = ret.into_iter().map(|a| a.unwrap()).collect_vec();
        SartAdvBatch {
            obs: s,
            action: a,
            reward: r,
            advantage: adv,
            returns: ret,
            terminal: t,
        }
    }
}
