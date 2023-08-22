use bevy::prelude::Vec2;

use crate::{Action, Observation};

use super::{replay_buffer::ReplayBuffer, FrameStack};

pub mod ppo;

pub trait Thinker {
    fn act(&self, obs: FrameStack) -> Action;
    fn learn(&mut self, b: &mut ReplayBuffer) -> f32;
}

pub struct RandomThinker;

impl Thinker for RandomThinker {
    fn act(&self, _obs: FrameStack) -> Action {
        Action {
            lin_force: Vec2::new(
                rand::random::<f32>() * 2.0 - 1.0,
                rand::random::<f32>() * 2.0 - 1.0,
            ),
            ang_force: rand::random::<f32>() * 2.0 - 1.0,
            shoot: rand::random::<f32>() * 2.0 - 1.0,
        }
    }
    fn learn(&mut self, _b: &mut ReplayBuffer) -> f32 {
        0.0
    }
}
