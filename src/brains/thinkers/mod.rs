use std::path::Path;

use bevy::prelude::Vec2;

use crate::Action;

use super::{replay_buffer::SartAdvBuffer, FrameStack};

pub mod ppo;
pub mod stats;

pub trait Thinker {
    fn act(&mut self, obs: FrameStack) -> Action;
    fn learn(&mut self, b: &SartAdvBuffer);
    fn save(&self, path: impl AsRef<Path>) -> Result<(), Box<dyn std::error::Error>>;
}

pub struct RandomThinker;

impl Thinker for RandomThinker {
    fn act(&mut self, _obs: FrameStack) -> Action {
        Action {
            lin_force: Vec2::new(
                rand::random::<f32>() * 2.0 - 1.0,
                rand::random::<f32>() * 2.0 - 1.0,
            ),
            ang_force: rand::random::<f32>() * 2.0 - 1.0,
            shoot: rand::random::<f32>() * 2.0 - 1.0,
            metadata: None,
        }
    }
    fn learn(&mut self, _b: &SartAdvBuffer) {}
    fn save(&self, _path: impl AsRef<Path>) -> Result<(), Box<dyn std::error::Error>> {
        Ok(())
    }
}
