use bevy::prelude::Vec2;

use crate::{Action, Observation};

use super::Brain;

pub trait Thinker {
    fn act(&self, obs: Observation) -> Action;
    fn learn(&mut self, b: &Brain<Self>);
}

pub struct RandomThinker;

impl Thinker for RandomThinker {
    fn act(&self, obs: Observation) -> Action {
        Action {
            lin_force: Vec2::new(
                rand::random::<f32>() * 2.0 - 1.0,
                rand::random::<f32>() * 2.0 - 1.0,
            ),
            ang_force: rand::random::<f32>() * 2.0 - 1.0,
            shoot: rand::random::<bool>(),
        }
    }
    fn learn(&mut self, b: &Brain<Self>) {}
}
