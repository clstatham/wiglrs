use bevy::{ecs::schedule::SystemConfigs, prelude::*};

use crate::FrameStack;

pub mod ffa;
pub mod maps;
pub mod modules;
pub mod tdm;

pub trait Action<E: Env + ?Sized>: Clone + Default {
    type Metadata: Default;
    fn as_slice(&self, params: &E::Params) -> Box<[f32]>;
    fn from_slice(v: &[f32], metadata: Self::Metadata, params: &E::Params) -> Self;
    fn metadata(&self) -> Self::Metadata;
}

pub trait Observation: Clone
where
    Self: Sized,
{
    fn as_slice<P: Params>(&self, params: &P) -> Box<[f32]>;
}

pub trait DefaultFrameStack<E: Env + ?Sized>: Observation {
    fn default_frame_stack(params: &E::Params) -> FrameStack<Self>;
}

pub trait Params {
    fn agent_radius(&self) -> f32;
    fn agent_max_health(&self) -> f32;
    fn num_agents(&self) -> usize;
    fn agent_frame_stack_len(&self) -> usize;
}

pub trait Env: Resource {
    type Params: Params + Default + Resource + Send + Sync;
    type Observation: Observation + DefaultFrameStack<Self> + Component + Send + Sync;
    type Action: Action<Self> + Component + Send + Sync;

    fn init() -> Self;

    fn setup_system() -> SystemConfigs;
    fn observation_system() -> SystemConfigs;
    fn action_system() -> SystemConfigs;
    fn reward_system() -> SystemConfigs;
    fn terminal_system() -> SystemConfigs;
    fn update_system() -> SystemConfigs;
    fn learn_system() -> SystemConfigs;
    fn ui_system() -> SystemConfigs;

    fn main_system() -> SystemConfigs {
        (
            Self::observation_system(),
            Self::action_system(),
            Self::reward_system(),
            Self::terminal_system(),
            Self::update_system(),
            Self::learn_system(),
        )
            .chain()
    }
}
