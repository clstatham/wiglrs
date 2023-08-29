use bevy::{ecs::schedule::SystemConfigs, prelude::*};

use crate::FrameStack;

use self::maps::Map;

pub mod ffa;
pub mod maps;
pub mod tdm;

pub trait Action<E: Env> {
    type Metadata: Default;
    fn as_vec(&self, params: &E::Params) -> Vec<f32>;
    fn from_slice(v: &[f32], metadata: Self::Metadata, params: &E::Params) -> Self;
    fn metadata(&self) -> Self::Metadata;
}

pub trait Observation<E: Env>
where
    Self: Sized,
{
    fn as_vec(&self, params: &E::Params) -> Vec<f32>;
    fn new_frame_stack(params: &E::Params) -> FrameStack<Self>;
}

pub trait Env: Resource
where
    Self: Sized,
{
    type Params: Default + Resource + Send + Sync;
    type Observation: Observation<Self> + Send + Sync + Clone;
    type Action: Action<Self> + Default + Send + Sync + Clone;

    fn init() -> Self;

    fn setup_system<M: Map>() -> SystemConfigs;
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
