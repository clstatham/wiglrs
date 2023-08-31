use std::{
    fs::File,
    io::{Read, Write},
    path::Path,
};

use crate::FrameStack;
use bevy::{ecs::schedule::SystemConfigs, prelude::*};
use burn_tensor::{backend::Backend, Tensor};
use serde::{de::DeserializeOwned, Deserialize, Serialize};

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
    fn as_slice(&self) -> Box<[f32]>;
    fn as_tensor<B: Backend>(&self) -> Tensor<B, 1> {
        Tensor::from_floats(self.as_slice().to_vec().as_slice())
    }
}

pub trait DefaultFrameStack<E: Env + ?Sized>: Observation {
    fn default_frame_stack(params: &E::Params) -> FrameStack<Self>;
}

pub trait Params {
    fn agent_radius(&self) -> f32;
    fn agent_max_health(&self) -> f32;
    fn num_agents(&self) -> usize;
    fn agent_frame_stack_len(&self) -> usize;
    fn to_yaml(&self) -> Result<String, Box<dyn std::error::Error>>
    where
        Self: Serialize,
    {
        let s = serde_yaml::to_string(self)?;
        Ok(s)
    }
    fn to_yaml_file(&self, path: impl AsRef<Path>) -> Result<(), Box<dyn std::error::Error>>
    where
        Self: Serialize,
    {
        let mut f = File::create(path)?;
        let s = self.to_yaml()?;
        write!(f, "{}", s)?;
        Ok(())
    }
    fn from_yaml<'a>(json: &'a str) -> Result<Self, Box<dyn std::error::Error>>
    where
        Self: Deserialize<'a>,
    {
        let this = serde_yaml::from_str(json)?;
        Ok(this)
    }
    fn from_yaml_file(path: impl AsRef<Path>) -> Result<Self, Box<dyn std::error::Error>>
    where
        Self: DeserializeOwned,
    {
        let mut f = File::open(path)?;
        let mut s = String::new();
        f.read_to_string(&mut s)?;
        let this = Self::from_yaml(s.as_str())?;
        Ok(this)
    }
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
