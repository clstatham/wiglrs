use bevy::prelude::{Component, Resource};
use bevy_prng::ChaCha8Rng;
use bevy_rand::prelude::EntropyComponent;
use candle_core::Device;

use crate::{
    envs::{Env, StepMetadata},
    FrameStack, TbWriter,
};

use super::models::{Policy, ValueEstimator};

pub mod maddpg;
// pub mod coma;
// pub mod ppo;
pub mod utils;

lazy_static::lazy_static! {
    pub static ref DEVICE: Device = Device::Cpu;
}

pub trait Status {
    fn log(&self, writer: &mut TbWriter, step: usize);
}

impl Status for () {
    fn log(&self, _writer: &mut TbWriter, _step: usize) {}
}

pub trait Learner<E: Env, P: Policy, V: ValueEstimator>: Resource {
    type Buffer: Buffer<E>;
    type Status: Status + Clone + Default;

    fn learn(&mut self, policies: &[P], values: &[V]);
    fn status(&self) -> Self::Status;
}

#[derive(Clone)]
pub struct Sart<E: Env, M: StepMetadata> {
    pub obs: FrameStack<Box<[f32]>>,
    pub action: E::Action,
    pub reward: f32,
    pub terminal: bool,
    pub metadata: M,
}

impl<E: Env, M: StepMetadata> Sart<E, M> {
    pub fn unzip(self) -> (FrameStack<Box<[f32]>>, E::Action, f32, bool, M) {
        (
            self.obs,
            self.action,
            self.reward,
            self.terminal,
            self.metadata,
        )
    }
}

pub trait Buffer<E: Env>: Clone + Resource {
    type Metadata: StepMetadata;
    fn remember_sart(&mut self, step: Sart<E, Self::Metadata>);
}

pub trait OnPolicyBuffer<E: Env>: Buffer<E> {
    fn finish_trajectory(&mut self, final_val: Option<f32>);
    fn shuffled_and_batched(&self, batch_size: usize) -> Vec<Self>;
}

pub trait OffPolicyBuffer<E: Env>: Buffer<E> {
    fn sample(&self, batch_size: usize) -> Self;
}
