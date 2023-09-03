use std::{
    marker::PhantomData,
    path::Path,
    sync::{Arc, Mutex, MutexGuard},
};

use bevy::prelude::Component;
use bevy_prng::ChaCha8Rng;
use bevy_rand::prelude::EntropyComponent;
use candle_core::Device;
use rand_distr::Distribution;

use crate::{
    envs::{Action, Env, StepMetadata},
    FrameStack, TbWriter,
};

use super::models::{Policy, ValueEstimator};

pub mod ppo;
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

pub trait Learner<E: Env>: Component {
    type Buffer: Buffer<E>;
    type Status: Status + Clone + Default + Component;

    fn learn<P: Policy, V: ValueEstimator>(
        &mut self,
        policy: &P,
        value: &V,
        buffer: &Self::Buffer,
        rng: &mut EntropyComponent<ChaCha8Rng>,
    );
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

pub trait Buffer<E: Env>: Clone + Component {
    type Metadata: StepMetadata;
    fn remember_sart(&mut self, step: Sart<E, Self::Metadata>);
    fn finish_trajectory(&mut self, final_val: Option<f32>);
    fn shuffled_and_batched(
        &self,
        batch_size: usize,
        rng: &mut EntropyComponent<ChaCha8Rng>,
    ) -> Vec<Self>;
}
