use std::{
    marker::PhantomData,
    path::Path,
    sync::{Arc, Mutex, MutexGuard},
};

use bevy_prng::ChaCha8Rng;
use bevy_rand::prelude::EntropyComponent;
use rand_distr::Distribution;

use crate::{
    envs::{Action, Env},
    FrameStack, TbWriter,
};

use super::replay_buffer::{PpoBuffer, PpoMetadata};

// pub mod linalg;
// pub mod ncp;
pub mod ppo;

pub trait Status {
    fn log(&self, writer: &mut TbWriter, step: usize);
}

impl Status for () {
    fn log(&self, _writer: &mut TbWriter, _step: usize) {}
}

pub trait Thinker<E: Env>: Send + Sync + 'static {
    type Metadata: Clone + Send + Sync;
    type Status: Status + Clone + Default + Send + Sync;
    type ActionMetadata: Clone + Default + Send + Sync;
    fn act(
        &mut self,
        obs: &FrameStack<Box<[f32]>>,
        metadata: &mut Self::Metadata,
        params: &E::Params,
        rng: &mut EntropyComponent<ChaCha8Rng>,
    ) -> E::Action;
    fn learn(
        &mut self,
        b: &mut PpoBuffer<E>,
        params: &E::Params,
        rng: &mut EntropyComponent<ChaCha8Rng>,
    ) where
        E::Action: Action<E, Metadata = PpoMetadata>;
    fn save(&self, path: impl AsRef<Path>) -> Result<(), Box<dyn std::error::Error>>;
    fn init_metadata(&self, batch_size: usize) -> Self::Metadata;
    fn status(&self) -> Self::Status;
}

pub struct SharedThinker<E: Env, T: Thinker<E>> {
    thinker: Arc<Mutex<T>>,
    _e: PhantomData<E>,
}

impl<E: Env, T: Thinker<E>> Default for SharedThinker<E, T>
where
    T: Default,
{
    fn default() -> Self {
        Self::new(T::default())
    }
}

impl<E: Env, T: Thinker<E>> Clone for SharedThinker<E, T> {
    fn clone(&self) -> Self {
        Self {
            thinker: self.thinker.clone(),
            _e: PhantomData,
        }
    }
}

impl<E: Env, T: Thinker<E>> SharedThinker<E, T> {
    pub fn new(thinker: T) -> Self {
        Self {
            thinker: Arc::new(Mutex::new(thinker)),
            _e: PhantomData,
        }
    }

    pub fn lock(&self) -> MutexGuard<'_, T> {
        self.thinker.lock().unwrap()
    }
}

impl<E: Env, T: Thinker<E>> Thinker<E> for SharedThinker<E, T> {
    type Metadata = T::Metadata;
    type ActionMetadata = T::ActionMetadata;
    type Status = T::Status;
    fn act(
        &mut self,
        obs: &FrameStack<Box<[f32]>>,
        metadata: &mut Self::Metadata,
        params: &E::Params,
        rng: &mut EntropyComponent<ChaCha8Rng>,
    ) -> E::Action {
        self.lock().act(obs, metadata, params, rng)
    }
    fn learn(
        &mut self,
        b: &mut PpoBuffer<E>,
        params: &E::Params,
        rng: &mut EntropyComponent<ChaCha8Rng>,
    ) where
        E::Action: Action<E, Metadata = PpoMetadata>,
    {
        self.lock().learn(b, params, rng)
    }
    fn save(&self, path: impl AsRef<Path>) -> Result<(), Box<dyn std::error::Error>> {
        self.lock().save(path)
    }
    fn init_metadata(&self, batch_size: usize) -> Self::Metadata {
        self.lock().init_metadata(batch_size)
    }
    fn status(&self) -> Self::Status {
        self.lock().status()
    }
}

pub struct RandomThinker;

impl<E: Env> Thinker<E> for RandomThinker {
    type Metadata = ();
    type ActionMetadata = ();
    type Status = ();

    fn act(
        &mut self,
        _obs: &FrameStack<Box<[f32]>>,
        _metadata: &mut Self::Metadata,
        params: &<E as Env>::Params,
        rng: &mut EntropyComponent<ChaCha8Rng>,
    ) -> <E as Env>::Action {
        let len = E::Action::default().as_slice(params).len();
        let dist = rand_distr::Uniform::new(-1.0, 1.0);
        let mut out = vec![];
        for _ in 0..len {
            out.push(dist.sample(rng));
        }
        E::Action::from_slice(&out, <E::Action as Action<E>>::Metadata::default(), params)
    }

    fn learn(
        &mut self,
        _b: &mut PpoBuffer<E>,
        _params: &<E as Env>::Params,
        _rng: &mut EntropyComponent<ChaCha8Rng>,
    ) where
        E::Action: Action<E, Metadata = PpoMetadata>,
    {
    }

    fn save(&self, _path: impl AsRef<Path>) -> Result<(), Box<dyn std::error::Error>> {
        Ok(())
    }

    fn init_metadata(&self, _batch_size: usize) -> Self::Metadata {}

    fn status(&self) -> Self::Status {}
}
