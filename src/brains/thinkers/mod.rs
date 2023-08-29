use std::{
    marker::PhantomData,
    path::Path,
    sync::{Arc, Mutex, MutexGuard},
};



use crate::{envs::Env, FrameStack, TbWriter};

use super::replay_buffer::PpoBuffer;

pub mod ncp;
pub mod ppo;
pub mod stats;

pub trait Status {
    fn log(&self, writer: &mut TbWriter, step: usize);
}

impl Status for () {
    fn log(&self, _writer: &mut TbWriter, _step: usize) {}
}

pub trait Thinker<E: Env> {
    type Metadata: Clone;
    type Status: Status + Clone + Default;
    fn act(
        &mut self,
        obs: &FrameStack<E::Observation>,
        metadata: &mut Self::Metadata,
        params: &E::Params,
    ) -> Option<E::Action>;
    fn learn(&mut self, b: &PpoBuffer<E>, params: &E::Params);
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
    type Status = T::Status;
    fn act(
        &mut self,
        obs: &FrameStack<E::Observation>,
        metadata: &mut Self::Metadata,
        params: &E::Params,
    ) -> Option<E::Action> {
        self.lock().act(obs, metadata, params)
    }
    fn learn(&mut self, b: &PpoBuffer<E>, params: &E::Params) {
        self.lock().learn(b, params)
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
