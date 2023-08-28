use std::{
    path::Path,
    sync::{Arc, Mutex, MutexGuard},
};

use bevy::prelude::Vec2;

use crate::{Action, TbWriter};

use super::{replay_buffer::PpoBuffer, FrameStack};

pub mod ncp;
pub mod ppo;
pub mod stats;

pub trait Status {
    fn log(&self, writer: &mut TbWriter, step: usize);
}

impl Status for () {
    fn log(&self, _writer: &mut TbWriter, _step: usize) {}
}

pub trait Thinker {
    type Metadata: Clone;
    type Status: Status + Clone + Default;
    fn act(&mut self, obs: FrameStack, metadata: &mut Self::Metadata) -> Action;
    fn learn(&mut self, b: &PpoBuffer);
    fn save(&self, path: impl AsRef<Path>) -> Result<(), Box<dyn std::error::Error>>;
    fn init_metadata(&self, batch_size: usize) -> Self::Metadata;
    fn status(&self) -> Self::Status;
}

pub struct RandomThinker;

impl Thinker for RandomThinker {
    type Metadata = ();
    type Status = ();
    fn act(&mut self, _obs: FrameStack, _metadata: &mut ()) -> Action {
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
    fn learn(&mut self, _b: &PpoBuffer) {}
    fn save(&self, _path: impl AsRef<Path>) -> Result<(), Box<dyn std::error::Error>> {
        Ok(())
    }
    fn init_metadata(&self, _batch_size: usize) -> Self::Metadata {}
    fn status(&self) -> Self::Status {}
}

pub struct SharedThinker<T: Thinker> {
    thinker: Arc<Mutex<T>>,
}

impl<T: Thinker> Default for SharedThinker<T>
where
    T: Default,
{
    fn default() -> Self {
        Self::new(T::default())
    }
}

impl<T: Thinker> Clone for SharedThinker<T> {
    fn clone(&self) -> Self {
        Self {
            thinker: self.thinker.clone(),
        }
    }
}

impl<T: Thinker> SharedThinker<T> {
    pub fn new(thinker: T) -> Self {
        Self {
            thinker: Arc::new(Mutex::new(thinker)),
        }
    }

    pub fn lock(&self) -> MutexGuard<'_, T> {
        self.thinker.lock().unwrap()
    }
}

impl<T: Thinker> Thinker for SharedThinker<T> {
    type Metadata = T::Metadata;
    type Status = T::Status;
    fn act(&mut self, obs: FrameStack, metadata: &mut Self::Metadata) -> Action {
        self.lock().act(obs, metadata)
    }
    fn learn(&mut self, b: &PpoBuffer) {
        self.lock().learn(b)
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
