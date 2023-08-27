use std::{
    path::Path,
    sync::{Arc, Mutex, MutexGuard},
};

use bevy::prelude::Vec2;

use crate::Action;

use super::{replay_buffer::SartAdvBuffer, FrameStack};

pub mod ncp;
pub mod ppo;
pub mod stats;

pub trait Thinker {
    type Metadata;
    fn act(&mut self, obs: FrameStack, metadata: &mut Self::Metadata) -> Action;
    fn learn(&mut self, b: &SartAdvBuffer);
    fn save(&self, path: impl AsRef<Path>) -> Result<(), Box<dyn std::error::Error>>;
    fn init_metadata(&self, batch_size: usize) -> Self::Metadata;
}

pub struct RandomThinker;

impl Thinker for RandomThinker {
    type Metadata = ();
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
    fn learn(&mut self, _b: &SartAdvBuffer) {}
    fn save(&self, _path: impl AsRef<Path>) -> Result<(), Box<dyn std::error::Error>> {
        Ok(())
    }
    fn init_metadata(&self, _batch_size: usize) -> Self::Metadata {}
}

pub struct SharedThinker<T: Thinker> {
    thinker: Arc<Mutex<T>>,
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
        self.thinker.try_lock().unwrap()
    }
}

impl<T: Thinker> Thinker for SharedThinker<T> {
    type Metadata = T::Metadata;
    fn act(&mut self, obs: FrameStack, metadata: &mut Self::Metadata) -> Action {
        self.lock().act(obs, metadata)
    }
    fn learn(&mut self, b: &SartAdvBuffer) {
        self.lock().learn(b)
    }
    fn save(&self, path: impl AsRef<Path>) -> Result<(), Box<dyn std::error::Error>> {
        self.lock().save(path)
    }
    fn init_metadata(&self, batch_size: usize) -> Self::Metadata {
        self.lock().init_metadata(batch_size)
    }
}
