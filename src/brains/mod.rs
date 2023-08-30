use bevy::prelude::*;
use bevy_prng::ChaCha8Rng;
use bevy_rand::prelude::EntropyComponent;

use self::{
    replay_buffer::{PpoBuffer, PpoMetadata},
    thinkers::{ppo::PpoThinker, Status, Thinker},
};
use crate::{
    envs::{Action, Env},
    FrameStack, TbWriter, Timestamp,
};

pub mod replay_buffer;
pub mod thinkers;

#[derive(Component)]
pub struct Brain<E: Env, T: Thinker<E>> {
    pub name: String,
    pub timestamp: Timestamp,
    pub last_action: E::Action,
    pub thinker: T,
    pub writer: TbWriter,
    pub metadata: T::Metadata,
}

impl<E: Env, T: Thinker<E>> Brain<E, T> {
    pub fn new(thinker: T, name: String, timestamp: Timestamp) -> Self {
        let mut writer = TbWriter::default();
        writer.init(Some(name.as_str()), &timestamp);
        Self {
            metadata: thinker.init_metadata(1),
            timestamp: timestamp.to_owned(),
            last_action: E::Action::default(),
            thinker,
            writer,
            name,
        }
    }

    pub fn save(&self) -> Result<(), Box<dyn std::error::Error>> {
        let path = format!("training/{}/{}", self.timestamp, self.name);
        std::fs::create_dir_all(&path).ok();
        self.thinker.save(path)?;
        Ok(())
    }
}

impl<E: Env, T: Thinker<E>> Brain<E, T> {
    pub fn act(
        &mut self,
        obs: &FrameStack<E::Observation>,
        params: &E::Params,
        rng: &mut EntropyComponent<ChaCha8Rng>,
    ) -> Option<E::Action> {
        let action = self.thinker.act(obs, &mut self.metadata, params, rng);
        if let Some(ref action) = action {
            self.last_action = action.clone();
        }
        // self.writer
        //     .add_scalar("Entropy", self.thinker.recent_entropy, frame_count);
        action
    }

    pub fn learn(
        &mut self,
        frame_count: usize,
        rb: &mut PpoBuffer<E>,
        params: &E::Params,
        rng: &mut EntropyComponent<ChaCha8Rng>,
    ) where
        E::Action: Action<E, Metadata = PpoMetadata>,
    {
        self.thinker.learn(rb, params, rng);
        let status = self.thinker.status();
        status.log(&mut self.writer, frame_count);
    }
}

pub type AgentThinker = PpoThinker;

#[derive(Debug)]
pub struct ThinkerStatus<E: Env, T: Thinker<E>> {
    pub last_action: Option<E::Action>,
    pub status: Option<T::Status>,
    pub meta: Option<T::Metadata>,
}

impl<E: Env, T: Thinker<E>> Default for ThinkerStatus<E, T>
where
    T::Status: Default,
{
    fn default() -> Self {
        Self {
            last_action: Some(E::Action::default()),
            status: Some(T::Status::default()),
            meta: None,
        }
    }
}

impl<E: Env, T: Thinker<E>> Clone for ThinkerStatus<E, T>
where
    T::Status: Clone,
{
    fn clone(&self) -> Self {
        Self {
            last_action: self.last_action.clone(),
            status: self.status.clone(),
            meta: self.meta.clone(),
        }
    }
}
