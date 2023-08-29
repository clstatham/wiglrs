use bevy::{ecs::schedule::SystemConfigs, prelude::*};

pub mod tdm;

pub trait Map: Resource {
    fn setup_system() -> SystemConfigs;
}
