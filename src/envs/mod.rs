use bevy::prelude::*;

pub trait Env {
    type SetupResources;
    type RewardResources;

    fn setup(
        &mut self,
        commands: Commands,
        asset_server: Res<AssetServer>,
        resources: Self::SetupResources,
    );
    fn reward(
        &mut self,
        for_ent: Entity,
        commands: Commands,
        asset_server: Res<AssetServer>,
        resources: Self::RewardResources,
    ) -> f32;
}

pub struct Ffa {
    pub n_players: usize,
}

impl Env for Ffa {
    type SetupResources = ();
    fn setup(
        &mut self,
        _commands: Commands,
        _asset_server: Res<AssetServer>,
        _resources: Self::SetupResources,
    ) {
    }

    type RewardResources = ();
    fn reward(
        &mut self,
        _for_ent: Entity,
        _commands: Commands,
        _asset_server: Res<AssetServer>,
        _resources: Self::RewardResources,
    ) -> f32 {
        todo!()
    }
}
