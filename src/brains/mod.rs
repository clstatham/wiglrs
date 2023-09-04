use bevy::prelude::Resource;

use self::{
    learners::maddpg::Maddpg,
    models::{
        deterministic_mlp::{DeterministicMlpActor, DeterministicMlpCritic},
        linear_resnet::{LinResActor, LinResCritic},
        CentralizedCritic, CompoundPolicy, CriticWithTarget, Policy, PolicyWithTarget,
        ValueEstimator,
    },
};

pub mod learners;
pub mod models;

pub type AgentPolicy = PolicyWithTarget<DeterministicMlpActor>;
pub type AgentValue = CriticWithTarget<DeterministicMlpCritic>;
pub type AgentLearner<E> = Maddpg<E>;

#[derive(Resource)]
pub struct Policies<P: Policy>(pub Vec<P>);

#[derive(Resource)]
pub struct ValueEstimators<V: ValueEstimator>(pub Vec<V>);
