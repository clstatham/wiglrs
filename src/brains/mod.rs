use bevy::prelude::Resource;

use crate::envs::Action;

use self::{
    learners::maddpg::Maddpg,
    models::{
        deterministic_mlp::DeterministicMlpActor,
        linear_resnet::{LinResActor, LinResCritic},
        CriticWithTarget, Policy, PolicyWithTarget, ValueEstimator,
    },
};

pub mod learners;
pub mod models;

pub type AgentPolicy = PolicyWithTarget<DeterministicMlpActor>;
pub type AgentValue = CriticWithTarget<LinResCritic>;
pub type AgentLearner<E> = Maddpg<E>;

#[derive(Resource)]
pub struct Policies<P: Policy>(pub Vec<P>);

#[derive(Resource)]
pub struct Actions<A: Action>(pub Vec<A>);

#[derive(Resource)]
pub struct ValueEstimators<V: ValueEstimator>(pub Vec<V>);
