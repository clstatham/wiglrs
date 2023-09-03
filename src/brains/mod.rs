use self::{
    learners::ppo::Ppo,
    models::linear_resnet::{LinResActor, LinResCritic},
};

pub mod learners;
pub mod models;

pub type AgentPolicy = LinResActor;
pub type AgentValue = LinResCritic;
pub type AgentLearner = Ppo;
