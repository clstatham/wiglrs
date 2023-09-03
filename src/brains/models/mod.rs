use std::{ops::Deref, sync::Arc};

use bevy::prelude::Component;
use candle_core::{backprop::GradStore, Result, Tensor};

pub mod linear_resnet;

pub trait Policy: Component {
    type Logits;
    type Status: Clone;
    fn action_logits(&self, obs: &Tensor) -> Result<Self::Logits>;
    fn act(&self, obs: &Tensor) -> Result<(Tensor, Self::Logits)>;
    fn log_prob(&self, logits: &Self::Logits, action: &Tensor) -> Result<Tensor>;
    fn entropy(&self, logits: &Self::Logits) -> Result<Tensor>;
    fn apply_gradients(&self, grads: &GradStore) -> Result<()>;
    fn status(&self) -> Option<Self::Status>;
}

pub trait ValueEstimator: Component {
    fn estimate_value(&self, obs: &Tensor) -> Result<Tensor>;
    fn apply_gradients(&self, grads: &GradStore) -> Result<()>;
}

#[derive(Component)]
pub struct CentralizedCritic<C: ValueEstimator> {
    critic: Arc<C>,
}

impl<C: ValueEstimator> Clone for CentralizedCritic<C> {
    fn clone(&self) -> Self {
        Self {
            critic: self.critic.clone(),
        }
    }
}

impl<C: ValueEstimator> Deref for CentralizedCritic<C> {
    type Target = C;
    fn deref(&self) -> &Self::Target {
        self.critic.as_ref()
    }
}

impl<C: ValueEstimator> ValueEstimator for CentralizedCritic<C> {
    fn estimate_value(&self, obs: &Tensor) -> Result<Tensor> {
        self.critic.estimate_value(obs)
    }

    fn apply_gradients(&self, grads: &GradStore) -> Result<()> {
        self.critic.apply_gradients(grads)
    }
}
