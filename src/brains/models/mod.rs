use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard};

use bevy::prelude::Component;
use candle_core::{backprop::GradStore, Result, Tensor};

pub mod deterministic_mlp;
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
    fn estimate_value(&self, obs: &Tensor, action: Option<&Tensor>) -> Result<Tensor>;
    fn apply_gradients(&self, grads: &GradStore) -> Result<()>;
}

#[derive(Component)]
pub struct CentralizedCritic<C: ValueEstimator> {
    critic: Arc<RwLock<C>>,
}

impl<C: ValueEstimator> CentralizedCritic<C> {
    pub fn new(critic: C) -> Self {
        Self {
            critic: Arc::new(RwLock::new(critic)),
        }
    }

    pub fn get_mut(&self) -> RwLockWriteGuard<'_, C> {
        self.critic.write().unwrap()
    }

    pub fn get(&self) -> RwLockReadGuard<'_, C> {
        self.critic.read().unwrap()
    }
}

impl<C: ValueEstimator> Clone for CentralizedCritic<C> {
    fn clone(&self) -> Self {
        Self {
            critic: self.critic.clone(),
        }
    }
}

impl<C: ValueEstimator> ValueEstimator for CentralizedCritic<C> {
    fn estimate_value(&self, obs: &Tensor, action: Option<&Tensor>) -> Result<Tensor> {
        self.get().estimate_value(obs, action)
    }

    fn apply_gradients(&self, grads: &GradStore) -> Result<()> {
        self.get().apply_gradients(grads)
    }
}

pub trait CopyWeights {
    fn soft_update(&self, other: &Self, tau: f32);
    fn hard_update(&self, other: &Self);
}

#[derive(Component)]
pub struct PolicyWithTarget<P: Policy + CopyWeights> {
    pub policy: P,
    pub target_policy: P,
}

impl<P: Policy + CopyWeights> PolicyWithTarget<P> {
    pub fn soft_update(&self, tau: f32) {
        self.target_policy.soft_update(&self.policy, tau);
    }
}

impl<P: Policy + CopyWeights> Policy for PolicyWithTarget<P> {
    type Logits = P::Logits;

    type Status = P::Status;

    fn action_logits(&self, obs: &Tensor) -> Result<Self::Logits> {
        self.policy.action_logits(obs)
    }

    fn act(&self, obs: &Tensor) -> Result<(Tensor, Self::Logits)> {
        self.policy.act(obs)
    }

    fn log_prob(&self, logits: &Self::Logits, action: &Tensor) -> Result<Tensor> {
        self.policy.log_prob(logits, action)
    }

    fn entropy(&self, logits: &Self::Logits) -> Result<Tensor> {
        self.policy.entropy(logits)
    }

    fn apply_gradients(&self, grads: &GradStore) -> Result<()> {
        self.policy.apply_gradients(grads)
    }

    fn status(&self) -> Option<Self::Status> {
        self.policy.status()
    }
}

#[derive(Component)]
pub struct CriticWithTarget<V: ValueEstimator + CopyWeights> {
    pub critic: V,
    pub target_critic: V,
}

impl<V: ValueEstimator + CopyWeights> ValueEstimator for CriticWithTarget<V> {
    fn estimate_value(&self, obs: &Tensor, action: Option<&Tensor>) -> Result<Tensor> {
        self.critic.estimate_value(obs, action)
    }

    fn apply_gradients(&self, grads: &GradStore) -> Result<()> {
        self.critic.apply_gradients(grads)
    }
}

impl<V: ValueEstimator + CopyWeights> CriticWithTarget<V> {
    pub fn soft_update(&self, tau: f32) {
        self.target_critic.soft_update(&self.critic, tau);
    }
}
