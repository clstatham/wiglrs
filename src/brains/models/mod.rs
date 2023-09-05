use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard};

use bevy::prelude::Component;
use candle_core::{backprop::GradStore, Result, Tensor};
use candle_nn::VarMap;

pub mod deterministic_mlp;
pub mod linear_resnet;

pub trait Policy: Send + Sync + 'static {
    type Logits;
    type Status: Clone;
    fn action_logits(&self, obs: &Tensor) -> Result<Self::Logits>;
    fn act(&self, obs: &Tensor) -> Result<(Tensor, Self::Logits)>;
    fn log_prob(&self, logits: &Self::Logits, action: &Tensor) -> Result<Tensor>;
    fn entropy(&self, logits: &Self::Logits) -> Result<Tensor>;
    fn apply_gradients(&self, grads: &GradStore) -> Result<()>;
    fn status(&self) -> Option<Self::Status>;
    fn varmap(&self) -> &VarMap;

    fn soft_update(&self, other: &Self, tau: f32) {
        for (varname, my_var) in self.varmap().data().lock().unwrap().iter() {
            let other_var = &other.varmap().data().lock().unwrap()[varname];
            let new_var = my_var
                .affine(1.0 - tau as f64, 0.0)
                .unwrap()
                .add(&other_var.affine(tau as f64, 0.0).unwrap())
                .unwrap();
            my_var.set(&new_var).unwrap();
        }
    }

    fn hard_update(&self, other: &Self) {
        for (varname, my_var) in self.varmap().data().lock().unwrap().iter() {
            my_var
                .set(&other.varmap().data().lock().unwrap()[varname])
                .unwrap();
        }
    }
}

pub trait ValueEstimator: Send + Sync + 'static {
    fn estimate_value(&self, obs: &Tensor, action: Option<&Tensor>) -> Result<Tensor>;
    fn apply_gradients(&self, grads: &GradStore) -> Result<()>;
    fn varmap(&self) -> &VarMap;

    fn soft_update(&self, other: &Self, tau: f32) {
        for (varname, my_var) in self.varmap().data().lock().unwrap().iter() {
            let other_var = &other.varmap().data().lock().unwrap()[varname];
            let new_var = my_var
                .affine(1.0 - tau as f64, 0.0)
                .unwrap()
                .add(&other_var.affine(tau as f64, 0.0).unwrap())
                .unwrap();
            my_var.set(&new_var).unwrap();
        }
    }

    fn hard_update(&self, other: &Self) {
        for (varname, my_var) in self.varmap().data().lock().unwrap().iter() {
            my_var
                .set(&other.varmap().data().lock().unwrap()[varname])
                .unwrap();
        }
    }
}

#[derive(Component)]
pub struct PolicyWithTarget<P: Policy> {
    pub policy: P,
    pub target_policy: P,
}

impl<P: Policy> PolicyWithTarget<P> {
    pub fn soft_update(&self, tau: f32) {
        self.target_policy.soft_update(&self.policy, tau);
    }
}

impl<P: Policy> Policy for PolicyWithTarget<P> {
    type Logits = P::Logits;

    type Status = P::Status;

    fn varmap(&self) -> &VarMap {
        unimplemented!()
    }

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
pub struct CriticWithTarget<V: ValueEstimator> {
    pub critic: V,
    pub target_critic: V,
}

impl<V: ValueEstimator> ValueEstimator for CriticWithTarget<V> {
    fn varmap(&self) -> &VarMap {
        unimplemented!()
    }

    fn estimate_value(&self, obs: &Tensor, action: Option<&Tensor>) -> Result<Tensor> {
        self.critic.estimate_value(obs, action)
    }

    fn apply_gradients(&self, grads: &GradStore) -> Result<()> {
        self.critic.apply_gradients(grads)
    }
}

impl<V: ValueEstimator> CriticWithTarget<V> {
    pub fn soft_update(&self, tau: f32) {
        self.target_critic.soft_update(&self.critic, tau);
    }
}
