use std::sync::Mutex;

use bevy::prelude::Component;
use candle_core::{backprop::GradStore, Module, Result, Tensor};
use candle_nn::{AdamW, Linear, Optimizer, VarBuilder, VarMap};

use crate::{
    brains::learners::{
        utils::{linear, MvNormal, ResBlock},
        DEVICE,
    },
    envs::Action,
    FrameStack,
};

use super::{Policy, ValueEstimator};

#[derive(Component)]
pub struct LinResActor {
    common1: Linear,
    common2: ResBlock,
    common3: ResBlock,
    mu_head: Linear,
    cov_head: Linear,
    varmap: VarMap,
    optim: Mutex<AdamW>,
}

impl LinResActor {
    pub fn new(obs_len: usize, hidden_len: usize, action_len: usize, lr: f64) -> Result<Self> {
        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &DEVICE);
        let common1 = linear(obs_len, hidden_len, 5. / 3., vs.pp("common1"));
        let common2 = ResBlock::new(hidden_len, hidden_len / 2, vs.pp("common2"));
        let common3 = ResBlock::new(hidden_len, hidden_len / 2, vs.pp("common3"));
        let mu_head = linear(hidden_len, action_len, 0.01, vs.pp("mu_head"));
        let cov_head = linear(hidden_len, action_len, 0.01, vs.pp("cov_head"));
        let optim = AdamW::new_lr(varmap.all_vars(), lr)?;
        Ok(Self {
            varmap,
            common1,
            common2,
            common3,
            mu_head,
            cov_head,
            optim: Mutex::new(optim),
        })
    }
}

impl Policy for LinResActor {
    type Logits = (Tensor, Tensor);

    fn action_logits(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        let x = x.flatten(1, 2)?;

        let x = self.common1.forward(&x)?.tanh()?;
        let x = self.common2.forward(&x)?;
        let x = self.common3.forward(&x)?;

        let mu = self.mu_head.forward(&x)?.tanh()?;

        let cov = self.cov_head.forward(&x)?.exp()?;
        // let cov = (self.cov_head.exp()? + 1.0)?
        //     .log()?
        //     .repeat(mu.shape().dims()[0])?;
        // let cov = (cov.exp()? + 1.0)?.log()?;

        Ok((mu, cov))
    }

    fn act(&self, obs: &Tensor) -> Result<(Tensor, Self::Logits)> {
        let (mu, cov) = self.action_logits(obs)?;
        let dist = MvNormal {
            mu: mu.clone(),
            cov: cov.clone(),
        };
        let action = dist.sample()?;
        Ok((action, (mu, cov)))
    }

    fn log_prob(&self, logits: &Self::Logits, action: &Tensor) -> Result<Tensor> {
        let (mu, cov) = logits.to_owned();
        let dist = MvNormal { mu, cov };
        Ok(dist.log_prob(action))
    }

    fn entropy(&self, logits: &Self::Logits) -> Result<Tensor> {
        let (mu, cov) = logits.to_owned();
        let dist = MvNormal { mu, cov };
        dist.entropy()
    }

    fn apply_gradients(&self, grads: &GradStore) -> Result<()> {
        self.optim.lock().unwrap().step(grads)
    }
}

#[derive(Component)]
pub struct LinResCritic {
    l1: Linear,
    l2: ResBlock,
    l3: ResBlock,
    head: Linear,
    varmap: VarMap,
    optim: Mutex<AdamW>,
}

impl LinResCritic {
    pub fn new(obs_len: usize, hidden_len: usize, lr: f64) -> Result<Self> {
        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &DEVICE);
        let l1 = linear(obs_len, hidden_len, 5. / 3., vs.pp("l1"));
        let l2 = ResBlock::new(hidden_len, hidden_len / 2, vs.pp("l2"));
        let l3 = ResBlock::new(hidden_len, hidden_len / 2, vs.pp("l3"));
        let head = linear(hidden_len, 1, 1.0, vs.pp("head"));
        let optim = AdamW::new_lr(varmap.all_vars(), lr)?;
        Ok(Self {
            varmap,
            l1,
            l2,
            l3,
            head,
            optim: Mutex::new(optim),
        })
    }
}
impl ValueEstimator for LinResCritic {
    fn estimate_value(&self, x: &Tensor) -> Result<Tensor> {
        let x = x.flatten(1, 2)?;
        let x = self.l1.forward(&x)?.tanh()?;
        let x = self.l2.forward(&x)?;
        let x = self.l3.forward(&x)?;

        let x = self.head.forward(&x)?.squeeze(1)?;
        Ok(x)
    }

    fn apply_gradients(&self, grads: &GradStore) -> Result<()> {
        self.optim.lock().unwrap().step(grads)
    }
}
