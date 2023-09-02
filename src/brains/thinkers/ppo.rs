#![allow(clippy::single_range_in_vec_init)]

use bevy::prelude::Component;
use bevy_prng::ChaCha8Rng;
use bevy_rand::prelude::EntropyComponent;

use itertools::Itertools;
use rand_distr::Distribution;
use std::f32::consts::PI;

use crate::{
    brains::replay_buffer::{PpoBuffer, PpoMetadata},
    envs::Params,
};
use crate::{
    envs::{Action, Env},
    FrameStack,
};

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{AdamW, Linear, Module, Optimizer, VarBuilder, VarMap};

use super::{Status, Thinker};

lazy_static::lazy_static! {
    pub static ref DEVICE: Device = Device::Cpu;
}

pub struct MvNormal {
    pub mu: Tensor,
    pub cov: Tensor,
}

impl MvNormal {
    pub fn sample(&self, rng: &mut EntropyComponent<ChaCha8Rng>) -> Result<Tensor> {
        let elems = self.mu.shape().elem_count();
        let dist = rand_distr::StandardNormal;
        let samples = (0..elems)
            .map(|_| Distribution::<f32>::sample(&dist, rng))
            .collect_vec();
        let z = Tensor::from_vec(samples, self.mu.shape(), self.mu.device())?;

        z.broadcast_mul(&self.cov)?.broadcast_add(&self.mu)
    }

    // https://online.stat.psu.edu/stat505/book/export/html/636
    pub fn log_prob(&self, x: &Tensor) -> Tensor {
        let _nbatch = self.mu.shape().dims()[0];
        let x = x.to_device(self.mu.device()).unwrap();
        let first = self
            .cov
            .ones_like()
            .unwrap()
            .broadcast_div(
                &self
                    .cov
                    .broadcast_mul(&Tensor::new(2.0 * PI, &DEVICE).unwrap())
                    .unwrap(),
            )
            .unwrap()
            .sqrt()
            .unwrap();
        let second = {
            let a = self
                .cov
                .ones_like()
                .unwrap()
                .neg()
                .unwrap()
                .div(
                    &self
                        .cov
                        .broadcast_mul(&Tensor::new(2.0f32, &DEVICE).unwrap())
                        .unwrap(),
                )
                .unwrap();
            let b = x.sub(&self.mu).unwrap();
            (a.mul(&b).unwrap().mul(&b).unwrap()).exp().unwrap()
        };
        let factors = first.mul(&second).unwrap();

        factors.log().unwrap().sum(1).unwrap()
    }

    pub fn entropy(&self) -> Result<Tensor> {
        // let cov = self.cov.to_data().value;
        // let [_, nfeat] = self.mu.shape().dims;
        // assert!(cov.iter().all(|f| *f > 0.0), "{:?}", cov);
        // let nbatch = self.mu.shape().dims[0];
        // let mut g = Tensor::ones([nbatch, 1]).to_device(&self.cov.device());
        // for i in 0..nfeat {
        //     g = g * self.cov.clone().slice([0..nbatch, i..i + 1]);
        // }
        // let second_term = (nfeat as f32 * 0.5) * (1.0 + (2.0 * PI).ln());
        // g.log() * 0.5 + second_term
        todo!()
    }
}

pub struct PpoActor {
    common1: Linear,
    common2: Linear,
    mu_head: Linear,
    cov_head: Linear,
}

impl PpoActor {
    pub fn forward(
        &mut self,
        x: &Tensor,
        // common_h: &mut Tensor<B, 2>,
    ) -> Result<(Tensor, Tensor)> {
        let (_, nseq, _) = x.shape().dims3().unwrap();
        let x = x
            .index_select(
                &Tensor::from_slice::<_, i64>(&[nseq as i64 - 1], (1,), &DEVICE)?,
                1,
            )?
            .squeeze(1)
            .unwrap();
        let x = self.common1.forward(&x)?.tanh()?;
        let x = self.common2.forward(&x)?.tanh()?;

        let mu = self.mu_head.forward(&x)?.tanh()?;

        let cov = self.cov_head.forward(&x)?;
        let cov = (cov.exp()? + 1.0)?.log()?;

        Ok((mu, cov))
    }
}

impl PpoActor {
    pub fn new(
        obs_len: usize,
        hidden_len: usize,
        action_len: usize,
        vs: VarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            common1: candle_nn::linear(obs_len, hidden_len, vs.pp("common1"))?,
            common2: candle_nn::linear(hidden_len, hidden_len, vs.pp("common2"))?,
            mu_head: candle_nn::linear(hidden_len, action_len, vs.pp("mu_head"))?,
            cov_head: candle_nn::linear(hidden_len, action_len, vs.pp("cov_head"))?,
        })
    }
}

pub struct PpoCritic {
    common1: Linear,
    common2: Linear,
    head: Linear,
}

impl PpoCritic {
    pub fn new(obs_len: usize, hidden_len: usize, vs: VarBuilder) -> Result<Self> {
        Ok(Self {
            common1: candle_nn::linear(obs_len, hidden_len, vs.pp("common1"))?,
            common2: candle_nn::linear(hidden_len, hidden_len, vs.pp("common2"))?,
            head: candle_nn::linear(hidden_len, 1, vs.push_prefix("head"))?,
        })
    }
}

impl PpoCritic {
    pub fn forward(&mut self, x: &Tensor) -> Result<Tensor> {
        let (_, nseq, _) = x.shape().dims3().unwrap();
        let x = x
            .index_select(
                &Tensor::from_slice::<_, i64>(&[nseq as i64 - 1], (1,), &DEVICE)?,
                1,
            )?
            .squeeze(1)
            .unwrap();
        let x = self.common1.forward(&x)?.tanh()?;
        let x = (&x + self.common2.forward(&x)?.tanh()?)?;

        let x = self.head.forward(&x)?.squeeze(1)?;
        Ok(x)
    }
}

#[derive(Debug, Clone)]
pub struct HiddenStates {
    // pub actor_com_h: Tensor<B, 2>,
    // pub critic_h: Tensor<B, 2>,
}

#[derive(Debug, Clone, Default)]
pub struct PpoStatus {
    pub mu: Box<[f32]>,
    pub cov: Box<[f32]>,
    pub recent_entropy: f32,
    pub policy_loss: f32,
    pub value_loss: f32,
    pub entropy_loss: f32,
    pub nclamp: f32,
    pub kl: f32,
    pub explained_var: f32,
    pub policy_batches_completed: f32,
}

impl Status for PpoStatus {
    fn log(&self, writer: &mut crate::TbWriter, step: usize) {
        writer.add_scalar("Policy/Loss", self.policy_loss, step);
        writer.add_scalar("Policy/Entropy", self.entropy_loss, step);
        writer.add_scalar("Policy/ClampRatio", self.nclamp, step);
        writer.add_scalar("Policy/KL", self.kl, step);
        writer.add_scalar(
            "Policy/BatchesCompleted",
            self.policy_batches_completed,
            step,
        );
        writer.add_scalar("Value/Loss", self.value_loss, step);
        writer.add_scalar("Value/ExplainedVariance", self.explained_var, step);
    }
}

pub trait PpoParams: Params {
    fn actor_lr(&self) -> f64;
    fn critic_lr(&self) -> f64;
    fn entropy_beta(&self) -> f32;
    fn training_batch_size(&self) -> usize;
    fn training_epochs(&self) -> usize;
    fn agent_rb_max_len(&self) -> usize;
    fn agent_update_interval(&self) -> usize;
}

#[derive(Debug, Component)]
pub struct RmsNormalize {
    mean: Tensor,
    var: Tensor,
    ret_mean: f32,
    ret_var: f32,
    count: f32,
    ret_count: f32,
    returns: f32,
}

impl RmsNormalize {
    pub fn new(shape: &[usize]) -> Result<Self> {
        Ok(Self {
            mean: Tensor::zeros(shape, DType::F32, &DEVICE)?,
            var: Tensor::ones(shape, DType::F32, &DEVICE)?,
            ret_mean: 0.0,
            ret_var: 0.0,
            count: 1e-8,
            ret_count: 1e-8,
            returns: 0.0,
        })
    }

    // https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/common/running_mean_std.py#L12
    fn update_obs(&mut self, x: &Tensor) -> Result<()> {
        // let batch_count = x.shape().dims()[0] as f32;
        // assert_eq!(x.shape().dims()[0], 1);
        let batch_mean = x;

        let delta = (batch_mean - &self.mean)?;
        let count = Tensor::new(self.count, &DEVICE)?;
        let tot_count = Tensor::new(1.0 + self.count, &DEVICE)?;
        self.mean = (&self.mean + delta.broadcast_div(&tot_count)?)?;
        let m_a = self.var.broadcast_mul(&count)?;
        // let m_b = batch_var * batch_count;
        let m2 = (m_a + ((&delta * &delta)?.broadcast_mul(&(&count / &tot_count)?))?)?;
        self.var = m2.broadcast_div(&tot_count)?;
        self.count += 1.0;
        Ok(())
    }

    pub fn forward_obs(&mut self, x: &Tensor) -> Result<Tensor> {
        self.update_obs(x)?;
        (x - self.mean.clone()) / (self.var.clone() + 1e-8)?.sqrt()?
    }

    fn update_ret(&mut self, ret: f32) {
        let batch_mean = ret;
        let delta = batch_mean - self.ret_mean;
        let tot_count = 1.0 + self.ret_count;
        self.ret_mean += delta / tot_count;
        let m_a = self.ret_var * self.ret_count;
        let m2 = m_a + (delta * delta) * self.ret_count / tot_count;
        self.ret_var = m2 / tot_count;
        self.ret_count = tot_count;
    }

    pub fn forward_ret(&mut self, rew: f32) -> f32 {
        self.returns = self.returns * 0.99 + rew;
        self.update_ret(self.returns);
        rew / (self.ret_var + 1e-8).sqrt()
    }
}

pub struct PpoThinker {
    device: Device,
    actor: PpoActor,
    critic: PpoCritic,
    optim: AdamW,
    status: PpoStatus,
    training_epochs: usize,
    training_batch_size: usize,
    entropy_beta: f32,
    varmap: VarMap,
    action_len: usize,
}

impl PpoThinker {
    pub fn new(
        obs_len: usize,
        hidden_len: usize,
        action_len: usize,
        training_epochs: usize,
        training_batch_size: usize,
        entropy_beta: f32,
        lr: f64,
        _rng: &mut EntropyComponent<ChaCha8Rng>,
    ) -> Result<Self> {
        let device = DEVICE.to_owned();
        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let actor = PpoActor::new(obs_len, hidden_len, action_len, vs.pp("actor"))?;
        let critic = PpoCritic::new(obs_len, hidden_len, vs.pp("critic"))?;
        let optim = AdamW::new_lr(varmap.all_vars(), lr)?;
        Ok(Self {
            action_len,
            device,
            varmap,
            actor,
            critic,
            optim,
            status: PpoStatus::default(),
            training_batch_size,
            training_epochs,
            entropy_beta,
        })
    }
}

impl<E: Env> Thinker<E> for PpoThinker
where
    E::Action: Action<E, Metadata = PpoMetadata>,
{
    type Metadata = HiddenStates;
    type Status = PpoStatus;
    type ActionMetadata = PpoMetadata;

    fn status(&self) -> Self::Status {
        self.status.clone()
    }

    fn init_metadata(&self, _batch_size: usize) -> Self::Metadata {
        // let dev = self.actor.devices()[0];
        HiddenStates {}
    }

    fn act(
        &mut self,
        obs: &FrameStack<Box<[f32]>>,
        metadata: &mut Self::Metadata,
        params: &E::Params,
        rng: &mut EntropyComponent<ChaCha8Rng>,
    ) -> E::Action {
        let hiddens = metadata.clone(); //.to_device(&TchDEVICE);
        let obs = obs
            .as_vec()
            .into_iter()
            .map(|o| Tensor::from_slice(&o, (o.len(),), &self.device).unwrap())
            .collect::<Vec<_>>();
        let obs = Tensor::stack(obs.as_slice(), 0)
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        let (mu, cov) = self.actor.forward(&obs).unwrap();
        self.status.mu = mu
            .reshape((self.action_len,))
            .unwrap()
            .to_vec1()
            .unwrap()
            .into_boxed_slice();
        self.status.cov = cov
            .reshape((self.action_len,))
            .unwrap()
            .to_vec1()
            .unwrap()
            .into_boxed_slice();
        let dist = MvNormal { mu, cov };
        // self.status.recent_entropy = dist.entropy().into_scalar();
        self.status.recent_entropy = 0.0;
        let action = dist.sample(rng).unwrap();
        let val = self.critic.forward(&obs).unwrap();

        let action_vec = action
            .reshape((self.action_len,))
            .unwrap()
            .to_vec1()
            .unwrap();

        let logp = dist
            .log_prob(&action)
            .reshape(())
            .unwrap()
            .to_scalar()
            .unwrap();

        E::Action::from_slice(
            action_vec.as_slice(),
            PpoMetadata {
                val: val.reshape(()).unwrap().to_scalar().unwrap(),
                logp,
                hiddens: Some(hiddens),
            },
            params,
        )
    }

    fn learn(
        &mut self,
        rb: &mut PpoBuffer<E>,
        params: &E::Params,
        rng: &mut EntropyComponent<ChaCha8Rng>,
    ) {
        use kdam::{tqdm, BarExt};
        let mean = |l: &[f32]| l.iter().sum::<f32>() / l.len() as f32;
        let mut total_pi_loss: Vec<f32> = vec![];
        let mut total_val_loss: Vec<f32> = vec![];
        let mut total_entropy_loss = vec![];
        let mut total_nclamp = vec![];
        let mut total_kl = vec![];
        let mut total_explained_var = vec![];
        let mut total_policy_batches = 0;
        let mut total_batches = 0;
        let mut it = tqdm!(total = self.training_epochs, desc = "Training");
        for _epoch in 0..self.training_epochs {
            let batches = rb.shuffled_and_batched(self.training_batch_size, rng);
            for (_batch_i, batch) in batches.iter().enumerate() {
                let s = batch
                    .obs
                    .iter()
                    .map(|stack| {
                        Tensor::stack(
                            stack
                                .as_vec()
                                .into_iter()
                                .map(|x| Tensor::new(&*x, &self.device).unwrap())
                                .collect::<Vec<_>>()
                                .as_slice(),
                            0,
                        )
                        .unwrap()
                    })
                    .collect::<Vec<_>>();
                let s = Tensor::stack(s.as_slice(), 0).unwrap();
                let a = batch
                    .action
                    .iter()
                    .map(|action| Tensor::new(&*action.as_slice(params), &self.device).unwrap())
                    .collect::<Vec<_>>();
                let a = Tensor::stack(a.as_slice(), 0).unwrap();
                let old_lp = batch
                    .action
                    .iter()
                    .map(|action| action.metadata().logp)
                    .collect::<Vec<_>>();
                let old_lp = Tensor::new(old_lp.as_slice(), &self.device).unwrap();
                let old_val = batch
                    .action
                    .iter()
                    .map(|action| action.metadata().val)
                    .collect::<Vec<_>>();
                let old_val = Tensor::new(old_val.as_slice(), &self.device).unwrap();

                let advantage = Tensor::new(
                    batch
                        .advantage
                        .iter()
                        .copied()
                        .map(|a| a.unwrap())
                        .collect_vec()
                        .as_slice(),
                    &self.device,
                )
                .unwrap();
                // let advantage = (advantage - advantage.mean(0).unwrap()).unwrap()
                //     / (advantage.variance + 1e-8).sqrt();
                let returns = (&advantage + &old_val).unwrap();

                let (mu, cov) = self.actor.forward(&s).unwrap();
                let dist = MvNormal {
                    mu: mu.clone(),
                    cov: cov.clone(),
                };
                // let entropy = dist.entropy();
                let lp = dist.log_prob(&a);
                let log_ratio = (lp - old_lp).unwrap();

                let ratio = log_ratio.exp().unwrap();
                let surr1 = (&ratio * &advantage).unwrap();
                // let nclamp = ratio
                //     .zeros_like()
                //     .unwrap()
                //     .where_cond(, on_false)
                //     .mask_fill(ratio.clone().lower_elem(0.8), 1.0);
                // let nclamp = nclamp.mask_fill(ratio.clone().greater_elem(1.2), 1.0);
                // total_nclamp.push(nclamp.mean().into_scalar());
                total_nclamp.push(0.0);
                let surr2 = (ratio
                    .broadcast_minimum(&Tensor::new(1.2f32, &self.device).unwrap())
                    .unwrap()
                    .broadcast_maximum(&Tensor::new(0.8f32, &self.device).unwrap())
                    .unwrap()
                    * advantage.clone())
                .unwrap();
                let masked = surr2.minimum(&surr1).unwrap();
                let policy_loss = masked.mean(0).unwrap().neg().unwrap();

                // let entropy_loss = entropy.mean();
                // total_entropy_loss.push(entropy_loss.clone().into_scalar());
                total_entropy_loss.push(0.0);
                // let policy_loss = policy_loss - entropy_loss * self.entropy_beta;
                let kl = log_ratio
                    .neg()
                    .unwrap()
                    .mean(0)
                    .unwrap()
                    .to_scalar()
                    .unwrap();

                total_kl.push(kl);
                let pl = policy_loss.to_scalar().unwrap();

                total_pi_loss.push(pl);

                let val = self.critic.forward(&s).unwrap();
                let val_err = (val - returns).unwrap();
                let value_loss = (&val_err * &val_err).unwrap().mean(0).unwrap();
                let vl = value_loss.to_scalar().unwrap();
                total_val_loss.push(vl);

                let loss = (policy_loss + value_loss).unwrap();
                if kl <= 0.2 {
                    total_policy_batches += 1;
                    let grads = loss.backward().unwrap();

                    self.optim.step(&grads).unwrap();
                } else {
                    // println!("Maximum KL reached after {batch_i} batches");
                    // break;
                }

                // let y_true = returns.clone().squeeze(1);
                // let explained_var = if y_true.clone().var(0).into_scalar() == 0.0 {
                //     f32::NAN
                // } else {
                //     1.0 - ((y_true.clone() - val.clone().detach()).var(0) / y_true.var(0))
                //         .into_scalar()
                // };
                total_explained_var.push(0.0);

                total_batches += 1;
            }
            it.set_postfix(format!(
                "pl={} vl={} kl={}",
                mean(&total_pi_loss),
                mean(&total_val_loss),
                mean(&total_kl),
            ));
            it.update(1).ok();
        }
        self.status.policy_batches_completed = total_policy_batches as f32 / total_batches as f32;
        self.status.policy_loss = mean(&total_pi_loss);
        self.status.value_loss = mean(&total_val_loss);
        self.status.entropy_loss = mean(&total_entropy_loss);
        self.status.nclamp = mean(&total_nclamp);
        self.status.kl = mean(&total_kl);
        self.status.explained_var = mean(&total_explained_var);
    }

    fn save(
        &self,
        _path: impl AsRef<std::path::Path>,
    ) -> std::result::Result<(), std::boxed::Box<(dyn std::error::Error + 'static)>> {
        // self.actor.clone().save_file(
        //     path.as_ref().join("actor"),
        //     &BinGzFileRecorder::<FullPrecisionSettings>::new(),
        // )?;
        // self.critic.clone().save_file(
        //     path.as_ref().join("critic"),
        //     &BinGzFileRecorder::<FullPrecisionSettings>::new(),
        // )?;
        Ok(())
    }
}
