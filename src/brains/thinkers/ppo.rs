#![allow(clippy::single_range_in_vec_init)]

use bevy::prelude::Component;
use bevy_prng::ChaCha8Rng;
use bevy_rand::prelude::EntropyComponent;

use itertools::Itertools;
use rand_distr::Distribution;
use std::{f32::consts::PI, ops::Mul};

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

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_log_prob() -> Result<()> {
        let mu = Tensor::new(&[-0.0966f32, -0.5324, 0.1607, 0.4173], &DEVICE)?.unsqueeze(0)?;
        let cov = Tensor::new(&[1.3077f32, 3.7718, 1.7900, 23.6713], &DEVICE)?.unsqueeze(0)?;
        let action = Tensor::new(&[1.0094f32, -6.3756, -2.0827, 75.4585], &DEVICE)?.unsqueeze(0)?;
        // let action =
        //     Tensor::new(&[-1.7375f32, -0.1554, -2.1293, -2.1105], &DEVICE)?.unsqueeze(0)?;
        let dist = MvNormal { mu, cov };
        let lp = dist.log_prob(&action).reshape(())?.to_scalar::<f32>()?;
        let difference = lp - -131.6919;
        assert!(difference.abs() < 1e-4, "{:?}", lp);
        Ok(())
    }
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

    // https://github.com/pytorch/pytorch/blob/ba9acbebfc54b6a81d3db40c58e49b12041980ae/torch/distributions/normal.py#L77C10-L77C10
    pub fn log_prob(&self, x: &Tensor) -> Tensor {
        let x = x.to_device(self.mu.device()).unwrap();
        let log_scale = (self.cov.sqrt().unwrap() + 1e-8).unwrap().log().unwrap();
        let first = ((x - &self.mu).unwrap().sqr().unwrap().neg().unwrap()
            / self.cov.affine(2.0, 1e-8).unwrap())
        .unwrap();
        let third = (2.0 * std::f64::consts::PI).sqrt().ln();

        let individuals = (first - log_scale).unwrap().affine(1.0, -third).unwrap();
        individuals.sum(1).unwrap()
    }

    pub fn entropy(&self) -> Result<Tensor> {
        let (_, nfeat) = self.mu.shape().dims2().unwrap();
        let second_term = (nfeat as f32 * 0.5) * (1.0 + (2.0 * PI).ln());
        self.cov
            .log()
            .unwrap()
            .sum(1)
            .unwrap()
            .mul(0.5)
            .unwrap()
            .broadcast_add(&Tensor::new(second_term, &DEVICE).unwrap())
    }
}

pub fn linear(in_len: usize, out_len: usize, gain: f64, vs: VarBuilder) -> Linear {
    let w_init = candle_nn::init::Init::Kaiming {
        dist: candle_nn::init::NormalOrUniform::Normal,
        fan: candle_nn::init::FanInOut::FanIn,
        non_linearity: candle_nn::init::NonLinearity::ExplicitGain(gain),
    };
    let b_init = candle_nn::init::Init::Const(0.0);
    let weight = vs
        .get_with_hints((out_len, in_len), "weight", w_init)
        .unwrap();
    let bias = vs.get_with_hints((out_len,), "bias", b_init).unwrap();
    Linear::new(weight, Some(bias))
}

pub struct ResBlock {
    pub l1: Linear,
    pub l2: Linear,
}

impl ResBlock {
    pub fn new(in_out_len: usize, hidden_len: usize, vs: VarBuilder) -> ResBlock {
        ResBlock {
            l1: linear(in_out_len, hidden_len, 5. / 3., vs.pp("l1")),
            l2: linear(hidden_len, in_out_len, 5. / 3., vs.pp("l2")),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x1 = self.l1.forward(x)?.tanh()?;
        let x2 = self.l2.forward(&x1)?;
        (x2 + x)?.tanh()
    }
}

pub struct PpoActor {
    common1: Linear,
    common2: ResBlock,
    common3: ResBlock,
    mu_head: Linear,
    cov_head: Linear,
}

impl PpoActor {
    pub fn forward(&mut self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        // let (_, nseq, _) = x.shape().dims3().unwrap();
        // let x = x
        //     .index_select(
        //         &Tensor::from_slice::<_, i64>(&[nseq as i64 - 1], (1,), &DEVICE)?,
        //         1,
        //     )?
        //     .squeeze(1)
        //     .unwrap();
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
}

impl PpoActor {
    pub fn new(
        obs_len: usize,
        hidden_len: usize,
        action_len: usize,
        vs: VarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            common1: linear(obs_len, hidden_len, 5. / 3., vs.pp("common1")),
            common2: ResBlock::new(hidden_len, hidden_len / 2, vs.pp("common2")),
            common3: ResBlock::new(hidden_len, hidden_len / 2, vs.pp("common3")),
            mu_head: linear(hidden_len, action_len, 0.01, vs.pp("mu_head")),
            // cov_head: vs.get_with_hints((1, action_len), "cov_head", Init::Const(0.0))?,
            cov_head: linear(hidden_len, action_len, 0.01, vs.pp("cov_head")),
        })
    }
}

pub struct PpoCritic {
    l1: Linear,
    l2: ResBlock,
    l3: ResBlock,
    head: Linear,
}

impl PpoCritic {
    pub fn new(obs_len: usize, hidden_len: usize, vs: VarBuilder) -> Result<Self> {
        Ok(Self {
            l1: linear(obs_len, hidden_len, 5. / 3., vs.pp("l1")),
            l2: ResBlock::new(hidden_len, hidden_len / 2, vs.pp("l2")),
            l3: ResBlock::new(hidden_len, hidden_len / 2, vs.pp("l3")),
            head: linear(hidden_len, 1, 1.0, vs.pp("head")),
        })
    }
}

impl PpoCritic {
    pub fn forward(&mut self, x: &Tensor) -> Result<Tensor> {
        // let (_, nseq, _) = x.shape().dims3().unwrap();
        // let x = x
        //     .index_select(
        //         &Tensor::from_slice::<_, i64>(&[nseq as i64 - 1], (1,), &DEVICE)?,
        //         1,
        //     )?
        //     .squeeze(1)
        //     .unwrap();
        let x = x.flatten(1, 2)?;
        let x = self.l1.forward(&x)?.tanh()?;
        let x = self.l2.forward(&x)?;
        let x = self.l3.forward(&x)?;

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
    fn agent_warmup(&self) -> usize;
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
        frame_stack_len: usize,
        lr: f64,
        _rng: &mut EntropyComponent<ChaCha8Rng>,
    ) -> Result<Self> {
        let obs_len = obs_len * frame_stack_len;
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
        self.status.recent_entropy = dist
            .entropy()
            .unwrap()
            .reshape(())
            .unwrap()
            .to_scalar()
            .unwrap();
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
        fn variance(a: &Tensor) -> Tensor {
            let a_mean = a.mean(0).unwrap();
            let n = Tensor::new(a.shape().dims()[0] as f32, &DEVICE).unwrap();
            a.broadcast_sub(&a_mean)
                .unwrap()
                .sqr()
                .unwrap()
                .sum_keepdim(0)
                .unwrap()
                .broadcast_div(&n)
                .unwrap()
        }
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

                let adv_mean = advantage.mean_keepdim(0).unwrap();
                let adv_std = variance(&advantage).sqrt().unwrap();
                let advantage = (advantage
                    .broadcast_sub(&adv_mean)
                    .unwrap()
                    .broadcast_div(&(adv_std + 1e-8).unwrap()))
                .unwrap();
                let returns = (&advantage + &old_val).unwrap();

                let (mu, cov) = self.actor.forward(&s).unwrap();
                let dist = MvNormal {
                    mu: mu.clone(),
                    cov: cov.clone(),
                };
                let entropy = dist.entropy().unwrap();
                total_entropy_loss.push(
                    entropy
                        .detach()
                        .unwrap()
                        .mean(0)
                        .unwrap()
                        .to_scalar()
                        .unwrap(),
                );
                let lp = dist.log_prob(&a);
                let log_ratio = (&lp - &old_lp).unwrap();

                let ratio = log_ratio.exp().unwrap();
                let surr1 = (&ratio * &advantage).unwrap();

                let lo = Tensor::new(0.8f32, &self.device).unwrap();
                let hi = Tensor::new(1.2f32, &self.device).unwrap();

                let nclamp: f32 = (ratio
                    .detach()
                    .unwrap()
                    .lt(&lo.broadcast_left(ratio.shape()).unwrap())
                    .unwrap()
                    + ratio
                        .detach()
                        .unwrap()
                        .gt(&hi.broadcast_left(ratio.shape()).unwrap())
                        .unwrap())
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap()
                .mean(0)
                .unwrap()
                .to_scalar()
                .unwrap();
                total_nclamp.push(nclamp);

                let surr2 = (ratio
                    .broadcast_minimum(&hi)
                    .unwrap()
                    .broadcast_maximum(&lo)
                    .unwrap()
                    * advantage.clone())
                .unwrap();
                let masked = surr2.minimum(&surr1).unwrap();
                let policy_loss = masked.neg().unwrap().mean(0).unwrap();

                let kl: f32 = log_ratio
                    .detach()
                    .unwrap()
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
                let value_loss = (&val - &returns).unwrap().sqr().unwrap().mean(0).unwrap();
                let vl = value_loss.to_scalar().unwrap();
                total_val_loss.push(vl);

                let y_true = returns.detach().unwrap();
                let explained_var: f32 = (Tensor::new(&[1.0f32], &DEVICE).unwrap()
                    - (variance(&(&y_true - &val.detach().unwrap()).unwrap())
                        .div(&(variance(&y_true) + 1e-8).unwrap())))
                .unwrap()
                .reshape(())
                .unwrap()
                .to_scalar()
                .unwrap();
                total_explained_var.push(explained_var);

                if pl.is_normal() && vl.is_normal() && kl.is_finite() && !kl.is_nan() {
                    if kl <= 0.02 * 1.5 {
                        let loss = (policy_loss + (value_loss * 0.5).unwrap()).unwrap();

                        total_policy_batches += 1;
                        let mut grads = loss.backward().unwrap();

                        // gradient clip-by-norm
                        for var in self.varmap.all_vars().iter() {
                            let var = var.as_tensor();
                            let grad = grads.get(var).unwrap();
                            let norm: f32 = grad
                                .sqr()
                                .unwrap()
                                .sum_all()
                                .unwrap()
                                .sqrt()
                                .unwrap()
                                .to_scalar()
                                .unwrap();
                            let grad = if norm >= 0.5 {
                                (grad * 0.5)
                                    .unwrap()
                                    .affine((norm as f64).recip(), 0.0)
                                    .unwrap()
                            } else {
                                grad.to_owned()
                            };
                            grads.insert(var, grad);
                        }

                        self.optim.step(&grads).unwrap();
                    }
                } else {
                    eprintln!("kl: {:?}\n", kl);
                    eprintln!("s: {:?}\n", s.to_vec3::<f32>().ok());
                    eprintln!("a: {:?}\n", a.to_vec2::<f32>().ok());
                    eprintln!("adv: {:?}\n", advantage.to_vec1::<f32>().ok());
                    eprintln!("ret: {:?}\n", returns.to_vec1::<f32>().ok());
                    eprintln!("mu: {:?}\n", mu.to_vec2::<f32>().ok());
                    eprintln!("cov: {:?}\n", cov.to_vec2::<f32>().ok());
                    eprintln!("lp: {:?}\n", lp.to_vec1::<f32>().ok());

                    if !pl.is_normal() {
                        panic!("pl={pl}");
                    }
                    if !vl.is_normal() {
                        panic!("vl={vl}");
                    }
                    if !kl.is_normal() {
                        panic!("kl={kl}");
                    }
                }

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
        path: impl AsRef<std::path::Path>,
    ) -> std::result::Result<(), std::boxed::Box<(dyn std::error::Error + 'static)>> {
        self.varmap.save(path.as_ref().join("model.safetensors"))?;

        Ok(())
    }
}
