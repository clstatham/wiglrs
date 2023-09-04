#![allow(clippy::single_range_in_vec_init)]

use bevy::prelude::{Component, Resource};
use bevy_prng::ChaCha8Rng;
use bevy_rand::prelude::EntropyComponent;

use itertools::Itertools;

use crate::brains::learners::OnPolicyBuffer;
use crate::brains::models::{Policy, ValueEstimator};
use crate::envs::{Action, Env};

use candle_core::{DType, Result, Tensor};

use self::rollout_buffer::PpoBuffer;

use super::Buffer;
use super::{Learner, Status, DEVICE};

pub mod rollout_buffer;

#[derive(Debug, Clone)]
pub struct HiddenStates {}

#[derive(Debug, Clone, Default, Component)]
pub struct PpoStatus {
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

#[derive(Resource)]
pub struct Ppo {
    status: PpoStatus,
    training_epochs: usize,
    training_batch_size: usize,
    entropy_beta: f32,
}

impl Ppo {
    pub fn new(
        training_epochs: usize,
        training_batch_size: usize,
        entropy_beta: f32,
    ) -> Result<Self> {
        Ok(Self {
            status: PpoStatus::default(),
            training_batch_size,
            training_epochs,
            entropy_beta,
        })
    }
}

impl<E: Env, P: Policy, V: ValueEstimator> Learner<E, P, V> for Ppo {
    type Status = PpoStatus;
    type Buffer = PpoBuffer<E>;

    fn status(&self) -> Self::Status {
        self.status.clone()
    }

    fn learn(
        &mut self,
        policy: &P,
        value: &V,
        buffer: &Self::Buffer,
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
            let batches = buffer.shuffled_and_batched(self.training_batch_size, rng);
            for (_batch_i, batch) in batches.iter().enumerate() {
                let s = batch
                    .obs
                    .iter()
                    .map(|stack| {
                        Tensor::stack(
                            stack
                                .as_vec()
                                .into_iter()
                                .map(|x| Tensor::new(&*x, &DEVICE).unwrap())
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
                    .map(|action| Tensor::new(&*action.as_slice(), &DEVICE).unwrap())
                    .collect::<Vec<_>>();
                let a = Tensor::stack(a.as_slice(), 0).unwrap();
                let old_lp = batch
                    .metadata
                    .iter()
                    .map(|action| action.logp)
                    .collect::<Vec<_>>();
                let old_lp = Tensor::new(old_lp.as_slice(), &DEVICE).unwrap();
                let old_val = batch
                    .metadata
                    .iter()
                    .map(|action| action.val)
                    .collect::<Vec<_>>();
                let old_val = Tensor::new(old_val.as_slice(), &DEVICE).unwrap();

                let advantage = Tensor::new(
                    batch
                        .advantage
                        .iter()
                        .copied()
                        .map(|a| a.unwrap())
                        .collect_vec()
                        .as_slice(),
                    &DEVICE,
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

                let logits = policy.action_logits(&s).unwrap();
                let entropy = policy.entropy(&logits).unwrap();
                total_entropy_loss.push(
                    entropy
                        .detach()
                        .unwrap()
                        .mean(0)
                        .unwrap()
                        .to_scalar()
                        .unwrap(),
                );
                let lp = policy.log_prob(&logits, &a).unwrap();
                let log_ratio = (&lp - &old_lp).unwrap();

                let ratio = log_ratio.exp().unwrap();
                let surr1 = (&ratio * &advantage).unwrap();

                let lo = Tensor::new(0.1f32, &DEVICE).unwrap();
                let hi = Tensor::new(1.1f32, &DEVICE).unwrap();

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

                let val = value.estimate_value(&s, Some(&a)).unwrap();
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

                if pl.is_finite() && vl.is_finite() && kl.is_finite() {
                    if kl <= 0.02 * 1.5 {
                        let loss = (policy_loss + (value_loss * 0.5).unwrap()).unwrap();

                        total_policy_batches += 1;
                        let grads = loss.backward().unwrap();

                        // // gradient clip-by-norm
                        // for var in self.varmap.all_vars().iter() {
                        //     let var = var.as_tensor();
                        //     let grad = grads.get(var).unwrap();
                        //     let norm: f32 = grad
                        //         .sqr()
                        //         .unwrap()
                        //         .sum_all()
                        //         .unwrap()
                        //         .sqrt()
                        //         .unwrap()
                        //         .to_scalar()
                        //         .unwrap();
                        //     let grad = if norm >= 0.1 {
                        //         (grad * 0.1)
                        //             .unwrap()
                        //             .affine((norm as f64).recip(), 0.0)
                        //             .unwrap()
                        //     } else {
                        //         grad.to_owned()
                        //     };
                        //     grads.insert(var, grad);
                        // }
                        policy.apply_gradients(&grads).unwrap();
                        value.apply_gradients(&grads).unwrap();
                    }
                } else {
                    eprintln!("kl: {:?}\n", kl);
                    eprintln!("s: {:?}\n", s.to_vec3::<f32>().ok());
                    eprintln!("a: {:?}\n", a.to_vec2::<f32>().ok());
                    eprintln!("adv: {:?}\n", advantage.to_vec1::<f32>().ok());
                    eprintln!("ret: {:?}\n", returns.to_vec1::<f32>().ok());
                    // eprintln!("mu: {:?}\n", mu.to_vec2::<f32>().ok());
                    // eprintln!("cov: {:?}\n", cov.to_vec2::<f32>().ok());
                    eprintln!("lp: {:?}\n", lp.to_vec1::<f32>().ok());

                    if !pl.is_finite() {
                        panic!("pl={pl}");
                    }
                    if !vl.is_finite() {
                        panic!("vl={vl}");
                    }
                    if !kl.is_finite() {
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
}
