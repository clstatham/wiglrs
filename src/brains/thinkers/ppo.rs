use dfdx::optim::Adam;
use dfdx::prelude::*;
use std::collections::VecDeque;
use std::f32::consts::PI;
use std::ops::{Add, Mul};

use crate::brains::replay_buffer::ReplayBuffer;
use crate::hparams::{AGENT_ACTOR_LR, AGENT_EMBED_DIM, AGENT_OPTIM_BATCH_SIZE, AGENT_OPTIM_EPOCHS};
use crate::{Action, Observation, ACTION_LEN, OBS_LEN};

use super::Thinker;

pub type Device = AutoDevice;

pub type Tensor<S, T = NoneTape> = dfdx::tensor::Tensor<S, f32, Device, T>;

pub struct MvNormal<const K: usize, T: Tape<f32, Device> + Merge<NoneTape>> {
    pub mu: Tensor<Rank1<K>, T>,
    pub cov_diag: Tensor<Rank1<K>, T>,
}

impl<const K: usize, T: Tape<f32, Device>> MvNormal<K, T>
where
    T: Merge<NoneTape>,
{
    pub fn sample(self) -> Tensor<Rank1<K>, T> {
        let cov_mat = nalgebra::DMatrix::from_diagonal(&nalgebra::DVector::from_iterator(
            K,
            self.cov_diag.as_vec().into_iter(),
        ));
        let a = cov_mat.cholesky().unwrap().unpack();
        let z = self.mu.device().sample_normal::<Rank1<K>>();
        let z = nalgebra::DVector::from_iterator(K, z.as_vec().into_iter());
        let x = a.mul(&z);
        let x = self.mu.device().tensor(x.data.as_vec().clone());
        self.mu + x
    }

    pub fn log_prob(self, x: Tensor<Rank1<K>>) -> Tensor<Rank1<1>, T> {
        let cov_mat = nalgebra::DMatrix::from_diagonal(&nalgebra::DVector::from_iterator(
            K,
            self.cov_diag.as_vec().into_iter(),
        ));
        let a = (-self.mu.with_empty_tape()).add(x.clone());
        let b = x.device().tensor_from_vec(
            cov_mat.clone().try_inverse().unwrap().data.as_vec().clone(),
            (Const::<K>, Const::<K>),
        );
        let c = ((-self.mu).add(x)).broadcast::<Rank2<K, 1>, _>();
        let g = cov_mat.determinant();
        let d = a.matmul(b);
        let e = d.matmul(c);
        let numer = (e * -0.5).exp();
        let f = (2.0 * PI).powf(K as f32);
        let denom = (f * g).sqrt();
        let pdf = numer / denom;
        pdf.ln()
    }
}

pub struct BatchMvNormal<const K: usize, T: Tape<f32, Device> + Merge<NoneTape>> {
    batches: VecDeque<MvNormal<K, T>>,
}

impl<const K: usize, T: Tape<f32, Device> + Merge<NoneTape>> BatchMvNormal<K, T> {
    pub fn new(mu: Tensor<(usize, Const<K>), T>, var: Tensor<(usize, Const<K>), T>) -> Self {
        let cpu = Cpu::default();
        let mu = mu.to_device(&cpu);
        let var = var.to_device(&cpu);
        let mut batches = VecDeque::new();

        for i in 0..mu.shape().0 {}

        Self { batches }
    }

    pub fn batch_log_prob(mut self, x: Tensor<(usize, Const<K>)>) -> Tensor<(usize, Const<1>), T> {
        // let mut out = cpu.zeros();
        let mut out = vec![];
        for i in 0..x.shape().0 {
            let action = x.clone().select(x.device().tensor(i));
            let lp = self.batches.pop_front().unwrap().log_prob(action);
            out.push(lp);
        }
        out.stack()
    }
}

#[derive(Default, Clone)]
pub struct Softplus;

impl ZeroSizedModule for Softplus {}
impl<S: Shape, D: dfdx::tensor_ops::Device<f32>, T: Tape<f32, D>>
    Module<dfdx::tensor::Tensor<S, f32, D, T>> for Softplus
{
    type Error = D::Err;
    type Output = dfdx::tensor::Tensor<S, f32, D, T>;
    fn try_forward(
        &self,
        input: dfdx::tensor::Tensor<S, f32, D, T>,
    ) -> Result<Self::Output, Self::Error> {
        Ok(input.exp().add(1.0).ln())
    }
}

type Policy = (
    (Linear<OBS_LEN, AGENT_EMBED_DIM>, ReLU),
    (Linear<AGENT_EMBED_DIM, AGENT_EMBED_DIM>, ReLU),
    SplitInto<(
        (Linear<AGENT_EMBED_DIM, ACTION_LEN>, Tanh),
        (Linear<AGENT_EMBED_DIM, ACTION_LEN>, Softplus),
    )>,
);

type Value = (
    (Linear<OBS_LEN, AGENT_EMBED_DIM>, ReLU),
    (Linear<AGENT_EMBED_DIM, AGENT_EMBED_DIM>, ReLU),
    Linear<AGENT_EMBED_DIM, 1>,
);

type PpoNet = SplitInto<(Policy, Value)>;

type BuiltPpoNet = <PpoNet as BuildOnDevice<Device, f32>>::Built;

pub struct PpoThinker {
    net: BuiltPpoNet,
    target_net: BuiltPpoNet,
    net_grads: Gradients<f32, Device>,
    optim: Adam<BuiltPpoNet, f32, Device>,
    device: Device,
    cpu: Cpu,
}

impl PpoThinker {
    pub fn new() -> Self {
        let device = Device::seed_from_u64(rand::random());
        let net = device.build_module::<PpoNet, f32>();
        let target_net = net.clone();
        let net_grads = net.alloc_grads();
        Self {
            optim: Adam::new(
                &net,
                AdamConfig {
                    lr: AGENT_ACTOR_LR,
                    ..Default::default()
                },
            ),
            net,
            net_grads,
            target_net,
            device,
            cpu: Cpu::seed_from_u64(rand::random()),
        }
    }
}

impl Default for PpoThinker {
    fn default() -> Self {
        Self::new()
    }
}

impl Thinker for PpoThinker {
    fn act(&self, obs: Observation) -> Action {
        let obs: Tensor<Rank1<OBS_LEN>> = self.device.tensor(obs.as_vec());
        let ((mu, var), _) = self.net.forward(obs);
        let dist = MvNormal { mu, cov_diag: var };
        let action = dist.sample();

        Action::from_slice(action.as_vec().as_slice())
    }

    fn learn(&mut self, rb: &mut ReplayBuffer) {
        let mut discounted_reward = 0.0;
        let mut rewards = vec![];
        let nstep = rb.buf.len();
        for step in rb.buf.iter().rev() {
            if step.terminal {
                discounted_reward = 0.0;
            }
            discounted_reward = step.reward + 0.99 * discounted_reward;
            rewards.insert(0, discounted_reward);
        }
        let rewards = self.device.tensor_from_vec(rewards, (nstep,));
        let rewards = rewards.normalize(1e-7).as_vec();

        let mut total_loss = 0.0;

        for _ in 0..AGENT_OPTIM_EPOCHS {
            for (step, reward) in rb.buf.iter().zip(rewards.iter().copied()) {
                let s: Tensor<Rank1<OBS_LEN>> = self.device.tensor(step.obs.as_vec());
                let a: Tensor<Rank1<ACTION_LEN>> = self.device.tensor(step.action.as_vec());

                let ((old_mu, old_var), old_val) = self.target_net.forward(s.clone());
                let old_dist = MvNormal {
                    mu: old_mu,
                    cov_diag: old_var,
                };
                let old_lp = old_dist.log_prob(a.clone());
                let advantage = (-old_val) + reward;

                let ((mu, var), val) = self.net.forward(s.trace(self.net_grads.to_owned()));
                let dist = MvNormal { mu, cov_diag: var };
                let lp = dist.log_prob(a);
                let ratio = (lp - old_lp).exp();
                let surr1 = ratio.with_empty_tape() * advantage.clone();
                let surr2 = ratio.clamp(0.8, 1.2) * advantage;
                let policy_loss = -surr2.minimum(surr1).mean();
                let value_loss = (val.square() - reward * reward).mean();
                let loss = policy_loss + value_loss;
                total_loss += loss.array() / nstep as f32;
                self.net_grads = loss.backward();
                self.optim.update(&mut self.net, &self.net_grads).unwrap();
                self.net.zero_grads(&mut self.net_grads);
            }
            self.target_net.clone_from(&self.net);
        }
        println!("Loss: {total_loss}");
        rb.buf.clear();
    }
}
