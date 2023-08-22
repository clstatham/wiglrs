use dfdx::nn::modules::Linear;
use dfdx::optim::Adam;
use dfdx::prelude::*;
use std::collections::VecDeque;
use std::ops::{Add, Mul};

use crate::brains::replay_buffer::ReplayBuffer;
use crate::brains::FrameStack;
use crate::hparams::{
    AGENT_HIDDEN_DIM, AGENT_LR, AGENT_OPTIM_BATCH_SIZE, AGENT_OPTIM_EPOCHS, N_FRAME_STACK,
};
use crate::{Action, ACTION_LEN, OBS_LEN};

use super::Thinker;

pub type Device = Cuda;
type Float = f32;
const PI: Float = std::f32::consts::PI;
pub type FTensor<S, T = NoneTape> = Tensor<S, Float, Device, T>;

pub struct MvNormal<const K: usize, T: Tape<Float, Device> + Merge<NoneTape>> {
    pub mu: FTensor<Rank1<K>, T>,
    pub cov_diag: FTensor<Rank1<K>, T>,
}

impl<const K: usize, T: Tape<Float, Device>> MvNormal<K, T>
where
    T: Merge<NoneTape>,
{
    pub fn sample(self) -> FTensor<Rank1<K>, T> {
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

    pub fn log_prob(self, x: FTensor<Rank1<K>>) -> (FTensor<Rank1<1>, T>, Float) {
        let cov_mat = nalgebra::DMatrix::from_diagonal(&nalgebra::DVector::from_iterator(
            K,
            self.cov_diag.as_vec().into_iter(),
        ));
        let a = (-self.mu.with_empty_tape())
            .add(x.clone())
            .broadcast::<Rank2<1, K>, _>();
        let b = x.device().tensor_from_vec(
            cov_mat.clone().try_inverse().unwrap().data.as_vec().clone(),
            (Const::<K>, Const::<K>),
        );
        let c = ((-self.mu).add(x)).broadcast::<Rank2<K, 1>, _>();
        let g = cov_mat.determinant();
        let d = a.matmul(b);
        let e = d.matmul(c);
        let numer = (e * -0.5).exp();
        let f = (2.0 * PI).powf(K as Float);
        let denom = (f * g).sqrt();
        let pdf = numer / denom;
        let logpdf = pdf.ln().reshape();
        let entropy = (K as Float / 2.0) + ((K as Float / 2.0) * (2.0 * PI).ln()) + (g.ln() / 2.0);
        (logpdf, entropy)
    }
}

fn mvn_batch_log_prob<const K: usize, T: Tape<Float, Device> + Merge<NoneTape>>(
    mu: FTensor<(usize, Const<K>), T>,
    var: FTensor<(usize, Const<K>), T>,
    x: FTensor<(usize, Const<K>)>,
) -> (FTensor<(usize, Const<1>), T>, FTensor<(usize, Const<1>)>) {
    let nbatch = x.shape().0;
    assert_eq!(mu.shape().0, nbatch);
    assert_eq!(var.shape().0, nbatch);
    let mut out_lp = vec![];
    let mut out_e = vec![];
    for i in 0..nbatch - 1 {
        let idx = mu.device().tensor(i);
        let mu = mu.with_empty_tape().select(idx.clone());
        let var = var.with_empty_tape().select(idx.clone());
        let x = x.with_empty_tape().select(idx.clone());
        let dist = MvNormal { mu, cov_diag: var };
        let (lp, e) = dist.log_prob(x);
        out_lp.push(lp);
        out_e.push(e);
    }
    let dev = mu.device().clone();
    let idx = mu.device().tensor(nbatch - 1);
    let mu = mu.select(idx.clone());
    let var = var.select(idx.clone());
    let x = x.select(idx.clone());
    let dist = MvNormal { mu, cov_diag: var };
    let (lp, e) = dist.log_prob(x);
    out_lp.push(lp);
    out_e.push(e);
    (out_lp.stack(), dev.tensor_from_vec(out_e, (nbatch, Const)))
}

fn softplus<
    S: Shape,
    E: Dtype + num_traits::Float,
    D: dfdx::tensor_ops::Device<E>,
    T: Tape<E, D>,
>(
    input: Tensor<S, E, D, T>,
) -> Tensor<S, E, D, T> {
    input.exp().add(E::from(1.0).unwrap()).ln()
}

#[derive(Clone)]
struct PpoNet<
    const OBS_LEN: usize,
    const HIDDEN_DIM: usize,
    const ACTION_LEN: usize,
    E: Dtype,
    D: dfdx::tensor_ops::Device<E>,
> {
    common: (
        Linear<OBS_LEN, HIDDEN_DIM, E, D>,
        ReLU,
        Linear<HIDDEN_DIM, HIDDEN_DIM, E, D>,
        ReLU,
        Linear<HIDDEN_DIM, HIDDEN_DIM, E, D>,
        ReLU,
    ),
    mu_head: (Linear<HIDDEN_DIM, ACTION_LEN, E, D>, Tanh),
    var_head: Linear<HIDDEN_DIM, ACTION_LEN, E, D>,
    val_head: Linear<HIDDEN_DIM, 1, E, D>,
}

impl<
        const OBS_LEN: usize,
        const HIDDEN_DIM: usize,
        const ACTION_LEN: usize,
        E: Dtype + num_traits::Float + rand_distr::uniform::SampleUniform,
        D: dfdx::tensor_ops::Device<E>,
    > TensorCollection<E, D> for PpoNet<OBS_LEN, HIDDEN_DIM, ACTION_LEN, E, D>
{
    type To<E2: Dtype, D2: dfdx::tensor_ops::Device<E2>> =
        PpoNet<OBS_LEN, HIDDEN_DIM, ACTION_LEN, E2, D2>;

    fn iter_tensors<V: ModuleVisitor<Self, E, D>>(
        visitor: &mut V,
    ) -> Result<Option<Self::To<V::E2, V::D2>>, V::Err> {
        visitor.visit_fields(
            (
                Self::module("common", |s| &s.common, |s| &mut s.common),
                Self::module("mu_head", |s| &s.mu_head, |s| &mut s.mu_head),
                Self::module("var_head", |s| &s.var_head, |s| &mut s.var_head),
                Self::module("val_head", |s| &s.val_head, |s| &mut s.val_head),
            ),
            |(c, mu, var, val)| Self::To {
                common: c,
                mu_head: mu,
                var_head: var,
                val_head: val,
            },
        )
    }
}

impl<
        const OBS_LEN: usize,
        const HIDDEN_DIM: usize,
        const ACTION_LEN: usize,
        E: Dtype + num_traits::Float,
        D: dfdx::tensor_ops::Device<E>,
        T: Tape<E, D>,
    > Module<Tensor<(usize, Const<OBS_LEN>), E, D, T>>
    for PpoNet<OBS_LEN, HIDDEN_DIM, ACTION_LEN, E, D>
{
    type Error = D::Err;
    type Output = (
        Tensor<(Const<ACTION_LEN>,), E, D, T>,
        Tensor<(Const<ACTION_LEN>,), E, D, T>,
        Tensor<(Const<1>,), E, D, T>,
    );
    fn try_forward(
        &self,
        input: Tensor<(usize, Const<OBS_LEN>), E, D, T>,
    ) -> Result<Self::Output, Self::Error> {
        let nstack = input.shape().concrete()[0];
        let x = self.common.try_forward(input)?;
        let idx = x.device().tensor(nstack - 1);
        let x = x.try_select(idx)?;
        let val = self.val_head.try_forward(x.with_empty_tape())?;
        let var = self.var_head.try_forward(x.with_empty_tape())?;
        let var = softplus(var);
        let mu = self.mu_head.try_forward(x)?;
        Ok((mu, var, val))
    }
}

impl<
        const OBS_LEN: usize,
        const HIDDEN_DIM: usize,
        const ACTION_LEN: usize,
        E: Dtype + num_traits::Float,
        D: dfdx::tensor_ops::Device<E>,
        T: Tape<E, D>,
    > Module<Tensor<(usize, usize, Const<OBS_LEN>), E, D, T>>
    for PpoNet<OBS_LEN, HIDDEN_DIM, ACTION_LEN, E, D>
{
    type Error = D::Err;
    type Output = (
        Tensor<(usize, Const<ACTION_LEN>), E, D, T>,
        Tensor<(usize, Const<ACTION_LEN>), E, D, T>,
        Tensor<(usize, Const<1>), E, D, T>,
    );
    fn try_forward(
        &self,
        input: Tensor<(usize, usize, Const<OBS_LEN>), E, D, T>,
    ) -> Result<Self::Output, Self::Error> {
        let nbatch = input.shape().0;
        let nstack = input.shape().1;
        let x = self.common.try_forward(input)?.relu();
        let x = x
            .slice((0..nbatch, nstack - 1..nstack, 0..HIDDEN_DIM))
            .reshape_like(&(nbatch, Const::<HIDDEN_DIM>));
        let val = self.val_head.try_forward(x.with_empty_tape())?;
        let var = self.var_head.try_forward(x.with_empty_tape())?;
        let var = softplus(var);
        let mu = self.mu_head.try_forward(x)?.tanh();
        Ok((mu, var, val))
    }
}

type Ppo = PpoNet<OBS_LEN, AGENT_HIDDEN_DIM, ACTION_LEN, Float, Device>;

pub struct PpoThinker {
    net: Ppo,
    target_net: Ppo,
    net_grads: Option<Gradients<Float, Device>>,
    optim: Adam<Ppo, Float, Device>,
    device: Device,
    pub recent_mu: Vec<f32>,
    pub recent_std: Vec<f32>,
}

impl PpoThinker {
    pub fn new() -> Self {
        let device = Device::seed_from_u64(rand::random());
        let net = Ppo::build(&device);
        let target_net = net.clone();
        let net_grads = net.alloc_grads();
        Self {
            optim: Adam::new(
                &net,
                AdamConfig {
                    lr: AGENT_LR,
                    ..Default::default()
                },
            ),
            net,
            net_grads: Some(net_grads),
            target_net,
            device,
            recent_mu: Vec::new(),
            recent_std: Vec::new(),
        }
    }
}

impl Default for PpoThinker {
    fn default() -> Self {
        Self::new()
    }
}

impl Thinker for PpoThinker {
    fn act(&mut self, obs: FrameStack) -> Action {
        let obs: FTensor<(usize, Const<OBS_LEN>)> = (obs
            .as_vec()
            .into_iter()
            .map(|x| self.device.tensor(x.as_vec()))
            .collect::<Vec<FTensor<Rank1<OBS_LEN>>>>())
        .stack();
        let (mu, std, _) = self.net.forward(obs);
        self.recent_mu = mu.as_vec();
        self.recent_std = std.as_vec();
        let dist = MvNormal {
            mu,
            cov_diag: std.square(),
        };
        let action = dist.sample();

        Action::from_slice(
            action
                .as_vec()
                .into_iter()
                .map(|x| x as f32)
                .collect::<Vec<_>>()
                .as_slice(),
        )
    }

    fn learn(&mut self, rb: &mut ReplayBuffer) -> f32 {
        let mut discounted_reward = 0.0;
        let mut rewards = VecDeque::new();
        let nstep = rb.buf.len();
        for step in rb.buf.iter().rev() {
            if step.terminal {
                discounted_reward = 0.0;
            }
            discounted_reward = step.reward + 0.99 * discounted_reward;
            rewards.push_front(discounted_reward);
        }
        let rewards = self.device.tensor_from_vec(rewards.into(), (nstep,));
        let rewards = rewards.normalize(1e-7).as_vec();

        let mut total_loss = 0.0;
        let mut net_grads = self.net_grads.take().unwrap();
        let rb: Vec<_> = rb.buf.clone().into();
        for _ in 0..AGENT_OPTIM_EPOCHS {
            for (step, reward) in rb
                .chunks(AGENT_OPTIM_BATCH_SIZE)
                .zip(rewards.chunks(AGENT_OPTIM_BATCH_SIZE))
            {
                let nbatch = reward.len();
                let reward = self
                    .device
                    .tensor_from_vec(reward.to_vec(), (nbatch, Const::<1>));
                let s = step
                    .iter()
                    .map(|step| {
                        step.obs
                            .as_vec()
                            .into_iter()
                            .map(|x| self.device.tensor(x.as_vec()))
                            .collect::<Vec<FTensor<Rank1<OBS_LEN>>>>()
                            .stack()
                    })
                    .collect::<Vec<FTensor<(usize, Const<OBS_LEN>)>>>()
                    .stack();
                let a = step
                    .iter()
                    .map(|step| {
                        self.device.tensor(
                            step.action
                                .as_vec()
                                .into_iter()
                                .map(|x| x as Float)
                                .collect::<Vec<Float>>(),
                        )
                    })
                    .collect::<Vec<FTensor<Rank1<ACTION_LEN>>>>()
                    .stack();
                let (old_mu, old_std, old_val) = self.target_net.forward(s.clone());

                let (old_lp, _old_entropy) =
                    mvn_batch_log_prob(old_mu, old_std.square(), a.clone());
                let advantage = (-old_val) + reward.clone();

                let (mu, std, val) = self.net.forward(s.trace(net_grads));

                let (lp, entropy) = mvn_batch_log_prob(mu, std.square(), a);
                let ratio = (lp - old_lp).exp();
                let surr1 = ratio.with_empty_tape() * advantage.clone();
                let surr2 = ratio.clamp(0.8, 1.2) * advantage;
                let policy_loss = -(surr2.minimum(surr1).mean());

                let value_loss = (val - reward).square().mean();
                let loss = policy_loss + value_loss + entropy.clone().mean() * 0.01;
                // let e = entropy.as_vec();
                // dbg!(e.iter().sum::<f32>() / e.len() as f32);

                total_loss += loss.array();

                net_grads = loss.backward();
            }
            self.optim.update(&mut self.net, &net_grads).unwrap();
            self.net.zero_grads(&mut net_grads);
        }

        self.target_net.clone_from(&self.net);
        self.net_grads = Some(net_grads);
        // rb.buf.clear();
        total_loss as f32 / AGENT_OPTIM_EPOCHS as f32
    }
}
