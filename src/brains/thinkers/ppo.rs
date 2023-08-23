use dfdx::nn::modules::Linear as ModLinear;
use dfdx::optim::Adam;
use dfdx::prelude::*;
use std::collections::VecDeque;
use std::ops::{Add, Mul};

use crate::brains::replay_buffer::ReplayBuffer;
use crate::brains::FrameStack;
use crate::hparams::{
    AGENT_ACTOR_LR, AGENT_CRITIC_LR, AGENT_HIDDEN_DIM, AGENT_OPTIM_BATCH_SIZE, AGENT_OPTIM_EPOCHS,
    N_FRAME_STACK,
};
use crate::{Action, ActionMetadata, ACTION_LEN, OBS_LEN};

use super::Thinker;

pub type Device = Cuda;
type Float = f32;
const PI: Float = std::f32::consts::PI;
pub type FTensor<S, T = NoneTape> = Tensor<S, Float, Device, T>;

pub struct MvNormal<
    const K: usize,
    D: dfdx::tensor_ops::Device<Float>,
    T: Tape<Float, D> + Merge<NoneTape>,
> {
    pub mu: Tensor<Rank1<K>, Float, D, T>,
    pub cov_diag: Tensor<Rank1<K>, Float, D, T>,
}

impl<const K: usize, D: dfdx::tensor_ops::Device<Float>> MvNormal<K, D, NoneTape> {
    pub fn sample(&self) -> Tensor<Rank1<K>, Float, D, NoneTape> {
        let cov_mat = nalgebra::DMatrix::from_diagonal(&nalgebra::DVector::from_iterator(
            K,
            self.cov_diag.as_vec().into_iter(),
        ));
        let a = cov_mat.cholesky().unwrap().unpack();
        let z = self.mu.device().sample_normal::<Rank1<K>>();
        let z = nalgebra::DVector::from_iterator(K, z.as_vec().into_iter());
        let x = a.mul(&z);
        let x = self.mu.device().tensor(x.data.as_vec().clone());
        self.mu.clone() + x
    }
}

impl<const K: usize, D: dfdx::tensor_ops::Device<Float>, T: Tape<Float, D>> MvNormal<K, D, T>
where
    T: Merge<NoneTape>,
{
    pub fn log_prob(self, x: Tensor<Rank1<K>, Float, D>) -> (Tensor<Rank1<1>, Float, D, T>, Float) {
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

    pub fn entropy(&self) -> Float {
        let cov_mat = nalgebra::DMatrix::from_diagonal(&nalgebra::DVector::from_iterator(
            K,
            self.cov_diag.as_vec().into_iter(),
        ));
        let g = cov_mat.determinant();
        (K as Float / 2.0) + ((K as Float / 2.0) * (2.0 * PI).ln()) + (g.ln() / 2.0)
    }
}

fn mvn_batch_log_prob<
    const K: usize,
    D: dfdx::tensor_ops::Device<Float>,
    T: Tape<Float, D> + Merge<NoneTape>,
>(
    mu: Tensor<(usize, Const<K>), Float, D, T>,
    var: Tensor<(usize, Const<K>), Float, D, T>,
    x: Tensor<(usize, Const<K>), Float, D>,
) -> (
    Tensor<(usize, Const<1>), Float, D, T>,
    Tensor<(usize, Const<1>), Float, D>,
) {
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
struct PpoActor<
    const OBS_LEN: usize,
    const HIDDEN_DIM: usize,
    const ACTION_LEN: usize,
    E: Dtype,
    D: dfdx::tensor_ops::Device<E>,
> {
    common: (
        ModLinear<OBS_LEN, HIDDEN_DIM, E, D>,
        ReLU,
        ModLinear<HIDDEN_DIM, HIDDEN_DIM, E, D>,
        ReLU,
    ),
    mu_head: (
        ModLinear<HIDDEN_DIM, HIDDEN_DIM, E, D>,
        ReLU,
        ModLinear<HIDDEN_DIM, ACTION_LEN, E, D>,
        Tanh,
    ),
    std_head: (
        ModLinear<HIDDEN_DIM, HIDDEN_DIM, E, D>,
        ReLU,
        ModLinear<HIDDEN_DIM, ACTION_LEN, E, D>,
    ),
}

impl<
        const OBS_LEN: usize,
        const HIDDEN_DIM: usize,
        const ACTION_LEN: usize,
        E: Dtype + num_traits::Float + rand_distr::uniform::SampleUniform,
        D: dfdx::tensor_ops::Device<E>,
    > TensorCollection<E, D> for PpoActor<OBS_LEN, HIDDEN_DIM, ACTION_LEN, E, D>
{
    type To<E2: Dtype, D2: dfdx::tensor_ops::Device<E2>> =
        PpoActor<OBS_LEN, HIDDEN_DIM, ACTION_LEN, E2, D2>;

    fn iter_tensors<V: ModuleVisitor<Self, E, D>>(
        visitor: &mut V,
    ) -> Result<Option<Self::To<V::E2, V::D2>>, V::Err> {
        visitor.visit_fields(
            (
                Self::module("common", |s| &s.common, |s| &mut s.common),
                Self::module("mu_head", |s| &s.mu_head, |s| &mut s.mu_head),
                Self::module("std_head", |s| &s.std_head, |s| &mut s.std_head),
            ),
            |(c, mu, std)| Self::To {
                common: c,
                mu_head: mu,
                std_head: std,
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
    for PpoActor<OBS_LEN, HIDDEN_DIM, ACTION_LEN, E, D>
{
    type Error = D::Err;
    type Output = (
        Tensor<(Const<ACTION_LEN>,), E, D, T>,
        Tensor<(Const<ACTION_LEN>,), E, D, T>,
    );
    fn try_forward(
        &self,
        input: Tensor<(usize, Const<OBS_LEN>), E, D, T>,
    ) -> Result<Self::Output, Self::Error> {
        let nstack = input.shape().concrete()[0];
        let x = self.common.try_forward(input)?;
        let idx = x.device().tensor(nstack - 1);
        let x = x.try_select(idx)?;
        let std = self.std_head.try_forward(x.with_empty_tape())?;
        let std = softplus(std);
        let mu = self.mu_head.try_forward(x)?;
        Ok((mu, std))
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
    for PpoActor<OBS_LEN, HIDDEN_DIM, ACTION_LEN, E, D>
{
    type Error = D::Err;
    type Output = (
        Tensor<(usize, Const<ACTION_LEN>), E, D, T>,
        Tensor<(usize, Const<ACTION_LEN>), E, D, T>,
    );
    fn try_forward(
        &self,
        input: Tensor<(usize, usize, Const<OBS_LEN>), E, D, T>,
    ) -> Result<Self::Output, Self::Error> {
        let nbatch = input.shape().0;
        let nstack = input.shape().1;
        let x = self.common.try_forward(input)?;
        let x = x
            .slice((0..nbatch, nstack - 1..nstack, 0..HIDDEN_DIM))
            .reshape_like(&(nbatch, Const::<HIDDEN_DIM>));
        let std = self.std_head.try_forward(x.with_empty_tape())?;
        let std = softplus(std);
        let mu = self.mu_head.try_forward(x)?;
        Ok((mu, std))
    }
}

type PpoA = PpoActor<OBS_LEN, AGENT_HIDDEN_DIM, ACTION_LEN, Float, Device>;

#[rustfmt::skip]
type PpoCritic = (
    (Linear<OBS_LEN, AGENT_HIDDEN_DIM>, ReLU),
    (Linear<AGENT_HIDDEN_DIM, AGENT_HIDDEN_DIM>, ReLU,),
    (Linear<AGENT_HIDDEN_DIM, AGENT_HIDDEN_DIM>, ReLU,),
    (Linear<AGENT_HIDDEN_DIM, 1>,),
    //
);
type PpoC = <PpoCritic as BuildOnDevice<Device, Float>>::Built;

pub struct PpoThinker {
    actor: PpoA,
    critic: PpoC,
    actor_grads: Option<Gradients<Float, Device>>,
    critic_grads: Option<Gradients<Float, Device>>,
    actor_optim: Adam<PpoA, Float, Device>,
    critic_optim: Adam<PpoC, Float, Device>,
    device: Device,
    cpu: Cpu,
    pub recent_mu: Vec<f32>,
    pub recent_std: Vec<f32>,
    pub recent_entropy: f32,
    pub recent_policy_loss: f32,
    pub recent_value_loss: f32,
    pub recent_entropy_loss: f32,
}

impl PpoThinker {
    pub fn new() -> Self {
        let device = Device::seed_from_u64(rand::random());
        let actor = PpoA::build(&device);
        let actor_grads = actor.alloc_grads();
        let critic = device.build_module::<PpoCritic, Float>();
        let critic_grads = critic.alloc_grads();

        Self {
            cpu: Cpu::seed_from_u64(rand::random()),
            actor_optim: Adam::new(
                &actor,
                AdamConfig {
                    lr: AGENT_ACTOR_LR,
                    ..Default::default()
                },
            ),
            actor,
            actor_grads: Some(actor_grads),
            // target_actor,
            critic_optim: Adam::new(
                &critic,
                AdamConfig {
                    lr: AGENT_CRITIC_LR,
                    ..Default::default()
                },
            ),
            critic,
            critic_grads: Some(critic_grads),
            // target_critic,
            device,
            recent_mu: Vec::new(),
            recent_std: Vec::new(),
            recent_entropy: 0.0,
            recent_policy_loss: 0.0,
            recent_value_loss: 0.0,
            recent_entropy_loss: 0.0,
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
        let (mu, std) = self.actor.forward(obs.clone());
        let mu = mu.to_device(&self.cpu);
        let std = std.to_device(&self.cpu);
        self.recent_mu = mu.as_vec();
        self.recent_std = std.as_vec();
        let dist = MvNormal {
            mu,
            cov_diag: std.square(),
        };
        self.recent_entropy = dist.entropy();
        let action = dist.sample();
        let val = self.critic.forward(obs);
        let val = val
            .slice((N_FRAME_STACK - 1..N_FRAME_STACK, 0..1))
            .reshape_like(&(Const::<1>,));

        Action::from_slice(
            action.as_vec().as_slice(),
            Some(ActionMetadata {
                logp: dist.log_prob(action).0.array()[0],
                val: val.array()[0],
            }),
        )
    }

    fn learn<const MAX_LEN: usize>(&mut self, rb: &mut ReplayBuffer<MAX_LEN>) {
        let nstep = rb.buf.len();
        // if nstep < AGENT_RB_MAX_LEN {
        //     return; // not enough data to train yet
        // }
        let mut discounted_reward = 0.0;
        let mut rewards = VecDeque::new();

        for step in rb.buf.iter().rev() {
            if step.terminal {
                discounted_reward = 0.0;
            }
            discounted_reward = step.reward + 0.99 * discounted_reward;
            rewards.push_front(discounted_reward);
        }
        let rewards = self.device.tensor_from_vec(rewards.into(), (nstep,));
        let rewards = rewards.normalize(1e-7).as_vec();

        let mut total_pi_loss = 0.0;
        let mut total_val_loss = 0.0;
        let mut total_entropy_loss = 0.0;
        let mut actor_grads = self.actor_grads.take().unwrap();
        let mut critic_grads = self.critic_grads.take().unwrap();
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
                let old_lp = step
                    .iter()
                    .map(|step| {
                        self.device
                            .tensor(step.action.metadata.unwrap().logp)
                            .broadcast()
                    })
                    .collect::<Vec<FTensor<Rank1<1>>>>()
                    .stack();
                let old_val = step
                    .iter()
                    .map(|step| {
                        self.device
                            .tensor(step.action.metadata.unwrap().val)
                            .broadcast()
                    })
                    .collect::<Vec<FTensor<Rank1<1>>>>()
                    .stack();

                // let (old_mu, old_std) = self.target_actor.forward(s.clone());
                // let old_val = self.target_critic.forward(s.clone());
                // let old_val = old_val
                //     .slice((0..nbatch, N_FRAME_STACK - 1..N_FRAME_STACK, 0..1))
                //     .reshape_like(&(nbatch, Const));

                // let (old_lp, _old_entropy) =
                //     mvn_batch_log_prob(old_mu, old_std.square(), a.clone());
                let advantage = (-old_val) + reward.clone();

                let (mu, std) = self.actor.forward(s.trace(actor_grads));
                let val = self.critic.forward(s.trace(critic_grads));
                let val = val
                    .slice((0..nbatch, N_FRAME_STACK - 1..N_FRAME_STACK, 0..1))
                    .reshape_like(&(nbatch, Const));

                let (lp, entropy) = mvn_batch_log_prob(mu, std.square(), a);
                let ratio = (lp - old_lp).exp();
                let surr1 = ratio.with_empty_tape() * advantage.clone();
                let surr2 = ratio.clamp(0.8, 1.2) * advantage;
                let policy_loss = -(surr2.minimum(surr1).mean());
                let value_loss = (val - reward).square().mean();
                total_val_loss += value_loss.array();
                total_pi_loss += policy_loss.array();
                let entropy_loss = entropy.mean() * 1e-5;
                total_entropy_loss += entropy_loss.array();
                let policy_loss = policy_loss - entropy_loss;

                actor_grads = policy_loss.backward();
                self.actor_optim
                    .update(&mut self.actor, &actor_grads)
                    .unwrap();
                self.actor.zero_grads(&mut actor_grads);

                critic_grads = value_loss.backward();
                self.critic_optim
                    .update(&mut self.critic, &critic_grads)
                    .unwrap();
                self.critic.zero_grads(&mut critic_grads);
            }
        }
        self.recent_policy_loss = total_pi_loss / nstep as f32;
        self.recent_value_loss = total_val_loss / nstep as f32;
        self.recent_entropy_loss = total_entropy_loss / nstep as f32;
        self.actor_grads = Some(actor_grads);
        self.critic_grads = Some(critic_grads);
        // rb.buf.clear();
    }

    fn save(&self, path: impl AsRef<std::path::Path>) -> Result<(), Box<dyn std::error::Error>> {
        self.actor
            .save_safetensors(path.as_ref().join("actor.safetensors"))?;
        self.critic
            .save_safetensors(path.as_ref().join("critic.safetensors"))?;
        Ok(())
    }
}
