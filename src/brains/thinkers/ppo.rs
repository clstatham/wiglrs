#![allow(clippy::single_range_in_vec_init)]

use bevy::prelude::{Component, Resource};
use bevy_prng::ChaCha8Rng;
use bevy_rand::prelude::EntropyComponent;
use burn::{
    config::Config,
    module::{ADModule, Module, ModuleVisitor, Param, ParamId},
    nn::{
        transformer::{TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput},
        Initializer, Linear, LinearConfig,
    },
    optim::{adaptor::OptimizerAdaptor, GradientsParams, Optimizer, RMSProp, RMSPropConfig},
    record::{BinGzFileRecorder, FullPrecisionSettings},
    tensor::{backend::Backend, Tensor},
};

use burn_tch::{TchBackend, TchDevice};

use itertools::Itertools;
use std::{f32::consts::PI, marker::PhantomData};

use crate::{
    brains::{
        replay_buffer::{PpoBuffer, PpoMetadata},
        thinkers::stats::{diag, diag2d},
    },
    envs::{Observation, Params},
};
use crate::{
    envs::{Action, Env},
    FrameStack,
};

use super::{Status, Thinker};

pub type Be = burn_autodiff::ADBackendDecorator<burn_tch::TchBackend<f32>>;

pub struct MvNormal<B: Backend<FloatElem = f32>> {
    pub mu: Tensor<B, 2>,
    pub std: Tensor<B, 2>,
}

impl<B: Backend<FloatElem = f32>> MvNormal<B> {
    pub fn sample(&self, rng: &mut EntropyComponent<ChaCha8Rng>) -> Tensor<B, 2> {
        let std = self.std.clone();
        let [_, nfeat] = self.mu.shape().dims;
        let cov = std.to_data().value;
        assert!(cov.iter().all(|f| *f > 0.0), "{:?}", cov);
        let nbatch = self.mu.shape().dims[0];
        let mut sampler = burn_tensor::Distribution::<f32>::Normal(0.0, 1.0).sampler(rng);
        let samples = (0..nbatch * nfeat).map(|_| sampler.sample()).collect_vec();
        let z = Tensor::from_floats(samples.as_slice())
            .reshape([nbatch, nfeat])
            .to_device(&self.mu.device());
        // let z = Tensor::random([nbatch, nfeat]).to_device(&self.mu.device());

        self.mu.clone() + z * std
    }

    // https://online.stat.psu.edu/stat505/book/export/html/636
    pub fn log_prob(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let [nbatch, nfeat] = self.mu.shape().dims;
        let x = x.to_device(&self.mu.device());
        let first = self.std.ones_like() / (self.std.clone() * 2.0 * PI).sqrt();
        let second = {
            let a = self.std.ones_like().neg() / (self.std.clone() * 2.0);
            let b = x.clone() - self.mu.clone();
            (a * b.clone() * b).exp()
        };
        let factors = first * second;

        let mut prod = Tensor::<B, 2>::ones([nbatch, 1]).to_device(&self.mu.device());
        for i in 0..nfeat {
            prod = prod * factors.clone().slice([0..nbatch, i..i + 1]);
        }
        prod.log()
    }

    pub fn entropy(&self) -> Tensor<B, 2> {
        let cov = self.std.to_data().value;
        let [_, nfeat] = self.mu.shape().dims;
        assert!(cov.iter().all(|f| *f > 0.0), "{:?}", cov);
        let nbatch = self.mu.shape().dims[0];
        let mut g = Tensor::ones([nbatch, 1]).to_device(&self.std.device());
        for i in 0..nfeat {
            g = g * self.std.clone().slice([0..nbatch, i..i + 1]);
        }
        let second_term = (nfeat as f32 * 0.5) * (1.0 + (2.0 * PI).ln());
        g.log() * 0.5 + second_term
    }
}

pub fn sigmoid<B: Backend, const ND: usize>(x: Tensor<B, ND>) -> Tensor<B, ND> {
    x.ones_like() / (x.ones_like() + x.clone().neg().exp())
}

#[derive(Module, Debug)]
pub struct GruCell<B: Backend> {
    w_xz: Param<Tensor<B, 2>>,
    w_hz: Param<Tensor<B, 2>>,
    b_z: Param<Tensor<B, 1>>,
    w_xr: Param<Tensor<B, 2>>,
    w_hr: Param<Tensor<B, 2>>,
    b_r: Param<Tensor<B, 1>>,
    w_xh: Param<Tensor<B, 2>>,
    w_hh: Param<Tensor<B, 2>>,
    b_h: Param<Tensor<B, 1>>,
}

#[derive(Config, Debug)]
pub struct GruCellConfig {
    input_len: usize,
    hidden_len: usize,
}

impl GruCellConfig {
    pub fn init<B: Backend>(&self) -> GruCell<B> {
        let initializer = Initializer::XavierNormal { gain: 1.0 };
        GruCell {
            w_xz: Param::new(
                ParamId::new(),
                initializer.init_with(
                    [self.input_len, self.hidden_len],
                    Some(self.input_len),
                    Some(self.hidden_len),
                ),
            ),
            w_hz: Param::new(
                ParamId::new(),
                initializer.init_with(
                    [self.hidden_len, self.hidden_len],
                    Some(self.hidden_len),
                    Some(self.hidden_len),
                ),
            ),
            b_z: Param::new(ParamId::new(), Tensor::zeros([self.hidden_len])),
            w_xr: Param::new(
                ParamId::new(),
                initializer.init_with(
                    [self.input_len, self.hidden_len],
                    Some(self.input_len),
                    Some(self.hidden_len),
                ),
            ),
            w_hr: Param::new(
                ParamId::new(),
                initializer.init_with(
                    [self.hidden_len, self.hidden_len],
                    Some(self.hidden_len),
                    Some(self.hidden_len),
                ),
            ),
            b_r: Param::new(ParamId::new(), Tensor::zeros([self.hidden_len])),
            w_xh: Param::new(
                ParamId::new(),
                initializer.init_with(
                    [self.input_len, self.hidden_len],
                    Some(self.input_len),
                    Some(self.hidden_len),
                ),
            ),
            w_hh: Param::new(
                ParamId::new(),
                initializer.init_with(
                    [self.hidden_len, self.hidden_len],
                    Some(self.hidden_len),
                    Some(self.hidden_len),
                ),
            ),
            b_h: Param::new(ParamId::new(), Tensor::zeros([self.hidden_len])),
        }
    }
}

impl<B: Backend> GruCell<B> {
    pub fn forward(
        &self,
        x: Tensor<B, 2>,
        h: Option<Tensor<B, 2>>,
    ) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let dev = &self.devices()[0];
        let [nbatch, _] = x.shape().dims;
        let [nhidden] = self.b_h.shape().dims;
        let mut h = h.unwrap_or(Tensor::zeros_device([nbatch, nhidden], dev));

        let z = sigmoid(
            x.clone().matmul(self.w_xz.val())
                + h.clone().matmul(self.w_hz.val())
                + self.b_z.val().reshape([1, nhidden]),
        );
        let r = sigmoid(
            x.clone().matmul(self.w_xr.val())
                + h.clone().matmul(self.w_hr.val())
                + self.b_r.val().reshape([1, nhidden]),
        );
        let h_tilde = (x.clone().matmul(self.w_xh.val())
            + (r * h.clone()).matmul(self.w_hh.val())
            + self.b_h.val().reshape([1, nhidden]))
        .tanh();
        h = z.clone() * h + (z.ones_like() - z) * h_tilde;

        (h.clone(), h)
    }
}

pub struct CheckNanWeights;
impl<B: Backend<FloatElem = f32>> ModuleVisitor<B> for CheckNanWeights {
    fn visit<const D: usize>(&mut self, id: &ParamId, tensor: &Tensor<B, D>) {
        if tensor.to_data().value.into_iter().any(|s| s.is_nan()) {
            panic!("NaN weight detected in param id {}", id);
        }
        if tensor.to_data().value.into_iter().any(|s| !s.is_finite()) {
            panic!("Inf weight detected in param id {}", id);
        }
    }
}

#[derive(Module, Debug)]
pub struct PpoActor<B: Backend> {
    common1: Linear<B>,
    common2: Linear<B>,
    mu_head: Linear<B>,
    // std_head: Param<Tensor<B, 1>>,
    std_head: Linear<B>,
    obs_len: usize,
}

#[derive(Config, Debug)]
pub struct PpoActorConfig {
    obs_len: usize,
    hidden_len: usize,
    action_len: usize,
}

impl PpoActorConfig {
    pub fn init<B: Backend<FloatElem = f32>>(
        &self,
        _rng: &mut EntropyComponent<ChaCha8Rng>,
    ) -> PpoActor<B> {
        PpoActor {
            obs_len: self.obs_len,
            // common: TransformerEncoderConfig::new(self.obs_len, self.hidden_len, 1, 3).init(),
            common1: LinearConfig::new(self.obs_len, self.hidden_len).init(),
            common2: LinearConfig::new(self.hidden_len, self.hidden_len).init(),
            mu_head: LinearConfig::new(self.hidden_len, self.action_len)
                .with_initializer(Initializer::XavierUniform { gain: 1.0 })
                .init(),
            std_head: LinearConfig::new(self.hidden_len, self.action_len)
                .with_initializer(Initializer::XavierUniform { gain: 1.0 })
                .init(),
            // std_head: Param::new(ParamId::new(), Initializer::Zeros.init([self.action_len])),
        }
    }
}
impl<B: Backend<FloatElem = f32>> PpoActor<B> {
    pub fn forward(
        &mut self,
        x: Tensor<B, 3>,
        // common_h: &mut Tensor<B, 2>,
    ) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let x = x.to_device(&self.devices()[0]);
        let [nbatch, nstack, nfeat] = x.shape().dims;
        let x = x
            .clone()
            .slice([0..nbatch, nstack - 1..nstack, 0..nfeat])
            .squeeze(1);
        let x = self.common1.forward(x).tanh();
        let x = self.common2.forward(x).tanh();
        // let x = x.tanh();
        let mu = self.mu_head.forward(x.clone());

        // let mu = mu.tanh();

        let std = self.std_head.forward(x).exp();
        // let std = std + 1e-7;
        // let [nbatch, nstack, nfeat] = std.shape().dims;
        // let std = std
        //     .slice([0..nbatch, nstack - 1..nstack, 0..nfeat])
        //     .squeeze(1);
        // softplus
        // let std: Tensor<B, 2> = (std.exp() + 1.0).log() + 1e-7;
        (mu, std)
    }
}

#[derive(Module, Debug)]
pub struct PpoCritic<B: Backend> {
    common1: Linear<B>,
    common2: Linear<B>,
    head: Linear<B>,
    obs_len: usize,
}

#[derive(Debug, Config)]
pub struct PpoCriticConfig {
    obs_len: usize,
    hidden_len: usize,
}

impl PpoCriticConfig {
    pub fn init<B: Backend<FloatElem = f32>>(
        &self,
        _rng: &mut EntropyComponent<ChaCha8Rng>,
    ) -> PpoCritic<B> {
        PpoCritic {
            obs_len: self.obs_len,
            // common: TransformerEncoderConfig::new(self.obs_len, self.hidden_len, 1, 3).init(),
            common1: LinearConfig::new(self.obs_len, self.hidden_len).init(),
            common2: LinearConfig::new(self.hidden_len, self.hidden_len).init(),
            head: LinearConfig::new(self.hidden_len, 1).init(),
        }
    }
}

impl<B: Backend> PpoCritic<B> {
    pub fn forward(&mut self, x: Tensor<B, 3>) -> Tensor<B, 1> {
        let x = x.to_device(&self.devices()[0]);
        let [nbatch, nstack, nfeat] = x.shape().dims;
        let x: Tensor<B, 2> = x
            .slice([0..nbatch, nstack - 1..nstack, 0..nfeat])
            .squeeze(1);
        let x = self.common1.forward(x).tanh();
        let x = self.common2.forward(x).tanh();

        let x = self.head.forward(x);
        x.squeeze(1)
    }
}

#[derive(Debug, Clone)]
pub struct HiddenStates<B: Backend> {
    phantom: PhantomData<B>,
    // pub actor_com_h: Tensor<B, 2>,
    // pub critic_h: Tensor<B, 2>,
}

#[derive(Debug, Clone, Default)]
pub struct PpoStatus {
    pub recent_mu: Box<[f32]>,
    pub recent_std: Box<[f32]>,
    pub recent_entropy: f32,
    pub recent_policy_loss: f32,
    pub recent_value_loss: f32,
    pub recent_entropy_loss: f32,
    pub recent_nclamp: f32,
    pub recent_kl: f32,
}

impl Status for PpoStatus {
    fn log(&self, writer: &mut crate::TbWriter, step: usize) {
        writer.add_scalar("Policy/Loss", self.recent_policy_loss, step);
        writer.add_scalar("Policy/Entropy", self.recent_entropy_loss, step);
        writer.add_scalar("Policy/ClampRatio", self.recent_nclamp, step);
        writer.add_scalar("Value/Loss", self.recent_value_loss, step);
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
pub struct RmsNormalize<B: Backend, const D: usize> {
    mean: Tensor<B, D>,
    var: Tensor<B, D>,
    count: f32,
}

impl<B: Backend, const D: usize> RmsNormalize<B, D> {
    pub fn new(shape: burn_tensor::Shape<D>) -> Self {
        Self {
            mean: Tensor::zeros(shape.clone()),
            var: Tensor::ones(shape),
            count: 1e-4,
        }
    }

    // https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/common/running_mean_std.py#L12
    fn update(&mut self, x: Tensor<B, D>) {
        let batch_count = x.shape().dims[0] as f32;
        let batch_mean = x.mean_dim(0);

        let delta = batch_mean.clone() - self.mean.clone();
        let tot_count = batch_count + self.count;
        self.mean = self.mean.clone() + delta.clone() * batch_count / tot_count;
        let m_a = self.var.clone() * self.count;
        // let m_b = batch_var * batch_count;
        let m2 = m_a + (delta.clone() * delta) * self.count * batch_count / tot_count;
        self.var = m2 / tot_count;
        self.count = tot_count;
    }

    pub fn forward(&mut self, x: Tensor<B, D>) -> Tensor<B, D> {
        self.update(x.clone());
        (x - self.mean.clone()) / (self.var.clone() + 1e-4).sqrt()
    }
}

pub struct PpoThinker {
    actor: PpoActor<Be>,
    critic: PpoCritic<Be>,
    actor_optim: OptimizerAdaptor<RMSProp<TchBackend<f32>>, PpoActor<Be>, Be>,
    critic_optim: OptimizerAdaptor<RMSProp<TchBackend<f32>>, PpoCritic<Be>, Be>,
    status: PpoStatus,
    training_epochs: usize,
    training_batch_size: usize,
    entropy_beta: f32,
    actor_lr: f64,
    critic_lr: f64,
}

impl PpoThinker {
    pub fn new(
        obs_len: usize,
        hidden_len: usize,
        action_len: usize,
        training_epochs: usize,
        training_batch_size: usize,
        entropy_beta: f32,
        actor_lr: f64,
        critic_lr: f64,
        rng: &mut EntropyComponent<ChaCha8Rng>,
    ) -> Self {
        let actor = PpoActorConfig {
            obs_len,
            hidden_len,
            action_len,
        }
        .init(rng);
        // .fork(&TchDevice::Cuda(0));
        dbg!(actor.num_params());
        let critic = PpoCriticConfig {
            obs_len,
            hidden_len,
        }
        .init(rng);
        // .fork(&TchDevice::Cuda(0));
        dbg!(critic.num_params());
        let actor_optim = RMSPropConfig::new().with_momentum(0.0).init();
        let critic_optim = RMSPropConfig::new().with_momentum(0.0).init();
        Self {
            actor,
            critic,
            actor_optim,
            critic_optim,
            status: PpoStatus::default(),
            training_batch_size,
            training_epochs,
            entropy_beta,
            actor_lr,
            critic_lr,
        }
    }
}

impl<E: Env> Thinker<E> for PpoThinker
where
    E::Action: Action<E, Metadata = PpoMetadata>,
{
    type Metadata = HiddenStates<TchBackend<f32>>;
    type Status = PpoStatus;
    type ActionMetadata = PpoMetadata;

    fn status(&self) -> Self::Status {
        self.status.clone()
    }

    fn init_metadata(&self, _batch_size: usize) -> Self::Metadata {
        // let dev = self.actor.devices()[0];
        HiddenStates {
            phantom: PhantomData,
        }
    }

    fn act(
        &mut self,
        obs: &FrameStack<Box<[f32]>>,
        metadata: &mut Self::Metadata,
        params: &E::Params,
        rng: &mut EntropyComponent<ChaCha8Rng>,
    ) -> E::Action {
        let hiddens = metadata.clone(); //.to_device(&TchDevice::Cpu);
        let obs = obs
            .as_vec()
            .into_iter()
            .map(|o| Tensor::from_floats(&*o).unsqueeze::<2>())
            .collect::<Vec<_>>();
        let obs = Tensor::cat(obs, 0).unsqueeze();
        let (mu, std) = self.actor.valid().forward(obs.clone());
        self.status.recent_mu = mu.to_data().value.into_boxed_slice();
        self.status.recent_std = std.to_data().value.into_boxed_slice();
        let dist = MvNormal { mu, std };
        self.status.recent_entropy = dist.entropy().into_scalar();
        let action = dist.sample(rng).tanh();
        let val = self.critic.valid().forward(obs);

        let action_vec = action.to_data().value;

        let logp = dist.log_prob(action.clone()).into_scalar();
        let logp2 = dist.log_prob(action).into_scalar();
        assert_eq!(logp, logp2);

        E::Action::from_slice(
            action_vec.as_slice(),
            PpoMetadata {
                val: val.into_scalar(),
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
        let mut nstep = 0;

        let mut total_pi_loss = 0.0;
        let mut total_val_loss = 0.0;
        let mut total_entropy_loss = 0.0;
        let mut total_nclamp = 0.0;
        for _epoch in kdam::tqdm!(0..self.training_epochs, desc = "Training") {
            nstep += 1;

            let step = rb.sample_batch(self.training_batch_size, rng).unwrap();
            let s = step
                .obs
                .iter()
                .map(|stack| {
                    Tensor::cat(
                        stack
                            .as_vec()
                            .into_iter()
                            .map(|x| Tensor::from_floats(&*x).unsqueeze())
                            .collect::<Vec<_>>(),
                        1,
                    )
                })
                .collect::<Vec<_>>();
            let s: Tensor<Be, 3> = Tensor::cat(s, 0);
            let a = step
                .action
                .iter()
                .map(|action| Tensor::from_floats(&*action.as_slice(params)).unsqueeze())
                .collect::<Vec<_>>();
            let a: Tensor<Be, 2> = Tensor::cat(a, 0);
            let old_lp = step
                .action
                .iter()
                .map(|action| action.metadata().logp)
                .collect::<Vec<_>>();
            let old_lp = Tensor::from_floats(old_lp.as_slice())
                .to_device(&self.actor.devices()[0])
                .reshape([0, 1]);
            let old_val = step
                .action
                .iter()
                .map(|action| action.metadata().val)
                .collect::<Vec<_>>();
            let old_val: Tensor<Be, 2> = Tensor::from_floats(old_val.as_slice())
                .to_device(&self.actor.devices()[0])
                .reshape([0, 1]);

            let advantage = Tensor::from_floats(
                step.advantage
                    .iter()
                    .copied()
                    .map(|a| a.unwrap())
                    .collect_vec()
                    .as_slice(),
            )
            .to_device(&self.actor.devices()[0])
            .reshape([0, 1]);
            let advantage = (advantage.clone() - advantage.clone().mean().unsqueeze())
                / (advantage.var(0).sqrt() + 1e-7);
            let returns = advantage.clone() + old_val;

            let (mu, std) = self.actor.forward(s.clone());
            let dist = MvNormal { mu, std };
            let entropy = dist.entropy();
            let lp = dist.log_prob(a);
            let ratio = (lp - old_lp).exp();
            let surr1 = ratio.clone() * advantage.clone();
            let nclamp = ratio
                .clone()
                .zeros_like()
                .mask_fill(ratio.clone().lower_elem(0.8), 1.0);
            let nclamp = nclamp.mask_fill(ratio.clone().greater_elem(1.2), 1.0);
            total_nclamp += nclamp.mean().into_scalar();

            let surr2 = ratio.clamp(0.8, 1.2) * advantage;
            let masked = surr2.clone().mask_where(surr1.clone().lower(surr2), surr1);
            let policy_loss = -masked.mean();

            total_pi_loss += policy_loss.clone().into_scalar();
            let entropy_loss = entropy.mean();
            total_entropy_loss += entropy_loss.clone().into_scalar();
            let policy_loss = policy_loss - entropy_loss * self.entropy_beta;

            let actor_grads = policy_loss.backward();

            // TchTensor::new(todo!())
            //     .tensor
            //     .isnan()
            //     .any()
            //     .iter()
            //     .map(|t| t.map(|t| println!("{}", t)));
            let mut visitor = CheckNanWeights;
            self.actor = self.actor_optim.step(
                self.actor_lr,
                self.actor.clone(),
                GradientsParams::from_grads(actor_grads, &self.actor),
            );
            self.actor.visit(&mut visitor);
            let val = self.critic.forward(s);
            let val_err = val - returns.squeeze(1);
            let value_loss = (val_err.clone() * val_err).mean();
            total_val_loss += value_loss.clone().into_scalar();
            let critic_grads = value_loss.backward();
            self.critic = self.critic_optim.step(
                self.critic_lr,
                self.critic.clone(),
                GradientsParams::from_grads(critic_grads, &self.critic),
            );
            self.critic.visit(&mut visitor);
        }

        self.status.recent_policy_loss = total_pi_loss / nstep as f32;
        self.status.recent_value_loss = total_val_loss / nstep as f32;
        self.status.recent_entropy_loss = total_entropy_loss / nstep as f32;
        self.status.recent_nclamp = total_nclamp / nstep as f32;
    }

    fn save(&self, path: impl AsRef<std::path::Path>) -> Result<(), Box<dyn std::error::Error>> {
        self.actor.clone().save_file(
            path.as_ref().join("actor"),
            &BinGzFileRecorder::<FullPrecisionSettings>::new(),
        )?;
        self.critic.clone().save_file(
            path.as_ref().join("critic"),
            &BinGzFileRecorder::<FullPrecisionSettings>::new(),
        )?;
        Ok(())
    }
}
