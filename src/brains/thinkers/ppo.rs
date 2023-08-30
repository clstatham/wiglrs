#![allow(clippy::single_range_in_vec_init)]

use burn::{
    config::Config,
    module::{ADModule, Module, ModuleVisitor, Param, ParamId},
    nn::{Initializer, Linear, LinearConfig},
    optim::{adaptor::OptimizerAdaptor, GradientsParams, Optimizer, RMSProp, RMSPropConfig},
    record::{BinGzFileRecorder, FullPrecisionSettings},
    tensor::{backend::Backend, Tensor},
};

use burn_tch::{TchBackend, TchDevice};

use itertools::Itertools;
use std::f32::consts::PI;

use crate::{
    brains::replay_buffer::{PpoBuffer, PpoMetadata},
    envs::{Observation, Params},
};
use crate::{
    envs::{Action, Env},
    FrameStack,
};

use super::{
    ncp::{Cfc, CfcCellMode, CfcConfig, CfcMode, FullyConnected, WiringConfig},
    Status, Thinker,
};

pub type Be = burn_autodiff::ADBackendDecorator<burn_tch::TchBackend<f32>>;

pub struct MvNormal<B: Backend<FloatElem = f32>> {
    pub mu: Tensor<B, 2>,
    pub cov_diag: Tensor<B, 2>,
}

impl<B: Backend<FloatElem = f32>> MvNormal<B> {
    pub fn sample(&self) -> Tensor<B, 2> {
        let std = self.cov_diag.clone().sqrt();
        let [_, nfeat] = self.mu.shape().dims;
        let cov = std.to_data().value;
        assert!(cov.iter().all(|f| *f > 0.0), "{:?}", cov);
        let nbatch = self.mu.shape().dims[0];
        let z = Tensor::random([nbatch, nfeat], burn_tensor::Distribution::Normal(0.0, 1.0))
            .to_device(&self.mu.device());
        self.mu.clone() + z * std
    }

    pub fn log_prob(self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let [_, nfeat] = self.mu.shape().dims;
        let x = x.to_device(&self.mu.device());
        let a = x.clone() - self.mu.clone();
        let cov = self.cov_diag.to_data().value;
        assert!(cov.iter().all(|f| *f > 0.0), "{:?}", cov);
        let cov_diag = self.cov_diag.clone();
        let nbatch = self.mu.shape().dims[0];
        let mut det = Tensor::<B, 2>::ones([nbatch, 1]).to_device(&self.cov_diag.device());
        for i in 0..nfeat {
            det = det * cov_diag.clone().slice([0..nbatch, i..i + 1]);
        }
        let d = a.clone() / cov_diag;
        let e = (d * a).sum_dim(1);
        let numer = (e * -0.5).exp().reshape([nbatch, 1]);
        let f = (2.0 * PI).powf(nfeat as f32);
        let denom = (det * f).sqrt();
        let pdf = numer / denom;
        pdf.log()
    }

    pub fn entropy(&self) -> Tensor<B, 2> {
        let cov = self.cov_diag.to_data().value;
        let [_, nfeat] = self.mu.shape().dims;
        assert!(cov.iter().all(|f| *f > 0.0), "{:?}", cov);
        let nbatch = self.mu.shape().dims[0];
        let mut g = Tensor::ones([nbatch, 1]).to_device(&self.cov_diag.device());
        for i in 0..nfeat {
            g = g * self.cov_diag.clone().slice([0..nbatch, i..i + 1]);
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
    common: Cfc<B>,
    mu_head: Linear<B>,
    std_head: Linear<B>,
    obs_len: usize,
    com_units: usize,
}

#[derive(Config, Debug)]
pub struct PpoActorConfig {
    obs_len: usize,
    hidden_len: usize,
    action_len: usize,
}

impl PpoActorConfig {
    pub fn init<B: Backend<FloatElem = f32>>(&self) -> PpoActor<B> {
        let wiring_com = FullyConnected::new(self.obs_len, self.hidden_len, true);

        PpoActor {
            obs_len: self.obs_len,
            com_units: wiring_com.units(),
            common: CfcConfig::new(self.obs_len, self.hidden_len).init(
                wiring_com,
                CfcMode::MixedMemory,
                CfcCellMode::Default,
            ),
            mu_head: LinearConfig::new(self.hidden_len, self.action_len).init(),
            std_head: LinearConfig::new(self.hidden_len, self.action_len).init(),
        }
    }
}
impl<B: Backend<FloatElem = f32>> PpoActor<B> {
    pub fn forward(
        &mut self,
        x: Tensor<B, 3>,
        common_h: &mut Tensor<B, 2>,
    ) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let x = x.to_device(&self.devices()[0]);

        let (x, ch) = self.common.forward(x, Some(common_h.clone()));
        *common_h = ch;
        // let x = x.tanh();
        let mu = self.mu_head.forward(x.clone());
        let [nbatch, nstack, nfeat] = mu.shape().dims;
        let mu = mu
            .clone()
            .slice([0..nbatch, nstack - 1..nstack, 0..nfeat])
            .squeeze(1);
        let mu = mu.tanh();

        let std = self.std_head.forward(x);
        let [nbatch, nstack, nfeat] = std.shape().dims;
        let std = std
            .slice([0..nbatch, nstack - 1..nstack, 0..nfeat])
            .squeeze(1);
        // softplus
        let std: Tensor<B, 2> = (std.exp() + 1.0).log() + 1e-7;
        (mu, std)
    }
}

#[derive(Module, Debug)]
pub struct PpoCritic<B: Backend> {
    rnn: Cfc<B>,
    obs_len: usize,
    rnn_units: usize,
}

#[derive(Debug, Config)]
pub struct PpoCriticConfig {
    obs_len: usize,
    hidden_len: usize,
}

impl PpoCriticConfig {
    pub fn init<B: Backend<FloatElem = f32>>(&self) -> PpoCritic<B> {
        let wiring = FullyConnected::new(self.obs_len, self.hidden_len, true);
        PpoCritic {
            obs_len: self.obs_len,
            rnn_units: wiring.units(),
            rnn: CfcConfig::new(self.obs_len, self.hidden_len)
                .with_projected_len(1)
                .init(wiring, CfcMode::MixedMemory, CfcCellMode::Default),
        }
    }
}

impl<B: Backend> PpoCritic<B> {
    pub fn forward(&mut self, x: Tensor<B, 3>, h: &mut Tensor<B, 2>) -> Tensor<B, 1> {
        let x = x.to_device(&self.devices()[0]);
        let (x, h_new) = self.rnn.forward(x, Some(h.clone()));
        *h = h_new;
        let [nbatch, nstack, nfeat] = x.shape().dims;
        let x: Tensor<B, 2> = x
            .slice([0..nbatch, nstack - 1..nstack, 0..nfeat])
            .squeeze(1);
        x.squeeze(1)
    }
}

#[derive(Debug, Module)]
pub struct HiddenStates<B: Backend> {
    pub actor_com_h: Tensor<B, 2>,
    pub critic_h: Tensor<B, 2>,
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
    ) -> Self {
        let actor = PpoActorConfig {
            obs_len,
            hidden_len,
            action_len,
        }
        .init()
        .fork(&TchDevice::Cuda(0));
        dbg!(actor.num_params());
        let critic = PpoCriticConfig {
            obs_len,
            hidden_len,
        }
        .init()
        .fork(&TchDevice::Cuda(0));
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

    fn init_metadata(&self, batch_size: usize) -> Self::Metadata {
        let dev = self.actor.devices()[0];
        HiddenStates {
            actor_com_h: Tensor::zeros_device([batch_size, self.actor.com_units], &dev),
            critic_h: Tensor::zeros_device([batch_size, self.critic.rnn_units], &dev),
        }
    }

    fn act(
        &mut self,
        obs: &FrameStack<E::Observation>,
        metadata: &mut Self::Metadata,
        params: &E::Params,
    ) -> Option<E::Action> {
        let hiddens = metadata.clone().to_device(&TchDevice::Cpu);
        let obs = obs
            .as_vec()
            .into_iter()
            .map(|o| Tensor::from_floats(&*o.as_slice(params)).unsqueeze::<2>())
            .collect::<Vec<_>>();
        let obs = Tensor::cat(obs, 0).unsqueeze();
        let (mu, std) = self
            .actor
            .valid()
            .forward(obs.clone(), &mut metadata.actor_com_h);
        self.status.recent_mu = mu.to_data().value.into_boxed_slice();
        self.status.recent_std = std.to_data().value.into_boxed_slice();
        let dist = MvNormal {
            mu,
            cov_diag: std.clone() * std,
        };
        self.status.recent_entropy = dist.entropy().into_scalar();
        let action = dist.sample().clamp(-1.0, 1.0);
        let val = self.critic.valid().forward(obs, &mut metadata.critic_h);

        let action_vec = action.to_data().value;
        Some(E::Action::from_slice(
            action_vec.as_slice(),
            PpoMetadata {
                val: val.into_scalar(),
                logp: dist.log_prob(action).into_scalar(),
                hiddens: Some(hiddens),
            },
            params,
        ))
    }

    fn learn(&mut self, rb: &PpoBuffer<E>, params: &E::Params) {
        let mut nstep = 0;

        let mut total_pi_loss = 0.0;
        let mut total_val_loss = 0.0;
        let mut total_entropy_loss = 0.0;
        let mut total_nclamp = 0.0;
        for _epoch in kdam::tqdm!(0..self.training_epochs, desc = "Training") {
            nstep += 1;

            let step = rb.sample_batch(self.training_batch_size).unwrap();
            let returns = Tensor::from_floats(
                step.returns
                    .iter()
                    .copied()
                    .map(|r| r.unwrap())
                    .collect_vec()
                    .as_slice(),
            )
            .to_device(&self.actor.devices()[0]);
            let returns = (returns.clone() - returns.clone().mean().unsqueeze())
                / (returns.var(0).sqrt() + 1e-7);

            let s = step
                .obs
                .iter()
                .map(|stack| {
                    Tensor::cat(
                        stack
                            .as_vec()
                            .into_iter()
                            .map(|x| Tensor::from_floats(&*x.as_slice(params)).unsqueeze())
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
                .reshape([self.training_batch_size, 1]);

            let advantage = Tensor::from_floats(
                step.advantage
                    .iter()
                    .copied()
                    .map(|a| a.unwrap())
                    .collect_vec()
                    .as_slice(),
            )
            .to_device(&self.actor.devices()[0])
            .reshape([self.training_batch_size, 1]);
            let advantage = (advantage.clone() - advantage.clone().mean().unsqueeze())
                / (advantage.var(0).sqrt() + 1e-7);

            let (actor_com_h, critic_h): (Vec<_>, Vec<_>) = step
                .action
                .into_iter()
                .map(|hs| {
                    let hiddens = hs.metadata().hiddens.as_ref().unwrap().to_owned();
                    (hiddens.actor_com_h, hiddens.critic_h)
                })
                .multiunzip();

            let dev = &self.actor.devices()[0];
            let mut actor_com_h = Tensor::from_inner(Tensor::cat(actor_com_h, 0)).to_device(dev);
            let mut critic_h = Tensor::from_inner(Tensor::cat(critic_h, 0)).to_device(dev);
            let (mu, std) = self.actor.forward(s.clone(), &mut actor_com_h);
            let dist = MvNormal {
                mu,
                cov_diag: std.clone() * std,
            };
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
            let val = self.critic.forward(s, &mut critic_h);
            let val_err = val - returns;
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
