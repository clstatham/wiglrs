#![allow(clippy::single_range_in_vec_init)]

use burn::{
    config::Config,
    module::{Module, Param, ParamId},
    nn::{
        transformer::{TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput},
        Initializer, Linear, LinearConfig,
    },
    optim::{adaptor::OptimizerAdaptor, GradientsParams, Optimizer, RMSProp, RMSPropConfig},
    record::{BinGzFileRecorder, FullPrecisionSettings},
    tensor::{backend::Backend, Tensor},
};

use burn_tch::{TchBackend, TchDevice};

use burn_tensor::backend::ADBackend;
use std::{
    collections::VecDeque,
    f32::consts::{E, PI},
};

use crate::brains::replay_buffer::ReplayBuffer;
use crate::brains::FrameStack;
use crate::hparams::{
    AGENT_ACTOR_LR, AGENT_CRITIC_LR, AGENT_HIDDEN_DIM, AGENT_OPTIM_BATCH_SIZE, AGENT_OPTIM_EPOCHS,
};
use crate::{Action, ActionMetadata, ACTION_LEN, OBS_LEN};

use super::Thinker;

pub type Be = burn_autodiff::ADBackendDecorator<burn_tch::TchBackend<f32>>;

pub struct MvNormal<const K: usize> {
    pub mu: Tensor<Be, 2>,
    pub cov_diag: Tensor<Be, 2>,
}

impl<const K: usize> MvNormal<K> {
    pub fn sample(&self) -> Tensor<Be, 2> {
        let std = self.cov_diag.clone().sqrt();
        let cov = std.to_data().value;
        assert!(cov.iter().all(|f| *f > 0.0), "{:?}", cov);
        let nbatch = self.mu.shape().dims[0];
        let z = Tensor::random([nbatch, K], burn_tensor::Distribution::Normal(0.0, 1.0))
            .to_device(&self.mu.device());
        self.mu.clone() + z * std
    }

    pub fn log_prob(self, x: Tensor<Be, 2>) -> Tensor<Be, 2> {
        let x = x.to_device(&self.mu.device());
        let a = x.clone() - self.mu.clone();
        let cov = self.cov_diag.to_data().value;
        assert!(cov.iter().all(|f| *f > 0.0), "{:?}", cov);
        let cov_diag = self.cov_diag.clone();
        let nbatch = self.mu.shape().dims[0];
        let mut det = Tensor::ones([nbatch, 1]).to_device(&self.cov_diag.device());
        for i in 0..K {
            det = det * cov_diag.clone().slice([0..nbatch, i..i + 1]);
        }
        let d = a.clone() / cov_diag;
        let e = (d * a).sum_dim(1);
        let numer = (e * -0.5).exp().reshape([nbatch, 1]);
        let f = (2.0 * PI).powf(K as f32);
        let denom = (det * f).sqrt();
        let pdf = numer / denom;
        // let non_nans = pdf.clone().equal(pdf.clone()); // filter nans
        // let ones = pdf.ones_like();
        // let pdf = ones.mask_where(non_nans, pdf);
        // pdf.log()
        pdf.log()
    }

    pub fn entropy(&self) -> Tensor<Be, 2> {
        let cov = self.cov_diag.to_data().value;
        assert!(cov.iter().all(|f| *f > 0.0), "{:?}", cov);
        let nbatch = self.mu.shape().dims[0];
        let mut g = Tensor::ones([nbatch, 1]).to_device(&self.cov_diag.device());
        let two_pi_e_sigma = self.cov_diag.clone() * E * 2.0 * PI;
        for i in 0..K {
            g = g * two_pi_e_sigma.clone().slice([0..nbatch, i..i + 1]);
        }
        g.log() * 0.5
    }
}

pub fn sigmoid<B: Backend, const ND: usize>(x: Tensor<B, ND>) -> Tensor<B, ND> {
    x.ones_like() / (x.ones_like() + x.clone().neg().exp())
}

#[derive(Module, Debug)]
pub struct Gru<B: Backend> {
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
pub struct GruConfig {
    input_len: usize,
    hidden_len: usize,
}

impl GruConfig {
    pub fn init<B: ADBackend>(&self) -> Gru<B> {
        let initializer = Initializer::XavierNormal { gain: 1.0 };
        Gru {
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

impl<B: Backend> Gru<B> {
    pub fn forward(
        &self,
        xs: Tensor<B, 3>,
        h: Option<Tensor<B, 2>>,
    ) -> (Tensor<B, 3>, Tensor<B, 2>) {
        let dev = &self.devices()[0];
        let [nbatch, nseq, nfeat] = xs.shape().dims;
        let [nhidden] = self.b_h.shape().dims;
        let mut h = h.unwrap_or(Tensor::zeros([nbatch, nhidden])).to_device(dev);
        let mut outputs = Tensor::zeros([nbatch, nseq, nhidden]).to_device(dev);

        for i in 0..nseq {
            let x: Tensor<B, 2> = xs.clone().slice([0..nbatch, i..i + 1, 0..nfeat]).squeeze(1);
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
            outputs = outputs.slice_assign(
                [0..nbatch, i..i + 1, 0..nhidden],
                h.clone().reshape([nbatch, 1, nhidden]),
            );
        }

        (outputs, h)
    }
}

#[derive(Module, Debug)]
pub struct PpoActor<B: Backend> {
    common: Gru<B>,
    mu_head1: TransformerEncoder<B>,
    mu_head2: Linear<B>,
    std_head1: TransformerEncoder<B>,
    std_head2: Linear<B>,
}

#[derive(Config, Debug)]
pub struct PpoActorConfig {
    obs_len: usize,
    hidden_len: usize,
    action_len: usize,
}

impl PpoActorConfig {
    pub fn init<B: ADBackend>(&self) -> PpoActor<B> {
        PpoActor {
            common: GruConfig::new(self.obs_len, self.hidden_len).init(),
            mu_head1: TransformerEncoderConfig::new(self.hidden_len, self.hidden_len, 1, 1).init(),
            mu_head2: LinearConfig::new(self.hidden_len, self.action_len).init(),
            std_head1: TransformerEncoderConfig::new(self.hidden_len, self.hidden_len, 1, 1).init(),
            std_head2: LinearConfig::new(self.hidden_len, self.action_len).init(),
        }
    }
}
impl<B: Backend> PpoActor<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let x = x.to_device(&self.devices()[0]);

        let (x, _) = self.common.forward(x, None);

        let mu = self
            .mu_head1
            .forward(TransformerEncoderInput::new(x.clone()));
        let [nbatch, nstack, nfeat] = mu.shape().dims;
        let mu = mu
            .slice([0..nbatch, nstack - 1..nstack, 0..nfeat])
            .squeeze(1);
        let mu = self.mu_head2.forward(mu).tanh();

        let std = self.std_head1.forward(TransformerEncoderInput::new(x));
        let [nbatch, nstack, nfeat] = std.shape().dims;
        let std = std
            .slice([0..nbatch, nstack - 1..nstack, 0..nfeat])
            .squeeze(1);
        let std = self.std_head2.forward(std);
        // softplus
        let std: Tensor<B, 2> = (std.exp() + 1.0).log();

        (mu, std)
    }
}

#[derive(Module, Debug)]
pub struct PpoCritic<B: Backend> {
    rnn: Gru<B>,
    transformer: TransformerEncoder<B>,
    head: Linear<B>,
}

#[derive(Debug, Config)]
pub struct PpoCriticConfig {
    obs_len: usize,
    hidden_len: usize,
}

impl PpoCriticConfig {
    pub fn init<B: ADBackend>(&self) -> PpoCritic<B> {
        PpoCritic {
            rnn: GruConfig::new(self.obs_len, self.hidden_len).init(),
            transformer: TransformerEncoderConfig::new(self.hidden_len, self.hidden_len, 1, 1)
                .init(),
            head: LinearConfig::new(self.hidden_len, 1).init(),
        }
    }
}

impl<B: Backend> PpoCritic<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 1> {
        let x = x.to_device(&self.devices()[0]);
        let (x, _) = self.rnn.forward(x, None);
        let x = self.transformer.forward(TransformerEncoderInput::new(x));
        let [nbatch, nstack, nfeat] = x.shape().dims;
        let x: Tensor<B, 2> = x
            .slice([0..nbatch, nstack - 1..nstack, 0..nfeat])
            .squeeze(1);
        let x = self.head.forward(x);
        x.squeeze(1)
    }
}

pub struct PpoThinker {
    actor: PpoActor<Be>,
    critic: PpoCritic<Be>,
    actor_optim: OptimizerAdaptor<RMSProp<TchBackend<f32>>, PpoActor<Be>, Be>,
    critic_optim: OptimizerAdaptor<RMSProp<TchBackend<f32>>, PpoCritic<Be>, Be>,
    pub recent_mu: Vec<f32>,
    pub recent_std: Vec<f32>,
    pub recent_entropy: f32,
    pub recent_policy_loss: f32,
    pub recent_value_loss: f32,
    pub recent_entropy_loss: f32,
    pub recent_nclamp: f32,
    pub recent_kl: f32,
}

impl PpoThinker {
    pub fn new() -> Self {
        let actor = PpoActorConfig {
            obs_len: OBS_LEN,
            hidden_len: AGENT_HIDDEN_DIM,
            action_len: ACTION_LEN,
        }
        .init()
        .fork(&TchDevice::Cuda(0));
        let critic = PpoCriticConfig {
            obs_len: OBS_LEN,
            hidden_len: AGENT_HIDDEN_DIM,
        }
        .init()
        .fork(&TchDevice::Cuda(0));
        let actor_optim = RMSPropConfig::new().with_momentum(0.0).init();
        let critic_optim = RMSPropConfig::new().with_momentum(0.0).init();
        Self {
            actor,
            critic,
            actor_optim,
            critic_optim,
            recent_mu: Vec::new(),
            recent_std: Vec::new(),
            recent_entropy: 0.0,
            recent_policy_loss: 0.0,
            recent_value_loss: 0.0,
            recent_entropy_loss: 0.0,
            recent_nclamp: 0.0,
            recent_kl: 0.0,
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
        let obs = obs
            .as_vec()
            .into_iter()
            .map(|o| Tensor::from_floats(o.as_vec().as_slice()).unsqueeze::<2>())
            .collect::<Vec<_>>();
        let obs = Tensor::cat(obs, 0).unsqueeze();
        let (mu, std) = self.actor.forward(obs.clone());

        self.recent_mu = mu.to_data().value;
        self.recent_std = std.to_data().value;
        let dist: MvNormal<ACTION_LEN> = MvNormal {
            mu,
            cov_diag: std.clone() * std,
        };
        self.recent_entropy = dist.entropy().into_scalar();
        let action = dist.sample().clamp(-1.0, 1.0);
        let val = self.critic.forward(obs);

        Action::from_slice(
            action.to_data().value.as_slice(),
            Some(ActionMetadata {
                logp: dist.log_prob(action).into_scalar(),
                val: val.into_scalar(),
            }),
        )
    }

    fn learn<const MAX_LEN: usize>(&mut self, rb: &mut ReplayBuffer<MAX_LEN>) {
        let nstep = rb.buf.len() - 1;
        let mut gae = 0.0;
        let mut returns = VecDeque::new();

        for i in (0..nstep).rev() {
            let mask = if rb.buf[i].terminal { 0.0 } else { 1.0 };
            let delta = rb.buf[i].reward + 0.99 * rb.buf[i + 1].action.metadata.unwrap().val * mask
                - rb.buf[i].action.metadata.unwrap().val;
            gae = delta + 0.99 * 0.95 * mask * gae;
            returns.push_front(gae + rb.buf[i].action.metadata.unwrap().val);
        }
        let returns: Vec<f32> = returns.into();

        let returns: Tensor<Be, 1> = Tensor::from_floats(returns.as_slice());
        let returns = (returns.clone() - returns.clone().mean()) / (returns.var(0) + 1e-7);
        let returns = returns.to_data().value;

        let mut total_pi_loss = 0.0;
        let mut total_val_loss = 0.0;
        let mut total_entropy_loss = 0.0;
        let mut total_nclamp = 0.0;
        let total_batches =
            (nstep as f32 / AGENT_OPTIM_BATCH_SIZE as f32).ceil() as usize * AGENT_OPTIM_EPOCHS;
        let rb: Vec<_> = rb.buf.clone().into();
        for epoch in kdam::tqdm!(0..AGENT_OPTIM_EPOCHS, desc = "Training", position = 0) {
            let dsc = format!("Epoch {}", epoch + 1);
            for (step, returns) in kdam::tqdm!(
                rb[..nstep]
                    .chunks(AGENT_OPTIM_BATCH_SIZE)
                    .zip(returns.chunks(AGENT_OPTIM_BATCH_SIZE)),
                desc = dsc,
                position = 1
            ) {
                let nbatch = returns.len();
                let returns = Tensor::from_floats(returns).to_device(&self.actor.devices()[0]);
                let s = step
                    .iter()
                    .map(|step| {
                        Tensor::cat(
                            step.obs
                                .as_vec()
                                .into_iter()
                                .map(|x| Tensor::from_floats(x.as_vec().as_slice()).unsqueeze())
                                .collect::<Vec<_>>(),
                            1,
                        )
                    })
                    .collect::<Vec<_>>();
                let s: Tensor<Be, 3> = Tensor::cat(s, 0);
                let a = step
                    .iter()
                    .map(|step| Tensor::from_floats(step.action.as_vec().as_slice()).unsqueeze())
                    .collect::<Vec<_>>();
                let a: Tensor<Be, 2> = Tensor::cat(a, 0);
                let old_lp = step
                    .iter()
                    .map(|step| step.action.metadata.unwrap().logp)
                    .collect::<Vec<_>>();
                let old_lp = Tensor::from_floats(old_lp.as_slice())
                    .to_device(&self.actor.devices()[0])
                    .reshape([nbatch, 1]);
                let old_val = step
                    .iter()
                    .map(|step| step.action.metadata.unwrap().val)
                    .collect::<Vec<_>>();
                let old_val: Tensor<_, 1> =
                    Tensor::from_floats(old_val.as_slice()).to_device(&self.actor.devices()[0]);

                let advantage = (returns.clone() - old_val).reshape([nbatch, 1]);

                let s = s.require_grad();
                let (mu, std) = self.actor.forward(s.clone());
                let dist = MvNormal::<ACTION_LEN> {
                    mu,
                    cov_diag: std.clone() * std,
                };
                let entropy = dist.entropy();
                let lp = dist.log_prob(a);
                // dbg!(old_lp.clone().to_data(), lp.clone().to_data());
                // let kl = (old_lp.clone() - lp.clone()).mean();
                // total_kl += kl.into_scalar();
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
                let entropy_loss = entropy.mean() * 1e-3;
                total_entropy_loss += entropy_loss.clone().into_scalar();
                let policy_loss = policy_loss - entropy_loss;

                let actor_grads = policy_loss.backward();
                self.actor = self.actor_optim.step(
                    AGENT_ACTOR_LR,
                    self.actor.clone(),
                    GradientsParams::from_grads(actor_grads, &self.actor),
                );
                let val = self.critic.forward(s.require_grad());
                let val_err = val - returns;
                let value_loss = (val_err.clone() * val_err).mean();
                total_val_loss += value_loss.clone().into_scalar();
                let critic_grads = value_loss.backward();
                self.critic = self.critic_optim.step(
                    AGENT_CRITIC_LR,
                    self.critic.clone(),
                    GradientsParams::from_grads(critic_grads, &self.critic),
                );
            }
        }
        self.recent_policy_loss = total_pi_loss / total_batches as f32;
        self.recent_value_loss = total_val_loss / total_batches as f32;
        self.recent_entropy_loss = total_entropy_loss / total_batches as f32;
        self.recent_nclamp = total_nclamp / total_batches as f32;
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
