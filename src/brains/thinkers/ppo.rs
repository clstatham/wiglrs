#![allow(clippy::single_range_in_vec_init)]

use burn::{
    config::Config,
    module::{ADModule, Module, Param, ParamId},
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
use std::f32::consts::PI;

use crate::brains::FrameStack;
use crate::hparams::{
    AGENT_ACTOR_LR, AGENT_CRITIC_LR, AGENT_HIDDEN_DIM, AGENT_OPTIM_BATCH_SIZE, AGENT_OPTIM_EPOCHS,
};
use crate::{brains::replay_buffer::SartAdvBuffer, hparams::AGENT_ENTROPY_BETA};
use crate::{Action, ActionMetadata, ACTION_LEN, OBS_LEN};

use super::{
    ncp::{Cfc, Ltc, LtcCellConfig, Ncp, WiredCfcCellConfig},
    Thinker,
};

pub type Be = burn_autodiff::ADBackendDecorator<burn_tch::TchBackend<f32>>;

pub struct MvNormal<B: Backend<FloatElem = f32>, const K: usize> {
    pub mu: Tensor<B, 2>,
    pub cov_diag: Tensor<B, 2>,
}

impl<B: Backend<FloatElem = f32>, const K: usize> MvNormal<B, K> {
    pub fn sample(&self) -> Tensor<B, 2> {
        let std = self.cov_diag.clone().sqrt();
        let cov = std.to_data().value;
        assert!(cov.iter().all(|f| *f > 0.0), "{:?}", cov);
        let nbatch = self.mu.shape().dims[0];
        let z = Tensor::random([nbatch, K], burn_tensor::Distribution::Normal(0.0, 1.0))
            .to_device(&self.mu.device());
        self.mu.clone() + z * std
    }

    pub fn log_prob(self, x: Tensor<B, 2>) -> Tensor<B, 2> {
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

        pdf.log()
    }

    pub fn entropy(&self) -> Tensor<B, 2> {
        let cov = self.cov_diag.to_data().value;
        assert!(cov.iter().all(|f| *f > 0.0), "{:?}", cov);
        let nbatch = self.mu.shape().dims[0];
        let mut g = Tensor::ones([nbatch, 1]).to_device(&self.cov_diag.device());
        for i in 0..K {
            g = g * self.cov_diag.clone().slice([0..nbatch, i..i + 1]);
        }
        let second_term = (K as f32 * 0.5) * (1.0 + (2.0 * PI).ln());
        g.log() * 0.5 + second_term
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
    pub fn init<B: Backend>(&self) -> Gru<B> {
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
    common: TransformerEncoder<B>,
    mu_head: Ltc<B>,
    std_head: Ltc<B>,
    mu_h: Tensor<B, 2>,
    std_h: Tensor<B, 2>,
}

#[derive(Config, Debug)]
pub struct PpoActorConfig {
    obs_len: usize,
    hidden_len: usize,
    action_len: usize,
}

impl PpoActorConfig {
    pub fn init<B: Backend<FloatElem = f32>>(&self) -> PpoActor<B> {
        let wiring1 = Ncp::auto(self.obs_len, self.hidden_len, self.action_len, None);
        let wiring2 = Ncp::auto(self.obs_len, self.hidden_len, self.action_len, None);
        PpoActor {
            common: TransformerEncoderConfig::new(self.obs_len, self.hidden_len, 2, 1).init(),
            mu_h: Tensor::zeros([1, wiring1.wiring.units()]),
            std_h: Tensor::zeros([1, wiring1.wiring.units()]),
            mu_head: Ltc {
                cell: LtcCellConfig::new().init(wiring1),
            },
            std_head: Ltc {
                cell: LtcCellConfig::new().init(wiring2),
            },
        }
    }
}
impl<B: Backend> PpoActor<B> {
    pub fn forward(&mut self, x: Tensor<B, 3>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let x = x.to_device(&self.devices()[0]);

        let x = self.common.forward(TransformerEncoderInput::new(x));

        let (mu, mu_h) = self.mu_head.forward(x.clone(), Some(self.mu_h.clone()));
        self.mu_h = mu_h;
        let [nbatch, nstack, nfeat] = mu.shape().dims;
        let mu = mu
            .clone()
            .slice([0..nbatch, nstack - 1..nstack, 0..nfeat])
            .squeeze(1)
            .tanh();

        let (std, std_h) = self.std_head.forward(x, Some(self.std_h.clone()));
        self.std_h = std_h;
        let std = std
            .slice([0..nbatch, nstack - 1..nstack, 0..nfeat])
            .squeeze(1);
        // softplus
        let std: Tensor<B, 2> = (std.exp() + 1.0).log();

        (mu, std)
    }

    pub fn reset_h(&mut self, batch_len: usize) {
        self.mu_h =
            Tensor::zeros_device([batch_len, self.mu_h.shape().dims[1]], &self.mu_h.device());
        self.std_h = Tensor::zeros_device(
            [batch_len, self.std_h.shape().dims[1]],
            &self.std_h.device(),
        );
    }
}

#[derive(Module, Debug)]
pub struct PpoCritic<B: Backend> {
    rnn: Cfc<B>,
    h: Tensor<B, 2>,
    // head: Linear<B>,
}

#[derive(Debug, Config)]
pub struct PpoCriticConfig {
    obs_len: usize,
    hidden_len: usize,
}

impl PpoCriticConfig {
    pub fn init<B: Backend<FloatElem = f32>>(&self) -> PpoCritic<B> {
        let wiring = Ncp::auto(self.obs_len, self.hidden_len, 1, None);
        PpoCritic {
            h: Tensor::zeros([1, wiring.wiring.units()]),
            // rnn: GruConfig::new(self.obs_len, self.hidden_len).init(),
            rnn: Cfc {
                cell: WiredCfcCellConfig::new().init(wiring),
            },
            // head: LinearConfig::new(self.hidden_len, 1).init(),
        }
    }
}

impl<B: Backend> PpoCritic<B> {
    pub fn forward(&mut self, x: Tensor<B, 3>) -> Tensor<B, 1> {
        let x = x.to_device(&self.devices()[0]);
        let (x, h) = self.rnn.forward(x, Some(self.h.clone()));
        self.h = h;
        let [nbatch, nstack, nfeat] = x.shape().dims;
        let x: Tensor<B, 2> = x
            .slice([0..nbatch, nstack - 1..nstack, 0..nfeat])
            .squeeze(1);
        // let x = self.head.forward(x);
        x.squeeze(1)
    }

    pub fn reset_h(&mut self, batch_len: usize) {
        self.h = Tensor::zeros_device([batch_len, self.h.shape().dims[1]], &self.h.device());
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
        let (mu, std) = self.actor.valid().forward(obs.clone());
        self.recent_mu = mu.to_data().value;
        self.recent_std = std.to_data().value;
        let dist: MvNormal<_, ACTION_LEN> = MvNormal {
            mu,
            cov_diag: std.clone() * std,
        };
        self.recent_entropy = dist.entropy().into_scalar();
        let action = dist.sample().clamp(-1.0, 1.0);
        let val = self.critic.valid().forward(obs);

        Action::from_slice(
            action.to_data().value.as_slice(),
            Some(ActionMetadata {
                logp: dist.log_prob(action).into_scalar(),
                val: val.into_scalar(),
            }),
        )
    }

    fn learn(&mut self, rb: &SartAdvBuffer) {
        let mut nstep = 0;

        let mut total_pi_loss = 0.0;
        let mut total_val_loss = 0.0;
        let mut total_entropy_loss = 0.0;
        let mut total_nclamp = 0.0;
        for _epoch in kdam::tqdm!(0..AGENT_OPTIM_EPOCHS, desc = "Training") {
            nstep += AGENT_OPTIM_BATCH_SIZE;

            let step = rb.sample_batch(AGENT_OPTIM_BATCH_SIZE).unwrap();
            let returns = Tensor::from_floats(
                step.returns
                    .iter()
                    .copied()
                    .map(|r| r.unwrap())
                    .collect_vec()
                    .as_slice(),
            )
            .to_device(&self.actor.devices()[0]);
            let s = step
                .obs
                .iter()
                .map(|stack| {
                    Tensor::cat(
                        stack
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
                .action
                .iter()
                .map(|action| Tensor::from_floats(action.as_vec().as_slice()).unsqueeze())
                .collect::<Vec<_>>();
            let a: Tensor<Be, 2> = Tensor::cat(a, 0);
            let old_lp = step
                .action
                .iter()
                .map(|action| action.metadata.unwrap().logp)
                .collect::<Vec<_>>();
            let old_lp = Tensor::from_floats(old_lp.as_slice())
                .to_device(&self.actor.devices()[0])
                .reshape([AGENT_OPTIM_BATCH_SIZE, 1]);

            let advantage = Tensor::from_floats(
                step.advantage
                    .iter()
                    .copied()
                    .map(|a| a.unwrap())
                    .collect_vec()
                    .as_slice(),
            )
            .to_device(&self.actor.devices()[0])
            .reshape([AGENT_OPTIM_BATCH_SIZE, 1]);

            // let s = s.require_grad();
            self.actor.reset_h(AGENT_OPTIM_BATCH_SIZE);
            let (mu, std) = self.actor.forward(s.clone());
            let dist = MvNormal::<Be, ACTION_LEN> {
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
            let policy_loss = policy_loss - entropy_loss * AGENT_ENTROPY_BETA;

            let actor_grads = policy_loss.backward();
            self.actor = self.actor_optim.step(
                AGENT_ACTOR_LR,
                self.actor.clone(),
                GradientsParams::from_grads(actor_grads, &self.actor),
            );
            self.critic.reset_h(AGENT_OPTIM_BATCH_SIZE);
            let val = self.critic.forward(s);
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
        self.actor.reset_h(1);
        self.critic.reset_h(1);
        self.recent_policy_loss = total_pi_loss / nstep as f32;
        self.recent_value_loss = total_val_loss / nstep as f32;
        self.recent_entropy_loss = total_entropy_loss / nstep as f32;
        self.recent_nclamp = total_nclamp / nstep as f32;
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
