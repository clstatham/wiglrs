use burn::{
    config::Config,
    module::{ADModule, Module, ModuleMapper},
    nn::{Linear, LinearConfig, ReLU},
    optim::{
        adaptor::OptimizerAdaptor, Adam, AdamConfig, GradientsParams, Optimizer, Sgd, SgdConfig,
        SimpleOptimizer,
    },
    record::{BinGzFileRecorder, FullPrecisionSettings},
    tensor::{backend::Backend, Tensor},
};
use burn_tch::TchBackend;
use burn_tensor::{backend::ADBackend, TensorKind};

use std::ops::{Add, Mul};
use std::{collections::VecDeque, marker::PhantomData};

use crate::brains::replay_buffer::ReplayBuffer;
use crate::brains::FrameStack;
use crate::hparams::{
    AGENT_ACTOR_LR, AGENT_CRITIC_LR, AGENT_HIDDEN_DIM, AGENT_OPTIM_BATCH_SIZE, AGENT_OPTIM_EPOCHS,
    AGENT_RB_MAX_LEN, N_FRAME_STACK,
};
use crate::{Action, ActionMetadata, ACTION_LEN, OBS_LEN};

use super::Thinker;

pub type Be = burn_autodiff::ADBackendDecorator<burn_tch::TchBackend<f32>>;

pub struct MvNormal<const K: usize, B: Backend> {
    pub mu: Tensor<B, 1>,
    pub cov_diag: Tensor<B, 1>,
}

impl<const K: usize, B: Backend> MvNormal<K, B>
where
    f32: std::convert::From<<B as Backend>::FloatElem>,
{
    pub fn sample(&self) -> Tensor<B, 1> {
        use rand_distr::{Distribution, StandardNormal};
        let cov_mat = nalgebra::DMatrix::from_diagonal(&nalgebra::DVector::from_iterator(
            K,
            self.cov_diag
                .to_data()
                .value
                .into_iter()
                .map(|f| f32::from(f)),
        ));
        let a = cov_mat.cholesky().unwrap().unpack();
        let z = nalgebra::DVector::from_iterator(
            K,
            (0..K).map(|_| StandardNormal.sample(&mut rand::thread_rng())),
        );
        let x = a * &z;
        let x = Tensor::from_floats(
            x.data
                .as_slice()
                .into_iter()
                .map(|f| (*f).into())
                .collect::<Vec<_>>()
                .as_slice(),
        );
        self.mu.clone() + x
    }

    pub fn log_prob(self, x: Tensor<B, 1>) -> Tensor<B, 1> {
        let cov_mat = nalgebra::DMatrix::from_diagonal(&nalgebra::DVector::from_iterator(
            K,
            self.cov_diag
                .to_data()
                .value
                .into_iter()
                .map(|f| f32::from(f)),
        ));
        let a = (-self.mu.clone()).add(x.clone()).reshape([1, K]);
        let b = Tensor::from_floats(
            cov_mat
                .clone()
                .try_inverse()
                .unwrap()
                .data
                .as_slice()
                .into_iter()
                .map(|f| (*f).into())
                .collect::<Vec<_>>()
                .as_slice(),
        )
        .reshape([K, K]);
        let c = ((-self.mu).add(x)).reshape([K, 1]);
        let g = cov_mat.determinant();
        let d = a.matmul(b);
        let e = d.matmul(c);
        let numer = (e * -0.5).exp();
        let f = (2.0 * std::f32::consts::PI).powf(K as f32);
        let denom = (f * g).sqrt();
        let pdf = numer / denom;
        let logpdf = pdf.log();
        logpdf.reshape([1])
    }

    pub fn entropy(&self) -> f32 {
        let cov_mat = nalgebra::DMatrix::from_diagonal(&nalgebra::DVector::from_iterator(
            K,
            self.cov_diag
                .to_data()
                .value
                .into_iter()
                .map(|f| f32::from(f)),
        ));
        let g = cov_mat.determinant();
        (K as f32 / 2.0) + ((K as f32 / 2.0) * (2.0 * std::f32::consts::PI).ln()) + (g.ln() / 2.0)
    }
}

fn mvn_batch_log_prob<const K: usize>(
    mu: Tensor<Be, 2>,
    var: Tensor<Be, 2>,
    x: Tensor<Be, 2>,
) -> (Tensor<Be, 1>, Tensor<Be, 1>) {
    let nbatch = x.shape().dims[0];
    assert_eq!(mu.shape().dims[0], nbatch);
    assert_eq!(var.shape().dims[0], nbatch);
    let mut out_lp = vec![];
    let mut out_e = vec![];
    for i in 0..nbatch {
        let idx = Tensor::from_ints([i as i32]);
        let mu = mu.clone().select(0, idx.clone()).squeeze(0);
        let var = var.clone().select(0, idx.clone()).squeeze(0);
        let x = x.clone().select(0, idx.clone()).squeeze(0);
        let dist: MvNormal<ACTION_LEN, Be> = MvNormal { mu, cov_diag: var };
        let e = dist.entropy();
        let lp = dist.log_prob(x).unsqueeze();
        out_lp.push(lp);
        out_e.push(e);
    }
    (
        Tensor::cat(out_lp, 0),
        Tensor::from_floats(out_e.as_slice()),
    )
}

// fn softplus<
//     S: Shape,
//     E: Dtype + num_traits::Float,
//     D: dfdx::tensor_ops::Device<E>,
//     T: Tape<E, D>,
// >(
//     input: Tensor<S, E, D, T>,
// ) -> Tensor<S, E, D, T> {
//     input.exp().add(E::from(1.0).unwrap()).ln()
// }

pub struct FooMapper;

impl ModuleMapper<Be> for FooMapper {
    fn map<const D: usize>(
        &mut self,
        id: &burn::module::ParamId,
        tensor: Tensor<Be, D>,
    ) -> Tensor<Be, D> {
        dbg!(tensor.is_require_grad());
        tensor
    }
}

#[derive(Module, Debug)]
pub struct PpoActor<B: Backend> {
    com1: Linear<B>,
    com2: Linear<B>,
    mu_head1: Linear<B>,
    mu_head2: Linear<B>,
    std_head1: Linear<B>,
    std_head2: Linear<B>,
    relu: ReLU,
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
            com1: LinearConfig::new(self.obs_len, self.hidden_len).init(),
            com2: LinearConfig::new(self.hidden_len, self.hidden_len).init(),
            mu_head1: LinearConfig::new(self.hidden_len, self.hidden_len).init(),
            mu_head2: LinearConfig::new(self.hidden_len, self.action_len).init(),
            std_head1: LinearConfig::new(self.hidden_len, self.hidden_len).init(),
            std_head2: LinearConfig::new(self.hidden_len, self.action_len).init(),
            relu: ReLU::new(),
        }
    }
}
impl<B: Backend> PpoActor<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let x = self.com1.forward(x);
        let x = self.relu.forward(x);
        let x = self.com2.forward(x);
        let x = self.relu.forward(x);

        let [nbatch, nstack, nhidden] = x.shape().dims;

        let x = x
            .slice([0..nbatch, nstack - 1..nstack, 0..nhidden])
            .squeeze(1);
        let mu = self.mu_head1.forward(x.clone());
        let mu = self.relu.forward(mu);
        let mu = self.mu_head2.forward(mu);
        let mu = mu.tanh();

        let std = self.std_head1.forward(x);
        let std = self.relu.forward(std);
        let std = self.std_head2.forward(std);
        // softplus
        let std = (std.exp() + 1.0).log();

        (mu, std)
    }
}

#[derive(Module, Debug)]
pub struct PpoCritic<B: Backend> {
    lin1: Linear<B>,
    lin2: Linear<B>,
    lin3: Linear<B>,
    lin4: Linear<B>,
    relu: ReLU,
}

#[derive(Debug, Config)]
pub struct PpoCriticConfig {
    obs_len: usize,
    hidden_len: usize,
}

impl PpoCriticConfig {
    pub fn init<B: ADBackend>(&self) -> PpoCritic<B> {
        PpoCritic {
            lin1: LinearConfig::new(self.obs_len, self.hidden_len).init(),
            lin2: LinearConfig::new(self.hidden_len, self.hidden_len).init(),
            lin3: LinearConfig::new(self.hidden_len, self.hidden_len).init(),
            lin4: LinearConfig::new(self.hidden_len, 1).init(),
            relu: ReLU::new(),
        }
    }
}

impl<B: Backend> PpoCritic<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 1> {
        let x = self.lin1.forward(x);
        let x = self.relu.forward(x);
        let x = self.lin2.forward(x);
        let x = self.relu.forward(x);
        let x = self.lin3.forward(x);
        let x = self.relu.forward(x);
        let x = self.lin4.forward(x);
        let [nbatch, nstack, nfeat] = x.shape().dims;
        x.slice([0..nbatch, nstack - 1..nstack, 0..nfeat])
            .squeeze::<2>(1)
            .squeeze(1)
    }
}

pub struct PpoThinker {
    actor: PpoActor<Be>,
    critic: PpoCritic<Be>,
    actor_optim: OptimizerAdaptor<Sgd<TchBackend<f32>>, PpoActor<Be>, Be>,
    critic_optim: OptimizerAdaptor<Sgd<TchBackend<f32>>, PpoCritic<Be>, Be>,
    pub recent_mu: Vec<f32>,
    pub recent_std: Vec<f32>,
    pub recent_entropy: f32,
    pub recent_policy_loss: f32,
    pub recent_value_loss: f32,
    pub recent_entropy_loss: f32,
}

impl PpoThinker {
    pub fn new() -> Self {
        let actor = PpoActorConfig {
            obs_len: OBS_LEN,
            hidden_len: AGENT_HIDDEN_DIM,
            action_len: ACTION_LEN,
        }
        .init();
        let critic = PpoCriticConfig {
            obs_len: OBS_LEN,
            hidden_len: AGENT_HIDDEN_DIM,
        }
        .init();
        let actor_optim = SgdConfig::new().init();
        let critic_optim = SgdConfig::new().init();
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
        let mu = mu.squeeze(0);
        let std = std.squeeze(0);
        let dist: MvNormal<ACTION_LEN, TchBackend<f32>> = MvNormal {
            mu,
            cov_diag: std.clone() * std,
        };
        self.recent_entropy = dist.entropy();
        let action = dist.sample();
        let val = self.critic.valid().forward(obs);

        Action::from_slice(
            action.to_data().value.as_slice(),
            Some(ActionMetadata {
                logp: dist.log_prob(action).to_data().value[0],
                val: val.to_data().value[0],
            }),
        )
    }

    fn learn<const MAX_LEN: usize>(&mut self, rb: &mut ReplayBuffer<MAX_LEN>) {
        let nstep = rb.buf.len() - 1;
        // if nstep < AGENT_RB_MAX_LEN - 1 {
        //     return; // not enough data to train yet
        // }
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
                let returns = Tensor::from_floats(returns);
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
                let old_lp = Tensor::from_floats(old_lp.as_slice());
                let old_val = step
                    .iter()
                    .map(|step| step.action.metadata.unwrap().val)
                    .collect::<Vec<_>>();
                let old_val: Tensor<_, 1> = Tensor::from_floats(old_val.as_slice());

                let advantage = returns.clone() - old_val;
                let s = s.require_grad();
                let (mu, std) = self.actor.forward(s.clone());
                // assert!(mu.is_require_grad());
                // assert!(std.is_require_grad());

                // assert!(val.is_require_grad());

                let (lp, entropy) = mvn_batch_log_prob::<ACTION_LEN>(mu, std.clone() * std, a);
                let ratio = (lp - old_lp).exp();
                let surr1 = ratio.clone() * advantage.clone();
                let surr2 = ratio.clamp(0.8, 1.2) * advantage;
                let masked = surr2.clone().mask_where(surr2.lower(surr1.clone()), surr1);
                let policy_loss = -masked.mean();

                total_pi_loss += policy_loss.clone().into_scalar() * nbatch as f32;
                let entropy_loss = entropy.mean() * 1e-5;
                total_entropy_loss += entropy_loss.clone().into_scalar() * nbatch as f32;
                let policy_loss = policy_loss - entropy_loss;

                let actor_grads = policy_loss.backward();
                self.actor = self.actor_optim.step(
                    AGENT_ACTOR_LR,
                    self.actor.clone(),
                    GradientsParams::from_grads(actor_grads, &self.actor),
                );
                // self.actor.zero_grads(&mut actor_grads);
                let val = self.critic.forward(s.require_grad());
                let val_err = val - returns;
                let value_loss = (val_err.clone() * val_err).mean();
                total_val_loss += value_loss.clone().into_scalar() * nbatch as f32;
                let critic_grads = value_loss.backward();
                self.critic = self.critic_optim.step(
                    AGENT_CRITIC_LR,
                    self.critic.clone(),
                    GradientsParams::from_grads(critic_grads, &self.critic),
                );
            }
        }
        self.recent_policy_loss = total_pi_loss / nstep as f32;
        self.recent_value_loss = total_val_loss / nstep as f32;
        self.recent_entropy_loss = total_entropy_loss / nstep as f32;
        // rb.buf.clear();
    }

    fn save(&self, path: impl AsRef<std::path::Path>) -> Result<(), Box<dyn std::error::Error>> {
        self.actor.clone().save_file(
            path.as_ref().join("actor.bin.gz"),
            &BinGzFileRecorder::<FullPrecisionSettings>::new(),
        )?;
        self.critic.clone().save_file(
            path.as_ref().join("critic.bin.gz"),
            &BinGzFileRecorder::<FullPrecisionSettings>::new(),
        )?;
        Ok(())
    }
}
