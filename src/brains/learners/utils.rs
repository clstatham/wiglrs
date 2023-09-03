use std::{f32::consts::PI, ops::Mul};

use bevy::prelude::*;
use bevy_prng::ChaCha8Rng;
use bevy_rand::prelude::EntropyComponent;
use candle_core::{DType, Module, Result, Tensor};
use candle_nn::{Linear, VarBuilder};
use itertools::Itertools;
use rand::thread_rng;
use rand_distr::Distribution;

use super::DEVICE;

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
        (x2 + (x * 0.1)?)?.tanh()
    }
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
    pub fn sample(&self) -> Result<Tensor> {
        let elems = self.mu.shape().elem_count();
        let dist = rand_distr::StandardNormal;
        let samples = (0..elems)
            .map(|_| Distribution::<f32>::sample(&dist, &mut thread_rng()))
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
