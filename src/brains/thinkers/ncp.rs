//! implementation of https://github.com/mlech26l/ncps/

use std::{collections::BTreeMap, fs::File, path::Path};

use burn::{
    config::Config,
    module::{Module, Param, ParamId},
    nn::{Linear, LinearConfig},
    tensor::{backend::Backend, Tensor},
};

use burn_tensor::Distribution;
use itertools::Itertools;
use petgraph::dot::Dot;
use rand::{seq::SliceRandom, thread_rng};

use super::ppo::sigmoid;

pub type Neuron = usize;

#[derive(Debug, Clone, Copy)]
pub enum Polarity {
    Positive,
    Negative,
}

impl From<Polarity> for f32 {
    fn from(val: Polarity) -> Self {
        match val {
            Polarity::Negative => -1.0,
            Polarity::Positive => 1.0,
        }
    }
}

#[derive(Debug)]
pub struct Wiring<B: Backend> {
    adj_matrix: Tensor<B, 2>,
    sensory_adj_matrix: Tensor<B, 2>,
    num_layers: usize,
}

impl<B: Backend> Wiring<B> {
    pub fn fully_connected(input_dim: usize, output_dim: usize, self_connections: bool) -> Self {
        let mut this = Self {
            adj_matrix: Tensor::zeros([output_dim, output_dim]),
            sensory_adj_matrix: Tensor::zeros([input_dim, output_dim]),
            num_layers: 1,
        };
        for from in 0..output_dim {
            for to in 0..output_dim {
                if from == to && !self_connections {
                    continue;
                }
                let polarity = [Polarity::Negative, Polarity::Positive, Polarity::Positive]
                    .choose(&mut thread_rng())
                    .copied()
                    .unwrap();
                this.add_synapse(from, to, polarity);
            }
        }
        for from in 0..input_dim {
            for to in 0..output_dim {
                let polarity = [Polarity::Negative, Polarity::Positive, Polarity::Positive]
                    .choose(&mut thread_rng())
                    .copied()
                    .unwrap();
                this.add_sensory_synapse(from, to, polarity);
            }
        }
        this
    }

    pub fn add_synapse(&mut self, from: Neuron, to: Neuron, polarity: Polarity) {
        let dev = self.adj_matrix.device();
        self.adj_matrix = self.adj_matrix.clone().slice_assign(
            [from..from + 1, to..to + 1],
            Tensor::full_device::<_, f32>([1, 1], polarity.into(), &dev),
        );
    }

    pub fn add_sensory_synapse(&mut self, from: Neuron, to: Neuron, polarity: Polarity) {
        let dev = self.sensory_adj_matrix.device();
        self.sensory_adj_matrix = self.sensory_adj_matrix.clone().slice_assign(
            [from..from + 1, to..to + 1],
            Tensor::full_device::<_, f32>([1, 1], polarity.into(), &dev),
        );
    }

    pub fn input_dim(&self) -> usize {
        self.sensory_adj_matrix.shape().dims[0]
    }

    pub fn output_dim(&self) -> usize {
        self.adj_matrix.shape().dims[1]
    }

    pub fn units(&self) -> usize {
        self.output_dim()
    }

    pub fn to_petgraph(&self) -> petgraph::prelude::DiGraph<&'static str, f32>
    where
        B::FloatElem: PartialEq<f32> + Into<f32>,
    {
        use petgraph::prelude::*;
        let mut g = DiGraph::new();
        let mut my_adj_to_g_idx = BTreeMap::default();
        let mut my_sens_adj_to_g_idx = BTreeMap::default();
        for i in 0..self.units() {
            my_adj_to_g_idx.insert(i, g.add_node("n"));
        }
        for i in 0..self.input_dim() {
            my_sens_adj_to_g_idx.insert(i, g.add_node("s"));
        }
        for src in 0..self.input_dim() {
            for dest in 0..self.units() {
                let weight = self
                    .sensory_adj_matrix
                    .clone()
                    .slice([src..src + 1, dest..dest + 1])
                    .into_scalar();
                if weight != 0.0 {
                    g.add_edge(
                        my_sens_adj_to_g_idx[&src],
                        my_adj_to_g_idx[&dest],
                        weight.into(),
                    );
                }
            }
        }
        for src in 0..self.units() {
            for dest in 0..self.units() {
                let weight = self
                    .adj_matrix
                    .clone()
                    .slice([src..src + 1, dest..dest + 1])
                    .into_scalar();
                if weight != 0.0 {
                    g.add_edge(my_adj_to_g_idx[&src], my_adj_to_g_idx[&dest], weight.into());
                }
            }
        }

        g
    }

    pub fn write_dot(&self, path: impl AsRef<Path>) -> Result<(), Box<dyn std::error::Error>>
    where
        B::FloatElem: PartialEq<f32> + Into<f32>,
    {
        let g = self.to_petgraph();
        let dot = Dot::new(&g);
        let mut f = File::create(path.as_ref())?;
        use std::io::Write;
        write!(f, "{:?}", dot)?;
        Ok(())
    }
}

#[derive(Debug)]
pub struct Ncp<B: Backend> {
    pub wiring: Wiring<B>,
    pub sensory_neurons: Vec<Neuron>,
    pub command_neurons: Vec<Neuron>,
    pub inter_neurons: Vec<Neuron>,
    pub motor_neurons: Vec<Neuron>,
}

impl<B: Backend> Ncp<B> {
    pub fn new(
        sensory_neurons: usize,
        inter_neurons: usize,
        command_neurons: usize,
        motor_neurons: usize,
        sensory_fanout: usize,
        inter_fanout: usize,
        recurrent_command_synapses: usize,
        motor_fanin: usize,
    ) -> Self {
        let units = inter_neurons + command_neurons + motor_neurons;
        let mut this = Wiring {
            adj_matrix: Tensor::zeros([units, units]),
            sensory_adj_matrix: Tensor::zeros([sensory_neurons, units]),
            num_layers: 3,
        };

        let sensory = (0..sensory_neurons).collect_vec();
        let motor = (0..motor_neurons).collect_vec();
        let command = (motor_neurons..motor_neurons + command_neurons).collect_vec();
        let inter = (motor_neurons + command_neurons
            ..motor_neurons + command_neurons + inter_neurons)
            .collect_vec();

        // sensory -> inter layer
        let mut unreachable_inter_neurons = inter.clone();
        for src in sensory.iter() {
            for dest in inter.choose_multiple(&mut thread_rng(), sensory_fanout) {
                if unreachable_inter_neurons.contains(dest) {
                    unreachable_inter_neurons.retain(|n| *n != *dest);
                }
                let polarity = [Polarity::Negative, Polarity::Positive]
                    .choose(&mut thread_rng())
                    .unwrap();
                this.add_sensory_synapse(*src, *dest, *polarity);
            }
        }
        let mean_inter_neuron_fanin =
            (sensory_neurons as f32 * sensory_fanout as f32 / inter_neurons as f32) as usize;
        let mean_inter_neuron_fanin = mean_inter_neuron_fanin.clamp(1, sensory_neurons);
        for dest in unreachable_inter_neurons {
            for src in sensory.choose_multiple(&mut thread_rng(), mean_inter_neuron_fanin) {
                let polarity = [Polarity::Negative, Polarity::Positive]
                    .choose(&mut thread_rng())
                    .unwrap();
                this.add_sensory_synapse(*src, dest, *polarity);
            }
        }

        // inter -> command layer
        let mut unreachable_command_neurons = command.clone();
        for src in inter.iter() {
            for dest in command.choose_multiple(&mut thread_rng(), inter_fanout) {
                if unreachable_command_neurons.contains(dest) {
                    unreachable_command_neurons.retain(|n| *n != *dest);
                }
                let polarity = [Polarity::Negative, Polarity::Positive]
                    .choose(&mut thread_rng())
                    .unwrap();
                this.add_synapse(*src, *dest, *polarity);
            }
        }
        let mean_command_neuron_fanin =
            (inter_neurons as f32 * inter_fanout as f32 / command_neurons as f32) as usize;
        let mean_command_neuron_fanin = mean_command_neuron_fanin.clamp(1, command_neurons);
        for dest in unreachable_command_neurons {
            for src in inter.choose_multiple(&mut thread_rng(), mean_command_neuron_fanin) {
                let polarity = [Polarity::Negative, Polarity::Positive]
                    .choose(&mut thread_rng())
                    .unwrap();
                this.add_synapse(*src, dest, *polarity);
            }
        }

        // recurrent command layer
        for _ in 0..recurrent_command_synapses {
            let src = command.choose(&mut thread_rng()).unwrap();
            let dest = command.choose(&mut thread_rng()).unwrap();
            let polarity = [Polarity::Negative, Polarity::Positive]
                .choose(&mut thread_rng())
                .unwrap();
            this.add_synapse(*src, *dest, *polarity);
        }

        // command -> motor layer
        let mut unreachable_command_neurons = command.clone();
        for dest in motor.iter() {
            for src in command.choose_multiple(&mut thread_rng(), motor_fanin) {
                if unreachable_command_neurons.contains(src) {
                    unreachable_command_neurons.retain(|n| *n != *src);
                }
                let polarity = [Polarity::Negative, Polarity::Positive]
                    .choose(&mut thread_rng())
                    .unwrap();
                this.add_synapse(*src, *dest, *polarity);
            }
        }
        let mean_command_fanout =
            (motor_neurons as f32 * motor_fanin as f32 / command_neurons as f32) as usize;
        let mean_command_fanout = mean_command_fanout.clamp(1, motor_neurons);
        for src in unreachable_command_neurons {
            for dest in motor.choose_multiple(&mut thread_rng(), mean_command_fanout) {
                let polarity = [Polarity::Negative, Polarity::Positive]
                    .choose(&mut thread_rng())
                    .unwrap();
                this.add_synapse(src, *dest, *polarity);
            }
        }

        Self {
            wiring: this,
            sensory_neurons: sensory,
            command_neurons: command,
            inter_neurons: inter,
            motor_neurons: motor,
        }
    }

    pub fn auto(
        input_dim: usize,
        units: usize,
        output_dim: usize,
        sparsity_level: Option<f32>,
    ) -> Self {
        let sparsity_level = sparsity_level.unwrap_or(0.5);
        let density_level = 1.0 - sparsity_level;
        let inter_and_command_neurons = units - output_dim;
        let command_neurons = ((0.4 * inter_and_command_neurons as f32) as usize).max(1);
        let inter_neurons = inter_and_command_neurons - command_neurons;
        let sensory_fanout = ((inter_neurons as f32 * density_level) as usize).max(1);
        let inter_fanout = ((command_neurons as f32 * density_level) as usize).max(1);
        let recurrent_command_synapses =
            ((command_neurons as f32 * density_level * 2.0) as usize).max(1);
        let motor_fanin = ((command_neurons as f32 * density_level) as usize).max(1);
        Self::new(
            input_dim,
            inter_neurons,
            command_neurons,
            output_dim,
            sensory_fanout,
            inter_fanout,
            recurrent_command_synapses,
            motor_fanin,
        )
    }
}

fn ltc_sigmoid<B: Backend>(
    v_pre: Tensor<B, 2>,
    mu: Tensor<B, 2>,
    sigma: Tensor<B, 2>,
) -> Tensor<B, 3> {
    let [nbatch, nfeat] = v_pre.shape().dims;
    let [nfeat2, nout] = mu.shape().dims;
    assert_eq!(nfeat2, nfeat);
    let v_pre = v_pre.reshape([nbatch, nfeat, 1]).repeat(2, nout);

    let mues = v_pre - mu.reshape([1, nfeat, nout]);
    let x = sigma.reshape([1, nfeat, nout]) * mues;
    super::ppo::sigmoid(x)
}

pub fn softplus<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    (x.exp() + 1.0).log()
}

#[derive(Module, Debug)]
pub struct LtcCell<B: Backend> {
    pub gleak: Param<Tensor<B, 1>>,
    pub vleak: Param<Tensor<B, 1>>,
    pub cm: Param<Tensor<B, 1>>,
    pub sigma: Param<Tensor<B, 2>>,
    pub mu: Param<Tensor<B, 2>>,
    pub w: Param<Tensor<B, 2>>,
    pub erev: Param<Tensor<B, 2>>,
    pub sensory_sigma: Param<Tensor<B, 2>>,
    pub sensory_mu: Param<Tensor<B, 2>>,
    pub sensory_w: Param<Tensor<B, 2>>,
    pub sensory_erev: Param<Tensor<B, 2>>,
    pub sparsity_mask: Tensor<B, 2>,
    pub sensory_sparsity_mask: Tensor<B, 2>,
    pub input_w: Param<Tensor<B, 1>>,
    pub input_b: Param<Tensor<B, 1>>,
    pub output_w: Param<Tensor<B, 1>>,
    pub output_b: Param<Tensor<B, 1>>,
    pub ode_unfolds: usize,
    pub units: usize,
    pub output_len: usize,
}

impl<B: Backend> LtcCell<B> {
    fn ode_solver(
        &self,
        inputs: Tensor<B, 2>,
        state: Tensor<B, 2>,
        elapsed_time: Option<f32>,
    ) -> Tensor<B, 2> {
        let elapsed_time = elapsed_time.unwrap_or(1.0);
        let mut v_pre = state;
        let sensory_w_activation: Tensor<B, 3> = softplus(self.sensory_w.val()).unsqueeze()
            * ltc_sigmoid(inputs, self.sensory_mu.val(), self.sensory_sigma.val())
            * self.sensory_sparsity_mask.clone().unsqueeze();
        let sensory_rev_activation =
            sensory_w_activation.clone() * self.sensory_erev.val().unsqueeze();
        let w_numerator_sensory: Tensor<B, 2> = sensory_rev_activation.sum_dim(1).squeeze(1);
        let w_denominator_sensory: Tensor<B, 2> = sensory_w_activation.sum_dim(1).squeeze(1);
        let cm_t = softplus(self.cm.val()) / (elapsed_time / self.ode_unfolds as f32);
        let w_param = softplus(self.w.val());
        for _t in 0..self.ode_unfolds {
            let w_activation = w_param.clone().unsqueeze()
                * ltc_sigmoid(v_pre.clone(), self.mu.val(), self.sigma.val())
                * self.sparsity_mask.clone().unsqueeze();
            let rev_activation = w_activation.clone() * self.erev.val().unsqueeze();
            let w_numerator: Tensor<B, 2> =
                rev_activation.sum_dim(1).squeeze(1) + w_numerator_sensory.clone();
            let w_denominator: Tensor<B, 2> =
                w_activation.sum_dim(1).squeeze(1) + w_denominator_sensory.clone();

            let gleak = softplus(self.gleak.val());
            let numerator = cm_t.clone().unsqueeze() * v_pre.clone()
                + gleak.clone().unsqueeze() * self.vleak.val().unsqueeze()
                + w_numerator;
            let denominator = cm_t.clone().unsqueeze() + gleak.clone().unsqueeze() * w_denominator;
            v_pre = numerator / (denominator + 1e-7);
        }

        v_pre
    }

    pub fn forward(
        &self,
        inputs: Tensor<B, 2>,
        state: Tensor<B, 2>,
        elapsed_time: Option<f32>,
    ) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let inputs = inputs * self.input_w.val().unsqueeze() + self.input_b.val().unsqueeze();
        let next_state = self.ode_solver(inputs, state, elapsed_time);
        let outputs =
            next_state.clone() * self.output_w.val().unsqueeze() + self.output_b.val().unsqueeze();
        let [nbatch, _nfeat] = outputs.shape().dims;
        let outputs = outputs.slice([0..nbatch, 0..self.output_len]);
        (outputs, next_state)
    }
}

#[derive(Debug, Config)]
pub struct LtcCellConfig {
    pub input_len: usize,
    pub hidden_len: usize,
    pub output_len: usize,
    pub ode_unfolds: usize,
}

impl LtcCellConfig {
    pub fn init<B: Backend<FloatElem = f32>>(&self, wiring: Ncp<B>) -> LtcCell<B> {
        let Ncp { wiring, .. } = wiring;
        LtcCell {
            output_len: self.output_len,
            gleak: Param::new(
                ParamId::new(),
                Tensor::random([wiring.units()], Distribution::Uniform(0.001, 1.0)),
            ),
            vleak: Param::new(
                ParamId::new(),
                Tensor::random([wiring.units()], Distribution::Uniform(-0.2, 0.2)),
            ),
            cm: Param::new(
                ParamId::new(),
                Tensor::random([wiring.units()], Distribution::Uniform(-0.4, 0.4)),
            ),
            sigma: Param::new(
                ParamId::new(),
                Tensor::random(
                    [wiring.units(), wiring.units()],
                    Distribution::Uniform(3.0, 8.0),
                ),
            ),
            mu: Param::new(
                ParamId::new(),
                Tensor::random(
                    [wiring.units(), wiring.units()],
                    Distribution::Uniform(0.3, 0.8),
                ),
            ),
            w: Param::new(
                ParamId::new(),
                Tensor::random(
                    [wiring.units(), wiring.units()],
                    Distribution::Uniform(0.001, 1.0),
                ),
            ),
            erev: Param::new(ParamId::new(), wiring.adj_matrix.clone()),
            sensory_sigma: Param::new(
                ParamId::new(),
                Tensor::random(
                    [wiring.input_dim(), wiring.units()],
                    Distribution::Uniform(3.0, 8.0),
                ),
            ),
            sensory_mu: Param::new(
                ParamId::new(),
                Tensor::random(
                    [wiring.input_dim(), wiring.units()],
                    Distribution::Uniform(0.3, 0.8),
                ),
            ),
            sensory_w: Param::new(
                ParamId::new(),
                Tensor::random(
                    [wiring.input_dim(), wiring.units()],
                    Distribution::Uniform(0.001, 1.0),
                ),
            ),
            sensory_erev: Param::new(ParamId::new(), wiring.sensory_adj_matrix.clone()),
            sparsity_mask: wiring.adj_matrix.clone().abs(),
            sensory_sparsity_mask: wiring.sensory_adj_matrix.clone().abs(),
            input_w: Param::new(ParamId::new(), Tensor::ones([wiring.input_dim()])),
            input_b: Param::new(ParamId::new(), Tensor::zeros([wiring.input_dim()])),
            output_w: Param::new(ParamId::new(), Tensor::ones([wiring.output_dim()])),
            output_b: Param::new(ParamId::new(), Tensor::zeros([wiring.output_dim()])),
            ode_unfolds: self.ode_unfolds,
            // wiring,
            units: wiring.units(),
        }
    }
}

#[derive(Debug, Module)]
pub struct Ltc<B: Backend> {
    pub cell: LtcCell<B>,
}

impl<B: Backend> Ltc<B> {
    pub fn forward(
        &self,
        xs: Tensor<B, 3>,
        h: Option<Tensor<B, 2>>,
    ) -> (Tensor<B, 3>, Tensor<B, 2>) {
        let dev = &self.devices()[0];
        let [nbatch, nseq, nfeat] = xs.shape().dims;
        let nout = self.cell.output_len;
        let mut h = h
            .unwrap_or(Tensor::zeros([nbatch, self.cell.units]))
            .to_device(dev);
        let mut outputs = Tensor::zeros([nbatch, nseq, nout]).to_device(dev);

        for i in 0..nseq {
            let x: Tensor<B, 2> = xs.clone().slice([0..nbatch, i..i + 1, 0..nfeat]).squeeze(1);
            let (out, h_next) = self.cell.forward(x, h, None);
            h = h_next;
            outputs = outputs.slice_assign(
                [0..nbatch, i..i + 1, 0..nout],
                out.clone().reshape([nbatch, 1, nout]),
            );
        }

        (outputs, h)
    }
}

#[derive(Debug, Module)]
pub struct CfcCell<B: Backend> {
    pub ff1: Linear<B>,
    pub ff2: Linear<B>,
    pub time_a: Linear<B>,
    pub time_b: Linear<B>,
    pub sparsity_mask: Tensor<B, 2>,
}

impl<B: Backend> CfcCell<B> {
    pub fn forward(
        &self,
        input: Tensor<B, 2>,
        hx: Tensor<B, 2>,
        ts: Tensor<B, 1>,
    ) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let x = Tensor::cat(vec![input, hx], 1);
        let ff1 = x
            .clone()
            .matmul(self.ff1.weight.val() * self.sparsity_mask.clone())
            + self.ff1.bias.as_ref().unwrap().val().unsqueeze();
        let ff2 = x
            .clone()
            .matmul(self.ff2.weight.val() * self.sparsity_mask.clone())
            + self.ff2.bias.as_ref().unwrap().val().unsqueeze();
        let ff1 = ff1.tanh();
        let ff2 = ff2.tanh();
        let t_a = self.time_a.forward(x.clone());
        let t_b = self.time_b.forward(x.clone());
        let t_interp = sigmoid(t_a * ts.unsqueeze() + t_b);
        let new_hidden = ff1 * (-t_interp.clone() + 1.0) + t_interp * ff2;
        (new_hidden.clone(), new_hidden)
    }
}

#[derive(Debug, Module)]
pub struct WiredCfcCell<B: Backend> {
    pub layers: Vec<CfcCell<B>>,
    pub inter_neurons: Vec<Neuron>,
    pub command_neurons: Vec<Neuron>,
    pub motor_neurons: Vec<Neuron>,
}

impl<B: Backend> WiredCfcCell<B> {
    pub fn layer_sizes(&self) -> Vec<usize> {
        vec![
            self.inter_neurons.len(),
            self.command_neurons.len(),
            self.motor_neurons.len(),
        ]
    }

    pub fn forward(
        &self,
        mut input: Tensor<B, 2>,
        hx: Tensor<B, 2>,
        timespans: Tensor<B, 1>,
    ) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let [nbatch, _] = hx.shape().dims;
        let layer_sizes = [
            self.inter_neurons.len(),
            self.command_neurons.len(),
            self.motor_neurons.len(),
        ];
        let h_state = vec![
            hx.clone().slice([0..nbatch, 0..layer_sizes[0]]),
            hx.clone()
                .slice([0..nbatch, layer_sizes[0]..layer_sizes[0] + layer_sizes[1]]),
            hx.clone().slice([
                0..nbatch,
                layer_sizes[0] + layer_sizes[1]..layer_sizes[0] + layer_sizes[1] + layer_sizes[2],
            ]),
        ];
        let mut new_h_state = vec![];
        for (layer, h) in self.layers.iter().zip(h_state.into_iter()) {
            let (h, _) = layer.forward(input, h, timespans.clone());
            input = h.clone();
            new_h_state.push(h);
        }
        let last_h = new_h_state.last().cloned().unwrap();
        let new_h_state = Tensor::cat(new_h_state, 1);
        (last_h, new_h_state)
    }
}

#[derive(Debug, Config)]
pub struct WiredCfcCellConfig;

impl WiredCfcCellConfig {
    pub fn init<B: Backend>(&self, wiring: Ncp<B>) -> WiredCfcCell<B> {
        let mut layers = vec![];
        let mut in_features = wiring.wiring.input_dim();
        for (i, neurons) in [
            wiring.inter_neurons.clone(),
            wiring.command_neurons.clone(),
            wiring.motor_neurons.clone(),
        ]
        .into_iter()
        .enumerate()
        {
            let hidden_units = neurons
                .iter()
                .map(|n| Tensor::from_ints([*n as i32]))
                .collect_vec();
            let input_sparsity = if i == 0 {
                wiring
                    .wiring
                    .sensory_adj_matrix
                    .clone()
                    .select(1, Tensor::cat(hidden_units, 0))
            } else {
                let prev_neurons = match i - 1 {
                    0 => wiring.inter_neurons.clone(),
                    1 => wiring.command_neurons.clone(),
                    _ => unreachable!(),
                };
                let prev_hidden_units = prev_neurons
                    .into_iter()
                    .map(|n| Tensor::from_ints([n as i32]))
                    .collect_vec();
                let input_sparsity = wiring
                    .wiring
                    .adj_matrix
                    .clone()
                    .select(1, Tensor::cat(hidden_units, 0));
                input_sparsity.select(0, Tensor::cat(prev_hidden_units, 0))
            };
            let input_sparsity = Tensor::cat(
                vec![input_sparsity, Tensor::ones([neurons.len(), neurons.len()])],
                0,
            );

            let cell = CfcCell {
                sparsity_mask: input_sparsity,
                ff1: LinearConfig::new(in_features + neurons.len(), neurons.len())
                    .with_initializer(burn::nn::Initializer::XavierUniform { gain: 1.0 })
                    .init(),
                ff2: LinearConfig::new(in_features + neurons.len(), neurons.len())
                    .with_initializer(burn::nn::Initializer::XavierUniform { gain: 1.0 })
                    .init(),
                time_a: LinearConfig::new(in_features + neurons.len(), neurons.len())
                    .with_initializer(burn::nn::Initializer::XavierUniform { gain: 1.0 })
                    .init(),
                time_b: LinearConfig::new(in_features + neurons.len(), neurons.len())
                    .with_initializer(burn::nn::Initializer::XavierUniform { gain: 1.0 })
                    .init(),
            };
            layers.push(cell);
            in_features = neurons.len();
        }
        let Ncp {
            inter_neurons,
            command_neurons,
            motor_neurons,
            ..
        } = wiring;
        WiredCfcCell {
            layers,
            inter_neurons,
            command_neurons,
            motor_neurons,
        }
    }
}

#[derive(Debug, Module)]
pub struct Cfc<B: Backend> {
    pub cell: WiredCfcCell<B>,
}

impl<B: Backend> Cfc<B> {
    pub fn forward(
        &self,
        xs: Tensor<B, 3>,
        h: Option<Tensor<B, 2>>,
    ) -> (Tensor<B, 3>, Tensor<B, 2>) {
        let dev = &self.devices()[0];
        let [nbatch, nseq, nfeat] = xs.shape().dims;
        let nout = self.cell.motor_neurons.len();
        let mut h = h.unwrap_or(Tensor::zeros([nbatch, nout])).to_device(dev);
        let mut outputs = Tensor::zeros([nbatch, nseq, nout]).to_device(dev);

        for i in 0..nseq {
            let x: Tensor<B, 2> = xs.clone().slice([0..nbatch, i..i + 1, 0..nfeat]).squeeze(1);
            let (out, h_next) = self.cell.forward(x, h, Tensor::ones_device([1], dev));
            h = h_next;
            outputs = outputs.slice_assign(
                [0..nbatch, i..i + 1, 0..nout],
                out.clone().reshape([nbatch, 1, nout]),
            );
        }

        (outputs, h)
    }
}