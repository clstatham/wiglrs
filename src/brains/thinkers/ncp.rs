//! implementation of https://github.com/mlech26l/ncps/

use burn::{
    config::Config,
    module::{Module, Param, ParamId},
    nn::{Linear, LinearConfig},
    tensor::{backend::Backend, Tensor},
};

use burn_tensor::Distribution;
use itertools::Itertools;

use rand::{seq::SliceRandom, thread_rng};

use super::ppo::{sigmoid, GruCell, GruCellConfig};

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

pub trait WiringConfig<B: Backend>
where
    Self: Send + Sync,
{
    fn input_dim(&self) -> usize;
    fn output_dim(&self) -> usize;
    fn units(&self) -> usize;

    fn num_layers(&self) -> usize;
    fn dim_for_layer(&self, l: usize) -> Option<usize> {
        self.neurons_for_layer(l).map(|l| l.len())
    }
    fn neurons_for_layer(&self, l: usize) -> Option<Vec<Neuron>>;
    fn adj_matrix(&self) -> &Tensor<B, 2>;
    fn sensory_adj_matrix(&self) -> &Tensor<B, 2>;
    fn num_neurons(&self) -> usize {
        (0..self.num_layers())
            .map(|l| self.dim_for_layer(l).unwrap())
            .sum::<usize>()
    }
}

#[derive(Debug, Clone)]
pub struct FullyConnected<B: Backend> {
    pub wiring: Wiring<B>,
    pub input_dim: usize,
    pub output_dim: usize,
}
impl<B: Backend> WiringConfig<B> for FullyConnected<B> {
    fn input_dim(&self) -> usize {
        self.input_dim
    }

    fn output_dim(&self) -> usize {
        self.output_dim
    }

    fn units(&self) -> usize {
        self.output_dim
    }

    fn num_layers(&self) -> usize {
        1
    }

    fn neurons_for_layer(&self, l: usize) -> Option<Vec<Neuron>> {
        match l {
            0 => Some((0..self.output_dim).collect()),
            _ => None,
        }
    }

    fn adj_matrix(&self) -> &Tensor<B, 2> {
        &self.wiring.adj_matrix
    }

    fn sensory_adj_matrix(&self) -> &Tensor<B, 2> {
        &self.wiring.sensory_adj_matrix
    }
}

impl<B: Backend> FullyConnected<B> {
    pub fn new(input_dim: usize, output_dim: usize, self_connections: bool) -> Self {
        let mut this = Wiring {
            adj_matrix: Tensor::zeros([output_dim, output_dim]),
            sensory_adj_matrix: Tensor::zeros([input_dim, output_dim]),
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
        Self {
            wiring: this,
            input_dim,
            output_dim,
        }
    }
}

#[derive(Debug, Module)]
pub struct Wiring<B: Backend> {
    adj_matrix: Tensor<B, 2>,
    sensory_adj_matrix: Tensor<B, 2>,
}

impl<B: Backend> Wiring<B> {
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
}

#[derive(Debug, Clone)]
pub struct Ncp<B: Backend> {
    pub wiring: Wiring<B>,
    pub sensory_neurons: Vec<Neuron>,
    pub command_neurons: Vec<Neuron>,
    pub inter_neurons: Vec<Neuron>,
    pub motor_neurons: Vec<Neuron>,
}

impl<B: Backend> WiringConfig<B> for Ncp<B> {
    fn input_dim(&self) -> usize {
        self.sensory_neurons.len()
    }

    fn output_dim(&self) -> usize {
        self.motor_neurons.len()
    }

    fn units(&self) -> usize {
        self.inter_neurons.len() + self.command_neurons.len() + self.motor_neurons.len()
    }

    fn num_layers(&self) -> usize {
        3
    }

    fn neurons_for_layer(&self, l: usize) -> Option<Vec<Neuron>> {
        match l {
            0 => Some(self.inter_neurons.clone()),
            1 => Some(self.command_neurons.clone()),
            2 => Some(self.motor_neurons.clone()),
            _ => None,
        }
    }

    fn adj_matrix(&self) -> &Tensor<B, 2> {
        &self.wiring.adj_matrix
    }

    fn sensory_adj_matrix(&self) -> &Tensor<B, 2> {
        &self.wiring.sensory_adj_matrix
    }
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
    ) -> Ncp<B> {
        let units = inter_neurons + command_neurons + motor_neurons;
        let sensory = (0..sensory_neurons).collect_vec();
        let motor = (0..motor_neurons).collect_vec();
        let command = (motor_neurons..motor_neurons + command_neurons).collect_vec();
        let inter = (motor_neurons + command_neurons
            ..motor_neurons + command_neurons + inter_neurons)
            .collect_vec();
        let mut this = Wiring {
            adj_matrix: Tensor::zeros([units, units]),
            sensory_adj_matrix: Tensor::zeros([sensory_neurons, units]),
        };

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

        Ncp {
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
    ) -> Ncp<B> {
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
            let denominator = cm_t.clone().unsqueeze() + gleak.clone().unsqueeze() + w_denominator;
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
    #[config(default = "6")]
    pub ode_unfolds: usize,
}

impl LtcCellConfig {
    pub fn init<B: Backend<FloatElem = f32>>(&self, wiring: impl WiringConfig<B>) -> LtcCell<B> {
        LtcCell {
            output_len: wiring.output_dim(),
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
            erev: Param::new(ParamId::new(), wiring.adj_matrix().clone()),
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
            sensory_erev: Param::new(ParamId::new(), wiring.sensory_adj_matrix().clone()),
            sparsity_mask: wiring.adj_matrix().clone().abs(),
            sensory_sparsity_mask: wiring.sensory_adj_matrix().clone().abs(),
            input_w: Param::new(ParamId::new(), Tensor::ones([wiring.input_dim()])),
            input_b: Param::new(ParamId::new(), Tensor::zeros([wiring.input_dim()])),
            output_w: Param::new(ParamId::new(), Tensor::ones([wiring.output_dim()])),
            output_b: Param::new(ParamId::new(), Tensor::zeros([wiring.output_dim()])),
            ode_unfolds: self.ode_unfolds,
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

#[derive(Debug, Clone, Copy, Module, Default)]
pub enum CfcCellMode {
    #[default]
    Default,
    Pure,
    NoGate,
}

#[derive(Debug, Module)]
pub struct CfcCell<B: Backend> {
    pub ff1: Linear<B>,
    pub a: Param<Tensor<B, 2>>,
    pub w_tau: Param<Tensor<B, 2>>,
    pub ff2: Linear<B>,
    pub time_a: Linear<B>,
    pub time_b: Linear<B>,
    pub sparsity_mask: Tensor<B, 2>,
    pub ninputs: usize,
    pub nhidden: usize,
    pub mode: CfcCellMode,
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
        if let CfcCellMode::Pure = self.mode {
            let new_hidden = -self.a.val()
                * (-ts.unsqueeze() * (self.w_tau.val().abs() + ff1.clone().abs())).exp()
                * ff1
                + self.a.val();
            (new_hidden.clone(), new_hidden)
        } else {
            let ff2 = x
                .clone()
                .matmul(self.ff2.weight.val() * self.sparsity_mask.clone())
                + self.ff2.bias.as_ref().unwrap().val().unsqueeze();
            let ff1 = ff1.tanh();
            let ff2 = ff2.tanh();
            let t_a = self.time_a.forward(x.clone());
            let t_b = self.time_b.forward(x.clone());
            let t_interp = sigmoid(t_a * ts.unsqueeze() + t_b);
            let new_hidden = if let CfcCellMode::NoGate = self.mode {
                ff1 + t_interp * ff2
            } else {
                ff1 * (t_interp.ones_like() - t_interp.clone()) + t_interp * ff2
            };
            (new_hidden.clone(), new_hidden)
        }
    }
}

#[derive(Debug, Module)]
pub struct WiredCfcCell<B: Backend> {
    pub layers: Vec<CfcCell<B>>,
}

impl<B: Backend> WiredCfcCell<B> {
    pub fn forward(
        &self,
        mut input: Tensor<B, 2>,
        hx: Tensor<B, 2>,
        timespans: Tensor<B, 1>,
    ) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let [nbatch, _] = hx.shape().dims;
        let mut h_state = vec![];
        let mut accum = 0;
        for layer in self.layers.iter() {
            let nhidden = layer.ff1.weight.shape().dims[1];
            h_state.push(hx.clone().slice([0..nbatch, accum..accum + nhidden]));
            accum += nhidden;
        }
        let mut new_h_state = vec![];
        for (layer, h) in self.layers.iter().zip(h_state.into_iter()) {
            let (h_new, _) = layer.forward(input, h, timespans.clone());
            input = h_new.clone();
            new_h_state.push(h_new);
        }
        let last_h = new_h_state.last().cloned().unwrap();
        let new_h_state = Tensor::cat(new_h_state, 1);
        (last_h, new_h_state)
    }
}

#[derive(Debug, Config)]
pub struct WiredCfcCellConfig;

impl WiredCfcCellConfig {
    pub fn init<B: Backend>(
        &self,
        wiring: impl WiringConfig<B>,
        mode: CfcCellMode,
    ) -> WiredCfcCell<B> {
        let mut layers = vec![];
        let mut in_features = wiring.input_dim();
        for l in 0..wiring.num_layers() {
            let neurons = wiring.neurons_for_layer(l).unwrap();
            let hidden_units = neurons
                .iter()
                .map(|n| Tensor::from_ints([*n as i32]))
                .collect_vec();
            let input_sparsity = if l == 0 {
                wiring
                    .sensory_adj_matrix()
                    .clone()
                    .select(1, Tensor::cat(hidden_units, 0))
            } else {
                let prev_neurons = wiring.neurons_for_layer(l - 1).unwrap();
                let prev_hidden_units = prev_neurons
                    .into_iter()
                    .map(|n| Tensor::from_ints([n as i32]))
                    .collect_vec();
                let input_sparsity = wiring
                    .adj_matrix()
                    .clone()
                    .select(1, Tensor::cat(hidden_units, 0));
                input_sparsity.select(0, Tensor::cat(prev_hidden_units, 0))
            };
            let input_sparsity = Tensor::cat(
                vec![input_sparsity, Tensor::ones([neurons.len(), neurons.len()])],
                0,
            );
            let init = burn::nn::Initializer::XavierUniform { gain: 1.0 };
            let cell = CfcCell {
                mode,
                sparsity_mask: input_sparsity,
                ninputs: in_features,
                nhidden: neurons.len(),
                a: Param::new(
                    ParamId::new(),
                    init.init_with([1, neurons.len()], Some(neurons.len()), Some(neurons.len())),
                ),
                w_tau: Param::new(
                    ParamId::new(),
                    init.init_with([1, neurons.len()], Some(neurons.len()), Some(neurons.len())),
                ),
                ff1: LinearConfig::new(in_features + neurons.len(), neurons.len())
                    .with_initializer(init.clone())
                    .init(),
                ff2: LinearConfig::new(in_features + neurons.len(), neurons.len())
                    .with_initializer(init.clone())
                    .init(),
                time_a: LinearConfig::new(in_features + neurons.len(), neurons.len())
                    .with_initializer(init.clone())
                    .init(),
                time_b: LinearConfig::new(in_features + neurons.len(), neurons.len())
                    .with_initializer(init.clone())
                    .init(),
            };
            layers.push(cell);
            in_features = neurons.len();
        }
        WiredCfcCell { layers }
    }
}

#[derive(Debug, Module)]
pub struct Cfc<B: Backend> {
    pub cell: WiredCfcCell<B>,
    pub rnn: GruCell<B>,
    pub fc: Linear<B>,
    pub mode: CfcMode,
}

impl<B: Backend> Cfc<B> {
    pub fn forward(
        &self,
        xs: Tensor<B, 3>,
        h: Option<Tensor<B, 2>>,
    ) -> (Tensor<B, 3>, Tensor<B, 2>) {
        let dev = &self.devices()[0];
        let [nbatch, nseq, nfeat] = xs.shape().dims;
        let nout = self.fc.weight.shape().dims[1];
        let nhidden = self
            .cell
            .layers
            .iter()
            .map(|l| l.ff1.weight.shape().dims[1])
            .sum::<usize>();
        let mut h = h.unwrap_or(Tensor::zeros([nbatch, nhidden])).to_device(dev);
        let mut outputs = Tensor::zeros([nbatch, nseq, nout]).to_device(dev);

        for i in 0..nseq {
            let x: Tensor<B, 2> = xs.clone().slice([0..nbatch, i..i + 1, 0..nfeat]).squeeze(1);
            let (out, h_next) = if let CfcMode::MixedMemory = self.mode {
                let (h, _) = self.rnn.forward(x.clone(), Some(h));
                self.cell.forward(x, h, Tensor::ones_device([1], dev))
            } else {
                self.cell.forward(x, h, Tensor::ones_device([1], dev))
            };
            h = h_next;
            let out = self
                .fc
                .forward(out.to_device(dev))
                .reshape([nbatch, 1, nout]);
            outputs = outputs.slice_assign([0..nbatch, i..i + 1, 0..nout], out);
        }
        (outputs, h)
    }
}

#[derive(Debug, Clone, Module)]
pub enum CfcMode {
    Default,
    MixedMemory,
}

#[derive(Debug, Config)]
pub struct CfcConfig {
    pub input_len: usize,
    pub hidden_len: usize,
    #[config(default = "hidden_len")]
    pub projected_len: usize,
}

impl CfcConfig {
    pub fn init<B: Backend>(
        &self,
        wiring: impl WiringConfig<B>,
        mode: CfcMode,
        cell_mode: CfcCellMode,
    ) -> Cfc<B> {
        Cfc {
            cell: WiredCfcCellConfig::new().init(wiring, cell_mode),
            rnn: GruCellConfig::new(self.input_len, self.hidden_len).init(),
            fc: LinearConfig::new(self.hidden_len, self.projected_len).init(),
            mode,
        }
    }
}
