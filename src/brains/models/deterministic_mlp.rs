use std::sync::Mutex;

use bevy::prelude::*;
use bevy_egui::{
    egui::{
        self,
        plot::{Bar, BarChart, Line},
    },
    EguiContexts,
};
use candle_core::{backprop::GradStore, Module, Result, Tensor};
use candle_nn::{AdamW, Linear, Optimizer, VarBuilder, VarMap};
use itertools::Itertools;

use crate::{
    brains::{
        learners::{utils::linear, DEVICE},
        Policies,
    },
    envs::{Agent, AgentId, Env, Name},
};

use super::{CopyWeights, Policy, PolicyWithTarget, ValueEstimator};

#[derive(Clone)]
pub struct DeterministicMlpActorStatus {
    pub action: Box<[f32]>,
    pub std: f32,
}

#[derive(Component)]
pub struct DeterministicMlpActor {
    pub layers: Vec<Linear>,
    pub varmap: VarMap,
    pub optim: Mutex<AdamW>,
    pub status: Mutex<DeterministicMlpActorStatus>,
}

impl DeterministicMlpActor {
    pub fn new(hidden_units: &[usize], lr: f64) -> Self {
        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &DEVICE);
        let mut layers = vec![linear(
            hidden_units[0],
            hidden_units[1],
            5.0 / 3.0,
            vs.pp("l1"),
        )];
        let mut last_out = hidden_units[1];
        for (i, next_out) in hidden_units.iter().enumerate().skip(2) {
            layers.push(linear(
                last_out,
                *next_out,
                5.0 / 3.0,
                vs.pp(format!("l{i}")),
            ));
            last_out = *next_out;
        }
        let optim = Mutex::new(AdamW::new_lr(varmap.all_vars(), lr).unwrap());
        Self {
            varmap,
            layers,
            optim,
            status: Mutex::new(DeterministicMlpActorStatus {
                std: 1.0,
                action: vec![0.0f32; last_out].into_boxed_slice(),
            }),
        }
    }
}

impl Policy for DeterministicMlpActor {
    type Logits = Tensor;

    type Status = DeterministicMlpActorStatus;

    fn action_logits(&self, obs: &Tensor) -> Result<Self::Logits> {
        let mut x = obs.flatten_from(1).unwrap();
        let n_layers = self.layers.len();
        for layer in self.layers[..n_layers - 1].iter() {
            x = layer.forward(&x)?.tanh()?;
        }
        x = self.layers[n_layers - 1].forward(&x)?.tanh()?;
        Ok(x)
    }

    fn act(&self, obs: &Tensor) -> Result<(Tensor, Self::Logits)> {
        let logits = self.action_logits(obs)?;
        let mut status = self.status.lock().unwrap();
        status.action = logits.squeeze(0)?.to_vec1()?.into_boxed_slice();
        let actions = (&logits + logits.randn_like(0.0, status.std as f64)?)?;
        if status.std > 0.05 {
            status.std *= 0.99998;
        }
        Ok((actions, logits))
    }

    fn log_prob(&self, _logits: &Self::Logits, _action: &Tensor) -> Result<Tensor> {
        unimplemented!()
    }

    fn entropy(&self, _logits: &Self::Logits) -> Result<Tensor> {
        unimplemented!()
    }

    fn apply_gradients(&self, grads: &GradStore) -> Result<()> {
        self.optim.lock().unwrap().step(grads)
    }

    fn status(&self) -> Option<Self::Status> {
        Some(self.status.lock().unwrap().clone())
    }
}

impl CopyWeights for DeterministicMlpActor {
    fn soft_update(&self, other: &Self, tau: f32) {
        for (varname, my_var) in self.varmap.data().lock().unwrap().iter() {
            let other_var = &other.varmap.data().lock().unwrap()[varname];
            let new_var = my_var
                .affine(1.0 - tau as f64, 0.0)
                .unwrap()
                .add(&other_var.affine(tau as f64, 0.0).unwrap())
                .unwrap();
            my_var.set(&new_var).unwrap();
        }
    }

    fn hard_update(&self, other: &Self) {
        for (varname, my_var) in self.varmap.data().lock().unwrap().iter() {
            my_var
                .set(&other.varmap.data().lock().unwrap()[varname])
                .unwrap();
        }
    }
}

#[derive(Component)]
pub struct DeterministicMlpCritic {
    pub layers: Vec<Linear>,
    pub varmap: VarMap,
    pub optim: Mutex<AdamW>,
}

impl DeterministicMlpCritic {
    pub fn new(hidden_units: &[usize], lr: f64) -> Self {
        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &DEVICE);
        let mut layers = vec![linear(
            hidden_units[0],
            hidden_units[1],
            5.0 / 3.0,
            vs.pp("l1"),
        )];
        let mut last_out = hidden_units[1];
        for (i, next_out) in hidden_units.iter().enumerate().skip(2) {
            layers.push(linear(
                last_out,
                *next_out,
                5.0 / 3.0,
                vs.pp(format!("l{i}")),
            ));
            last_out = *next_out;
        }
        let optim = Mutex::new(AdamW::new_lr(varmap.all_vars(), lr).unwrap());
        Self {
            varmap,
            layers,
            optim,
        }
    }
}

impl ValueEstimator for DeterministicMlpCritic {
    fn estimate_value(&self, obs: &Tensor, action: Option<&Tensor>) -> Result<Tensor> {
        let action = action.unwrap().flatten_from(1).unwrap();
        let obs = obs.flatten_from(1).unwrap();
        let mut x = Tensor::cat(&[obs, action], 1).unwrap();
        let n_layers = self.layers.len();
        for layer in self.layers[..n_layers - 1].iter() {
            x = layer.forward(&x)?.tanh()?;
        }
        x = self.layers[n_layers - 1].forward(&x)?.squeeze(1)?;
        Ok(x)
    }
    fn apply_gradients(&self, grads: &GradStore) -> Result<()> {
        self.optim.lock().unwrap().step(grads)
    }
}

impl CopyWeights for DeterministicMlpCritic {
    fn soft_update(&self, other: &Self, tau: f32) {
        for (varname, my_var) in self.varmap.data().lock().unwrap().iter() {
            let other_var = &other.varmap.data().lock().unwrap()[varname];
            let new_var = my_var
                .affine(1.0 - tau as f64, 0.0)
                .unwrap()
                .add(&other_var.affine(tau as f64, 0.0).unwrap())
                .unwrap();
            my_var.set(&new_var).unwrap();
        }
    }

    fn hard_update(&self, other: &Self) {
        for (varname, my_var) in self.varmap.data().lock().unwrap().iter() {
            my_var
                .set(&other.varmap.data().lock().unwrap()[varname])
                .unwrap();
        }
    }
}

pub fn action_space_ui<E: Env>(
    mut cxs: EguiContexts,
    agents: Query<Entity, With<Agent>>,
    names: Query<&Name, With<Agent>>,
    ids: Query<&AgentId, With<Agent>>,
    policies: Res<Policies<PolicyWithTarget<DeterministicMlpActor>>>,
) {
    egui::Window::new("Actions")
        .min_height(200.0)
        .min_width(1200.0)
        // .auto_sized()
        .scroll2([true, false])
        .resizable(true)
        .show(cxs.ctx_mut(), |ui| {
            ui.with_layout(
                egui::Layout::left_to_right(egui::Align::Min).with_main_wrap(false),
                |ui| {
                    // egui::Grid::new("mean/std grid").show(ui, |ui| {
                    for (_i, agent) in agents.iter().enumerate() {
                        ui.group(|ui| {
                            ui.vertical(|ui| {
                                ui.heading(&names.get(agent).unwrap().0);
                                // ui.group(|ui| {
                                let status = policies
                                    .0
                                    .get(ids.get(agent).unwrap().0)
                                    .unwrap()
                                    .status()
                                    .unwrap();
                                // if let Some(status) = status {
                                let mut action = "action:".to_owned();
                                for a in status.action.iter() {
                                    action.push_str(&format!(" {:.4}", a));
                                }
                                ui.label(action);
                                ui.label(format!("std: {}", status.std));
                                // });

                                // ui.horizontal_top(|ui| {

                                let ms = status
                                    .action
                                    .iter()
                                    .enumerate()
                                    .map(|(i, a)| {
                                        let s = Line::new(vec![
                                            [i as f64, *a as f64 - status.std as f64],
                                            [i as f64, *a as f64 + status.std as f64],
                                        ])
                                        .stroke(egui::Stroke::new(4.0, egui::Color32::LIGHT_GREEN));
                                        let m =
                                            Bar::new(i as f64, *a as f64).fill(egui::Color32::RED);
                                        (m, s)
                                    })
                                    .collect_vec();

                                ui.group(|ui| {
                                    ui.horizontal(|ui| {
                                        ui.vertical(|ui| {
                                            ui.label("Action Space");
                                            egui::plot::Plot::new(format!(
                                                "ActionSpace{}",
                                                names.get(agent).unwrap().0,
                                            ))
                                            .center_y_axis(true)
                                            .data_aspect(1.0 / 2.0)
                                            .height(80.0)
                                            .width(220.0)
                                            .show(
                                                ui,
                                                |plot| {
                                                    let (m, s): (Vec<_>, Vec<_>) =
                                                        ms.into_iter().multiunzip();
                                                    plot.bar_chart(BarChart::new(m));
                                                    for s in s {
                                                        plot.line(s);
                                                    }
                                                },
                                            );
                                        });
                                    });
                                });
                                // }
                            });
                        });
                    }
                },
            );
        });
}
