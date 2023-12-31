use std::sync::Mutex;

use bevy::prelude::*;
use bevy_egui::{
    egui::{
        self,
        plot::{Bar, BarChart, Line},
    },
    EguiContexts,
};
use candle_core::{backprop::GradStore, IndexOp, Module, Result, Tensor};
use candle_nn::{AdamW, Linear, Optimizer, VarBuilder, VarMap};
use itertools::Itertools;

use crate::{
    brains::{
        learners::{
            utils::{adam, linear, MvNormal, ResBlock},
            DEVICE,
        },
        Policies,
    },
    envs::{Agent, AgentId, Env, Name},
};

use super::{Policy, PolicyWithTarget, ValueEstimator};

#[derive(Clone, Debug)]
pub struct LinearResnetStatus {
    pub mu: Box<[f32]>,
    pub cov: Box<[f32]>,
    pub entropy: f32,
}

#[derive(Component)]
pub struct LinResActor {
    action_len: usize,
    common1: Linear,
    common2: ResBlock,
    common3: ResBlock,
    mu_head: Linear,
    cov_head: Linear,
    varmap: VarMap,
    optim: Mutex<AdamW>,
    status: Mutex<Option<LinearResnetStatus>>,
}

impl LinResActor {
    pub fn new(obs_len: usize, hidden_len: usize, action_len: usize, lr: f64) -> Result<Self> {
        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &DEVICE);
        let common1 = linear(obs_len, hidden_len, 5. / 3., vs.pp("common1"));
        let common2 = ResBlock::new(hidden_len, hidden_len / 2, vs.pp("common2"));
        let common3 = ResBlock::new(hidden_len, hidden_len / 2, vs.pp("common3"));
        let mu_head = linear(hidden_len, action_len, 5. / 3., vs.pp("mu_head"));
        let cov_head = linear(hidden_len, action_len, 5. / 3., vs.pp("cov_head"));
        let optim = adam(varmap.all_vars(), lr)?;
        Ok(Self {
            action_len,
            varmap,
            common1,
            common2,
            common3,
            mu_head,
            cov_head,
            optim: Mutex::new(optim),
            status: Mutex::new(None),
        })
    }
}

impl Policy for LinResActor {
    type Logits = Tensor;
    type Status = LinearResnetStatus;

    fn varmap(&self) -> &VarMap {
        &self.varmap
    }

    fn action_logits(&self, x: &Tensor) -> Result<Tensor> {
        let x = x.flatten_from(1)?;

        let x = self.common1.forward(&x)?.tanh()?;
        let x = self.common2.forward(&x)?;
        let x = self.common3.forward(&x)?;

        let mu = self.mu_head.forward(&x)?.tanh()?;

        let cov = self.cov_head.forward(&x)?;
        let logits = Tensor::cat(&[mu, cov], 1)?;
        Ok(logits)
    }

    fn act(&self, obs: &Tensor) -> Result<(Tensor, Self::Logits)> {
        let logits = self.action_logits(obs)?;
        let mu = logits.i((.., ..self.action_len))?;
        let cov = logits.i((.., self.action_len..))?;
        let cov = (cov.exp()? + 1.0)?.log()?;
        let dist = MvNormal {
            mu: mu.clone(),
            cov: cov.clone(),
        };
        *self.status.lock().unwrap() = Some(LinearResnetStatus {
            mu: mu.squeeze(0)?.to_vec1::<f32>()?.into_boxed_slice(),
            cov: cov.squeeze(0)?.to_vec1::<f32>()?.into_boxed_slice(),
            entropy: dist.entropy()?.reshape(())?.to_scalar::<f32>()?,
        });
        let action = dist.sample()?;
        Ok((action, logits.squeeze(0)?))
    }

    fn log_prob(&self, logits: &Self::Logits, action: &Tensor) -> Result<Tensor> {
        let mu = logits.i((.., ..self.action_len))?;
        let cov = logits.i((.., self.action_len..))?;
        let cov = (cov.exp()? + 1.0)?.log()?;
        let dist = MvNormal { mu, cov };
        Ok(dist.log_prob(action))
    }

    fn entropy(&self, logits: &Self::Logits) -> Result<Tensor> {
        let mu = logits.i((.., ..self.action_len))?;
        let cov = logits.i((.., self.action_len..))?;
        let cov = (cov.exp()? + 1.0)?.log()?;
        let dist = MvNormal { mu, cov };
        dist.entropy()
    }

    fn apply_gradients(&self, grads: &GradStore) -> Result<()> {
        self.optim.lock().unwrap().step(grads)
    }

    fn status(&self) -> Option<Self::Status> {
        self.status.lock().unwrap().as_ref().cloned()
    }
}

#[derive(Component)]
pub struct LinResCritic {
    l1: Linear,
    l2: ResBlock,
    l3: ResBlock,
    head: Linear,
    varmap: VarMap,
    optim: Mutex<AdamW>,
}

impl LinResCritic {
    pub fn new(obs_len: usize, hidden_len: usize, lr: f64) -> Result<Self> {
        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &DEVICE);
        let l1 = linear(obs_len, hidden_len, 5. / 3., vs.pp("l1"));
        let l2 = ResBlock::new(hidden_len, hidden_len / 2, vs.pp("l2"));
        let l3 = ResBlock::new(hidden_len, hidden_len / 2, vs.pp("l3"));
        let head = linear(hidden_len, 1, 1.0, vs.pp("head"));
        let optim = adam(varmap.all_vars(), lr)?;
        Ok(Self {
            varmap,
            l1,
            l2,
            l3,
            head,
            optim: Mutex::new(optim),
        })
    }
}
impl ValueEstimator for LinResCritic {
    fn varmap(&self) -> &VarMap {
        &self.varmap
    }

    fn estimate_value(&self, x: &Tensor, action: Option<&Tensor>) -> Result<Tensor> {
        let mut x = x.flatten_from(1)?;
        if let Some(action) = action {
            x = Tensor::cat(&[x, action.flatten_from(1)?], 1)?;
        }

        let x = self.l1.forward(&x)?.tanh()?;
        let x = self.l2.forward(&x)?;
        let x = self.l3.forward(&x)?;

        let x = self.head.forward(&x)?.squeeze(1)?;
        Ok(x)
    }

    fn apply_gradients(&self, grads: &GradStore) -> Result<()> {
        self.optim.lock().unwrap().step(grads)
    }
}

pub fn action_space_ui<E: Env>(
    mut cxs: EguiContexts,
    agents: Query<Entity, With<Agent>>,
    names: Query<&Name, With<Agent>>,
    ids: Query<&AgentId, With<Agent>>,
    policies: Res<Policies<PolicyWithTarget<LinResActor>>>,
) {
    egui::Window::new("Action Mean/Std/Entropy")
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
                                let status =
                                    policies.0.get(ids.get(agent).unwrap().0).unwrap().status();
                                if let Some(status) = status {
                                    let mut mu = "mu:".to_owned();
                                    for m in status.mu.iter() {
                                        mu.push_str(&format!(" {:.4}", m));
                                    }
                                    ui.label(mu);
                                    let mut cov = "cov:".to_owned();
                                    for s in status.cov.iter() {
                                        cov.push_str(&format!(" {:.4}", s));
                                    }
                                    ui.label(cov);
                                    ui.label(format!("ent: {}", status.entropy));
                                    // });

                                    // ui.horizontal_top(|ui| {

                                    let ms = status
                                        .mu
                                        .iter()
                                        .zip(status.cov.iter())
                                        .enumerate()
                                        .map(|(i, (mu, cov))| {
                                            let m = Bar::new(i as f64, *mu as f64)
                                                // .fill(Color32::from_rgb(
                                                //     (rg.x * 255.0) as u8,
                                                //     (rg.y * 255.0) as u8,
                                                //     0,
                                                // ));
                                                .fill(egui::Color32::RED);
                                            let std = cov.sqrt();
                                            let s = Line::new(vec![
                                                [i as f64, *mu as f64 - std as f64],
                                                [i as f64, *mu as f64 + std as f64],
                                            ])
                                            .stroke(egui::Stroke::new(
                                                4.0,
                                                egui::Color32::LIGHT_GREEN,
                                            ));
                                            (m, s)
                                            // .width(1.0 - *std as f64 / 6.0)
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
                                                .show(ui, |plot| {
                                                    let (mu, std): (Vec<Bar>, Vec<Line>) =
                                                        ms.into_iter().multiunzip();
                                                    plot.bar_chart(BarChart::new(mu));
                                                    for std in std {
                                                        plot.line(std);
                                                    }
                                                });
                                            });
                                        });
                                    });
                                }
                            });
                        });
                    }
                },
            );
        });
}
