use std::collections::VecDeque;

use bevy::prelude::*;
use bevy_egui::egui;
use bevy_egui::egui::plot::{Bar, BarChart, Line};
use bevy_egui::EguiContexts;
use itertools::Itertools;

use crate::brains::thinkers::ppo::PpoStatus;
use crate::envs::ffa::{Name, RunningReward};
use crate::{
    brains::{thinkers::Thinker, Brain},
    envs::{
        ffa::{Agent, Deaths, Kills},
        Env,
    },
};

#[derive(Debug, Default, Resource)]
pub struct LogText(pub VecDeque<String>);
impl LogText {
    pub fn push(&mut self, s: String) {
        if self.0.len() >= 1000 {
            self.0.pop_front();
        }
        info!("{}", &s);
        self.0.push_back(s);
    }
}

impl std::fmt::Display for LogText {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for line in self.0.iter().rev() {
            writeln!(f, "{}", line)?;
        }
        Ok(())
    }
}

pub fn kdr<E: Env, T: Thinker<E>>(
    mut cxs: EguiContexts,
    agents: Query<Entity, With<Agent>>,
    kills: Query<&Kills, With<Agent>>,
    deaths: Query<&Deaths, With<Agent>>,
    names: Query<&Name, With<Agent>>,
) {
    egui::Window::new("KDR").show(cxs.ctx_mut(), |ui| {
        ui.vertical(|ui| {
            for handle in agents
                .iter()
                .sorted_by_key(|b| kills.get(*b).unwrap().0)
                .rev()
            {
                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new(format!("{}", names.get(handle).unwrap().0,))
                            .text_style(egui::TextStyle::Heading),
                    );
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::TOP), |ui| {
                        ui.label(
                            egui::RichText::new(format!(
                                "{}-{}",
                                kills.get(handle).unwrap().0,
                                deaths.get(handle).unwrap().0,
                            ))
                            .text_style(egui::TextStyle::Heading),
                        );
                    });
                });
            }
        });
    });
}

pub fn action_space<E: Env, T: Thinker<E, Status = PpoStatus>>(
    mut cxs: EguiContexts,
    agents: Query<Entity, With<Agent>>,
    names: Query<&Name, With<Agent>>,
    brains: Query<&Brain<E, T>, With<Agent>>,
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
                    for (_i, brain) in agents.iter().enumerate() {
                        ui.group(|ui| {
                            ui.vertical(|ui| {
                                ui.heading(&names.get(brain).unwrap().0);
                                // ui.group(|ui| {
                                let status = T::status(&brains.get(brain).unwrap().thinker);
                                let mut mu = "mu:".to_owned();
                                for m in status.recent_mu.iter() {
                                    mu.push_str(&format!(" {:.4}", m));
                                }
                                ui.label(mu);
                                let mut cov = "cov:".to_owned();
                                for s in status.recent_cov.iter() {
                                    cov.push_str(&format!(" {:.4}", s));
                                }
                                ui.label(cov);
                                ui.label(format!("ent: {}", status.recent_entropy));
                                // });

                                // ui.horizontal_top(|ui| {

                                let ms = status
                                    .recent_mu
                                    .iter()
                                    .zip(status.recent_cov.iter())
                                    .enumerate()
                                    .map(|(i, (mu, cov))| {
                                        // https://www.desmos.com/calculator/rkoehr8rve
                                        let scale = cov.sqrt() * 3.0;
                                        let _rg =
                                            Vec2::new(scale.exp(), (1.0 / scale).exp()).normalize();
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
                                        .stroke(egui::Stroke::new(4.0, egui::Color32::LIGHT_GREEN));
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
                                                names.get(brain).unwrap().0,
                                            ))
                                            .center_y_axis(true)
                                            .data_aspect(1.0 / 2.0)
                                            .height(80.0)
                                            .width(220.0)
                                            .show(
                                                ui,
                                                |plot| {
                                                    let (mu, std): (Vec<Bar>, Vec<Line>) =
                                                        ms.into_iter().multiunzip();
                                                    plot.bar_chart(BarChart::new(mu));
                                                    for std in std {
                                                        plot.line(std);
                                                    }
                                                },
                                            );
                                        });
                                    });
                                });
                            });
                        });
                    }
                },
            );
        });
}

pub fn log(mut cxs: EguiContexts, log: Res<LogText>) {
    egui::Window::new("Log")
        .vscroll(true)
        .hscroll(true)
        .show(cxs.ctx_mut(), |ui| {
            let s = format!("{}", *log);
            ui.add_sized(
                ui.available_size(),
                egui::TextEdit::multiline(&mut s.as_str()).desired_rows(20),
            );
        });
}

pub fn running_reward(mut cxs: EguiContexts, queries: Query<(&Name, &RunningReward), With<Agent>>) {
    egui::Window::new("Running Reward")
        .vscroll(true)
        .hscroll(true)
        .show(cxs.ctx_mut(), |ui| {
            ui.with_layout(egui::Layout::left_to_right(egui::Align::Min), |ui| {
                for (name, reward) in queries.iter() {
                    ui.group(|ui| {
                        ui.vertical(|ui| {
                            ui.heading(&name.0);
                            ui.label(format!("{:.4}", reward.get().unwrap_or_default()));

                            egui::plot::Plot::new(format!("RunningReward{}", &name.0))
                                .height(80.0)
                                .width(220.0)
                                .show(ui, |plot| {
                                    let line = Line::new(
                                        (0..reward.history_max_len)
                                            .zip(reward.history.iter())
                                            .map(|(x, y)| [x as f64, *y as f64])
                                            .collect_vec(),
                                    )
                                    .color(egui::Color32::YELLOW);
                                    plot.line(line);
                                });
                        });
                    });
                }
            });
        });
}
