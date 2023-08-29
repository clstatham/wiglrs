use std::collections::VecDeque;

use bevy::prelude::*;
use bevy_egui::{
    egui::{
        self,
        plot::{Bar, BarChart, Line},
        Color32, Layout, Stroke,
    },
    EguiContexts,
};

use itertools::Itertools;

use crate::envs::ffa::{BrainId, Deaths, Ffa, FfaAgent, Kills, Name};

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

pub fn ui(
    mut cxs: EguiContexts,
    mut env: ResMut<Ffa>,
    log: ResMut<LogText>,
    agents: Query<Entity, With<FfaAgent>>,
    kills: Query<&Kills, With<FfaAgent>>,
    deaths: Query<&Deaths, With<FfaAgent>>,
    brain_ids: Query<&BrainId, With<FfaAgent>>,
    names: Query<&Name, With<FfaAgent>>,
) {
    egui::Window::new("Scores").show(cxs.ctx_mut(), |ui| {
        ui.vertical(|ui| {
            for handle in agents
                .iter()
                .sorted_by_key(|b| kills.get(*b).unwrap().0)
                .rev()
            {
                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new(format!(
                            "{} {}",
                            brain_ids.get(handle).unwrap().0,
                            names.get(handle).unwrap().0,
                        ))
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
    egui::Window::new("Action Mean/Std/Entropy")
        .min_height(200.0)
        .min_width(1200.0)
        // .auto_sized()
        .scroll2([true, false])
        .resizable(true)
        .show(cxs.ctx_mut(), |ui| {
            ui.with_layout(
                Layout::left_to_right(egui::Align::Min).with_main_wrap(false),
                |ui| {
                    // egui::Grid::new("mean/std grid").show(ui, |ui| {
                    for (_i, brain) in agents
                        .iter()
                        .sorted_by_key(|h| brain_ids.get(*h).unwrap().0)
                        .enumerate()
                    {
                        ui.group(|ui| {
                            ui.vertical(|ui| {
                                ui.heading(&names.get(brain).unwrap().0);
                                // ui.group(|ui| {
                                let status = env
                                    .brains
                                    .get_status(brain_ids.get(brain).unwrap().0)
                                    .unwrap_or_default();
                                let status = status.status.unwrap();
                                let mut mu = "mu:".to_owned();
                                for m in status.recent_mu.iter() {
                                    mu.push_str(&format!(" {:.4}", m));
                                }
                                ui.label(mu);
                                let mut std = "std:".to_owned();
                                for s in status.recent_std.iter() {
                                    std.push_str(&format!(" {:.4}", s));
                                }
                                ui.label(std);
                                ui.label(format!("ent: {}", status.recent_entropy));
                                // });

                                // ui.horizontal_top(|ui| {

                                let ms = status
                                    .recent_mu
                                    .iter()
                                    .zip(status.recent_std.iter())
                                    .enumerate()
                                    .map(|(i, (mu, std))| {
                                        // https://www.desmos.com/calculator/rkoehr8rve
                                        let scale = std * 3.0;
                                        let _rg =
                                            Vec2::new(scale.exp(), (1.0 / scale).exp()).normalize();
                                        let m = Bar::new(i as f64, *mu as f64)
                                            // .fill(Color32::from_rgb(
                                            //     (rg.x * 255.0) as u8,
                                            //     (rg.y * 255.0) as u8,
                                            //     0,
                                            // ));
                                            .fill(Color32::RED);
                                        let var = std * std;
                                        let s = Line::new(vec![
                                            [i as f64, *mu as f64 - var as f64],
                                            [i as f64, *mu as f64 + var as f64],
                                        ])
                                        .stroke(Stroke::new(4.0, Color32::LIGHT_GREEN));
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
                                                brain_ids.get(brain).unwrap().0,
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
