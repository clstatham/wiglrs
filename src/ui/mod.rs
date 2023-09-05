use std::collections::VecDeque;

use bevy::prelude::*;
use bevy_egui::egui;
use bevy_egui::egui::plot::Line;
use bevy_egui::EguiContexts;
use itertools::Itertools;

use crate::brains::models::{Policy, ValueEstimator};

use crate::envs::{Name, RunningReturn};
use crate::{
    brains::learners::Learner,
    envs::{Agent, Deaths, Env, Kills},
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

pub fn kdr<E: Env>(
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

pub fn running_reward(mut cxs: EguiContexts, queries: Query<(&Name, &RunningReturn), With<Agent>>) {
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
                                        (0..reward.max_len)
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
