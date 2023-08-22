use std::collections::VecDeque;

use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts};
use itertools::Itertools;

use crate::brains::BrainBank;

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

pub fn ui(mut cxs: EguiContexts, brains: NonSend<BrainBank>, log: ResMut<LogText>) {
    egui::Window::new("Scores").show(cxs.ctx_mut(), |ui| {
        ui.vertical(|ui| {
            for (ent, brain) in brains.iter().sorted_by_key(|(_, b)| b.kills).rev() {
                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new(format!("{} {}", brain.id, &brain.name))
                            .text_style(egui::TextStyle::Heading),
                    );
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(
                            egui::RichText::new(format!("{}-{}", brain.kills, brain.version))
                                .text_style(egui::TextStyle::Heading),
                        );
                    });
                });
            }
        });
    });
    egui::Window::new("Action Mean/Std").show(cxs.ctx_mut(), |ui| {
        ui.vertical(|ui| {
            for (ent, brain) in brains.iter().sorted_by_key(|(_, b)| b.id) {
                ui.group(|ui| {
                    ui.vertical(|ui| {
                        ui.heading(&brain.name);
                        let mut mu = "mu:".to_owned();
                        for m in brain.thinker.recent_mu.iter() {
                            mu.push_str(&format!(" {:.4}", m));
                        }
                        ui.label(mu);
                        let mut std = "std:".to_owned();
                        for s in brain.thinker.recent_std.iter() {
                            std.push_str(&format!(" {:.4}", s));
                        }
                        ui.label(std);
                    });
                });
            }
        });
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
