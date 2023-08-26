use std::collections::VecDeque;

use bevy::prelude::*;
use bevy_egui::{
    egui::{
        self,
        plot::{Bar, BarChart},
        Color32, Layout,
    },
    EguiContexts,
};
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
            for (_ent, brain) in brains.iter().sorted_by_key(|(_, b)| b.kills).rev() {
                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new(format!("{} {}", brain.id, &brain.name))
                            .text_style(egui::TextStyle::Heading),
                    );
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::TOP), |ui| {
                        ui.label(
                            egui::RichText::new(format!("{}-{}", brain.kills, brain.deaths))
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
                    for (_i, (_ent, brain)) in
                        brains.iter().sorted_by_key(|(_, b)| b.id).enumerate()
                    {
                        ui.group(|ui| {
                            ui.vertical(|ui| {
                                ui.heading(&brain.name);
                                // ui.group(|ui| {
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
                                ui.label(format!("ent: {}", brain.thinker.recent_entropy));
                                // });

                                // ui.horizontal_top(|ui| {
                                let mu = brain
                                    .thinker
                                    .recent_mu
                                    .iter()
                                    .zip(brain.thinker.recent_std.iter())
                                    .enumerate()
                                    .map(|(i, (mu, std))| {
                                        let rg = Vec2::new(*std, 1.0 / *std).normalize();
                                        Bar::new(i as f64, *mu as f64).fill(Color32::from_rgb(
                                            (rg.x * 255.0) as u8,
                                            (rg.y * 255.0) as u8,
                                            0,
                                        ))
                                        // .width(1.0 - *std as f64 / 6.0)
                                    })
                                    .collect_vec();
                                // let stddev = brain
                                //     .thinker
                                //     .recent_std
                                //     .iter()
                                //     .enumerate()
                                //     .map(|(i, val)| {
                                //         Bar::new(i as f64, *val as f64).fill(Color32::GREEN)
                                //     })
                                //     .collect_vec();
                                ui.group(|ui| {
                                    ui.horizontal(|ui| {
                                        ui.vertical(|ui| {
                                            ui.label("Action Space");
                                            egui::plot::Plot::new("ActionSpace")
                                                .center_y_axis(true)
                                                .height(80.0)
                                                .width(220.0)
                                                .show(ui, |plot| plot.bar_chart(BarChart::new(mu)));
                                        });
                                        // ui.vertical(|ui| {
                                        //     ui.label("Action Stddev");
                                        //     egui::plot::Plot::new("Std")
                                        //         .auto_bounds_y()
                                        //         .data_aspect(1.0 / 6.0)
                                        //         .height(50.0)
                                        //         .width(100.0)
                                        //         .show(ui, |plot| {
                                        //             plot.bar_chart(BarChart::new(stddev))
                                        //         });
                                        // })
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
