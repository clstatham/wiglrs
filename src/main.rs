#![feature(return_position_impl_trait_in_trait)]
#![allow(clippy::type_complexity, clippy::too_many_arguments)]

use std::{collections::VecDeque, fmt, sync::Arc};

use bevy::{
    prelude::*,
    window::{PresentMode, WindowMode},
    winit::WinitSettings,
};
use bevy_egui::EguiPlugin;
use bevy_rapier2d::prelude::*;

use burn_tensor::backend::Backend;
use envs::{
    ffa::Ffa,
    maps::{tdm::TdmMap, Map},
    tdm::Tdm,
    Env,
};
use tensorboard_rs::summary_writer::SummaryWriter;
use ui::LogText;

pub mod brains;
pub mod envs;
pub mod names;
pub mod ui;

pub struct FrameStack<O>(pub VecDeque<O>);

impl<O> Clone for FrameStack<O>
where
    O: Clone,
{
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<O> fmt::Debug for FrameStack<O> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "FrameStack")
    }
}

impl<O> FrameStack<O> {
    pub fn push(&mut self, s: O, max_len: Option<usize>) {
        if let Some(max_len) = max_len {
            while self.0.len() >= max_len {
                self.0.pop_front();
            }
        }

        self.0.push_back(s);
    }
}
impl<O> FrameStack<O>
where
    O: Clone,
{
    pub fn as_vec(&self) -> Vec<O> {
        self.0.clone().into()
    }
}

#[derive(Component)]
pub struct Wall;

fn setup(mut commands: Commands) {
    commands.spawn(Camera2dBundle {
        transform: Transform::from_xyz(0.0, 0.0, 500.0),
        ..Default::default()
    });
}

#[derive(Resource, Clone)]
pub struct Timestamp(Arc<String>);

impl Default for Timestamp {
    fn default() -> Self {
        use chrono::prelude::*;
        Self(Arc::new(format!("{}", Local::now().format("%Y%m%d%H%M%S"))))
    }
}

impl std::fmt::Display for Timestamp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Default)]
pub struct TbWriter(Option<SummaryWriter>);

impl TbWriter {
    pub fn init(&mut self, subdir: Option<&str>, timestamp: &Timestamp) {
        let dir = if let Some(subdir) = subdir {
            format!("training/{}/{}", timestamp, subdir)
        } else {
            format!("training/{}", timestamp)
        };
        self.0 = Some(SummaryWriter::new(dir));
    }

    pub fn add_scalar(&mut self, label: impl AsRef<str>, scalar: f32, step: usize) {
        if let Some(writer) = self.0.as_mut() {
            writer.add_scalar(label.as_ref(), scalar, step);
        } else {
            warn!("Attempted to write to uninitialized TbWriter");
        }
    }
}

impl Drop for TbWriter {
    fn drop(&mut self) {
        if let Some(w) = self.0.as_mut() {
            w.flush();
        }
    }
}

fn handle_input(
    mut window: Query<&mut Window>,
    keys: Res<Input<KeyCode>>,
    mut log_text: ResMut<LogText>,
) {
    let mut window = window.get_single_mut().unwrap();
    if keys.just_pressed(KeyCode::Space) {
        if window.present_mode == PresentMode::AutoNoVsync {
            log_text.push("VSync On".into());
            window.present_mode = PresentMode::AutoVsync;
        } else if window.present_mode == PresentMode::AutoVsync {
            log_text.push("VSync Off".into());
            window.present_mode = PresentMode::AutoNoVsync;
        }
    }
}

fn run_env<E: Env, M: Map>() {
    App::new()
        .insert_resource(Msaa::default())
        .insert_resource(WinitSettings {
            focused_mode: bevy::winit::UpdateMode::Continuous,
            ..default()
        })
        .insert_resource(Timestamp::default())
        .insert_resource(ui::LogText::default())
        .insert_resource(ClearColor(Color::DARK_GRAY))
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                present_mode: bevy::window::PresentMode::AutoNoVsync,
                title: "wiglrs".to_owned(),
                mode: WindowMode::Windowed,
                ..default()
            }),
            ..Default::default()
        }))
        .add_plugins(bevy_framepace::FramepacePlugin)
        .insert_resource(bevy_framepace::FramepaceSettings {
            // limiter: bevy_framepace::Limiter::Manual(Duration::from_secs_f64(1.0 / 144.0)),
            limiter: bevy_framepace::Limiter::Off,
        })
        // .add_plugins(LogDiagnosticsPlugin::default())
        // .add_plugins(FrameTimeDiagnosticsPlugin)
        .insert_resource(RapierConfiguration {
            gravity: Vec2::ZERO,
            timestep_mode: TimestepMode::Fixed {
                dt: 1.0 / 60.0,
                substeps: 1,
            },
            ..Default::default()
        })
        .add_plugins(RapierPhysicsPlugin::<NoUserData>::pixels_per_meter(100.0))
        // .add_plugins(RapierDebugRenderPlugin::default())
        .insert_resource(E::init())
        .add_plugins(EguiPlugin)
        .add_systems(Startup, setup)
        .add_systems(Startup, E::setup_system::<M>())
        .add_systems(Update, E::ui_system())
        .add_systems(Update, handle_input)
        .add_systems(Update, E::main_system())
        .run();
}

fn main() {
    burn_tch::TchBackend::<f32>::seed(rand::random());
    run_env::<Tdm, TdmMap>();
}
