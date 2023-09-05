#![feature(return_position_impl_trait_in_trait)]
#![allow(clippy::type_complexity, clippy::too_many_arguments)]

use std::{collections::VecDeque, f32::consts::PI, fmt, path::Path, sync::Arc};

use bevy::{
    prelude::*,
    window::{PresentMode, WindowMode},
    winit::WinitSettings,
};
use bevy_egui::EguiPlugin;
use bevy_prng::ChaCha8Rng;
use bevy_rapier2d::prelude::*;

use bevy_rand::prelude::*;
use brains::{
    learners::{
        maddpg::{replay_buffer::MaddpgBuffer, MaddpgStatus},
        DEVICE,
    },
    AgentLearner, AgentPolicy, AgentValue, Policies, ValueEstimators,
};
use candle_core::Tensor;
use envs::{maps::tdm::TdmMap, tdm::Tdm, Env, Params};
use tensorboard_rs::summary_writer::SummaryWriter;
use ui::LogText;

pub mod brains;
pub mod envs;
pub mod names;
pub mod ui;

/// Something that stores a state with interior mutability.
pub trait Stateful: Sized {
    type State;
    fn replace_state(&self, s: Self::State) -> Option<Self::State>;
    fn get_state(&self) -> Option<&Self::State>;
}

#[inline(always)]
pub fn transform_angle_for_agent(x: f32) -> f32 {
    // rotate to be local to +Y
    let mut x = x + PI / 2.0;
    if x > PI {
        x -= 2.0 * PI;
    }
    if x < -PI {
        x += 2.0 * PI;
    }
    x
}

#[derive(Component)]
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

impl FrameStack<Box<[f32]>> {
    pub fn as_tensor(&self) -> Tensor {
        let mut obs = vec![];
        for frame in self.0.iter() {
            obs.push(Tensor::new(&**frame, &DEVICE).unwrap());
        }
        Tensor::stack(&obs, 0).unwrap().unsqueeze(0).unwrap()
    }

    pub fn as_flat_tensor(&self) -> Tensor {
        self.as_tensor().flatten_all().unwrap()
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

#[derive(Default, Component)]
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

fn run_env<E: crate::envs::Env, M: crate::envs::maps::Map>(
    seed: [u8; 32],
    env: E,
    _map: M,
    params: Params,
) {
    let ts = Timestamp::default();
    let mut p = Path::new("./training").to_path_buf();
    p.push(ts.to_string());
    std::fs::create_dir_all(&p).ok();
    p.push("env.yaml");
    params.to_yaml_file(p).ok();

    let learner: AgentLearner<E> = AgentLearner {
        gamma: 0.99,
        tau: 0.01,
        soft_update_interval: 1,
        steps_done: 0,
        batch_size: params.get_int("agent_training_batch_size").unwrap() as usize,
        buffer: MaddpgBuffer::new(
            params.get_int("num_agents").unwrap() as usize,
            params.get_int("agent_rb_max_len").unwrap() as usize,
        ),
        status: MaddpgStatus::default(),
    };

    let mut app = App::new();
    app.insert_resource(Msaa::default())
        .insert_resource(WinitSettings {
            focused_mode: bevy::winit::UpdateMode::Continuous,
            ..default()
        })
        .insert_resource(ts)
        .insert_resource(Policies(Vec::<AgentPolicy>::new()))
        .insert_resource(ValueEstimators(Vec::<AgentValue>::new()))
        .insert_resource(learner)
        .insert_resource(ui::LogText::default())
        .insert_resource(ClearColor(Color::DARK_GRAY))
        .add_plugins(EntropyPlugin::<ChaCha8Rng>::with_seed(seed))
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
        .add_plugins(
            RapierPhysicsPlugin::<NoUserData>::pixels_per_meter(100.0), // .with_default_system_setup(false),
        )
        // .add_plugins(RapierDebugRenderPlugin::default())
        .add_plugins(EguiPlugin)
        .insert_resource(env)
        .insert_resource(params)
        .add_systems(Startup, setup)
        .add_systems(Startup, (M::setup_system(), E::setup_system()).chain())
        .add_systems(Update, E::ui_system())
        .add_systems(Update, handle_input);
    E::add_main_systems(&mut app);
    app.run();
}

pub enum EnvKind {
    Tdm,
}

pub enum MapKind {
    Tdm,
}

fn main() {
    let _tch_seed: u64 = 0xcafebabe;
    let bevy_seed: [u8; 32] = [42; 32];

    let env = EnvKind::Tdm;

    match env {
        EnvKind::Tdm => {
            let params = Params::from_yaml_file("tdm.yaml").unwrap();
            run_env(bevy_seed, Tdm::init(), TdmMap, params);
        }
    }
}
