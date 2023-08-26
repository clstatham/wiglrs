#![allow(clippy::type_complexity, clippy::too_many_arguments)]

use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

use bevy::{
    app::ScheduleRunnerPlugin,
    core::FrameCount,
    math::Vec3Swizzles,
    prelude::*,
    sprite::{Anchor, MaterialMesh2dBundle},
    window::{PresentMode, WindowMode},
    winit::WinitSettings,
};
use bevy_egui::EguiPlugin;
use bevy_rapier2d::prelude::*;
use brains::{
    replay_buffer::SavedStep,
    thinkers::{self, ppo::PpoThinker, Thinker},
    Brain, BrainBank,
};
use hparams::{
    AGENT_ANG_MOVE_FORCE, AGENT_LIN_MOVE_FORCE, AGENT_MAX_HEALTH, AGENT_OPTIM_EPOCHS, AGENT_RADIUS,
    AGENT_RB_MAX_LEN, AGENT_SHOOT_DISTANCE, AGENT_TICK_RATE, NUM_AGENTS,
};
use itertools::Itertools;
use serde::{Deserialize, Serialize};
use tensorboard_rs::summary_writer::SummaryWriter;
use ui::{ui, LogText};

pub mod brains;
pub mod names;
pub mod ui;

#[derive(Default, Clone, Copy, Serialize, Deserialize)]
pub struct OtherState {
    pub rel_pos: Vec2,
    pub linvel: Vec2,
    pub direction_dotprod: f32,
    pub firing: bool,
}

pub const OTHER_STATE_LEN: usize = 6;

#[derive(Clone, Serialize, Deserialize)]
pub struct Observation {
    pub pos: Vec2,
    pub linvel: Vec2,
    pub direction: Vec2,
    pub health: f32,
    pub other_states: Vec<OtherState>,
}

pub const OBS_LEN: usize = OTHER_STATE_LEN * (NUM_AGENTS - 1) + 7;

impl Default for Observation {
    fn default() -> Self {
        Self {
            pos: Vec2::default(),
            linvel: Vec2::default(),
            direction: Vec2::default(),
            health: 0.0,
            other_states: vec![OtherState::default(); NUM_AGENTS - 1],
        }
    }
}

impl Observation {
    pub fn as_vec(&self) -> Vec<f32> {
        let mut out = vec![
            self.pos.x / 2000.0,
            self.pos.y / 2000.0,
            self.linvel.x / 2000.0,
            self.linvel.y / 2000.0,
            self.direction.x,
            self.direction.y,
            self.health / AGENT_MAX_HEALTH,
            // self.dt,
        ];
        for other in &self.other_states {
            out.extend_from_slice(&[
                other.rel_pos.x / 2000.0,
                other.rel_pos.y / 2000.0,
                other.linvel.x / 2000.0,
                other.linvel.y / 2000.0,
                other.direction_dotprod,
                if other.firing { 1.0 } else { 0.0 },
            ]);
        }
        out
    }

    pub fn dim() -> usize {
        Self::default().as_vec().len()
    }
}

pub mod hparams;

#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize)]
pub struct ActionMetadata {
    pub val: f32,
    pub logp: f32,
}

#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize)]
pub struct Action {
    lin_force: Vec2,
    ang_force: f32,
    shoot: f32,
    metadata: Option<ActionMetadata>,
}

pub const ACTION_LEN: usize = 4;

impl Action {
    pub fn from_slice(action: &[f32], metadata: Option<ActionMetadata>) -> Self {
        Self {
            lin_force: Vec2::new(action[0], action[1]).clamp(Vec2::splat(-1.0), Vec2::splat(1.0)),
            ang_force: action[2].clamp(-1.0, 1.0),
            shoot: action[3].clamp(-1.0, 1.0),
            metadata,
        }
    }

    pub fn as_vec(&self) -> Vec<f32> {
        vec![
            self.lin_force.x,
            self.lin_force.y,
            self.ang_force,
            self.shoot,
        ]
    }

    pub fn dim() -> usize {
        Self::default().as_vec().len()
    }
}

#[derive(Component)]
pub struct NameText {
    entity_following: Entity,
}

#[derive(Component)]
pub struct Health(pub f32);

#[derive(Component)]
pub struct HealthBar {
    entity_following: Entity,
}

#[derive(Component)]
pub struct Agent;

#[derive(Component, Default)]
pub struct ShootyLine;

#[derive(Bundle, Default)]
pub struct ShootyLineBundle {
    mesh: MaterialMesh2dBundle<ColorMaterial>,
    s: ShootyLine,
}

#[derive(Bundle)]
pub struct AgentBundle {
    rb: RigidBody,
    col: Collider,
    rest: Restitution,
    friction: Friction,
    gravity: GravityScale,
    velocity: Velocity,
    damping: Damping,
    force: ExternalForce,
    impulse: ExternalImpulse,
    mesh: MaterialMesh2dBundle<ColorMaterial>,
    health: Health,
    _a: Agent,
}
impl AgentBundle {
    pub fn new(
        pos: Vec3,
        color: Option<Color>,
        meshes: &mut ResMut<Assets<Mesh>>,
        materials: &mut ResMut<Assets<ColorMaterial>>,
    ) -> Self {
        Self {
            rb: RigidBody::Dynamic,
            col: Collider::ball(AGENT_RADIUS),
            rest: Restitution::coefficient(0.5),
            friction: Friction {
                coefficient: 0.0,
                combine_rule: CoefficientCombineRule::Min,
            },
            gravity: GravityScale(0.0),
            velocity: Velocity::default(),
            damping: Damping {
                angular_damping: 30.0,
                linear_damping: 10.0,
            },
            force: ExternalForce::default(),
            impulse: ExternalImpulse::default(),

            mesh: MaterialMesh2dBundle {
                material: materials.add(ColorMaterial::from(color.unwrap_or(Color::PURPLE))),
                mesh: meshes.add(shape::Circle::new(AGENT_RADIUS).into()).into(),
                transform: Transform::from_translation(pos),
                ..Default::default()
            },

            health: Health(AGENT_MAX_HEALTH),
            _a: Agent,
        }
    }
}

#[derive(Event)]
struct TrainBrain(usize);

#[derive(Event)]
struct DoneTraining;

#[derive(Component)]
struct TrainingText(usize, Timer);

fn check_train_brains(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut tx: EventWriter<TrainBrain>,
    mut rx: EventReader<DoneTraining>,
    mut text: Query<(Entity, &mut TrainingText)>,
    frame_count: Res<FrameCount>,
    time: Res<Time>,
) {
    if frame_count.0 as usize % AGENT_RB_MAX_LEN == AGENT_RB_MAX_LEN - 1 {
        commands.spawn((
            Text2dBundle {
                text: Text::from_section(
                    format!("Training (1/{})", NUM_AGENTS),
                    TextStyle {
                        font: asset_server.load("fonts/FiraSans-Bold.ttf"),
                        font_size: 72.0,
                        color: Color::YELLOW,
                    },
                ),
                transform: Transform::from_translation(Vec3::splat(0.0)),
                text_anchor: Anchor::Center,
                ..Default::default()
            },
            TrainingText(0, Timer::from_seconds(0.1, TimerMode::Once)),
        ));
    }
    for (text_ent, mut text) in text.iter_mut() {
        text.1.tick(time.delta());
        if text.1.just_finished() {
            tx.send(TrainBrain(text.0));
        }
        if rx.iter().next().is_some() {
            // for ent in text_ent.iter() {
            let id = text.0;
            commands.entity(text_ent).despawn();
            if id + 1 < NUM_AGENTS {
                commands.spawn((
                    Text2dBundle {
                        text: Text::from_section(
                            format!("Training ({}/{})", id + 2, NUM_AGENTS),
                            TextStyle {
                                font: asset_server.load("fonts/FiraSans-Bold.ttf"),
                                font_size: 72.0,
                                color: Color::YELLOW,
                            },
                        ),
                        transform: Transform::from_translation(Vec3::splat(0.0)),
                        text_anchor: Anchor::Center,
                        ..Default::default()
                    },
                    TrainingText(id + 1, Timer::from_seconds(0.1, TimerMode::Once)),
                ));
            }

            // }
        }
    }
}

fn train_brains(
    mut brains: NonSendMut<BrainBank>,
    mut log: ResMut<LogText>,
    frame_count: Res<FrameCount>,
    mut rx: EventReader<TrainBrain>,
    mut tx: EventWriter<DoneTraining>,
) {
    if let Some(id) = rx.iter().next() {
        // for brain in brains.values_mut() {
        // if brain.rb.buf.len() >= AGENT_RB_MAX_LEN {
        let brain = brains
            .iter_mut()
            .find_map(|(_, v)| if v.id == id.0 as u64 { Some(v) } else { None })
            .unwrap();
        log.push(format!(
            "{} {} replay buffer full, training for {} epochs...",
            brain.id, brain.name, AGENT_OPTIM_EPOCHS
        ));
        brain.learn(frame_count.0 as usize);
        log.push(format!(
            "{} {} Policy Loss: {}",
            brain.id, &brain.name, brain.thinker.recent_policy_loss
        ));
        log.push(format!(
            "{} {} Value Loss: {}",
            brain.id, &brain.name, brain.thinker.recent_value_loss
        ));
        log.push(format!(
            "{} {} Policy Clamp Ratio: {}",
            brain.id, &brain.name, brain.thinker.recent_nclamp
        ));
        // log.push(format!(
        //     "{} {} Policy KL Divergence: {}",
        //     brain.id, &brain.name, brain.thinker.recent_kl
        // ));
        brain.rb.buf.clear();
        // }
        // }
        tx.send(DoneTraining);
    }
}

fn check_respawn_all(
    mut commands: Commands,
    mut brains: NonSendMut<BrainBank>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    asset_server: Res<AssetServer>,
) {
    for agent in brains.keys().copied().collect::<Vec<_>>() {
        if commands.get_entity(agent).is_none() {
            let mut brain = brains.remove(&agent).unwrap();

            // brain.learn(frame_count.0 as usize);

            brain.deaths += 1;

            spawn_agent(
                brain,
                &mut commands,
                &mut meshes,
                &mut materials,
                &mut brains,
                &asset_server,
            );
            // avg_kills.0 = brains.values().map(|b| b.kills as f32).sum::<f32>() / NUM_AGENTS as f32;
        }
    }
}

#[derive(Component)]
pub struct Wall;

fn spawn_agent(
    brain: Brain<PpoThinker>,
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<ColorMaterial>>,
    brains: &mut NonSendMut<BrainBank>,
    asset_server: &Res<AssetServer>,
) {
    let agent_pos = Vec3::new(
        (rand::random::<f32>() - 0.5) * 500.0,
        (rand::random::<f32>() - 0.5) * 500.0,
        0.0,
    );
    let id = commands
        .spawn(AgentBundle::new(
            agent_pos,
            Some(brain.color),
            meshes,
            materials,
        ))
        .insert(ActiveEvents::all())
        .with_children(|parent| {
            parent.spawn(ShootyLineBundle {
                mesh: MaterialMesh2dBundle {
                    mesh: meshes
                        .add(Mesh::from(shape::Box::new(3.0, 1.0, 0.0)))
                        .into(),
                    material: materials.add(ColorMaterial::from(Color::WHITE)),
                    transform: Transform::from_translation(Vec3::new(0.0, 0.0, 0.0)),
                    visibility: Visibility::Hidden,
                    ..Default::default()
                },
                ..Default::default()
            });
        })
        .id();
    commands.spawn((
        Text2dBundle {
            text: Text::from_section(
                format!("{} {} {}.0", brain.id, &brain.name, brain.deaths),
                TextStyle {
                    font: asset_server.load("fonts/FiraSans-Bold.ttf"),
                    font_size: 20.0,
                    color: Color::WHITE,
                },
            ),
            transform: Transform::from_translation(agent_pos + Vec3::new(0.0, 0.0, 2.0)),
            ..Default::default()
        },
        NameText {
            entity_following: id,
        },
    ));
    commands.spawn((
        HealthBar {
            entity_following: id,
        },
        MaterialMesh2dBundle {
            mesh: meshes.add(shape::Box::new(1.0, 6.0, 0.0).into()).into(),
            material: materials.add(ColorMaterial::from(Color::RED)),
            transform: Transform::from_translation(agent_pos + Vec3::new(0.0, -AGENT_RADIUS, 2.0)),
            ..Default::default()
        },
    ));
    brains.insert(id, brain);
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    mut brains: NonSendMut<BrainBank>,
    asset_server: Res<AssetServer>,
    timestamp: Res<Timestamp>,
) {
    // writer.init(None, &timestamp);

    commands.spawn(Camera2dBundle {
        transform: Transform::from_xyz(0.0, 0.0, 500.0),
        ..Default::default()
    });

    commands
        .spawn(Collider::cuboid(500.0, 10.0))
        .insert(SpriteBundle {
            sprite: Sprite {
                color: Color::BLACK,
                custom_size: Some(Vec2::new(1000.0, 20.0)),
                ..default()
            },
            ..default()
        })
        .insert(Wall)
        .insert(ActiveEvents::all())
        .insert(TransformBundle::from(Transform::from_xyz(0.0, -300.0, 0.0)));

    commands
        .spawn(Collider::cuboid(500.0, 10.0))
        .insert(SpriteBundle {
            sprite: Sprite {
                color: Color::BLACK,
                custom_size: Some(Vec2::new(1000.0, 20.0)),
                ..default()
            },
            ..default()
        })
        .insert(Wall)
        .insert(ActiveEvents::all())
        .insert(TransformBundle::from(Transform::from_xyz(0.0, 300.0, 0.0)));
    commands
        .spawn(Collider::cuboid(10.0, 300.0))
        .insert(SpriteBundle {
            sprite: Sprite {
                color: Color::BLACK,
                custom_size: Some(Vec2::new(20.0, 600.0)),
                ..default()
            },
            ..default()
        })
        .insert(Wall)
        .insert(ActiveEvents::all())
        .insert(TransformBundle::from(Transform::from_xyz(-500.0, 0.0, 0.0)));
    commands
        .spawn(Collider::cuboid(10.0, 300.0))
        .insert(SpriteBundle {
            sprite: Sprite {
                color: Color::BLACK,
                custom_size: Some(Vec2::new(20.0, 600.0)),
                ..default()
            },
            ..default()
        })
        .insert(Wall)
        .insert(ActiveEvents::all())
        .insert(TransformBundle::from(Transform::from_xyz(500.0, 0.0, 0.0)));

    for _ in 0..NUM_AGENTS {
        let brain = Brain::new(thinkers::ppo::PpoThinker::default(), &timestamp);
        spawn_agent(
            brain,
            &mut commands,
            &mut meshes,
            &mut materials,
            &mut brains,
            &asset_server,
        );
    }
}

fn update(
    mut commands: Commands,
    mut agents: Query<
        (Entity, &mut ExternalForce, &mut Velocity, &Transform),
        (With<Agent>, Without<NameText>),
    >,
    mut brains: NonSendMut<BrainBank>,
    mut health: Query<&mut Health>,
    agents_shootin: Query<(&Transform, &Children), (With<Agent>, Without<ShootyLine>)>,
    mut line_vis: Query<
        (&mut Visibility, &mut Transform),
        (With<ShootyLine>, Without<Agent>, Without<NameText>),
    >,
    mut name_text_t: Query<
        (Entity, &mut Transform, &mut Text, &NameText),
        (Without<Agent>, Without<ShootyLine>),
    >,
    mut health_bar_t: Query<
        (Entity, &mut Transform, &HealthBar),
        (Without<Agent>, Without<ShootyLine>, Without<NameText>),
    >,
    cx: Res<RapierContext>,
    _collision_events: EventReader<ContactForceEvent>,
    _walls: Query<&Collider, (Without<Agent>, With<Wall>)>,
    _keys: Res<Input<KeyCode>>,
    mut log: ResMut<LogText>,
    frame_count: Res<FrameCount>,
) {
    let mut all_states = BTreeMap::new();
    let mut all_actions = BTreeMap::new();
    let mut all_rewards = BTreeMap::new();
    let mut all_terminals = BTreeMap::new();

    for (t_ent, mut t, mut text, text_comp) in name_text_t.iter_mut() {
        if let Ok(agent) = agents.get_component::<Transform>(text_comp.entity_following) {
            t.translation = agent.translation + Vec3::new(0.0, 40.0, 2.0);
            let brain = &brains[&text_comp.entity_following];
            text.sections[0].value = format!(
                "{} {} {}-{}",
                brain.id, &brain.name, brain.kills, brain.deaths
            );
        } else {
            commands.entity(t_ent).despawn();
        }
    }
    for (t_ent, mut t, hb) in health_bar_t.iter_mut() {
        if let Ok(agent) = agents.get_component::<Transform>(hb.entity_following) {
            t.translation = agent.translation + Vec3::new(0.0, 25.0, 2.0);
            let health = health.get_component::<Health>(hb.entity_following).unwrap();
            t.scale = Vec3::new(health.0 / AGENT_MAX_HEALTH * 100.0, 1.0, 1.0);
        } else {
            commands.entity(t_ent).despawn();
        }
    }

    for (agent, _, velocity, transform) in agents.iter() {
        let mut my_state = Observation {
            pos: transform.translation.xy(),
            linvel: velocity.linvel,
            direction: transform.local_y().xy(),
            health: health.get_component::<Health>(agent).unwrap().0,
            other_states: vec![OtherState::default(); NUM_AGENTS - 1],
        };

        for (i, (other, _, other_vel, other_transform)) in agents
            .iter()
            .filter(|a| a.0 != agent)
            .sorted_by_key(|a| a.3.translation.distance(transform.translation) as i64)
            .enumerate()
        {
            let other_state = OtherState {
                rel_pos: transform.translation.xy() - other_transform.translation.xy(),
                linvel: other_vel.linvel,
                direction_dotprod: transform.local_y().xy().dot(other_transform.local_y().xy()),
                firing: brains[&other].last_action.shoot > 0.0,
            };
            my_state.other_states[i] = other_state;
        }

        all_states.insert(agent, my_state.clone());
        let action = if frame_count.0 as usize % AGENT_TICK_RATE == 0 {
            brains
                .get_mut(&agent)
                .unwrap()
                .act(my_state, frame_count.0 as usize)
        } else {
            brains.get(&agent).unwrap().last_action
        };

        all_actions.insert(agent, action);
        all_rewards.insert(agent, 0.0);
        all_terminals.insert(agent, false);
    }
    let mut dead_agents = BTreeSet::default();
    for (agent, mut force, _velocity, _transform) in agents.iter_mut() {
        if !dead_agents.contains(&agent) {
            if all_actions[&agent].shoot > 0.0 && !dead_agents.contains(&agent) {
                let (ray_dir, ray_pos) = {
                    let (transform, childs) = agents_shootin.get(agent).unwrap();
                    let ray_dir = transform.local_y().xy();
                    let ray_pos = transform.translation.xy() + ray_dir * (AGENT_RADIUS + 2.0);
                    for child in childs.iter() {
                        if let Ok(mut line) = line_vis.get_mut(*child) {
                            *line.0 = Visibility::Visible;
                        }
                    }
                    (ray_dir, ray_pos)
                };

                let mut filter = QueryFilter::default().exclude_collider(agent);
                for dead in dead_agents.iter() {
                    filter = filter.exclude_collider(*dead);
                }

                if let Some((hit_entity, toi)) =
                    cx.cast_ray(ray_pos, ray_dir, AGENT_SHOOT_DISTANCE, false, filter)
                {
                    let (_, childs) = agents_shootin.get(agent).unwrap();
                    for child in childs.iter() {
                        if let Ok((_, mut t)) = line_vis.get_mut(*child) {
                            t.scale = Vec3::new(1.0, toi, 1.0);
                            *t = t.with_translation(Vec3::new(0.0, toi / 2.0, 0.0));
                        }
                    }
                    if let Ok(mut health) = health.get_mut(hit_entity) {
                        health.0 -= 5.0;
                        *all_rewards.get_mut(&agent).unwrap() += 1.0;
                        *all_rewards.get_mut(&hit_entity).unwrap() -= 1.0;
                        if health.0 <= 0.0 {
                            dead_agents.insert(hit_entity);
                            *all_terminals.get_mut(&hit_entity).unwrap() = true;
                            *all_rewards.get_mut(&agent).unwrap() += 100.0;
                            *all_rewards.get_mut(&hit_entity).unwrap() -= 100.0;
                            brains.get_mut(&agent).unwrap().kills += 1;
                            let msg = format!(
                                "{} killed {}! Nice!",
                                &brains[&agent].name, &brains[&hit_entity].name
                            );
                            log.push(msg);
                        }
                    }
                } else {
                    let (_, childs) = agents_shootin.get(agent).unwrap();
                    for child in childs.iter() {
                        if let Ok((_, mut t)) = line_vis.get_mut(*child) {
                            t.scale = Vec3::new(1.0, AGENT_SHOOT_DISTANCE, 1.0);
                            *t =
                                t.with_translation(Vec3::new(0.0, AGENT_SHOOT_DISTANCE / 2.0, 0.0));
                        }
                    }
                    *all_rewards.get_mut(&agent).unwrap() -= 1.0;
                }
            } else {
                let (_, childs) = agents_shootin.get(agent).unwrap();
                for child in childs.iter() {
                    if let Ok(mut line) = line_vis.get_mut(*child) {
                        *line.0 = Visibility::Hidden;
                    }
                }
            }

            force.force = all_actions[&agent].lin_force * AGENT_LIN_MOVE_FORCE;
            force.torque = all_actions[&agent].ang_force * AGENT_ANG_MOVE_FORCE;
        }
    }

    for (agent, _, _, _) in agents.iter() {
        let fs = brains.get(&agent).unwrap().fs.clone();
        // for brain in brains.values_mut() {
        brains.get_mut(&agent).unwrap().rb.remember(SavedStep {
            obs: fs.clone(),
            action: all_actions.get(&agent).unwrap().to_owned(),
            reward: all_rewards.get(&agent).unwrap().to_owned(),
            terminal: all_terminals.get(&agent).unwrap().to_owned(),
        });
        // }
    }

    for agent in dead_agents {
        commands.entity(agent).despawn_recursive();
    }
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

impl<T: Thinker> Drop for Brain<T> {
    fn drop(&mut self) {
        warn!("Saving {}...", &self.name);
        if let Err(e) = self.save() {
            error!("Failed to save {}: {:?}", &self.name, e);
        }
    }
}

#[derive(Resource, Default)]
pub struct AvgAgentKills(f32);

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

fn main() {
    App::new()
        .insert_resource(Msaa::default())
        .insert_resource(WinitSettings {
            focused_mode: bevy::winit::UpdateMode::Continuous,
            ..default()
        })
        .insert_resource(Timestamp::default())
        .insert_resource(AvgAgentKills::default())
        .insert_resource(ui::LogText::default())
        .insert_resource(ClearColor(Color::DARK_GRAY))
        .insert_non_send_resource(BrainBank::default())
        .add_plugins(ScheduleRunnerPlugin {
            run_mode: bevy::app::RunMode::Loop {
                // wait: Some(Duration::from_secs_f64(1.0 / 60.0)),
                wait: None,
            },
        })
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                present_mode: bevy::window::PresentMode::AutoNoVsync,
                title: "wiglrs".to_owned(),
                mode: WindowMode::BorderlessFullscreen,
                ..default()
            }),
            ..Default::default()
        }))
        .insert_resource(RapierConfiguration {
            gravity: Vec2::ZERO,
            timestep_mode: TimestepMode::Fixed {
                dt: 1.0 / 60.0,
                substeps: 2,
            },
            ..Default::default()
        })
        .add_plugins(RapierPhysicsPlugin::<NoUserData>::pixels_per_meter(100.0))
        .add_plugins(RapierDebugRenderPlugin::default())
        .add_plugins(EguiPlugin)
        .add_systems(Startup, setup)
        .add_systems(Update, update)
        .add_systems(Update, check_respawn_all)
        .add_systems(PostUpdate, check_train_brains)
        .add_systems(Update, train_brains)
        .add_systems(Update, ui)
        .add_systems(Update, handle_input)
        .add_event::<TrainBrain>()
        .add_event::<DoneTraining>()
        .run();
}
