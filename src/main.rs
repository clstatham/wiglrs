#![allow(clippy::type_complexity, clippy::too_many_arguments)]

use std::{
    collections::{BTreeMap, VecDeque},
    sync::atomic::AtomicU64,
};

use bevy::{
    core::FrameCount, math::Vec3Swizzles, prelude::*, sprite::MaterialMesh2dBundle,
    winit::WinitSettings,
};
use bevy_rapier2d::prelude::*;
use brains::{
    replay_buffer::SavedStep,
    thinkers::{self, Thinker},
    Brain, BrainBank,
};
use hparams::{
    AGENT_ANG_MOVE_FORCE, AGENT_LIN_MOVE_FORCE, AGENT_MAX_HEALTH, AGENT_RADIUS,
    AGENT_SHOOT_DISTANCE, NUM_AGENTS,
};
use tensorboard_rs::summary_writer::SummaryWriter;

pub mod brains;
pub mod names;

#[derive(Default, Clone, Copy)]
pub struct OtherState {
    pub rel_pos: Vec2,
    pub linvel: Vec2,
    pub direction: Vec2,
}

#[derive(Default, Clone, Copy)]
pub struct Observation {
    pub pos: Vec2,
    pub linvel: Vec2,
    pub direction: Vec2,
    pub dt: f32,
    pub other_states: [OtherState; NUM_AGENTS],
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
            // self.dt,
        ];
        for other in &self.other_states {
            out.extend_from_slice(&[
                other.rel_pos.x / 2000.0,
                other.rel_pos.y / 2000.0,
                other.linvel.x / 2000.0,
                other.linvel.y / 2000.0,
                other.direction.x,
                other.direction.y,
            ]);
        }
        out
    }

    pub fn dim() -> usize {
        Self::default().as_vec().len()
    }
}

pub mod hparams;

#[derive(Debug, Default, Clone, Copy)]
pub struct Action {
    lin_force: Vec2,
    ang_force: f32,
    shoot: bool,
}

impl Action {
    pub fn from_slice(action: &[f32]) -> Self {
        Self {
            lin_force: Vec2::new(
                action[0] * AGENT_LIN_MOVE_FORCE,
                action[1] * AGENT_LIN_MOVE_FORCE,
            ),
            ang_force: action[2] * AGENT_ANG_MOVE_FORCE,
            shoot: action[3] > 0.0,
        }
    }

    pub fn as_vec(&self) -> Vec<f32> {
        vec![
            self.lin_force.x / AGENT_LIN_MOVE_FORCE,
            self.lin_force.y / AGENT_LIN_MOVE_FORCE,
            self.ang_force / AGENT_ANG_MOVE_FORCE,
            if self.shoot { 1.0 } else { 0.0 },
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

fn check_respawn_all(
    mut commands: Commands,
    mut brains: NonSendMut<brains::BrainBank>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    mut writer: NonSendMut<TbWriter>,
    asset_server: Res<AssetServer>,
    frame_count: Res<FrameCount>,
) {
    for agent in brains.keys().copied().collect::<Vec<_>>() {
        if commands.get_entity(agent).is_none() {
            let all_rbs = brains.values().map(|b| b.rb.clone()).collect::<Vec<_>>();
            let mut brain = brains.remove(&agent).unwrap();
            // for (a, brain) in brains.iter_mut() {
            let mean_reward =
                brain.rb.buf.iter().map(|s| s.reward).sum::<f32>() / brain.rb.buf.len() as f32;
            println!("{} reward: {mean_reward}", &brain.name);
            writer.0.add_scalar(
                &format!("Reward/{}", brain.id),
                mean_reward,
                frame_count.0 as usize,
            );
            let mut total_loss = 0.0;
            let mut n_losses = 0;
            // for rb in all_rbs.iter() {
            //     if let Some((_reward, loss)) = brain.learn(Some(rb.clone())) {
            //         total_loss += loss;
            //         // total_reward += reward;
            //         n_losses += 1;
            //     }
            // }
            // writer.0.add_scalar(
            //     &format!("Loss/{}", brain.id),
            //     total_loss / n_losses as f32,
            //     frame_count.0 as usize,
            // );
            // println!("{} loss: {}", &brain.name, total_loss / n_losses as f32);
            // }
            // brain transplant

            brain.rb.buf.clear();
            brain.version += 1;

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
}

#[derive(Component)]
pub struct Wall;

fn spawn_agent(
    brain: Brain<thinkers::RandomThinker>,
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
                format!("{} {}.0", &brain.name, brain.version),
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
    brains.insert(id, brain);
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    mut brains: NonSendMut<BrainBank>,
    asset_server: Res<AssetServer>,
) {
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
        let brain = Brain::new(thinkers::RandomThinker);
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
        (Entity, &mut Transform, &NameText),
        (Without<Agent>, Without<ShootyLine>),
    >,
    cx: Res<RapierContext>,
    _collision_events: EventReader<ContactForceEvent>,
    _walls: Query<&Collider, (Without<Agent>, With<Wall>)>,
    _keys: Res<Input<KeyCode>>,
    time: Res<Time>,
) {
    let mut all_states = BTreeMap::new();
    let mut all_actions = BTreeMap::new();
    let mut all_rewards = BTreeMap::new();
    let mut all_terminals = BTreeMap::new();

    for (t_ent, mut t, text) in name_text_t.iter_mut() {
        if let Ok(agent) = agents.get_component::<Transform>(text.entity_following) {
            t.translation = agent.translation + Vec3::new(0.0, 40.0, 0.0);
        } else {
            commands.entity(t_ent).despawn();
        }
    }

    for (agent, _, velocity, transform) in agents.iter() {
        let mut my_state = Observation {
            pos: transform.translation.xy(),
            linvel: velocity.linvel,
            direction: transform.local_y().xy(),
            dt: time.delta_seconds(),
            other_states: [OtherState::default(); NUM_AGENTS],
        };

        for (other, _, other_vel, other_transform) in agents.iter() {
            let other_state = OtherState {
                rel_pos: transform.translation.xy() - other_transform.translation.xy(),
                linvel: other_vel.linvel,
                direction: other_transform.local_y().xy(),
            };
            my_state.other_states[brains[&other].id as usize] = other_state;
        }

        // brains.get_mut(&agent).unwrap().frame_stack.push(my_state);

        // let state = brains.get(&agent).unwrap().frame_stack.as_tensor();
        all_states.insert(agent, my_state);
        let action = brains.get_mut(&agent).unwrap().act(my_state);

        all_actions.insert(agent, action);
        all_rewards.insert(agent, 0.0);
        all_terminals.insert(agent, false);
    }
    let mut dead_agents = vec![];
    for (agent, mut force, _velocity, transform) in agents.iter_mut() {
        let distance_to_center = transform.translation.distance(Vec3::splat(0.0));
        if distance_to_center >= 100.0 {
            let mut hp = health.get_mut(agent).unwrap();
            hp.0 -= distance_to_center / 10000.0;
            *all_rewards.get_mut(&agent).unwrap() -= distance_to_center / 1000.0;
            if hp.0 <= 0.0 {
                dead_agents.push(agent);
                *all_terminals.get_mut(&agent).unwrap() = true;
            }
        }

        if all_actions[&agent].shoot && !dead_agents.contains(&agent) {
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

            if let Some((hit_entity, toi)) = cx.cast_ray(
                ray_pos,
                ray_dir,
                AGENT_SHOOT_DISTANCE,
                false,
                QueryFilter::default().exclude_collider(agent),
            ) {
                let (_, childs) = agents_shootin.get(agent).unwrap();
                for child in childs.iter() {
                    if let Ok((_, mut t)) = line_vis.get_mut(*child) {
                        t.scale = Vec3::new(1.0, toi, 1.0);
                        *t = t.with_translation(Vec3::new(0.0, toi / 2.0, 0.0));
                    }
                }
                if let Ok(mut health) = health.get_mut(hit_entity) {
                    health.0 -= 1.0;
                    *all_rewards.get_mut(&agent).unwrap() += 2.0;
                    *all_rewards.get_mut(&hit_entity).unwrap() -= 1.0;
                    if health.0 <= 0.0 {
                        dead_agents.push(hit_entity);
                        *all_terminals.get_mut(&hit_entity).unwrap() = true;
                        *all_rewards.get_mut(&agent).unwrap() += 1000.0;
                        *all_rewards.get_mut(&hit_entity).unwrap() -= 1000.0;
                        println!(
                            "{} killed {}! Nice!",
                            &brains[&agent].name, &brains[&hit_entity].name
                        );
                    }
                }
            } else {
                let (_, childs) = agents_shootin.get(agent).unwrap();
                for child in childs.iter() {
                    if let Ok((_, mut t)) = line_vis.get_mut(*child) {
                        t.scale = Vec3::new(1.0, AGENT_SHOOT_DISTANCE, 1.0);
                        *t = t.with_translation(Vec3::new(0.0, AGENT_SHOOT_DISTANCE / 2.0, 0.0));
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

        force.force = all_actions[&agent].lin_force;
        force.torque = all_actions[&agent].ang_force;

        // let mut applied_force = Vec2::default();
        // let mut applied_torque = 0.0;

        // if keys.pressed(KeyCode::A) {
        //     applied_force += Vec2::NEG_X * AGENT_LIN_MOVE_FORCE;
        // }
        // if keys.pressed(KeyCode::D) {
        //     applied_force += Vec2::X * AGENT_LIN_MOVE_FORCE;
        // }
        // if keys.pressed(KeyCode::W) {
        //     applied_force += Vec2::Y * AGENT_LIN_MOVE_FORCE;
        // }
        // if keys.pressed(KeyCode::S) {
        //     applied_force += Vec2::NEG_Y * AGENT_LIN_MOVE_FORCE;
        // }

        // if keys.pressed(KeyCode::Q) {
        //     applied_torque += AGENT_ANG_MOVE_FORCE;
        // }
        // if keys.pressed(KeyCode::E) {
        //     applied_torque -= AGENT_ANG_MOVE_FORCE;
        // }
        // force.force = applied_force;
        // force.torque = applied_torque;

        // clamp velocity
        // velocity.linvel = velocity.linvel.clamp(
        //     Vec2::new(-AGENT_MAX_LIN_VEL, -AGENT_MAX_LIN_VEL),
        //     Vec2::new(AGENT_MAX_LIN_VEL, AGENT_MAX_LIN_VEL),
        // );
        // velocity.angvel = velocity.angvel.clamp(-AGENT_MAX_ANG_VEL, AGENT_MAX_ANG_VEL);
    }

    for (agent, _, _, _) in agents.iter() {
        brains.get_mut(&agent).unwrap().rb.remember(SavedStep {
            obs: all_states.remove(&agent).unwrap(),
            action: all_actions.remove(&agent).unwrap(),
            reward: all_rewards.remove(&agent).unwrap(),
            terminal: all_terminals.remove(&agent).unwrap(),
        });
    }

    for agent in dead_agents {
        commands.entity(agent).despawn_recursive();
    }
}

pub struct TbWriter(SummaryWriter);

impl Default for TbWriter {
    fn default() -> Self {
        use chrono::prelude::*;
        let timestamp = Local::now().format("%Y%m%d%H%M%S");
        Self(tensorboard_rs::summary_writer::SummaryWriter::new(format!(
            "training/{timestamp}"
        )))
    }
}

fn main() {
    App::new()
        .insert_resource(Msaa::default())
        .insert_resource(WinitSettings {
            focused_mode: bevy::winit::UpdateMode::Continuous,
            ..default()
        })
        .insert_resource(ClearColor(Color::DARK_GRAY))
        .insert_non_send_resource(TbWriter::default())
        .insert_non_send_resource(BrainBank::default())
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                present_mode: bevy::window::PresentMode::AutoVsync,
                title: "wiglrs".to_owned(),
                ..default()
            }),
            ..Default::default()
        }))
        .add_plugins(RapierPhysicsPlugin::<NoUserData>::pixels_per_meter(100.0))
        .add_plugins(RapierDebugRenderPlugin::default())
        .add_systems(Startup, setup)
        .add_systems(Update, update)
        .add_systems(Update, check_respawn_all)
        .run();
}
