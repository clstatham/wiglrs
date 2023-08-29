use std::{collections::BTreeMap, f32::consts::PI};

use bevy::{
    core::FrameCount, ecs::schedule::SystemConfigs, math::Vec3Swizzles, prelude::*,
    sprite::MaterialMesh2dBundle,
};
use bevy_rapier2d::prelude::*;
use bevy_tasks::AsyncComputeTaskPool;
use itertools::Itertools;

use crate::{
    brains::{
        replay_buffer::{PpoBuffer, PpoMetadata, Sart},
        thinkers::ppo::PpoThinker,
        AgentThinker, Brain, BrainBank,
    },
    ui::LogText,
    FrameStack, Timestamp, Wall,
};

use super::{Action, Env, Observation};

#[derive(Debug, Resource, Clone, Copy)]
pub struct FfaParams {
    pub num_agents: usize,
    pub agent_hidden_dim: usize,
    pub agent_actor_lr: f64,
    pub agent_critic_lr: f64,
    pub agent_training_epochs: usize,
    pub agent_training_batch_size: usize,
    pub agent_entropy_beta: f32,
    pub agent_update_interval: usize,
    pub agent_rb_max_len: usize,
    pub agent_frame_stack_len: usize,
    pub agent_radius: f32,
    pub agent_lin_move_force: f32,
    pub agent_ang_move_force: f32,
    pub agent_max_health: f32,
    pub agent_shoot_distance: f32,
}

impl Default for FfaParams {
    fn default() -> Self {
        Self {
            num_agents: 6,
            agent_hidden_dim: 128,
            agent_actor_lr: 1e-5,
            agent_critic_lr: 1e-4,
            agent_training_epochs: 25,
            agent_training_batch_size: 128,
            agent_entropy_beta: 0.001,
            agent_update_interval: 2_000,
            agent_rb_max_len: 100_000,
            agent_frame_stack_len: 5,
            agent_radius: 20.0,
            agent_lin_move_force: 600.0,
            agent_ang_move_force: 1.0,
            agent_max_health: 100.0,
            agent_shoot_distance: 500.0,
        }
    }
}

#[derive(Default, Debug, Clone, Copy)]
pub struct OtherState {
    pub rel_pos: Vec2,
    pub linvel: Vec2,
    pub direction: Vec2,
    pub firing: bool,
}

pub const OTHER_STATE_LEN: usize = 7;

#[derive(Clone, Debug)]
pub struct FfaObs {
    pub pos: Vec2,
    pub linvel: Vec2,
    pub direction: Vec2,
    pub health: f32,
    pub up_wall_dist: f32,
    pub down_wall_dist: f32,
    pub left_wall_dist: f32,
    pub right_wall_dist: f32,
    pub other_states: Vec<OtherState>,
}

pub const BASE_STATE_LEN: usize = 11;

impl Observation<Ffa> for FfaObs {
    fn new_frame_stack(params: &FfaParams) -> FrameStack<Self> {
        FrameStack(
            vec![
                Self {
                    pos: Vec2::ZERO,
                    linvel: Vec2::ZERO,
                    direction: Vec2::ZERO,
                    health: params.agent_max_health,
                    up_wall_dist: 0.0,
                    left_wall_dist: 0.0,
                    right_wall_dist: 0.0,
                    down_wall_dist: 0.0,
                    other_states: vec![OtherState::default(); params.num_agents - 1],
                };
                params.agent_frame_stack_len
            ]
            .into(),
        )
    }

    fn as_vec(&self, params: &FfaParams) -> Vec<f32> {
        let mut out = vec![
            self.pos.x / 2000.0,
            self.pos.y / 2000.0,
            self.linvel.x / 2000.0,
            self.linvel.y / 2000.0,
            self.direction.x,
            self.direction.y,
            self.up_wall_dist / 2000.0,
            self.down_wall_dist / 2000.0,
            self.left_wall_dist / 2000.0,
            self.right_wall_dist / 2000.0,
            self.health / params.agent_max_health,
        ];
        for other in &self.other_states {
            out.extend_from_slice(&[
                other.rel_pos.x / 2000.0,
                other.rel_pos.y / 2000.0,
                other.linvel.x / 2000.0,
                other.linvel.y / 2000.0,
                other.direction.x,
                other.direction.y,
                if other.firing { 1.0 } else { 0.0 },
            ]);
        }
        out
    }
}

#[derive(Debug, Clone, Default)]
pub struct FfaAction {
    lin_force: Vec2,
    ang_force: f32,
    shoot: f32,
    metadata: PpoMetadata,
}

pub const ACTION_LEN: usize = 4;

impl Action<Ffa> for FfaAction {
    type Metadata = PpoMetadata;

    fn from_slice(action: &[f32], metadata: Self::Metadata, _params: &FfaParams) -> Self {
        Self {
            lin_force: Vec2::new(action[0], action[1]).clamp(Vec2::splat(-1.0), Vec2::splat(1.0)),
            ang_force: action[2].clamp(-1.0, 1.0),
            shoot: action[3].clamp(-1.0, 1.0),
            metadata,
        }
    }

    fn as_vec(&self, _params: &FfaParams) -> Vec<f32> {
        vec![
            self.lin_force.x,
            self.lin_force.y,
            self.ang_force,
            self.shoot,
        ]
    }

    fn metadata(&self) -> Self::Metadata {
        self.metadata.clone()
    }
}

#[derive(Component)]
pub struct ShootyLine;

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
pub struct FfaAgent;

#[derive(Component)]
pub struct Kills(pub usize);

#[derive(Component)]
pub struct Deaths(pub usize);

#[derive(Component)]
pub struct BrainId(pub usize);

#[derive(Component)]
pub struct Name(pub String);

#[derive(Bundle)]
pub struct FfaAgentBundle {
    pub rb: RigidBody,
    pub col: Collider,
    pub rest: Restitution,
    pub friction: Friction,
    pub gravity: GravityScale,
    pub velocity: Velocity,
    pub damping: Damping,
    pub force: ExternalForce,
    pub impulse: ExternalImpulse,
    pub mesh: MaterialMesh2dBundle<ColorMaterial>,
    pub health: Health,
    pub kills: Kills,
    pub deaths: Deaths,
    pub brain_id: BrainId,
    pub name: Name,
    pub marker: FfaAgent,
}
impl FfaAgentBundle {
    pub fn new(
        pos: Vec3,
        color: Option<Color>,
        name: String,
        brain_id: usize,
        meshes: &mut ResMut<Assets<Mesh>>,
        materials: &mut ResMut<Assets<ColorMaterial>>,
        params: &FfaParams,
    ) -> Self {
        Self {
            marker: FfaAgent,
            rb: RigidBody::Dynamic,
            col: Collider::ball(params.agent_radius),
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
                mesh: meshes
                    .add(shape::Circle::new(params.agent_radius).into())
                    .into(),
                transform: Transform::from_translation(pos),
                ..Default::default()
            },
            health: Health(params.agent_max_health),
            name: Name(name),
            kills: Kills(0),
            deaths: Deaths(0),
            brain_id: BrainId(brain_id),
        }
    }
}

#[derive(Resource)]
pub struct Ffa {
    pub params: FfaParams,
    pub brains: BrainBank<Ffa, AgentThinker>,
    pub rbs: BTreeMap<Entity, PpoBuffer<Ffa>>,
    pub observations: BTreeMap<Entity, FrameStack<FfaObs>>,
    pub actions: BTreeMap<Entity, FfaAction>,
    pub rewards: BTreeMap<Entity, f32>,
    pub terminals: BTreeMap<Entity, bool>,
}

impl Env for Ffa {
    type Params = FfaParams;
    type Observation = FfaObs;
    type Action = FfaAction;

    fn init() -> Self {
        Self {
            params: Default::default(),
            brains: Default::default(),
            rbs: Default::default(),
            observations: Default::default(),
            actions: Default::default(),
            rewards: Default::default(),
            terminals: Default::default(),
        }
    }

    fn setup_system() -> SystemConfigs {
        setup.chain()
    }

    fn observation_system() -> SystemConfigs {
        get_observation.chain()
    }

    fn action_system() -> SystemConfigs {
        get_action.chain()
    }

    fn reward_system() -> SystemConfigs {
        get_reward.chain()
    }

    fn terminal_system() -> SystemConfigs {
        get_terminal.chain()
    }

    fn update_system() -> SystemConfigs {
        update.chain()
    }

    fn learn_system() -> SystemConfigs {
        learn.chain()
    }
}

fn setup(
    mut env: ResMut<Ffa>,
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    timestamp: Res<Timestamp>,
) {
    // bottom wall
    commands
        .spawn(Collider::cuboid(500.0, 10.0))
        .insert(SpriteBundle {
            sprite: Sprite {
                color: Color::BEIGE,
                custom_size: Some(Vec2::new(1000.0, 20.0)),
                ..default()
            },
            ..default()
        })
        .insert(Wall)
        .insert(ActiveEvents::all())
        .insert(TransformBundle::from(Transform::from_xyz(0.0, -250.0, 0.0)));

    // top wall
    commands
        .spawn(Collider::cuboid(500.0, 10.0))
        .insert(SpriteBundle {
            sprite: Sprite {
                color: Color::BEIGE,
                custom_size: Some(Vec2::new(1000.0, 20.0)),
                ..default()
            },
            ..default()
        })
        .insert(Wall)
        .insert(ActiveEvents::all())
        .insert(TransformBundle::from(Transform::from_xyz(0.0, 250.0, 0.0)));

    // left wall
    commands
        .spawn(Collider::cuboid(10.0, 300.0))
        .insert(SpriteBundle {
            sprite: Sprite {
                color: Color::BLUE,
                custom_size: Some(Vec2::new(20.0, 600.0)),
                ..default()
            },
            ..default()
        })
        .insert(Wall)
        .insert(ActiveEvents::all())
        .insert(TransformBundle::from(Transform::from_xyz(-500.0, 0.0, 0.0)));

    // right wall
    commands
        .spawn(Collider::cuboid(10.0, 300.0))
        .insert(SpriteBundle {
            sprite: Sprite {
                color: Color::RED,
                custom_size: Some(Vec2::new(20.0, 600.0)),
                ..default()
            },
            ..default()
        })
        .insert(Wall)
        .insert(ActiveEvents::all())
        .insert(TransformBundle::from(Transform::from_xyz(500.0, 0.0, 0.0)));

    // right-middle wall
    commands
        .spawn(Collider::cuboid(100.0, 10.0))
        .insert(SpriteBundle {
            sprite: Sprite {
                color: Color::RED,
                custom_size: Some(Vec2::new(200.0, 20.0)),
                ..default()
            },
            ..default()
        })
        .insert(Wall)
        .insert(ActiveEvents::all())
        .insert(TransformBundle::from(Transform::from_xyz(400.0, 0.0, 0.0)));

    // left-middle wall
    commands
        .spawn(Collider::cuboid(100.0, 10.0))
        .insert(SpriteBundle {
            sprite: Sprite {
                color: Color::BLUE,
                custom_size: Some(Vec2::new(200.0, 20.0)),
                ..default()
            },
            ..default()
        })
        .insert(Wall)
        .insert(ActiveEvents::all())
        .insert(TransformBundle::from(Transform::from_xyz(-400.0, 0.0, 0.0)));

    // top-middle wall
    commands
        .spawn(Collider::cuboid(10.0, 100.0))
        .insert(SpriteBundle {
            sprite: Sprite {
                color: Color::BEIGE,
                custom_size: Some(Vec2::new(20.0, 200.0)),
                ..default()
            },
            ..default()
        })
        .insert(Wall)
        .insert(ActiveEvents::all())
        .insert(TransformBundle::from(Transform::from_xyz(0.0, 200.0, 0.0)));

    // bottom-middle wall
    commands
        .spawn(Collider::cuboid(10.0, 100.0))
        .insert(SpriteBundle {
            sprite: Sprite {
                color: Color::BEIGE,
                custom_size: Some(Vec2::new(20.0, 200.0)),
                ..default()
            },
            ..default()
        })
        .insert(Wall)
        .insert(ActiveEvents::all())
        .insert(TransformBundle::from(Transform::from_xyz(0.0, -200.0, 0.0)));

    // bottom-left corner wall
    commands
        .spawn(Collider::cuboid(10.0, 120.0))
        .insert(SpriteBundle {
            sprite: Sprite {
                color: Color::BLUE,
                custom_size: Some(Vec2::new(20.0, 240.0)),
                ..default()
            },
            ..default()
        })
        .insert(Wall)
        .insert(ActiveEvents::all())
        .insert(TransformBundle::from(
            Transform::from_rotation(Quat::from_axis_angle(Vec3::Z, 45.0 / 180.0 * PI))
                .with_translation(Vec3::new(-400.0, -200.0, 0.0)),
        ));

    // top-right corner wall
    commands
        .spawn(Collider::cuboid(10.0, 120.0))
        .insert(SpriteBundle {
            sprite: Sprite {
                color: Color::RED,
                custom_size: Some(Vec2::new(20.0, 240.0)),
                ..default()
            },
            ..default()
        })
        .insert(Wall)
        .insert(ActiveEvents::all())
        .insert(TransformBundle::from(
            Transform::from_rotation(Quat::from_axis_angle(Vec3::Z, 45.0 / 180.0 * PI))
                .with_translation(Vec3::new(400.0, 200.0, 0.0)),
        ));

    // top-left corner wall
    commands
        .spawn(Collider::cuboid(10.0, 120.0))
        .insert(SpriteBundle {
            sprite: Sprite {
                color: Color::BLUE,
                custom_size: Some(Vec2::new(20.0, 240.0)),
                ..default()
            },
            ..default()
        })
        .insert(Wall)
        .insert(ActiveEvents::all())
        .insert(TransformBundle::from(
            Transform::from_rotation(Quat::from_axis_angle(Vec3::Z, -45.0 / 180.0 * PI))
                .with_translation(Vec3::new(-400.0, 200.0, 0.0)),
        ));

    // bottom-right corner wall
    commands
        .spawn(Collider::cuboid(10.0, 120.0))
        .insert(SpriteBundle {
            sprite: Sprite {
                color: Color::RED,
                custom_size: Some(Vec2::new(20.0, 240.0)),
                ..default()
            },
            ..default()
        })
        .insert(Wall)
        .insert(ActiveEvents::all())
        .insert(TransformBundle::from(
            Transform::from_rotation(Quat::from_axis_angle(Vec3::Z, -45.0 / 180.0 * PI))
                .with_translation(Vec3::new(400.0, -200.0, 0.0)),
        ));

    let mut taken_names = vec![];
    let obs_len = OTHER_STATE_LEN * (env.params.num_agents - 1) + BASE_STATE_LEN;

    for _ in 0..env.params.num_agents {
        let ts = timestamp.clone();
        let mut name = crate::names::random_name();
        while taken_names.contains(&name) {
            name = crate::names::random_name();
        }
        taken_names.push(name.clone());
        let brain_name = name.clone();
        let thinker = PpoThinker::new(
            obs_len,
            env.params.agent_hidden_dim,
            ACTION_LEN,
            env.params.agent_training_epochs,
            env.params.agent_training_batch_size,
            env.params.agent_entropy_beta,
            env.params.agent_actor_lr,
            env.params.agent_critic_lr,
        );
        let brain_id = env
            .brains
            .spawn(|rx| Brain::new(thinker, brain_name, ts, rx));
        let agent_pos = Vec3::new(
            (rand::random::<f32>() - 0.5) * 500.0,
            (rand::random::<f32>() - 0.5) * 500.0,
            0.0,
        );
        let color = Color::rgb(rand::random(), rand::random(), rand::random());
        let agent = FfaAgentBundle::new(
            agent_pos,
            Some(color),
            name,
            brain_id,
            &mut meshes,
            &mut materials,
            &env.params,
        );
        let id = commands
            .spawn(agent)
            .insert(ActiveEvents::all())
            .with_children(|parent| {
                parent
                    .spawn(MaterialMesh2dBundle {
                        material: materials.add(ColorMaterial::from(Color::WHITE)),
                        mesh: meshes
                            .add(Mesh::from(shape::Box::new(3.0, 1.0, 0.0)))
                            .into(),
                        transform: Transform::from_translation(Vec3::new(0.0, 0.0, 0.0)),
                        ..Default::default()
                    })
                    .insert(ShootyLine);

                parent.spawn(MaterialMesh2dBundle {
                    mesh: meshes.add(Mesh::from(shape::Circle::new(3.0))).into(),
                    material: materials.add(ColorMaterial::from(Color::BLACK)),
                    transform: Transform::from_translation(Vec3::new(
                        -5.0,
                        env.params.agent_radius - 5.0,
                        0.1,
                    )),
                    ..Default::default()
                });
                parent.spawn(MaterialMesh2dBundle {
                    mesh: meshes.add(Mesh::from(shape::Circle::new(3.0))).into(),
                    material: materials.add(ColorMaterial::from(Color::BLACK)),
                    transform: Transform::from_translation(Vec3::new(
                        5.0,
                        env.params.agent_radius - 5.0,
                        0.1,
                    )),
                    ..Default::default()
                });
            })
            .id();
        commands.spawn((
            Text2dBundle {
                text: Text::from_section(
                    "",
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
                transform: Transform::from_translation(
                    agent_pos + Vec3::new(0.0, -env.params.agent_radius, 2.0),
                ),
                ..Default::default()
            },
        ));
        let params = env.params;
        env.observations
            .insert(id, FfaObs::new_frame_stack(&params));
        env.rbs.insert(id, PpoBuffer::default());
        env.brains.assign_entity(brain_id, id);
    }
}

fn get_observation(
    mut env: ResMut<Ffa>,
    agents: Query<Entity, With<FfaAgent>>,
    agent_velocity: Query<&Velocity, With<FfaAgent>>,
    agent_transform: Query<&Transform, With<FfaAgent>>,
    agent_health: Query<&Health, With<FfaAgent>>,
    brain_ids: Query<&BrainId, With<FfaAgent>>,
    cx: Res<RapierContext>,
) {
    for agent_ent in agents.iter() {
        let my_t = agent_transform.get(agent_ent).unwrap();
        let my_v = agent_velocity.get(agent_ent).unwrap();
        let my_h = agent_health.get(agent_ent).unwrap();
        let mut my_state = FfaObs {
            pos: my_t.translation.xy(),
            linvel: my_v.linvel,
            direction: my_t.local_y().xy(),
            health: my_h.0,
            other_states: vec![OtherState::default(); env.params.num_agents - 1],
            down_wall_dist: 0.0,
            up_wall_dist: 0.0,
            left_wall_dist: 0.0,
            right_wall_dist: 0.0,
        };

        let filter = QueryFilter::only_fixed();
        if let Some((_, toi)) = cx.cast_ray(my_t.translation.xy(), Vec2::Y, 2000.0, true, filter) {
            my_state.up_wall_dist = toi;
        }
        if let Some((_, toi)) =
            cx.cast_ray(my_t.translation.xy(), Vec2::NEG_Y, 2000.0, true, filter)
        {
            my_state.down_wall_dist = toi;
        }
        if let Some((_, toi)) = cx.cast_ray(my_t.translation.xy(), Vec2::X, 2000.0, true, filter) {
            my_state.right_wall_dist = toi;
        }
        if let Some((_, toi)) =
            cx.cast_ray(my_t.translation.xy(), Vec2::NEG_X, 2000.0, true, filter)
        {
            my_state.left_wall_dist = toi;
        }

        for (i, other_ent) in agents
            .iter()
            .filter(|a| *a != agent_ent)
            .sorted_by_key(|a| {
                let other_t = agent_transform.get(*a).unwrap();
                other_t.translation.distance(my_t.translation) as i64
            })
            .enumerate()
        {
            let other_id = brain_ids.get(other_ent).unwrap();
            let other_t = agent_transform.get(other_ent).unwrap();
            let other_v = agent_velocity.get(other_ent).unwrap();
            let status = env.brains.get_status(other_id.0);
            let other_state = OtherState {
                rel_pos: my_t.translation.xy() - other_t.translation.xy(),
                linvel: other_v.linvel,
                direction: other_t.local_y().xy(),
                firing: status
                    .unwrap_or_default()
                    .last_action
                    .unwrap_or_default()
                    .shoot
                    > 0.0,
            };
            my_state.other_states[i] = other_state;
        }

        let max_len = env.params.agent_frame_stack_len;
        env.observations
            .get_mut(&agent_ent)
            .unwrap()
            .push(my_state, Some(max_len));
    }
}

fn get_action(
    mut env: ResMut<Ffa>,
    frame_count: Res<FrameCount>,
    agents: Query<Entity, With<FfaAgent>>,
    brain_ids: Query<&BrainId, With<FfaAgent>>,
) {
    if frame_count.0 as usize % env.params.agent_frame_stack_len == 0 {
        for agent_ent in agents.iter() {
            let brain_id = brain_ids.get(agent_ent).unwrap().0;
            env.brains.send_obs(
                brain_id,
                env.observations[&agent_ent].clone(),
                frame_count.0 as usize,
                env.params,
            );
            let status = env.brains.get_status(brain_id);
            if let Some(status) = status {
                status
                    .last_action
                    .and_then(|action| env.actions.insert(agent_ent, action));
            }
        }
    }
}

fn get_reward(
    mut env: ResMut<Ffa>,
    _commands: Commands,
    cx: Res<RapierContext>,
    agents: Query<Entity, With<FfaAgent>>,
    agent_transform: Query<&Transform, With<FfaAgent>>,
    childs: Query<&Children, With<FfaAgent>>,
    mut health: Query<&mut Health, With<FfaAgent>>,
    mut kills: Query<&mut Kills, With<FfaAgent>>,
    mut force: Query<&mut ExternalForce, With<FfaAgent>>,
    mut line_vis: Query<&mut Visibility, With<ShootyLine>>,
    mut line_transform: Query<&mut Transform, (With<ShootyLine>, Without<FfaAgent>)>,
    names: Query<&Name, With<FfaAgent>>,
    mut log: ResMut<LogText>,
) {
    for agent_ent in agents.iter() {
        env.rewards.insert(agent_ent, 0.0);
        let my_health = health.get(agent_ent).unwrap();
        let my_t = agent_transform.get(agent_ent).unwrap();
        if let Some(action) = env.actions.get(&agent_ent).cloned() {
            if action.shoot > 0.0 && my_health.0 > 0.0 {
                let (ray_dir, ray_pos) = {
                    let ray_dir = my_t.local_y().xy();
                    let ray_pos = my_t.translation.xy() + ray_dir * (env.params.agent_radius + 2.0);

                    for child in childs.get(agent_ent).unwrap().iter() {
                        if let Ok(mut vis) = line_vis.get_mut(*child) {
                            *vis = Visibility::Visible;
                        }
                    }
                    (ray_dir, ray_pos)
                };

                let filter = QueryFilter::default().exclude_collider(agent_ent);

                if let Some((hit_entity, toi)) = cx.cast_ray(
                    ray_pos,
                    ray_dir,
                    env.params.agent_shoot_distance,
                    false,
                    filter,
                ) {
                    for child in childs.get(agent_ent).unwrap().iter() {
                        if let Ok(mut line) = line_transform.get_mut(*child) {
                            line.scale = Vec3::new(1.0, toi, 1.0);
                            line.translation = Vec3::new(0.0, toi / 2.0, 0.0);
                        }
                    }

                    if let Ok(mut health) = health.get_component_mut::<Health>(hit_entity) {
                        if health.0 > 0.0 {
                            health.0 -= 5.0;
                            *env.rewards.get_mut(&agent_ent).unwrap() += 1.0;
                            *env.rewards.get_mut(&hit_entity).unwrap() -= 1.0;
                            if health.0 <= 0.0 {
                                *env.rewards.get_mut(&agent_ent).unwrap() += 100.0;
                                *env.rewards.get_mut(&hit_entity).unwrap() -= 100.0;
                                kills.get_mut(agent_ent).unwrap().0 += 1;
                                let msg = format!(
                                    "{} killed {}! Nice!",
                                    &names.get(agent_ent).unwrap().0,
                                    &names.get(hit_entity).unwrap().0
                                );
                                log.push(msg);
                            }
                        }
                    }
                } else {
                    for child in childs.get(agent_ent).unwrap().iter() {
                        if let Ok(mut line) = line_transform.get_mut(*child) {
                            line.scale = Vec3::new(1.0, env.params.agent_shoot_distance, 1.0);
                            line.translation =
                                Vec3::new(0.0, env.params.agent_shoot_distance / 2.0, 0.0);
                        }
                    }
                    *env.rewards.get_mut(&agent_ent).unwrap() -= 4.0;
                }
            } else {
                for child in childs.get(agent_ent).unwrap().iter() {
                    if let Ok(mut vis) = line_vis.get_mut(*child) {
                        *vis = Visibility::Hidden;
                    }
                }
            }

            let mut my_force = force.get_mut(agent_ent).unwrap();
            my_force.force = action.lin_force * env.params.agent_lin_move_force;
            my_force.torque = action.ang_force * env.params.agent_ang_move_force;
        }
    }
}

fn get_terminal(
    mut env: ResMut<Ffa>,
    agents: Query<Entity, With<FfaAgent>>,
    health: Query<&Health, With<FfaAgent>>,
) {
    for agent_ent in agents.iter() {
        if health.get(agent_ent).unwrap().0 <= 0.0 {
            env.terminals.insert(agent_ent, true);
        } else {
            env.terminals.insert(agent_ent, false);
        }
    }
}

fn update(
    mut commands: Commands,
    mut env: ResMut<Ffa>,
    mut name_text_t: Query<
        (Entity, &mut Transform, &mut Text, &mut NameText),
        (With<NameText>, Without<FfaAgent>),
    >,
    mut health_bar_t: Query<
        (Entity, &mut Transform, &HealthBar),
        (Without<NameText>, Without<FfaAgent>),
    >,
    agents: Query<Entity, With<FfaAgent>>,
    brain_ids: Query<&BrainId, With<FfaAgent>>,
    names: Query<&Name, With<FfaAgent>>,
    kills: Query<&Kills, With<FfaAgent>>,
    mut deaths: Query<&mut Deaths, With<FfaAgent>>,
    mut health: Query<&mut Health, With<FfaAgent>>,
    mut agent_transform: Query<&mut Transform, With<FfaAgent>>,
) {
    for (t_ent, mut t, mut text, text_comp) in name_text_t.iter_mut() {
        if let Ok(agent) = agent_transform.get(text_comp.entity_following) {
            t.translation = agent.translation + Vec3::new(0.0, env.params.agent_radius + 20.0, 2.0);
            text.sections[0].value = format!(
                "{} {} {}-{}",
                brain_ids.get(text_comp.entity_following).unwrap().0,
                names.get(text_comp.entity_following).unwrap().0,
                kills.get(text_comp.entity_following).unwrap().0,
                deaths.get(text_comp.entity_following).unwrap().0,
            );
        } else {
            commands.entity(t_ent).despawn();
        }
    }
    for (t_ent, mut t, hb) in health_bar_t.iter_mut() {
        if let Ok(agent) = agent_transform.get(hb.entity_following) {
            t.translation = agent.translation + Vec3::new(0.0, env.params.agent_radius + 5.0, 2.0);
            let health = health.get(hb.entity_following).unwrap();
            t.scale = Vec3::new(health.0 / env.params.agent_max_health * 100.0, 1.0, 1.0);
        } else {
            commands.entity(t_ent).despawn();
        }
    }

    for agent_ent in agents.iter() {
        let _status = env.brains.get_status(brain_ids.get(agent_ent).unwrap().0);
        let (action, reward, terminal) = (
            env.actions.get(&agent_ent).cloned(),
            env.rewards.get(&agent_ent).copied(),
            env.terminals.get(&agent_ent).copied(),
        );
        let obs = env.observations[&agent_ent].clone();
        let max_len = env.params.agent_rb_max_len;
        // if let Some(status) = status {
        if let Some(rb) = env.rbs.get_mut(&agent_ent) {
            // if fresh {
            if let (Some(action), Some(reward), Some(terminal)) = (action, reward, terminal) {
                rb.remember_sart(
                    Sart {
                        obs,
                        action: action.to_owned(),
                        reward,
                        terminal,
                    },
                    Some(max_len),
                );
            }
        }
        // }
    }

    for agent_ent in agents.iter() {
        let mut my_health = health.get_mut(agent_ent).unwrap();
        if my_health.0 <= 0.0 {
            if let Some(rb) = env.rbs.get_mut(&agent_ent) {
                rb.finish_trajectory();
            }

            // let mut ent = commands.entity(agent);
            deaths.get_mut(agent_ent).unwrap().0 += 1;
            my_health.0 = env.params.agent_max_health;
            let agent_pos = Vec3::new(
                (rand::random::<f32>() - 0.5) * 500.0,
                (rand::random::<f32>() - 0.5) * 500.0,
                0.0,
            );
            agent_transform.get_mut(agent_ent).unwrap().translation = agent_pos;
        }
    }
}

fn learn(
    mut env: ResMut<Ffa>,
    frame_count: Res<FrameCount>,
    agents: Query<Entity, With<FfaAgent>>,
    brain_ids: Query<&BrainId, With<FfaAgent>>,
    deaths: Query<&Deaths, With<FfaAgent>>,
    names: Query<&Name, With<FfaAgent>>,
    mut log: ResMut<LogText>,
) {
    if frame_count.0 > 1 && frame_count.0 as usize % env.params.agent_update_interval == 0 {
        for agent_ent in agents.iter() {
            if deaths.get(agent_ent).unwrap().0 > 0 {
                let rb = env.rbs[&agent_ent].clone();
                let id = brain_ids.get(agent_ent).unwrap().0;
                let name = &names.get(agent_ent).unwrap().0;
                log.push(format!("Training {id} {name}..."));

                AsyncComputeTaskPool::get().scope(|scope| {
                    scope.spawn(async {
                        env.brains
                            .learn(id, frame_count.0 as usize, rb, env.params)
                            .await;
                    });
                });
                let status = env.brains.get_status(id);
                if let Some(status) = status.and_then(|s| s.status) {
                    log.push(format!(
                        "{} {} Policy Loss: {}",
                        id, name, status.recent_policy_loss
                    ));
                    log.push(format!(
                        "{} {} Policy Entropy: {}",
                        id, name, status.recent_entropy_loss
                    ));
                    log.push(format!(
                        "{} {} Policy Clip Ratio: {}",
                        id, name, status.recent_nclamp
                    ));
                    log.push(format!(
                        "{} {} Value Loss: {}",
                        id, name, status.recent_value_loss
                    ));
                }
            }
        }
    }
}
