use bevy_egui::{
    egui::{
        self,
        plot::{Bar, BarChart, Line},
        Color32, Layout, Stroke,
    },
    EguiContexts,
};

use itertools::Itertools;

use bevy::{
    core::FrameCount, ecs::schedule::SystemConfigs, math::Vec3Swizzles, prelude::*,
    sprite::MaterialMesh2dBundle,
};
use bevy_rapier2d::prelude::*;
use bevy_tasks::AsyncComputeTaskPool;

use crate::{
    brains::{
        replay_buffer::{PpoBuffer, PpoMetadata, Sart},
        thinkers::ppo::PpoThinker,
        Brain, BrainBank,
    },
    ui::LogText,
    FrameStack, Timestamp,
};

use super::{
    maps::Map,
    modules::{
        map_interaction::MapInteractionProperties, Behavior, CombatBehaviors, CombatProperties,
        PhysicalBehaviors, PhysicalProperties, Property,
    },
    Action, DefaultFrameStack, Env, Observation, Params,
};

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
    pub distance_scaling: f32,
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
            agent_rb_max_len: 50_000,
            agent_frame_stack_len: 5,
            agent_radius: 20.0,
            agent_lin_move_force: 600.0,
            agent_ang_move_force: 1.0,
            agent_max_health: 100.0,
            agent_shoot_distance: 500.0,
            distance_scaling: 1.0 / 2000.0,
        }
    }
}

impl Params for FfaParams {
    fn agent_radius(&self) -> f32 {
        self.agent_radius
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct OtherState {
    pub phys: PhysicalProperties,
    pub combat: CombatProperties,
    pub map_inter: MapInteractionProperties,
    pub firing: bool,
}

impl Observation<Ffa> for OtherState {
    fn as_slice(&self, _params: &FfaParams) -> Box<[f32]> {
        let mut out = self.phys.as_slice().to_vec();
        out.extend_from_slice(&self.combat.as_slice());
        out.extend_from_slice(&self.map_inter.as_slice());
        out.push(if self.firing { 1.0 } else { 0.0 });
        out.into_boxed_slice()
    }
}

pub const OTHER_STATE_LEN: usize = 6 + 1 + 4 + 1;

#[derive(Clone, Debug, Component)]
pub struct FfaObs {
    pub phys: PhysicalProperties,
    pub combat: CombatProperties,
    pub map_inter: MapInteractionProperties,
    pub other_states: Vec<OtherState>,
}

pub const BASE_STATE_LEN: usize = 6 + 1 + 4;

impl Observation<Ffa> for FfaObs {
    fn as_slice(&self, params: &FfaParams) -> Box<[f32]> {
        let mut out = self.phys.as_slice().to_vec();
        out.extend_from_slice(&self.combat.as_slice());
        out.extend_from_slice(&self.map_inter.as_slice());
        for other in &self.other_states {
            out.extend_from_slice(&other.as_slice(params));
        }
        out.into_boxed_slice()
    }
}

impl DefaultFrameStack<Ffa> for FfaObs {
    fn default_frame_stack(params: &FfaParams) -> FrameStack<Self> {
        FrameStack(
            vec![
                Self {
                    phys: Default::default(),
                    combat: Default::default(),
                    map_inter: Default::default(),
                    other_states: vec![OtherState::default(); params.num_agents - 1],
                };
                params.agent_frame_stack_len
            ]
            .into(),
        )
    }
}

#[derive(Debug, Clone, Default, Component)]
pub struct FfaAction {
    pub phys: PhysicalBehaviors,
    pub combat: CombatBehaviors,
    pub metadata: PpoMetadata,
}

pub const ACTION_LEN: usize = 4;

impl Action<Ffa> for FfaAction {
    type Metadata = PpoMetadata;

    fn from_slice(action: &[f32], metadata: Self::Metadata, _params: &FfaParams) -> Self {
        Self {
            phys: PhysicalBehaviors {
                force: Vec2::new(action[0], action[1]),
                torque: action[2],
            },
            combat: CombatBehaviors {
                shoot: action[3] > 0.0,
            },
            metadata,
        }
    }

    fn as_slice(&self, _params: &FfaParams) -> Box<[f32]> {
        self.phys
            .as_slice()
            .iter()
            .chain(self.combat.as_slice().iter())
            .copied()
            .collect_vec()
            .into_boxed_slice()
    }

    fn metadata(&self) -> Self::Metadata {
        self.metadata.clone()
    }
}

#[derive(Component)]
pub struct ShootyLine;

#[derive(Bundle)]
pub struct ShootyLineBundle {
    pub mesh: MaterialMesh2dBundle<ColorMaterial>,
    pub shooty_line: ShootyLine,
}

impl ShootyLineBundle {
    pub fn new(mut materials: Mut<Assets<ColorMaterial>>, mut meshes: Mut<Assets<Mesh>>) -> Self {
        Self {
            mesh: MaterialMesh2dBundle {
                material: materials.add(ColorMaterial::from(Color::WHITE)),
                mesh: meshes
                    .add(Mesh::from(shape::Box::new(3.0, 1.0, 0.0)))
                    .into(),
                transform: Transform::from_translation(Vec3::new(0.0, 0.0, 0.0)),
                ..Default::default()
            },
            shooty_line: ShootyLine,
        }
    }
}

pub struct Eyeballs;

impl Eyeballs {
    pub fn spawn(
        parent: &mut ChildBuilder,
        mut meshes: Mut<Assets<Mesh>>,
        mut materials: Mut<Assets<ColorMaterial>>,
        agent_radius: f32,
    ) {
        parent.spawn(MaterialMesh2dBundle {
            mesh: meshes.add(Mesh::from(shape::Circle::new(3.0))).into(),
            material: materials.add(ColorMaterial::from(Color::BLACK)),
            transform: Transform::from_translation(Vec3::new(-5.0, agent_radius - 5.0, 0.1)),
            ..Default::default()
        });
        parent.spawn(MaterialMesh2dBundle {
            mesh: meshes.add(Mesh::from(shape::Circle::new(3.0))).into(),
            material: materials.add(ColorMaterial::from(Color::BLACK)),
            transform: Transform::from_translation(Vec3::new(5.0, agent_radius - 5.0, 0.1)),
            ..Default::default()
        });
    }
}

#[derive(Component)]
pub struct NameText {
    pub entity_following: Entity,
}

#[derive(Bundle)]
pub struct NameTextBundle {
    pub text: Text2dBundle,
    pub name_text: NameText,
}

impl NameTextBundle {
    pub fn new(asset_server: &AssetServer, entity_following: Entity) -> Self {
        Self {
            text: Text2dBundle {
                text: Text::from_section(
                    "",
                    TextStyle {
                        font: asset_server.load("fonts/FiraSans-Bold.ttf"),
                        font_size: 20.0,
                        color: Color::WHITE,
                    },
                ),
                transform: Transform::default(),
                ..Default::default()
            },
            name_text: NameText { entity_following },
        }
    }
}

#[derive(Component)]
pub struct Health(pub f32);

#[derive(Component)]
pub struct HealthBar {
    pub entity_following: Entity,
}

#[derive(Bundle)]
pub struct HealthBarBundle {
    pub mesh: MaterialMesh2dBundle<ColorMaterial>,
    pub health_bar: HealthBar,
}

impl HealthBarBundle {
    pub fn new(
        mut meshes: Mut<Assets<Mesh>>,
        mut materials: Mut<Assets<ColorMaterial>>,
        entity_following: Entity,
    ) -> Self {
        Self {
            mesh: MaterialMesh2dBundle {
                mesh: meshes.add(shape::Box::new(1.0, 6.0, 0.0).into()).into(),
                material: materials.add(ColorMaterial::from(Color::RED)),
                ..Default::default()
            },
            health_bar: HealthBar { entity_following },
        }
    }
}

#[derive(Component)]
pub struct Agent;

#[derive(Component)]
pub struct Reward(pub f32);

#[derive(Component)]
pub struct Terminal(pub bool);

#[derive(Component)]
pub struct Kills(pub usize);

#[derive(Component)]
pub struct Deaths(pub usize);

#[derive(Component)]
pub struct BrainId(pub usize);

#[derive(Component)]
pub struct Name(pub String);

#[derive(Bundle)]
pub struct AgentBundle<E: Env> {
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
    pub obs: FrameStack<E::Observation>,
    pub action: E::Action,
    pub replay_buffer: PpoBuffer<E>,
    pub reward: Reward,
    pub terminal: Terminal,
    marker: Agent,
}
impl<E: Env> AgentBundle<E> {
    pub fn new(
        pos: Vec3,
        color: Option<Color>,
        name: String,
        brain_id: usize,
        meshes: &mut ResMut<Assets<Mesh>>,
        materials: &mut ResMut<Assets<ColorMaterial>>,
        params: &E::Params,
    ) -> Self {
        Self {
            obs: E::Observation::default_frame_stack(params),
            action: E::Action::default(),
            replay_buffer: PpoBuffer::default(),
            reward: Reward(0.0),
            terminal: Terminal(false),
            marker: Agent,
            rb: RigidBody::Dynamic,
            col: Collider::ball(params.agent_radius()),
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
                    .add(shape::Circle::new(params.agent_radius()).into())
                    .into(),
                transform: Transform::from_translation(pos),
                ..Default::default()
            },
            health: Health(0.0),
            name: Name(name),
            kills: Kills(0),
            deaths: Deaths(0),
            brain_id: BrainId(brain_id),
        }
    }
}

#[derive(Resource, Default)]
pub struct Ffa;

impl Env for Ffa {
    type Params = FfaParams;
    type Observation = FfaObs;
    type Action = FfaAction;

    fn init() -> Self {
        Self
    }

    fn setup_system<M: Map>() -> SystemConfigs {
        (M::setup_system(), setup).chain()
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
        (update, store_sarts, check_dead).chain()
    }

    fn learn_system() -> SystemConfigs {
        learn.chain()
    }

    fn ui_system() -> SystemConfigs {
        ui.chain()
    }
}

fn setup(
    params: Res<FfaParams>,
    mut brains: ResMut<BrainBank<Ffa, PpoThinker>>,
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    timestamp: Res<Timestamp>,
) {
    let mut taken_names = vec![];
    let obs_len = OTHER_STATE_LEN * (params.num_agents - 1) + BASE_STATE_LEN;

    for _ in 0..params.num_agents {
        let ts = timestamp.clone();
        let mut name = crate::names::random_name();
        while taken_names.contains(&name) {
            name = crate::names::random_name();
        }
        taken_names.push(name.clone());
        let brain_name = name.clone();
        let thinker = PpoThinker::new(
            obs_len,
            params.agent_hidden_dim,
            ACTION_LEN,
            params.agent_training_epochs,
            params.agent_training_batch_size,
            params.agent_entropy_beta,
            params.agent_actor_lr,
            params.agent_critic_lr,
        );
        // let thinker = RandomThinker;
        let brain_id = brains.spawn(|rx| Brain::new(thinker, brain_name, ts, rx));
        let agent_pos = Vec3::new(
            (rand::random::<f32>() - 0.5) * 500.0,
            (rand::random::<f32>() - 0.5) * 500.0,
            0.0,
        );
        let color = Color::rgb(rand::random(), rand::random(), rand::random());
        let mut agent = AgentBundle::<Ffa>::new(
            agent_pos,
            Some(color),
            name,
            brain_id,
            &mut meshes,
            &mut materials,
            &params,
        );
        agent.health = Health(params.agent_max_health);
        let id = commands
            .spawn(agent)
            .insert(ActiveEvents::all())
            .with_children(|parent| {
                parent.spawn(ShootyLineBundle::new(
                    materials.reborrow(),
                    meshes.reborrow(),
                ));
                Eyeballs::spawn(
                    parent,
                    meshes.reborrow(),
                    materials.reborrow(),
                    params.agent_radius,
                );
            })
            .id();
        commands.spawn(NameTextBundle::new(&asset_server, id));
        commands.spawn(HealthBarBundle::new(
            meshes.reborrow(),
            materials.reborrow(),
            id,
        ));

        brains.assign_entity(brain_id, id);
    }
}

fn get_observation(
    params: Res<FfaParams>,
    mut brains: ResMut<BrainBank<Ffa, PpoThinker>>,
    mut observations: Query<&mut FrameStack<FfaObs>, With<Agent>>,
    agents: Query<Entity, With<Agent>>,
    agent_velocity: Query<&Velocity, With<Agent>>,
    agent_transform: Query<&Transform, With<Agent>>,
    agent_health: Query<&Health, With<Agent>>,
    brain_ids: Query<&BrainId, With<Agent>>,
    cx: Res<RapierContext>,
) {
    let phys_scaling = PhysicalProperties {
        position: Vec2::splat(params.distance_scaling),
        direction: Vec2::splat(1.0),
        linvel: Vec2::splat(params.distance_scaling),
    };
    let map_scaling = MapInteractionProperties {
        up_wall_dist: params.distance_scaling,
        down_wall_dist: params.distance_scaling,
        left_wall_dist: params.distance_scaling,
        right_wall_dist: params.distance_scaling,
    };
    for agent_ent in agents.iter() {
        let my_t = agent_transform.get(agent_ent).unwrap();
        let my_v = agent_velocity.get(agent_ent).unwrap();
        let my_h = agent_health.get(agent_ent).unwrap();
        let mut my_state = FfaObs {
            phys: PhysicalProperties {
                position: my_t.translation.xy(),
                direction: my_t.local_y().xy(),
                linvel: my_v.linvel,
            }
            .scaled_by(&phys_scaling),
            combat: CombatProperties { health: my_h.0 }.scaled_by(&CombatProperties {
                health: 1.0 / params.agent_max_health,
            }),
            other_states: vec![],
            map_inter: MapInteractionProperties::new(my_t, &cx).scaled_by(&map_scaling),
        };

        for other_ent in agents
            .iter()
            .filter(|a| *a != agent_ent)
            .sorted_by_key(|a| {
                let other_t = agent_transform.get(*a).unwrap();
                other_t.translation.distance(my_t.translation) as i64
            })
        {
            let other_id = brain_ids.get(other_ent).unwrap();
            let other_t = agent_transform.get(other_ent).unwrap();
            let other_v = agent_velocity.get(other_ent).unwrap();
            let other_h = agent_health.get(other_ent).unwrap();
            let status = brains.get_status(other_id.0);
            let other_state = OtherState {
                phys: PhysicalProperties {
                    position: my_t.translation.xy() - other_t.translation.xy(),
                    direction: other_t.local_y().xy(),
                    linvel: other_v.linvel,
                }
                .scaled_by(&phys_scaling),
                combat: CombatProperties { health: other_h.0 }.scaled_by(&CombatProperties {
                    health: 1.0 / params.agent_max_health,
                }),
                map_inter: MapInteractionProperties::new(other_t, &cx).scaled_by(&map_scaling),

                firing: status
                    .unwrap_or_default()
                    .last_action
                    .unwrap_or_default()
                    .combat
                    .shoot,
            };

            my_state.other_states.push(other_state);
        }

        let max_len = params.agent_frame_stack_len;
        observations
            .get_mut(agent_ent)
            .unwrap()
            .push(my_state, Some(max_len));
    }
}

fn get_action(
    params: Res<FfaParams>,
    mut brains: ResMut<BrainBank<Ffa, PpoThinker>>,
    mut actions: Query<&mut FfaAction, With<Agent>>,
    observations: Query<&FrameStack<FfaObs>, With<Agent>>,
    frame_count: Res<FrameCount>,
    agents: Query<Entity, With<Agent>>,
    brain_ids: Query<&BrainId, With<Agent>>,
) {
    if frame_count.0 as usize % params.agent_frame_stack_len == 0 {
        for agent_ent in agents.iter() {
            let brain_id = brain_ids.get(agent_ent).unwrap().0;
            brains.send_obs(
                brain_id,
                observations.get(agent_ent).unwrap().clone(),
                frame_count.0 as usize,
                *params,
            );
            let status = brains.get_status(brain_id);
            if let Some(status) = status {
                // if let Some(action) = status.last_action {
                *actions.get_mut(agent_ent).unwrap() = status.last_action.unwrap();
                // }
            }
        }
    }
}

fn get_reward(
    params: Res<FfaParams>,
    mut rewards: Query<&mut Reward, With<Agent>>,
    actions: Query<&FfaAction, With<Agent>>,
    cx: Res<RapierContext>,
    agents: Query<Entity, With<Agent>>,
    agent_transform: Query<&Transform, With<Agent>>,
    childs: Query<&Children, With<Agent>>,
    mut health: Query<&mut Health, With<Agent>>,
    mut kills: Query<&mut Kills, With<Agent>>,
    mut force: Query<&mut ExternalForce, With<Agent>>,
    mut line_vis: Query<&mut Visibility, With<ShootyLine>>,
    mut line_transform: Query<&mut Transform, (With<ShootyLine>, Without<Agent>)>,
    names: Query<&Name, With<Agent>>,
    mut log: ResMut<LogText>,
) {
    for agent_ent in agents.iter() {
        rewards.get_mut(agent_ent).unwrap().0 = 0.0;
        let my_health = health.get(agent_ent).unwrap();
        let my_t = agent_transform.get(agent_ent).unwrap();
        if let Ok(action) = actions.get(agent_ent).cloned() {
            if action.combat.shoot && my_health.0 > 0.0 {
                let (ray_dir, ray_pos) = {
                    let ray_dir = my_t.local_y().xy();
                    let ray_pos = my_t.translation.xy() + ray_dir * (params.agent_radius + 2.0);

                    for child in childs.get(agent_ent).unwrap().iter() {
                        if let Ok(mut vis) = line_vis.get_mut(*child) {
                            *vis = Visibility::Visible;
                        }
                    }
                    (ray_dir, ray_pos)
                };

                let filter = QueryFilter::default().exclude_collider(agent_ent);

                if let Some((hit_entity, toi)) =
                    cx.cast_ray(ray_pos, ray_dir, params.agent_shoot_distance, false, filter)
                {
                    for child in childs.get(agent_ent).unwrap().iter() {
                        if let Ok(mut line) = line_transform.get_mut(*child) {
                            line.scale = Vec3::new(1.0, toi, 1.0);
                            line.translation = Vec3::new(0.0, toi / 2.0, 0.0);
                        }
                    }

                    if let Ok(mut health) = health.get_component_mut::<Health>(hit_entity) {
                        if health.0 > 0.0 {
                            health.0 -= 5.0;
                            rewards.get_mut(agent_ent).unwrap().0 += 1.0;
                            rewards.get_mut(hit_entity).unwrap().0 -= 1.0;
                            if health.0 <= 0.0 {
                                rewards.get_mut(agent_ent).unwrap().0 += 100.0;
                                rewards.get_mut(hit_entity).unwrap().0 -= 100.0;
                                kills.get_mut(agent_ent).unwrap().0 += 1;
                                let msg = format!(
                                    "{} killed {}! Nice!",
                                    &names.get(agent_ent).unwrap().0,
                                    &names.get(hit_entity).unwrap().0
                                );
                                log.push(msg);
                            }
                        }
                    } else {
                        // hit a wall
                        rewards.get_mut(agent_ent).unwrap().0 -= 4.0;
                    }
                } else {
                    // hit nothing
                    for child in childs.get(agent_ent).unwrap().iter() {
                        if let Ok(mut line) = line_transform.get_mut(*child) {
                            line.scale = Vec3::new(1.0, params.agent_shoot_distance, 1.0);
                            line.translation =
                                Vec3::new(0.0, params.agent_shoot_distance / 2.0, 0.0);
                        }
                    }
                    rewards.get_mut(agent_ent).unwrap().0 -= 4.0;
                }
            } else {
                for child in childs.get(agent_ent).unwrap().iter() {
                    if let Ok(mut vis) = line_vis.get_mut(*child) {
                        *vis = Visibility::Hidden;
                    }
                }
            }

            let mut my_force = force.get_mut(agent_ent).unwrap();
            my_force.force = action.phys.force * params.agent_lin_move_force;
            my_force.torque = action.phys.torque * params.agent_ang_move_force;
        }
    }
}

fn get_terminal(
    mut terminals: Query<&mut Terminal, With<Agent>>,
    agents: Query<Entity, With<Agent>>,
    health: Query<&Health, With<Agent>>,
) {
    for agent_ent in agents.iter() {
        terminals.get_mut(agent_ent).unwrap().0 = health.get(agent_ent).unwrap().0 <= 0.0;
    }
}

fn store_sarts(
    params: Res<FfaParams>,
    observations: Query<&FrameStack<FfaObs>, With<Agent>>,
    actions: Query<&FfaAction, With<Agent>>,
    rewards: Query<&Reward, With<Agent>>,
    mut rbs: Query<&mut PpoBuffer<Ffa>, With<Agent>>,
    terminals: Query<&Terminal, With<Agent>>,
    agents: Query<Entity, With<Agent>>,
) {
    for agent_ent in agents.iter() {
        let (action, reward, terminal) = (
            actions.get(agent_ent).unwrap().clone(),
            rewards.get(agent_ent).unwrap().0,
            terminals.get(agent_ent).unwrap().0,
        );
        let obs = observations.get(agent_ent).unwrap().clone();
        let max_len = params.agent_rb_max_len;
        rbs.get_mut(agent_ent).unwrap().remember_sart(
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

fn update(
    mut commands: Commands,
    params: Res<FfaParams>,
    mut name_text_t: Query<
        (Entity, &mut Transform, &mut Text, &mut NameText),
        (With<NameText>, Without<Agent>),
    >,
    mut health_bar_t: Query<
        (Entity, &mut Transform, &HealthBar),
        (Without<NameText>, Without<Agent>),
    >,
    brain_ids: Query<&BrainId, With<Agent>>,
    names: Query<&Name, With<Agent>>,
    kills: Query<&Kills, With<Agent>>,
    deaths: Query<&Deaths, With<Agent>>,
    health: Query<&Health, With<Agent>>,
    agent_transform: Query<&Transform, With<Agent>>,
) {
    for (t_ent, mut t, mut text, text_comp) in name_text_t.iter_mut() {
        if let Ok(agent) = agent_transform.get(text_comp.entity_following) {
            t.translation = agent.translation + Vec3::new(0.0, params.agent_radius + 20.0, 2.0);
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
            t.translation = agent.translation + Vec3::new(0.0, params.agent_radius + 5.0, 2.0);
            let health = health.get(hb.entity_following).unwrap();
            t.scale = Vec3::new(health.0 / params.agent_max_health * 100.0, 1.0, 1.0);
        } else {
            commands.entity(t_ent).despawn();
        }
    }
}

fn check_dead(
    params: Res<FfaParams>,
    agents: Query<Entity, With<Agent>>,
    mut health: Query<&mut Health, With<Agent>>,
    mut deaths: Query<&mut Deaths, With<Agent>>,
    mut rbs: Query<&mut PpoBuffer<Ffa>, With<Agent>>,
    mut agent_transform: Query<&mut Transform, With<Agent>>,
) {
    for agent_ent in agents.iter() {
        let mut my_health = health.get_mut(agent_ent).unwrap();
        if my_health.0 <= 0.0 {
            if let Ok(mut rb) = rbs.get_mut(agent_ent) {
                rb.finish_trajectory();
            }

            // let mut ent = commands.entity(agent);
            deaths.get_mut(agent_ent).unwrap().0 += 1;
            my_health.0 = params.agent_max_health;
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
    params: Res<FfaParams>,
    rbs: Query<&PpoBuffer<Ffa>, With<Agent>>,
    mut brains: ResMut<BrainBank<Ffa, PpoThinker>>,
    frame_count: Res<FrameCount>,
    agents: Query<Entity, With<Agent>>,
    brain_ids: Query<&BrainId, With<Agent>>,
    deaths: Query<&Deaths, With<Agent>>,
    names: Query<&Name, With<Agent>>,
    mut log: ResMut<LogText>,
) {
    if frame_count.0 > 1 && frame_count.0 as usize % params.agent_update_interval == 0 {
        for agent_ent in agents.iter() {
            if deaths.get(agent_ent).unwrap().0 > 0 {
                let rb = rbs.get(agent_ent).unwrap().clone();
                let id = brain_ids.get(agent_ent).unwrap().0;
                let name = &names.get(agent_ent).unwrap().0;
                log.push(format!("Training {id} {name}..."));

                AsyncComputeTaskPool::get().scope(|scope| {
                    scope.spawn(async {
                        brains.learn(id, frame_count.0 as usize, rb, *params).await;
                    });
                });
                let status = brains.get_status(id);
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

pub fn ui(
    mut cxs: EguiContexts,
    log: ResMut<LogText>,
    agents: Query<Entity, With<Agent>>,
    kills: Query<&Kills, With<Agent>>,
    deaths: Query<&Deaths, With<Agent>>,
    brain_ids: Query<&BrainId, With<Agent>>,
    names: Query<&Name, With<Agent>>,
    mut brains: ResMut<BrainBank<Ffa, PpoThinker>>,
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
                                let status = brains
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