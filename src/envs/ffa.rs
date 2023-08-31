use std::collections::VecDeque;

use bevy_prng::ChaCha8Rng;
use bevy_rand::{prelude::EntropyComponent, resource::GlobalEntropy};
use burn_tch::TchBackend;
use burn_tensor::Tensor;
use itertools::Itertools;

use bevy::{
    core::FrameCount, ecs::schedule::SystemConfigs, math::Vec3Swizzles, prelude::*,
    sprite::MaterialMesh2dBundle,
};
use bevy_rapier2d::prelude::*;
use rand_distr::{Distribution, Uniform};
use serde::{Deserialize, Serialize};

use crate::{
    brains::{
        replay_buffer::{store_sarts, PpoBuffer, PpoMetadata},
        thinkers::{
            ppo::{PpoParams, PpoStatus, PpoThinker, RmsNormalize},
            Thinker,
        },
        Brain,
    },
    ui::LogText,
    FrameStack, Timestamp,
};

use super::{
    modules::{
        map_interaction::MapInteractionProperties, Behavior, CombatBehaviors, CombatProperties,
        PhysicalBehaviors, PhysicalProperties, Property,
    },
    Action, DefaultFrameStack, Env, Observation, Params,
};

#[derive(Debug, Resource, Clone, Copy, Serialize, Deserialize)]
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
    pub reward_for_kill: f32,
    pub reward_for_death: f32,
    pub reward_for_hit: f32,
    pub reward_for_getting_hit: f32,
    pub reward_for_miss: f32,
}

impl Default for FfaParams {
    fn default() -> Self {
        Self {
            num_agents: 6,
            agent_hidden_dim: 128,
            agent_actor_lr: 0.001,
            agent_critic_lr: 0.01,
            agent_training_epochs: 10,
            agent_training_batch_size: 512,
            agent_entropy_beta: 0.00001,
            agent_update_interval: 1_000,
            agent_rb_max_len: 100_000,
            agent_frame_stack_len: 5,
            agent_radius: 20.0,
            agent_lin_move_force: 600.0,
            agent_ang_move_force: 1.0,
            agent_max_health: 100.0,
            agent_shoot_distance: 1000.0,
            distance_scaling: 1.0 / 2000.0,
            reward_for_kill: 100.0,
            reward_for_death: -50.0,
            reward_for_hit: 1.0,
            reward_for_getting_hit: -0.5,
            reward_for_miss: -2.0,
        }
    }
}

impl Params for FfaParams {
    fn num_agents(&self) -> usize {
        self.num_agents
    }

    fn agent_frame_stack_len(&self) -> usize {
        self.agent_frame_stack_len
    }

    fn agent_radius(&self) -> f32 {
        self.agent_radius
    }

    fn agent_max_health(&self) -> f32 {
        self.agent_max_health
    }
}

impl PpoParams for FfaParams {
    fn actor_lr(&self) -> f64 {
        self.agent_actor_lr
    }

    fn agent_rb_max_len(&self) -> usize {
        self.agent_rb_max_len
    }

    fn critic_lr(&self) -> f64 {
        self.agent_critic_lr
    }

    fn entropy_beta(&self) -> f32 {
        self.agent_entropy_beta
    }

    fn training_batch_size(&self) -> usize {
        self.agent_training_batch_size
    }

    fn training_epochs(&self) -> usize {
        self.agent_training_epochs
    }

    fn agent_update_interval(&self) -> usize {
        self.agent_update_interval
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct OtherState {
    pub phys: PhysicalProperties,
    pub combat: CombatProperties,
    pub map_inter: MapInteractionProperties,
    pub firing: bool,
}

impl Observation for OtherState {
    fn as_slice(&self) -> Box<[f32]> {
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

impl Observation for FfaObs {
    fn as_slice(&self) -> Box<[f32]> {
        let mut out = self.phys.as_slice().to_vec();
        out.extend_from_slice(&self.combat.as_slice());
        out.extend_from_slice(&self.map_inter.as_slice());
        for other in &self.other_states {
            out.extend_from_slice(&other.as_slice());
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
                    other_states: vec![OtherState::default(); params.num_agents() - 1],
                };
                params.agent_frame_stack_len()
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
            combat: CombatBehaviors { shoot: action[3] },
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
pub struct RunningReward {
    pub rewards: VecDeque<f32>,
    pub history: VecDeque<f32>,
    pub max_len: usize,
    pub history_max_len: usize,
}

impl RunningReward {
    pub fn update(&mut self, reward: f32) -> Option<f32> {
        if self.rewards.len() >= self.max_len {
            self.rewards.pop_front();
        }
        self.rewards.push_back(reward);
        if let Some(r) = self.get() {
            if self.history.len() >= self.history_max_len {
                self.history.pop_front();
            }
            self.history.push_back(r);
            Some(r)
        } else {
            None
        }
    }

    pub fn get(&self) -> Option<f32> {
        if self.rewards.len() == self.max_len {
            Some(self.rewards.iter().sum::<f32>() / self.max_len as f32)
        } else {
            None
        }
    }
}

#[derive(Component)]
pub struct Terminal(pub bool);

#[derive(Component)]
pub struct Kills(pub usize);

#[derive(Component)]
pub struct Deaths(pub usize);

#[derive(Component)]
pub struct Name(pub String);

#[derive(Bundle)]
pub struct AgentBundle<E: Env, T: Thinker<E>>
where
    E::Action: Action<E, Metadata = PpoMetadata>,
{
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
    pub name: Name,
    pub brain: Brain<E, T>,
    pub obs: FrameStack<Box<[f32]>>,
    pub obs_norm: RmsNormalize<TchBackend<f32>, 2>,
    pub action: E::Action,
    pub replay_buffer: PpoBuffer<E>,
    pub reward: Reward,
    pub running_reward: RunningReward,
    pub terminal: Terminal,
    pub rng: EntropyComponent<ChaCha8Rng>,
    marker: Agent,
}
impl<E: Env, T: Thinker<E>> AgentBundle<E, T>
where
    E::Params: PpoParams,
    E::Action: Action<E, Metadata = PpoMetadata>,
{
    pub fn new(
        pos: Vec3,
        color: Option<Color>,
        name: String,
        brain: Brain<E, T>,
        mut meshes: Mut<Assets<Mesh>>,
        mut materials: Mut<Assets<ColorMaterial>>,
        params: &E::Params,
        obs_len: usize,
        rng: &mut ResMut<GlobalEntropy<ChaCha8Rng>>,
    ) -> Self {
        Self {
            rng: EntropyComponent::from(rng),
            obs: FrameStack(
                vec![vec![0.0; obs_len].into_boxed_slice(); params.agent_frame_stack_len()].into(),
            ),
            obs_norm: RmsNormalize::new([1, obs_len].into()),
            action: E::Action::default(),
            replay_buffer: PpoBuffer::new(Some(params.agent_rb_max_len())),
            reward: Reward(0.0),
            running_reward: RunningReward {
                rewards: VecDeque::new(),
                history: VecDeque::new(),
                max_len: params.agent_frame_stack_len() * 100,
                history_max_len: params.agent_frame_stack_len() * 100,
            },
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
            brain,
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

    fn setup_system() -> SystemConfigs {
        setup.chain()
    }

    fn observation_system() -> SystemConfigs {
        get_observation.chain()
    }

    fn action_system() -> SystemConfigs {
        get_action::<Ffa, PpoThinker>.chain()
    }

    fn reward_system() -> SystemConfigs {
        (get_reward, send_reward::<Ffa, PpoThinker>).chain()
    }

    fn terminal_system() -> SystemConfigs {
        get_terminal.chain()
    }

    fn update_system() -> SystemConfigs {
        (update::<Ffa>, store_sarts::<Ffa>, check_dead::<Ffa>).chain()
    }

    fn learn_system() -> SystemConfigs {
        learn::<Ffa, PpoThinker>.chain()
    }

    fn ui_system() -> SystemConfigs {
        use crate::ui::*;
        (
            kdr::<Ffa, PpoThinker>,
            action_space::<Ffa, PpoThinker>,
            log,
            running_reward,
        )
            .chain()
    }
}

fn setup(
    params: Res<FfaParams>,
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    timestamp: Res<Timestamp>,
    mut rng: ResMut<GlobalEntropy<ChaCha8Rng>>,
) {
    let mut taken_names = vec![];
    let obs_len = OTHER_STATE_LEN * (params.num_agents - 1) + BASE_STATE_LEN;
    for _ in 0..params.num_agents {
        let mut rng_comp = EntropyComponent::from(&mut rng);
        let ts = timestamp.clone();
        let mut name = crate::names::random_name(&mut rng_comp);
        while taken_names.contains(&name) {
            name = crate::names::random_name(&mut rng_comp);
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
            &mut EntropyComponent::from(&mut rng),
        );
        let dist = Uniform::new(-250.0, 250.0);
        let agent_pos = Vec3::new(dist.sample(&mut rng_comp), dist.sample(&mut rng_comp), 0.0);
        let color = Color::rgb(rand::random(), rand::random(), rand::random());
        let mut agent = AgentBundle::<Ffa, PpoThinker>::new(
            agent_pos,
            Some(color),
            name,
            Brain::new(thinker, brain_name, ts),
            meshes.reborrow(),
            materials.reborrow(),
            &params,
            obs_len,
            &mut rng,
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
    }
}

fn get_observation(
    params: Res<FfaParams>,
    mut observations: Query<
        (
            &mut FrameStack<Box<[f32]>>,
            &mut RmsNormalize<TchBackend<f32>, 2>,
        ),
        With<Agent>,
    >,
    actions: Query<&FfaAction, With<Agent>>,
    agents: Query<Entity, With<Agent>>,
    agent_velocity: Query<&Velocity, With<Agent>>,
    agent_transform: Query<&Transform, With<Agent>>,
    agent_health: Query<&Health, With<Agent>>,
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
            },
            // .scaled_by(&phys_scaling),
            combat: CombatProperties { health: my_h.0 }, //.scaled_by(&CombatProperties {
            // health: 1.0 / params.agent_max_health,
            // }),
            other_states: vec![],
            map_inter: MapInteractionProperties::new(my_t, &cx), //.scaled_by(&map_scaling),
        };

        for other_ent in agents
            .iter()
            .filter(|a| *a != agent_ent)
            .sorted_by_key(|a| {
                let other_t = agent_transform.get(*a).unwrap();
                other_t.translation.distance(my_t.translation) as i64
            })
        {
            let other_t = agent_transform.get(other_ent).unwrap();
            let other_v = agent_velocity.get(other_ent).unwrap();
            let other_h = agent_health.get(other_ent).unwrap();
            // let status = brains.with_brain(other_id.0, |brain| brain.thinker.status());
            let other_state = OtherState {
                phys: PhysicalProperties {
                    position: my_t.translation.xy() - other_t.translation.xy(),
                    direction: other_t.local_y().xy(),
                    linvel: other_v.linvel,
                },
                //.scaled_by(&phys_scaling),
                combat: CombatProperties { health: other_h.0 }, //.scaled_by(&CombatProperties {
                // health: 1.0 / params.agent_max_health,
                // }),
                map_inter: MapInteractionProperties::new(other_t, &cx), //.scaled_by(&map_scaling),

                firing: actions.get(other_ent).unwrap().combat.shoot > 0.0,
            };

            my_state.other_states.push(other_state);
        }

        let max_len = params.agent_frame_stack_len;

        let obs = observations
            .get_mut(agent_ent)
            .unwrap()
            .1
            .forward(Tensor::from_floats(&*my_state.as_slice()).unsqueeze())
            .into_data()
            .value
            .into_boxed_slice();
        observations
            .get_mut(agent_ent)
            .unwrap()
            .0
            .push(obs, Some(max_len));
    }
}

pub fn get_action<E: Env, T: Thinker<E>>(
    params: Res<E::Params>,
    mut obs_brains_actions: Query<
        (
            &FrameStack<Box<[f32]>>,
            &mut Brain<E, T>,
            &mut E::Action,
            &mut EntropyComponent<ChaCha8Rng>,
        ),
        With<Agent>,
    >,
    frame_count: Res<FrameCount>,
) {
    if frame_count.0 as usize % params.agent_frame_stack_len() == 0 {
        obs_brains_actions
            .par_iter_mut()
            .for_each_mut(|(obs, mut brain, mut actions, mut rng)| {
                let action = brain.act(&obs, &*params, &mut rng);
                // if let Some(action) = action {
                // if let Some(action) = status.last_action {
                *actions = action;
                // }
                // }
            });
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
        let my_health = health.get(agent_ent).unwrap();
        let my_t = agent_transform.get(agent_ent).unwrap();
        if let Ok(action) = actions.get(agent_ent).cloned() {
            if action.combat.shoot > 0.0 && my_health.0 > 0.0 {
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
            my_force.force = action.phys.force.clamp(Vec2::splat(-1.0), Vec2::splat(1.0))
                * params.agent_lin_move_force;
            my_force.torque = action.phys.torque.clamp(-1.0, 1.0) * params.agent_ang_move_force;
        }
    }
}

pub fn send_reward<E: Env, T: Thinker<E>>(
    params: Res<E::Params>,
    agents: Query<Entity, With<Agent>>,
    frame_count: Res<FrameCount>,
    rewards: Query<&Reward, With<Agent>>,
    mut running_rewards: Query<&mut RunningReward, With<Agent>>,
    mut brains: Query<&mut Brain<E, T>, With<Agent>>,
) {
    if frame_count.0 as usize % params.agent_frame_stack_len() == 0 {
        for agent_ent in agents.iter() {
            let reward = rewards.get(agent_ent).unwrap().0;
            brains.get_mut(agent_ent).unwrap().writer.add_scalar(
                "Reward/Frame",
                reward,
                frame_count.0 as usize,
            );
            if let Some(running_reward) = running_rewards.get_mut(agent_ent).unwrap().update(reward)
            {
                brains.get_mut(agent_ent).unwrap().writer.add_scalar(
                    "Reward/Running",
                    running_reward,
                    frame_count.0 as usize,
                );
            }
        }
    }
}

pub fn get_terminal(
    mut terminals: Query<&mut Terminal, With<Agent>>,
    agents: Query<Entity, With<Agent>>,
    health: Query<&Health, With<Agent>>,
) {
    for agent_ent in agents.iter() {
        terminals.get_mut(agent_ent).unwrap().0 = health.get(agent_ent).unwrap().0 <= 0.0;
    }
}

pub fn update<E: Env>(
    mut commands: Commands,
    params: Res<E::Params>,
    mut name_text_t: Query<
        (Entity, &mut Transform, &mut Text, &mut NameText),
        (With<NameText>, Without<Agent>),
    >,
    mut health_bar_t: Query<
        (Entity, &mut Transform, &HealthBar),
        (Without<NameText>, Without<Agent>),
    >,
    names: Query<&Name, With<Agent>>,
    kills: Query<&Kills, With<Agent>>,
    deaths: Query<&Deaths, With<Agent>>,
    health: Query<&Health, With<Agent>>,
    agent_transform: Query<&Transform, With<Agent>>,
) {
    for (t_ent, mut t, mut text, text_comp) in name_text_t.iter_mut() {
        if let Ok(agent) = agent_transform.get(text_comp.entity_following) {
            t.translation = agent.translation + Vec3::new(0.0, params.agent_radius() + 20.0, 2.0);
            text.sections[0].value = format!(
                "{} {}-{}",
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
            t.translation = agent.translation + Vec3::new(0.0, params.agent_radius() + 5.0, 2.0);
            let health = health.get(hb.entity_following).unwrap();
            t.scale = Vec3::new(health.0 / params.agent_max_health() * 100.0, 1.0, 1.0);
        } else {
            commands.entity(t_ent).despawn();
        }
    }
}

pub fn check_dead<E: Env>(
    params: Res<E::Params>,
    agents: Query<Entity, With<Agent>>,
    mut health: Query<&mut Health, With<Agent>>,
    mut deaths: Query<&mut Deaths, With<Agent>>,
    mut agent_transform: Query<&mut Transform, With<Agent>>,
    mut rng: ResMut<GlobalEntropy<ChaCha8Rng>>,
) where
    E::Action: Action<E, Metadata = PpoMetadata>,
{
    for agent_ent in agents.iter() {
        let mut my_health = health.get_mut(agent_ent).unwrap();
        if my_health.0 <= 0.0 {
            // if let Ok(mut rb) = rbs.get_mut(agent_ent) {
            //     let final_val = actions.get(agent_ent).unwrap().metadata().val;
            //     rb.finish_trajectory(Some(final_val));
            // }

            deaths.get_mut(agent_ent).unwrap().0 += 1;
            my_health.0 = params.agent_max_health();
            let dist = Uniform::new(-250.0, 250.0);
            let mut rng_comp = EntropyComponent::from(&mut rng);
            let agent_pos = Vec3::new(dist.sample(&mut rng_comp), dist.sample(&mut rng_comp), 0.0);
            agent_transform.get_mut(agent_ent).unwrap().translation = agent_pos;
        }
    }
}

pub fn learn<E: Env, T>(
    params: Res<E::Params>,
    mut query: Query<
        (
            &mut EntropyComponent<ChaCha8Rng>,
            &mut PpoBuffer<E>,
            &E::Action,
            &Name,
            &mut Brain<E, T>,
        ),
        With<Agent>,
    >,
    frame_count: Res<FrameCount>,
    mut log: ResMut<LogText>,
) where
    E::Params: PpoParams,
    E::Action: Action<E, Metadata = PpoMetadata>,
    T: Thinker<E, Status = PpoStatus>,
{
    if frame_count.0 as usize >= params.training_batch_size()
        && frame_count.0 as usize % params.agent_update_interval() == 0
    {
        query
            .par_iter_mut()
            .for_each_mut(|(mut rng, mut rb, action, _name, mut brain)| {
                // if deaths.0 > 0 {
                rb.finish_trajectory(Some(action.metadata().val));
                brain.learn(frame_count.0 as usize, &mut *rb, &*params, &mut rng);
                // }
            });
        for (_, _, _, name, brain) in query.iter() {
            let status = Thinker::<E>::status(&brain.thinker);
            log.push(format!(
                "{} Policy Loss: {}",
                name.0, status.recent_policy_loss
            ));
            log.push(format!(
                "{} Policy Entropy: {}",
                name.0, status.recent_entropy_loss
            ));
            log.push(format!(
                "{} Policy Clip Ratio: {}",
                name.0, status.recent_nclamp
            ));
            log.push(format!(
                "{} Value Loss: {}",
                name.0, status.recent_value_loss
            ));
        }
    }
}
