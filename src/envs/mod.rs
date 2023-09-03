use std::{
    collections::VecDeque,
    fs::File,
    io::{Read, Write},
    path::Path,
};

use crate::{
    brains::{
        learners::{
            ppo::{rollout_buffer::PpoMetadata, PpoStatus},
            utils::RmsNormalize,
            Learner, Sart, DEVICE,
        },
        models::{Policy, ValueEstimator},
    },
    ui::LogText,
    FrameStack, TbWriter, Timestamp,
};
use bevy::{
    core::FrameCount, ecs::schedule::SystemConfigs, prelude::*, sprite::MaterialMesh2dBundle,
};
use bevy_prng::ChaCha8Rng;
use bevy_rand::{prelude::EntropyComponent, resource::GlobalEntropy};
use bevy_rapier2d::prelude::*;
use candle_core::Tensor;
use rand_distr::{Distribution, Uniform};
use serde::{de::DeserializeOwned, Deserialize, Serialize};

pub mod maps;
pub mod modules;
pub mod tdm;

pub trait StepMetadata: Default {
    fn calculate<E: Env, P: Policy, V: ValueEstimator>(
        obs: &FrameStack<Box<[f32]>>,
        action: &E::Action,
        policy: &P,
        value: &V,
    ) -> Self
    where
        E::Action: Action<E, Logits = P::Logits>;
}

pub trait Action<E: Env + ?Sized>: Clone + Default {
    type Logits;
    fn as_slice(&self) -> Box<[f32]>;
    fn from_slice(v: &[f32], logits: Self::Logits) -> Self;
    fn logits(&self) -> Option<&Self::Logits>;
    fn as_tensor(&self) -> Tensor {
        Tensor::new(&*self.as_slice(), &DEVICE).unwrap()
    }
}

pub trait Observation: Clone
where
    Self: Sized,
{
    fn as_slice(&self) -> Box<[f32]>;
    fn as_tensor(&self) -> Tensor {
        Tensor::new(&*self.as_slice(), &DEVICE).unwrap()
    }
}

pub trait DefaultFrameStack<E: Env + ?Sized>: Observation {
    fn default_frame_stack(params: &E::Params) -> FrameStack<Self>;
}

pub trait Params {
    fn agent_radius(&self) -> f32;
    fn agent_max_health(&self) -> f32;
    fn num_agents(&self) -> usize;
    fn agent_frame_stack_len(&self) -> usize;
    fn actor_lr(&self) -> f64;
    fn critic_lr(&self) -> f64;
    fn entropy_beta(&self) -> f32;
    fn training_batch_size(&self) -> usize;
    fn training_epochs(&self) -> usize;
    fn agent_rb_max_len(&self) -> usize;
    fn agent_warmup(&self) -> usize;
    fn agent_update_interval(&self) -> usize;

    fn to_yaml(&self) -> Result<String, Box<dyn std::error::Error>>
    where
        Self: Serialize,
    {
        let s = serde_yaml::to_string(self)?;
        Ok(s)
    }
    fn to_yaml_file(&self, path: impl AsRef<Path>) -> Result<(), Box<dyn std::error::Error>>
    where
        Self: Serialize,
    {
        let mut f = File::create(path)?;
        let s = self.to_yaml()?;
        write!(f, "{}", s)?;
        Ok(())
    }
    fn from_yaml<'a>(json: &'a str) -> Result<Self, Box<dyn std::error::Error>>
    where
        Self: Deserialize<'a>,
    {
        let this = serde_yaml::from_str(json)?;
        Ok(this)
    }
    fn from_yaml_file(path: impl AsRef<Path>) -> Result<Self, Box<dyn std::error::Error>>
    where
        Self: DeserializeOwned,
    {
        let mut f = File::open(path)?;
        let mut s = String::new();
        f.read_to_string(&mut s)?;
        let this = Self::from_yaml(s.as_str())?;
        Ok(this)
    }
}

pub trait Env: Resource {
    type Params: Params + Default + Resource + Send + Sync;
    type Observation: Observation + DefaultFrameStack<Self> + Component + Send + Sync;
    type Action: Action<Self> + Component + Send + Sync;

    fn init() -> Self;

    fn setup_system() -> SystemConfigs;
    fn observation_system() -> SystemConfigs;
    fn action_system() -> SystemConfigs;
    fn reward_system() -> SystemConfigs;
    fn terminal_system() -> SystemConfigs;
    fn update_system() -> SystemConfigs;
    fn learn_system() -> SystemConfigs;
    fn ui_system() -> SystemConfigs;

    fn add_main_systems(app: &mut App) {
        app.add_systems(Startup, Self::observation_system());
        app.add_systems(
            Update,
            (
                Self::action_system(),
                Self::reward_system(),
                Self::terminal_system(),
                Self::update_system(),
                Self::observation_system(),
                Self::learn_system(),
            )
                .chain()
                .after(PhysicsSet::Writeback),
        );
    }
}

#[derive(Debug, Clone, Copy, Component)]
pub struct AgentId(pub usize);

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
    pub agent_warmup: usize,
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
            agent_critic_lr: 0.001,
            agent_training_epochs: 10,
            agent_training_batch_size: 64,
            agent_entropy_beta: 0.00001,
            agent_update_interval: 2048,
            agent_warmup: 4096,
            agent_rb_max_len: 2048,
            agent_frame_stack_len: 4,
            agent_radius: 20.0,
            agent_lin_move_force: 600.0,
            agent_ang_move_force: 1.0,
            agent_max_health: 100.0,
            agent_shoot_distance: 1000.0,
            distance_scaling: 1.0 / 2000.0,
            reward_for_kill: 1.0,
            reward_for_death: -1.0,
            reward_for_hit: 0.1,
            reward_for_getting_hit: -0.1,
            reward_for_miss: -0.1,
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

    fn agent_warmup(&self) -> usize {
        self.agent_warmup
    }

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
pub struct RunningReturn {
    pub buf: VecDeque<f32>,
    // pub returns: VecDeque<f32>,
    pub history: VecDeque<f32>,
    pub max_len: usize,
}

impl RunningReturn {
    pub fn new(max_len: usize) -> Self {
        Self {
            buf: VecDeque::default(),
            // returns: VecDeque::default(),
            history: VecDeque::default(),
            max_len,
        }
    }
}

impl RunningReturn {
    pub fn update(&mut self, reward: f32) -> Option<f32> {
        if self.buf.len() >= self.max_len {
            self.buf.pop_front();
        }
        self.buf.push_back(reward);

        // self.returns.clear();
        let mut val = 0.0;
        for r in self.buf.iter().rev() {
            val = *r + 0.99 * val;
            // self.returns.push_front(val);
        }
        if self.history.len() >= self.max_len {
            self.history.pop_front();
        }
        self.history.push_back(val);
        Some(val)
    }

    pub fn get(&self) -> Option<f32> {
        self.history.back().copied()
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
pub struct AgentBundle<E: Env, P: Policy, V: ValueEstimator, L: Learner<E>> {
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
    pub policy: P,
    pub value: V,
    pub writer: TbWriter,
    pub obs: FrameStack<Box<[f32]>>,
    pub obs_norm: RmsNormalize,
    pub action: E::Action,
    pub learner: L,
    pub replay_buffer: L::Buffer,
    pub reward: Reward,
    pub running_reward: RunningReturn,
    pub terminal: Terminal,
    pub rng: EntropyComponent<ChaCha8Rng>,
    marker: Agent,
}
impl<E: Env, P: Policy, V: ValueEstimator, L: Learner<E>> AgentBundle<E, P, V, L> {
    pub fn new(
        pos: Vec3,
        color: Option<Color>,
        name: String,
        policy: P,
        value: V,
        learner: L,
        buffer: L::Buffer,
        timestamp: &Timestamp,
        mut meshes: Mut<Assets<Mesh>>,
        mut materials: Mut<Assets<ColorMaterial>>,
        params: &E::Params,
        obs_len: usize,
        rng: &mut ResMut<GlobalEntropy<ChaCha8Rng>>,
    ) -> Self {
        let mut writer = TbWriter::default();
        writer.init(Some(name.as_str()), timestamp);
        Self {
            policy,
            value,
            writer,
            learner,
            rng: EntropyComponent::from(rng),
            obs: FrameStack(
                vec![vec![0.0; obs_len].into_boxed_slice(); params.agent_frame_stack_len()].into(),
            ),
            obs_norm: RmsNormalize::new(&[obs_len]).unwrap(),
            action: E::Action::default(),
            replay_buffer: buffer,
            reward: Reward(0.0),
            running_reward: RunningReturn::new(params.agent_update_interval()),
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
                angular_damping: 120.0,
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
        }
    }
}

pub fn get_action<E: Env, P: Policy, V: ValueEstimator>(
    params: Res<E::Params>,
    mut obs_brains_actions: Query<
        (
            &FrameStack<Box<[f32]>>,
            &P,
            &V,
            &mut E::Action,
            &mut EntropyComponent<ChaCha8Rng>,
        ),
        With<Agent>,
    >,
    frame_count: Res<FrameCount>,
) where
    E::Action: Action<E, Logits = P::Logits>,
{
    if frame_count.0 as usize % params.agent_frame_stack_len() == 0 {
        obs_brains_actions.par_iter_mut().for_each_mut(
            |(fs, policy, value, mut actions, mut rng)| {
                let obs = fs.as_tensor();
                let (action, logits) = policy.act(&obs).unwrap();
                *actions = <E::Action as Action<E>>::from_slice(
                    action.squeeze(0).unwrap().to_vec1().unwrap().as_slice(),
                    logits,
                );
            },
        );
    }
}

pub fn send_reward<E: Env, T: Learner<E>>(
    params: Res<E::Params>,
    agents: Query<Entity, With<Agent>>,
    frame_count: Res<FrameCount>,
    rewards: Query<&Reward, With<Agent>>,
    mut running_rewards: Query<&mut RunningReturn, With<Agent>>,
    mut writers: Query<&mut TbWriter, With<Agent>>,
) {
    for agent_ent in agents.iter() {
        let reward = rewards.get(agent_ent).unwrap().0;
        let running = running_rewards.get_mut(agent_ent).unwrap().update(reward);
        if frame_count.0 as usize >= params.agent_warmup()
            && frame_count.0 as usize % params.agent_frame_stack_len() == 0
        {
            writers.get_mut(agent_ent).unwrap().add_scalar(
                "Reward/Frame",
                reward,
                frame_count.0 as usize,
            );
            if let Some(running_reward) = running {
                writers.get_mut(agent_ent).unwrap().add_scalar(
                    "Reward/Running",
                    running_reward,
                    frame_count.0 as usize,
                );
            }
        }
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
) {
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

pub fn learn<E: Env, L: Learner<E>, P: Policy, V: ValueEstimator>(
    params: Res<E::Params>,
    obs: Query<&FrameStack<Box<[f32]>>, With<Agent>>,
    mut query: Query<
        (
            Entity,
            &mut EntropyComponent<ChaCha8Rng>,
            &Name,
            &mut L,
            &mut L::Buffer,
            &P,
            &V,
        ),
        With<Agent>,
    >,
    frame_count: Res<FrameCount>,
    mut log: ResMut<LogText>,
) where
    L: Learner<E, Status = PpoStatus>,
{
    if frame_count.0 as usize >= params.agent_warmup()
        && frame_count.0 as usize % params.agent_update_interval() == 0
    {
        query.par_iter_mut().for_each_mut(
            |(ent, mut rng, _name, mut learner, mut rb, policy, value)| {
                let val: f32 = value
                    .estimate_value(&obs.get(ent).unwrap().as_tensor())
                    .unwrap()
                    .reshape(())
                    .unwrap()
                    .to_scalar()
                    .unwrap();
                use crate::brains::learners::Buffer;
                rb.finish_trajectory(Some(val));
                learner.learn(policy, value, &*rb, &mut rng);
            },
        );
        for (_, _, name, learner, _, _, _) in query.iter() {
            let status = learner.status();
            log.push(format!("{} Policy Loss: {}", name.0, status.policy_loss));
            log.push(format!(
                "{} Policy Entropy: {}",
                name.0, status.entropy_loss
            ));
            log.push(format!("{} Policy Clip Ratio: {}", name.0, status.nclamp));
            log.push(format!("{} Value Loss: {}", name.0, status.value_loss));
        }
    }
}
