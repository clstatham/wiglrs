use std::{
    collections::VecDeque,
    fs::File,
    io::{Read, Write},
    path::Path,
};

use crate::{
    brains::{
        learners::{utils::RmsNormalize, Learner, OffPolicyBuffer, DEVICE},
        models::{Policy, ValueEstimator},
        Policies, ValueEstimators,
    },
    FrameStack, TbWriter, Timestamp,
};
use bevy::{
    core::FrameCount, ecs::schedule::SystemConfigs, prelude::*, sprite::MaterialMesh2dBundle,
    utils::HashMap,
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
    fn calculate<A, P: Policy, V: ValueEstimator>(
        obs: &FrameStack<Box<[f32]>>,
        action: &A,
        policy: &P,
        value: &V,
    ) -> Self
    where
        A: Action<Logits = P::Logits>;
}

impl StepMetadata for () {
    fn calculate<A, P: Policy, V: ValueEstimator>(
        _obs: &FrameStack<Box<[f32]>>,
        _action: &A,
        _policy: &P,
        _value: &V,
    ) -> Self
    where
        A: Action<Logits = P::Logits>,
    {
    }
}

pub trait Action: Clone + Default + Component {
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

pub trait DefaultFrameStack: Observation {
    fn default_frame_stack(params: &Params) -> FrameStack<Self>;
}

#[derive(Debug, Clone, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Param {
    Int(isize),
    Float(f64),
    String(String),
}

impl Param {
    pub fn as_int(&self) -> Option<isize> {
        if let Self::Int(i) = self {
            Some(*i)
        } else {
            None
        }
    }

    pub fn as_float(&self) -> Option<f64> {
        if let Self::Float(f) = self {
            Some(*f)
        } else {
            None
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        if let Self::String(s) = self {
            Some(s)
        } else {
            None
        }
    }
}

#[derive(Debug, Resource, Serialize, Deserialize)]
pub struct Params {
    params: HashMap<String, Param>,
}

impl Params {
    pub fn get(&self, key: impl AsRef<str>) -> Option<&Param> {
        self.params.get(key.as_ref())
    }

    pub fn get_int(&self, key: impl AsRef<str>) -> Option<isize> {
        self.get(key.as_ref()).and_then(|p| p.as_int())
    }

    pub fn get_float(&self, key: impl AsRef<str>) -> Option<f64> {
        self.get(key.as_ref()).and_then(|p| p.as_float())
    }

    pub fn get_str(&self, key: impl AsRef<str>) -> Option<&str> {
        self.get(key.as_ref()).and_then(|p| p.as_str())
    }

    pub fn to_yaml(&self) -> Result<String, Box<dyn std::error::Error>>
    where
        Self: Serialize,
    {
        let s = serde_yaml::to_string(self)?;
        Ok(s)
    }
    pub fn to_yaml_file(&self, path: impl AsRef<Path>) -> Result<(), Box<dyn std::error::Error>>
    where
        Self: Serialize,
    {
        let mut f = File::create(path)?;
        let s = self.to_yaml()?;
        write!(f, "{}", s)?;
        Ok(())
    }
    pub fn from_yaml<'a>(json: &'a str) -> Result<Self, Box<dyn std::error::Error>>
    where
        Self: Deserialize<'a>,
    {
        let this = serde_yaml::from_str(json)?;
        Ok(this)
    }
    pub fn from_yaml_file(path: impl AsRef<Path>) -> Result<Self, Box<dyn std::error::Error>>
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
    type Observation: Observation + DefaultFrameStack + Component + Send + Sync;
    type Action: Action + Component + Send + Sync;

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
                Self::update_system(),
                Self::action_system(),
                Self::reward_system(),
                Self::terminal_system(),
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
pub struct Kills(pub isize);

#[derive(Component)]
pub struct Deaths(pub usize);

#[derive(Component)]
pub struct Name(pub String);

#[derive(Component)]
pub struct NextObservation(pub Option<FrameStack<Box<[f32]>>>);

#[derive(Component)]
pub struct CurrentObservation(pub FrameStack<Box<[f32]>>);

#[derive(Bundle)]
pub struct AgentBundle<A: Action> {
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
    pub writer: TbWriter,
    pub action: A,
    pub next_obs: NextObservation,
    pub current_obs: CurrentObservation,
    pub obs_norm: RmsNormalize,
    pub reward: Reward,
    pub running_reward: RunningReturn,
    pub terminal: Terminal,
    pub rng: EntropyComponent<ChaCha8Rng>,
    marker: Agent,
}
impl<A: Action> AgentBundle<A> {
    pub fn new(
        pos: Vec3,
        color: Option<Color>,
        name: String,
        timestamp: &Timestamp,
        mut meshes: Mut<Assets<Mesh>>,
        mut materials: Mut<Assets<ColorMaterial>>,
        params: &Params,
        obs_len: usize,
        rng: &mut ResMut<GlobalEntropy<ChaCha8Rng>>,
    ) -> Self {
        let mut writer = TbWriter::default();
        writer.init(Some(name.as_str()), timestamp);
        let obs = FrameStack(
            vec![
                vec![0.0; obs_len].into_boxed_slice();
                params.get_int("agent_frame_stack_len").unwrap() as usize
            ]
            .into(),
        );
        Self {
            action: A::default(),
            writer,
            rng: EntropyComponent::from(rng),
            next_obs: NextObservation(Some(obs.clone())),
            current_obs: CurrentObservation(obs),
            obs_norm: RmsNormalize::new(&[obs_len]).unwrap(),
            reward: Reward(0.0),
            running_reward: RunningReturn::new(
                params.get_int("agent_update_interval").unwrap() as usize
            ),
            terminal: Terminal(false),
            marker: Agent,
            rb: RigidBody::Dynamic,
            col: Collider::ball(params.get_float("agent_radius").unwrap() as f32),
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
                    .add(
                        shape::Circle::new(params.get_float("agent_radius").unwrap() as f32).into(),
                    )
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

pub fn get_action<A, P: Policy>(
    params: Res<Params>,
    policies: Res<Policies<P>>,
    mut obs_brains_actions: Query<(&CurrentObservation, &AgentId, &mut A), With<Agent>>,
    frame_count: Res<FrameCount>,
) where
    A: Action<Logits = P::Logits>,
{
    if frame_count.0 as usize % params.get_int("agent_frame_stack_len").unwrap() as usize == 0 {
        obs_brains_actions
            .par_iter_mut()
            .for_each_mut(|(fs, id, mut actions)| {
                let obs = fs.0.as_tensor();
                let (action, logits) = policies.0.get(id.0).unwrap().act(&obs).unwrap();
                *actions = A::from_slice(
                    action.squeeze(0).unwrap().to_vec1().unwrap().as_slice(),
                    logits,
                );
            });
    }
}

pub fn send_reward(
    params: Res<Params>,
    agents: Query<Entity, With<Agent>>,
    frame_count: Res<FrameCount>,
    rewards: Query<&Reward, With<Agent>>,
    mut running_rewards: Query<&mut RunningReturn, With<Agent>>,
    mut writers: Query<&mut TbWriter, With<Agent>>,
) {
    for agent_ent in agents.iter() {
        let reward = rewards.get(agent_ent).unwrap().0;
        let running = running_rewards.get_mut(agent_ent).unwrap().update(reward);
        if frame_count.0 as usize > params.get_int("agent_warmup").unwrap() as usize
            && frame_count.0 as usize % params.get_int("agent_frame_stack_len").unwrap() as usize
                == 0
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

pub fn update(
    mut commands: Commands,
    params: Res<Params>,
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
    mut obs: Query<(&mut CurrentObservation, &mut NextObservation), With<Agent>>,
) {
    for (t_ent, mut t, mut text, text_comp) in name_text_t.iter_mut() {
        if let Ok(agent) = agent_transform.get(text_comp.entity_following) {
            t.translation = agent.translation
                + Vec3::new(
                    0.0,
                    params.get_float("agent_radius").unwrap() as f32 + 20.0,
                    2.0,
                );
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
            t.translation = agent.translation
                + Vec3::new(
                    0.0,
                    params.get_float("agent_radius").unwrap() as f32 + 5.0,
                    2.0,
                );
            let health = health.get(hb.entity_following).unwrap();
            t.scale = Vec3::new(
                health.0 / params.get_float("agent_max_health").unwrap() as f32 * 100.0,
                1.0,
                1.0,
            );
        } else {
            commands.entity(t_ent).despawn();
        }
    }

    for (mut cur, mut next) in obs.iter_mut() {
        cur.0 = next.0.take().unwrap();
    }
}

pub fn check_dead(
    params: Res<Params>,
    agents: Query<Entity, With<Agent>>,
    mut health: Query<&mut Health, With<Agent>>,
    mut deaths: Query<&mut Deaths, With<Agent>>,
    mut agent_transform: Query<&mut Transform, With<Agent>>,
    mut rng: ResMut<GlobalEntropy<ChaCha8Rng>>,
) {
    for agent_ent in agents.iter() {
        let mut my_health = health.get_mut(agent_ent).unwrap();
        if my_health.0 <= 0.0 {
            deaths.get_mut(agent_ent).unwrap().0 += 1;
            my_health.0 = params.get_float("agent_max_health").unwrap() as f32;
            let dist = Uniform::new(-250.0, 250.0);
            let mut rng_comp = EntropyComponent::from(&mut rng);
            let agent_pos = Vec3::new(dist.sample(&mut rng_comp), dist.sample(&mut rng_comp), 0.0);
            agent_transform.get_mut(agent_ent).unwrap().translation = agent_pos;
        }
    }
}

pub fn learn<E: Env, P: Policy, V: ValueEstimator, L: Learner<E, P, V>>(
    params: Res<Params>,
    mut learner: ResMut<L>,
    policies: Res<Policies<P>>,
    values: Res<ValueEstimators<V>>,
    query: Query<(Entity, &Name), With<Agent>>,
    frame_count: Res<FrameCount>,
    mut writers: Query<&mut TbWriter, With<Agent>>,
) where
    L: Learner<E, P, V>,
    L::Buffer: OffPolicyBuffer<E>,
{
    if frame_count.0 as usize > params.get_int("agent_warmup").unwrap() as usize
        && frame_count.0 as usize % params.get_int("agent_update_interval").unwrap() as usize == 0
    {
        learner.learn(policies.0.as_slice(), values.0.as_slice());
        for (agent, _name) in query.iter() {
            let status = learner.status();
            use crate::brains::learners::Status;
            status.log(&mut writers.get_mut(agent).unwrap(), frame_count.0 as usize);
        }
    }
}
