#![allow(clippy::type_complexity, clippy::too_many_arguments)]

use std::{collections::BTreeMap, f32::consts::PI, sync::atomic::AtomicU64};

use bevy::{core::FrameCount, math::Vec3Swizzles, prelude::*, sprite::MaterialMesh2dBundle};
use bevy_rapier2d::prelude::*;
use rand::seq::SliceRandom;
use tch::{
    kind::INT64_CPU,
    nn::{self, OptimizerConfig},
    Tensor,
};

lazy_static::lazy_static! {
    pub static ref NAMES: Vec<String> = {
        let names = "Voli
Plen
Autumn
Rascal
Kevin
Morty
Lumi
Savannah
Horton
Sandi
McCullough
Kent
Key
Terri
Rush
Leo
Sanders
Eric
Bassett
Beverly
Curtis
Deb
Conley
Mel
Potts
Jodi
Brady
Gayle
Courtney
Rosemary
FitzPatrick
Sandra
Hansen
Jeffery
McCall
Freddie
Atkins
Irma
Sheridan
Gloria
Cantu
Donna
Cross
Santiago
Combs
Kari
Larson
Carrie
Stokes
Hugo
Day
Wilma
Drake
Lupe
Fox
Thomas
Manning
Cara
Briggs
Lorena
Clayton
Carlos
Fish
Dwayne
Cole
Cummings
Angelo
Mathews
Brittany
Shelton
Joshua
Stuart
Daryl
Bradley
Thelma
Cahill
Ryan
Coffey
Richard
Sheldon
Bob
Conway
Helen
Riggs
Jim
Wilson
Curtis
Snell
Laverne
Walton
Joyce
Norton
Javier
Martinez
Van
Keenan".split_ascii_whitespace();
        names.into_iter().map(str::trim).map(ToOwned::to_owned).collect()
    };
}

pub fn random_name() -> String {
    NAMES.choose(&mut rand::thread_rng()).unwrap().to_owned()
}

#[derive(Default, Clone, Copy)]
pub struct State {
    pub position: Vec2,
    pub linvel: Vec2,
    pub angle: f32,
    pub angvel: f32,
}

impl State {
    pub fn as_vec(&self) -> Vec<f32> {
        vec![
            self.position.x / 1000.0,
            self.position.y / 1000.0,
            self.linvel.x / AGENT_MAX_LIN_VEL,
            self.linvel.y / AGENT_MAX_LIN_VEL,
            self.angle / PI,
            self.angvel / PI,
        ]
    }

    pub fn dim() -> usize {
        Self::default().as_vec().len()
    }
}

pub const NUM_AGENTS: usize = 10;
pub const AGENT_EMBED_DIM: i64 = 512;
pub const AGENT_LR: f64 = 1e-4;
pub const AGENT_OPTIM_EPOCHS: usize = 100;
pub const AGENT_OPTIM_BATCH_SIZE: i64 = 64;

pub const AGENT_RADIUS: f32 = 20.0;
pub const AGENT_MAX_LIN_VEL: f32 = 300.0;
pub const AGENT_MAX_ANG_VEL: f32 = 2.0;
pub const AGENT_LIN_MOVE_FORCE: f32 = 300.0;
pub const AGENT_ANG_MOVE_FORCE: f32 = 1.0;

pub const AGENT_MAX_HEALTH: f32 = 100.0;
pub const AGENT_SHOOT_DISTANCE: f32 = 200.0;

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

pub struct SavedStep {
    pub state: Vec<f32>,
    pub action: Vec<f32>,
    pub reward: f32,
    pub terminal: bool,
}

impl SavedStep {
    pub fn unzip(self) -> (Vec<f32>, Vec<f32>, f32, bool) {
        (self.state, self.action, self.reward, self.terminal)
    }
}

static BRAIN_IDS: AtomicU64 = AtomicU64::new(0);

pub struct Actor {
    pub vs: nn::VarStore,
    pub model: nn::Sequential,
    pub opt: nn::Optimizer,
    pub device: tch::Device,
}

impl Default for Actor {
    fn default() -> Self {
        let dev = tch::Device::cuda_if_available();
        let vs = nn::VarStore::new(dev);
        let p = &vs.root();
        Self {
            model: nn::seq()
                .add(nn::linear(
                    p / "al1",
                    (State::dim() * NUM_AGENTS) as i64,
                    AGENT_EMBED_DIM,
                    Default::default(),
                ))
                .add_fn(|xs| xs.relu())
                .add(nn::linear(
                    p / "al2",
                    AGENT_EMBED_DIM,
                    AGENT_EMBED_DIM,
                    Default::default(),
                ))
                .add_fn(|xs| xs.relu())
                .add(nn::linear(
                    p / "al3",
                    AGENT_EMBED_DIM,
                    Action::dim() as i64,
                    Default::default(),
                ))
                .add_fn(|xs| xs.tanh()),
            opt: nn::Adam::default().build(&vs, AGENT_LR).unwrap(),
            vs,
            device: dev,
        }
    }
}

impl Actor {
    pub fn forward(&self, state: &Tensor) -> Tensor {
        state.to(self.device).apply(&self.model)
    }
}

impl Clone for Actor {
    fn clone(&self) -> Self {
        let mut new = Self::default();
        new.vs.copy(&self.vs).unwrap();
        new
    }
}

pub struct Critic {
    pub vs: nn::VarStore,
    pub model: nn::Sequential,
    pub opt: nn::Optimizer,
    pub device: tch::Device,
}

impl Default for Critic {
    fn default() -> Self {
        let dev = tch::Device::cuda_if_available();
        let vs = nn::VarStore::new(dev);
        let p = &vs.root();
        Self {
            model: nn::seq()
                .add(nn::linear(
                    p / "cl1",
                    (State::dim() * NUM_AGENTS) as i64 + (Action::dim() as i64),
                    AGENT_EMBED_DIM,
                    Default::default(),
                ))
                .add_fn(|xs| xs.relu())
                .add(nn::linear(
                    p / "cl2",
                    AGENT_EMBED_DIM,
                    AGENT_EMBED_DIM,
                    Default::default(),
                ))
                .add_fn(|xs| xs.relu())
                .add(nn::linear(
                    p / "cl3",
                    AGENT_EMBED_DIM,
                    1,
                    Default::default(),
                )),
            opt: nn::Adam::default().build(&vs, AGENT_LR).unwrap(),
            vs,
            device: dev,
        }
    }
}
impl Critic {
    pub fn forward(&self, state: &Tensor, action: &Tensor) -> Tensor {
        let xs = Tensor::cat(&[action.copy(), state.copy()], 1);
        xs.to(self.device).apply(&self.model)
    }
}

impl Clone for Critic {
    fn clone(&self) -> Self {
        let mut new = Self::default();
        new.vs.copy(&self.vs).unwrap();
        new
    }
}

fn track(dest: &mut nn::VarStore, src: &nn::VarStore, tau: f64) {
    tch::no_grad(|| {
        for (dest, src) in dest
            .trainable_variables()
            .iter_mut()
            .zip(src.trainable_variables().iter())
        {
            dest.copy_(&(tau * src + (1.0 - tau) * &*dest));
        }
    })
}

pub struct Brain {
    pub name: String,
    pub color: Color,
    pub id: u64,
    pub actor: Actor,
    pub critic: Critic,
    pub target_actor: Actor,
    pub target_critic: Critic,
    pub rb: Vec<SavedStep>,
}

impl Brain {
    pub fn new() -> Self {
        let id = BRAIN_IDS.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let name = random_name();

        let actor = Actor::default();
        let critic = Critic::default();
        Self {
            name,
            color: Color::rgb(rand::random(), rand::random(), rand::random()),
            id,
            target_actor: actor.clone(),
            target_critic: critic.clone(),
            actor,
            critic,
            rb: Vec::new(),
        }
    }

    pub fn act(&mut self, state: &Tensor) -> Tensor {
        tch::no_grad(|| self.actor.forward(state))
    }

    pub fn learn(&mut self) {
        let device = self.actor.device;
        let nsteps = self.rb.len() as i64;
        if nsteps <= 1 {
            return; // sometimes bevy is slow to despawn things
        }
        let (states, actions, rewards, terminals): (
            Vec<Tensor>,
            Vec<Tensor>,
            Vec<Tensor>,
            Vec<Tensor>,
        ) = itertools::multiunzip(self.rb.drain(..).map(|s| s.unzip()).map(|(s, a, r, t)| {
            (
                Tensor::from_slice(&s),
                Tensor::from_slice(&a),
                Tensor::from_slice(&[r]),
                Tensor::from_slice(&[if t { 1.0 } else { 0.0 }]),
            )
        }));
        let (states, actions, rewards, terminals) = (
            Tensor::stack(&states, 0).to(device),
            Tensor::stack(&actions, 0).to(device),
            Tensor::stack(&rewards, 0).to(device),
            Tensor::stack(&terminals, 0).to(device),
        );

        let mut total_loss = 0.0f32;
        for _ in 0..AGENT_OPTIM_EPOCHS {
            let batch_idxs =
                Tensor::randint(nsteps - 1, [AGENT_OPTIM_BATCH_SIZE], INT64_CPU).to(device);
            let state = states.index_select(0, &batch_idxs);
            let actions = actions.index_select(0, &batch_idxs);
            let rewards = rewards.index_select(0, &batch_idxs);
            let next_state = states.index_select(0, &(batch_idxs + 1));

            let mut q_target = self
                .target_critic
                .forward(&next_state, &self.target_actor.forward(&next_state));
            q_target = rewards + (0.99f32 * q_target).detach();

            let q = self.critic.forward(&state, &actions);

            let critic_loss = q.mse_loss(&q_target, tch::Reduction::Mean);

            self.critic.opt.zero_grad();
            critic_loss.backward();
            self.critic.opt.step();

            let actor_loss = -self
                .critic
                .forward(&state, &self.actor.forward(&state))
                .mean(tch::Kind::Float);

            self.actor.opt.zero_grad();
            actor_loss.backward();
            self.actor.opt.step();

            track(&mut self.target_actor.vs, &self.actor.vs, 0.005);
            track(&mut self.target_critic.vs, &self.critic.vs, 0.005);

            let actor_loss: f32 = actor_loss.try_into().unwrap();
            let critic_loss: f32 = critic_loss.try_into().unwrap();
            let loss = actor_loss + critic_loss;

            total_loss += loss;
        }
        println!("{} loss: {total_loss}", self.name);
    }
}

impl Default for Brain {
    fn default() -> Self {
        Self::new()
    }
}

type BrainBank = BTreeMap<Entity, Brain>;

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

fn setup_brains(world: &mut World) {
    let bank = BrainBank::default();
    world.insert_non_send_resource(bank);
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
            // brain transplant
            let mut brain = brains.remove(&agent).unwrap();
            brain.rb.clear();

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

fn spawn_agent(
    brain: Brain,
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
                &brain.name,
                TextStyle {
                    font: asset_server.load("fonts/FiraSans-Bold.ttf"),
                    font_size: 20.0,
                    color: Color::BLACK,
                },
            ),
            transform: Transform::from_translation(agent_pos),
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

    for _ in 0..NUM_AGENTS {
        let brain = Brain::new();
        spawn_agent(
            brain,
            &mut commands,
            &mut meshes,
            &mut materials,
            &mut brains,
            &asset_server,
        );
    }

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
        .insert(TransformBundle::from(Transform::from_xyz(500.0, 0.0, 0.0)));
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
) {
    let mut all_states = BTreeMap::new(); // you're in good hands...?
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
        let mut state = vec![State::default(); NUM_AGENTS];
        let my_state = State {
            position: transform.translation.xy(),
            linvel: velocity.linvel,
            angle: transform.rotation.to_euler(EulerRot::XYZ).2,
            angvel: velocity.angvel,
        };
        state[brains[&agent].id as usize] = my_state;

        for (other, _, other_vel, other_transform) in agents.iter().filter(|a| a.0 != agent) {
            let other_state = State {
                position: other_transform.translation.xy() - transform.translation.xy(),
                linvel: other_vel.linvel,
                angle: other_transform.rotation.to_euler(EulerRot::XYZ).2,
                angvel: other_vel.angvel,
            };
            state[brains[&other].id as usize] = other_state;
        }

        let state = state
            .into_iter()
            .map(|s| s.as_vec())
            .collect::<Vec<_>>()
            .concat();
        let action = brains
            .get_mut(&agent)
            .unwrap()
            .act(&Tensor::from_slice(&state));

        let action: Vec<f32> = action.try_into().unwrap();
        all_states.insert(agent, state);
        all_actions.insert(agent, Action::from_slice(&action));
        all_rewards.insert(agent, 0.0f32);
        all_terminals.insert(agent, false);
    }
    for (agent, mut force, mut velocity, _) in agents.iter_mut() {
        if all_actions[&agent].shoot {
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
                    *all_rewards.get_mut(&agent).unwrap() += 1.0;
                    *all_rewards.get_mut(&hit_entity).unwrap() -= 1.0;
                    if health.0 <= 0.0 {
                        commands.entity(hit_entity).despawn_recursive();
                        *all_terminals.get_mut(&hit_entity).unwrap() = true;
                        *all_rewards.get_mut(&agent).unwrap() += 100.0;
                        *all_rewards.get_mut(&hit_entity).unwrap() -= 100.0;
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

        // clamp velocity
        velocity.linvel = velocity.linvel.clamp(
            Vec2::new(-AGENT_MAX_LIN_VEL, -AGENT_MAX_LIN_VEL),
            Vec2::new(AGENT_MAX_LIN_VEL, AGENT_MAX_LIN_VEL),
        );
        velocity.angvel = velocity.angvel.clamp(-AGENT_MAX_ANG_VEL, AGENT_MAX_ANG_VEL);
    }

    for (agent, _, _, _) in agents.iter() {
        brains.get_mut(&agent).unwrap().rb.push(SavedStep {
            state: all_states.remove(&agent).unwrap(),
            action: all_actions.remove(&agent).unwrap().as_vec(),
            reward: all_rewards.remove(&agent).unwrap(),
            terminal: all_terminals.remove(&agent).unwrap(),
        })
    }
}

fn learn(mut commands: Commands, mut brains: NonSendMut<BrainBank>, frame_count: Res<FrameCount>) {
    if frame_count.0 % 5000 == 0 {
        for (ent, brain) in brains.iter_mut() {
            if commands.get_entity(*ent).is_some() {
                brain.learn();
            }
        }
    }
}

fn main() {
    App::new()
        .insert_resource(Msaa::default())
        .add_plugins(DefaultPlugins)
        .add_plugins(RapierPhysicsPlugin::<NoUserData>::pixels_per_meter(100.0))
        .add_plugins(RapierDebugRenderPlugin::default())
        .add_systems(Startup, setup_brains)
        .add_systems(Startup, setup)
        .add_systems(Update, update)
        .add_systems(Update, check_respawn_all)
        .add_systems(Update, learn)
        .run();
}
