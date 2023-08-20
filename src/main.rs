#![allow(clippy::type_complexity)]

use std::{collections::BTreeMap, f32::consts::PI, sync::atomic::AtomicU64};

use bevy::{math::Vec3Swizzles, prelude::*, sprite::MaterialMesh2dBundle};
use bevy_rapier2d::prelude::*;
use rand_distr::Distribution;
use tch::{
    kind::{FLOAT_CPU, INT64_CPU},
    nn::{self, OptimizerConfig},
    Tensor,
};

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
            self.angle / PI,
            self.angvel / PI,
        ]
    }

    pub fn dim() -> usize {
        Self::default().as_vec().len()
    }
}

pub const NUM_AGENTS: usize = 10;
pub const AGENT_EMBED_DIM: i64 = 256;
pub const AGENT_LR: f64 = 1e-4;
pub const AGENT_OPTIM_EPOCHS: usize = 10;
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
    pub state_value: f32,
    pub action: Vec<f32>,
    pub reward: f32,
    pub terminal: bool,
}

impl SavedStep {
    pub fn unzip(self) -> (Vec<f32>, f32, Vec<f32>, f32, bool) {
        (
            self.state,
            self.state_value,
            self.action,
            self.reward,
            self.terminal,
        )
    }
}

pub fn logprob(mu: &Tensor, var: &Tensor, actions: &Tensor) -> Tensor {
    let p1 = (mu - actions).square() / (2.0 * var);
    let p2 = (2.0 * PI * var).sqrt().log();
    p1 + p2
}

static BRAIN_IDS: AtomicU64 = AtomicU64::new(0);

// #[derive(Component)]
pub struct Brain {
    pub id: u64,
    pub model: Box<dyn Fn(&Tensor) -> (Tensor, Tensor, Tensor)>,
    pub rb: Vec<SavedStep>,
    pub opt: nn::Optimizer,
}

impl Brain {
    pub fn new(vs: &nn::VarStore, state_dim: &[i64], nact: i64, embed_dim: i64) -> Self {
        let id = BRAIN_IDS.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let in_dim = state_dim.iter().product::<i64>();
        let lin_cfg = nn::LinearConfig::default();
        let p = &vs.root();
        let model = nn::seq()
            .add_fn(|xs| xs.flat_view())
            .add(nn::linear(p / id / "e1", in_dim, embed_dim, lin_cfg))
            .add_fn(|xs| xs.relu())
            .add(nn::linear(p / id / "e2", embed_dim, embed_dim, lin_cfg))
            .add_fn(|xs| xs.relu())
            .add(nn::linear(p / id / "e3", embed_dim, embed_dim, lin_cfg))
            .add_fn(|xs| xs.relu());
        let critic = nn::linear(p / id / "c1", embed_dim, 1, lin_cfg);
        let mu = nn::seq()
            .add(nn::linear(p / id / "mu1", embed_dim, nact, lin_cfg))
            .add_fn(|xs| xs.tanh());
        let var = nn::seq()
            .add(nn::linear(p / id / "var1", embed_dim, nact, lin_cfg))
            .add_fn(|xs| xs.softplus());
        let device = p.device();
        Self {
            id,
            model: Box::new(move |xs: &Tensor| {
                let xs = xs.to_device(device).apply(&model);
                (xs.apply(&mu), xs.apply(&var), xs.apply(&critic))
            }),
            rb: Vec::new(),
            opt: nn::Adam::default().build(vs, AGENT_LR).unwrap(),
        }
    }

    pub fn act(&mut self, state: &[f32], grads: bool) -> (Tensor, Tensor) {
        let (mu, var, state_value) = if grads {
            (self.model)(&tch::Tensor::from_slice(state).unsqueeze(0))
        } else {
            tch::no_grad(|| (self.model)(&tch::Tensor::from_slice(state).unsqueeze(0)))
        };

        let mut action = vec![];
        let mu: Vec<f32> = mu.squeeze().try_into().unwrap();
        let var: Vec<f32> = var.squeeze().try_into().unwrap();
        for (mu, var) in mu.into_iter().zip(var.into_iter()) {
            let distr = rand_distr::Normal::<f32>::new(mu, var.sqrt()).unwrap();
            // action.push();
            action.push(distr.sample(&mut rand::thread_rng()));
        }
        let action_t = Tensor::from_slice(&action);

        (action_t.squeeze(), state_value.squeeze())
    }

    pub fn learn(&mut self, device: tch::Device) {
        let nsteps = self.rb.len() as i64;
        if nsteps == 0 {
            return; // sometimes bevy is slow to despawn things
        }
        let (states, state_values, actions, rewards, terminals): (
            Vec<Tensor>,
            Vec<Tensor>,
            Vec<Tensor>,
            Vec<Tensor>,
            Vec<Tensor>,
        ) = itertools::multiunzip(self.rb.drain(..).map(|s| s.unzip()).map(|(s, v, a, r, t)| {
            (
                Tensor::from_slice(&s),
                Tensor::from_slice(&[v]),
                Tensor::from_slice(&a),
                Tensor::from_slice(&[r]),
                Tensor::from_slice(&[if t { 1.0 } else { 0.0 }]),
            )
        }));
        let (states, old_state_value, actions, rewards, terminals) = (
            Tensor::stack(&states, 0).to(device),
            Tensor::stack(&state_values, 0).to(device),
            Tensor::stack(&actions, 0).to(device),
            Tensor::stack(&rewards, 0).to(device),
            Tensor::stack(&terminals, 0).to(device),
        );

        let returns = {
            let r = Tensor::zeros([nsteps, 1], FLOAT_CPU).to(device);
            let mut r_s = Tensor::zeros([1], FLOAT_CPU).to(device);
            for s in (0i64..nsteps).rev() {
                let is_terminal: f32 = terminals.get(s).try_into().unwrap();
                if is_terminal > 0.0 {
                    r_s = r_s.zero_();
                }
                r_s = rewards.get(s) + r_s * 0.99;
                r.get(s).copy_(&r_s);
            }
            r.narrow(0, 0, nsteps).view([nsteps, 1])
        };
        let mut total_loss = 0.0f32;
        for _ in 0..AGENT_OPTIM_EPOCHS {
            let batch_idxs =
                Tensor::randint(nsteps, [AGENT_OPTIM_BATCH_SIZE], INT64_CPU).to(device);
            let states = states.index_select(0, &batch_idxs);
            let actions = actions.index_select(0, &batch_idxs);
            let returns = returns.index_select(0, &batch_idxs);
            let old_state_value = old_state_value.index_select(0, &batch_idxs);

            self.opt.zero_grad();
            let (mu, var, state_value) = (self.model)(&states);
            let value_loss = state_value.mse_loss(&returns, tch::Reduction::Mean);
            let advantage = returns - old_state_value.detach();
            let log_prob = advantage * logprob(&mu, &var, &actions);
            let policy_loss = -log_prob.mean(None);
            let entropy_loss = 1e-4 * ((-(2.0 * PI * var).log() + 1.0) / 2.0).mean(None);
            let loss: Tensor = value_loss + policy_loss + entropy_loss;
            loss.backward();
            self.opt.step();
            let loss: f32 = loss.try_into().unwrap();
            total_loss += loss;
        }
        println!("Total loss: {total_loss}");
    }
}

type BrainBank = BTreeMap<Entity, Brain>;

#[derive(Component)]
pub struct Health(pub f32);

// #[derive(Event)]
// pub struct Shoot {
//     pub firing: bool,
//     pub shooter: Entity,
//     pub damage: f32,
//     pub distance: f32,
// }

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
    let device = tch::Device::cuda_if_available();
    if device.is_cuda() {
        println!("Using CUDA.");
    }
    let vs = nn::VarStore::new(device);
    world.insert_non_send_resource(device);
    world.insert_non_send_resource(vs);
}

fn check_respawn_all(
    mut commands: Commands,
    mut brains: NonSendMut<BrainBank>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
) {
    // if brains.keys().any(|a| commands.get_entity(*a).is_none()) {
    // for brain in brains.values_mut() {
    //     brain.rb.clear();
    // }
    for agent in brains.keys().copied().collect::<Vec<_>>() {
        if commands.get_entity(agent).is_none() {
            // brain transplant
            let mut brain = brains.remove(&agent).unwrap();
            brain.rb.clear();

            let shooty = commands
                .spawn(ShootyLineBundle {
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
                })
                .id();
            let id = commands
                .spawn(AgentBundle::new(
                    Vec3::new(
                        (rand::random::<f32>() - 0.5) * 500.0,
                        (rand::random::<f32>() - 0.5) * 500.0,
                        0.0,
                    ),
                    Some(Color::rgb(rand::random(), rand::random(), rand::random())),
                    &mut meshes,
                    &mut materials,
                ))
                .add_child(shooty)
                .id();
            brains.insert(id, brain);
        }
    }
    // }
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    mut brains: NonSendMut<BrainBank>,
    vs: NonSend<nn::VarStore>,
) {
    commands.spawn(Camera2dBundle {
        transform: Transform::from_xyz(0.0, 0.0, 5.0),
        ..Default::default()
    });

    for _ in 0..NUM_AGENTS {
        let shooty = commands
            .spawn(ShootyLineBundle {
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
            })
            .id();
        let id = commands
            .spawn(AgentBundle::new(
                Vec3::new(
                    (rand::random::<f32>() - 0.5) * 500.0,
                    (rand::random::<f32>() - 0.5) * 500.0,
                    0.0,
                ),
                Some(Color::rgb(rand::random(), rand::random(), rand::random())),
                &mut meshes,
                &mut materials,
            ))
            .add_child(shooty)
            .id();
        brains.insert(
            id,
            Brain::new(
                &vs,
                &[(State::dim() * NUM_AGENTS) as i64],
                Action::dim() as i64,
                AGENT_EMBED_DIM,
            ),
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
    mut agents: Query<(Entity, &mut ExternalForce, &mut Velocity, &Transform), With<Agent>>,
    mut brains: NonSendMut<BrainBank>,
    mut health: Query<&mut Health>,
    agents_shootin: Query<(&Transform, &Children), (With<Agent>, Without<ShootyLine>)>,
    mut line_vis: Query<(&mut Visibility, &mut Transform), (With<ShootyLine>, Without<Agent>)>,
    cx: Res<RapierContext>,
) {
    let mut all_states = BTreeMap::new(); // you're in good hands...?
    let mut all_actions = BTreeMap::new();
    let mut all_state_values: BTreeMap<Entity, f32> = BTreeMap::new();
    let mut all_rewards = BTreeMap::new();
    let mut all_terminals = BTreeMap::new();

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
        let (action, state_value) = brains.get_mut(&agent).unwrap().act(&state, false);

        let action: Vec<f32> = action.try_into().unwrap();
        all_states.insert(agent, state);
        all_actions.insert(agent, Action::from_slice(&action));
        all_rewards.insert(agent, 0.0f32);
        all_state_values.insert(agent, state_value.try_into().unwrap());
        all_terminals.insert(agent, false);
    }
    for (agent, mut force, mut velocity, _) in agents.iter_mut() {
        if all_actions[&agent].shoot {
            let (ray_dir, ray_pos) = {
                let (transform, lines) = agents_shootin.get(agent).unwrap();
                let ray_dir = transform.local_y().xy();
                let ray_pos = transform.translation.xy() + ray_dir * (AGENT_RADIUS + 2.0);
                // println!("Pew pew at {:?} {:?}!", ray_pos, ray_dir);
                for line in lines.iter() {
                    *line_vis.get_mut(*line).unwrap().0 = Visibility::Visible;
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
                let (_, lines) = agents_shootin.get(agent).unwrap();
                for line in lines.iter() {
                    let mut t = line_vis.get_mut(*line).unwrap().1;
                    t.scale = Vec3::new(1.0, toi, 1.0);
                    *t = t.with_translation(Vec3::new(0.0, toi / 2.0, 0.0));
                }
                if let Ok(mut health) = health.get_mut(hit_entity) {
                    health.0 -= 1.0;
                    if health.0 <= 0.0 {
                        commands.entity(hit_entity).despawn_recursive();
                        *all_terminals.get_mut(&hit_entity).unwrap() = true;
                        *all_rewards.get_mut(&agent).unwrap() += 10.0;
                        println!("Kill! Nice!");
                    }
                }
            } else {
                let (_, lines) = agents_shootin.get(agent).unwrap();
                for line in lines.iter() {
                    let mut t = line_vis.get_mut(*line).unwrap().1;
                    t.scale = Vec3::new(1.0, AGENT_SHOOT_DISTANCE, 1.0);
                    *t = t.with_translation(Vec3::new(0.0, AGENT_SHOOT_DISTANCE / 2.0, 0.0));
                }
            }
        } else {
            let (_, lines) = agents_shootin.get(agent).unwrap();
            for line in lines.iter() {
                *line_vis.get_mut(*line).unwrap().0 = Visibility::Hidden;
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
            state_value: all_state_values.remove(&agent).unwrap(),
            action: all_actions.remove(&agent).unwrap().as_vec(),
            reward: all_rewards.remove(&agent).unwrap(),
            terminal: all_terminals.remove(&agent).unwrap(),
        })
    }
}

fn learn(
    mut commands: Commands,
    mut brains: NonSendMut<BrainBank>,
    device: NonSend<tch::Device>,
    time: Res<Time>,
    fixed_time: Res<FixedTime>,
) {
    if time.elapsed_seconds_f64() >= 3.0 {
        for (ent, brain) in brains.iter_mut() {
            if commands.get_entity(*ent).is_some() {
                brain.learn(*device);
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
        .add_systems(FixedUpdate, learn)
        .insert_resource(FixedTime::new_from_secs(1.0))
        .run();
}
