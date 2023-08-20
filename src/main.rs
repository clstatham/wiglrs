#![allow(clippy::type_complexity)]

use std::{collections::BTreeMap, f32::consts::PI, sync::atomic::AtomicU64};

use bevy::{math::Vec3Swizzles, prelude::*, sprite::MaterialMesh2dBundle};
use bevy_rapier2d::prelude::*;
use tch::{
    nn::{self},
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

pub const NUM_AGENTS: usize = 5;
pub const AGENT_EMBED_DIM: i64 = 256;

pub const AGENT_RADIUS: f32 = 20.0;
pub const AGENT_MAX_LIN_VEL: f32 = 300.0;
pub const AGENT_MAX_ANG_VEL: f32 = 2.0;
pub const AGENT_LIN_MOVE_FORCE: f32 = 300.0;
pub const AGENT_ANG_MOVE_FORCE: f32 = 1.0;

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

static BRAIN_IDS: AtomicU64 = AtomicU64::new(0);

// #[derive(Component)]
pub struct Brain {
    pub id: u64,
    pub model: Box<dyn Fn(&Tensor) -> (Tensor, Tensor)>,
}

impl Brain {
    pub fn new(p: &nn::Path, state_dim: &[i64], nact: i64, embed_dim: i64) -> Self {
        let id = BRAIN_IDS.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let in_dim = state_dim.iter().product::<i64>();
        let lin_cfg = nn::LinearConfig::default();
        let model = nn::seq()
            .add_fn(|xs| xs.flat_view())
            .add(nn::linear(p / id / "e1", in_dim, embed_dim, lin_cfg))
            .add_fn(|xs| xs.relu());
        let critic = nn::linear(p / id / "c1", embed_dim, 1, lin_cfg);
        let actor = nn::seq()
            .add(nn::linear(p / id / "a1", embed_dim, nact, lin_cfg))
            .add_fn(|xs| xs.tanh());
        let device = p.device();
        Self {
            id,
            model: Box::new(move |xs: &Tensor| {
                let xs = xs.to_device(device).apply(&model);
                (xs.apply(&actor), xs.apply(&critic))
            }),
        }
    }
}

type BrainBank = BTreeMap<Entity, Brain>;

#[derive(Component)]
pub struct Health(pub f32);

#[derive(Event)]
pub struct Shoot {
    pub firing: bool,
    pub shooter: Entity,
    pub damage: f32,
    pub distance: f32,
}

#[derive(Component)]
pub struct Agent;

#[derive(Component)]
pub struct ShootyLine;

#[derive(Bundle)]
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

            health: Health(100.0),
            _a: Agent,
        }
    }
}

fn shoot(
    mut health: Query<&mut Health>,
    agents: Query<(&Transform, &Children), (With<Agent>, Without<ShootyLine>)>,
    mut line_vis: Query<(&mut Visibility, &mut Transform), (With<ShootyLine>, Without<Agent>)>,
    cx: Res<RapierContext>,
    mut ev: EventReader<Shoot>,
) {
    for ev in ev.iter() {
        if ev.firing {
            let (ray_dir, ray_pos) = {
                let (transform, lines) = agents.get(ev.shooter).unwrap();
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
                ev.distance,
                false,
                QueryFilter::default().exclude_collider(ev.shooter),
            ) {
                let (_, lines) = agents.get(ev.shooter).unwrap();
                for line in lines.iter() {
                    let mut t = line_vis.get_mut(*line).unwrap().1;
                    t.scale = Vec3::new(1.0, toi, 1.0);
                    *t = t.with_translation(Vec3::new(0.0, toi / 2.0, 0.0));
                }
                if let Ok(mut health) = health.get_mut(hit_entity) {
                    health.0 -= ev.damage;
                }
            } else {
                let (_, lines) = agents.get(ev.shooter).unwrap();
                for line in lines.iter() {
                    let mut t = line_vis.get_mut(*line).unwrap().1;
                    t.scale = Vec3::new(1.0, ev.distance, 1.0);
                    *t = t.with_translation(Vec3::new(0.0, ev.distance / 2.0, 0.0));
                }
            }
        } else {
            let (_, lines) = agents.get(ev.shooter).unwrap();
            for line in lines.iter() {
                *line_vis.get_mut(*line).unwrap().0 = Visibility::Hidden;
            }
        }
    }
}

fn setup_brains(world: &mut World) {
    let bank = BrainBank::default();
    world.insert_non_send_resource(bank);
    let device = tch::Device::cuda_if_available();
    let vs = nn::VarStore::new(device);
    world.insert_non_send_resource(device);
    world.insert_non_send_resource(vs);
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
                s: ShootyLine,
            })
            .id();
        let id = commands
            .spawn(AgentBundle::new(
                Vec3::new(0.0, 100.0, 0.0),
                None,
                &mut meshes,
                &mut materials,
            ))
            .add_child(shooty)
            .id();
        brains.insert(
            id,
            Brain::new(
                &vs.root(),
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
    mut agents: Query<(Entity, &mut ExternalForce, &mut Velocity, &Transform), With<Agent>>,
    mut shooter: EventWriter<Shoot>,
    brains: NonSend<BrainBank>,
) {
    let mut all_actions = BTreeMap::new();
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
                position: other_transform.translation.xy(),
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
        let (action, _) = (brains[&agent].model)(&tch::Tensor::from_slice(&state).unsqueeze(0));
        let action = Vec::<f32>::try_from(&action.squeeze()).unwrap();
        all_actions.insert(agent, Action::from_slice(&action));
    }
    for (agent, mut force, mut velocity, _) in agents.iter_mut() {
        if all_actions[&agent].shoot {
            shooter.send(Shoot {
                firing: true,
                shooter: agent,
                damage: 1.0,
                distance: 100.0,
            });
        } else {
            shooter.send(Shoot {
                firing: false,
                shooter: agent,
                damage: 0.0,
                distance: 0.0,
            });
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
}

fn main() {
    App::new()
        .insert_resource(Msaa::default())
        .add_plugins(DefaultPlugins)
        .add_plugins(RapierPhysicsPlugin::<NoUserData>::pixels_per_meter(100.0))
        .add_plugins(RapierDebugRenderPlugin::default())
        .add_systems(Startup, setup_brains)
        .add_systems(Startup, setup)
        .add_systems(Update, (update, shoot))
        .add_event::<Shoot>()
        .run();
}
