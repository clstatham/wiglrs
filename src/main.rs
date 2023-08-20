#![allow(clippy::type_complexity)]

use bevy::{math::Vec3Swizzles, prelude::*, sprite::MaterialMesh2dBundle};
use bevy_rapier2d::prelude::*;

pub const ACTOR_RADIUS: f32 = 20.0;
pub const ACTOR_MAX_LIN_VEL: f32 = 300.0;
pub const ACTOR_MAX_ANG_VEL: f32 = 2.0;
pub const ACTOR_LIN_MOVE_FORCE: f32 = 300.0;
pub const ACTOR_ANG_MOVE_FORCE: f32 = 1.0;

#[derive(Component)]
pub struct Health(pub f32);

#[derive(Event)]
pub struct TakeDamage(pub Entity, pub f32);

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
            col: Collider::ball(ACTOR_RADIUS),
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
                mesh: meshes.add(shape::Circle::new(ACTOR_RADIUS).into()).into(),
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
                let ray_pos = transform.translation.xy() + ray_dir * (ACTOR_RADIUS + 2.0);
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
                // lines.line_colored(
                //     ray_pos.extend(0.0),
                //     (ray_pos + ray_dir * ev.distance).extend(0.0),
                //     0.1,
                //     Color::RED,
                // );
            }
        } else {
            let (_, lines) = agents.get(ev.shooter).unwrap();
            for line in lines.iter() {
                *line_vis.get_mut(*line).unwrap().0 = Visibility::Hidden;
            }
        }
    }
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
) {
    commands.spawn(Camera2dBundle {
        transform: Transform::from_xyz(0.0, 0.0, 5.0),
        ..Default::default()
    });

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

    commands
        .spawn(AgentBundle::new(
            Vec3::new(0.0, 100.0, 0.0),
            None,
            &mut meshes,
            &mut materials,
        ))
        .add_child(shooty);

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
    mut actors: Query<(Entity, &mut ExternalForce, &mut Velocity), With<Agent>>,
    keys: Res<Input<KeyCode>>,
    mut shooter: EventWriter<Shoot>,
) {
    for (actor, mut force, mut velocity) in actors.iter_mut() {
        if keys.pressed(KeyCode::Space) {
            shooter.send(Shoot {
                firing: true,
                shooter: actor,
                damage: 1.0,
                distance: 100.0,
            });
        } else {
            shooter.send(Shoot {
                firing: false,
                shooter: actor,
                damage: 0.0,
                distance: 0.0,
            });
        }

        force.force = Vec2::new(ACTOR_LIN_MOVE_FORCE, 0.0);
        force.torque = ACTOR_ANG_MOVE_FORCE;
        // clamp velocity
        velocity.linvel = velocity.linvel.clamp(
            Vec2::new(-ACTOR_MAX_LIN_VEL, -ACTOR_MAX_LIN_VEL),
            Vec2::new(ACTOR_MAX_LIN_VEL, ACTOR_MAX_LIN_VEL),
        );
        velocity.angvel = velocity.angvel.clamp(-ACTOR_MAX_ANG_VEL, ACTOR_MAX_ANG_VEL);
    }
}

fn main() {
    App::new()
        .insert_resource(Msaa::default())
        .add_plugins(DefaultPlugins)
        .add_plugins(RapierPhysicsPlugin::<NoUserData>::pixels_per_meter(100.0))
        .add_plugins(RapierDebugRenderPlugin::default())
        .add_systems(Startup, setup)
        .add_systems(Update, (update, shoot))
        .add_event::<TakeDamage>()
        .add_event::<Shoot>()
        .run();
}
