use std::f32::consts::PI;

use crate::Wall;

use super::Map;
use bevy::prelude::*;
use bevy_rapier2d::prelude::*;

#[derive(Resource)]
pub struct TdmMap;

impl Map for TdmMap {
    fn setup_system() -> bevy::ecs::schedule::SystemConfigs {
        fn setup(mut commands: Commands) {
            // bottom wall
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
                .insert(TransformBundle::from(Transform::from_xyz(0.0, -250.0, 0.0)));

            // top wall
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
                .insert(TransformBundle::from(Transform::from_xyz(0.0, 250.0, 0.0)));

            // left wall
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

            // right wall
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

            // right-middle wall
            commands
                .spawn(Collider::cuboid(100.0, 10.0))
                .insert(SpriteBundle {
                    sprite: Sprite {
                        color: Color::BLACK,
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
                        color: Color::BLACK,
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
                        color: Color::BLACK,
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
                        color: Color::BLACK,
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
                        color: Color::BLACK,
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
                        color: Color::BLACK,
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
                        color: Color::BLACK,
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
                        color: Color::BLACK,
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
        }
        setup.chain()
    }
}
