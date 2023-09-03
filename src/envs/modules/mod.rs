use bevy::math::Vec3Swizzles;
use bevy::prelude::*;
use bevy_rapier2d::prelude::Velocity;

use crate::transform_angle_for_agent;

pub mod map_interaction;

/// Like Observation, but independent of a specific Env
pub trait Property: std::fmt::Debug + Clone + Default {
    fn as_slice(&self) -> Box<[f32]>;
    fn from_slice(s: &[f32]) -> Self;
    fn len() -> usize;
}

/// Like Action, but independent of a specific Env
pub trait Behavior: std::fmt::Debug + Clone + Default {
    fn as_slice(&self) -> Box<[f32]>;
    fn from_slice(s: &[f32]) -> Self;
    fn len() -> usize;
}

#[derive(Debug, Clone, Default)]
pub struct IdentityEmbedding {
    pub embed: Box<[f32]>,
}

impl IdentityEmbedding {
    pub fn new(id: usize, max_id: usize) -> Self {
        let mut embed = vec![0.0; max_id];
        embed[id] = 1.0;
        Self {
            embed: embed.into_boxed_slice(),
        }
    }
}

impl Property for IdentityEmbedding {
    fn as_slice(&self) -> Box<[f32]> {
        self.embed.clone()
    }

    fn from_slice(s: &[f32]) -> Self {
        Self { embed: s.into() }
    }

    fn len() -> usize {
        unimplemented!()
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct RelativePhysicalProperties {
    pub angle_direction: f32,
    pub distance: f32,
    pub direction_dot_product: f32,
}

impl RelativePhysicalProperties {
    pub fn new(pov: &Transform, relative_to: &Transform) -> Self {
        let relative_direction = (relative_to.translation.xy() - pov.translation.xy())
            .try_normalize()
            .unwrap();
        Self {
            angle_direction: transform_angle_for_agent(f32::atan2(
                relative_direction.y,
                relative_direction.x,
            )),
            distance: pov.translation.xy().distance(relative_to.translation.xy()),
            direction_dot_product: pov.local_y().xy().dot(relative_direction),
        }
    }
}

impl Property for RelativePhysicalProperties {
    fn len() -> usize {
        3
    }

    fn as_slice(&self) -> Box<[f32]> {
        Box::new([
            self.angle_direction,
            self.distance,
            self.direction_dot_product,
        ])
    }

    fn from_slice(s: &[f32]) -> Self {
        Self {
            angle_direction: s[0],
            distance: s[1],
            direction_dot_product: s[2],
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct PhysicalProperties {
    pub position: Vec2,
    pub angle: f32,
    pub linvel: Vec2,
}

impl PhysicalProperties {
    pub fn new(transform: &Transform, velocity: &Velocity) -> Self {
        Self {
            position: transform.translation.xy(),
            angle: transform_angle_for_agent(transform.rotation.to_euler(EulerRot::XYZ).2),
            linvel: velocity.linvel,
        }
    }
}

impl Property for PhysicalProperties {
    fn len() -> usize {
        5
    }

    fn as_slice(&self) -> Box<[f32]> {
        Box::new([
            self.position.x,
            self.position.y,
            self.angle,
            self.linvel.x,
            self.linvel.y,
        ])
    }

    fn from_slice(s: &[f32]) -> Self {
        Self {
            position: Vec2::new(s[0], s[1]),
            angle: s[2],
            linvel: Vec2::new(s[3], s[4]),
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct CombatProperties {
    pub health: f32,
}

impl Property for CombatProperties {
    fn len() -> usize {
        1
    }

    fn as_slice(&self) -> Box<[f32]> {
        Box::new([self.health])
    }

    fn from_slice(s: &[f32]) -> Self {
        Self { health: s[0] }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct PhysicalBehaviors {
    pub direction: f32,
    pub thrust: f32,
    pub desired_rotation: f32,
}

impl Behavior for PhysicalBehaviors {
    fn len() -> usize {
        3
    }

    fn as_slice(&self) -> Box<[f32]> {
        Box::new([self.direction, self.thrust, self.desired_rotation])
    }

    fn from_slice(s: &[f32]) -> Self {
        Self {
            direction: s[0],
            thrust: s[1],
            desired_rotation: s[2],
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct CombatBehaviors {
    pub shoot: f32,
}

impl Behavior for CombatBehaviors {
    fn len() -> usize {
        1
    }

    fn as_slice(&self) -> Box<[f32]> {
        Box::new([self.shoot])
    }

    fn from_slice(s: &[f32]) -> Self {
        Self { shoot: s[0] }
    }
}
