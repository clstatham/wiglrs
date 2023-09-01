use bevy::math::Vec3Swizzles;
use bevy::prelude::*;
use bevy_rapier2d::prelude::Velocity;

pub mod map_interaction;

/// Like Observation, but independent of a specific Env
pub trait Property: std::fmt::Debug + Clone + Default {
    fn as_slice(&self) -> Box<[f32]>;
    fn from_slice(s: &[f32]) -> Self;
    fn scaled_by(&self, scaling: &Self) -> Self;
    fn len() -> usize;
}

/// Like Action, but independent of a specific Env
pub trait Behavior: std::fmt::Debug + Clone + Default {
    fn as_slice(&self) -> Box<[f32]>;
    fn from_slice(s: &[f32]) -> Self;
    fn scaled_by(&self, scaling: &Self) -> Self;
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

    fn scaled_by(&self, _scaling: &Self) -> Self {
        unimplemented!()
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct PhysicalProperties {
    pub position: Vec2,
    pub direction: Vec2,
    pub linvel: Vec2,
}

impl PhysicalProperties {
    pub fn new(transform: &Transform, velocity: &Velocity) -> Self {
        Self {
            position: transform.translation.xy(),
            direction: transform.local_y().xy(),
            linvel: velocity.linvel,
        }
    }
}

impl Property for PhysicalProperties {
    fn len() -> usize {
        6
    }

    fn as_slice(&self) -> Box<[f32]> {
        Box::new([
            self.position.x,
            self.position.y,
            self.direction.x,
            self.direction.y,
            self.linvel.x,
            self.linvel.y,
        ])
    }

    fn from_slice(s: &[f32]) -> Self {
        Self {
            position: Vec2::new(s[0], s[1]),
            direction: Vec2::new(s[2], s[3]),
            linvel: Vec2::new(s[4], s[5]),
        }
    }

    fn scaled_by(&self, scaling: &Self) -> Self {
        Self {
            position: self.position * scaling.position,
            direction: self.direction * scaling.direction,
            linvel: self.linvel * scaling.linvel,
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

    fn scaled_by(&self, scaling: &Self) -> Self {
        Self {
            health: self.health * scaling.health,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct PhysicalBehaviors {
    pub force: Vec2,
    pub torque: f32,
}

impl Behavior for PhysicalBehaviors {
    fn len() -> usize {
        3
    }

    fn as_slice(&self) -> Box<[f32]> {
        Box::new([self.force.x, self.force.y, self.torque])
    }

    fn from_slice(s: &[f32]) -> Self {
        Self {
            force: Vec2::new(s[0], s[1]),
            torque: s[2],
        }
    }
    fn scaled_by(&self, scaling: &Self) -> Self {
        Self {
            force: self.force * scaling.force,
            torque: self.torque * scaling.torque,
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

    fn scaled_by(&self, _scaling: &Self) -> Self {
        unimplemented!()
    }
}
