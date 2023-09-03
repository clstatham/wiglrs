use bevy::math::Vec3Swizzles;
use bevy::prelude::*;
use bevy_rapier2d::prelude::{QueryFilter, RapierContext};

use super::Property;

#[derive(Debug, Component, Clone, Copy, Default)]
pub struct MapInteractionProperties {
    pub up_wall_dist: f32,
    pub down_wall_dist: f32,
    pub left_wall_dist: f32,
    pub right_wall_dist: f32,
}

impl MapInteractionProperties {
    pub fn new(transform: &Transform, cx: &RapierContext) -> Self {
        let mut this = Self::default();
        let filter = QueryFilter::only_fixed();
        if let Some((_, toi)) =
            cx.cast_ray(transform.translation.xy(), Vec2::Y, 2000.0, true, filter)
        {
            this.up_wall_dist = toi;
        }
        if let Some((_, toi)) = cx.cast_ray(
            transform.translation.xy(),
            Vec2::NEG_Y,
            2000.0,
            true,
            filter,
        ) {
            this.down_wall_dist = toi;
        }
        if let Some((_, toi)) =
            cx.cast_ray(transform.translation.xy(), Vec2::X, 2000.0, true, filter)
        {
            this.right_wall_dist = toi;
        }
        if let Some((_, toi)) = cx.cast_ray(
            transform.translation.xy(),
            Vec2::NEG_X,
            2000.0,
            true,
            filter,
        ) {
            this.left_wall_dist = toi;
        }
        this
    }
}

impl Property for MapInteractionProperties {
    fn len() -> usize {
        4
    }

    fn as_slice(&self) -> Box<[f32]> {
        Box::new([
            self.up_wall_dist / 2000.0,
            self.down_wall_dist / 2000.0,
            self.left_wall_dist / 2000.0,
            self.right_wall_dist / 2000.0,
        ])
    }

    fn from_slice(s: &[f32]) -> Self {
        Self {
            up_wall_dist: s[0],
            down_wall_dist: s[1],
            left_wall_dist: s[2],
            right_wall_dist: s[3],
        }
    }
}
