use bevy::prelude::*;
use bevy_rapier2d::prelude::*;

use crate::brains::replay_buffer::PpoMetadata;

use super::{
    ffa::FfaParams,
    modules::{
        map_interaction::MapInteractionProperties, Behavior, CombatBehaviors, CombatProperties,
        PhysicalBehaviors, PhysicalProperties, Property,
    },
    Action, DefaultFrameStack, Env, Observation, Params,
};

#[derive(Resource, Debug, Clone)]
pub struct TdmParams {
    pub ffa_params: FfaParams,
    pub num_teams: usize,
    pub team_colors: Vec<Color>,
}

impl TdmParams {
    pub fn agents_per_team(&self) -> usize {
        self.ffa_params.num_agents / self.num_teams
    }
}

impl Params for TdmParams {
    fn agent_radius(&self) -> f32 {
        self.ffa_params.agent_radius
    }
}

impl Default for TdmParams {
    fn default() -> Self {
        Self {
            ffa_params: FfaParams {
                num_agents: 4,
                ..Default::default()
            },
            num_teams: 2,
            team_colors: vec![Color::RED, Color::BLUE],
        }
    }
}

#[derive(Component, Clone, Default)]
pub struct TeammateObs {
    pub phys: PhysicalProperties,
    pub combat: CombatProperties,
    pub map_interaction: MapInteractionProperties,
}

#[derive(Component, Clone, Default)]
pub struct EnemyObs {
    pub phys: PhysicalProperties,
    pub combat: CombatProperties,
    pub map_interaction: MapInteractionProperties,
}

#[derive(Component, Clone)]
pub struct TdmObs {
    pub phys: PhysicalProperties,
    pub combat: CombatProperties,
    pub map_interaction: MapInteractionProperties,
    pub teammates: Vec<TeammateObs>,
    pub enemies: Vec<EnemyObs>,
}

impl Observation<Tdm> for TdmObs {
    fn as_slice(&self, params: &<Tdm as Env>::Params) -> Box<[f32]> {
        let mut out = self.phys.as_slice().to_vec();
        out.extend_from_slice(&self.combat.as_slice());
        out.extend_from_slice(&self.map_interaction.as_slice());
        for teammate in self.teammates.iter() {
            out.extend_from_slice(&teammate.phys.as_slice());
            out.extend_from_slice(&teammate.combat.as_slice());
            out.extend_from_slice(&teammate.map_interaction.as_slice());
        }
        for enemy in self.enemies.iter() {
            out.extend_from_slice(&enemy.phys.as_slice());
            out.extend_from_slice(&enemy.combat.as_slice());
            out.extend_from_slice(&enemy.map_interaction.as_slice());
        }
        out.into_boxed_slice()
    }
}

impl DefaultFrameStack<Tdm> for TdmObs {
    fn default_frame_stack(params: &<Tdm as Env>::Params) -> crate::FrameStack<Self> {
        crate::FrameStack(
            vec![
                Self {
                    phys: Default::default(),
                    combat: Default::default(),
                    map_interaction: Default::default(),
                    teammates: vec![Default::default(); params.agents_per_team() - 1],
                    enemies: vec![
                        Default::default();
                        params.agents_per_team() * (params.num_teams - 1)
                    ],
                };
                params.ffa_params.agent_frame_stack_len
            ]
            .into(),
        )
    }
}

#[derive(Component, Default, Clone)]
pub struct TdmAction {
    pub phys: PhysicalBehaviors,
    pub combat: CombatBehaviors,
    pub metadata: PpoMetadata,
}

impl Action<Tdm> for TdmAction {
    type Metadata = PpoMetadata;

    fn as_slice(&self, params: &<Tdm as Env>::Params) -> Box<[f32]> {
        let mut out = vec![];
        out.extend_from_slice(&self.phys.as_slice());
        out.extend_from_slice(&self.combat.as_slice());
        out.into_boxed_slice()
    }

    fn from_slice(v: &[f32], metadata: Self::Metadata, params: &<Tdm as Env>::Params) -> Self {
        Self {
            phys: PhysicalBehaviors::from_slice(&v[0..PhysicalBehaviors::len()]),
            combat: CombatBehaviors::from_slice(&v[PhysicalBehaviors::len()..]),
            metadata,
        }
    }

    fn metadata(&self) -> Self::Metadata {
        self.metadata.clone()
    }
}

#[derive(Resource)]
pub struct Tdm;

impl Env for Tdm {
    type Params = TdmParams;

    type Observation = TdmObs;

    type Action = TdmAction;

    fn init() -> Self {
        Self
    }

    fn setup_system() -> bevy::ecs::schedule::SystemConfigs {
        todo!()
    }

    fn observation_system() -> bevy::ecs::schedule::SystemConfigs {
        todo!()
    }

    fn action_system() -> bevy::ecs::schedule::SystemConfigs {
        todo!()
    }

    fn reward_system() -> bevy::ecs::schedule::SystemConfigs {
        todo!()
    }

    fn terminal_system() -> bevy::ecs::schedule::SystemConfigs {
        todo!()
    }

    fn update_system() -> bevy::ecs::schedule::SystemConfigs {
        todo!()
    }

    fn learn_system() -> bevy::ecs::schedule::SystemConfigs {
        todo!()
    }

    fn ui_system() -> bevy::ecs::schedule::SystemConfigs {
        todo!()
    }
}
