use crate::brains::thinkers::ppo::{PpoParams, RmsNormalize};
use crate::ui::LogText;
use crate::{
    brains::{
        replay_buffer::{store_sarts, PpoMetadata},
        thinkers::{ppo::PpoThinker, SharedThinker},
        Brain,
    },
    envs::ffa::{check_dead, update},
    names, FrameStack, Timestamp,
};
use bevy::ecs::schedule::SystemConfigs;
use bevy::math::Vec3Swizzles;
use bevy::prelude::*;
use bevy_prng::ChaCha8Rng;
use bevy_rand::prelude::EntropyComponent;
use bevy_rand::resource::GlobalEntropy;
use bevy_rapier2d::prelude::*;
use burn_tch::TchBackend;
use burn_tensor::Tensor;
use itertools::Itertools;
use rand_distr::{Distribution, Uniform};
use serde::{Deserialize, Serialize};

use super::{
    ffa::{
        get_action, learn, send_reward, Agent, AgentBundle, Eyeballs, FfaParams, Health,
        HealthBarBundle, Kills, Name, NameTextBundle, Reward, Terminal,
    },
    modules::{
        map_interaction::MapInteractionProperties, Behavior, CombatBehaviors, CombatProperties,
        IdentityEmbedding, PhysicalBehaviors, PhysicalProperties, Property,
    },
    Action, DefaultFrameStack, Env, Observation, Params,
};

#[derive(Debug, Clone, Copy, Component)]
pub struct AgentId(pub usize);

#[derive(Resource, Debug, Clone, Serialize, Deserialize)]
pub struct BasicParams {
    pub ffa_params: FfaParams,
    pub distance_reward_mult: f32,
}

impl Params for BasicParams {
    fn num_agents(&self) -> usize {
        self.ffa_params.num_agents
    }

    fn agent_frame_stack_len(&self) -> usize {
        self.ffa_params.agent_frame_stack_len
    }

    fn agent_radius(&self) -> f32 {
        self.ffa_params.agent_radius
    }

    fn agent_max_health(&self) -> f32 {
        self.ffa_params.agent_max_health
    }
}

impl PpoParams for BasicParams {
    fn actor_lr(&self) -> f64 {
        self.ffa_params.agent_actor_lr
    }

    fn agent_rb_max_len(&self) -> usize {
        self.ffa_params.agent_rb_max_len
    }

    fn critic_lr(&self) -> f64 {
        self.ffa_params.agent_critic_lr
    }

    fn entropy_beta(&self) -> f32 {
        self.ffa_params.agent_entropy_beta
    }

    fn training_batch_size(&self) -> usize {
        self.ffa_params.agent_training_batch_size
    }

    fn training_epochs(&self) -> usize {
        self.ffa_params.agent_training_epochs
    }

    fn agent_update_interval(&self) -> usize {
        self.ffa_params.agent_update_interval
    }
}

impl Default for BasicParams {
    fn default() -> Self {
        Self {
            ffa_params: FfaParams {
                num_agents: 6,
                ..Default::default()
            },
            distance_reward_mult: 1.0,
        }
    }
}

#[derive(Component, Clone, Default)]
pub struct OtherObs {
    pub identity: IdentityEmbedding,
    pub phys: PhysicalProperties,
    pub map_interaction: MapInteractionProperties,
}

#[derive(Component, Clone, Default)]
pub struct BasicObs {
    pub identity: IdentityEmbedding,
    pub phys: PhysicalProperties,
    pub map_interaction: MapInteractionProperties,
    pub others: Vec<OtherObs>,
}

lazy_static::lazy_static! {
    pub static ref OTHER_OBS_LEN: usize = PhysicalProperties::len() + MapInteractionProperties::len();
    pub static ref BASE_OBS_LEN: usize = PhysicalProperties::len() + MapInteractionProperties::len();
}

impl Observation for BasicObs {
    fn as_slice(&self) -> Box<[f32]> {
        let mut out = self.identity.as_slice().to_vec();
        out.extend_from_slice(&self.phys.as_slice());
        out.extend_from_slice(&self.map_interaction.as_slice());
        for other in self.others.iter() {
            out.extend_from_slice(&other.identity.as_slice());
            out.extend_from_slice(&other.phys.as_slice());
            out.extend_from_slice(&other.map_interaction.as_slice());
        }
        out.into_boxed_slice()
    }
}

impl DefaultFrameStack<Basic> for BasicObs {
    fn default_frame_stack(params: &<Basic as Env>::Params) -> FrameStack<Self> {
        let this = Self {
            identity: IdentityEmbedding::new(0, params.num_agents()),
            phys: Default::default(),
            map_interaction: Default::default(),
            others: vec![Default::default(); params.num_agents() - 1],
        };
        FrameStack(vec![this; params.ffa_params.agent_frame_stack_len].into())
    }
}

#[derive(Component, Default, Clone)]
pub struct BasicAction {
    pub phys: PhysicalBehaviors,
    pub metadata: PpoMetadata,
}

lazy_static::lazy_static! {
    pub static ref ACTION_LEN: usize = PhysicalBehaviors::len();
}

impl Action<Basic> for BasicAction {
    type Metadata = PpoMetadata;

    fn as_slice(&self, params: &<Basic as Env>::Params) -> Box<[f32]> {
        self.phys.as_slice()
    }

    fn from_slice(v: &[f32], metadata: Self::Metadata, params: &<Basic as Env>::Params) -> Self {
        Self {
            phys: PhysicalBehaviors::from_slice(v),
            metadata,
        }
    }

    fn metadata(&self) -> Self::Metadata {
        self.metadata.clone()
    }
}

#[derive(Resource)]
pub struct Basic;

impl Env for Basic {
    type Params = BasicParams;

    type Observation = BasicObs;

    type Action = BasicAction;

    fn init() -> Self {
        Self
    }

    fn setup_system() -> SystemConfigs {
        setup.chain()
    }

    fn observation_system() -> SystemConfigs {
        observation.chain()
    }

    fn action_system() -> SystemConfigs {
        get_action::<Basic, SharedThinker<Basic, PpoThinker>>.chain()
    }

    fn reward_system() -> SystemConfigs {
        (
            get_reward,
            send_reward::<Basic, SharedThinker<Basic, PpoThinker>>,
        )
            .chain()
    }

    fn terminal_system() -> SystemConfigs {
        get_terminal.chain()
    }

    fn update_system() -> SystemConfigs {
        (update::<Basic>, store_sarts::<Basic>, check_dead::<Basic>).chain()
    }

    fn learn_system() -> SystemConfigs {
        learn::<Basic, SharedThinker<Basic, PpoThinker>>.chain()
    }

    fn ui_system() -> SystemConfigs {
        use crate::ui::*;
        (
            action_space::<Basic, SharedThinker<Basic, PpoThinker>>,
            log,
            running_reward,
        )
            .chain()
    }
}

fn setup(
    params: Res<BasicParams>,
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    timestamp: Res<Timestamp>,
    mut rng: ResMut<GlobalEntropy<ChaCha8Rng>>,
) {
    let mut taken_names = vec![];
    let obs_len = *BASE_OBS_LEN
        + params.num_agents()
        + (*OTHER_OBS_LEN + params.num_agents()) * (params.num_agents() - 1);
    let mut rng_comp = EntropyComponent::from(&mut rng);

    for agent_id in 0..params.num_agents() {
        let thinker = SharedThinker::<Basic, _>::new(PpoThinker::new(
            obs_len,
            params.ffa_params.agent_hidden_dim,
            *ACTION_LEN,
            params.ffa_params.agent_training_epochs,
            params.ffa_params.agent_training_batch_size,
            params.ffa_params.agent_entropy_beta,
            params.ffa_params.agent_actor_lr,
            params.ffa_params.agent_critic_lr,
            &mut rng_comp,
        ));
        let mut name = names::random_name(&mut rng_comp);
        while taken_names.contains(&name) {
            name = names::random_name(&mut rng_comp);
        }
        let dist = Uniform::new(-250.0, 250.0);
        let agent_pos = Vec3::new(dist.sample(&mut rng_comp), dist.sample(&mut rng_comp), 0.0);
        let color = if agent_id % 2 == 0 {
            Color::RED
        } else {
            Color::BLUE
        };
        taken_names.push(name.clone());
        let mut agent = AgentBundle::<Basic, _>::new(
            agent_pos,
            Some(color),
            name.clone(),
            Brain::<Basic, _>::new(thinker.clone(), name, timestamp.clone()),
            meshes.reborrow(),
            materials.reborrow(),
            &*params,
            obs_len,
            &mut rng,
        );
        agent.health = Health(params.ffa_params.agent_max_health);

        let id = commands
            .spawn(agent)
            .insert(AgentId(agent_id))
            .with_children(|parent| {
                // parent.spawn(ShootyLineBundle::new(
                //     materials.reborrow(),
                //     meshes.reborrow(),
                // ));
                Eyeballs::spawn(
                    parent,
                    meshes.reborrow(),
                    materials.reborrow(),
                    params.ffa_params.agent_radius,
                );
            })
            .id();
        commands.spawn(NameTextBundle::new(&asset_server, id));
        commands.spawn(HealthBarBundle::new(
            meshes.reborrow(),
            materials.reborrow(),
            id,
        ));
    }
}

fn observation(
    params: Res<BasicParams>,
    cx: Res<RapierContext>,
    mut fs: Query<
        (
            &mut FrameStack<Box<[f32]>>,
            &mut RmsNormalize<TchBackend<f32>, 2>,
        ),
        With<Agent>,
    >,
    queries: Query<(Entity, &AgentId, &Velocity, &Transform), With<Agent>>,
    mut gizmos: Gizmos,
) {
    queries
        .iter()
        .for_each(|(agent, agent_id, velocity, transform)| {
            // draw their vision cones
            let my_pos = transform.translation.xy();
            // let required_angle = if agent_id.0 % 2 == 0 {
            //     45.0f32.to_radians()
            // } else {
            //     80.0f32.to_radians()
            // };
            // let required_angle_positive =
            //     transform.local_y().xy().angle_between(Vec2::Y) + required_angle;
            // let required_angle_negative =
            //     transform.local_y().xy().angle_between(Vec2::Y) - required_angle;

            // let point1 = my_pos
            //     + 100.0 * Vec2::new(required_angle_positive.sin(), required_angle_positive.cos());
            // let point2 = my_pos
            //     + 100.0 * Vec2::new(required_angle_negative.sin(), required_angle_negative.cos());
            // gizmos.line_2d(my_pos, point1, Color::WHITE);
            // gizmos.line_2d(my_pos, point2, Color::WHITE);

            let mut my_state = BasicObs {
                identity: IdentityEmbedding::new(agent_id.0, params.num_agents()),
                phys: PhysicalProperties::new(transform, velocity),
                map_interaction: MapInteractionProperties::new(transform, &cx),
                others: Vec::new(),
            };
            for (other, other_id, other_v, other_t) in queries
                .iter()
                .filter(|a| a.0 != agent)
                .sorted_by_key(|o| o.1 .0)
            {
                let my_forward = transform.local_y().xy();
                let other_loc_relative =
                    (other_t.translation.xy() - transform.translation.xy()).normalize();
                // gizmos.line_2d(my_pos, my_pos + other_loc_relative * 100.0, Color::GREEN);

                // check line of sight
                let filter = QueryFilter::new().exclude_collider(agent);
                // for (ent, ent_id, _, _) in queries.iter() {
                //     if agent_id.0 % 2 == ent_id.0 % 2 {
                //         filter = filter.exclude_collider(ent);
                //     }
                // }
                if let Some((hit_ent, _)) = cx.cast_ray(
                    transform.translation.xy(),
                    (other_t.translation.xy() - transform.translation.xy()).normalize(),
                    Real::MAX,
                    true,
                    filter,
                ) {
                    // is it an agent or a wall?
                    if queries.get(hit_ent).is_ok() {
                        // check if they're in front of us
                        // let angle = my_forward.dot(other_loc_relative).acos();
                        // if angle <= required_angle {
                        // we can see the enemy
                        gizmos.line_2d(
                            my_pos,
                            my_pos + other_loc_relative * 100.0,
                            if other_id.0 % 2 == 0 {
                                Color::RED
                            } else {
                                Color::BLUE
                            },
                        );
                        my_state.others.push(OtherObs {
                            identity: IdentityEmbedding::new(other_id.0, params.num_agents()),
                            phys: PhysicalProperties::new(other_t, other_v),
                            map_interaction: MapInteractionProperties::new(other_t, &cx),
                        });
                        // } else {
                        //     // behind us
                        //     my_state.others.push(OtherObs {
                        //         identity: IdentityEmbedding::new(other_id.0, params.num_agents()),
                        //         ..Default::default()
                        //     });
                        // }
                    } else {
                        // obscured by a wall
                        my_state.others.push(OtherObs {
                            identity: IdentityEmbedding::new(other_id.0, params.num_agents()),
                            ..Default::default()
                        });
                    }
                } else {
                    // ray hit nothing???
                    my_state.others.push(OtherObs {
                        identity: IdentityEmbedding::new(other_id.0, params.num_agents()),
                        ..Default::default()
                    });
                }
            }
            let obs = fs
                .get_mut(agent)
                .unwrap()
                .1
                .forward_obs(Tensor::from_floats(&*my_state.as_slice()).unsqueeze())
                .into_data()
                .value
                .into_boxed_slice();
            fs.get_mut(agent)
                .unwrap()
                .0
                .push(obs, Some(params.agent_frame_stack_len()));
        });
}

fn get_reward(
    params: Res<BasicParams>,
    mut rewards: Query<(&mut Reward, &mut RmsNormalize<TchBackend<f32>, 2>), With<Agent>>,
    actions: Query<&BasicAction, With<Agent>>,
    agents: Query<Entity, With<Agent>>,
    agents_ids: Query<&AgentId, With<Agent>>,
    agent_transform: Query<&Transform, With<Agent>>,
    mut force: Query<&mut ExternalForce, With<Agent>>,
) {
    for agent_ent in agents.iter() {
        let action = actions.get(agent_ent).unwrap();
        let mut my_force = force.get_mut(agent_ent).unwrap();
        my_force.force = action.phys.force.clamp(Vec2::splat(-1.0), Vec2::splat(1.0))
            * params.ffa_params.agent_lin_move_force;
        my_force.torque =
            action.phys.torque.clamp(-1.0, 1.0) * params.ffa_params.agent_ang_move_force;
        let my_id = agents_ids.get(agent_ent).unwrap();
        let my_t = agent_transform.get(agent_ent).unwrap();
        for other_agent in agents.iter().filter(|a| *a != agent_ent) {
            let other_t = agent_transform.get(other_agent).unwrap();
            let other_id = agents_ids.get(other_agent).unwrap();
            let reward =
                my_t.translation.distance(other_t.translation) * params.distance_reward_mult;
            if my_id.0 % 2 == 0 && other_id.0 % 2 == 1 {
                rewards.get_mut(agent_ent).unwrap().0 .0 -= reward;
            } else if my_id.0 % 2 == 1 && other_id.0 % 2 == 0 {
                rewards.get_mut(agent_ent).unwrap().0 .0 += reward;
            }
        }
    }

    for (mut rew, mut norm) in rewards.iter_mut() {
        rew.0 = norm.forward_ret(Tensor::from_floats([rew.0])).into_scalar();
    }
}

fn get_terminal(mut terminals: Query<&mut Terminal, With<Agent>>) {
    for mut t in terminals.iter_mut() {
        t.0 = false;
    }
}
