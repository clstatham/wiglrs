use std::f32::consts::PI;

use crate::brains::learners::ppo::rollout_buffer::PpoBuffer;
use crate::brains::learners::ppo::Ppo;
use crate::brains::learners::utils::RmsNormalize;
use crate::brains::learners::DEVICE;
use crate::brains::models::linear_resnet::{LinResActor, LinResCritic};
use crate::brains::{AgentLearner, AgentPolicy, AgentValue};
use crate::ui::LogText;
use crate::{
    brains::learners::ppo::rollout_buffer::{store_sarts, PpoMetadata},
    envs::{check_dead, update},
    names, FrameStack, Timestamp,
};
use bevy::ecs::schedule::SystemConfigs;
use bevy::math::Vec3Swizzles;
use bevy::prelude::*;
use bevy_prng::ChaCha8Rng;
use bevy_rand::prelude::EntropyComponent;
use bevy_rand::resource::GlobalEntropy;
use bevy_rapier2d::prelude::*;
use candle_core::Tensor;
use itertools::Itertools;
use rand_distr::{Distribution, Uniform};
use serde::{Deserialize, Serialize};

use super::modules::RelativePhysicalProperties;
use super::AgentId;
use super::{
    modules::{
        map_interaction::MapInteractionProperties, Behavior, CombatBehaviors, CombatProperties,
        PhysicalBehaviors, PhysicalProperties, Property,
    },
    Action, DefaultFrameStack, Env, Observation, Params,
    {
        get_action, learn, send_reward, Agent, AgentBundle, Eyeballs, FfaParams, Health,
        HealthBarBundle, Kills, Name, NameTextBundle, Reward, Terminal,
    },
};

#[derive(Resource, Debug, Clone, Serialize, Deserialize)]
pub struct TdmParams {
    pub ffa_params: FfaParams,
    pub reward_for_friendly_fire: f32,
    pub num_teams: usize,
    pub team_colors: Vec<Color>,
}

impl TdmParams {
    pub fn agents_per_team(&self) -> usize {
        self.ffa_params.num_agents / self.num_teams
    }
}

impl Params for TdmParams {
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

    fn agent_warmup(&self) -> usize {
        self.ffa_params.agent_warmup
    }

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

impl Default for TdmParams {
    fn default() -> Self {
        Self {
            ffa_params: FfaParams {
                num_agents: 6,
                ..Default::default()
            },
            reward_for_friendly_fire: -10.0,
            num_teams: 3,
            team_colors: vec![
                Color::RED,
                Color::BLUE,
                Color::GREEN,
                Color::YELLOW,
                Color::FUCHSIA,
                Color::CYAN,
            ],
        }
    }
}

#[derive(Component, Clone, Default)]
pub struct TeammateObs {
    pub phys: RelativePhysicalProperties,
    pub combat: CombatProperties,
    pub map_interaction: MapInteractionProperties,
    pub firing: bool,
}

#[derive(Component, Clone, Default)]
pub struct EnemyObs {
    pub phys: RelativePhysicalProperties,
    pub combat: CombatProperties,
    pub map_interaction: MapInteractionProperties,
    pub firing: bool,
}

#[derive(Component, Clone)]
pub struct TdmObs {
    pub phys: PhysicalProperties,
    pub combat: CombatProperties,
    pub map_interaction: MapInteractionProperties,
    pub teammates: Vec<TeammateObs>,
    pub enemies: Vec<EnemyObs>,
}

lazy_static::lazy_static! {
    pub static ref TEAMMATE_OBS_LEN: usize = RelativePhysicalProperties::len() + CombatProperties::len() + MapInteractionProperties::len() + 1;
    pub static ref ENEMY_OBS_LEN: usize = RelativePhysicalProperties::len() + CombatProperties::len() + MapInteractionProperties::len() + 1;
    pub static ref BASE_OBS_LEN: usize = PhysicalProperties::len() + CombatProperties::len() + MapInteractionProperties::len();
}

impl Observation for TdmObs {
    fn as_slice(&self) -> Box<[f32]> {
        let mut out = self.phys.as_slice().to_vec();
        out.extend_from_slice(&self.combat.as_slice());
        out.extend_from_slice(&self.map_interaction.as_slice());
        for teammate in self.teammates.iter() {
            out.extend_from_slice(&teammate.phys.as_slice());
            out.extend_from_slice(&teammate.combat.as_slice());
            out.extend_from_slice(&teammate.map_interaction.as_slice());
            out.push(if teammate.firing { 1.0 } else { 0.0 });
        }
        for enemy in self.enemies.iter() {
            out.extend_from_slice(&enemy.phys.as_slice());
            out.extend_from_slice(&enemy.combat.as_slice());
            out.extend_from_slice(&enemy.map_interaction.as_slice());
            out.push(if enemy.firing { 1.0 } else { 0.0 });
        }
        out.into_boxed_slice()
    }
}

impl DefaultFrameStack<Tdm> for TdmObs {
    fn default_frame_stack(params: &TdmParams) -> crate::FrameStack<Self> {
        let this = Self {
            phys: Default::default(),
            combat: Default::default(),
            map_interaction: Default::default(),
            teammates: vec![Default::default(); params.agents_per_team() - 1],
            enemies: vec![Default::default(); params.agents_per_team() * (params.num_teams - 1)],
        };
        crate::FrameStack(vec![this; params.ffa_params.agent_frame_stack_len].into())
    }
}

#[derive(Component, Default, Clone)]
pub struct TdmAction
where
    Self: Action<Tdm>,
{
    pub phys: PhysicalBehaviors,
    pub combat: CombatBehaviors,
    pub logits: Option<<Self as Action<Tdm>>::Logits>,
}

lazy_static::lazy_static! {
    pub static ref ACTION_LEN: usize = PhysicalBehaviors::len() + CombatBehaviors::len();
}

impl Action<Tdm> for TdmAction {
    type Logits = (Tensor, Tensor);
    fn as_slice(&self) -> Box<[f32]> {
        let mut out = vec![];
        out.extend_from_slice(&self.phys.as_slice());
        out.extend_from_slice(&self.combat.as_slice());
        out.into_boxed_slice()
    }

    fn from_slice(v: &[f32], logits: Self::Logits) -> Self {
        Self {
            phys: PhysicalBehaviors::from_slice(&v[0..PhysicalBehaviors::len()]),
            combat: CombatBehaviors::from_slice(&v[PhysicalBehaviors::len()..]),
            logits: Some(logits),
        }
    }

    fn logits(&self) -> Option<&Self::Logits> {
        self.logits.as_ref()
    }
}

#[derive(Component)]
pub struct TeamId(pub i32);

#[derive(Resource)]
pub struct Tdm;

impl Env for Tdm {
    type Params = TdmParams;

    type Observation = TdmObs;

    type Action = TdmAction;

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
        get_action::<Tdm, AgentPolicy, AgentValue>.chain()
    }

    fn reward_system() -> SystemConfigs {
        (get_reward, send_reward::<Tdm, AgentLearner>).chain()
    }

    fn terminal_system() -> SystemConfigs {
        get_terminal.chain()
    }

    fn update_system() -> SystemConfigs {
        (
            update::<Tdm>,
            store_sarts::<Tdm, AgentPolicy, AgentValue>,
            check_dead::<Tdm>,
        )
            .chain()
    }

    fn learn_system() -> SystemConfigs {
        learn::<Tdm, AgentLearner, AgentPolicy, AgentValue>.chain()
    }

    fn ui_system() -> SystemConfigs {
        use crate::ui::*;
        (
            kdr::<Tdm, AgentLearner>,
            action_space::<Tdm>,
            log,
            running_reward,
        )
            .chain()
    }
}

fn setup(
    params: Res<TdmParams>,
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    timestamp: Res<Timestamp>,
    mut rng: ResMut<GlobalEntropy<ChaCha8Rng>>,
) {
    let mut taken_names = vec![];
    let obs_len = *BASE_OBS_LEN
        + *TEAMMATE_OBS_LEN * (params.agents_per_team() - 1)
        + *ENEMY_OBS_LEN * params.agents_per_team() * (params.num_teams - 1);
    let mut agent_id = 0;
    for team_id in 0..params.num_teams {
        for _ in 0..params.agents_per_team() {
            let mut rng_comp = EntropyComponent::from(&mut rng);
            let learner = AgentLearner::new(
                params.ffa_params.agent_training_epochs,
                params.ffa_params.agent_training_batch_size,
                params.ffa_params.agent_entropy_beta,
            )
            .unwrap();
            let mut name = names::random_name(&mut rng_comp);
            while taken_names.contains(&name) {
                name = names::random_name(&mut rng_comp);
            }
            let dist = Uniform::new(-250.0, 250.0);
            let agent_pos = Vec3::new(dist.sample(&mut rng_comp), dist.sample(&mut rng_comp), 0.0);
            taken_names.push(name.clone());
            let mut agent = AgentBundle::<Tdm, AgentPolicy, AgentValue, AgentLearner>::new(
                agent_pos,
                Some(params.team_colors[team_id]),
                name.clone(),
                AgentPolicy::new(
                    obs_len * params.agent_frame_stack_len(),
                    params.ffa_params.agent_hidden_dim,
                    *ACTION_LEN,
                    params.ffa_params.agent_actor_lr,
                )
                .unwrap(),
                AgentValue::new(
                    obs_len * params.agent_frame_stack_len(),
                    params.ffa_params.agent_hidden_dim,
                    params.ffa_params.agent_critic_lr,
                )
                .unwrap(),
                learner,
                PpoBuffer::new(Some(params.ffa_params.agent_rb_max_len)),
                &timestamp,
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
                .insert(TeamId(team_id as i32))
                .with_children(|parent| {
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

            agent_id += 1;
        }
    }
}

fn observation(
    params: Res<TdmParams>,
    cx: Res<RapierContext>,
    mut fs: Query<(&mut FrameStack<Box<[f32]>>, &mut RmsNormalize), With<Agent>>,
    queries: Query<
        (
            Entity,
            &AgentId,
            &TeamId,
            &TdmAction,
            &Velocity,
            &Transform,
            &Health,
        ),
        With<Agent>,
    >,
    mut gizmos: Gizmos,
) {
    queries.iter().for_each(
        |(agent, _agent_id, my_team, _action, velocity, transform, health)| {
            let mut my_state = TdmObs {
                phys: PhysicalProperties::new(transform, velocity), //.scaled_by(&phys_scaling),
                combat: CombatProperties {
                    health: health.0, // / params.ffa_params.agent_max_health,
                },
                map_interaction: MapInteractionProperties::new(transform, &cx),
                // .scaled_by(&map_scaling),
                teammates: vec![],
                enemies: vec![],
            };
            for (
                _other,
                _other_id,
                other_team,
                other_action,
                other_vel,
                other_transform,
                other_health,
            ) in queries
                .iter()
                .filter(|q| q.0 != agent)
                .sorted_by_key(|(_, id, _, _, _, _t, _)| id.0)
            {
                if my_team.0 == other_team.0 {
                    my_state.teammates.push(TeammateObs {
                        phys: RelativePhysicalProperties::new(transform, other_transform),
                        combat: CombatProperties {
                            health: other_health.0,
                        },
                        map_interaction: MapInteractionProperties::new(other_transform, &cx),
                        firing: other_action.combat.shoot > 0.0,
                    });
                } else {
                    my_state.enemies.push(EnemyObs {
                        phys: RelativePhysicalProperties::new(transform, other_transform),
                        combat: CombatProperties {
                            health: other_health.0,
                        },
                        map_interaction: MapInteractionProperties::new(other_transform, &cx),
                        firing: other_action.combat.shoot > 0.0,
                    });
                }
            }

            let obs = fs
                .get_mut(agent)
                .unwrap()
                .1
                .forward_obs(&Tensor::new(&*my_state.as_slice(), &DEVICE).unwrap())
                .unwrap()
                .to_vec1()
                .unwrap()
                .into_boxed_slice();
            fs.get_mut(agent)
                .unwrap()
                .0
                .push(obs, Some(params.agent_frame_stack_len()));
        },
    );
}

fn get_reward(
    params: Res<TdmParams>,
    mut rewards: Query<(&mut Reward, &mut RmsNormalize), With<Agent>>,
    actions: Query<&TdmAction, With<Agent>>,
    cx: Res<RapierContext>,
    agents: Query<Entity, With<Agent>>,
    agent_transform: Query<&Transform, With<Agent>>,
    team_id: Query<&TeamId, With<Agent>>,
    mut health: Query<&mut Health, With<Agent>>,
    mut kills: Query<&mut Kills, With<Agent>>,
    mut force: Query<&mut ExternalImpulse, With<Agent>>,
    names: Query<&Name, With<Agent>>,
    mut log: ResMut<LogText>,
    mut gizmos: Gizmos,
) {
    macro_rules! reward_team {
        ($team:expr, $reward:expr) => {
            for agent in agents.iter() {
                if team_id.get(agent).unwrap().0 == $team {
                    rewards.get_component_mut::<Reward>(agent).unwrap().0 += $reward;
                }
            }
        };
    }
    for agent_ent in agents.iter() {
        let my_health = health.get(agent_ent).unwrap();
        let my_t = agent_transform.get(agent_ent).unwrap();

        if let Ok(action) = actions.get(agent_ent) {
            let mut my_force = force.get_mut(agent_ent).unwrap();
            let x_direction = (action.phys.direction * PI - PI / 2.0).sin();
            let y_direction = (action.phys.direction * PI - PI / 2.0).cos();
            let force = Vec2::new(x_direction, y_direction) * action.phys.thrust.clamp(-1.0, 1.0);
            my_force.impulse = force * params.ffa_params.agent_lin_move_force;
            // let direction_normalized = action.phys.desired_direction.try_normalize().unwrap();

            // let desired_angle = f32::atan2(direction_normalized.y, direction_normalized.x);
            let desired_angle = action.phys.desired_rotation * PI;
            let current_angle = my_t.rotation.to_euler(EulerRot::XYZ).2 + PI / 2.0;
            // my_t.rotation = Quat::from_euler(EulerRot::XYZ, 0.0, 0.0, desired_angle - PI / 2.0);
            let mut delta = desired_angle - current_angle;
            if delta > PI {
                delta -= 2.0 * PI;
            }
            if delta < -PI {
                delta += 2.0 * PI;
            }
            my_force.torque_impulse = delta * params.ffa_params.agent_ang_move_force;
            /*
                       if action.combat.shoot > 0.0 && my_health.0 > 0.0 {
                           let my_team = team_id.get(agent_ent).unwrap();
                           let (ray_dir, ray_pos) = {
                               let ray_dir = my_t.local_y().xy();
                               let ray_pos = my_t.translation.xy();
                               (ray_dir, ray_pos)
                           };

                           let filter = QueryFilter::default().exclude_collider(agent_ent);

                           if let Some((hit_entity, toi)) = cx.cast_ray(
                               ray_pos,
                               ray_dir,
                               params.ffa_params.agent_shoot_distance,
                               true,
                               filter,
                           ) {
                               gizmos.line_2d(
                                   ray_pos,
                                   ray_pos + ray_dir * toi,
                                   params.team_colors[team_id.get(agent_ent).unwrap().0 as usize],
                               );

                               if let Ok(mut health) = health.get_mut(hit_entity) {
                                   if let Ok(other_team) = team_id.get(hit_entity) {
                                       if my_team.0 == other_team.0 {
                                           // friendly fire!
                                           rewards.get_component_mut::<Reward>(agent_ent).unwrap().0 +=
                                               params.reward_for_friendly_fire;
                                           // kills.get_mut(agent_ent).unwrap().0 -= 1;
                                       } else if health.0 > 0.0 {
                                           health.0 -= 5.0;
                                           rewards.get_component_mut::<Reward>(agent_ent).unwrap().0 +=
                                               params.ffa_params.reward_for_hit;
                                           rewards.get_component_mut::<Reward>(hit_entity).unwrap().0 +=
                                               params.ffa_params.reward_for_getting_hit;
                                           // kills.get_mut(agent_ent).unwrap().0 += 1;
                                           if health.0 <= 0.0 {
                                               rewards.get_mut(agent_ent).unwrap().0 .0 +=
                                                   params.ffa_params.reward_for_kill;
                                               rewards.get_mut(hit_entity).unwrap().0 .0 +=
                                                   params.ffa_params.reward_for_death;

                                               // reward_team!(my_team.0, params.ffa_params.reward_for_kill);
                                               // reward_team!(other_team.0, params.ffa_params.reward_for_death);
                                               kills.get_mut(agent_ent).unwrap().0 += 1;
                                               let msg = format!(
                                                   "{} killed {}! Nice!",
                                                   &names.get(agent_ent).unwrap().0,
                                                   &names.get(hit_entity).unwrap().0
                                               );
                                               log.push(msg);
                                           }
                                       }
                                   }
                               } else {
                                   // hit a wall
                                   rewards.get_component_mut::<Reward>(agent_ent).unwrap().0 +=
                                       params.ffa_params.reward_for_miss;
                               }
                           } else {
                               gizmos.line_2d(
                                   ray_pos,
                                   ray_pos + ray_dir * params.ffa_params.agent_shoot_distance,
                                   params.team_colors[team_id.get(agent_ent).unwrap().0 as usize],
                               );
                               // hit nothing
                               rewards.get_component_mut::<Reward>(agent_ent).unwrap().0 +=
                                   params.ffa_params.reward_for_miss;
                           }
                       }
            */
            for other in agents.iter() {
                let my_t = agent_transform.get(agent_ent).unwrap();
                let other_t = agent_transform.get(other).unwrap();
                if other_t.translation != my_t.translation {
                    let dist = other_t.translation.distance_squared(my_t.translation);
                    let dot = my_t
                        .local_y()
                        .xy()
                        .dot((other_t.translation.xy() - my_t.translation.xy()).normalize());
                    if action.combat.shoot > 0.0 {
                        if dot > 0.9 {
                            rewards.get_mut(agent_ent).unwrap().0 .0 += dot;
                            rewards.get_mut(other).unwrap().0 .0 -= dot;

                            gizmos.line_2d(
                                my_t.translation.xy(),
                                my_t.translation.xy()
                                    // + (other_t.translation.xy() - my_t.translation.xy())
                                        // .normalize()
                                        + my_t.local_y().xy()
                                        * dot
                                        * dist.sqrt(),
                                params.team_colors[team_id.get(agent_ent).unwrap().0 as usize],
                                // Color::WHITE,
                            );
                        } else {
                            rewards.get_mut(agent_ent).unwrap().0 .0 -= 1.0;
                        }
                    }
                }
            }
        }
    }

    for (mut rew, mut norm) in rewards.iter_mut() {
        rew.0 = norm.forward_ret(rew.0);
    }
}

fn get_terminal(
    mut terminals: Query<&mut Terminal, With<Agent>>,
    agents: Query<Entity, With<Agent>>,
    health: Query<&Health, With<Agent>>,
) {
    for agent_ent in agents.iter() {
        terminals.get_mut(agent_ent).unwrap().0 = health.get(agent_ent).unwrap().0 <= 0.0;
    }
}
