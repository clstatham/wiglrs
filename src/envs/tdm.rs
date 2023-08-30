use bevy::{core::FrameCount, math::Vec3Swizzles, prelude::*};
use bevy_egui::{
    egui::{
        self,
        plot::{Bar, BarChart, Line},
        Color32, Layout, Stroke,
    },
    EguiContexts,
};
use bevy_rapier2d::prelude::*;
use bevy_tasks::AsyncComputeTaskPool;
use itertools::Itertools;
use std::collections::BTreeMap;

use crate::{
    brains::{
        replay_buffer::{PpoBuffer, Sart},
        thinkers::{ppo::PpoThinker, SharedThinker},
        AgentThinker, Brain, BrainBank,
    },
    ui::LogText,
    FrameStack, Timestamp,
};

use super::{
    ffa::{
        self, Agent, BrainId, Deaths, Eyeballs, Health, HealthBar, HealthBarBundle, Kills, Name,
        NameText, NameTextBundle, ShootyLine, ShootyLineBundle,
    },
    maps::Map,
    Action, DefaultFrameStack, Env, Observation,
};

#[derive(Debug, Resource, Clone)]
pub struct TdmParams {
    pub ffa_params: ffa::FfaParams,
    pub num_teams: usize,
    pub team_colors: Vec<Color>,
}

impl Default for TdmParams {
    fn default() -> Self {
        Self {
            ffa_params: ffa::FfaParams {
                num_agents: 4,
                ..Default::default()
            },
            num_teams: 2,
            team_colors: vec![Color::RED, Color::BLUE],
        }
    }
}

impl TdmParams {
    pub fn agents_per_team(&self) -> usize {
        self.ffa_params.num_agents / self.num_teams
    }
}

#[derive(Default, Debug, Clone, Copy)]
pub struct TeammateState {
    pub ffa_state: ffa::OtherState,
}

pub const TEAMMATE_STATE_LEN: usize = ffa::OTHER_STATE_LEN;

#[derive(Default, Debug, Clone, Copy)]
pub struct EnemyState {
    pub ffa_state: ffa::OtherState,
}

pub const ENEMY_STATE_LEN: usize = ffa::OTHER_STATE_LEN;

#[derive(Debug, Clone)]
pub struct TdmObs {
    pub ffa_state: ffa::FfaObs,
    pub teammate_states: Vec<TeammateState>,
    pub enemy_states: Vec<EnemyState>,
}

pub const BASE_STATE_LEN: usize = ffa::BASE_STATE_LEN;

impl Observation<Tdm> for TdmObs {
    fn as_slice(&self, params: &<Tdm as Env>::Params) -> Box<[f32]> {
        let mut out = self.ffa_state.as_slice(&params.ffa_params).to_vec();
        for other in &self.teammate_states {
            out.extend_from_slice(&other.ffa_state.as_slice(&params.ffa_params));
        }
        for other in &self.enemy_states {
            out.extend_from_slice(&other.ffa_state.as_slice(&params.ffa_params));
        }

        out.into_boxed_slice()
    }
}

impl DefaultFrameStack<Tdm> for TdmObs {
    fn default_frame_stack(params: &<Tdm as Env>::Params) -> crate::FrameStack<Self> {
        let ffa_fs = ffa::FfaObs::default_frame_stack(&params.ffa_params);
        let mut out = vec![];
        for mut ffa_fs in ffa_fs.as_vec() {
            ffa_fs.other_states.clear();
            out.push(Self {
                ffa_state: ffa_fs,
                teammate_states: vec![TeammateState::default(); params.agents_per_team() - 1],
                enemy_states: vec![
                    EnemyState::default();
                    params.agents_per_team() * (params.num_teams - 1)
                ],
            })
        }
        FrameStack(out.into())
    }
}

#[derive(Debug, Clone, Default)]
pub struct TdmAction {
    pub ffa_action: ffa::FfaAction,
}

pub const ACTION_LEN: usize = ffa::ACTION_LEN;

impl Action<Tdm> for TdmAction {
    type Metadata = <ffa::FfaAction as Action<ffa::Ffa>>::Metadata;

    fn from_slice(v: &[f32], metadata: Self::Metadata, params: &<Tdm as Env>::Params) -> Self {
        Self {
            ffa_action: ffa::FfaAction::from_slice(v, metadata, &params.ffa_params),
        }
    }

    fn as_slice(&self, params: &<Tdm as Env>::Params) -> Box<[f32]> {
        self.ffa_action.as_slice(&params.ffa_params)
    }

    fn metadata(&self) -> Self::Metadata {
        self.ffa_action.metadata()
    }
}

#[derive(Component)]
pub struct TeamId(pub usize);

#[derive(Bundle)]
pub struct TdmAgentBundle {
    pub ffa_agent: ffa::FfaAgentBundle,
    pub team_id: TeamId,
}

impl TdmAgentBundle {
    pub fn new(
        pos: Vec3,
        team_id: usize,
        name: String,
        brain_id: usize,
        meshes: &mut ResMut<Assets<Mesh>>,
        materials: &mut ResMut<Assets<ColorMaterial>>,
        params: &TdmParams,
    ) -> Self {
        let color = params.team_colors[team_id];
        Self {
            team_id: TeamId(team_id),
            ffa_agent: ffa::FfaAgentBundle::new(
                pos,
                Some(color),
                name,
                brain_id,
                meshes,
                materials,
                &params.ffa_params,
            ),
        }
    }
}

#[derive(Resource, Default)]
pub struct Tdm {
    pub params: TdmParams,
    pub brains: BrainBank<Tdm, SharedThinker<Tdm, AgentThinker>>,
    pub rbs: BTreeMap<Entity, PpoBuffer<Tdm>>,
    pub observations: BTreeMap<Entity, FrameStack<TdmObs>>,
    pub actions: BTreeMap<Entity, TdmAction>,
    pub rewards: BTreeMap<Entity, f32>,
    pub terminals: BTreeMap<Entity, bool>,
}

impl Env for Tdm {
    type Params = TdmParams;

    type Observation = TdmObs;

    type Action = TdmAction;

    fn init() -> Self {
        Self::default()
    }

    fn setup_system<M: Map>() -> bevy::ecs::schedule::SystemConfigs {
        (M::setup_system(), setup).chain()
    }

    fn observation_system() -> bevy::ecs::schedule::SystemConfigs {
        get_observation.chain()
    }

    fn action_system() -> bevy::ecs::schedule::SystemConfigs {
        get_action.chain()
    }

    fn reward_system() -> bevy::ecs::schedule::SystemConfigs {
        get_reward.chain()
    }

    fn terminal_system() -> bevy::ecs::schedule::SystemConfigs {
        get_terminal.chain()
    }

    fn update_system() -> bevy::ecs::schedule::SystemConfigs {
        update.chain()
    }

    fn learn_system() -> bevy::ecs::schedule::SystemConfigs {
        learn.chain()
    }

    fn ui_system() -> bevy::ecs::schedule::SystemConfigs {
        ui.chain()
    }
}

fn setup(
    mut env: ResMut<Tdm>,
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    timestamp: Res<Timestamp>,
) {
    let mut taken_names = vec![];
    let obs_len = TdmObs::default_frame_stack(&env.params).0[0]
        .as_slice(&env.params)
        .len();

    for team_id in 0..env.params.num_teams {
        let thinker = PpoThinker::new(
            obs_len,
            env.params.ffa_params.agent_hidden_dim,
            ACTION_LEN,
            env.params.ffa_params.agent_training_epochs,
            env.params.ffa_params.agent_training_batch_size,
            env.params.ffa_params.agent_entropy_beta,
            env.params.ffa_params.agent_actor_lr,
            env.params.ffa_params.agent_critic_lr,
        );
        let shared_thinker = SharedThinker::new(thinker);
        for _ in 0..env.params.agents_per_team() {
            let ts = timestamp.clone();
            let mut name = crate::names::random_name();
            while taken_names.contains(&name) {
                name = crate::names::random_name();
            }
            taken_names.push(name.clone());
            let brain_name = name.clone();
            let thinker = shared_thinker.clone();
            let brain_id = env
                .brains
                .spawn(|rx| Brain::new(thinker, brain_name, ts, rx));
            let agent_pos = Vec3::new(
                (rand::random::<f32>() - 0.5) * 500.0,
                (rand::random::<f32>() - 0.5) * 500.0,
                0.0,
            );
            let agent = TdmAgentBundle::new(
                agent_pos,
                team_id,
                name,
                brain_id,
                &mut meshes,
                &mut materials,
                &env.params,
            );
            let id = commands
                .spawn(agent)
                .insert(ActiveEvents::all())
                .with_children(|parent| {
                    parent.spawn(ShootyLineBundle::new(
                        materials.reborrow(),
                        meshes.reborrow(),
                    ));
                    Eyeballs::spawn(
                        parent,
                        meshes.reborrow(),
                        materials.reborrow(),
                        env.params.ffa_params.agent_radius,
                    );
                })
                .id();
            commands.spawn(NameTextBundle::new(&asset_server, id));
            commands.spawn(HealthBarBundle::new(
                meshes.reborrow(),
                materials.reborrow(),
                id,
            ));

            let params = env.params.clone();
            env.observations
                .insert(id, TdmObs::default_frame_stack(&params));
            env.rbs.insert(id, PpoBuffer::default());
            env.brains.assign_entity(brain_id, id);
        }
    }
}

fn get_observation(
    mut env: ResMut<Tdm>,
    agents: Query<Entity, With<Agent>>,
    team_ids: Query<&TeamId, With<Agent>>,
    agent_velocity: Query<&Velocity, With<Agent>>,
    agent_transform: Query<&Transform, With<Agent>>,
    agent_health: Query<&Health, With<Agent>>,
    brain_ids: Query<&BrainId, With<Agent>>,
    cx: Res<RapierContext>,
) {
    for agent_ent in agents.iter() {
        let my_t = agent_transform.get(agent_ent).unwrap();
        let my_v = agent_velocity.get(agent_ent).unwrap();
        let my_h = agent_health.get(agent_ent).unwrap();
        let mut my_state = TdmObs {
            ffa_state: ffa::FfaObs {
                pos: my_t.translation.xy(),
                linvel: my_v.linvel,
                direction: my_t.local_y().xy(),
                health: my_h.0,
                other_states: vec![],
                down_wall_dist: 0.0,
                up_wall_dist: 0.0,
                left_wall_dist: 0.0,
                right_wall_dist: 0.0,
            },
            teammate_states: vec![],
            enemy_states: vec![],
        };

        let my_team = team_ids.get(agent_ent).unwrap();

        let filter = QueryFilter::only_fixed();
        if let Some((_, toi)) = cx.cast_ray(my_t.translation.xy(), Vec2::Y, 2000.0, true, filter) {
            my_state.ffa_state.up_wall_dist = toi;
        }
        if let Some((_, toi)) =
            cx.cast_ray(my_t.translation.xy(), Vec2::NEG_Y, 2000.0, true, filter)
        {
            my_state.ffa_state.down_wall_dist = toi;
        }
        if let Some((_, toi)) = cx.cast_ray(my_t.translation.xy(), Vec2::X, 2000.0, true, filter) {
            my_state.ffa_state.right_wall_dist = toi;
        }
        if let Some((_, toi)) =
            cx.cast_ray(my_t.translation.xy(), Vec2::NEG_X, 2000.0, true, filter)
        {
            my_state.ffa_state.left_wall_dist = toi;
        }

        for other_ent in agents
            .iter()
            .filter(|a| *a != agent_ent)
            .sorted_by_key(|a| {
                let other_t = agent_transform.get(*a).unwrap();
                other_t.translation.distance(my_t.translation) as i64
            })
        {
            let other_id = brain_ids.get(other_ent).unwrap();
            let other_t = agent_transform.get(other_ent).unwrap();
            let other_v = agent_velocity.get(other_ent).unwrap();
            let status = env.brains.get_status(other_id.0);
            let other_state = ffa::OtherState {
                rel_pos: my_t.translation.xy() - other_t.translation.xy(),
                linvel: other_v.linvel,
                direction: other_t.local_y().xy(),
                firing: status
                    .unwrap_or_default()
                    .last_action
                    .unwrap_or_default()
                    .ffa_action
                    .shoot
                    > 0.0,
            };
            let other_team = team_ids.get(other_ent).unwrap();

            if my_team.0 == other_team.0 {
                my_state.teammate_states.push(TeammateState {
                    ffa_state: other_state,
                });
            } else {
                my_state.enemy_states.push(EnemyState {
                    ffa_state: other_state,
                });
            }
        }

        let max_len = env.params.ffa_params.agent_frame_stack_len;
        env.observations
            .get_mut(&agent_ent)
            .unwrap()
            .push(my_state, Some(max_len));
    }
}

fn get_action(
    mut env: ResMut<Tdm>,
    frame_count: Res<FrameCount>,
    agents: Query<Entity, With<Agent>>,
    brain_ids: Query<&BrainId, With<Agent>>,
) {
    if frame_count.0 as usize % env.params.ffa_params.agent_frame_stack_len == 0 {
        for agent_ent in agents.iter() {
            let brain_id = brain_ids.get(agent_ent).unwrap().0;
            env.brains.send_obs(
                brain_id,
                env.observations[&agent_ent].clone(),
                frame_count.0 as usize,
                env.params.clone(),
            );
            let status = env.brains.get_status(brain_id);
            if let Some(status) = status {
                status
                    .last_action
                    .and_then(|action| env.actions.insert(agent_ent, action));
            }
        }
    }
}

fn get_reward(
    mut env: ResMut<Tdm>,
    _commands: Commands,
    cx: Res<RapierContext>,
    agents: Query<Entity, With<Agent>>,
    agent_transform: Query<&Transform, With<Agent>>,
    childs: Query<&Children, With<Agent>>,
    mut health: Query<&mut Health, With<Agent>>,
    mut kills: Query<&mut Kills, With<Agent>>,
    mut force: Query<&mut ExternalForce, With<Agent>>,
    mut line_vis: Query<&mut Visibility, With<ShootyLine>>,
    mut line_transform: Query<&mut Transform, (With<ShootyLine>, Without<Agent>)>,
    names: Query<&Name, With<Agent>>,
    team_ids: Query<&TeamId, With<Agent>>,
    mut log: ResMut<LogText>,
) {
    for agent_ent in agents.iter() {
        env.rewards.insert(agent_ent, 0.0);
        let my_health = health.get(agent_ent).unwrap();
        let my_t = agent_transform.get(agent_ent).unwrap();
        if let Some(action) = env.actions.get(&agent_ent).cloned() {
            if action.ffa_action.shoot > 0.0 && my_health.0 > 0.0 {
                let (ray_dir, ray_pos) = {
                    let ray_dir = my_t.local_y().xy();
                    let ray_pos = my_t.translation.xy()
                        + ray_dir * (env.params.ffa_params.agent_radius + 2.0);

                    for child in childs.get(agent_ent).unwrap().iter() {
                        if let Ok(mut vis) = line_vis.get_mut(*child) {
                            *vis = Visibility::Visible;
                        }
                    }
                    (ray_dir, ray_pos)
                };

                let filter = QueryFilter::default().exclude_collider(agent_ent);

                if let Some((hit_entity, toi)) = cx.cast_ray(
                    ray_pos,
                    ray_dir,
                    env.params.ffa_params.agent_shoot_distance,
                    false,
                    filter,
                ) {
                    for child in childs.get(agent_ent).unwrap().iter() {
                        if let Ok(mut line) = line_transform.get_mut(*child) {
                            line.scale = Vec3::new(1.0, toi, 1.0);
                            line.translation = Vec3::new(0.0, toi / 2.0, 0.0);
                        }
                    }

                    // no friendly fire damage (todo: make this an option)
                    if let Ok(other_team) = team_ids.get(hit_entity) {
                        if team_ids.get(agent_ent).unwrap().0 != other_team.0 {
                            if let Ok(mut health) = health.get_component_mut::<Health>(hit_entity) {
                                if health.0 > 0.0 {
                                    health.0 -= 5.0;
                                    *env.rewards.get_mut(&agent_ent).unwrap() += 1.0;
                                    *env.rewards.get_mut(&hit_entity).unwrap() -= 1.0;
                                    if health.0 <= 0.0 {
                                        *env.rewards.get_mut(&agent_ent).unwrap() += 100.0;
                                        *env.rewards.get_mut(&hit_entity).unwrap() -= 100.0;
                                        kills.get_mut(agent_ent).unwrap().0 += 1;
                                        let msg = format!(
                                            "{} killed {}! Nice!",
                                            &names.get(agent_ent).unwrap().0,
                                            &names.get(hit_entity).unwrap().0
                                        );
                                        log.push(msg);
                                    }
                                } else {
                                    // hit a wall
                                    *env.rewards.get_mut(&agent_ent).unwrap() -= 4.0;
                                }
                            }
                        } else {
                            // penalize friendly fire
                            *env.rewards.get_mut(&agent_ent).unwrap() -= 10.0;
                        }
                    }
                } else {
                    // hit nothing
                    for child in childs.get(agent_ent).unwrap().iter() {
                        if let Ok(mut line) = line_transform.get_mut(*child) {
                            line.scale =
                                Vec3::new(1.0, env.params.ffa_params.agent_shoot_distance, 1.0);
                            line.translation = Vec3::new(
                                0.0,
                                env.params.ffa_params.agent_shoot_distance / 2.0,
                                0.0,
                            );
                        }
                    }
                    *env.rewards.get_mut(&agent_ent).unwrap() -= 4.0;
                }
            } else {
                for child in childs.get(agent_ent).unwrap().iter() {
                    if let Ok(mut vis) = line_vis.get_mut(*child) {
                        *vis = Visibility::Hidden;
                    }
                }
            }

            let mut my_force = force.get_mut(agent_ent).unwrap();
            my_force.force =
                action.ffa_action.lin_force * env.params.ffa_params.agent_lin_move_force;
            my_force.torque =
                action.ffa_action.ang_force * env.params.ffa_params.agent_ang_move_force;
        }
    }
}

fn get_terminal(
    mut env: ResMut<Tdm>,
    agents: Query<Entity, With<Agent>>,
    health: Query<&Health, With<Agent>>,
) {
    for agent_ent in agents.iter() {
        if health.get(agent_ent).unwrap().0 <= 0.0 {
            env.terminals.insert(agent_ent, true);
        } else {
            env.terminals.insert(agent_ent, false);
        }
    }
}

fn update(
    mut commands: Commands,
    mut env: ResMut<Tdm>,
    mut name_text_t: Query<
        (Entity, &mut Transform, &mut Text, &mut NameText),
        (With<NameText>, Without<Agent>),
    >,
    mut health_bar_t: Query<
        (Entity, &mut Transform, &HealthBar),
        (Without<NameText>, Without<Agent>),
    >,
    agents: Query<Entity, With<Agent>>,
    brain_ids: Query<&BrainId, With<Agent>>,
    names: Query<&Name, With<Agent>>,
    kills: Query<&Kills, With<Agent>>,
    mut deaths: Query<&mut Deaths, With<Agent>>,
    mut health: Query<&mut Health, With<Agent>>,
    mut agent_transform: Query<&mut Transform, With<Agent>>,
) {
    for (t_ent, mut t, mut text, text_comp) in name_text_t.iter_mut() {
        if let Ok(agent) = agent_transform.get(text_comp.entity_following) {
            t.translation =
                agent.translation + Vec3::new(0.0, env.params.ffa_params.agent_radius + 20.0, 2.0);
            text.sections[0].value = format!(
                "{} {} {}-{}",
                brain_ids.get(text_comp.entity_following).unwrap().0,
                names.get(text_comp.entity_following).unwrap().0,
                kills.get(text_comp.entity_following).unwrap().0,
                deaths.get(text_comp.entity_following).unwrap().0,
            );
        } else {
            commands.entity(t_ent).despawn();
        }
    }
    for (t_ent, mut t, hb) in health_bar_t.iter_mut() {
        if let Ok(agent) = agent_transform.get(hb.entity_following) {
            t.translation =
                agent.translation + Vec3::new(0.0, env.params.ffa_params.agent_radius + 5.0, 2.0);
            let health = health.get(hb.entity_following).unwrap();
            t.scale = Vec3::new(
                health.0 / env.params.ffa_params.agent_max_health * 100.0,
                1.0,
                1.0,
            );
        } else {
            commands.entity(t_ent).despawn();
        }
    }

    for agent_ent in agents.iter() {
        let _status = env.brains.get_status(brain_ids.get(agent_ent).unwrap().0);
        let (action, reward, terminal) = (
            env.actions.get(&agent_ent).cloned(),
            env.rewards.get(&agent_ent).copied(),
            env.terminals.get(&agent_ent).copied(),
        );
        let obs = env.observations[&agent_ent].clone();
        let max_len = env.params.ffa_params.agent_rb_max_len;
        // if let Some(status) = status {
        if let Some(rb) = env.rbs.get_mut(&agent_ent) {
            // if fresh {
            if let (Some(action), Some(reward), Some(terminal)) = (action, reward, terminal) {
                rb.remember_sart(
                    Sart {
                        obs,
                        action: action.to_owned(),
                        reward,
                        terminal,
                    },
                    Some(max_len),
                );
            }
        }
        // }
    }

    for agent_ent in agents.iter() {
        let mut my_health = health.get_mut(agent_ent).unwrap();
        if my_health.0 <= 0.0 {
            if let Some(rb) = env.rbs.get_mut(&agent_ent) {
                rb.finish_trajectory();
            }

            // let mut ent = commands.entity(agent);
            deaths.get_mut(agent_ent).unwrap().0 += 1;
            my_health.0 = env.params.ffa_params.agent_max_health;
            let agent_pos = Vec3::new(
                (rand::random::<f32>() - 0.5) * 500.0,
                (rand::random::<f32>() - 0.5) * 500.0,
                0.0,
            );
            agent_transform.get_mut(agent_ent).unwrap().translation = agent_pos;
        }
    }
}

fn learn(
    mut env: ResMut<Tdm>,
    frame_count: Res<FrameCount>,
    agents: Query<Entity, With<Agent>>,
    brain_ids: Query<&BrainId, With<Agent>>,
    deaths: Query<&Deaths, With<Agent>>,
    names: Query<&Name, With<Agent>>,
    mut log: ResMut<LogText>,
) {
    if frame_count.0 > 1
        && frame_count.0 as usize % env.params.ffa_params.agent_update_interval == 0
    {
        for agent_ent in agents.iter() {
            if deaths.get(agent_ent).unwrap().0 > 0 {
                let rb = env.rbs[&agent_ent].clone();
                let id = brain_ids.get(agent_ent).unwrap().0;
                let name = &names.get(agent_ent).unwrap().0;
                log.push(format!("Training {id} {name}..."));

                AsyncComputeTaskPool::get().scope(|scope| {
                    scope.spawn(async {
                        env.brains
                            .learn(id, frame_count.0 as usize, rb, env.params.clone())
                            .await;
                    });
                });
                let status = env.brains.get_status(id);
                if let Some(status) = status.and_then(|s| s.status) {
                    log.push(format!(
                        "{} {} Policy Loss: {}",
                        id, name, status.recent_policy_loss
                    ));
                    log.push(format!(
                        "{} {} Policy Entropy: {}",
                        id, name, status.recent_entropy_loss
                    ));
                    log.push(format!(
                        "{} {} Policy Clip Ratio: {}",
                        id, name, status.recent_nclamp
                    ));
                    log.push(format!(
                        "{} {} Value Loss: {}",
                        id, name, status.recent_value_loss
                    ));
                }
            }
        }
    }
}

pub fn ui(
    mut cxs: EguiContexts,
    mut env: ResMut<Tdm>,
    log: ResMut<LogText>,
    agents: Query<Entity, With<Agent>>,
    kills: Query<&Kills, With<Agent>>,
    deaths: Query<&Deaths, With<Agent>>,
    brain_ids: Query<&BrainId, With<Agent>>,
    names: Query<&Name, With<Agent>>,
) {
    egui::Window::new("Scores").show(cxs.ctx_mut(), |ui| {
        ui.vertical(|ui| {
            for handle in agents
                .iter()
                .sorted_by_key(|b| kills.get(*b).unwrap().0)
                .rev()
            {
                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new(format!(
                            "{} {}",
                            brain_ids.get(handle).unwrap().0,
                            names.get(handle).unwrap().0,
                        ))
                        .text_style(egui::TextStyle::Heading),
                    );
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::TOP), |ui| {
                        ui.label(
                            egui::RichText::new(format!(
                                "{}-{}",
                                kills.get(handle).unwrap().0,
                                deaths.get(handle).unwrap().0,
                            ))
                            .text_style(egui::TextStyle::Heading),
                        );
                    });
                });
            }
        });
    });
    egui::Window::new("Action Mean/Std/Entropy")
        .min_height(200.0)
        .min_width(1200.0)
        // .auto_sized()
        .scroll2([true, false])
        .resizable(true)
        .show(cxs.ctx_mut(), |ui| {
            ui.with_layout(
                Layout::left_to_right(egui::Align::Min).with_main_wrap(false),
                |ui| {
                    // egui::Grid::new("mean/std grid").show(ui, |ui| {
                    for (_i, brain) in agents
                        .iter()
                        .sorted_by_key(|h| brain_ids.get(*h).unwrap().0)
                        .enumerate()
                    {
                        ui.group(|ui| {
                            ui.vertical(|ui| {
                                ui.heading(&names.get(brain).unwrap().0);
                                // ui.group(|ui| {
                                let status = env
                                    .brains
                                    .get_status(brain_ids.get(brain).unwrap().0)
                                    .unwrap_or_default();
                                let status = status.status.unwrap();
                                let mut mu = "mu:".to_owned();
                                for m in status.recent_mu.iter() {
                                    mu.push_str(&format!(" {:.4}", m));
                                }
                                ui.label(mu);
                                let mut std = "std:".to_owned();
                                for s in status.recent_std.iter() {
                                    std.push_str(&format!(" {:.4}", s));
                                }
                                ui.label(std);
                                ui.label(format!("ent: {}", status.recent_entropy));
                                // });

                                // ui.horizontal_top(|ui| {

                                let ms = status
                                    .recent_mu
                                    .iter()
                                    .zip(status.recent_std.iter())
                                    .enumerate()
                                    .map(|(i, (mu, std))| {
                                        // https://www.desmos.com/calculator/rkoehr8rve
                                        let scale = std * 3.0;
                                        let _rg =
                                            Vec2::new(scale.exp(), (1.0 / scale).exp()).normalize();
                                        let m = Bar::new(i as f64, *mu as f64)
                                            // .fill(Color32::from_rgb(
                                            //     (rg.x * 255.0) as u8,
                                            //     (rg.y * 255.0) as u8,
                                            //     0,
                                            // ));
                                            .fill(Color32::RED);
                                        let var = std * std;
                                        let s = Line::new(vec![
                                            [i as f64, *mu as f64 - var as f64],
                                            [i as f64, *mu as f64 + var as f64],
                                        ])
                                        .stroke(Stroke::new(4.0, Color32::LIGHT_GREEN));
                                        (m, s)
                                        // .width(1.0 - *std as f64 / 6.0)
                                    })
                                    .collect_vec();

                                ui.group(|ui| {
                                    ui.horizontal(|ui| {
                                        ui.vertical(|ui| {
                                            ui.label("Action Space");
                                            egui::plot::Plot::new(format!(
                                                "ActionSpace{}",
                                                brain_ids.get(brain).unwrap().0,
                                            ))
                                            .center_y_axis(true)
                                            .data_aspect(1.0 / 2.0)
                                            .height(80.0)
                                            .width(220.0)
                                            .show(
                                                ui,
                                                |plot| {
                                                    let (mu, std): (Vec<Bar>, Vec<Line>) =
                                                        ms.into_iter().multiunzip();
                                                    plot.bar_chart(BarChart::new(mu));
                                                    for std in std {
                                                        plot.line(std);
                                                    }
                                                },
                                            );
                                        });
                                    });
                                });
                            });
                        });
                    }
                },
            );
        });
    egui::Window::new("Log")
        .vscroll(true)
        .hscroll(true)
        .show(cxs.ctx_mut(), |ui| {
            let s = format!("{}", *log);
            ui.add_sized(
                ui.available_size(),
                egui::TextEdit::multiline(&mut s.as_str()).desired_rows(20),
            );
        });
}