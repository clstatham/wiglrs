pub const NUM_AGENTS: usize = 4;
pub const AGENT_HIDDEN_DIM: usize = 128;
pub const AGENT_ACTOR_LR: f64 = 1e-4;
pub const AGENT_CRITIC_LR: f64 = 1e-3;
pub const AGENT_OPTIM_EPOCHS: usize = 25;
pub const AGENT_OPTIM_BATCH_SIZE: usize = 128;
pub const AGENT_ENTROPY_BETA: f32 = 0.0;

pub const AGENT_UPDATE_INTERVAL: usize = 2_000;
pub const AGENT_RB_MAX_LEN: usize = 100_000;
pub const N_FRAME_STACK: usize = 2;

pub const AGENT_RADIUS: f32 = 20.0;
pub const AGENT_LIN_MOVE_FORCE: f32 = 600.0;
pub const AGENT_ANG_MOVE_FORCE: f32 = 1.0;

pub const AGENT_MAX_HEALTH: f32 = 50.0;
pub const AGENT_SHOOT_DISTANCE: f32 = 100.0;
