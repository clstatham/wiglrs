pub const NUM_AGENTS: usize = 4;
pub const AGENT_HIDDEN_DIM: usize = 128;
pub const AGENT_ACTOR_LR: f64 = 1e-5;
pub const AGENT_CRITIC_LR: f64 = 1e-4;
pub const AGENT_OPTIM_EPOCHS: usize = 5;
pub const AGENT_OPTIM_BATCH_SIZE: usize = 128;

pub const AGENT_TICK_RATE: usize = 3;
pub const AGENT_RB_MAX_LEN: usize = 1_000;
pub const N_FRAME_STACK: usize = 3;

pub const AGENT_RADIUS: f32 = 20.0;
pub const AGENT_LIN_MOVE_FORCE: f32 = 600.0;
pub const AGENT_ANG_MOVE_FORCE: f32 = 1.0;

pub const AGENT_MAX_HEALTH: f32 = 100.0;
pub const AGENT_SHOOT_DISTANCE: f32 = 500.0;
