pub const NUM_AGENTS: usize = 8;
pub const AGENT_EMBED_DIM: usize = 512;
pub const AGENT_ACTOR_LR: f64 = 1e-4;
pub const AGENT_CRITIC_LR: f64 = 1e-4;
pub const AGENT_OPTIM_EPOCHS: usize = 1;
pub const AGENT_OPTIM_BATCH_SIZE: usize = 128;

// pub const N_FRAME_STACK: usize = 4;

pub const AGENT_RADIUS: f32 = 20.0;
pub const AGENT_MAX_LIN_VEL: f32 = 300.0;
// pub const AGENT_MAX_ANG_VEL: f32 = 2.0;
pub const AGENT_LIN_MOVE_FORCE: f32 = 300.0;
pub const AGENT_ANG_MOVE_FORCE: f32 = 1.0;

pub const AGENT_MAX_HEALTH: f32 = 100.0;
pub const AGENT_SHOOT_DISTANCE: f32 = 200.0;
