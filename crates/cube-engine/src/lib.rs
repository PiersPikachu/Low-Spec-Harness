//! The CPU inference engine: runs a cube vault (dense, Qwen3-MoE, gpt-oss) streaming its weights from disk.

pub mod backend;
pub mod kernels;
pub mod model;
pub mod sample;
pub mod serve;
pub mod simd;
pub mod store;

pub use backend::{Backend, LayerPrefix};
pub use model::{End, Engine, GenStats, Options, Timing};
