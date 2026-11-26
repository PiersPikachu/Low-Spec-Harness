//! The CPU inference engine: runs a cube vault (dense, Qwen3-MoE, gpt-oss) streaming its weights from disk.

pub mod kernels;
pub mod model;
pub mod sample;
pub mod serve;
pub mod simd;
pub mod store;

pub use model::{Engine, Options, Timing};
