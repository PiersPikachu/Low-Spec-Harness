//! `ModelMeta`: the typed model description in the `META` section (replaces the key/value lines of `vault.idx`).

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Arch {
    /// q/k/v/o + gate/up/down per layer (Qwen2/3, Llama, Mistral without sliding window)
    Dense,
    /// Qwen3-MoE: q/k/v/o + router per layer, experts gate/up/down (bf16), softmax-then-top-k routing
    QwenMoe,
    /// gpt-oss: q/k/v/o + router (+ biases, attention sinks), MXFP4 experts, top-k-then-softmax, clamped SwiGLU, sliding layers
    Gptoss,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Codec {
    Huff4,
    Rans4,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Rope {
    /// inv_freq = theta^(-2i/d), cos/sin computed in f64
    Default { theta: f64 },
    /// Llama 3 frequency scaling
    Llama3 { theta: f64, factor: f64, low: f64, high: f64, orig: f64 },
    /// inverse frequencies given (the `ROPE` section, f32), cos/sin in f32 times `scale` (gpt-oss YaRN)
    Table { scale: f64 },
}

/// Offsets of the per-layer resident vectors inside one layer's slice of `RESI` (in f32 elements).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct LayerResident {
    pub len: usize,
    pub ln_in: usize,
    pub ln_post: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub q_norm: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub k_norm: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bq: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bk: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bv: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bo: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bg: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bu: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bd: Option<usize>,
    /// gpt-oss attention sinks (one per head)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sinks: Option<usize>,
    /// gpt-oss router bias (one per expert)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub router_bias: Option<usize>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Moe {
    pub experts: usize,
    pub top_k: usize,
    /// expert hidden size
    pub inter: usize,
    /// renormalise the top-k weights (Qwen3-MoE `norm_topk_prob`)
    pub norm_topk: bool,
    /// index of the first expert stream; expert (l, e) is stream `expert_stream0 + l * experts + e`
    pub expert_stream0: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Gptoss {
    pub sliding_window: usize,
    /// 1 = sliding-window attention layer
    pub layer_sliding: Vec<u8>,
    pub swiglu_limit: f64,
    pub swiglu_alpha: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Source {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repo: Option<String>,
    pub packer: String,
    /// the v3-v7 format this vault was converted from
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub converted_from: Option<u32>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ModelMeta {
    pub format: u32,
    pub arch: Arch,
    pub codec: Codec,
    pub hidden: usize,
    /// dense MLP size (the expert size is in `moe.inter`)
    pub inter: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub vocab: usize,
    pub tie_embeddings: bool,
    /// end-of-sequence token ids (the engine stops on any of them)
    pub eos: Vec<i64>,
    pub rms_eps: f64,
    pub rope: Rope,
    pub layer_resident: LayerResident,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub moe: Option<Moe>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gptoss: Option<Gptoss>,
    /// stream index of the embedding and of the lm_head (equal when tied)
    pub embed_stream: usize,
    pub lm_head_stream: usize,
    pub lossy: bool,
    pub source: Source,
}

impl ModelMeta {
    pub fn expert_stream(&self, layer: usize, expert: usize) -> Option<usize> {
        self.moe.as_ref().map(|m| m.expert_stream0 + layer * m.experts + expert)
    }
    /// f32 count of `RESI`: per-layer vectors then the final norm
    pub fn resident_len(&self) -> usize {
        self.layers * self.layer_resident.len + self.hidden
    }
}
