//! Readers for the v3-v7 vault layout (`vault.idx` + `vault.bin` + `resident.bin` + loose tokenizer files), as written by
//! `pack_universal_v3.py` and `pack_gptoss.py` and read by `nvme_engine_u.c`.

use crate::meta::*;
use crate::stream::StreamEntry;
use anyhow::{bail, ensure, Context, Result};
use std::collections::HashMap;
use std::path::Path;

pub struct Legacy {
    pub version: u32,
    pub meta: ModelMeta,
    pub streams: Vec<StreamEntry>,
    pub rope_inv: Vec<f32>,
}

/// Parse `vault.idx` exactly as the C engine does (unknown keys ignored).
pub fn read_idx(dir: &Path) -> Result<Legacy> {
    let text = std::fs::read_to_string(dir.join("vault.idx")).with_context(|| format!("no vault.idx in {}", dir.display()))?;
    let mut ver = 0u32;
    let mut kv: HashMap<String, f64> = HashMap::new();
    let mut streams: Vec<StreamEntry> = Vec::new();
    let mut layer_types: Vec<u8> = Vec::new();
    let mut rope_inv: Vec<f32> = Vec::new();
    for line in text.lines() {
        let line = line.trim_end_matches('\r');
        let mut it = line.split_whitespace();
        let Some(key) = it.next() else { continue };
        match key {
            "WVAULT" => ver = it.next().context("WVAULT without version")?.parse()?,
            "layer_types" => layer_types = it.map(|x| x.parse::<u8>()).collect::<Result<_, _>>()?,
            "rope_inv" => rope_inv = it.map(|x| x.parse::<f32>()).collect::<Result<_, _>>()?,
            "stream" => {
                let v: Vec<&str> = it.collect();
                ensure!(v.len() == 6, "bad stream line: {line}");
                let i: usize = v[0].parse()?;
                if streams.len() <= i {
                    streams.resize(i + 1, StreamEntry::default());
                }
                streams[i] = StreamEntry {
                    off: v[1].parse()?,
                    size: v[2].parse()?,
                    padded: v[3].parse()?,
                    raw: v[4].parse()?,
                    hash: u64::from_str_radix(v[5], 16)?,
                };
            }
            k => {
                if let Some(v) = it.next() {
                    if let Ok(x) = v.parse::<f64>() {
                        kv.insert(k.to_string(), x);
                    }
                }
            }
        }
    }
    ensure!(matches!(ver, 3 | 4 | 6 | 7), "vault format {ver} unknown (expected WVAULT 3, 4, 6 or 7)");
    let g = |k: &str| kv.get(k).copied();
    let gi = |k: &str| kv.get(k).map(|&v| v as usize);
    let req = |k: &str| gi(k).with_context(|| format!("vault.idx: missing '{k}'"));
    let hidden = req("hidden")?;
    let head_dim = req("head_dim")?;
    let mut lr = LayerResident {
        len: gi("lres").unwrap_or(0),
        ln_in: gi("r_ln_in").unwrap_or(usize::MAX),
        ln_post: gi("r_ln_post").unwrap_or(usize::MAX),
        q_norm: gi("r_qn"),
        k_norm: gi("r_kn"),
        bq: gi("r_bq"),
        bk: gi("r_bk"),
        bv: gi("r_bv"),
        bo: gi("r_bo"),
        bg: gi("r_bg"),
        bu: gi("r_bu"),
        bd: gi("r_bd"),
        sinks: gi("r_sinks"),
        router_bias: gi("r_rbias"),
    };
    let mut supported = gi("supported").unwrap_or(0) == 1;
    if ver == 3 {
        // v3: fixed Qwen3 layout ln_in, ln_post, q_norm, k_norm
        lr = LayerResident {
            len: 2 * hidden + 2 * head_dim,
            ln_in: 0,
            ln_post: hidden,
            q_norm: Some(2 * hidden),
            k_norm: Some(2 * hidden + head_dim),
            ..Default::default()
        };
        supported = true;
    }
    if !supported || lr.ln_in == usize::MAX || lr.ln_post == usize::MAX || lr.len == 0 {
        bail!("vault layout not supported by this engine (see engine_u.reasons in vault.json)");
    }
    let codec = match gi("codec").unwrap_or(0) {
        0 => Codec::Huff4,
        1 => Codec::Rans4,
        c => bail!("unknown weight codec {c}"),
    };
    let layers = req("layers")?;
    let tie = gi("tie").unwrap_or(0) == 1;
    let gptoss_on = gi("gptoss").unwrap_or(0) == 1;
    let moe_on = gi("moe").unwrap_or(0) == 1;
    let rope = if gptoss_on {
        Rope::Table { scale: g("rope_scale").unwrap_or(1.0) }
    } else if gi("rope_llama3").unwrap_or(0) == 1 {
        Rope::Llama3 {
            theta: g("theta").unwrap_or(0.0),
            factor: g("rope_factor").unwrap_or(1.0),
            low: g("rope_low").unwrap_or(1.0),
            high: g("rope_high").unwrap_or(1.0),
            orig: g("rope_orig").unwrap_or(1.0),
        }
    } else {
        Rope::Default { theta: g("theta").unwrap_or(0.0) }
    };
    let moe = if moe_on {
        Some(Moe {
            experts: req("n_experts")?,
            top_k: req("top_k")?,
            inter: req("moe_inter")?,
            norm_topk: gi("norm_topk").unwrap_or(0) == 1,
            expert_stream0: req("expert_stream0")?,
        })
    } else {
        None
    };
    let gptoss = if gptoss_on {
        let mut ls = layer_types.clone();
        ls.resize(layers, 0);
        Some(Gptoss {
            sliding_window: gi("sliding_window").unwrap_or(0),
            layer_sliding: ls,
            swiglu_limit: g("swiglu_limit").unwrap_or(7.0),
            swiglu_alpha: g("swiglu_alpha").unwrap_or(1.702),
        })
    } else {
        None
    };
    let eos = {
        let (a, b) = (g("eos").unwrap_or(-1.0) as i64, g("eos2").unwrap_or(-1.0) as i64);
        if a == b {
            vec![a]
        } else {
            vec![a, b]
        }
    };
    let arch = if gptoss_on {
        Arch::Gptoss
    } else if moe_on {
        Arch::QwenMoe
    } else {
        Arch::Dense
    };
    ensure!(streams.len() > layers, "vault.idx lists {} streams for {layers} layers", streams.len());
    let meta = ModelMeta {
        format: 8,
        arch,
        codec,
        hidden,
        inter: req("inter")?,
        layers,
        heads: req("heads")?,
        kv_heads: req("kv_heads")?,
        head_dim,
        vocab: req("vocab")?,
        tie_embeddings: tie,
        eos,
        rms_eps: g("eps").unwrap_or(1e-6),
        rope,
        layer_resident: lr,
        moe,
        gptoss,
        embed_stream: layers,
        lm_head_stream: if tie { layers } else { layers + 1 },
        lossy: false,
        source: Source { repo: None, packer: format!("legacy WVAULT {ver}"), converted_from: Some(ver) },
        tensors: Vec::new(),
    };
    Ok(Legacy { version: ver, meta, streams, rope_inv })
}

/// `resident.bin` as f32.
pub fn read_resident(dir: &Path, want: usize) -> Result<Vec<f32>> {
    let b = std::fs::read(dir.join("resident.bin")).context("resident.bin missing")?;
    ensure!(b.len() >= 4 * want, "resident.bin has {} values, the model needs {want}", b.len() / 4);
    Ok(b[..4 * want].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())
}

/// The per-tensor table of a v3/v4/v6 `vault.json` (gpt-oss v7 packs have none).
pub fn read_tensors(dir: &Path) -> Vec<TensorRef> {
    let Some(v) = std::fs::read(dir.join("vault.json")).ok().and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok()) else {
        return Vec::new();
    };
    let Some(ts) = v.get("tensors").and_then(|t| t.as_array()) else { return Vec::new() };
    ts.iter()
        .filter_map(|t| {
            let store = t.get("store")?.as_str()?;
            let u = |k: &str| t.get(k).and_then(|x| x.as_u64()).map(|x| x as usize);
            Some(TensorRef {
                name: t.get("name")?.as_str()?.to_string(),
                shape: t.get("shape")?.as_array()?.iter().filter_map(|x| x.as_u64().map(|x| x as usize)).collect(),
                src_dtype: t.get("src_dtype").and_then(|x| x.as_str()).unwrap_or("").to_string(),
                store: match store {
                    "resident_f32" => "resident".into(),
                    s => s.into(),
                },
                stream: u("stream"),
                sub: u("sub"),
                resident: if store == "resident_f32" { Some((u("res_off")?, u("count")?)) } else { None },
                alias_of: t.get("alias_of").and_then(|x| x.as_str()).map(String::from),
            })
        })
        .collect()
}

/// `lossy` flag from `vault.json`, when present.
pub fn read_lossy(dir: &Path) -> bool {
    std::fs::read(dir.join("vault.json"))
        .ok()
        .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
        .and_then(|v| v.get("lossy").and_then(|x| x.as_bool()))
        .unwrap_or(false)
}
