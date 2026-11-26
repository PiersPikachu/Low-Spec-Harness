//! Pack a Hugging Face checkpoint into a v8 vault: `vault.bin` (streams) + `vault.cv8` (manifest).
//!
//! Dense and Qwen3-MoE checkpoints follow `pack_universal_v3.py` stream for stream (same planning, same order, same codec
//! bytes), so `vault.bin` is byte-identical to the Python packer's. gpt-oss follows `pack_gptoss.py`. Every matrix is decoded
//! again and compared with the source before it is written.

pub mod safetensors;

use anyhow::{bail, ensure, Context, Result};
use cube_vault::hash::{xxh3, HashKind};
use cube_vault::meta::*;
use cube_vault::stream::{self, StreamEntry};
use cube_vault::v8::{Manifest, Role, StreamInfo, FILE_SECTIONS};
use cube_vault::{huff4, mxf, rans4, MANIFEST};
use rayon::prelude::*;
use regex::Regex;
use safetensors::{Checkpoint, TensorInfo};
use serde_json::Value;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::io::Write;
use std::path::Path;

pub struct Options {
    pub codec: Codec,
    /// allow F16/F32 matrices (rounded to bf16: lossy, flagged in the vault)
    pub convert: bool,
    /// MXF2 (packed scales) for gpt-oss experts, else MXF4 as released
    pub mxf2: bool,
    pub repo: Option<String>,
}

impl Default for Options {
    fn default() -> Self {
        Options { codec: Codec::Huff4, convert: false, mxf2: true, repo: None }
    }
}

pub struct Report {
    pub streams: usize,
    pub raw: u64,
    pub stored: u64,
    pub lossy: bool,
    pub arch: Arch,
}

const KNOWN_MAT: [&str; 7] =
    ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"];
const KNOWN_NORM: [&str; 4] = ["input_layernorm", "post_attention_layernorm", "self_attn.q_norm", "self_attn.k_norm"];

#[derive(Clone, Debug)]
struct PStream {
    kind: &'static str, // layer | embed | lm_head | extra | expert
    layer: Option<usize>,
    expert: Option<usize>,
    mats: Vec<String>,
    stems: Vec<String>,
}

fn rank(stem: &str, order: &[&str]) -> usize {
    order.iter().position(|x| *x == stem).unwrap_or(order.len())
}
fn stem(rest: &str) -> String {
    rest.strip_suffix(".weight").or_else(|| rest.strip_suffix(".bias")).unwrap_or(rest).to_string()
}

fn text_config(cfg: &Value) -> &Value {
    cfg.get("text_config").unwrap_or(cfg)
}

/// `plan_streams` of the Python packer.
fn plan_streams(ck: &Checkpoint) -> Result<(Vec<PStream>, Vec<String>, usize)> {
    let tc = text_config(&ck.config);
    let nl = tc.get("num_hidden_layers").and_then(|x| x.as_u64()).map(|x| x as usize);
    let layer_re = Regex::new(r"^(?P<pre>.*?)(?P<cont>layers|h|blocks)\.(?P<idx>\d+)\.(?P<rest>.+)$")?;
    let expert_re = Regex::new(r"^mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")?;
    let embed_re = Regex::new(r"(^|\.)(embed_tokens|wte|tok_embeddings|word_embeddings)\.weight$")?;
    let head_re = Regex::new(r"(^|\.)(lm_head|output)\.weight$")?;
    let mut groups: BTreeMap<String, BTreeMap<usize, Vec<(String, String)>>> = BTreeMap::new();
    let mut gorder: Vec<String> = Vec::new();
    for name in ck.tensors.keys() {
        if let Some(c) = layer_re.captures(name) {
            let g = format!("{}{}", &c["pre"], &c["cont"]);
            if !groups.contains_key(&g) {
                gorder.push(g.clone());
            }
            groups.entry(g).or_default().entry(c["idx"].parse()?).or_default().push((name.clone(), c["rest"].to_string()));
        }
    }
    let shape = |n: &str| &ck.tensors[n].shape;
    let main = if groups.is_empty() {
        None
    } else {
        let mut cands: Vec<&String> = groups.keys().filter(|k| nl.map_or(false, |n| groups[*k].len() == n)).collect();
        if !cands.is_empty() {
            cands.sort_by_key(|k| (k.as_str() != "model.layers", (*k).clone()));
            Some(cands[0].clone())
        } else if nl.is_none() {
            gorder.iter().max_by_key(|k| groups[*k].len()).cloned()
        } else {
            bail!("no layer group with num_hidden_layers={:?}", nl);
        }
    };
    let mut layers = Vec::new();
    let mut experts = Vec::new();
    let mut used: HashSet<String> = HashSet::new();
    let mut resident: Vec<String> = Vec::new();
    if let Some(mk) = &main {
        let g = &groups[mk];
        ensure!(g.keys().copied().eq(0..g.len()), "layer indices are not contiguous from 0");
        for li in 0..g.len() {
            let ents = &g[&li];
            let mut ex: BTreeMap<usize, HashMap<String, String>> = BTreeMap::new();
            for (n, r) in ents {
                if let Some(c) = expert_re.captures(r) {
                    ex.entry(c[1].parse()?).or_default().insert(c[2].to_string(), n.clone());
                }
            }
            for (e, m) in &ex {
                experts.push(PStream {
                    kind: "expert",
                    layer: Some(li),
                    expert: Some(*e),
                    mats: vec![m["gate_proj"].clone(), m["up_proj"].clone(), m["down_proj"].clone()],
                    stems: vec![],
                });
            }
            let mut mats: Vec<(String, String)> =
                ents.iter().filter(|(n, r)| shape(n).len() >= 2 && !expert_re.is_match(r)).cloned().collect();
            mats.sort_by_key(|(n, r)| (rank(&stem(r), &KNOWN_MAT), n.clone()));
            let mut norms: Vec<(String, String)> = ents.iter().filter(|(n, _)| shape(n).len() < 2).cloned().collect();
            norms.sort_by_key(|(n, r)| (rank(&stem(r), &KNOWN_NORM), n.clone()));
            layers.push(PStream {
                kind: "layer",
                layer: Some(li),
                expert: None,
                mats: mats.iter().map(|x| x.0.clone()).collect(),
                stems: mats.iter().map(|x| stem(&x.1)).collect(),
            });
            resident.extend(norms.iter().map(|x| x.0.clone()));
            used.extend(ents.iter().map(|x| x.0.clone()));
        }
    }
    let rest: Vec<String> = ck.tensors.keys().filter(|n| !used.contains(*n)).cloned().collect();
    let rest_mats: Vec<String> = rest.iter().filter(|n| shape(n).len() >= 2).cloned().collect();
    resident.extend(rest.iter().filter(|n| shape(n).len() < 2).cloned());
    let embed = rest_mats.iter().find(|n| embed_re.is_match(n)).cloned();
    let head = rest_mats.iter().find(|n| head_re.is_match(n) && Some(*n) != embed.as_ref()).cloned();
    let nlayers = layers.len();
    let mut streams = layers;
    if let Some(e) = &embed {
        streams.push(PStream { kind: "embed", layer: None, expert: None, mats: vec![e.clone()], stems: vec![] });
    }
    if let Some(h) = &head {
        streams.push(PStream { kind: "lm_head", layer: None, expert: None, mats: vec![h.clone()], stems: vec![] });
    }
    for n in &rest_mats {
        if Some(n) != embed.as_ref() && Some(n) != head.as_ref() {
            streams.push(PStream { kind: "extra", layer: None, expert: None, mats: vec![n.clone()], stems: vec![] });
        }
    }
    if main.is_some() {
        streams.extend(experts);
    }
    Ok((streams, resident, nlayers))
}

fn as_usize(v: &Value, k: &str) -> Option<usize> {
    v.get(k).and_then(|x| x.as_u64()).map(|x| x as usize)
}
fn as_f64(v: &Value, k: &str) -> Option<f64> {
    v.get(k).and_then(|x| x.as_f64())
}

fn ids_of(v: Option<&Value>) -> Vec<i64> {
    match v {
        Some(Value::Array(a)) => a.iter().filter_map(|x| x.as_i64()).collect(),
        Some(x) => x.as_i64().into_iter().collect(),
        None => vec![],
    }
}

/// `tokenizer_config.eos_token` looked up in `tokenizer.json`
fn tokenizer_eos(dir: &Path) -> Option<i64> {
    let tc: Value = serde_json::from_slice(&std::fs::read(dir.join("tokenizer_config.json")).ok()?).ok()?;
    let e = tc.get("eos_token")?;
    let e = e.get("content").unwrap_or(e).as_str()?.to_string();
    let tj: Value = serde_json::from_slice(&std::fs::read(dir.join("tokenizer.json")).ok()?).ok()?;
    if let Some(a) = tj.get("added_tokens").and_then(|x| x.as_array()) {
        for t in a {
            if t.get("content").and_then(|x| x.as_str()) == Some(e.as_str()) {
                return t.get("id").and_then(|x| x.as_i64());
            }
        }
    }
    tj.get("model")?.get("vocab")?.get(&e)?.as_i64()
}

/// eos ids: generation_config first (it overrides config), then the tokenizer's eos token.
fn eos_ids(dir: &Path, tc: &Value) -> Vec<i64> {
    let mut eos = ids_of(tc.get("eos_token_id"));
    if let Ok(b) = std::fs::read(dir.join("generation_config.json")) {
        if let Ok(g) = serde_json::from_slice::<Value>(&b) {
            if g.get("eos_token_id").is_some() {
                eos = ids_of(g.get("eos_token_id"));
            }
        }
    }
    if eos.is_empty() {
        eos.push(-1);
    }
    if let Some(t) = tokenizer_eos(dir) {
        if !eos.contains(&t) {
            eos.push(t);
        }
    }
    eos
}

fn rope_of(tc: &Value) -> Result<Rope> {
    let rp = tc.get("rope_parameters").filter(|x| x.is_object());
    let theta = as_f64(tc, "rope_theta").or_else(|| rp.and_then(|r| as_f64(r, "rope_theta"))).unwrap_or(0.0);
    let rs = tc.get("rope_scaling").filter(|x| x.is_object()).or(rp);
    let rt = rs.and_then(|r| r.get("rope_type").or_else(|| r.get("type"))).and_then(|x| x.as_str());
    match rt {
        None | Some("default") => Ok(Rope::Default { theta }),
        Some("llama3") => {
            let r = rs.unwrap();
            let g = |k: &str| as_f64(r, k).with_context(|| format!("llama3 rope missing {k}"));
            Ok(Rope::Llama3 {
                theta,
                factor: g("factor")?,
                low: g("low_freq_factor")?,
                high: g("high_freq_factor")?,
                orig: g("original_max_position_embeddings")?,
            })
        }
        Some(t) => bail!("rope scaling type {t} is not supported"),
    }
}

/// Resident offsets of one dense/MoE layer, with the checks of `engine_u_layout` / `engine_moe_layout`.
fn layer_layout(streams: &[PStream], nl: usize, res: &[String], ck: &Checkpoint, moe: bool, dims: (usize, usize, usize, usize, usize)) -> Result<LayerResident> {
    let (hid, hd, nh, nkv, inter) = dims;
    let role_of = |rest: &str| -> Option<&'static str> {
        Some(match rest {
            "input_layernorm.weight" => "ln_in",
            "post_attention_layernorm.weight" => "ln_post",
            "self_attn.q_norm.weight" => "qn",
            "self_attn.k_norm.weight" => "kn",
            "self_attn.q_proj.bias" if !moe => "bq",
            "self_attn.k_proj.bias" if !moe => "bk",
            "self_attn.v_proj.bias" if !moe => "bv",
            "self_attn.o_proj.bias" if !moe => "bo",
            "mlp.gate_proj.bias" if !moe => "bg",
            "mlp.up_proj.bias" if !moe => "bu",
            "mlp.down_proj.bias" if !moe => "bd",
            _ => return None,
        })
    };
    let want = |r: &str| match r {
        "ln_in" | "ln_post" | "bo" | "bd" => hid,
        "qn" | "kn" => hd,
        "bq" => nh * hd,
        "bk" | "bv" => nkv * hd,
        _ => inter,
    };
    let mut layout: Option<(Vec<(&'static str, usize)>, usize)> = None;
    let mut in_layers: HashSet<&String> = HashSet::new();
    for (li, s) in streams.iter().take(nl).enumerate() {
        let want_stems: Vec<&str> = if moe {
            vec!["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate"]
        } else {
            KNOWN_MAT.to_vec()
        };
        ensure!(s.stems == want_stems, "layer {li} matrices {:?} are not {:?}", s.stems, want_stems);
        let pre = format!("{}.", s.mats[0].rsplit_once(".self_attn.").map(|x| x.0).unwrap_or(""));
        let mut cur = Vec::new();
        let mut o = 0;
        for n in res.iter().filter(|n| n.starts_with(&pre)) {
            in_layers.insert(n);
            let r = role_of(&n[pre.len()..]).with_context(|| format!("unsupported per-layer tensor {}", &n[pre.len()..]))?;
            let cnt = ck.tensors[n].numel();
            ensure!(cnt == want(r), "{n} has {cnt} values, the engine expects {}", want(r));
            cur.push((r, o));
            o += cnt;
        }
        match &layout {
            None => layout = Some((cur, o)),
            Some(l) => ensure!(*l == (cur, o), "layer {li} resident layout differs from layer 0"),
        }
    }
    let (cur, len) = layout.context("no layers")?;
    let get = |r: &str| cur.iter().find(|x| x.0 == r).map(|x| x.1);
    let tail: Vec<&String> = res.iter().filter(|n| !in_layers.contains(n)).collect();
    ensure!(tail.len() == 1 && tail[0].ends_with("norm.weight"), "expected exactly one final norm outside the layers, found {tail:?}");
    ensure!(get("qn").is_some() == get("kn").is_some(), "q_norm without k_norm (or vice versa)");
    Ok(LayerResident {
        len,
        ln_in: get("ln_in").context("missing input_layernorm")?,
        ln_post: get("ln_post").context("missing post_attention_layernorm")?,
        q_norm: get("qn"),
        k_norm: get("kn"),
        bq: get("bq"),
        bk: get("bk"),
        bv: get("bv"),
        bo: get("bo"),
        bg: get("bg"),
        bu: get("bu"),
        bd: get("bd"),
        sinks: None,
        router_bias: None,
    })
}

fn copy_files(dir: &Path) -> Vec<([u8; 4], Vec<u8>)> {
    FILE_SECTIONS.iter().filter_map(|(t, n)| std::fs::read(dir.join(n)).ok().map(|b| (**t, b))).collect()
}

struct Writer {
    f: std::io::BufWriter<std::fs::File>,
    pos: u64,
    streams: Vec<StreamInfo>,
    raw: u64,
    stored: u64,
}

impl Writer {
    fn new(path: &Path) -> Result<Writer> {
        Ok(Writer { f: std::io::BufWriter::with_capacity(64 << 20, std::fs::File::create(path)?), pos: 0, streams: vec![], raw: 0, stored: 0 })
    }
    fn put(&mut self, subs: &[Vec<u8>], raw: u64, role: Role, layer: Option<usize>, expert: Option<usize>) -> Result<()> {
        let refs: Vec<&[u8]> = subs.iter().map(|b| b.as_slice()).collect();
        let (buf, size) = stream::build(&refs);
        self.f.write_all(&buf)?;
        self.streams.push(StreamInfo {
            e: StreamEntry { off: self.pos, size, padded: buf.len() as u64, raw, hash: xxh3(&buf) },
            hash_kind: HashKind::Xxh3,
            role,
            nsub: subs.len() as u16,
            layer: layer.map_or(u32::MAX, |x| x as u32),
            expert: expert.map_or(u32::MAX, |x| x as u32),
        });
        self.pos += buf.len() as u64;
        self.raw += raw;
        self.stored += size;
        Ok(())
    }
    fn finish(mut self) -> Result<(Vec<StreamInfo>, u64, u64)> {
        self.f.flush()?;
        self.f.get_ref().sync_all()?;
        Ok((self.streams, self.raw, self.stored))
    }
}

fn encode(w: &[u16], rows: usize, cols: usize, codec: Codec) -> Result<Vec<u8>> {
    match codec {
        Codec::Huff4 => huff4::compress(w, rows, cols),
        Codec::Rans4 => rans4::encode(w, rows, cols, 12, 16),
    }
}

/// Pack a dense or Qwen3-MoE checkpoint.
pub fn pack(model: &Path, out: &Path, opt: &Options, log: &dyn Fn(&str)) -> Result<Report> {
    let ck = Checkpoint::open(model)?;
    let tc = text_config(&ck.config).clone();
    if tc.get("model_type").and_then(|x| x.as_str()) == Some("gpt_oss") {
        return pack_gptoss(&ck, out, opt, log);
    }
    std::fs::create_dir_all(out)?;
    let (mut streams, res_names, nl) = plan_streams(&ck)?;
    let g = |k: &str| as_usize(&tc, k);
    let hid = g("hidden_size").context("hidden_size")?;
    let heads = g("num_attention_heads").context("num_attention_heads")?;
    let kv = g("num_key_value_heads").unwrap_or(heads);
    let hd = g("head_dim").unwrap_or(hid / heads);
    let tie = tc.get("tie_word_embeddings").and_then(|x| x.as_bool()).unwrap_or(false);
    let mut tensors: Vec<TensorRef> = Vec::new();
    let mut aliases: Vec<(String, String)> = Vec::new();
    if tie {
        if let Some(hi) = streams.iter().position(|s| s.kind == "lm_head") {
            let h = streams[hi].mats[0].clone();
            if let Some(e) = streams.iter().find(|s| s.kind == "embed").map(|s| s.mats[0].clone()) {
                let (th, te) = (ck.get(&h)?, ck.get(&e)?);
                if th.shape == te.shape && th.dtype == te.dtype && ck.bytes(th) == ck.bytes(te) {
                    streams.remove(hi);
                    log(&format!("tie_word_embeddings: {h} is byte-identical to {e}; stored once"));
                    aliases.push((h, e));
                }
            }
        }
    }
    let moe = streams.iter().any(|s| s.kind == "expert");
    let act = tc.get("hidden_act").and_then(|x| x.as_str()).unwrap_or("silu");
    ensure!(act == "silu" || act == "swish", "activation {act}: the engine has SiLU only");
    ensure!(!tc.get("use_sliding_window").and_then(|x| x.as_bool()).unwrap_or(false), "sliding-window attention is not supported for this architecture");
    let prf = as_f64(&tc, "partial_rotary_factor").unwrap_or(1.0);
    ensure!(prf == 1.0, "partial rotary embeddings are not supported");
    let inter = g("intermediate_size").unwrap_or(0);
    let lr = layer_layout(&streams, nl, &res_names, &ck, moe, (hid, hd, heads, kv, inter))?;
    let ne = streams.iter().filter(|s| s.kind != "expert").count();
    let n_ex = streams.len() - ne;
    let moe_meta = if moe {
        let e = g("num_experts").or_else(|| g("num_local_experts")).context("num_experts")?;
        ensure!(n_ex == nl * e, "{n_ex} expert streams, expected {nl} x {e}");
        ensure!(tc.get("mlp_only_layers").and_then(|x| x.as_array()).map_or(true, |a| a.is_empty()) && g("decoder_sparse_step").unwrap_or(1) == 1, "mixed dense/MoE layers are not supported");
        ensure!(g("shared_expert_intermediate_size").unwrap_or(0) == 0, "shared experts are not supported");
        Some(Moe {
            experts: e,
            top_k: g("num_experts_per_tok").context("num_experts_per_tok")?,
            inter: g("moe_intermediate_size").context("moe_intermediate_size")?,
            norm_topk: tc.get("norm_topk_prob").and_then(|x| x.as_bool()).unwrap_or(false),
            expert_stream0: ne,
        })
    } else {
        None
    };
    let has_head = streams.iter().any(|s| s.kind == "lm_head");
    ensure!(ne == nl + 1 + usize::from(has_head), "extra non-layer matrices are not supported");
    ensure!(tie || has_head, "untied model without an lm_head");
    ensure!(streams.get(nl).map(|s| s.kind) == Some("embed"), "no embed_tokens stream after the layers");
    let mut lossy = false;
    let mut w = Writer::new(&out.join(cube_vault::DATA))?;
    let total = streams.len();
    for (si, s) in streams.iter().enumerate() {
        let infos: Vec<&TensorInfo> = s.mats.iter().map(|n| ck.get(n)).collect::<Result<_>>()?;
        lossy |= infos.iter().any(|t| t.dtype != "BF16");
        let blobs: Vec<Vec<u8>> = infos
            .par_iter()
            .map(|t| {
                let bits = ck.bf16_bits(t, opt.convert)?;
                let (r, c) = t.matrix();
                encode(&bits, r, c, opt.codec)
            })
            .collect::<Result<_>>()?;
        let raw: u64 = infos.iter().map(|t| 2 * t.numel() as u64).sum();
        for (k, t) in infos.iter().enumerate() {
            tensors.push(TensorRef {
                name: t.name.clone(),
                shape: t.shape.clone(),
                src_dtype: t.dtype.clone(),
                store: match opt.codec {
                    Codec::Huff4 => "huff4".into(),
                    Codec::Rans4 => "rans4".into(),
                },
                stream: Some(si),
                sub: Some(k),
                resident: None,
                alias_of: None,
            });
        }
        let role = match s.kind {
            "layer" => Role::Layer,
            "embed" => Role::Embed,
            "lm_head" => Role::LmHead,
            "expert" => Role::Expert,
            _ => Role::Extra,
        };
        w.put(&blobs, raw, role, s.layer, s.expert)?;
        log(&format!("stream {}/{total}", si + 1));
    }
    let (sinfo, raw, stored) = w.finish()?;
    for (h, e) in aliases {
        let t = ck.get(&h)?;
        tensors.push(TensorRef { name: h, shape: t.shape.clone(), src_dtype: t.dtype.clone(), store: "alias".into(), stream: None, sub: None, resident: None, alias_of: Some(e) });
    }
    let mut resident = Vec::new();
    for n in &res_names {
        let t = ck.get(n)?;
        let v = ck.f32(t)?;
        tensors.push(TensorRef { name: n.clone(), shape: t.shape.clone(), src_dtype: t.dtype.clone(), store: "resident".into(), stream: None, sub: None, resident: Some((resident.len(), v.len())), alias_of: None });
        resident.extend(v);
    }
    let meta = ModelMeta {
        format: 8,
        arch: if moe { Arch::QwenMoe } else { Arch::Dense },
        codec: opt.codec,
        hidden: hid,
        inter: if moe { moe_meta.as_ref().unwrap().inter } else { inter },
        layers: nl,
        heads,
        kv_heads: kv,
        head_dim: hd,
        vocab: g("vocab_size").context("vocab_size")?,
        tie_embeddings: tie,
        eos: eos_ids(model, &tc),
        rms_eps: as_f64(&tc, "rms_norm_eps").unwrap_or(1e-6),
        rope: rope_of(&tc)?,
        layer_resident: lr,
        moe: moe_meta,
        gptoss: None,
        embed_stream: nl,
        lm_head_stream: if has_head { nl + 1 } else { nl },
        lossy,
        source: Source { repo: opt.repo.clone(), packer: format!("cube-pack {}", env!("CARGO_PKG_VERSION")), converted_from: None },
        tensors,
    };
    ensure!(resident.len() == meta.resident_len(), "resident tensors: {} values, layout needs {}", resident.len(), meta.resident_len());
    let m = Manifest { flags: 0, meta, streams: sinfo, resident, rope_inv: vec![], files: copy_files(model) };
    m.write(&out.join(MANIFEST))?;
    Ok(Report { streams: total, raw, stored, lossy, arch: m.meta.arch })
}

/// YaRN inverse frequencies and attention factor, computed in f32 the way transformers does.
pub fn yarn(tc: &Value, head_dim: usize) -> Result<(Vec<f32>, f64)> {
    let rp = tc.get("rope_scaling").filter(|x| x.is_object()).or_else(|| tc.get("rope_parameters").filter(|x| x.is_object())).context("no rope_scaling")?;
    let base = as_f64(tc, "rope_theta").or_else(|| as_f64(rp, "rope_theta")).context("rope_theta")?;
    let factor = as_f64(rp, "factor").context("factor")?;
    let orig = as_f64(rp, "original_max_position_embeddings").context("original_max_position_embeddings")?;
    let get_mscale = |s: f64, m: f64| if s <= 1.0 { 1.0 } else { 0.1 * m * s.ln() + 1.0 };
    let att = match as_f64(rp, "attention_factor") {
        Some(a) => a,
        None => match (as_f64(rp, "mscale"), as_f64(rp, "mscale_all_dim")) {
            (Some(m), Some(ma)) if m != 0.0 && ma != 0.0 => get_mscale(factor, m) / get_mscale(factor, ma),
            _ => get_mscale(factor, 1.0),
        },
    };
    let bf = as_f64(rp, "beta_fast").filter(|x| *x != 0.0).unwrap_or(32.0);
    let bs = as_f64(rp, "beta_slow").filter(|x| *x != 0.0).unwrap_or(1.0);
    let dim = head_dim;
    let corr = |rot: f64| (dim as f64 * (orig / (rot * 2.0 * std::f64::consts::PI)).ln()) / (2.0 * base.ln());
    let truncate = rp.get("truncate").and_then(|x| x.as_bool()).unwrap_or(true);
    let (mut low, mut high) = (corr(bf), corr(bs));
    if truncate {
        low = low.floor();
        high = high.ceil();
    }
    let (low, mut high) = (low.max(0.0), high.min(dim as f64 - 1.0));
    if low == high {
        high += 0.001;
    }
    let half = dim / 2;
    let mut inv = Vec::with_capacity(half);
    for i in 0..half {
        let e = (2 * i) as f32 / dim as f32;
        let pos = (base as f32).powf(e);
        let ext = 1.0f32 / pos;
        let itp = 1.0f32 / (factor as f32 * pos);
        let ramp = ((i as f32 - low as f32) / ((high - low) as f32)).clamp(0.0, 1.0);
        let ef = 1.0f32 - ramp;
        inv.push(itp * (1.0f32 - ef) + ext * ef);
    }
    Ok((inv, att))
}

/// Pack an OpenAI gpt-oss checkpoint (bf16 attention/router/embeddings huffman4, MXFP4 experts as MXF2 or MXF4).
fn pack_gptoss(ck: &Checkpoint, out: &Path, opt: &Options, log: &dyn Fn(&str)) -> Result<Report> {
    std::fs::create_dir_all(out)?;
    let c = &ck.config;
    let g = |k: &str| as_usize(c, k).with_context(|| format!("config: {k}"));
    let (nl, ne, k) = (g("num_hidden_layers")?, g("num_local_experts")?, g("num_experts_per_tok")?);
    let (hid, nh, nkv, hd, inter) = (g("hidden_size")?, g("num_attention_heads")?, g("num_key_value_heads")?, g("head_dim")?, g("intermediate_size")?);
    let mut w = Writer::new(&out.join(cube_vault::DATA))?;
    let mut resident: Vec<f32> = Vec::new();
    let huff = |name: &str| -> Result<(Vec<u8>, u64)> {
        let t = ck.get(name)?;
        ensure!(t.dtype == "BF16", "{name} is {}, expected BF16", t.dtype);
        let (r, cc) = t.matrix();
        Ok((encode(&ck.bf16_bits(t, false)?, r, cc, opt.codec)?, 2 * t.numel() as u64))
    };
    for l in 0..nl {
        let p = format!("model.layers.{l}.");
        let parts: Vec<(Vec<u8>, u64)> = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.router"]
            .par_iter()
            .map(|m| huff(&format!("{p}{m}.weight")))
            .collect::<Result<_>>()?;
        let raw = parts.iter().map(|x| x.1).sum();
        w.put(&parts.into_iter().map(|x| x.0).collect::<Vec<_>>(), raw, Role::Layer, Some(l), None)?;
        for n in ["input_layernorm.weight", "post_attention_layernorm.weight", "self_attn.q_proj.bias", "self_attn.k_proj.bias", "self_attn.v_proj.bias", "self_attn.o_proj.bias", "self_attn.sinks", "mlp.router.bias"] {
            resident.extend(ck.f32(ck.get(&format!("{p}{n}"))?)?);
        }
        log(&format!("layer streams {}/{nl}", l + 1));
    }
    for (n, role) in [("model.embed_tokens.weight", Role::Embed), ("lm_head.weight", Role::LmHead)] {
        let (b, r) = huff(n)?;
        w.put(&[b], r, role, None, None)?;
    }
    let e0 = w.streams.len();
    for l in 0..nl {
        let p = format!("model.layers.{l}.mlp.experts.");
        let t = |s: &str| ck.get(&format!("{p}{s}"));
        let (gb, gs, gbi, db, ds, dbi) = (t("gate_up_proj_blocks")?, t("gate_up_proj_scales")?, t("gate_up_proj_bias")?, t("down_proj_blocks")?, t("down_proj_scales")?, t("down_proj_bias")?);
        ensure!(gb.dtype == "U8" && gs.dtype == "U8" && db.dtype == "U8" && ds.dtype == "U8", "expert tensors are not MXFP4 (U8 blocks/scales)");
        let (gbv, gsv, dbv, dsv) = (ck.bytes(gb), ck.bytes(gs), ck.bytes(db), ck.bytes(ds));
        let (gbias, dbias) = (ck.f32(gbi)?, ck.f32(dbi)?);
        let (grow, gnb) = (gs.shape[1], gs.shape[2]);
        let (drow, dnb) = (ds.shape[1], ds.shape[2]);
        let per = |e: usize| -> Result<(Vec<Vec<u8>>, u64)> {
            let gbl = &gbv[e * grow * gnb * 16..(e + 1) * grow * gnb * 16];
            let gsc = &gsv[e * grow * gnb..(e + 1) * grow * gnb];
            let dbl = &dbv[e * drow * dnb * 16..(e + 1) * drow * dnb * 16];
            let dsc = &dsv[e * drow * dnb..(e + 1) * drow * dnb];
            let mx = |b: &[u8], s: &[u8], r: usize, nb: usize| -> Result<Vec<u8>> {
                if opt.mxf2 { mxf::mxf2_blob(b, s, r, nb) } else { Ok(mxf::mxf4_blob(b, s, r, nb)) }
            };
            let gl = gbias.len() / ne;
            let dl = dbias.len() / ne;
            let subs = vec![
                mx(gbl, gsc, grow, gnb)?,
                mxf::f32_blob(&gbias[e * gl..(e + 1) * gl]),
                mx(dbl, dsc, drow, dnb)?,
                mxf::f32_blob(&dbias[e * dl..(e + 1) * dl]),
            ];
            Ok((subs, (gbl.len() + gsc.len() + dbl.len() + dsc.len()) as u64))
        };
        let all: Vec<(Vec<Vec<u8>>, u64)> = (0..ne).into_par_iter().map(per).collect::<Result<_>>()?;
        for (e, (subs, raw)) in all.into_iter().enumerate() {
            w.put(&subs, raw, Role::Expert, Some(l), Some(e))?;
        }
        log(&format!("expert layers {}/{nl}", l + 1));
    }
    resident.extend(ck.f32(ck.get("model.norm.weight")?)?);
    let (rope_inv, ascale) = yarn(c, hd)?;
    let ls: Vec<u8> = match c.get("layer_types").and_then(|x| x.as_array()) {
        Some(a) => a.iter().map(|x| u8::from(x.as_str() == Some("sliding_attention"))).collect(),
        None => vec![0; nl],
    };
    let mut eos = ids_of(c.get("eos_token_id"));
    if let Ok(b) = std::fs::read(ck.dir.join("generation_config.json")) {
        let gc: Value = serde_json::from_slice(&b)?;
        if gc.get("eos_token_id").is_some() {
            eos = ids_of(gc.get("eos_token_id"));
        }
    }
    let lres = 2 * hid + nh * hd + 2 * nkv * hd + hid + nh + ne;
    let bo = 2 * hid + nh * hd + 2 * nkv * hd;
    let (sinfo, raw, stored) = w.finish()?;
    let meta = ModelMeta {
        format: 8,
        arch: Arch::Gptoss,
        codec: opt.codec,
        hidden: hid,
        inter,
        layers: nl,
        heads: nh,
        kv_heads: nkv,
        head_dim: hd,
        vocab: g("vocab_size")?,
        tie_embeddings: false,
        eos,
        rms_eps: as_f64(c, "rms_norm_eps").unwrap_or(1e-5),
        rope: Rope::Table { scale: ascale },
        layer_resident: LayerResident {
            len: lres,
            ln_in: 0,
            ln_post: hid,
            bq: Some(2 * hid),
            bk: Some(2 * hid + nh * hd),
            bv: Some(2 * hid + nh * hd + nkv * hd),
            bo: Some(bo),
            sinks: Some(bo + hid),
            router_bias: Some(bo + hid + nh),
            ..Default::default()
        },
        moe: Some(Moe { experts: ne, top_k: k, inter, norm_topk: false, expert_stream0: e0 }),
        gptoss: Some(Gptoss {
            sliding_window: as_usize(c, "sliding_window").unwrap_or(0),
            layer_sliding: ls,
            swiglu_limit: as_f64(c, "swiglu_limit").unwrap_or(7.0),
            swiglu_alpha: 1.702,
        }),
        embed_stream: nl,
        lm_head_stream: nl + 1,
        lossy: false,
        source: Source { repo: opt.repo.clone(), packer: format!("cube-pack {}", env!("CARGO_PKG_VERSION")), converted_from: None },
        tensors: vec![],
    };
    ensure!(resident.len() == meta.resident_len(), "gpt-oss resident: {} values, layout needs {}", resident.len(), meta.resident_len());
    let m = Manifest { flags: 0, meta, streams: sinfo, resident, rope_inv, files: copy_files(&ck.dir) };
    m.write(&out.join(MANIFEST))?;
    Ok(Report { streams: m.streams.len(), raw, stored, lossy: false, arch: Arch::Gptoss })
}

/// Compare every tensor of a vault with the source checkpoint (decoded bits vs source bits; resident f32 vs source).
pub fn verify_against_source(vault: &cube_vault::Vault, model: &Path) -> Result<(usize, usize)> {
    let ck = Checkpoint::open(model)?;
    let m = vault.meta();
    ensure!(!m.tensors.is_empty(), "this vault has no tensor table");
    let bad = std::sync::atomic::AtomicUsize::new(0);
    m.tensors.par_iter().try_for_each(|t| -> Result<()> {
        let src = ck.get(&t.name)?;
        let ok = match t.store.as_str() {
            "resident" => {
                let (o, n) = t.resident.context("resident offset")?;
                ck.f32(src)?.iter().map(|x| x.to_bits()).eq(vault.resident()[o..o + n].iter().map(|x| x.to_bits()))
            }
            "huff4" | "rans4" | "alias" => {
                let tr = if t.store == "alias" { m.tensors.iter().find(|x| Some(&x.name) == t.alias_of.as_ref()).context("alias target")? } else { t };
                let buf = vault.read_stream(tr.stream.context("stream")?)?;
                let subs = stream::sub_blobs(&buf)?;
                let blob = subs[tr.sub.context("sub")?];
                let dec = if m.codec == Codec::Huff4 { huff4::Huff4::parse(blob)?.decode_bits() } else { rans4::Rans4::parse(blob)?.decode_bits() };
                dec == ck.bf16_bits(src, true)?
            }
            _ => true,
        };
        if !ok {
            eprintln!("MISMATCH {}", t.name);
            bad.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        Ok(())
    })?;
    Ok((m.tensors.len(), bad.into_inner()))
}
