//! The model's own recommended generation settings (model card values per family, then the vault's generation_config.json),
//! a context length sized to an affordable KV cache, and the settings the Models tab tuner measured (tuned.json).

use cube_vault::meta::Arch;
use cube_vault::Vault;
use serde_json::{json, Map, Value as J};
use std::path::Path;

pub fn recommended(vault_dir: &Path) -> anyhow::Result<(Map<String, J>, Vec<String>)> {
    let v = Vault::open(vault_dir)?;
    let m = v.meta().clone();
    let jl = |n: &str| -> J { v.file(n).and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or(J::Null) };
    let mut cfgj = jl("config.json");
    if cfgj.is_null() {
        // a v3-v7 vault keeps the source config inside vault.json
        cfgj = std::fs::read(vault_dir.join("vault.json")).ok().and_then(|b| serde_json::from_slice::<J>(&b).ok()).map(|x| x["config"].clone()).unwrap_or(J::Null);
    }
    let gen = jl("generation_config.json");
    let name = format!("{} {}", cfgj["_name_or_path"].as_str().unwrap_or(""), vault_dir.file_name().unwrap_or_default().to_string_lossy()).to_lowercase();
    let mtype = cfgj["model_type"].as_str().unwrap_or("").to_lowercase();
    let tmpl = v.file("chat_template.jinja").map(|b| String::from_utf8_lossy(&b).into_owned()).unwrap_or_default() + jl("tokenizer_config.json")["chat_template"].as_str().unwrap_or("");
    let moe = m.moe.is_some();
    let gptoss = m.arch == Arch::Gptoss || mtype == "gpt_oss";
    let qwen = mtype.starts_with("qwen3") || name.contains("qwen3");
    let hermes = name.contains("hermes");
    let think = gptoss || tmpl.contains("enable_thinking") || tmpl.contains("<think>");
    let mut notes: Vec<String> = Vec::new();
    let mut s: Map<String, J>;
    if gptoss {
        s = json!({"temp": 1.0, "top_p": 1.0, "top_k": 0, "max_new": 2048, "ctx": 8192}).as_object().unwrap().clone();
        notes.push("gpt-oss: OpenAI recommends temperature 1.0, top_p 1.0, top_k 0 (off; this engine caps top-k at 128).".into());
        notes.push("It reasons before every answer, so replies need room: max new 2048.".into());
        notes.push("OpenAI suggests >= 16k context: the context is at least that, longer when RAM is left after the expert cache.".into());
    } else if hermes {
        s = json!({"temp": 0.6, "top_p": 0.95, "top_k": 20, "max_new": 2048, "ctx": 8192}).as_object().unwrap().clone();
        notes.push("Hermes 4 (model card): temperature 0.6, top_p 0.95, top_k 20; reasons inside <think>.".into());
    } else if qwen {
        s = if think { json!({"temp": 0.6, "top_p": 0.95, "top_k": 20, "ctx": 8192}) } else { json!({"temp": 0.7, "top_p": 0.8, "top_k": 20, "ctx": 8192}) }.as_object().unwrap().clone();
        notes.push("Qwen3 (model card): thinking 0.6 / 0.95 / top-k 20, non-thinking 0.7 / 0.8 / 20; never greedy with thinking on (it can loop).".into());
    } else {
        s = json!({"temp": 0.7, "top_p": 0.9, "top_k": 40, "ctx": 4096}).as_object().unwrap().clone();
    }
    let mut shipped = Vec::new();
    for (k, g) in [("temp", "temperature"), ("top_p", "top_p"), ("top_k", "top_k")] {
        if let Some(x) = gen.get(g).filter(|x| !x.is_null()) {
            s.insert(k.into(), x.clone());
            shipped.push(format!("{k}={x}"));
        }
    }
    if !shipped.is_empty() {
        notes.push(format!("Sampling from the model's own generation_config.json: {}.", shipped.join(", ")));
    } else if !(gptoss || hermes || qwen) {
        notes.push("No recommendation shipped with this model: generic 0.7 / 0.9 / 40.".into());
    }
    if !s.contains_key("max_new") {
        s.insert("max_new".into(), json!(if think { 2048 } else { gen["max_new_tokens"].as_i64().unwrap_or(1024) }));
    }
    // context from the real KV cost: full-attention layers only, fp16 for gpt-oss
    let sliding: usize = m.gptoss.as_ref().map_or(0, |g| g.layer_sliding.iter().filter(|&&x| x == 1).count());
    let full = m.layers - sliding;
    let kv16 = gptoss;
    let bpe = if kv16 { 2 } else { 4 };
    let per_tok = (2 * full * m.kv_heads * m.head_dim * bpe) as f64;
    let native = cfgj["max_position_embeddings"].as_i64().unwrap_or(32768).max(2048) as f64;
    let ex_gb: f64 = m.moe.as_ref().map_or(0.0, |mo| (mo.expert_stream0..v.n_streams()).map(|i| v.stream(i).padded as f64).sum::<f64>() / 1e9);
    let kv_of = |ctx: usize| crate::hw::kv_bytes(&m, ctx, kv16);
    let (cap, fits) = if let Some(mo) = m.moe.as_ref() {
        // MoE: the context gets the RAM left after the expert cache can hold every expert the VRAM tier cannot
        // (Session sizes the cache as (free RAM - 1.5 GB - KV) x 0.8 - pinned attention)
        let nx = m.layers * mo.experts;
        let xsz = (0..nx).map(|i| v.stream(mo.expert_stream0 + i).padded).max().unwrap_or(1).max(1) as f64;
        let gpu = gptoss && cube_gpu::gpu().is_some();
        let vfree = if gpu { cube_gpu::vram().map_or(0.0, |x| x.0 as f64) } else { 0.0 };
        let head = if gpu && cube_gpu::GpuHead::vram_needed(&v) as f64 + 600e6 < vfree { cube_gpu::GpuHead::vram_needed(&v) as f64 } else { 0.0 };
        let stride = ((xsz as u64 + 255) & !255) as f64;
        let vslots = if gpu { ((vfree - 600e6 - head).max(0.0) / stride).floor().min(nx as f64) } else { 0.0 };
        let ram_need = (nx as f64 - vslots) * xsz;
        let mut pinned: f64 = (0..m.layers).map(|i| v.stream(i).padded as f64).sum();
        if head == 0.0 && m.lm_head_stream != m.embed_stream {
            pinned += v.stream(m.lm_head_stream).padded as f64;
        }
        let ram_mb = crate::hw::ram_mb().1;
        let kv_max = ((ram_mb - 1500.0) * 1e6 - (ram_need + pinned) / 0.8).min(3.5e9);
        let floor = if gptoss { 16384 } else { 2048 };
        let mut c = floor;
        while c + 1024 <= native as usize && kv_of(c + 1024) <= kv_max {
            c += 1024;
        }
        let fits = kv_of(floor) <= kv_max;
        notes.push(format!(
            "Experts: {} of {nx} fit in VRAM{}, the RAM cache needs {:.1} GB for the rest; {:.1} GB of free RAM now.",
            vslots as usize,
            if head > 0.0 { " (after the output layer)" } else { "" },
            ram_need / 1e9,
            ram_mb / 1e3
        ));
        (c as i64, fits)
    } else {
        let budget = 2.0e9;
        ((2048f64).max(native.min(((budget / per_tok) as i64 / 1024 * 1024) as f64)) as i64, true)
    };
    s.insert("ctx".into(), json!(cap));
    s.insert("kv16".into(), json!(kv16));
    notes.push(format!(
        "Context {} tokens (model maximum {}): KV cache {}, {:.0} KB per token{}, {:.1} GB at full length.",
        crate::session::fmt_int(cap),
        crate::session::fmt_int(native as i64),
        if kv16 { "fp16" } else { "fp32" },
        per_tok / 1024.0,
        if sliding > 0 { ", sliding-window layers excluded" } else { "" },
        per_tok * cap as f64 / 1e9
    ));
    if moe && !fits {
        notes.push(format!("Not every expert fits in RAM + VRAM even at the minimum context, so some are read from the SSD: every GB given to the context removes ~{:.0} cached experts. Raise Context only if you need it.", 0.8 * (m.layers * m.moe.as_ref().map_or(1, |x| x.experts)) as f64 / ex_gb.max(1e-9)));
    } else if moe {
        notes.push("The context is as long as RAM allows while every expert stays cached (RAM + VRAM); a longer context evicts experts and slows decoding.".into());
    }
    let mn = s["max_new"].as_i64().unwrap_or(2048).min(cap / 2);
    s.insert("max_new".into(), json!(mn));
    if gptoss {
        let x = if fits { 8192 } else { 4096 };
        s.insert("max_new".into(), json!(x));
        notes.push(format!("Max reply {} tokens (large files are written in 3,000-character tool calls, so this mostly bounds reasoning + answer).", crate::session::fmt_int(x)));
    }
    s.insert("think".into(), json!(think));
    s.insert("greedy".into(), json!(false));
    s.insert("mode".into(), json!("auto"));
    s.insert("draft".into(), json!(if moe { "none" } else { "auto" }));
    notes.push(if think { format!("Thinking: on ({}).", if gptoss { "harmony analysis channel" } else { "<think>" }) } else { "Thinking: off (this model's template has no reasoning mode).".into() });
    if let Ok(b) = std::fs::read(vault_dir.join("tuned.json")) {
        if let Ok(t) = serde_json::from_slice::<J>(&b) {
            if let Some(o) = t["settings"].as_object() {
                for (k, x) in o {
                    s.insert(k.clone(), x.clone());
                }
                notes.push(format!("Tuned on this machine ({}): {} tok/s with {} - re-run Tune in the Models tab after hardware changes.", t["date"].as_str().unwrap_or(""), t["tok_s"], t["settings"]));
            }
        }
    }
    let mut all = vec![format!("Recommended settings for {}:", vault_dir.file_name().unwrap_or_default().to_string_lossy())];
    all.extend(notes);
    Ok((s, all))
}
