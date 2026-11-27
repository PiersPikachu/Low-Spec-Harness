//! Hugging Face models this harness can run on THIS machine: search, assess (architecture -> weights format -> compressed
//! size -> disk -> memory tier -> speed -> verdict), install (download -> pack losslessly -> check every stream -> tune),
//! and tune (short benchmark of the candidate engine settings, the fastest sane one saved to <vault>/tuned.json).

use crate::paths::{default_model_dir, Config, Paths};
use anyhow::{anyhow, bail, Context, Result};
use serde_json::{json, Value as J};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

pub const DENSE: [&str; 4] = ["llama", "qwen2", "qwen3", "mistral"];
pub const MOE: [&str; 2] = ["qwen3_moe", "gpt_oss"];
const K_MOE_MXF: f64 = 0.065; // s per billion active parameters, MXFP4 experts
const K_MOE_BF16: f64 = 0.136; // s per billion active parameters, huffman bf16 experts (CPU)
const GPU_GBPS: f64 = 45e9;

fn bytes_of(dt: &str) -> f64 {
    match dt {
        "BF16" | "F16" => 2.0,
        "F32" | "I32" => 4.0,
        "U8" | "I8" | "F8_E4M3" => 1.0,
        "I64" => 8.0,
        _ => 2.0,
    }
}

pub fn token() -> Option<String> {
    if let Ok(t) = std::env::var("HF_TOKEN") {
        if !t.trim().is_empty() {
            return Some(t.trim().to_string());
        }
    }
    let home = std::env::var("USERPROFILE").or_else(|_| std::env::var("HOME")).ok()?;
    let p = PathBuf::from(home).join(".cache").join("huggingface").join("token");
    std::fs::read_to_string(p).ok().map(|s| s.trim().to_string()).filter(|s| !s.is_empty())
}

fn agent() -> ureq::Agent {
    ureq::AgentBuilder::new().timeout_connect(Duration::from_secs(20)).timeout_read(Duration::from_secs(120)).try_proxy_from_env(true).build()
}

fn get(url: &str) -> Result<ureq::Response> {
    let mut r = agent().get(url).set("User-Agent", "cube-harness-rs/0.1");
    if let Some(t) = token() {
        r = r.set("Authorization", &format!("Bearer {t}"));
    }
    r.call().map_err(|e| match e {
        ureq::Error::Status(c, resp) => anyhow!("HTTP {c} for {url}: {}", resp.into_string().unwrap_or_default().chars().take(200).collect::<String>()),
        e => anyhow!("{url}: {e}"),
    })
}

fn get_json(url: &str) -> Result<J> {
    Ok(get(url)?.into_json()?)
}

#[derive(Clone, Debug, Default)]
pub struct Found {
    pub repo: String,
    pub downloads: u64,
    pub likes: u64,
    pub gated: bool,
    pub params: Option<f64>,
}

/// Text-generation models, most downloaded first.
pub fn search(query: &str, limit: usize) -> Result<Vec<Found>> {
    let q: String = query.chars().map(|c| if c.is_ascii_alphanumeric() || "-_.".contains(c) { c.to_string() } else { format!("%{:02X}", c as u32 & 0xFF) }).collect();
    let url = format!(
        "https://huggingface.co/api/models?search={q}&pipeline_tag=text-generation&sort=downloads&direction=-1&limit={limit}&expand[]=downloads&expand[]=likes&expand[]=gated&expand[]=safetensors"
    );
    let v = get_json(&url)?;
    Ok(v.as_array()
        .cloned()
        .unwrap_or_default()
        .into_iter()
        .map(|m| Found {
            repo: m["id"].as_str().unwrap_or("").to_string(),
            downloads: m["downloads"].as_u64().unwrap_or(0),
            likes: m["likes"].as_u64().unwrap_or(0),
            gated: m["gated"].as_bool().unwrap_or(false) || m["gated"].is_string(),
            params: m["safetensors"]["total"].as_f64(),
        })
        .collect())
}

#[derive(Clone, Debug)]
pub struct Hw {
    pub ram_w: f64,
    pub vram: f64,
    pub vault_dir: PathBuf,
    pub dl_dir: PathBuf,
    pub vault_free: f64,
    pub dl_free: f64,
    pub disk_bps: f64,
    pub cpu_bps: f64,
}

pub fn hardware(p: &Paths, c: &Config) -> Hw {
    let (tot, _) = crate::hw::ram_mb();
    let gpu = crate::hw::gpu();
    let vd = default_model_dir(p, c);
    let dd = match c.str("hub_download_dir") {
        s if !s.is_empty() && s != "null" => PathBuf::from(s),
        _ => {
            if cfg!(windows) && Path::new("D:/").exists() {
                PathBuf::from("D:\\models")
            } else {
                p.data.join("downloads")
            }
        }
    };
    let _ = std::fs::create_dir_all(&vd);
    let _ = std::fs::create_dir_all(&dd);
    let prof: J = std::fs::read(p.data.join("hw_profile.json")).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or(J::Null);
    Hw {
        ram_w: ((tot - 6500.0) * 1048576.0).max(2e9),
        vram: gpu.map_or(0.0, |g| ((g.vram_free_mb - 900.0) * 1048576.0).max(0.0)),
        vault_free: crate::storage::free_bytes(&vd) as f64,
        dl_free: crate::storage::free_bytes(&dd) as f64,
        vault_dir: vd,
        dl_dir: dd,
        disk_bps: 2.5e9,
        cpu_bps: prof["cpu_decode_GBps"].as_f64().unwrap_or(14.5) * 1e9,
    }
}

#[derive(Clone, Debug, Default)]
pub struct Assessment {
    pub repo: String,
    pub ok: bool,
    /// "dense" | "moe" | "gptoss"; None = cannot be installed
    pub kind: Option<String>,
    pub est_tok_s: Option<f64>,
    pub vault_gb: Option<f64>,
    pub download_gb: Option<f64>,
    pub verdict: String,
    /// (question, answer, "ok" | "warn" | "bad")
    pub tree: Vec<(String, String, String)>,
}

pub fn assess(repo: &str, hw: &Hw) -> Assessment {
    let mut r = Assessment { repo: repo.into(), ..Default::default() };
    let step = |r: &mut Assessment, q: &str, a: String, s: &str| r.tree.push((q.into(), a, s.into()));
    let info = match get_json(&format!("https://huggingface.co/api/models/{repo}?expand[]=safetensors&expand[]=gated&expand[]=config")) {
        Ok(v) => v,
        Err(e) => {
            step(&mut r, "Repository", format!("cannot read it: {e}"), "bad");
            r.verdict = "unavailable".into();
            return r;
        }
    };
    let gated = info["gated"].as_bool().unwrap_or(false) || info["gated"].is_string();
    if gated {
        step(&mut r, "Access", "gated: accept the licence on huggingface.co and put your token in HF_TOKEN (or ~/.cache/huggingface/token) before installing".into(), "warn");
    }
    let st = &info["safetensors"];
    if st.is_null() {
        step(&mut r, "Weights", "no safetensors weights (GGUF-only or PyTorch .bin) - this harness packs safetensors".into(), "bad");
        r.verdict = "not compatible".into();
        return r;
    }
    let c = match get_json(&format!("https://huggingface.co/{repo}/resolve/main/config.json")) {
        Ok(v) => v,
        Err(e) => {
            if gated {
                step(&mut r, "config.json", "hidden until you accept the licence and log in, so compatibility cannot be checked yet".into(), "warn");
                r.verdict = "needs Hugging Face login".into();
            } else {
                step(&mut r, "config.json", format!("missing or unreadable: {}", e.to_string().chars().take(160).collect::<String>()), "bad");
                r.verdict = "not compatible".into();
            }
            return r;
        }
    };
    let c = if c["text_config"].is_object() { c["text_config"].clone() } else { c };
    let mt = c["model_type"].as_str().unwrap_or("?").to_string();
    let q = if c["quantization_config"].is_object() { c["quantization_config"].clone() } else { info["config"]["quantization_config"].clone() };
    let kind = if MOE.contains(&mt.as_str()) {
        if mt == "gpt_oss" { "gptoss" } else { "moe" }
    } else if DENSE.contains(&mt.as_str()) && !(mt == "mistral" && !c["sliding_window"].is_null()) {
        "dense"
    } else {
        let mut sup: Vec<&str> = DENSE.iter().chain(MOE.iter()).copied().collect();
        sup.sort();
        step(&mut r, "Architecture", format!("{mt}: not supported by this engine (supported: {})", sup.join(", ")), "bad");
        r.verdict = "not compatible".into();
        return r;
    };
    step(&mut r, "Architecture", format!("{mt} ({}) - supported", if kind != "dense" { "mixture of experts" } else { "dense decoder" }), "ok");
    let params: Vec<(String, f64)> = st["parameters"].as_object().map(|o| o.iter().map(|(k, v)| (k.clone(), v.as_f64().unwrap_or(0.0))).collect()).unwrap_or_default();
    let total = st["total"].as_f64().unwrap_or_else(|| params.iter().map(|x| x.1).sum());
    let qm = q["quant_method"].as_str().unwrap_or("").to_lowercase();
    if !qm.is_empty() && !(qm == "mxfp4" && kind == "gptoss") {
        step(&mut r, "Weights format", format!("pre-quantized ({qm}): the packer needs bf16/fp16 (or gpt-oss MXFP4) weights - look for the original repo"), "bad");
        r.verdict = "not compatible".into();
        return r;
    }
    // Hugging Face counts gpt-oss MXFP4 tensors (U8) in weights, not bytes: 4.25 bits each on disk
    let raw: f64 = params.iter().map(|(k, n)| n * if k == "U8" && kind == "gptoss" { 4.25 / 8.0 } else { bytes_of(k) }).sum();
    step(&mut r, "Weights format", params.iter().map(|(k, n)| format!("{k} {:.2}B", n / 1e9)).collect::<Vec<_>>().join(", ") + &format!(" = {:.1} GB to download", raw / 1e9), "ok");
    let pget = |k: &str| params.iter().find(|x| x.0 == k).map_or(0.0, |x| x.1);
    let vault = if kind == "gptoss" {
        let v = pget("BF16") * 2.0 * 0.67 + pget("U8") * 4.25 / 8.0 * 0.965;
        step(&mut r, "Compression", format!("bf16 parts huffman4 (lossless, ~67%), MXFP4 experts MXF2 (exact to the release, ~96.5%) -> vault ~{:.1} GB", v / 1e9), "ok");
        v
    } else {
        let v = raw * if params.iter().any(|x| x.0 == "BF16" || x.0 == "F16") { 0.67 } else { 0.335 };
        step(&mut r, "Compression", format!("huffman4, lossless (every matrix verified bit-exact) -> vault ~{:.1} GB (~67% of bf16)", v / 1e9), "ok");
        v
    };
    r.kind = Some(kind.into());
    r.vault_gb = Some(vault / 1e9);
    r.download_gb = Some(raw / 1e9);
    let mut ok_disk = true;
    if hw.dl_free < raw * 1.05 + 2e9 {
        step(&mut r, "Disk: download", format!("{} has {:.0} GB free, needs ~{:.0} GB", hw.dl_dir.display(), hw.dl_free / 1e9, raw / 1e9 * 1.05 + 2.0), "bad");
        ok_disk = false;
    } else {
        step(&mut r, "Disk: download", format!("{}: {:.0} GB free", hw.dl_dir.display(), hw.dl_free / 1e9), "ok");
    }
    if hw.vault_free < vault + 2e9 {
        step(&mut r, "Disk: vault", format!("{} has {:.0} GB free, needs ~{:.0} GB (fast NVMe)", hw.vault_dir.display(), hw.vault_free / 1e9, vault / 1e9 + 2.0), "bad");
        ok_disk = false;
    } else {
        step(&mut r, "Disk: vault", format!("{}: {:.0} GB free", hw.vault_dir.display(), hw.vault_free / 1e9), "ok");
    }
    let l = c["num_hidden_layers"].as_f64().unwrap_or(0.0);
    let hid = c["hidden_size"].as_f64().unwrap_or(0.0);
    let fast = hw.ram_w + hw.vram;
    let (tier, tps);
    if kind == "dense" {
        if vault <= hw.vram - 0.5e9 && crate::gpu_available() {
            tier = "fits in GPU memory".to_string();
            tps = GPU_GBPS / vault;
        } else if vault <= hw.ram_w {
            tier = "fits in RAM (CPU decode)".to_string();
            tps = 0.95 * hw.cpu_bps / vault;
        } else {
            tier = format!("streams {:.1} GB per pass from the SSD", (vault - hw.ram_w) / 1e9);
            let t_pass = (vault - hw.ram_w) / hw.disk_bps + 0.6 * vault / hw.cpu_bps;
            tps = 1.0 / t_pass;
        }
    } else {
        let e = c["num_experts"].as_f64().or_else(|| c["num_local_experts"].as_f64()).unwrap_or(1.0);
        let k = c["num_experts_per_tok"].as_f64().unwrap_or(1.0);
        let i = c["moe_intermediate_size"].as_f64().or_else(|| c["intermediate_size"].as_f64()).unwrap_or(0.0);
        let per_exp = 3.0 * i * hid;
        let ex_total = l * e * per_exp;
        let active = if total > ex_total { total - ex_total + l * k * per_exp } else { total };
        let bpp = if kind == "gptoss" { 4.25 / 8.0 * 0.965 } else { 2.0 * 0.67 };
        let ex_bytes = ex_total * bpp;
        let c_frac = ((fast - (vault - ex_bytes)) / ex_bytes.max(1.0)).clamp(0.0, 1.0);
        let hit = if c_frac >= 1.0 { 1.0 } else if c_frac < 0.525 { 0.95 * (c_frac / 0.525).sqrt() } else { 0.95 + 0.05 * (c_frac - 0.525) / 0.475 };
        let io = (1.0 - hit) * l * k * per_exp * bpp / hw.disk_bps;
        let comp = active / 1e9 * if kind == "gptoss" { K_MOE_MXF } else { K_MOE_BF16 };
        tps = 1.0 / (io.max(comp) + 0.3 * io.min(comp) + 0.05);
        tier = format!("{e:.0} experts/layer, top {k:.0}, ~{:.1}B active; RAM + VRAM hold ~{:.0}% of experts (~{:.0}% hits)", active / 1e9, 100.0 * c_frac, 100.0 * hit);
    }
    step(&mut r, "Memory", tier, if tps >= 0.5 { "ok" } else { "warn" });
    step(&mut r, "Speed (estimate +-30%)", format!("~{tps:.2} tokens/s on this machine"), if tps >= 2.0 { "ok" } else if tps >= 0.5 { "warn" } else { "bad" });
    if let Some(m) = c["max_position_embeddings"].as_i64() {
        step(&mut r, "Context", format!("model maximum {} tokens; the harness sizes it to the KV cache it can afford", crate::session::fmt_int(m)), "ok");
    }
    r.est_tok_s = Some(tps);
    r.ok = ok_disk && tps >= 0.5;
    r.verdict = format!("{}{}", if tps >= 2.0 { "runs well" } else if tps >= 0.5 { "runs slowly" } else { "too slow" }, if ok_disk { "" } else { " - not enough disk space" });
    r
}

fn wanted(path: &str) -> bool {
    let name = path.rsplit('/').next().unwrap_or(path);
    if path.starts_with("original/") || path.starts_with("metal/") || name.ends_with(".pth") {
        return false;
    }
    name.ends_with(".json") || name.ends_with(".safetensors") || name.ends_with(".jinja") || name.starts_with("tokenizer") || name.ends_with(".model") || name.ends_with(".txt")
}

/// Download the files a vault needs (resumable: a partial file continues where it stopped).
pub fn download(repo: &str, dst: &Path, log: &mut dyn FnMut(String), stop: &dyn Fn() -> bool) -> Result<()> {
    let tree = get_json(&format!("https://huggingface.co/api/models/{repo}/tree/main?recursive=1"))?;
    let files: Vec<(String, u64)> = tree
        .as_array()
        .context("file list")?
        .iter()
        .filter(|f| f["type"] == "file")
        .map(|f| (f["path"].as_str().unwrap_or("").to_string(), f["lfs"]["size"].as_u64().or_else(|| f["size"].as_u64()).unwrap_or(0)))
        .filter(|(p, _)| wanted(p))
        .collect();
    let total: u64 = files.iter().map(|f| f.1).sum();
    log(format!("{} files, {:.2} GB", files.len(), total as f64 / 1e9));
    std::fs::create_dir_all(dst)?;
    let mut done_all = 0u64;
    let t0 = Instant::now();
    for (path, size) in files {
        let out = dst.join(&path);
        if let Some(d) = out.parent() {
            std::fs::create_dir_all(d)?;
        }
        let have = std::fs::metadata(&out).map(|m| m.len()).unwrap_or(0);
        if size > 0 && have == size {
            done_all += size;
            continue;
        }
        let url = format!("https://huggingface.co/{repo}/resolve/main/{path}");
        let mut req = agent().get(&url).set("User-Agent", "cube-harness-rs/0.1");
        if let Some(t) = token() {
            req = req.set("Authorization", &format!("Bearer {t}"));
        }
        let resume = have > 0 && have < size;
        if resume {
            req = req.set("Range", &format!("bytes={have}-"));
        }
        let resp = req.call().map_err(|e| anyhow!("download {path}: {e}"))?;
        let partial = resp.status() == 206;
        let mut f = std::fs::OpenOptions::new().create(true).write(true).append(partial).truncate(!partial).open(&out)?;
        let mut rd = resp.into_reader();
        let mut buf = vec![0u8; 1 << 20];
        let mut got = if partial { have } else { 0 };
        let mut last = Instant::now();
        loop {
            if stop() {
                bail!("stopped");
            }
            let n = rd.read(&mut buf)?;
            if n == 0 {
                break;
            }
            f.write_all(&buf[..n])?;
            got += n as u64;
            if last.elapsed().as_secs_f64() > 2.0 {
                last = Instant::now();
                let all = done_all + got;
                log(format!("  {path}: {:.0}/{:.0} MB  (total {:.1}/{:.1} GB, {:.0} MB/s)", got as f64 / 1e6, size as f64 / 1e6, all as f64 / 1e9, total as f64 / 1e9, all as f64 / 1e6 / t0.elapsed().as_secs_f64()));
            }
        }
        anyhow::ensure!(size == 0 || got == size, "{path}: got {got} bytes, expected {size}");
        done_all += size.max(got);
    }
    log(format!("downloaded {:.2} GB in {:.0} s", total as f64 / 1e9, t0.elapsed().as_secs_f64()));
    Ok(())
}

pub fn vault_name(repo: &str, kind: &str) -> String {
    let name: String = repo.rsplit('/').next().unwrap_or(repo).chars().map(|c| if c.is_ascii_alphanumeric() || "._-".contains(c) { c } else { '_' }).collect();
    format!("vault_{}{}", name.to_lowercase().replace('.', "_"), if kind == "gptoss" { "_mxf2" } else { "_huff" })
}

/// A downloaded Hugging Face checkpoint (config.json + safetensors) found in a model folder, maybe not compressed yet.
#[derive(Clone, Debug)]
pub struct Local {
    pub dir: PathBuf,
    pub name: String,
    pub model_type: String,
    /// "dense" | "moe" | "gptoss"; None = this engine cannot run it
    pub kind: Option<String>,
    pub bytes: u64,
    /// its vault, when one with the expected name exists in a model folder
    pub vault: Option<PathBuf>,
}

pub fn kind_of(cfg: &J) -> (String, Option<String>) {
    let c = if cfg["text_config"].is_object() { &cfg["text_config"] } else { cfg };
    let mt = c["model_type"].as_str().unwrap_or("?").to_string();
    let k = if mt == "gpt_oss" {
        Some("gptoss")
    } else if mt == "qwen3_moe" {
        Some("moe")
    } else if DENSE.contains(&mt.as_str()) && !(mt == "mistral" && !c["sliding_window"].is_null()) {
        Some("dense")
    } else {
        None
    };
    (mt, k.map(str::to_string))
}

/// Checkpoint folders directly inside every model folder and the download folder.
pub fn local_checkpoints(p: &Paths, c: &Config) -> Vec<Local> {
    let hw = hardware(p, c);
    let mut bases: Vec<PathBuf> = crate::paths::model_dirs(p, c).into_iter().map(|m| m.path).collect();
    if !bases.contains(&hw.dl_dir) {
        bases.push(hw.dl_dir.clone());
    }
    let vaults = crate::paths::find_vaults(p, c);
    let mut out: Vec<Local> = Vec::new();
    for b in bases {
        let Ok(rd) = std::fs::read_dir(&b) else { continue };
        let mut ds: Vec<PathBuf> = rd.flatten().map(|e| e.path()).filter(|d| d.is_dir()).collect();
        ds.sort();
        for d in ds {
            if crate::paths::is_vault(&d) || !d.join("config.json").exists() || out.iter().any(|x| x.dir == d) {
                continue;
            }
            let st: Vec<PathBuf> = std::fs::read_dir(&d).map(|r| r.flatten().map(|e| e.path()).filter(|f| f.extension().map_or(false, |x| x == "safetensors")).collect()).unwrap_or_default();
            if st.is_empty() {
                continue;
            }
            let cfg: J = std::fs::read(d.join("config.json")).ok().and_then(|x| serde_json::from_slice(&x).ok()).unwrap_or(J::Null);
            let (mt, kind) = kind_of(&cfg);
            let name = d.file_name().unwrap().to_string_lossy().into_owned();
            let vn = vault_name(&name, kind.as_deref().unwrap_or(""));
            out.push(Local {
                bytes: st.iter().map(|f| std::fs::metadata(f).map(|m| m.len()).unwrap_or(0)).sum(),
                vault: vaults.iter().find(|(n, _)| *n == vn).map(|x| x.1.clone()),
                dir: d,
                name,
                model_type: mt,
                kind,
            });
        }
    }
    out
}

/// pack -> check -> tune a checkpoint folder into `dst_dir`; returns the vault path
pub fn compress_local(p: &Paths, c: &Config, src: &Path, dst_dir: &Path, repo: Option<&str>, tune_after: bool, log: &mut dyn FnMut(String), stop: &(dyn Fn() -> bool + Sync)) -> Result<PathBuf> {
    let cfg: J = serde_json::from_slice(&std::fs::read(src.join("config.json")).context("config.json")?)?;
    let (mt, kind) = kind_of(&cfg);
    let kind = kind.ok_or_else(|| anyhow!("{mt}: not supported by this engine"))?;
    let name = src.file_name().unwrap().to_string_lossy().into_owned();
    let vd = dst_dir.join(vault_name(&name, &kind));
    let need: u64 = std::fs::read_dir(src)?.flatten().filter(|e| e.path().extension().map_or(false, |x| x == "safetensors")).map(|e| e.metadata().map(|m| m.len()).unwrap_or(0)).sum();
    std::fs::create_dir_all(dst_dir)?;
    let free = crate::storage::free_bytes(dst_dir);
    anyhow::ensure!(free > need + 2_000_000_000, "{} has {:.1} GB free; the vault needs up to {:.1} GB + 2 GB margin", dst_dir.display(), free as f64 / 1e9, need as f64 / 1e9);
    log(format!("[1/3] compressing {} into {} ({})", src.display(), vd.display(), if kind == "gptoss" { "huffman4 + MXF2 experts, exact to the release" } else { "huffman4, lossless" }));
    if vd.exists() {
        std::fs::remove_dir_all(&vd)?;
    }
    let opt = cube_pack::Options { codec: cube_vault::Codec::Huff4, convert: false, mxf2: true, repo: repo.map(str::to_string).or(Some(name.clone())) };
    let t0 = Instant::now();
    let last = std::cell::Cell::new(Instant::now());
    let lg = std::cell::RefCell::new(&mut *log);
    let r = cube_pack::pack(src, &vd, &opt, &|s: &str| {
        if last.get().elapsed().as_secs_f64() > 2.0 {
            last.set(Instant::now());
            (lg.borrow_mut())(format!("  {}", s.trim()));
        }
    });
    drop(lg);
    let r = match r {
        Ok(r) => r,
        Err(e) => {
            let _ = std::fs::remove_dir_all(&vd);
            return Err(e);
        }
    };
    log(format!("  packed {:?}: {} streams, {:.2} GB -> {:.2} GB ({:.4}), lossless={}, {:.1} min", r.arch, r.streams, r.raw as f64 / 1e9, r.stored as f64 / 1e9, r.stored as f64 / r.raw.max(1) as f64, !r.lossy, t0.elapsed().as_secs_f64() / 60.0));
    log("[2/3] checking every stream of the vault".into());
    let (_, bad) = crate::storage::check_vault(&vd, log)?;
    anyhow::ensure!(bad == 0, "{bad} streams failed their checksum");
    if tune_after && !stop() {
        log("[3/3] tuning the engine settings for this machine".into());
        tune(p, c, &vd, 64, log)?;
    }
    log(format!("ready: {}  (the source checkpoint {} is no longer needed to run it)", vd.display(), src.display()));
    Ok(vd)
}

/// download -> pack -> check -> tune; returns the vault path
pub fn install(p: &Paths, c: &Config, repo: &str, tune_after: bool, log: &mut dyn FnMut(String), stop: &(dyn Fn() -> bool + Sync)) -> Result<PathBuf> {
    let hw = hardware(p, c);
    let a = assess(repo, &hw);
    a.kind.clone().ok_or_else(|| anyhow!("{repo}: {}", a.verdict))?;
    let name: String = repo.rsplit('/').next().unwrap_or(repo).chars().map(|c| if c.is_ascii_alphanumeric() || "._-".contains(c) { c } else { '_' }).collect();
    let src = hw.dl_dir.join(&name);
    log(format!("downloading {repo} (~{:.1} GB) to {}", a.download_gb.unwrap_or(0.0), src.display()));
    download(repo, &src, log, stop)?;
    if stop() {
        bail!("stopped");
    }
    compress_local(p, c, &src, &hw.vault_dir, Some(repo), tune_after, log, stop)
}

const PROMPTS: [&str; 2] = ["In two sentences, explain what a compiler does.", "Write a Python function that reverses a string."];

fn loops(text: &str) -> bool {
    let t: Vec<char> = text.split_whitespace().collect::<Vec<_>>().join(" ").chars().collect();
    if t.len() <= 200 {
        return false;
    }
    let s: String = t.iter().collect();
    (0..t.len().saturating_sub(24)).step_by(12).any(|i| {
        let w: String = t[i..i + 24].iter().collect();
        s.matches(&w).count() >= 5
    })
}

/// Measure the candidate engine settings on short greedy replies; the fastest sane one goes to <vault>/tuned.json.
pub fn tune(p: &Paths, base: &Config, vault: &Path, tokens: i64, log: &mut dyn FnMut(String)) -> Result<Option<J>> {
    let (rec, _) = crate::settings::recommended(vault)?;
    let mut cfg = base.clone();
    for (k, v) in rec {
        cfg.set(&k, v);
    }
    cfg.set("vault", json!(vault.to_string_lossy()));
    cfg.set("greedy", json!(true));
    cfg.set("max_new", json!(tokens));
    cfg.set("think", json!(false));
    let n = crate::hw::threads();
    let mut cands: Vec<J> = vec![json!({"threads": n.saturating_sub(2).max(1)}), json!({"threads": n.saturating_sub(1).max(1)}), json!({"threads": n})];
    cands.dedup();
    if let Some(sp) = crate::storage::read_speed(p, vault.parent().unwrap_or(vault), false) {
        if sp < 500.0 {
            log(format!("note: {} is on a slow drive ({sp:.0} MB/s) - tuning will measure the drive; move it to an SSD folder first for real numbers", vault.display()));
        }
    }
    log(format!("{} configuration(s) to test: {}", cands.len(), cands.iter().map(|c| c.to_string()).collect::<Vec<_>>().join("; ")));
    let tools = crate::tools::Tools { list: Vec::new(), artifacts: std::sync::Arc::new(crate::tools::artifact::Artifacts::new(p.artifacts())) };
    let mut results = Vec::new();
    let mut best = 0.0f64;
    for cd in &cands {
        let mut c2 = cfg.clone();
        for (k, v) in cd.as_object().unwrap() {
            c2.set(k, v.clone());
        }
        let t0 = Instant::now();
        let r = (|| -> Result<J> {
            let mut s = crate::session::Session::open(p, &c2)?;
            let (mut speeds, mut texts, mut early) = (Vec::new(), Vec::new(), false);
            for (i, pr) in PROMPTS.iter().enumerate() {
                let (new, st) = s.reply(p, &[json!({"role": "user", "content": pr})], "", &tools, &[], 0, &mut |_| {}, &|| false)?;
                let sp = st.split(" tok/s").next().and_then(|x| x.rsplit(' ').next()).and_then(|x| x.parse::<f64>().ok()).unwrap_or(0.0);
                speeds.push(sp);
                let last = new.last().cloned().unwrap_or(J::Null);
                texts.push(format!("{}{}", last["content"].as_str().unwrap_or(""), last["thinking"].as_str().unwrap_or("")));
                if i == 0 && best > 0.0 && speeds[0] < 0.6 * best {
                    early = true;
                    break;
                }
            }
            let sane = texts.iter().all(|t| !t.trim().is_empty()) && !texts.iter().any(|t| loops(t));
            let tok_s = (speeds.iter().sum::<f64>() / speeds.len() as f64 * 1000.0).round() / 1000.0;
            Ok(json!({"settings": cd, "tok_s": tok_s, "sane": sane, "how": s.how, "secs": t0.elapsed().as_secs(), "early_stop": early}))
        })()
        .unwrap_or_else(|e| json!({"settings": cd, "tok_s": 0.0, "sane": false, "error": e.to_string(), "secs": t0.elapsed().as_secs()}));
        if r["sane"] == true {
            best = best.max(r["tok_s"].as_f64().unwrap_or(0.0));
        }
        log(format!(
            "  {} {}: {} tok/s in {} s{}{}",
            r["how"].as_str().unwrap_or(""),
            cd,
            r["tok_s"],
            r["secs"],
            if r["early_stop"] == true { "  (stopped early: slower)" } else { "" },
            if r["sane"] == true { String::new() } else { format!("  (rejected: {})", r["error"].as_str().unwrap_or("empty or looping reply")) }
        ));
        results.push(r);
    }
    let good: Vec<&J> = results.iter().filter(|r| r["sane"] == true).collect();
    let Some(b) = good.iter().max_by(|a, b| a["tok_s"].as_f64().partial_cmp(&b["tok_s"].as_f64()).unwrap()) else {
        log("tuning found no working setting - keeping the recommended defaults".into());
        return Ok(None);
    };
    let out = json!({"date": chrono::Local::now().format("%Y-%m-%d %H:%M").to_string(), "settings": b["settings"], "tok_s": b["tok_s"], "results": results});
    std::fs::write(vault.join("tuned.json"), serde_json::to_string_pretty(&out)?)?;
    log(format!("best: {} -> {} tok/s (saved to {}/tuned.json)", b["settings"], b["tok_s"], vault.file_name().unwrap().to_string_lossy()));
    Ok(Some(out))
}
