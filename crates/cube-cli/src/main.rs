//! `cube`: command-line tool for the Rust Cube Harness.

use anyhow::{bail, Context, Result};
use cube_vault::{huff4::Huff4, meta::Arch, mxf, rans4::Rans4, stream, v8::Role, Codec, Vault};
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

const USAGE: &str = "usage:
  cube info <vault>                     model description and streams
  cube check <vault> [--decode]         verify every stream checksum (--decode: also decode every matrix)
  cube convert <vault>                  write vault.cv8 next to a v3-v7 vault (streams untouched)
  cube reencode <vault>                 re-encode every matrix and compare with the stored bytes (codec parity)
  cube pack <model_dir> <vault> [--rans4] [--mxf4] [--convert-matrices]
                                        pack a Hugging Face checkpoint (lossless; --convert-matrices rounds F16/F32 to bf16)
  cube verify-source <vault> <model_dir>   compare every tensor of a vault with its source checkpoint
  cube serve <vault> [--ctx N] [--threads N] [--batch N] [--resident-mb MB] [--expert-mb MB] [--kv16] [--log FILE] [--gpu | --gpu-layers N]
                                        run the engine on the nvme_engine_u line protocol (stdin/stdout)
  cube chat <vault> [--prompt TEXT] [--tools a,b] [--think] [--max-new N] [--ctx N] [--greedy] [--seed N]
                                        talk to a model (with tools); without --prompt reads lines from stdin
  cube gpu-check <vault> [--ctx N] [--tokens N] [--layers N]   GPU engine vs CPU engine: logits, hidden states, greedy text, speed
  cube bench <vault> [--prompt N] [--tokens N] [serve options]   greedy prefill + decode timing (synthetic tokens)
  cube chat-bench <vault> [--mode auto|cpu] [--rounds N] [--max-new N] [--threads N] [--cpu-head]
                                        real chat replies in one session, as the app runs them: tok/s per reply, expert tier hits
  cube tune <vault>                     what the Models tab's Tune does: measure candidate settings, write <vault>/tuned.json
  cube mem                              free RAM, this process, the largest processes
  cube recommend <vault>                the settings the app picks for this vault on this machine now (context sized to RAM)";

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e:#}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    match a.get(1).map(String::as_str) {
        Some("info") => info(a.get(2).context(USAGE)?),
        Some("check") => check(a.get(2).context(USAGE)?, a.iter().any(|x| x == "--decode")),
        Some("convert") => {
            let n = Vault::convert(a.get(2).context(USAGE)?)?;
            let v = Vault::open(a.get(2).unwrap())?;
            println!("wrote vault.cv8 ({n} bytes): {:?}, {} layers, {} streams", v.meta().arch, v.meta().layers, v.n_streams());
            Ok(())
        }
        Some("reencode") => reencode(a.get(2).context(USAGE)?),
        Some("pack") => {
            let (m, o) = (a.get(2).context(USAGE)?, a.get(3).context(USAGE)?);
            let opt = cube_pack::Options {
                codec: if a.iter().any(|x| x == "--rans4") { Codec::Rans4 } else { Codec::Huff4 },
                convert: a.iter().any(|x| x == "--convert-matrices"),
                mxf2: !a.iter().any(|x| x == "--mxf4"),
                repo: None,
            };
            let t0 = Instant::now();
            let r = cube_pack::pack(std::path::Path::new(m), std::path::Path::new(o), &opt, &|s: &str| eprint!("\r{s}      "))?;
            eprintln!();
            println!(
                "packed {:?}: {} streams, raw {:.1} MB -> {:.1} MB ({:.4}), lossy={}, {:.1}s",
                r.arch, r.streams, r.raw as f64 / 1e6, r.stored as f64 / 1e6, r.stored as f64 / r.raw.max(1) as f64, r.lossy, t0.elapsed().as_secs_f64()
            );
            Ok(())
        }
        Some("yarn") => {
            let c: serde_json::Value = serde_json::from_slice(&std::fs::read(a.get(2).context(USAGE)?)?)?;
            let hd = c["head_dim"].as_u64().context("head_dim")? as usize;
            let (inv, att) = cube_pack::yarn(&c, hd)?;
            println!("{att:.17}");
            println!("{}", inv.iter().map(|x| format!("{:08x}", x.to_bits())).collect::<Vec<_>>().join(" "));
            Ok(())
        }
        Some("serve") => serve(&a),
        Some("bench") => bench(&a),
        Some("chat-bench") => chat_bench(&a),
        Some("recommend") => {
            let (s, notes) = cube_core::settings::recommended(std::path::Path::new(a.get(2).context(USAGE)?))?;
            for n in notes {
                println!("{n}");
            }
            println!("{}", serde_json::Value::Object(s));
            Ok(())
        }
        Some("tune") => {
            let p = cube_core::paths::Paths::discover();
            p.ensure_dirs();
            let cfg = cube_core::paths::Config::load(&p);
            let v = cube_core::paths::canon(std::path::Path::new(a.get(2).context(USAGE)?));
            cube_core::hub::tune(&p, &cfg, &v, 64, &mut |s| println!("{s}"))?;
            Ok(())
        }
        Some("mem") => {
            println!("{}", cube_core::hw::mem_report());
            Ok(())
        }
        Some("chat") => chat(&a),
        Some("gpu-check") => gpu_check(&a),
        Some("verify-source") => {
            let v = Vault::open(a.get(2).context(USAGE)?)?;
            let (n, bad) = cube_pack::verify_against_source(&v, std::path::Path::new(a.get(3).context(USAGE)?))?;
            println!("verify-source: {n} tensors compared, {bad} mismatches");
            if bad > 0 {
                bail!("{bad} mismatches");
            }
            Ok(())
        }
        _ => {
            println!("{USAGE}");
            Ok(())
        }
    }
}

fn info(dir: &str) -> Result<()> {
    let v = Vault::open(dir)?;
    let m = v.meta();
    println!(
        "{}: {} (v8 manifest: {})",
        dir,
        match v.legacy_version {
            Some(x) => format!("WVAULT {x}"),
            None => "v8".into(),
        },
        v.legacy_version.is_none()
    );
    println!("{}", serde_json::to_string_pretty(m)?);
    let total: u64 = v.manifest.streams.iter().map(|s| s.e.padded).sum();
    println!("{} streams, {:.3} GB; resident {} floats; rope table {}", v.n_streams(), total as f64 / 1e9, v.resident().len(), v.manifest.rope_inv.len());
    Ok(())
}

/// What one sub-blob is, from the model description and the stream role.
fn sub_kind(v: &Vault, role: Role, sub: usize) -> &'static str {
    match (v.meta().arch, role, sub) {
        (Arch::Gptoss, Role::Expert, 0 | 2) => "mxf",
        (Arch::Gptoss, Role::Expert, _) => "f32",
        _ => match v.meta().codec {
            Codec::Huff4 => "huff4",
            Codec::Rans4 => "rans4",
        },
    }
}

fn check(dir: &str, decode: bool) -> Result<()> {
    let v = Vault::open(dir)?;
    let t0 = Instant::now();
    let bad = AtomicUsize::new(0);
    let mats = AtomicUsize::new(0);
    let bytes: u64 = v.manifest.streams.iter().map(|s| s.e.padded).sum();
    (0..v.n_streams()).into_par_iter().try_for_each(|i| -> Result<()> {
        let buf = match v.read_stream_checked(i) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("{e:#}");
                bad.fetch_add(1, Ordering::Relaxed);
                return Ok(());
            }
        };
        if decode {
            let role = v.manifest.streams[i].role;
            for (k, b) in stream::sub_blobs(&buf)?.iter().enumerate() {
                match sub_kind(&v, role, k) {
                    "huff4" => {
                        Huff4::parse(b)?.decode_bits();
                    }
                    "rans4" => {
                        Rans4::parse(b)?.decode_bits();
                    }
                    "mxf" => {
                        mxf::Mxf::parse(b)?.all_scales();
                    }
                    _ => {
                        mxf::f32_view(b)?;
                    }
                }
                mats.fetch_add(1, Ordering::Relaxed);
            }
        }
        Ok(())
    })?;
    let s = t0.elapsed().as_secs_f64();
    let bad = bad.into_inner();
    println!(
        "vault {}: {} streams, {:.0} MB in {:.1}s ({:.0} MB/s){}, bad={}",
        if bad == 0 { "OK" } else { "CORRUPT" },
        v.n_streams(),
        bytes as f64 / 1e6,
        s,
        bytes as f64 / 1e6 / s,
        if decode { format!(", {} sub-blobs decoded", mats.into_inner()) } else { String::new() },
        bad
    );
    if bad > 0 {
        bail!("{bad} bad streams");
    }
    Ok(())
}

/// Decode every matrix and encode it again with the Rust encoders; the bytes must equal the stored blob (made by the
/// Python packers), and the whole rebuilt stream must equal the stored stream.
fn reencode(dir: &str) -> Result<()> {
    let v = Vault::open(dir)?;
    let t0 = Instant::now();
    let same = AtomicUsize::new(0);
    let diff = AtomicUsize::new(0);
    (0..v.n_streams()).into_par_iter().try_for_each(|i| -> Result<()> {
        let buf = v.read_stream(i)?;
        let role = v.manifest.streams[i].role;
        let subs = stream::sub_blobs(&buf)?;
        let mut rebuilt: Vec<Vec<u8>> = Vec::new();
        for (k, b) in subs.iter().enumerate() {
            let nb = match sub_kind(&v, role, k) {
                "huff4" => {
                    let h = Huff4::parse(b)?;
                    cube_vault::huff4::compress(&h.decode_bits(), h.rows, h.cols)?
                }
                "rans4" => {
                    let r = Rans4::parse(b)?;
                    cube_vault::rans4::encode(&r.decode_bits(), r.rows, r.cols, r.mbits, r.r)?
                }
                "mxf" => {
                    let m = mxf::Mxf::parse(b)?;
                    if &b[0..4] == b"MXF2" {
                        mxf::mxf2_blob(m.blocks(), &m.all_scales(), m.rows, m.nb)?
                    } else {
                        mxf::mxf4_blob(m.blocks(), &m.all_scales(), m.rows, m.nb)
                    }
                }
                _ => mxf::f32_blob(&mxf::f32_view(b)?),
            };
            rebuilt.push(nb);
        }
        let refs: Vec<&[u8]> = rebuilt.iter().map(|x| x.as_slice()).collect();
        let (sb, size) = stream::build(&refs);
        if sb == buf && size == v.stream(i).size {
            same.fetch_add(1, Ordering::Relaxed);
        } else {
            eprintln!("stream {i}: rebuilt bytes differ");
            diff.fetch_add(1, Ordering::Relaxed);
        }
        Ok(())
    })?;
    let (s, d) = (same.into_inner(), diff.into_inner());
    println!("reencode: {s} streams byte-identical, {d} different ({:.1}s)", t0.elapsed().as_secs_f64());
    if d > 0 {
        bail!("{d} streams differ");
    }
    Ok(())
}

fn engine_opts(a: &[String], from: usize, extra: &mut Vec<(String, String)>) -> Result<(cube_engine::Options, usize, Option<String>)> {
    let mut o = cube_engine::Options::default();
    let mut threads = 0usize;
    let mut log: Option<String> = None;
    let mut i = from;
    let val = |i: usize| -> Result<&String> { a.get(i + 1).with_context(|| format!("{} needs a value", a[i])) };
    while i < a.len() {
        match a[i].as_str() {
            "--ctx" => { o.ctx = val(i)?.parse()?; i += 1; }
            "--threads" => { threads = val(i)?.parse()?; i += 1; }
            "--batch" => { o.batch = val(i)?.parse::<usize>()?.max(1); i += 1; }
            "--resident-mb" => { o.resident_mb = if val(i)? == "auto" { -1.0 } else { val(i)?.parse()? }; i += 1; }
            "--expert-mb" => { o.expert_cache_mb = val(i)?.parse()?; i += 1; }
            "--kv16" => o.kv16 = true,
            "--log" => { log = Some(val(i)?.clone()); i += 1; }
            // accepted for compatibility with nvme_engine_u; no effect here
            "--cached" | "--gpu" => {}
            "--gpu-layers" => { i += 1; }
            "--io-threads" | "--io-chunk" | "--log-detail" | "--vram-mb" | "--dev-experts-cpu" => { i += 1; }
            "--cpu-head" | "--dev-head-cpu" => {}
            "--prefetch" => { if a.get(i + 1).map_or(false, |x| x.starts_with(|c: char| c.is_ascii_digit())) { i += 1; } }
            "--prompt" | "--tokens" => { extra.push((a[i].clone(), val(i)?.clone())); i += 1; }
            x => bail!("unknown option {x}"),
        }
        i += 1;
    }
    if let Ok(m) = std::env::var("NVME_KV_MARGIN") { o.kv_margin = m.parse().unwrap_or(1024); }
    if threads == 0 {
        threads = std::thread::available_parallelism().map_or(1, |n| n.get()).saturating_sub(2).max(1);
    }
    o.threads = threads;
    rayon::ThreadPoolBuilder::new().num_threads(threads).build_global().ok();
    Ok((o, threads, log))
}

/// gpt-oss: put a device expert tier under the RAM cache. `--gpu` (VRAM, budget `--vram-mb` or free VRAM - 600 MB) or
/// `--dev-experts-cpu N` (N slots in host memory, the same tiering without a GPU, for tests). Returns a description.
fn attach_dev_experts(a: &[String], e: &mut cube_engine::Engine) -> Result<Option<String>> {
    if a.iter().any(|x| x == "--dev-head-cpu") {
        let h = cube_engine::model::HostHead::new(&e.vault)?;
        e.attach_head(Box::new(h));
    }
    if e.m.arch != cube_vault::meta::Arch::Gptoss {
        return Ok(None);
    }
    let num = |k: &str| a.iter().position(|x| x == k).and_then(|i| a.get(i + 1)).and_then(|x| x.parse::<u64>().ok());
    if let Some(n) = num("--dev-experts-cpu") {
        let d = cube_engine::model::HostExperts::new(&e.m, n as usize)?;
        let s = cube_engine::store::ExpertDevice::describe(&d);
        e.attach_expert_device(Box::new(d));
        return Ok(Some(s));
    }
    if a.iter().any(|x| x == "--gpu") {
        let budget = num("--vram-mb").map(|mb| mb * 1_000_000);
        return Ok(Some(cube_gpu::attach_moe_tiers(e, !a.iter().any(|x| x == "--cpu-head"), 600_000_000, budget)?));
    }
    Ok(None)
}

/// `--resident-mb auto`: what the app pins (free RAM - 1.5 GB - KV cache, x 0.8); MoE models also get the app's prompt batch.
fn auto_ram(v: &Vault, o: &mut cube_engine::Options) {
    if o.resident_mb < 0.0 {
        let kv = cube_core::hw::kv_bytes(v.meta(), o.ctx, o.kv16) / 1e6;
        o.resident_mb = ((cube_core::hw::ram_mb().1 - 1500.0 - kv).max(0.0)) * 0.8;
        if v.meta().moe.is_some() {
            o.batch = 128;
        }
    }
}

fn tiers_line(e: &dyn cube_engine::Backend) -> Option<String> {
    e.expert_tiers().map(|(r, d, m, rc, dc)| format!("M ram_hits={r} vram_hits={d} misses={m} ram_slots={rc} vram_slots={dc} hit_pct={:.1}", 100.0 * (r + d) as f64 / (r + d + m).max(1) as f64))
}

fn serve(a: &[String]) -> Result<()> {
    let vdir = a.get(2).context(USAGE)?;
    let (o, threads, log) = engine_opts(a, 3, &mut Vec::new())?;
    let t0 = Instant::now();
    let v = Vault::open(vdir)?;
    let mut o = o;
    auto_ram(&v, &mut o);
    let (gpu_all, gpu_layers) = (a.iter().any(|x| x == "--gpu"), a.iter().position(|x| x == "--gpu-layers").and_then(|i| a.get(i + 1)).and_then(|x| x.parse::<usize>().ok()).unwrap_or(0));
    let gptoss = v.meta().arch == cube_vault::meta::Arch::Gptoss;
    let (e, ring_margin): (Box<dyn cube_engine::Backend>, Option<usize>) = if gpu_all && !gptoss {
        (Box::new(cube_gpu::GpuDense::open(&v, o.ctx, None)?), None)
    } else {
        let mut e = cube_engine::Engine::open(v, o)?;
        if let Some(d) = attach_dev_experts(a, &mut e)? {
            eprintln!("expert tier: {d}");
        }
        if gpu_layers > 0 {
            let g = cube_gpu::GpuDense::open(&e.vault, e.opt.ctx, Some(gpu_layers))?;
            e.attach_prefix(Box::new(g));
        }
        let rm = if e.has_ring() { Some(e.opt.kv_margin) } else { None };
        (Box::new(e), rm)
    };
    let logf = match &log { Some(p) => Some(std::fs::OpenOptions::new().create(true).append(true).open(p)?), None => None };
    let dump = match std::env::var("NVME_DUMP_H") { Ok(p) => Some(std::fs::File::create(p)?), Err(_) => None };
    eprintln!("engine ready: {} layers, {threads} threads, {:.0} MB pinned, {:.2}s", e.meta().layers, e.pinned_bytes() as f64 / 1e6, t0.elapsed().as_secs_f64());
    let mut s = cube_engine::serve::Server { e, ring_margin, log: logf, dump, ignore_eos: std::env::var("NVME_IGNORE_EOS").is_ok() };
    let rx = cube_engine::serve::stdin_lines();
    let stdout = std::io::stdout();
    let mut out = std::io::BufWriter::new(stdout.lock());
    s.run(rx, &mut out)
}

fn bench(a: &[String]) -> Result<()> {
    let vdir = a.get(2).context(USAGE)?;
    let mut ex = Vec::new();
    let (o, threads, _) = engine_opts(a, 3, &mut ex)?;
    let get = |k: &str, d: usize| ex.iter().find(|(x, _)| x == k).and_then(|(_, v)| v.parse().ok()).unwrap_or(d);
    let (np, nt) = (get("--prompt", 8), get("--tokens", 16));
    let t0 = Instant::now();
    let v = Vault::open(vdir)?;
    let mut o = o;
    auto_ram(&v, &mut o);
    let mut e = cube_engine::Engine::open(v, o)?;
    if let Some(d) = attach_dev_experts(a, &mut e)? {
        println!("expert tier: {d}");
    }
    println!("BENCH {:?} layers={} threads={threads} open_s={:.2} pinned_MB={:.0} resident_MB={:.0}", e.m.arch, e.m.layers, t0.elapsed().as_secs_f64(), e.pinned_bytes() as f64 / 1e6, e.opt.resident_mb);
    let prompt: Vec<u32> = (0..np as u32).map(|i| 100 + i * 7).collect();
    let tp = Instant::now();
    let b = e.opt.batch;
    for (i, ch) in prompt.chunks(b).enumerate() {
        e.forward(ch, u8::from((i + 1) * b >= np), false)?;
    }
    let pms = tp.elapsed().as_secs_f64() * 1e3;
    let mut tok = cube_engine::sample::argmax(&e.logits) as u32;
    let (mut ms, mut w, mut r, mut by, mut at, mut cm, mut lm) = (Vec::new(), 0.0, 0.0, 0u64, 0.0, 0.0, 0.0);
    for _ in 0..nt {
        let t = Instant::now();
        e.forward(&[tok], 1, false)?;
        tok = cube_engine::sample::argmax(&e.logits) as u32;
        ms.push(t.elapsed().as_secs_f64() * 1e3);
        w += e.timing.wait_ms; r += e.timing.read_ms; cm += e.timing.comp_ms; lm += e.timing.lm_ms; by += e.timing.bytes_read; at += e.timing.attn_ms;
        println!("T {tok}");
    }
    let tot: f64 = ms.iter().sum();
    let late: f64 = ms[ms.len() / 2..].iter().sum::<f64>();
    let late_tok_s = (ms.len() - ms.len() / 2) as f64 * 1e3 / late.max(1e-9);
    ms.sort_by(|x, y| x.partial_cmp(y).unwrap());
    println!("S prompt={np} prefill_ms={pms:.0} prefill_tok_s={:.2} tok_s={:.3} ms_tok={:.1} p50_ms={:.1} stall_pct={:.0} read_MBps={:.0} attn_pct={:.1} streamed_MB_tok={:.0}",
        np as f64 * 1e3 / pms, nt as f64 * 1e3 / tot, tot / nt as f64, ms[ms.len() / 2], 100.0 * w / tot, if r > 0.0 { by as f64 / 1e3 / r } else { 0.0 }, 100.0 * at / tot, by as f64 / 1e6 / nt as f64);
    println!("D comp_ms_tok={:.1} lm_ms_tok={:.1} wait_ms_tok={:.1} second_half_tok_s={late_tok_s:.3}", cm / nt as f64, lm / nt as f64, w / nt as f64);
    if let Some(l) = tiers_line(&e) { println!("{l}"); }
    Ok(())
}

fn chat(a: &[String]) -> Result<()> {
    use cube_core::{paths, session::{Event, Session}, tools::Tools};
    let p = paths::Paths::discover();
    p.ensure_dirs();
    let mut cfg = paths::Config::load(&p);
    let vault = a.get(2).context(USAGE)?;
    cfg.set("vault", serde_json::json!(cube_core::paths::canon(std::path::Path::new(vault)).to_string_lossy().into_owned()));
    let mut prompt: Option<String> = None;
    let mut i = 3;
    while i < a.len() {
        let v = a.get(i + 1).cloned().unwrap_or_default();
        match a[i].as_str() {
            "--prompt" => { prompt = Some(v); i += 1; }
            "--tools" => { cfg.set("enabled_plugins", serde_json::json!(v.split(',').filter(|x| !x.is_empty()).collect::<Vec<_>>())); i += 1; }
            "--think" => cfg.set("think", serde_json::json!(true)),
            "--greedy" => cfg.set("greedy", serde_json::json!(true)),
            "--max-new" => { cfg.set("max_new", serde_json::json!(v.parse::<i64>()?)); i += 1; }
            "--ctx" => { cfg.set("ctx", serde_json::json!(v.parse::<i64>()?)); i += 1; }
            "--seed" => { cfg.set("seed", serde_json::json!(v.parse::<u64>()?)); i += 1; }
            "--threads" => { cfg.set("threads", serde_json::json!(v.parse::<i64>()?)); i += 1; }
            "--system" => { cfg.set("system", serde_json::json!(v)); i += 1; }
            x => bail!("unknown option {x}"),
        }
        i += 1;
    }
    let th = cfg.i64("threads");
    if th > 0 { rayon::ThreadPoolBuilder::new().num_threads(th as usize).build_global().ok(); }
    let t0 = Instant::now();
    let mut s = Session::open(&p, &cfg)?;
    eprintln!("{}\nloaded in {:.1}s - {}", s.banner, t0.elapsed().as_secs_f64(), s.how);
    let tools = Tools::load(&p, &cfg.str("python"));
    let enabled = cfg.enabled_plugins();
    let mut history: Vec<serde_json::Value> = Vec::new();
    let lines: Vec<String> = match &prompt { Some(x) => vec![x.clone()], None => std::io::stdin().lines().map_while(|l| l.ok()).collect() };
    for user in lines {
        if user.trim().is_empty() { continue; }
        history.push(serde_json::json!({"role": "user", "content": user}));
        let on = |e: Event| match e {
            Event::Thinking(t) => eprint!("\x1b[2m{t}\x1b[0m"),
            Event::Assistant(t) => print!("{t}"),
            Event::AssistantNext => println!(),
            Event::Tool(t) => eprint!("\x1b[33m{t}\x1b[0m"),
            Event::Act(_) => {}
        };
        use std::io::Write;
        let (new, stats) = s.reply(&p, &history, &cfg.str("system"), &tools, &enabled, cfg.i64("max_tool_rounds").max(0) as usize, &mut |e| { on(e); let _ = std::io::stdout().flush(); }, &|| false)?;
        println!("\n[{stats}]");
        history.extend(new);
    }
    Ok(())
}

fn gpu_check(a: &[String]) -> Result<()> {
    use cube_engine::Backend;
    let vdir = a.get(2).context(USAGE)?;
    let arg = |k: &str, d: usize| a.iter().position(|x| x == k).and_then(|i| a.get(i + 1)).and_then(|x| x.parse().ok()).unwrap_or(d);
    let (ctx, ntok) = (arg("--ctx", 512), arg("--tokens", 32));
    let g = cube_gpu::gpu().context("no CUDA device found (NVIDIA driver not loaded)")?;
    println!("GPU: {} (compute {}.{}), VRAM free {:.2} GB", g.name, g.cc.0, g.cc.1, cube_gpu::vram().map_or(0.0, |v| v.0 as f64 / 1e9));
    let v = Vault::open(vdir)?;
    let m = v.meta().clone();
    let half = arg("--layers", m.layers / 2).max(1);
    println!("model: {:?} {} layers, codec {:?}; VRAM needed whole {:.2} GB, {half} layers {:.2} GB", m.arch, m.layers, m.codec,
        cube_gpu::vram_needed(&v, ctx, None) as f64 / 1e9, cube_gpu::vram_needed(&v, ctx, Some(half)) as f64 / 1e9);
    let prompt: Vec<u32> = (0..24u32).map(|i| (1000 + 7919 * i) % m.vocab as u32).collect();
    let o = cube_engine::Options { ctx, ..Default::default() };
    let t = Instant::now();
    let mut cpu = cube_engine::Engine::open(Vault::open(vdir)?, o.clone())?;
    let t_open = t.elapsed().as_secs_f64();
    let cmp = |x: &[f32], y: &[f32]| -> (f64, f64, bool) {
        let (mut d2, mut n2, mut mx) = (0f64, 0f64, 0f64);
        for (p, q) in x.iter().zip(y) {
            let d = (*p as f64 - *q as f64).abs();
            d2 += d * d;
            n2 += (*p as f64) * (*p as f64);
            mx = mx.max(d);
        }
        ((d2 / n2.max(1e-30)).sqrt(), mx, cube_engine::sample::argmax(x) == cube_engine::sample::argmax(y))
    };
    if m.arch == cube_vault::meta::Arch::Gptoss {
        return gpu_check_experts(a, vdir, &prompt, o, ntok, cpu, &cmp);
    }
    let free = cube_gpu::vram().map_or(0, |v| v.0 as u64);
    let whole = cube_gpu::vram_needed(&v, ctx, None) + 300_000_000 < free;
    let mut ok = true;
    let oc: Vec<u32>;
    let gen = |b: &mut dyn Backend| -> Result<(Vec<u32>, f64)> {
        b.reset();
        let (out, _, st) = cube_engine::backend::generate(b, &prompt, ntok, 0.0, 1.0, 1, 1, &[], &mut |_| false)?;
        Ok((out, st.tok_s))
    };
    if whole {
        let t = Instant::now();
        let mut gpu = cube_gpu::GpuDense::open(&v, ctx, None)?;
        println!("opened: CPU {t_open:.1}s, GPU {:.1}s ({:.0} MB VRAM)", t.elapsed().as_secs_f64(), gpu.vram_bytes() as f64 / 1e6);
        // 1. prompt logits + per-layer hidden states
        cpu.forward(&prompt, 1, true)?;
        Backend::forward(&mut gpu, &prompt, 1, true)?;
        let (rl, mx, am) = cmp(&cpu.logits, gpu.logits());
        println!("prompt logits: relL2 {rl:.2e}  maxabs {mx:.2e}  argmax {}", if am { "same" } else { "DIFFERENT" });
        ok &= rl < 1e-3 && am;
        if let (Some(hc), Some(hg)) = (cpu.dump_hidden.as_ref(), gpu.dump_hidden()) {
            let hid = m.hidden;
            let worst = (0..m.layers).map(|l| cmp(&hc[l * hid..(l + 1) * hid], &hg[l * hid..(l + 1) * hid]).0).fold(0f64, f64::max);
            println!("hidden states, worst layer relL2 {worst:.2e}");
            ok &= worst < 1e-3;
        }
        // 2. greedy continuation
        let (c, sc) = gen(&mut cpu)?;
        let (og, sg) = gen(&mut gpu)?;
        let same = c.iter().zip(&og).take_while(|(a, b)| a == b).count();
        println!("greedy {ntok} tokens: {same} identical before the first difference; CPU {sc:.2} tok/s, GPU {sg:.2} tok/s");
        ok &= same >= ntok.min(8);
        oc = c;
    } else {
        println!("the whole model does not fit in free VRAM ({:.2} GB): GPU-resident check skipped, split mode only", free as f64 / 1e9);
        let (c, sc) = gen(&mut cpu)?;
        println!("CPU greedy {ntok} tokens: {sc:.2} tok/s");
        oc = c;
    }
    // 3. split mode
    let mut sp = cube_engine::Engine::open(Vault::open(vdir)?, o)?;
    sp.attach_prefix(Box::new(cube_gpu::GpuDense::open(&v, ctx, Some(half))?));
    let (os, ss) = gen(&mut sp)?;
    let same2 = oc.iter().zip(&os).take_while(|(a, b)| a == b).count();
    cpu.reset();
    sp.reset();
    cpu.forward(&prompt, 1, false)?;
    sp.forward(&prompt, 1, false)?;
    let (rl2, _, am2) = cmp(&cpu.logits, &sp.logits);
    println!("split ({half} layers on GPU): logits relL2 {rl2:.2e} argmax {}, greedy {same2}/{ntok} identical, {ss:.2} tok/s", if am2 { "same" } else { "DIFFERENT" });
    ok &= rl2 < 1e-3 && am2;
    println!("GPU CHECK {}", if ok { "PASS" } else { "FAIL" });
    if !ok {
        bail!("GPU results differ from the CPU engine");
    }
    Ok(())
}

/// gpt-oss: the CPU engine alone against CPU + a VRAM expert tier with a tiny RAM cache, so nearly every expert is computed on the GPU.
fn gpu_check_experts(a: &[String], vdir: &str, prompt: &[u32], o: cube_engine::Options, ntok: usize, mut cpu: cube_engine::Engine, cmp: &dyn Fn(&[f32], &[f32]) -> (f64, f64, bool)) -> Result<()> {
    use cube_engine::Backend;
    let mut ok = true;
    let mut t = cube_engine::Engine::open(Vault::open(vdir)?, cube_engine::Options { expert_cache_mb: 1.0, ..o })?;
    let desc = match attach_dev_experts(a, &mut t)? {
        Some(d) => d,
        None => cube_gpu::attach_moe_tiers(&mut t, !a.iter().any(|x| x == "--cpu-head"), 600_000_000, None)?,
    };
    println!("expert tier: {desc}; RAM cache forced to ~1 MB so evicted experts run from the device tier");
    // pass 1 fills the tiers, pass 2 runs (mostly) from the device tier
    for pass in 1..=2 {
        cpu.reset();
        t.reset();
        cpu.forward(prompt, 1, false)?;
        t.forward(prompt, 1, false)?;
        let (rl, mx, am) = cmp(&cpu.logits, &t.logits);
        println!("pass {pass} prompt logits: relL2 {rl:.2e} maxabs {mx:.2e} argmax {}  {}", if am { "same" } else { "DIFFERENT" }, tiers_line(&t).unwrap_or_default());
        ok &= rl < 1e-3 && am;
    }
    let gen = |b: &mut dyn cube_engine::Backend| -> Result<(Vec<u32>, f64)> {
        b.reset();
        let (out, _, st) = cube_engine::backend::generate(b, prompt, ntok, 0.0, 1.0, 1, 1, &[], &mut |_| false)?;
        Ok((out, st.tok_s))
    };
    let (c, sc) = gen(&mut cpu)?;
    let (g, sg) = gen(&mut t)?;
    let same = c.iter().zip(&g).take_while(|(x, y)| x == y).count();
    println!("greedy {ntok} tokens: {same} identical before the first difference; CPU {sc:.2} tok/s, tiered {sg:.2} tok/s");
    println!("{}", tiers_line(&t).unwrap_or_default());
    ok &= same >= ntok.min(8);
    println!("GPU CHECK {}", if ok { "PASS" } else { "FAIL" });
    if !ok {
        bail!("device-tier results differ from the CPU engine");
    }
    Ok(())
}

const CHAT_BENCH: [&str; 6] = [
    "In two sentences, explain what a compiler does.",
    "Write a Python function that reverses a string.",
    "What were the main causes of the First World War?",
    "Give me a recipe for a simple tomato soup.",
    "Explain the difference between TCP and UDP to a beginner.",
    "Write a short poem about autumn rain.",
];

/// chat-bench <vault> [--mode auto|cpu|gpu|split] [--rounds N] [--max-new N] [--threads N]: real chat replies in one session
/// (the app's engine choice and RAM budget), tok/s per reply and the expert tiers' hits - warm-cache speed on real text.
fn chat_bench(a: &[String]) -> Result<()> {
    use cube_core::{paths, session::Session, tools::Tools};
    let p = paths::Paths::discover();
    p.ensure_dirs();
    let mut cfg = paths::Config::load(&p);
    let vault = a.get(2).context(USAGE)?;
    cfg.set("vault", serde_json::json!(cube_core::paths::canon(std::path::Path::new(vault)).to_string_lossy().into_owned()));
    cfg.set("greedy", serde_json::json!(true));
    cfg.set("think", serde_json::json!(false));
    cfg.set("max_new", serde_json::json!(128));
    let mut rounds = 2usize;
    let mut i = 3;
    while i < a.len() {
        let v = a.get(i + 1).cloned().unwrap_or_default();
        match a[i].as_str() {
            "--mode" => cfg.set("mode", serde_json::json!(v)),
            "--rounds" => rounds = v.parse()?,
            "--max-new" => cfg.set("max_new", serde_json::json!(v.parse::<i64>()?)),
            "--threads" => cfg.set("threads", serde_json::json!(v.parse::<i64>()?)),
            "--cpu-head" => {
                cfg.set("cpu_head", serde_json::json!(true));
                i += 1;
                continue;
            }
            x => bail!("unknown option {x}"),
        }
        i += 2;
    }
    let t0 = Instant::now();
    let mut s = Session::open(&p, &cfg)?;
    println!("CHAT-BENCH {} mode={} open_s={:.1}\n  {}", s.name, cfg.str("mode"), t0.elapsed().as_secs_f64(), s.how);
    let tools = Tools { list: Vec::new(), artifacts: std::sync::Arc::new(cube_core::tools::artifact::Artifacts::new(p.artifacts())) };
    let mut per_round = Vec::new();
    for r in 1..=rounds {
        let mut sp = Vec::new();
        for pr in CHAT_BENCH {
            let (_, st) = s.reply(&p, &[serde_json::json!({"role": "user", "content": pr})], "", &tools, &[], 0, &mut |_| {}, &|| false)?;
            let x = st.split(" tok/s").next().and_then(|x| x.rsplit(' ').next()).and_then(|x| x.parse::<f64>().ok()).unwrap_or(0.0);
            sp.push(x);
            println!("round {r}: {x:6.3} tok/s  {:<60}  {}", pr, tiers_line(s.eng.as_ref()).unwrap_or_default());
        }
        let avg = sp.iter().sum::<f64>() / sp.len() as f64;
        println!("round {r} average {avg:.3} tok/s");
        per_round.push(avg);
    }
    println!("RESULT {}", per_round.iter().enumerate().map(|(i, x)| format!("round{}={x:.3}", i + 1)).collect::<Vec<_>>().join(" "));
    Ok(())
}
