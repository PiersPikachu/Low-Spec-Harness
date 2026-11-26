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
  cube serve <vault> [--ctx N] [--threads N] [--batch N] [--resident-mb MB] [--expert-mb MB] [--kv16] [--log FILE]
                                        run the engine on the nvme_engine_u line protocol (stdin/stdout)
  cube bench <vault> [--prompt N] [--tokens N] [serve options]   greedy prefill + decode timing";

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
            "--resident-mb" => { o.resident_mb = val(i)?.parse()?; i += 1; }
            "--expert-mb" => { o.expert_cache_mb = val(i)?.parse()?; i += 1; }
            "--kv16" => o.kv16 = true,
            "--log" => { log = Some(val(i)?.clone()); i += 1; }
            // accepted for compatibility with nvme_engine_u; no effect here
            "--cached" => {}
            "--io-threads" | "--io-chunk" | "--log-detail" | "--vram-mb" => { i += 1; }
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

fn serve(a: &[String]) -> Result<()> {
    let vdir = a.get(2).context(USAGE)?;
    let (o, threads, log) = engine_opts(a, 3, &mut Vec::new())?;
    let t0 = Instant::now();
    let v = Vault::open(vdir)?;
    let e = cube_engine::Engine::open(v, o)?;
    let logf = match &log { Some(p) => Some(std::fs::OpenOptions::new().create(true).append(true).open(p)?), None => None };
    let dump = match std::env::var("NVME_DUMP_H") { Ok(p) => Some(std::fs::File::create(p)?), Err(_) => None };
    eprintln!("engine ready: {} layers, {threads} threads, {:.0} MB pinned, {:.2}s", e.m.layers, e.pinned_bytes() as f64 / 1e6, t0.elapsed().as_secs_f64());
    let mut s = cube_engine::serve::Server { e, log: logf, dump, ignore_eos: std::env::var("NVME_IGNORE_EOS").is_ok() };
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
    let mut e = cube_engine::Engine::open(Vault::open(vdir)?, o)?;
    println!("BENCH {:?} layers={} threads={threads} open_s={:.2} pinned_MB={:.0}", e.m.arch, e.m.layers, t0.elapsed().as_secs_f64(), e.pinned_bytes() as f64 / 1e6);
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
    ms.sort_by(|x, y| x.partial_cmp(y).unwrap());
    println!("S prompt={np} prefill_ms={pms:.0} prefill_tok_s={:.2} tok_s={:.3} ms_tok={:.1} p50_ms={:.1} stall_pct={:.0} read_MBps={:.0} attn_pct={:.1} streamed_MB_tok={:.0}",
        np as f64 * 1e3 / pms, nt as f64 * 1e3 / tot, tot / nt as f64, ms[ms.len() / 2], 100.0 * w / tot, if r > 0.0 { by as f64 / 1e3 / r } else { 0.0 }, 100.0 * at / tot, by as f64 / 1e6 / nt as f64);
    println!("D comp_ms_tok={:.1} lm_ms_tok={:.1} wait_ms_tok={:.1}", cm / nt as f64, lm / nt as f64, w / nt as f64);
    if let Some((h, m, c)) = e.expert_stats() { println!("M hits={h} misses={m} slots={c}"); }
    Ok(())
}
