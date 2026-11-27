//! The forward pass: dense decoders (Qwen2/3, Llama), Qwen3-MoE and gpt-oss, reading weights straight from a vault.

use crate::kernels::{self, f16_to_f32, f32_to_f16, Mat};
use crate::store::{ExpertCache, StreamStore};
use anyhow::{bail, ensure, Context, Result};
use cube_vault::huff4::{build_lut, LMAX};
use cube_vault::meta::{Arch, ModelMeta};
use cube_vault::{mxf, stream, Codec, Rope, Vault};
use rayon::prelude::*;
use std::sync::Arc;
use std::time::Instant;

#[derive(Clone, Debug)]
pub struct Options {
    pub ctx: usize,
    /// prompt tokens per weight pass
    pub batch: usize,
    /// RAM for pinned streams (MB); experts use what is left of it for their cache
    pub resident_mb: f64,
    /// RAM for the expert cache (MB) on MoE models (0 = from resident_mb)
    pub expert_cache_mb: f64,
    /// store K and V as fp16
    pub kv16: bool,
    /// positions a sliding-window ring keeps beyond the window (allows rolling back this far)
    pub kv_margin: usize,
    pub threads: usize,
}

impl Default for Options {
    fn default() -> Self {
        Options { ctx: 2048, batch: 32, resident_mb: 0.0, expert_cache_mb: 0.0, kv16: false, kv_margin: 1024, threads: 0 }
    }
}

enum Kv {
    F32 { k: Vec<f32>, v: Vec<f32> },
    F16 { k: Vec<u16>, v: Vec<u16> },
}
struct KvLayer {
    cap: usize,
    kv: Kv,
}

/// Row-by-row access to the embedding without loading it (one row is read per token).
struct Embed {
    codec: Codec,
    rows: usize,
    cols: usize,
    /// absolute file offsets
    es0: u64,
    sm0: u64,
    off: Vec<u32>,
    lut: Vec<u16>,
    // rans4
    r: usize,
    freq: [u32; 256],
}

#[derive(Default, Clone, Debug)]
pub struct Timing {
    pub wait_ms: f64,
    pub read_ms: f64,
    pub comp_ms: f64,
    pub lm_ms: f64,
    pub attn_ms: f64,
    pub bytes_read: u64,
}

pub struct Engine {
    /// set from any thread to abort a running prefill / generation at the next chunk or token (the call returns an error)
    pub abort: Arc<std::sync::atomic::AtomicBool>,
    pub vault: Vault,
    pub m: ModelMeta,
    pub opt: Options,
    half: usize,
    cos: Vec<f32>,
    sin: Vec<f32>,
    kv: Vec<KvLayer>,
    has_ring: bool,
    pub pos: usize,
    store: StreamStore,
    experts: Option<ExpertCache>,
    embed: Embed,
    pub logits: Vec<f32>,
    /// logits of every position of the last forward with `all_logits` (n x vocab)
    pub all_logits: Vec<f32>,
    /// last-token hidden state after every layer of the last forward (for verification)
    pub dump_hidden: Option<Vec<f32>>,
    pub timing: Timing,
    /// layers [0, n) run by another device (split mode)
    pub prefix: Option<Box<dyn crate::backend::LayerPrefix>>,
}

impl Engine {
    pub fn open(vault: Vault, opt: Options) -> Result<Engine> {
        let m = vault.meta().clone();
        ensure!(m.layer_resident.len > 0, "vault has no layer layout");
        ensure!(m.heads % m.kv_heads == 0, "heads {} not a multiple of kv_heads {}", m.heads, m.kv_heads);
        let (hd, kvd) = (m.head_dim, m.kv_heads * m.head_dim);
        ensure!(!opt.kv16 || kvd % 8 == 0, "fp16 KV needs kv_heads * head_dim divisible by 8");
        let half = hd / 2;
        let ctx = opt.ctx;
        let (cos, sin) = rope_tables(&m, &vault.manifest.rope_inv, ctx)?;
        // KV
        let mut kv = Vec::with_capacity(m.layers);
        let mut has_ring = false;
        for l in 0..m.layers {
            let sliding = m.gptoss.as_ref().map_or(false, |g| g.sliding_window > 0 && g.layer_sliding.get(l).copied() == Some(1));
            let sw = m.gptoss.as_ref().map_or(0, |g| g.sliding_window);
            let cap = if sliding && sw + opt.kv_margin < ctx { sw + opt.kv_margin } else { ctx };
            has_ring |= cap < ctx;
            kv.push(KvLayer {
                cap,
                kv: if opt.kv16 {
                    Kv::F16 { k: vec![0; cap * kvd], v: vec![0; cap * kvd] }
                } else {
                    Kv::F32 { k: vec![0.0; cap * kvd], v: vec![0.0; cap * kvd] }
                },
            });
        }
        let kv_bytes: f64 = kv.iter().map(|l| 2.0 * (l.cap * kvd) as f64 * if opt.kv16 { 2.0 } else { 4.0 }).sum();
        let store = StreamStore::new(&vault, &m, opt.resident_mb * 1e6, kv_bytes)?;
        let experts = if let Some(mo) = &m.moe {
            let budget = if opt.expert_cache_mb > 0.0 { opt.expert_cache_mb * 1e6 } else { (opt.resident_mb * 1e6 - store.pinned_bytes() as f64).max(0.0) };
            Some(ExpertCache::new(&vault, mo, m.layers, budget))
        } else {
            None
        };
        let embed = Embed::open(&vault, &m)?;
        Ok(Engine {
            abort: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            logits: vec![0.0; m.vocab],
            vault,
            m,
            opt,
            half,
            cos,
            sin,
            kv,
            has_ring,
            pos: 0,
            store,
            experts,
            embed,
            all_logits: Vec::new(),
            dump_hidden: None,
            timing: Timing::default(),
            prefix: None,
        })
    }

    pub fn has_ring(&self) -> bool {
        self.has_ring
    }
    pub fn pinned_bytes(&self) -> u64 {
        self.store.pinned_bytes()
    }
    pub fn expert_read_ms(&self) -> f64 {
        self.experts.as_ref().map_or(0.0, |e| e.read_ms)
    }
    pub fn expert_stats(&self) -> Option<(u64, u64, usize)> {
        self.experts.as_ref().map(|e| (e.hits, e.misses, e.capacity()))
    }

    /// Move the position back (keeps the KV cache up to `p`). Fails for a sliding ring rolled back past its margin.
    pub fn set_pos(&mut self, p: usize) -> Result<()> {
        ensure!(p <= self.pos, "setpos {p} beyond position {}", self.pos);
        ensure!(!self.has_ring || self.pos - p <= self.opt.kv_margin, "ring {p} (pos {}, rollback limit {})", self.pos, self.opt.kv_margin);
        self.pos = p;
        Ok(())
    }

    /// Run `tok` (n tokens) through the model at the current position. `logits`: 0 none, 1 last token, 2 every token.
    pub fn forward(&mut self, tok: &[u32], logits: u8, dump: bool) -> Result<()> {
        let n = tok.len();
        let m = self.m.clone();
        let hid = m.hidden;
        ensure!(self.pos + n <= self.opt.ctx, "context full ({} + {n} > {})", self.pos, self.opt.ctx);
        let mut h = vec![0f32; n * hid];
        for (t, &x) in tok.iter().enumerate() {
            self.embed.row(&self.vault, x as usize, &mut h[t * hid..(t + 1) * hid])?;
        }
        self.timing = Timing::default();
        let mut dumps = if dump { Some(Vec::with_capacity(m.layers * hid)) } else { None };
        let pos0 = self.pos;
        let l0 = match self.prefix.as_mut() {
            Some(p) => {
                let t0 = Instant::now();
                p.run(&mut h, n, pos0, dumps.as_mut())?;
                self.timing.comp_ms += t0.elapsed().as_secs_f64() * 1e3;
                p.layers().min(m.layers)
            }
            None => 0,
        };
        let ns = m.layers + usize::from(logits > 0);
        let sid = |i: usize| if i < m.layers { i } else { m.lm_head_stream };
        let next_of = |i: usize, st: &StreamStore| -> Option<usize> {
            ((i + 1)..ns).map(sid).find(|&s| !st.is_pinned(s)).or_else(|| (l0..m.layers).find(|&s| !st.is_pinned(s)))
        };
        for l in l0..m.layers {
            let t0 = Instant::now();
            let nx = next_of(l, &self.store);
            let (blob, bytes, rms) = self.store.get(l, nx)?;
            self.timing.wait_ms += t0.elapsed().as_secs_f64() * 1e3;
            self.timing.read_ms += rms;
            self.timing.bytes_read += bytes;
            let t1 = Instant::now();
            self.layer(l, &blob, &mut h, n, pos0)?;
            self.timing.comp_ms += t1.elapsed().as_secs_f64() * 1e3;
            if let Some(d) = dumps.as_mut() {
                d.extend_from_slice(&h[(n - 1) * hid..n * hid]);
            }
        }
        self.dump_hidden = dumps;
        if logits > 0 {
            let t0 = Instant::now();
            let nx = next_of(ns - 1, &self.store);
            let (blob, bytes, rms) = self.store.get(m.lm_head_stream, nx)?;
            self.timing.wait_ms += t0.elapsed().as_secs_f64() * 1e3;
            self.timing.read_ms += rms;
            self.timing.bytes_read += bytes;
            let t1 = Instant::now();
            let subs = stream::sub_blobs(&blob)?;
            let head = self.mat(subs[0])?;
            let fnrm = self.vault.final_norm();
            let eps = m.rms_eps as f32;
            if logits == 2 {
                let mut xn = vec![0f32; n * hid];
                for t in 0..n {
                    kernels::rmsnorm(&mut xn[t * hid..(t + 1) * hid], &h[t * hid..(t + 1) * hid], fnrm, eps);
                }
                self.all_logits.resize(n * m.vocab, 0.0);
                head.mv(&xn, &mut self.all_logits, n);
                self.logits.copy_from_slice(&self.all_logits[(n - 1) * m.vocab..]);
            } else {
                let mut xn = vec![0f32; hid];
                kernels::rmsnorm(&mut xn, &h[(n - 1) * hid..n * hid], fnrm, eps);
                head.mv(&xn, &mut self.logits, 1);
            }
            self.timing.lm_ms = t1.elapsed().as_secs_f64() * 1e3;
            self.timing.comp_ms += self.timing.lm_ms;
        }
        self.pos += n;
        Ok(())
    }

    fn mat<'a>(&self, blob: &'a [u8]) -> Result<Mat<'a>> {
        Ok(match self.m.codec {
            Codec::Huff4 => Mat::Huff(cube_vault::huff4::Huff4::parse(blob)?),
            Codec::Rans4 => Mat::Rans(cube_vault::rans4::Rans4::parse(blob)?),
        })
    }

    fn layer(&mut self, l: usize, blob: &[u8], h: &mut [f32], n: usize, pos0: usize) -> Result<()> {
        let m = &self.m;
        let (hid, hd, nh, nkv) = (m.hidden, m.head_dim, m.heads, m.kv_heads);
        let (qd, kvd) = (nh * hd, nkv * hd);
        let eps = m.rms_eps as f32;
        let subs = stream::sub_blobs(blob)?;
        let rl: Vec<f32> = self.vault.layer_resident(l).to_vec();
        let lr = &m.layer_resident;
        let sl = |o: Option<usize>, len: usize| o.map(|o| &rl[o..o + len]);
        let (lni, lnp) = (&rl[lr.ln_in..lr.ln_in + hid], &rl[lr.ln_post..lr.ln_post + hid]);
        let (qn, kn) = (sl(lr.q_norm, hd), sl(lr.k_norm, hd));
        let (bq, bk, bv, bo) = (sl(lr.bq, qd), sl(lr.bk, kvd), sl(lr.bv, kvd), sl(lr.bo, hid));
        let mut xn = vec![0f32; n * hid];
        for t in 0..n {
            kernels::rmsnorm(&mut xn[t * hid..(t + 1) * hid], &h[t * hid..(t + 1) * hid], lni, eps);
        }
        let (mut q, mut k, mut v) = (vec![0f32; n * qd], vec![0f32; n * kvd], vec![0f32; n * kvd]);
        self.mat(subs[0])?.mv(&xn, &mut q, n);
        self.mat(subs[1])?.mv(&xn, &mut k, n);
        self.mat(subs[2])?.mv(&xn, &mut v, n);
        add_bias(&mut q, n, qd, bq);
        add_bias(&mut k, n, kvd, bk);
        add_bias(&mut v, n, kvd, bv);
        let ta = Instant::now();
        let mut att = vec![0f32; n * qd];
        let grp = nh / nkv;
        let scale = 1.0f32 / (hd as f32).sqrt();
        let gpt = m.gptoss.as_ref();
        let sliding = gpt.map_or(false, |g| g.sliding_window > 0 && g.layer_sliding.get(l).copied() == Some(1));
        let swin = gpt.map_or(0, |g| g.sliding_window);
        let sinks = lr.sinks.map(|o| &rl[o..o + nh]);
        let kvl = &mut self.kv[l];
        let cap = kvl.cap;
        let ring = cap < self.opt.ctx;
        for t in 0..n {
            let p = pos0 + t;
            let qt = &mut q[t * qd..(t + 1) * qd];
            let kt = &mut k[t * kvd..(t + 1) * kvd];
            if let Some(w) = qn {
                for hh in 0..nh {
                    kernels::rmsnorm_inplace(&mut qt[hh * hd..(hh + 1) * hd], w, eps);
                }
            }
            if let Some(w) = kn {
                for hh in 0..nkv {
                    kernels::rmsnorm_inplace(&mut kt[hh * hd..(hh + 1) * hd], w, eps);
                }
            }
            let cs = &self.cos[p * self.half..(p + 1) * self.half];
            let sn = &self.sin[p * self.half..(p + 1) * self.half];
            for hh in 0..nh {
                kernels::rope_head(&mut qt[hh * hd..(hh + 1) * hd], cs, sn);
            }
            for hh in 0..nkv {
                kernels::rope_head(&mut kt[hh * hd..(hh + 1) * hd], cs, sn);
            }
            let slot = (if ring { p % cap } else { p }) * kvd;
            let vt = &v[t * kvd..(t + 1) * kvd];
            match &mut kvl.kv {
                Kv::F32 { k: kc, v: vc } => {
                    kc[slot..slot + kvd].copy_from_slice(kt);
                    vc[slot..slot + kvd].copy_from_slice(vt);
                }
                Kv::F16 { k: kc, v: vc } => {
                    for d in 0..kvd {
                        kc[slot + d] = f32_to_f16(kt[d]);
                        vc[slot + d] = f32_to_f16(vt[d]);
                    }
                }
            }
            let t0 = if sliding && p + 1 > swin { p + 1 - swin } else { 0 };
            let qt: &[f32] = qt;
            let kvref = &kvl.kv;
            let out: Vec<Vec<f32>> = (0..nh)
                .into_par_iter()
                .map(|hh| {
                    let qh = &qt[hh * hd..(hh + 1) * hd];
                    let kvh = hh / grp;
                    let mut sc = vec![0f32; p + 1 - t0];
                    let mut mx = -1e30f32;
                    let snk = sinks.map_or(0.0, |s| s[hh]);
                    if sinks.is_some() && snk > mx {
                        mx = snk;
                    }
                    let idx = |tt: usize| (if ring { tt % cap } else { tt }) * kvd + kvh * hd;
                    match kvref {
                        Kv::F32 { k: kc, .. } => {
                            for tt in t0..=p {
                                let kk = &kc[idx(tt)..idx(tt) + hd];
                                let mut s = 0f32;
                                for d in 0..hd {
                                    s += qh[d] * kk[d];
                                }
                                s *= scale;
                                sc[tt - t0] = s;
                                if s > mx {
                                    mx = s;
                                }
                            }
                        }
                        Kv::F16 { k: kc, .. } => {
                            for tt in t0..=p {
                                let kk = &kc[idx(tt)..idx(tt) + hd];
                                let mut a = [0f32; 8];
                                let mut d = 0;
                                while d < hd {
                                    for ln in 0..8 {
                                        a[ln] = qh[d + ln].mul_add(f16_to_f32(kk[d + ln]), a[ln]);
                                    }
                                    d += 8;
                                }
                                let s = (((a[0] + a[4]) + (a[1] + a[5])) + ((a[2] + a[6]) + (a[3] + a[7]))) * scale;
                                sc[tt - t0] = s;
                                if s > mx {
                                    mx = s;
                                }
                            }
                        }
                    }
                    let mut sum = if sinks.is_some() { (snk - mx).exp() } else { 0.0 };
                    for s in sc.iter_mut() {
                        *s = (*s - mx).exp();
                        sum += *s;
                    }
                    let inv = 1.0f32 / sum;
                    let mut o = vec![0f32; hd];
                    match kvref {
                        Kv::F32 { v: vc, .. } => {
                            for tt in t0..=p {
                                let vv = &vc[idx(tt)..idx(tt) + hd];
                                let w = sc[tt - t0] * inv;
                                for d in 0..hd {
                                    o[d] += w * vv[d];
                                }
                            }
                        }
                        Kv::F16 { v: vc, .. } => {
                            for tt in t0..=p {
                                let vv = &vc[idx(tt)..idx(tt) + hd];
                                let w = sc[tt - t0] * inv;
                                for d in 0..hd {
                                    o[d] = w.mul_add(f16_to_f32(vv[d]), o[d]);
                                }
                            }
                        }
                    }
                    o
                })
                .collect();
            for (hh, o) in out.into_iter().enumerate() {
                att[t * qd + hh * hd..t * qd + (hh + 1) * hd].copy_from_slice(&o);
            }
        }
        self.timing.attn_ms += ta.elapsed().as_secs_f64() * 1e3;
        let mut o = vec![0f32; n * hid];
        self.mat(subs[3])?.mv(&att, &mut o, n);
        add_bias(&mut o, n, hid, bo);
        for i in 0..n * hid {
            h[i] += o[i];
        }
        for t in 0..n {
            kernels::rmsnorm(&mut xn[t * hid..(t + 1) * hid], &h[t * hid..(t + 1) * hid], lnp, eps);
        }
        match m.arch {
            Arch::Gptoss => self.gptoss_moe(l, subs[4], &rl, &xn, h, n),
            Arch::QwenMoe => self.qwen_moe(l, subs[4], &xn, h, n),
            Arch::Dense => {
                let inter = m.inter;
                let (bg, bu, bd) = (sl(lr.bg, inter), sl(lr.bu, inter), sl(lr.bd, hid));
                let (mut g, mut u) = (vec![0f32; n * inter], vec![0f32; n * inter]);
                self.mat(subs[4])?.mv(&xn, &mut g, n);
                self.mat(subs[5])?.mv(&xn, &mut u, n);
                add_bias(&mut g, n, inter, bg);
                add_bias(&mut u, n, inter, bu);
                for i in 0..n * inter {
                    let x = g[i];
                    g[i] = x / (1.0f32 + (-x).exp()) * u[i];
                }
                self.mat(subs[6])?.mv(&g, &mut o, n);
                add_bias(&mut o, n, hid, bd);
                for i in 0..n * hid {
                    h[i] += o[i];
                }
                Ok(())
            }
        }
    }

    /// top-k by repeated first-maximum (the C engine's selection order and tie-breaking)
    fn topk(z: &[f32], k: usize) -> Vec<usize> {
        let mut sel: Vec<usize> = Vec::with_capacity(k);
        for _ in 0..k {
            let mut b: Option<usize> = None;
            for (e, &v) in z.iter().enumerate() {
                if sel.contains(&e) {
                    continue;
                }
                if b.map_or(true, |bb| v > z[bb]) {
                    b = Some(e);
                }
            }
            sel.push(b.unwrap());
        }
        sel
    }

    /// group (token, weight) rows by expert, ascending expert id
    fn plan(sel: &[Vec<usize>], wt: &[Vec<f32>], ne: usize) -> Vec<(usize, Vec<usize>, Vec<f32>)> {
        let mut out = Vec::new();
        for e in 0..ne {
            let (mut rows, mut ws) = (Vec::new(), Vec::new());
            for (t, s) in sel.iter().enumerate() {
                for (j, &x) in s.iter().enumerate() {
                    if x == e {
                        rows.push(t);
                        ws.push(wt[t][j]);
                    }
                }
            }
            if !rows.is_empty() {
                out.push((e, rows, ws));
            }
        }
        out
    }

    fn qwen_moe(&mut self, l: usize, router: &[u8], xn: &[f32], h: &mut [f32], n: usize) -> Result<()> {
        let m = self.m.clone();
        let mo = m.moe.as_ref().unwrap();
        let (hid, ne, kk, inter) = (m.hidden, mo.experts, mo.top_k, mo.inter);
        let mut lgt = vec![0f32; n * ne];
        self.mat(router)?.mv(xn, &mut lgt, n);
        let mut sel = Vec::with_capacity(n);
        let mut wt = Vec::with_capacity(n);
        for t in 0..n {
            let z = &mut lgt[t * ne..(t + 1) * ne];
            let mut mx = z[0];
            for &v in z.iter().skip(1) {
                if v > mx {
                    mx = v;
                }
            }
            let mut sum = 0f32;
            for v in z.iter_mut() {
                *v = (*v - mx).exp();
                sum += *v;
            }
            for v in z.iter_mut() {
                *v /= sum;
            }
            let s = Self::topk(z, kk);
            let mut w: Vec<f32> = s.iter().map(|&e| z[e]).collect();
            let mut ws = 0f32;
            for &x in &w {
                ws += x;
            }
            if mo.norm_topk {
                for x in w.iter_mut() {
                    *x /= ws;
                }
            }
            sel.push(s);
            wt.push(w);
        }
        let plan = Self::plan(&sel, &wt, ne);
        let ids: Vec<usize> = plan.iter().map(|p| p.0).collect();
        let blobs: Vec<Blob> = self.experts.as_mut().unwrap().get_many(&self.vault, l, &ids, &mut self.timing)?.into_iter().map(|b| b.expect("Qwen MoE experts have no device tier")).collect();
        let codec = m.codec;
        fn mk(codec: Codec, b: &[u8]) -> Result<Mat<'_>> {
            Ok(match codec {
                Codec::Huff4 => Mat::Huff(cube_vault::huff4::Huff4::parse(b)?),
                Codec::Rans4 => Mat::Rans(cube_vault::rans4::Rans4::parse(b)?),
            })
        }
        let mut o = vec![0f32; n * hid];
        let mut dall: Vec<Vec<f32>> = Vec::with_capacity(plan.len());
        for ((_, rows, _), blob) in plan.iter().zip(&blobs) {
            let subs = stream::sub_blobs(blob)?;
            let mm = rows.len();
            let mut xs = vec![0f32; mm * hid];
            for (i, &r) in rows.iter().enumerate() {
                xs[i * hid..(i + 1) * hid].copy_from_slice(&xn[r * hid..(r + 1) * hid]);
            }
            let (mut g, mut u) = (vec![0f32; mm * inter], vec![0f32; mm * inter]);
            mk(codec, subs[0])?.mv(&xs, &mut g, mm);
            mk(codec, subs[1])?.mv(&xs, &mut u, mm);
            for i in 0..mm * inter {
                let x = g[i];
                g[i] = x / (1.0f32 + (-x).exp()) * u[i];
            }
            let mut d = vec![0f32; mm * hid];
            mk(codec, subs[2])?.mv(&g, &mut d, mm);
            dall.push(d);
        }
        for ((_, rows, ws), d) in plan.iter().zip(&dall) {
            for (i, &r) in rows.iter().enumerate() {
                let w = ws[i];
                for kx in 0..hid {
                    o[r * hid + kx] += w * d[i * hid + kx];
                }
            }
        }
        for i in 0..n * hid {
            h[i] += o[i];
        }
        Ok(())
    }

    fn gptoss_moe(&mut self, l: usize, router: &[u8], rl: &[f32], xn: &[f32], h: &mut [f32], n: usize) -> Result<()> {
        let m = self.m.clone();
        let mo = m.moe.as_ref().unwrap();
        let g = m.gptoss.as_ref().unwrap();
        let (hid, ne, kk, inter) = (m.hidden, mo.experts, mo.top_k, mo.inter);
        let (lim, al) = (g.swiglu_limit as f32, g.swiglu_alpha as f32);
        let rb = &rl[m.layer_resident.router_bias.context("gpt-oss vault without router bias")?..][..ne];
        let mut lgt = vec![0f32; n * ne];
        self.mat(router)?.mv(xn, &mut lgt, n);
        let mut sel = Vec::with_capacity(n);
        let mut wt = Vec::with_capacity(n);
        for t in 0..n {
            let z = &mut lgt[t * ne..(t + 1) * ne];
            for e in 0..ne {
                z[e] += rb[e];
            }
            let s = Self::topk(z, kk);
            let mx = z[s[0]];
            let mut w: Vec<f32> = s.iter().map(|&e| (z[e] - mx).exp()).collect();
            let mut sum = 0f32;
            for &x in &w {
                sum += x;
            }
            for x in w.iter_mut() {
                *x /= sum;
            }
            sel.push(s);
            wt.push(w);
        }
        let plan = Self::plan(&sel, &wt, ne);
        let ids: Vec<usize> = plan.iter().map(|p| p.0).collect();
        let blobs = self.experts.as_mut().unwrap().get_many(&self.vault, l, &ids, &mut self.timing)?;
        let mut dall: Vec<Vec<f32>> = Vec::with_capacity(plan.len());
        for ((e, rows, _), blob) in plan.iter().zip(&blobs) {
            let Some(blob) = blob else {
                // held in VRAM: computed on the GPU
                let ec = self.experts.as_mut().unwrap();
                let key = ec.key(l, *e);
                let mm = rows.len();
                let mut xs = vec![0f32; mm * hid];
                for (i, &r) in rows.iter().enumerate() {
                    xs[i * hid..(i + 1) * hid].copy_from_slice(&xn[r * hid..(r + 1) * hid]);
                }
                let mut d = vec![0f32; mm * hid];
                ec.dev.as_mut().unwrap().run(key, &xs, mm, &mut d)?;
                dall.push(d);
                continue;
            };
            let mm = rows.len();
            let mut xs = vec![0f32; mm * hid];
            for (i, &r) in rows.iter().enumerate() {
                xs[i * hid..(i + 1) * hid].copy_from_slice(&xn[r * hid..(r + 1) * hid]);
            }
            let mut d = vec![0f32; mm * hid];
            gptoss_expert(blob, &xs, mm, hid, inter, lim, al, &mut d)?;
            dall.push(d);
        }
        let mut o = vec![0f32; n * hid];
        for ((_, rows, ws), d) in plan.iter().zip(&dall) {
            for (i, &r) in rows.iter().enumerate() {
                let w = ws[i];
                for kx in 0..hid {
                    o[r * hid + kx] += w * d[i * hid + kx];
                }
            }
        }
        for i in 0..n * hid {
            h[i] += o[i];
        }
        Ok(())
    }
}

/// One gpt-oss expert on the CPU: `out` (m x hid) = down(clamped SwiGLU(gate_up(xs) + b_gu)) + b_down.
#[allow(clippy::too_many_arguments)]
pub fn gptoss_expert(blob: &[u8], xs: &[f32], mm: usize, hid: usize, inter: usize, lim: f32, al: f32, d: &mut [f32]) -> Result<()> {
    let subs = stream::sub_blobs(blob)?;
    let bgu = mxf::f32_view(subs[1])?;
    let bd = mxf::f32_view(subs[3])?;
    let gu_m = Mat::Mxf(mxf::Mxf::parse(subs[0])?);
    let mut gu = vec![0f32; mm * 2 * inter];
    gu_m.mv(xs, &mut gu, mm);
    let mut a = vec![0f32; mm * inter];
    for i in 0..mm {
        for j in 0..inter {
            let mut gt = gu[i * 2 * inter + 2 * j] + bgu[2 * j];
            let mut up = gu[i * 2 * inter + 2 * j + 1] + bgu[2 * j + 1];
            if gt > lim {
                gt = lim;
            }
            if up > lim {
                up = lim;
            }
            if up < -lim {
                up = -lim;
            }
            a[i * inter + j] = (up + 1.0f32) * (gt / (1.0f32 + (-al * gt).exp()));
        }
    }
    Mat::Mxf(mxf::Mxf::parse(subs[2])?).mv(&a, d, mm);
    for i in 0..mm {
        for kx in 0..hid {
            d[i * hid + kx] += bd[kx];
        }
    }
    Ok(())
}

/// Stand-in device tier that holds experts in host memory and computes them with `gptoss_expert`: exercises the tiering
/// logic exactly as the VRAM tier would, on machines without a GPU (tests, `--dev-experts-cpu`).
pub struct HostExperts {
    pub map: std::collections::HashMap<usize, (Vec<u8>, u64)>,
    pub cap: usize,
    protected: Vec<usize>,
    clock: u64,
    hid: usize,
    inter: usize,
    lim: f32,
    al: f32,
}

impl HostExperts {
    pub fn new(m: &ModelMeta, cap: usize) -> Result<HostExperts> {
        let mo = m.moe.as_ref().context("no MoE metadata")?;
        let go = m.gptoss.as_ref().context("no gpt-oss metadata")?;
        Ok(HostExperts { map: Default::default(), cap, protected: vec![], clock: 0, hid: m.hidden, inter: mo.inter, lim: go.swiglu_limit as f32, al: go.swiglu_alpha as f32 })
    }
}

impl crate::store::ExpertDevice for HostExperts {
    fn contains(&self, key: usize) -> bool {
        self.map.contains_key(&key)
    }
    fn protect(&mut self, keys: &[usize]) {
        self.clock += 1;
        for k in keys {
            if let Some(x) = self.map.get_mut(k) {
                x.1 = self.clock;
            }
        }
        self.protected = keys.to_vec();
    }
    fn insert(&mut self, key: usize, blob: &[u8]) -> Result<bool> {
        if self.map.contains_key(&key) || self.cap == 0 {
            return Ok(false);
        }
        if self.map.len() >= self.cap {
            let Some(vk) = self.map.iter().filter(|(k, _)| !self.protected.contains(k)).min_by_key(|(_, v)| v.1).map(|(k, _)| *k) else { return Ok(false) };
            self.map.remove(&vk);
        }
        self.map.insert(key, (blob.to_vec(), self.clock));
        Ok(true)
    }
    fn run(&mut self, key: usize, xs: &[f32], m: usize, out: &mut [f32]) -> Result<()> {
        let b = &self.map.get(&key).context("expert not held")?.0;
        gptoss_expert(b, xs, m, self.hid, self.inter, self.lim, self.al, out)
    }
    fn capacity(&self) -> usize {
        self.cap
    }
    fn describe(&self) -> String {
        format!("{} expert slots in host memory (test tier)", self.cap)
    }
}

/// cos/sin tables (ctx x head_dim/2) for the model's rotary embedding, computed like the C engine.
pub fn rope_tables(m: &ModelMeta, rope_inv: &[f32], ctx: usize) -> Result<(Vec<f32>, Vec<f32>)> {
    let hd = m.head_dim;
    let half = hd / 2;
    let mut cos = vec![0f32; ctx * half];
    let mut sin = vec![0f32; ctx * half];
    for i in 0..half {
        match &m.rope {
            Rope::Table { scale } => {
                let rf = *rope_inv.get(i).context("rope table shorter than head_dim/2")?;
                let sc = *scale as f32;
                for p in 0..ctx {
                    let a = p as f32 * rf;
                    cos[p * half + i] = a.cos() * sc;
                    sin[p * half + i] = a.sin() * sc;
                }
            }
            rope => {
                let theta = match rope {
                    Rope::Default { theta } | Rope::Llama3 { theta, .. } => *theta,
                    _ => unreachable!(),
                };
                let mut fr = theta.powf(-2.0 * i as f64 / hd as f64);
                if let Rope::Llama3 { factor, low, high, orig, .. } = rope {
                    let wl = 2.0 * std::f64::consts::PI / fr;
                    let (lo, hi) = (orig / low, orig / high);
                    if wl > lo {
                        fr /= factor;
                    } else if wl >= hi {
                        let sm = (orig / wl - low) / (high - low);
                        fr = (1.0 - sm) * fr / factor + sm * fr;
                    }
                }
                for p in 0..ctx {
                    let a = p as f64 * fr;
                    cos[p * half + i] = a.cos() as f32;
                    sin[p * half + i] = a.sin() as f32;
                }
            }
        }
    }
    Ok((cos, sin))
}

fn add_bias(y: &mut [f32], n: usize, d: usize, b: Option<&[f32]>) {
    if let Some(b) = b {
        for t in 0..n {
            for i in 0..d {
                y[t * d + i] += b[i];
            }
        }
    }
}

impl Embed {
    fn open(v: &Vault, m: &ModelMeta) -> Result<Embed> {
        let s = *v.stream(m.embed_stream);
        let mut h8 = [0u8; 16];
        v.read_at(s.off, &mut h8)?;
        ensure!(u32::from_le_bytes(h8[0..4].try_into()?) == stream::MAGIC, "embedding stream: bad magic");
        let mut o = [0u8; 8];
        v.read_at(s.off + 8, &mut o)?;
        let sub = s.off + u64::from_le_bytes(o);
        match m.codec {
            Codec::Huff4 => {
                let mut hdr = vec![0u8; 264];
                v.read_at(sub, &mut hdr)?;
                let rows = u32::from_le_bytes(hdr[0..4].try_into()?) as usize;
                let cols = u32::from_le_bytes(hdr[4..8].try_into()?) as usize;
                let no4 = 4 * rows + 1;
                let mut ob = vec![0u8; 4 * no4];
                v.read_at(sub + 264, &mut ob)?;
                let off: Vec<u32> = ob.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
                let mut len = [0u8; 256];
                len.copy_from_slice(&hdr[8..264]);
                let es0 = sub + 264 + 4 * no4 as u64;
                let sm0 = es0 + off[4 * rows] as u64 + 8;
                Ok(Embed { codec: Codec::Huff4, rows, cols, es0, sm0, off, lut: build_lut(&len), r: 0, freq: [0; 256] })
            }
            Codec::Rans4 => {
                let mut hdr = vec![0u8; 524];
                v.read_at(sub, &mut hdr)?;
                let rows = u32::from_le_bytes(hdr[0..4].try_into()?) as usize;
                let cols = u32::from_le_bytes(hdr[4..8].try_into()?) as usize;
                ensure!(hdr[8] == 12 && hdr[9] >= 1, "embedding: unsupported rans4 parameters");
                let r = hdr[9] as usize;
                let nblk = rows.div_ceil(r);
                let no = 4 * nblk + 1;
                let mut ob = vec![0u8; 4 * no];
                v.read_at(sub + 524, &mut ob)?;
                let off: Vec<u32> = ob.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
                let mut freq = [0u32; 256];
                for s in 0..256 {
                    freq[s] = u16::from_le_bytes([hdr[12 + 2 * s], hdr[13 + 2 * s]]) as u32;
                }
                let es0 = sub + 524 + 4 * no as u64;
                let sm0 = es0 + off[4 * nblk] as u64 + 8;
                Ok(Embed { codec: Codec::Rans4, rows, cols, es0, sm0, off, lut: vec![], r, freq })
            }
        }
    }

    fn row(&self, v: &Vault, tok: usize, out: &mut [f32]) -> Result<()> {
        if tok >= self.rows {
            bail!("token id {tok} out of range ({} rows)", self.rows);
        }
        let cols = self.cols;
        let mut sm = vec![0u8; cols];
        v.read_at(self.sm0 + (tok * cols) as u64, &mut sm)?;
        let mut e = vec![0u8; cols];
        match self.codec {
            Codec::Huff4 => {
                let o = &self.off[4 * tok..4 * tok + 5];
                let a = o[0];
                let mut es = vec![0u8; (o[4] - a) as usize + 8];
                v.read_at(self.es0 + a as u64, &mut es)?;
                // single-symbol decode, the same as the C engine's embed_row
                let mut p = [0usize, (o[1] - a) as usize, (o[2] - a) as usize, (o[3] - a) as usize];
                let mut b = [0u64; 4];
                let mut c = [0u32; 4];
                for j in 0..cols {
                    let k = j & 3;
                    while c[k] < LMAX {
                        b[k] |= (es.get(p[k]).copied().unwrap_or(0) as u64) << c[k];
                        p[k] += 1;
                        c[k] += 8;
                    }
                    let ent = self.lut[(b[k] & ((1 << LMAX) - 1)) as usize];
                    let l = (ent >> 8) as u32;
                    b[k] >>= l;
                    c[k] -= l;
                    e[j] = ent as u8;
                }
            }
            Codec::Rans4 => {
                let (bk, i) = (tok / self.r, tok % self.r);
                let o = &self.off[4 * bk..4 * bk + 5];
                let a = o[0];
                let mut cs = vec![0u8; (o[4] - a) as usize + 8];
                v.read_at(self.es0 + a as u64, &mut cs)?;
                let mut cum = [0u32; 256];
                let mut acc = 0;
                for s in 0..256 {
                    cum[s] = acc;
                    acc += self.freq[s];
                }
                let mut s2s = vec![0u8; 4096];
                for s in 0..256 {
                    for q in 0..self.freq[s] {
                        s2s[(cum[s] + q) as usize] = s as u8;
                    }
                }
                let mut ee = vec![0u8; (i + 1) * cols];
                for k in 0..4 {
                    let mut p = (o[k] - a) as usize;
                    let mut x = u32::from_le_bytes(cs[p..p + 4].try_into()?) as u64;
                    p += 4;
                    for rr in 0..=i {
                        let mut j = k;
                        while j < cols {
                            let slot = x & 4095;
                            let s = s2s[slot as usize] as usize;
                            x = self.freq[s] as u64 * (x >> 12) + slot - cum[s] as u64;
                            while x < (1 << 23) {
                                x = (x << 8) | cs[p] as u64;
                                p += 1;
                            }
                            ee[rr * cols + j] = s as u8;
                            j += 4;
                        }
                    }
                }
                e.copy_from_slice(&ee[i * cols..(i + 1) * cols]);
            }
        }
        for j in 0..cols {
            out[j] = f32::from_bits((((sm[j] as u32) & 0x80) << 24) | ((e[j] as u32) << 23) | (((sm[j] as u32) & 0x7F) << 16));
        }
        Ok(())
    }
}

pub type Blob = Arc<Vec<u8>>;

/// How a generation ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum End {
    Eos,
    Max,
    Stop,
    Ctx,
}
impl End {
    pub fn as_str(&self) -> &'static str {
        match self {
            End::Eos => "eos",
            End::Max => "max",
            End::Stop => "stop",
            End::Ctx => "ctx",
        }
    }
}

/// Timing of one generation (the fields of the serve protocol's `S` line).
#[derive(Clone, Debug, Default)]
pub struct GenStats {
    pub tok_s: f64,
    pub ms_tok: f64,
    pub p50_ms: f64,
    pub stall_pct: f64,
    pub read_mbps: f64,
    pub prefill_ms: f64,
    pub streamed_mb_tok: f64,
    pub resident_mb: f64,
    pub decode_tokens: usize,
}

impl GenStats {
    pub fn line(&self) -> String {
        if self.decode_tokens == 0 {
            return format!("tok_s=0 ms_tok=0 stall_pct=0 read_MBps=0 prefill_ms={:.0}", self.prefill_ms);
        }
        format!(
            "tok_s={:.3} ms_tok={:.0} p50_ms={:.0} stall_pct={:.0} read_MBps={:.0} prefill_ms={:.0} streamed_MB_tok={:.0} resident_MB={:.0}",
            self.tok_s, self.ms_tok, self.p50_ms, self.stall_pct, self.read_mbps, self.prefill_ms, self.streamed_mb_tok, self.resident_mb
        )
    }
}

impl Engine {
    /// Prefill `prompt` at the current position (chunks of the batch size, logits for the last one).
    pub fn prefill(&mut self, prompt: &[u32]) -> Result<()> {
        crate::backend::prefill(self, prompt)
    }

    /// The serve protocol's GEN, in process (see [`crate::backend::generate`]).
    #[allow(clippy::too_many_arguments)]
    pub fn generate(
        &mut self,
        prompt: &[u32],
        max_new: usize,
        temp: f32,
        topp: f32,
        topk: i32,
        seed: u64,
        eos: &[u32],
        on_token: &mut dyn FnMut(&[u32]) -> bool,
    ) -> Result<(Vec<u32>, End, GenStats)> {
        crate::backend::generate(self, prompt, max_new, temp, topp, topk, seed, eos, on_token)
    }

    /// Add a device tier under the RAM expert cache (gpt-oss experts in VRAM).
    pub fn attach_expert_device(&mut self, d: Box<dyn crate::store::ExpertDevice>) {
        if let Some(e) = self.experts.as_mut() {
            e.dev = Some(d);
        }
    }
    /// (RAM hits, device hits, misses, RAM slots, device slots)
    pub fn expert_tiers(&self) -> Option<(u64, u64, u64, usize, usize)> {
        self.experts.as_ref().map(|e| (e.hits - e.dev_hits, e.dev_hits, e.misses, e.capacity(), e.dev.as_ref().map_or(0, |d| d.capacity())))
    }

    /// Run layers `[0, n)` elsewhere (the GPU) from now on; their pinned streams are released.
    pub fn attach_prefix(&mut self, p: Box<dyn crate::backend::LayerPrefix>) {
        for l in 0..p.layers().min(self.m.layers) {
            self.store.unpin(l);
        }
        self.prefix = Some(p);
    }
}

impl crate::backend::Backend for Engine {
    fn forward(&mut self, tok: &[u32], logits: u8, dump: bool) -> Result<()> {
        Engine::forward(self, tok, logits, dump)
    }
    fn logits(&self) -> &[f32] {
        &self.logits
    }
    fn all_logits(&self) -> &[f32] {
        &self.all_logits
    }
    fn dump_hidden(&self) -> Option<&[f32]> {
        self.dump_hidden.as_deref()
    }
    fn pos(&self) -> usize {
        self.pos
    }
    fn set_pos(&mut self, p: usize) -> Result<()> {
        Engine::set_pos(self, p)
    }
    fn reset(&mut self) {
        self.pos = 0;
    }
    fn ctx(&self) -> usize {
        self.opt.ctx
    }
    fn batch(&self) -> usize {
        self.opt.batch
    }
    fn abort_flag(&self) -> Arc<std::sync::atomic::AtomicBool> {
        self.abort.clone()
    }
    fn timing(&self) -> Timing {
        self.timing.clone()
    }
    fn pinned_bytes(&self) -> u64 {
        self.store.pinned_bytes()
    }
    fn meta(&self) -> &ModelMeta {
        &self.m
    }
    fn expert_stats(&self) -> Option<(u64, u64, usize)> {
        Engine::expert_stats(self)
    }
}
