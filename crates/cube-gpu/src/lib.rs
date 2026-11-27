//! The GPU engine (CUDA via cudarc): weights stay compressed in VRAM and are decoded inside every matvec — the port of the C
//! harness's gpu_engine.cu. Two uses:
//!
//! * [`GpuDense`] as a [`Backend`]: the whole dense model in VRAM (GPU-resident mode).
//! * [`GpuDense`] as a [`LayerPrefix`]: layers `[0, n)` in VRAM, the CPU engine runs the rest (split mode).
//!
//! The kernels (kernels.cu) are compiled to PTX by nvcc at build time when it is installed, otherwise by NVRTC the first time
//! the GPU is used. The CUDA driver and NVRTC are loaded at run time, so the program runs on machines without CUDA (the GPU
//! is then simply not offered). fp32 throughout; sums run in a different order than on the CPU, so logits agree to ~1e-5,
//! not bit for bit.

use anyhow::{anyhow, bail, ensure, Context, Result};
use cube_engine::backend::{Backend, LayerPrefix};
use cube_engine::model::{rope_tables, Timing};
use cube_vault::meta::{Arch, ModelMeta};
use cube_vault::{stream, Codec, Vault};
use cudarc::driver::result as cu;
use cudarc::driver::sys::{CUdeviceptr, CUfunction, CUmodule};
use cudarc::driver::CudaContext;
use std::ffi::{c_void, CString};
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, OnceLock};
use std::time::Instant;

const NBMAX: usize = 16;
const KERNELS: &str = include_str!("kernels.cu");
const PTX_BUILT: &str = include_str!(concat!(env!("OUT_DIR"), "/kernels.ptx"));

/// The CUDA device, if this machine has one and its driver loads.
pub struct Gpu {
    pub ctx: Arc<CudaContext>,
    pub name: String,
    pub cc: (i32, i32),
}

static GPU: OnceLock<Option<Arc<Gpu>>> = OnceLock::new();

/// The first CUDA device (None when there is no NVIDIA driver / device). Probed once.
pub fn gpu() -> Option<Arc<Gpu>> {
    GPU.get_or_init(|| {
        // no NVIDIA driver library: no GPU (checked first so nothing panics)
        if !unsafe { cudarc::driver::sys::is_culib_present() } {
            return None;
        }
        let r = std::panic::catch_unwind(|| -> Result<Gpu> {
            let ctx = CudaContext::new(0).map_err(|e| anyhow!("{e:?}"))?;
            Ok(Gpu { name: ctx.name().unwrap_or_default(), cc: ctx.compute_capability().unwrap_or((0, 0)), ctx })
        });
        match r {
            Ok(Ok(g)) => Some(Arc::new(g)),
            _ => None,
        }
    })
    .clone()
}

/// (free, total) bytes of VRAM.
pub fn vram() -> Option<(usize, usize)> {
    let g = gpu()?;
    g.ctx.bind_to_thread().ok()?;
    cu::mem_get_info().ok()
}

struct Kernels {
    _module: CUmodule,
    f: std::collections::HashMap<&'static str, CUfunction>,
}
unsafe impl Send for Kernels {}
unsafe impl Sync for Kernels {}

const NAMES: [&str; 15] = [
    "mvb_huff1", "mvb_huff4", "mvb_huff16", "embed_huff", "mvb_rans1", "mvb_rans4", "mvb_rans16", "embed_rans", "rms", "rope", "attn", "silu_mul", "mxf4", "act_gptoss", "add_bias",
];

static KER: OnceLock<std::result::Result<Arc<Kernels>, String>> = OnceLock::new();

/// PTX of kernels.cu: built by nvcc at compile time, else compiled now by NVRTC.
fn ptx(cc: (i32, i32)) -> Result<String> {
    if !PTX_BUILT.trim().is_empty() {
        return Ok(PTX_BUILT.to_string());
    }
    if !unsafe { cudarc::nvrtc::sys::is_culib_present() } {
        bail!("the GPU kernels need NVRTC (nvrtc64_*.dll from the CUDA toolkit) on PATH, or a build with nvcc");
    }
    let arch: &'static str = match cc.0 * 10 + cc.1 {
        x if x >= 90 => "compute_90",
        x if x >= 89 => "compute_89",
        x if x >= 86 => "compute_86",
        x if x >= 80 => "compute_80",
        x if x >= 75 => "compute_75",
        x if x >= 70 => "compute_70",
        _ => "compute_61",
    };
    let opts = cudarc::nvrtc::CompileOptions { arch: Some(arch), fmad: Some(true), options: vec!["--std=c++14".into(), "-default-device".into()], ..Default::default() };
    let p = cudarc::nvrtc::compile_ptx_with_opts(KERNELS, opts).map_err(|e| anyhow!("NVRTC could not compile the GPU kernels (is the CUDA toolkit's nvrtc on PATH?): {e:?}"))?;
    Ok(p.to_src())
}

fn kernels(g: &Gpu) -> Result<Arc<Kernels>> {
    KER.get_or_init(|| {
        (|| -> Result<Arc<Kernels>> {
            g.ctx.bind_to_thread().map_err(|e| anyhow!("{e:?}"))?;
            let src = CString::new(ptx(g.cc)?)?;
            let module = unsafe { cu::module::load_data(src.as_ptr() as *const c_void) }.map_err(|e| anyhow!("loading the GPU kernels: {e:?}"))?;
            let mut f = std::collections::HashMap::new();
            for n in NAMES {
                let h = unsafe { cu::module::get_function(module, CString::new(n)?) }.map_err(|e| anyhow!("kernel {n}: {e:?}"))?;
                f.insert(n, h);
            }
            Ok(Arc::new(Kernels { _module: module, f }))
        })()
        .map_err(|e| format!("{e:#}"))
    })
    .clone()
    .map_err(|e| anyhow!(e))
}

/// A matrix in VRAM (device addresses of its parts).
#[derive(Clone, Copy, Debug, Default)]
struct DMat {
    es: u64,
    off: u64,
    sm: u64,
    /// huffman4: 4096 x u16 decode table; rans4: 4096 x u32 slot table
    tab: u64,
    rows: i32,
    cols: i32,
    /// rans4 rows per chain block (0 = huffman4)
    r: i32,
}

/// Owner of every device allocation of one model.
struct Mem {
    ptrs: Vec<CUdeviceptr>,
    bytes: u64,
}
impl Mem {
    fn alloc(&mut self, n: usize) -> Result<u64> {
        let p = unsafe { cu::malloc_sync(n.max(4)) }.map_err(|e| anyhow!("VRAM allocation of {:.0} MB failed: {e:?}", n as f64 / 1e6))?;
        unsafe { cu::memset_d8_sync(p, 0, n.max(4)) }.map_err(|e| anyhow!("{e:?}"))?;
        self.ptrs.push(p);
        self.bytes += n as u64;
        Ok(p)
    }
    fn upload<T>(&mut self, v: &[T]) -> Result<u64> {
        let p = self.alloc(std::mem::size_of_val(v))?;
        if !v.is_empty() {
            unsafe { cu::memcpy_htod_sync(p, v) }.map_err(|e| anyhow!("{e:?}"))?;
        }
        Ok(p)
    }
}
impl Drop for Mem {
    fn drop(&mut self) {
        for p in self.ptrs.drain(..) {
            unsafe {
                let _ = cu::free_sync(p);
            }
        }
    }
}

fn u32_at(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes(b[o..o + 4].try_into().unwrap())
}

/// Upload one stream (layer / embedding / lm_head) and describe its matrices.
fn upload_stream(mem: &mut Mem, v: &Vault, si: usize, codec: Codec) -> Result<Vec<DMat>> {
    let host = v.read_stream(si)?;
    let subs = stream::sub_blobs(&host)?;
    let base = mem.upload(&host)?;
    let mut out = Vec::new();
    for (i, b) in subs.iter().enumerate() {
        let o = stream::sub_offset(&host, i);
        let rows = u32_at(b, 0) as usize;
        let cols = u32_at(b, 4) as usize;
        match codec {
            Codec::Huff4 => {
                let no4 = 4 * rows + 1;
                let p_off = 264usize;
                let p_es = p_off + 4 * no4;
                let es_len = u32_at(b, p_off + 4 * (no4 - 1)) as usize;
                let p_sm = p_es + es_len + 8;
                let mut len = [0u8; 256];
                len.copy_from_slice(&b[8..264]);
                let lut = cube_vault::huff4::build_lut(&len);
                let tab = mem.upload(&lut)?;
                out.push(DMat { es: base + o + p_es as u64, off: base + o + p_off as u64, sm: base + o + p_sm as u64, tab, rows: rows as i32, cols: cols as i32, r: 0 });
            }
            Codec::Rans4 => {
                let r = b[9] as usize;
                ensure!(b[8] == 12 && r >= 1, "stream {si}: unsupported rans4 parameters (the GPU engine needs M = 2^12)");
                let nblk = rows.div_ceil(r);
                let p_off = 524usize;
                let p_cs = p_off + 4 * (4 * nblk + 1);
                let cs_len = u32_at(b, p_off + 4 * (4 * nblk)) as usize;
                let p_sm = p_cs + cs_len + 8;
                let mut freq = [0u32; 256];
                for s in 0..256 {
                    freq[s] = u16::from_le_bytes([b[12 + 2 * s], b[13 + 2 * s]]) as u32;
                }
                ensure!(freq.iter().sum::<u32>() == 4096, "stream {si}: bad rans4 frequency table");
                let t = cube_engine::kernels::rans_tab(&freq);
                let tab = mem.upload(&t)?;
                out.push(DMat { es: base + o + p_cs as u64, off: base + o + p_off as u64, sm: base + o + p_sm as u64, tab, rows: rows as i32, cols: cols as i32, r: r as i32 });
            }
        }
    }
    Ok(out)
}

/// A dense decoder in VRAM: all layers (+ embedding and lm_head) for GPU-resident mode, or the first `n` layers for split mode.
pub struct GpuDense {
    gpu: Arc<Gpu>,
    k: Arc<Kernels>,
    mem: Mem,
    pub m: ModelMeta,
    layers: Vec<Vec<DMat>>,
    emb: Option<DMat>,
    lm: Option<DMat>,
    res: u64,
    fnrm: u64,
    cs: u64,
    sn: u64,
    h: u64,
    xn: u64,
    q: u64,
    kb: u64,
    vb: u64,
    att: u64,
    g: u64,
    u: u64,
    lg: u64,
    kc: u64,
    vc: u64,
    scr: u64,
    ctx: usize,
    pub pos: usize,
    logits: Vec<f32>,
    all_logits: Vec<f32>,
    dump: Option<Vec<f32>>,
    abort: Arc<AtomicBool>,
    timing: Timing,
}

// all device state is owned by this struct and used from one thread at a time (the session's)
unsafe impl Send for GpuDense {}

/// Bytes of VRAM a dense model needs (weights + tables + KV + work buffers) for `layers` layers (None = whole model).
pub fn vram_needed(v: &Vault, ctx: usize, layers: Option<usize>) -> u64 {
    let m = v.meta();
    let nl = layers.unwrap_or(m.layers).min(m.layers);
    let mut b: u64 = (0..nl).map(|l| v.stream(l).padded).sum();
    if layers.is_none() {
        b += v.stream(m.embed_stream).padded;
        if m.lm_head_stream != m.embed_stream {
            b += v.stream(m.lm_head_stream).padded;
        }
        b += m.vocab as u64 * 4 * NBMAX as u64;
    }
    let tabs = (nl as u64 + 2) * 7 * 16384;
    let kvd = (m.kv_heads * m.head_dim) as u64;
    let kv = 2 * nl as u64 * ctx as u64 * kvd * 4;
    let bufs = 4 * NBMAX as u64 * (3 * m.hidden as u64 + 2 * (m.heads * m.head_dim) as u64 + 2 * kvd + 2 * m.inter as u64) + 4 * (m.heads * ctx) as u64;
    b + tabs + kv + bufs + (m.layers * m.layer_resident.len + m.hidden) as u64 * 4 + 2 * (ctx * m.head_dim / 2) as u64 * 4
}

macro_rules! args {
    ($($x:expr),* $(,)?) => {{
        let v: Vec<Box<dyn std::any::Any>> = vec![$(Box::new($x)),*];
        v
    }};
}

impl GpuDense {
    /// `layers`: None = the whole model (embedding, all layers, lm_head); Some(n) = only layers [0, n).
    pub fn open(v: &Vault, ctx: usize, layers: Option<usize>) -> Result<GpuDense> {
        let g = gpu().context("no CUDA device")?;
        g.ctx.bind_to_thread().map_err(|e| anyhow!("{e:?}"))?;
        let k = kernels(&g)?;
        let m = v.meta().clone();
        ensure!(m.arch == Arch::Dense, "the GPU engine runs dense decoders (this vault is {:?})", m.arch);
        ensure!(m.head_dim % 2 == 0 && m.head_dim <= 256, "unsupported head_dim {}", m.head_dim);
        let nl = layers.unwrap_or(m.layers).min(m.layers);
        let mut mem = Mem { ptrs: Vec::new(), bytes: 0 };
        let mut ls = Vec::with_capacity(nl);
        for l in 0..nl {
            let mats = upload_stream(&mut mem, v, l, m.codec)?;
            ensure!(mats.len() == 7, "layer {l} has {} matrices, the GPU engine needs 7", mats.len());
            ls.push(mats);
        }
        let (emb, lm) = if layers.is_none() {
            let e = upload_stream(&mut mem, v, m.embed_stream, m.codec)?[0];
            let l = if m.lm_head_stream == m.embed_stream { e } else { upload_stream(&mut mem, v, m.lm_head_stream, m.codec)?[0] };
            (Some(e), Some(l))
        } else {
            (None, None)
        };
        let res = mem.upload(v.resident())?;
        let fnrm = res + (m.layers * m.layer_resident.len * 4) as u64;
        let (cs, sn) = rope_tables(&m, &v.manifest.rope_inv, ctx)?;
        let csd = mem.upload(&cs)?;
        let snd = mem.upload(&sn)?;
        let (hid, qd, kvd) = (m.hidden, m.heads * m.head_dim, m.kv_heads * m.head_dim);
        let f = |n: usize| n * 4;
        let h = mem.alloc(f(NBMAX * hid))?;
        let xn = mem.alloc(f(NBMAX * hid))?;
        let q = mem.alloc(f(NBMAX * qd))?;
        let kb = mem.alloc(f(NBMAX * kvd))?;
        let vb = mem.alloc(f(NBMAX * kvd))?;
        let att = mem.alloc(f(NBMAX * qd))?;
        let gg = mem.alloc(f(NBMAX * m.inter))?;
        let u = mem.alloc(f(NBMAX * m.inter))?;
        let lg = if layers.is_none() { mem.alloc(f(NBMAX * m.vocab))? } else { 0 };
        let kc = mem.alloc(f(nl * ctx * kvd))?;
        let vc = mem.alloc(f(nl * ctx * kvd))?;
        let scr = mem.alloc(f(m.heads * ctx))?;
        Ok(GpuDense {
            gpu: g,
            k,
            mem,
            logits: vec![0.0; m.vocab],
            all_logits: Vec::new(),
            layers: ls,
            emb,
            lm,
            res,
            fnrm,
            cs: csd,
            sn: snd,
            h,
            xn,
            q,
            kb,
            vb,
            att,
            g: gg,
            u,
            lg,
            kc,
            vc,
            scr,
            ctx,
            pos: 0,
            dump: None,
            abort: Arc::new(AtomicBool::new(false)),
            timing: Timing::default(),
            m,
        })
    }

    pub fn vram_bytes(&self) -> u64 {
        self.mem.bytes
    }
    pub fn device_name(&self) -> String {
        self.gpu.name.clone()
    }

    fn launch(&self, name: &str, grid: u32, block: u32, a: Vec<Box<dyn std::any::Any>>) -> Result<()> {
        let f = *self.k.f.get(name).context("kernel")?;
        // kernel parameters: pointers to each argument's value (u64 device pointers, i32, f32)
        let mut ptrs: Vec<*mut c_void> = Vec::with_capacity(a.len());
        let mut a = a;
        for x in a.iter_mut() {
            let p: *mut c_void = if let Some(v) = x.downcast_mut::<u64>() {
                v as *mut u64 as *mut c_void
            } else if let Some(v) = x.downcast_mut::<i32>() {
                v as *mut i32 as *mut c_void
            } else if let Some(v) = x.downcast_mut::<f32>() {
                v as *mut f32 as *mut c_void
            } else {
                bail!("kernel {name}: unsupported argument type");
            };
            ptrs.push(p);
        }
        unsafe { cu::launch_kernel(f, (grid.max(1), 1, 1), (block, 1, 1), 0, std::ptr::null_mut(), &mut ptrs) }.map_err(|e| anyhow!("kernel {name}: {e:?}"))
    }

    /// Y (n x rows) (+)= X (n x cols) W^T + bias
    fn mvb(&self, w: &DMat, x: u64, y: u64, bias: u64, acc: bool, n: usize) -> Result<()> {
        let nb = if n == 1 { 1 } else if n <= 4 { 4 } else { 16 };
        if w.r > 0 {
            let chains = 4 * ((w.rows + w.r - 1) / w.r) as u32;
            self.launch(&format!("mvb_rans{nb}"), chains.div_ceil(128), 128, args![w.es, w.off, w.sm, w.tab, w.rows, w.cols, w.r, x, y, bias, i32::from(acc), n as i32])
        } else {
            self.launch(&format!("mvb_huff{nb}"), (w.rows as u32).div_ceil(16), 64, args![w.es, w.off, w.sm, w.tab, w.rows, w.cols, x, y, bias, i32::from(acc), n as i32])
        }
    }

    fn rl(&self, l: usize, o: Option<usize>) -> u64 {
        o.map_or(0, |o| self.res + ((l * self.m.layer_resident.len + o) * 4) as u64)
    }

    /// n tokens (n <= 16) at positions pos0.. through layer l; H holds n rows
    fn layer(&self, l: usize, pos0: usize, n: usize) -> Result<()> {
        let m = &self.m;
        let (hid, nh, nkv, hd) = (m.hidden, m.heads, m.kv_heads, m.head_dim);
        let (qd, kvd, inter, half) = (nh * hd, nkv * hd, m.inter, hd / 2);
        let eps = m.rms_eps as f32;
        let scale = 1.0f32 / (hd as f32).sqrt();
        let lr = &m.layer_resident;
        let s = &self.layers[l];
        self.launch("rms", n as u32, 256, args![self.h, self.rl(l, Some(lr.ln_in)), self.xn, hid as i32, eps])?;
        self.mvb(&s[0], self.xn, self.q, self.rl(l, lr.bq), false, n)?;
        self.mvb(&s[1], self.xn, self.kb, self.rl(l, lr.bk), false, n)?;
        self.mvb(&s[2], self.xn, self.vb, self.rl(l, lr.bv), false, n)?;
        if lr.q_norm.is_some() {
            self.launch("rms", (n * nh) as u32, 128, args![self.q, self.rl(l, lr.q_norm), self.q, hd as i32, eps])?;
        }
        if lr.k_norm.is_some() {
            self.launch("rms", (n * nkv) as u32, 128, args![self.kb, self.rl(l, lr.k_norm), self.kb, hd as i32, eps])?;
        }
        let lkv = (l * self.ctx * kvd * 4) as u64;
        for t in 0..n {
            let p = pos0 + t;
            let slot = ((l * self.ctx + p) * kvd * 4) as u64;
            self.launch(
                "rope",
                (nh + nkv) as u32,
                64,
                args![self.q + (t * qd * 4) as u64, self.kb + (t * kvd * 4) as u64, self.vb + (t * kvd * 4) as u64, self.kc + slot, self.vc + slot, self.cs + (p * half * 4) as u64, self.sn + (p * half * 4) as u64, nh as i32, hd as i32],
            )?;
        }
        for t in 0..n {
            let p = pos0 + t;
            self.launch("attn", nh as u32, 128, args![self.q + (t * qd * 4) as u64, self.kc + lkv, self.vc + lkv, self.att + (t * qd * 4) as u64, self.scr, nh as i32, nkv as i32, hd as i32, (p + 1) as i32, scale])?;
        }
        self.mvb(&s[3], self.att, self.h, self.rl(l, lr.bo), true, n)?;
        self.launch("rms", n as u32, 256, args![self.h, self.rl(l, Some(lr.ln_post)), self.xn, hid as i32, eps])?;
        self.mvb(&s[4], self.xn, self.g, self.rl(l, lr.bg), false, n)?;
        self.mvb(&s[5], self.xn, self.u, self.rl(l, lr.bu), false, n)?;
        self.launch("silu_mul", ((n * inter) as u32).div_ceil(256), 256, args![self.g, self.u, (n * inter) as i32])?;
        self.mvb(&s[6], self.g, self.h, self.rl(l, lr.bd), true, n)
    }

    fn sync(&self) -> Result<()> {
        cu::ctx::synchronize().map_err(|e| anyhow!("GPU error: {e:?}"))
    }

    fn last_hidden(&self, n: usize) -> Result<Vec<f32>> {
        let hid = self.m.hidden;
        let mut v = vec![0f32; hid];
        unsafe { cu::memcpy_dtoh_sync(&mut v, self.h + ((n - 1) * hid * 4) as u64) }.map_err(|e| anyhow!("{e:?}"))?;
        Ok(v)
    }
}

impl Backend for GpuDense {
    fn forward(&mut self, tok: &[u32], logits: u8, dump: bool) -> Result<()> {
        self.gpu.ctx.bind_to_thread().map_err(|e| anyhow!("{e:?}"))?;
        let (emb, lm) = (self.emb.context("layer-prefix GPU model has no embedding")?, self.lm.unwrap());
        ensure!(self.pos + tok.len() <= self.ctx, "context full ({} + {} > {})", self.pos, tok.len(), self.ctx);
        let t0 = Instant::now();
        let (hid, vocab) = (self.m.hidden, self.m.vocab);
        let mut dumps = if dump { Some(Vec::new()) } else { None };
        if logits == 2 {
            self.all_logits.clear();
        }
        let mut i = 0;
        while i < tok.len() {
            let n = (tok.len() - i).min(NBMAX);
            for t in 0..n {
                let out = self.h + (t * hid * 4) as u64;
                let x = tok[i + t] as i32;
                ensure!((x as usize) < emb.rows as usize, "token id {x} out of range");
                if emb.r > 0 {
                    self.launch("embed_rans", 1, 4, args![emb.es, emb.off, emb.sm, emb.tab, emb.cols, emb.r, x, out])?;
                } else {
                    self.launch("embed_huff", 1, 4, args![emb.es, emb.off, emb.sm, emb.tab, emb.cols, x, out])?;
                }
            }
            let last_chunk = i + n == tok.len();
            for l in 0..self.layers.len() {
                self.layer(l, self.pos, n)?;
                if last_chunk {
                    if let Some(d) = dumps.as_mut() {
                        d.extend(self.last_hidden(n)?);
                    }
                }
            }
            self.pos += n;
            if logits == 2 {
                self.launch("rms", n as u32, 256, args![self.h, self.fnrm, self.xn, hid as i32, self.m.rms_eps as f32])?;
                self.mvb(&lm, self.xn, self.lg, 0, false, n)?;
                let mut v = vec![0f32; n * vocab];
                unsafe { cu::memcpy_dtoh_sync(&mut v, self.lg) }.map_err(|e| anyhow!("{e:?}"))?;
                self.all_logits.extend(v);
            } else if logits == 1 && last_chunk {
                self.launch("rms", 1, 256, args![self.h + ((n - 1) * hid * 4) as u64, self.fnrm, self.xn, hid as i32, self.m.rms_eps as f32])?;
                self.mvb(&lm, self.xn, self.lg, 0, false, 1)?;
                unsafe { cu::memcpy_dtoh_sync(&mut self.logits, self.lg) }.map_err(|e| anyhow!("{e:?}"))?;
            }
            i += n;
        }
        if logits == 2 {
            let n = tok.len();
            self.logits.copy_from_slice(&self.all_logits[(n - 1) * vocab..n * vocab]);
        }
        self.sync()?;
        self.dump = dumps;
        self.timing = Timing { comp_ms: t0.elapsed().as_secs_f64() * 1e3, ..Default::default() };
        Ok(())
    }
    fn logits(&self) -> &[f32] {
        &self.logits
    }
    fn all_logits(&self) -> &[f32] {
        &self.all_logits
    }
    fn dump_hidden(&self) -> Option<&[f32]> {
        self.dump.as_deref()
    }
    fn pos(&self) -> usize {
        self.pos
    }
    fn set_pos(&mut self, p: usize) -> Result<()> {
        ensure!(p <= self.pos, "setpos {p} beyond position {}", self.pos);
        self.pos = p;
        Ok(())
    }
    fn reset(&mut self) {
        self.pos = 0;
    }
    fn ctx(&self) -> usize {
        self.ctx
    }
    fn batch(&self) -> usize {
        NBMAX * 8
    }
    fn abort_flag(&self) -> Arc<AtomicBool> {
        self.abort.clone()
    }
    fn timing(&self) -> Timing {
        self.timing.clone()
    }
    fn pinned_bytes(&self) -> u64 {
        self.mem.bytes
    }
    fn meta(&self) -> &ModelMeta {
        &self.m
    }
}

impl LayerPrefix for GpuDense {
    fn layers(&self) -> usize {
        self.layers.len()
    }
    fn run(&mut self, h: &mut [f32], n: usize, pos0: usize, dump: Option<&mut Vec<f32>>) -> Result<()> {
        self.gpu.ctx.bind_to_thread().map_err(|e| anyhow!("{e:?}"))?;
        let hid = self.m.hidden;
        ensure!(pos0 + n <= self.ctx, "position {} exceeds ctx {}", pos0 + n, self.ctx);
        let mut dump = dump;
        let mut i = 0;
        while i < n {
            let m = (n - i).min(NBMAX);
            unsafe { cu::memcpy_htod_sync(self.h, &h[i * hid..(i + m) * hid]) }.map_err(|e| anyhow!("{e:?}"))?;
            let last = i + m == n;
            for l in 0..self.layers.len() {
                self.layer(l, pos0 + i, m)?;
                if last {
                    if let Some(d) = dump.as_mut() {
                        d.extend(self.last_hidden(m)?);
                    }
                }
            }
            unsafe { cu::memcpy_dtoh_sync(&mut h[i * hid..(i + m) * hid], self.h) }.map_err(|e| anyhow!("{e:?}"))?;
            i += m;
        }
        self.sync()
    }
}

// ---------------------------------------------------------------- gpt-oss experts in VRAM

/// gpt-oss experts held in VRAM (MXF4/MXF2 blobs exactly as in the vault) and computed there; the tier under the RAM cache.
pub struct GpuExperts {
    gpu: Arc<Gpu>,
    k: Arc<Kernels>,
    mem: Mem,
    pool: u64,
    stride: u64,
    slots: Vec<Option<usize>>,
    /// per slot: device addresses of gate_up MXF, gate_up bias, down MXF, down bias
    offs: Vec<[u64; 4]>,
    /// key -> (slot, last use)
    map: std::collections::HashMap<usize, (usize, u64)>,
    protected: Vec<usize>,
    clock: u64,
    hid: usize,
    inter: usize,
    lim: f32,
    al: f32,
    mmax: usize,
    xb: u64,
    gub: u64,
    ab: u64,
    db: u64,
}

unsafe impl Send for GpuExperts {}

impl GpuExperts {
    /// Slots for as many experts as fit in `budget` bytes of VRAM.
    pub fn open(v: &Vault, budget: u64) -> Result<GpuExperts> {
        let g = gpu().context("no CUDA device")?;
        g.ctx.bind_to_thread().map_err(|e| anyhow!("{e:?}"))?;
        let k = kernels(&g)?;
        let m = v.meta();
        ensure!(m.arch == Arch::Gptoss, "the VRAM expert tier is for gpt-oss vaults");
        let mo = m.moe.as_ref().context("no MoE metadata")?;
        let go = m.gptoss.as_ref().context("no gpt-oss metadata")?;
        let nx = m.layers * mo.experts;
        let xsz = (0..nx).map(|i| v.stream(mo.expert_stream0 + i).padded).max().unwrap_or(0);
        let stride = (xsz + 255) & !255;
        let mmax = 128usize;
        let work = (4 * mmax * (2 * m.hidden + 3 * mo.inter)) as u64;
        ensure!(budget > work + stride, "not enough VRAM for even one expert");
        let n = (((budget - work) / stride) as usize).min(nx);
        let mut mem = Mem { ptrs: Vec::new(), bytes: 0 };
        let pool = mem.alloc((n as u64 * stride) as usize)?;
        let xb = mem.alloc(4 * mmax * m.hidden)?;
        let gub = mem.alloc(4 * mmax * 2 * mo.inter)?;
        let ab = mem.alloc(4 * mmax * mo.inter)?;
        let db = mem.alloc(4 * mmax * m.hidden)?;
        Ok(GpuExperts {
            gpu: g,
            k,
            mem,
            pool,
            stride,
            slots: vec![None; n],
            offs: vec![[0; 4]; n],
            map: std::collections::HashMap::new(),
            protected: Vec::new(),
            clock: 0,
            hid: m.hidden,
            inter: mo.inter,
            lim: go.swiglu_limit as f32,
            al: go.swiglu_alpha as f32,
            mmax,
            xb,
            gub,
            ab,
            db,
        })
    }

    pub fn vram_bytes(&self) -> u64 {
        self.mem.bytes
    }

    pub fn describe_short(&self) -> String {
        format!("{} experts, {:.1} GB", self.slots.len(), self.mem.bytes as f64 / 1e9)
    }

    fn launch(&self, name: &str, grid: u32, block: u32, a: Vec<Box<dyn std::any::Any>>) -> Result<()> {
        let f = *self.k.f.get(name).context("kernel")?;
        let mut a = a;
        let mut ptrs: Vec<*mut c_void> = Vec::with_capacity(a.len());
        for x in a.iter_mut() {
            let p: *mut c_void = if let Some(v) = x.downcast_mut::<u64>() {
                v as *mut u64 as *mut c_void
            } else if let Some(v) = x.downcast_mut::<i32>() {
                v as *mut i32 as *mut c_void
            } else if let Some(v) = x.downcast_mut::<f32>() {
                v as *mut f32 as *mut c_void
            } else {
                bail!("kernel {name}: unsupported argument type");
            };
            ptrs.push(p);
        }
        unsafe { cu::launch_kernel(f, (grid.max(1), 1, 1), (block, 1, 1), 0, std::ptr::null_mut(), &mut ptrs) }.map_err(|e| anyhow!("kernel {name}: {e:?}"))
    }
}

impl cube_engine::store::ExpertDevice for GpuExperts {
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
        if self.map.contains_key(&key) || blob.len() as u64 > self.stride || self.slots.is_empty() {
            return Ok(false);
        }
        let subs = cube_vault::stream::sub_blobs(blob)?;
        ensure!(subs.len() == 4 && subs[0].starts_with(b"MXF") && subs[2].starts_with(b"MXF"), "not a gpt-oss expert stream");
        ensure!(subs[1].starts_with(b"F32 ") && subs[3].starts_with(b"F32 "), "gpt-oss expert without F32 biases");
        let so = |i: usize| cube_vault::stream::sub_offset(blob, i);
        let slot = match self.slots.iter().position(|s| s.is_none()) {
            Some(i) => i,
            None => {
                let victim = self.map.iter().filter(|(k, _)| !self.protected.contains(k)).min_by_key(|(_, v)| v.1).map(|(k, v)| (*k, v.0));
                let Some((vk, vs)) = victim else { return Ok(false) };
                self.map.remove(&vk);
                vs
            }
        };
        self.gpu.ctx.bind_to_thread().map_err(|e| anyhow!("{e:?}"))?;
        unsafe { cu::memcpy_htod_sync(self.pool + slot as u64 * self.stride, blob) }.map_err(|e| anyhow!("{e:?}"))?;
        let base = self.pool + slot as u64 * self.stride;
        self.offs[slot] = [base + so(0), base + so(1) + 8, base + so(2), base + so(3) + 8];
        self.slots[slot] = Some(key);
        self.map.insert(key, (slot, self.clock));
        Ok(true)
    }
    fn run(&mut self, key: usize, xs: &[f32], m: usize, out: &mut [f32]) -> Result<()> {
        let (slot, _) = *self.map.get(&key).context("expert not in VRAM")?;
        self.gpu.ctx.bind_to_thread().map_err(|e| anyhow!("{e:?}"))?;
        let [gu, bgu, dn, bd] = self.offs[slot];
        let (hid, inter) = (self.hid, self.inter);
        let mut t = 0;
        while t < m {
            let n = (m - t).min(self.mmax);
            unsafe { cu::memcpy_htod_sync(self.xb, &xs[t * hid..(t + n) * hid]) }.map_err(|e| anyhow!("{e:?}"))?;
            let rows_gu = 2 * inter;
            self.launch("mxf4", (rows_gu as u32).div_ceil(8), 256, args![gu, rows_gu as i32, hid as i32, self.xb, self.gub, n as i32])?;
            self.launch("act_gptoss", ((n * inter) as u32).div_ceil(256), 256, args![self.gub, bgu, self.ab, n as i32, inter as i32, self.lim, self.al])?;
            self.launch("mxf4", (hid as u32).div_ceil(8), 256, args![dn, hid as i32, inter as i32, self.ab, self.db, n as i32])?;
            self.launch("add_bias", ((n * hid) as u32).div_ceil(256), 256, args![self.db, bd, n as i32, hid as i32])?;
            unsafe { cu::memcpy_dtoh_sync(&mut out[t * hid..(t + n) * hid], self.db) }.map_err(|e| anyhow!("{e:?}"))?;
            t += n;
        }
        Ok(())
    }
    fn capacity(&self) -> usize {
        self.slots.len()
    }
    fn describe(&self) -> String {
        format!("{} expert slots in VRAM ({:.0} MB) on {}", self.slots.len(), self.mem.bytes as f64 / 1e6, self.gpu.name)
    }
}
