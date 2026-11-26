//! What a session needs from an engine (CPU, GPU-resident or split), and generation on top of it.

use crate::model::{End, GenStats, Timing};
use anyhow::{ensure, Result};
use cube_vault::meta::ModelMeta;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

pub trait Backend: Send {
    /// Run `tok` at the current position. `logits`: 0 none, 1 last token, 2 every token.
    fn forward(&mut self, tok: &[u32], logits: u8, dump: bool) -> Result<()>;
    fn logits(&self) -> &[f32];
    fn all_logits(&self) -> &[f32];
    fn dump_hidden(&self) -> Option<&[f32]> {
        None
    }
    fn pos(&self) -> usize;
    fn set_pos(&mut self, p: usize) -> Result<()>;
    fn reset(&mut self);
    fn ctx(&self) -> usize;
    fn batch(&self) -> usize;
    fn abort_flag(&self) -> Arc<AtomicBool>;
    fn timing(&self) -> Timing;
    fn pinned_bytes(&self) -> u64;
    fn meta(&self) -> &ModelMeta;
    fn expert_stats(&self) -> Option<(u64, u64, usize)> {
        None
    }
}

/// Layers `[0, layers())` of a model run on another device (split mode): hidden states in, hidden states out.
pub trait LayerPrefix: Send {
    fn layers(&self) -> usize;
    /// `h`: n x hidden at positions pos0..pos0+n; `dump`: append the last token's hidden state after each layer
    fn run(&mut self, h: &mut [f32], n: usize, pos0: usize, dump: Option<&mut Vec<f32>>) -> Result<()>;
}

/// Prefill `prompt` at the current position (chunks of the batch size, logits for the last one).
pub fn prefill(b: &mut dyn Backend, prompt: &[u32]) -> Result<()> {
    let bs = b.batch().max(1);
    let abort = b.abort_flag();
    let mut i = 0;
    while i < prompt.len() {
        ensure!(!abort.load(Ordering::SeqCst), "aborted");
        let n = (prompt.len() - i).min(bs);
        let last = i + n == prompt.len();
        b.forward(&prompt[i..i + n], u8::from(last), false)?;
        i += n;
    }
    Ok(())
}

/// The serve protocol's GEN, in process: prefill, then sample up to `max_new` tokens. `on_token(tokens so far)` runs after
/// every token and returns true to stop. Same token sequence as `nvme_engine_u` for the same seed and logits.
#[allow(clippy::too_many_arguments)]
pub fn generate(
    b: &mut dyn Backend,
    prompt: &[u32],
    max_new: usize,
    temp: f32,
    topp: f32,
    topk: i32,
    seed: u64,
    eos: &[u32],
    on_token: &mut dyn FnMut(&[u32]) -> bool,
) -> Result<(Vec<u32>, End, GenStats)> {
    let ctx = b.ctx();
    ensure!(b.pos() + prompt.len() + 1 <= ctx, "context full: position {} + {} prompt tokens > {ctx}", b.pos(), prompt.len());
    let abort = b.abort_flag();
    let mut rng = crate::sample::Rng::new(seed);
    let tp = Instant::now();
    prefill(b, prompt)?;
    let prefill_ms = tp.elapsed().as_secs_f64() * 1e3;
    let mut tok = crate::sample::sample(b.logits(), temp, topp, topk, &mut rng) as u32;
    let mut out: Vec<u32> = Vec::new();
    let mut tms: Vec<f64> = Vec::new();
    let (mut sw, mut sr, mut sb) = (0f64, 0f64, 0f64);
    let end;
    loop {
        ensure!(!abort.load(Ordering::SeqCst), "aborted");
        out.push(tok);
        let stop = on_token(&out);
        if eos.contains(&tok) {
            end = End::Eos;
            break;
        }
        if out.len() >= max_new {
            b.forward(&[tok], 0, false)?;
            end = End::Max;
            break;
        }
        if stop {
            b.forward(&[tok], 0, false)?;
            end = End::Stop;
            break;
        }
        if b.pos() + 1 >= ctx {
            end = End::Ctx;
            break;
        }
        let t1 = Instant::now();
        b.forward(&[tok], 1, false)?;
        tok = crate::sample::sample(b.logits(), temp, topp, topk, &mut rng) as u32;
        tms.push(t1.elapsed().as_secs_f64() * 1e3);
        let t = b.timing();
        sw += t.wait_ms;
        sr += t.read_ms;
        sb += t.bytes_read as f64;
    }
    let nd = tms.len();
    let mut st = GenStats { prefill_ms, decode_tokens: nd, resident_mb: b.pinned_bytes() as f64 / 1e6, ..Default::default() };
    if nd > 0 {
        let dec: f64 = tms.iter().sum();
        let mut s = tms.clone();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        st.tok_s = 1000.0 * nd as f64 / dec;
        st.ms_tok = dec / nd as f64;
        st.p50_ms = s[((nd - 1) as f64 * 0.5) as usize];
        st.stall_pct = if dec > 0.0 { 100.0 * sw / dec } else { 0.0 };
        st.read_mbps = if sr > 0.0 { sb / 1e3 / sr } else { 0.0 };
        st.streamed_mb_tok = sb / 1e6 / nd as f64;
    }
    Ok((out, end, st))
}
