//! `serve`: the line protocol of `nvme_engine_u.exe serve`, so the existing tools and the app can drive this engine.
//!
//! ```text
//! -> READY
//! <- PING                      -> PONG
//! <- RESET                     -> OK
//! <- SETPOS p                  -> OK p | ERR ...
//! <- LOGITS n \n tokens        -> L v1 .. vV \n END dbg 0 pos
//! <- GEN max temp topp topk seed n \n tokens
//!                              -> T tok (per token) \n S tok_s=... \n END reason new pos   (STOP line aborts)
//! <- VERIFY n \n tokens        -> A argmax1 .. argmaxN \n END verify n pos ms
//! <- VERIFYP n temp topp topk \n tokens -> P k id p .. (per token) \n END verifyp n pos ms
//! <- MOESTATS                  -> M hits misses read_ms slots 0 0 0 0 0 0 0 0 0
//! <- QUIT
//! ```

use crate::backend::Backend;
use crate::sample::{self, Rng};
use anyhow::Result;
use std::fs::File;
use std::io::{BufRead, Write};
use std::sync::mpsc::{Receiver, TryRecvError};
use std::time::Instant;

/// C `printf("%.9g")` for an f32 (9 significant digits, %g style).
pub fn g9(v: f32) -> String {
    let v = v as f64;
    if v.is_nan() {
        return if v.is_sign_negative() { "-nan".into() } else { "nan".into() };
    }
    if v.is_infinite() {
        return if v < 0.0 { "-inf".into() } else { "inf".into() };
    }
    if v == 0.0 {
        return if v.is_sign_negative() { "-0".into() } else { "0".into() };
    }
    let e = format!("{:.8e}", v);
    let (mant, exp) = e.split_once('e').unwrap();
    let x: i32 = exp.parse().unwrap();
    if x < -4 || x >= 9 {
        let mut m = mant.to_string();
        if m.contains('.') {
            while m.ends_with('0') {
                m.pop();
            }
            if m.ends_with('.') {
                m.pop();
            }
        }
        format!("{}e{}{:02}", m, if x < 0 { '-' } else { '+' }, x.abs())
    } else {
        let prec = (8 - x).max(0) as usize;
        let mut s = format!("{:.*}", prec, v);
        if s.contains('.') {
            while s.ends_with('0') {
                s.pop();
            }
            if s.ends_with('.') {
                s.pop();
            }
        }
        s
    }
}

fn parse_toks(line: &str, n: usize) -> Vec<u32> {
    line.split_whitespace().take(n).map(|t| t.parse::<i64>().unwrap_or(0) as u32).collect()
}

fn pct(v: &[f64], q: f64) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    let mut t = v.to_vec();
    t.sort_by(|a, b| a.partial_cmp(b).unwrap());
    t[((t.len() - 1) as f64 * q) as usize]
}

pub struct Server {
    pub e: Box<dyn Backend>,
    /// a sliding-window ring can roll back at most this far (None: no ring)
    pub ring_margin: Option<usize>,
    pub log: Option<File>,
    pub dump: Option<File>,
    pub ignore_eos: bool,
}

impl Server {
    fn lg(&mut self, s: &str) {
        if let Some(f) = self.log.as_mut() {
            let _ = writeln!(f, "{s}");
        }
    }

    fn fwd(&mut self, t: &[u32], logits: u8) -> Result<()> {
        let dump = self.dump.is_some();
        self.e.forward(t, logits, dump)?;
        if let (Some(f), Some(h)) = (self.dump.as_mut(), self.e.dump_hidden()) {
            let b: Vec<u8> = h.iter().flat_map(|x| x.to_le_bytes()).collect();
            f.write_all(&b)?;
        }
        Ok(())
    }

    /// prefill in chunks of the batch size; logits for the last chunk
    fn prefill(&mut self, pt: &[u32]) -> Result<()> {
        let b = self.e.batch().max(1);
        let mut i = 0;
        while i < pt.len() {
            let n = (pt.len() - i).min(b);
            let last = i + n == pt.len();
            self.fwd(&pt[i..i + n], u8::from(last))?;
            i += n;
        }
        Ok(())
    }

    pub fn run(&mut self, rx: Receiver<String>, out: &mut impl Write) -> Result<()> {
        writeln!(out, "READY")?;
        out.flush()?;
        let mut pending: Vec<String> = Vec::new();
        let next = |pending: &mut Vec<String>| -> Option<String> {
            if !pending.is_empty() {
                return Some(pending.remove(0));
            }
            rx.recv().ok()
        };
        let ctx = self.e.ctx();
        while let Some(line) = next(&mut pending) {
            let l = line.trim_end_matches(['\r', '\n']);
            let w: Vec<&str> = l.split_whitespace().collect();
            let cmd = w.first().copied().unwrap_or("");
            match cmd {
                "" => {}
                "QUIT" => break,
                "PING" => writeln!(out, "PONG")?,
                "RESET" => {
                    self.e.reset();
                    self.lg("[reset] kv cache cleared");
                    writeln!(out, "OK")?
                }
                "SETPOS" => {
                    let p: i64 = w.get(1).and_then(|x| x.parse().ok()).unwrap_or(-1);
                    let pos = self.e.pos();
                    if p < 0 || p as usize > pos {
                        writeln!(out, "ERR setpos {p} (pos {pos})")?
                    } else if self.ring_margin.map_or(false, |mg| pos - p as usize > mg) {
                        writeln!(out, "ERR ring {p} (pos {pos}, rollback limit {})", self.ring_margin.unwrap())?
                    } else {
                        self.e.set_pos(p as usize)?;
                        writeln!(out, "OK {p}")?
                    }
                }
                "MOESTATS" => {
                    let (h, m, c) = self.e.expert_stats().unwrap_or((0, 0, 0));
                    let ms = 0.0f64;
                    writeln!(out, "M {h} {m} {ms:.1} {c} 0 0 0 0.0 0 0.0 0 0 0")?
                }
                "VERIFY" | "VERIFYP" => {
                    let p = cmd == "VERIFYP";
                    let n: usize = w.get(1).and_then(|x| x.parse().ok()).unwrap_or(0);
                    let (temp, topp, topk) = (
                        w.get(2).and_then(|x| x.parse().ok()).unwrap_or(0.7f32),
                        w.get(3).and_then(|x| x.parse().ok()).unwrap_or(0.8f32),
                        w.get(4).and_then(|x| x.parse().ok()).unwrap_or(20i32),
                    );
                    if p && w.len() != 5 {
                        writeln!(out, "ERR bad VERIFYP")?;
                        let _ = next(&mut pending);
                        out.flush()?;
                        continue;
                    }
                    let Some(tl) = next(&mut pending) else { break };
                    if n < 1 || n > self.e.batch() || self.e.pos() + n > ctx {
                        writeln!(out, "ERR verify n={n} pos={} batch={} ctx={ctx}", self.e.pos(), self.e.batch())?;
                        out.flush()?;
                        continue;
                    }
                    let pt = parse_toks(&tl, n);
                    let t0 = Instant::now();
                    self.fwd(&pt, 2)?;
                    let v = self.e.meta().vocab;
                    if p {
                        for t in 0..n {
                            let d = sample::distribution(&self.e.all_logits()[t * v..(t + 1) * v], temp, topp, topk);
                            let mut s = format!("P {}", d.len());
                            for (id, pr) in d {
                                s.push_str(&format!(" {id} {}", g9(pr)));
                            }
                            writeln!(out, "{s}")?;
                        }
                        writeln!(out, "END verifyp {n} {} {:.0}", self.e.pos(), t0.elapsed().as_secs_f64() * 1e3)?;
                    } else {
                        let mut s = String::from("A");
                        for t in 0..n {
                            s.push_str(&format!(" {}", sample::argmax(&self.e.all_logits()[t * v..(t + 1) * v])));
                        }
                        writeln!(out, "{s}\nEND verify {n} {} {:.0}", self.e.pos(), t0.elapsed().as_secs_f64() * 1e3)?;
                    }
                }
                "LOGITS" => {
                    let n: usize = w.get(1).and_then(|x| x.parse().ok()).unwrap_or(0);
                    let Some(tl) = next(&mut pending) else { break };
                    let pt = parse_toks(&tl, n);
                    if let Err(er) = self.prefill(&pt) {
                        writeln!(out, "ERR {er}")?;
                        out.flush()?;
                        continue;
                    }
                    if let Some(f) = self.dump.as_mut() {
                        f.flush()?;
                    }
                    let mut s = String::with_capacity(self.e.meta().vocab * 14);
                    s.push('L');
                    for &x in self.e.logits() {
                        s.push(' ');
                        s.push_str(&g9(x));
                    }
                    writeln!(out, "{s}\nEND dbg 0 {}", self.e.pos())?;
                }
                "GEN" => {
                    if w.len() != 7 {
                        writeln!(out, "ERR bad GEN")?;
                        out.flush()?;
                        continue;
                    }
                    let max_new: usize = w[1].parse().unwrap_or(0);
                    let temp: f32 = w[2].parse().unwrap_or(0.7);
                    let topp: f32 = w[3].parse().unwrap_or(0.8);
                    let topk: i32 = w[4].parse().unwrap_or(20);
                    let seed: u64 = w[5].parse().unwrap_or(0);
                    let n: usize = w[6].parse().unwrap_or(0);
                    let Some(tl) = next(&mut pending) else { break };
                    let pt = parse_toks(&tl, n);
                    self.gen(&pt, max_new, temp, topp, topk, seed, &rx, &mut pending, out)?;
                }
                _ => writeln!(out, "ERR unknown command")?,
            }
            out.flush()?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn gen(
        &mut self,
        pt: &[u32],
        max_new: usize,
        temp: f32,
        topp: f32,
        topk: i32,
        seed: u64,
        rx: &Receiver<String>,
        pending: &mut Vec<String>,
        out: &mut impl Write,
    ) -> Result<()> {
        let ctx = self.e.ctx();
        if self.e.pos() + pt.len() + 1 > ctx {
            writeln!(out, "ERR ctx {} {ctx}", self.e.pos())?;
            return Ok(());
        }
        let mut rng = Rng::new(seed);
        let tp = Instant::now();
        self.prefill(pt)?;
        let prefill_ms = tp.elapsed().as_secs_f64() * 1e3;
        let eos: Vec<i64> = self.e.meta().eos.clone();
        let mut tok = sample::sample(self.e.logits(), temp, topp, topk, &mut rng);
        let mut new_tok = 0usize;
        let reason;
        let mut tms: Vec<f64> = Vec::new();
        let (mut sw, mut sr, mut sb) = (0f64, 0f64, 0f64);
        let stop_seen = |pending: &mut Vec<String>| -> bool {
            loop {
                match rx.try_recv() {
                    Ok(l) if l.trim_start().starts_with("STOP") => return true,
                    Ok(l) => pending.push(l),
                    Err(TryRecvError::Empty) | Err(TryRecvError::Disconnected) => return false,
                }
            }
        };
        loop {
            writeln!(out, "T {tok}")?;
            out.flush()?;
            new_tok += 1;
            if !self.ignore_eos && eos.contains(&(tok as i64)) {
                reason = "eos";
                break;
            }
            if new_tok >= max_new {
                self.fwd(&[tok as u32], 0)?;
                reason = "max";
                break;
            }
            if stop_seen(pending) {
                self.fwd(&[tok as u32], 0)?;
                reason = "stop";
                break;
            }
            if self.e.pos() + 1 >= ctx {
                reason = "ctx";
                break;
            }
            let t1 = Instant::now();
            self.fwd(&[tok as u32], 1)?;
            tok = sample::sample(self.e.logits(), temp, topp, topk, &mut rng);
            tms.push(t1.elapsed().as_secs_f64() * 1e3);
            let t = self.e.timing();
            sw += t.wait_ms;
            sr += t.read_ms;
            sb += t.bytes_read as f64;
        }
        let nd = tms.len();
        if nd > 0 {
            let dec: f64 = tms.iter().sum();
            let stall = if dec > 0.0 { 100.0 * sw / dec } else { 0.0 };
            writeln!(
                out,
                "S tok_s={:.3} ms_tok={:.0} p50_ms={:.0} stall_pct={:.0} read_MBps={:.0} prefill_ms={:.0} streamed_MB_tok={:.0} resident_MB={:.0}",
                1000.0 * nd as f64 / dec,
                dec / nd as f64,
                pct(&tms, 0.5),
                stall,
                if sr > 0.0 { sb / 1e3 / sr } else { 0.0 },
                prefill_ms,
                sb / 1e6 / nd as f64,
                self.e.pinned_bytes() as f64 / 1e6
            )?;
        } else {
            writeln!(out, "S tok_s=0 ms_tok=0 stall_pct=0 read_MBps=0 prefill_ms={prefill_ms:.0}")?;
        }
        self.lg(&format!("[gen] prompt_tokens={} new_tokens={new_tok} reason={reason} pos_after={}", pt.len(), self.e.pos()));
        writeln!(out, "END {reason} {new_tok} {}", self.e.pos())?;
        out.flush()?;
        Ok(())
    }
}

/// Lines of stdin on a channel (lets GEN notice a STOP line without blocking).
pub fn stdin_lines() -> Receiver<String> {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let stdin = std::io::stdin();
        let mut lock = stdin.lock();
        loop {
            let mut s = String::new();
            match lock.read_line(&mut s) {
                Ok(0) | Err(_) => break,
                Ok(_) => {
                    if tx.send(s).is_err() {
                        break;
                    }
                }
            }
        }
    });
    rx
}

#[cfg(test)]
mod tests {
    use super::g9;
    #[test]
    fn g9_like_printf() {
        assert_eq!(g9(1.0), "1");
        assert_eq!(g9(0.1), "0.100000001");
        assert_eq!(g9(-12.5), "-12.5");
        assert_eq!(g9(1e-5), "9.99999975e-06");
        assert_eq!(g9(123456789.0), "123456792");
        assert_eq!(g9(1e10), "1e+10");
        assert_eq!(g9(3.0e-4), "0.000300000014");
    }
}
