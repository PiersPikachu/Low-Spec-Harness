//! Weight residency: pinned streams, a one-ahead prefetch of the next streamed stream, and the MoE expert cache.

use crate::model::{Blob, Timing};
use anyhow::{Context, Result};
use cube_vault::meta::{ModelMeta, Moe};
use cube_vault::stream::StreamEntry;
use cube_vault::Vault;
use std::collections::HashMap;
use std::fs::File;
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Instant;

#[cfg(unix)]
pub fn pread(f: &File, off: u64, buf: &mut [u8]) -> std::io::Result<()> {
    use std::os::unix::fs::FileExt;
    f.read_exact_at(buf, off)
}
#[cfg(windows)]
pub fn pread(f: &File, mut off: u64, mut buf: &mut [u8]) -> std::io::Result<()> {
    use std::os::windows::fs::FileExt;
    while !buf.is_empty() {
        let n = f.seek_read(buf, off)?;
        if n == 0 {
            return Err(std::io::ErrorKind::UnexpectedEof.into());
        }
        off += n as u64;
        buf = &mut buf[n..];
    }
    Ok(())
}

/// Read a padded stream: large ones in 8 MB pieces on up to `io_threads` plain threads (never on the compute pool).
pub fn read_entry(f: &File, e: &StreamEntry, io_threads: usize) -> Result<Vec<u8>> {
    let mut buf = Vec::new();
    read_entry_into(f, e, io_threads, &mut buf)?;
    Ok(buf)
}

/// [`read_entry`] into a reused buffer (no page faults once it has grown to the largest stream).
pub fn read_entry_into(f: &File, e: &StreamEntry, io_threads: usize, buf: &mut Vec<u8>) -> Result<()> {
    let n = e.padded as usize;
    buf.resize(n, 0);
    const CH: usize = 8 << 20;
    if n <= CH || io_threads <= 1 {
        pread(f, e.off, buf)?;
        return Ok(());
    }
    let per = n.div_ceil(io_threads).div_ceil(CH) * CH;
    std::thread::scope(|sc| -> Result<()> {
        let hs: Vec<_> = buf.chunks_mut(per).enumerate().map(|(i, c)| sc.spawn(move || pread(f, e.off + (i * per) as u64, c))).collect();
        for h in hs {
            h.join().map_err(|_| anyhow::anyhow!("read thread panicked"))??;
        }
        Ok(())
    })?;
    Ok(())
}

pub struct StreamStore {
    file: Arc<File>,
    entries: Vec<StreamEntry>,
    pinned: HashMap<usize, Blob>,
    pinned_bytes: u64,
    pending: Option<(usize, JoinHandle<Result<(Vec<u8>, f64)>>)>,
    /// the last streamed blob handed out; reclaimed for reuse once the caller drops it
    inflight: Vec<Blob>,
    spare: Vec<Vec<u8>>,
}

impl StreamStore {
    /// Pin streams within `budget` bytes: lm_head first, then layers in van der Corput order (any prefix of the order
    /// is spread evenly through the network), as the C engine does.
    pub fn new(v: &Vault, m: &ModelMeta, budget: f64, _kv_bytes: f64) -> Result<StreamStore> {
        let file = Arc::new(File::open(v.data_path()).with_context(|| format!("cannot open {}", v.data_path().display()))?);
        let entries: Vec<StreamEntry> = (0..v.n_streams()).map(|i| *v.stream(i)).collect();
        let mut order = vec![m.lm_head_stream];
        let mut nb = 1;
        while (1usize << nb) < m.layers {
            nb += 1;
        }
        for k in 0..(1usize << nb) {
            let mut r = 0;
            for q in 0..nb {
                if k & (1 << q) != 0 {
                    r |= 1 << (nb - 1 - q);
                }
            }
            if r < m.layers {
                order.push(r);
            }
        }
        let mut pinned = HashMap::new();
        let mut used = 0u64;
        if budget > 0.0 {
            for s in order {
                let e = entries[s];
                if (used + e.padded) as f64 > budget {
                    continue;
                }
                pinned.insert(s, Arc::new(read_entry(&file, &e, 2)?));
                used += e.padded;
            }
        }
        Ok(StreamStore { file, entries, pinned, pinned_bytes: used, pending: None, inflight: Vec::new(), spare: Vec::new() })
    }

    pub fn pinned_bytes(&self) -> u64 {
        self.pinned_bytes
    }
    pub fn unpin(&mut self, s: usize) {
        if let Some(b) = self.pinned.remove(&s) {
            self.pinned_bytes -= b.len() as u64;
        }
    }
    pub fn is_pinned(&self, s: usize) -> bool {
        self.pinned.contains_key(&s)
    }

    /// Stream `s`, then start reading `next` (when it is not pinned) in the background.
    /// Returns the blob, bytes read from disk for it (0 when pinned) and the ms spent reading it.
    pub fn get(&mut self, s: usize, next: Option<usize>) -> Result<(Blob, u64, f64)> {
        let mut keep = Vec::new();
        for b in self.inflight.drain(..) {
            match Arc::try_unwrap(b) {
                Ok(v) => self.spare.push(v),
                Err(b) => keep.push(b),
            }
        }
        self.inflight = keep;
        let (blob, bytes, read_ms) = if let Some(b) = self.pinned.get(&s) {
            (b.clone(), 0, 0.0)
        } else {
            let got = match self.pending.take() {
                Some((ps, h)) if ps == s => Some(h.join().map_err(|_| anyhow::anyhow!("prefetch thread panicked"))??),
                Some((_, h)) => {
                    let _ = h.join();
                    None
                }
                None => None,
            };
            let (buf, ms) = match got {
                Some(x) => x,
                None => {
                    let t = Instant::now();
                    let mut b = self.spare.pop().unwrap_or_default();
                    read_entry_into(&self.file, &self.entries[s], 2, &mut b)?;
                    (b, t.elapsed().as_secs_f64() * 1e3)
                }
            };
            let a = Arc::new(buf);
            self.inflight.push(a.clone());
            (a, self.entries[s].padded, ms)
        };
        if self.pending.is_none() {
            if let Some(nx) = next.filter(|n| !self.pinned.contains_key(n)) {
                let f = self.file.clone();
                let e = self.entries[nx];
                let mut b = self.spare.pop().unwrap_or_default();
                self.pending = Some((
                    nx,
                    std::thread::spawn(move || {
                        let t = Instant::now();
                        read_entry_into(&f, &e, 2, &mut b)?;
                        Ok((b, t.elapsed().as_secs_f64() * 1e3))
                    }),
                ));
            }
        }
        Ok((blob, bytes, read_ms))
    }
}

/// LRU cache of expert streams.
pub struct ExpertCache {
    file: Arc<File>,
    s0: usize,
    ne: usize,
    entries: Vec<StreamEntry>,
    cap: usize,
    map: HashMap<usize, (Blob, u64)>,
    tick: u64,
    pub hits: u64,
    pub misses: u64,
    pub read_ms: f64,
}

impl ExpertCache {
    pub fn new(v: &Vault, mo: &Moe, layers: usize, budget: f64) -> ExpertCache {
        let nx = layers * mo.experts;
        let entries: Vec<StreamEntry> = (0..nx).map(|i| *v.stream(mo.expert_stream0 + i)).collect();
        let xsz = entries.iter().map(|e| e.padded).max().unwrap_or(1).max(1);
        let cap = ((budget / xsz as f64) as usize).max(mo.top_k).min(nx);
        ExpertCache {
            file: Arc::new(File::open(v.data_path()).expect("vault.bin")),
            s0: mo.expert_stream0,
            ne: mo.experts,
            entries,
            cap,
            map: HashMap::new(),
            tick: 0,
            hits: 0,
            misses: 0,
            read_ms: 0.0,
        }
    }
    pub fn capacity(&self) -> usize {
        self.cap
    }
    pub fn first_stream(&self) -> usize {
        self.s0
    }

    /// The experts `ids` of layer `l`; misses are read in parallel. Experts used by this call are never evicted by it.
    pub fn get_many(&mut self, _v: &Vault, l: usize, ids: &[usize], t: &mut Timing) -> Result<Vec<Blob>> {
        self.tick += 1;
        let stamp = self.tick;
        let keys: Vec<usize> = ids.iter().map(|&e| l * self.ne + e).collect();
        let miss: Vec<usize> = keys.iter().copied().filter(|k| !self.map.contains_key(k)).collect();
        self.hits += (keys.len() - miss.len()) as u64;
        self.misses += miss.len() as u64;
        for k in &keys {
            if let Some(x) = self.map.get_mut(k) {
                x.1 = stamp;
            }
        }
        let t0 = Instant::now();
        let f = &self.file;
        let ents = &self.entries;
        // misses in parallel on plain threads (blocking reads must not occupy the compute pool)
        let got: Vec<(usize, Vec<u8>)> = std::thread::scope(|sc| -> Result<Vec<(usize, Vec<u8>)>> {
            let mut out = Vec::with_capacity(miss.len());
            for grp in miss.chunks(8) {
                let hs: Vec<_> = grp.iter().map(|&k| sc.spawn(move || read_entry(f, &ents[k], 1).map(|b| (k, b)))).collect();
                for h in hs {
                    out.push(h.join().map_err(|_| anyhow::anyhow!("read thread panicked"))??);
                }
            }
            Ok(out)
        })?;
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        self.read_ms += ms;
        t.read_ms += ms;
        let mut fresh: HashMap<usize, Blob> = HashMap::new();
        for (k, b) in got {
            t.bytes_read += b.len() as u64;
            fresh.insert(k, Arc::new(b));
        }
        let out: Vec<Blob> = keys.iter().map(|k| self.map.get(k).map(|x| x.0.clone()).unwrap_or_else(|| fresh[k].clone())).collect();
        for (k, b) in fresh {
            if self.map.len() >= self.cap {
                let victim = self.map.iter().filter(|(_, v)| v.1 < stamp).min_by_key(|(_, v)| v.1).map(|(k, _)| *k);
                match victim {
                    Some(vk) => {
                        self.map.remove(&vk);
                    }
                    None => continue,
                }
            }
            self.map.insert(k, (b, stamp));
        }
        Ok(out)
    }
}
