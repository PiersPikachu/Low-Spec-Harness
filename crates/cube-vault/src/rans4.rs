//! rans4 ("cube-rANS"): bf16 matrices with the exponent byte rANS-coded (static per-matrix table, precision 2^mbits),
//! R rows per chain, 4 interleaved column chains per row block; sign + mantissa stored raw.
//!
//! Blob: `u32 rows, u32 cols, u8 mbits, u8 R, u16 0, u16 freq[256], u32 off[4*nblk+1]`, chain bytes, 8 pad, `u8 sm[rows*cols]`.
//! Each chain starts with its 4-byte final state, then renormalisation bytes in decode order. Mirrors `tools/rans4.py`.

use crate::huff4::{exponent, join, sign_mantissa};
use anyhow::{ensure, Result};

const RANS_L: u64 = 1 << 23;

pub struct Rans4<'a> {
    pub rows: usize,
    pub cols: usize,
    pub mbits: u32,
    pub r: usize,
    pub freq: [u32; 256],
    cum: [u32; 256],
    s2s: Vec<u8>,
    off: &'a [u8],
    cs: &'a [u8],
    sm: &'a [u8],
}

impl<'a> Rans4<'a> {
    pub fn parse(blob: &'a [u8]) -> Result<Self> {
        ensure!(blob.len() >= 524, "rans4 blob too short");
        let rows = u32::from_le_bytes(blob[0..4].try_into()?) as usize;
        let cols = u32::from_le_bytes(blob[4..8].try_into()?) as usize;
        let mbits = blob[8] as u32;
        let r = blob[9] as usize;
        ensure!(r >= 1 && (1..=16).contains(&mbits), "rans4: bad parameters");
        let mut freq = [0u32; 256];
        for s in 0..256 {
            freq[s] = u16::from_le_bytes([blob[12 + 2 * s], blob[13 + 2 * s]]) as u32;
        }
        let total: u32 = freq.iter().sum();
        ensure!(total == 1 << mbits, "rans4: frequencies sum to {total}, not 2^{mbits}");
        let mut cum = [0u32; 256];
        let mut c = 0u32;
        for s in 0..256 {
            cum[s] = c;
            c += freq[s];
        }
        let mut s2s = vec![0u8; 1 << mbits];
        for s in 0..256 {
            for k in 0..freq[s] {
                s2s[(cum[s] + k) as usize] = s as u8;
            }
        }
        let nblk = rows.div_ceil(r);
        let o0 = 524;
        let c0 = o0 + 4 * (4 * nblk + 1);
        ensure!(blob.len() >= c0, "rans4 blob truncated (chain table)");
        let off = &blob[o0..c0];
        let n = u32::from_le_bytes(off[4 * 4 * nblk..4 * 4 * nblk + 4].try_into()?) as usize;
        let p_sm = c0 + n + 8;
        ensure!(blob.len() >= p_sm + rows * cols, "rans4 blob truncated (data)");
        Ok(Rans4 { rows, cols, mbits, r, freq, cum, s2s, off, cs: &blob[c0..c0 + n + 8], sm: &blob[p_sm..p_sm + rows * cols] })
    }

    fn chain_off(&self, i: usize) -> usize {
        u32::from_le_bytes(self.off[4 * i..4 * i + 4].try_into().unwrap()) as usize
    }

    /// Decode the exponent bytes of row block `b` (rows `b*R .. min(rows, (b+1)*R)`), row-major, into `e`.
    pub fn decode_block_exponents(&self, b: usize, e: &mut [u8]) {
        let (cols, m1) = (self.cols, (1u64 << self.mbits) - 1);
        let r0 = b * self.r;
        let nr = (self.rows - r0).min(self.r);
        for k in 0..4 {
            let mut p = self.chain_off(4 * b + k);
            let c = self.cs;
            let mut x = c[p] as u64 | (c[p + 1] as u64) << 8 | (c[p + 2] as u64) << 16 | (c[p + 3] as u64) << 24;
            p += 4;
            for i in 0..nr {
                let mut j = k;
                while j < cols {
                    let slot = x & m1;
                    let s = self.s2s[slot as usize] as usize;
                    x = self.freq[s] as u64 * (x >> self.mbits) + slot - self.cum[s] as u64;
                    while x < RANS_L {
                        x = (x << 8) | c[p] as u64;
                        p += 1;
                    }
                    e[i * cols + j] = s as u8;
                    j += 4;
                }
            }
        }
    }

    pub fn sm_row(&self, r: usize) -> &'a [u8] {
        &self.sm[r * self.cols..(r + 1) * self.cols]
    }
    /// all chain bytes (plus the 8 pad bytes)
    pub fn chains(&self) -> &'a [u8] {
        self.cs
    }
    /// start of the 4 chains of row block `b` inside [`chains`](Self::chains)
    pub fn block_offsets(&self, b: usize) -> [usize; 4] {
        [self.chain_off(4 * b), self.chain_off(4 * b + 1), self.chain_off(4 * b + 2), self.chain_off(4 * b + 3)]
    }

    pub fn decode_bits(&self) -> Vec<u16> {
        let mut out = vec![0u16; self.rows * self.cols];
        let mut e = vec![0u8; self.r * self.cols];
        for b in 0..self.rows.div_ceil(self.r) {
            self.decode_block_exponents(b, &mut e);
            let r0 = b * self.r;
            for i in 0..(self.rows - r0).min(self.r) {
                let sm = self.sm_row(r0 + i);
                for j in 0..self.cols {
                    out[(r0 + i) * self.cols + j] = join(e[i * self.cols + j], sm[j]);
                }
            }
        }
        out
    }
}

/// Frequencies summing to exactly 2^mbits, every present symbol >= 1 (mirrors `rans4.quantize`, float64 arithmetic included).
pub fn quantize(cnt: &[u64; 256], mbits: u32) -> [u32; 256] {
    let m = 1u64 << mbits;
    let n: u64 = cnt.iter().sum();
    let mut f = [0i64; 256];
    let nz: Vec<usize> = (0..256).filter(|&i| cnt[i] > 0).collect();
    if nz.len() == 1 {
        f[nz[0]] = m as i64;
    } else {
        for &i in &nz {
            f[i] = ((cnt[i] as f64 * m as f64 / n as f64 + 0.5).floor() as i64).max(1);
        }
        loop {
            let sum: i64 = f.iter().sum();
            if sum == m as i64 {
                break;
            }
            if (m as i64) - sum > 0 {
                let mut best = nz[0];
                for &i in &nz {
                    if cnt[i] as f64 / f[i] as f64 > cnt[best] as f64 / f[best] as f64 {
                        best = i;
                    }
                }
                f[best] += 1;
            } else {
                let cand: Vec<usize> = nz.iter().copied().filter(|&i| f[i] > 1).collect();
                let mut best = cand[0];
                for &i in &cand {
                    if (cnt[i] as f64 / f[i] as f64) < cnt[best] as f64 / f[best] as f64 {
                        best = i;
                    }
                }
                f[best] -= 1;
            }
        }
    }
    let mut out = [0u32; 256];
    for i in 0..256 {
        out[i] = f[i] as u32;
    }
    out
}

/// Encode a bf16 matrix (mirrors `rans4.encode`). Verified bit-exact before returning.
pub fn encode(w: &[u16], rows: usize, cols: usize, mbits: u32, r: usize) -> Result<Vec<u8>> {
    ensure!(w.len() == rows * cols && r >= 1 && r < 256);
    let mut cnt = [0u64; 256];
    for &x in w {
        cnt[exponent(x) as usize] += 1;
    }
    let freq = quantize(&cnt, mbits);
    let mut cum = [0u64; 256];
    let mut c = 0u64;
    for s in 0..256 {
        cum[s] = c;
        c += freq[s] as u64;
    }
    let nblk = rows.div_ceil(r);
    let mut off = vec![0u32; 4 * nblk + 1];
    let mut chains: Vec<u8> = Vec::with_capacity(rows * cols);
    let mut tmp: Vec<u8> = Vec::new();
    for b in 0..nblk {
        let (r0, r1) = (b * r, ((b + 1) * r).min(rows));
        for k in 0..4 {
            tmp.clear();
            let mut x = RANS_L;
            for row in (r0..r1).rev() {
                if k >= cols {
                    continue;
                }
                let jl = k + ((cols - 1 - k) / 4) * 4;
                let mut j = jl as isize;
                while j >= k as isize {
                    let s = exponent(w[row * cols + j as usize]) as usize;
                    let f = freq[s] as u64;
                    let xmax = ((RANS_L >> mbits) << 8) * f;
                    while x >= xmax {
                        tmp.push((x & 255) as u8);
                        x >>= 8;
                    }
                    x = ((x / f) << mbits) + (x % f) + cum[s];
                    j -= 4;
                }
            }
            for bb in 0..4 {
                tmp.push(((x >> (8 * (3 - bb))) & 255) as u8);
            }
            off[4 * b + k] = chains.len() as u32;
            chains.extend(tmp.iter().rev());
        }
    }
    off[4 * nblk] = chains.len() as u32;
    let mut blob = Vec::with_capacity(524 + off.len() * 4 + chains.len() + 8 + rows * cols);
    blob.extend_from_slice(&(rows as u32).to_le_bytes());
    blob.extend_from_slice(&(cols as u32).to_le_bytes());
    blob.push(mbits as u8);
    blob.push(r as u8);
    blob.extend_from_slice(&0u16.to_le_bytes());
    for s in 0..256 {
        blob.extend_from_slice(&(freq[s] as u16).to_le_bytes());
    }
    for o in &off {
        blob.extend_from_slice(&o.to_le_bytes());
    }
    blob.extend_from_slice(&chains);
    blob.extend_from_slice(&[0u8; 8]);
    blob.extend(w.iter().map(|&x| sign_mantissa(x)));
    ensure!(Rans4::parse(&blob)?.decode_bits() == w, "rans4 verify failed");
    Ok(blob)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn roundtrip() {
        let mut s = 1u64;
        for &(rows, cols) in &[(37, 101), (5, 3), (1, 1), (64, 512), (9, 2050)] {
            for &(m, r) in &[(12, 1), (12, 16), (14, 3)] {
                let w: Vec<u16> = (0..rows * cols)
                    .map(|_| {
                        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                        let e = 118 + ((s >> 33) % 12) as u16;
                        ((s >> 50) as u16 & 0x80) << 8 | e << 7 | (s >> 40) as u16 & 0x7F
                    })
                    .collect();
                let b = encode(&w, rows, cols, m, r).unwrap();
                assert_eq!(Rans4::parse(&b).unwrap().decode_bits(), w);
            }
        }
    }
}
