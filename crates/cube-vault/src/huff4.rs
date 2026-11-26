//! huffman4-v3: lossless bf16 matrices. The exponent byte of every weight is Huffman-coded (code length <= 12, one table per
//! matrix) in 4 interleaved column streams per row; the sign + mantissa byte is stored raw (it is incompressible).
//!
//! Blob: `u32 rows, u32 cols, u8 len[256], u32 off[4*rows+1], exponent bit-streams, 8 zero bytes, u8 sm[rows*cols]`.
//! The encoder mirrors `pack_universal_v3.py` (and `nvme_engine.c`) exactly, so a blob made here is byte-identical to one
//! made by the Python packer.

use anyhow::{bail, ensure, Result};

pub const LMAX: u32 = 12;
pub const LUT_SIZE: usize = 1 << LMAX;
const HDR: usize = 264;

/// Huffman code lengths for 256 symbol frequencies, limited to [`LMAX`] bits by halving the frequencies until they fit.
/// Tie-breaking follows the Python packer node for node, so the lengths (and the blob) are identical.
pub fn huff_lengths(freq: &[u64; 256]) -> [u8; 256] {
    let mut f = *freq;
    loop {
        let mut ln = [0u8; 256];
        let idx: Vec<usize> = (0..256).filter(|&i| f[i] != 0).collect();
        let n = idx.len();
        if n == 0 {
            return ln;
        }
        if n == 1 {
            ln[idx[0]] = 1;
            return ln;
        }
        let mut w = vec![0u64; 2 * n + 2];
        let mut par = vec![-1i64; 2 * n + 2];
        let mut al: Vec<usize> = (0..n).collect();
        let mut na = n;
        let mut m = n;
        for k in 0..n {
            w[k] = f[idx[k]];
        }
        while na > 1 {
            let mut a = 0usize;
            for k in 1..na {
                if w[al[k]] < w[al[a]] {
                    a = k;
                }
            }
            let mut b: isize = -1;
            for k in 0..na {
                if k == a {
                    continue;
                }
                if b < 0 || w[al[k]] < w[al[b as usize]] {
                    b = k as isize;
                }
            }
            let b = b as usize;
            w[m] = w[al[a]] + w[al[b]];
            par[m] = -1;
            par[al[a]] = m as i64;
            par[al[b]] = m as i64;
            al[a] = m;
            al[b] = al[na - 1];
            na -= 1;
            m += 1;
        }
        let mut mx = 0u32;
        for k in 0..n {
            let mut d = 0u32;
            let mut p = par[k];
            while p >= 0 {
                d += 1;
                p = par[p as usize];
            }
            ln[idx[k]] = d as u8;
            mx = mx.max(d);
        }
        if mx <= LMAX {
            return ln;
        }
        for x in f.iter_mut() {
            if *x != 0 {
                *x = (*x + 1) >> 1;
            }
        }
    }
}

/// Canonical codes, bit-reversed (the streams are read least-significant bit first).
pub fn make_rcodes(ln: &[u8; 256]) -> [u32; 256] {
    let mut cnt = [0u32; LMAX as usize + 2];
    let mut nxt = [0u32; LMAX as usize + 2];
    for &l in ln.iter() {
        if l != 0 {
            cnt[l as usize] += 1;
        }
    }
    let mut c = 0u32;
    for l in 1..=LMAX as usize {
        c = (c + cnt[l - 1]) << 1;
        nxt[l] = c;
    }
    let mut rc = [0u32; 256];
    for s in 0..256 {
        let l = ln[s] as u32;
        if l == 0 {
            continue;
        }
        let code = nxt[l as usize];
        nxt[l as usize] += 1;
        let mut r = 0u32;
        for i in 0..l {
            r |= ((code >> i) & 1) << (l - 1 - i);
        }
        rc[s] = r;
    }
    rc
}

/// Decode table: 12-bit window -> `symbol | length << 8` (0 = no code).
pub fn build_lut(ln: &[u8; 256]) -> Vec<u16> {
    let rc = make_rcodes(ln);
    let mut lut = vec![0u16; LUT_SIZE];
    for s in 0..256 {
        let l = ln[s] as u32;
        if l == 0 {
            continue;
        }
        for ks in 0..(1u32 << (LMAX - l)) {
            lut[(rc[s] | (ks << l)) as usize] = (s as u16) | ((l as u16) << 8);
        }
    }
    lut
}

#[inline]
pub fn exponent(w: u16) -> u8 {
    ((w >> 7) & 255) as u8
}
#[inline]
pub fn sign_mantissa(w: u16) -> u8 {
    (((w >> 8) & 0x80) | (w & 0x7F)) as u8
}
#[inline]
pub fn join(e: u8, sm: u8) -> u16 {
    (((sm as u16) & 0x80) << 8) | ((e as u16) << 7) | ((sm as u16) & 0x7F)
}
#[inline]
pub fn join_f32(e: u8, sm: u8) -> f32 {
    f32::from_bits((((sm as u32) & 0x80) << 24) | ((e as u32) << 23) | (((sm as u32) & 0x7F) << 16))
}

/// Encode a row-major bf16 matrix (raw `u16` bits). Verifies the result decodes bit-exactly before returning it.
pub fn compress(w: &[u16], rows: usize, cols: usize) -> Result<Vec<u8>> {
    ensure!(w.len() == rows * cols, "matrix size {} != {rows} x {cols}", w.len());
    let mut freq = [0u64; 256];
    for &x in w {
        freq[exponent(x) as usize] += 1;
    }
    let ln = huff_lengths(&freq);
    let rc = make_rcodes(&ln);
    let no4 = 4 * rows + 1;
    let mut off = vec![0u64; no4];
    let mut pos = 0u64;
    for r in 0..rows {
        let row = &w[r * cols..(r + 1) * cols];
        for k in 0..4 {
            off[4 * r + k] = pos;
            let mut bits = 0u64;
            let mut j = k;
            while j < cols {
                bits += ln[exponent(row[j]) as usize] as u64;
                j += 4;
            }
            pos += (bits + 7) >> 3;
        }
    }
    off[4 * rows] = pos;
    if pos > 0xFFFF_FFF0 {
        bail!("matrix too large for 32-bit stream offsets");
    }
    let p_off = HDR;
    let p_es = p_off + 4 * no4;
    let p_sm = p_es + pos as usize + 8;
    let n = rows * cols;
    let mut blob = vec![0u8; p_sm + n];
    blob[0..4].copy_from_slice(&(rows as u32).to_le_bytes());
    blob[4..8].copy_from_slice(&(cols as u32).to_le_bytes());
    blob[8..HDR].copy_from_slice(&ln);
    for (i, &o) in off.iter().enumerate() {
        blob[p_off + 4 * i..p_off + 4 * i + 4].copy_from_slice(&(o as u32).to_le_bytes());
    }
    {
        let (head, sm_part) = blob.split_at_mut(p_sm);
        let es = &mut head[p_es..p_es + pos as usize];
        for r in 0..rows {
            let row = &w[r * cols..(r + 1) * cols];
            for k in 0..4 {
                let mut p = off[4 * r + k] as usize;
                let mut acc: u64 = 0;
                let mut nb = 0u32;
                let mut j = k;
                while j < cols {
                    let e = exponent(row[j]) as usize;
                    acc |= (rc[e] as u64) << nb;
                    nb += ln[e] as u32;
                    while nb >= 8 {
                        es[p] = (acc & 255) as u8;
                        p += 1;
                        acc >>= 8;
                        nb -= 8;
                    }
                    j += 4;
                }
                if nb > 0 {
                    es[p] = (acc & 255) as u8;
                }
            }
            for j in 0..cols {
                sm_part[r * cols + j] = sign_mantissa(row[j]);
            }
        }
    }
    let v = Huff4::parse(&blob)?;
    let mut dec = vec![0u16; cols];
    for r in 0..rows {
        v.decode_row_bits(r, &mut dec);
        ensure!(dec[..] == w[r * cols..(r + 1) * cols], "huffman4 verify failed at row {r}");
    }
    Ok(blob)
}

/// A parsed huffman4 blob (borrowed).
pub struct Huff4<'a> {
    pub rows: usize,
    pub cols: usize,
    pub len: [u8; 256],
    off: &'a [u8],
    es: &'a [u8],
    sm: &'a [u8],
    lut: Vec<u16>,
}

impl<'a> Huff4<'a> {
    pub fn parse(blob: &'a [u8]) -> Result<Self> {
        ensure!(blob.len() >= HDR, "huffman4 blob too short");
        let rows = u32::from_le_bytes(blob[0..4].try_into()?) as usize;
        let cols = u32::from_le_bytes(blob[4..8].try_into()?) as usize;
        let mut len = [0u8; 256];
        len.copy_from_slice(&blob[8..HDR]);
        ensure!(len.iter().all(|&l| l as u32 <= LMAX), "huffman4: code length > 12");
        let no4 = 4 * rows + 1;
        let p_es = HDR + 4 * no4;
        ensure!(blob.len() >= p_es, "huffman4 blob truncated (row table)");
        let off = &blob[HDR..p_es];
        let pos = u32::from_le_bytes(off[4 * (no4 - 1)..4 * no4].try_into()?) as usize;
        let p_sm = p_es + pos + 8;
        ensure!(blob.len() >= p_sm + rows * cols, "huffman4 blob truncated (data)");
        let lut = build_lut(&len);
        Ok(Huff4 { rows, cols, len, off, es: &blob[p_es..p_es + pos + 8], sm: &blob[p_sm..p_sm + rows * cols], lut })
    }

    #[inline]
    pub fn row_offsets(&self, r: usize) -> [u32; 4] {
        let mut o = [0u32; 4];
        for k in 0..4 {
            let i = 4 * (4 * r + k);
            o[k] = u32::from_le_bytes(self.off[i..i + 4].try_into().unwrap());
        }
        o
    }
    pub fn lut(&self) -> &[u16] {
        &self.lut
    }
    pub fn es(&self) -> &'a [u8] {
        self.es
    }
    pub fn sm_row(&self, r: usize) -> &'a [u8] {
        &self.sm[r * self.cols..(r + 1) * self.cols]
    }

    /// Decode the exponent bytes of row `r` (reference path: one bit at a time from each of the 4 streams).
    pub fn decode_row_exponents(&self, r: usize, e_out: &mut [u8]) {
        let cols = self.cols;
        let o = self.row_offsets(r);
        let mut pos = [o[0] as usize, o[1] as usize, o[2] as usize, o[3] as usize];
        let mut acc = [0u64; 4];
        let mut nb = [0u32; 4];
        for j in 0..cols {
            let k = j & 3;
            while nb[k] < LMAX {
                let b = self.es.get(pos[k]).copied().unwrap_or(0);
                acc[k] |= (b as u64) << nb[k];
                pos[k] += 1;
                nb[k] += 8;
            }
            let ent = self.lut[(acc[k] & (LUT_SIZE as u64 - 1)) as usize];
            let l = (ent >> 8) as u32;
            acc[k] >>= l;
            nb[k] -= l;
            e_out[j] = (ent & 255) as u8;
        }
    }

    /// Decode row `r` to raw bf16 bits.
    pub fn decode_row_bits(&self, r: usize, out: &mut [u16]) {
        let mut e = vec![0u8; self.cols];
        self.decode_row_exponents(r, &mut e);
        let sm = self.sm_row(r);
        for j in 0..self.cols {
            out[j] = join(e[j], sm[j]);
        }
    }

    /// Decode the whole matrix to raw bf16 bits.
    pub fn decode_bits(&self) -> Vec<u16> {
        let mut out = vec![0u16; self.rows * self.cols];
        for r in 0..self.rows {
            self.decode_row_bits(r, &mut out[r * self.cols..(r + 1) * self.cols]);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rng(s: &mut u64) -> u64 {
        *s ^= *s >> 12;
        *s ^= *s << 25;
        *s ^= *s >> 27;
        s.wrapping_mul(2685821657736338717)
    }

    #[test]
    fn roundtrip_shapes_and_patterns() {
        let mut s = 88172645463325252u64;
        for &(r, c) in &[(1, 1), (7, 13), (5, 4097), (64, 64), (9, 2563), (16, 4096), (3, 5)] {
            for mode in 0..4 {
                let w: Vec<u16> = (0..r * c)
                    .map(|_| {
                        let x = rng(&mut s);
                        let e: u16 = match mode {
                            0 => 127,
                            1 => (x & 255) as u16,
                            2 => ((x | 0x8000_0000).trailing_zeros().min(255)) as u16,
                            _ => 120 + (x & 7) as u16,
                        };
                        let sm = ((x >> 20) & 0xFF) as u16;
                        ((sm & 0x80) << 8) | (e << 7) | (sm & 0x7F)
                    })
                    .collect();
                let b = compress(&w, r, c).unwrap();
                assert_eq!(Huff4::parse(&b).unwrap().decode_bits(), w, "{r}x{c} mode {mode}");
            }
        }
    }
}
