//! MXFP4 expert matrices (gpt-oss): FP4 E2M1 codes, two per byte (low nibble first), one E8M0 scale per 32 weights.
//!
//! * `MXF4`: as released: `"MXF4", u32 rows, u32 cols, u8 scales[rows][cols/32], u8 blocks[rows][cols/2]`.
//! * `MXF2`: lossless repacking of the scales: per row `base + 2-bit offset` when the row's scale range is <= 3 (98.5% of rows
//!   in gpt-oss-120b), else the row's raw scales in an exception table. Codes unchanged.
//!   `"MXF2", rows, cols, pb, nexc, blocks_off | u32 desc[rows] | u8 packed[rows][pb] | u8 exc[nexc][nb] | pad 16 | blocks`.

use anyhow::{bail, ensure, Result};

pub const FP4: [f32; 16] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0];

pub enum Scales<'a> {
    Raw(&'a [u8]),
    Packed { pb: usize, desc: &'a [u8], packed: &'a [u8], exc: &'a [u8] },
}

pub struct Mxf<'a> {
    pub rows: usize,
    pub cols: usize,
    pub nb: usize,
    scales: Scales<'a>,
    blocks: &'a [u8],
}

fn u32at(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes(b[o..o + 4].try_into().unwrap())
}

impl<'a> Mxf<'a> {
    pub fn parse(blob: &'a [u8]) -> Result<Self> {
        ensure!(blob.len() >= 12, "MXF blob too short");
        let rows = u32at(blob, 4) as usize;
        let cols = u32at(blob, 8) as usize;
        ensure!(cols % 32 == 0, "MXF: cols {cols} not a multiple of 32");
        let nb = cols / 32;
        match &blob[0..4] {
            b"MXF4" => {
                let s0 = 12;
                let b0 = s0 + rows * nb;
                ensure!(blob.len() >= b0 + rows * nb * 16, "MXF4 blob truncated");
                Ok(Mxf { rows, cols, nb, scales: Scales::Raw(&blob[s0..b0]), blocks: &blob[b0..b0 + rows * nb * 16] })
            }
            b"MXF2" => {
                ensure!(blob.len() >= 24, "MXF2 blob too short");
                let pb = u32at(blob, 12) as usize;
                let nexc = u32at(blob, 16) as usize;
                let boff = u32at(blob, 20) as usize;
                let d0 = 24;
                let p0 = d0 + 4 * rows;
                let e0 = p0 + rows * pb;
                let e1 = e0 + nexc * nb;
                ensure!(pb * 4 >= nb && e1 <= boff && blob.len() >= boff + rows * nb * 16, "MXF2 blob inconsistent");
                Ok(Mxf {
                    rows,
                    cols,
                    nb,
                    scales: Scales::Packed { pb, desc: &blob[d0..p0], packed: &blob[p0..e0], exc: &blob[e0..e1] },
                    blocks: &blob[boff..boff + rows * nb * 16],
                })
            }
            m => bail!("not an MXF blob (magic {:?})", m),
        }
    }

    /// The E8M0 scale bytes of row `r` (unpacked for MXF2).
    pub fn row_scales(&self, r: usize, out: &mut [u8]) {
        match &self.scales {
            Scales::Raw(s) => out[..self.nb].copy_from_slice(&s[r * self.nb..(r + 1) * self.nb]),
            Scales::Packed { pb, desc, packed, exc } => {
                let d = u32at(desc, 4 * r);
                if d >> 31 != 0 {
                    let i = (d & 0x7fff_ffff) as usize;
                    out[..self.nb].copy_from_slice(&exc[i * self.nb..(i + 1) * self.nb]);
                } else {
                    let pk = &packed[r * pb..(r + 1) * pb];
                    for b in 0..self.nb {
                        out[b] = ((d & 255) + ((pk[b >> 2] as u32 >> ((b & 3) * 2)) & 3)) as u8;
                    }
                }
            }
        }
    }

    /// FP4 code bytes of row `r` (`nb * 16` bytes).
    pub fn row_blocks(&self, r: usize) -> &'a [u8] {
        &self.blocks[r * self.nb * 16..(r + 1) * self.nb * 16]
    }

    /// Row `r` as f32 (reference dequantisation: code value * 2^(scale - 127)).
    pub fn dequant_row(&self, r: usize, out: &mut [f32]) {
        let mut sc = vec![0u8; self.nb];
        self.row_scales(r, &mut sc);
        let bl = self.row_blocks(r);
        for b in 0..self.nb {
            let s = 2f32.powi(sc[b] as i32 - 127);
            for j in 0..16 {
                let q = bl[b * 16 + j];
                out[b * 32 + 2 * j] = FP4[(q & 15) as usize] * s;
                out[b * 32 + 2 * j + 1] = FP4[(q >> 4) as usize] * s;
            }
        }
    }

    /// All scales, row-major `[rows][nb]` (for repacking).
    pub fn all_scales(&self) -> Vec<u8> {
        let mut out = vec![0u8; self.rows * self.nb];
        for r in 0..self.rows {
            self.row_scales(r, &mut out[r * self.nb..(r + 1) * self.nb]);
        }
        out
    }
    pub fn blocks(&self) -> &'a [u8] {
        self.blocks
    }
}

/// Build an MXF4 blob from released tensors (`blocks [rows][nb][16]`, `scales [rows][nb]`).
pub fn mxf4_blob(blocks: &[u8], scales: &[u8], rows: usize, nb: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(12 + scales.len() + blocks.len());
    out.extend_from_slice(b"MXF4");
    out.extend_from_slice(&(rows as u32).to_le_bytes());
    out.extend_from_slice(&((nb * 32) as u32).to_le_bytes());
    out.extend_from_slice(scales);
    out.extend_from_slice(blocks);
    out
}

/// Build an MXF2 blob (byte-identical to `pack_gptoss.mxf2_blob`); verified before returning.
pub fn mxf2_blob(blocks: &[u8], scales: &[u8], rows: usize, nb: usize) -> Result<Vec<u8>> {
    ensure!(scales.len() == rows * nb && blocks.len() == rows * nb * 16);
    let pb = (nb.div_ceil(4) + 3) & !3;
    let mut desc = vec![0u32; rows];
    let mut packed = vec![0u8; rows * pb];
    let mut exc_rows: Vec<usize> = Vec::new();
    for r in 0..rows {
        let s = &scales[r * nb..(r + 1) * nb];
        let lo = *s.iter().min().unwrap_or(&0);
        let hi = *s.iter().max().unwrap_or(&0);
        if (hi as i32 - lo as i32) <= 3 {
            desc[r] = lo as u32;
            for b in 0..nb {
                packed[r * pb + (b >> 2)] |= ((s[b] - lo) & 3) << ((b & 3) * 2);
            }
        } else {
            desc[r] = (1 << 31) | exc_rows.len() as u32;
            exc_rows.push(r);
        }
    }
    let hdr = 24usize;
    let body = 4 * rows + rows * pb + exc_rows.len() * nb;
    let boff = (hdr + body + 15) & !15;
    let mut out = Vec::with_capacity(boff + blocks.len());
    out.extend_from_slice(b"MXF2");
    for v in [rows as u32, (nb * 32) as u32, pb as u32, exc_rows.len() as u32, boff as u32] {
        out.extend_from_slice(&v.to_le_bytes());
    }
    for d in &desc {
        out.extend_from_slice(&d.to_le_bytes());
    }
    out.extend_from_slice(&packed);
    for &r in &exc_rows {
        out.extend_from_slice(&scales[r * nb..(r + 1) * nb]);
    }
    out.resize(boff, 0);
    out.extend_from_slice(blocks);
    let m = Mxf::parse(&out)?;
    ensure!(m.all_scales() == scales && m.blocks() == blocks, "MXF2 packing not lossless");
    Ok(out)
}

/// `"F32 ", u32 n, f32[n]` (expert biases).
pub fn f32_view(blob: &[u8]) -> Result<Vec<f32>> {
    ensure!(blob.len() >= 8 && &blob[0..4] == b"F32 ", "not an F32 blob");
    let n = u32at(blob, 4) as usize;
    ensure!(blob.len() >= 8 + 4 * n, "F32 blob truncated");
    Ok(blob[8..8 + 4 * n].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())
}

pub fn f32_blob(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(8 + 4 * v.len());
    out.extend_from_slice(b"F32 ");
    out.extend_from_slice(&(v.len() as u32).to_le_bytes());
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn mxf2_roundtrip_with_exceptions() {
        let (rows, nb) = (9, 5);
        let mut sc: Vec<u8> = (0..rows * nb).map(|i| 118 + (i % 4) as u8).collect();
        sc[3 * nb + 2] = 100; // row 3 becomes an exception
        let bl: Vec<u8> = (0..rows * nb * 16).map(|i| (i * 37 % 256) as u8).collect();
        let b2 = mxf2_blob(&bl, &sc, rows, nb).unwrap();
        let b4 = mxf4_blob(&bl, &sc, rows, nb);
        let (m2, m4) = (Mxf::parse(&b2).unwrap(), Mxf::parse(&b4).unwrap());
        let (mut x, mut y) = (vec![0f32; nb * 32], vec![0f32; nb * 32]);
        for r in 0..rows {
            m2.dequant_row(r, &mut x);
            m4.dequant_row(r, &mut y);
            assert_eq!(x, y);
        }
    }
}
