//! Numeric kernels. Every reduction uses the same operation order as `nvme_engine_u.c` built with `-march=native
//! -ffp-contract=off` (its AVX2 paths), so results are bit-identical to the C engine wherever both use the same libm:
//!
//! * bf16 matrix x vector: 8 lane accumulators, multiply then add (no FMA), tail into lane 0,
//!   `((a0+a1)+(a2+a3))+((a4+a5)+(a6+a7))`
//! * MXFP4 matrix x vector: per 32-block `d = w0*x0; d = fma(w1,x1,d); ...` over 8 lanes, `acc = fma(scale, d, acc)`,
//!   horizontal sum `((l0+l4)+(l1+l5))+((l2+l6)+(l3+l7))`
//! * RMSNorm sums squares in f64

use cube_vault::huff4::{Huff4, LMAX};
use cube_vault::mxf::Mxf;
use cube_vault::rans4::Rans4;
use rayon::prelude::*;

/// 8-lane dot product in the C engine's order.
#[inline]
pub fn dot8(w: &[f32], x: &[f32]) -> f32 {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
    return crate::simd::dot8(w, x);
    #[allow(unreachable_code)]
    dot8_scalar(w, x)
}

pub fn dot8_scalar(w: &[f32], x: &[f32]) -> f32 {
    let n = w.len();
    let x = &x[..n];
    let mut a = [0f32; 8];
    let (wc, xc) = (w.chunks_exact(8), x.chunks_exact(8));
    let (wr, xr) = (wc.remainder(), xc.remainder());
    for (wv, xv) in wc.zip(xc) {
        for l in 0..8 {
            a[l] += wv[l] * xv[l];
        }
    }
    for j in 0..wr.len() {
        a[0] += wr[j] * xr[j];
    }
    ((a[0] + a[1]) + (a[2] + a[3])) + ((a[4] + a[5]) + (a[6] + a[7]))
}

#[inline]
#[allow(dead_code)]
fn bf16_to_f32_parts(e: u8, sm: u8) -> f32 {
    f32::from_bits((((sm as u32) & 0x80) << 24) | ((e as u32) << 23) | (((sm as u32) & 0x7F) << 16))
}

/// Multi-symbol table: 12-bit window -> up to 3 exponent symbols. entry = syms(24) | bits_used(4)<<24 | count(2)<<28
pub fn build_lut3(lut: &[u16]) -> Vec<u32> {
    let mask = (1u32 << LMAX) - 1;
    (0..1u32 << LMAX)
        .map(|v| {
            let (mut bits, mut used, mut cnt, mut syms) = (v, 0u32, 0u32, 0u32);
            while cnt < 3 {
                let e = lut[(bits & mask) as usize] as u32;
                let l = e >> 8;
                if l == 0 || used + l > LMAX {
                    break;
                }
                syms |= (e & 255) << (8 * cnt);
                cnt += 1;
                used += l;
                bits >>= l;
            }
            if cnt == 0 {
                syms = 0;
                used = 1;
                cnt = 1;
            }
            syms | (used << 24) | (cnt << 28)
        })
        .collect()
}

#[inline(always)]
fn rd32(b: &[u8], p: usize) -> u32 {
    match b.get(p..p + 4) {
        Some(s) => u32::from_le_bytes([s[0], s[1], s[2], s[3]]),
        None => {
            let mut t = [0u8; 4];
            for i in 0..4 {
                t[i] = b.get(p + i).copied().unwrap_or(0);
            }
            u32::from_le_bytes(t)
        }
    }
}

/// Decode the 4 interleaved exponent streams of one huffman4 row into `eb` (4 arrays of `cap` bytes; stream k holds
/// columns j % 4 == k), two multi-symbol lookups per stream per refill — the C engine's `dec4`.
#[inline]
fn dec4(l3: &[u32], es: &[u8], o4: [u32; 4], cols: usize, eb: &mut [u8], cap: usize) {
    assert!(l3.len() == 1 << LMAX && eb.len() >= 4 * cap && cap >= (cols >> 2) + 8);
    // every stream pointer advances at most 4 bytes per 2 symbols: with this margin no read leaves `es`
    let lim = cols / 2 + 16;
    if o4.iter().any(|&o| o as usize + lim > es.len()) {
        return dec4_checked(l3, es, o4, cols, eb, cap);
    }
    let mask = (1u64 << LMAX) - 1;
    let nn = [(cols + 3) >> 2, (cols + 2) >> 2, (cols + 1) >> 2, cols >> 2];
    unsafe {
        let pe = es.as_ptr();
        let po = eb.as_mut_ptr();
        let pl = l3.as_ptr();
        let mut p = [pe.add(o4[0] as usize), pe.add(o4[1] as usize), pe.add(o4[2] as usize), pe.add(o4[3] as usize)];
        let mut b = [0u64; 4];
        let mut c = [0u32; 4];
        let mut q = [0usize; 4];
        macro_rules! rf {
            ($k:expr) => {
                if c[$k] < 24 {
                    b[$k] |= ((p[$k] as *const u32).read_unaligned().to_le() as u64) << c[$k];
                    p[$k] = p[$k].add(4);
                    c[$k] += 32;
                }
            };
        }
        macro_rules! lk {
            ($k:expr) => {{
                let t = *pl.add((b[$k] & mask) as usize);
                (po.add($k * cap + q[$k]) as *mut u32).write_unaligned(t.to_le());
                let u = (t >> 24) & 15;
                b[$k] >>= u;
                c[$k] -= u;
                q[$k] += ((t >> 28) & 3) as usize;
            }};
        }
        while q[0] < nn[0] && q[1] < nn[1] && q[2] < nn[2] && q[3] < nn[3] {
            rf!(0);
            rf!(1);
            rf!(2);
            rf!(3);
            lk!(0);
            lk!(1);
            lk!(2);
            lk!(3);
            lk!(0);
            lk!(1);
            lk!(2);
            lk!(3);
        }
        // constant indices only: a variable index would keep b/c/p/q in memory for the whole loop
        while q[0] < nn[0] {
            rf!(0);
            lk!(0);
        }
        while q[1] < nn[1] {
            rf!(1);
            lk!(1);
        }
        while q[2] < nn[2] {
            rf!(2);
            lk!(2);
        }
        while q[3] < nn[3] {
            rf!(3);
            lk!(3);
        }
    }
}

/// bounds-checked `dec4` (malformed offsets cannot read outside the blob)
#[inline(never)]
fn dec4_checked(l3: &[u32], es: &[u8], o4: [u32; 4], cols: usize, eb: &mut [u8], cap: usize) {
    let mask = (1u64 << LMAX) - 1;
    let nn = [(cols + 3) >> 2, (cols + 2) >> 2, (cols + 1) >> 2, cols >> 2];
    let mut p = [o4[0] as usize, o4[1] as usize, o4[2] as usize, o4[3] as usize];
    let mut b = [0u64; 4];
    let mut c = [0u32; 4];
    let mut q = [0usize; 4];
    macro_rules! rf {
        ($k:expr) => {
            if c[$k] < 24 {
                b[$k] |= (rd32(es, p[$k]) as u64) << c[$k];
                p[$k] += 4;
                c[$k] += 32;
            }
        };
    }
    macro_rules! lk {
        ($k:expr) => {{
            let t = l3[(b[$k] & mask) as usize];
            let o = $k * cap + q[$k];
            eb[o..o + 4].copy_from_slice(&t.to_le_bytes());
            let u = (t >> 24) & 15;
            b[$k] >>= u;
            c[$k] -= u;
            q[$k] += ((t >> 28) & 3) as usize;
        }};
    }
    while q[0] < nn[0] && q[1] < nn[1] && q[2] < nn[2] && q[3] < nn[3] {
        rf!(0);
        rf!(1);
        rf!(2);
        rf!(3);
        lk!(0);
        lk!(1);
        lk!(2);
        lk!(3);
        lk!(0);
        lk!(1);
        lk!(2);
        lk!(3);
    }
    for k in 0..4 {
        while q[k] < nn[k] {
            rf!(k);
            lk!(k);
        }
    }
}

/// bf16 weights of one huffman4 row from the 4 exponent arrays and the sign+mantissa bytes
#[inline]
#[allow(dead_code)]
fn assemble_huff(eb: &[u8], cap: usize, sm: &[u8], w: &mut [f32]) {
    let cols = w.len();
    let (e0, rest) = eb.split_at(cap);
    let (e1, rest) = rest.split_at(cap);
    let (e2, e3) = rest.split_at(cap);
    let full = cols / 4;
    for i in 0..full {
        let j = 4 * i;
        w[j] = bf16_to_f32_parts(e0[i], sm[j]);
        w[j + 1] = bf16_to_f32_parts(e1[i], sm[j + 1]);
        w[j + 2] = bf16_to_f32_parts(e2[i], sm[j + 2]);
        w[j + 3] = bf16_to_f32_parts(e3[i], sm[j + 3]);
    }
    for j in 4 * full..cols {
        let e = [e0, e1, e2, e3][j & 3][j >> 2];
        w[j] = bf16_to_f32_parts(e, sm[j]);
    }
}

/// rANS table: slot -> sym | (freq-1)<<8 | (slot-cum)<<20 (M = 2^12)
pub fn rans_tab(freq: &[u32; 256]) -> Vec<u32> {
    let mut t = vec![0u32; 4096];
    let mut cum = 0u32;
    for s in 0..256 {
        let f = freq[s];
        for k in 0..f {
            t[(cum + k) as usize] = s as u32 | ((f.wrapping_sub(1) & 4095) << 8) | (k << 20);
        }
        cum += f;
    }
    t
}

/// Decode `nr` rows of one rans4 row block, the 4 chains interleaved (the C engine's `rans_rows`).
#[inline]
fn rans_rows(tab: &[u32], cs: &[u8], o4: [usize; 4], cols: usize, nr: usize, e: &mut [u8]) {
    // each step renormalises by at most 2 bytes: with this margin no read leaves `cs`
    let lim = 4 + nr * cols.div_ceil(4) * 2 + 8;
    if tab.len() == 4096 && e.len() >= nr * cols && o4.iter().all(|&o| o + lim <= cs.len()) {
        rans_rows_g::<false>(tab, cs, o4, cols, nr, e)
    } else {
        rans_rows_g::<true>(tab, cs, o4, cols, nr, e)
    }
}

#[inline(always)]
fn rans_rows_g<const CHECKED: bool>(tab: &[u32], cs: &[u8], o4: [usize; 4], cols: usize, nr: usize, e: &mut [u8]) {
    let mut x = [rd32(cs, o4[0]), rd32(cs, o4[1]), rd32(cs, o4[2]), rd32(cs, o4[3])];
    let mut p = [o4[0] + 4, o4[1] + 4, o4[2] + 4, o4[3] + 4];
    macro_rules! step {
        ($k:expr, $dst:expr) => {{
            let t = if CHECKED { tab[(x[$k] & 4095) as usize] } else { unsafe { *tab.get_unchecked((x[$k] & 4095) as usize) } };
            $dst = t as u8;
            x[$k] = (((t >> 8) & 4095) + 1).wrapping_mul(x[$k] >> 12).wrapping_add(t >> 20);
            while x[$k] < (1 << 23) {
                let byte = if CHECKED { cs.get(p[$k]).copied().unwrap_or(0) } else { unsafe { *cs.get_unchecked(p[$k]) } };
                x[$k] = (x[$k] << 8) | byte as u32;
                p[$k] += 1;
                if CHECKED && p[$k] > cs.len() + 8 {
                    break;
                }
            }
        }};
    }
    for r in 0..nr {
        let er = &mut e[r * cols..(r + 1) * cols];
        let mut j = 0;
        while j + 4 <= cols {
            step!(0, er[j]);
            step!(1, er[j + 1]);
            step!(2, er[j + 2]);
            step!(3, er[j + 3]);
            j += 4;
        }
        // tail (cols % 4 columns): constant chain indices keep the states in registers
        let tail = cols - j;
        if tail > 0 {
            step!(0, er[j]);
        }
        if tail > 1 {
            step!(1, er[j + 1]);
        }
        if tail > 2 {
            step!(2, er[j + 2]);
        }
    }
}

/// A weight matrix as stored in a vault.
pub enum Mat<'a> {
    Huff(Huff4<'a>),
    Rans(Rans4<'a>),
    Mxf(Mxf<'a>),
}

/// Rows computed into `tmp` (row-major, `rows x n`), then scattered into `y` (`n x rows`).
fn scatter(tmp: &[f32], y: &mut [f32], rows: usize, n: usize) {
    for r in 0..rows {
        for t in 0..n {
            y[t * rows + r] = tmp[r * n + t];
        }
    }
}

impl<'a> Mat<'a> {
    pub fn rows(&self) -> usize {
        match self {
            Mat::Huff(h) => h.rows,
            Mat::Rans(r) => r.rows,
            Mat::Mxf(m) => m.rows,
        }
    }
    pub fn cols(&self) -> usize {
        match self {
            Mat::Huff(h) => h.cols,
            Mat::Rans(r) => r.cols,
            Mat::Mxf(m) => m.cols,
        }
    }

    /// `Y[t][r] = sum_j W[r][j] * X[t][j]` for `n` vectors (`X` is `n x cols`, `Y` is `n x rows`).
    pub fn mv(&self, x: &[f32], y: &mut [f32], n: usize) {
        let (rows, cols) = (self.rows(), self.cols());
        assert!(x.len() >= n * cols && y.len() >= n * rows);
        let mut tmp = if n == 1 { Vec::new() } else { vec![0f32; rows * n] };
        {
            let out: &mut [f32] = if n == 1 { &mut y[..rows] } else { &mut tmp };
            match self {
                Mat::Huff(h) => {
                    let l3 = build_lut3(h.lut());
                    let es = h.es();
                    let cap = (cols >> 2) + 16;
                    out.par_chunks_mut(n).enumerate().with_min_len(8).for_each_init(
                        || (vec![0u8; 4 * cap], vec![0f32; cols]),
                        |(eb, w), (r, o)| {
                            dec4(&l3, es, h.row_offsets(r), cols, eb, cap);
                            #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
                            {
                                if n == 1 {
                                    o[0] = crate::simd::dot_huff(eb, cap, h.sm_row(r), x);
                                    return;
                                }
                                crate::simd::assemble_huff(eb, cap, h.sm_row(r), w);
                            }
                            #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma")))]
                            assemble_huff(eb, cap, h.sm_row(r), w);
                            for t in 0..n {
                                o[t] = dot8(w, &x[t * cols..(t + 1) * cols]);
                            }
                        },
                    );
                }
                Mat::Rans(rr) => {
                    assert!(rr.mbits == 12, "rans4: the engine needs M = 2^12");
                    let tab = rans_tab(&rr.freq);
                    let cs = rr.chains();
                    let rb = rr.r;
                    out.par_chunks_mut(rb * n).enumerate().for_each_init(
                        || (vec![0u8; rb * cols], vec![0f32; cols]),
                        |(e, w), (b, o)| {
                            let r0 = b * rb;
                            let nr = (rows - r0).min(rb);
                            rans_rows(&tab, cs, rr.block_offsets(b), cols, nr, e);
                            for i in 0..nr {
                                let sm = rr.sm_row(r0 + i);
                                let er = &e[i * cols..(i + 1) * cols];
                                #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
                                {
                                    if n == 1 {
                                        o[i] = crate::simd::dot_rows(er, sm, x);
                                        continue;
                                    }
                                    crate::simd::assemble_rows(er, sm, w);
                                }
                                #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma")))]
                                for j in 0..cols {
                                    w[j] = bf16_to_f32_parts(er[j], sm[j]);
                                }
                                for t in 0..n {
                                    o[i * n + t] = dot8(w, &x[t * cols..(t + 1) * cols]);
                                }
                            }
                        },
                    );
                }
                Mat::Mxf(m) => mxf_rows(m, x, out, n),
            }
        }
        if n > 1 {
            scatter(&tmp, y, rows, n);
        }
    }
}

/// 2 x FP4 value (exact integers 0, +-1, +-2, +-3, +-4, +-6, +-8, +-12); the scale carries the factor 0.5.
const FP4X2: [f32; 16] = [0., 1., 2., 3., 4., 6., 8., 12., 0., -1., -2., -3., -4., -6., -8., -12.];

fn mxf_rows(m: &Mxf, x: &[f32], out: &mut [f32], n: usize) {
    let (cols, nb) = (m.cols, m.nb);
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
    {
        out.par_chunks_mut(n).enumerate().with_min_len(16).for_each_init(
            || (vec![0u8; nb], crate::simd::zero_acc(n)),
            |(sc, acc), (r, o)| {
                m.row_scales(r, sc);
                crate::simd::mxf_row(m.row_blocks(r), sc, x, cols, n, acc, o);
            },
        );
        return;
    }
    #[allow(unreachable_code)]
    out.par_chunks_mut(n).enumerate().with_min_len(16).for_each_init(
        || (vec![0u8; nb], vec![[0f32; 8]; n]),
        |(sc, acc), (r, o)| {
            m.row_scales(r, sc);
            let bl = m.row_blocks(r);
            for a in acc.iter_mut() {
                *a = [0f32; 8];
            }
            let mut w = [0f32; 32];
            for b in 0..nb {
                let q = &bl[b * 16..b * 16 + 16];
                for j in 0..16 {
                    w[2 * j] = FP4X2[(q[j] & 15) as usize];
                    w[2 * j + 1] = FP4X2[(q[j] >> 4) as usize];
                }
                let scl = ldexp_half(sc[b]);
                for t in 0..n {
                    let xb = &x[t * cols + b * 32..t * cols + b * 32 + 32];
                    let a = &mut acc[t];
                    for l in 0..8 {
                        let mut d = w[l] * xb[l];
                        d = w[8 + l].mul_add(xb[8 + l], d);
                        d = w[16 + l].mul_add(xb[16 + l], d);
                        d = w[24 + l].mul_add(xb[24 + l], d);
                        a[l] = scl.mul_add(d, a[l]);
                    }
                }
            }
            for t in 0..n {
                let a = &acc[t];
                o[t] = ((a[0] + a[4]) + (a[1] + a[5])) + ((a[2] + a[6]) + (a[3] + a[7]));
            }
        },
    );
}

/// `ldexpf(0.5, s - 127)`, exact (including the subnormal range).
#[inline]
pub(crate) fn ldexp_half(s: u8) -> f32 {
    let e = s as i32 - 128;
    if e >= -126 {
        f32::from_bits(((e + 127) as u32) << 23)
    } else {
        f32::from_bits(1u32 << (e + 149).max(0))
    }
}

pub fn rmsnorm(out: &mut [f32], x: &[f32], w: &[f32], eps: f32) {
    let n = x.len();
    let mut ss = 0f64;
    for &v in x {
        ss += v as f64 * v as f64;
    }
    let inv = 1.0f32 / ((ss / n as f64) as f32 + eps).sqrt();
    for i in 0..n {
        out[i] = w[i] * (x[i] * inv);
    }
}

pub fn rmsnorm_inplace(x: &mut [f32], w: &[f32], eps: f32) {
    let n = x.len();
    let mut ss = 0f64;
    for &v in x.iter() {
        ss += v as f64 * v as f64;
    }
    let inv = 1.0f32 / ((ss / n as f64) as f32 + eps).sqrt();
    for i in 0..n {
        x[i] = w[i] * (x[i] * inv);
    }
}

#[inline]
pub fn rope_head(v: &mut [f32], cs: &[f32], sn: &[f32]) {
    let half = cs.len();
    for i in 0..half {
        let (x0, x1) = (v[i], v[i + half]);
        v[i] = x0 * cs[i] - x1 * sn[i];
        v[i + half] = x1 * cs[i] + x0 * sn[i];
    }
}

/// f32 -> IEEE half, round to nearest even (what `_mm256_cvtps_ph(x, 0)` does)
pub fn f32_to_f16(f: f32) -> u16 {
    let x = f.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let e = ((x >> 23) & 0xFF) as i32;
    let m = x & 0x7F_FFFF;
    if e == 255 {
        return sign | 0x7C00 | if m != 0 { 0x200 | (m >> 13) as u16 } else { 0 };
    }
    let e16 = e - 127 + 15;
    if e16 >= 31 {
        return sign | 0x7C00;
    }
    if e16 <= 0 {
        if e16 < -10 {
            return sign;
        }
        let mm = m | 0x80_0000;
        let shift = (14 - e16) as u32;
        let half = 1u32 << (shift - 1);
        let rem = mm & ((1u32 << shift) - 1);
        let mut r = mm >> shift;
        if rem > half || (rem == half && (r & 1) == 1) {
            r += 1;
        }
        return sign | r as u16;
    }
    let mut r = ((e16 as u32) << 10) | (m >> 13);
    let rem = m & 0x1FFF;
    if rem > 0x1000 || (rem == 0x1000 && (r & 1) == 1) {
        r += 1;
    }
    sign | r as u16
}

pub fn f16_to_f32(h: u16) -> f32 {
    let s = ((h >> 15) as u32) << 31;
    let e = ((h >> 10) & 0x1F) as u32;
    let m = (h & 0x3FF) as u32;
    let bits = if e == 0 {
        if m == 0 {
            s
        } else {
            let mut e2: i32 = 113;
            let mut m2 = m;
            while m2 & 0x400 == 0 {
                m2 <<= 1;
                e2 -= 1;
            }
            s | ((e2 as u32) << 23) | ((m2 & 0x3FF) << 13)
        }
    } else if e == 31 {
        s | 0x7F80_0000 | (m << 13)
    } else {
        s | ((e + 112) << 23) | (m << 13)
    };
    f32::from_bits(bits)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn f16_roundtrip_matches_rne() {
        for bits in (0u32..=0xFFFF_FFFF).step_by(65_521) {
            let f = f32::from_bits(bits);
            if f.is_nan() {
                continue;
            }
            let h = f32_to_f16(f);
            let back = f16_to_f32(h);
            // the nearest representable half: no other half is closer
            if back.is_finite() && f.is_finite() {
                let up = f16_to_f32(h.wrapping_add(1));
                let dn = f16_to_f32(h.wrapping_sub(1));
                let d = (back as f64 - f as f64).abs();
                assert!(d <= (up as f64 - f as f64).abs() || !up.is_finite(), "{f} -> {h:#x}");
                assert!(d <= (dn as f64 - f as f64).abs() || !dn.is_finite() || h & 0x7FFF == 0, "{f} -> {h:#x}");
            }
        }
    }
    #[test]
    fn dot8_simd_matches_scalar() {
        let mut st = 12345u64;
        let mut r = || {
            st ^= st << 13;
            st ^= st >> 7;
            st ^= st << 17;
            ((st >> 40) as f32 / 16777216.0 - 0.5) * 3.0
        };
        for n in [1usize, 7, 8, 9, 31, 1000, 1027] {
            let w: Vec<f32> = (0..n).map(|_| r()).collect();
            let x: Vec<f32> = (0..n).map(|_| r()).collect();
            assert_eq!(dot8(&w, &x).to_bits(), dot8_scalar(&w, &x).to_bits(), "n={n}");
        }
    }
    #[test]
    fn ldexp_half_exact() {
        for s in 0..=255u8 {
            let want = 0.5f64 * 2f64.powi(s as i32 - 127);
            assert_eq!(ldexp_half(s) as f64, want, "s={s}");
        }
    }
}
