//! AVX2/FMA versions of the hot loops, written with the same intrinsics (and so the same rounding) as the C engine.
//! Compiled when the build enables AVX2 + FMA (`.cargo/config.toml` builds for the native CPU); the scalar code in
//! `kernels` gives identical results without them.

#![allow(unsafe_op_in_unsafe_fn)]
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub const ON: bool = cfg!(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"));

#[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
mod imp {
    use super::*;

    #[inline(always)]
    unsafe fn hsum_tree(acc: __m256) -> [f32; 8] {
        let mut a = [0f32; 8];
        _mm256_storeu_ps(a.as_mut_ptr(), acc);
        a
    }

    /// 8 lanes, mul then add, tail into lane 0, `((a0+a1)+(a2+a3))+((a4+a5)+(a6+a7))`
    #[inline]
    pub fn dot8(w: &[f32], x: &[f32]) -> f32 {
        let n = w.len();
        assert!(x.len() >= n);
        unsafe {
            let mut acc = _mm256_setzero_ps();
            let mut j = 0;
            while j + 8 <= n {
                acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_loadu_ps(w.as_ptr().add(j)), _mm256_loadu_ps(x.as_ptr().add(j))));
                j += 8;
            }
            let mut a = hsum_tree(acc);
            while j < n {
                a[0] += w[j] * x[j];
                j += 1;
            }
            ((a[0] + a[1]) + (a[2] + a[3])) + ((a[4] + a[5]) + (a[6] + a[7]))
        }
    }

    /// 8 bf16 weights from 8 exponent bytes and 8 sign+mantissa bytes
    #[inline(always)]
    unsafe fn asm8(e8: __m128i, sm: *const u8) -> __m256 {
        let s8 = _mm_loadl_epi64(sm as *const __m128i);
        let e32 = _mm256_cvtepu8_epi32(e8);
        let s32 = _mm256_cvtepu8_epi32(s8);
        let bits = _mm256_or_si256(
            _mm256_or_si256(_mm256_slli_epi32(_mm256_and_si256(s32, _mm256_set1_epi32(0x80)), 24), _mm256_slli_epi32(e32, 23)),
            _mm256_slli_epi32(_mm256_and_si256(s32, _mm256_set1_epi32(0x7F)), 16),
        );
        _mm256_castsi256_ps(bits)
    }

    /// exponents of weights 4i..4i+15 from the 4 stream arrays, in weight order
    #[inline(always)]
    unsafe fn ev16(eb: *const u8, cap: usize, i: usize) -> (__m128i, __m128i) {
        let rd = |p: *const u8| (p as *const i32).read_unaligned();
        let a = rd(eb.add(i));
        let b = rd(eb.add(cap + i));
        let c = rd(eb.add(2 * cap + i));
        let d = rd(eb.add(3 * cap + i));
        let ab = _mm_unpacklo_epi8(_mm_cvtsi32_si128(a), _mm_cvtsi32_si128(b));
        let cd = _mm_unpacklo_epi8(_mm_cvtsi32_si128(c), _mm_cvtsi32_si128(d));
        let v = _mm_unpacklo_epi16(ab, cd);
        (v, _mm_srli_si128(v, 8))
    }

    #[inline(always)]
    fn bf(e: u8, s: u8) -> f32 {
        f32::from_bits((((s as u32) & 0x80) << 24) | ((e as u32) << 23) | (((s as u32) & 0x7F) << 16))
    }

    /// one huffman4 row (exponents already split into 4 arrays) dot x — the C engine's `dot_decode3` order
    #[inline]
    pub fn dot_huff(eb: &[u8], cap: usize, sm: &[u8], x: &[f32]) -> f32 {
        let cols = sm.len();
        assert!(eb.len() >= 4 * cap && cap >= (cols >> 2) + 4 && x.len() >= cols);
        unsafe {
            let (pe, ps, px) = (eb.as_ptr(), sm.as_ptr(), x.as_ptr());
            let mut acc = _mm256_setzero_ps();
            let mut j = 0;
            while j + 16 <= cols {
                let (lo, hi) = ev16(pe, cap, j >> 2);
                acc = _mm256_add_ps(acc, _mm256_mul_ps(asm8(lo, ps.add(j)), _mm256_loadu_ps(px.add(j))));
                acc = _mm256_add_ps(acc, _mm256_mul_ps(asm8(hi, ps.add(j + 8)), _mm256_loadu_ps(px.add(j + 8))));
                j += 16;
            }
            if j + 8 <= cols {
                let (lo, _) = ev16(pe, cap, j >> 2);
                acc = _mm256_add_ps(acc, _mm256_mul_ps(asm8(lo, ps.add(j)), _mm256_loadu_ps(px.add(j))));
                j += 8;
            }
            let mut a = hsum_tree(acc);
            while j < cols {
                a[0] += bf(eb[(j & 3) * cap + (j >> 2)], sm[j]) * x[j];
                j += 1;
            }
            ((a[0] + a[1]) + (a[2] + a[3])) + ((a[4] + a[5]) + (a[6] + a[7]))
        }
    }

    /// huffman4 row -> f32 weights
    #[inline]
    pub fn assemble_huff(eb: &[u8], cap: usize, sm: &[u8], w: &mut [f32]) {
        let cols = sm.len();
        assert!(eb.len() >= 4 * cap && cap >= (cols >> 2) + 4 && w.len() >= cols);
        unsafe {
            let (pe, ps, pw) = (eb.as_ptr(), sm.as_ptr(), w.as_mut_ptr());
            let mut j = 0;
            while j + 8 <= cols {
                let (lo, _) = ev16(pe, cap, j >> 2);
                _mm256_storeu_ps(pw.add(j), asm8(lo, ps.add(j)));
                j += 8;
            }
            while j < cols {
                w[j] = bf(eb[(j & 3) * cap + (j >> 2)], sm[j]);
                j += 1;
            }
        }
    }

    /// row-major exponent bytes + sign/mantissa dot x (the C engine's `dot_rans_row`)
    #[inline]
    pub fn dot_rows(er: &[u8], sm: &[u8], x: &[f32]) -> f32 {
        let cols = sm.len();
        assert!(er.len() >= cols && x.len() >= cols);
        unsafe {
            let mut acc = _mm256_setzero_ps();
            let mut j = 0;
            while j + 8 <= cols {
                let e8 = _mm_loadl_epi64(er.as_ptr().add(j) as *const __m128i);
                acc = _mm256_add_ps(acc, _mm256_mul_ps(asm8(e8, sm.as_ptr().add(j)), _mm256_loadu_ps(x.as_ptr().add(j))));
                j += 8;
            }
            let mut a = hsum_tree(acc);
            while j < cols {
                a[0] += bf(er[j], sm[j]) * x[j];
                j += 1;
            }
            ((a[0] + a[1]) + (a[2] + a[3])) + ((a[4] + a[5]) + (a[6] + a[7]))
        }
    }

    #[inline]
    pub fn assemble_rows(er: &[u8], sm: &[u8], w: &mut [f32]) {
        let cols = sm.len();
        assert!(er.len() >= cols && w.len() >= cols);
        unsafe {
            let mut j = 0;
            while j + 8 <= cols {
                let e8 = _mm_loadl_epi64(er.as_ptr().add(j) as *const __m128i);
                _mm256_storeu_ps(w.as_mut_ptr().add(j), asm8(e8, sm.as_ptr().add(j)));
                j += 8;
            }
            while j < cols {
                w[j] = bf(er[j], sm[j]);
                j += 1;
            }
        }
    }

    /// One MXFP4 row against n vectors (the C engine's AVX2 `mxf4_mv` inner loop). `acc` has n entries.
    #[inline]
    pub fn mxf_row(blocks: &[u8], sc: &[u8], x: &[f32], cols: usize, n: usize, acc: &mut [__m256], out: &mut [f32]) {
        let nb = sc.len();
        assert!(blocks.len() >= nb * 16 && x.len() >= n * cols && acc.len() >= n && out.len() >= n && nb * 32 <= cols);
        unsafe {
            let lut = _mm_setr_epi8(0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12);
            let m4 = _mm_set1_epi8(15);
            for a in acc.iter_mut().take(n) {
                *a = _mm256_setzero_ps();
            }
            for b in 0..nb {
                let q = _mm_loadu_si128(blocks.as_ptr().add(b * 16) as *const __m128i);
                let lo = _mm_and_si128(q, m4);
                let hi = _mm_and_si128(_mm_srli_epi16(q, 4), m4);
                let v0 = _mm_shuffle_epi8(lut, _mm_unpacklo_epi8(lo, hi));
                let v1 = _mm_shuffle_epi8(lut, _mm_unpackhi_epi8(lo, hi));
                let w0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(v0));
                let w1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(v0, 8)));
                let w2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(v1));
                let w3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(v1, 8)));
                let scl = _mm256_set1_ps(crate::kernels::ldexp_half(sc[b]));
                for t in 0..n {
                    let px = x.as_ptr().add(t * cols + b * 32);
                    let mut d = _mm256_mul_ps(w0, _mm256_loadu_ps(px));
                    d = _mm256_fmadd_ps(w1, _mm256_loadu_ps(px.add(8)), d);
                    d = _mm256_fmadd_ps(w2, _mm256_loadu_ps(px.add(16)), d);
                    d = _mm256_fmadd_ps(w3, _mm256_loadu_ps(px.add(24)), d);
                    acc[t] = _mm256_fmadd_ps(scl, d, acc[t]);
                }
            }
            for t in 0..n {
                let a = hsum_tree(acc[t]);
                out[t] = ((a[0] + a[4]) + (a[1] + a[5])) + ((a[2] + a[6]) + (a[3] + a[7]));
            }
        }
    }

    pub fn zero_acc(n: usize) -> Vec<__m256> {
        unsafe { vec![_mm256_setzero_ps(); n] }
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
pub use imp::*;
