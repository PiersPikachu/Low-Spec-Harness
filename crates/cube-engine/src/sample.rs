//! Sampling, bit-compatible with the C engine (same RNG, same top-k insertion order, same arithmetic).

pub const DEFAULT_SEED: u64 = 88172645463325252;

pub struct Rng(pub u64);
impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(if seed == 0 { DEFAULT_SEED } else { seed })
    }
    pub fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(2685821657736338717)
    }
    pub fn f(&mut self) -> f32 {
        (self.next() >> 40) as f32 / 16777216.0f32
    }
}

pub fn argmax(l: &[f32]) -> usize {
    let mut b = 0;
    for i in 1..l.len() {
        if l[i] > l[b] {
            b = i;
        }
    }
    b
}

/// The distribution `sample` draws from: (ids, probabilities before the top-p renormalisation, kept count, kept sum).
fn dist(l: &[f32], temp: f32, topp: f32, topk: i32) -> (Vec<usize>, Vec<f32>, usize, f32) {
    let v = l.len();
    let mut topk = if topk < 1 || topk > 128 { 128 } else { topk as usize };
    if topk > v {
        topk = v;
    }
    let mut idx = [0usize; 128];
    let mut val = [0f32; 128];
    let mut k = 0usize;
    for (i, &x) in l.iter().enumerate() {
        if k == topk && x <= val[k - 1] {
            continue;
        }
        let mut j = if k < topk {
            k += 1;
            k - 1
        } else {
            k - 1
        };
        while j > 0 && val[j - 1] < x {
            val[j] = val[j - 1];
            idx[j] = idx[j - 1];
            j -= 1;
        }
        val[j] = x;
        idx[j] = i;
    }
    let mut p = vec![0f32; k];
    let (mut sum, mx) = (0f32, val[0]);
    for j in 0..k {
        p[j] = ((val[j] - mx) / temp).exp();
        sum += p[j];
    }
    for x in p.iter_mut() {
        *x /= sum;
    }
    let mut cum = 0f32;
    let mut kk = k;
    for j in 0..k {
        cum += p[j];
        if cum >= topp {
            kk = j + 1;
            break;
        }
    }
    let mut s2 = 0f32;
    for j in 0..kk {
        s2 += p[j];
    }
    (idx[..k].to_vec(), p, kk, s2)
}

pub fn sample(l: &[f32], temp: f32, topp: f32, topk: i32, rng: &mut Rng) -> usize {
    if temp <= 0.0 {
        return argmax(l);
    }
    let (idx, p, kk, s2) = dist(l, temp, topp, topk);
    let r = rng.f() * s2;
    let mut c2 = 0f32;
    for j in 0..kk {
        c2 += p[j];
        if r <= c2 {
            return idx[j];
        }
    }
    idx[kk - 1]
}

/// (id, probability) pairs of the distribution `sample` draws from (VERIFYP).
pub fn distribution(l: &[f32], temp: f32, topp: f32, topk: i32) -> Vec<(usize, f32)> {
    if temp <= 0.0 {
        return vec![(argmax(l), 1.0)];
    }
    let (idx, p, kk, s2) = dist(l, temp, topp, topk);
    (0..kk).map(|j| (idx[j], p[j] / s2)).collect()
}
