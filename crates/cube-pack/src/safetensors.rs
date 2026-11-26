//! Minimal safetensors reader (memory-mapped, one shard at a time is enough for the packer).

use anyhow::{bail, ensure, Context, Result};
use memmap2::Mmap;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug)]
pub struct TensorInfo {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub file: usize,
    pub start: usize,
    pub end: usize,
}

impl TensorInfo {
    pub fn numel(&self) -> usize {
        self.shape.iter().product::<usize>().max(if self.shape.is_empty() { 1 } else { 0 })
    }
    /// rows, cols of the 2-D view (leading dimension, product of the rest)
    pub fn matrix(&self) -> (usize, usize) {
        let rows = self.shape[0];
        (rows, self.shape[1..].iter().product())
    }
}

pub struct Checkpoint {
    pub dir: PathBuf,
    pub files: Vec<PathBuf>,
    maps: Vec<Mmap>,
    pub tensors: BTreeMap<String, TensorInfo>,
    pub config: serde_json::Value,
}

impl Checkpoint {
    pub fn open(dir: &Path) -> Result<Checkpoint> {
        let config: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("config.json")).context("config.json missing")?)?;
        let idx = dir.join("model.safetensors.index.json");
        let mut names: Vec<String> = if idx.exists() {
            let v: serde_json::Value = serde_json::from_slice(&std::fs::read(&idx)?)?;
            let mut s: Vec<String> = v["weight_map"].as_object().context("weight_map")?.values().filter_map(|x| x.as_str().map(String::from)).collect();
            s.sort();
            s.dedup();
            s
        } else {
            let mut s: Vec<String> = std::fs::read_dir(dir)?
                .filter_map(|e| e.ok().map(|e| e.file_name().to_string_lossy().to_string()))
                .filter(|n| n.ends_with(".safetensors"))
                .collect();
            s.sort();
            s
        };
        ensure!(!names.is_empty(), "no .safetensors files in {}", dir.display());
        let mut files = Vec::new();
        let mut maps = Vec::new();
        let mut tensors = BTreeMap::new();
        for (fi, n) in names.drain(..).enumerate() {
            let p = dir.join(&n);
            let f = std::fs::File::open(&p).with_context(|| format!("cannot open {}", p.display()))?;
            let m = unsafe { Mmap::map(&f)? };
            ensure!(m.len() >= 8, "{n}: too short");
            let hl = u64::from_le_bytes(m[0..8].try_into()?) as usize;
            let hdr: serde_json::Value = serde_json::from_slice(&m[8..8 + hl])?;
            let data0 = 8 + hl;
            for (k, v) in hdr.as_object().context("bad header")? {
                if k == "__metadata__" {
                    continue;
                }
                let o = v["data_offsets"].as_array().context("data_offsets")?;
                let (s, e) = (o[0].as_u64().unwrap() as usize, o[1].as_u64().unwrap() as usize);
                tensors.insert(
                    k.clone(),
                    TensorInfo {
                        name: k.clone(),
                        dtype: v["dtype"].as_str().unwrap_or("").to_string(),
                        shape: v["shape"].as_array().context("shape")?.iter().map(|x| x.as_u64().unwrap() as usize).collect(),
                        file: fi,
                        start: data0 + s,
                        end: data0 + e,
                    },
                );
            }
            files.push(p);
            maps.push(m);
        }
        Ok(Checkpoint { dir: dir.to_path_buf(), files, maps, tensors, config })
    }

    pub fn get(&self, name: &str) -> Result<&TensorInfo> {
        self.tensors.get(name).with_context(|| format!("tensor {name} missing"))
    }
    pub fn bytes(&self, t: &TensorInfo) -> &[u8] {
        &self.maps[t.file][t.start..t.end]
    }

    /// bf16 bits of a matrix; F16/F32 only when `convert` (rounded to nearest even, NaN -> 0x7FC0: lossy)
    pub fn bf16_bits(&self, t: &TensorInfo, convert: bool) -> Result<Vec<u16>> {
        let b = self.bytes(t);
        match t.dtype.as_str() {
            "BF16" => Ok(b.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()),
            "F16" | "F32" if convert => Ok(self.f32(t)?.into_iter().map(f32_to_bf16).collect()),
            d => bail!("{} is {d}: matrices must be BF16 to stay lossless (convert = rounding to bf16, lossy)", t.name),
        }
    }

    pub fn f32(&self, t: &TensorInfo) -> Result<Vec<f32>> {
        let b = self.bytes(t);
        Ok(match t.dtype.as_str() {
            "BF16" => b.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect(),
            "F16" => b.chunks_exact(2).map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect(),
            "F32" => b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect(),
            d => bail!("{}: unsupported dtype {d}", t.name),
        })
    }
}

pub fn f32_to_bf16(f: f32) -> u16 {
    if f.is_nan() {
        return 0x7FC0;
    }
    let u = f.to_bits() as u64;
    ((u + ((u >> 16) & 1) + 0x7FFF) >> 16) as u16
}

pub fn f16_to_f32(h: u16) -> f32 {
    let s = ((h >> 15) as u32) << 31;
    let e = ((h >> 10) & 0x1F) as u32;
    let m = (h & 0x3FF) as u32;
    let bits = if e == 0 {
        if m == 0 {
            s
        } else {
            // subnormal: normalise
            let mut e2 = 127 - 15 + 1;
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
        s | ((e + 127 - 15) << 23) | (m << 13)
    };
    f32::from_bits(bits)
}
