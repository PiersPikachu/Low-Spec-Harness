//! Cube vaults: lossless compressed model weights.
//!
//! [`Vault::open`] reads a v8 vault (`vault.cv8` + `vault.bin`) or, when there is no manifest, a v3-v7 vault (`vault.idx` +
//! `vault.bin` + `resident.bin`), which it converts in memory. [`Vault::convert`] writes the v8 manifest next to an existing
//! v3-v7 vault: the weight streams are untouched, so the conversion is instant and lossless.

pub mod hash;
pub mod huff4;
pub mod legacy;
pub mod meta;
pub mod mxf;
pub mod rans4;
pub mod stream;
pub mod v8;

use anyhow::{ensure, Context, Result};
use hash::HashKind;
use meta::{Arch, ModelMeta};
use std::fs::File;
use std::path::{Path, PathBuf};
use stream::StreamEntry;
use v8::{Manifest, Role, StreamInfo, FILE_SECTIONS};

pub use meta::{Codec, Rope};

pub const MANIFEST: &str = "vault.cv8";
pub const DATA: &str = "vault.bin";

pub struct Vault {
    pub dir: PathBuf,
    pub manifest: Manifest,
    /// the v3-v7 version when opened without a manifest
    pub legacy_version: Option<u32>,
    data: File,
}

fn role_of(meta: &ModelMeta, i: usize) -> (Role, u32, u32) {
    if let Some(m) = &meta.moe {
        if i >= m.expert_stream0 {
            let k = i - m.expert_stream0;
            return (Role::Expert, (k / m.experts) as u32, (k % m.experts) as u32);
        }
    }
    if i < meta.layers {
        (Role::Layer, i as u32, u32::MAX)
    } else if i == meta.embed_stream {
        (Role::Embed, u32::MAX, u32::MAX)
    } else if i == meta.lm_head_stream {
        (Role::LmHead, u32::MAX, u32::MAX)
    } else {
        (Role::Extra, u32::MAX, u32::MAX)
    }
}

/// Build a v8 manifest from a v3-v7 vault folder.
pub fn manifest_from_legacy(dir: &Path) -> Result<(Manifest, u32)> {
    let lg = legacy::read_idx(dir)?;
    let mut meta = lg.meta;
    meta.lossy = legacy::read_lossy(dir);
    meta.tensors = legacy::read_tensors(dir);
    let resident = legacy::read_resident(dir, meta.resident_len())?;
    if meta.arch == Arch::Gptoss {
        ensure!(lg.rope_inv.len() == meta.head_dim / 2, "gpt-oss vault.idx: rope_inv has {} values, need {}", lg.rope_inv.len(), meta.head_dim / 2);
    }
    let streams = lg
        .streams
        .iter()
        .enumerate()
        .map(|(i, e)| {
            let (role, layer, expert) = role_of(&meta, i);
            StreamInfo { e: *e, hash_kind: HashKind::Legacy, role, nsub: 0, layer, expert }
        })
        .collect();
    let mut files = Vec::new();
    for (tag, name) in FILE_SECTIONS {
        if let Ok(b) = std::fs::read(dir.join(name)) {
            files.push((**tag, b));
        }
    }
    let m = Manifest { flags: v8::FLAG_CONVERTED, meta, streams, resident, rope_inv: lg.rope_inv, files };
    Ok((m, lg.version))
}

impl Vault {
    pub fn open(dir: impl AsRef<Path>) -> Result<Vault> {
        let dir = dir.as_ref().to_path_buf();
        let (manifest, legacy_version) = if dir.join(MANIFEST).exists() {
            (Manifest::read(&dir.join(MANIFEST))?, None)
        } else {
            let (m, v) = manifest_from_legacy(&dir)?;
            (m, Some(v))
        };
        let data = File::open(dir.join(DATA)).with_context(|| format!("cannot open {}", dir.join(DATA).display()))?;
        let len = data.metadata()?.len();
        if let Some(last) = manifest.streams.iter().map(|s| s.e.off + s.e.padded).max() {
            ensure!(last <= len, "vault.bin is {len} bytes, the stream table needs {last} (incomplete download?)");
        }
        Ok(Vault { dir, manifest, legacy_version, data })
    }

    /// Write `vault.cv8` next to a v3-v7 vault. Returns the manifest size in bytes.
    pub fn convert(dir: impl AsRef<Path>) -> Result<u64> {
        let dir = dir.as_ref();
        let (m, _) = manifest_from_legacy(dir)?;
        m.write(&dir.join(MANIFEST))?;
        Ok(std::fs::metadata(dir.join(MANIFEST))?.len())
    }

    pub fn meta(&self) -> &ModelMeta {
        &self.manifest.meta
    }
    pub fn stream(&self, i: usize) -> &StreamEntry {
        &self.manifest.streams[i].e
    }
    pub fn n_streams(&self) -> usize {
        self.manifest.streams.len()
    }
    pub fn resident(&self) -> &[f32] {
        &self.manifest.resident
    }
    /// per-layer resident slice
    pub fn layer_resident(&self, l: usize) -> &[f32] {
        let n = self.meta().layer_resident.len;
        &self.manifest.resident[l * n..(l + 1) * n]
    }
    pub fn final_norm(&self) -> &[f32] {
        let m = self.meta();
        &self.manifest.resident[m.layers * m.layer_resident.len..]
    }
    pub fn data_path(&self) -> PathBuf {
        self.dir.join(DATA)
    }

    /// Read bytes of `vault.bin` at an absolute offset.
    pub fn read_at(&self, off: u64, buf: &mut [u8]) -> Result<()> {
        read_exact_at(&self.data, off, buf).with_context(|| format!("vault.bin read failed at {off}"))
    }

    /// Read a whole (padded) stream.
    pub fn read_stream(&self, i: usize) -> Result<Vec<u8>> {
        let e = self.stream(i);
        let mut buf = vec![0u8; e.padded as usize];
        self.read_at(e.off, &mut buf)?;
        Ok(buf)
    }

    /// Read a stream and check its hash.
    pub fn read_stream_checked(&self, i: usize) -> Result<Vec<u8>> {
        let buf = self.read_stream(i)?;
        let s = &self.manifest.streams[i];
        ensure!(s.hash_kind.of(&buf) == s.e.hash, "stream {i}: checksum mismatch");
        Ok(buf)
    }

    /// Tokenizer or template file: from the manifest, else the loose file in the vault folder.
    pub fn file(&self, name: &str) -> Option<Vec<u8>> {
        FILE_SECTIONS
            .iter()
            .find(|(_, n)| *n == name)
            .and_then(|(t, _)| self.manifest.file(t).map(|b| b.to_vec()))
            .or_else(|| std::fs::read(self.dir.join(name)).ok())
    }
}

#[cfg(unix)]
fn read_exact_at(f: &File, off: u64, buf: &mut [u8]) -> std::io::Result<()> {
    use std::os::unix::fs::FileExt;
    f.read_exact_at(buf, off)
}

#[cfg(windows)]
fn read_exact_at(f: &File, mut off: u64, mut buf: &mut [u8]) -> std::io::Result<()> {
    use std::os::windows::fs::FileExt;
    while !buf.is_empty() {
        let n = f.seek_read(buf, off)?;
        if n == 0 {
            return Err(std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "short read"));
        }
        off += n as u64;
        buf = &mut buf[n..];
    }
    Ok(())
}
