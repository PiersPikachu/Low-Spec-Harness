//! `vault.cv8`: header + checksummed sections (see docs/VAULT_V8.md).

use crate::hash::{xxh3, HashKind};
use crate::meta::ModelMeta;
use crate::stream::StreamEntry;
use anyhow::{bail, ensure, Context, Result};
use std::io::Write;
use std::path::Path;

pub const MAGIC: &[u8; 8] = b"CUBEVLT8";
pub const VERSION: u32 = 8;
pub const FLAG_EMBEDDED_DATA: u32 = 1;
pub const FLAG_CONVERTED: u32 = 2;
const HEADER: usize = 64;
const ENTRY: usize = 32;
const STRM_ENTRY: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Role {
    Layer = 0,
    Embed = 1,
    LmHead = 2,
    Expert = 3,
    Extra = 4,
}

#[derive(Clone, Debug)]
pub struct StreamInfo {
    pub e: StreamEntry,
    pub hash_kind: HashKind,
    pub role: Role,
    pub nsub: u16,
    pub layer: u32,
    pub expert: u32,
}

/// Everything in a manifest.
#[derive(Clone, Debug)]
pub struct Manifest {
    pub flags: u32,
    pub meta: ModelMeta,
    pub streams: Vec<StreamInfo>,
    pub resident: Vec<f32>,
    pub rope_inv: Vec<f32>,
    /// named text/binary sections: TOKJ, TOKC, CHAT, GENC, SRCC
    pub files: Vec<([u8; 4], Vec<u8>)>,
}

pub const FILE_SECTIONS: &[(&[u8; 4], &str)] = &[
    (b"TOKJ", "tokenizer.json"),
    (b"TOKC", "tokenizer_config.json"),
    (b"CHAT", "chat_template.jinja"),
    (b"GENC", "generation_config.json"),
    (b"SRCC", "config.json"),
];

impl Manifest {
    pub fn file(&self, tag: &[u8; 4]) -> Option<&[u8]> {
        self.files.iter().find(|(t, _)| t == tag).map(|(_, b)| b.as_slice())
    }

    fn strm_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.streams.len() * STRM_ENTRY);
        for s in &self.streams {
            for v in [s.e.off, s.e.size, s.e.padded, s.e.raw, s.e.hash] {
                out.extend_from_slice(&v.to_le_bytes());
            }
            out.push(s.hash_kind as u8);
            out.push(s.role as u8);
            out.extend_from_slice(&s.nsub.to_le_bytes());
            out.extend_from_slice(&s.layer.to_le_bytes());
            out.extend_from_slice(&s.expert.to_le_bytes());
            out.extend_from_slice(&[0u8; 12]);
        }
        out
    }

    /// Serialise to bytes (without embedded data).
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut secs: Vec<([u8; 4], Vec<u8>)> = vec![(*b"META", serde_json::to_vec_pretty(&self.meta)?), (*b"STRM", self.strm_bytes())];
        secs.push((*b"RESI", self.resident.iter().flat_map(|x| x.to_le_bytes()).collect()));
        if !self.rope_inv.is_empty() {
            secs.push((*b"ROPE", self.rope_inv.iter().flat_map(|x| x.to_le_bytes()).collect()));
        }
        for (t, b) in &self.files {
            secs.push((*t, b.clone()));
        }
        let table_off = HEADER;
        let mut pos = (table_off + ENTRY * secs.len() + 63) & !63;
        let mut table = Vec::with_capacity(ENTRY * secs.len());
        let mut offs = Vec::new();
        for (t, b) in &secs {
            offs.push(pos);
            table.extend_from_slice(t);
            table.extend_from_slice(&0u32.to_le_bytes());
            table.extend_from_slice(&(pos as u64).to_le_bytes());
            table.extend_from_slice(&(b.len() as u64).to_le_bytes());
            table.extend_from_slice(&xxh3(b).to_le_bytes());
            pos = (pos + b.len() + 63) & !63;
        }
        let total = pos;
        let mut out = vec![0u8; total];
        out[0..8].copy_from_slice(MAGIC);
        out[8..12].copy_from_slice(&VERSION.to_le_bytes());
        out[12..16].copy_from_slice(&self.flags.to_le_bytes());
        out[16..24].copy_from_slice(&(table_off as u64).to_le_bytes());
        out[24..28].copy_from_slice(&(secs.len() as u32).to_le_bytes());
        out[32..40].copy_from_slice(&(total as u64).to_le_bytes());
        out[40..48].copy_from_slice(&xxh3(&table).to_le_bytes());
        out[table_off..table_off + table.len()].copy_from_slice(&table);
        for ((_, b), &o) in secs.iter().zip(&offs) {
            out[o..o + b.len()].copy_from_slice(b);
        }
        Ok(out)
    }

    pub fn write(&self, path: &Path) -> Result<()> {
        let tmp = path.with_extension("cv8.tmp");
        {
            let mut f = std::fs::File::create(&tmp).with_context(|| format!("cannot write {}", tmp.display()))?;
            f.write_all(&self.to_bytes()?)?;
            f.sync_all()?;
        }
        std::fs::rename(&tmp, path)?;
        Ok(())
    }

    /// Parse and verify every checksum.
    pub fn from_bytes(b: &[u8]) -> Result<Manifest> {
        ensure!(b.len() >= HEADER && &b[0..8] == MAGIC, "not a v8 vault manifest");
        let rd32 = |o: usize| u32::from_le_bytes(b[o..o + 4].try_into().unwrap());
        let rd64 = |o: usize| u64::from_le_bytes(b[o..o + 8].try_into().unwrap());
        let ver = rd32(8);
        ensure!(ver == VERSION, "manifest version {ver}, this build reads {VERSION}");
        let flags = rd32(12);
        let (toff, n) = (rd64(16) as usize, rd32(24) as usize);
        ensure!(rd64(32) as usize <= b.len(), "manifest truncated");
        ensure!(toff + ENTRY * n <= b.len(), "section table out of range");
        let table = &b[toff..toff + ENTRY * n];
        ensure!(xxh3(table) == rd64(40), "section table checksum mismatch");
        let mut get: Vec<([u8; 4], &[u8])> = Vec::new();
        for i in 0..n {
            let e = &table[ENTRY * i..ENTRY * (i + 1)];
            let tag: [u8; 4] = e[0..4].try_into()?;
            let (o, l, h) = (
                u64::from_le_bytes(e[8..16].try_into()?) as usize,
                u64::from_le_bytes(e[16..24].try_into()?) as usize,
                u64::from_le_bytes(e[24..32].try_into()?),
            );
            ensure!(o + l <= b.len(), "section {} out of range", String::from_utf8_lossy(&tag));
            let s = &b[o..o + l];
            ensure!(xxh3(s) == h, "section {} checksum mismatch", String::from_utf8_lossy(&tag));
            get.push((tag, s));
        }
        let find = |t: &[u8; 4]| get.iter().find(|(x, _)| x == t).map(|(_, s)| *s);
        let meta: ModelMeta = serde_json::from_slice(find(b"META").context("no META section")?)?;
        let strm = find(b"STRM").context("no STRM section")?;
        ensure!(strm.len() % STRM_ENTRY == 0, "STRM size");
        let mut streams = Vec::with_capacity(strm.len() / STRM_ENTRY);
        for c in strm.chunks_exact(STRM_ENTRY) {
            let r = |o: usize| u64::from_le_bytes(c[o..o + 8].try_into().unwrap());
            let role = match c[41] {
                0 => Role::Layer,
                1 => Role::Embed,
                2 => Role::LmHead,
                3 => Role::Expert,
                4 => Role::Extra,
                x => bail!("unknown stream role {x}"),
            };
            streams.push(StreamInfo {
                e: StreamEntry { off: r(0), size: r(8), padded: r(16), raw: r(24), hash: r(32) },
                hash_kind: HashKind::from_u8(c[40]).context("unknown hash kind")?,
                role,
                nsub: u16::from_le_bytes([c[42], c[43]]),
                layer: u32::from_le_bytes(c[44..48].try_into()?),
                expert: u32::from_le_bytes(c[48..52].try_into()?),
            });
        }
        let f32s = |s: &[u8]| -> Vec<f32> { s.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect() };
        let resident = f32s(find(b"RESI").context("no RESI section")?);
        ensure!(resident.len() == meta.resident_len(), "RESI has {} values, the model needs {}", resident.len(), meta.resident_len());
        let rope_inv = find(b"ROPE").map(f32s).unwrap_or_default();
        let files = FILE_SECTIONS.iter().filter_map(|(t, _)| find(t).map(|s| (**t, s.to_vec()))).collect();
        Ok(Manifest { flags, meta, streams, resident, rope_inv, files })
    }

    pub fn read(path: &Path) -> Result<Manifest> {
        let b = std::fs::read(path).with_context(|| format!("cannot read {}", path.display()))?;
        Manifest::from_bytes(&b).with_context(|| format!("{}", path.display()))
    }
}
