//! Streams: one super-blob per layer / embedding / lm_head / expert, zero-padded to the 4096-byte sector size.
//! `u32 magic 0x534C5657 ("WVLS"), u32 n, u64 off[n]` (from the stream start), sub-blobs 64-aligned.

use anyhow::{ensure, Result};

pub const MAGIC: u32 = 0x534C_5657;
pub const SECT: usize = 4096;

/// One entry of the stream table.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct StreamEntry {
    pub off: u64,
    pub size: u64,
    pub padded: u64,
    pub raw: u64,
    pub hash: u64,
}

/// The sub-blobs of a stream buffer.
pub fn sub_blobs(buf: &[u8]) -> Result<Vec<&[u8]>> {
    ensure!(buf.len() >= 8, "stream too short");
    let magic = u32::from_le_bytes(buf[0..4].try_into()?);
    ensure!(magic == MAGIC, "bad stream magic {magic:#x}");
    let n = u32::from_le_bytes(buf[4..8].try_into()?) as usize;
    ensure!(buf.len() >= 8 + 8 * n, "stream header truncated");
    let offs: Vec<usize> = (0..n).map(|i| u64::from_le_bytes(buf[8 + 8 * i..16 + 8 * i].try_into().unwrap()) as usize).collect();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let end = if i + 1 < n { offs[i + 1] } else { buf.len() };
        ensure!(offs[i] <= end && end <= buf.len(), "sub-blob {i} out of range");
        out.push(&buf[offs[i]..end]);
    }
    Ok(out)
}

/// Offset of sub-blob `i` inside a stream (reads only the header).
pub fn sub_offset(buf: &[u8], i: usize) -> u64 {
    u64::from_le_bytes(buf[8 + 8 * i..16 + 8 * i].try_into().unwrap())
}

/// Build a stream from sub-blobs; returns (padded buffer, unpadded size). Byte-identical to the Python packers.
pub fn build(subs: &[&[u8]]) -> (Vec<u8>, u64) {
    let n = subs.len();
    let hdr = 128.max((8 + 8 * n + 63) & !63);
    let mut pos = hdr;
    let mut offs = Vec::with_capacity(n);
    for b in subs {
        pos = (pos + 63) & !63;
        offs.push(pos);
        pos += b.len();
    }
    let padded = (pos + SECT - 1) & !(SECT - 1);
    let mut buf = vec![0u8; padded];
    buf[0..4].copy_from_slice(&MAGIC.to_le_bytes());
    buf[4..8].copy_from_slice(&(n as u32).to_le_bytes());
    for (i, (&o, b)) in offs.iter().zip(subs).enumerate() {
        buf[8 + 8 * i..16 + 8 * i].copy_from_slice(&(o as u64).to_le_bytes());
        buf[o..o + b.len()].copy_from_slice(b);
    }
    (buf, pos as u64)
}
