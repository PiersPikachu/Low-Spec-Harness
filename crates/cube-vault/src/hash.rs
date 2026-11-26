//! Stream checksums.
//!
//! * [`legacy_hash64`]: the FNV-style hash written into `vault.idx` by vaults v3-v7 (kept so converted vaults still verify
//!   against the hashes made when they were packed).
//! * [`xxh3`]: the checksum of every v8 section and of streams in new packs.

/// The v3-v7 stream hash: 8-byte words `h ^= w; h *= P; h ^= h >> 29`, then the tail bytes `h ^= b; h *= P`.
pub fn legacy_hash64(buf: &[u8]) -> u64 {
    const P: u64 = 1_099_511_628_211;
    let mut h: u64 = 1_469_598_103_934_665_603;
    let mut chunks = buf.chunks_exact(8);
    for c in &mut chunks {
        let w = u64::from_le_bytes(c.try_into().unwrap());
        h ^= w;
        h = h.wrapping_mul(P);
        h ^= h >> 29;
    }
    for &b in chunks.remainder() {
        h ^= b as u64;
        h = h.wrapping_mul(P);
    }
    h
}

/// xxh3-64 (seed 0).
pub fn xxh3(buf: &[u8]) -> u64 {
    xxhash_rust::xxh3::xxh3_64(buf)
}

/// Which hash a stream carries.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[repr(u8)]
pub enum HashKind {
    Xxh3 = 1,
    Legacy = 2,
}

impl HashKind {
    pub fn of(self, buf: &[u8]) -> u64 {
        match self {
            HashKind::Xxh3 => xxh3(buf),
            HashKind::Legacy => legacy_hash64(buf),
        }
    }
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            1 => Some(HashKind::Xxh3),
            2 => Some(HashKind::Legacy),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn legacy_matches_python_reference() {
        // values from pack_universal_v3.hash64_ref
        assert_eq!(legacy_hash64(b""), 1_469_598_103_934_665_603);
        let mut h: u64 = 1_469_598_103_934_665_603;
        for &b in b"abc" {
            h ^= b as u64;
            h = h.wrapping_mul(1_099_511_628_211);
        }
        assert_eq!(legacy_hash64(b"abc"), h);
    }
}
