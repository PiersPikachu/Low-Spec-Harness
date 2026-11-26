//! Conversations: data/conversations/<id>.json ({id, title, created, updated, vault, messages}), same files as the Python harness.

use serde_json::{json, Value as J};
use std::path::Path;

pub fn now_f() -> f64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0)
}

pub fn new_id() -> String {
    let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let x = (t as u64) ^ ((t >> 64) as u64) ^ (std::process::id() as u64).rotate_left(32);
    let mut s = x.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    s ^= s >> 29;
    format!("{:012x}", s & 0xFFFF_FFFF_FFFF)
}

pub fn new_conversation(vault: &str) -> J {
    let t = now_f();
    json!({"id": new_id(), "title": "New chat", "created": t, "updated": t, "vault": vault, "messages": []})
}

pub fn list(dir: &Path) -> Vec<J> {
    let mut v: Vec<J> = std::fs::read_dir(dir)
        .map(|r| r.flatten().filter(|e| e.path().extension().map_or(false, |x| x == "json")).filter_map(|e| std::fs::read(e.path()).ok().and_then(|b| serde_json::from_slice(&b).ok())).collect())
        .unwrap_or_default();
    v.sort_by(|a, b| b["updated"].as_f64().unwrap_or(0.0).partial_cmp(&a["updated"].as_f64().unwrap_or(0.0)).unwrap());
    v
}

pub fn save(dir: &Path, c: &mut J) -> std::io::Result<()> {
    c["updated"] = json!(now_f());
    let _ = std::fs::create_dir_all(dir);
    std::fs::write(dir.join(format!("{}.json", c["id"].as_str().unwrap_or("x"))), serde_json::to_string_pretty(c).unwrap())
}

pub fn delete(dir: &Path, id: &str) {
    let _ = std::fs::remove_file(dir.join(format!("{id}.json")));
}
