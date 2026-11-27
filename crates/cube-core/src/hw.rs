//! This machine: RAM, CPU threads, GPU (nvidia-smi), and the KV-cache cost of a model.

use cube_vault::meta::ModelMeta;

/// (total MB, available MB)
pub fn ram_mb() -> (f64, f64) {
    let mut s = sysinfo::System::new();
    s.refresh_memory();
    (s.total_memory() as f64 / 1048576.0, s.available_memory() as f64 / 1048576.0)
}

/// One line on where RAM is: free RAM, this process, and the largest processes (working sets).
pub fn mem_report() -> String {
    use sysinfo::{ProcessRefreshKind, ProcessesToUpdate, System};
    let mut s = System::new();
    s.refresh_memory();
    s.refresh_processes_specifics(ProcessesToUpdate::All, true, ProcessRefreshKind::nothing().with_memory());
    let me = sysinfo::get_current_pid().ok().and_then(|p| s.process(p)).map_or(0, |p| p.memory());
    let mut ps: Vec<(String, u64)> = s.processes().values().map(|p| (p.name().to_string_lossy().into_owned(), p.memory())).collect();
    ps.sort_by(|a, b| b.1.cmp(&a.1));
    let top: Vec<String> = ps.iter().take(4).map(|(n, m)| format!("{n} {:.0}", *m as f64 / 1048576.0)).collect();
    format!(
        "RAM: {:.0} MB free of {:.0}; this process {:.0} MB; largest (MB): {}",
        s.available_memory() as f64 / 1048576.0,
        s.total_memory() as f64 / 1048576.0,
        me as f64 / 1048576.0,
        top.join(", ")
    )
}

pub fn threads() -> usize {
    std::thread::available_parallelism().map_or(1, |n| n.get())
}

/// KV bytes for `ctx` positions (sliding-window layers hold window + 1024).
pub fn kv_bytes(m: &ModelMeta, ctx: usize, kv16: bool) -> f64 {
    let kvd = (m.kv_heads * m.head_dim) as f64;
    let bpe = if kv16 { 2.0 } else { 4.0 };
    (0..m.layers)
        .map(|l| {
            let sliding = m.gptoss.as_ref().map_or(false, |g| g.sliding_window > 0 && g.layer_sliding.get(l).copied() == Some(1));
            let sw = m.gptoss.as_ref().map_or(0, |g| g.sliding_window);
            let cap = if sliding && sw + 1024 < ctx { sw + 1024 } else { ctx };
            2.0 * cap as f64 * kvd * bpe
        })
        .sum()
}

#[derive(Clone, Debug, Default)]
pub struct Gpu {
    pub name: String,
    pub vram_total_mb: f64,
    pub vram_free_mb: f64,
    pub driver: String,
}

/// The first NVIDIA GPU, from nvidia-smi (None without one).
pub fn gpu() -> Option<Gpu> {
    let mut c = std::process::Command::new("nvidia-smi");
    c.args(["--query-gpu=name,memory.total,memory.free,driver_version", "--format=csv,noheader,nounits"]);
    crate::tools::external::no_window(&mut c);
    let o = c.output().ok()?;
    let s = String::from_utf8_lossy(&o.stdout);
    let l = s.lines().next()?;
    let f: Vec<&str> = l.split(',').map(str::trim).collect();
    if f.len() < 4 {
        return None;
    }
    Some(Gpu { name: f[0].into(), vram_total_mb: f[1].parse().ok()?, vram_free_mb: f[2].parse().ok()?, driver: f[3].into() })
}
