//! Model folders: what kind of drive each is on, how fast it reads, what it holds, and moving a vault between them.
//! A vault can live in any model folder; the engine streams weights from it while it runs, so a folder on an external
//! HDD/USB drive is fine for keeping models but slow for running them.

use crate::paths::{is_vault, Paths};
use serde_json::{json, Value as J};
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Free bytes on the volume holding `p`.
pub fn free_bytes(p: &Path) -> u64 {
    let p = p.canonicalize().unwrap_or_else(|_| p.to_path_buf());
    let disks = sysinfo::Disks::new_with_refreshed_list();
    disks.list().iter().filter(|d| p.starts_with(d.mount_point())).max_by_key(|d| d.mount_point().as_os_str().len()).map_or(0, |d| d.available_space())
}

/// Drive letter on Windows ("C:"), mount point elsewhere.
pub fn drive_of(p: &Path) -> String {
    let p = p.canonicalize().unwrap_or_else(|_| p.to_path_buf());
    let s = p.to_string_lossy().replace("\\\\?\\", "");
    if s.len() >= 2 && s.as_bytes()[1] == b':' {
        return s[..2].to_uppercase();
    }
    let disks = sysinfo::Disks::new_with_refreshed_list();
    disks.list().iter().filter(|d| p.starts_with(d.mount_point())).max_by_key(|d| d.mount_point().as_os_str().len()).map_or("/".into(), |d| d.mount_point().to_string_lossy().into_owned())
}

#[derive(Clone, Debug, Default)]
pub struct FolderSummary {
    pub exists: bool,
    pub free: u64,
    pub vaults: Vec<String>,
    pub size: u64,
}

pub fn vault_size(v: &Path) -> u64 {
    std::fs::metadata(v.join("vault.bin")).map(|m| m.len()).unwrap_or(0)
}

pub fn folder_summary(p: &Path) -> FolderSummary {
    if !p.exists() {
        return FolderSummary::default();
    }
    let mut vs: Vec<PathBuf> = std::fs::read_dir(p).map(|r| r.flatten().map(|e| e.path()).filter(|d| is_vault(d)).collect()).unwrap_or_default();
    vs.sort();
    FolderSummary { exists: true, free: free_bytes(p), size: vs.iter().map(|v| vault_size(v)).sum(), vaults: vs.iter().map(|v| v.file_name().unwrap().to_string_lossy().into_owned()).collect() }
}

fn cache_path(p: &Paths) -> PathBuf {
    p.data.join("drive_info.json")
}

/// (media, bus, model) of the physical disk holding `path`, e.g. ("SSD", "NVMe", "Samsung ..."); cached per drive.
pub fn drive_kind(p: &Paths, path: &Path) -> (String, String, String) {
    let d = drive_of(path).trim_end_matches(':').to_string();
    let mut c: J = std::fs::read(cache_path(p)).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or(json!({}));
    if let Some(a) = c.get(&d).and_then(|x| x.as_array()) {
        let g = |i: usize| a.get(i).and_then(|x| x.as_str()).unwrap_or("?").to_string();
        return (g(0), g(1), g(2));
    }
    let (media, bus, model) = if cfg!(windows) && d.len() == 1 {
        let ps = format!("$d = Get-Partition -DriveLetter {d} -ErrorAction Stop | Get-Disk; $p = Get-PhysicalDisk | Where-Object DeviceId -eq $d.Number; \"$($p.MediaType)|$($p.BusType)|$($d.FriendlyName)\"");
        let mut cmd = std::process::Command::new("powershell");
        cmd.args(["-NoProfile", "-Command", &ps]);
        crate::tools::external::no_window(&mut cmd);
        match cmd.output() {
            Ok(o) => {
                let s = String::from_utf8_lossy(&o.stdout);
                let l = s.trim().lines().last().unwrap_or("").to_string();
                let mut f: Vec<String> = l.split('|').map(str::to_string).collect();
                f.resize(3, String::new());
                let media = if f[0].is_empty() || f[0] == "Unspecified" { "?".to_string() } else { f[0].clone() };
                (media, if f[1].is_empty() { "?".into() } else { f[1].clone() }, if f[2].is_empty() { "?".into() } else { f[2].clone() })
            }
            Err(_) => ("?".into(), "?".into(), "?".into()),
        }
    } else {
        let disks = sysinfo::Disks::new_with_refreshed_list();
        let kind = disks
            .list()
            .iter()
            .filter(|x| path.starts_with(x.mount_point()))
            .max_by_key(|x| x.mount_point().as_os_str().len())
            .map(|x| match x.kind() {
                sysinfo::DiskKind::SSD => "SSD".to_string(),
                sysinfo::DiskKind::HDD => "HDD".to_string(),
                _ => "?".to_string(),
            })
            .unwrap_or_else(|| "?".into());
        (kind, "?".into(), "?".into())
    };
    c[&d] = json!([media, bus, model]);
    let _ = std::fs::write(cache_path(p), serde_json::to_string_pretty(&c).unwrap());
    (media, bus, model)
}

fn profile(p: &Paths) -> J {
    std::fs::read(p.data.join("hw_profile.json")).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or(json!({}))
}

/// MB/s of the drive holding `folder`: the stored measurement, or (measure) read a sample of its largest vault.
pub fn read_speed(p: &Paths, folder: &Path, measure: bool) -> Option<f64> {
    let d = drive_of(folder);
    let mut prof = profile(p);
    let known = prof["disk_MBps"][&d].as_f64();
    if !measure {
        return known;
    }
    let mut vs: Vec<PathBuf> = std::fs::read_dir(folder).map(|r| r.flatten().map(|e| e.path()).filter(|x| is_vault(x)).collect()).unwrap_or_default();
    vs.sort_by_key(|v| std::cmp::Reverse(vault_size(v)));
    let v = vs.first()?;
    let mbps = measure_file(&v.join("vault.bin"), 1 << 30).ok()?;
    if !prof["disk_MBps"].is_object() {
        prof["disk_MBps"] = json!({});
    }
    prof["disk_MBps"][&d] = json!(mbps.round());
    let _ = std::fs::write(p.data.join("hw_profile.json"), serde_json::to_string_pretty(&prof).unwrap());
    Some(mbps)
}

/// Sequential read speed of a file (up to `max` bytes, 8 MB pieces, bypassing the OS cache where possible).
pub fn measure_file(f: &Path, max: u64) -> std::io::Result<f64> {
    let len = std::fs::metadata(f)?.len();
    let n = len.min(max) & !(4095);
    if n == 0 {
        return Ok(0.0);
    }
    let file = open_direct(f)?;
    const CH: usize = 8 << 20;
    // 4096-aligned buffer for unbuffered reads
    let mut raw = vec![0u8; CH + 4096];
    let a = raw.as_ptr().align_offset(4096);
    let buf = &mut raw[a..a + CH];
    // start past the first quarter, where an earlier run is least likely to have left pages cached
    let start = ((len - n) / 2) & !4095;
    let t = Instant::now();
    let mut done = 0u64;
    while done < n {
        let k = (CH as u64).min(n - done) as usize;
        crate::storage::pread_exact(&file, &mut buf[..k], start + done)?;
        done += k as u64;
    }
    Ok(n as f64 / 1e6 / t.elapsed().as_secs_f64().max(1e-6))
}

#[cfg(windows)]
fn open_direct(f: &Path) -> std::io::Result<std::fs::File> {
    use std::os::windows::fs::OpenOptionsExt;
    std::fs::OpenOptions::new().read(true).custom_flags(0x2000_0000 /* FILE_FLAG_NO_BUFFERING */).open(f)
}
#[cfg(target_os = "linux")]
fn open_direct(f: &Path) -> std::io::Result<std::fs::File> {
    use std::os::unix::fs::OpenOptionsExt;
    std::fs::OpenOptions::new().read(true).custom_flags(0o40000 /* O_DIRECT */).open(f).or_else(|_| std::fs::File::open(f))
}
#[cfg(not(any(windows, target_os = "linux")))]
fn open_direct(f: &Path) -> std::io::Result<std::fs::File> {
    std::fs::File::open(f)
}

#[cfg(unix)]
pub fn pread_exact(f: &std::fs::File, buf: &mut [u8], off: u64) -> std::io::Result<()> {
    use std::os::unix::fs::FileExt;
    f.read_exact_at(buf, off)
}
#[cfg(windows)]
pub fn pread_exact(f: &std::fs::File, mut buf: &mut [u8], mut off: u64) -> std::io::Result<()> {
    use std::os::windows::fs::FileExt;
    while !buf.is_empty() {
        let k = f.seek_read(buf, off)?;
        if k == 0 {
            return Err(std::io::ErrorKind::UnexpectedEof.into());
        }
        off += k as u64;
        buf = &mut buf[k..];
    }
    Ok(())
}

/// Every stream checksum of a vault (the engine's `check`): (streams, bad).
pub fn check_vault(v: &Path, log: &mut dyn FnMut(String)) -> anyhow::Result<(usize, usize)> {
    let vault = cube_vault::Vault::open(v)?;
    let n = vault.n_streams();
    let mut bad = 0;
    let t = Instant::now();
    for i in 0..n {
        if vault.read_stream_checked(i).is_err() {
            bad += 1;
            log(format!("  stream {i}: checksum mismatch"));
        }
        if t.elapsed().as_secs() > 0 && i % 64 == 0 {
            log(format!("  checked {}/{n}", i + 1));
        }
    }
    log(format!("vault {}: streams={n} bad={bad}", if bad == 0 { "OK" } else { "BAD" }));
    Ok((n, bad))
}

fn copy_dir(src: &Path, dst: &Path, log: &mut dyn FnMut(String)) -> std::io::Result<u64> {
    std::fs::create_dir_all(dst)?;
    let mut n = 0;
    for e in std::fs::read_dir(src)?.flatten() {
        let (s, d) = (e.path(), dst.join(e.file_name()));
        if s.is_dir() {
            n += copy_dir(&s, &d, log)?;
        } else {
            let sz = std::fs::metadata(&s)?.len();
            if sz > 1 << 30 {
                log(format!("  copying {} ({:.1} GB)", e.file_name().to_string_lossy(), sz as f64 / 1e9));
            }
            n += std::fs::copy(&s, &d)?;
        }
    }
    Ok(n)
}

/// Copy a vault to another model folder, verify every stream of the copy, then delete the original.
pub fn move_vault(src: &Path, dst_dir: &Path, log: &mut dyn FnMut(String)) -> anyhow::Result<PathBuf> {
    let dst = dst_dir.join(src.file_name().unwrap());
    let t0 = Instant::now();
    anyhow::ensure!(!dst.exists(), "{} already exists", dst.display());
    fn size(d: &Path) -> u64 {
        std::fs::read_dir(d).map(|r| r.flatten().map(|e| if e.path().is_dir() { size(&e.path()) } else { e.metadata().map(|m| m.len()).unwrap_or(0) }).sum()).unwrap_or(0)
    }
    let need = size(src);
    std::fs::create_dir_all(dst_dir)?;
    anyhow::ensure!(free_bytes(dst_dir) >= need + 2_000_000_000, "not enough space in {}: needs {:.1} GB + 2 GB margin", dst_dir.display(), need as f64 / 1e9);
    log(format!("copying {} ({:.1} GB) to {} ...", src.file_name().unwrap().to_string_lossy(), need as f64 / 1e9, dst_dir.display()));
    copy_dir(src, &dst, log)?;
    log(format!("copied in {:.1} min - verifying every stream of the copy ...", t0.elapsed().as_secs_f64() / 60.0));
    let (_, bad) = check_vault(&dst, log)?;
    anyhow::ensure!(bad == 0, "verification failed - the original was kept, the copy left for inspection");
    std::fs::remove_dir_all(src)?;
    log(format!("moved: {}  ({:.1} min)", dst.display(), t0.elapsed().as_secs_f64() / 60.0));
    Ok(dst)
}
