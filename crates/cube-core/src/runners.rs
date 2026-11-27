//! Artifact types and their runners.
//!
//! A type is one JSON file in `data/artifact_types/` (the harness writes its built-in ones there and updates them while
//! they keep `"managed": "cube-harness"`; edit one, or add your own, and it is yours). A type names its main file,
//! starter files, how to tell its toolchain is installed (`detect`) and its commands (`build`, `run`, `test`, ...).
//!
//! Commands run in one of two places:
//! * **this PC**: `cmd /C` in the artifact's folder, with the toolchains you have installed;
//! * **the sandbox**: the WSL distro `cube-sandbox` (sandbox_setup.bat), which has no Windows drives mounted and cannot
//!   start Windows programs. The artifact's files are copied in, the command runs there, and new or changed files are
//!   copied back. Code in there cannot read or change anything on Windows outside what it was given.
//!
//! Every run has a time limit, can be stopped, and its output is capped.

use serde_json::{json, Value as J};
use std::collections::HashMap;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant, SystemTime};

pub const SANDBOX: &str = "cube-sandbox";
const BUILTIN: &str = include_str!("artifact_types.json");
/// folders that are build output or caches: not copied into the sandbox, not reported as changed
const SKIP_DIRS: [&str; 8] = [".versions", "node_modules", "target", ".git", "__pycache__", "obj", ".pytest_cache", ".venv"];
const MAX_OUTPUT: usize = 2 << 20;
const MAX_COPY_BACK: u64 = 50 << 20;

#[derive(Clone, Debug, Default)]
pub struct Kind {
    pub name: String,
    pub title: String,
    pub description: String,
    pub main: String,
    pub starters: Vec<(String, String)>,
    pub detect: Vec<String>,
    pub detect_sandbox: Vec<String>,
    pub commands: Vec<(String, String)>,
    pub sandbox_commands: Vec<(String, String)>,
    pub linux_only: bool,
    pub windows_only: bool,
    pub open: String,
    pub install: String,
    pub timeout_s: u64,
    order: i64,
}

fn pairs(v: &J) -> Vec<(String, String)> {
    match v {
        J::Array(a) => a.iter().filter_map(|p| Some((p.get(0)?.as_str()?.to_string(), p.get(1)?.as_str()?.to_string()))).collect(),
        J::Object(o) => o.iter().filter_map(|(k, v)| Some((k.clone(), v.as_str()?.to_string()))).collect(),
        _ => Vec::new(),
    }
}

fn strs(v: &J) -> Vec<String> {
    v.as_array().map(|a| a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect()).unwrap_or_default()
}

impl Kind {
    pub fn from_json(j: &J) -> Option<Kind> {
        let s = |k: &str| j.get(k).and_then(|v| v.as_str()).unwrap_or("").to_string();
        let name = s("name");
        if name.is_empty() || !name.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-' || c == '_') {
            return None;
        }
        Some(Kind {
            title: if s("title").is_empty() { name.clone() } else { s("title") },
            description: s("description"),
            main: if s("main").is_empty() { "README.md".into() } else { s("main") },
            starters: pairs(&j["starters"]),
            detect: strs(&j["detect"]),
            detect_sandbox: strs(&j["detect_sandbox"]),
            commands: pairs(&j["commands"]),
            sandbox_commands: pairs(&j["sandbox_commands"]),
            linux_only: j["linux_only"].as_bool().unwrap_or(false),
            windows_only: j["windows_only"].as_bool().unwrap_or(false),
            open: s("open"),
            install: s("install"),
            timeout_s: j["timeout_s"].as_u64().unwrap_or(300).clamp(5, 3600),
            order: j["order"].as_i64().unwrap_or(100),
            name,
        })
    }

    /// command names available where the code would run (sandbox: its own commands, else the Windows ones)
    pub fn command_names(&self, sandbox: bool) -> Vec<String> {
        self.commands_for(sandbox).iter().map(|c| c.0.clone()).collect()
    }

    pub fn commands_for(&self, sandbox: bool) -> &[(String, String)] {
        if sandbox && !self.sandbox_commands.is_empty() {
            &self.sandbox_commands
        } else {
            &self.commands
        }
    }

    /// where this type's code runs, given the user's preference
    pub fn in_sandbox(&self, prefer_sandbox: bool) -> bool {
        if self.linux_only {
            true
        } else if self.windows_only {
            false
        } else {
            prefer_sandbox
        }
    }

    pub fn runnable(&self) -> bool {
        !self.commands.is_empty() || !self.sandbox_commands.is_empty()
    }
}

/// Write the built-in types into `dir` (new ones, and updates to ones still marked managed).
pub fn seed(dir: &Path) {
    let _ = std::fs::create_dir_all(dir);
    let Ok(J::Array(all)) = serde_json::from_str::<J>(BUILTIN) else { return };
    for mut t in all {
        let Some(name) = t["name"].as_str().map(str::to_string) else { continue };
        t["managed"] = json!("cube-harness");
        let body = serde_json::to_string_pretty(&t).unwrap() + "\n";
        let f = dir.join(format!("{name}.json"));
        let replace = match std::fs::read_to_string(&f) {
            Err(_) => true,
            Ok(old) => old != body && serde_json::from_str::<J>(&old).map_or(false, |o| o["managed"] == "cube-harness"),
        };
        if replace {
            let _ = std::fs::write(&f, body);
        }
    }
}

/// The types in `dir` (falls back to the built-in list), in display order.
pub fn load(dir: &Path) -> Vec<Kind> {
    let mut v: Vec<Kind> = Vec::new();
    if let Ok(r) = std::fs::read_dir(dir) {
        for e in r.flatten() {
            if e.path().extension().map_or(false, |x| x == "json") {
                if let Some(k) = std::fs::read_to_string(e.path()).ok().and_then(|s| serde_json::from_str::<J>(&s).ok()).and_then(|j| Kind::from_json(&j)) {
                    if !v.iter().any(|x| x.name == k.name) {
                        v.push(k);
                    }
                }
            }
        }
    }
    if v.is_empty() {
        if let Ok(J::Array(all)) = serde_json::from_str::<J>(BUILTIN) {
            v = all.iter().filter_map(Kind::from_json).collect();
        }
    }
    v.sort_by(|a, b| a.order.cmp(&b.order).then(a.name.cmp(&b.name)));
    v
}

// ------------------------------------------------------------------------------------------------ processes

#[cfg(windows)]
fn hide(c: &mut Command) {
    use std::os::windows::process::CommandExt;
    c.creation_flags(0x0800_0000); // CREATE_NO_WINDOW: the harness is a window app; no console pops up
}
#[cfg(not(windows))]
fn hide(_: &mut Command) {}

/// own process group on Unix, so the whole tree can be killed (a shell's children keep the output pipes open)
#[cfg(unix)]
fn group(c: &mut Command) {
    use std::os::unix::process::CommandExt;
    c.process_group(0);
}
#[cfg(not(unix))]
fn group(_: &mut Command) {}

fn kill_tree(ch: &mut Child) {
    #[cfg(windows)]
    {
        let mut k = Command::new("taskkill");
        k.args(["/T", "/F", "/PID", &ch.id().to_string()]).stdout(Stdio::null()).stderr(Stdio::null());
        hide(&mut k);
        let _ = k.status();
    }
    #[cfg(unix)]
    {
        let _ = Command::new("kill").args(["-KILL", "--", &format!("-{}", ch.id())]).stdout(Stdio::null()).stderr(Stdio::null()).status();
    }
    let _ = ch.kill();
}

/// Run to completion (or timeout / stop): (exit code, combined stdout+stderr, timed out, stopped).
fn drive(mut c: Command, stdin: Option<Vec<u8>>, timeout: Duration, stop: &AtomicBool, live: Option<&Arc<Mutex<String>>>) -> std::io::Result<(Option<i32>, Vec<u8>, bool, bool)> {
    hide(&mut c);
    group(&mut c);
    c.stdin(if stdin.is_some() { Stdio::piped() } else { Stdio::null() }).stdout(Stdio::piped()).stderr(Stdio::piped());
    let mut ch = c.spawn()?;
    if let (Some(data), Some(mut si)) = (stdin, ch.stdin.take()) {
        std::thread::spawn(move || {
            let _ = si.write_all(&data);
        });
    }
    let buf = Arc::new(Mutex::new(Vec::<u8>::new()));
    let mut readers = Vec::new();
    for src in [ch.stdout.take().map(|x| Box::new(x) as Box<dyn Read + Send>), ch.stderr.take().map(|x| Box::new(x) as Box<dyn Read + Send>)].into_iter().flatten() {
        let (b, l) = (buf.clone(), live.cloned());
        readers.push(std::thread::spawn(move || {
            let mut src = src;
            let mut chunk = [0u8; 8192];
            while let Ok(n) = src.read(&mut chunk) {
                if n == 0 {
                    break;
                }
                let mut g = b.lock().unwrap();
                if g.len() < MAX_OUTPUT {
                    g.extend_from_slice(&chunk[..n]);
                }
                if let Some(l) = &l {
                    let mut s = l.lock().unwrap();
                    if s.len() < MAX_OUTPUT {
                        s.push_str(&String::from_utf8_lossy(&chunk[..n]));
                    }
                }
            }
        }));
    }
    let t0 = Instant::now();
    let (mut timed_out, mut stopped) = (false, false);
    let code = loop {
        if let Some(st) = ch.try_wait()? {
            break st.code();
        }
        if stop.load(Ordering::Relaxed) {
            stopped = true;
        } else if t0.elapsed() > timeout {
            timed_out = true;
        }
        if stopped || timed_out {
            kill_tree(&mut ch);
            let _ = ch.wait();
            break None;
        }
        std::thread::sleep(Duration::from_millis(40));
    };
    // a leftover grandchild can hold the pipes open: give the readers a moment, then go on without them
    let t1 = Instant::now();
    while readers.iter().any(|r| !r.is_finished()) && t1.elapsed() < Duration::from_secs(3) {
        std::thread::sleep(Duration::from_millis(20));
    }
    let out = std::mem::take(&mut *buf.lock().unwrap());
    Ok((code, out, timed_out, stopped))
}

/// `wsl -d cube-sandbox -u runner --exec bash -c <script>`; CUBE_SANDBOX_LOCAL=<dir> runs the same script with plain
/// bash and HOME=<dir> instead (tests on Linux).
fn sandbox_cmd(script: &str) -> Command {
    use base64::Engine;
    let b64 = base64::engine::general_purpose::STANDARD.encode(script);
    // only [A-Za-z0-9+/=|;>-/ ] reach the command line: nothing for any quoting layer to mangle
    let line = format!("echo {b64}|base64 -d>/tmp/cube_run.sh;bash /tmp/cube_run.sh");
    if let Ok(home) = std::env::var("CUBE_SANDBOX_LOCAL") {
        let mut c = Command::new("bash");
        c.arg("-c").arg(line.replace("/tmp/cube_run.sh", &format!("{home}/.cube_run.sh"))).env("HOME", home);
        return c;
    }
    let mut c = Command::new("wsl.exe");
    c.args(["-d", SANDBOX, "-u", "runner", "--exec", "bash", "-c", &line]);
    c
}

/// Is the sandbox there?
pub fn sandbox_ready() -> bool {
    static C: OnceLock<Mutex<Option<(Instant, bool)>>> = OnceLock::new();
    let m = C.get_or_init(|| Mutex::new(None));
    if let Some((t, ok)) = *m.lock().unwrap() {
        if t.elapsed() < Duration::from_secs(30) {
            return ok;
        }
    }
    let stop = AtomicBool::new(false);
    let ok = cfg!(windows) || std::env::var("CUBE_SANDBOX_LOCAL").is_ok();
    let ok = ok && matches!(drive(sandbox_cmd("echo cube-sandbox-ok"), None, Duration::from_secs(60), &stop, None), Ok((Some(0), ref o, _, _)) if String::from_utf8_lossy(o).contains("cube-sandbox-ok"));
    *m.lock().unwrap() = Some((Instant::now(), ok));
    ok
}

/// Ok(version line) when the type's toolchain is there, Err(what to do) when not.
pub fn available(k: &Kind, sandbox: bool) -> Result<String, String> {
    static C: OnceLock<Mutex<HashMap<(String, bool), (Instant, Result<String, String>)>>> = OnceLock::new();
    let m = C.get_or_init(|| Mutex::new(HashMap::new()));
    let key = (k.name.clone(), sandbox);
    if let Some((t, r)) = m.lock().unwrap().get(&key) {
        if t.elapsed() < Duration::from_secs(120) {
            return r.clone();
        }
    }
    let r = probe(k, sandbox);
    m.lock().unwrap().insert(key, (Instant::now(), r.clone()));
    r
}

fn probe(k: &Kind, sandbox: bool) -> Result<String, String> {
    if !k.runnable() {
        return Err("nothing to run for this type".into());
    }
    if sandbox && k.windows_only {
        return Err(format!("{} runs on this PC only (not in the sandbox)", k.title));
    }
    if !sandbox && k.linux_only {
        return Err(format!("{} runs in the Linux sandbox only - set it up with sandbox_setup.bat", k.title));
    }
    let stop = AtomicBool::new(false);
    let line = |o: &[u8]| String::from_utf8_lossy(o).lines().map(str::trim).find(|l| !l.is_empty()).unwrap_or("").chars().take(120).collect::<String>();
    if sandbox {
        if !sandbox_ready() {
            return Err("the sandbox is not set up - run sandbox_setup.bat".into());
        }
        let d = if k.detect_sandbox.is_empty() { &k.detect } else { &k.detect_sandbox };
        if d.is_empty() {
            return Ok("ready".into());
        }
        let q: Vec<String> = d.iter().map(|a| format!("'{}'", a.replace('\'', "'\\''"))).collect();
        return match drive(sandbox_cmd(&format!("{} 2>&1", q.join(" "))), None, Duration::from_secs(60), &stop, None) {
            Ok((Some(0), o, _, _)) => Ok(line(&o)),
            _ => Err(format!("'{}' is not installed in the sandbox (wsl -d {SANDBOX} -u root, then apt install it)", d[0])),
        };
    }
    if k.detect.is_empty() {
        return Ok("ready".into());
    }
    // through the shell, as the commands run: npm-installed tools are .cmd files that only cmd finds
    match run_here(Path::new("."), &k.detect.join(" "), Duration::from_secs(30), &stop, None) {
        Ok((Some(0), o, _, _)) => Ok(line(&o)),
        Ok((Some(9009 | 127), _, _, _)) => Err(format!("{} is not installed{}", k.detect[0], if k.install.is_empty() { String::new() } else { format!(" - install: {}", k.install) })),
        Ok((code, o, _, _)) => Err(format!("'{}' did not run ({}){}", k.detect.join(" "), code.map_or("no exit code".into(), |c| format!("exit {c}")), {
            let l = line(&o);
            if l.is_empty() { String::new() } else { format!(": {l}") }
        })),
        Err(e) => Err(format!("{} could not be started: {e}", k.detect[0])),
    }
}

// ------------------------------------------------------------------------------------------------ running

#[derive(Clone, Debug, Default)]
pub struct RunResult {
    pub command: String,
    pub name: String,
    pub sandbox: bool,
    pub exit: Option<i32>,
    pub secs: f64,
    pub output: String,
    pub changed: Vec<String>,
    pub timed_out: bool,
    pub stopped: bool,
    pub error: String,
}

impl RunResult {
    pub fn ok(&self) -> bool {
        self.error.is_empty() && self.exit == Some(0)
    }

    pub fn status(&self) -> String {
        if !self.error.is_empty() {
            return format!("could not run: {}", self.error);
        }
        let how = if self.timed_out {
            "stopped at the time limit".to_string()
        } else if self.stopped {
            "stopped".to_string()
        } else {
            match self.exit {
                Some(0) => "exit 0 (ok)".into(),
                Some(c) => format!("exit {c} (failed)"),
                None => "ended without an exit code".into(),
            }
        };
        format!("{} '{}' {} in {:.1} s, {}", self.name, self.command, if self.sandbox { "in the sandbox" } else { "on this PC" }, self.secs, how)
    }

    /// what the model gets back: status, output (head and tail when long), files the run created or changed
    pub fn report(&self, max: usize) -> String {
        let mut out = self.output.trim_end().to_string();
        let n = out.chars().count();
        if n > max {
            let head: String = out.chars().take(max / 4).collect();
            let tail: String = out.chars().skip(n - max * 3 / 4).collect();
            out = format!("{head}\n[... {} characters cut ...]\n{tail}", n - max);
        }
        let mut s = format!("{}\n--- output ---\n{}\n--- end of output ---", self.status(), if out.is_empty() { "(no output)" } else { &out });
        if !self.changed.is_empty() {
            s += &format!("\nfiles created or changed: {}", self.changed.join(", "));
        }
        s
    }
}

fn safe_path(p: &str) -> bool {
    !p.is_empty() && !p.starts_with('/') && !p.split('/').any(|x| x == ".." || x.is_empty()) && p.chars().all(|c| c.is_ascii_alphanumeric() || "._-/".contains(c))
}

/// Fill {main} {stem} {name} {crate} {title} into a command; the file names must be plain (no shell characters).
pub fn fill(cmd: &str, main: &str, name: &str) -> Result<String, String> {
    if !safe_path(main) {
        return Err(format!("the main file '{main}' has characters that cannot go into a command (use letters, digits, . _ - /)"));
    }
    if !safe_path(name) {
        return Err(format!("the artifact name '{name}' cannot go into a command"));
    }
    let stem = Path::new(main).file_stem().map(|x| x.to_string_lossy().into_owned()).unwrap_or_default();
    Ok(cmd.replace("{main}", main).replace("{stem}", &stem).replace("{name}", name).replace("{crate}", &name.replace('-', "_")))
}

/// {files:.c} -> the artifact's .c files (sorted, '/' paths): no shell globbing needed (cmd.exe has none)
pub fn fill_files(cmd: &str, dir: &Path) -> Result<String, String> {
    let mut out = cmd.to_string();
    while let Some(i) = out.find("{files:") {
        let Some(j) = out[i..].find('}') else { break };
        let ext = out[i + 7..i + j].to_string();
        let mut fs: Vec<String> = snapshot(dir).into_keys().filter(|f| f.ends_with(&ext)).collect();
        fs.sort();
        if let Some(bad) = fs.iter().find(|f| !safe_path(f)) {
            return Err(format!("the file name '{bad}' has characters that cannot go into a command (use letters, digits, . _ - /)"));
        }
        if fs.is_empty() {
            return Err(format!("no {ext} files in the artifact"));
        }
        out = format!("{}{}{}", &out[..i], fs.join(" "), &out[i + j + 1..]);
    }
    Ok(out)
}

type Snap = HashMap<String, (u64, SystemTime)>;

fn skip_dir(rel: &str) -> bool {
    rel.split('/').next().map_or(false, |top| SKIP_DIRS.contains(&top))
}

fn snapshot(dir: &Path) -> Snap {
    fn walk(b: &Path, d: &Path, m: &mut Snap) {
        if let Ok(r) = std::fs::read_dir(d) {
            for e in r.flatten() {
                let p = e.path();
                let rel = p.strip_prefix(b).unwrap().to_string_lossy().replace('\\', "/");
                if skip_dir(&rel) || rel == ".artifact.json" {
                    continue;
                }
                if p.is_dir() {
                    walk(b, &p, m);
                } else if let Ok(md) = e.metadata() {
                    m.insert(rel, (md.len(), md.modified().unwrap_or(SystemTime::UNIX_EPOCH)));
                }
            }
        }
    }
    let mut m = HashMap::new();
    walk(dir, dir, &mut m);
    m
}

fn changed(before: &Snap, after: &Snap) -> Vec<String> {
    let mut v: Vec<String> = after.iter().filter(|(k, v)| before.get(*k) != Some(*v)).map(|(k, _)| k.clone()).collect();
    v.sort();
    v.truncate(40);
    v
}

fn find_vcvars() -> Option<PathBuf> {
    let roots = [r"C:\Program Files (x86)\Microsoft Visual Studio", r"C:\Program Files\Microsoft Visual Studio"];
    for r in roots {
        let Ok(vers) = std::fs::read_dir(r) else { continue };
        let mut vers: Vec<PathBuf> = vers.flatten().map(|e| e.path()).collect();
        vers.sort();
        for v in vers.iter().rev() {
            if let Ok(eds) = std::fs::read_dir(v) {
                for e in eds.flatten() {
                    let p = e.path().join(r"VC\Auxiliary\Build\vcvars64.bat");
                    if p.exists() {
                        return Some(p);
                    }
                }
            }
        }
    }
    None
}

/// Run command `cmd_name` of type `k` for the artifact in `dir`. `live` receives the output as it comes.
pub fn run(k: &Kind, dir: &Path, artifact: &str, main: &str, cmd_name: &str, sandbox: bool, stop: &AtomicBool, live: Option<&Arc<Mutex<String>>>) -> RunResult {
    let mut r = RunResult { name: artifact.into(), sandbox, ..Default::default() };
    let cmds = k.commands_for(sandbox);
    let Some((_, raw)) = cmds.iter().find(|c| c.0 == cmd_name) else {
        r.command = cmd_name.into();
        r.error = format!("type '{}' has no '{cmd_name}' command{} (it has: {})", k.name, if sandbox { " in the sandbox" } else { "" }, k.command_names(sandbox).join(", "));
        return r;
    };
    let cmd = match fill(raw, main, artifact).and_then(|c| fill_files(&c, dir)) {
        Ok(c) => c,
        Err(e) => {
            r.command = raw.clone();
            r.error = e;
            return r;
        }
    };
    r.command = cmd.clone();
    if let Err(e) = available(k, sandbox) {
        r.error = e;
        return r;
    }
    stop.store(false, Ordering::Relaxed);
    let t0 = Instant::now();
    let timeout = Duration::from_secs(k.timeout_s);
    let before = snapshot(dir);
    let res = if sandbox { run_sandbox(dir, &cmd, timeout, stop, live, &mut r) } else { run_here(dir, &cmd, timeout, stop, live) };
    r.secs = t0.elapsed().as_secs_f64();
    match res {
        Ok((code, out, to, st)) => {
            r.exit = code;
            r.output = String::from_utf8_lossy(&out).replace("\r\n", "\n");
            r.timed_out = to;
            r.stopped = st;
        }
        Err(e) => r.error = e,
    }
    r.changed = changed(&before, &snapshot(dir));
    r
}

fn run_here(dir: &Path, cmd: &str, timeout: Duration, stop: &AtomicBool, live: Option<&Arc<Mutex<String>>>) -> Result<(Option<i32>, Vec<u8>, bool, bool), String> {
    #[cfg(windows)]
    let mut c = {
        use std::os::windows::process::CommandExt;
        let mut c = Command::new("cmd");
        c.raw_arg(format!("/D /S /C \"{cmd}\""));
        if let Some(v) = find_vcvars() {
            c.env("VCVARS64", v);
        }
        c
    };
    #[cfg(not(windows))]
    let mut c = {
        let mut c = Command::new("sh");
        c.arg("-c").arg(cmd);
        let _ = find_vcvars;
        c
    };
    c.current_dir(dir).env("PYTHONIOENCODING", "utf-8").env("PYTHONUNBUFFERED", "1").env("NO_COLOR", "1").env("CARGO_TERM_COLOR", "never");
    drive(c, None, timeout, stop, live).map_err(|e| e.to_string())
}

/// Copy the artifact into the sandbox (tar over stdin), run there, copy new / changed files back.
fn run_sandbox(dir: &Path, cmd: &str, timeout: Duration, stop: &AtomicBool, live: Option<&Arc<Mutex<String>>>, r: &mut RunResult) -> Result<(Option<i32>, Vec<u8>, bool, bool), String> {
    let mut tb = tar::Builder::new(Vec::new());
    for (rel, _) in snapshot(dir) {
        tb.append_path_with_name(dir.join(&rel), &rel).map_err(|e| format!("packing {rel}: {e}"))?;
    }
    let data = tb.into_inner().map_err(|e| e.to_string())?;
    let secs = timeout.as_secs();
    let script = format!(
        "set -e\nrm -rf \"$HOME/work\"; mkdir -p \"$HOME/work\"; cd \"$HOME/work\"; tar -x\nset +e\ncat > \"$HOME/.cube_cmd.sh\" <<'CUBE_CMD_EOF'\n{cmd}\nCUBE_CMD_EOF\ntimeout -k 5 {secs} bash \"$HOME/.cube_cmd.sh\" </dev/null 2>&1\n"
    );
    let res = drive(sandbox_cmd(&script), Some(data), timeout + Duration::from_secs(30), stop, live).map_err(|e| e.to_string())?;
    if (res.2 || res.3) && std::env::var("CUBE_SANDBOX_LOCAL").is_err() {
        // wsl.exe is gone; make sure nothing of the run keeps going inside (the sandbox user runs nothing else)
        let _ = drive(sandbox_cmd("pkill -KILL -f cube_cmd.sh; true"), None, Duration::from_secs(20), &AtomicBool::new(false), None);
    }
    let excl: String = SKIP_DIRS.iter().map(|d| format!(" --exclude=./{d}")).collect();
    let back = drive(sandbox_cmd(&format!("cd \"$HOME/work\" && tar -c{excl} .")), None, Duration::from_secs(120), &AtomicBool::new(false), None).map_err(|e| e.to_string())?;
    if back.0 == Some(0) {
        let mut ar = tar::Archive::new(&back.1[..]);
        if let Ok(entries) = ar.entries() {
            for mut e in entries.flatten() {
                if !e.header().entry_type().is_file() || e.size() > MAX_COPY_BACK {
                    continue;
                }
                let Ok(p) = e.path() else { continue };
                let rel = p.to_string_lossy().trim_start_matches("./").replace('\\', "/");
                if !safe_rel(&rel) || skip_dir(&rel) || rel == ".artifact.json" {
                    continue;
                }
                let mut bytes = Vec::new();
                if e.read_to_end(&mut bytes).is_err() {
                    continue;
                }
                let dst = dir.join(&rel);
                if std::fs::read(&dst).ok().as_deref() != Some(&bytes[..]) {
                    if let Some(d) = dst.parent() {
                        let _ = std::fs::create_dir_all(d);
                    }
                    let _ = std::fs::write(&dst, &bytes);
                }
            }
        }
    }
    let _ = r;
    Ok(res)
}

/// a relative path from the sandbox that stays inside the artifact
fn safe_rel(p: &str) -> bool {
    !p.is_empty() && !p.starts_with('/') && !p.contains(':') && !p.split('/').any(|x| x == "..")
}

#[cfg(test)]
mod tests {
    use crate::tools::artifact::{Artifacts, RunPolicy};
    use serde_json::json;
    use std::sync::Arc;

    fn arts(tag: &str) -> (Artifacts, std::path::PathBuf) {
        let base = std::env::temp_dir().join(format!("cube_run_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(base.join("artifacts")).unwrap();
        super::seed(&base.join("artifact_types"));
        (Artifacts::new(base.join("artifacts")), base)
    }

    #[test]
    fn policy_off_ask_auto_and_native_runs() {
        let (a, base) = arts("pol");
        let r = a.act(&json!({"action": "create", "name": "hello-py", "type": "python"}));
        assert!(r.contains("created artifact 'hello-py' (python)") && r.contains("Commands: run, test"), "{r}");
        // off: refused
        let r = a.act(&json!({"action": "run", "name": "hello-py"}));
        assert!(r.contains("turned off"), "{r}");
        // ask: denied, then allowed
        *a.policy.lock().unwrap() = RunPolicy { model: "ask".into(), sandbox: false };
        let r = a.act(&json!({"action": "run", "name": "hello-py"}));
        assert!(r.contains("no one can approve"), "{r}");
        *a.approver.lock().unwrap() = Some(Arc::new(|q: &str| !q.contains("hello-py")));
        let r = a.act(&json!({"action": "run", "name": "hello-py"}));
        assert!(r.contains("did not allow"), "{r}");
        *a.approver.lock().unwrap() = Some(Arc::new(|_: &str| true));
        // python on this box is python3; the type says python: point it there for the test
        if std::process::Command::new("python").arg("--version").output().is_ok() {
            let r = a.act(&json!({"action": "run", "name": "hello-py"}));
            assert!(r.contains("exit 0 (ok)") && r.contains("hello from hello-py"), "{r}");
        }
        // auto + a failing C program: compile error comes back to the model
        *a.policy.lock().unwrap() = RunPolicy { model: "auto".into(), sandbox: false };
        a.act(&json!({"action": "create", "name": "bad-c", "type": "c", "content": "int main(void) { return x; }\n"}));
        let r = a.act(&json!({"action": "run", "name": "bad-c"}));
        assert!(r.contains("failed") && r.contains("'x' undeclared"), "{r}");
        a.act(&json!({"action": "create", "name": "good-c", "type": "c"}));
        // the Windows command ends in 'app.exe', which cmd finds in the folder and sh does not: build here, run there
        let r = a.act(&json!({"action": "run", "name": "good-c", "command": if cfg!(windows) { "run" } else { "build" }}));
        assert!(r.contains("exit 0") && r.contains("app.exe") && (!cfg!(windows) || r.contains("hello from good-c")), "{r}");
        let r = a.act(&json!({"action": "run", "name": "good-c", "command": "deploy"}));
        assert!(r.contains("no 'deploy' command"), "{r}");
        let r = a.act(&json!({"action": "run", "name": "missing"}));
        assert!(r.contains("error") || r.contains("no artifact"), "{r}");
        let t = a.act(&json!({"action": "types"}));
        assert!(t.contains("python  main main.py") && t.contains("shell") && t.contains("[files only]"), "{t}");
        let _ = std::fs::remove_dir_all(base);
    }

    #[test]
    fn timeout_and_stop() {
        let (a, base) = arts("to");
        // cmd has no 'sleep'; ping waits ~1 s per echo request
        let slow = if cfg!(windows) { "ping -n 31 127.0.0.1 >nul" } else { "sleep 30" };
        let k = super::Kind { name: "slow".into(), commands: vec![("run".into(), slow.into())], timeout_s: 1, ..Default::default() };
        std::fs::write(a.types_dir.join("slow.json"), serde_json::to_string(&json!({"name": "slow", "main": "x.txt", "commands": [["run", slow]], "timeout_s": 5})).unwrap()).unwrap();
        a.act(&json!({"action": "create", "name": "s", "type": "slow", "content": "x"}));
        let d = a.root.join("s");
        let t0 = std::time::Instant::now();
        let mut k5 = k.clone();
        k5.timeout_s = 1;
        let r = super::run(&k5, &d, "s", "x.txt", "run", false, &std::sync::atomic::AtomicBool::new(false), None);
        assert!(r.timed_out && t0.elapsed().as_secs() < 10, "{r:?}");
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let s2 = stop.clone();
        std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(500));
            s2.store(true, std::sync::atomic::Ordering::Relaxed);
        });
        let k9 = super::Kind { timeout_s: 60, ..k };
        let t0 = std::time::Instant::now();
        // run() resets the flag at start, so stop from the side after it began
        let r = super::run(&k9, &d, "s", "x.txt", "run", false, &stop, None);
        assert!(r.stopped && t0.elapsed().as_secs() < 10, "{r:?}");
        let _ = std::fs::remove_dir_all(base);
    }

    /// the sandbox's copy-in / run / copy-back with plain bash standing in for WSL (Unix only: on Windows 'bash' is
    /// WSL's launcher; the real sandbox is checked by sandbox_setup.bat)
    #[cfg(unix)]
    #[test]
    fn sandbox_copy_in_run_copy_back() {
        let (a, base) = arts("sb");
        let home = base.join("sandbox_home");
        std::fs::create_dir_all(&home).unwrap();
        std::env::set_var("CUBE_SANDBOX_LOCAL", &home);
        *a.policy.lock().unwrap() = RunPolicy { model: "auto".into(), sandbox: true };
        a.act(&json!({"action": "create", "name": "sh-art", "type": "shell", "content": "echo made > out.txt; mkdir -p sub; echo deep > sub/d.txt; cat input.txt; pwd\n"}));
        a.act(&json!({"action": "write", "name": "sh-art", "path": "input.txt", "content": "from windows side"}));
        let r = a.act(&json!({"action": "run", "name": "sh-art"}));
        assert!(r.contains("in the sandbox") && r.contains("exit 0") && r.contains("from windows side"), "{r}");
        assert!(r.contains(&format!("{}/work", home.display())), "ran in the sandbox's own folder: {r}");
        let d = a.root.join("sh-art");
        assert_eq!(std::fs::read_to_string(d.join("out.txt")).unwrap().trim(), "made");
        assert_eq!(std::fs::read_to_string(d.join("sub/d.txt")).unwrap().trim(), "deep");
        assert!(r.contains("out.txt") && r.contains("sub/d.txt"), "{r}");
        std::env::remove_var("CUBE_SANDBOX_LOCAL");
        let _ = std::fs::remove_dir_all(base);
    }

    /// every built-in type whose toolchain is on this machine: create it from its starters and run its main command
    #[test]
    fn starters_build_and_run_where_installed() {
        let (a, base) = arts("all");
        *a.policy.lock().unwrap() = RunPolicy { model: "auto".into(), sandbox: false };
        let mut ran = Vec::new();
        for k in a.kinds() {
            if !k.runnable() || k.linux_only || super::available(&k, false).is_err() {
                continue;
            }
            let n = format!("t-{}", k.name);
            a.act(&json!({"action": "create", "name": n, "type": k.name}));
            // 'run' where there is one (Windows-only 'app.exe' runs are built instead off Windows)
            let cmd = if k.command_names(false).contains(&"run".to_string()) && !(cfg!(unix) && k.commands.iter().any(|c| c.1.contains("app.exe"))) { "run" } else { "build" };
            let r = a.act(&json!({"action": "run", "name": n, "command": cmd}));
            assert!(r.contains("exit 0"), "{}: {r}", k.name);
            if cmd == "run" {
                // every starter says "hello from <artifact>"; the CUDA one says it from each GPU thread
                let want = if k.name == "cuda" { "hello from thread 3".to_string() } else { format!("hello from {n}") };
                assert!(r.contains(&want), "{}: {r}", k.name);
            }
            ran.push(format!("{}:{cmd}", k.name));
        }
        eprintln!("ran: {}", ran.join(" "));
        let _ = std::fs::remove_dir_all(base);
    }
}
