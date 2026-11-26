//! Plugins written in Python (data/plugins/*.py, the Python harness's format), run in a subprocess per call.

use super::{Tool, BUILTIN};
use serde_json::{json, Value as J};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

const HOST: &str = r#"
import json, sys, importlib.util, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
f, what = sys.argv[1], sys.argv[2]
try:
    spec = importlib.util.spec_from_file_location("plugin_" + f.replace("\\", "/").rsplit("/", 1)[-1][:-3], f)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
except Exception as ex:
    print(json.dumps({"error": f"failed to load: {ex}"})); sys.exit(0)
if what == "describe":
    if not all(hasattr(m, a) for a in ("NAME", "DESCRIPTION", "PARAMETERS", "run")):
        print(json.dumps({"error": "not a plugin (needs NAME, DESCRIPTION, PARAMETERS and run())"})); sys.exit(0)
    print(json.dumps({"NAME": m.NAME, "DESCRIPTION": m.DESCRIPTION, "PARAMETERS": m.PARAMETERS}))
else:
    a = json.loads(sys.stdin.buffer.read().decode("utf-8") or "{}")
    try: print(json.dumps({"result": str(m.run(**a))}))
    except Exception as ex: print(json.dumps({"result": f"error: {ex}"}))
"#;

pub fn python_cmd(python: &str) -> Option<Vec<String>> {
    let cands: Vec<Vec<String>> = if !python.is_empty() {
        vec![python.split_whitespace().map(str::to_string).collect()]
    } else if cfg!(windows) {
        vec![vec!["python".into()], vec!["py".into(), "-3".into()], vec!["python3".into()]]
    } else {
        vec![vec!["python3".into()], vec!["python".into()]]
    };
    cands.into_iter().find(|c| {
        let mut cmd = Command::new(&c[0]);
        cmd.args(&c[1..]).arg("-c").arg("import sys").stdout(Stdio::null()).stderr(Stdio::null());
        no_window(&mut cmd);
        cmd.status().map(|s| s.success()).unwrap_or(false)
    })
}

#[cfg(windows)]
pub fn no_window(c: &mut Command) {
    use std::os::windows::process::CommandExt;
    c.creation_flags(0x0800_0000);
}
#[cfg(not(windows))]
pub fn no_window(_: &mut Command) {}

fn host(py: &[String], file: &Path, what: &str, input: &str, timeout_s: u64) -> Result<J, String> {
    let mut c = Command::new(&py[0]);
    c.args(&py[1..]).arg("-c").arg(HOST).arg(file).arg(what).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped());
    if let Some(d) = file.parent() {
        c.current_dir(d);
    }
    no_window(&mut c);
    let mut ch = c.spawn().map_err(|e| format!("cannot start Python: {e}"))?;
    ch.stdin.take().unwrap().write_all(input.as_bytes()).map_err(|e| e.to_string())?;
    let t0 = std::time::Instant::now();
    loop {
        if let Some(_) = ch.try_wait().map_err(|e| e.to_string())? {
            break;
        }
        if t0.elapsed().as_secs() >= timeout_s {
            let _ = ch.kill();
            return Err(format!("plugin timed out after {timeout_s} s"));
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    let o = ch.wait_with_output().map_err(|e| e.to_string())?;
    let s = String::from_utf8_lossy(&o.stdout);
    let line = s.lines().rev().find(|l| l.trim_start().starts_with('{')).unwrap_or("");
    serde_json::from_str(line).map_err(|_| format!("plugin produced no result: {}", String::from_utf8_lossy(&o.stderr).lines().last().unwrap_or("")))
}

pub struct PyPlugin {
    pub file: PathBuf,
    py: Option<Vec<String>>,
    name: String,
    desc: String,
    params: J,
    broken: Option<String>,
}

impl Tool for PyPlugin {
    fn name(&self) -> String {
        self.name.clone()
    }
    fn description(&self) -> String {
        self.desc.clone()
    }
    fn parameters(&self) -> J {
        self.params.clone()
    }
    fn broken(&self) -> Option<String> {
        self.broken.clone()
    }
    fn builtin(&self) -> bool {
        false
    }
    fn run(&self, a: &J) -> String {
        let Some(py) = &self.py else { return "error: Python is not installed, so Python plugins cannot run".into() };
        match host(py, &self.file, "run", &a.to_string(), 120) {
            Ok(v) => v["result"].as_str().map(str::to_string).unwrap_or_else(|| v.to_string()),
            Err(e) => format!("error: {e}"),
        }
    }
}

pub fn describe(file: &Path, py: Option<Vec<String>>) -> PyPlugin {
    let stem = file.file_stem().unwrap().to_string_lossy().into_owned();
    let broken = |msg: String, py| PyPlugin { file: file.to_path_buf(), py, name: stem.clone(), desc: format!("({msg})"), params: json!({}), broken: Some(msg) };
    let Some(p) = py.clone() else { return broken("Python is not installed".into(), None) };
    match host(&p, file, "describe", "", 30) {
        Ok(v) if v.get("error").is_some() => broken(v["error"].as_str().unwrap_or("").to_string(), py),
        Ok(v) => PyPlugin {
            file: file.to_path_buf(),
            py,
            name: v["NAME"].as_str().unwrap_or(&stem).to_string(),
            desc: v["DESCRIPTION"].as_str().unwrap_or("").to_string(),
            params: v["PARAMETERS"].clone(),
            broken: None,
        },
        Err(e) => broken(format!("failed to load: {e}"), py),
    }
}

/// Python plugins in a folder (the built-in tool names are skipped: the Rust versions replace them).
pub fn load_dir(dir: &Path, python: &str) -> Vec<std::sync::Arc<dyn Tool>> {
    let mut fs: Vec<PathBuf> = std::fs::read_dir(dir).map(|r| r.flatten().map(|e| e.path()).filter(|p| p.extension().map_or(false, |e| e == "py")).collect()).unwrap_or_default();
    fs.retain(|f| !BUILTIN.contains(&f.file_stem().unwrap().to_string_lossy().as_ref()));
    fs.sort();
    if fs.is_empty() {
        return Vec::new();
    }
    let py = python_cmd(python);
    fs.iter().map(|f| std::sync::Arc::new(describe(f, py.clone())) as std::sync::Arc<dyn Tool>).collect()
}
