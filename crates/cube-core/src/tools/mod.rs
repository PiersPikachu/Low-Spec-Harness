//! Tools the model may call: built-in Rust tools (artifact, calculator, clock, read_file, harness_dev) and plugins from
//! data/plugins/*.py (the Python harness's plugin format: NAME, DESCRIPTION, PARAMETERS, run(**args)), run in a Python
//! subprocess so a plugin can never crash or block the harness.

pub mod artifact;
pub mod calc;
pub mod dev;
pub mod external;

use crate::paths::Paths;
use serde_json::{json, Value as J};
use std::sync::Arc;

pub trait Tool: Send + Sync {
    fn name(&self) -> String;
    fn description(&self) -> String;
    fn parameters(&self) -> J;
    fn run(&self, args: &J) -> String;
    /// a plugin that failed to load (shown in the Plugins tab, never offered to the model)
    fn broken(&self) -> Option<String> {
        None
    }
    fn builtin(&self) -> bool {
        true
    }
}

pub const BUILTIN: [&str; 5] = ["artifact", "calculator", "clock", "read_file", "harness_dev"];

struct Calculator;
impl Tool for Calculator {
    fn name(&self) -> String {
        calc::NAME.into()
    }
    fn description(&self) -> String {
        calc::DESCRIPTION.into()
    }
    fn parameters(&self) -> J {
        json!({"type": "object", "properties": {"expression": {"type": "string", "description": "the expression to evaluate"}}, "required": ["expression"]})
    }
    fn run(&self, a: &J) -> String {
        calc::run(a.get("expression").and_then(|x| x.as_str()).unwrap_or(""))
    }
}

struct Clock;
impl Tool for Clock {
    fn name(&self) -> String {
        "clock".into()
    }
    fn description(&self) -> String {
        "Return the current local date, time, weekday and UTC offset of this computer.".into()
    }
    fn parameters(&self) -> J {
        json!({"type": "object", "properties": {}, "required": []})
    }
    fn run(&self, _: &J) -> String {
        chrono::Local::now().format("%A %Y-%m-%d %H:%M:%S (UTC%z)").to_string()
    }
}

struct ReadFile(std::path::PathBuf);
impl Tool for ReadFile {
    fn name(&self) -> String {
        "read_file".into()
    }
    fn description(&self) -> String {
        "Read a UTF-8 text file from the harness workspace folder. Path is relative to the workspace, e.g. 'notes.txt'. Returns at most 8000 characters. Use path '.' to list the files.".into()
    }
    fn parameters(&self) -> J {
        json!({"type": "object", "properties": {"path": {"type": "string", "description": "file path relative to the workspace folder, or '.' to list files"}}, "required": ["path"]})
    }
    fn run(&self, a: &J) -> String {
        let ws = &self.0;
        let _ = std::fs::create_dir_all(ws);
        let path = a.get("path").and_then(|x| x.as_str()).unwrap_or(".");
        let ws_c = ws.canonicalize().unwrap_or(ws.clone());
        let mut p = ws_c.clone();
        for part in path.replace('\\', "/").split('/') {
            match part {
                "" | "." => {}
                ".." => {
                    p.pop();
                }
                x => p.push(x),
            }
        }
        let p = p.canonicalize().unwrap_or(p);
        if !p.starts_with(&ws_c) {
            return "error: outside the workspace folder".into();
        }
        if p.is_dir() {
            let mut v = Vec::new();
            fn walk(b: &std::path::Path, d: &std::path::Path, v: &mut Vec<String>) {
                if let Ok(r) = std::fs::read_dir(d) {
                    for e in r.flatten() {
                        if e.path().is_dir() {
                            walk(b, &e.path(), v)
                        } else {
                            v.push(e.path().strip_prefix(b).unwrap().to_string_lossy().replace('\\', "/"))
                        }
                    }
                }
            }
            walk(&ws_c, &p, &mut v);
            v.sort();
            v.truncate(200);
            return if v.is_empty() { "(workspace is empty)".into() } else { v.join("\n") };
        }
        if !p.exists() {
            return format!("error: {path} not found");
        }
        match std::fs::read(&p) {
            Ok(b) => String::from_utf8_lossy(&b).chars().take(8000).collect(),
            Err(e) => format!("error: {e}"),
        }
    }
}

/// Every tool the harness knows, in name order.
pub struct Tools {
    pub list: Vec<Arc<dyn Tool>>,
    pub artifacts: Arc<artifact::Artifacts>,
}

impl Tools {
    pub fn load(p: &Paths, python: &str) -> Tools {
        let artifacts = Arc::new(artifact::Artifacts::new(p.artifacts()));
        let mut list: Vec<Arc<dyn Tool>> = vec![
            Arc::new(artifact::ArtifactTool(artifacts.clone())),
            Arc::new(Calculator),
            Arc::new(Clock),
            Arc::new(ReadFile(p.workspace())),
            Arc::new(dev::HarnessDev::new(p.clone(), python.to_string())),
        ];
        for t in external::load_dir(&p.plugins(), python) {
            if !list.iter().any(|x| x.name() == t.name()) {
                list.push(t);
            }
        }
        list.sort_by_key(|t| t.name());
        Tools { list, artifacts }
    }

    pub fn get(&self, n: &str) -> Option<&Arc<dyn Tool>> {
        self.list.iter().find(|t| t.name() == n)
    }

    pub fn usable(&self, enabled: &[String]) -> Vec<&Arc<dyn Tool>> {
        self.list.iter().filter(|t| t.broken().is_none() && enabled.iter().any(|e| *e == t.name())).collect()
    }

    /// OpenAI-style function schemas of the enabled tools (what the chat template receives as `tools`)
    pub fn schemas(&self, enabled: &[String]) -> Vec<J> {
        self.usable(enabled)
            .iter()
            .map(|t| json!({"type": "function", "function": {"name": t.name(), "description": t.description(), "parameters": t.parameters()}}))
            .collect()
    }

    /// (name, actions) of the enabled tools, for resolving decorated call names
    pub fn actions(&self, enabled: &[String]) -> Vec<(String, Vec<String>)> {
        self.usable(enabled)
            .iter()
            .map(|t| {
                let p = t.parameters();
                let acts = p["properties"]["action"]["enum"].as_array().map(|a| a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect()).unwrap_or_default();
                (t.name(), acts)
            })
            .collect()
    }

    pub fn run(&self, name: &str, args: &J) -> String {
        match self.get(name).filter(|t| t.broken().is_none()) {
            Some(t) => t.run(args),
            None => {
                let have: Vec<String> = self.list.iter().filter(|t| t.broken().is_none()).map(|t| t.name()).collect();
                format!(
                    "error: no tool named '{name}'. Available tools: {}. Call a tool by its name and put the operation in its arguments (e.g. artifact with action='create').",
                    have.join(", ")
                )
            }
        }
    }
}
