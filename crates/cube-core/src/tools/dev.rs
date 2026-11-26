//! harness_dev - lets the model extend the harness with its own skills and plugins, with the user in control.
//!
//! skills   written to skills/<name>.md with 'author: model' and always_on: false - usable as /name; the user decides in the
//!          Skills tab whether one is always on (and can edit or delete it there)
//! plugins  written to plugins/pending/<name>.py and NOT loaded: Python runs with the user's permissions, so a model-written
//!          plugin only becomes a tool after the user reads it and approves it in the Plugins tab.
//! The model may only change or delete skills it wrote itself, and only pending plugins.

use super::{Tool, BUILTIN};
use crate::paths::Paths;
use crate::skills;
use regex::Regex;
use serde_json::{json, Value as J};

pub const DESCRIPTION: &str = "Call this tool by the name 'harness_dev' with an 'action'. Extend the harness: write reusable skills (instruction snippets the user can invoke as /name) and new tool plugins (Python). Plugins are saved for the user's review and are not usable until the user approves them in the Plugins tab. Actions: list | write_skill | delete_skill | write_plugin | read_plugin | discard_plugin. A plugin is a Python module with NAME (same as its name), DESCRIPTION (str), PARAMETERS (JSON schema dict) and def run(**args) -> str, using only the standard library.";

pub struct HarnessDev {
    p: Paths,
    python: String,
}

impl HarnessDev {
    pub fn new(p: Paths, python: String) -> Self {
        HarnessDev { p, python }
    }
}

fn name_ok(n: &str, plugin: bool) -> Result<String, String> {
    let mut n = n.trim().to_lowercase();
    if plugin {
        n = n.replace('-', "_");
    }
    if !Regex::new(r"^[a-z][a-z0-9_\-]{0,40}$").unwrap().is_match(&n) {
        return Err("name must start with a letter and use only a-z, 0-9, '_' or '-' (max 41 chars)".into());
    }
    Ok(n)
}

const CHECK: &str = r#"
import ast, sys, json
name, code = sys.argv[1], sys.stdin.buffer.read().decode("utf-8")
def chk():
    try: tree = ast.parse(code)
    except SyntaxError as ex: return f"syntax error line {ex.lineno}: {ex.msg}"
    names = {}; has_run = False
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name): names[t.id] = node.value
        if isinstance(node, ast.FunctionDef) and node.name == "run": has_run = True
    for need in ("NAME", "DESCRIPTION", "PARAMETERS"):
        if need not in names: return f"missing module-level {need} = ..."
    if not (isinstance(names["NAME"], ast.Constant) and names["NAME"].value == name): return f"NAME must be the string '{name}'"
    if not has_run: return "missing def run(**args) -> str"
    return None
print(json.dumps({"err": chk()}))
"#;

/// static checks only - the code is never executed
pub fn check_plugin(python: &str, name: &str, code: &str) -> Option<String> {
    if code.len() > 100_000 {
        return Some("plugin source is larger than 100 KB".into());
    }
    if let Some(py) = super::external::python_cmd(python) {
        use std::io::Write;
        let mut c = std::process::Command::new(&py[0]);
        c.args(&py[1..]).arg("-c").arg(CHECK).arg(name).stdin(std::process::Stdio::piped()).stdout(std::process::Stdio::piped()).stderr(std::process::Stdio::null());
        super::external::no_window(&mut c);
        if let Ok(mut ch) = c.spawn() {
            let _ = ch.stdin.take().unwrap().write_all(code.as_bytes());
            if let Ok(o) = ch.wait_with_output() {
                if let Ok(v) = serde_json::from_slice::<J>(String::from_utf8_lossy(&o.stdout).trim().as_bytes()) {
                    return v["err"].as_str().map(str::to_string);
                }
            }
        }
    }
    // no Python here: the same checks by pattern
    let has = |p: &str| Regex::new(p).unwrap().is_match(code);
    for need in ["NAME", "DESCRIPTION", "PARAMETERS"] {
        if !has(&format!(r"(?m)^{need}\s*=")) {
            return Some(format!("missing module-level {need} = ..."));
        }
    }
    if !has(&format!(r#"(?m)^NAME\s*=\s*["']{}["']\s*$"#, regex::escape(name))) {
        return Some(format!("NAME must be the string '{name}'"));
    }
    if !has(r"(?m)^def\s+run\s*\(") {
        return Some("missing def run(**args) -> str".into());
    }
    None
}

impl Tool for HarnessDev {
    fn name(&self) -> String {
        "harness_dev".into()
    }
    fn description(&self) -> String {
        DESCRIPTION.into()
    }
    fn parameters(&self) -> J {
        json!({"type": "object", "properties": {
            "action": {"type": "string", "enum": ["list", "write_skill", "delete_skill", "write_plugin", "read_plugin", "discard_plugin"], "description": "what to do"},
            "name": {"type": "string", "description": "skill or plugin name: lowercase letters, digits, '_' or '-'"},
            "description": {"type": "string", "description": "write_skill: one line saying when to use the skill"},
            "body": {"type": "string", "description": "write_skill: the skill's instructions (markdown)"},
            "code": {"type": "string", "description": "write_plugin: the complete Python source of the plugin module"}
        }, "required": ["action"]})
    }
    fn run(&self, a: &J) -> String {
        match self.go(a) {
            Ok(s) => s,
            Err(e) => format!("error: {e}"),
        }
    }
}

impl HarnessDev {
    fn go(&self, a: &J) -> Result<String, String> {
        let s = |k: &str| a.get(k).and_then(|v| v.as_str()).map(str::to_string);
        let (sk, pl) = (self.p.skills(), self.p.plugins());
        let pe = pl.join("pending");
        let _ = std::fs::create_dir_all(&sk);
        let _ = std::fs::create_dir_all(&pe);
        let stems = |d: &std::path::Path, ext: &str| -> Vec<String> {
            let mut v: Vec<String> = std::fs::read_dir(d)
                .map(|r| r.flatten().map(|e| e.path()).filter(|p| p.extension().map_or(false, |e| e == ext)).map(|p| p.file_stem().unwrap().to_string_lossy().into_owned()).collect())
                .unwrap_or_default();
            v.sort();
            v
        };
        let action = s("action").unwrap_or_else(|| "list".into());
        let nm = s("name").unwrap_or_default();
        match action.as_str() {
            "list" => {
                let skl: Vec<String> = stems(&sk, "md").into_iter().map(|n| format!("/{n}") + if skills::author(&sk.join(format!("{n}.md"))) == "model" { "  (written by model)" } else { "" }).collect();
                let mut loaded: Vec<String> = BUILTIN.iter().map(|x| x.to_string()).chain(stems(&pl, "py")).collect();
                loaded.sort();
                loaded.dedup();
                let pend = stems(&pe, "py");
                let j = |v: Vec<String>| if v.is_empty() { "none".to_string() } else { v.join(", ") };
                Ok(format!("skills: {}\nplugins (loaded): {}\nplugins waiting for the user's approval: {}", j(skl), j(loaded), j(pend)))
            }
            "write_skill" => {
                let n = name_ok(&nm, false)?;
                let f = sk.join(format!("{n}.md"));
                if f.exists() && skills::author(&f) != "model" {
                    return Ok(format!("error: skill '{n}' was written by the user - pick another name"));
                }
                let Some(body) = s("body").filter(|b| !b.is_empty()) else { return Ok("error: give body (the skill's instructions)".into()) };
                let desc: String = s("description").unwrap_or_default().split_whitespace().collect::<Vec<_>>().join(" ").chars().take(200).collect();
                std::fs::write(&f, format!("---\nname: {n}\ndescription: {desc}\nalways_on: false\nauthor: model\nwritten: {}\n---\n{}\n", super::artifact::now(), body.trim())).map_err(|e| e.to_string())?;
                Ok(format!("skill /{n} saved (off by default; the user can use it as /{n} or switch it on in the Skills tab)"))
            }
            "delete_skill" => {
                let n = name_ok(&nm, false)?;
                let f = sk.join(format!("{n}.md"));
                if !f.exists() {
                    return Ok(format!("error: no skill '{n}'"));
                }
                if skills::author(&f) != "model" {
                    return Ok(format!("error: skill '{n}' was written by the user; only the user can delete it"));
                }
                std::fs::remove_file(f).map_err(|e| e.to_string())?;
                Ok(format!("skill /{n} deleted"))
            }
            "write_plugin" => {
                let n = name_ok(&nm, true)?;
                if BUILTIN.contains(&n.as_str()) || pl.join(format!("{n}.py")).exists() {
                    return Ok(format!("error: a plugin named '{n}' already exists - pick another name"));
                }
                let Some(code) = s("code").filter(|c| !c.is_empty()) else { return Ok("error: give code (the full Python module)".into()) };
                if let Some(e) = check_plugin(&self.python, &n, &code) {
                    return Ok(format!("error: plugin not saved - {e}"));
                }
                std::fs::write(pe.join(format!("{n}.py")), format!("# written by the model on {} - review before enabling\n{code}", super::artifact::now())).map_err(|e| e.to_string())?;
                Ok(format!("plugin '{n}' saved for review. It is NOT active: the user must open the Plugins tab and press 'Approve and enable'."))
            }
            "read_plugin" => {
                let n = name_ok(&nm, true)?;
                let mut f = pe.join(format!("{n}.py"));
                if !f.exists() {
                    f = pl.join(format!("{n}.py"));
                }
                Ok(std::fs::read_to_string(&f).map(|t| t.chars().take(12000).collect()).unwrap_or_else(|_| format!("error: no plugin '{n}'")))
            }
            "discard_plugin" => {
                let n = name_ok(&nm, true)?;
                let f = pe.join(format!("{n}.py"));
                if !f.exists() {
                    return Ok(format!("error: no pending plugin '{n}' (approved plugins can only be removed by the user)"));
                }
                std::fs::remove_file(f).map_err(|e| e.to_string())?;
                Ok(format!("pending plugin '{n}' discarded"))
            }
            _ => Ok(format!("error: unknown action '{action}'")),
        }
    }
}
