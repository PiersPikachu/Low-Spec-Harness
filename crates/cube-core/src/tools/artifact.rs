//! artifact - persistent, editable artifacts the model creates and maintains across chats (notes, HTML pages, multi-file sites,
//! codebases, data), stored in data/artifacts/<name>/. Every change keeps the previous version of the file in
//! .versions/ so any edit can be reverted (by the model with action=revert, or from the Artifacts tab).
//! The model never touches anything outside data/artifacts/.

use super::Tool;
use serde_json::{json, Value as J};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

pub const NAME: &str = "artifact";
pub const TYPES: [(&str, &str); 5] = [("note", "notes.md"), ("page", "index.html"), ("site", "index.html"), ("code", "README.md"), ("data", "data.csv")];
pub const MAX_WRITE: usize = 2_000_000;
pub const MAX_FILES: usize = 500;
pub const DESCRIPTION: &str = "Call this tool by the name 'artifact' and put the operation in the 'action' argument, e.g. artifact(action='create', name='my-notes', type='note'). Create and maintain persistent artifacts saved in the harness (kept across chats): notes, a single HTML page, a multi-file website, a codebase, or data files. Use 'list' to see what exists, 'read' before editing, and 'edit' for small changes (old text must match exactly once). Actions: list | create | files | read | write | edit | append | delete_file | history | revert. Types for create: note (notes.md), page (one self-contained index.html), site (index.html + other files), code (any files), data (csv/json). Paths are relative to the artifact, e.g. 'index.html' or 'src/app.js'. Every change can be undone with revert.";

pub fn parameters() -> J {
    json!({"type": "object", "properties": {
        "action": {"type": "string", "enum": ["list", "create", "files", "read", "write", "edit", "append", "delete_file", "history", "revert"], "description": "what to do"},
        "name": {"type": "string", "description": "artifact name (lowercase words joined by '-', e.g. 'trip-notes'); not needed for list"},
        "type": {"type": "string", "enum": ["note", "page", "site", "code", "data"], "description": "create only: kind of artifact"},
        "title": {"type": "string", "description": "create only: human-readable title"},
        "path": {"type": "string", "description": "file inside the artifact; defaults to its main file"},
        "content": {"type": "string", "description": "create / write / append: the text to put in the file"},
        "old": {"type": "string", "description": "edit: exact existing text to replace (must occur exactly once)"},
        "new": {"type": "string", "description": "edit: replacement text"},
        "version": {"type": "integer", "description": "revert: change number from history (restores the file as it was before that change)"},
        "start_line": {"type": "integer", "description": "read: first line to show (1-based), for long files"}
    }, "required": ["action"]})
}

pub fn now() -> String {
    chrono::Local::now().format("%Y-%m-%dT%H:%M:%S").to_string()
}

pub fn slug(s: &str) -> String {
    let mut out = String::new();
    let mut dash = false;
    for c in s.to_lowercase().chars() {
        if c.is_ascii_lowercase() || c.is_ascii_digit() {
            out.push(c);
            dash = false;
        } else if !dash {
            out.push('-');
            dash = true;
        }
    }
    let t: String = out.trim_matches('-').chars().take(60).collect();
    t
}

fn main_for(t: &str) -> &'static str {
    TYPES.iter().find(|(k, _)| *k == t).map(|x| x.1).unwrap_or("README.md")
}

struct LiveState {
    dir: PathBuf,
    path: PathBuf,
    rel: String,
    existed_art: bool,
    existed: bool,
    base: String,
}

pub struct Artifacts {
    pub root: PathBuf,
    live: Mutex<HashMap<(String, String, String), LiveState>>,
}

type R<T> = Result<T, String>;

fn read_text(p: &Path) -> String {
    std::fs::read(p).map(|b| String::from_utf8_lossy(&b).into_owned()).unwrap_or_default()
}
fn write_text(p: &Path, s: &str) -> R<()> {
    if let Some(d) = p.parent() {
        std::fs::create_dir_all(d).map_err(|e| e.to_string())?;
    }
    std::fs::write(p, s).map_err(|e| e.to_string())
}

impl Artifacts {
    pub fn new(root: PathBuf) -> Artifacts {
        Artifacts { root, live: Mutex::new(HashMap::new()) }
    }

    pub fn meta(&self, d: &Path) -> J {
        std::fs::read(d.join(".artifact.json"))
            .ok()
            .and_then(|b| serde_json::from_slice(&b).ok())
            .unwrap_or_else(|| json!({"title": d.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default(), "type": "code", "main": "README.md", "created": "", "updated": "", "changes": []}))
    }
    fn save_meta(&self, d: &Path, m: &mut J) -> R<()> {
        m["updated"] = J::String(now());
        std::fs::write(d.join(".artifact.json"), serde_json::to_string_pretty(m).unwrap()).map_err(|e| e.to_string())
    }

    pub fn names(&self) -> Vec<String> {
        let mut v: Vec<String> = std::fs::read_dir(&self.root)
            .map(|r| r.filter_map(|e| e.ok()).filter(|e| e.path().join(".artifact.json").exists()).map(|e| e.file_name().to_string_lossy().into_owned()).collect())
            .unwrap_or_default();
        v.sort();
        v
    }

    pub fn dir(&self, name: &str, must_exist: bool) -> R<PathBuf> {
        let n = slug(name);
        if n.is_empty() {
            return Err("give the artifact a name, e.g. name='trip-notes'".into());
        }
        let d = self.root.join(&n);
        if must_exist && !d.join(".artifact.json").exists() {
            let have = self.names().join(", ");
            return Err(format!("no artifact named '{n}'") + &if have.is_empty() { "; create it first with action=create".to_string() } else { format!("; existing: {have}") });
        }
        Ok(d)
    }

    /// (absolute file, relative path) of a file inside an artifact
    pub fn file(&self, d: &Path, path: Option<&str>, m: &J) -> R<(PathBuf, String)> {
        let rel = path.filter(|s| !s.is_empty()).map(str::to_string).or_else(|| m["main"].as_str().map(str::to_string)).unwrap_or_default().replace('\\', "/");
        let rel = rel.trim_matches('/').to_string();
        if rel.is_empty() || rel.starts_with(".versions") || rel == ".artifact.json" {
            return Err("give a file path inside the artifact".into());
        }
        let mut p = d.to_path_buf();
        for part in rel.split('/') {
            match part {
                "" | "." => {}
                ".." => {
                    if !p.pop() {
                        return Err("path must stay inside the artifact folder".into());
                    }
                }
                x => p.push(x),
            }
        }
        if !p.starts_with(d) || p == d {
            return Err("path must stay inside the artifact folder".into());
        }
        Ok((p, rel))
    }

    fn snapshot(&self, d: &Path, m: &mut J, rel: &str, p: &Path, action: &str) -> R<u64> {
        let n = m["changes"].as_array().map_or(0, |a| a.len()) as u64 + 1;
        let vd = d.join(".versions");
        std::fs::create_dir_all(&vd).map_err(|e| e.to_string())?;
        let existed = p.exists();
        if existed {
            std::fs::copy(p, vd.join(format!("{n:05}.prev"))).map_err(|e| e.to_string())?;
        }
        if !m["changes"].is_array() {
            m["changes"] = json!([]);
        }
        m["changes"].as_array_mut().unwrap().push(json!({"n": n, "time": now(), "action": action, "path": rel, "existed": existed}));
        Ok(n)
    }

    pub fn files(&self, d: &Path) -> Vec<String> {
        fn walk(base: &Path, d: &Path, out: &mut Vec<String>) {
            if let Ok(rd) = std::fs::read_dir(d) {
                for e in rd.flatten() {
                    let p = e.path();
                    let n = e.file_name().to_string_lossy().into_owned();
                    if p.is_dir() {
                        if n != ".versions" {
                            walk(base, &p, out);
                        }
                    } else if n != ".artifact.json" {
                        out.push(p.strip_prefix(base).unwrap().to_string_lossy().replace('\\', "/"));
                    }
                }
            }
        }
        let mut v = Vec::new();
        walk(d, d, &mut v);
        v.sort();
        v
    }

    // ------------------------------------------------------------ live saving (used by the harness, not a tool action)
    /// While the model streams a write/append/create call, save the content-so-far into the file, so the work is on disk even
    /// if the reply is cut off. `live_end` restores the pre-call state before the completed call runs normally.
    pub fn live_update(&self, name: &str, path: Option<&str>, action: &str, text: &str, title: Option<&str>, typ: Option<&str>) -> Option<String> {
        let n = slug(name);
        if n.is_empty() {
            return None;
        }
        let d = self.root.join(&n);
        let key = (n.clone(), path.unwrap_or("").replace('\\', "/").trim_matches('/').to_string(), action.to_string());
        let mut live = self.live.lock().unwrap();
        if !live.contains_key(&key) {
            let existed_art = d.join(".artifact.json").exists();
            if !existed_art && action != "create" {
                return None;
            }
            if !existed_art {
                let t = typ.filter(|t| TYPES.iter().any(|x| x.0 == *t)).unwrap_or("page");
                let main = if key.1.is_empty() { main_for(t).to_string() } else { key.1.clone() };
                let tm = now();
                std::fs::create_dir_all(&d).ok()?;
                std::fs::write(
                    d.join(".artifact.json"),
                    serde_json::to_string_pretty(&json!({"title": title.unwrap_or(name), "type": t, "main": main, "created": tm, "updated": tm, "changes": []})).unwrap(),
                )
                .ok()?;
            }
            let m = self.meta(&d);
            let (p, rel) = self.file(&d, Some(if key.1.is_empty() { m["main"].as_str().unwrap_or("") } else { &key.1 }), &m).ok()?;
            let existed = p.exists();
            let base = if existed { read_text(&p) } else { String::new() };
            live.insert(key.clone(), LiveState { dir: d.clone(), path: p, rel, existed_art, existed, base });
        }
        let st = live.get(&key)?;
        let body = if action == "append" {
            format!("{}{}{}", st.base, if st.base.is_empty() || st.base.ends_with('\n') { "" } else { "\n" }, text)
        } else {
            text.to_string()
        };
        if body.len() > MAX_WRITE {
            return Some(st.rel.clone());
        }
        write_text(&st.path, &body).ok()?;
        Some(st.rel.clone())
    }

    pub fn live_end(&self, name: &str, path: Option<&str>, action: &str) {
        let key = (slug(name), path.unwrap_or("").replace('\\', "/").trim_matches('/').to_string(), action.to_string());
        let st = self.live.lock().unwrap().remove(&key);
        if let Some(st) = st {
            if !st.existed_art {
                let _ = std::fs::remove_dir_all(&st.dir);
                return;
            }
            if st.existed {
                let _ = std::fs::write(&st.path, &st.base);
            } else if st.path.exists() {
                let _ = std::fs::remove_file(&st.path);
            }
        }
    }

    /// force stop / crash path: keep whatever was saved live, forget the restore points
    pub fn live_keep(&self) {
        self.live.lock().unwrap().clear();
    }

    /// The user's edits in the Artifacts tab: one revertible change per editing session, later saves update it in place.
    pub fn autosave(&self, name: &str, path: Option<&str>, text: &str, session: &str) -> R<String> {
        let d = self.dir(name, true)?;
        let mut m = self.meta(&d);
        let (p, rel) = self.file(&d, path, &m)?;
        let last = m["changes"].as_array().and_then(|a| a.last()).cloned();
        if let Some(l) = last {
            if l["session"].as_str() == Some(session) && l["path"].as_str() == Some(&rel) {
                write_text(&p, text)?;
                self.save_meta(&d, &mut m)?;
                return Ok(format!("autosaved {rel}"));
            }
        }
        let n = self.snapshot(&d, &mut m, &rel, &p, "user edit")?;
        if let Some(l) = m["changes"].as_array_mut().and_then(|a| a.last_mut()) {
            l["session"] = J::String(session.into());
        }
        write_text(&p, text)?;
        self.save_meta(&d, &mut m)?;
        Ok(format!("saved {rel} (change #{n})"))
    }

    pub fn delete_artifact(&self, name: &str) -> R<()> {
        let d = self.dir(name, true)?;
        std::fs::remove_dir_all(d).map_err(|e| e.to_string())
    }

    pub fn act(&self, a: &J) -> String {
        match self.act_r(a) {
            Ok(s) => s,
            Err(e) => format!("error: {e}"),
        }
    }

    fn act_r(&self, a: &J) -> R<String> {
        let s = |k: &str| a.get(k).and_then(|v| v.as_str()).map(str::to_string);
        let int = |k: &str| a.get(k).and_then(|v| v.as_i64().or_else(|| v.as_f64().map(|f| f as i64)).or_else(|| v.as_str().and_then(|x| x.trim().parse().ok())));
        let action = s("action").unwrap_or_else(|| "list".into());
        let name = s("name").unwrap_or_default();
        let _ = std::fs::create_dir_all(&self.root);
        if action == "list" {
            let rows: Vec<String> = self
                .names()
                .into_iter()
                .map(|n| {
                    let d = self.root.join(&n);
                    let m = self.meta(&d);
                    format!(
                        "{n}  [{}]  '{}'  {} file(s), main {}, updated {}",
                        m["type"].as_str().unwrap_or(""),
                        m["title"].as_str().unwrap_or(""),
                        self.files(&d).len(),
                        m["main"].as_str().unwrap_or(""),
                        m["updated"].as_str().unwrap_or("")
                    )
                })
                .collect();
            return Ok(if rows.is_empty() { "no artifacts yet - create one with action=create".into() } else { rows.join("\n") });
        }
        if action == "create" {
            let d = self.dir(&name, false)?;
            let dn = d.file_name().unwrap().to_string_lossy().into_owned();
            if d.join(".artifact.json").exists() {
                return Ok(format!("artifact '{dn}' already exists - use write/edit to change it"));
            }
            let t = s("type").filter(|t| TYPES.iter().any(|x| x.0 == t)).unwrap_or_else(|| "note".into());
            let main = s("path").map(|p| p.replace('\\', "/").trim_matches('/').to_string()).filter(|p| !p.is_empty()).unwrap_or_else(|| main_for(&t).to_string());
            std::fs::create_dir_all(&d).map_err(|e| e.to_string())?;
            let tm = now();
            let title = s("title").unwrap_or_else(|| name.clone());
            let mut m = json!({"title": title, "type": t, "main": main, "created": tm, "updated": tm, "changes": []});
            let (p, rel) = self.file(&d, Some(&main), &m)?;
            let body = match s("content") {
                Some(c) => c,
                None => match t.as_str() {
                    "note" => format!("# {title}\n\n"),
                    "data" => String::new(),
                    "page" | "site" => format!("<!doctype html>\n<html><head><meta charset='utf-8'><title>{title}</title></head><body>\n</body></html>\n"),
                    _ => format!("# {title}\n"),
                },
            };
            if body.len() > MAX_WRITE {
                return Ok(format!("error: content larger than {MAX_WRITE} bytes"));
            }
            write_text(&p, &body)?;
            self.save_meta(&d, &mut m)?;
            return Ok(format!("created artifact '{dn}' ({t}) with {rel} ({} chars). Open it in the Artifacts tab.", body.chars().count()));
        }
        let d = self.dir(&name, true)?;
        let mut m = self.meta(&d);
        match action.as_str() {
            "files" => {
                let fs: Vec<String> = self.files(&d).into_iter().map(|f| format!("{f}  {} bytes", std::fs::metadata(d.join(&f)).map(|x| x.len()).unwrap_or(0))).collect();
                return Ok(if fs.is_empty() { "(no files)".into() } else { fs.join("\n") });
            }
            "history" => {
                let ch = m["changes"].as_array().cloned().unwrap_or_default();
                let tail = &ch[ch.len().saturating_sub(30)..];
                let rows: Vec<String> = tail
                    .iter()
                    .map(|c| format!("#{}  {}  {}  {}", c["n"], c["time"].as_str().unwrap_or(""), c["action"].as_str().unwrap_or(""), c["path"].as_str().unwrap_or("")))
                    .collect();
                return Ok(if rows.is_empty() { "no changes recorded yet".into() } else { rows.join("\n") });
            }
            "revert" => {
                let Some(v) = int("version") else { return Ok("error: give version (a change number from action=history)".into()) };
                let Some(c) = m["changes"].as_array().and_then(|a| a.iter().find(|c| c["n"].as_i64() == Some(v)).cloned()) else {
                    return Ok(format!("error: no change #{v}"));
                };
                let (p, rel) = self.file(&d, c["path"].as_str(), &m)?;
                let n = self.snapshot(&d, &mut m, &rel, &p, &format!("revert #{v}"))?;
                let prev = d.join(".versions").join(format!("{v:05}.prev"));
                let existed = c["existed"].as_bool().unwrap_or(false);
                if existed {
                    if let Some(pp) = p.parent() {
                        let _ = std::fs::create_dir_all(pp);
                    }
                    std::fs::copy(&prev, &p).map_err(|e| e.to_string())?;
                } else if p.exists() {
                    let _ = std::fs::remove_file(&p);
                }
                self.save_meta(&d, &mut m)?;
                return Ok(if existed { format!("{rel} restored to how it was before change #{v}") } else { format!("{rel} removed (it did not exist before change #{v})") } + &format!(" (recorded as change #{n})"));
            }
            _ => {}
        }
        let (p, rel) = self.file(&d, s("path").as_deref(), &m)?;
        match action.as_str() {
            "read" => {
                if !p.exists() {
                    let fs = self.files(&d).join(", ");
                    return Ok(format!("error: {rel} does not exist; files: {}", if fs.is_empty() { "(none)".into() } else { fs }));
                }
                let text = read_text(&p);
                let lines = crate::reply::py_splitlines(&text);
                let s0 = int("start_line").unwrap_or(1).max(1) as usize;
                let (mut out, mut size) = (Vec::new(), 0usize);
                for i in (s0 - 1)..lines.len() {
                    let row = format!("{:5}| {}", i + 1, lines[i]);
                    size += row.chars().count() + 1;
                    if size > 12000 {
                        out.push(format!("... ({} more lines; read again with start_line={})", lines.len() - i, i + 1));
                        break;
                    }
                    out.push(row);
                }
                Ok(format!("{rel} ({} lines):\n{}", lines.len(), out.join("\n")))
            }
            "write" | "append" => {
                let Some(content) = s("content") else { return Ok("error: give content".into()) };
                if action == "write" && !p.exists() && self.files(&d).len() >= MAX_FILES {
                    return Ok(format!("error: artifact already has {MAX_FILES} files"));
                }
                let new_text = if action == "write" {
                    content
                } else {
                    let old = if p.exists() { read_text(&p) } else { String::new() };
                    let sep = if p.exists() && !old.ends_with('\n') { "\n" } else { "" };
                    format!("{old}{sep}{content}")
                };
                if new_text.len() > MAX_WRITE {
                    return Ok(format!("error: file would exceed {MAX_WRITE} bytes"));
                }
                let n = self.snapshot(&d, &mut m, &rel, &p, &action)?;
                write_text(&p, &new_text)?;
                self.save_meta(&d, &mut m)?;
                Ok(format!("{action} {rel}: {} chars, {} lines (change #{n})", new_text.chars().count(), new_text.matches('\n').count() + 1))
            }
            "edit" => {
                let (Some(old), Some(new)) = (s("old"), s("new")) else { return Ok("error: give old and new".into()) };
                if !p.exists() {
                    return Ok(format!("error: {rel} does not exist"));
                }
                let text = read_text(&p);
                let k = if old.is_empty() { text.chars().count() + 1 } else { text.matches(&old).count() };
                if k != 1 {
                    return Ok(format!("error: old text found {k} times in {rel} - it must match exactly once (read the file and copy a longer, unique piece)"));
                }
                let n = self.snapshot(&d, &mut m, &rel, &p, "edit")?;
                write_text(&p, &text.replacen(&old, &new, 1))?;
                self.save_meta(&d, &mut m)?;
                Ok(format!("edited {rel} (change #{n}): replaced {} chars with {}", old.chars().count(), new.chars().count()))
            }
            "delete_file" => {
                if !p.exists() {
                    return Ok(format!("error: {rel} does not exist"));
                }
                let n = self.snapshot(&d, &mut m, &rel, &p, "delete")?;
                std::fs::remove_file(&p).map_err(|e| e.to_string())?;
                self.save_meta(&d, &mut m)?;
                Ok(format!("deleted {rel} (change #{n}, revertible)"))
            }
            _ => Ok(format!("error: unknown action '{action}'")),
        }
    }
}

pub struct ArtifactTool(pub std::sync::Arc<Artifacts>);
impl Tool for ArtifactTool {
    fn name(&self) -> String {
        NAME.into()
    }
    fn description(&self) -> String {
        DESCRIPTION.into()
    }
    fn parameters(&self) -> J {
        parameters()
    }
    fn run(&self, args: &J) -> String {
        self.0.act(args)
    }
}
