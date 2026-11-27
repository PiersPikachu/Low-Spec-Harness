//! Every path the harness uses, resolved from the folder it runs in (the whole folder can be moved or copied), and the
//! settings file. Same layout and `harness_config.json` keys as the Python harness, so a data folder can be shared.
//!
//! ```text
//! cube-harness-rs/        ROOT (holds cube.exe / cube-harness.exe, or set CUBE_HARNESS_ROOT)
//!   data/
//!     vaults/ conversations/ skills/ plugins/ artifacts/ workspace/ hub_cache/ packages/ logs/
//!     harness_config.json  settings; paths inside ROOT are stored relative to it
//! ```

use serde_json::{json, Map, Value as J};
use std::path::{Path, PathBuf};

/// `canonicalize` without Windows' `\\?\` verbatim prefix (which would leak into settings files and messages).
pub fn canon(p: &Path) -> PathBuf {
    strip_verbatim(&p.canonicalize().unwrap_or_else(|_| p.to_path_buf()))
}

pub fn strip_verbatim(p: &Path) -> PathBuf {
    let s = p.to_string_lossy();
    if let Some(r) = s.strip_prefix(r"\\?\UNC\") {
        PathBuf::from(format!(r"\\{r}"))
    } else if let Some(r) = s.strip_prefix(r"\\?\") {
        PathBuf::from(r)
    } else {
        p.to_path_buf()
    }
}

#[derive(Clone, Debug)]
pub struct Paths {
    pub root: PathBuf,
    pub data: PathBuf,
}

impl Paths {
    /// ROOT: `CUBE_HARNESS_ROOT`, else the first folder upward from the executable that has a `data` folder, else the exe folder.
    pub fn discover() -> Paths {
        if let Ok(r) = std::env::var("CUBE_HARNESS_ROOT") {
            return Paths::at(PathBuf::from(r));
        }
        let exe = std::env::current_exe().ok().and_then(|p| p.parent().map(Path::to_path_buf)).unwrap_or_else(|| PathBuf::from("."));
        let mut d = Some(exe.as_path());
        while let Some(x) = d {
            let repo = x.join("Cargo.toml").exists() && x.join("crates").is_dir();
            if repo || (x.join("data").is_dir() && (x.join("cube-harness.bat").exists() || x.join("data").join("harness_config.json").exists())) {
                return Paths::at(x.to_path_buf());
            }
            d = x.parent();
        }
        Paths::at(exe)
    }
    pub fn at(root: PathBuf) -> Paths {
        let root = canon(&root);
        Paths { data: root.join("data"), root }
    }
    pub fn config(&self) -> PathBuf {
        self.data.join("harness_config.json")
    }
    pub fn vaults(&self) -> PathBuf {
        self.data.join("vaults")
    }
    pub fn conversations(&self) -> PathBuf {
        self.data.join("conversations")
    }
    pub fn skills(&self) -> PathBuf {
        self.data.join("skills")
    }
    pub fn plugins(&self) -> PathBuf {
        self.data.join("plugins")
    }
    pub fn artifacts(&self) -> PathBuf {
        self.data.join("artifacts")
    }
    pub fn workspace(&self) -> PathBuf {
        self.data.join("workspace")
    }
    pub fn hub_cache(&self) -> PathBuf {
        self.data.join("hub_cache")
    }
    pub fn logs(&self) -> PathBuf {
        self.data.join("logs")
    }
    pub fn packages(&self) -> PathBuf {
        self.data.join("packages")
    }
    pub fn ensure_dirs(&self) {
        for d in [self.data.clone(), self.vaults(), self.conversations(), self.skills(), self.plugins(), self.artifacts(), self.workspace(), self.hub_cache(), self.logs(), self.packages()] {
            let _ = std::fs::create_dir_all(d);
        }
    }

    /// store a path relative to ROOT when it lives inside the harness folder
    pub fn rel(&self, p: &Path) -> String {
        let p = canon(p);
        match p.strip_prefix(&self.root) {
            Ok(r) => r.to_string_lossy().into_owned(),
            Err(_) => p.to_string_lossy().into_owned(),
        }
    }
    pub fn absolute(&self, p: &str) -> PathBuf {
        let q = strip_verbatim(Path::new(p));
        if q.is_absolute() {
            q
        } else {
            canon(&self.root.join(q))
        }
    }
}

pub fn defaults() -> Map<String, J> {
    let v = json!({
        "vault": "", "mode": "auto", "gpu_layers": 0, "ctx": 2048, "max_new": 512, "kv16": false, "max_file_growth": 0,
        "temp": 0.7, "top_p": 0.8, "top_k": 20, "greedy": false, "seed": null,
        "draft": "auto", "k": 8, "think": false, "system": "",
        "model_dirs": [], "enabled_plugins": ["calculator", "clock", "artifact"], "max_tool_rounds": 4,
        "resident_mb": 0, "threads": 0
    });
    v.as_object().unwrap().clone()
}

/// Settings: defaults overlaid with `harness_config.json`.
#[derive(Clone, Debug)]
pub struct Config(pub Map<String, J>);

impl Config {
    pub fn load(p: &Paths) -> Config {
        let mut c = defaults();
        if let Ok(b) = std::fs::read(p.config()) {
            if let Ok(J::Object(o)) = serde_json::from_slice::<J>(&b) {
                for (k, v) in o {
                    c.insert(k, v);
                }
            }
        }
        // older configs: extra_vault_dirs / hub_vault_dir become entries of model_dirs
        let mut md: Vec<J> = c.get("model_dirs").and_then(|x| x.as_array().cloned()).unwrap_or_default();
        if let Some(J::Array(x)) = c.remove("extra_vault_dirs") {
            for e in x {
                if !md.iter().any(|m| m["path"] == e) {
                    md.push(json!({"path": e, "default": false}));
                }
            }
        }
        if let Some(hv) = c.remove("hub_vault_dir").filter(|v| !v.is_null()) {
            if !md.iter().any(|m| m["path"] == hv) {
                md.push(json!({"path": hv, "default": true}));
            }
        }
        c.insert("model_dirs".into(), J::Array(md));
        Config(c)
    }
    pub fn save(&self, p: &Paths) -> std::io::Result<()> {
        std::fs::create_dir_all(&p.data)?;
        std::fs::write(p.config(), serde_json::to_string_pretty(&J::Object(self.0.clone())).unwrap())
    }
    pub fn get(&self, k: &str) -> &J {
        self.0.get(k).unwrap_or(&J::Null)
    }
    pub fn str(&self, k: &str) -> String {
        match self.get(k) {
            J::String(s) => s.clone(),
            J::Null => String::new(),
            v => v.to_string(),
        }
    }
    pub fn f64(&self, k: &str) -> f64 {
        match self.get(k) {
            J::Number(n) => n.as_f64().unwrap_or(0.0),
            J::String(s) => s.trim().parse().unwrap_or(0.0),
            J::Bool(b) => f64::from(u8::from(*b)),
            _ => 0.0,
        }
    }
    pub fn i64(&self, k: &str) -> i64 {
        self.f64(k) as i64
    }
    pub fn bool(&self, k: &str) -> bool {
        match self.get(k) {
            J::Bool(b) => *b,
            J::Number(n) => n.as_f64().unwrap_or(0.0) != 0.0,
            J::String(s) => matches!(s.as_str(), "1" | "true" | "True" | "yes"),
            _ => false,
        }
    }
    pub fn set(&mut self, k: &str, v: J) {
        self.0.insert(k.into(), v);
    }
    pub fn enabled_plugins(&self) -> Vec<String> {
        self.get("enabled_plugins").as_array().map(|a| a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect()).unwrap_or_default()
    }
}

/// A folder vaults are kept in.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelDir {
    pub path: PathBuf,
    pub default: bool,
    pub builtin: bool,
}

pub fn is_vault(d: &Path) -> bool {
    d.is_dir() && d.join("vault.bin").exists() && (d.join("vault.cv8").exists() || d.join("vault.idx").exists())
}

/// The built-in data/vaults first, then the folders added in the Model folders tab.
pub fn model_dirs(p: &Paths, c: &Config) -> Vec<ModelDir> {
    let mut out = vec![ModelDir { path: p.vaults(), default: false, builtin: true }];
    for e in c.get("model_dirs").as_array().cloned().unwrap_or_default() {
        let Some(s) = e["path"].as_str() else { continue };
        let q = p.absolute(s);
        let def = e["default"].as_bool().unwrap_or(false);
        if q == p.vaults() {
            out[0].default = def;
        } else if !out.iter().any(|o| o.path == q) {
            out.push(ModelDir { path: q, default: def, builtin: false });
        }
    }
    if !out.iter().any(|o| o.default) {
        out[0].default = true;
    }
    out
}

pub fn default_model_dir(p: &Paths, c: &Config) -> PathBuf {
    model_dirs(p, c).into_iter().find(|o| o.default).map(|o| o.path).unwrap_or_else(|| p.vaults())
}

pub fn add_model_dir(p: &Paths, c: &mut Config, path: &Path, default: bool) {
    let r = p.rel(path);
    let mut md: Vec<J> = c.get("model_dirs").as_array().cloned().unwrap_or_default().into_iter().filter(|e| e["path"].as_str().map(|s| p.absolute(s)) != Some(p.absolute(&r))).collect();
    if default {
        for e in md.iter_mut() {
            e["default"] = J::Bool(false);
        }
    }
    md.push(json!({"path": r, "default": default}));
    c.set("model_dirs", J::Array(md));
}

pub fn remove_model_dir(p: &Paths, c: &mut Config, path: &Path) {
    let md: Vec<J> = c.get("model_dirs").as_array().cloned().unwrap_or_default().into_iter().filter(|e| e["path"].as_str().map(|s| p.absolute(s)).as_deref() != Some(path)).collect();
    c.set("model_dirs", J::Array(md));
}

/// `{name: path}` of every vault found; earlier model folders win on a name clash.
pub fn find_vaults(p: &Paths, c: &Config) -> Vec<(String, PathBuf)> {
    let mut found: Vec<(String, PathBuf)> = Vec::new();
    let add = |n: String, d: PathBuf, f: &mut Vec<(String, PathBuf)>| {
        if !f.iter().any(|(x, _)| *x == n) {
            f.push((n, d));
        }
    };
    let mut bases: Vec<PathBuf> = model_dirs(p, c).into_iter().map(|m| m.path).filter(|d| d.exists()).collect();
    bases.push(p.root.clone());
    for base in bases {
        if let Ok(rd) = std::fs::read_dir(&base) {
            let mut ds: Vec<PathBuf> = rd.filter_map(|e| e.ok().map(|e| e.path())).collect();
            ds.sort();
            for d in ds {
                if is_vault(&d) {
                    add(d.file_name().unwrap().to_string_lossy().into_owned(), d, &mut found);
                }
            }
        }
        if is_vault(&base) {
            add(base.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default(), base.clone(), &mut found);
        }
    }
    if let Ok(t) = std::fs::read_to_string(p.data.join("drafts.txt")) {
        for l in t.lines().map(str::trim).filter(|l| !l.is_empty() && !l.starts_with('#')) {
            let d = p.absolute(l);
            if is_vault(&d) {
                add(d.file_name().unwrap().to_string_lossy().into_owned(), d, &mut found);
            }
        }
    }
    found
}

/// A vault by name (searched in every model folder) or by path (absolute or relative to ROOT).
pub fn resolve_vault(p: &Paths, c: &Config, name_or_path: &str) -> anyhow::Result<PathBuf> {
    let q = p.absolute(name_or_path);
    if is_vault(&q) {
        return Ok(q);
    }
    let nm = Path::new(name_or_path).file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    find_vaults(p, c).into_iter().find(|(n, _)| *n == nm).map(|x| x.1).ok_or_else(|| anyhow::anyhow!("vault '{name_or_path}' not found in any model folder"))
}

#[cfg(test)]
mod tests {
    use super::strip_verbatim;
    use std::path::Path;
    #[test]
    fn verbatim() {
        assert_eq!(strip_verbatim(Path::new(r"\\?\D:\models")), Path::new(r"D:\models"));
        assert_eq!(strip_verbatim(Path::new(r"\\?\UNC\srv\share\x")), Path::new(r"\\srv\share\x"));
        assert_eq!(strip_verbatim(Path::new("/home/x")), Path::new("/home/x"));
    }
}
