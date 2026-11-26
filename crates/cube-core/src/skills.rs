//! Skills: instruction snippets in data/skills/*.md (front matter + body). '/name' at the start of a message inserts one;
//! 'always_on' adds it to the system prompt; 'applies_when: tools' skills are added whenever tools are on.

use regex::Regex;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Skill {
    pub name: String,
    pub description: String,
    pub always_on: bool,
    pub author: String,
    pub applies_when: String,
    pub body: String,
    pub file: PathBuf,
}

pub fn parse(text: &str) -> (Vec<(String, String)>, String) {
    let re = Regex::new(r"(?s)^---\s*\n(.*?)\n---\s*\n?(.*)$").unwrap();
    if let Some(c) = re.captures(text) {
        let meta = c[1]
            .lines()
            .filter_map(|l| l.split_once(':').map(|(k, v)| (k.trim().to_string(), v.trim().to_string())))
            .collect();
        return (meta, c[2].trim().to_string());
    }
    (Vec::new(), text.trim().to_string())
}

pub fn author(f: &Path) -> String {
    let t = std::fs::read_to_string(f).unwrap_or_default();
    parse(&t).0.into_iter().find(|(k, _)| k == "author").map(|x| x.1).unwrap_or_default()
}

pub fn load(dir: &Path) -> Vec<Skill> {
    let mut fs: Vec<PathBuf> = std::fs::read_dir(dir).map(|r| r.flatten().map(|e| e.path()).filter(|p| p.extension().map_or(false, |e| e == "md")).collect()).unwrap_or_default();
    fs.sort();
    let mut out: Vec<Skill> = Vec::new();
    for f in fs {
        let t = std::fs::read_to_string(&f).unwrap_or_default();
        let (meta, body) = parse(&t);
        let g = |k: &str| meta.iter().find(|(a, _)| a == k).map(|x| x.1.clone());
        let name = g("name").unwrap_or_else(|| f.file_stem().unwrap().to_string_lossy().into_owned());
        let s = Skill {
            description: g("description").unwrap_or_default(),
            always_on: g("always_on").map_or(false, |v| v.to_lowercase() == "true"),
            author: g("author").unwrap_or_default(),
            applies_when: g("applies_when").unwrap_or_default(),
            body,
            file: f,
            name: name.clone(),
        };
        out.retain(|x| x.name != name);
        out.push(s);
    }
    out
}

pub fn save(dir: &Path, name: &str, description: &str, always_on: bool, body: &str, old_name: Option<&str>) -> std::io::Result<String> {
    let mut n: String = name.trim().chars().map(|c| if c.is_ascii_alphanumeric() || c == '_' || c == '-' { c } else { '-' }).collect();
    if n.is_empty() {
        n = "skill".into();
    }
    let _ = std::fs::create_dir_all(dir);
    let mut keep = String::new();
    let mut srcs = vec![dir.join(format!("{n}.md"))];
    if let Some(o) = old_name {
        srcs.push(dir.join(format!("{o}.md")));
    }
    for s in &srcs {
        if s.exists() {
            let (meta, _) = parse(&std::fs::read_to_string(s).unwrap_or_default());
            for k in ["author", "applies_when"] {
                if let Some((_, v)) = meta.iter().find(|(a, _)| a == k).filter(|x| !x.1.is_empty()) {
                    keep.push_str(&format!("{k}: {v}\n"));
                }
            }
            break;
        }
    }
    if let Some(o) = old_name {
        if o != n && dir.join(format!("{o}.md")).exists() {
            std::fs::remove_file(dir.join(format!("{o}.md")))?;
        }
    }
    std::fs::write(
        dir.join(format!("{n}.md")),
        format!("---\nname: {n}\ndescription: {}\nalways_on: {}\n{keep}---\n{}\n", description.trim(), if always_on { "true" } else { "false" }, body.trim()),
    )?;
    Ok(n)
}

pub fn delete(dir: &Path, name: &str) {
    let _ = std::fs::remove_file(dir.join(format!("{name}.md")));
}

/// '/name rest of message' -> skill body + rest; several /names may lead the message.
pub fn expand(text: &str, skills: &[Skill]) -> (String, Vec<String>) {
    let re = Regex::new(r"^\s*/([A-Za-z0-9_\-]+)\b\s*").unwrap();
    let mut used: Vec<String> = Vec::new();
    let mut t = text.to_string();
    loop {
        let Some(c) = re.captures(&t) else { break };
        let n = c[1].to_string();
        if !skills.iter().any(|s| s.name == n) {
            break;
        }
        let end = c.get(0).unwrap().end();
        used.push(n);
        t = t[end..].to_string();
    }
    if used.is_empty() {
        return (t, used);
    }
    let bodies: Vec<String> = used.iter().map(|u| skills.iter().find(|s| s.name == *u).unwrap().body.clone()).collect();
    (bodies.join("\n\n") + &if t.trim().is_empty() { String::new() } else { format!("\n\n{t}") }, used)
}

/// Bodies of 'applies_when: tools' skills with {tools} filled in, keeping only the sections for this model's call format.
pub fn tool_skill_text(dir: &Path, tools: &[serde_json::Value], harmony: bool) -> String {
    let mut lines = Vec::new();
    for t in tools {
        let f = &t["function"];
        let acts = f["parameters"]["properties"]["action"]["enum"].as_array().map(|a| a.iter().filter_map(|x| x.as_str()).collect::<Vec<_>>().join(", ")).unwrap_or_default();
        let d = f["description"].as_str().unwrap_or("");
        let first: String = d.split(". ").next().unwrap_or("").chars().take(140).collect();
        lines.push(format!("- {}{}: {first}", f["name"].as_str().unwrap_or(""), if acts.is_empty() { String::new() } else { format!(" (actions: {acts})") }));
    }
    let fmt = Regex::new(r"^##\s*format:\s*(\w+)").unwrap();
    let mut out = Vec::new();
    for sk in load(dir) {
        if sk.applies_when != "tools" {
            continue;
        }
        let body = sk.body.replace("{tools}", &lines.join("\n"));
        let mut keep = Vec::new();
        let mut on = true;
        for ln in body.lines() {
            if let Some(c) = fmt.captures(ln) {
                on = (&c[1] == "harmony") == harmony;
                continue;
            }
            if ln.starts_with("## ") {
                on = true;
            }
            if on {
                keep.push(ln);
            }
        }
        out.push(keep.join("\n").trim().to_string());
    }
    out.join("\n\n")
}
