//! Tokenizer + chat template, rendered the way `transformers.apply_chat_template` renders it (Jinja with trim_blocks and
//! lstrip_blocks, `tojson` = Python `json.dumps(ensure_ascii=False)`, `raise_exception`, `strftime_now`, Python string and
//! dict methods), so prompts are token-for-token those of the Python harness.

use anyhow::{anyhow, Context, Result};
use minijinja::value::{Kwargs, Value, ValueKind};
use minijinja::{Environment, Error, ErrorKind};
use serde_json::Value as J;
use std::path::Path;

pub struct Tok {
    pub tk: tokenizers::Tokenizer,
    pub chat_template: Option<String>,
    /// special tokens passed to the template (bos_token, eos_token, ...)
    pub special: serde_json::Map<String, J>,
}

fn tok_str(v: &J) -> Option<String> {
    match v {
        J::String(s) => Some(s.clone()),
        J::Object(o) => o.get("content").and_then(|c| c.as_str()).map(str::to_string),
        _ => None,
    }
}

impl Tok {
    /// From a vault (manifest sections or loose files).
    pub fn from_vault(v: &cube_vault::Vault) -> Result<Tok> {
        let tj = v.file("tokenizer.json").context("the vault has no tokenizer.json")?;
        let cfg: J = v.file("tokenizer_config.json").and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or(J::Null);
        let jinja = v.file("chat_template.jinja").map(|b| String::from_utf8_lossy(&b).into_owned());
        Self::build(&tj, &cfg, jinja, v.dir.join("special_tokens_map.json").as_path())
    }

    /// From a Hugging Face model folder.
    pub fn from_dir(d: &Path) -> Result<Tok> {
        let tj = std::fs::read(d.join("tokenizer.json")).with_context(|| format!("no tokenizer.json in {}", d.display()))?;
        let cfg: J = std::fs::read(d.join("tokenizer_config.json")).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or(J::Null);
        let jinja = std::fs::read_to_string(d.join("chat_template.jinja")).ok();
        Self::build(&tj, &cfg, jinja, &d.join("special_tokens_map.json"))
    }

    fn build(tj: &[u8], cfg: &J, jinja: Option<String>, stm: &Path) -> Result<Tok> {
        let tk = tokenizers::Tokenizer::from_bytes(tj).map_err(|e| anyhow!("tokenizer.json: {e}"))?;
        // transformers: chat_template.jinja wins over tokenizer_config.json; a list of named templates -> "default"
        let ct = jinja.or_else(|| match cfg.get("chat_template") {
            Some(J::String(s)) => Some(s.clone()),
            Some(J::Array(a)) => a
                .iter()
                .find(|t| t.get("name").and_then(|n| n.as_str()) == Some("default"))
                .or_else(|| a.first())
                .and_then(|t| t.get("template").and_then(|x| x.as_str()).map(str::to_string)),
            _ => None,
        });
        let mut special = serde_json::Map::new();
        let smap: J = std::fs::read(stm).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or(J::Null);
        for k in ["bos_token", "eos_token", "unk_token", "pad_token", "sep_token", "cls_token", "mask_token"] {
            if let Some(s) = cfg.get(k).and_then(tok_str).or_else(|| smap.get(k).and_then(tok_str)) {
                special.insert(k.into(), J::String(s));
            }
        }
        Ok(Tok { tk, chat_template: ct, special })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        Ok(self.tk.encode(text, false).map_err(|e| anyhow!("tokenize: {e}"))?.get_ids().to_vec())
    }
    pub fn decode(&self, ids: &[u32], skip_special: bool) -> String {
        self.tk.decode(ids, skip_special).unwrap_or_default()
    }
    pub fn token_id(&self, s: &str) -> Option<u32> {
        self.tk.token_to_id(s)
    }

    /// Render the chat template. `extra`: template variables such as enable_thinking.
    pub fn render(&self, messages: &J, tools: Option<&J>, add_generation_prompt: bool, extra: &serde_json::Map<String, J>) -> Result<String> {
        let src = self.chat_template.as_deref().context("this model has no chat template")?;
        render_template(src, messages, tools, add_generation_prompt, &self.special, extra)
    }

    pub fn apply(&self, messages: &J, tools: Option<&J>, extra: &serde_json::Map<String, J>) -> Result<Vec<u32>> {
        let s = self.render(messages, tools, true, extra)?;
        self.encode(&s)
    }
}

// ---------------------------------------------------------------- Python json.dumps
fn py_str(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

fn py_float(f: f64) -> String {
    if f.is_nan() {
        return "NaN".into();
    }
    if f.is_infinite() {
        return if f > 0.0 { "Infinity".into() } else { "-Infinity".into() };
    }
    crate::util::py_float(f)
}

fn dumps(v: &Value, indent: Option<usize>, sort: bool, seps: (&str, &str), depth: usize, out: &mut String) -> Result<(), Error> {
    let nl = |out: &mut String, d: usize| {
        if let Some(n) = indent {
            out.push('\n');
            for _ in 0..n * d {
                out.push(' ');
            }
        }
    };
    match v.kind() {
        ValueKind::Undefined | ValueKind::None => out.push_str("null"),
        ValueKind::Bool => out.push_str(if v.is_true() { "true" } else { "false" }),
        ValueKind::Number => {
            if let Some(i) = v.as_i64() {
                if v.to_string().contains('.') {
                    out.push_str(&py_float(i as f64));
                } else {
                    out.push_str(&i.to_string());
                }
            } else {
                let f: f64 = f64::try_from(v.clone()).unwrap_or(0.0);
                out.push_str(&py_float(f));
            }
        }
        ValueKind::String => py_str(v.as_str().unwrap_or(""), out),
        ValueKind::Seq | ValueKind::Iterable => {
            let items: Vec<Value> = v.try_iter()?.collect();
            if items.is_empty() {
                out.push_str("[]");
                return Ok(());
            }
            out.push('[');
            for (i, x) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(if indent.is_some() { seps.0.trim_end() } else { seps.0 });
                }
                nl(out, depth + 1);
                dumps(x, indent, sort, seps, depth + 1, out)?;
            }
            nl(out, depth);
            out.push(']');
        }
        ValueKind::Map => {
            let mut keys: Vec<Value> = v.try_iter()?.collect();
            if keys.is_empty() {
                out.push_str("{}");
                return Ok(());
            }
            if sort {
                keys.sort_by_key(|k| k.to_string());
            }
            out.push('{');
            for (i, k) in keys.iter().enumerate() {
                if i > 0 {
                    out.push_str(if indent.is_some() { seps.0.trim_end() } else { seps.0 });
                }
                nl(out, depth + 1);
                match k.as_str() {
                    Some(s) => py_str(s, out),
                    None => py_str(&k.to_string(), out),
                }
                out.push_str(seps.1);
                dumps(&v.get_item(k)?, indent, sort, seps, depth + 1, out)?;
            }
            nl(out, depth);
            out.push('}');
        }
        _ => py_str(&v.to_string(), out),
    }
    Ok(())
}

fn tojson(v: Value, kw: Kwargs) -> Result<Value, Error> {
    let indent: Option<usize> = kw.get("indent")?;
    let sort: Option<bool> = kw.get("sort_keys")?;
    let seps: Option<Vec<String>> = kw.get("separators")?;
    let _: Option<bool> = kw.get("ensure_ascii")?;
    kw.assert_all_used()?;
    let (a, b) = match &seps {
        Some(s) if s.len() == 2 => (s[0].clone(), s[1].clone()),
        _ => (if indent.is_some() { ",".into() } else { ", ".into() }, ": ".into()),
    };
    let mut out = String::new();
    dumps(&v, indent, sort.unwrap_or(false), (&a, &b), 0, &mut out)?;
    Ok(Value::from_safe_string(out))
}

pub fn render_template(src: &str, messages: &J, tools: Option<&J>, add_generation_prompt: bool, special: &serde_json::Map<String, J>, extra: &serde_json::Map<String, J>) -> Result<String> {
    let mut env = Environment::new();
    env.set_trim_blocks(true);
    env.set_lstrip_blocks(true);
    env.set_keep_trailing_newline(false);
    env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
    env.add_filter("tojson", tojson);
    env.add_function("raise_exception", |msg: String| -> Result<Value, Error> { Err(Error::new(ErrorKind::InvalidOperation, msg)) });
    env.add_function("strftime_now", |fmt: String| -> String { chrono::Local::now().format(&fmt).to_string() });
    env.add_template("chat", src).map_err(|e| anyhow!("chat template: {e:#}"))?;
    let mut ctx = serde_json::Map::new();
    for (k, v) in special {
        ctx.insert(k.clone(), v.clone());
    }
    ctx.insert("messages".into(), messages.clone());
    if let Some(t) = tools {
        ctx.insert("tools".into(), t.clone());
    }
    ctx.insert("add_generation_prompt".into(), J::Bool(add_generation_prompt));
    for (k, v) in extra {
        ctx.insert(k.clone(), v.clone());
    }
    let t = env.get_template("chat")?;
    t.render(Value::from_serialize(&J::Object(ctx))).map_err(|e| anyhow!("chat template: {e:#}"))
}
