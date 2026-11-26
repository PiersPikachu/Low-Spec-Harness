//! Reading a model's reply: reasoning / answer / tool calls for the harmony format (gpt-oss) and the `<think>` +
//! `<tool_call>` format (Qwen3, Hermes 4); the tool call still being written (for live file saving); chunk cutting.

use regex::Regex;
use serde_json::{json, Map, Value as J};
use std::sync::OnceLock;

pub const HARMONY_END: [&str; 3] = ["<|end|>", "<|return|>", "<|call|>"];
pub const CUT_MSG: &str = "cut off before it finished. Your call was too long for one reply: write the file in parts - artifact action 'write' with the first part (under ~3000 characters), then action 'append' for each next part.";

pub const CHUNK: usize = 3000;
pub const HARD_CHUNK: usize = 12000;
pub const MAX_CHUNKS: usize = 60;
pub const LIVE_EVERY: f64 = 0.5;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Reply {
    pub thinking: String,
    pub answer: String,
    /// (tool name, arguments object)
    pub calls: Vec<(String, J)>,
    pub harmony: bool,
}

fn re(s: &'static OnceLock<Regex>, p: &str) -> &'static Regex {
    s.get_or_init(|| Regex::new(p).unwrap())
}
static TOOL_RE: OnceLock<Regex> = OnceLock::new();
static CH_RE: OnceLock<Regex> = OnceLock::new();
static TO_RE: OnceLock<Regex> = OnceLock::new();

/// Python `json.loads(s, strict=False)`: control characters allowed inside strings.
pub fn loads_lenient(s: &str) -> Result<J, serde_json::Error> {
    match serde_json::from_str::<J>(s) {
        Ok(v) => Ok(v),
        Err(e) => {
            // escape raw control characters inside strings, then retry
            let mut out = String::with_capacity(s.len() + 16);
            let mut in_str = false;
            let mut esc = false;
            for c in s.chars() {
                if in_str {
                    if esc {
                        esc = false;
                        out.push(c);
                        continue;
                    }
                    match c {
                        '\\' => {
                            esc = true;
                            out.push(c)
                        }
                        '"' => {
                            in_str = false;
                            out.push(c)
                        }
                        '\n' => out.push_str("\\n"),
                        '\r' => out.push_str("\\r"),
                        '\t' => out.push_str("\\t"),
                        c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
                        c => out.push(c),
                    }
                } else {
                    if c == '"' {
                        in_str = true;
                    }
                    out.push(c);
                }
            }
            serde_json::from_str::<J>(&out).map_err(|_| e)
        }
    }
}

fn err_args(msg: String, cut: bool) -> J {
    json!({"_error": msg, "_cut": cut})
}

pub fn parse_tool_calls(text: &str) -> Vec<(String, J)> {
    let r = re(&TOOL_RE, r"(?s)<tool_call>\s*(\{.*?\})\s*</tool_call>");
    let mut calls = Vec::new();
    for m in r.captures_iter(text) {
        let blob = &m[1];
        match loads_lenient(blob) {
            Ok(c) => {
                let name = c.get("name").and_then(|n| n.as_str()).unwrap_or("").to_string();
                let mut args = c.get("arguments").cloned().unwrap_or(json!({}));
                if let J::String(s) = &args {
                    args = loads_lenient(if s.is_empty() { "{}" } else { s }).unwrap_or_else(|_| json!({"_error": format!("could not parse tool call JSON: {}", trunc(blob, 200))}));
                }
                calls.push((name, args));
            }
            Err(_) => calls.push((String::new(), json!({"_error": format!("could not parse tool call JSON: {}", trunc(blob, 200))}))),
        }
    }
    if let Some(i) = text.rfind("<tool_call>") {
        if !text[i..].contains("</tool_call>") {
            calls.push((String::new(), err_args(format!("the tool call was {CUT_MSG}"), true)));
        }
    }
    calls
}

pub fn trunc(s: &str, n: usize) -> &str {
    match s.char_indices().nth(n) {
        Some((i, _)) => &s[..i],
        None => s,
    }
}

pub fn parse_reply(raw: &str) -> Reply {
    if raw.contains("<|channel|>") || raw.contains("<|message|>") {
        let (mut think, mut ans, mut calls) = (Vec::new(), Vec::new(), Vec::new());
        let ch_re = re(&CH_RE, r"<\|channel\|>(\w+)");
        let to_re = re(&TO_RE, r"to=functions\.([\w.\-]+)");
        for seg in raw.split("<|start|>") {
            let Some((head, body)) = seg.split_once("<|message|>") else { continue };
            let mut body = body;
            for e in HARMONY_END {
                if let Some((b, _)) = body.split_once(e) {
                    body = b;
                }
            }
            let ch = ch_re.captures(head).map(|c| c[1].to_string());
            if let Some(to) = to_re.captures(head) {
                let b = body.trim();
                let args = match loads_lenient(if b.is_empty() { "{}" } else { b }) {
                    Ok(J::Object(o)) => J::Object(o),
                    Ok(v) => json!({ "value": v }),
                    Err(ex) => {
                        let cut = !b.ends_with('}');
                        err_args(
                            if cut {
                                format!("the tool call was {CUT_MSG}")
                            } else {
                                format!("tool arguments are not valid JSON ({ex}). Escape quotes as \\\" and new lines as \\n inside strings.")
                            },
                            cut,
                        )
                    }
                };
                calls.push((to[1].to_string(), args));
            } else if ch.as_deref() == Some("analysis") {
                think.push(body.to_string());
            } else {
                ans.push(body.to_string());
            }
        }
        return Reply { thinking: think.join("\n").trim().to_string(), answer: ans.concat().trim().to_string(), calls, harmony: true };
    }
    let (th, ans) = if let Some((a, b)) = raw.split_once("</think>") {
        (a.rsplit_once("<think>").map_or(a, |x| x.1).to_string(), b.to_string())
    } else if let Some((_, b)) = raw.split_once("<think>") {
        (b.to_string(), String::new())
    } else {
        (String::new(), raw.to_string())
    };
    let answer = ans.trim().to_string();
    Reply { thinking: th.trim().to_string(), calls: parse_tool_calls(&answer), answer, harmony: false }
}

/// Value of `"key": "..."` in possibly unterminated JSON text, decoded as far as it goes (None if the key has not started).
pub fn partial_json_str(text: &str, key: &str) -> Option<String> {
    let r = Regex::new(&format!(r#""{}"\s*:\s*""#, regex::escape(key))).ok()?;
    let m = r.find(text)?;
    let b: Vec<char> = text[m.end()..].chars().collect();
    let mut out = String::new();
    let mut i = 0;
    while i < b.len() {
        let ch = b[i];
        if ch == '"' {
            break;
        }
        if ch == '\\' {
            if i + 1 >= b.len() {
                break;
            }
            let c = b[i + 1];
            if c == 'u' {
                if i + 6 > b.len() {
                    break;
                }
                let h: String = b[i + 2..i + 6].iter().collect();
                if let Ok(v) = u32::from_str_radix(&h, 16) {
                    if let Some(c) = char::from_u32(v) {
                        out.push(c);
                    }
                }
                i += 6;
                continue;
            }
            out.push(match c {
                'n' => '\n',
                't' => '\t',
                'r' => '\r',
                'b' => '\u{8}',
                'f' => '\u{c}',
                c => c,
            });
            i += 2;
            continue;
        }
        out.push(ch);
        i += 1;
    }
    Some(out)
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct PartialCall {
    pub name: String,
    pub action: Option<String>,
    pub target: Option<String>,
    pub path: Option<String>,
    pub text: Option<String>,
    pub chars: usize,
    pub kind: Option<String>,
    pub title: Option<String>,
}

/// The tool call being written at the end of `raw`, if any.
pub fn partial_call(raw: &str) -> Option<PartialCall> {
    let (name, body): (String, String);
    let harmony = raw.contains("<|channel|>") || raw.contains("<|message|>");
    if harmony {
        let seg = raw.rsplit("<|start|>").next().unwrap_or("");
        let (head, b) = seg.split_once("<|message|>")?;
        let to = re(&TO_RE, r"to=functions\.([\w.\-]+)").captures(head)?;
        if HARMONY_END.iter().any(|e| b.contains(e)) {
            return None;
        }
        name = to[1].to_string();
        body = b.to_string();
    } else {
        let i = raw.rfind("<tool_call>")?;
        if raw[i..].contains("</tool_call>") {
            return None;
        }
        body = raw[i + "<tool_call>".len()..].to_string();
        name = partial_json_str(&body, "name").filter(|s| !s.is_empty()).unwrap_or_else(|| "?".into());
    }
    // fields of the arguments object (in the <tool_call> format the body also holds the tool's own "name" first; the Python
    // harness read that one as the artifact name, so live saves of Qwen-style calls went to an artifact called 'artifact')
    let args_body = match body.find("\"arguments\"") {
        Some(i) if !harmony => body[i + "\"arguments\"".len()..].to_string(),
        _ if !harmony => String::new(),
        _ => body.clone(),
    };
    let f = |k: &str| partial_json_str(&args_body, k);
    let text = f("content").or_else(|| f("new")).or_else(|| f("code")).or_else(|| f("body"));
    Some(PartialCall {
        target: if name != "?" { f("name") } else { None },
        action: f("action"),
        path: f("path"),
        text,
        chars: body.chars().count(),
        kind: f("type"),
        title: f("title"),
        name,
    })
}

/// Where to cut streamed file content (in chars): end of the line that crosses CHUNK (or HARD_CHUNK); final: last newline.
pub fn chunk_cut(text: &str, fin: bool) -> Option<usize> {
    let c: Vec<char> = text.chars().collect();
    if fin {
        return Some(c.iter().rposition(|&x| x == '\n').map_or(c.len(), |i| i + 1));
    }
    if c.len() >= HARD_CHUNK {
        return Some(HARD_CHUNK);
    }
    if c.len() < CHUNK {
        return None;
    }
    c[CHUNK..].iter().position(|&x| x == '\n').map(|i| CHUNK + i + 1)
}

/// Longest prefix (in chars) that fits in `room` characters and ends on a whole line (or exactly room without a newline).
pub fn cap_cut(text: &str, room: i64) -> usize {
    if room <= 0 {
        return 0;
    }
    let c: Vec<char> = text.chars().collect();
    let room = room as usize;
    if c.len() <= room {
        return c.len();
    }
    c[..room].iter().rposition(|&x| x == '\n').map_or(room, |i| i + 1)
}

pub fn char_prefix(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}

static OUTLINE: OnceLock<Regex> = OnceLock::new();

/// Outline (headings / def / class / function / main HTML elements) + the last `last` lines of a file.
pub fn file_context(text: &str, last: usize, max_items: usize) -> String {
    let r = re(
        &OUTLINE,
        r"^\s*(#{1,6}\s+\S.*|(?:async\s+)?def\s+\w+.*|class\s+\w+.*|(?:export\s+)?(?:async\s+)?function\s*\*?\s*\w+.*|(?:export\s+)?class\s+\w+.*|(?:const|let|var)\s+\w+\s*=\s*(?:async\s*)?(?:\(|function|class).*|<(?:h[1-6]|section|header|footer|main|nav|form|canvas|svg|script|style|template)\b[^>]*>.*)$",
    );
    let lines: Vec<&str> = py_splitlines(text);
    let mut out: Vec<String> = Vec::new();
    for (i, ln) in lines.iter().enumerate() {
        if r.is_match(ln) {
            out.push(format!("  L{}: {}", i + 1, trunc(ln.trim(), 100)));
        }
    }
    if out.len() > max_items {
        let n = out.len();
        let mut o2: Vec<String> = out[..max_items / 2].to_vec();
        o2.push(format!("  ... {} more ...", n - max_items));
        o2.extend_from_slice(&out[n - max_items / 2..]);
        out = o2;
    }
    let start = lines.len().saturating_sub(last);
    let tail: Vec<String> = lines[start..].iter().enumerate().map(|(k, l)| format!("{:5}| {}", start + k + 1, l)).collect();
    format!(
        "File so far: {} lines, {} characters.\nOutline:\n{}\nLast {} lines:\n{}",
        lines.len(),
        text.chars().count(),
        if out.is_empty() { "  (no headings, functions or sections yet)".to_string() } else { out.join("\n") },
        last.min(lines.len()),
        tail.join("\n")
    )
}

/// Python `str.splitlines()` for the common separators.
pub fn py_splitlines(text: &str) -> Vec<&str> {
    let mut v = Vec::new();
    let mut start = 0;
    let b = text.as_bytes();
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'\n' || b[i] == b'\r' {
            v.push(&text[start..i]);
            if b[i] == b'\r' && i + 1 < b.len() && b[i + 1] == b'\n' {
                i += 1;
            }
            start = i + 1;
        }
        i += 1;
    }
    if start < b.len() {
        v.push(&text[start..]);
    }
    v
}

static BIG_ARG: OnceLock<Regex> = OnceLock::new();

/// Shorten the file content of older tool calls (the files hold it); the newest `keep_last` calls stay complete.
pub fn compact_history(msgs: &[J], keep_last: usize) -> Vec<J> {
    let idx: Vec<usize> = msgs
        .iter()
        .enumerate()
        .filter(|(_, m)| {
            m.get("tool_calls").map_or(false, |t| t.as_array().map_or(false, |a| !a.is_empty()))
                || (m["role"] == "assistant" && m.get("content").and_then(|c| c.as_str()).map_or(false, |c| c.contains("<tool_call>")))
        })
        .map(|(i, _)| i)
        .collect();
    let mut out = msgs.to_vec();
    let upto = idx.len().saturating_sub(keep_last);
    let shorten = |s: &str| format!("{} ... [{} characters, saved by the tool]", char_prefix(s, 120), s.chars().count());
    for &i in &idx[..if keep_last > 0 { upto } else { idx.len() }] {
        let m = &mut out[i];
        if let Some(J::Array(tcs)) = m.get_mut("tool_calls") {
            for tc in tcs.iter_mut() {
                if let Some(J::Object(a)) = tc.get_mut("function").and_then(|f| f.get_mut("arguments")) {
                    for k in ["content", "new", "code", "body"] {
                        if let Some(J::String(s)) = a.get(k) {
                            if s.chars().count() > 300 {
                                let v = shorten(s);
                                a.insert(k.into(), J::String(v));
                            }
                        }
                    }
                }
            }
        } else if let Some(J::String(c)) = m.get("content") {
            let r = re(&BIG_ARG, r#"("(?:content|new|code|body)"\s*:\s*")((?:[^"\\]|\\.){300,})""#);
            let v = r
                .replace_all(c, |g: &regex::Captures| format!("{}{} ... [{} characters, saved by the tool]\"", &g[1], char_prefix(&g[2], 120), g[2].chars().count()))
                .into_owned();
            m["content"] = J::String(v);
        }
    }
    out
}

/// `(plugin name, args)` for a call whose name may be decorated or may be an action of a plugin.
pub fn resolve_tool(tools: &[(String, Vec<String>)], name: &str, args: &J) -> (String, J) {
    let mut args = match args {
        J::Object(o) => J::Object(o.clone()),
        _ => J::Object(Map::new()),
    };
    let mut n = name.trim();
    for pre in ["functions.", "tools.", "tool."] {
        if let Some(r) = n.strip_prefix(pre) {
            n = r;
        }
    }
    let has = |x: &str| tools.iter().any(|(t, _)| t == x);
    if has(n) {
        return (n.to_string(), args);
    }
    for sep in ['.', '_', '/', ':'] {
        if let Some((head, tail)) = n.split_once(sep) {
            if let Some((_, acts)) = tools.iter().find(|(t, _)| t == head) {
                if acts.iter().any(|a| a == tail) {
                    args.as_object_mut().unwrap().entry("action").or_insert(J::String(tail.into()));
                }
                return (head.to_string(), args);
            }
        }
    }
    let owners: Vec<&String> = tools.iter().filter(|(_, a)| a.iter().any(|x| x == n)).map(|(t, _)| t).collect();
    if owners.len() == 1 {
        args.as_object_mut().unwrap().entry("action").or_insert(J::String(n.into()));
        return (owners[0].clone(), args);
    }
    (n.to_string(), args)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn think_and_calls() {
        let r = parse_reply("<think>\nplan\n</think>\n\nOK <tool_call>\n{\"name\": \"calculator\", \"arguments\": {\"expression\": \"2+2\"}}\n</tool_call>");
        assert_eq!(r.thinking, "plan");
        assert_eq!(r.calls, vec![("calculator".to_string(), json!({"expression": "2+2"}))]);
        let r = parse_reply("<|channel|>analysis<|message|>think<|end|><|start|>assistant<|channel|>commentary to=functions.artifact <|constrain|>json<|message|>{\"action\": \"list\"}<|call|>");
        assert!(r.harmony);
        assert_eq!(r.thinking, "think");
        assert_eq!(r.calls[0].0, "artifact");
        let r = parse_reply("x <tool_call>{\"name\": \"a\", \"arguments\": {\"content\": \"abc");
        assert!(r.calls[0].1["_cut"] == true);
    }
    #[test]
    fn partial() {
        let p = partial_call("<tool_call>\n{\"name\": \"artifact\", \"arguments\": {\"action\": \"write\", \"name\": \"site\", \"path\": \"index.html\", \"content\": \"<h1>\\\"hi\\\"\\nline").unwrap();
        assert_eq!(p.name, "artifact");
        assert_eq!(p.target.as_deref(), Some("site"));
        assert_eq!(p.text.as_deref(), Some("<h1>\"hi\"\nline"));
        assert_eq!(chunk_cut("a\nb", true), Some(2));
        assert_eq!(cap_cut("ab\ncd\nef", 7), 6);
    }
    #[test]
    fn lenient() {
        assert_eq!(loads_lenient("{\"a\": \"x\ny\"}").unwrap(), json!({"a": "x\ny"}));
    }
    #[test]
    fn resolve() {
        let t = vec![("artifact".to_string(), vec!["list".to_string(), "create".to_string()]), ("clock".to_string(), vec![])];
        assert_eq!(resolve_tool(&t, "functions.artifact", &json!({})).0, "artifact");
        assert_eq!(resolve_tool(&t, "artifact.create", &json!({})), ("artifact".to_string(), json!({"action": "create"})));
        assert_eq!(resolve_tool(&t, "list", &json!({})), ("artifact".to_string(), json!({"action": "list"})));
    }
}
