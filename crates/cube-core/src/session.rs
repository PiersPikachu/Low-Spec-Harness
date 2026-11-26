//! One loaded model: the engine (in process), its tokenizer and chat template, and the reply loop that runs tool calls —
//! a port of `harness_core.Session` (KV reuse across turns, context trimming into a memory artifact, streamed file writes cut
//! into 3000-character calls, per-reply file growth cap, live saving of a file being written).

use crate::paths::{Config, Paths};
use crate::reply::{self, cap_cut, char_prefix, chunk_cut, compact_history, file_context, parse_reply, partial_call, resolve_tool, PartialCall, LIVE_EVERY, MAX_CHUNKS};
use crate::template::Tok;
use crate::tools::Tools;
use anyhow::{Context, Result};
use cube_engine::{End, Engine, Options};
use cube_vault::Vault;
use serde_json::{json, Map, Value as J};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

/// What the reply loop reports while it runs (the Python harness's `on_text(kind, payload)`).
#[derive(Clone, Debug)]
pub enum Event {
    Thinking(String),
    Assistant(String),
    AssistantNext,
    Tool(String),
    /// {"ev": "round" | "call_stream" | "call" | "result", ...}
    Act(J),
}

pub struct Session {
    pub vault_path: PathBuf,
    pub name: String,
    pub tok: Tok,
    pub eng: Engine,
    pub eos: Vec<u32>,
    pub chat_mode: bool,
    pub cfg: Config,
    pub banner: String,
    pub how: String,
    /// tokens currently in the engine's KV cache
    fed: Vec<u32>,
    /// set from any thread to end the running generation after the next token
    pub stop: Arc<AtomicBool>,
}

fn kw_think(think: bool) -> Map<String, J> {
    let mut m = Map::new();
    m.insert("enable_thinking".into(), J::Bool(think));
    m.insert("thinking".into(), J::Bool(think));
    m
}

pub fn sha1_hex8(s: &str) -> String {
    let mut h = sha1_smol::Sha1::new();
    h.update(s.as_bytes());
    h.digest().to_string()[..8].to_string()
}

impl Session {
    pub fn open(p: &Paths, cfg: &Config) -> Result<Session> {
        let vp = crate::paths::resolve_vault(p, cfg, &cfg.str("vault"))?;
        let vault = Vault::open(&vp).with_context(|| format!("cannot open vault {}", vp.display()))?;
        let tok = Tok::from_vault(&vault).with_context(|| format!("no tokenizer in {} (repack, or copy the model's tokenizer files into the vault folder)", vp.display()))?;
        let m = vault.meta().clone();
        let eos: Vec<u32> = m.eos.iter().filter(|&&e| e >= 0).map(|&e| e as u32).collect();
        let mut o = Options { ctx: cfg.i64("ctx").max(64) as usize, kv16: cfg.bool("kv16"), ..Default::default() };
        let ram = crate::hw::ram_mb();
        let budget_mb = cfg.f64("resident_mb");
        let kv_mb = crate::hw::kv_bytes(&m, o.ctx, o.kv16) / 1e6;
        // pin what fits: the configured amount, else free RAM minus a 1.5 GB reserve and the KV cache
        o.resident_mb = if budget_mb > 0.0 { budget_mb } else { (ram.1 - 1500.0 - kv_mb).max(0.0) * 0.8 };
        o.threads = cfg.i64("threads").max(0) as usize;
        if m.moe.is_some() {
            o.batch = 128;
        }
        let total_mb: f64 = (0..vault.n_streams()).map(|i| vault.stream(i).padded as f64).sum::<f64>() / 1e6;
        let how = if m.moe.is_some() {
            format!("mixture of experts: attention + output pinned, experts in a RAM LRU cache ({:.0} MB), the rest read from the SSD", o.resident_mb)
        } else if o.resident_mb >= total_mb {
            "CPU, all weights pinned in RAM".to_string()
        } else {
            format!("CPU, {:.0} of {:.0} MB pinned in RAM, the rest streamed from the SSD", o.resident_mb.min(total_mb), total_mb)
        };
        let banner = format!(
            "{}: {:?}, {} layers, hidden {}, vocab {}, vault {:.2} GB ({}){}",
            vp.file_name().unwrap().to_string_lossy(),
            m.arch,
            m.layers,
            m.hidden,
            m.vocab,
            total_mb / 1e3,
            if m.lossy { "lossy" } else { "lossless" },
            m.moe.as_ref().map(|x| format!(", {} experts top {}", x.experts, x.top_k)).unwrap_or_default()
        );
        let chat_mode = tok.chat_template.is_some();
        let eng = Engine::open(vault, o)?;
        Ok(Session {
            name: vp.file_name().unwrap().to_string_lossy().into_owned(),
            vault_path: vp,
            tok,
            eng,
            eos,
            chat_mode,
            cfg: cfg.clone(),
            banner,
            how,
            fed: Vec::new(),
            stop: Arc::new(AtomicBool::new(false)),
        })
    }

    /// True for models that call tools in the harmony format (gpt-oss).
    pub fn harmony(&self) -> bool {
        self.tok.chat_template.as_deref().map_or(false, |t| t.contains("<|channel|>"))
    }

    pub fn reset(&mut self) {
        self.eng.pos = 0;
        self.fed.clear();
    }

    fn ids(&self, msgs: &[J], tools: &[J]) -> Result<Vec<u32>> {
        let t = if tools.is_empty() { None } else { Some(J::Array(tools.to_vec())) };
        self.tok.apply(&J::Array(msgs.to_vec()), t.as_ref(), &kw_think(self.cfg.bool("think")))
    }

    fn decode(&self, out: &[u32]) -> String {
        let v: Vec<u32> = out.iter().copied().filter(|t| !self.eos.contains(t)).collect();
        self.tok.decode(&v, false).replace("<|endoftext|>", "")
    }

    /// Generate from `ids`, reusing the KV cache up to the longest common prefix with what was fed before.
    #[allow(clippy::too_many_arguments)]
    pub fn generate(&mut self, ids: &[u32], max_new: usize, temp: f32, topp: f32, topk: i32, seed: u64, on_tok: &mut dyn FnMut(&[u32])) -> Result<(Vec<u32>, End, cube_engine::GenStats)> {
        gen_with(&mut self.eng, &mut self.fed, &self.stop, &self.eos, ids, max_new, temp, topp, topk, seed, on_tok)
    }

    /// Append trimmed turns verbatim to the artifact memory-<id> and return the note that replaces them.
    fn archive(&self, history: &[J], dropped: &[J], tools: &Tools) -> Option<String> {
        let a = &tools.artifacts;
        let first = history.iter().find(|m| m["role"] == "user").and_then(|m| m["content"].as_str()).filter(|s| !s.is_empty()).unwrap_or("chat");
        let name = format!("memory-{}", sha1_hex8(first));
        let mut lines = Vec::new();
        for m in dropped {
            let mut body = m["content"].as_str().unwrap_or("").to_string();
            if let Some(tc) = m.get("tool_calls").and_then(|t| t.as_array()).filter(|a| !a.is_empty()) {
                let f: Vec<J> = tc.iter().map(|t| t.get("function").cloned().unwrap_or_else(|| t.clone())).collect();
                body = if body.is_empty() { String::new() } else { body + "\n" } + &serde_json::to_string(&f).unwrap();
            }
            lines.push(format!("### {}\n{}\n", m["role"].as_str().unwrap_or(""), body));
        }
        if !a.root.join(&name).join(".artifact.json").exists() {
            a.act(&json!({"action": "create", "name": name, "type": "note", "title": "Earlier conversation (moved out of the context window)"}));
        }
        a.act(&json!({"action": "append", "name": name, "content": lines.join("\n")}));
        Some(format!(
            "[Harness note: {} earlier message(s) of this conversation were moved out of the context window into the artifact '{name}' (notes.md), word for word. If you need them, read it with the artifact tool (action 'read', name '{name}').]",
            dropped.len()
        ))
    }

    /// history: [{role, content, ...}] including the new user message. Streams through `on`; runs the tool calls the model
    /// makes (enabled tools only) and continues. Returns the new messages to append and a stats line.
    pub fn reply(&mut self, p: &Paths, history: &[J], system: &str, tools: &Tools, enabled: &[String], max_tool_rounds: usize, on: &mut dyn FnMut(Event), should_stop: &dyn Fn() -> bool) -> Result<(Vec<J>, String)> {
        let cfg = self.cfg.clone();
        let schemas = tools.schemas(enabled);
        let acts = tools.actions(enabled);
        let mut system = system.to_string();
        if !schemas.is_empty() {
            let auto = crate::skills::tool_skill_text(&p.skills(), &schemas, self.harmony());
            if !auto.is_empty() {
                system = if system.is_empty() { auto } else { format!("{}\n\n{auto}", system).trim().to_string() };
            }
        }
        let max_new = cfg.i64("max_new").max(1) as usize;
        let ctx = cfg.i64("ctx").max(64) as usize;
        let mut new_msgs: Vec<J> = Vec::new();
        let mut stats = String::new();
        let (mut rnd, mut budget, mut chunks, mut conts) = (-1i64, max_tool_rounds as i64 + 1, 0usize, 0usize);
        let mut cont: Vec<u32> = Vec::new();
        let mut carry: Option<(String, String)> = None;
        let cap_chars: i64 = { let g = cfg.i64("max_file_growth"); if g > 0 { g } else { 4 * max_new as i64 } };
        let mut written: i64 = 0;
        let mut capped = false;
        tools.artifacts.live_keep();
        while rnd + 1 < budget {
            rnd += 1;
            let mut msgs: Vec<J> = if system.is_empty() { Vec::new() } else { vec![json!({"role": "system", "content": system})] };
            msgs.extend(history.iter().chain(new_msgs.iter()).filter(|m| matches!(m["role"].as_str(), Some("user" | "assistant" | "tool"))).cloned());
            let mut msgs = compact_history(&msgs, 1);
            let mut ids: Vec<u32>;
            if self.chat_mode {
                ids = self.ids(&msgs, &schemas)?;
                let s0 = usize::from(!system.is_empty());
                let mut last_user = msgs.iter().rposition(|m| m["role"] == "user").unwrap_or(0);
                let mut dropped: Vec<J> = Vec::new();
                while ids.len() + cont.len() + max_new > ctx {
                    if s0 >= last_user {
                        break; // never drop the current request
                    }
                    dropped.push(msgs.remove(s0));
                    last_user -= 1;
                    ids = self.ids(&msgs, &schemas)?;
                }
                if !dropped.is_empty() {
                    if let Some(note) = self.archive(history, &dropped, tools) {
                        msgs.insert(s0, json!({"role": "user", "content": note}));
                        ids = self.ids(&msgs, &schemas)?;
                    }
                }
            } else {
                let text: String = msgs.iter().map(|m| m["content"].as_str().unwrap_or("")).collect();
                ids = self.tok.tk.encode(text.as_str(), true).map_err(|e| anyhow::anyhow!("{e}"))?.get_ids().to_vec();
                let keep = ctx.saturating_sub(max_new + 1);
                if ids.len() > keep {
                    ids = ids[ids.len() - keep..].to_vec();
                }
            }
            let pre = cont.clone();
            if self.chat_mode {
                ids.extend_from_slice(&pre);
            }
            let (mut shown_t, mut shown_a) = carry.take().unwrap_or_default();
            let mut last_pc = (Instant::now() - std::time::Duration::from_secs(10), -1i64);
            let mut cut_at: Option<usize> = None;
            let mut cut_pc: Option<PartialCall> = None;
            let mut cut_capped = false;
            let mut cut_ui = false;
            let mut live_t = Instant::now() - std::time::Duration::from_secs(10);
            let mut lkey: Option<(String, Option<String>, String)> = None;
            on(Event::Act(json!({"ev": "round", "n": rnd + 1})));
            let seed: u64 = match cfg.get("seed") {
                J::Number(n) => n.as_u64().unwrap_or(1),
                J::String(s) if !s.trim().is_empty() => s.trim().parse().unwrap_or(1),
                _ => (std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_micros() % (1u128 << 62)) as u64,
            };
            let (temp, topp, topk) = if cfg.bool("greedy") { (0.0f32, 1.0f32, 1i32) } else { (cfg.f64("temp") as f32, cfg.f64("top_p") as f32, cfg.i64("top_k") as i32) };
            let t0 = Instant::now();
            let stop = self.stop.clone();
            let has_tools = !schemas.is_empty();
            // stream reasoning and answer separately; each only ever grows
            let (out, end, st) = {
                let tok_ref = &self.tok;
                let eos = self.eos.clone();
                let arts = tools.artifacts.clone();
                let acts_ref = &acts;
                let mut on_tok = |o: &[u32]| {
                    let mut all = pre.clone();
                    all.extend_from_slice(o);
                    let v: Vec<u32> = all.iter().copied().filter(|t| !eos.contains(t)).collect();
                    let raw_now = tok_ref.decode(&v, false).replace("<|endoftext|>", "");
                    let r = parse_reply(&raw_now);
                    let pc = if has_tools { partial_call(&raw_now) } else { None };
                    if let Some(pc) = &pc {
                        if last_pc.0.elapsed().as_secs_f64() > 0.12 || pc.chars as i64 - last_pc.1 > 400 {
                            last_pc = (Instant::now(), pc.chars as i64);
                            on(Event::Act(json!({"ev": "call_stream", "name": pc.name, "action": pc.action, "target": pc.target, "path": pc.path, "text": pc.text, "chars": pc.chars, "type": pc.kind, "title": pc.title})));
                        }
                    }
                    let art = pc.as_ref().filter(|pc| {
                        pc.text.is_some() && pc.target.is_some() && matches!(pc.action.as_deref(), Some("write" | "append" | "create")) && resolve_tool(acts_ref, &pc.name, &json!({})).0 == "artifact"
                    });
                    if let (Some(pc), None) = (art, cut_at) {
                        let text = pc.text.as_deref().unwrap();
                        let room = cap_chars - written;
                        let k = if text.chars().count() as i64 >= room {
                            cut_capped = true;
                            Some(cap_cut(text, room))
                        } else {
                            chunk_cut(text, false)
                        };
                        if let Some(k) = k {
                            // enough for one call: stop at the end of this line
                            cut_at = Some(k);
                            cut_pc = Some(pc.clone());
                            stop.store(true, Ordering::SeqCst);
                        } else if live_t.elapsed().as_secs_f64() > LIVE_EVERY {
                            live_t = Instant::now();
                            let t = pc.target.clone().unwrap();
                            lkey = Some((t.clone(), pc.path.clone(), pc.action.clone().unwrap()));
                            arts.live_update(&t, pc.path.as_deref(), pc.action.as_deref().unwrap(), text, pc.title.as_deref(), pc.kind.as_deref());
                        }
                    }
                    if should_stop() && !cut_ui {
                        cut_ui = true;
                        stop.store(true, Ordering::SeqCst);
                    }
                    if r.thinking.starts_with(&shown_t) && r.thinking.len() > shown_t.len() {
                        on(Event::Thinking(r.thinking[shown_t.len()..].to_string()));
                        shown_t = r.thinking.clone();
                    }
                    if r.answer.starts_with(&shown_a) && r.answer.len() > shown_a.len() {
                        on(Event::Assistant(r.answer[shown_a.len()..].to_string()));
                        shown_a = r.answer.clone();
                    }
                };
                gen_with(&mut self.eng, &mut self.fed, &self.stop, &self.eos, &ids, max_new, temp, topp, topk, seed, &mut on_tok)?
            };
            let mut out_all = pre.clone();
            out_all.extend_from_slice(&out);
            cont.clear();
            let mut reason = end.as_str().to_string();
            if end == End::Ctx && self.chat_mode && conts < 4 && cut_at.is_none() && !cut_ui {
                cont = out_all; // compact the history and carry on
                conts += 1;
                budget += 1;
                carry = Some((shown_t, shown_a));
                on(Event::Act(json!({"ev": "round", "n": rnd + 2})));
                continue;
            }
            let raw = self.decode(&out_all);
            let r = parse_reply(&raw);
            if cut_at.is_none() && end == End::Max && has_tools {
                // the token limit hit mid-file: save what is complete
                if let Some(pc) = partial_call(&raw) {
                    if pc.text.as_deref().map_or(false, |t| !t.is_empty())
                        && pc.target.is_some()
                        && matches!(pc.action.as_deref(), Some("write" | "append" | "create"))
                        && resolve_tool(&acts, &pc.name, &json!({})).0 == "artifact"
                    {
                        let text = pc.text.clone().unwrap();
                        let cc = cap_cut(&text, cap_chars - written);
                        let k = chunk_cut(&text, true).unwrap().min(cc);
                        cut_at = Some(k);
                        if k < text.trim_end_matches('\n').chars().count() && k == cc {
                            cut_capped = true;
                        }
                        cut_pc = Some(pc);
                    }
                }
            }
            if let Some((t, pth, a)) = lkey.take() {
                tools.artifacts.live_end(&t, pth.as_deref(), &a);
            }
            stats = format!("{} tokens in {:.1}s, {:.3} tok/s decode, stop: {}", out_all.len(), t0.elapsed().as_secs_f64(), st.tok_s, reason);
            let mut calls: Vec<(String, J)> = if has_tools { r.calls.iter().map(|(n, a)| resolve_tool(&acts, n, a)).collect() } else { Vec::new() };
            let mut chunk: Option<String> = None;
            if cut_at == Some(0) && cut_capped {
                cut_at = None; // nothing fits any more
                capped = true;
            }
            if let (Some(k), true) = (cut_at.filter(|&k| k > 0), chunks < MAX_CHUNKS) {
                // build the call ourselves from the streamed part
                let pc = cut_pc.clone().unwrap();
                let part = char_prefix(pc.text.as_deref().unwrap_or(""), k);
                let act = pc.action.clone().unwrap_or_default();
                let mut args = json!({"action": act, "name": pc.target, "content": part});
                if let Some(pp) = &pc.path {
                    args["path"] = J::String(pp.clone());
                }
                if act == "create" {
                    if let Some(t) = &pc.kind {
                        args["type"] = J::String(t.clone());
                    }
                    if let Some(t) = &pc.title {
                        args["title"] = J::String(t.clone());
                    }
                }
                calls = vec![("artifact".into(), args)];
                chunk = Some(part);
                chunks += 1;
                budget += 1;
                if reason == "stop" {
                    reason = "chunk".into();
                }
            }
            if reason == "max" && chunk.is_none() {
                for (_, a) in calls.iter_mut() {
                    if let Some(e) = a.get("_error").and_then(|x| x.as_str()).map(str::to_string) {
                        a["_error"] = J::String(format!("{e} (this reply stopped at the max-new-tokens limit of {max_new})"));
                    }
                }
            }
            let mut msg = json!({"role": "assistant", "content": r.answer, "meta": {"stats": stats}});
            if !r.thinking.is_empty() {
                msg["thinking"] = J::String(r.thinking.clone());
            }
            if !calls.is_empty() && r.harmony {
                // gpt-oss: structured call; the chat template renders it as 'to=functions.NAME ... <|call|>'
                calls.truncate(1);
                msg["content"] = J::String(String::new());
                msg["tool_calls"] = J::Array(calls.iter().map(|(n, a)| json!({"type": "function", "function": {"name": n, "arguments": a}})).collect());
            } else if chunk.is_some() {
                // <tool_call> models: store the call we completed, not the cut-off text
                let before = parse_reply(raw.split("<tool_call>").next().unwrap_or("")).answer;
                msg["content"] = J::String(format!(
                    "{}<tool_call>\n{}\n</tool_call>",
                    if before.is_empty() { String::new() } else { before + "\n" },
                    serde_json::to_string(&json!({"name": calls[0].0, "arguments": calls[0].1})).unwrap()
                ));
            }
            new_msgs.push(msg);
            if calls.is_empty() || cut_ui || should_stop() {
                break;
            }
            for i_c in 0..calls.len() {
                let (name, mut args) = calls[i_c].clone();
                if name == "artifact" && args.get("_error").is_none() {
                    let key = if args["action"] == "edit" { "new" } else { "content" };
                    if matches!(args["action"].as_str(), Some("write" | "append" | "create" | "edit")) && args[key].is_string() {
                        let room = cap_chars - written;
                        let body = args[key].as_str().unwrap().to_string();
                        if capped || room <= 0 {
                            args["_error"] = J::String(format!(
                                "the per-reply file growth limit ({} characters) is already reached. Do not write more in this reply: tell the user the file is incomplete; they can say 'continue' in their next message.",
                                fmt_int(cap_chars)
                            ));
                        } else if body.chars().count() as i64 > room {
                            let k2 = cap_cut(&body, room);
                            args[key] = J::String(char_prefix(&body, k2));
                            capped = true;
                            if chunk.is_some() {
                                chunk = Some(args[key].as_str().unwrap().to_string());
                            }
                        }
                        if args.get("_error").is_none() {
                            written += args[key].as_str().unwrap().chars().count() as i64;
                        }
                        calls[i_c] = (name.clone(), args.clone());
                        if cut_capped {
                            capped = true;
                        }
                    }
                }
                on(Event::Act(json!({"ev": "call", "name": name, "args": args})));
                // only enabled tools run (the Python harness ran any loaded plugin a model named, enabled or not)
                let mut res = match args.get("_error").and_then(|e| e.as_str()) {
                    Some(e) => format!("error: {e}"),
                    None if enabled.contains(&name) => tools.run(&name, &args),
                    None => format!(
                        "error: no tool named '{name}'. Available tools: {}. Call a tool by its name and put the operation in its arguments (e.g. artifact with action='create').",
                        acts.iter().map(|a| a.0.clone()).collect::<Vec<_>>().join(", ")
                    ),
                };
                let mut fctx = String::new();
                if name == "artifact" && !res.starts_with("error") && matches!(args["action"].as_str(), Some("write" | "append" | "create" | "edit")) {
                    if let Ok(d) = tools.artifacts.dir(args["name"].as_str().unwrap_or(""), true) {
                        let m = tools.artifacts.meta(&d);
                        if let Ok((fp, _)) = tools.artifacts.file(&d, args["path"].as_str(), &m) {
                            fctx = file_context(&String::from_utf8_lossy(&std::fs::read(fp).unwrap_or_default()), 40, 60);
                        }
                    }
                }
                if capped && !res.starts_with("error") {
                    res += &format!(
                        "\nThe per-reply file growth limit ({} characters) is reached, so the harness saved the content up to the last whole line that fits. Stop writing now. Tell the user the file is incomplete, what remains to be written, and that they can say 'continue' in their next message to resume from the last saved line.\n{fctx}",
                        fmt_int(cap_chars)
                    );
                } else if let Some(c) = chunk.as_ref().filter(|_| !res.starts_with("error")) {
                    res += &format!(
                        "\nThat call was ended by the harness after {} characters (the end of a line), to keep each call small. Continue the same file now: call artifact with action 'append', name '{}'{}, with content that starts right after the last saved line shown below. Keep it consistent with the outline (same names, styles and structure). Do not repeat anything already saved. When the file is complete, stop calling tools and tell the user.\n{fctx}",
                        c.chars().count(),
                        args["name"].as_str().unwrap_or(""),
                        args["path"].as_str().map(|p| format!(", path '{p}'")).unwrap_or_default()
                    );
                }
                on(Event::Tool(format!("-> {name}({})\n", reply::trunc(&serde_json::to_string(&args).unwrap(), 160))));
                new_msgs.push(json!({"role": "tool", "content": res, "meta": {"tool": name, "args": args}}));
                on(Event::Tool(format!("[{name}] {res}\n")));
                on(Event::Act(json!({"ev": "result", "name": name, "args": args, "text": res})));
            }
            on(Event::AssistantNext);
        }
        Ok((new_msgs, stats))
    }
}

#[allow(clippy::too_many_arguments)]
fn gen_with(
    eng: &mut Engine,
    fed: &mut Vec<u32>,
    stop: &Arc<AtomicBool>,
    eos: &[u32],
    ids: &[u32],
    max_new: usize,
    temp: f32,
    topp: f32,
    topk: i32,
    seed: u64,
    on_tok: &mut dyn FnMut(&[u32]),
) -> Result<(Vec<u32>, End, cube_engine::GenStats)> {
    let mut lcp = fed.iter().zip(ids).take_while(|(a, b)| a == b).count();
    lcp = lcp.min(ids.len().saturating_sub(1));
    if lcp > 0 {
        if eng.set_pos(lcp).is_err() {
            eng.pos = 0; // e.g. a sliding-window ring cannot roll back that far
            fed.clear();
            lcp = 0;
        }
    } else if !fed.is_empty() {
        eng.pos = 0;
        fed.clear();
    }
    let new = &ids[lcp..];
    stop.store(false, Ordering::SeqCst);
    let st2 = stop.clone();
    let mut cb = |o: &[u32]| -> bool {
        on_tok(o);
        st2.load(Ordering::SeqCst)
    };
    let (out, end, st) = match eng.generate(new, max_new, temp, topp, topk, seed, eos, &mut cb) {
        Ok(x) => x,
        Err(e) => {
            eng.pos = 0;
            fed.clear();
            return Err(e);
        }
    };
    // what the engine actually forwarded: everything except a final token that ended the turn (eos / ctx)
    *fed = ids.to_vec();
    if matches!(end, End::Max | End::Stop) {
        fed.extend_from_slice(&out);
    } else {
        fed.extend_from_slice(&out[..out.len().saturating_sub(1)]);
    }
    Ok((out, end, st))
}

/// 12345 -> "12,345" (Python's `{:,}`)
pub fn fmt_int(n: i64) -> String {
    let s = n.abs().to_string();
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    if n < 0 {
        format!("-{out}")
    } else {
        out
    }
}
