//! The tabs of the window, their state, dialogs and background jobs.

use crate::{App, Job, Tab, ACCENT, ASST, BAD, DIM, THINK, TOOL};
use cube_core::paths::{self, is_vault};
use eframe::egui::{self, Color32, RichText, ScrollArea, TextEdit};
use egui_extras::{Column, TableBuilder};
use serde_json::{json, Value as J};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

#[derive(Clone, Debug)]
pub enum Confirm {
    DeleteChat(String),
    DeleteArtifact(String),
    Revert(String, i64),
    Install(String),
    Compress(PathBuf, PathBuf),
    Move(PathBuf, PathBuf),
    RemoveFolder(PathBuf),
    Approve(String),
    Discard(String),
    DeleteSkill(String),
}

#[derive(Clone, Debug)]
pub enum PromptKind {
    NewArtifact,
    NewFile,
    RenameChat,
    TestPlugin(String),
}

#[derive(Clone, Debug)]
pub struct Prompt {
    pub title: String,
    pub fields: Vec<(String, String)>,
    pub kind: PromptKind,
}

pub struct FolderRow {
    pub path: PathBuf,
    pub drive: String,
    pub kind: String,
    pub speed: String,
    pub free: String,
    pub models: String,
    pub role: String,
    pub builtin: bool,
}

pub struct Installed {
    pub name: String,
    pub path: PathBuf,
    pub size: String,
    pub kind: String,
    pub tuned: String,
}

#[derive(Default)]
pub struct State {
    pub prompt: Option<Prompt>,
    pub view: Option<(String, String)>,
    // artifacts
    pub art_names: Vec<(String, String, String)>,
    pub art_name: Option<String>,
    pub art_path: Option<String>,
    pub art_files: Vec<String>,
    pub art_hist: Vec<J>,
    pub art_text: String,
    pub art_head: String,
    pub art_file_label: String,
    pub art_auto: bool,
    pub art_follow: bool,
    pub art_typed: Option<Instant>,
    pub art_session: Option<(String, String, String)>,
    pub art_hist_sel: Option<i64>,
    pub art_live: bool,
    /// (artifact, file) the model was last seen writing; selected when its call completes
    pub art_live_target: Option<(String, Option<String>)>,
    // models
    pub hub_q: String,
    pub hub_fit: bool,
    pub hub_hw: String,
    pub hub_res: Vec<cube_core::hub::Found>,
    pub hub_assessed: HashMap<String, cube_core::hub::Assessment>,
    pub hub_sel: Option<String>,
    pub hub_log: String,
    pub hub_busy: bool,
    pub hub_stop: Arc<AtomicBool>,
    pub inst: Vec<Installed>,
    pub inst_sel: Option<PathBuf>,
    pub loc: Vec<cube_core::hub::Local>,
    pub loc_sel: Option<PathBuf>,
    pub loc_dst: Option<PathBuf>,
    pub loc_tune: bool,
    // folders
    pub st_rows: Vec<FolderRow>,
    pub st_sel: Option<PathBuf>,
    pub st_vault: String,
    pub st_dl: String,
    pub st_log: String,
    pub st_busy: bool,
    // config
    pub cf: HashMap<String, String>,
    pub cf_bool: HashMap<String, bool>,
    pub cf_sys: String,
    pub cf_plan: String,
    // plugins
    pub pl_sel: Option<String>,
    pub pend: Vec<(String, u64, String, String)>,
    pub pend_sel: Option<String>,
    // skills
    pub sk_sel: Option<String>,
    pub sk_old: Option<String>,
    pub sk_name: String,
    pub sk_desc: String,
    pub sk_on: bool,
    pub sk_body: String,
}

pub fn init(a: &mut App) {
    a.t.art_auto = true;
    a.t.art_follow = true;
    a.t.hub_q = "qwen3".into();
    a.t.st_dl = a.cfg.str("hub_download_dir");
    if a.t.st_dl == "null" {
        a.t.st_dl.clear();
    }
    a.t.cf_plan = "The plan for the loaded model appears here after it loads, or press 'Model defaults'.".into();
    artifacts_refresh(a);
    hub_installed(a);
    folders_refresh(a);
    config_fill(a);
    plugins_refresh_pending(a);
}

fn head(ui: &mut egui::Ui, t: &str) {
    ui.label(RichText::new(t).size(15.0).strong());
}

fn open_path(p: &std::path::Path) {
    let _ = open::that_detached(p);
}

// ====================================================================== Messages
const FIELDS_HINT: &str = "Ctrl+Enter sends.";

pub fn messages(a: &mut App, ctx: &egui::Context) {
    egui::SidePanel::left("convs").resizable(true).default_width(260.0).show(ctx, |ui| {
        ui.add_space(6.0);
        if ui.add_sized([ui.available_width(), 28.0], egui::Button::new(RichText::new("+  New chat").color(ACCENT))).clicked() {
            a.new_chat();
        }
        ui.horizontal(|ui| {
            if ui.button("Rename").clicked() {
                if let Some(c) = &a.conv {
                    a.t.prompt = Some(Prompt { title: "Rename chat".into(), fields: vec![("New title".into(), c["title"].as_str().unwrap_or("").into())], kind: PromptKind::RenameChat });
                }
            }
            if ui.button("Delete").clicked() && !a.busy {
                if let Some(c) = &a.conv {
                    let (id, t) = (c["id"].as_str().unwrap_or("").to_string(), c["title"].as_str().unwrap_or("").to_string());
                    a.ask("Delete chat", &format!("Delete '{t}' permanently?"), Confirm::DeleteChat(id));
                }
            }
        });
        ui.separator();
        ScrollArea::vertical().auto_shrink([false; 2]).show(ui, |ui| {
            let cur = a.conv.as_ref().map(|c| c["id"].clone());
            let mut pick = None;
            for c in &a.convs {
                let t: String = c["title"].as_str().unwrap_or("").chars().take(34).collect();
                if ui.selectable_label(Some(&c["id"]) == cur.as_ref(), t).clicked() && !a.busy {
                    pick = Some(c.clone());
                }
            }
            if let Some(c) = pick {
                a.conv = Some(c);
            }
        });
    });
    egui::CentralPanel::default().show(ctx, |ui| {
        ui.horizontal(|ui| {
            ui.label(RichText::new(&a.model_label).size(15.0).strong());
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.add_enabled(!a.busy, egui::Button::new("Unload")).clicked() {
                    a.unload_model();
                }
                if ui.add_enabled(!a.busy, egui::Button::new("Load model")).clicked() {
                    a.load_model();
                }
            });
        });
        let bottom_h = 132.0;
        let avail = ui.available_height() - bottom_h;
        ScrollArea::vertical().id_salt("transcript").stick_to_bottom(true).auto_shrink([false; 2]).max_height(avail.max(100.0)).show(ui, |ui| {
            ui.set_width(ui.available_width());
            transcript(a, ui);
        });
        ui.separator();
        ui.horizontal(|ui| {
            let w = ui.available_width() - 110.0;
            let r = ui.add_sized([w, 90.0], TextEdit::multiline(&mut a.input).hint_text("Message the model ... (start with /skill-name to insert a skill)"));
            let send = r.has_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter) && i.modifiers.command);
            ui.vertical(|ui| {
                if ui.add_enabled(!a.busy, egui::Button::new(RichText::new("Send").color(ACCENT)).min_size([96.0, 28.0].into())).clicked() || (send && !a.busy) {
                    if send {
                        // the Enter of Ctrl+Enter was typed into the box
                        while a.input.ends_with('\n') {
                            a.input.pop();
                        }
                    }
                    a.send();
                }
                if ui.add_enabled(a.busy, egui::Button::new("Stop").min_size([96.0, 24.0].into())).clicked() {
                    a.worker.request_stop();
                    a.status = "stopping after the current step ...".into();
                }
                if ui.add_enabled(a.busy, egui::Button::new("Force stop").min_size([96.0, 24.0].into())).clicked() {
                    a.forced = true;
                    a.worker.force_stop();
                    a.status = "force stop: the engine aborts at the next token".into();
                }
            });
        });
        let tools: Vec<String> = {
            let t = a.tools();
            a.cfg.enabled_plugins().into_iter().filter(|n| t.get(n).is_some()).collect()
        };
        let sk: Vec<String> = a.skills.iter().map(|s| format!("/{}", s.name)).collect();
        ui.label(RichText::new(format!("{FIELDS_HINT}   Skills: {}   Tools on: {}", if sk.is_empty() { "none".into() } else { sk.join(" ") }, if tools.is_empty() { "none".into() } else { tools.join(", ") })).color(DIM).small());
    });
}

fn think_preview(ui: &mut egui::Ui, id: egui::Id, text: &str) {
    let words = text.split_whitespace().count();
    let flat: String = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let headtxt: String = flat.chars().take(110).collect();
    let ell = if flat.chars().count() > 110 { "…" } else { "" };
    egui::CollapsingHeader::new(RichText::new(format!("Thought for {words} words — {headtxt}{ell}")).color(Color32::from_rgb(0xa7, 0xad, 0xb8)).small())
        .id_salt(id)
        .default_open(false)
        .show(ui, |ui| {
            ui.label(RichText::new(text.trim()).italics().color(THINK));
        });
}

fn transcript(a: &mut App, ui: &mut egui::Ui) {
    let name = a.conv.as_ref().and_then(|c| c["vault"].as_str().map(str::to_string)).filter(|s| !s.is_empty()).unwrap_or_else(|| a.cfg.str("vault"));
    let name = std::path::Path::new(&name).file_name().map(|x| x.to_string_lossy().into_owned()).unwrap_or_else(|| "model".into());
    let msgs: Vec<J> = a.conv.as_ref().and_then(|c| c["messages"].as_array().cloned()).unwrap_or_default();
    for (i, m) in msgs.iter().enumerate() {
        let meta = &m["meta"];
        match m["role"].as_str().unwrap_or("") {
            "user" => {
                ui.add_space(8.0);
                ui.label(RichText::new("You").color(ACCENT).strong());
                ui.label(meta["typed"].as_str().or(m["content"].as_str()).unwrap_or(""));
                if let Some(s) = meta["skills"].as_array().filter(|s| !s.is_empty()) {
                    ui.label(RichText::new(format!("skills: {}", s.iter().filter_map(|x| x.as_str()).map(|x| format!("/{x}")).collect::<Vec<_>>().join(", "))).color(DIM).italics().small());
                }
            }
            "assistant" => {
                ui.add_space(8.0);
                ui.label(RichText::new(&name).color(ASST).strong());
                if let Some(t) = m["thinking"].as_str().filter(|t| !t.is_empty()) {
                    think_preview(ui, egui::Id::new(("think", a.conv.as_ref().map(|c| c["id"].to_string()), i)), t);
                }
                for tc in m["tool_calls"].as_array().cloned().unwrap_or_default() {
                    let f = if tc["function"].is_object() { tc["function"].clone() } else { tc.clone() };
                    let s: String = serde_json::to_string(&f["arguments"]).unwrap_or_default().chars().take(160).collect();
                    ui.label(RichText::new(format!("-> {}({s})", f["name"].as_str().unwrap_or(""))).color(TOOL).monospace());
                }
                if let Some(c) = m["content"].as_str().filter(|c| !c.is_empty()) {
                    ui.label(c);
                }
                if let Some(s) = meta["stats"].as_str() {
                    ui.label(RichText::new(s).color(DIM).italics().small());
                }
            }
            "tool" => {
                let full = m["content"].as_str().unwrap_or("");
                let short: String = full.chars().take(600).collect();
                let r = ui.label(RichText::new(format!("[{}] {short}{}", meta["tool"].as_str().unwrap_or("tool"), if full.chars().count() > 600 { " …" } else { "" })).color(TOOL).monospace());
                if full.chars().count() > 600 && r.clicked() {
                    a.t.view = Some(("tool result".into(), full.to_string()));
                }
            }
            _ => {}
        }
    }
    if let Some(l) = &a.live {
        ui.add_space(8.0);
        ui.label(RichText::new(if l.model.is_empty() { &name } else { &l.model }).color(ASST).strong());
        // activity: rounds, thinking, tool calls (click an item for its full text)
        let mut open_detail = None;
        egui::Frame::new().fill(Color32::from_rgb(0x31, 0x33, 0x38)).inner_margin(6.0).corner_radius(4.0).show(ui, |ui| {
            ui.set_width(ui.available_width());
            if l.rounds.is_empty() {
                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.label(RichText::new("reading the prompt ...").color(DIM));
                });
            }
            for (ri, r) in l.rounds.iter().enumerate() {
                egui::CollapsingHeader::new(format!("Round {}   {}", r.n, r.status)).id_salt(("round", ri)).default_open(true).show(ui, |ui| {
                    for it in &r.items {
                        ui.horizontal(|ui| {
                            if ui.link(RichText::new(&it.label).monospace()).clicked() && !it.detail.is_empty() {
                                open_detail = Some((it.label.clone(), it.detail.clone()));
                            }
                            ui.label(RichText::new(&it.status).color(DIM).small());
                        });
                    }
                });
            }
        });
        if let Some(d) = open_detail {
            a.t.view = Some(d);
        }
        for (k, t) in &l.parts {
            match k.as_str() {
                "think" => {
                    ui.label(RichText::new(t).italics().color(THINK));
                }
                "answer" => {
                    ui.label(t.as_str());
                }
                "tool" => {
                    ui.label(RichText::new(t.trim_end()).color(TOOL).monospace());
                }
                "head" => {
                    ui.label(RichText::new(if l.model.is_empty() { &name } else { &l.model }).color(ASST).strong());
                }
                _ => {}
            }
        }
    }
}

// ====================================================================== Artifacts
pub fn artifacts_refresh(a: &mut App) {
    let arts = a.tools().artifacts.clone();
    let _ = std::fs::create_dir_all(&arts.root);
    let mut v: Vec<(String, std::time::SystemTime)> = arts.names().into_iter().map(|n| {
        let t = std::fs::metadata(arts.root.join(&n).join(".artifact.json")).and_then(|m| m.modified()).unwrap_or(std::time::UNIX_EPOCH);
        (n, t)
    }).collect();
    v.sort_by(|x, y| y.1.cmp(&x.1));
    a.t.art_names = v.into_iter().map(|(n, _)| {
        let m = arts.meta(&arts.root.join(&n));
        (n, m["title"].as_str().unwrap_or("").to_string(), m["type"].as_str().unwrap_or("").to_string())
    }).collect();
    // a refresh happens when a call has completed (or by hand): leave the live view and show what was saved,
    // selecting the artifact and file the model was writing
    let was_live = std::mem::take(&mut a.t.art_live);
    if let Some((n, p)) = a.t.art_live_target.take() {
        if a.t.art_names.iter().any(|x| x.0 == n) {
            a.t.art_name = Some(n);
            if p.is_some() {
                a.t.art_path = p;
            } else if was_live {
                a.t.art_path = None;
            }
        }
    }
    if let Some(n) = a.t.art_name.clone() {
        if a.t.art_names.iter().any(|x| x.0 == n) {
            let keep = a.t.art_path.is_some();
            art_show(a, &n, keep);
        }
    }
}

fn art_autosave(a: &mut App) {
    a.t.art_typed = None;
    let (Some(n), Some(p)) = (a.t.art_name.clone(), a.t.art_path.clone()) else { return };
    let need_new = a.t.art_session.as_ref().map_or(true, |s| s.0 != n || s.1 != p);
    if need_new {
        a.t.art_session = Some((n.clone(), p.clone(), format!("{}", cube_core::convs::now_f() as u64)));
    }
    let sid = a.t.art_session.as_ref().unwrap().2.clone();
    a.status = match a.tools().artifacts.autosave(&n, Some(&p), &a.t.art_text, &sid) {
        Ok(s) => s,
        Err(e) => format!("autosave failed: {e}"),
    };
}

fn art_show(a: &mut App, name: &str, keep_file: bool) {
    let arts = a.tools().artifacts.clone();
    let d = arts.root.join(name);
    let m = arts.meta(&d);
    a.t.art_name = Some(name.into());
    a.t.art_head = format!("{}   [{}]   artifacts/{name}   updated {}", m["title"].as_str().unwrap_or(name), m["type"].as_str().unwrap_or(""), m["updated"].as_str().unwrap_or(""));
    a.t.art_files = arts.files(&d);
    a.t.art_hist = m["changes"].as_array().cloned().unwrap_or_default().into_iter().rev().take(200).collect();
    let main = m["main"].as_str().unwrap_or("").to_string();
    let target = if keep_file && a.t.art_path.as_ref().map_or(false, |p| a.t.art_files.contains(p)) {
        a.t.art_path.clone()
    } else if a.t.art_files.contains(&main) {
        Some(main)
    } else {
        a.t.art_files.first().cloned()
    };
    match target {
        Some(t) => art_load(a, &t),
        None => {
            a.t.art_path = None;
            a.t.art_text.clear();
            a.t.art_file_label = "(no files)".into();
        }
    }
}

fn art_load(a: &mut App, rel: &str) {
    if a.t.art_typed.is_some() {
        art_autosave(a);
    }
    a.t.art_session = None;
    let Some(n) = a.t.art_name.clone() else { return };
    let p = a.tools().artifacts.root.join(&n).join(rel);
    a.t.art_path = Some(rel.into());
    a.t.art_text = match std::fs::read(&p) {
        Ok(b) => String::from_utf8(b).unwrap_or_else(|e| format!("(cannot show this file as text: {e})")),
        Err(e) => format!("(cannot show this file as text: {e})"),
    };
    a.t.art_live = false;
    a.t.art_file_label = format!("{rel}   {} bytes   - edit here and press Save file (versioned, revertible)", std::fs::metadata(&p).map(|m| m.len()).unwrap_or(0));
}

/// show an artifact file while the model is still writing it (the call's content as decoded so far)
pub fn artifacts_live(a: &mut App, ev: &J) {
    if !a.t.art_follow {
        return;
    }
    let Some(text) = ev["text"].as_str() else { return };
    if a.t.art_typed.is_some() {
        art_autosave(a);
    }
    let arts = a.tools().artifacts.clone();
    let name = cube_core::tools::artifact::slug(ev["target"].as_str().unwrap_or(""));
    let mut rel = ev["path"].as_str().unwrap_or("").replace('\\', "/").trim_matches('/').to_string();
    if !name.is_empty() && arts.root.join(&name).join(".artifact.json").exists() {
        if rel.is_empty() {
            rel = arts.meta(&arts.root.join(&name))["main"].as_str().unwrap_or("").to_string();
        }
        a.t.art_name = Some(name.clone());
        if !rel.is_empty() {
            a.t.art_path = Some(rel.clone());
        }
    }
    if !name.is_empty() {
        a.t.art_live_target = Some((name.clone(), if rel.is_empty() { None } else { Some(rel.clone()) }));
    }
    let verb = match ev["action"].as_str().unwrap_or("") {
        "create" => "creating",
        "write" => "writing",
        "append" => "appending to",
        "edit" => "editing (new text)",
        x => x,
    };
    a.t.art_head = format!("● LIVE  {}{}  -  model is {verb}  ({} chars)", if name.is_empty() { "(new artifact)" } else { &name }, if rel.is_empty() { String::new() } else { format!("/{rel}") }, text.chars().count());
    a.t.art_file_label = "live view of the model's tool call - the file is saved when the call completes".into();
    a.t.art_text = text.to_string();
    a.t.art_live = true;
}

pub fn artifacts(a: &mut App, ctx: &egui::Context) {
    if let Some(t) = a.t.art_typed {
        if a.t.art_auto && t.elapsed().as_secs_f64() > 1.5 {
            art_autosave(a);
        } else {
            ctx.request_repaint_after(std::time::Duration::from_millis(300));
        }
    }
    egui::SidePanel::left("arts").resizable(true).default_width(270.0).show(ctx, |ui| {
        ui.add_space(6.0);
        if ui.add_sized([ui.available_width(), 28.0], egui::Button::new(RichText::new("+  New artifact").color(ACCENT))).clicked() {
            a.t.prompt = Some(Prompt { title: "New artifact".into(), fields: vec![("Name (e.g. trip-notes)".into(), String::new()), ("Type: note, page, site, code or data".into(), "note".into())], kind: PromptKind::NewArtifact });
        }
        ui.horizontal(|ui| {
            if ui.button("Refresh").clicked() {
                artifacts_refresh(a);
            }
            if ui.button("Delete").clicked() {
                if let Some(n) = a.t.art_name.clone() {
                    a.ask("Delete artifact", &format!("Delete artifact '{n}' and all its files and history permanently?"), Confirm::DeleteArtifact(n));
                }
            }
        });
        ui.separator();
        ScrollArea::vertical().auto_shrink([false; 2]).show(ui, |ui| {
            let mut pick = None;
            for (n, t, ty) in &a.t.art_names {
                if ui.selectable_label(a.t.art_name.as_deref() == Some(n), format!("{}  [{ty}]", if t.is_empty() { n } else { t })).clicked() {
                    pick = Some(n.clone());
                }
            }
            if let Some(n) = pick {
                art_show(a, &n, false);
            }
        });
    });
    egui::CentralPanel::default().show(ctx, |ui| {
        head(ui, if a.t.art_head.is_empty() { "no artifact selected - the model creates them with the 'artifact' tool, or press New" } else { &a.t.art_head });
        ui.horizontal(|ui| {
            let arts = a.tools().artifacts.clone();
            if ui.button("Open in browser").clicked() {
                if let Some(n) = &a.t.art_name {
                    let d = arts.root.join(n);
                    let m = arts.meta(&d);
                    let rel = a.t.art_path.clone().filter(|p| p.to_lowercase().ends_with(".html") || p.to_lowercase().ends_with(".htm")).or_else(|| m["main"].as_str().map(str::to_string)).unwrap_or_default();
                    let p = d.join(rel);
                    if p.is_file() {
                        open_path(&p);
                        a.status = format!("opened {} in the browser", p.file_name().unwrap().to_string_lossy());
                    } else {
                        a.status = "This artifact has no HTML file to open.".into();
                    }
                }
            }
            if ui.button("Open folder").clicked() {
                if let Some(n) = &a.t.art_name {
                    open_path(&arts.root.join(n));
                }
            }
            if ui.button("Export .zip").clicked() {
                if let Some(n) = a.t.art_name.clone() {
                    if let Some(dst) = rfd::FileDialog::new().set_file_name(format!("{n}.zip")).add_filter("zip", &["zip"]).save_file() {
                        a.status = match zip_artifact(&arts, &n, &dst) {
                            Ok(()) => format!("exported {n} to {} (current files, no history)", dst.display()),
                            Err(e) => format!("export failed: {e}"),
                        };
                    }
                }
            }
            if ui.button("New file...").clicked() && a.t.art_name.is_some() {
                a.t.prompt = Some(Prompt { title: "New file".into(), fields: vec![("Path inside the artifact (e.g. css/style.css)".into(), String::new())], kind: PromptKind::NewFile });
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.button(RichText::new("Save file").color(ACCENT)).clicked() {
                    if let (Some(n), Some(p)) = (a.t.art_name.clone(), a.t.art_path.clone()) {
                        a.status = arts.act(&json!({"action": "write", "name": n, "path": p, "content": a.t.art_text}));
                        a.t.art_typed = None;
                        artifacts_refresh(a);
                    }
                }
                ui.checkbox(&mut a.t.art_auto, "Autosave");
                ui.checkbox(&mut a.t.art_follow, "Follow the model live");
            });
        });
        egui::SidePanel::left("artfl").resizable(true).default_width(240.0).show_inside(ui, |ui| {
            ui.label(RichText::new("Files").color(DIM));
            let mut pick = None;
            ScrollArea::vertical().id_salt("artfiles").max_height((ui.available_height() - 230.0).max(80.0)).show(ui, |ui| {
                for f in &a.t.art_files {
                    if ui.selectable_label(a.t.art_path.as_deref() == Some(f), RichText::new(f).monospace()).clicked() {
                        pick = Some(f.clone());
                    }
                }
            });
            if let Some(f) = pick {
                art_load(a, &f);
            }
            ui.label(RichText::new("History (newest first)").color(DIM));
            ScrollArea::vertical().id_salt("arthist").max_height(160.0).show(ui, |ui| {
                for c in &a.t.art_hist {
                    let n = c["n"].as_i64().unwrap_or(0);
                    let t = c["time"].as_str().unwrap_or("");
                    let t = if t.len() >= 16 { &t[5..16] } else { t };
                    if ui.selectable_label(a.t.art_hist_sel == Some(n), RichText::new(format!("#{n} {t} {} {}", c["action"].as_str().unwrap_or(""), c["path"].as_str().unwrap_or(""))).monospace().small()).clicked() {
                        a.t.art_hist_sel = Some(n);
                    }
                }
            });
            if ui.button("Revert selected change").clicked() {
                if let (Some(n), Some(v)) = (a.t.art_name.clone(), a.t.art_hist_sel) {
                    a.ask("Revert", &format!("Restore the file as it was before change #{v}? (The current version stays in the history.)"), Confirm::Revert(n, v));
                }
            }
        });
        egui::CentralPanel::default().show_inside(ui, |ui| {
            ui.label(RichText::new(&a.t.art_file_label).color(DIM).small());
            ScrollArea::both().id_salt("arted").auto_shrink([false; 2]).show(ui, |ui| {
                let rows = ((ui.available_height() / 15.0) as usize).max(10);
                let r = ui.add(TextEdit::multiline(&mut a.t.art_text).code_editor().desired_width(f32::INFINITY).desired_rows(rows));
                if r.changed() && !a.t.art_live {
                    a.t.art_typed = Some(Instant::now());
                }
            });
        });
    });
}

fn zip_artifact(arts: &cube_core::tools::artifact::Artifacts, n: &str, dst: &std::path::Path) -> anyhow::Result<()> {
    use std::io::Write;
    let d = arts.root.join(n);
    let f = std::fs::File::create(dst)?;
    let mut z = zip::ZipWriter::new(f);
    let o = zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Deflated);
    for rel in arts.files(&d) {
        z.start_file(rel.clone(), o)?;
        z.write_all(&std::fs::read(d.join(&rel))?)?;
    }
    z.finish()?;
    Ok(())
}

// ====================================================================== Models (Hugging Face -> compressed vault)
fn slow_folder(a: &App, d: &std::path::Path) -> bool {
    let (media, bus, _) = cube_core::storage::drive_kind(&a.p, d);
    media == "HDD" || bus == "USB" || cube_core::storage::read_speed(&a.p, d, false).map_or(false, |s| s < 500.0)
}

pub fn hub_installed(a: &mut App) {
    a.t.loc = cube_core::hub::local_checkpoints(&a.p, &a.cfg);
    if a.t.loc_dst.as_ref().map_or(true, |d| !d.exists()) {
        // compress onto a fast drive: the engine streams weights from the vault while it runs
        let dirs: Vec<PathBuf> = paths::model_dirs(&a.p, &a.cfg).into_iter().map(|m| m.path).filter(|d| d.exists()).collect();
        a.t.loc_dst = dirs.iter().find(|d| !slow_folder(a, d)).cloned().or_else(|| dirs.first().cloned());
        a.t.loc_tune = true;
    }
    a.t.inst = paths::find_vaults(&a.p, &a.cfg)
        .into_iter()
        .filter_map(|(name, path)| {
            let v = cube_vault::Vault::open(&path).ok()?;
            let m = v.meta();
            let kind = match m.arch {
                cube_vault::meta::Arch::Gptoss => "gpt-oss (MoE)",
                cube_vault::meta::Arch::QwenMoe => "MoE",
                cube_vault::meta::Arch::Dense => "dense",
            };
            let tuned = std::fs::read(path.join("tuned.json"))
                .ok()
                .and_then(|b| serde_json::from_slice::<J>(&b).ok())
                .map(|d| format!("{} tok/s  {}  ({})", d["tok_s"], d["settings"], d["date"].as_str().unwrap_or("")))
                .unwrap_or_else(|| "not tuned yet".into());
            Some(Installed { size: format!("{:.1} GB", cube_core::storage::vault_size(&path) as f64 / 1e9), kind: format!("{kind}{}", if v.legacy_version.is_some() { " (v3-v7)" } else { "" }), tuned, name, path })
        })
        .collect();
}

fn hub_search(a: &mut App) {
    if a.t.hub_busy {
        return;
    }
    let q = a.t.hub_q.trim().to_string();
    a.t.hub_res.clear();
    a.t.hub_assessed.clear();
    a.t.hub_busy = true;
    a.status = format!("searching Hugging Face for '{q}' ...");
    let (p, c, tx) = (a.p.clone(), a.cfg.clone(), a.jtx.clone());
    std::thread::spawn(move || {
        let hw = cube_core::hub::hardware(&p, &c);
        match cube_core::hub::search(&q, 30) {
            Ok(res) => {
                let _ = tx.send(Job::Found(res.clone(), hw.clone()));
                for m in res {
                    let _ = tx.send(Job::Assessed(cube_core::hub::assess(&m.repo, &hw)));
                }
                let _ = tx.send(Job::Status("search done - select a model to see its decision tree".into()));
            }
            Err(e) => {
                let _ = tx.send(Job::Status(format!("search failed: {e}")));
            }
        }
        let _ = tx.send(Job::Done(Tab::Models));
    });
}

fn hub_log(a: &mut App, s: &str) {
    a.t.hub_log.push_str(s.trim_end());
    a.t.hub_log.push('\n');
}

fn release_chat(a: &mut App) -> bool {
    if a.busy {
        a.status = "Wait for the current reply to finish (or Force stop it) first.".into();
        return false;
    }
    let _ = a.worker.tx.send(crate::worker::Cmd::Unload);
    true
}

fn hub_install(a: &mut App, repo: String) {
    if a.t.hub_busy || !release_chat(a) {
        return;
    }
    a.t.hub_busy = true;
    a.t.hub_stop.store(false, Ordering::SeqCst);
    let (p, c, tx, stop) = (a.p.clone(), a.cfg.clone(), a.jtx.clone(), a.t.hub_stop.clone());
    std::thread::spawn(move || {
        let t2 = tx.clone();
        let r = cube_core::hub::install(&p, &c, &repo, true, &mut |s| {
            let _ = t2.send(Job::Log(Tab::Models, s));
        }, &|| stop.load(Ordering::SeqCst));
        let _ = tx.send(Job::Log(Tab::Models, match r {
            Ok(v) => format!("done: {}", v.display()),
            Err(e) => format!("stopped: {e:#}"),
        }));
        let _ = tx.send(Job::Done(Tab::Models));
    });
}

fn hub_compress(a: &mut App, src: PathBuf, dst: PathBuf) {
    if a.t.hub_busy || !release_chat(a) {
        return;
    }
    a.t.hub_busy = true;
    a.t.hub_stop.store(false, Ordering::SeqCst);
    let (p, c, tx, stop, tune) = (a.p.clone(), a.cfg.clone(), a.jtx.clone(), a.t.hub_stop.clone(), a.t.loc_tune);
    std::thread::spawn(move || {
        let t2 = tx.clone();
        let r = cube_core::hub::compress_local(&p, &c, &src, &dst, None, tune, &mut |s| {
            let _ = t2.send(Job::Log(Tab::Models, s));
        }, &|| stop.load(Ordering::SeqCst));
        let _ = tx.send(Job::Log(Tab::Models, match r {
            Ok(v) => format!("done: {} - select it under Installed vaults and press 'Use in chat'", v.display()),
            Err(e) => format!("stopped: {e:#}"),
        }));
        let _ = tx.send(Job::Done(Tab::Models));
    });
}

fn hub_tune(a: &mut App, v: PathBuf) {
    if a.t.hub_busy || !release_chat(a) {
        return;
    }
    a.t.hub_busy = true;
    hub_log(a, &format!("tuning {} (short test replies with each candidate setting) ...", v.file_name().unwrap_or_default().to_string_lossy()));
    let (p, c, tx) = (a.p.clone(), a.cfg.clone(), a.jtx.clone());
    std::thread::spawn(move || {
        let t2 = tx.clone();
        if let Err(e) = cube_core::hub::tune(&p, &c, &v, 64, &mut |s| {
            let _ = t2.send(Job::Log(Tab::Models, s));
        }) {
            let _ = tx.send(Job::Log(Tab::Models, format!("tuning failed: {e:#}")));
        }
        let _ = tx.send(Job::Done(Tab::Models));
    });
}

fn use_vault(a: &mut App, name: &str, path: &std::path::Path) {
    a.cfg.set("vault", json!(name));
    if let Ok((rec, notes)) = cube_core::settings::recommended(path) {
        for (k, v) in rec {
            a.cfg.set(&k, v);
        }
        a.t.cf_plan = notes.join("\n");
    }
    a.save_cfg();
    config_fill(a);
    a.tab = Tab::Messages;
    a.status = format!("{name} selected with its recommended / tuned settings - it loads with your next message");
}

pub fn models(a: &mut App, ctx: &egui::Context) {
    egui::TopBottomPanel::top("hubtop").show(ctx, |ui| {
        ui.horizontal(|ui| {
            ui.label("Search Hugging Face:");
            let r = ui.add(TextEdit::singleline(&mut a.t.hub_q).desired_width(280.0));
            if (r.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter))) || ui.add_enabled(!a.t.hub_busy, egui::Button::new(RichText::new("Search").color(ACCENT))).clicked() {
                hub_search(a);
            }
            ui.checkbox(&mut a.t.hub_fit, "Only models that run well here");
            ui.label(RichText::new(&a.t.hub_hw).color(DIM).small());
        });
    });
    egui::SidePanel::right("hubright").resizable(true).default_width(520.0).show(ctx, |ui| {
        head(ui, "Decision tree for the selected model");
        let sel = a.t.hub_sel.clone();
        match sel.as_ref().and_then(|s| a.t.hub_assessed.get(s)) {
            Some(x) => {
                ui.label(RichText::new(format!("{}  →  {}", x.repo.rsplit('/').next().unwrap_or(""), x.verdict)).strong());
                for (q, ans, st) in &x.tree {
                    let (icon, col) = match st.as_str() {
                        "ok" => ("✓", ASST),
                        "warn" => ("!", TOOL),
                        _ => ("✗", BAD),
                    };
                    ui.horizontal_wrapped(|ui| {
                        ui.label(RichText::new(format!("{icon} {q}:")).color(col));
                        ui.label(ans);
                    });
                }
            }
            None => {
                ui.label(RichText::new(if sel.is_some() { "checking ..." } else { "select a model" }).color(DIM));
            }
        }
        ui.horizontal(|ui| {
            let can = sel.as_ref().and_then(|s| a.t.hub_assessed.get(s)).map_or(false, |x| x.kind.is_some());
            if ui.add_enabled(can && !a.t.hub_busy, egui::Button::new(RichText::new("Install: download, compress, tune").color(ACCENT))).clicked() {
                let x = a.t.hub_assessed.get(sel.as_ref().unwrap()).unwrap().clone();
                let hw = cube_core::hub::hardware(&a.p, &a.cfg);
                let msg = format!(
                    "Install {}?\n\n1. Download ~{:.1} GB from huggingface.co to {}\n2. Compress it into a ~{:.1} GB vault in {}\n3. Check every stream of the vault\n4. Tune its engine settings on this machine (a few short test replies)\n\nVerdict: {}{}",
                    x.repo, x.download_gb.unwrap_or(0.0), hw.dl_dir.display(), x.vault_gb.unwrap_or(0.0), hw.vault_dir.display(), x.verdict,
                    if x.verdict.contains("disk") { "\n\nWARNING: the disk check failed - see the decision tree." } else { "" }
                );
                a.ask("Install model", &msg, Confirm::Install(x.repo));
            }
            if ui.button("Open on Hugging Face").clicked() {
                if let Some(s) = &sel {
                    let _ = open::that_detached(format!("https://huggingface.co/{s}"));
                }
            }
            if ui.add_enabled(a.t.hub_busy, egui::Button::new("Stop")).clicked() {
                a.t.hub_stop.store(true, Ordering::SeqCst);
            }
        });
        ScrollArea::vertical().id_salt("hublog").stick_to_bottom(true).auto_shrink([false; 2]).show(ui, |ui| {
            ui.label(RichText::new(&a.t.hub_log).monospace().small());
        });
    });
    egui::CentralPanel::default().show(ctx, |ui| {
        let h = ui.available_height();
        let rows: Vec<cube_core::hub::Found> = a.t.hub_res.iter().filter(|m| !a.t.hub_fit || a.t.hub_assessed.get(&m.repo).map_or(true, |x| x.verdict.starts_with("runs well"))).cloned().collect();
        let mut pick = None;
        ui.push_id("hubres", |ui| {
            TableBuilder::new(ui)
                .striped(true)
                .max_scroll_height(h * 0.6)
                .column(Column::remainder().at_least(240.0))
                .columns(Column::auto().at_least(70.0), 4)
                .column(Column::auto().at_least(90.0))
                .header(20.0, |mut hd| {
                    for t in ["Model", "Params", "Vault", "Est. tok/s", "Verdict", "Downloads"] {
                        hd.col(|ui| {
                            ui.strong(t);
                        });
                    }
                })
                .body(|mut b| {
                    for m in &rows {
                        let x = a.t.hub_assessed.get(&m.repo);
                        b.row(20.0, |mut r| {
                            r.col(|ui| {
                                if ui.selectable_label(a.t.hub_sel.as_deref() == Some(&m.repo), &m.repo).clicked() {
                                    pick = Some(m.repo.clone());
                                }
                            });
                            r.col(|ui| {
                                ui.label(m.params.map_or("?".into(), |p| format!("{:.1}B", p / 1e9)));
                            });
                            r.col(|ui| {
                                ui.label(x.and_then(|x| x.vault_gb).map_or(String::new(), |g| format!("{g:.1} GB")));
                            });
                            r.col(|ui| {
                                ui.label(x.and_then(|x| x.est_tok_s).map_or(String::new(), |t| format!("{t:.2}")));
                            });
                            r.col(|ui| {
                                ui.label(x.map_or("checking ...".to_string(), |x| x.verdict.clone()));
                            });
                            r.col(|ui| {
                                ui.label(cube_core::session::fmt_int(m.downloads as i64));
                            });
                        });
                    }
                });
        });
        if let Some(p) = pick {
            a.t.hub_sel = Some(p);
        }
        ui.separator();
        ui.label(RichText::new("Downloaded checkpoints in your model folders (compress one to run it)").color(DIM));
        let mut pick = None;
        ui.push_id("hubloc", |ui| {
            TableBuilder::new(ui)
                .striped(true)
                .max_scroll_height(130.0)
                .column(Column::remainder().at_least(240.0))
                .column(Column::auto().at_least(90.0))
                .column(Column::auto().at_least(70.0))
                .column(Column::remainder().at_least(200.0))
                .header(20.0, |mut hd| {
                    for t in ["Checkpoint", "Type", "Size", "Status"] {
                        hd.col(|ui| {
                            ui.strong(t);
                        });
                    }
                })
                .body(|mut b| {
                    for l in &a.t.loc {
                        b.row(20.0, |mut r| {
                            r.col(|ui| {
                                if ui.selectable_label(a.t.loc_sel.as_ref() == Some(&l.dir), l.dir.display().to_string()).clicked() {
                                    pick = Some(l.dir.clone());
                                }
                            });
                            r.col(|ui| {
                                ui.label(&l.model_type);
                            });
                            r.col(|ui| {
                                ui.label(format!("{:.1} GB", l.bytes as f64 / 1e9));
                            });
                            r.col(|ui| {
                                ui.label(match (&l.kind, &l.vault) {
                                    (None, _) => RichText::new("not supported by this engine").color(BAD),
                                    (_, Some(v)) => RichText::new(format!("compressed: {}", v.display())).color(ASST),
                                    _ => RichText::new("not compressed yet").color(TOOL),
                                });
                            });
                        });
                    }
                });
        });
        if let Some(p) = pick {
            a.t.loc_sel = Some(p);
        }
        ui.horizontal(|ui| {
            ui.label("Into:");
            let dirs: Vec<PathBuf> = paths::model_dirs(&a.p, &a.cfg).into_iter().map(|m| m.path).filter(|d| d.exists()).collect();
            let cur = a.t.loc_dst.clone().map(|d| d.display().to_string()).unwrap_or_default();
            let mut choice = None;
            egui::ComboBox::from_id_salt("locdst").width(360.0).selected_text(cur).show_ui(ui, |ui| {
                for d in &dirs {
                    let tag = if slow_folder(a, d) { "  (slow drive: fine for storage, slow to run)" } else { "" };
                    if ui.selectable_label(a.t.loc_dst.as_ref() == Some(d), format!("{}{tag}", d.display())).clicked() {
                        choice = Some(d.clone());
                    }
                }
            });
            if let Some(d) = choice {
                a.t.loc_dst = Some(d);
            }
            ui.checkbox(&mut a.t.loc_tune, "Tune after");
            let sel = a.t.loc_sel.clone().and_then(|s| a.t.loc.iter().find(|l| l.dir == s).cloned());
            let can = sel.as_ref().map_or(false, |l| l.kind.is_some()) && a.t.loc_dst.is_some() && !a.t.hub_busy;
            if ui.add_enabled(can, egui::Button::new(RichText::new("Compress selected").color(ACCENT))).clicked() {
                let l = sel.unwrap();
                let dst = a.t.loc_dst.clone().unwrap();
                let est = l.bytes as f64 / 1e9 * if l.kind.as_deref() == Some("gptoss") { 0.97 } else { 0.67 };
                let msg = format!(
                    "Compress {}\ninto {}?\n\nThe vault will be about {est:.1} GB. Every stream is checked after packing{}. The checkpoint itself is not changed.{}",
                    l.dir.display(),
                    dst.display(),
                    if a.t.loc_tune { ", then the settings are tuned" } else { "" },
                    if l.vault.is_some() { "\n\nA vault for it already exists and will be replaced." } else { "" }
                );
                a.ask("Compress checkpoint", &msg, Confirm::Compress(l.dir, dst));
            }
        });
        ui.separator();
        ui.label(RichText::new("Installed vaults").color(DIM));
        let mut pick = None;
        ui.push_id("hubinst", |ui| {
            TableBuilder::new(ui)
                .striped(true)
                .max_scroll_height(180.0)
                .column(Column::remainder().at_least(240.0))
                .column(Column::auto().at_least(80.0))
                .column(Column::auto().at_least(110.0))
                .column(Column::remainder().at_least(200.0))
                .header(20.0, |mut hd| {
                    for t in ["Vault", "Size", "Type", "Tuned"] {
                        hd.col(|ui| {
                            ui.strong(t);
                        });
                    }
                })
                .body(|mut b| {
                    for v in &a.t.inst {
                        b.row(20.0, |mut r| {
                            r.col(|ui| {
                                if ui.selectable_label(a.t.inst_sel.as_ref() == Some(&v.path), &v.name).clicked() {
                                    pick = Some(v.path.clone());
                                }
                            });
                            r.col(|ui| {
                                ui.label(&v.size);
                            });
                            r.col(|ui| {
                                ui.label(&v.kind);
                            });
                            r.col(|ui| {
                                ui.label(RichText::new(&v.tuned).small());
                            });
                        });
                    }
                });
        });
        if let Some(p) = pick {
            a.t.inst_sel = Some(p);
        }
        ui.horizontal(|ui| {
            if ui.add_enabled(!a.t.hub_busy, egui::Button::new("Tune selected")).clicked() {
                if let Some(v) = a.t.inst_sel.clone() {
                    hub_tune(a, v);
                }
            }
            if ui.button(RichText::new("Use in chat").color(ACCENT)).clicked() {
                if let Some(v) = a.t.inst_sel.clone() {
                    let n = a.t.inst.iter().find(|x| x.path == v).map(|x| x.name.clone()).unwrap_or_default();
                    use_vault(a, &n, &v);
                }
            }
            if ui.button("Refresh").clicked() {
                hub_installed(a);
            }
        });
    });
}

// ====================================================================== Model folders
pub fn folders_refresh(a: &mut App) {
    a.cfg = paths::Config::load(&a.p);
    a.t.st_rows = paths::model_dirs(&a.p, &a.cfg)
        .into_iter()
        .map(|o| {
            let sm = cube_core::storage::folder_summary(&o.path);
            let role = format!("{}{}", if o.default { "install target" } else { "" }, if o.builtin { ", built-in" } else { "" }).trim_start_matches(", ").to_string();
            if sm.exists {
                let (media, bus, model) = cube_core::storage::drive_kind(&a.p, &o.path);
                let sp = cube_core::storage::read_speed(&a.p, &o.path, false);
                let slow = sp.map_or(false, |s| s < 500.0) || media == "HDD" || bus == "USB";
                FolderRow {
                    drive: cube_core::storage::drive_of(&o.path),
                    kind: format!("{} {}{}{}", media, if bus == "?" { "" } else { &bus }, if model != "?" && !model.is_empty() { format!(" - {model}") } else { String::new() }, if slow { "  (slow: storage)" } else { "" }),
                    speed: sp.map_or("not measured".into(), |s| format!("{} MB/s", cube_core::session::fmt_int(s as i64))),
                    free: format!("{:.0} GB", sm.free as f64 / 1e9),
                    models: format!("{} ({:.1} GB)", sm.vaults.len(), sm.size as f64 / 1e9),
                    role,
                    builtin: o.builtin,
                    path: o.path,
                }
            } else {
                FolderRow { drive: String::new(), kind: "NOT FOUND (drive disconnected?)".into(), speed: String::new(), free: String::new(), models: String::new(), role, builtin: o.builtin, path: o.path }
            }
        })
        .collect();
    hub_installed(a);
}

fn st_log(a: &mut App, s: &str) {
    a.t.st_log.push_str(s.trim_end());
    a.t.st_log.push('\n');
}

pub fn folders(a: &mut App, ctx: &egui::Context) {
    egui::CentralPanel::default().show(ctx, |ui| {
        ui.horizontal(|ui| {
            head(ui, "Folders that hold compressed models");
            ui.label(RichText::new("SSD folders run models at full speed; HDD / USB folders are fine for keeping models - move one to an SSD folder before chatting with it.").color(DIM).small());
        });
        let mut pick = None;
        ui.push_id("folders", |ui| {
            TableBuilder::new(ui)
                .striped(true)
                .max_scroll_height(240.0)
                .column(Column::remainder().at_least(300.0))
                .column(Column::auto().at_least(50.0))
                .column(Column::auto().at_least(200.0))
                .columns(Column::auto().at_least(90.0), 4)
                .header(20.0, |mut hd| {
                    for t in ["Folder", "Drive", "Disk", "Read speed", "Free", "Models", "Role"] {
                        hd.col(|ui| {
                            ui.strong(t);
                        });
                    }
                })
                .body(|mut b| {
                    for f in &a.t.st_rows {
                        b.row(20.0, |mut r| {
                            r.col(|ui| {
                                if ui.selectable_label(a.t.st_sel.as_ref() == Some(&f.path), f.path.display().to_string()).clicked() {
                                    pick = Some(f.path.clone());
                                }
                            });
                            for v in [&f.drive, &f.kind, &f.speed, &f.free, &f.models, &f.role] {
                                r.col(|ui| {
                                    ui.label(v.as_str());
                                });
                            }
                        });
                    }
                });
        });
        if let Some(p) = pick {
            a.t.st_sel = Some(p);
        }
        ui.horizontal(|ui| {
            if ui.button("Add folder...").clicked() {
                if let Some(d) = rfd::FileDialog::new().set_title("Folder for compressed models (any drive)").pick_folder() {
                    let mut c = paths::Config::load(&a.p);
                    paths::add_model_dir(&a.p, &mut c, &d, false);
                    let _ = c.save(&a.p);
                    folders_refresh(a);
                    a.status = format!("model folder added: {}", d.display());
                }
            }
            if ui.button("Remove from list").clicked() {
                if let Some(p) = a.t.st_sel.clone() {
                    if a.t.st_rows.iter().any(|r| r.path == p && r.builtin) {
                        a.status = "The built-in folder data/vaults cannot be removed.".into();
                    } else {
                        a.ask("Model folders", &format!("Remove {} from the list?\n\nNothing is deleted: its models just stop showing in the harness.", p.display()), Confirm::RemoveFolder(p));
                    }
                }
            }
            if ui.button("Use for new installs").clicked() {
                if let Some(p) = a.t.st_sel.clone() {
                    let mut c = paths::Config::load(&a.p);
                    paths::add_model_dir(&a.p, &mut c, &p, true);
                    let _ = c.save(&a.p);
                    folders_refresh(a);
                    a.status = format!("new installs go to {}", p.display());
                }
            }
            if ui.button("Open").clicked() {
                if let Some(p) = &a.t.st_sel {
                    open_path(p);
                }
            }
            if ui.add_enabled(!a.t.st_busy, egui::Button::new("Measure read speed")).clicked() {
                if let Some(p) = a.t.st_sel.clone() {
                    a.t.st_busy = true;
                    st_log(a, &format!("measuring {} (reads 1 GB of its largest model) ...", p.display()));
                    let (pp, tx) = (a.p.clone(), a.jtx.clone());
                    std::thread::spawn(move || {
                        let r = cube_core::storage::read_speed(&pp, &p, true);
                        let _ = tx.send(Job::Log(Tab::Folders, match r {
                            Some(s) => format!("{}: {} MB/s", p.display(), cube_core::session::fmt_int(s as i64)),
                            None => format!("{}: no model in it to measure with", p.display()),
                        }));
                        let _ = tx.send(Job::Done(Tab::Folders));
                    });
                }
            }
            if ui.button("Refresh").clicked() {
                folders_refresh(a);
            }
        });
        ui.separator();
        ui.label("Move a model to the selected folder (copy, verify every stream, then delete the original)");
        ui.horizontal(|ui| {
            let names: Vec<(String, PathBuf)> = paths::find_vaults(&a.p, &a.cfg);
            egui::ComboBox::from_id_salt("mvvault").width(420.0).selected_text(if a.t.st_vault.is_empty() { "(pick a model)".to_string() } else { a.t.st_vault.clone() }).show_ui(ui, |ui| {
                for (n, v) in &names {
                    ui.selectable_value(&mut a.t.st_vault, n.clone(), format!("{n}   ({})", v.parent().map(|x| x.display().to_string()).unwrap_or_default()));
                }
            });
            if ui.add_enabled(!a.t.st_busy, egui::Button::new(RichText::new("Move here").color(ACCENT))).clicked() {
                let src = names.iter().find(|x| x.0 == a.t.st_vault).map(|x| x.1.clone());
                match (a.t.st_sel.clone(), src) {
                    (Some(dst), Some(src)) if src.parent() != Some(dst.as_path()) => {
                        a.ask("Move model", &format!("Move {}\nfrom {}\nto {}?\n\nIt is copied, every stream of the copy is verified, then the original is deleted.", a.t.st_vault, src.parent().unwrap().display(), dst.display()), Confirm::Move(src, dst));
                    }
                    _ => a.status = "Select a destination folder in the list and a model in the box.".into(),
                }
            }
        });
        ui.label("Download folder for new models (the original weights; can be deleted after install)");
        ui.horizontal(|ui| {
            ui.add(TextEdit::singleline(&mut a.t.st_dl).desired_width(480.0));
            if ui.button("Browse...").clicked() {
                if let Some(d) = rfd::FileDialog::new().set_title("Download folder for new models").pick_folder() {
                    a.t.st_dl = d.display().to_string();
                }
            }
            if ui.button("Save").clicked() {
                a.cfg = paths::Config::load(&a.p);
                a.cfg.set("hub_download_dir", if a.t.st_dl.trim().is_empty() { J::Null } else { json!(a.t.st_dl.trim()) });
                a.save_cfg();
                a.status = "download folder saved".into();
            }
        });
        ScrollArea::vertical().id_salt("stlog").stick_to_bottom(true).auto_shrink([false; 2]).show(ui, |ui| {
            ui.label(RichText::new(&a.t.st_log).monospace().small());
        });
    });
}

// ====================================================================== Config
const FIELDS: [(&str, &str, &str); 17] = [
    ("vault", "Model vault", "vault"),
    ("mode", "Execution mode (auto, cpu, gpu, split)", "mode"),
    ("gpu_layers", "GPU layers in split mode (0 = as many as fit)", "int"),
    ("ctx", "Context length (tokens)", "int"),
    ("max_new", "Max reply tokens", "int"),
    ("max_file_growth", "Max file growth per reply (chars, 0 = 4 x max reply)", "int"),
    ("temp", "Temperature", "float"),
    ("top_p", "Top-p", "float"),
    ("top_k", "Top-k (0 = off)", "int"),
    ("greedy", "Greedy (deterministic)", "bool"),
    ("seed", "Seed (blank = random)", "str"),
    ("think", "Let the model think first (Qwen3 / Hermes)", "bool"),
    ("kv16", "KV cache in fp16 (half the memory, ~2e-4 rounding)", "bool"),
    ("max_tool_rounds", "Max tool calls per reply", "int"),
    ("resident_mb", "RAM for pinned weights, MB (0 = automatic)", "int"),
    ("threads", "Compute threads (0 = all but 2)", "int"),
    ("python", "Python for .py plugins (blank = find it)", "str"),
];

pub fn config_fill(a: &mut App) {
    for (k, _, kind) in FIELDS {
        if kind == "bool" {
            a.t.cf_bool.insert(k.into(), a.cfg.bool(k));
        } else {
            let v = match a.cfg.get(k) {
                J::Null => String::new(),
                J::String(s) => s.clone(),
                v => v.to_string(),
            };
            a.t.cf.insert(k.into(), v);
        }
    }
    a.t.cf_sys = a.cfg.str("system");
}

fn config_save(a: &mut App) -> bool {
    let mut c = paths::Config::load(&a.p);
    for (k, label, kind) in FIELDS {
        let v = a.t.cf.get(k).cloned().unwrap_or_default();
        let val = match kind {
            "bool" => json!(a.t.cf_bool.get(k).copied().unwrap_or(false)),
            "int" => match v.trim().parse::<i64>() {
                Ok(x) => json!(x),
                Err(_) => {
                    a.status = format!("Bad value for '{label}': {v}");
                    return false;
                }
            },
            "float" => match v.trim().parse::<f64>() {
                Ok(x) => json!(x),
                Err(_) => {
                    a.status = format!("Bad value for '{label}': {v}");
                    return false;
                }
            },
            _ if k == "seed" => {
                if v.trim().is_empty() {
                    J::Null
                } else {
                    match v.trim().parse::<u64>() {
                        Ok(x) => json!(x),
                        Err(_) => {
                            a.status = format!("Bad value for '{label}': {v}");
                            return false;
                        }
                    }
                }
            }
            _ => json!(v.trim()),
        };
        c.set(k, val);
    }
    c.set("system", json!(a.t.cf_sys.trim()));
    a.cfg = c;
    a.save_cfg();
    a.reload_tools();
    a.status = "settings saved to harness_config.json".into();
    true
}

pub fn config(a: &mut App, ctx: &egui::Context) {
    egui::SidePanel::left("cfgleft").resizable(true).default_width(560.0).show(ctx, |ui| {
        ScrollArea::vertical().show(ui, |ui| {
            head(ui, "Settings");
            let vaults: Vec<String> = paths::find_vaults(&a.p, &a.cfg).into_iter().map(|x| x.0).collect();
            egui::Grid::new("cfg").num_columns(2).spacing([12.0, 6.0]).show(ui, |ui| {
                for (k, label, kind) in FIELDS {
                    ui.label(label);
                    match kind {
                        "bool" => {
                            let b = a.t.cf_bool.entry(k.into()).or_default();
                            ui.checkbox(b, "");
                        }
                        "mode" => {
                            let v = a.t.cf.entry(k.into()).or_default();
                            if v.is_empty() {
                                *v = "auto".into();
                            }
                            egui::ComboBox::from_id_salt("cfgmode").width(120.0).selected_text(v.clone()).show_ui(ui, |ui| {
                                for n in ["auto", "cpu", "gpu", "split"] {
                                    ui.selectable_value(v, n.to_string(), n);
                                }
                            });
                        }
                        "vault" => {
                            let v = a.t.cf.entry(k.into()).or_default();
                            egui::ComboBox::from_id_salt("cfgvault").width(300.0).selected_text(v.clone()).show_ui(ui, |ui| {
                                for n in &vaults {
                                    ui.selectable_value(v, n.clone(), n);
                                }
                            });
                        }
                        _ => {
                            let v = a.t.cf.entry(k.into()).or_default();
                            ui.add(TextEdit::singleline(v).desired_width(if kind == "str" { 300.0 } else { 120.0 }));
                        }
                    }
                    ui.end_row();
                }
                ui.label("System prompt");
                ui.add(TextEdit::multiline(&mut a.t.cf_sys).desired_rows(6).desired_width(340.0));
                ui.end_row();
            });
            ui.horizontal(|ui| {
                if ui.button(RichText::new("Save").color(ACCENT)).clicked() {
                    config_save(a);
                }
                if ui.button("Save and reload model").clicked() && config_save(a) {
                    a.unload_model();
                    a.load_model();
                }
                if ui.button("Revert").clicked() {
                    a.cfg = paths::Config::load(&a.p);
                    config_fill(a);
                }
                if ui.button("Model defaults").clicked() {
                    let name = a.t.cf.get("vault").cloned().unwrap_or_default();
                    match paths::resolve_vault(&a.p, &a.cfg, &name).and_then(|v| cube_core::settings::recommended(&v)) {
                        Ok((rec, notes)) => {
                            for (k, v) in &rec {
                                if FIELDS.iter().any(|f| f.0 == k && f.2 == "bool") {
                                    a.t.cf_bool.insert(k.clone(), v.as_bool().unwrap_or(false));
                                } else if FIELDS.iter().any(|f| f.0 == k) {
                                    a.t.cf.insert(k.clone(), match v {
                                        J::String(s) => s.clone(),
                                        v => v.to_string(),
                                    });
                                } else {
                                    a.cfg.set(k, v.clone());
                                }
                            }
                            a.t.cf_plan = format!(
                                "{}\n\nApplied to the fields on the left: {}\nPress Save (or Save and reload model) to keep them.",
                                notes.join("\n"),
                                rec.iter().map(|(k, v)| format!("{k}={v}")).collect::<Vec<_>>().join(", ")
                            );
                            a.status = "model defaults filled in - press Save to keep them".into();
                        }
                        Err(e) => a.status = format!("Model defaults: {e}"),
                    }
                }
            });
        });
    });
    egui::CentralPanel::default().show(ctx, |ui| {
        ui.horizontal(|ui| {
            head(ui, "Hardware and execution plan");
            if ui.button("Show machine").clicked() {
                a.t.cf_plan = machine_text(a);
            }
        });
        ScrollArea::vertical().auto_shrink([false; 2]).show(ui, |ui| {
            if !a.banner.is_empty() {
                ui.label(RichText::new(format!("{}\n{}", a.banner, a.how)).monospace());
                ui.separator();
            }
            ui.label(RichText::new(&a.t.cf_plan).monospace());
        });
    });
}

fn machine_text(a: &App) -> String {
    let (tot, av) = cube_core::hw::ram_mb();
    let g = cube_core::hw::gpu();
    let mut s = format!(
        "CPU threads: {}\nRAM: {:.1} GB total, {:.1} GB available\nGPU: {}\n",
        cube_core::hw::threads(),
        tot / 1024.0,
        av / 1024.0,
        g.map_or("none found (nvidia-smi)".to_string(), |g| format!("{} - {:.1} GB VRAM, {:.1} GB free, driver {}", g.name, g.vram_total_mb / 1024.0, g.vram_free_mb / 1024.0, g.driver))
    );
    s.push_str(&format!("CUDA engine: {}\n", match cube_gpu::gpu() { Some(g) => format!("{} (compute {}.{}), VRAM free {:.1} GB", g.name, g.cc.0, g.cc.1, cube_gpu::vram().map_or(0.0, |v| v.0 as f64 / 1e9)), None => "not available (no NVIDIA driver)".into() }));
    s.push_str(&format!("CPU features: avx2={} fma={} (engine built {})\n", is_x86_feature("avx2"), is_x86_feature("fma"), if cube_engine::simd::ON { "with AVX2/FMA kernels" } else { "with portable kernels" }));
    if let Ok(v) = paths::resolve_vault(&a.p, &a.cfg, &a.cfg.str("vault")) {
        if let Ok(vault) = cube_vault::Vault::open(&v) {
            let m = vault.meta();
            let total: f64 = (0..vault.n_streams()).map(|i| vault.stream(i).padded as f64).sum();
            let kv = cube_core::hw::kv_bytes(m, a.cfg.i64("ctx").max(64) as usize, a.cfg.bool("kv16"));
            s.push_str(&format!(
                "\nSelected vault {}: {:?}, {} layers, {:.2} GB stored\nKV cache at ctx {}: {:.2} GB ({})\nPinned automatically: free RAM - 1.5 GB - KV ({:.1} GB available now); the rest streams from the drive each token.",
                v.display(),
                m.arch,
                m.layers,
                total / 1e9,
                a.cfg.i64("ctx"),
                kv / 1e9,
                if a.cfg.bool("kv16") { "fp16" } else { "fp32" },
                (av - 1500.0 - kv / 1e6).max(0.0) / 1024.0
            ));
        }
    }
    s
}

fn is_x86_feature(f: &str) -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        match f {
            "avx2" => std::arch::is_x86_feature_detected!("avx2"),
            "fma" => std::arch::is_x86_feature_detected!("fma"),
            _ => false,
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = f;
        false
    }
}

// ====================================================================== Plugins
pub fn plugins_refresh_pending(a: &mut App) {
    let d = a.p.plugins().join("pending");
    let _ = std::fs::create_dir_all(&d);
    let re = regex_lite(r#"(?m)^DESCRIPTION\s*=\s*\(?\s*["'](.+?)["']"#);
    let mut v: Vec<(String, u64, String, String)> = std::fs::read_dir(&d)
        .map(|r| {
            r.flatten()
                .map(|e| e.path())
                .filter(|p| p.extension().map_or(false, |x| x == "py"))
                .map(|p| {
                    let src = String::from_utf8_lossy(&std::fs::read(&p).unwrap_or_default()).into_owned();
                    let md = std::fs::metadata(&p).ok();
                    let when = md.as_ref().and_then(|m| m.modified().ok()).map(|t| chrono_like(t)).unwrap_or_default();
                    (p.file_stem().unwrap().to_string_lossy().into_owned(), md.map_or(0, |m| m.len()), when, re(&src).chars().take(120).collect())
                })
                .collect()
        })
        .unwrap_or_default();
    v.sort();
    a.t.pend = v;
}

fn regex_lite(p: &str) -> impl Fn(&str) -> String {
    let r = cube_core::reply_regex(p);
    move |s: &str| r.captures(s).map(|c| c[1].to_string()).unwrap_or_default()
}

fn chrono_like(t: std::time::SystemTime) -> String {
    cube_core::local_time(t)
}

pub fn plugins(a: &mut App, ctx: &egui::Context) {
    egui::CentralPanel::default().show(ctx, |ui| {
        let tl = a.tools();
        head(ui, "Tools the model can call");
        let enabled = a.cfg.enabled_plugins();
        let mut toggle = None;
        let mut pick = None;
        ui.push_id("tools", |ui| {
            TableBuilder::new(ui)
                .striped(true)
                .max_scroll_height(220.0)
                .column(Column::auto().at_least(70.0))
                .column(Column::auto().at_least(130.0))
                .column(Column::remainder())
                .header(20.0, |mut hd| {
                    for t in ["Enabled", "Plugin", "What it does"] {
                        hd.col(|ui| {
                            ui.strong(t);
                        });
                    }
                })
                .body(|mut b| {
                    for t in &tl.list {
                        let n = t.name();
                        b.row(22.0, |mut r| {
                            r.col(|ui| {
                                let mut on = enabled.contains(&n);
                                if ui.add_enabled(t.broken().is_none(), egui::Checkbox::new(&mut on, "")).changed() {
                                    toggle = Some(n.clone());
                                }
                            });
                            r.col(|ui| {
                                if ui.selectable_label(a.t.pl_sel.as_deref() == Some(&n), format!("{n}{}", if t.builtin() { "" } else { "  (.py)" })).clicked() {
                                    pick = Some(n.clone());
                                }
                            });
                            r.col(|ui| {
                                let d = t.broken().map(|b| format!("({b})")).unwrap_or_else(|| t.description());
                                ui.label(RichText::new(d.chars().take(220).collect::<String>()).small()).on_hover_text(t.description());
                            });
                        });
                    }
                });
        });
        if let Some(p) = pick {
            a.t.pl_sel = Some(p);
        }
        if let Some(n) = toggle {
            let mut on = a.cfg.enabled_plugins();
            if let Some(i) = on.iter().position(|x| *x == n) {
                on.remove(i);
            } else {
                on.push(n);
            }
            a.cfg.set("enabled_plugins", json!(on));
            a.save_cfg();
        }
        ui.horizontal(|ui| {
            if ui.button("Test...").clicked() {
                if let Some(n) = a.t.pl_sel.clone() {
                    if let Some(t) = tl.get(&n) {
                        let props: Vec<(String, String)> = t.parameters()["properties"].as_object().map(|o| o.iter().map(|(k, v)| (format!("{k}: {}", v["description"].as_str().unwrap_or("")), String::new())).collect()).unwrap_or_default();
                        a.t.prompt = Some(Prompt { title: format!("Test {n}"), fields: props, kind: PromptKind::TestPlugin(n) });
                    }
                }
            }
            if ui.button("Reload plugins").clicked() {
                a.reload_tools();
                a.status = format!("{} tools loaded", a.tools().list.len());
            }
            if ui.button("Open plugins folder").clicked() {
                open_path(&a.p.plugins());
            }
            ui.label(RichText::new("A .py plugin defines NAME, DESCRIPTION, PARAMETERS (JSON schema) and run(**args) -> str; it runs in its own Python process.").color(DIM).small());
        });
        ui.separator();
        head(ui, "Written by the model - not loaded until you approve (plugins/pending)");
        let mut pick = None;
        ui.push_id("pend", |ui| {
            TableBuilder::new(ui)
                .striped(true)
                .max_scroll_height(120.0)
                .column(Column::auto().at_least(130.0))
                .column(Column::auto().at_least(70.0))
                .column(Column::auto().at_least(140.0))
                .column(Column::remainder())
                .header(20.0, |mut hd| {
                    for t in ["Plugin", "Size", "Written", "Description (from its source)"] {
                        hd.col(|ui| {
                            ui.strong(t);
                        });
                    }
                })
                .body(|mut b| {
                    for (n, sz, when, d) in &a.t.pend {
                        b.row(20.0, |mut r| {
                            r.col(|ui| {
                                if ui.selectable_label(a.t.pend_sel.as_deref() == Some(n), n).clicked() {
                                    pick = Some(n.clone());
                                }
                            });
                            r.col(|ui| {
                                ui.label(format!("{sz} B"));
                            });
                            r.col(|ui| {
                                ui.label(when);
                            });
                            r.col(|ui| {
                                ui.label(d);
                            });
                        });
                    }
                });
        });
        if let Some(p) = pick {
            a.t.pend_sel = Some(p);
        }
        ui.horizontal(|ui| {
            if ui.button("View code").clicked() {
                if let Some(n) = &a.t.pend_sel {
                    let src = std::fs::read_to_string(a.p.plugins().join("pending").join(format!("{n}.py"))).unwrap_or_default();
                    a.t.view = Some((format!("{n}.py - model-written plugin (not loaded)"), src));
                }
            }
            if ui.button(RichText::new("Approve and enable").color(ACCENT)).clicked() {
                if let Some(n) = a.t.pend_sel.clone() {
                    a.ask("Approve plugin", &format!("Load '{n}' as a tool the model can call?\n\nIt will run as Python with your permissions. Only approve code you have read and trust."), Confirm::Approve(n));
                }
            }
            if ui.button("Discard").clicked() {
                if let Some(n) = a.t.pend_sel.clone() {
                    a.ask("Discard", &format!("Delete the pending plugin '{n}'?"), Confirm::Discard(n));
                }
            }
            ui.label(RichText::new("Plugins run as you, with full access to this computer: read the code before approving.").color(DIM).small());
        });
        ui.separator();
        head(ui, "Connections (all relative to the harness folder)");
        ScrollArea::vertical().id_salt("conns").auto_shrink([false; 2]).show(ui, |ui| {
            egui::Grid::new("conns").num_columns(3).striped(true).spacing([16.0, 4.0]).show(ui, |ui| {
                let row = |ui: &mut egui::Ui, a: &str, b: String, c: String| {
                    ui.label(a);
                    ui.label(RichText::new(b).monospace().small());
                    ui.label(c);
                    ui.end_row();
                };
                row(ui, "Engine", "built in (cube-engine, CPU)".into(), if cube_engine::simd::ON { "AVX2/FMA kernels".into() } else { "portable kernels".into() });
                for (name, p) in paths::find_vaults(&a.p, &a.cfg) {
                    let st = match cube_vault::Vault::open(&p) {
                        Ok(v) => format!(
                            "{:.2} GB {:?}, {} layers{}{}",
                            cube_core::storage::vault_size(&p) as f64 / 1e9,
                            v.meta().codec,
                            v.meta().layers,
                            if v.file("tokenizer.json").is_some() { "" } else { ", NO TOKENIZER" },
                            if v.legacy_version.is_some() { ", v3-v7 (Convert to v8 with: cube convert)" } else { ", v8" }
                        ),
                        Err(e) => format!("unreadable: {e}"),
                    };
                    row(ui, &format!("vault  {name}"), a.p.rel(&p), st);
                }
                for (label, d) in [("conversations", a.p.conversations()), ("skills", a.p.skills()), ("plugins", a.p.plugins()), ("artifacts", a.p.artifacts()), ("logs", a.p.logs()), ("settings", a.p.config())] {
                    let ok = d.exists();
                    row(ui, label, a.p.rel(&d), if ok { "found".into() } else { "missing".into() });
                }
            });
            ui.horizontal(|ui| {
                if ui.button("Open harness folder").clicked() {
                    open_path(&a.p.root);
                }
                if ui.button("Open logs").clicked() {
                    open_path(&a.p.logs());
                }
            });
        });
    });
}

// ====================================================================== Skills
pub fn skills(a: &mut App, ctx: &egui::Context) {
    egui::SidePanel::left("skills").resizable(true).default_width(250.0).show(ctx, |ui| {
        ui.add_space(6.0);
        if ui.add_sized([ui.available_width(), 28.0], egui::Button::new(RichText::new("+  New skill").color(ACCENT))).clicked() {
            a.t.sk_old = None;
            a.t.sk_sel = None;
            a.t.sk_name = "new-skill".into();
            a.t.sk_desc.clear();
            a.t.sk_on = false;
            a.t.sk_body.clear();
        }
        ScrollArea::vertical().auto_shrink([false; 2]).show(ui, |ui| {
            let mut pick = None;
            for s in &a.skills {
                let lab = format!(
                    "{}/{}{}{}",
                    if s.always_on { "* " } else { "" },
                    s.name,
                    if s.author == "model" { "   [by model]" } else { "" },
                    if s.applies_when == "tools" { "   [auto when tools are on]" } else { "" }
                );
                if ui.selectable_label(a.t.sk_sel.as_deref() == Some(&s.name), lab).clicked() {
                    pick = Some(s.clone());
                }
            }
            if let Some(s) = pick {
                a.t.sk_sel = Some(s.name.clone());
                a.t.sk_old = Some(s.name.clone());
                a.t.sk_name = s.name;
                a.t.sk_desc = s.description;
                a.t.sk_on = s.always_on;
                a.t.sk_body = s.body;
            }
        });
    });
    egui::CentralPanel::default().show(ctx, |ui| {
        egui::Grid::new("sk").num_columns(2).spacing([12.0, 6.0]).show(ui, |ui| {
            ui.label("Name (use as /name)");
            ui.add(TextEdit::singleline(&mut a.t.sk_name).desired_width(260.0));
            ui.end_row();
            ui.label("Description");
            ui.add(TextEdit::singleline(&mut a.t.sk_desc).desired_width(f32::INFINITY));
            ui.end_row();
            ui.label("");
            ui.checkbox(&mut a.t.sk_on, "Always on (added to every system prompt)");
            ui.end_row();
        });
        ui.label("Instructions");
        let h = ui.available_height() - 40.0;
        ScrollArea::vertical().id_salt("skbody").max_height(h).show(ui, |ui| {
            ui.add(TextEdit::multiline(&mut a.t.sk_body).desired_width(f32::INFINITY).desired_rows(24));
        });
        ui.horizontal(|ui| {
            if ui.button(RichText::new("Save").color(ACCENT)).clicked() {
                if a.t.sk_body.trim().is_empty() {
                    a.status = "Write some instructions first.".into();
                } else {
                    match cube_core::skills::save(&a.p.skills(), &a.t.sk_name, &a.t.sk_desc, a.t.sk_on, &a.t.sk_body, a.t.sk_old.as_deref()) {
                        Ok(n) => {
                            a.t.sk_old = Some(n.clone());
                            a.t.sk_sel = Some(n.clone());
                            a.t.sk_name = n.clone();
                            a.skills = cube_core::skills::load(&a.p.skills());
                            a.status = format!("skill /{n} saved");
                        }
                        Err(e) => a.status = format!("cannot save the skill: {e}"),
                    }
                }
            }
            if ui.button("Delete").clicked() {
                if let Some(n) = a.t.sk_old.clone() {
                    a.ask("Skill", &format!("Delete /{n}?"), Confirm::DeleteSkill(n));
                }
            }
            if ui.button("Use in next message").clicked() {
                if let Some(n) = &a.t.sk_old {
                    a.input = format!("/{n} {}", a.input);
                    a.tab = Tab::Messages;
                }
            }
            ui.label(RichText::new("Start a message with /name to insert a skill; several can be chained: /concise /verify ...").color(DIM).small());
        });
    });
}

// ====================================================================== dialogs and jobs
pub fn on_job(a: &mut App, j: Job) {
    match j {
        Job::Log(Tab::Models, s) => hub_log(a, &s),
        Job::Log(_, s) => st_log(a, &s),
        Job::Found(res, hw) => {
            a.t.hub_hw = format!(
                "this machine: {:.0} GB RAM for weights, {:.1} GB VRAM, vaults {:.0} GB free, downloads {:.0} GB free",
                hw.ram_w / 1e9,
                hw.vram / 1e9,
                hw.vault_free / 1e9,
                hw.dl_free / 1e9
            );
            a.status = format!("{} models - checking each against this machine ...", res.len());
            a.t.hub_res = res;
        }
        Job::Assessed(x) => {
            a.t.hub_assessed.insert(x.repo.clone(), x);
        }
        Job::Status(s) => a.status = s,
        Job::Done(Tab::Models) => {
            a.t.hub_busy = false;
            hub_installed(a);
        }
        Job::Done(_) => {
            a.t.st_busy = false;
            folders_refresh(a);
        }
    }
}

fn do_confirm(a: &mut App, c: Confirm) {
    match c {
        Confirm::DeleteChat(id) => {
            cube_core::convs::delete(&a.p.conversations(), &id);
            a.conv = None;
            a.refresh_convs(true);
        }
        Confirm::DeleteArtifact(n) => {
            let _ = a.tools().artifacts.delete_artifact(&n);
            a.t.art_name = None;
            a.t.art_path = None;
            a.t.art_text.clear();
            a.t.art_head = "artifact deleted".into();
            a.t.art_files.clear();
            a.t.art_hist.clear();
            artifacts_refresh(a);
        }
        Confirm::Revert(n, v) => {
            a.status = a.tools().artifacts.act(&json!({"action": "revert", "name": n, "version": v}));
            artifacts_refresh(a);
        }
        Confirm::Install(repo) => hub_install(a, repo),
        Confirm::Compress(src, dst) => hub_compress(a, src, dst),
        Confirm::Move(src, dst) => {
            if !release_chat(a) {
                return;
            }
            a.t.st_busy = true;
            let tx = a.jtx.clone();
            std::thread::spawn(move || {
                let t2 = tx.clone();
                if let Err(e) = cube_core::storage::move_vault(&src, &dst, &mut |s| {
                    let _ = t2.send(Job::Log(Tab::Folders, s));
                }) {
                    let _ = tx.send(Job::Log(Tab::Folders, format!("move stopped: {e:#}")));
                }
                let _ = tx.send(Job::Done(Tab::Folders));
            });
        }
        Confirm::RemoveFolder(p) => {
            let mut c = paths::Config::load(&a.p);
            paths::remove_model_dir(&a.p, &mut c, &p);
            let _ = c.save(&a.p);
            folders_refresh(a);
        }
        Confirm::Approve(n) => {
            let src = a.p.plugins().join("pending").join(format!("{n}.py"));
            let dst = a.p.plugins().join(format!("{n}.py"));
            if dst.exists() {
                a.status = format!("plugins/{n}.py already exists.");
                return;
            }
            if let Err(e) = std::fs::rename(&src, &dst) {
                a.status = format!("cannot approve: {e}");
                return;
            }
            a.reload_tools();
            let ok = a.tools().get(&n).map_or(false, |t| t.broken().is_none());
            if ok {
                let mut on = a.cfg.enabled_plugins();
                if !on.contains(&n) {
                    on.push(n.clone());
                }
                a.cfg.set("enabled_plugins", json!(on));
                a.save_cfg();
                a.status = format!("plugin '{n}' approved and enabled");
            } else {
                a.status = format!("plugin '{n}' failed to load: {}", a.tools().get(&n).and_then(|t| t.broken()).unwrap_or_else(|| "not loaded".into()));
            }
            plugins_refresh_pending(a);
        }
        Confirm::Discard(n) => {
            let _ = std::fs::remove_file(a.p.plugins().join("pending").join(format!("{n}.py")));
            plugins_refresh_pending(a);
        }
        Confirm::DeleteSkill(n) => {
            cube_core::skills::delete(&a.p.skills(), &n);
            a.t.sk_old = None;
            a.t.sk_sel = None;
            a.t.sk_name.clear();
            a.t.sk_desc.clear();
            a.t.sk_body.clear();
            a.skills = cube_core::skills::load(&a.p.skills());
        }
    }
}

fn do_prompt(a: &mut App, pr: Prompt) {
    let f = |i: usize| pr.fields.get(i).map(|x| x.1.trim().to_string()).unwrap_or_default();
    match pr.kind {
        PromptKind::RenameChat => {
            if !f(0).is_empty() {
                if let Some(c) = a.conv.as_mut() {
                    c["title"] = json!(f(0));
                }
                a.save_conv();
            }
        }
        PromptKind::NewArtifact => {
            if f(0).is_empty() {
                return;
            }
            let t = if f(1).is_empty() { "note".to_string() } else { f(1).to_lowercase() };
            a.status = a.tools().artifacts.act(&json!({"action": "create", "name": f(0), "type": t, "title": f(0)}));
            let n = cube_core::tools::artifact::slug(&f(0));
            a.t.art_name = Some(n.clone());
            artifacts_refresh(a);
            art_show(a, &n, false);
        }
        PromptKind::NewFile => {
            if let (Some(n), false) = (a.t.art_name.clone(), f(0).is_empty()) {
                a.status = a.tools().artifacts.act(&json!({"action": "write", "name": n, "path": f(0), "content": ""}));
                a.t.art_path = Some(f(0).replace('\\', "/").trim_matches('/').to_string());
                artifacts_refresh(a);
            }
        }
        PromptKind::TestPlugin(n) => {
            let tl = a.tools();
            let props: Vec<String> = tl.get(&n).and_then(|t| t.parameters()["properties"].as_object().map(|o| o.keys().cloned().collect())).unwrap_or_default();
            let mut args = serde_json::Map::new();
            for (i, k) in props.iter().enumerate() {
                let v = f(i);
                if !v.is_empty() {
                    args.insert(k.clone(), serde_json::from_str::<J>(&v).ok().filter(|x| !x.is_string() || v.starts_with('"')).unwrap_or(json!(v)));
                }
            }
            let r = tl.run(&n, &J::Object(args));
            a.t.view = Some((format!("{n} result"), r.chars().take(3000).collect()));
        }
    }
}

pub fn dialogs(a: &mut App, ctx: &egui::Context) {
    if let Some((title, text, what)) = a.confirm.clone() {
        let mut done: Option<bool> = None;
        egui::Window::new(title).collapsible(false).resizable(false).anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0]).show(ctx, |ui| {
            ui.label(&text);
            ui.horizontal(|ui| {
                if ui.button(RichText::new("Yes").color(ACCENT)).clicked() {
                    done = Some(true);
                }
                if ui.button("No").clicked() {
                    done = Some(false);
                }
            });
        });
        if let Some(y) = done {
            a.confirm = None;
            if y {
                do_confirm(a, what);
            }
        }
    }
    if let Some(mut pr) = a.t.prompt.clone() {
        let mut done: Option<bool> = None;
        egui::Window::new(pr.title.clone()).collapsible(false).resizable(false).anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0]).show(ctx, |ui| {
            for (label, v) in pr.fields.iter_mut() {
                ui.label(label.as_str());
                let r = ui.add(TextEdit::singleline(v).desired_width(380.0));
                if r.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                    done = Some(true);
                }
            }
            ui.horizontal(|ui| {
                if ui.button(RichText::new("OK").color(ACCENT)).clicked() {
                    done = Some(true);
                }
                if ui.button("Cancel").clicked() {
                    done = Some(false);
                }
            });
        });
        match done {
            Some(true) => {
                a.t.prompt = None;
                do_prompt(a, pr);
            }
            Some(false) => a.t.prompt = None,
            None => a.t.prompt = Some(pr),
        }
    }
    if let Some((title, text)) = a.t.view.clone() {
        let mut open = true;
        egui::Window::new(title).open(&mut open).default_size([820.0, 520.0]).vscroll(true).show(ctx, |ui| {
            let mut t = text.clone();
            ui.add(TextEdit::multiline(&mut t).code_editor().desired_width(f32::INFINITY).interactive(true));
        });
        if !open {
            a.t.view = None;
        }
    }
    let _ = is_vault;
}
