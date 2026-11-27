//! Cube Harness — the window (egui). Tabs: Messages, Artifacts, Models, Model folders, Config, Plugins, Skills.
#![cfg_attr(all(windows, not(debug_assertions)), windows_subsystem = "windows")]

mod tabs;
mod worker;

use cube_core::paths::{Config, Paths};
use cube_core::session::Event;
use cube_core::tools::Tools;
use eframe::egui::{self, Color32, RichText};
use serde_json::{json, Value as J};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};
use worker::{Cmd, Msg, Worker};

pub const ACCENT: Color32 = Color32::from_rgb(0x7a, 0xa2, 0xff);
pub const DIM: Color32 = Color32::from_rgb(0x94, 0x9b, 0xa4);
pub const TOOL: Color32 = Color32::from_rgb(0xc9, 0xa2, 0x6b);
pub const ASST: Color32 = Color32::from_rgb(0x8f, 0xd1, 0x9e);
pub const THINK: Color32 = Color32::from_rgb(0x8a, 0x8f, 0x99);
pub const BAD: Color32 = Color32::from_rgb(0xe0, 0x6c, 0x75);

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Tab {
    Messages,
    Artifacts,
    Models,
    Folders,
    Config,
    Plugins,
    Skills,
}

/// Messages from background jobs (hub search, install, tune, folder measure/move) to the window.
pub enum Job {
    Log(Tab, String),
    Found(Vec<cube_core::hub::Found>, cube_core::hub::Hw),
    Assessed(cube_core::hub::Assessment),
    Done(Tab),
    Status(String),
}

/// One item of the activity tree shown while a reply runs.
#[derive(Clone, Default)]
pub struct ActItem {
    pub label: String,
    pub status: String,
    pub detail: String,
}
#[derive(Clone, Default)]
pub struct Round {
    pub n: i64,
    pub status: String,
    pub items: Vec<ActItem>,
    pub think: Option<usize>,
    pub call: Option<usize>,
}

/// The reply being streamed.
#[derive(Default)]
pub struct Live {
    pub model: String,
    pub rounds: Vec<Round>,
    /// (kind, text) pieces in arrival order: "think", "answer", "tool", "head"
    pub parts: Vec<(String, String)>,
    pub answer: String,
    pub think: String,
}

pub struct App {
    pub p: Paths,
    pub cfg: Config,
    pub tab: Tab,
    pub status: String,
    pub worker: Worker,
    pub tools: Arc<Mutex<Arc<Tools>>>,
    pub skills: Vec<cube_core::skills::Skill>,
    pub busy: bool,
    pub forced: bool,
    pub model_label: String,
    pub banner: String,
    pub how: String,
    // messages
    pub convs: Vec<J>,
    pub conv: Option<J>,
    pub input: String,
    pub live: Option<Live>,
    pub detail: Option<(String, String)>,
    pub rename: Option<String>,
    pub confirm: Option<(String, String, tabs::Confirm)>,
    // jobs
    pub jtx: Sender<Job>,
    pub jrx: Receiver<Job>,
    pub t: tabs::State,
}

impl App {
    fn new(cc: &eframe::CreationContext) -> App {
        let p = Paths::discover();
        p.ensure_dirs();
        seed_data(&p);
        let cfg = Config::load(&p);
        style(&cc.egui_ctx);
        let tools = Arc::new(Mutex::new(Arc::new(Tools::load(&p, &cfg.str("python")))));
        let worker = Worker::start(p.clone(), tools.clone(), cc.egui_ctx.clone());
        let (jtx, jrx) = channel();
        let mut a = App {
            skills: cube_core::skills::load(&p.skills()),
            p,
            cfg,
            tab: Tab::Messages,
            status: if instance_lock() { "ready - no model loaded".into() } else { "WARNING: another Cube Harness window is open - each loads its own model into the same RAM and VRAM; close one".into() },
            worker,
            tools,
            busy: false,
            forced: false,
            model_label: "no model loaded".into(),
            banner: String::new(),
            how: String::new(),
            convs: Vec::new(),
            conv: None,
            input: String::new(),
            live: None,
            detail: None,
            rename: None,
            confirm: None,
            jtx,
            jrx,
            t: tabs::State::default(),
        };
        a.refresh_convs(true);
        tabs::init(&mut a);
        // smoke tests: send one message at startup
        if let Ok(m) = std::env::var("CUBE_HARNESS_AUTOSEND") {
            a.input = m;
            a.send();
        }
        a
    }

    pub fn tools(&self) -> Arc<Tools> {
        self.tools.lock().unwrap().clone()
    }

    pub fn reload_tools(&mut self) {
        *self.tools.lock().unwrap() = Arc::new(Tools::load(&self.p, &self.cfg.str("python")));
    }

    pub fn save_cfg(&mut self) {
        if let Err(e) = self.cfg.save(&self.p) {
            self.status = format!("cannot save settings: {e}");
        }
    }

    pub fn refresh_convs(&mut self, select_first: bool) {
        self.convs = cube_core::convs::list(&self.p.conversations());
        if select_first {
            if let Some(c) = self.convs.first() {
                self.conv = Some(c.clone());
            } else {
                self.new_chat();
            }
        }
    }

    pub fn new_chat(&mut self) {
        if self.busy {
            return;
        }
        let mut c = cube_core::convs::new_conversation(&self.cfg.str("vault"));
        let _ = cube_core::convs::save(&self.p.conversations(), &mut c);
        self.conv = Some(c);
        self.convs = cube_core::convs::list(&self.p.conversations());
    }

    pub fn save_conv(&mut self) {
        if let Some(c) = self.conv.as_mut() {
            let _ = cube_core::convs::save(&self.p.conversations(), c);
        }
        let id = self.conv.as_ref().map(|c| c["id"].clone());
        self.convs = cube_core::convs::list(&self.p.conversations());
        if let Some(id) = id {
            if let Some(c) = self.convs.iter().find(|c| c["id"] == id) {
                self.conv = Some(c.clone());
            }
        }
    }

    pub fn system(&self) -> String {
        let mut parts = vec![self.cfg.str("system").trim().to_string()];
        parts.extend(self.skills.iter().filter(|s| s.always_on).map(|s| s.body.clone()));
        parts.into_iter().filter(|x| !x.is_empty()).collect::<Vec<_>>().join("\n\n")
    }

    fn ensure_vault(&mut self) -> bool {
        if self.cfg.str("vault").is_empty() {
            match cube_core::paths::find_vaults(&self.p, &self.cfg).first() {
                Some((n, _)) => {
                    self.cfg.set("vault", json!(n));
                    self.save_cfg();
                }
                None => {
                    self.status = "no vault found - install one in the Models tab or add a model folder".into();
                    return false;
                }
            }
        }
        true
    }

    pub fn load_model(&mut self) {
        if self.busy || !self.ensure_vault() {
            return;
        }
        self.begin();
        let _ = self.worker.tx.send(Cmd::Load(self.cfg.clone()));
    }

    pub fn unload_model(&mut self) {
        if self.busy {
            return;
        }
        let _ = self.worker.tx.send(Cmd::Unload(None));
    }

    pub fn send(&mut self) {
        if self.busy {
            return;
        }
        let typed = self.input.trim().to_string();
        if typed.is_empty() || !self.ensure_vault() {
            return;
        }
        if self.conv.is_none() {
            self.new_chat();
        }
        self.input.clear();
        let (content, used) = cube_core::skills::expand(&typed, &self.skills);
        let vault = self.cfg.str("vault");
        {
            let c = self.conv.as_mut().unwrap();
            c["messages"].as_array_mut().unwrap().push(json!({"role": "user", "content": content, "meta": {"typed": typed, "skills": used}}));
            if c["title"] == "New chat" {
                c["title"] = json!(typed.lines().next().unwrap_or("").chars().take(40).collect::<String>());
            }
            c["vault"] = json!(vault);
        }
        self.save_conv();
        let history: Vec<J> = self.conv.as_ref().unwrap()["messages"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| {
                let mut m = m.clone();
                if let Some(o) = m.as_object_mut() {
                    o.remove("meta");
                }
                m
            })
            .collect();
        self.begin();
        self.live = Some(Live::default());
        let _ = self.worker.tx.send(Cmd::Reply {
            cfg: self.cfg.clone(),
            history,
            system: self.system(),
            enabled: self.cfg.enabled_plugins(),
            max_rounds: self.cfg.i64("max_tool_rounds").max(0) as usize,
        });
    }

    fn begin(&mut self) {
        self.busy = true;
        self.forced = false;
        self.worker.stop.store(false, std::sync::atomic::Ordering::SeqCst);
        self.status = "working ...".into();
    }

    fn idle(&mut self) {
        self.busy = false;
    }

    fn on_msg(&mut self, m: Msg) {
        match m {
            Msg::Info(t) => self.status = t.trim().to_string(),
            Msg::Loaded { name, how, banner, secs } => {
                self.model_label = format!("{name} - {how}");
                self.status = format!("{name} - {how}  (ready in {secs:.0}s)");
                self.banner = banner;
                self.how = how;
            }
            Msg::Unloaded => {
                self.model_label = "no model loaded".into();
                self.status = "model unloaded".into();
            }
            Msg::Start(name) => {
                if let Some(l) = self.live.as_mut() {
                    l.model = name;
                }
            }
            Msg::Ev(e) => self.on_event(e),
            Msg::Done(new, stats) => {
                if let Some(c) = self.conv.as_mut() {
                    c["messages"].as_array_mut().unwrap().extend(new);
                }
                self.save_conv();
                self.live = None;
                self.status = format!("{}  |  last reply: {stats}", self.how);
                self.refresh_after_reply();
                self.idle();
            }
            Msg::Error(e) => {
                if self.live.is_some() {
                    let l = self.live.take().unwrap();
                    if self.forced && (!l.answer.is_empty() || !l.think.is_empty()) {
                        let mut m = json!({"role": "assistant", "content": format!("{}\n[force-stopped]", l.answer.trim()), "meta": {"stats": "force-stopped"}});
                        if !l.think.is_empty() {
                            m["thinking"] = json!(l.think);
                        }
                        if let Some(c) = self.conv.as_mut() {
                            c["messages"].as_array_mut().unwrap().push(m);
                        }
                        self.save_conv();
                    }
                }
                self.tools().artifacts.live_keep();
                self.status = if self.forced { "force-stopped".into() } else { format!("error: {}", e.lines().last().unwrap_or("")) };
                let _ = std::fs::write(self.p.logs().join("harness_error.txt"), &e);
                self.refresh_after_reply();
                self.idle();
            }
            Msg::Idle => self.idle(),
        }
    }

    fn refresh_after_reply(&mut self) {
        self.skills = cube_core::skills::load(&self.p.skills());
        tabs::artifacts_refresh(self);
        tabs::plugins_refresh_pending(self);
    }

    fn on_event(&mut self, e: Event) {
        let mut live_ev: Option<J> = None;
        let mut refresh_art = false;
        if let Event::Act(ev) = &e {
            if ev["ev"] == "call" && ev["name"] == "artifact" {
                let n = cube_core::tools::artifact::slug(ev["args"]["name"].as_str().unwrap_or(""));
                if !n.is_empty() {
                    let p = ev["args"]["path"].as_str().map(|p| p.replace('\\', "/").trim_matches('/').to_string()).filter(|p| !p.is_empty());
                    self.t.art_live_target = Some((n, p));
                }
            }
        }
        self.on_event_inner(e, &mut live_ev, &mut refresh_art);
        if let Some(ev) = live_ev {
            tabs::artifacts_live(self, &ev);
        }
        if refresh_art {
            tabs::artifacts_refresh(self);
        }
    }

    fn on_event_inner(&mut self, e: Event, live_ev: &mut Option<J>, refresh_art: &mut bool) {
        let Some(l) = self.live.as_mut() else { return };
        let push = |l: &mut Live, kind: &str, t: &str| {
            if let Some(last) = l.parts.last_mut() {
                if last.0 == kind {
                    last.1.push_str(t);
                    return;
                }
            }
            l.parts.push((kind.to_string(), t.to_string()));
        };
        match e {
            Event::Thinking(t) => {
                l.think.push_str(&t);
                push(l, "think", &t);
                let all: String = l.parts.iter().rev().take_while(|p| p.0 == "think").map(|p| p.1.clone()).collect();
                let r = cur_round(l);
                let label = format!("○ thinking ({} words)", all.split_whitespace().count());
                let tail: String = all.split_whitespace().collect::<Vec<_>>().join(" ");
                let tail: String = tail.chars().rev().take(70).collect::<Vec<_>>().into_iter().rev().collect();
                match r.think {
                    Some(i) => {
                        r.items[i].label = label;
                        r.items[i].status = tail;
                        r.items[i].detail = all;
                    }
                    None => {
                        r.items.push(ActItem { label, status: tail, detail: all });
                        r.think = Some(r.items.len() - 1);
                    }
                }
            }
            Event::Assistant(t) => {
                l.answer.push_str(&t);
                push(l, "answer", &t);
                cur_round(l).think = None;
            }
            Event::AssistantNext => push(l, "head", ""),
            Event::Tool(t) => push(l, "tool", &t),
            Event::Act(ev) => {
                let k = ev["ev"].as_str().unwrap_or("");
                if k == "round" {
                    if let Some(r) = l.rounds.last_mut() {
                        r.status = "done".into();
                    }
                    l.rounds.push(Round { n: ev["n"].as_i64().unwrap_or(1), status: "working ...".into(), ..Default::default() });
                    return;
                }
                let r = cur_round(l);
                match k {
                    "call_stream" => {
                        let wh = format!("{}{}", ev["target"].as_str().unwrap_or(""), ev["path"].as_str().map(|p| format!("/{p}")).unwrap_or_default());
                        let label = format!("⚙ {} · {} · {wh}", ev["name"].as_str().unwrap_or(""), ev["action"].as_str().unwrap_or("..."));
                        let st = match ev["text"].as_str() {
                            Some(t) => format!("writing ... {} chars", t.chars().count()),
                            None => format!("composing call ... {} chars", ev["chars"]),
                        };
                        let det = ev["text"].as_str().unwrap_or("").to_string();
                        match r.call {
                            Some(i) => {
                                r.items[i].label = label;
                                r.items[i].status = st;
                                if !det.is_empty() {
                                    r.items[i].detail = det.clone();
                                }
                            }
                            None => {
                                r.items.push(ActItem { label, status: st, detail: det.clone() });
                                r.call = Some(r.items.len() - 1);
                            }
                        }
                        if ev["name"] == "artifact" {
                            *live_ev = Some(ev.clone());
                        }
                    }
                    "call" => {
                        let a = &ev["args"];
                        let wh = format!("{}{}", a["name"].as_str().unwrap_or(""), a["path"].as_str().map(|p| format!("/{p}")).unwrap_or_default());
                        let label = format!("⚙ {} · {} · {wh}", ev["name"].as_str().unwrap_or(""), a["action"].as_str().unwrap_or(""));
                        let det = serde_json::to_string_pretty(a).unwrap_or_default();
                        match r.call {
                            Some(i) => {
                                r.items[i].label = label;
                                r.items[i].status = "running ...".into();
                                r.items[i].detail = det;
                            }
                            None => {
                                r.items.push(ActItem { label, status: "running ...".into(), detail: det });
                                r.call = Some(r.items.len() - 1);
                            }
                        }
                    }
                    "result" => {
                        let res = ev["text"].as_str().unwrap_or("").to_string();
                        let ok = !res.starts_with("error");
                        let i = match r.call {
                            Some(i) => i,
                            None => {
                                r.items.push(ActItem { label: format!("⚙ {}", ev["name"].as_str().unwrap_or("")), ..Default::default() });
                                r.items.len() - 1
                            }
                        };
                        let short: String = res.split_whitespace().collect::<Vec<_>>().join(" ").chars().take(70).collect();
                        r.items[i].status = format!("{} {short}", if ok { "✓" } else { "✗" });
                        r.items[i].detail.push_str(&format!("\n\n--- result ---\n{res}"));
                        r.call = None;
                        r.think = None;
                        if ev["name"] == "artifact" {
                            *refresh_art = true;
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    pub fn ask(&mut self, title: &str, text: &str, what: tabs::Confirm) {
        self.confirm = Some((title.into(), text.into(), what));
    }
}

fn cur_round(l: &mut Live) -> &mut Round {
    if l.rounds.is_empty() {
        l.rounds.push(Round { n: 1, status: "working ...".into(), ..Default::default() });
    }
    l.rounds.last_mut().unwrap()
}

/// First run: the default skill that teaches the tool-call format.
fn seed_data(p: &Paths) {
    let f = p.skills().join("tool-calls.md");
    if !f.exists() && std::fs::read_dir(p.skills()).map(|mut r| r.next().is_none()).unwrap_or(true) {
        let _ = std::fs::write(f, include_str!("../assets/tool-calls.md"));
    }
}

fn style(ctx: &egui::Context) {
    let mut v = egui::Visuals::dark();
    v.panel_fill = Color32::from_rgb(0x1e, 0x1f, 0x22);
    v.window_fill = Color32::from_rgb(0x2b, 0x2d, 0x31);
    v.extreme_bg_color = Color32::from_rgb(0x31, 0x33, 0x38);
    v.selection.bg_fill = Color32::from_rgb(0x40, 0x45, 0x52);
    v.hyperlink_color = ACCENT;
    ctx.set_visuals(v);
    let mut s = (*ctx.style()).clone();
    s.spacing.item_spacing = egui::vec2(8.0, 6.0);
    s.spacing.button_padding = egui::vec2(10.0, 4.0);
    ctx.set_style(s);
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _: &mut eframe::Frame) {
        while let Ok(m) = self.worker.rx.try_recv() {
            self.on_msg(m);
        }
        while let Ok(j) = self.jrx.try_recv() {
            tabs::on_job(self, j);
        }
        egui::TopBottomPanel::top("tabs").show(ctx, |ui| {
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                for (t, n) in [
                    (Tab::Messages, "Messages"),
                    (Tab::Artifacts, "Artifacts"),
                    (Tab::Models, "Models"),
                    (Tab::Folders, "Model folders"),
                    (Tab::Config, "Config"),
                    (Tab::Plugins, "Plugins"),
                    (Tab::Skills, "Skills"),
                ] {
                    if ui.selectable_label(self.tab == t, RichText::new(format!("  {n}  ")).size(14.0)).clicked() {
                        self.tab = t;
                    }
                }
            });
            ui.add_space(2.0);
        });
        egui::TopBottomPanel::bottom("status").show(ctx, |ui| {
            ui.horizontal(|ui| {
                if self.busy {
                    ui.spinner();
                }
                ui.label(RichText::new(&self.status).color(DIM));
            });
        });
        match self.tab {
            Tab::Messages => tabs::messages(self, ctx),
            Tab::Artifacts => tabs::artifacts(self, ctx),
            Tab::Models => tabs::models(self, ctx),
            Tab::Folders => tabs::folders(self, ctx),
            Tab::Config => tabs::config(self, ctx),
            Tab::Plugins => tabs::plugins(self, ctx),
            Tab::Skills => tabs::skills(self, ctx),
        }
        tabs::dialogs(self, ctx);
        if self.busy || self.t.hub_busy || self.t.st_busy {
            ctx.request_repaint_after(std::time::Duration::from_millis(250));
        }
    }
}

/// true when this is the only window: holds a localhost port for the life of the process
fn instance_lock() -> bool {
    static L: std::sync::OnceLock<Option<std::net::TcpListener>> = std::sync::OnceLock::new();
    L.get_or_init(|| std::net::TcpListener::bind(("127.0.0.1", 47613)).ok()).is_some()
}

fn main() -> eframe::Result<()> {
    let opts = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_title("Cube Harness").with_inner_size([1240.0, 820.0]).with_min_inner_size([960.0, 620.0]),
        ..Default::default()
    };
    eframe::run_native("Cube Harness", opts, Box::new(|cc| Ok(Box::new(App::new(cc)))))
}
