//! The model worker: one thread owns the loaded Session and runs replies; the window talks to it through channels.

use cube_core::paths::{Config, Paths};
use cube_core::session::{Event, Session};
use cube_core::tools::Tools;
use serde_json::Value as J;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;
use std::time::Instant;

pub enum Cmd {
    Load(Config),
    /// drop the loaded model; the sender (if any) is told once its RAM and VRAM are free
    Unload(Option<std::sync::mpsc::Sender<()>>),
    Reply { cfg: Config, history: Vec<J>, system: String, enabled: Vec<String>, max_rounds: usize },
}

pub enum Msg {
    Info(String),
    Loaded { name: String, how: String, banner: String, secs: f64 },
    Unloaded,
    Start(String),
    Ev(Event),
    Done(Vec<J>, String),
    Error(String),
    Idle,
}

/// What identifies a loaded engine: a change to any of these reloads the model.
pub fn key(c: &Config) -> String {
    ["vault", "ctx", "kv16", "resident_mb", "threads", "mode", "gpu_layers", "cpu_head"].iter().map(|k| c.get(k).to_string()).collect::<Vec<_>>().join("|")
}

pub struct Worker {
    pub tx: Sender<Cmd>,
    pub rx: Receiver<Msg>,
    /// "Stop": end after the current step (the reply loop's should_stop)
    pub stop: Arc<AtomicBool>,
    /// "Force stop": the engine aborts at the next token or prefill chunk
    pub abort: Arc<std::sync::Mutex<Option<(Arc<AtomicBool>, Arc<AtomicBool>)>>>,
}

impl Worker {
    pub fn start(p: Paths, tools: Arc<std::sync::Mutex<Arc<Tools>>>, ctx: eframe::egui::Context) -> Worker {
        let (tx, crx) = channel::<Cmd>();
        let (mtx, rx) = channel::<Msg>();
        let stop = Arc::new(AtomicBool::new(false));
        let abort: Arc<std::sync::Mutex<Option<(Arc<AtomicBool>, Arc<AtomicBool>)>>> = Arc::new(std::sync::Mutex::new(None));
        let (stop2, abort2) = (stop.clone(), abort.clone());
        std::thread::Builder::new()
            .name("model".into())
            .stack_size(16 << 20)
            .spawn(move || {
                let mut sess: Option<Session> = None;
                let mut skey = String::new();
                let send = |m: Msg| {
                    let _ = mtx.send(m);
                    ctx.request_repaint();
                };
                let ensure = |cfg: &Config, sess: &mut Option<Session>, skey: &mut String| -> anyhow::Result<()> {
                    if sess.is_some() && *skey == key(cfg) {
                        sess.as_mut().unwrap().cfg = cfg.clone();
                        return Ok(());
                    }
                    *sess = None;
                    *abort2.lock().unwrap() = None;
                    anyhow::ensure!(!cfg.str("vault").is_empty(), "no model selected - pick one in Config or the Models tab");
                    send(Msg::Info(format!("loading {} ...\n", cfg.str("vault"))));
                    let t0 = Instant::now();
                    let s = Session::open(&p, cfg)?;
                    *abort2.lock().unwrap() = Some((s.eng.abort_flag(), s.stop.clone()));
                    send(Msg::Loaded { name: s.name.clone(), how: s.how.clone(), banner: s.banner.clone(), secs: t0.elapsed().as_secs_f64() });
                    *skey = key(cfg);
                    *sess = Some(s);
                    Ok(())
                };
                while let Ok(cmd) = crx.recv() {
                    match cmd {
                        Cmd::Load(cfg) => {
                            match ensure(&cfg, &mut sess, &mut skey) {
                                Ok(()) => {}
                                Err(e) => send(Msg::Error(format!("{e:#}"))),
                            }
                            send(Msg::Idle);
                        }
                        Cmd::Unload(done) => {
                            sess = None;
                            skey.clear();
                            *abort2.lock().unwrap() = None;
                            if let Some(d) = done {
                                let _ = d.send(());
                            }
                            send(Msg::Unloaded);
                        }
                        Cmd::Reply { cfg, history, system, enabled, max_rounds } => {
                            if let Err(e) = ensure(&cfg, &mut sess, &mut skey) {
                                send(Msg::Error(format!("{e:#}")));
                                continue;
                            }
                            let s = sess.as_mut().unwrap();
                            send(Msg::Start(s.name.clone()));
                            let tl = tools.lock().unwrap().clone();
                            let enabled = if s.chat_mode { enabled } else { Vec::new() };
                            let st = stop2.clone();
                            let r = s.reply(&p, &history, &system, &tl, &enabled, max_rounds, &mut |e| send(Msg::Ev(e)), &|| st.load(Ordering::SeqCst));
                            match r {
                                Ok((new, stats)) => send(Msg::Done(new, stats)),
                                Err(e) => send(Msg::Error(format!("{e:#}"))),
                            }
                        }
                    }
                }
            })
            .expect("model thread");
        Worker { tx, rx, stop, abort }
    }

    pub fn request_stop(&self) {
        self.stop.store(true, Ordering::SeqCst);
    }
    pub fn force_stop(&self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some((a, s)) = self.abort.lock().unwrap().as_ref() {
            a.store(true, Ordering::SeqCst);
            s.store(true, Ordering::SeqCst);
        }
    }
}
