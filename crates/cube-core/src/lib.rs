//! The harness core: everything the app does that is not drawing — paths and settings, tokenizer and chat templates,
//! reply parsing, tools, skills, conversations and the model session that runs the engine.

pub mod convs;
pub mod hub;
pub mod hw;
pub mod session;
pub mod paths;
pub mod reply;
pub mod runners;
pub mod settings;
pub mod skills;
pub mod storage;
pub mod template;
pub mod tools;
pub mod util;

/// True when a CUDA engine is available in this build (the GPU tier is not wired up yet).
pub fn gpu_available() -> bool {
    false
}

/// A compiled regular expression (for front ends that do not depend on `regex` themselves).
pub fn reply_regex(p: &str) -> regex::Regex {
    regex::Regex::new(p).unwrap()
}

/// "YYYY-MM-DD HH:MM" in local time.
pub fn local_time(t: std::time::SystemTime) -> String {
    let dt: chrono::DateTime<chrono::Local> = t.into();
    dt.format("%Y-%m-%d %H:%M").to_string()
}
