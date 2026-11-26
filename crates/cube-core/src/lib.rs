//! The harness core: everything the app does that is not drawing — paths and settings, tokenizer and chat templates,
//! reply parsing, tools, skills, conversations and the model session that runs the engine.

pub mod convs;
pub mod hw;
pub mod session;
pub mod paths;
pub mod reply;
pub mod skills;
pub mod template;
pub mod tools;
pub mod util;
