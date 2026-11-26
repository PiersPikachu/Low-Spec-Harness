// render chat template cases: render <model_dir> <cases.json>  ->  JSON list of {text, ids}
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let t = cube_core::template::Tok::from_dir(std::path::Path::new(&a[1])).unwrap();
    let cases: serde_json::Value = serde_json::from_slice(&std::fs::read(&a[2]).unwrap()).unwrap();
    let mut out = vec![];
    for c in cases.as_array().unwrap() {
        let extra = c["extra"].as_object().cloned().unwrap_or_default();
        let tools = c.get("tools").filter(|x| !x.is_null());
        match t.render(&c["messages"], tools, true, &extra) {
            Ok(s) => { let ids = t.encode(&s).unwrap(); out.push(serde_json::json!({"text": s, "ids": ids, "decoded": t.decode(&ids, false)})); }
            Err(e) => out.push(serde_json::json!({"error": format!("{e:#}")})),
        }
    }
    println!("{}", serde_json::to_string(&out).unwrap());
}
