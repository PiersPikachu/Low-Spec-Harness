fn main() {
    let a: Vec<String> = std::env::args().collect();
    let p = cube_core::paths::Paths::at(std::path::PathBuf::from("/home/claude/hroot"));
    let c = cube_core::paths::Config::load(&p);
    match a[1].as_str() {
        "search" => for m in cube_core::hub::search(&a[2], 8).unwrap() { println!("{} {} {:?}", m.repo, m.downloads, m.params); },
        "assess" => { let hw = cube_core::hub::hardware(&p, &c); let r = cube_core::hub::assess(&a[2], &hw); println!("{} -> {}", r.repo, r.verdict); for (q, x, s) in r.tree { println!("  [{s}] {q}: {x}"); } }
        "install" => { let v = cube_core::hub::install(&p, &c, &a[2], false, &mut |s| println!("{s}"), &|| false).unwrap(); println!("{}", v.display()); }
        "local" => for l in cube_core::hub::local_checkpoints(&p, &c) { println!("{} {} {:?} {} {:?}", l.dir.display(), l.model_type, l.kind, l.bytes, l.vault); },
        "compress" => { let v = cube_core::hub::compress_local(&p, &c, std::path::Path::new(&a[2]), std::path::Path::new(&a[3]), None, false, &mut |s| println!("{s}"), &|| false).unwrap(); println!("{}", v.display()); }
        _ => {}
    }
}
