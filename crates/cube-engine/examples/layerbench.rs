// time one full matvec pass over every layer stream (pinned), n=1, as the engine would see it
use cube_engine::kernels::Mat;
use cube_vault::{huff4::Huff4, stream, Vault};
use std::time::Instant;
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let v = Vault::open(&a[1]).unwrap();
    let th: usize = a[2].parse().unwrap();
    rayon::ThreadPoolBuilder::new().num_threads(th).build_global().unwrap();
    let bufs: Vec<Vec<u8>> = (0..v.meta().layers).map(|l| v.read_stream(l).unwrap()).collect();
    let x: Vec<f32> = (0..8192).map(|i| ((i * 37 % 101) as f32 - 50.0) * 0.01).collect();
    let mut y = vec![0f32; 8192];
    for it in 0..3 {
        let t = Instant::now();
        let mut tp = 0.0;
        for b in &bufs {
            for sb in stream::sub_blobs(b).unwrap().into_iter().take(7) {
                let t1 = Instant::now();
                let m = Mat::Huff(Huff4::parse(sb).unwrap());
                tp += t1.elapsed().as_secs_f64();
                m.mv(&x, &mut y, 1);
            }
        }
        println!("pass {it}: {:.1} ms (parse {:.1} ms)", t.elapsed().as_secs_f64() * 1e3, tp * 1e3);
    }
}
