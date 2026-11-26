use cube_engine::kernels::Mat;
use cube_vault::{huff4::Huff4, stream, Vault};
use std::time::Instant;
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let v = Vault::open(&a[1]).unwrap();
    let th: usize = a[2].parse().unwrap();
    rayon::ThreadPoolBuilder::new().num_threads(th).build_global().unwrap();
    let buf = v.read_stream(v.meta().lm_head_stream).unwrap();
    let subs = stream::sub_blobs(&buf).unwrap();
    let m = Mat::Huff(Huff4::parse(subs[0]).unwrap());
    let x: Vec<f32> = (0..m.cols()).map(|i| ((i * 37 % 101) as f32 - 50.0) * 0.01).collect();
    let mut y = vec![0f32; m.rows()];
    for _ in 0..2 { m.mv(&x, &mut y, 1); }
    let t = Instant::now(); for _ in 0..5 { m.mv(&x, &mut y, 1); }
    println!("threads {th}: lm_head {:.1} ms", t.elapsed().as_secs_f64() * 1e3 / 5.0);
}
