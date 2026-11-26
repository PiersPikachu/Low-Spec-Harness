use cube_engine::kernels::Mat;
use cube_vault::{huff4::Huff4, rans4::Rans4, stream, Codec, Vault};
use std::time::Instant;
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let v = Vault::open(&a[1]).unwrap();
    let th: usize = a.get(2).map_or(2, |x| x.parse().unwrap());
    rayon::ThreadPoolBuilder::new().num_threads(th).build_global().unwrap();
    let buf = v.read_stream(0).unwrap();
    let subs = stream::sub_blobs(&buf).unwrap();
    for (k, sb) in subs.iter().enumerate().take(7) {
        let m = match v.meta().codec { Codec::Huff4 => Mat::Huff(Huff4::parse(sb).unwrap()), Codec::Rans4 => Mat::Rans(Rans4::parse(sb).unwrap()) };
        let (r, c) = (m.rows(), m.cols());
        for n in [1usize, 16] {
            let x: Vec<f32> = (0..n * c).map(|i| ((i * 37 % 101) as f32 - 50.0) * 0.01).collect();
            let mut y = vec![0f32; n * r];
            m.mv(&x, &mut y, n);
            let t = Instant::now();
            let it = if n == 1 { 20 } else { 4 };
            for _ in 0..it { m.mv(&x, &mut y, n); }
            let ms = t.elapsed().as_secs_f64() * 1e3 / it as f64;
            println!("sub {k} {r}x{c} n={n}: {ms:.2} ms  ({:.0} Mweights/s)", (r * c) as f64 / ms / 1e3);
        }
    }
}
