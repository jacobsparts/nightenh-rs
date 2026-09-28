//! Bisect the op list for the first non-deterministic op.
//!
//!     cargo run --release --example op_determinism [h] [w]
//!
//! Runs the GPU executor twice over the SAME input with the plan's op list
//! truncated to n ops, for every n, and reports the first n at which the two runs
//! disagree; the op at index n-1 is then the one that is not deterministic.
//!
//! Use it rather than guessing: at 64x64 it names the offending op with the max
//! |a - b| at that point, where a tolerance check at the output shows only an
//! intermittent difference that may not reproduce. The failure this shape catches
//! is a kernel that reads a block-reduced value out of shared memory and then
//! reuses the same array for the next reduction; the test `the_gpu_is_deterministic`
//! in tests/reference.rs is the guard for the one that is fixed, and this example
//! locates a new one.

use nightenh::config::Variant;
use nightenh::host::Host;
use nightenh::model;
use nightenh::weights::Weights;

fn main() {
    let w = Weights::open("../models/nightenh-lol.safetensors").unwrap();
    let v = Variant::from_weights(&w).unwrap();
    println!("variant {} n_res {} ngf {}", v.variant, v.n_res, v.ngf);
    let a: Vec<String> = std::env::args().skip(1).collect();
    let h: usize = a.first().map(|s| s.parse().unwrap()).unwrap_or(16);
    let wd: usize = a.get(1).map(|s| s.parse().unwrap()).unwrap_or(h);
    let plan = model::build(&w, h, wd).unwrap();
    let input: Vec<f32> = (0..3 * h * wd).map(|i| ((i % 97) as f32 / 97.0) - 0.5).collect();
    let n_ops = plan.ops.len();
    println!("{n_ops} ops");

    for n in 1..=n_ops {
        let mut a = Vec::new();
        let mut b = Vec::new();
        for pass in 0..2 {
            let mut p = plan.clone();
            p.ops.truncate(n);
            let mut gpu = nightenh::exec_gpu::Gpu::new(&p, &w).expect("gpu");
            let mut host = Host::new(p, input.clone(), true, [0, 0, 0, 0]);
            gpu.run(&mut host).unwrap();
            if pass == 0 { a = host.arena.clone() } else { b = host.arena.clone() }
        }
        let d = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
        let kind = plan.ops[n - 1].kind();
        if d > 1e-6 {
            println!("FIRST DIVERGENCE at n={n} op {kind}");
            println!("  max |a - b| = {d:.3e}");
            println!("  op debug: {:?}", plan.ops[n - 1]);
            return;
        }
        println!("n={n:3} {kind:<16} ok");
    }
    println!("no divergence in {n_ops} ops");
}
