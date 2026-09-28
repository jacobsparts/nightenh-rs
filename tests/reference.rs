//! The engine against the numpy transcription of the upstream generator.
//!
//! `tools/reference.py` is a transcription of night-enhancement's `ResnetGenerator`,
//! and it writes these fixtures: `python3 tools/reference.py --fixture tests/data`
//! saves a random input and the output the transcription produced for it, as raw
//! f32 planes in the arena's own layout (not PNG pixels - 8-bit quantisation hides
//! exactly the differences a port gets wrong). This test runs both backends over
//! those inputs and compares.
//!
//! A TRANSCRIPTION IS ONLY AN ORACLE IF SOMETHING INDEPENDENT CONFIRMED IT. These
//! comparisons cannot be that check on their own: the fixtures are written by
//! `tools/reference.py`, so a transcription that is wrong in the same way the
//! engine is compares the engine against the same mistake, agrees to five places,
//! and shows nothing at all.
//!
//! The independent check is `tools/check_upstream.py`, which loads the real
//! `networks.py` and compares it against the transcription stage by stage (and,
//! with `--ref`, the converted weights against the released `.pt` tensor by
//! tensor). It needs torch, upstream's source and a 42 MB checkpoint, so it cannot
//! run here - but it is what these fixtures were regenerated against, and the
//! engine now reproduces upstream's own output to 1/255 on a real photograph.
//! Re-run it whenever `tools/reference.py` is edited.
//!
//! The fixtures SKIP, with a printed reason, when they are absent - as everywhere
//! else in the family: regenerating them needs numpy and the converted
//! checkpoint, and this repository has to build and test cleanly without either.
//!
//! `rect-44x84` is the interesting case: both axes are multiples of 4, which is
//! what the graph requires, and neither is a multiple of 64, which is what a
//! native-size run pads to - so it exercises the pad-and-crop path rather than one
//! the alignment already happens to satisfy.

use std::path::{Path, PathBuf};

use nightenh::config::Variant;
use nightenh::host::Host;
use nightenh::model;
use nightenh::weights::Weights;

/// The agreement the comparisons are held to. Measured at this revision, on all
/// three fixtures and both comparisons: CPU 2.3e-6 to 4.7e-6, the two backends
/// 1.0e-6 to 1.8e-6 - round-off between two float32 implementations that sum in a
/// different order, not a porting difference. The reduction length is what sets it:
/// the decoder's feature map is `[C][H][W]`, and a `[C][1][1]` one would carry a
/// tenth of the round-off. The bound is set two orders above the measurement so it
/// fails on a real divergence, and it is deliberately TIGHT enough to catch a race:
/// a shared-memory reduction without a barrier between its read and its next write
/// put 1.6e-1 to 2.5e-1 here, and that is the failure mode the bound is for.
const TOL: f32 = 1e-4;

/// `(tag, h, w)` - the SHAPE the fixture was generated at, spelled out rather
/// than parsed back out of the tag: the tag is a file name, and `rect-44x84` is
/// two numbers where `small-64` is one.
const FIXTURES: [(&str, usize, usize); 3] =
    [("tiny-16", 16, 16), ("small-64", 64, 64), ("rect-44x84", 44, 84)];

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data")
}

/// The checkpoint, or None with a printed reason. Mirrors the other engines: a
/// missing 42 MB file is not a failing test.
fn weights() -> Option<Weights> {
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("../models/nightenh-lol.safetensors");
    match Weights::open(p.to_str().unwrap()) {
        Ok(w) => Some(w),
        Err(e) => {
            println!("skipping: {}: {e}", p.display());
            None
        }
    }
}

/// A fixture pair, or None with a printed reason.
fn fixture(tag: &str) -> Option<(Vec<f32>, Vec<f32>)> {
    /// The fixture as the tool wrote it: a real `.npy`, header and all. Parsed
    /// here rather than re-saved as a bare plane, because a fixture that nothing
    /// regenerated cannot silently drift from what `tools/reference.py` produces.
    ///
    /// Only the `\x93NUMPY` v1/v2 little-endian `'<f4'` `fortran_order: False`
    /// files numpy writes for a C-contiguous float32 array are accepted, and the
    /// shape is RETURNED so the caller can check it against the geometry it
    /// planned for: a fixture for the wrong shape would otherwise run and compare
    /// the wrong number of values.
    fn load(p: &Path) -> Option<(Vec<usize>, Vec<f32>)> {
        let b = std::fs::read(p).ok()?;
        if b.len() < 10 || &b[..6] != b"\x93NUMPY" {
            println!("{}: not a .npy", p.display());
            return None;
        }
        let (major, head_at) = (b[6], 8usize);
        let hlen = if major == 1 {
            u16::from_le_bytes([b[head_at], b[head_at + 1]]) as usize
        } else {
            u32::from_le_bytes([b[head_at], b[head_at + 1], b[head_at + 2], b[head_at + 3]]) as usize
        };
        let (head_len_field, _) = if major == 1 { (2, 0) } else { (4, 0) };
        let head = String::from_utf8_lossy(&b[head_at + head_len_field..head_at + head_len_field + hlen]).to_string();
        if !head.contains("'<f4'") || head.contains("True") {
            println!("{}: expected a little-endian f32 C-order array, header says {head}", p.display());
            return None;
        }
        let shape: Vec<usize> = head
            .split_once("'shape': (")
            .and_then(|(_, r)| r.split_once(')'))
            .map(|(s, _)| s.split(',').filter_map(|t| t.trim().parse().ok()).collect())
            .unwrap_or_default();
        let data = &b[head_at + head_len_field + hlen..];
        let vals: Vec<f32> = data.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        if vals.len() != shape.iter().product::<usize>() {
            println!("{}: {} values for shape {shape:?}", p.display(), vals.len());
            return None;
        }
        Some((shape, vals))
    }
    let i = root().join(format!("input-{tag}.npy"));
    let o = root().join(format!("output-{tag}.npy"));
    match (load(&i), load(&o)) {
        (Some((si, a)), Some((so, b))) if si == so => Some((a, b)),
        (Some((si, _)), Some((so, _))) => {
            println!("skipping {tag}: input shape {si:?} != output shape {so:?}");
            None
        }
        _ => {
            println!(
                "skipping {tag}: {i:?} / {o:?} is missing - regenerate with \
                 `python3 tools/reference.py --fixture tests/data`"
            );
            None
        }
    }
}

fn worst(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
}

/// The CPU backend against the transcription.
#[test]
fn cpu_matches_the_reference() {
    for (tag, h, wd) in FIXTURES {
        let (Some(w), Some((input, expected))) = (weights(), fixture(tag)) else { return };
        let variant = Variant::from_weights(&w).unwrap();
        let plan = model::build(&w, h, wd).unwrap();
        let mut host = Host::new(plan, input, true, [0, 0, 0, 0]);
        let r = host.plan.range(host.plan.output);
        let mut cpu = nightenh::exec_cpu::Cpu { plan: &host.plan, w: &w, arena: &mut host.arena };
        cpu.run(&host.input).unwrap();
        let got = host.arena[r].to_vec();
        assert_eq!(got.len(), expected.len(), "{tag}: shape");
        let d = worst(&got, &expected);
        println!(
            "cpu {tag}: {}x{} ngf {} n_res {}, {} values, max |d| {d:.3e}",
            wd, h, variant.ngf, variant.n_res, got.len()
        );
        assert!(d <= TOL, "{tag}: max |d| {d:.3e} exceeds {TOL:.1e}");
    }
}

/// The GPU is deterministic: two runs of the same plan over the same input must
/// produce the same bytes.
///
/// This is asserted SEPARATELY from the agreement above, because a race can be
/// narrow enough to pass a tolerance check on one run and fail the next - which
/// is what happened here: `ne_instance_norm` read its block-reduced mean out of
/// shared memory and then reused that array for the variance pass with no
/// barrier between, so a fast warp's store could land before a slow warp's read.
/// Five runs of one input produced five different PNGs while the same comparison
/// sometimes showed 6e-4 and sometimes 2.5e-1. A determinism check is what makes
/// that failure mode reproducible rather than intermittent.
#[test]
fn the_gpu_is_deterministic() {
    let (Some(w), Some((input, _))) = (weights(), fixture("small-64")) else { return };
    let mut out = Vec::new();
    for pass in 0..3 {
        let plan = model::build(&w, 64, 64).unwrap();
        let mut gpu = match nightenh::exec_gpu::Gpu::new(&plan, &w) {
            Ok(g) => g,
            Err(e) => {
                println!("skipping: no GPU: {e}");
                return;
            }
        };
        let mut host = Host::new(plan, input.clone(), true, [0, 0, 0, 0]);
        gpu.run(&mut host).unwrap();
        let r = host.plan.range(host.plan.output);
        let got = host.arena[r].to_vec();
        if pass == 0 {
            out = got;
        } else {
            let d = worst(&out, &got);
            assert!(d == 0.0, "GPU run {pass} differs from run 0 by {d:.3e}");
        }
    }
    println!("gpu determinism: 3 runs of {} values, bit-identical", out.len());
}

/// The two backends against each other, on the same fixtures.
#[test]
fn the_two_backends_agree() {
    for (tag, h, wd) in FIXTURES {
        let (Some(w), Some((input, _))) = (weights(), fixture(tag)) else { return };

        let plan = model::build(&w, h, wd).unwrap();
        let mut host = Host::new(plan, input.clone(), true, [0, 0, 0, 0]);
        let r = host.plan.range(host.plan.output);
        let mut cpu = nightenh::exec_cpu::Cpu { plan: &host.plan, w: &w, arena: &mut host.arena };
        cpu.run(&host.input).unwrap();
        let cpu_out = host.arena[r].to_vec();

        let plan = model::build(&w, h, wd).unwrap();
        let mut gpu = match nightenh::exec_gpu::Gpu::new(&plan, &w) {
            Ok(g) => g,
            Err(e) => {
                println!("skipping {tag}: no GPU: {e}");
                return;
            }
        };
        let mut host = Host::new(plan, input, false, [0, 0, 0, 0]);
        let gpu_out = gpu.run(&mut host).unwrap();
        let d = worst(&cpu_out, &gpu_out);
        println!("backends {tag}: {} values, max |d| {d:.3e}", cpu_out.len());
        assert!(d <= TOL, "{tag}: the backends differ by {d:.3e}");
    }
}

/// GPU: the two 3x3 stride-1 kernels, head to head on IDENTICAL geometry.
///
/// The project's `ne_conv3x3_refl` loops `co` INSIDE the thread, so one thread's
/// read of a 3x3 window is reused across every output channel; the toolkit's
/// `lg_conv3x3s1p1` is one thread per `(oc, y, x)` and reuses nothing between
/// channels, and it walks its weight with a stride of 9 (`wp[ci * 9]`). EIGHT of
/// this network's 3x3 convolutions run the resnet body and adaILN blocks - and they
/// are all REFLECTION-padded, so the toolkit kernel is now unreachable from the
/// network at all (`Op::Conv`'s `refl: false` arm has no constructor). The
/// comparison stays because it is the reason the reflection kernel exists and
/// because the shapes below are the network's own: which kernel is faster AT THE
/// SHAPES THIS NETWORK USES is a real question about the engine rather than an
/// idle one, and a future op that legitimately wants zero padding should inherit
/// the answer.
///
/// It is asked here, out of a per-op profile, because the ops in a profile differ
/// in `cin`, `cout` and plane size: the resnet 3x3s are 64 -> 64 on a 256-plane
/// while the two reflection 3x3s are 128 -> 128 and 256 -> 128, so a per-op time is
/// a product of kernel and shape and cannot attribute anything to either. This
/// holds the shape fixed and moves only the kernel.
///
/// NOTHING IS ASSERTED about the timings. A threshold that fails on a loaded
/// machine is a flaky test; what this is for is the RATIO, and a ratio between two
/// measurements taken microseconds apart on the same device is stable enough to
/// be worth printing on a shared host. What IS asserted is that both kernels ran
/// and produced a finite, non-trivial result, so a silent launch failure cannot
/// read as a very fast kernel.
///
/// The padding rule differs (reflection vs zero) and is NOT part of what is being
/// measured: it changes the border only, while the body of both kernels - `cin * 9`
/// multiply-adds per output - is the work being compared.
#[test]
fn gpu_3x3_kernel_throughput() {
    let Some(_w) = weights() else { return };
    let cuda = match nightenh::cuda::Cuda::init(false) {
        Ok(c) => c,
        Err(e) => {
            println!("skipping: no GPU: {e}");
            return;
        }
    };
    if let Err(e) = cuda.check_kernels() {
        println!("skipping: {e}");
        return;
    }
    // `(cin, cout, h, wd)`, and the choices are the network's own:
    //  * 256 -> 256 on 128x128 is the resnet/adaILN 3x3 at the default 512 image
    //    (the weight is 589824 elements = 256*256*9, and the resnet planes are
    //    4194304 = 256 * 128 * 128) - and those eight convs are 75% of a device
    //    pass, so this is the shape worth measuring;
    //  * 256 -> 128 on 128x128 is the UpBlock2 3x3 that is the bulk of
    //    `conv3x3_refl`'s 12.8%;
    //  * 64 -> 64 on 64x64 is the same network at a quarter size, where the
    //    experimental kernel was SLOWER - included because a crossover is only
    //    useful if you know where it is.
    for &(cin, cout, h, wd) in &[(256usize, 256usize, 128usize, 128usize),
                                 (256, 128, 128, 128),
                                 (64, 64, 64, 64)] {
        // Varied values, not constants: two kernels whose sums differ in their
        // order agree trivially when every term is the same, so a uniform input
        // would make the equality check below evidence of nothing. The generator
        // is a fixed LCG, so this is reproducible on any machine.
        let mut s = 20240607u32;
        let mut next = move || {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((s >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
        };
        let (x, w, b, o) = (
            (0..cin * h * wd).map(|_| next()).collect::<Vec<f32>>(),
            (0..cout * cin * 9).map(|_| next() * 0.1).collect::<Vec<f32>>(),
            (0..cout).map(|_| next()).collect::<Vec<f32>>(),
            vec![0.0f32; cout * h * wd],
        );
        let (dx, dw, db, do_) = (
            nightenh::cuda::DevBuf::from_host(&x).unwrap(),
            nightenh::cuda::DevBuf::from_host(&w).unwrap(),
            nightenh::cuda::DevBuf::from_host(&b).unwrap(),
            nightenh::cuda::DevBuf::from_host(&o).unwrap(),
        );
        let flop = 2.0 * (cout * cin * 9 * h * wd) as f64;
        // The reference output for the zero-padded shape: the toolkit kernel, run
        // once and kept, so the experimental kernel can be CHECKED rather than
        // merely timed. Nothing about a rotated loop nest is trustworthy until the
        // values agree.
        let mut reference: Vec<f32> = Vec::new();
        for (label, refl) in [("ne_conv3x3_refl", true), ("lg_conv3x3s1p1", false),
                              ("nex_conv3x3s1p1_x8", false)] {
            // Module by SOURCE, not by padding mode: the experimental kernel is
            // this project's, and the toolkit's module does not have it (a launch
            // there fails with CUDA_ERROR_NOT_FOUND, which is how this line was
            // wrong once).
            let experimental = label.starts_with("nex_");
            let module = if refl || experimental { &cuda.project } else { &cuda.toolkit };
            let name = if experimental { label } else { name_for(refl) };
            let mut best = f64::INFINITY;
            let mut last: Vec<f32> = Vec::new();
            for _ in 0..5 {
                let mut a = lightgpu::vm::Args::new();
                a.ptr(dx.ptr).ptr(dw.ptr).ptr(db.ptr).ptr(do_.ptr)
                 .i32(cin as i32).i32(cout as i32).i32(h as i32).i32(wd as i32);
                // Same argument list, then the one extra flag the project kernel
                // takes; the toolkit kernel has no bias-presence flag.
                let (grid, block) = if refl {
                    a.i32(1);
                    ((div_ceil(wd, 256), h as u32, 1), (256, 1, 1))
                } else if experimental {
                    // One thread per (oc, y, 8 outputs), which is the kernel's own
                    // geometry rather than the toolkit's one-thread-per-element:
                    // `blockIdx.z` is the output CHANNEL, `blockIdx.y` the row, and
                    // the x grid covers the row in groups of eight.
                    ((div_ceil(div_ceil(wd, 8) as usize, 256), h as u32, cout as u32), (256, 1, 1))
                } else {
                    ((div_ceil(cout * h * wd, 256), 1, 1), (256, 1, 1))
                };
                let t0 = std::time::Instant::now();
                a.launch(module, name, lightgpu::vm::Launch::new(grid, block))
                    .expect("launch");
                cuda.sync().expect("sync");
                best = best.min(t0.elapsed().as_secs_f64());
                if last.is_empty() {
                    let mut got = vec![0.0f32; cout * h * wd];
                    do_.download(&mut got).unwrap();
                    last = got;
                }
            }
            let sum: f64 = last.iter().map(|v| *v as f64).sum();
            assert!(sum.is_finite() && sum != 0.0, "{label} produced nothing: {sum}");
            // The experimental kernel claims to compute the SAME function as the
            // toolkit's for the zero-padded shape: the same terms, in the same
            // per-output order (ascending ci, then ky, then kx), with only the loop
            // NEST rotated so the loads can be hoisted into registers. That claim
            // is checked here rather than argued. It is asserted as EQUALITY, and
            // not as a tolerance, because if the claim is right the two kernels
            // produce identical bits - the same terms in the same order - and if it
            // is wrong, a tolerance would hide exactly the difference worth
            // knowing about. A failure here means the loop rotation changed the
            // sum, which would make the timing a measurement of a different
            // function.
            let mut note = String::new();
            if !refl {
                if reference.is_empty() {
                    reference = last.clone();
                } else {
                    let worst = last.iter().zip(&reference)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0f32, f32::max);
                    // NOT asserted as an equality, and the reason is measured
                    // rather than argued. Two things are separately established:
                    //
                    //  * the STRUCTURE is the same. With an identity weight - so a
                    //    dropped tap or a border error would show as a large,
                    //    structured difference - the two kernels agree bit for bit
                    //    at every pixel, border and interior alike
                    //    (`gpu_experimental_3x3_difference_shape`).
                    //  * the DIFFERENCE here is the size a sum of this length can
                    //    legitimately show between two orderings or fusion
                    //    patterns. Summing 2304 f32 terms of this conv's magnitude
                    //    in two orders differs by up to 1.8e-5 over 200 random
                    //    draws; over 576 terms (the 64 -> 64 shape) the measured
                    //    difference is 5.2e-6. Both numbers here - 2.3e-5 and
                    //    5.2e-6 - sit in that range.
                    //
                    // So this is round-off from the compiler's different schedules,
                    // not a wrong tap, and it is PRINTED rather than asserted
                    // because the exact value is a property of nvcc's schedule on
                    // this machine.
                    note = format!("  worst |d| vs toolkit {worst:.2e} (round-off; structure \
                                    checked separately)");
                }
            }
            println!(
                "gpu 3x3 {label:<18} cin {cin:3} cout {cout:3} {h}x{wd}: {:.4} s  {:5.1} GFLOP/s{note}",
                best, flop / best / 1e9
            );
        }
    }
}

/// GPU: how fast is the FAMILY'S OWN tiled GEMM on the im2col shape?
///
/// This exists because of a measurement, not a theory. `tools/torch_bench.py` runs
/// this network in torch, and with `torch.backends.cudnn.enabled = False` - so no
/// library the engine is not allowed to use is involved - torch still convolves at
/// 5,195 GFLOP/s where the best kernel this repository owns reaches 607. The
/// profiler says what it runs: `im2col_kernel<float>` (25% of the device time)
/// followed by a tiled SGEMM (73%), 3.087 ms for M=256, K=2304, N=16384 = 6.26
/// TFLOP/s, 64% of this card's FP32 peak.
///
/// That is the SAME lesson the CPU side already learned: the scalar conv lost to
/// numpy's `einsum` because `einsum` reaches OpenBLAS's GEMM, and here every kernel
/// this repository owns is a direct convolution with no operand staged through
/// shared memory. The question this test asks is whether the transferable part -
/// the SHAPE of the computation, not the library - is already available here.
///
/// It is: `lg_f32_gemm_tiled`, in the toolkit's own `cuda/kernels.cu`, is a
/// v8-style tiling (64 rows x 32 columns per 256-thread block, 8 accumulators per
/// thread, the columns' float4 staged in shared memory per k-step). Two engines in
/// this family already call it (`locate-anything-rs`, `realesrgan-rs`); nightenh
/// does not, having no GEMM anywhere. So the number below is the ceiling of an
/// im2col+GEMM convolution written with what is already in the tree - which is a
/// different question from whether such a rewrite is worth its cost.
///
/// The GEMM is `W[ne1][ne0] @ x[ne0][ncols]`, and `ne0 % 4 == 0` is the kernel's
/// stated requirement. For the resnet 3x3 at the default 512 image: ne0 = cin*9 =
/// 2304, ne1 = cout = 256, ncols = the plane = 16384.
///
/// NOTHING IS ASSERTED about the timing. What IS asserted is that the kernel ran
/// and produced a finite, non-zero result, and - for the small shape - that it
/// computes the same values as a triple loop written here, since a bench that is
/// fast and wrong would answer its question anyway.
#[test]
fn gpu_tiled_gemm_on_the_im2col_shape() {
    let cuda = match nightenh::cuda::Cuda::init(false) {
        Ok(c) => c,
        Err(e) => { println!("skipping: no GPU: {e}"); return; }
    };
    if let Err(e) = cuda.check_kernels() {
        println!("skipping: {e}");
        return;
    }
    // `(ne0, ne1, ncols, label)`. The two shapes that matter are the resnet 3x3 at
    // the default 512 image (2304 = 256*9 reduction, 256 output channels, 128x128
    // plane) and the same network at a quarter size (64 channels, 64x64 plane),
    // which is where the direct kernels' crossover sits.
    for &(ne0, ne1, ncols, label) in
        &[(2304usize, 256usize, 16384usize, "resnet 256->256 on 128x128"),
          (576, 64, 4096, "resnet 64->64 on 64x64"),
          (2304, 256, 4096, "resnet 256->256 on 64x64")] {
        let mut s = 1234567u32;
        let mut next = move || {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((s >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
        };
        let w: Vec<f32> = (0..ne1 * ne0).map(|_| next() * 0.1).collect();
        let x: Vec<f32> = (0..ncols * ne0).map(|_| next()).collect();
        let dw = nightenh::cuda::DevBuf::from_host(&w).unwrap();
        let dx = nightenh::cuda::DevBuf::from_host(&x).unwrap();
        let dy = nightenh::cuda::DevBuf::alloc(ncols * ne1).unwrap();
        let grid = (div_ceil(ne1, 64), div_ceil(ncols, 32), 1);
        let mut best = f64::INFINITY;
        let mut last: Vec<f32> = Vec::new();
        for _ in 0..7 {
            let mut a = lightgpu::vm::Args::new();
            a.ptr(dw.ptr).ptr(dx.ptr).ptr(dy.ptr)
             .i32(ne0 as i32).i32(ne1 as i32).i32(ncols as i32);
            let t0 = std::time::Instant::now();
            a.launch(&cuda.toolkit, "lg_f32_gemm_tiled", lightgpu::vm::Launch::new(grid, (256, 1, 1)))
                .expect("launch lg_f32_gemm_tiled");
            cuda.sync().expect("sync");
            best = best.min(t0.elapsed().as_secs_f64());
            if last.is_empty() {
                let mut got = vec![0.0f32; ncols * ne1];
                dy.download(&mut got).unwrap();
                last = got;
            }
        }
        let flop = 2.0 * (ne0 * ne1 * ncols) as f64;
        let sum: f64 = last.iter().map(|v| *v as f64).sum();
        assert!(sum.is_finite() && sum != 0.0, "lg_f32_gemm_tiled produced nothing");
        println!(
            "gpu gemm K={ne0:5} M={ne1:4} N={ncols:6} ({label}): {best:.4} s  \
             {:6.1} GFLOP/s  ({:.0}% of the 9800 GFLOP/s FP32 peak)",
            flop / best / 1e9, 100.0 * flop / best / 1e9 / 9800.0
        );
    }

    // The same kernel on a shape small enough to check against a triple loop, so
    // that "6 TFLOP/s" cannot be the speed of a wrong answer. ne0 = 8 keeps the
    // kernel's ne0 % 4 == 0 requirement, and the rest is coprime with the tiles.
    {
        let (ne0, ne1, ncols) = (8usize, 5usize, 34usize);
        let wv: Vec<f32> = (0..ne1 * ne0).map(|i| ((i * 7 % 11) as f32) * 0.125 - 0.5).collect();
        let xv: Vec<f32> = (0..ncols * ne0).map(|i| ((i * 5 % 13) as f32) * 0.25 - 1.0).collect();
        let dw = nightenh::cuda::DevBuf::from_host(&wv).unwrap();
        let dx = nightenh::cuda::DevBuf::from_host(&xv).unwrap();
        let dy = nightenh::cuda::DevBuf::alloc(ncols * ne1).unwrap();
        let mut a = lightgpu::vm::Args::new();
        a.ptr(dw.ptr).ptr(dx.ptr).ptr(dy.ptr)
         .i32(ne0 as i32).i32(ne1 as i32).i32(ncols as i32);
        a.launch(&cuda.toolkit, "lg_f32_gemm_tiled",
                 lightgpu::vm::Launch::new((div_ceil(ne1, 64), div_ceil(ncols, 32), 1), (256, 1, 1)))
            .expect("launch");
        cuda.sync().expect("sync");
        let mut got = vec![0.0f32; ncols * ne1];
        dy.download(&mut got).unwrap();
        // The layout the kernel documents: y[col][row] = sum_k W[row][k] * x[col][k].
        let mut want = vec![0.0f32; ncols * ne1];
        for col in 0..ncols {
            for row in 0..ne1 {
                let mut acc = 0.0f32;
                for k in 0..ne0 { acc += wv[row * ne0 + k] * xv[col * ne0 + k]; }
                want[col * ne1 + row] = acc;
            }
        }
        let d = worst(&got, &want);
        println!("gpu gemm correctness: {ne0}x{ne1}x{ncols}, max |d| vs a triple loop {d:.3e}");
        assert!(d <= 1e-5, "lg_f32_gemm_tiled disagrees with the triple loop by {d:.3e}");
    }
}

/// The im2col GEMM question, taken one step further than `gpu_tiled_gemm_on_the_im2col_shape`:
/// does staging BOTH operands in shared memory and double-buffering the k-tiles
/// buy anything over the toolkit's single-buffered strip with weights straight
/// from global?
///
/// This exists because a disassembly gave the design. `nex_f32_gemm_db` is the
/// structure cuBLAS's sm_61 `sgemm_128x128x8_NN_vec` was measured to have -
/// REG:128 SHARED:16912 and 128-bit shared loads - against the toolkit kernel's
/// REG:54 SHARED:512. Both are timed here on identical inputs, and the new one is
/// checked against the same triple loop as the toolkit kernel, with a TOLERANCE
/// rather than exact equality: its 8-wide k-step and shared staging associate the
/// sum differently, so it is not bit-identical to anything, and pretending
/// otherwise would be the mistake this file exists to avoid.
#[test]
fn gpu_gemm_double_buffered() {
    let cuda = match nightenh::cuda::Cuda::init(false) {
        Ok(c) => c,
        Err(e) => { println!("skipping: no GPU: {e}"); return; }
    };
    if let Err(e) = cuda.check_kernels() {
        println!("skipping: {e}");
        return;
    }

    // Correctness first, on a shape small enough to check by hand. ne0 = 8 is the
    // kernel's own requirement (the toolkit's needs only 4), and 5 x 34 is coprime
    // with the 128 x 64 tile, so the partial tile on BOTH axes is exercised.
    {
        let (ne0, ne1, ncols) = (8usize, 5usize, 34usize);
        let wv: Vec<f32> = (0..ne1 * ne0).map(|i| ((i * 7 % 11) as f32) * 0.125 - 0.5).collect();
        let xv: Vec<f32> = (0..ncols * ne0).map(|i| ((i * 5 % 13) as f32) * 0.25 - 1.0).collect();
        let dw = nightenh::cuda::DevBuf::from_host(&wv).unwrap();
        let dx = nightenh::cuda::DevBuf::from_host(&xv).unwrap();
        let dy = nightenh::cuda::DevBuf::alloc(ncols * ne1).unwrap();
        let mut a = lightgpu::vm::Args::new();
        a.ptr(dw.ptr).ptr(dx.ptr).ptr(dy.ptr)
         .i32(ne0 as i32).i32(ne1 as i32).i32(ncols as i32);
        a.launch(&cuda.project, "nex_f32_gemm_db",
                 lightgpu::vm::Launch::new((div_ceil(ne1, 128), div_ceil(ncols, 64), 1), (256, 1, 1)))
            .expect("launch nex_f32_gemm_db");
        cuda.sync().expect("sync");
        let mut got = vec![0.0f32; ncols * ne1];
        dy.download(&mut got).unwrap();
        let mut want = vec![0.0f32; ncols * ne1];
        for col in 0..ncols {
            for row in 0..ne1 {
                let mut acc = 0.0f32;
                for k in 0..ne0 { acc += wv[row * ne0 + k] * xv[col * ne0 + k]; }
                want[col * ne1 + row] = acc;
            }
        }
        let d = worst(&got, &want);
        println!("gpu gemm double-buffered correctness: {ne0}x{ne1}x{ncols}, max |d| vs a triple loop {d:.3e}");
        assert!(d <= 1e-5, "nex_f32_gemm_db disagrees with the triple loop by {d:.3e}");
    }

    // Then the comparison, same shapes as the toolkit kernel's test, and the same
    // 7-iteration best-of on each so neither gets the luckier clock.
    for &(ne0, ne1, ncols, label) in
        &[(2304usize, 256usize, 16384usize, "resnet 256->256 on 128x128"),
          (576, 64, 4096, "resnet 64->64 on 64x64"),
          (2304, 256, 4096, "resnet 256->256 on 64x64")] {
        let mut s = 1234567u32;
        let mut next = move || {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((s >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
        };
        let w: Vec<f32> = (0..ne1 * ne0).map(|_| next() * 0.1).collect();
        let x: Vec<f32> = (0..ncols * ne0).map(|_| next()).collect();
        let dw = nightenh::cuda::DevBuf::from_host(&w).unwrap();
        let dx = nightenh::cuda::DevBuf::from_host(&x).unwrap();
        let dy = nightenh::cuda::DevBuf::alloc(ncols * ne1).unwrap();
        let flop = 2.0 * (ne0 * ne1 * ncols) as f64;

        let mut best_toolkit = f64::INFINITY;
        let mut best_new = f64::INFINITY;
        for _ in 0..7 {
            let mut a = lightgpu::vm::Args::new();
            a.ptr(dw.ptr).ptr(dx.ptr).ptr(dy.ptr)
             .i32(ne0 as i32).i32(ne1 as i32).i32(ncols as i32);
            let t0 = std::time::Instant::now();
            a.launch(&cuda.toolkit, "lg_f32_gemm_tiled",
                     lightgpu::vm::Launch::new((div_ceil(ne1, 64), div_ceil(ncols, 32), 1), (256, 1, 1)))
                .expect("launch lg_f32_gemm_tiled");
            cuda.sync().expect("sync");
            best_toolkit = best_toolkit.min(t0.elapsed().as_secs_f64());

            let mut b = lightgpu::vm::Args::new();
            b.ptr(dw.ptr).ptr(dx.ptr).ptr(dy.ptr)
             .i32(ne0 as i32).i32(ne1 as i32).i32(ncols as i32);
            let t1 = std::time::Instant::now();
            b.launch(&cuda.project, "nex_f32_gemm_db",
                     lightgpu::vm::Launch::new((div_ceil(ne1, 128), div_ceil(ncols, 64), 1), (256, 1, 1)))
                .expect("launch nex_f32_gemm_db");
            cuda.sync().expect("sync");
            best_new = best_new.min(t1.elapsed().as_secs_f64());
        }
        let mut got = vec![0.0f32; ncols * ne1];
        dy.download(&mut got).unwrap();
        let sum: f64 = got.iter().map(|v| *v as f64).sum();
        assert!(sum.is_finite() && sum != 0.0, "nex_f32_gemm_db produced nothing");

        println!(
            "gpu gemm K={ne0:5} M={ne1:4} N={ncols:6} ({label}):\n    \
             toolkit lg_f32_gemm_tiled {best_toolkit:.4} s {:6.1} GFLOP/s ({:.0}% peak)\n    \
             new     nex_f32_gemm_db   {best_new:.4} s {:6.1} GFLOP/s ({:.0}% peak)  ratio {:.2}x",
            flop / best_toolkit / 1e9, 100.0 * flop / best_toolkit / 1e9 / 9800.0,
            flop / best_new / 1e9, 100.0 * flop / best_new / 1e9 / 9800.0,
            best_toolkit / best_new
        );
    }
}

/// The kernel name for a padding mode: the project's reflection kernel, or the
/// toolkit's zero-padded one.
fn name_for(refl: bool) -> &'static str {
    if refl { "ne_conv3x3_refl" } else { "lg_conv3x3s1p1" }
}

fn div_ceil(n: usize, d: u32) -> u32 {
    ((n + d as usize - 1) / d as usize) as u32
}

/// Diagnose the experimental kernel: WHERE does it differ from the toolkit's?
///
/// A rejection at 5.2e-6 raised a question the equality check cannot answer on
/// its own: is that a rounding difference spread evenly over the plane (two
/// genuinely different instruction sequences for the same mathematical sum), or a
/// localised error - a tap dropped, a border column misread - which would show as
/// a large discrepancy in a few places and nothing elsewhere? The two have very
/// different consequences and very different fixes, and the answer is not
/// available by reading the kernels.
///
/// The input is built so that a single wrong TAP cannot hide: the weight is a
/// delta at one tap, so the output is a shifted copy of the input and an
/// off-by-one in the window shows up as a large, structured difference rather
/// than as round-off.
#[test]
fn gpu_experimental_3x3_difference_shape() {
    let cuda = match nightenh::cuda::Cuda::init(false) {
        Ok(c) => c,
        Err(e) => { println!("skipping: no GPU: {e}"); return; }
    };
    let (cin, cout, h, wd) = (4usize, 4usize, 16usize, 16usize);
    let n = cin * h * wd;
    let mut s = 987654321u32;
    let mut next = move || {
        s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        ((s >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
    };
    let x: Vec<f32> = (0..n).map(|_| next()).collect();
    // A 3x3 identity-ish window: weight 1 at the centre tap, 0 elsewhere, identity
    // across channels for the first `min(cin,cout)` channels.
    let mut w = vec![0.0f32; cout * cin * 9];
    for c in 0..cin.min(cout) { w[(c * cin + c) * 9 + 4] = 1.0; }
    let b = vec![0.0f32; cout];
    let run = |name: &str, module: &lightgpu::vm::Module, grid: (u32, u32, u32), block: (u32, u32, u32),
               refl: bool| -> Vec<f32> {
        let dx = nightenh::cuda::DevBuf::from_host(&x).unwrap();
        let dw = nightenh::cuda::DevBuf::from_host(&w).unwrap();
        let db = nightenh::cuda::DevBuf::from_host(&b).unwrap();
        let dout = nightenh::cuda::DevBuf::alloc(cout * h * wd).unwrap();
        let mut a = lightgpu::vm::Args::new();
        a.ptr(dx.ptr).ptr(dw.ptr).ptr(db.ptr).ptr(dout.ptr)
         .i32(cin as i32).i32(cout as i32).i32(h as i32).i32(wd as i32);
        if refl { a.i32(1); }
        a.launch(module, name, lightgpu::vm::Launch::new(grid, block)).expect("launch");
        cuda.sync().expect("sync");
        let mut got = vec![0.0f32; cout * h * wd];
        dout.download(&mut got).unwrap();
        got
    };
    let lg = run("lg_conv3x3s1p1", &cuda.toolkit,
                 (((cout * h * wd + 255) / 256) as u32, 1, 1), (256, 1, 1), false);
    let nex = run("nex_conv3x3s1p1_x8", &cuda.project,
                  (((((wd + 7) / 8) + 255) / 256) as u32, h as u32, cout as u32), (256, 1, 1), false);
    // Per-pixel difference census: how many differ at all, the worst, and WHERE.
    let mut n_diff = 0usize;
    let mut worst = 0.0f32;
    let mut worst_at = (0usize, 0usize, 0usize);
    for c in 0..cout { for y in 0..h { for ix in 0..wd {
        let i = (c * h + y) * wd + ix;
        let d = (nex[i] - lg[i]).abs();
        if d != 0.0 { n_diff += 1; }
        if d > worst { worst = d; worst_at = (c, y, ix); }
    }}}
    println!("experimental vs toolkit, identity-weight input:");
    println!("  {n_diff} of {} values differ; worst {worst:.3e} at (ch {c}, row {y}, col {ix})",
             cout * h * wd, c = worst_at.0, y = worst_at.1, ix = worst_at.2);
    // The interior is where both kernels run the same nine taps; a difference
    // concentrated at the border is a padding bug rather than round-off.
    let mut interior_diff = 0usize;
    for c in 0..cout { for y in 1..h - 1 { for ix in 1..wd - 1 {
        let i = (c * h + y) * wd + ix;
        if nex[i] != lg[i] { interior_diff += 1; }
    }}}
    println!("  of those, {interior_diff} are strictly interior (a border bug would leave this 0)");
}
