//! The command line.
//!
//! One binary, both backends: it runs on the GPU when the CUDA driver can be
//! brought up and on the CPU otherwise, so a machine with no NVIDIA driver at all
//! still gets an enhanced picture. `--device cpu|gpu` overrides that choice, and
//! `--gpu` additionally refuses to fall back.
//!
//! The parts of this file that are contract rather than convenience, all inherited
//! from the rest of the family:
//!
//! * A failed pass is NEVER retried on the CPU. The only thing that sends a run to
//!   the CPU is a driver that cannot be brought up, because a silent switch to a
//!   multi-gigabyte host allocation is worse than an error on a machine short of
//!   RAM - that machine does not fail, it swaps.
//! * The backend is chosen BEFORE the host arena is sized. A GPU run holds only
//!   the input in host memory, so sizing the arena first would allocate gigabytes
//!   the run never touches.
//! * The CPU backend's memory is checked before anything is allocated, and the
//!   refusal quotes the numbers it used.
//! * The development flags exist only in a `--features dev` build and are refused
//!   BY NAME outside it, so a script that asked for a dump cannot carry on as if
//!   it had one.
//! * `-i -` and `-o -` are stdin and stdout, and both are the defaults, so the
//!   tool can sit in a pipe with no arguments but `-m`.

use std::io::Read;

use nightenh::config::Variant;
use nightenh::host::Host;
use nightenh::image::{self, Image};
use nightenh::{memguard, model, weights::Weights, Error};

/// The resolution upstream's pipeline resizes to before the network. Not a
/// requirement of the graph (it is fully convolutional) but the size the released
/// weights were fitted at, and the size at which the reference's own numbers were
/// produced.
const PIPE_SIZE: usize = 512;

/// The alignment a NATIVE-size run pads each axis to. The graph needs a multiple
/// of 4 (two stride-2 stages, two 2x upsamples); 64 is a multiple of 4 and also
/// keeps every block's rows whole.
const PAD_ALIGN: usize = 64;

struct Opts {
    model: String,
    input: String,
    output: String,
    device: Option<String>,
    force_gpu: bool,
    quiet: bool,
    /// `None` means the upstream pipeline: resize to 512, run, resize back.
    native: bool,
    factor: usize,
    #[allow(dead_code)] // dev-only: a release build parses it but cannot set it
    dump: Option<String>,
    profile: bool,
}

fn usage(dev: bool) -> String {
    let mut s = String::from(
"nightenh - night image enhancement (LOL / delight-effects)

USAGE:
    nightenh -m <model.safetensors> [-i <in.png>] [-o <out.png>]

    -m, --model <path>    converted .safetensors checkpoint (see tools/convert.py)
    -i, --input <path>    input PNG, or - for stdin (default: stdin)
    -o, --output <path>   output PNG, or - for stdout (default: stdout)
        --device <dev>    gpu or cpu (default: gpu when the CUDA driver can be
                          brought up, cpu otherwise)
        --cpu             same as --device cpu
        --gpu             same as --device gpu, and refuses to fall back
        --native          run at the input's own size instead of resizing to
                          512x512 and back; each axis is padded up to a multiple
                          of 64 with a reflection and cropped back
    -q, --quiet           no progress output
    -h, --help            this text
    -V, --version         print the version
");
    if dev {
        s.push_str(
"DEVELOPMENT FLAGS (this build has them; a release build refuses them by name):
        --dump <dir>      write every named buffer as an .npy
        --profile         print the per-op census
        --factor <n>      pad native runs to a multiple of n instead of 64
        --verify-cpu      run every op twice and compare (slow, for bisecting)
");
    }
    s
}

fn next(args: &[String], i: &mut usize, what: &str) -> Result<String, Error> {
    *i += 1;
    args.get(*i).cloned().ok_or_else(|| Error(format!("{what} needs a value")))
}

fn parse(args: &[String]) -> Result<Opts, Error> {
    let mut o = Opts {
        model: String::new(), input: "-".into(), output: "-".into(),
        device: None, force_gpu: false, quiet: false, native: false,
        factor: PAD_ALIGN, dump: None, profile: false,
    };
    let mut i = 0;
    while i < args.len() {
        let a = args[i].as_str();
        match a {
            "-m" | "--model" => o.model = next(args, &mut i, a)?,
            "-i" | "--input" => o.input = next(args, &mut i, a)?,
            "-o" | "--output" => o.output = next(args, &mut i, a)?,
            "--device" => o.device = Some(next(args, &mut i, a)?),
            "--cpu" => o.device = Some("cpu".into()),
            "--gpu" => { o.device = Some("gpu".into()); o.force_gpu = true; }
            "--native" => o.native = true,
            "-q" | "--quiet" => o.quiet = true,
            "-h" | "--help" => { print!("{}", usage(cfg!(feature = "dev"))); std::process::exit(0); }
            "-V" | "--version" => {
                println!("nightenh {}", env!("CARGO_PKG_VERSION"));
                std::process::exit(0);
            }
            #[cfg(feature = "dev")]
            "--dump" => o.dump = Some(next(args, &mut i, a)?),
            #[cfg(feature = "dev")]
            "--profile" => o.profile = true,
            #[cfg(feature = "dev")]
            "--factor" => o.factor = next(args, &mut i, a)?.parse().unwrap_or(PAD_ALIGN),
            #[cfg(feature = "dev")]
            "--verify-cpu" => {
                o.profile = true;
                // The check itself lives in the CPU executor, which reads this
                // same variable: `--verify-cpu` is the flag's name, not a second
                // implementation of it.
                std::env::set_var("NIGHTENH_VERIFY_CPU", "1");
            }
            #[cfg(not(feature = "dev"))]
            "--dump" | "--profile" | "--factor" | "--verify-cpu" => {
                eprintln!("nightenh: `{a}` is a development flag and this is a release build");
                eprintln!("nightenh: rebuild with `cargo build --release --features dev` for");
                eprintln!("nightenh: --dump, --profile, --factor and --verify-cpu");
                std::process::exit(2);
            }
            other => {
                eprintln!("nightenh: unrecognised argument `{other}`");
                eprint!("{}", usage(cfg!(feature = "dev")));
                std::process::exit(2);
            }
        }
        i += 1;
    }
    if o.model.is_empty() {
        eprintln!("nightenh: -m <model.safetensors> is required");
        eprint!("{}", usage(cfg!(feature = "dev")));
        std::process::exit(2);
    }
    Ok(o)
}

fn read_input(path: &str) -> Result<Image, Error> {
    if path == "-" {
        let mut buf = Vec::new();
        std::io::stdin().read_to_end(&mut buf).map_err(|e| Error(format!("stdin: {e}")))?;
        image::read_png_stream(std::io::Cursor::new(buf), "stdin")
    } else {
        image::read_png(path)
    }
}

fn write_output(path: &str, img: &Image) -> Result<(), Error> {
    if path == "-" {
        image::write_png_stream(std::io::stdout().lock(), img, "stdout")
    } else {
        image::write_png(path, img)
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(()) => {}
        Err(e) => {
            eprintln!("nightenh: {e}");
            std::process::exit(1);
        }
    }
}

fn run(args: &[String]) -> Result<(), Error> {
    let o = parse(args)?;

    let w = Weights::open(&o.model)?;
    let variant = Variant::from_weights(&w)?;
    let img = read_input(&o.input)?;

    // Upstream's pipeline: resize to 512x512, run, resize the result back. With
    // --native the model runs at the caller's size instead, padded to a multiple
    // of `factor` with a reflection and cropped back.
    let (padded, crop, ran_at) = if o.native {
        let (p, crop) = image::pad_to_multiple(&img, o.factor)?;
        let dims = (p.h, p.w);
        (p, crop, dims)
    } else {
        let r = image::resize(&img, PIPE_SIZE, PIPE_SIZE);
        (r, [0, 0, 0, 0], (PIPE_SIZE, PIPE_SIZE))
    };
    // EVERY progress line goes to STDERR and stdout carries the image and
    // nothing else. That is the family's contract - nafnet and maxim reserve
    // stdout the same way - and it is what `-i -`/`-o -` are for: progress on
    // stdout would prefix the PNG with text and make a pipe unreadable, which
    // is exactly how this was found.
    if !o.quiet {
        eprintln!(
            "nightenh: {} checkpoint, n_res {}, ngf {}",
            variant.variant, variant.n_res, variant.ngf
        );
        eprintln!(
            "nightenh: {}x{} -> {}x{} padded{}, model runs at {}x{}",
            img.w, img.h, padded.w, padded.h,
            if o.native { "" } else { " (resize)" }, ran_at.1, ran_at.0
        );
    }

    // `with_dumps` below is the only mutation, and it exists in a dev build
    // only, so a release build would otherwise warn about the `mut`.
    #[cfg_attr(not(feature = "dev"), allow(unused_mut))]
    let mut plan = model::build(&w, padded.h, padded.w)?;
    // A dump needs every named buffer held to the end of the run, which inflates
    // the arena; a release build never asks for one and never pays for it.
    #[cfg(feature = "dev")]
    if o.dump.is_some() {
        plan.with_dumps();
        if std::env::var("NIGHTENH_PLAN").is_ok() {
            eprintln!("plan arena {} elements, dumps={}", plan.arena_len, plan.dumps);
            for (i, (n, ln)) in plan.weights.iter().enumerate() {
                eprintln!("  w{i:3} {n} ({ln})");
            }
            for op in plan.ops.iter() {
                if let nightenh::model::Op::AdaILn { c, hw, rho, gamma, beta, .. } = op {
                    eprintln!("  adailn c={c} hw={hw} rho={rho} gamma={gamma:?} beta={beta:?} (names {} {} {})", plan.weights[*rho].0, gamma.map(|g| plan.weights[g].0.clone()).unwrap_or_default(), beta.map(|b| plan.weights[b].0.clone()).unwrap_or_default());
                }
            }
            for (i, b) in plan.bufs.iter().enumerate() {
                eprintln!("  buf{i:3} {:>22} len={:6} off={:6} alias={:?}", b.name, b.len, plan.offset_of(i), b.alias_of);
            }
        }
    }
    let weights_bytes: usize = plan.weights.iter().map(|(_, n)| n * 4).sum();
    if !o.quiet {
        let census = plan.census();
        let total: usize = census.iter().map(|(_, n)| n).sum();
        eprintln!(
            "nightenh: plan {} ops, {} buffers, arena {:.1} MiB, weights {:.1} MiB",
            total,
            plan.bufs.len(),
            plan.arena_len as f64 * 4.0 / (1 << 20) as f64,
            weights_bytes as f64 / (1 << 20) as f64,
        );
        if o.profile {
            for (k, n) in &census {
                eprintln!("nightenh:   {k:<16} {n}");
            }
        }
    }

    let input = Host::preprocess(&padded);
    let out = run_backend(&o, &plan, &w, input, weights_bytes, &padded)?;

    let img_out = Host::postprocess(&out);
    let c = 3;
    let mut out_img = Image { c, h: plan.h, w: plan.wd, data: img_out };
    if crop != [0, 0, 0, 0] {
        out_img = image::crop(&out_img, crop);
    }
    if !o.native {
        out_img = image::resize(&out_img, img.h, img.w);
    }
    write_output(&o.output, &out_img)?;
    Ok(())
}

/// Choose a backend and run. See the module note for why the choice happens here
/// and not earlier.
fn run_backend(o: &Opts, plan: &model::Plan, w: &Weights, input: Vec<f32>,
               weights_bytes: usize, padded: &Image) -> Result<Vec<f32>, Error> {
    let want_gpu = match o.device.as_deref() {
        Some("cpu") => false,
        Some("gpu") => true,
        Some(other) => {
            return Err(Error(format!("--device {other}: expected `cpu` or `gpu`")))
        }
        // `--gpu` does not need to be tested here: it sets the device AND sets
        // force_gpu, and force_gpu is what refuses the fallback below. Writing
        // `&& !o.force_gpu` into this expression as well would be dead - it
        // cancels out - and it read as though it did something.
        None => cfg!(feature = "cuda"),
    };

    #[cfg(feature = "cuda")]
    if want_gpu {
        match nightenh::exec_gpu::Gpu::new(plan, w) {
            Ok(mut gpu) => {
                // The rest of the family names the device on a GPU run -
                // `nafnet: device ...` - and a run that silently used some other
                // card is a support question nobody can answer afterwards.
                if !o.quiet {
                    eprintln!("nightenh: device {}", gpu.device_name());
                }
                let host_arena = o.dump.is_some();
                let mut host = Host::new(clone_plan(plan), input, host_arena, [0, 0, 0, 0]);
                match gpu.run(&mut host) {
                    Ok(out) => {
                        #[cfg(feature = "dev")]
                        if let Some(dir) = o.dump.as_deref() {
                            dump_buffers(&host, dir)?;
                        }
                        return Ok(out);
                    }
                    Err(e) => {
                        // A GPU pass that failed AFTER the device came up is not
                        // retried on the CPU: a silent multi-gigabyte host
                        // allocation is worse than the error. There is no arm
                        // for `force_gpu` here because there is nothing to
                        // distinguish - the failure is returned either way.
                        return Err(e);
                    }
                }
            }
            Err(e) => {
                if o.force_gpu || matches!(o.device.as_deref(), Some("gpu")) {
                    return Err(e);
                }
                eprintln!("nightenh: cuda: {e}");
                eprintln!("nightenh: falling back to the CPU backend (--gpu forces the GPU)");
            }
        }
    }
    #[cfg(not(feature = "cuda"))]
    if want_gpu {
        if o.force_gpu || matches!(o.device.as_deref(), Some("gpu")) {
            return Err(Error("this build has no cuda feature; use --device cpu".into()));
        }
    }

    #[cfg(feature = "dev")]
    let dump = o.dump.as_deref();
    #[cfg(not(feature = "dev"))]
    let dump: Option<&str> = None;
    run_cpu_dump(plan, w, input, weights_bytes, padded, o.quiet, dump)
}

/// The CPU run, with an optional per-buffer dump.
///
/// The dump is keyed by the plan's own buffer names, so a comparison against the
/// reference is written as `d1.conv` vs `DownBlock.5` and not as an op index -
/// which is what makes a divergence readable instead of merely located.
#[allow(clippy::too_many_arguments)]
fn run_cpu_dump(plan: &model::Plan, w: &Weights, input: Vec<f32>, weights_bytes: usize,
                padded: &Image, quiet: bool, dump: Option<&str>) -> Result<Vec<f32>, Error> {
    // The guard runs before the allocation, because a machine out of RAM swaps
    // rather than fails: a run that swaps looks like a hang, not an error.
    memguard::check(
        &format!("{}x{}", padded.w, padded.h),
        plan.arena_len,
        weights_bytes,
        memguard::scratch_bytes(3, padded.h, padded.w),
    )?;
    if !quiet {
        let need = memguard::need_bytes(
            plan.arena_len, weights_bytes,
            memguard::scratch_bytes(3, padded.h, padded.w));
        eprintln!(
            "nightenh: cpu plan {:.1} MiB (arena {} + weights {} + scratch {}, plus {}% slack), {:.1} MiB available",
            need as f64 / (1 << 20) as f64,
            plan.arena_len as f64 * 4.0 / (1 << 20) as f64,
            weights_bytes as f64 / (1 << 20) as f64,
            memguard::scratch_bytes(3, padded.h, padded.w) as f64 / (1 << 20) as f64,
            memguard::SLACK_PCT,
            memguard::available_bytes()? as f64 / (1 << 20) as f64,
        );
    }
    let mut host = Host::new(clone_plan(plan), input, true, [0, 0, 0, 0]);
    let r = host.plan.range(host.plan.output);
    let mut cpu = nightenh::exec_cpu::Cpu { plan: &host.plan, w, arena: &mut host.arena };
    cpu.run(&host.input)?;
    if let Some(dir) = dump {
        dump_buffers(&host, dir)?;
    }
    Ok(host.arena[r].to_vec())
}

/// Write every named buffer as a `.npy` of f32 values.
///
/// `Buf.name` is the plan's own name for the buffer, so the files line up with
/// the reference's block names by construction rather than by an index that has
/// to be kept in step by hand.
fn dump_buffers(host: &Host, dir: &str) -> Result<(), Error> {
    let _ = std::fs::create_dir_all(dir);
    for (i, b) in host.plan.bufs.iter().enumerate() {
        if b.len == 0 || b.name.is_empty() { continue; }
        let r = host.plan.range(i);
        let v = &host.arena[r];
        let path = format!("{dir}/{}.npy", b.name);
        let mut hdr = Vec::new();
        hdr.extend_from_slice(b"\x93NUMPY");
        hdr.push(1);
        hdr.push(0);
        let dict = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({},), }}", v.len());
        let mut pad = 64 - ((10 + dict.len() + 1) % 64);
        if pad == 64 { pad = 0; }
        let mut hdr2 = dict.into_bytes();
        hdr2.extend(std::iter::repeat(b' ').take(pad));
        hdr2.push(b'\n');
        let n = hdr2.len() as u16;
        hdr.extend_from_slice(&n.to_le_bytes());
        hdr.extend_from_slice(&hdr2);
        let mut bytes = hdr;
        for x in v { bytes.extend_from_slice(&x.to_le_bytes()); }
        std::fs::write(&path, bytes).map_err(|e| Error(format!("{path}: {e}")))?;
    }
    Ok(())
}

/// The CLI owns the plan and the executors borrow it, so the host gets a copy.
/// It is small next to the arena it indexes (thousands of `Op`s against millions
/// of floats), and copying it keeps the ownership story in the run path instead
/// of threading a lifetime through every function.
fn clone_plan(plan: &model::Plan) -> model::Plan {
    plan.clone()
}
