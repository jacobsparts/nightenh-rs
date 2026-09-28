//! The GPU backend: the same op list, one kernel launch per op.
//!
//! This is the twin of [`crate::exec_cpu`], and it is written to be read against
//! it: every arm here mirrors the arm of the same name there, so a difference
//! between `--device cpu` and `--device gpu` is floating point and never a
//! different graph. The plumbing that makes that possible is small - one device
//! arena and one uploaded copy of each weight - and the launches are all the
//! `Match` below.
//!
//! EVERY convolution in this network is reflection-padded, so every `Op::Conv`
//! here launches one of this project's own `ne_*_refl` kernels, which fold the
//! reflection into the tap read - including the eight 3x3s inside the
//! `ResnetBlock`s and adaILN blocks, which upstream opens with
//! `nn.ReflectionPad2d(1)`. The toolkit's `lg_conv3x3s1p1` - exactly
//! `nn.Conv2d(3, 3, 1, padding=1)` with a zero border - is therefore unreachable
//! from the network. A wrong border survives a visual check, which is why the
//! choice stays an operand of `Op::Conv`, with the arm kept for a model that
//! genuinely wants zero padding, rather than a property of the file.

#![allow(clippy::too_many_arguments)]

use lightgpu::vm;

use crate::cuda::{Cuda, DevBuf};
use crate::host::Host;
use crate::model::{Op, Plan};
use crate::weights::Weights;
use crate::Error;

/// Threads per block for the kernels whose grid is one thread per element. The
/// toolkit's own convs use 256 and `lg_*`'s channel kernels use one block per
/// channel, so 256 is the family's number and the one the kernels' launch
/// bounds are written for.
const THREADS: u32 = 256;

/// The block shape for the one-thread-per-element kernels: `lg_relu`, `lg_add`,
/// `lg_conv1x1`, `lg_conv3x3s1p1`, the `ne_*_3x3_refl` convs, and the project's
/// channel kernels, all of which index `threadIdx.x` flat.
const FLAT: (u32, u32, u32) = (THREADS, 1, 1);

pub struct Gpu {
    cuda: Cuda,
    plan: Plan,
    /// One `DevBuf` per weight, in `Plan::weights` order, uploaded once. Weights
    /// are read-only for the whole run, so they never move and never alias - which
    /// is the same reason the arena packer leaves them out.
    weights: Vec<DevBuf>,
    /// The whole arena in one allocation: a buffer's device address is
    /// `arena.ptr + plan.offset_of(b) * 4`, so the packer's live ranges are what
    /// make this correct and an alias resolves for free.
    arena: DevBuf,
    /// The final output, kept so `run` can download exactly the plane the plan
    /// names rather than the whole arena.
    out_len: usize,
}

impl Gpu {
    pub fn new(plan: &Plan, w: &Weights) -> Result<Gpu, Error> {
        let cuda = Cuda::init(false).map_err(Error)?;
        cuda.check_kernels().map_err(Error)?;
        let mut weights = Vec::with_capacity(plan.weights.len());
        for (name, _) in plan.weights.iter() {
            let v = w.f32(name)?;
            weights.push(DevBuf::from_host(v).map_err(Error)?);
        }
        let arena = DevBuf::alloc(plan.arena_len).map_err(Error)?;
        Ok(Gpu { cuda, plan: plan.clone(), weights, arena, out_len: plan.out_len })
    }

    pub fn device_name(&self) -> String {
        format!("{} cc {}.{}", self.cuda.info.name, self.cuda.info.cc_major, self.cuda.info.cc_minor)
    }

    /// The device address of an arena buffer.
    fn buf(&self, b: usize) -> u64 {
        self.arena.ptr + (self.plan.offset_of(b) as u64) * 4
    }

    fn weight(&self, id: usize) -> u64 {
        self.weights[id].ptr
    }

    /// Upload the input into the plan's input buffer, run every op, download the
    /// output plane.
    pub fn run(&mut self, host: &mut Host) -> Result<Vec<f32>, Error> {
        let plan = self.plan.clone();
        if self.arena.len < plan.arena_len {
            return Err(Error(format!(
                "the GPU arena is {} elements but the plan needs {}",
                self.arena.len, plan.arena_len
            )));
        }
        // The input lands at the input buffer's own offset, which is where the
        // first op reads it - the same convention `exec_cpu::input_buf` states.
        let ib = plan.range(0).start;
        {
            let dst = self.arena.ptr + (ib as u64) * 4;
            let bytes: &[u8] = bytemuck(host.input.as_slice());
            vm::copy_htod(dst, bytes).map_err(Error)?;
        }
        // A per-op synchronise, behind an env var, because an illegal address
        // only surfaces at the NEXT sync: without this the error is reported at
        // the end of the run and names no op, which is a hunt rather than a
        // location. Off by default - a sync per op serialises the whole pass.
        let trace = std::env::var("NIGHTENH_GPU_TRACE").is_ok();
        // `NIGHTENH_GPU_TIME=1` is the device's answer to the question
        // `NIGHTENH_CPU_TIME` answers for the twin, and it exists because that
        // question was the whole reason the CPU path could be fixed: it reports
        // where a pass actually spends its time BY KIND instead of leaving the
        // device to be reasoned about. It costs a synchronise per op, which
        // serialises the pipeline and inflates the wall clock - so the SHARES are
        // the measurement and the absolute numbers are not. Dev-only, like the
        // rest.
        #[cfg(feature = "dev")]
        let time = std::env::var("NIGHTENH_GPU_TIME").is_ok();
        #[cfg(feature = "dev")]
        let mut times: Vec<(usize, f64)> = Vec::new();
        for (idx, op) in plan.ops.iter().enumerate() {
            #[cfg(feature = "dev")]
            let t0 = std::time::Instant::now();
            self.exec(op).map_err(|e| Error(format!("op {idx} ({}): {e}", op.kind())))?;
            #[cfg(feature = "dev")]
            if time {
                // The sync is INSIDE the timed region: a launch returns as soon as
                // it is queued, so timing without it would measure the host and
                // report every kernel as free.
                self.cuda.sync().map_err(Error)?;
                times.push((idx, t0.elapsed().as_secs_f64()));
            }
            if trace {
                eprintln!("gpu op{idx} {}{}", op.kind(), match self.cuda.sync() {
                    Ok(()) => String::new(),
                    Err(e) => format!("  <- FAILED: {e}"),
                });
                if self.cuda.sync().is_err() { break; }
            }
        }
        #[cfg(feature = "dev")]
        if time {
            let total: f64 = times.iter().map(|(_, t)| *t).sum();
            let mut by_kind: Vec<(&'static str, f64, usize)> = Vec::new();
            for (idx, t) in &times {
                let k = plan.ops[*idx].kind();
                match by_kind.iter_mut().find(|(n, _, _)| *n == k) {
                    Some((_, acc, n)) => { *acc += *t; *n += 1; }
                    None => by_kind.push((k, *t, 1)),
                }
            }
            by_kind.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            eprintln!("nightenh: gpu op time (serialised by a sync per op), {total:.3} s over {} ops", times.len());
            for (k, t, n) in by_kind {
                eprintln!("nightenh:   {k:<16} {t:8.4} s {n:4} ops {:5.1}%", 100.0 * t / total);
            }
        }
        if !trace {
            self.cuda.sync().map_err(Error)?;
        }
        // With `--dump` the host was given a FULL arena, and copying the device
        // arena back into it is what lets the CPU and GPU dumps go through the
        // same writer in main.rs - so a divergence is found by comparing two
        // directories, buffer by buffer, rather than by two formats. Off in a
        // normal run, where the host arena is only as long as the plan says.
        if host.arena.len() == plan.arena_len {
            self.arena.download(&mut host.arena).map_err(Error)?;
        }
        let mut out = vec![0.0f32; self.out_len];
        let ob = plan.range(plan.output).start;
        let src = self.arena.ptr + (ob as u64) * 4;
        vm::copy_dtoh(bytemuck_mut(&mut out), src).map_err(Error)?;
        let _ = host;
        Ok(out)
    }

    fn exec(&self, op: &Op) -> Result<(), String> {
        match *op {
            Op::Conv { dst, src, w, b, k, stride, cin, cout, h, wd, refl } => {
                let name = if refl {
                    match (k, stride) {
                        (3, 1) => "ne_conv3x3_refl",
                        (3, 2) => "ne_down3x3_refl",
                        (7, 1) => "ne_conv7x7_refl",
                        (7, 2) => "ne_down7x7_refl",
                        _ => return Err(format!("no reflection kernel for k={k} stride={stride}")),
                    }
                } else {
                    // The zero-padded arm: no op in this network reaches it, but a
                    // graph that wants a plain `nn.Conv2d(padding=1)` should not
                    // have to grow a kernel family to say so.
                    if !(k == 3 && stride == 1) {
                        return Err(format!("no zero-padded kernel for k={k} stride={stride}"));
                    }
                    "lg_conv3x3s1p1"
                };
                let (oh, ow) = out_dims(h, wd, k, stride);
                let module = if refl { Module::Project } else { Module::Toolkit };
                let mut a = vm::Args::new();
                let bias = match b { Some(id) => self.weight(id), None => 0 };
                a.ptr(self.buf(src)).ptr(self.weight(w)).ptr(bias).ptr(self.buf(dst))
                 .i32(cin as i32).i32(cout as i32).i32(h as i32).i32(wd as i32);
                if refl {
                    a.i32(if b.is_some() { 1 } else { 0 });
                }
                // The GRID is part of each kernel's contract, and every one of
                // these indexes `blockIdx.y` as the output ROW and `blockIdx.z`
                // as the batch, then loops over `cout` INSIDE the thread - so a
                // flat one-thread-per-output grid is wrong twice over: only row 0
                // would be written, and the output columns would be counted
                // `cout` times. The 3x3s take a flat blockDim over columns; the
                // 7x7s stage a 16x16 output tile in shared memory.
                let (grid, block): ((u32, u32, u32), (u32, u32, u32)) = if !refl {
                    // The toolkit's zero-padded conv is flat: one thread per
                    // (oc, y, x) of `cout * oh * ow`, exactly like the other
                    // `lg_*` elementwise kernels.
                    ((div_ceil(cout * oh * ow, THREADS as usize), 1, 1), FLAT)
                } else if k == 7 {
                    // The 7x7s stage a 16x16 output tile; `blockIdx.z` is unused
                    // here because the generator runs one sample.
                    ((div_ceil(ow, 16), div_ceil(oh, 16), 1), (16, 16, 1))
                } else {
                    // The 3x3s: one column per thread, one row per blockIdx.y,
                    // and they loop over `cout` inside the thread.
                    ((div_ceil(ow, THREADS as usize), oh as u32, 1), FLAT)
                };
                self.launch(module, name, grid, block, &mut a)?;
                Ok(())
            }
            Op::InstanceNorm { dst, src, gamma, beta, c, hw, unbiased } => {
                let g = match gamma { Some(i) => self.weight(i), None => 0 };
                let be = match beta { Some(i) => self.weight(i), None => 0 };
                let mut a = vm::Args::new();
                a.ptr(self.buf(src)).ptr(g).ptr(be).ptr(self.buf(dst))
                 .i32(c as i32).i32(hw as i32).i32(unbiased as i32);
                // One block per (sample, channel): the kernel reduces a plane
                // with 256 lanes and a halving tree.
                self.launch(Module::Project, "ne_instance_norm",
                            (c as u32, 1, 1), FLAT, &mut a)?;
                Ok(())
            }
            Op::AdaILn { dst, src, rho, gamma, beta, c, hw } => {
                let r = self.weight(rho);
                let g = match gamma { Some(i) => self.weight(i), None => 0 };
                let be = match beta { Some(i) => self.weight(i), None => 0 };
                let mut a = vm::Args::new();
                a.ptr(self.buf(src)).ptr(r).ptr(g).ptr(be).ptr(self.buf(dst))
                 .i32(c as i32).i32(hw as i32);
                // ONE block for the whole sample: the layer half's statistics are
                // per-sample scalars, so `blockIdx.x` is the sample (1 here) and
                // the kernel loops the channels inside. Its scratch is
                // `extern __shared__`: blockDim.x partials plus two scalars.
                self.launch_shared(Module::Project, "ne_adailn", (1, 1, 1), FLAT,
                                   (THREADS + 2) * 4, &mut a)?;
                Ok(())
            }
            Op::AdaILnBuf { dst, src, rho, gamma, beta, c, hw } => {
                // `rho` is a checkpoint tensor and `gamma`/`beta` are arena
                // buffers the CAM produced, which is exactly the distinction
                // `Op::AdaILnBuf` exists to make - the kernel sees three
                // pointers and does not care where they point.
                let r = self.weight(rho);
                let mut a = vm::Args::new();
                a.ptr(self.buf(src)).ptr(r).ptr(self.buf(gamma)).ptr(self.buf(beta))
                 .ptr(self.buf(dst)).i32(c as i32).i32(hw as i32);
                self.launch_shared(Module::Project, "ne_adailn", (1, 1, 1), FLAT,
                                   (THREADS + 2) * 4, &mut a)?;
                Ok(())
            }
            Op::ChannelAffine { dst, src, gamma, beta, c, hw } => {
                let g = match gamma { Some(i) => self.weight(i), None => 0 };
                let be = match beta { Some(i) => self.weight(i), None => 0 };
                let mut a = vm::Args::new();
                a.ptr(self.buf(src)).ptr(g).ptr(be).ptr(self.buf(dst))
                 .i32(c as i32).i32(hw as i32);
                self.launch(Module::Project, "ne_channel_affine", (c as u32, 1, 1), FLAT, &mut a)?;
                Ok(())
            }
            Op::ChannelMean { dst, src, c, hw } => {
                let mut a = vm::Args::new();
                a.ptr(self.buf(src)).ptr(self.buf(dst)).i32(c as i32).i32(hw as i32);
                self.launch(Module::Toolkit, "lg_channel_mean", (c as u32, 1, 1), FLAT, &mut a)?;
                Ok(())
            }
            Op::ChannelMax { dst, src, c, hw } => {
                let mut a = vm::Args::new();
                a.ptr(self.buf(src)).ptr(self.buf(dst)).i32(c as i32).i32(hw as i32);
                // One block per channel; the reduction's scratch is
                // `extern __shared__`, so the launch must reserve blockDim.x floats.
                self.launch_shared(Module::Project, "ne_channel_max", (c as u32, 1, 1), FLAT,
                                   THREADS * 4, &mut a)?;
                Ok(())
            }
            Op::ChannelMul { dst, src, s, c, hw } => {
                let mut a = vm::Args::new();
                a.ptr(self.buf(src)).ptr(self.weight(s)).ptr(self.buf(dst))
                 .i32(c as i32).i32(hw as i32);
                self.launch(Module::Project, "ne_channel_mul", (c as u32, 1, 1), FLAT, &mut a)?;
                Ok(())
            }
            Op::Linear { dst, src, w, b, rows, cin, cout } => {
                let bias = match b { Some(id) => self.weight(id), None => 0 };
                let mut a = vm::Args::new();
                a.ptr(self.buf(src)).ptr(self.weight(w)).ptr(bias).ptr(self.buf(dst))
                 .i32(rows as i32).i32(cin as i32).i32(cout as i32);
                // `lg_linear` tiles `threadIdx` as 16x16 over (row, output), so
                // the block is 16x16 and the grid is (ceil(cout/16),
                // ceil(rows/16)) - not a flat 256. The CAM's three layers are
                // 1 x 256, i.e. one row of 16 blocks.
                let grid = (div_ceil(cout, 16), div_ceil(rows, 16).max(1), 1);
                self.launch(Module::Toolkit, "lg_linear", grid, (16, 16, 1), &mut a)?;
                Ok(())
            }
            Op::Conv1x1 { dst, src, w, b, cin, cout, hw } => {
                let bias = match b { Some(id) => self.weight(id), None => 0 };
                let mut a = vm::Args::new();
                a.ptr(self.buf(src)).ptr(self.weight(w)).ptr(bias).ptr(self.buf(dst))
                 .i32(cin as i32).i32(cout as i32).i32(1).i32(hw as i32);
                let grid = (div_ceil(cout * hw, THREADS as usize), 1, 1);
                self.launch(Module::Toolkit, "lg_conv1x1", grid, FLAT, &mut a)?;
                Ok(())
            }
            Op::Relu { dst, src, n } => {
                let mut a = vm::Args::new();
                a.ptr(self.buf(src)).ptr(self.buf(dst)).i32(n as i32);
                self.launch(Module::Toolkit, "lg_relu", (div_ceil(n, THREADS as usize), 1, 1),
                            FLAT, &mut a)?;
                Ok(())
            }
            Op::Add { dst, a: x, b: z, n } => {
                let mut a = vm::Args::new();
                a.ptr(self.buf(x)).ptr(self.buf(z)).ptr(self.buf(dst)).i32(n as i32);
                self.launch(Module::Toolkit, "lg_add", (div_ceil(n, THREADS as usize), 1, 1),
                            FLAT, &mut a)?;
                Ok(())
            }
            Op::Copy { dst, src, n } => {
                // `lg_add` with a zero operand would need one, so the CAM's two
                // concatenation steps are a device-to-device copy of `n`
                // elements - the one op with no kernel of its own because the
                // driver already has it.
                let (d, s) = (self.buf(dst), self.buf(src));
                vm::copy_d2d(d, s, n * 4)?;
                Ok(())
            }
            Op::Upsample2x { dst, src, c, h, wd } => {
                let mut a = vm::Args::new();
                a.ptr(self.buf(src)).ptr(self.buf(dst)).i32(c as i32).i32(h as i32).i32(wd as i32);
                // One thread per (x, y) of the OUTPUT plane, blockDim 32x8 - the
                // shape the kernel's own comment specifies.
                self.launch(Module::Toolkit, "lg_upsample2x_nearest",
                            (div_ceil(wd * 2, 32), div_ceil(h * 2, 8), 1), (32, 8, 1), &mut a)?;
                Ok(())
            }
            Op::TanhAdd { dst, x, skip, n } => {
                let mut a = vm::Args::new();
                a.ptr(self.buf(x)).ptr(self.buf(skip)).ptr(self.buf(dst)).i64(n as i64);
                self.launch(Module::Project, "ne_tanh_add", (div_ceil(n, THREADS as usize), 1, 1),
                            FLAT, &mut a)?;
                Ok(())
            }
        }
    }

    /// Launch one kernel. The block SHAPE is the kernel's contract, not a
    /// preference: `lg_linear` tiles `threadIdx` as 16x16, `lg_upsample2x_nearest`
    /// as 32x8, and the one-thread-per-element kernels take a flat 256. Passing
    /// the wrong shape is a wrong answer rather than a slow one, so the shape
    /// travels with the launch and is visible at each call site.
    fn launch(&self, module: Module, name: &str, grid: (u32, u32, u32), block: (u32, u32, u32),
              a: &mut vm::Args) -> Result<(), String> {
        let m = match module {
            Module::Toolkit => &self.cuda.toolkit,
            Module::Project => &self.cuda.project,
        };
        a.launch(m, name, vm::Launch::new(grid, block))
    }

    /// Launch with DYNAMIC shared memory.
    ///
    /// `ne_channel_max` and `ne_adailn` declare `extern __shared__`, so their
    /// scratch is not part of the kernel image and a launch that does not reserve
    /// it writes into an unbacked shared window - which is an
    /// `CUDA_ERROR_ILLEGAL_ADDRESS`, not a wrong number, and one that only shows
    /// up at the next synchronise. The byte counts are the kernels' own
    /// (`blockDim.x` floats, plus two scalars for `ne_adailn`'s layer statistics).
    fn launch_shared(&self, module: Module, name: &str, grid: (u32, u32, u32),
                     block: (u32, u32, u32), shared: u32, a: &mut vm::Args)
                     -> Result<(), String> {
        let m = match module {
            Module::Toolkit => &self.cuda.toolkit,
            Module::Project => &self.cuda.project,
        };
        a.launch(m, name, vm::Launch::new(grid, block).shared(shared))
    }
}

#[derive(Clone, Copy)]
enum Module { Toolkit, Project }

/// The output size of a `k` x `k` stride-`s` convolution with `k / 2` of padding:
/// `floor((n + 2*(k/2) - k) / s) + 1`. The same rule as `model::stride_out`,
/// repeated here because the kernels need the OUTPUT size to size their grid and
/// the plan stores the input's.
fn out_dims(h: usize, wd: usize, k: u8, stride: u8) -> (usize, usize) {
    let (k, stride) = (k as usize, stride as usize);
    ((h + 2 * (k / 2) - k) / stride + 1, (wd + 2 * (k / 2) - k) / stride + 1)
}

fn div_ceil(n: usize, d: usize) -> u32 { ((n + d - 1) / d) as u32 }

/// Reinterpret an `&[f32]` as bytes for the driver's upload. `bytemuck` would be
/// a dependency for this one call site; `slice::from_raw_parts` is what it does.
fn bytemuck(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

fn bytemuck_mut(v: &mut [f32]) -> &mut [u8] {
    unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, std::mem::size_of_val(v)) }
}
