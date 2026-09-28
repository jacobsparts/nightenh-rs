//! The CPU backend: the twin of every op, in the same order as the kernels.
//!
//! This is not a test harness. It is what runs when there is no GPU, and the
//! engines in this family hold it to the same standard as the kernels: the op
//! list is walked once, each op's channels go across a rayon pool, and the
//! arithmetic matches the CUDA twin closely enough that a backend difference
//! means a real difference. The reflection index fold and the actual divisor are
//! shared functions here precisely so the two backends cannot disagree about
//! them.
//!
//! Parallelism is over CHANNELS and over spatial blocks, never over the
//! reduction itself: `instance_norm`, `adailn` and `channel_mean` each sum a
//! plane, and the ORDER the terms are added in is part of the contract with the
//! kernels. That order is not the natural one: every device reduction
//! (`ne_instance_norm`, `ne_adailn`, `lg_channel_mean`) splits a plane across
//! `LANES` threads that stride by the lane count, then folds the partials with a
//! halving tree. Summing serially instead is NOT the same arithmetic, and on
//! real activations it is not even close: a serial f32 sum over the 4.19M values
//! of a 256x128x128 plane puts a 5e-3 relative error in the layer variance,
//! where the tree puts in 1e-6 - which shows up as a 0.05 absolute difference in
//! a normalised output and grows with the plane. [`reduce`] reproduces the
//! device's lane order exactly, so a backend difference stays a real difference.

use rayon::prelude::*;

use crate::model::{stride_out, Op, Plan};
use crate::weights::Weights;
use crate::Error;

/// The reflection fold: the coordinate `i` mapped into `[0, n)` by mirroring
/// about the interior, i.e. `-1 -> 1` and `n -> n - 2`.
///
/// The same rule as `np.pad(mode="reflect")` and `F.pad(mode="reflect")`, and the
/// same one the CUDA twin uses. `symmetric` padding (which repeats the edge
/// sample) is a different function that would still look plausible in a picture.
#[inline]
fn refl_fold(i: isize, n: usize) -> usize {
    let n = n as isize;
    let mut i = i;
    while i < 0 || i >= n {
        i = if i < 0 { -i } else { 2 * n - 2 - i };
    }
    i as usize
}

/// Lanes in a device reduction: `blockDim.x` at every reduction launch.
///
/// Not a tuning knob here. It fixes the addition order of [`reduce`], and the
/// order is what the two backends are compared on.
pub const LANES: usize = 256;

/// Sum `x` (or the squared deviations from `mean`) in the DEVICE's order: `LANES`
/// lanes striding by `LANES`, then a halving tree.
///
/// This is the twin of the reduction every device kernel performs, and it is the
/// reason none of the normalisations below just call `iter().sum()`: a serial
/// accumulator is a different (and less accurate) function, whose error grows
/// with the plane, so at 512x512 it swamps the backend comparison it exists to
/// serve. `mean` selects the second pass (`sum (x - mean)^2`); `lanes` is scratch
/// the caller owns so this stays allocation-free in a per-channel loop.
#[inline]
fn reduce(x: &[f32], mean: Option<f32>, lanes: &mut [f32; LANES]) -> f32 {
    // The terms are visited in BLOCKS of `LANES` rather than lane-by-lane. That
    // is the same arithmetic - lane `l` still accumulates `x[l]`, `x[l + LANES]`,
    // ... in that order, which is what the device's `for (i = tid; i < n; i +=
    // blockDim.x)` does - but it walks memory sequentially instead of with a
    // 1 KB stride. The stride cost 3x on a 256x128x128 plane (313 s against
    // 114 s at 1024x1024), and the order is the part that has to be preserved.
    lanes.fill(0.0);
    let mut blocks = x.chunks_exact(LANES);
    match mean {
        None => {
            for blk in blocks.by_ref() {
                for (a, v) in lanes.iter_mut().zip(blk) { *a += *v; }
            }
            // A short final block touches only the low lanes; the tree above
            // folds the untouched ones as zeros, exactly as the device's own
            // tail threads do.
            for (a, v) in lanes.iter_mut().zip(blocks.remainder()) { *a += *v; }
        }
        Some(m) => {
            for blk in blocks.by_ref() {
                for (a, v) in lanes.iter_mut().zip(blk) { let d = *v - m; *a += d * d; }
            }
            for (a, v) in lanes.iter_mut().zip(blocks.remainder()) { let d = *v - m; *a += d * d; }
        }
    }
    let mut w = LANES;
    while w > 1 {
        let h = w / 2;
        for i in 0..h { lanes[i] += lanes[i + h]; }
        w = h;
    }
    lanes[0]
}

/// Everything the executor needs: the plan, the weights in host memory, and the
/// arena.
pub struct Cpu<'a> {
    pub plan: &'a Plan,
    pub w: &'a Weights,
    pub arena: &'a mut [f32],
}

impl<'a> Cpu<'a> {
    /// Run every op in order. `arena` must be `plan.arena_len` long.
    pub fn run(&mut self, input: &[f32]) -> Result<(), Error> {
        let plan = self.plan;
        if self.arena.len() < plan.arena_len {
            return Err(Error(format!(
                "the CPU backend needs the whole arena: {} elements, got {}",
                plan.arena_len, self.arena.len()
            )));
        }
        // The input buffer is filled here rather than uploaded, because on this
        // backend there is no upload: arena and input are the same `Vec`.
        {
            let r = plan.range(self.input_buf());
            self.arena[r].copy_from_slice(input);
        }
        // Per-op wall time, off unless asked for: `NIGHTENH_CPU_TIME=1` names
        // where a CPU pass actually spends it, the same question
        // `NIGHTENH_GPU_TRACE` answers for the device. A release build has no
        // probe at all.
        #[cfg(feature = "dev")]
        let mut times: Vec<(usize, f64)> = Vec::new();
        // `NIGHTENH_VERIFY_CPU=1` - what `--verify-cpu` sets - runs every op a
        // second time from the same inputs and requires the same BYTES in what it
        // writes. That is the check a bisect wants when the answer is "sometimes":
        // an op that is not a function of its inputs, or one that races.
        #[cfg(feature = "dev")]
        let verify = std::env::var("NIGHTENH_VERIFY_CPU").is_ok();
        for (idx, op) in plan.ops.iter().enumerate() {
            #[cfg(feature = "dev")]
            if std::env::var("NIGHTENH_TRACE").is_ok() {
                if let Op::Conv { w, src, k, cin, h, wd, .. } = *op {
                    let name = &self.plan.weights[w].0;
                    let wt = self.w.f32(name)?;
                    let sum: f64 = wt.iter().map(|v| *v as f64).sum();
                    let first: Vec<f32> = wt[..3.min(wt.len())].to_vec();
                    let xs = self.inp(src);
                    let xsum: f64 = xs.iter().map(|v| *v as f64).sum();
                    let bname = &self.plan.bufs[src].name;
                    eprintln!(
                        "op{idx} conv k{k} cin{cin} {h}x{wd}: weight[{}] \"{name}\" n={} sum={sum:.6} first={first:?}; src \"{bname}\" off={} sum={xsum:.6}",
                        w, wt.len(), self.plan.offset_of(src));
                }
            }
            #[cfg(feature = "dev")]
            let t0 = std::time::Instant::now();
            #[cfg(feature = "dev")]
            if verify {
                self.run_verified(idx, op)?;
            } else {
                self.exec(op).map_err(|e| Error(format!("op {idx} ({}): {e}", op.kind())))?;
            }
            #[cfg(not(feature = "dev"))]
            self.exec(op).map_err(|e| Error(format!("op {idx} ({}): {e}", op.kind())))?;
            #[cfg(feature = "dev")]
            times.push((idx, t0.elapsed().as_secs_f64()));
        }
        // `NIGHTENH_CPU_TIME=1` names where a CPU pass spends its time, by op
        // kind, the same question `NIGHTENH_GPU_TRACE` answers for the device.
        // A dev probe like the others: a release build contains none of it.
        #[cfg(feature = "dev")]
        if std::env::var("NIGHTENH_CPU_TIME").is_ok() {
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
            eprintln!("nightenh: cpu op time, {total:.2} s over {} ops", times.len());
            for (k, t, n) in by_kind {
                eprintln!("nightenh:   {k:<16} {t:8.3} s {n:4} ops {:5.1}%", 100.0 * t / total);
            }
        }
        Ok(())
    }

    /// Run `op` twice from the same inputs - `NIGHTENH_VERIFY_CPU`, which
    /// `--verify-cpu` sets - and require the same BYTES in what it writes.
    ///
    /// Dev-only. What it proves, said plainly: it compares an op against ITSELF, so
    /// it catches an op whose output depends on something other than its input - a
    /// race between the pool's tasks, an uninitialised read, a buffer the plan
    /// aliased when it should not have. It does NOT check any op against the
    /// reference or against the kernels; comparing a per-buffer `--dump` with the
    /// device is what does that.
    ///
    /// The inputs are snapshotted BEFORE the first run, and that is the whole
    /// subtlety here: most ops in this plan write their own input in place, so a
    /// snapshot taken afterwards hands the second run its own answer. An earlier
    /// version of this function did exactly that and duly reported `instnorm` as
    /// non-deterministic, at a difference of 2e-5 - a defect in the check, not in
    /// the kernel.
    ///
    /// Timing under this flag is doubled by construction - every op runs twice -
    /// so a pass measured with `--verify-cpu` is not a pass to quote.
    #[cfg(feature = "dev")]
    fn run_verified(&mut self, idx: usize, op: &Op) -> Result<(), Error> {
        let plan = self.plan;
        let (mut reads, mut writes) = (Vec::new(), Vec::new());
        op.operands(&mut reads, &mut writes);
        let before: Vec<(usize, Vec<f32>)> =
            reads.iter().map(|b| (*b, self.arena[plan.range(*b)].to_vec())).collect();
        self.exec(op).map_err(|e| Error(format!("op {idx} ({}): {e}", op.kind())))?;
        let first: Vec<(usize, Vec<f32>)> =
            writes.iter().map(|b| (*b, self.arena[plan.range(*b)].to_vec())).collect();
        for (b, v) in &before {
            let r = plan.range(*b);
            self.arena[r].copy_from_slice(v);
        }
        self.exec(op).map_err(|e| Error(format!("op {idx} ({}): {e}", op.kind())))?;
        for (bi, (b, want)) in first.iter().enumerate() {
            let got = &self.arena[plan.range(*b)];
            for (i, (a, c)) in want.iter().zip(got.iter()).enumerate() {
                if a != c {
                    return Err(Error(format!(
                        "op {idx} ({}) is not a function of its input: buffer \"{}\" element {i} \
                         is {a} after one run and {c} after another, on the same input",
                        op.kind(), plan.bufs[first[bi].0].name
                    )));
                }
            }
        }
        Ok(())
    }

    fn input_buf(&self) -> usize {
        // The first buffer a plan creates is the input; said in one place so the
        // convention is not repeated in every caller.
        0
    }

    /// The arena slice a buffer occupies, as `&mut`.
    fn out(&mut self, b: usize) -> &mut [f32] {
        let r = self.plan.range(b);
        &mut self.arena[r]
    }

    /// The arena slice a buffer occupies, as `&`.
    fn inp(&self, b: usize) -> &[f32] {
        let r = self.plan.range(b);
        &self.arena[r]
    }

    fn weight(&self, id: usize) -> Result<&[f32], Error> {
        let (name, _) = &self.plan.weights[id];
        self.w.f32(name)
    }

    fn exec(&mut self, op: &Op) -> Result<(), Error> {
        match *op {
            Op::Conv { dst, src, w, b, k, stride, cin, cout, h, wd, refl } => {
                let wt = self.weight(w)?.to_vec();
                let bias = match b { Some(b) => Some(self.weight(b)?.to_vec()), None => None };
                let x = self.inp(src).to_vec();
                let y = self.out(dst);
                conv_pad(&x, &wt, bias.as_deref(), y, k, stride, cin, cout, h, wd, refl);
            }
            Op::InstanceNorm { dst, src, gamma, beta, c, hw, unbiased } => {
                let g = match gamma { Some(i) => Some(self.weight(i)?.to_vec()), None => None };
                let be = match beta { Some(i) => Some(self.weight(i)?.to_vec()), None => None };
                // In place is allowed and is what the plan relies on for its
                // arena: every element is read once, after the statistics.
                let x = self.inp(src).to_vec();
                let y = self.out(dst);
                instance_norm(&x, g.as_deref(), be.as_deref(), y, c, hw, unbiased != 0);
            }
            Op::AdaILn { dst, src, rho, gamma, beta, c, hw } => {
                let r = self.weight(rho)?.to_vec();
                let g = match gamma { Some(i) => Some(self.weight(i)?.to_vec()), None => None };
                let be = match beta { Some(i) => Some(self.weight(i)?.to_vec()), None => None };
                let x = self.inp(src).to_vec();
                let y = self.out(dst);
                adailn(&x, &r, g.as_deref(), be.as_deref(), y, c, hw);
            }
            Op::AdaILnBuf { dst, src, rho, gamma, beta, c, hw } => {
                // `rho` is a checkpoint tensor; `gamma`/`beta` are arena buffers
                // the CAM produced. The two are different kinds of thing and the
                // op says which it wants - see `Op::AdaILnBuf`.
                let r = self.weight(rho)?.to_vec();
                let g = self.inp(gamma).to_vec();
                let be = self.inp(beta).to_vec();
                let x = self.inp(src).to_vec();
                let y = self.out(dst);
                adailn(&x, &r, Some(&g), Some(&be), y, c, hw);
            }
            Op::ChannelAffine { dst, src, gamma, beta, c, hw } => {
                let g = match gamma { Some(i) => Some(self.weight(i)?.to_vec()), None => None };
                let be = match beta { Some(i) => Some(self.weight(i)?.to_vec()), None => None };
                let x = self.inp(src).to_vec();
                let y = self.out(dst);
                channel_affine(&x, g.as_deref(), be.as_deref(), y, c, hw);
            }
            Op::ChannelMean { dst, src, c, hw } => {
                let x = self.inp(src).to_vec();
                let y = self.out(dst);
                channel_mean(&x, y, c, hw);
            }
            Op::ChannelMax { dst, src, c, hw } => {
                let x = self.inp(src).to_vec();
                let y = self.out(dst);
                channel_max(&x, y, c, hw);
            }
            Op::ChannelMul { dst, src, s, c, hw } => {
                let sv = self.weight(s)?.to_vec();
                let x = self.inp(src).to_vec();
                let y = self.out(dst);
                channel_mul(&x, &sv, y, c, hw);
            }
            Op::Linear { dst, src, w, b, rows, cin, cout } => {
                let wt = self.weight(w)?.to_vec();
                let bias = match b { Some(b) => Some(self.weight(b)?.to_vec()), None => None };
                let x = self.inp(src).to_vec();
                let y = self.out(dst);
                linear(&x, &wt, bias.as_deref(), y, rows, cin, cout);
            }
            Op::Conv1x1 { dst, src, w, b, cin, cout, hw } => {
                let wt = self.weight(w)?.to_vec();
                let bias = match b { Some(b) => Some(self.weight(b)?.to_vec()), None => None };
                let x = self.inp(src).to_vec();
                let y = self.out(dst);
                conv1x1(&x, &wt, bias.as_deref(), y, cin, cout, hw);
            }
            Op::Relu { dst, src, n } => {
                let x = self.inp(src).to_vec();
                let y = self.out(dst);
                for i in 0..n { y[i] = if x[i] < 0.0 { 0.0 } else { x[i] }; }
            }
            Op::Add { dst, a, b, n } => {
                let (x, z) = (self.inp(a).to_vec(), self.inp(b).to_vec());
                let y = self.out(dst);
                for i in 0..n { y[i] = x[i] + z[i]; }
            }
            Op::Copy { dst, src, n } => {
                let x = self.inp(src).to_vec();
                let y = self.out(dst);
                y[..n].copy_from_slice(&x[..n]);
            }
            Op::Upsample2x { dst, src, c, h, wd } => {
                let x = self.inp(src).to_vec();
                let y = self.out(dst);
                for ch in 0..c {
                    let s = &x[ch * h * wd..(ch + 1) * h * wd];
                    let d = &mut y[ch * 4 * h * wd..(ch + 1) * 4 * h * wd];
                    for oy in 0..2 * h {
                        for ox in 0..2 * wd {
                            d[oy * 2 * wd + ox] = s[(oy / 2) * wd + ox / 2];
                        }
                    }
                }
            }
            Op::TanhAdd { dst, x, skip, n } => {
                let (a, b) = (self.inp(x).to_vec(), self.inp(skip).to_vec());
                let y = self.out(dst);
                for i in 0..n { y[i] = (a[i] + b[i]).tanh(); }
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// The twins
// ---------------------------------------------------------------------------

/// Convolution with `ci, ky, kx` accumulation order - the order the CUDA kernels
/// use and the order the toolkit's convs document.
///
/// The channel is the OUTERMOST loop and the nine taps are the inner ones, which
/// is what the body below and every kernel in `cuda/nightenh.cu` do; since the
/// addition order is the thing the two backends are compared on, it is worth
/// stating exactly.
///
/// `refl` selects the border: reflection (upstream's `ReflectionPad2d`, which
/// EXCLUDES the edge sample) or zero. Both are used by this network, in different
/// blocks - see `Op::Conv` - so the flag is an operand of the op rather than a
/// property of the kernel.
///
/// The two differ only at the border. The accumulation ORDER is the same either
/// way, which is what keeps a backend difference meaningful.
///
/// Three shapes have a vector kernel, and between them they are every convolution
/// in this network except the single 1x1: the 3x3 stride-1 (sixteen of the
/// twenty-three, and 74% of a pass's FLOPs), the 3x3 stride-2 downsample (two
/// convolutions, 4.6% of the FLOPs but a third of a single-threaded pass, because
/// it is the one that runs at full 256-channel width on the LARGE plane), and the
/// 7x7 (two, 4.8% of the FLOPs and 12% of the time). Each is BIT-IDENTICAL to the
/// scalar one below, which the kernels' own comments argue and a test asserts.
fn conv_pad(x: &[f32], w: &[f32], b: Option<&[f32]>, y: &mut [f32], k: u8, stride: u8,
            cin: usize, cout: usize, h: usize, wd: usize, refl: bool) {
    // The `unsafe` below is entered only behind a runtime feature check, and the
    // kernels it calls are compiled with exactly the feature that check names. A
    // machine without AVX2 takes the scalar path: the instruction set is never
    // assumed by the build, so this needs no `-C target-cpu` and no baseline bump.
    #[cfg(target_arch = "x86_64")]
    if x86_avx2() {
        match (k, stride) {
            (3, 1) => { conv3x3_s1_vector(x, w, b, y, cin, cout, h, wd, refl); return; }
            (3, 2) => { conv3x3_s2_vector(x, w, b, y, cin, cout, h, wd, refl); return; }
            (7, 1) => { conv7x7_s1_vector(x, w, b, y, cin, cout, h, wd, refl); return; }
            _ => {}
        }
    }
    conv_pad_scalar(x, w, b, y, k, stride, cin, cout, h, wd, refl);
}

/// The scalar conv: every shape, one output element at a time.
///
/// The reference the three vector kernels are checked against, and what still runs
/// for the 1x1.
fn conv_pad_scalar(x: &[f32], w: &[f32], b: Option<&[f32]>, y: &mut [f32], k: u8, stride: u8,
                   cin: usize, _cout: usize, h: usize, wd: usize, refl: bool) {
    // `k` and `stride` are `u8` in `Op` so the enum stays compact; widened here
    // once rather than at every use.
    let (k, stride) = (k as usize, stride as usize);
    let (oh, ow) = ((h + 2 * (k / 2) - k) / stride + 1, (wd + 2 * (k / 2) - k) / stride + 1);
    let plane = h * wd;
    let oplane = oh * ow;
    let pad = (k / 2) as isize;
    y.par_chunks_mut(oplane).enumerate().for_each(|(co, out)| {
        let bias = b.map(|b| b[co]).unwrap_or(0.0);
        for oy in 0..oh {
            for ox in 0..ow {
                let mut acc = bias;
                for ci in 0..cin {
                    let xp = &x[ci * plane..(ci + 1) * plane];
                    let wp = &w[(co * cin + ci) * k * k..(co * cin + ci + 1) * k * k];
                    for ky in 0..k {
                        let sy = oy as isize * stride as isize - pad + ky as isize;
                        if !refl && (sy < 0 || sy >= h as isize) { continue; }
                        let iy = if refl { refl_fold(sy, h) } else { sy as usize };
                        for kx in 0..k {
                            let sx = ox as isize * stride as isize - pad + kx as isize;
                            if !refl && (sx < 0 || sx >= wd as isize) { continue; }
                            let ix = if refl { refl_fold(sx, wd) } else { sx as usize };
                            acc += wp[ky * k + kx] * xp[iy * wd + ix];
                        }
                    }
                }
                out[oy * ow + ox] = acc;
            }
        }
    });
}

/// `nn.InstanceNorm2d` (biased) or the `torch.var` form (unbiased).
///
/// Two passes, the second over the mean-subtracted values, which is the stable
/// form and the one the CUDA twin uses. `eps` is added to the VARIANCE.
fn instance_norm(x: &[f32], gamma: Option<&[f32]>, beta: Option<&[f32]>, y: &mut [f32],
                 _c: usize, hw: usize, unbiased: bool) {
    y.par_chunks_mut(hw).enumerate().for_each(|(ch, out)| {
        let xp = &x[ch * hw..(ch + 1) * hw];
        let mut lanes = [0.0f32; LANES];
        let sum = reduce(xp, None, &mut lanes);
        let mean = sum / hw as f32;
        let acc = reduce(xp, Some(mean), &mut lanes);
        let div = if unbiased { (hw - 1) as f32 } else { hw as f32 };
        let inv = 1.0 / (acc / div + 1e-5).sqrt();
        let g = gamma.map(|g| g[ch]).unwrap_or(1.0);
        let b = beta.map(|b| b[ch]).unwrap_or(0.0);
        for i in 0..hw { out[i] = (xp[i] - mean) * inv * g + b; }
    });
}

/// The adaptive instance-layer norm. Both halves are UNBIASED, because
/// `AdaILN.forward` calls `torch.var`, and the layer half's statistics are over
/// `c * hw` per sample.
///
/// The per-sample statistics are computed once and reused for every channel,
/// which is what the CUDA twin does too - so this is not an optimisation that
/// changes the arithmetic, only one that avoids recomputing a scalar.
fn adailn(x: &[f32], rho: &[f32], gamma: Option<&[f32]>, beta: Option<&[f32]>, y: &mut [f32],
          c: usize, hw: usize) {
    let total = c * hw;
    let mut lanes = [0.0f32; LANES];
    let sum = reduce(&x[..total], None, &mut lanes);
    let mean_l = sum / total as f32;
    let acc = reduce(&x[..total], Some(mean_l), &mut lanes);
    let inv_l = 1.0 / (acc / (total - 1) as f32 + 1e-5).sqrt();

    y.par_chunks_mut(hw).enumerate().for_each(|(ch, out)| {
        let xp = &x[ch * hw..(ch + 1) * hw];
        let mut lanes = [0.0f32; LANES];
        let s = reduce(xp, None, &mut lanes);
        let mean_i = s / hw as f32;
        let a = reduce(xp, Some(mean_i), &mut lanes);
        let inv_i = 1.0 / (a / (hw - 1) as f32 + 1e-5).sqrt();
        let r = rho[ch];
        let g = gamma.map(|g| g[ch]).unwrap_or(1.0);
        let b = beta.map(|b| b[ch]).unwrap_or(0.0);
        for i in 0..hw {
            let inst = (xp[i] - mean_i) * inv_i;
            let lay = (xp[i] - mean_l) * inv_l;
            // The MIX first, then the affine - `AdaILN.forward` scales
            // `rho * inst + (1 - rho) * lay`, not the layer half on its own.
            out[i] = (r * inst + (1.0 - r) * lay) * g + b;
        }
    });
}

/// `out[c][p] = in[c][p] * g[c] + b[c]`.
fn channel_affine(x: &[f32], gamma: Option<&[f32]>, beta: Option<&[f32]>, y: &mut [f32],
                  _c: usize, hw: usize) {
    y.par_chunks_mut(hw).enumerate().for_each(|(ch, out)| {
        let xp = &x[ch * hw..(ch + 1) * hw];
        let g = gamma.map(|g| g[ch]).unwrap_or(1.0);
        let b = beta.map(|b| b[ch]).unwrap_or(0.0);
        for i in 0..hw { out[i] = xp[i] * g + b; }
    });
}

/// Per-channel spatial mean. Serial within a channel, so the order the terms are
/// added in is the plan's and not the pool's.
fn channel_mean(x: &[f32], y: &mut [f32], c: usize, hw: usize) {
    y[..c].par_iter_mut().enumerate().for_each(|(ch, out)| {
        let xp = &x[ch * hw..(ch + 1) * hw];
        // `lg_channel_mean` reduces this plane with the same 256-lane tree, so
        // the CAM's pooled row is summed the same way here.
        let mut lanes = [0.0f32; LANES];
        *out = reduce(xp, None, &mut lanes) / hw as f32;
    });
}

/// Per-channel spatial maximum.
fn channel_max(x: &[f32], y: &mut [f32], c: usize, hw: usize) {
    y[..c].par_iter_mut().enumerate().for_each(|(ch, out)| {
        let xp = &x[ch * hw..(ch + 1) * hw];
        let mut m = f32::NEG_INFINITY;
        for v in xp { m = m.max(*v); }
        *out = m;
    });
}

/// `out[c][p] = in[c][p] * s[..]`: a per-channel weight scaling a plane, or a
/// `[c][1]` row scaled by a per-channel scalar.
fn channel_mul(x: &[f32], s: &[f32], y: &mut [f32], c: usize, hw: usize) {
    // `.. c * hw`, not the whole slice: the destination may be an alias or a
    // buffer packed next to something else, and the CUDA twin launches exactly `c`
    // blocks. Iterating every chunk of `y` would make the two backends disagree
    // whenever the buffer is longer than `c * hw` - a difference that would look
    // like floating point and would not be.
    y[..c * hw].par_chunks_mut(hw).enumerate().for_each(|(ch, out)| {
        let xp = &x[ch * hw..(ch + 1) * hw];
        let sv = s[ch];
        for i in 0..hw { out[i] = xp[i] * sv; }
    });
}

/// `y = x @ w.T (+ b)`: `w` is `[cout][cin]` and `x` is `[rows][cin]`.
fn linear(x: &[f32], w: &[f32], b: Option<&[f32]>, y: &mut [f32], rows: usize, cin: usize,
          cout: usize) {
    for r in 0..rows {
        let xr = &x[r * cin..(r + 1) * cin];
        let yr = &mut y[r * cout..(r + 1) * cout];
        for o in 0..cout {
            let mut acc = b.map(|b| b[o]).unwrap_or(0.0);
            let wo = &w[o * cin..(o + 1) * cin];
            for ci in 0..cin { acc += wo[ci] * xr[ci]; }
            yr[o] = acc;
        }
    }
}

/// 1x1 convolution: `cin -> cout` over `hw` spatial positions.
///
/// Accumulates over `ci` in ascending order, which is the order the toolkit's
/// `lg_conv1x1` documents; the CUDA twin does the same, so a difference here is a
/// real one.
fn conv1x1(x: &[f32], w: &[f32], b: Option<&[f32]>, y: &mut [f32], cin: usize, _cout: usize,
           hw: usize) {
    y.par_chunks_mut(hw).enumerate().for_each(|(co, out)| {
        let bias = b.map(|b| b[co]).unwrap_or(0.0);
        for i in 0..hw { out[i] = bias; }
        for ci in 0..cin {
            let xp = &x[ci * hw..(ci + 1) * hw];
            let wv = w[co * cin + ci];
            for i in 0..hw { out[i] += wv * xp[i]; }
        }
    });
}

/// Whether this CPU has AVX2.
///
/// The arrangement `ifan-rs` uses in this family, after realesrgan-rs: the vector
/// body exists only under `#[cfg(target_arch = "x86_64")]` at COMPILE time and is
/// reached only through this check at RUN time, so the instruction set is never
/// assumed by the build - no `-C target-cpu`, no baseline bump - and a machine
/// without AVX2 takes the scalar path instead of dying on an illegal instruction.
#[cfg(target_arch = "x86_64")]
#[inline]
fn x86_avx2() -> bool {
    is_x86_feature_detected!("avx2")
}

/// Drive the vectorised 3x3 stride-1 convolution: one rayon task per output
/// channel, each calling [`conv3x3_s1_avx2`] on its own plane.
///
/// The kernel is a separate `#[target_feature]` FUNCTION rather than the body of a
/// closure in here, because a closure does not inherit its enclosing function's
/// target features: written as one, the intrinsics either cannot be inlined or get
/// called out of line from code the compiler is not allowed to give AVX2 to.
#[cfg(target_arch = "x86_64")]
fn conv3x3_s1_vector(x: &[f32], w: &[f32], b: Option<&[f32]>, y: &mut [f32], cin: usize,
                     cout: usize, h: usize, wd: usize, refl: bool) {
    let plane = h * wd;
    debug_assert_eq!(y.len(), cout * plane);
    y.par_chunks_mut(plane).enumerate().for_each(|(co, out)| {
        // SAFETY: the only call path here is `conv_pad`, which came through
        // `x86_avx2()` - `is_x86_feature_detected!("avx2")`, the single feature
        // this kernel is compiled with.
        unsafe { conv3x3_s1_avx2(x, w, b, out, co, cin, h, wd, refl) };
    });
}

/// The 3x3 stride-1 convolution for one output channel, eight output COLUMNS at a
/// time.
///
/// Bit-identical to [`conv_pad_scalar`], by construction rather than by
/// measurement. Two properties do it:
///
///  * the vector is over output columns, so each SIMD lane carries its own
///    accumulator and every output element still accumulates its `cin * 9` terms
///    in the scalar twin's order: `ci` outermost, then `ky`, then `kx`;
///  * the multiply and the add are SEPARATE instructions (`mul` then `add`),
///    never a fused one, because the scalar twin rounds them separately - `acc +=
///    w * x` with fp-contract off, which is this crate's setting. A fused
///    multiply-add would be a real, if small, numeric change, and it would put the
///    CPU path on a different footing from the CUDA path.
///
/// Output rows are blocked (`RB` at a time) and each input row is loaded once and
/// routed to the up-to-three output rows that consume it. That is what takes the
/// kernel off the load ports: without the block, every output row fetches its
/// three input rows again, so a tap costs a load and a multiply-add; with it, one
/// fetch of a row feeds up to three output rows and nine multiply-adds.
///
/// Measured on the model's own resnet convs - sixteen `256 -> 256` 3x3s over
/// 64x64 planes, which are 74% of a pass's FLOPs - this kernel runs them at ~25
/// GFLOP/s single-threaded where the scalar twin below manages 2.5 (numpy's own
/// conv2d on the same shape, through OpenBLAS, is 35). A standalone harness of the
/// same vector body reaches 32 on that shape, so the block and the border gather
/// below are worth counting: going to 64-wide planes is what made the border
/// matter, and while the border was still scalar this op measured 6.2 GFLOP/s.
///
/// The BORDER is vectorised too, and that is not a detail: at 64 wide - the
/// resnet plane at 256x256 - two of the eight column blocks contain an edge, so a
/// scalar border would make a quarter of the busiest convolution in the network
/// scalar. What differs there is only how the three shifted vectors are BUILT: a
/// tap whose column leaves the plane is folded back in (reflection) or dropped
/// (zero padding), so the vectors are gathered a column at a time instead of
/// loaded. The accumulation that follows is the same code, in the same order,
/// which is what keeps the two paths bit-identical.
///
/// # Safety
///
/// The caller must have established `x86_avx2()`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn conv3x3_s1_avx2(x: &[f32], w: &[f32], b: Option<&[f32]>, out: &mut [f32], co: usize,
                          cin: usize, h: usize, wd: usize, refl: bool) {
    use std::arch::x86_64::*;
    /// Output rows per block. Eight keeps the eight accumulator registers live
    /// with room for the three shifted input vectors beside them.
    const RB: usize = 8;
    /// Output columns per vector: one lane per column.
    const LF: usize = 8;
    let plane = h * wd;
    let bias = b.map(|b| b[co]).unwrap_or(0.0);
    let wb = &w[co * cin * 9..(co + 1) * cin * 9];
    let mut oy0 = 0;
    while oy0 < h {
        let nr = RB.min(h - oy0);
        let mut ox0 = 0;
        while ox0 < wd {
            let ncol = LF.min(wd - ox0);
            // A vector block needs its columns AND their one-column halo inside
            // the plane, so that its three tap loads are ordinary loads and no
            // lane needs a folded index.
            if ncol == LF && ox0 >= 1 && ox0 + LF + 1 <= wd {
                let mut acc = [_mm256_set1_ps(bias); RB];
                for ci in 0..cin {
                    let xp = &x[ci * plane..(ci + 1) * plane];
                    let wp = &wb[ci * 9..ci * 9 + 9];
                    // The input rows these outputs read: one above the first
                    // output row of the block through one below the last.
                    let mut iy = oy0 as isize - 1;
                    while iy <= oy0 as isize + nr as isize {
                        if refl || (iy >= 0 && (iy as usize) < h) {
                            let iyr = if refl { refl_fold(iy, h) } else { iy as usize };
                            let row = xp.as_ptr().add(iyr * wd);
                            // SAFETY: `ox0 >= 1` and `ox0 + LF + 1 <= wd`, so these
                            // three loads touch columns `ox0 - 1` through
                            // `ox0 + LF` inclusive, all within this row.
                            let xv = [
                                _mm256_loadu_ps(row.offset(ox0 as isize - 1)),
                                _mm256_loadu_ps(row.add(ox0)),
                                _mm256_loadu_ps(row.add(ox0 + 1)),
                            ];
                            let mut ky = 0isize;
                            while ky < 3 {
                                // The output row this input row feeds at this
                                // `ky` - outside the block for the rows just above
                                // and just below it, which is why this is tested
                                // rather than assumed.
                                let rr = iy - oy0 as isize + 1 - ky;
                                if rr >= 0 && rr < nr as isize {
                                    let a = &mut acc[rr as usize];
                                    for kx in 0..3 {
                                        let wv = _mm256_set1_ps(wp[ky as usize * 3 + kx]);
                                        *a = _mm256_add_ps(_mm256_mul_ps(wv, xv[kx]), *a);
                                    }
                                }
                                ky += 1;
                            }
                        }
                        iy += 1;
                    }
                }
                let op = out.as_mut_ptr();
                for r in 0..nr {
                    // SAFETY: the block writes `LF` columns of row `oy0 + r`,
                    // every one of which the plane has.
                    _mm256_storeu_ps(op.add((oy0 + r) * wd + ox0), acc[r]);
                }
            } else {
                // The left and right edge columns, and the partial block at the
                // end of a row. A tap here does not read a column of the plane -
                // it is folded back in (reflection) or dropped (zero padding) - so
                // the three shifted vectors are gathered one column at a time
                // instead of loaded. That is the ONLY difference from the branch
                // above: the accumulation below is the same `ci`, `ky`, `kx` loop,
                // which is why a border cannot be a different function.
                //
                // This branch is not a rarity. At 64 wide - the resnet body's
                // plane at 256x256, where the sixteen `256 -> 256` 3x3s run - two
                // of the eight column blocks land here, and while they were going
                // through the fully scalar loop they cost far more than they
                // should: the op ran at 6.2 GFLOP/s where the same shape reaches
                // 32 with every block vectorised.
                let mut acc = [_mm256_set1_ps(bias); RB];
                for ci in 0..cin {
                    let xp = &x[ci * plane..(ci + 1) * plane];
                    let wp = &wb[ci * 9..ci * 9 + 9];
                    let mut iy = oy0 as isize - 1;
                    while iy <= oy0 as isize + nr as isize {
                        if refl || (iy >= 0 && (iy as usize) < h) {
                            let iyr = if refl { refl_fold(iy, h) } else { iy as usize };
                            let row = xp.as_ptr().add(iyr * wd);
                            let mut xv = [_mm256_setzero_ps(); 3];
                            for kx in 0..3 {
                                let mut t = [0.0f32; LF];
                                for (j, tv) in t.iter_mut().enumerate() {
                                    // Lanes past the end of a short final block are
                                    // never stored; the index is clamped so the read
                                    // is in bounds and the lane is dead.
                                    let col = if ox0 + j < wd { ox0 + j } else { wd - 1 };
                                    let sx = col as isize + kx as isize - 1;
                                    if !refl && (sx < 0 || sx >= wd as isize) { continue; }
                                    let ix = if refl { refl_fold(sx, wd) } else { sx as usize };
                                    // SAFETY: `ix` is in `0..wd` by the two lines
                                    // above, and `row` addresses `wd` values.
                                    *tv = *row.add(ix);
                                }
                                xv[kx] = _mm256_loadu_ps(t.as_ptr());
                            }
                            let mut ky = 0isize;
                            while ky < 3 {
                                let rr = iy - oy0 as isize + 1 - ky;
                                if rr >= 0 && rr < nr as isize {
                                    let a = &mut acc[rr as usize];
                                    for kx in 0..3 {
                                        let wv = _mm256_set1_ps(wp[ky as usize * 3 + kx]);
                                        *a = _mm256_add_ps(_mm256_mul_ps(wv, xv[kx]), *a);
                                    }
                                }
                                ky += 1;
                            }
                        }
                        iy += 1;
                    }
                }
                for r in 0..nr {
                    let dst = out.as_mut_ptr().add((oy0 + r) * wd + ox0);
                    if ncol == LF {
                        // SAFETY: the block writes `LF` columns of row `oy0 + r`,
                        // every one of which the plane has.
                        _mm256_storeu_ps(dst, acc[r]);
                    } else {
                        let mut t = [0.0f32; LF];
                        _mm256_storeu_ps(t.as_mut_ptr(), acc[r]);
                        out[(oy0 + r) * wd + ox0..(oy0 + r) * wd + ox0 + ncol]
                            .copy_from_slice(&t[..ncol]);
                    }
                }
            }
            ox0 += LF;
        }
        oy0 += RB;
    }
}

/// The two column-strided views a stride-2 3x3 reads, built once per op.
///
/// For output column `ox` the three taps read input columns `2*ox - 1`, `2*ox` and
/// `2*ox + 1`. Those are two arithmetic progressions - the even columns and the odd
/// ones - so splitting them out turns all three taps into ordinary CONTIGUOUS
/// loads, and the border can be folded in while the views are built instead of
/// during the accumulation.
///
/// That is the same trade the stride-1 kernel makes at its border, for the same
/// reason: a fold or a bounds test in the inner loop is what made the first
/// version of that kernel slow, and here it would cost more still, because a
/// stride-2 tap can never be a single shifted load. Layout is `[ci][iy][j]`, with
/// `n` columns per row and eight columns of slack so the last block's vector loads
/// stay inside the allocation (its lanes past `ow` are loaded and then never
/// stored).
///
/// `EV[j]` is what the `kx=1` tap reads for output column `j` (input column `2j`);
/// `OD[j]` is the `kx=0` tap (input column `2j - 1`), and `OD[j + 1]` the `kx=2`
/// tap (input column `2j + 1`). Reading the three taps from two arrays does not
/// reorder anything: the accumulation still runs `ci`, then `ky`, then `kx`.
fn strided_views(x: &[f32], cin: usize, h: usize, wd: usize, ow: usize, refl: bool)
    -> (Vec<f32>, Vec<f32>) {
    const SLACK: usize = 8;
    let n = ow + SLACK;
    let stride = h * n;
    // Zero-filled, which is exactly the value a zero-padded conv wants for a tap
    // outside the plane.
    let mut ev = vec![0.0f32; cin * stride];
    let mut od = vec![0.0f32; cin * stride];
    for ci in 0..cin {
        let xp = &x[ci * h * wd..(ci + 1) * h * wd];
        let (evc, odc) = (&mut ev[ci * stride..(ci + 1) * stride],
                          &mut od[ci * stride..(ci + 1) * stride]);
        for iy in 0..h {
            let row = &xp[iy * wd..(iy + 1) * wd];
            let (er, or) = (&mut evc[iy * n..iy * n + n], &mut odc[iy * n..iy * n + n]);
            // `..= ow` rather than `..ow`: the last output column's `kx=2` tap is
            // `OD[ow]`.
            for j in 0..=ow {
                let e = 2 * j;
                if e < wd { er[j] = row[e]; }
                let o = 2 * j as isize - 1;
                if refl {
                    or[j] = row[refl_fold(o, wd)];
                } else if o >= 0 && (o as usize) < wd {
                    or[j] = row[o as usize];
                }
            }
        }
    }
    (ev, od)
}

/// Drive the vectorised stride-2 3x3: the views are built once and then read by
/// every output channel's task, which is why this is not per-channel.
#[cfg(target_arch = "x86_64")]
fn conv3x3_s2_vector(x: &[f32], w: &[f32], b: Option<&[f32]>, y: &mut [f32], cin: usize,
                     cout: usize, h: usize, wd: usize, refl: bool) {
    // `stride_out` is `(n, k, s)` in that order - calling it `(h, 2, 3)` sizes the
    // destination too small and then hands `par_chunks_mut` more chunks than there
    // are output channels, which indexes the weight buffer off its end.
    let (oh, ow) = (stride_out(h, 3, 2), stride_out(wd, 3, 2));
    let oplane = oh * ow;
    debug_assert_eq!(y.len(), cout * oplane);
    let (ev, od) = strided_views(x, cin, h, wd, ow, refl);
    let n = ow + 8;
    y.par_chunks_mut(oplane).enumerate().for_each(|(co, out)| {
        // SAFETY: the only call path here is `conv_pad`, which came through
        // `x86_avx2()`.
        unsafe { conv3x3_s2_avx2(&ev, &od, w, b, out, co, cin, h, n, oh, ow, refl) };
    });
}

/// The stride-2 3x3 for one output channel, eight output COLUMNS at a time, over
/// the two pre-split column views.
///
/// Bit-identical to [`conv_pad_scalar`] for the same two reasons the stride-1
/// kernel is: the vector is over output columns, so each element keeps its own
/// accumulator and the `ci, ky, kx` order, and the multiply and add stay separate
/// instructions.
///
/// Rows are blocked, and here the routing is what pays: an output row reads input
/// rows `2*oy - 1 .. 2*oy + 1`, so one input row feeds at most TWO output rows
/// (its `ky = 0` and `ky = 2` taps) and the row is fetched once for both.
///
/// # Safety
///
/// The caller must have established `x86_avx2()`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn conv3x3_s2_avx2(ev: &[f32], od: &[f32], w: &[f32], b: Option<&[f32]>, out: &mut [f32],
                          co: usize, cin: usize, h: usize, n: usize, oh: usize, ow: usize,
                          refl: bool) {
    use std::arch::x86_64::*;
    /// Output rows per block.
    const RB: usize = 8;
    /// Output columns per vector.
    const LF: usize = 8;
    let stride = h * n;
    let bias = b.map(|b| b[co]).unwrap_or(0.0);
    let wb = &w[co * cin * 9..(co + 1) * cin * 9];
    let mut oy0 = 0;
    while oy0 < oh {
        let nr = RB.min(oh - oy0);
        let mut ox0 = 0;
        while ox0 < ow {
            let ncol = LF.min(ow - ox0);
            let mut acc = [_mm256_set1_ps(bias); RB];
            for ci in 0..cin {
                let evp = ev.as_ptr().add(ci * stride);
                let odp = od.as_ptr().add(ci * stride);
                let wp = &wb[ci * 9..ci * 9 + 9];
                // An output row `oy` reads input rows `2*oy - 1` through
                // `2*oy + 1`, so the block spans these:
                let mut iy = 2 * oy0 as isize - 1;
                while iy <= (2 * (oy0 + nr - 1) + 1) as isize {
                    if refl || (iy >= 0 && (iy as usize) < h) {
                        let iyr = if refl { refl_fold(iy, h) } else { iy as usize };
                        let base = iyr * n;
                        // SAFETY: `ox0 + 1 + LF <= ow + 8 == n` for every block
                        // this loop starts, so all three loads stay in the row.
                        let xv = [
                            _mm256_loadu_ps(odp.add(base + ox0)),
                            _mm256_loadu_ps(evp.add(base + ox0)),
                            _mm256_loadu_ps(odp.add(base + ox0 + 1)),
                        ];
                        let mut ky = 0isize;
                        while ky < 3 {
                            // An output row is fed by this input row only when
                            // `iy + 1 - ky` is even, and only for the one output
                            // row that quotient names.
                            let t = iy + 1 - ky;
                            if t & 1 == 0 {
                                let rr = t / 2 - oy0 as isize;
                                if rr >= 0 && rr < nr as isize {
                                    let a = &mut acc[rr as usize];
                                    for kx in 0..3 {
                                        let wv = _mm256_set1_ps(wp[ky as usize * 3 + kx]);
                                        *a = _mm256_add_ps(_mm256_mul_ps(wv, xv[kx]), *a);
                                    }
                                }
                            }
                            ky += 1;
                        }
                    }
                    iy += 1;
                }
            }
            for r in 0..nr {
                let dst = out.as_mut_ptr().add((oy0 + r) * ow + ox0);
                if ncol == LF {
                    // SAFETY: the block writes `LF` columns of row `oy0 + r`,
                    // every one of which the plane has.
                    _mm256_storeu_ps(dst, acc[r]);
                } else {
                    let mut t = [0.0f32; LF];
                    _mm256_storeu_ps(t.as_mut_ptr(), acc[r]);
                    out[(oy0 + r) * ow + ox0..(oy0 + r) * ow + ox0 + ncol]
                        .copy_from_slice(&t[..ncol]);
                }
            }
            ox0 += LF;
        }
        oy0 += RB;
    }
}

/// Drive the vectorised 7x7 stride-1 convolution, one rayon task per output
/// channel like the 3x3s.
#[cfg(target_arch = "x86_64")]
fn conv7x7_s1_vector(x: &[f32], w: &[f32], b: Option<&[f32]>, y: &mut [f32], cin: usize,
                     cout: usize, h: usize, wd: usize, refl: bool) {
    let plane = h * wd;
    debug_assert_eq!(y.len(), cout * plane);
    y.par_chunks_mut(plane).enumerate().for_each(|(co, out)| {
        // SAFETY: the only call path here is `conv_pad`, through `x86_avx2()`.
        unsafe { conv7x7_s1_avx2(x, w, b, out, co, cin, h, wd, refl) };
    });
}

/// The 7x7 stride-1 convolution for one output channel, eight output COLUMNS at a
/// time.
///
/// The same two properties as the 3x3 kernel make this bit-identical to
/// [`conv_pad_scalar`]: the vector is over output columns, and the multiply and the
/// add stay separate instructions.
///
/// The difference from the 3x3 is the halo. A 7x7 output block reads three columns
/// on each side of it, so an interior block needs `ox0 >= 3` and
/// `ox0 + 8 + 3 <= wd` for its seven shifted loads to be ordinary loads; blocks at
/// either edge, and the partial block at the end of a row, gather their seven
/// vectors a column at a time with the fold applied during the gather, as the
/// stride-1 kernel's border path does.
///
/// Rows are blocked four at a time rather than eight, because the seven shifted
/// input vectors have to stay live alongside the accumulators and eight of each
/// does not fit the register file - the spills cost more than the wider block
/// saves.
///
/// # Safety
///
/// The caller must have established `x86_avx2()`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn conv7x7_s1_avx2(x: &[f32], w: &[f32], b: Option<&[f32]>, out: &mut [f32], co: usize,
                          cin: usize, h: usize, wd: usize, refl: bool) {
    use std::arch::x86_64::*;
    /// Output rows per block: four accumulators beside seven input vectors.
    const RB: usize = 4;
    /// Output columns per vector.
    const LF: usize = 8;
    /// Taps per side.
    const HALO: usize = 3;
    const K: usize = 7;
    /// Taps per kernel element, `K * K`.
    const KT: usize = K * K;
    let plane = h * wd;
    let bias = b.map(|b| b[co]).unwrap_or(0.0);
    let wb = &w[co * cin * KT..(co + 1) * cin * KT];
    let mut oy0 = 0;
    while oy0 < h {
        let nr = RB.min(h - oy0);
        let mut ox0 = 0;
        while ox0 < wd {
            let ncol = LF.min(wd - ox0);
            let interior = ncol == LF && ox0 >= HALO && ox0 + LF + HALO <= wd;
            let mut acc = [_mm256_set1_ps(bias); RB];
            for ci in 0..cin {
                let xp = &x[ci * plane..(ci + 1) * plane];
                let wp = &wb[ci * KT..ci * KT + KT];
                let mut iy = oy0 as isize - HALO as isize;
                while iy <= (oy0 + nr - 1 + HALO) as isize {
                    if refl || (iy >= 0 && (iy as usize) < h) {
                        let iyr = if refl { refl_fold(iy, h) } else { iy as usize };
                        let row = xp.as_ptr().add(iyr * wd);
                        let mut xv = [_mm256_setzero_ps(); K];
                        if interior {
                            // SAFETY: `ox0 >= 3` and `ox0 + 8 + 3 <= wd`, so the
                            // seven loads touch columns `ox0 - 3` through
                            // `ox0 + 10` inclusive, all inside this row.
                            for kx in 0..K {
                                xv[kx] = _mm256_loadu_ps(
                                    row.offset(ox0 as isize - HALO as isize + kx as isize));
                            }
                        } else {
                            for kx in 0..K {
                                let mut t = [0.0f32; LF];
                                for (j, tv) in t.iter_mut().enumerate() {
                                    // Lanes past the end of a short final block are
                                    // never stored; clamping keeps the read in
                                    // bounds and the lane is dead.
                                    let col = if ox0 + j < wd { ox0 + j } else { wd - 1 };
                                    let sx = col as isize + kx as isize - HALO as isize;
                                    if !refl && (sx < 0 || sx >= wd as isize) { continue; }
                                    let ix = if refl { refl_fold(sx, wd) } else { sx as usize };
                                    // SAFETY: `ix` is in `0..wd` by the two lines
                                    // above, and `row` addresses `wd` values.
                                    *tv = *row.add(ix);
                                }
                                xv[kx] = _mm256_loadu_ps(t.as_ptr());
                            }
                        }
                        let mut ky = 0isize;
                        while ky < K as isize {
                            let rr = iy - oy0 as isize + HALO as isize - ky;
                            if rr >= 0 && rr < nr as isize {
                                let a = &mut acc[rr as usize];
                                for kx in 0..K {
                                    let wv = _mm256_set1_ps(wp[ky as usize * K + kx]);
                                    *a = _mm256_add_ps(_mm256_mul_ps(wv, xv[kx]), *a);
                                }
                            }
                            ky += 1;
                        }
                    }
                    iy += 1;
                }
            }
            for r in 0..nr {
                let dst = out.as_mut_ptr().add((oy0 + r) * wd + ox0);
                if ncol == LF {
                    // SAFETY: the block writes `LF` columns of row `oy0 + r`,
                    // every one of which the plane has.
                    _mm256_storeu_ps(dst, acc[r]);
                } else {
                    let mut t = [0.0f32; LF];
                    _mm256_storeu_ps(t.as_mut_ptr(), acc[r]);
                    out[(oy0 + r) * wd + ox0..(oy0 + r) * wd + ox0 + ncol]
                        .copy_from_slice(&t[..ncol]);
                }
            }
            ox0 += LF;
        }
        oy0 += RB;
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    /// A zero-padded 3x3 conv against values a KNOWN-GOOD implementation produced.
    ///
    /// The expectations are `tools/reference.py`'s output for this exact input
    /// and weight, not a hand-derivation: the first version of this test asserted
    /// a hand-derived value that left out four of the nine taps, failed, and
    /// looked like a kernel bug for a while. A test derived from a second
    /// implementation cannot make that mistake.
    #[test]
    fn conv_pad_matches_the_reference() {
        let x: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let w: Vec<f32> = (1..10).map(|i| i as f32).collect();
        // reference: conv2d(4x4, w, pad=1)      -> first row [83, 139, 178, 121]
        let mut y = vec![0.0f32; 16];
        conv_pad(&x, &w, None, &mut y, 3, 1, 1, 1, 4, 4, false);
        assert_eq!(&y[..4], &[83.0, 139.0, 178.0, 121.0]);
        // reference: conv2d(4x4, w, pad=1, reflect=True) -> first row
        // [150, 171, 216, 225]
        let mut yr = vec![0.0f32; 16];
        conv_pad(&x, &w, None, &mut yr, 3, 1, 1, 1, 4, 4, true);
        assert_eq!(&yr[..4], &[150.0, 171.0, 216.0, 225.0]);
        // A bias is added once, not per tap.
        let mut yb = vec![0.0f32; 16];
        let b = vec![7.0f32];
        conv_pad(&x, &w, Some(&b), &mut yb, 3, 1, 1, 1, 4, 4, false);
        assert_eq!(yb[0], 83.0 + 7.0);
    }

    /// The reflection fold, checked directly: the two rules that differ.
    #[test]
    fn reflection_fold_excludes_the_edge() {
        assert_eq!(refl_fold(-1, 5), 1);
        assert_eq!(refl_fold(-2, 5), 2);
        assert_eq!(refl_fold(5, 5), 3);
        assert_eq!(refl_fold(6, 5), 2);
        assert_eq!(refl_fold(0, 5), 0);
        assert_eq!(refl_fold(4, 5), 4);
    }

    /// A multi-channel, strided conv where the channel ORDER matters.
    ///
    /// Two input channels with disjoint supports: if the twin mixed up `ci` with
    /// `co`, or indexed the weight as `[ci][co]`, this fails where the
    /// single-channel case cannot.
    #[test]
    fn conv_pad_uses_co_then_ci() {
        // 2 channels, 2x2, channel 0 = 1, channel 1 = 10.
        let x = vec![1.0, 1.0, 1.0, 1.0, 10.0, 10.0, 10.0, 10.0];
        // 2 out, 2 in, 1x1 weights: out0 = in0, out1 = in1.
        let w = vec![1.0, 0.0, 0.0, 1.0];
        let mut y = vec![0.0f32; 8];
        conv_pad(&x, &w, None, &mut y, 1, 1, 2, 2, 2, 2, false);
        assert_eq!(&y[0..4], &[1.0, 1.0, 1.0, 1.0]);
        assert_eq!(&y[4..8], &[10.0, 10.0, 10.0, 10.0]);
    }

    /// The two variance conventions, on a set whose answers differ by a known
    /// factor. Getting this backwards is a 0.5% error on a 4-element plane and a
    /// real one at 512x512, so it is asserted rather than assumed.
    #[test]
    fn instance_norm_divisor() {
        let x = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut biased = vec![0.0f32; 4];
        instance_norm(&x, None, None, &mut biased, 1, 4, false);
        let mut unbiased = vec![0.0f32; 4];
        instance_norm(&x, None, None, &mut unbiased, 1, 4, true);
        // mean 2.5; biased var = 1.25 (divisor 4), unbiased var = 5/3 (divisor 3).
        // The UNBIASED variance is the LARGER, so its normalised values have the
        // larger magnitude - an easy thing to assert backwards.
        let (vb, vu) = (1.25f32 + 1e-5, 5.0f32 / 3.0 + 1e-5);
        assert!((biased[0] - (-1.5 / vb.sqrt())).abs() < 1e-5);
        assert!((unbiased[0] - (-1.5 / vu.sqrt())).abs() < 1e-5);
        // The unbiased variance is the LARGER of the two, so its normalised
        // values have the SMALLER magnitude - which is the direction that is
        // easy to assert backwards, hence the explicit comparison.
        assert!(biased[0].abs() > unbiased[0].abs(), "biased must be the larger magnitude");
    }

    /// The vectorised stride-2 3x3 against the scalar one, element for element.
    ///
    /// The stride-2 kernel is the one with the most room to go wrong quietly: its
    /// three taps do not come from shifted loads but from two pre-split column
    /// views, and the row/`ky` routing is a parity test rather than a shift. Both
    /// are the kind of thing that produces a plausible-looking image with one
    /// column of every second row wrong. Bit-equality against the scalar twin is
    /// the assertion that rules all of that out at once.
    ///
    /// The shapes include a width whose output is not a multiple of the
    /// eight-column block and an input whose size is odd, so the final block's
    /// dead lanes are exercised; both borders are run because the reflection fold
    /// is applied while the views are BUILT and a zero border leaves the entry at
    /// zero, which are different code paths.
    #[test]
    fn conv3x3_stride2_vector_is_bit_identical_to_scalar() {
        #[cfg(target_arch = "x86_64")]
        if !x86_avx2() {
            eprintln!("conv3x3_stride2_vector_is_bit_identical_to_scalar: no AVX2, scalar only");
            return;
        }
        let mut s = 987654321u32;
        let mut next = move || {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((s >> 8) as f32 / (1u32 << 24) as f32) * 4.0 - 2.0
        };
        for &(h, wd, cin, cout) in
            &[(17usize, 23usize, 3usize, 5usize), (32, 32, 4, 3), (26, 41, 2, 2)]
        {
            let x: Vec<f32> = (0..cin * h * wd).map(|_| next()).collect();
            let w: Vec<f32> = (0..cout * cin * 9).map(|_| next() * 0.25).collect();
            let b: Vec<f32> = (0..cout).map(|_| next()).collect();
            for refl in [false, true] {
                for with_bias in [false, true] {
                    let bias = if with_bias { Some(&b[..]) } else { None };
                    let oh = stride_out(h, 3, 2);
                    let ow = stride_out(wd, 3, 2);
                    let mut fast = vec![0.0f32; cout * oh * ow];
                    let mut slow = vec![0.0f32; cout * oh * ow];
                    conv_pad(&x, &w, bias, &mut fast, 3, 2, cin, cout, h, wd, refl);
                    conv_pad_scalar(&x, &w, bias, &mut slow, 3, 2, cin, cout, h, wd, refl);
                    assert_eq!(
                        fast, slow,
                        "{h}x{wd} cin{cin} cout{cout} refl={refl} bias={with_bias}: the                          stride-2 vector kernel must be the SAME function, not merely a close one"
                    );
                    assert!(fast.iter().any(|v| *v != 0.0), "the test data produced an all-zero plane");
                }
            }
        }
    }

    /// The vectorised 7x7 against the scalar one, element for element.
    ///
    /// Its risk is the halo: an interior block loads seven shifted vectors, and a
    /// block at either edge gathers them with the fold applied. The shapes below
    /// include both, deliberately a width that is not a multiple of eight (so the
    /// final block is short AND at an edge) and a width smaller than the kernel, so
    /// that every tap of every output needs folding.
    #[test]
    fn conv7x7_vector_is_bit_identical_to_scalar() {
        #[cfg(target_arch = "x86_64")]
        if !x86_avx2() {
            eprintln!("conv7x7_vector_is_bit_identical_to_scalar: no AVX2, scalar only");
            return;
        }
        let mut s = 2468013579u32;
        let mut next = move || {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((s >> 8) as f32 / (1u32 << 24) as f32) * 4.0 - 2.0
        };
        for &(h, wd, cin, cout) in
            &[(21usize, 35usize, 3usize, 4usize), (12, 12, 2, 3), (6, 5, 2, 2), (16, 24, 1, 1)]
        {
            let x: Vec<f32> = (0..cin * h * wd).map(|_| next()).collect();
            let w: Vec<f32> = (0..cout * cin * 49).map(|_| next() * 0.1).collect();
            let b: Vec<f32> = (0..cout).map(|_| next()).collect();
            for refl in [false, true] {
                for with_bias in [false, true] {
                    let bias = if with_bias { Some(&b[..]) } else { None };
                    let mut fast = vec![0.0f32; cout * h * wd];
                    let mut slow = vec![0.0f32; cout * h * wd];
                    conv_pad(&x, &w, bias, &mut fast, 7, 1, cin, cout, h, wd, refl);
                    conv_pad_scalar(&x, &w, bias, &mut slow, 7, 1, cin, cout, h, wd, refl);
                    assert_eq!(
                        fast, slow,
                        "{h}x{wd} cin{cin} cout{cout} refl={refl} bias={with_bias}: the 7x7                          vector kernel must be the SAME function, not merely a close one"
                    );
                    assert!(fast.iter().any(|v| *v != 0.0), "the test data produced an all-zero plane");
                }
            }
        }
    }

    /// The vectorised 3x3 path against the scalar one, element for element.
    ///
    /// This is the test the whole arrangement turns on. The vector kernel is
    /// allowed to be faster, but it is not allowed to be a different function: the
    /// CPU and CUDA backends are compared on their ADDITION ORDER, so a fast CPU
    /// path that sums in another order would quietly turn every backend comparison
    /// into noise. Two properties are supposed to make it exact - the vector is
    /// over output columns, so each element keeps its own accumulator and the
    /// `ci, ky, kx` order; and the multiply and add stay separate instructions.
    /// Either one slipping shows up here as a non-zero difference.
    ///
    /// The shape is chosen so every path runs: a width that is not a multiple of
    /// the eight-column block (so a partial block lands at the end of a row), a
    /// height that is not a multiple of the eight-row block, more than one channel
    /// in and out, and both borders.
    #[test]
    fn conv3x3_vector_is_bit_identical_to_scalar() {
        #[cfg(target_arch = "x86_64")]
        if !x86_avx2() {
            // Nothing to compare: without the feature both calls run the scalar
            // path. Said out loud rather than passed silently.
            eprintln!("conv3x3_vector_is_bit_identical_to_scalar: no AVX2, scalar only");
            return;
        }
        let (h, wd, cin, cout) = (19usize, 27usize, 3usize, 4usize);
        // A generator, not `rand`: a fixed pseudo-random sequence keeps this
        // reproducible on any machine, and the values need to be varied enough
        // that a wrong accumulation order cannot round to the same answer.
        let mut s = 12345u32;
        let mut next = move || {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((s >> 8) as f32 / (1u32 << 24) as f32) * 4.0 - 2.0
        };
        let x: Vec<f32> = (0..cin * h * wd).map(|_| next()).collect();
        let w: Vec<f32> = (0..cout * cin * 9).map(|_| next() * 0.25).collect();
        let b: Vec<f32> = (0..cout).map(|_| next()).collect();
        for refl in [false, true] {
            for with_bias in [false, true] {
                let bias = if with_bias { Some(&b[..]) } else { None };
                let mut fast = vec![0.0f32; cout * h * wd];
                let mut slow = vec![0.0f32; cout * h * wd];
                conv_pad(&x, &w, bias, &mut fast, 3, 1, cin, cout, h, wd, refl);
                conv_pad_scalar(&x, &w, bias, &mut slow, 3, 1, cin, cout, h, wd, refl);
                assert_eq!(
                    fast, slow,
                    "refl={refl} bias={with_bias}: the vector 3x3 must be the SAME function, \
                     not merely a close one"
                );
                assert!(fast.iter().any(|v| *v != 0.0), "the test data produced an all-zero plane");
            }
        }
    }
}
