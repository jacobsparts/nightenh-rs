//! The graph: one op list per input shape, and the arena it lives in.
//!
//! The plan is the engine's central artifact. Building it resolves every
//! parameter, sizes every buffer and packs them by live range into ONE
//! allocation, and each op's operand lists are what make that packing possible.
//! Both backends then walk the same list, so a CPU/GPU difference is floating
//! point and never a different graph.
//!
//! Buffers are indices into [`Plan::bufs`]. A buffer that is an ALIAS has no
//! bytes of its own: it points into another buffer's storage at an element
//! offset, which is how the CAM's `2C` row is seen as its two `C` halves for
//! free. Weights do not live in the arena at all - they are uploaded once and
//! referenced by index into [`Plan::weights`] - because pinning them into a
//! live-range arena would defeat the packing.
//!
//! This file is written to be read against `networks.py`: `encoder`, `resnets`,
//! `cam`, `adailn_blocks`, `up1`, `up2` are the reference's own block names, and
//! the places where this graph deliberately does NOT mirror the reference are
//! called out in comments where they occur:
//!
//! * `c_logit` and `heatmap` are computed and then not used - upstream's
//!   `predict.py` keeps only the generator's first return value. The two logits
//!   ARE built (two ops, a two-element buffer) so the op list still reads like
//!   `networks.py`; `heatmap`, a channel sum, is not, because it would need an op
//!   of its own for no other reason. Both take the same `gap_fc`/`gmp_fc` weights
//!   the map path below uses, in a different shape.
//! * `nn.InstanceNorm2d` (biased) and `torch.var` (unbiased) both appear, in
//!   different blocks; [`Op::InstanceNorm`] and [`Op::AdaILn`] carry that
//!   distinction explicitly rather than sharing a divisor.
//!
//! Two properties of upstream's graph are worth stating plainly, because both are
//! easy to get wrong and neither shows up as a failure: the CAM catenates two
//! SCALED FULL MAPS and not two pooled vectors, which is what lets the decoder
//! start from the 1x1 conv's output; and every convolution in this network is
//! reflection-padded, including the eight inside the blocks.

use crate::weights::Weights;
use crate::Error;

/// One op. Buffers are indices into `Plan::bufs`; `w`/`b`/`s`/`rho`/`gamma`/`beta`
/// are indices into `Plan::weights` when they name a checkpoint tensor.
#[derive(Debug, Clone)]
pub enum Op {
    /// k x k convolution, stride 1 or 2, with `refl` selecting the border.
    ///
    /// `refl` is true for EVERY convolution in this network, and the flag is
    /// kept because that is a measured fact rather than a rule: upstream's
    /// `ResnetBlock` and `ResnetAdaILNBlock` each open their 3x3 with
    /// `nn.ReflectionPad2d(1)`, and `nn.Conv2d(padding=0)` is used throughout
    /// `networks.py`, so the eight convolutions inside the blocks are reflection-
    /// padded too. A zero border would be a border-only difference that leaves a
    /// picture looking entirely plausible and shows up only as a number in a
    /// comparison against the reference.
    Conv { dst: usize, src: usize, w: usize, b: Option<usize>, k: u8, stride: u8,
           cin: usize, cout: usize, h: usize, wd: usize, refl: bool },
    /// `nn.InstanceNorm2d`-form normalisation of one sample's channels, WITH an
    /// optional per-channel gamma/beta. `unbiased` is 0 for the `H*W` divisor
    /// (InstanceNorm2d, the DownBlock/ResnetBlock form) and 1 for `H*W - 1`
    /// (torch.var, which is what AdaILN and ILN use).
    InstanceNorm { dst: usize, src: usize, gamma: Option<usize>, beta: Option<usize>,
                   c: usize, hw: usize, unbiased: u8 },
    /// The adaptive instance-layer norm: an instance half and a layer half mixed
    /// by a per-channel `rho`, with an optional per-channel gamma/beta applied to
    /// the MIXED result. Both halves use the UNBIASED divisor.
    ///
    /// `out = (rho * inst + (1 - rho) * lay) * gamma + beta`, which is
    /// `AdaILN.forward`'s expression. Applying the affine to the layer half
    /// inside the mix instead - `rho*inst + (1-rho)*lay*gamma + beta` - is a
    /// different function wherever gamma is not 1, and in the CAM-driven blocks
    /// gamma is a network OUTPUT rather than a parameter near 1.
    ///
    /// `gamma`/`beta` are CHECKPOINT TENSORS here - the form the two UpBlock2
    /// stages use, whose `rho`, `gamma` and `beta` are all learned per-channel
    /// parameters. Use [`Op::AdaILnBuf`] for the upstream `ResnetAdaILNBlock`,
    /// where the gamma/beta are the CAM's OUTPUT and are computed at run time.
    AdaILn { dst: usize, src: usize, rho: usize, gamma: Option<usize>, beta: Option<usize>,
             c: usize, hw: usize },
    /// [`Op::AdaILn`] whose gamma/beta are ARENA BUFFERS (the CAM's `cam.gamma`
    /// and `cam.beta` vectors) rather than checkpoint tensors.
    ///
    /// This variant exists because the difference is real: with one op resolving
    /// gamma/beta only through `Plan::weights`, the eight CAM-driven blocks would
    /// pass a weight index that happens to be a valid index - and would read a
    /// 589,824-element conv kernel as a 256-long per-channel vector, silently.
    /// Making the operand's SOURCE part of the op is what stops that from being
    /// expressible at all.
    AdaILnBuf { dst: usize, src: usize, rho: usize, gamma: usize, beta: usize,
                c: usize, hw: usize },
    /// `out[c][p] = in[c][p] * gamma[c] + beta[c]`, whole planes.
    ChannelAffine { dst: usize, src: usize, gamma: Option<usize>, beta: Option<usize>,
                    c: usize, hw: usize },
    /// Per-channel spatial mean: `out[c] = mean_p in[c][p]`.
    ChannelMean { dst: usize, src: usize, c: usize, hw: usize },
    /// Per-channel spatial maximum.
    ChannelMax { dst: usize, src: usize, c: usize, hw: usize },
    /// `out[c][p] = in[c][p] * s[c]`, whole planes, `s` a WEIGHT index - the
    /// CAM's `x * gap_weight.unsqueeze(2).unsqueeze(3)`.
    ///
    /// `gap_fc.weight` is a `[1][C]` `Linear`, so its flat element `c` is exactly
    /// channel `c`'s scale and a weight index is the right form: the op is
    /// `Op::ChannelAffine` with the addend dropped, and the `[1][C]` shape needs no
    /// reshaping because there is only one row.
    ///
    /// The op exists at all - rather than a `ChannelAffine` - because this is a
    /// MULTIPLY, and the sibling `ChannelAffine` op multiplies and adds. It takes
    /// no flag selecting `s[0]`: the CAM's shape is two scaled FULL maps, so no
    /// caller multiplies a pooled `[C][1][1]` row by a scalar, and an option no
    /// launch exercises would be a second code path nothing checks.
    ChannelMul { dst: usize, src: usize, s: usize, c: usize, hw: usize },
    /// `y = x @ w.T (+ b)` on a `[rows][cin]` buffer.
    Linear { dst: usize, src: usize, w: usize, b: Option<usize>, rows: usize,
             cin: usize, cout: usize },
    /// 1x1 convolution over an `[cin][hw]` plane.
    Conv1x1 { dst: usize, src: usize, w: usize, b: Option<usize>, cin: usize, cout: usize,
              hw: usize },
    /// `y = max(x, 0)`.
    Relu { dst: usize, src: usize, n: usize },
    /// `y = a + b`.
    Add { dst: usize, a: usize, b: usize, n: usize },
    /// `y = x`, `n` elements. The CAM's concatenation is two of these.
    Copy { dst: usize, src: usize, n: usize },
    /// Nearest 2x upsample: `out[o] = in[o / 2]`.
    Upsample2x { dst: usize, src: usize, c: usize, h: usize, wd: usize },
    /// `y = tanh(x + skip)`.
    TanhAdd { dst: usize, x: usize, skip: usize, n: usize },
}

impl Op {
    /// The buffers this op reads and writes, in that order of significance, for
    /// the live-range packer.
    ///
    /// A buffer that is both read and written is an in-place op, and the packer
    /// keeps it live from its first read to its last write - which is what makes
    /// an in-place normalisation actually save a plane instead of aliasing its own
    /// input by accident.
    pub fn operands(&self, reads: &mut Vec<usize>, writes: &mut Vec<usize>) {
        reads.clear();
        writes.clear();
        // Weight indices and buffer indices share the slot, so the weights a plan
        // holds are tracked separately (`is_weight`); the packer is only ever
        // handed the buffer indices, because a weight is live for the whole run.
        match *self {
            Op::Conv { dst, src, .. } => { writes.push(dst); reads.push(src); }
            Op::InstanceNorm { dst, src, .. } => { writes.push(dst); reads.push(src); }
            Op::AdaILn { dst, src, .. } => { writes.push(dst); reads.push(src); }
            Op::AdaILnBuf { dst, src, gamma, beta, .. } => {
                writes.push(dst); reads.push(src); reads.push(gamma); reads.push(beta);
            }
            Op::ChannelAffine { dst, src, .. } => { writes.push(dst); reads.push(src); }
            Op::ChannelMean { dst, src, .. } | Op::ChannelMax { dst, src, .. } => {
                writes.push(dst); reads.push(src);
            }
            Op::ChannelMul { dst, src, .. } => { writes.push(dst); reads.push(src); }
            Op::Linear { dst, src, .. } => { writes.push(dst); reads.push(src); }
            Op::Conv1x1 { dst, src, .. } => { writes.push(dst); reads.push(src); }
            Op::Relu { dst, src, .. } => { writes.push(dst); reads.push(src); }
            Op::Add { dst, a, b, .. } => { writes.push(dst); reads.push(a); reads.push(b); }
            Op::Copy { dst, src, .. } => { writes.push(dst); reads.push(src); }
            Op::Upsample2x { dst, src, .. } => { writes.push(dst); reads.push(src); }
            Op::TanhAdd { dst, x, skip, .. } => { writes.push(dst); reads.push(x); reads.push(skip); }
        }
    }

    /// A short name for the op census.
    pub fn kind(&self) -> &'static str {
        match self {
            Op::Conv { k: 7, stride: 2, .. } => "down7x7",
            Op::Conv { k: 3, stride: 2, .. } => "down3x3",
            Op::Conv { k: 7, refl: true, .. } => "conv7x7_refl",
            Op::Conv { k: 7, .. } => "conv7x7",
            Op::Conv { refl: true, .. } => "conv3x3_refl",
            Op::Conv { .. } => "conv3x3",
            Op::InstanceNorm { unbiased: 1, .. } => "instnorm_unbiased",
            Op::InstanceNorm { .. } => "instnorm",
            Op::AdaILn { .. } => "adailn",
            Op::AdaILnBuf { .. } => "adailn_buf",
            Op::ChannelAffine { .. } => "chan_affine",
            Op::ChannelMean { .. } => "chan_mean",
            Op::ChannelMax { .. } => "chan_max",
            Op::ChannelMul { .. } => "chan_mul",
            Op::Linear { .. } => "linear",
            Op::Conv1x1 { .. } => "conv1x1",
            Op::Relu { .. } => "relu",
            Op::Add { .. } => "add",
            Op::Copy { .. } => "copy",
            Op::Upsample2x { .. } => "upsample2x",
            Op::TanhAdd { .. } => "tanh_add",
        }
    }
}

/// A buffer in the arena.
#[derive(Debug, Clone)]
pub struct Buf {
    pub len: usize,
    pub offset: usize,
    pub name: String,
    /// `Some((of, at))`: an alias sharing `of`'s storage from element `at`.
    pub alias_of: Option<(usize, usize)>,
}

impl Buf {
    /// The buffer index whose storage owns this buffer's bytes - itself, unless it
    /// is an alias.
    ///
    /// Aliases are only ever made FROM a real buffer (never from another alias), so
    /// this is one step rather than a walk; `Plan::offset_of` relies on the same
    /// property.
    pub fn owner(&self) -> usize {
        match self.alias_of {
            None => usize::MAX,      // caller supplies its own index
            Some((of, _)) => of,
        }
    }
}

#[derive(Clone)]
pub struct Plan {
    pub bufs: Vec<Buf>,
    pub ops: Vec<Op>,
    /// `(checkpoint name, element count)`, in the order ops reference them.
    pub weights: Vec<(String, usize)>,
    pub input_len: usize,
    pub out_len: usize,
    pub c: usize,
    pub h: usize,
    pub wd: usize,
    pub arena_len: usize,
    /// Hold every NAMED buffer live to the end of the run, so `--dump` can
    /// snapshot it after the last op.
    ///
    /// This is a development setting and it is EXPENSIVE: a named buffer cannot
    /// share its slot with anything, so the arena grows to roughly a named
    /// buffer's own size each rather than their peak live set. A release build
    /// plans with `dumps == false`, and a dump taken with the flag off is NOT a
    /// snapshot of that buffer - it is whatever op last wrote to the slot it
    /// shares, which looks like a plausible tensor. That is the failure this
    /// field exists to prevent, and it is why the flag is a field with a comment
    /// rather than a debugging convention.
    pub dumps: bool,
    pub variant: String,
    pub n_res: usize,
    /// The buffer holding the final output.
    pub output: usize,
}

impl Plan {
    /// Element offset an alias resolves to in the arena.
    pub fn offset_of(&self, b: usize) -> usize {
        match self.bufs[b].alias_of {
            None => self.bufs[b].offset,
            Some((of, at)) => self.bufs[of].offset + at,
        }
    }

    /// The byte range a buffer occupies within the arena.
    pub fn range(&self, b: usize) -> std::ops::Range<usize> {
        let start = self.offset_of(b);
        start..start + self.bufs[b].len
    }

    /// Op census, for `--profile` and the plan line.
    pub fn census(&self) -> Vec<(&'static str, usize)> {
        let mut m: Vec<(&'static str, usize)> = Vec::new();
        for op in &self.ops {
            let k = op.kind();
            match m.iter_mut().find(|(name, _)| *name == k) {
                Some((_, n)) => *n += 1,
                None => m.push((k, 1)),
            }
        }
        m.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));
        m
    }

    /// Hold every named buffer live to the end of the run, for `--dump`. See the
    /// field's note: this inflates the arena and must not be on in a release plan.
    pub fn with_dumps(&mut self) {
        self.dumps = true;
        self.pack();
    }

    /// Pack the buffers into one arena by live range: the largest buffers first,
    /// with first-use as the tie-break, and each buffer placed at the first offset
    /// where its whole extent is free.
    ///
    /// Largest-first matters and was measured on the sibling engine: placing
    /// buffers in first-use order leaves a large buffer to squeeze past everything
    /// already placed, and the gaps it leaves behind are too small for anything
    /// that comes after it. Adding the space BELOW each placed block to the
    /// candidate list was tried there and changed nothing, so it is not here.
    ///
    /// Aliases are not packed at all - they resolve to their owner's offset - so a
    /// split or a view costs zero bytes.
    pub fn pack(&mut self) {
        let n = self.bufs.len();
        let mut first: Vec<usize> = vec![usize::MAX; n];
        let mut last: Vec<usize> = vec![0; n];
        let mut reads = Vec::new();
        let mut writes = Vec::new();
        for (i, op) in self.ops.iter().enumerate() {
            op.operands(&mut reads, &mut writes);
            for &b in reads.iter().chain(writes.iter()) {
                if self.bufs[b].alias_of.is_some() {
                    continue;
                }
                if first[b] == usize::MAX {
                    first[b] = i;
                }
                last[b] = i;
            }
        }
        // With dumps on, a NAMED buffer's last use is the end of the run - set
        // here, before placement, so that the overlap test below sees it. Setting
        // it only while placing the buffer would extend its own extent and not
        // stop a LATER buffer from being given an overlapping slot, which is the
        // half-applied form of this flag.
        if self.dumps {
            let end = self.ops.len();
            for b in 0..n {
                if !self.bufs[b].name.is_empty() && first[b] != usize::MAX {
                    last[b] = end;
                }
            }
        }
        // Weight references have `len == 0` and are not in the arena, so the
        // `len > 0` filter keeps them out of the packing entirely; their offsets
        // are never read. An alias is skipped because it resolves through its
        // owner (`offset_of`).
        let mut order: Vec<usize> = (0..n)
            .filter(|&b| self.bufs[b].alias_of.is_none() && self.bufs[b].len > 0)
            .collect();
        order.sort_by(|&a, &b| {
            self.bufs[b].len.cmp(&self.bufs[a].len).then(first[a].cmp(&first[b]))
        });
        let mut placed: Vec<(usize, usize, usize)> = Vec::new();  // (start, end, buf)
        let mut peak = 0usize;
        for &b in &order {
            // A weight is live for the whole run, and a named buffer (with
            // dumps on) was already extended above.
            let (live_from, live_to) = (first[b], last[b]);
            let len = self.bufs[b].len;
            let mut start = 0usize;
            loop {
                let mut moved = false;
                for &(s, e, other) in &placed {
                    if last[other] < live_from || first[other] > live_to {
                        continue;             // not overlapping in time
                    }
                    if start < e && start + len > s {
                        start = e;
                        moved = true;
                    }
                }
                if !moved {
                    break;
                }
            }
            self.bufs[b].offset = start;
            placed.push((start, start + len, b));
            peak = peak.max(start + len);
        }
        // Every real buffer was placed above, so the only offsets left as MAX are
        // the weight references', which are not arena indices. They are set to 0
        // so that printing a plan cannot show a sentinel as if it were a position.
        for b in 0..n {
            if self.bufs[b].offset == usize::MAX {
                self.bufs[b].offset = 0;
            }
        }
        self.arena_len = peak;
    }
}

// ---------------------------------------------------------------------------
// The graph builder
// ---------------------------------------------------------------------------

/// Builds one plan. Method names follow the reference's block names.
struct Graph {
    bufs: Vec<Buf>,
    ops: Vec<Op>,
    weights: Vec<(String, usize)>,
}

impl Graph {
    fn new() -> Graph {
        Graph { bufs: Vec::new(), ops: Vec::new(), weights: Vec::new() }
    }

    fn buf(&mut self, name: String, len: usize) -> usize {
        self.bufs.push(Buf { len, offset: usize::MAX, name, alias_of: None });
        self.bufs.len() - 1
    }

    fn alias(&mut self, name: String, of: usize, at: usize, len: usize) -> usize {
        self.bufs.push(Buf { len, offset: usize::MAX, name, alias_of: Some((of, at)) });
        self.bufs.len() - 1
    }

    /// Register a checkpoint tensor. Its element count is checked against `len`
    /// when `len != 0`, which is how a wrong width fails here rather than as a
    /// garbage read.
    fn weight(&mut self, w: &Weights, name: &str, expect: usize) -> Result<usize, Error> {
        let n = w.f32(name)?.len();
        if expect != 0 && n != expect {
            return Err(Error(format!(
                "{name}: {n} elements, but the graph needs {expect}"
            )));
        }
        let id = self.weights.len();
        self.weights.push((name.to_string(), n));
        Ok(id)
    }

    fn push(&mut self, op: Op) { self.ops.push(op); }
}

/// The generator's graph for a padded `h` x `wd` input, in the reference's order.
pub fn build(w: &Weights, h: usize, wd: usize) -> Result<Plan, Error> {
    let (ngf, n_res) = (w.ngf, w.n_res);
    let mut g = Graph::new();
    let hw = h * wd;

    // ---- encoder -----------------------------------------------------------
    // `DownBlock` is `ReflectionPad2d(3)`, the 7x7 conv, `nn.InstanceNorm2d(ngf)`
    // and `nn.ReLU(True)`. Both are load-bearing: without them the first three
    // stages of the encoder run unnormalised and every statistic downstream moves.
    // `unbiased: 0` because
    // this one is `nn.InstanceNorm2d`, whose divisor is H*W - not `torch.var`'s
    // H*W - 1, which is what the adaILN blocks use.
    let x_in = g.buf("input".into(), 3 * hw);
    let d1 = g.buf("d1.conv".into(), ngf * hw);
    let w1 = g.weight(w, "DownBlock.1.weight", ngf * 3 * 49)?;
    g.push(Op::Conv { dst: d1, src: x_in, w: w1, b: None, k: 7, stride: 1,
                          cin: 3, cout: ngf, h, wd, refl: true });
    let d1n = g.buf("d1.norm".into(), ngf * hw);
    g.push(Op::InstanceNorm { dst: d1n, src: d1, gamma: None, beta: None,
                              c: ngf, hw, unbiased: 0 });
    g.push(Op::Relu { dst: d1n, src: d1n, n: ngf * hw });

    // DownBlock.5: ReflectionPad2d(1), 3x3 conv stride 2, InstanceNorm2d, ReLU.
    let (h2, w2) = (stride_out(h, 3, 2), stride_out(wd, 3, 2));
    let hw2 = h2 * w2;
    let w5 = g.weight(w, "DownBlock.5.weight", 2 * ngf * ngf * 9)?;
    let d2c = g.buf("d2.conv".into(), 2 * ngf * hw2);
    g.push(Op::Conv { dst: d2c, src: d1n, w: w5, b: None, k: 3, stride: 2,
                          cin: ngf, cout: 2 * ngf, h, wd, refl: true });
    // In place: the norm reads each element once after its statistics are
    // computed, so writing over the input costs nothing and saves a plane.
    g.push(Op::InstanceNorm { dst: d2c, src: d2c, gamma: None, beta: None,
                              c: 2 * ngf, hw: hw2, unbiased: 0 });
    let d2 = g.buf("d2.relu".into(), 2 * ngf * hw2);
    g.push(Op::Relu { dst: d2, src: d2c, n: 2 * ngf * hw2 });

    // DownBlock.9: the second downsample, to 4*ngf.
    let (h4, w4) = (stride_out(h2, 3, 2), stride_out(w2, 3, 2));
    let hw4 = h4 * w4;
    let w9 = g.weight(w, "DownBlock.9.weight", 4 * ngf * 2 * ngf * 9)?;
    let d3c = g.buf("d3.conv".into(), 4 * ngf * hw4);
    g.push(Op::Conv { dst: d3c, src: d2, w: w9, b: None, k: 3, stride: 2,
                          cin: 2 * ngf, cout: 4 * ngf, h: h2, wd: w2, refl: true });
    g.push(Op::InstanceNorm { dst: d3c, src: d3c, gamma: None, beta: None,
                              c: 4 * ngf, hw: hw4, unbiased: 0 });
    let mut body = g.buf("d3.relu".into(), 4 * ngf * hw4);
    g.push(Op::Relu { dst: body, src: d3c, n: 4 * ngf * hw4 });

    // ---- ResnetBlocks (the encoder half of each ResnetAdaILNBlock) ----------
    let c4 = 4 * ngf;
    for i in 0..n_res {
        let base = 12 + i;
        let an = format!("DownBlock.{base}.conv_block.1.weight");
        let bn = format!("DownBlock.{base}.conv_block.5.weight");
        let wa = g.weight(w, &an, c4 * c4 * 9)?;
        let wb = g.weight(w, &bn, c4 * c4 * 9)?;
        let t1 = g.buf(format!("res{i}.conv1"), c4 * hw4);
        g.push(Op::Conv { dst: t1, src: body, w: wa, b: None, k: 3, stride: 1,
                              cin: c4, cout: c4, h: h4, wd: w4, refl: true });
        g.push(Op::InstanceNorm { dst: t1, src: t1, gamma: None, beta: None,
                                  c: c4, hw: hw4, unbiased: 0 });
        let t2 = g.buf(format!("res{i}.relu"), c4 * hw4);
        g.push(Op::Relu { dst: t2, src: t1, n: c4 * hw4 });
        let t3 = g.buf(format!("res{i}.conv2"), c4 * hw4);
        g.push(Op::Conv { dst: t3, src: t2, w: wb, b: None, k: 3, stride: 1,
                              cin: c4, cout: c4, h: h4, wd: w4, refl: true });
        g.push(Op::InstanceNorm { dst: t3, src: t3, gamma: None, beta: None,
                                  c: c4, hw: hw4, unbiased: 0 });
        let out = g.buf(format!("res{i}.out"), c4 * hw4);
        g.push(Op::Add { dst: out, a: body, b: t3, n: c4 * hw4 });
        body = out;
    }

    // ---- the CAM -----------------------------------------------------------
    // Upstream pools `body` twice - `adaptive_avg_pool2d(x, 1)` and
    // `adaptive_max_pool2d(x, 1)` - and uses each pooled vector TWICE: once as
    // the `Linear` argument that produces `gap_logit`/`gmp_logit`, and once as a
    // per-channel SCALE on the full feature map, `x * gap_weight.unsqueeze(2)
    // .unsqueeze(3)`. What the 1x1 conv then reduces is the concatenation of
    // those two scaled MAPS, `[2C][H][W]`, and its output is `[C][H][W]`.
    //
    // Concatenating the two pooled `[C][1][1]` vectors instead would be a
    // different function of a different rank, and it would also force the decoder
    // onto the resnet body: a 1x1 map cannot be reflection-padded by 1 and
    // convolved, so there would be nothing else for `x_up` to start from. One
    // cause, two symptoms - and the more serious of the two, because the pooled
    // form discards every spatial mode of the map the decoder is conditioned on.
    //
    // `gap_logit`/`gmp_logit` (and `heatmap`) are NOT on the released inference
    // path - upstream's `predict.py` calls the generator and keeps only
    // `out, _, _` - but they are built anyway: they are two ops and a 2-element
    // buffer, and leaving them out would make the plan's op list harder to check
    // against `networks.py` than it is worth. Nothing reads them, so the packer
    // gives them the smallest slot it has.
    let gap_w = g.weight(w, "gap_fc.weight", c4)?;
    let gmp_w = g.weight(w, "gmp_fc.weight", c4)?;
    let gap = g.buf("cam.gap".into(), c4);
    g.push(Op::ChannelMean { dst: gap, src: body, c: c4, hw: hw4 });
    let gmp = g.buf("cam.gmp".into(), c4);
    g.push(Op::ChannelMax { dst: gmp, src: body, c: c4, hw: hw4 });
    let logits = g.buf("cam.logits".into(), 2);
    g.push(Op::Linear { dst: logits, src: gap, w: gap_w, b: None, rows: 1, cin: c4, cout: 1 });
    let logits_hi = g.alias("cam.logits.hi".into(), logits, 1, 1);
    g.push(Op::Linear { dst: logits_hi, src: gmp, w: gmp_w, b: None, rows: 1, cin: c4, cout: 1 });

    // The two scaled maps, concatenated. `cat` is a real [2C][H][W] buffer and
    // `cat_hi` is an alias of its upper half, so `lg_conv1x1` sees one contiguous
    // `[2C][hw]` plane with the second C channels offset by C*hw - the same
    // `torch.cat((gap, gmp), 1)` upstream builds, without a copy of either half.
    let scaled_gap = g.buf("cam.gap.scaled".into(), c4 * hw4);
    g.push(Op::ChannelMul { dst: scaled_gap, src: body, s: gap_w, c: c4, hw: hw4 });
    let scaled_gmp = g.buf("cam.gmp.scaled".into(), c4 * hw4);
    g.push(Op::ChannelMul { dst: scaled_gmp, src: body, s: gmp_w, c: c4, hw: hw4 });
    let cat = g.buf("cam.cat".into(), 2 * c4 * hw4);
    g.push(Op::Copy { dst: cat, src: scaled_gap, n: c4 * hw4 });
    let cat_hi = g.alias("cam.cat.hi".into(), cat, c4 * hw4, c4 * hw4);
    g.push(Op::Copy { dst: cat_hi, src: scaled_gmp, n: c4 * hw4 });

    let w11 = g.weight(w, "conv1x1.weight", c4 * 2 * c4)?;
    let b11 = g.weight(w, "conv1x1.bias", c4)?;
    let feat = g.buf("cam.feat".into(), c4 * hw4);
    g.push(Op::Conv1x1 { dst: feat, src: cat, w: w11, b: Some(b11),
                         cin: 2 * c4, cout: c4, hw: hw4 });
    g.push(Op::Relu { dst: feat, src: feat, n: c4 * hw4 });

    // The reference then pools `feat` over space with `adaptive_avg_pool2d(x, 1)`
    // - which, with `feat` a map rather than a 1x1 tensor, is now a real op and
    // not an identity.
    let wf0 = g.weight(w, "FC.0.weight", c4 * c4)?;
    let wf2 = g.weight(w, "FC.2.weight", c4 * c4)?;
    let wg = g.weight(w, "gamma.weight", c4 * c4)?;
    let wb = g.weight(w, "beta.weight", c4 * c4)?;
    let fc_in = g.buf("cam.fc_in".into(), c4);
    g.push(Op::ChannelMean { dst: fc_in, src: feat, c: c4, hw: hw4 });
    let fc0 = g.buf("cam.fc0".into(), c4);
    g.push(Op::Linear { dst: fc0, src: fc_in, w: wf0, b: None, rows: 1, cin: c4, cout: c4 });
    g.push(Op::Relu { dst: fc0, src: fc0, n: c4 });
    let fc2 = g.buf("cam.fc2".into(), c4);
    g.push(Op::Linear { dst: fc2, src: fc0, w: wf2, b: None, rows: 1, cin: c4, cout: c4 });
    // `FC` ENDS IN A RELU - `FC = [Linear, ReLU, Linear, ReLU]` - and stopping
    // after the second Linear would not be a detail: gamma/beta below are what the
    // eight CAM-driven adaILN blocks scale and shift by, and without the last
    // nonlinearity a negative `fc2` goes through and gamma carries the opposite
    // sign of the function upstream computes.
    g.push(Op::Relu { dst: fc2, src: fc2, n: c4 });
    // gamma/beta are Linear layers with no bias, applied to the [1][C] row and
    // then broadcast over space by the adaILN - so they stay [C] vectors here and
    // the kernel indexes them by channel.
    let cam_gamma = g.buf("cam.gamma".into(), c4);
    g.push(Op::Linear { dst: cam_gamma, src: fc2, w: wg, b: None, rows: 1, cin: c4, cout: c4 });
    let cam_beta = g.buf("cam.beta".into(), c4);
    g.push(Op::Linear { dst: cam_beta, src: fc2, w: wb, b: None, rows: 1, cin: c4, cout: c4 });

    // ---- ResnetAdaILNBlocks (the decoder half) -----------------------------
    // Both norms of a block read the SAME CAM gamma/beta; only `rho` differs,
    // which is why the checkpoint carries norm1.rho and norm2.rho and nothing
    // else. Each block is residual: `x = x + block(x)`.
    //
    // `x_up` starts from the CAM's OUTPUT, not from the resnet body - upstream's
    // `x = self.relu(self.conv1x1(x))` is the `x` the block loop runs on.
    let mut x_up = feat;
    for i in 1..=n_res {
        let p = format!("UpBlock1_{i}");
        let wc1 = g.weight(w, &format!("{p}.conv1.weight"), c4 * c4 * 9)?;
        let wc2 = g.weight(w, &format!("{p}.conv2.weight"), c4 * c4 * 9)?;
        let r1 = g.weight(w, &format!("{p}.norm1.rho"), c4)?;
        let r2 = g.weight(w, &format!("{p}.norm2.rho"), c4)?;
        let t1 = g.buf(format!("up1_{i}.conv1"), c4 * hw4);
        g.push(Op::Conv { dst: t1, src: x_up, w: wc1, b: None, k: 3, stride: 1,
                              cin: c4, cout: c4, h: h4, wd: w4, refl: true });
        // The CAM's gamma/beta are the OUTPUTS of two Linear layers, so this is
        // the BUFFER form of the op - and that distinction is the whole reason
        // `AdaILnBuf` exists (see its doc comment).
        g.push(Op::AdaILnBuf { dst: t1, src: t1, rho: r1, gamma: cam_gamma,
                               beta: cam_beta, c: c4, hw: hw4 });
        g.push(Op::Relu { dst: t1, src: t1, n: c4 * hw4 });
        let t2 = g.buf(format!("up1_{i}.conv2"), c4 * hw4);
        g.push(Op::Conv { dst: t2, src: t1, w: wc2, b: None, k: 3, stride: 1,
                              cin: c4, cout: c4, h: h4, wd: w4, refl: true });
        g.push(Op::AdaILnBuf { dst: t2, src: t2, rho: r2, gamma: cam_gamma,
                               beta: cam_beta, c: c4, hw: hw4 });
        let out = g.buf(format!("up1_{i}.out"), c4 * hw4);
        g.push(Op::Add { dst: out, a: x_up, b: t2, n: c4 * hw4 });
        x_up = out;
    }

    // ---- the two UpBlock2 stages ------------------------------------------
    // upsample 2x, ReflectionPad2d(1), 3x3 conv, then ILN with its OWN
    // rho/gamma/beta (an AdaILn with all three supplied), then ReLU.
    let stages: [(usize, &str); 2] = [(2, "3"), (7, "8")];
    let mut cur = x_up;
    let (mut ch, mut chh, mut chw) = (c4, h4, w4);
    for (idx, iln) in stages {
        let (uh, uw) = (chh * 2, chw * 2);
        let up = g.buf(format!("up2_{idx}.upsample"), ch * uh * uw);
        g.push(Op::Upsample2x { dst: up, src: cur, c: ch, h: chh, wd: chw });
        let cout = ch / 2;
        let wc = g.weight(w, &format!("UpBlock2.{idx}.weight"), cout * ch * 9)?;
        let conv = g.buf(format!("up2_{idx}.conv"), cout * uh * uw);
        g.push(Op::Conv { dst: conv, src: up, w: wc, b: None, k: 3, stride: 1,
                              cin: ch, cout, h: uh, wd: uw, refl: true });
        let rho = g.weight(w, &format!("UpBlock2.{iln}.rho"), cout)?;
        let gam = g.weight(w, &format!("UpBlock2.{iln}.gamma"), cout)?;
        let bet = g.weight(w, &format!("UpBlock2.{iln}.beta"), cout)?;
        g.push(Op::AdaILn { dst: conv, src: conv, rho, gamma: Some(gam),
                            beta: Some(bet), c: cout, hw: uh * uw });
        g.push(Op::Relu { dst: conv, src: conv, n: cout * uh * uw });
        cur = conv;
        ch = cout;
        chh = uh;
        chw = uw;
    }

    // ---- the tail ----------------------------------------------------------
    // ReflectionPad2d(3) + 7x7 conv back to 3 channels, then tanh(x + input): the
    // residual is with the INPUT plane, not with the upsampled feature.
    let wlast = g.weight(w, "UpBlock2.11.weight", 3 * ch * 49)?;
    let tail = g.buf("tail.conv".into(), 3 * chh * chw);
    g.push(Op::Conv { dst: tail, src: cur, w: wlast, b: None, k: 7, stride: 1,
                          cin: ch, cout: 3, h: chh, wd: chw, refl: true });
    let out = g.buf("output".into(), 3 * chh * chw);
    g.push(Op::TanhAdd { dst: out, x: tail, skip: x_in, n: 3 * chh * chw });

    let mut plan = Plan {
        bufs: g.bufs, ops: g.ops, weights: g.weights,
        input_len: 3 * hw, out_len: 3 * chh * chw,
        c: 3, h, wd, arena_len: 0, dumps: false, variant: w.variant.clone(), n_res,
        output: out,
    };
    plan.pack();
    Ok(plan)
}

/// The output size of a `k` x `k` stride-`s` convolution whose input is
/// reflection-padded by `k / 2`, then convolved with no padding:
/// `floor((n + 2*(k/2) - k) / s) + 1`.
///
/// For k = 3, s = 2 and even n this is `n / 2`, which is what makes the network
/// shape-preserving when both dimensions are multiples of 4; for an odd n it
/// loses a sample, which is why `image::pad_to_multiple` exists.
pub fn stride_out(n: usize, k: usize, s: usize) -> usize {
    (n + 2 * (k / 2) - k) / s + 1
}
