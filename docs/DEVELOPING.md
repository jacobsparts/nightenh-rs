# Developing this engine

The same shape as the rest of the family: the plan is the artifact, both backends
walk it, and every flag below exists to make one of those two sentences
checkable rather than asserted.

## The development build

```sh
cargo build --release --features dev
```

`dev` is off by default and a release build **refuses the flags by name** rather
than ignoring them:

```
nightenh: `--dump` is a development flag and this is a release build
nightenh: rebuild with `cargo build --release --features dev` for
nightenh: --dump, --profile, --factor and --verify-cpu
```

A flag that compiled away silently would let a script believe it had produced a
dump it never wrote, so the release binary exits 2 and says which build to make.

The binary lands in the **family** target directory, not this crate's:
`.cargo/config.toml` at the family root sets `target-dir = "target"`, so the
artifact is `../target/release/nightenh` and that is the path to run - there is no
`nightenh-rs/target/`. Two habits are worth having: stat the binary against the
source after a build before believing a measurement, and remember that a
`--features dev` build and a plain one write to the SAME path, so the last build
wins.

| flag | what it does |
|---|---|
| `--dump <dir>` | write every named buffer of the plan as an `.npy` of f32 values, under the plan's own buffer names (`res2.conv1`, `cam.gap`, `output`, ...) |
| `--profile` | print the per-op census of the plan: `conv3x3_refl 18`, `relu 16`, `instnorm 11`, ... |
| `--factor <n>` | pad a native-size run to a multiple of `n` instead of the default 64 |
| `--verify-cpu` | run every op twice from the same inputs and require the same bytes (see below; slow) |

Three environment variables reach the same machinery when a flag would be in the
way. `NIGHTENH_PLAN=1` prints the whole plan - every weight with its length, every
buffer with its offset and alias - and it goes WITH `--dump`, because a plan whose
buffers are dumped is a different plan (aliasing is what a dump disables, so the
listing would otherwise describe an arena that is not the one being run).
`NIGHTENH_TRACE=1` prints each conv op's weight name, its first few values and the
sum of its source buffer, which tells "the wrong tensor was read" apart from "the
right tensor was read differently"; `NIGHTENH_GPU_TRACE=1` syncs after every
launch and names the op that failed, because an illegal address only surfaces at
the NEXT sync otherwise.

`NIGHTENH_CPU_TIME=1` is the CPU's answer to the same question, and the one to
reach for before optimising anything: it times every op and reports the total by
kind, which is how the finding that **98.6% of a single-threaded CPU pass is
convolution** was made rather than guessed. Take it single-threaded
(`RAYON_NUM_THREADS=1`), where the whole pass is 4.72 s of op time at 256x256 and
the shape of the problem is legible: `conv3x3_refl` 4.30 s over 18 ops (91.1%),
`down3x3` 0.24 s over 2 (5.1%), `conv7x7_refl` 0.07 s (1.4%), `conv1x1` 0.05 s
(1.0%), and every norm, relu, adailn, add, linear, upsample and copy together
0.07 s (1.4%). On 24 threads the same pass is 0.49 s of wall; the per-op split
there says something different again, because the two `down3x3` ops and the two
3x3s that run at 512x512 in the decoder parallelise less well than the sixteen
`256 -> 256` ones in the resnet body, which are the bulk of the work and the only
convolution at 91% pure enough to schedule trivially.
`NIGHTENH_VERIFY_CPU=1` is what `--verify-cpu` sets.

`NIGHTENH_GPU_TIME=1` is the device's twin of `NIGHTENH_CPU_TIME`, and it exists
because the CPU one is the reason the CPU path was fixable at all: it times every
op and reports the total by kind, so a target is named rather than guessed. Two
caveats come with it. It synchronises after every op, because a launch only QUEUES
work and timing without a sync measures the host rather than the kernel - so the
pass it describes is fully serialised and the absolute seconds are not a timing to
quote, while the SHARES are. And it answers a question about the ENGINE rather than
about a kernel: the ops in a profile differ in `cin`, `cout` and plane size, so a
per-op time is the product of a kernel and a shape, and attributing it to either
one needs the shape held fixed - which is what `tests/reference.rs`'s
`gpu_3x3_kernel_throughput` does for the two 3x3 kernels, with the result that the
toolkit's one-thread-per-output kernel is FASTER than the project's
reuse-across-channels one at this network's channel counts, the opposite of what
reading the two sources suggests. The same discipline applies to the GEMMs:
`gpu_tiled_gemm_on_the_im2col_shape` times the toolkit kernel on the three im2col
shapes the network actually produces, and `gpu_gemm_double_buffered` times the
experimental one on the SAME shapes in one run, so the two are always compared
under one load and one set of shapes - and it checks the new kernel against a
triple loop FIRST, which is what caught all four of the staging bugs its staging
had. When measuring this device, `--native` and a
logged load average both still apply.

`--verify-cpu` and its variable are **CPU-only**, and what they check is worth
stating precisely: each op runs twice from the same inputs and the bytes it writes
must match. That catches an op that is not a function of its input - a race
between the pool's tasks, an uninitialised read, an aliasing the plan should not
have made. It does NOT compare the CPU against the device or against the
reference; the per-buffer `--dump` comparison is what does that. Two things about
the implementation are easy to get wrong: the inputs must be snapshotted **before**
the first run (most ops here write their own input in place, so a snapshot taken
afterwards hands the second run its own answer and reports an op as
non-deterministic), and a pass measured under this flag is doubled by construction,
so its timing is not a timing to quote.

## Tests

```sh
cargo test --release
```

Fourteen tests, and the ones that need a 42 MB checkpoint SKIP with a printed
reason rather than failing - this repository has to build and test cleanly without
the weights:

* seven in `src/exec_cpu.rs`: a 3x3 in each padding mode against a stored
  expectation from `tools/reference.py`, the reflection fold's two rules, `co`-then-`ci`
  weight indexing on a case where mixing them changes the answer, the two variance
  conventions side by side, and **one bit-equality test per vector kernel** - the
  3x3 stride-1, the 3x3 stride-2 and the 7x7, each against `conv_pad_scalar`. See
  "The CPU convolutions" below for why those are the load-bearing tests of the
  arrangement rather than a nicety.
* four in `tests/reference.rs`, against the `.npy` fixtures `tools/reference.py`
  writes: the CPU backend against the transcription, the two backends against each
  other, **the GPU against itself** - three runs of the same plan over the same
  input, asserted bit-equal - and the register-hoisted 3x3 kernel against the
  toolkit's, for the difference it has to show.
* three measurements, which print their table and assert only that the ordering is
  stable, so the numbers in the README can be re-taken from a clone rather than
  believed: `gpu_3x3_kernel_throughput`, `gpu_tiled_gemm_on_the_im2col_shape` and
  `gpu_gemm_double_buffered`.

The bit-equality one is not decoration. A tolerance check passes or fails by luck
when the underlying op is racy: a race here showed 6e-4 on one run and 2.5e-1 on
the next. Bit-equality across three runs is the assertion that makes a race fail
every time.

Note what this list does NOT contain. Nothing in it compares the transcription
against upstream, because the fixtures are written BY the transcription and would
agree with it - so a mistake in `tools/reference.py` is visible to `cargo test` in
no way at all. `tools/check_upstream.py` is that check, and it needs torch and a
42 MB checkpoint.

## The CPU convolutions

The CPU twin walks the same plan as the device, and the obvious way to write the
convolution is the wrong one: one output element at a time, one accumulator per
element, chained through `cin * 9` sequential multiply-adds is a latency chain,
and it is what the scalar twin does. The twenty 3x3s are 96.6% of a pass's FLOPs,
with the sixteen `256 -> 256` ones - eight resnet and eight adaILN, one 3x3 each in
the latter - carrying 73.6% of the whole network between them, so the scalar twin
runs at about 2 GFLOP/s per core while the numpy reference - which reaches OpenBLAS
through `einsum` - runs the same shape at 35. Parallelising over all 24 cores does
not close that gap; the vector kernels are what does.

`conv_pad` now has a vector kernel for each of the three convolution shapes in
this network except the single 1x1, and all three are behind one dispatch:

| shape | where it is | count | share of FLOPs | kernel |
|---|---|---|---|---|
| 3x3 stride 1 | the resnet body and the decode 3x3s | 18 | 92.0% | `conv3x3_s1_avx2` |
| 3x3 stride 2 | the two `DownBlock` downsamples | 2 | 4.6% | `conv3x3_s2_avx2` |
| 7x7 stride 1 | the two `DownBlock` 7x7s | 2 | 2.3% | `conv7x7_s1_avx2` |
| 1x1 stride 1 | the CAM's `conv1x1` | 1 | 1.0% | the toolkit's |

The shares are measured, not derived: a forward hook on upstream's `Conv2d`s at
512x512 totals **420.0 GFLOP** and splits 92.0 / 4.6 / 2.3 / 1.0. Counting only
the sixteen `256 -> 256` resnet 3x3s would give 74% for the stride-1 row and would
be wrong - the eight adaILN-block 3x3s are the same shape and the same cost.

The share of FLOPs is not the share of TIME, which is why all three kernels
matter: the two downsamples are a THIRD of what remains in a single-threaded pass
and the 7x7s an eighth of it, on 4.6% and 2.3% of the FLOPs. Vectorising the
stride-1 3x3s makes them four times faster and still leaves them 91% of the pass,
and the single-threaded census at 256x256 is 4.72 s in total: `conv3x3_refl`
4.30 s, `down3x3` 0.24 s, `conv7x7_refl` 0.07 s, `conv1x1` 0.05 s.

Two properties make every one of them **bit-identical** to `conv_pad_scalar`, and
both are requirements rather than implementation details:

* **The vector is over output COLUMNS.** Eight neighbouring outputs share a
  weight and each SIMD lane carries its own accumulator, so every output element
  still accumulates its terms in the scalar twin's order - `ci` outermost, then
  `ky`, then `kx` - and the ORDER is what the two backends are compared on. A
  vector over `cout` or over `ci` would sum in another order and turn every
  backend comparison into noise.
* **The multiply and the add stay separate instructions.** `_mm256_mul_ps` then
  `_mm256_add_ps`, never `_mm256_fmadd_ps`. The scalar twin rounds them
  separately (fp-contract is off in this crate), so a fused multiply-add would put
  the CPU path on a different footing from the CUDA path for a speed gain that
  measured as nothing - 34.7 against 35.1 GFLOP/s, inside the noise of a shared
  machine.

Each kernel has a test asserting that equality element for element
(`conv3x3_vector_is_bit_identical_to_scalar`,
`conv3x3_stride2_vector_is_bit_identical_to_scalar`,
`conv7x7_vector_is_bit_identical_to_scalar`), and the whole model's PNG is
byte-identical to what the scalar path produces. If a kernel ever differs, one of
the two properties above has been broken; the test is the place that is noticed.

**The border is vectorised in all three, and that is the single most important
detail in this section.** A tap whose column leaves the plane is not a load: it is
folded back in (reflection) or dropped (zero padding). The first version of the
stride-1 kernel sent every such block through the scalar loop, which sounds
harmless and is not: at 64 wide - the resnet body's plane when the model runs at
256x256, where the sixteen busiest convolutions in the network run - two of the
eight column blocks
contain an edge, so a quarter of that work was scalar and the op measured 6.2
GFLOP/s against 25 with the border vectorised. The fix is not to change the
arithmetic but to change how the shifted vectors are BUILT: gather them a column at
a time with the fold applied during the gather. The accumulation that follows is
the same code in the same order, which is what keeps the two paths identical.

Row blocking is what takes the kernels off the load ports: without it every output
row fetches its own input rows, so a tap costs a load and a multiply-add; with it,
one fetch feeds up to three output rows (nine multiply-adds) at stride 1, two at
stride 2. That does not touch the addition order, because routing an input row to
an output row still performs that output's `ky` taps in ascending `ky`. The 7x7
blocks four rows rather than eight, because seven shifted input vectors have to
stay live beside the accumulators and eight of each does not fit the register file.

The stride-2 kernel at 3x3 is the odd one out, and it is worth knowing why it does
not look like the others. Its three taps read input columns `2*ox - 1`, `2*ox` and
`2*ox + 1` - two arithmetic progressions, the even columns and the odd ones - so it
builds those two views ONCE per op (`strided_views`, with the border folded in
while they are built) and then all three taps are ordinary contiguous loads with no
gather in the inner loop. A stride-2 tap can never be a single shifted load, so a
per-block gather there would have been the expensive thing the borders taught us to
avoid. The row routing is a parity test rather than a shift: input row `iy` feeds
the output row named by `iy + 1 - ky` only when that value is even.

`stride_out` is `(n, k, s)` in that order, and calling it `(h, 2, 3)` sizes the
destination too small, which makes `par_chunks_mut` hand out more chunks than there
are output channels and indexes the weight buffer off its end - a panic that names
a buffer, not a shape.

The dispatch is `is_x86_feature_detected!("avx2")` behind
`#[cfg(target_arch = "x86_64")]`, the arrangement `ifan-rs` uses in this family
after realesrgan-rs: the instruction set is never assumed by the build, so there
is no `-C target-cpu` to remember and a machine without AVX2 takes the scalar path
instead of dying on an illegal instruction. The `#[target_feature]` function is
also separate from the rayon driver on purpose - a closure does not inherit its
enclosing function's target features.

## Reproducing a divergence

When the two backends disagree by more than round-off, the order below finds the
op rather than guessing at it:

1. `cargo run --release --example op_determinism -- [h] [w]` truncates the plan's
   op list to `n` ops, runs the GPU executor twice over the same input for every
   `n`, and prints the first `n` at which the two runs disagree. Use it when the
   output itself is not reproducible (five runs, five different PNGs); the op it
   names is the one to read.
2. `cargo build --release --features dev` then run both backends with
   `--dump /tmp/cpu-d` and `--dump /tmp/gpu-d`, and compare buffer by buffer. The
   first buffer whose difference is above round-off is the op to read; the ones
   after it inherit it.
3. `NIGHTENH_TRACE=1` on the CPU prints each conv's weight name and an f64 sum of
   its values, which distinguishes "the wrong tensor was read" from "the right
   tensor was read differently".

A note on what to look for in a GPU kernel, because this one is not visible by
reading: **a block reduction writes its partials into shared memory and reads the
result out of index 0.** If the same array is reused for the next pass without a
`__syncthreads()` between the read and the write, the halving tree's first step can
fold an index that a faster warp has already overwritten, and the value every
thread then uses is a partial sum. That failure is invisible in the code and
obvious in two runs of the same input.

The engine's own kernels declare their shared arrays at a FIXED size (256
entries, the block width) rather than sized to `blockDim`, which is the same rule
`lg_layer_norm_warp` states in the toolkit: a kernel must not acquire a shared
memory requirement its callers cannot satisfy.

## Adding an op

Three places, which the compiler checks against each other:

1. `Op` in `src/model.rs`, with the buffer and weight indices the plan will
   record.
2. `exec_cpu`'s `match` arm, which is the definition of the op's arithmetic - the
   device reductions here reproduce the kernel's lane order exactly
   (`src/exec_cpu.rs`'s `reduce` and `LANES`), because a serial sum is a different
   and less accurate function whose error grows with the plane.
3. `exec_gpu`'s `match` arm, and the kernel's name in `PROJECT_KERNELS` or
   `TOOLKIT_KERNELS` in `build.rs`.

Both kernel lists are checked against the source they are compiled from in BOTH
directions, because a kernel missing from its `--entries` list is pruned from the
fatbin and fails at LAUNCH rather than at build time.

## Benchmarking against torch

`tools/torch_bench.py` runs the SAME generator in torch - same weights, same
block order, same numeric conventions - so "how fast is this network on PyTorch
CPU / PyTorch CUDA" is a measurement rather than a guess. It is not a third
reference implementation: every operator is a torch call and the weight names come
from `reference.read_safetensors`, so a disagreement with `reference.py` is torch's
arithmetic and not a second reading of upstream. Run it from an interpreter that
has torch (here: `python3.11`, not the default `python3`).

Two things to get right when reading its output:

* **Threads.** `torch.set_num_threads` defaults to the core count (16 on this
  24-core box), so torch uses multiple cores on CPU by default and the
  like-for-like comparison against this engine is torch's DEFAULT against the
  engine's default pool - `--threads 1` measures something else. Best-of-N with
  the load average logged, as everywhere else here: the 24-thread runs on a busy
  box were erratic in a way the best-of-N figure is not.
* **cuDNN.** A torch CUDA run is permitted to use cuDNN and the engine is
  prohibited from linking it, so a torch-CUDA/engine-GPU ratio measures the
  library, not the kernels. Worth knowing how large that is: on the resnet 3x3
  shape (256 -> 256, 3x3, 128x128) cuDNN 9.1 reports 13,537 GFLOP/s, which is
  ABOVE this card's 9,800 GFLOP/s FP32 peak and therefore not FP32 arithmetic at
  all. With `torch.backends.cudnn.enabled = False` the same call is 5,195
  GFLOP/s, i.e. 53% of peak in real FP32 - and still 43.8x the toolkit's
  `lg_conv3x3s1p1` (118.5) and 8.6x the register-hoisting experiment (607.4).
  So the honest reading of that experiment is that it is a fifth of the way to
  what a mature library reaches on this card in FP32, not that it closed the gap.

## What torch does that we do not (and what it would cost)

Worth writing down because the comparison is easy to misread twice in opposite
directions. With `torch.backends.cudnn.enabled = False` - so no library this
project may not link - torch convolves at **5,195 GFLOP/s** where the best kernel
in this tree reaches **607**. That is an 8.6x gap and it is not cuDNN's doing, so
the question "what is it doing" is a fair one. The profiler answers it:
`im2col_kernel<float>` (25% of device time) then `sgemm_128x128x8_NN_vec` (73%),
3.087 ms for M=256, K=2304, N=16384 = 6.26 TFLOP/s, 64% of this card's FP32 peak.
So torch materialises the patches and runs a tiled SGEMM.

Every kernel in this repository is a DIRECT convolution with no operand staged
through shared memory - which is the SAME lesson the CPU side already learned, where
the scalar conv lost to numpy only because `einsum` reaches OpenBLAS's GEMM. So the
obvious move is im2col + GEMM, and the toolkit already has the GEMM:
`lg_f32_gemm_tiled`, used by `locate-anything-rs` and `realesrgan-rs`.

It is worth knowing what that actually buys BEFORE rewriting anything, because
`tests/reference.rs`'s `gpu_tiled_gemm_on_the_im2col_shape` measures it: on the
im2col shape itself the toolkit's tiled GEMM runs at **1,137 GFLOP/s (12% of
peak)** where cuBLAS gets 6,260 (64%) on the identical GEMM. The family's GEMM is
5.5x behind cuBLAS's, so an im2col convolution built on it lands at about **1.9x**
over the best direct kernel here - and then pays the im2col copy, which was 25% of
torch's device time. That is a real but modest win, nothing like the 8.6x the
headline suggests, and it is the number that should decide the question.

| implementation, same work | GFLOP/s | % of FP32 peak |
|---|---|---|
| `ne_conv3x3_refl` (project, direct) | 82.9 | 0.8% |
| `lg_conv3x3s1p1` (toolkit, direct) | 118.5 | 1.2% |
| `nex_conv3x3s1p1_x8` (experimental, register-hoisted) | 607.4 | 6.2% |
| `lg_f32_gemm_tiled` (toolkit GEMM, on the im2col shape) | 1,136.7 | 11.6% |
| `nex_f32_gemm_db` (experimental, staged + double-buffered) | ~1,690 | 17.2% |
| torch, cuDNN off: im2col + cuBLAS SGEMM | 5,195 | 53.0% |

**The 5.5x has been investigated and is settled as far as this tree can take
it.** `nex_f32_gemm_db` in `cuda/nightenh.cu` (no op launches it) is the
experiment: a 128x64 tile per 256-thread block, 8x4 accumulators per thread, BOTH
operands staged in shared memory as `float4` in an `[index][k]` layout, k-tiles
double-buffered with one barrier per step, and the staging branch-free. It reaches
**1.4x** the toolkit's GEMM, which is real and is the number to quote - but it is
17% of peak where cuBLAS is at 53%.

Two further experiments came back NEGATIVE and they are the more useful result,
because they say which knob is NOT the binding one:

* **Instruction density is not the limit.** Reordering the consume loop to
  k-outermost (each staged operand read once per k instead of once per FMA, 256
  shared reads per k-tile down to 96) raised the FMA fraction of the SASS from
  **12.3% to 25.1%** and changed the runtime by nothing measurable. The kernel is
  not issue-limited, so "our kernel wastes 2.5x the instructions per unit of work
  against cuBLAS" was a true observation pointing at the wrong constraint.
* **A bigger tile is a loss, because of OCCUPANCY.** Copying cuBLAS's own
  configuration - a 128x128 tile, 8x8 accumulators per thread - reached a better
  FMA density (15.3%) at `REG:160` and was **2.1x SLOWER** (1625 -> 776 GFLOP/s).
  The cause is arithmetic, not spilling (`LDL`/`STL` count was 0 in both versions):
  160 registers x 256 threads is 40960 of the 65536-register file, so one block
  fits per SM against two at `REG:91`. On this card, at these FMA latencies,
  halving the resident warps costs more than the better instruction mix buys - so
  cuBLAS's tile shape is not transferable without cuBLAS's instruction schedule.

What is left is a different allocator and schedule, not another tile size. The
measured sweet spot here is the 128x64 tile at `REG:91`, two blocks per SM.

## The reference

`tools/reference.py` is a torch-free numpy transcription of upstream's
`networks.py`, one function per upstream class in the order they are defined, with
the upstream expression on the same line wherever a merge or a reassociation was
possible and `LOG` recording every decision that could have gone the other way.
The fixtures it writes are what `tests/reference.rs` compares against.

**A transcription is not an oracle until something independent has confirmed it.**
Every comparison in this repository is against values `tools/reference.py` itself
produced, so a divergence in it is invisible from inside the tree: the engine and
the fixtures agree, because they are the same reading of upstream. The independent
check is `tools/check_upstream.py`. Run it whenever this file is edited, or
whenever a checkpoint is converted:

```sh
python3 tools/reference.py --check                          # self-test + the shape rule
python3 tools/reference.py --fixture tests/data             # regenerate the fixtures
python3 tools/reference.py --image in.png --out out.png     # run the file itself

# Against the real upstream source. `--ref` also compares the converted weights
# with the released .pt TENSOR BY TENSOR, which is the converter's own check - the
# per-stage table cannot make it, because both implementations are fed the same
# file, so a bad conversion agrees with itself stage by stage.
/usr/local/bin/python3.11 tools/check_upstream.py \
    --networks ~/night-enhancement/networks.py \
    --weights ../models/lol.safetensors --ref ../models/LOL_params_0900000.pt
```

It needs torch, upstream's `networks.py` and a 42 MB checkpoint, so it is a tool
and not a test: `cargo test` must stay green on a machine with none of them. What
it reports first is the answer - the first stage that diverges is the whole
diagnosis, and everything below it is that divergence propagated.

## The converter

`tools/convert.py` reads the two released `.pt` files without torch, and they are
on **opposite sides of torch's 1.6 container change**: the LOL one is a legacy
pickle stream (five pickles, then the storage payloads back to back with no
header), the de-light-effects one is a zip (`data.pkl` plus one file per storage).
Which reader runs is decided by the file's first two bytes. Both are decoded by
measurement rather than by trusting a description, and the converter cross-checks
what it found against what the metadata it writes claims.
