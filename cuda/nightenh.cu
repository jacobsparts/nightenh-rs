//! This engine's kernel family: the reflection-padded convolutions, the two
//! normalisations the generator mixes, and the CAM's pooling and scaling steps.
//!
//! The toolkit's kernels are in `cuda/kernels.cu` and its own module. What is
//! here is here because the generator needs something the toolkit does not have
//! - see `build.rs` for why each toolkit kernel is absent - and every layout
//! obeys `cuda/CONVENTIONS.md`: NCHW, `[c_out][c_in][kh][kw]` weights, `c_in`
//! contiguous within a tap, `ky, kx, ci` accumulation order, and `extern "C"` so
//! the names match PROJECT_KERNELS exactly.
//!
//! TWO NUMERICS RULES, which are the reason several of these exist at all:
//!
//! 1. **Reflection, not zero, padding.** Every convolution in this network is
//!    wrapped in `nn.ReflectionPad2d` - 3 around the two 7x7s, 1 around every
//!    3x3 - and reflection EXCLUDES the edge sample (`|i|` mirrored about the
//!    first interior index). A zero-padded conv is wrong at the border, which is
//!    not a rounding difference: it changes the picture.
//! 2. **`nn.InstanceNorm2d` is BIASED and `torch.var` is UNBIASED.** Both appear
//!    in this one network. `ne_instance_norm` takes the divisor as an argument
//!    (`unbiased != 0` divides by n - 1), and `ne_adailn` computes both halves of
//!    its mix with the UNBIASED form, because `AdaILN.forward` calls
//!    `torch.var(..., [2, 3])` and `torch.var(..., [1, 2, 3])`. On a 512x512 map
//!    the two differ in the 4th significant figure of the normalised value.

#include <cuda_runtime.h>
#include <math_constants.h>

#define NE_EPS 1e-5f

// Reflection index for one axis: the coordinate `i` folded into `[0, n)` about
// the interior, i.e. `-1 -> 1` and `n -> n - 2`. The same rule as
// `np.pad(mode="reflect")` and `F.pad(mode="reflect")`; `symmetric` (which
// repeats the edge) is a different function and would pass a visual check.
__device__ __forceinline__ int ne_refl(int i, int n) {
    if (n == 1) return 0;
    while (i < 0 || i >= n) {
        i = (i < 0) ? -i : 2 * n - 2 - i;
    }
    return i;
}

// ---------------------------------------------------------------------------
// Reflection-padded convolutions
// ---------------------------------------------------------------------------
//
// All four take `x` and `y` as [n][c][h][w] with `c` as a stride of h*w, weights
// as [c_out][c_in][k][k] with `c_in` contiguous inside a tap, and accumulate in
// ky, kx, ci order - the order the CPU twins use and the order `lg_conv3x3s1p1`
// documents, so a mismatch is a numerics bug and not a reordering.
//
// One thread per output element for the 3x3s, which is what the toolkit's own
// conv does and is fine at these channel counts; the 7x7 has 49 taps, so it
// stages its input block in shared memory instead - 49 taps x c_in reads per
// output is the case where one-thread-per-output stops being free.

// 3x3, stride 1, ReflectionPad2d(1): out[h][w] is the tap window about it.
extern "C" __global__ void ne_conv3x3_refl(
    const float* __restrict__ x, const float* __restrict__ w,
    const float* __restrict__ b, float* __restrict__ y,
    int cin, int cout, int h, int wd, int has_bias)
{
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    if (ow >= wd || oh >= h) return;
    int plane = h * wd;
    long long base = (long long)blockIdx.z * cin * plane;
    for (int co = 0; co < cout; ++co) {
        float acc = has_bias ? b[co] : 0.0f;
        for (int ci = 0; ci < cin; ++ci) {
            const float* xp = x + base + (long long)ci * plane;
            const float* wp = w + ((long long)co * cin + ci) * 9;
            for (int ky = 0; ky < 3; ++ky) {
                int iy = ne_refl(oh - 1 + ky, h);
                for (int kx = 0; kx < 3; ++kx) {
                    int ix = ne_refl(ow - 1 + kx, wd);
                    acc += wp[ky * 3 + kx] * xp[(long long)iy * wd + ix];
                }
            }
        }
        y[base / cin * cout + (long long)co * plane + (long long)oh * wd + ow] = acc;
    }
}

// The same 3x3 with stride 2 and ReflectionPad2d(1): the downsample. Written
// separately rather than as a flag on the stride-1 kernel because the index
// arithmetic is different enough that sharing it would only hide the padding.
extern "C" __global__ void ne_down3x3_refl(
    const float* __restrict__ x, const float* __restrict__ w,
    const float* __restrict__ b, float* __restrict__ y,
    int cin, int cout, int h, int wd, int has_bias)
{
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int oh_full = (h + 2 - 3) / 2 + 1;
    int ow_full = (wd + 2 - 3) / 2 + 1;
    if (ow >= ow_full || oh >= oh_full) return;
    int plane = h * wd;
    int oplane = oh_full * ow_full;
    long long base = (long long)blockIdx.z * cin * plane;
    int oi = oh * 2 - 1, oj = ow * 2 - 1;
    for (int co = 0; co < cout; ++co) {
        float acc = has_bias ? b[co] : 0.0f;
        for (int ci = 0; ci < cin; ++ci) {
            const float* xp = x + base + (long long)ci * plane;
            const float* wp = w + ((long long)co * cin + ci) * 9;
            for (int ky = 0; ky < 3; ++ky) {
                int iy = ne_refl(oi + ky, h);
                for (int kx = 0; kx < 3; ++kx) {
                    int ix = ne_refl(oj + kx, wd);
                    acc += wp[ky * 3 + kx] * xp[(long long)iy * wd + ix];
                }
            }
        }
        y[(long long)blockIdx.z * cout * oplane + (long long)co * oplane +
          (long long)oh * ow_full + ow] = acc;
    }
}

// 7x7, stride 1, ReflectionPad2d(3), one thread per output, input staged in
// shared memory. The tile is 16x16 outputs and 22x22 inputs per channel, so a
// channel's halo is read from DRAM once per tile instead of 49 times per output.
#define NE_TILE 16
#define NE_HALO (NE_TILE + 6)
extern "C" __global__ void ne_conv7x7_refl(
    const float* __restrict__ x, const float* __restrict__ w,
    const float* __restrict__ b, float* __restrict__ y,
    int cin, int cout, int h, int wd, int has_bias)
{
    __shared__ float tile[NE_HALO * NE_HALO];
    int ox = blockIdx.x * NE_TILE, oy = blockIdx.y * NE_TILE;
    int tx = threadIdx.x, ty = threadIdx.y;
    int plane = h * wd;
    long long nbase = (long long)blockIdx.z * cin * plane;
    for (int co = 0; co < cout; ++co) {
        float acc = has_bias ? b[co] : 0.0f;
        for (int ci = 0; ci < cin; ++ci) {
            const float* xp = x + nbase + (long long)ci * plane;
            for (int i = ty; i < NE_HALO; i += NE_TILE) {
                int iy = ne_refl(oy + i - 3, h);
                for (int j = tx; j < NE_HALO; j += NE_TILE) {
                    int ix = ne_refl(ox + j - 3, wd);
                    tile[i * NE_HALO + j] = xp[(long long)iy * wd + ix];
                }
            }
            __syncthreads();
            const float* wp = w + ((long long)co * cin + ci) * 49;
            if (oy + ty < h && ox + tx < wd) {
                for (int ky = 0; ky < 7; ++ky) {
                    const float* row = tile + (ty + ky) * NE_HALO + tx;
                    const float* wr = wp + ky * 7;
                    for (int kx = 0; kx < 7; ++kx) acc += wr[kx] * row[kx];
                }
            }
            __syncthreads();
        }
        if (oy + ty < h && ox + tx < wd) {
            y[(long long)blockIdx.z * cout * plane + (long long)co * plane +
              (long long)(oy + ty) * wd + (ox + tx)] = acc;
        }
    }
}

// 7x7, stride 2, ReflectionPad2d(3) - the shape the generator has no instance of
// (upstream's first block is stride 1), kept because `CONVENTIONS.md` asks every
// op to be usable at both strides and because the CPU twin is symmetric with it.
// One thread per output: at 7x7 a shared-memory tile would be mostly halo.
extern "C" __global__ void ne_down7x7_refl(
    const float* __restrict__ x, const float* __restrict__ w,
    const float* __restrict__ b, float* __restrict__ y,
    int cin, int cout, int h, int wd, int has_bias)
{
    int oh_full = (h + 6 - 7) / 2 + 1;
    int ow_full = (wd + 6 - 7) / 2 + 1;
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    if (ow >= ow_full || oh >= oh_full) return;
    int plane = h * wd, oplane = oh_full * ow_full;
    long long base = (long long)blockIdx.z * cin * plane;
    int oi = oh * 2 - 3, oj = ow * 2 - 3;
    for (int co = 0; co < cout; ++co) {
        float acc = has_bias ? b[co] : 0.0f;
        for (int ci = 0; ci < cin; ++ci) {
            const float* xp = x + base + (long long)ci * plane;
            const float* wp = w + ((long long)co * cin + ci) * 49;
            for (int ky = 0; ky < 7; ++ky) {
                int iy = ne_refl(oi + ky, h);
                for (int kx = 0; kx < 7; ++kx) {
                    int ix = ne_refl(oj + kx, wd);
                    acc += wp[ky * 7 + kx] * xp[(long long)iy * wd + ix];
                }
            }
        }
        y[(long long)blockIdx.z * cout * oplane + (long long)co * oplane +
          (long long)oh * ow_full + ow] = acc;
    }
}

// ---------------------------------------------------------------------------
// The two normalisations
// ---------------------------------------------------------------------------

// `nn.InstanceNorm2d(...)` with `unbiased == 0` (its own definition: divide by
// H*W) and the UNBIASED form with `unbiased != 0` for the AdaILN/ILN halves.
//
// One block per (sample, channel). The mean and the variance are computed the
// stable way - two passes over the plane, the second on the mean-subtracted
// values - rather than as `E[x^2] - mean^2`, because a large mean with a small
// spread turns the latter into a difference of nearly equal numbers. `eps` is
// added to the VARIANCE before the square root, matching `torch.instance_norm`.
//
// The summation order is fixed (ascending within a thread, then a halving tree
// over 256 lanes) so the CPU twin can reproduce it exactly and a backend
// difference is a bug rather than a reassociation.
extern "C" __global__ void ne_instance_norm(
    const float* __restrict__ x, const float* __restrict__ gamma,
    const float* __restrict__ beta, float* __restrict__ y,
    int c, int hw, int unbiased)
{
    int ch = blockIdx.x;
    int n = blockIdx.y;
    int tid = threadIdx.x;
    const float* xp = x + ((long long)n * c + ch) * hw;
    float* yp = y + ((long long)n * c + ch) * hw;

    __shared__ float red[256];
    float local = 0.0f;
    for (int i = tid; i < hw; i += blockDim.x) local += xp[i];
    red[tid] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    float mean = red[0] / (float)hw;
    // THIS BARRIER IS LOAD-BEARING, and it is the reason the whole kernel is
    // deterministic or is not. Every thread reads `red[0]` above, and the
    // variance pass below writes `red[tid]` into the same array; the halving
    // tree folds `red[0]` on its FIRST step, so at `hw <= blockDim.x` every
    // thread crossing an implicit `__syncthreads()` boundary is the only thing
    // stopping a fast warp's store from landing on `red[0]` before a slow warp
    // has read it - which turns `mean` into a PARTIAL SUM and `var` into the
    // variance about that sum. There is no `__syncthreads()` inside the loop
    // below to close that window, so the read and the write must be separated
    // here. Measured before the fix: a two-run bisect of the op list made this
    // op the first non-deterministic one, at max |a - b| = 3.4 on a 64x64 plan,
    // and the divergence persisted into the output (five GPU runs of one input,
    // five different PNGs, against a bit-stable CPU).

    local = 0.0f;
    for (int i = tid; i < hw; i += blockDim.x) {
        float d = xp[i] - mean;
        local += d * d;
    }
    // ... and here is that barrier, the only thing between the read of `red[0]`
    // above and the store into `red[tid]` below. Every thread reaching it has
    // already performed its read.
    __syncthreads();
    red[tid] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    float var = red[0] / (float)(unbiased ? (hw - 1) : hw);
    float inv = rsqrtf(var + NE_EPS);
    float g = gamma ? gamma[ch] : 1.0f;
    float b = beta ? beta[ch] : 0.0f;
    for (int i = tid; i < hw; i += blockDim.x) yp[i] = (xp[i] - mean) * inv * g + b;
}

// `AdaILN`: one instance half and one layer half, mixed by a per-channel rho.
//
//   inst = (x - mean_hw)  / sqrt(var_hw + eps)        UNBIASED over the plane
//   lay  = (x - mean_chw) / sqrt(var_chw + eps)       UNBIASED over c*h*w
//   out  = (rho * inst + (1 - rho) * lay) * gamma + beta
//
// The layer half's statistics are SCALARS for the whole sample, so they are
// computed once per block (blockIdx.z is the sample, one block covers every
// channel) and the per-channel pass is a second kernel-level loop. Both halves
// use `torch.var`'s default, which is the UNBIASED divisor - the single most
// easily-wrong line in this engine, and the reason `ne_instance_norm` takes its
// divisor as an argument rather than assuming.
//
// The MIX comes first and the affine is applied to it, which is `AdaILN.forward`:
//   out = (rho * inst + (1 - rho) * lay) * gamma + beta
// The other reading - `rho*inst + (1-rho)*lay*gamma + beta`, gamma on the layer
// half - is NOT a rearrangement and must not be substituted: it puts the scale on
// one of the two normalised halves instead of on their mixture, and in the eight
// CAM-driven blocks gamma is a network OUTPUT rather than a parameter near 1, so
// the two expressions differ by a lot wherever it is not.
extern "C" __global__ void ne_adailn(
    const float* __restrict__ x, const float* __restrict__ rho,
    const float* __restrict__ gamma, const float* __restrict__ beta,
    float* __restrict__ y, int c, int hw)
{
    extern __shared__ float smem[];
    float* red = smem;                  // 256 partials
    float* scal = smem + 256;           // [mean_chw, var_chw]
    int n = blockIdx.x;
    int tid = threadIdx.x;
    const float* xp = x + (long long)n * c * hw;
    float* yp = y + (long long)n * c * hw;
    long long total = (long long)c * hw;

    // Layer half's statistics: one pass over the whole sample, then a second on
    // the mean-subtracted values. Ascending within a thread, halving tree after.
    float local = 0.0f;
    for (long long i = tid; i < total; i += blockDim.x) local += xp[i];
    red[tid] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    if (tid == 0) scal[0] = red[0] / (float)total;
    __syncthreads();
    float mean_l = scal[0];
    local = 0.0f;
    for (long long i = tid; i < total; i += blockDim.x) {
        float d = xp[i] - mean_l;
        local += d * d;
    }
    red[tid] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    if (tid == 0) scal[1] = red[0] / (float)(total - 1);
    __syncthreads();
    float mean_lc = scal[0], inv_l = rsqrtf(scal[1] + NE_EPS);

    // Instance half: per channel.
    for (int ch = 0; ch < c; ++ch) {
        const float* cp = xp + (long long)ch * hw;
        float* op = yp + (long long)ch * hw;
        float loc = 0.0f;
        for (int i = tid; i < hw; i += blockDim.x) loc += cp[i];
        red[tid] = loc;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) red[tid] += red[tid + s];
            __syncthreads();
        }
        float mean_i = red[0] / (float)hw;
        loc = 0.0f;
        for (int i = tid; i < hw; i += blockDim.x) {
            float d = cp[i] - mean_i;
            loc += d * d;
        }
        // The same load-bearing barrier as in `ne_instance_norm`, and the same
        // reasoning: at `hw <= blockDim.x` the tree's first step folds `red[0]`,
        // so a warp storing `red[tid]` before another has read `mean_i` turns
        // that mean into a partial sum. This kernel covers EVERY channel in one
        // block, so the window repeats c times per launch rather than once.
        __syncthreads();
        red[tid] = loc;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) red[tid] += red[tid + s];
            __syncthreads();
        }
        float var_i = red[0] / (float)(hw - 1);
        float inv_i = rsqrtf(var_i + NE_EPS);
        float r = rho[ch];
        float g = gamma ? gamma[ch] : 1.0f;
        float b = beta ? beta[ch] : 0.0f;
        for (int i = tid; i < hw; i += blockDim.x) {
            float inst = (cp[i] - mean_i) * inv_i;
            float lay = (cp[i] - mean_lc) * inv_l;
            op[i] = (r * inst + (1.0f - r) * lay) * g + b;
        }
        __syncthreads();
    }
}

// out[c][p] = in[c][p] * gamma[c] + beta[c], whole planes. The ILN's last step
// and the adaILN's, factored out so there is one place where gamma and beta are
// indexed - and so a weight of 1 / bias of 0 is a no-op the caller can express
// by passing nulls.
extern "C" __global__ void ne_channel_affine(
    const float* __restrict__ x, const float* __restrict__ gamma,
    const float* __restrict__ beta, float* __restrict__ y, int c, int hw)
{
    int ch = blockIdx.x;
    const float* xp = x + (long long)ch * hw;
    float* yp = y + (long long)ch * hw;
    float g = gamma ? gamma[ch] : 1.0f;
    float b = beta ? beta[ch] : 0.0f;
    for (int i = threadIdx.x; i < hw; i += blockDim.x) yp[i] = xp[i] * g + b;
}

// ---------------------------------------------------------------------------
// The CAM's pooling and scaling
// ---------------------------------------------------------------------------

// Per-channel global MAXIMUM over an NCHW plane, one block per channel. The
// twin of `lg_channel_mean` for the CAM's second weight. Ties and NaN are not
// specified beyond `fmaxf`'s own rule (which ignores a NaN operand); the graph
// never feeds it a NaN, and the check is in the fixture rather than here.
extern "C" __global__ void ne_channel_max(
    const float* __restrict__ x, float* __restrict__ out, int c, int hw)
{
    extern __shared__ float red[];
    int ch = blockIdx.x;
    int tid = threadIdx.x;
    const float* xp = x + (long long)ch * hw;
    float m = -CUDART_INF_F;
    for (int i = tid; i < hw; i += blockDim.x) m = fmaxf(m, xp[i]);
    red[tid] = m;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]);
        __syncthreads();
    }
    if (tid == 0) out[ch] = red[0];
}

// out[c][p] = in[c][p] * s[c] - a per-channel multiply with no add. `lg_mul` is
// plane-times-plane and `lg_channel_affine` always adds, so the CAM's
// `x * gap_weight` needs this on its own.
//
// `s` is a WEIGHT, and the arithmetic is `x * gap_weight.unsqueeze(2).unsqueeze(3)`
// from `networks.py`: `gap_fc.weight` is a `[1][C]` Linear, so its flat element `c`
// IS channel `c`'s scale on the full map. One multiply per element, one block per
// output channel.
//
// There is deliberately no flag selecting `s[0]`: the CAM's shape is two scaled
// FULL maps, so no caller multiplies a `[c][1][1]` pooled row by a single scalar,
// and a flag no launch exercises would be a second code path nothing checks.
extern "C" __global__ void ne_channel_mul(
    const float* __restrict__ x, const float* __restrict__ s,
    float* __restrict__ y, int c, int hw)
{
    int ch = blockIdx.x;
    if (ch >= c) return;
    float sv = s[ch];
    const float* xp = x + (long long)ch * hw;
    float* yp = y + (long long)ch * hw;
    for (int i = threadIdx.x; i < hw; i += blockDim.x) yp[i] = xp[i] * sv;
}

// y = tanh(x + skip), whole planes. The generator's output stage: the residual
// with the INPUT image (not the upsampled feature) and the final squash. Fused
// because splitting it costs a plane of the activation size to hold the sum.
extern "C" __global__ void ne_tanh_add(
    const float* __restrict__ x, const float* __restrict__ skip,
    float* __restrict__ y, long long n)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = tanhf(x[i] + skip[i]);
}

// ---------------------------------------------------------------------------
// EXPERIMENTAL: the zero-padded 3x3 with each input value fetched once.
// ---------------------------------------------------------------------------
//
// NOT LAUNCHED BY ANY OP. It exists to answer a measured question - how much
// headroom is left in the 3x3s, which are 75% of this device's time - and it is
// kept beside `tests/reference.rs`'s `gpu_3x3_kernel_throughput` rather than in
// `exec_gpu.rs` for a reason worth stating: its accumulation order is not the
// twin's, so it could not ship without a tolerance argument.
//
// Every other conv here - the CPU scalar and vector paths, the toolkit's
// `lg_conv3x3s1p1` - accumulates `ci` outermost, then `ky`, then `kx`, and that
// ORDER is what the two backends are compared on. This kernel keeps the same sum
// and the same relative order of the terms (ascending `ci`, then `ky`, then `kx`)
// but the LOADS are hoisted: each input value is read from DRAM once per thread
// and reused for every tap that needs it, which is the same argument the CPU
// kernel makes with row blocking and the toolkit's 7x7 makes with a shared-memory
// tile. Hoisting is only possible by walking taps before channels, so the two
// orders are not literally the same instruction sequence and they round
// differently in the last bit. That is why this is an experiment.
//
// One thread per (oc, y, 8 outputs). Three row segments of 10 floats are loaded
// per input channel - the eight outputs' window plus a halo column each side,
// three rows - and reused for all nine taps: 30 loads per thread feed 72
// multiply-adds per input channel, where the toolkit kernel issues 72 loads for
// the same work. The zero border is folded in by loading zeros for the lanes that
// fall outside the plane, so there is no per-tap branch.
#define NEX_E 8
#define NEX_X8(x) ((x) + (x) + (x) + (x))
extern "C" __global__ void nex_conv3x3s1p1_x8(
    const float* __restrict__ x, const float* __restrict__ w,
    const float* __restrict__ b, float* __restrict__ y,
    int cin, int cout, int h, int wd)
{
    // `oy`, not `y`: `y` is the kernel's OUTPUT POINTER, and naming a row `y`
    // here shadows it - which nvcc reports at the pointer arithmetic, two errors
    // away from the cause.
    const int x0 = (blockIdx.x * blockDim.x + threadIdx.x) * NEX_E;
    const int oy = blockIdx.y;
    const int oc = blockIdx.z;
    if (x0 >= wd || oy >= h) return;
    const size_t plane = (size_t)h * wd;
    float acc[NEX_E];
    const float bv = b ? b[oc] : 0.0f;
    for (int j = 0; j < NEX_E; ++j) acc[j] = bv;

    for (int ci = 0; ci < cin; ++ci) {
        const float* xp = x + (size_t)ci * plane;
        // Three rows x (8 outputs + 2 halo). Loaded once per input channel and
        // reused by all nine taps. `r0`/`r1`/`r2` are the `ky = 0/1/2` rows; the
        // indices inside them are compile-time constants, which is what keeps
        // this in registers rather than in local memory.
        float r0[NEX_E + 2], r1[NEX_E + 2], r2[NEX_E + 2];
        #pragma unroll
        for (int t = 0; t < NEX_E + 2; ++t) {
            const int ix = x0 - 1 + t;
            const bool in = (ix >= 0 && ix < wd);
            const int iy0 = oy - 1, iy1 = oy, iy2 = oy + 1;
            r0[t] = (in && iy0 >= 0)      ? xp[(size_t)iy0 * wd + ix] : 0.0f;
            r1[t] = (in)                  ? xp[(size_t)iy1 * wd + ix] : 0.0f;
            r2[t] = (in && iy2 < h)       ? xp[(size_t)iy2 * wd + ix] : 0.0f;
        }
        const float* rows[3] = { r0, r1, r2 };
        // `ky`, then `kx`, then the eight outputs: the tap order the twin uses,
        // with the row already in registers.
        for (int ky = 0; ky < 3; ++ky) {
            const float* row = rows[ky];
            const float* wp = w + ((size_t)oc * cin + ci) * 9 + ky * 3;
            for (int kx = 0; kx < 3; ++kx) {
                const float wv = wp[kx];
                #pragma unroll
                for (int j = 0; j < NEX_E; ++j) acc[j] += wv * row[j + kx];
            }
        }
    }
    float* out = y + (size_t)oc * plane + (size_t)oy * wd + x0;
    const int left = wd - x0;
    if (left >= NEX_E) {
        #pragma unroll
        for (int j = 0; j < NEX_E; ++j) out[j] = acc[j];
    } else {
        for (int j = 0; j < left; ++j) out[j] = acc[j];
    }
}

// ---------------------------------------------------------------------------
// EXPERIMENTAL, like `nex_conv3x3s1p1_x8`: no op launches this one. It exists
// because a disassembly said so, and it is the structure that disassembly
// justified - nothing more.
//
// Measured on this card (`cuobjdump -res-usage`, the kernel the profiler named):
// cuBLAS runs `sgemm_128x128x8_NN_vec` for the im2col GEMM at REG:128 and
// SHARED:16912, reading its staged operands with 128-bit shared loads (48 of its
// 63 LDS are LDS.U.128). The toolkit's `lg_f32_gemm_tiled` is REG:54 SHARED:512:
// one single-buffered 128-float activation strip, refilled and re-barriered EVERY
// k-step of 4, with the WEIGHT rows read straight from GLOBAL once per column
// slot and the row addresses recomputed per iteration.
//
// This kernel changes exactly two of those things and keeps the arithmetic order
// k-ascending:
//
//   * a 128 (rows of W) x 64 (columns of x) tile per 256-thread block, laid out
//     16x16 threads with 8 rows x 4 columns per thread = 32 accumulators;
//   * BOTH operands staged in shared as float4 in a [index][k] layout, so one
//     global load feeds the whole block - including the weights, which is the
//     change the toolkit kernel most obviously lacks;
//   * DOUBLE buffered: the k+1 tile is fetched while the k tile is consumed, and
//     the loop carries ONE __syncthreads() per k-step of 8 rather than two per
//     step of 4;
//   * every address expressible without k is hoisted above the loop.
//
// Same contract as `lg_f32_gemm_tiled`: y[col][row] = sum_k W[row][k] * x[col][k],
// W is [ne1][ne0], x is [ncols][ne0], grid = (ceil(ne1/128), ceil(ncols/64)).
// ne0 must be a multiple of 8 (the toolkit kernel requires only 4), so the two
// are NOT drop-in interchangeable.
//
// The accumulation order is k-ascending, but the 8-wide k-step and the shared
// staging mean the sum is taken in a different association from the toolkit's
// 4-wide version, so this is NOT bit-identical to it - which is why
// `tests/reference.rs` compares it against the same triple loop with a TOLERANCE
// rather than with exact equality.
#define NEXDB_KT    8
#define NEXDB_ROWS  128
#define NEXDB_COLS  64
#define NEXDB_RPT   8    // rows of W per thread
#define NEXDB_CPT   4    // columns of x per thread

extern "C" __global__ void nex_f32_gemm_db(
    const float *__restrict__ w, const float *__restrict__ x, float *__restrict__ y,
    int ne0, int ne1, int ncols)
{
    // [buffer][index][k]: k is the FAST axis, so every row of KT floats is a
    // whole number of float4s and every staged access below is 16-byte aligned.
    __shared__ float sw[2][NEXDB_ROWS][NEXDB_KT];   // 2 * ROWS * KT * 4 bytes
    __shared__ float sx[2][NEXDB_COLS][NEXDB_KT];   // 2 * COLS * KT * 4 bytes

    const int tid = threadIdx.x;
    const int tx  = tid & 15;                     // 0..15 across columns of x
    const int ty  = tid >> 4;                     // 0..15 down the rows of W
    // The tile base is a BLOCK-level quantity: every thread stages and reads the
    // same 128x64 window. The per-thread offsets (`ty * NEXDB_RPT`, `tx * NEXDB_CPT`)
    // belong to the accumulators only. Mixing the two into one index is the bug the
    // first version had, and it is silent: the staging just fills the wrong rows.
    const int rb  = blockIdx.x * NEXDB_ROWS;      // tile base, rows of W
    const int cb  = blockIdx.y * NEXDB_COLS;      // tile base, columns of x
    const int r0  = rb + ty * NEXDB_RPT;          // this thread's first row
    const int c0  = cb + tx * NEXDB_CPT;          // this thread's first column
    const int nk  = ne0 / NEXDB_KT;

    float acc[NEXDB_RPT][NEXDB_CPT];
    #pragma unroll
    for (int i = 0; i < NEXDB_RPT; ++i)
        #pragma unroll
        for (int j = 0; j < NEXDB_CPT; ++j) acc[i][j] = 0.0f;

    // Hoisted once: the W rows and x columns this thread owns never move, and
    // neither does the base of each k-tile within them.
    const float *wr[NEXDB_RPT];
    #pragma unroll
    for (int i = 0; i < NEXDB_RPT; ++i)
        wr[i] = w + (size_t)(r0 + i) * ne0;
    const float *xc[NEXDB_CPT];
    #pragma unroll
    for (int j = 0; j < NEXDB_CPT; ++j)
        xc[j] = x + (size_t)(c0 + j) * ne0;

    // STAGING. One flat index space of NEXDB_KT/4 float4 units per index, so the
    // unit count is ROWS*(KT/4) + COLS*(KT/4) and unit `u = pass * 256 + tid` covers
    // the W rows first and the x columns after them. The pass count is a CEILING
    // division: with KT = 8 there are 256 + 128 = 384 units, and plain `384 / 256`
    // is one pass, so the x half would never run - max |d| 2.656 with no other
    // symptom. The same shape of mistake is an `if (tid < 256) { W } else { x }`
    // form, where `tid` never reaches 256.
    //
    // The constant and this macro must change together, and neither may assume the
    // other's value: with KT = 16 a two-float4 form stages only k = 0..7 of each
    // tile and leaves the rest uninitialised shared memory (2.656, silent), and a
    // four-quarter form left in place while KT = 8 writes at offsets 0/4/8/12 of an
    // 8-float row and runs off the array (CUDA_ERROR_ILLEGAL_ADDRESS).
    //
    // NOTE for the next editor: comments cannot live INSIDE this macro body (the
    // backslash-joined lines swallow them, and nvcc then reports the error at the
    // `#define` with "'#' is not followed by a macro parameter"), and `#pragma`
    // must be written `_Pragma(...)` for the same reason.
    // QUARTERS PER ROW: a row of KT floats is KT/4 float4s, so the flat unit index
    // MUST be divided by this to get the row and taken modulo it to get the quarter.
    // Hard-coding `u >> 2` and `(u & 3) << 2` instead is the KT = 16 mapping and
    // goes out of bounds the moment KT is 8 (rows 0..255 into a 128-row array,
    // quarters 0/4/8/12 into an 8-float row).
    #define NEXDB_QPR (NEXDB_KT / 4)
    #define NEXDB_W_UNITS (NEXDB_ROWS * NEXDB_QPR)
    #define NEXDB_ALL_UNITS (NEXDB_W_UNITS + NEXDB_COLS * NEXDB_QPR)
    #define NEXDB_STAGE(buf, koff)                                            \
    {                                                                         \
        _Pragma("unroll")                                                     \
        for (int pass = 0; pass < (NEXDB_ALL_UNITS + 255) / 256; ++pass) {     \
            const int u = pass * 256 + tid;                                   \
            if (u >= NEXDB_ALL_UNITS) break;                                  \
            if (u < NEXDB_W_UNITS) {                                          \
                const int rr = u / NEXDB_QPR, q = (u % NEXDB_QPR) * 4;                      \
                float4 v = make_float4(0.f, 0.f, 0.f, 0.f);                   \
                if (rb + rr < ne1)                                            \
                    v = *reinterpret_cast<const float4 *>(                    \
                        (w + (size_t)(rb + rr) * ne0) + (koff) + q);          \
                *(float4 *)&sw[buf][rr][q] = v;                               \
            } else {                                                          \
                const int ux = u - NEXDB_W_UNITS;                             \
                const int cc = ux / NEXDB_QPR, q = (ux % NEXDB_QPR) * 4;                    \
                float4 v = make_float4(0.f, 0.f, 0.f, 0.f);                   \
                if (cb + cc < ncols)                                          \
                    v = *reinterpret_cast<const float4 *>(                    \
                        (x + (size_t)(cb + cc) * ne0) + (koff) + q);          \
                *(float4 *)&sx[buf][cc][q] = v;                               \
            }                                                                 \
        }                                                                     \
    }

    NEXDB_STAGE(0, 0);
    __syncthreads();

    for (int kt = 0; kt < nk; ++kt) {
        const int cur = kt & 1, nxt = cur ^ 1;
        if (kt + 1 < nk) NEXDB_STAGE(nxt, (kt + 1) * NEXDB_KT);

        // Consume `cur`, k OUTERMOST and each staged operand read ONCE per k.
        //
        // The obvious nesting (rows outer, then k, then the thread's columns) re-reads
        // `sx[cur][tx*CPT + j][k]` for every one of the RPT rows: 8 x 8 x 4 = 256
        // shared reads per k-tile for 256 FMAs, i.e. one shared read per FMA. That is
        // what the SASS histogram showed as "2.5x the instructions per unit of work
        // of cuBLAS", and the fix is a reordering, not a new algorithm: with k
        // outermost, the thread's CPT x values for that k are loaded once into
        // registers (CPT reads) and each W value once (RPT reads), so a k-step costs
        // RPT + CPT = 12 shared reads against RPT*CPT = 32 FMAs - 96 reads per k-tile
        // where there were 256. The accumulation order stays k-ascending, so every
        // index still receives the same sequence of terms as before, and the
        // `acc[i][j]` register block is untouched (no occupancy cost).
        #pragma unroll
        for (int k = 0; k < NEXDB_KT; ++k) {
            float xv[NEXDB_CPT];
            #pragma unroll
            for (int j = 0; j < NEXDB_CPT; ++j) xv[j] = sx[cur][tx * NEXDB_CPT + j][k];
            #pragma unroll
            for (int i = 0; i < NEXDB_RPT; ++i) {
                const float a = sw[cur][ty * NEXDB_RPT + i][k];
                #pragma unroll
                for (int j = 0; j < NEXDB_CPT; ++j) acc[i][j] = fmaf(a, xv[j], acc[i][j]);
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < NEXDB_RPT; ++i) {
        const int r = r0 + i;
        if (r >= ne1) continue;
        #pragma unroll
        for (int j = 0; j < NEXDB_CPT; ++j) {
            const int c = c0 + j;
            if (c >= ncols) continue;
            y[(size_t)c * ne1 + r] = acc[i][j];
        }
    }
}
