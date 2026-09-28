#!/usr/bin/env python3
"""A torch-free numpy transcription of night-enhancement's generator.

Upstream is https://github.com/jinyeying/night-enhancement - `networks.py`'s
`ResnetGenerator`, and only that: the released inference path is one
feed-forward generator, `ENHANCENET.test` resizes a picture to 512x512, runs the
generator, and resizes the result back. `net/` is a vendored Deep-Image-Prior
library, `disc/` is training-only, and the Matlab stages are a different stage of
the paper's decomposition; none of them are part of what this engine reproduces.

Why this file exists
--------------------
Every engine in the family checks its two backends against a transcription of the
upstream model. This one is written to be checkable by reading - one function per
upstream class, in the order `networks.py` defines them, with the upstream
expression on the same line wherever a merge or a reassociation was possible, and
with `LOG` recording every decision that could have gone the other way.

Reading it is necessary and not sufficient. A transcription is only an oracle
once something independent has confirmed it, and the fixtures for the Rust test
are written BY this file, so a mistake here is invisible to every comparison that
uses them - the engine and the fixtures agree, because they are the same mistake.
The confirmation is

    /usr/local/bin/python3.11 tools/check_upstream.py \
        --networks /path/to/night-enhancement/networks.py \
        --weights ../models/LOL_params_0900000.pt

which loads the real `networks.py` next to this module, feeds both the same
input, and reports the worst difference per tensor and per stage - the first
stage that diverges is the whole diagnosis, and reading is exactly what cannot be
relied on for it. It needs only the upstream source and the released `.pt`, both
of which are already on this machine. Run it whenever this file is edited, and
whenever a checkpoint is converted (`--ref` compares the converted file against
the `.pt` tensor by tensor, which is a check the per-stage table cannot make).

    python3 tools/reference.py --check                     # run the .safetensors
    python3 tools/reference.py --fixture ../tests/data/tiny
    python3 tools/reference.py --image in.png --out out.png

Shape rule: both spatial sizes must be a multiple of 4
-----------------------------------------------------
The generator is exactly shape-preserving only when each spatial dimension is a
multiple of 4, and 4 is a hard requirement rather than a preference. Two strided
convs with `ReflectionPad2d(1)` and k = 3, stride 2 map n -> floor((n + 2 - 3)/2) + 1
= floor((n - 1)/2) + 1, and two exact-2x upsamples multiply by 4, so the round trip
returns n only for even n that stay even through both stages. 33 -> 36 and
63 -> 64, measured; every multiple of 4 up to 128 round-trips exactly. Upstream
never meets the case because its pipeline resizes to 512x512 first, which is why
`ENHANCENET.test` looks like a size convention rather than a constraint. The
engine therefore resizes to 512 by default like upstream, and its native-size
path pads each axis up to a multiple of 4 and crops back.

The one thing that must NOT be simplified
-----------------------------------------
`torch.var` is UNBIASED (divides by n - 1) and `nn.InstanceNorm2d` is BIASED
(divides by n); the generator uses both, in different places, and on a 512x512
map they differ in the 4th significant figure of the normalised value:

    DownBlock / ResnetBlock   nn.InstanceNorm2d          var over H*W,   / (H*W)
    adaILN, ILN (instance)    torch.var(x, [2, 3])       var over H*W,   / (H*W - 1)
    adaILN, ILN (layer)       torch.var(x, [1, 2, 3])    var over C*H*W, / (C*H*W - 1)

`torch.var` also computes the variance of the mean-subtracted tensor, while
InstanceNorm2d is documented in terms of a biased estimate of the same thing;
both are `mean(x^2) - mean(x)^2` up to the divisor, and the divisor is the whole
point. Conflating the two is the mistake this file exists to make impossible.
"""
import argparse
import json
import os
import struct
import sys

import numpy as np

# Every "this could have been written the other way" decision, in order.
LOG = [
    "conv2d output size is computed from the PADDED height when reflect=True; "
    "computing it from the input height silently loses k - 1 samples per plane",
    "the strided convs need no asymmetric padding: n -> floor((n-1)/2) + 1 at "
    "stride 2 with pad 1, so an input that is a multiple of 4 round-trips",
    "instance_norm uses the mean-subtracted form (x - mu) / sqrt(var + eps), not "
    "sqrt(E[x^2] - mu^2 + eps)",
    "EVERY convolution is reflection-padded, including the eight 3x3s inside the "
    "residual blocks; `nn.Conv2d(padding=0)` is used throughout networks.py",
    "the CAM concatenates x*gap_fc.weight and x*gmp_fc.weight over the FULL map, "
    "so the 1x1 conv's input is [2C][H][W] and its output [C][H][W]",
    "the decoder starts from the CAM's 1x1 output, not from the resnet body",
    "FC ends in a ReLU: [Linear, ReLU, Linear, ReLU], not [Linear, ReLU, Linear]",
    "adaILN mixes FIRST and applies gamma/beta to the MIXED result",
    "DownBlock has InstanceNorm2d + ReLU after the first 7x7 conv",
]


# ---------------------------------------------------------------------------
# safetensors (read side)
# ---------------------------------------------------------------------------


def read_safetensors(path):
    """{name: [..] f32} plus the `__metadata__` map."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        blob = f.read()
    meta = header.pop("__metadata__", {})
    dtypes = {"F32": "<f4", "F16": "<f2", "F64": "<f8"}
    out = {}
    for name, info in header.items():
        dt = dtypes.get(info["dtype"])
        if dt is None:
            raise SystemExit(f"{name}: unsupported dtype {info['dtype']}")
        a, b = info["data_offsets"]
        out[name] = np.ascontiguousarray(
            np.frombuffer(blob[a:b], dtype=dt).reshape(info["shape"]), dtype=np.float32)
    return out, meta


# ---------------------------------------------------------------------------
# primitives
# ---------------------------------------------------------------------------


def reflect_pad2d(x, pad):
    """`nn.ReflectionPad2d(pad)` on the last two axes.

    Reflection EXCLUDES the edge sample, so index -1 of the padded axis is the
    second sample of the input, not the first: `np.pad(mode="reflect")` is the
    same rule, and `mode="symmetric"` is not (it repeats the edge).

    EVERY convolution in `ResnetGenerator` is reflection-padded, including the
    3x3s inside the residual blocks: upstream's `ResnetBlock` and
    `ResnetAdaILNBlock` each open their 3x3 with `nn.ReflectionPad2d(1)`, and
    `nn.Conv2d(padding=0)` is used throughout the file. The distinction is a
    border-only difference that leaves a picture looking entirely plausible, so
    nothing but a comparison against upstream will find it.
    """
    if pad == 0:
        return x
    return np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="reflect")


def conv2d(x, w, b=None, stride=1, pad=0, reflect=False):
    """NCHW convolution, the direct form.

    `w` is torch's `[c_out][c_in][kh][kw]`. The accumulation order is ky, kx, ci -
    the order `lg_conv3x3s1p1` and `lg_conv_kxk` document - so a mismatch against
    the Rust is a numeric difference and not a reordering of the terms.

    `reflect=True` applies `ReflectionPad2d(pad)` and then a padding-free conv,
    which is how upstream spells the three reflect-padded convolutions.
    """
    n, c_in, h, wd = x.shape
    c_out, c_in_w, kh, kw = w.shape
    assert c_in == c_in_w, (x.shape, w.shape)
    if reflect:
        x = reflect_pad2d(x, pad)
        h, wd = x.shape[2], x.shape[3]
        pad = 0
    oh = (h + 2 * pad - kh) // stride + 1
    ow = (wd + 2 * pad - kw) // stride + 1
    xp = x if pad == 0 else np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    acc = np.zeros((n, c_out, oh, ow), dtype=np.float32)
    for ky in range(kh):
        for kx in range(kw):
            patch = xp[:, :, ky:ky + oh * stride:stride, kx:kx + ow * stride:stride]
            acc += np.einsum("oi,nip->nop", w[:, :, ky, kx],
                             patch.reshape(n, c_in, -1), optimize=True
                             ).reshape(n, c_out, oh, ow)
    if b is not None:
        acc += b.reshape(1, -1, 1, 1)
    return acc


def instance_norm_mean_var(x, unbiased):
    """The two variance conventions the generator mixes, and which is which."""
    n = x.shape[2] * x.shape[3]
    mu = x.mean(axis=(2, 3), keepdims=True)
    var = ((x - mu) ** 2).sum(axis=(2, 3), keepdims=True) / (n - 1 if unbiased else n)
    return mu, var


def instance_norm(x, unbiased=False, eps=1e-5):
    """`nn.InstanceNorm2d` (affine=False, track_running_stats=False).

    The stable form of `torch.instance_norm`: subtract the mean, then divide by
    `sqrt(var + eps)` computed from the mean-subtracted tensor. Not
    `sqrt(E[x^2] - mean^2 + eps)`, which is the same number written in a way that
    a large mean and a small spread turn into a difference of nearly equal terms.
    """
    mu, var = instance_norm_mean_var(x, unbiased)
    return (x - mu) / np.sqrt(var + eps)


def layer_norm_chw(x, unbiased=True, eps=1e-5):
    """`torch.var(x, dim=[1, 2, 3], unbiased=True)` - the ILN half's norm.

    Over CHANNELS and space together, not over the channels at one position, so
    this is not a channel-axis LayerNorm: the mean and the variance are scalars
    for the whole sample. Separate from `instance_norm` rather than a flag,
    because the two differ in the axis as well as the divisor.
    """
    n = x.shape[1] * x.shape[2] * x.shape[3]
    mu = x.mean(axis=(1, 2, 3), keepdims=True)
    var = ((x - mu) ** 2).sum(axis=(1, 2, 3), keepdims=True) / (n - 1 if unbiased else n)
    return (x - mu) / np.sqrt(var + eps)


def upsample2x_nearest(x):
    """`nn.Upsample(scale_factor=2, mode="nearest")`.

    Output i takes input i // 2, which is what `lg_upsample2x_nearest` contracts
    to as well - so no new toolkit kernel is needed for this step, and the
    arithmetic is exactly reproducible rather than merely close.
    """
    return np.repeat(np.repeat(x, 2, axis=2), 2, axis=3)


def adaptive_avg_pool_1x1(x):
    """`nn.AdaptiveAvgPool2d(1)`: the mean over each channel's whole plane."""
    return x.mean(axis=(2, 3), keepdims=True)


def linear(x, w, b=None):
    """`nn.Linear`: y = x @ w.T (+ b), with w stored [out][in]."""
    y = x @ w.T
    return y if b is None else y + b


# ---------------------------------------------------------------------------
# the generator, class by class, in networks.py's order
# ---------------------------------------------------------------------------


def resnet_block(x, w, prefix, unbiased_instance):
    """`ResnetBlock`: pad, conv3x3, IN, ReLU, pad, conv3x3, IN, plus the input.

    The two 3x3s are REFLECT-padded, like the two 7x7s and the stride-2 stages.
    Zero padding here would be a border-only difference, but not a small one: it
    survives a visual check and shows up only against the reference.
    """
    out = conv2d(x, w[f"{prefix}.conv_block.1.weight"], pad=1, reflect=True)
    out = instance_norm(out, unbiased=unbiased_instance)
    out = np.maximum(out, 0.0)
    out = conv2d(out, w[f"{prefix}.conv_block.5.weight"], pad=1, reflect=True)
    out = instance_norm(out, unbiased=unbiased_instance)
    return x + out


def adailn(x, w, prefix, eps=1e-5):
    """`AdaILN` with its own `rho` and no affine.

    Both halves use `torch.var`'s default, i.e. UNBIASED, over `[2, 3]` and
    `[1, 2, 3]` respectively - not `InstanceNorm2d`. The `rho` is per channel.

    Unused by `Generator.forward`, which reaches the same expression through
    `_adailn_gb` (upstream's `AdaILN.forward` always has gamma and beta - they are
    `Parameter`s filled with 1 and 0, not optional). Kept because upstream's class
    is a real one and a reader looking for it should find it here.
    """
    rho = w[f"{prefix}.rho"]
    return _adailn_gb(x, rho, 1.0, 0.0, eps)


def iln(x, w, prefix, eps=1e-5):
    """`ILN`: the same two normals with a learned gamma/beta.

    This IS `AdaILN` - upstream's `ILN.forward` assigns `self.rho`, `self.gamma`
    and `self.beta` into the AdaILN it owns and calls it. Kept as its own function
    only so the call site reads like upstream's.
    """
    return _adailn_gb(x, w[f"{prefix}.rho"], w[f"{prefix}.gamma"], w[f"{prefix}.beta"], eps)


class Generator:
    """`ResnetGenerator`, the released inference path.

    `n_res` is not a weight shape - LOL is 4 and delight-effects is 6 - so it comes
    from the file's metadata, which the converter wrote after counting the
    `conv_block` modules.
    """

    def __init__(self, w, meta):
        self.w = w
        self.meta = meta
        self.n_res = int(meta["n_res"])
        self.ngf = int(meta["ngf"])

    def forward(self, x, trace=None):
        """x: [n][3][h][w], float32, already scaled to [-1, 1].

        `trace`, when a dict is passed, receives the intermediate tensors at the
        boundaries upstream's own `forward` has - keyed by the NAME of the
        upstream submodule that produces them (`DownBlock`, `conv1x1`, `FC`,
        `gamma`, `beta`, `UpBlock1_<i>`, `UpBlock2`, `output`). It exists for
        `tools/check_upstream.py`, which runs both implementations with hooks and
        reports the worst difference PER STAGE: when two forwards disagree, the
        first stage that diverges is the whole diagnosis - and it is the thing a
        reading of the two sources cannot be trusted to give.
        """
        w = self.w
        # `img_size` is the resolution the model was trained at, not a
        # requirement: every layer is fully convolutional, so any size runs and
        # the 512x512 in the paper is what the weights were fitted to rather
        # than what they need. The CLI still resizes by default, for parity with
        # upstream's pipeline; --native skips it.
        # `DownBlock[0..4]` is `ReflectionPad2d(3)`, the 7x7 conv, then
        # `nn.InstanceNorm2d(ngf)` and `nn.ReLU(True)`. The norm and the ReLU
        # after the 7x7 were MISSING from this file until it was checked against
        # upstream's `networks.py`: a forward that skips them leaves the first
        # three stages of the encoder unnormalised, and everything downstream
        # sees a different statistic. Both are BIASED (InstanceNorm2d's H*W
        # divisor, not torch.var's H*W - 1).
        h = conv2d(x, w["DownBlock.1.weight"], pad=3, reflect=True)
        h = instance_norm(h)
        h = np.maximum(h, 0.0)
        d1 = conv2d(h, w["DownBlock.5.weight"], stride=2, pad=1, reflect=True)
        d1 = instance_norm(d1)
        d1 = np.maximum(d1, 0.0)
        d2 = conv2d(d1, w["DownBlock.9.weight"], stride=2, pad=1, reflect=True)
        d2 = instance_norm(d2)
        d2 = np.maximum(d2, 0.0)

        body = d2
        for i in range(self.n_res):
            body = resnet_block(body, w, f"DownBlock.{12 + i}", unbiased_instance=False)
        if trace is not None:
            trace["DownBlock"] = body

        # The CAM. Upstream multiplies the FEATURE MAP by each pooled weight and
        # concatenates those two scaled maps
        # (`x * gap_weight.unsqueeze(2).unsqueeze(3)`), so the 1x1 conv sees a
        # `[2C][H][W]` tensor and returns a `[C][H][W]` one.
        #
        # Concatenating the two POOLED `[C][1][1]` vectors instead would be a
        # different function of a different rank: it would make `feat` a
        # `[C][1][1]` tensor, and that is not cosmetic - it is what would force
        # the DECODER below onto the resnet body, because a 1x1 map cannot be
        # reflection-padded by 1 and convolved. Two symptoms, one cause, and the
        # more serious of the two: the pooled form discards every spatial mode of
        # the map the decoder is supposed to be conditioned on.
        #
        # `gap_fc` and `gmp_fc` are used TWICE, and upstream does the same: once
        # as a Linear over the pooled vector to produce the (unused at inference)
        # attention logits, and once as a per-channel scale on the full map.
        gap = adaptive_avg_pool_1x1(body)
        gmp = body.max(axis=(2, 3), keepdims=True)
        gap_logit = linear(gap.reshape(1, -1), w["gap_fc.weight"])
        gmp_logit = linear(gmp.reshape(1, -1), w["gmp_fc.weight"])
        scaled_gap = body * w["gap_fc.weight"].reshape(1, -1, 1, 1)
        scaled_gmp = body * w["gmp_fc.weight"].reshape(1, -1, 1, 1)
        c_logit = np.concatenate((gap_logit, gmp_logit), axis=1)

        feat = np.maximum(
            conv2d(np.concatenate((scaled_gap, scaled_gmp), axis=1),
                   w["conv1x1.weight"], w["conv1x1.bias"]), 0.0)
        if trace is not None:
            trace["conv1x1"] = feat
        pooled = adaptive_avg_pool_1x1(feat).reshape(1, -1)
        pooled = np.maximum(linear(pooled, w["FC.0.weight"]), 0.0)
        pooled = linear(pooled, w["FC.2.weight"])
        # `FC` ENDS in a ReLU: `FC = [Linear, ReLU, Linear, ReLU]`. The affine
        # below is what the four adaILN blocks and both UpBlock2 stages scale and
        # shift by, so dropping the last nonlinearity would not be a detail - a
        # negative `pooled` would go through, and gamma would carry the opposite
        # sign of the function upstream computes.
        pooled = np.maximum(pooled, 0.0)
        if trace is not None:
            trace["FC"] = pooled
        gamma = linear(pooled, w["gamma.weight"]).reshape(1, -1, 1, 1)
        beta = linear(pooled, w["beta.weight"]).reshape(1, -1, 1, 1)
        if trace is not None:
            trace["gamma"], trace["beta"] = gamma, beta
        del c_logit

        # The decoder starts from the CAM's OUTPUT, not from the resnet body.
        x_up = feat
        for i in range(self.n_res):
            prefix = f"UpBlock1_{i + 1}"
            out = conv2d(x_up, w[f"{prefix}.conv1.weight"], pad=1, reflect=True)
            rho = w[f"{prefix}.norm1.rho"]
            # The adaILN here takes the CAM's gamma/beta, which is what makes the
            # four blocks adaptive rather than plain.
            out = _adailn_gb(out, rho, gamma, beta)
            out = np.maximum(out, 0.0)
            out = conv2d(out, w[f"{prefix}.conv2.weight"], pad=1, reflect=True)
            out = _adailn_gb(out, w[f"{prefix}.norm2.rho"], gamma, beta)
            x_up = x_up + out
            if trace is not None:
                trace[prefix] = x_up

        up = upsample2x_nearest(x_up)
        up = conv2d(up, w["UpBlock2.2.weight"], pad=1, reflect=True)
        up = _iln_gb(up, w["UpBlock2.3.rho"], w["UpBlock2.3.gamma"], w["UpBlock2.3.beta"])
        up = np.maximum(up, 0.0)
        up = upsample2x_nearest(up)
        up = conv2d(up, w["UpBlock2.7.weight"], pad=1, reflect=True)
        up = _iln_gb(up, w["UpBlock2.8.rho"], w["UpBlock2.8.gamma"], w["UpBlock2.8.beta"])
        up = np.maximum(up, 0.0)
        out = conv2d(up, w["UpBlock2.11.weight"], pad=3, reflect=True)
        if trace is not None:
            trace["UpBlock2"] = out
            trace["output"] = np.tanh(out + x)
        return np.tanh(out + x)


def _adailn_gb(x, rho, gamma, beta, eps=1e-5):
    """AdaILN with an EXTERNAL gamma/beta - the ResnetAdaILNBlock's form."""
    mu_i, var_i = instance_norm_mean_var(x, unbiased=True)
    n = x.shape[1] * x.shape[2] * x.shape[3]
    mu_l = x.mean(axis=(1, 2, 3), keepdims=True)
    var_l = ((x - mu_l) ** 2).sum(axis=(1, 2, 3), keepdims=True) / (n - 1)
    out_i = (x - mu_i) / np.sqrt(var_i + eps)
    out_l = (x - mu_l) / np.sqrt(var_l + eps)
    # The MIX comes first and the affine is applied to the MIXED result:
    # `AdaILN.forward` scales `rho * out_in + (1 - rho) * out_ln`, not the layer
    # half inside it. Written as `rho*out_i + (1-rho)*out_l*gamma + beta` the
    # affine lands on one of the two normalised halves instead of on their
    # mixture, which is a different function everywhere and a large one where
    # gamma is far from 1 - and gamma here is the CAM's OUTPUT, so it is not
    # near 1 in general.
    return (rho * out_i + (1.0 - rho) * out_l) * gamma + beta


def _iln_gb(x, rho, gamma, beta, eps=1e-5):
    """The UpBlock2 stages' ILN, whose gamma/beta come from its own weights."""
    return _adailn_gb(x, rho, gamma, beta, eps)


# ---------------------------------------------------------------------------
# preprocessing, matching predict.py
# ---------------------------------------------------------------------------


def preprocess(img01):
    """`Resize((512,512))` + `ToTensor` + `Normalize(0.5, 0.5)`.

    The tensor is `x / 127.5 - 1`, which is what `Normalize(mean=0.5, std=0.5)`
    does to a `ToTensor`-scaled [0, 1] image.
    """
    return (img01 - 0.5) / 0.5


def load(path):
    w, meta = read_safetensors(path)
    return Generator(w, meta)


def _bilinear_resize(img, h, w):
    """`cv2.resize(..., interpolation=cv2.INTER_LINEAR)`, half-pixel centres.

    `align_corners=False` in torch's spelling: src = (dst + 0.5) * scale - 0.5,
    then a linear blend of the two nearest samples, edges clamped.
    """
    c, ih, iw = img.shape
    ys = (np.arange(h, dtype=np.float32) + 0.5) * (ih / h) - 0.5
    xs = (np.arange(w, dtype=np.float32) + 0.5) * (iw / w) - 0.5
    y0 = np.floor(ys).astype(np.int64)
    x0 = np.floor(xs).astype(np.int64)
    wy = (ys - y0).astype(np.float32)
    wx = (xs - x0).astype(np.float32)
    y0c, y1c = np.clip(y0, 0, ih - 1), np.clip(y0 + 1, 0, ih - 1)
    x0c, x1c = np.clip(x0, 0, iw - 1), np.clip(x0 + 1, 0, iw - 1)
    out = np.empty((c, h, w), dtype=np.float32)
    for i in range(h):
        for j in range(w):
            a = img[:, y0c[i], x0c[j]] * (1 - wx[j]) + img[:, y0c[i], x1c[j]] * wx[j]
            b = img[:, y1c[i], x0c[j]] * (1 - wx[j]) + img[:, y1c[i], x1c[j]] * wx[j]
            out[:, i, j] = a * (1 - wy[i]) + b * wy[i]
    return out


class _Counting(dict):
    """A weight dict that records which keys were fetched.

    The check below wants to know whether every tensor in the file is actually
    read, and scanning this file's text for the names does not answer that: 24 of
    the 44 names are built by f-string (`f"UpBlock1_{i + 1}.conv1.weight"`), so a
    substring test reports them as unused when they are read on every run. Counting
    the lookups answers the question directly.
    """

    def __init__(self, d):
        super().__init__(d)
        self.reads = {}

    def __getitem__(self, k):
        self.reads[k] = self.reads.get(k, 0) + 1
        return dict.__getitem__(self, k)


def self_test(weights, size=64):
    """A shape and range check that does not need torch.

    The things worth asserting here, none of which need a second implementation:
    that every tensor in the file is READ (an unused convolution is a missing
    layer, and counting the lookups is the only check that catches it), that the
    output has the input's shape - the generator is exactly shape-preserving, so
    the residual at the end lines up and no padding is needed at any size - and
    that the result is a plausible image after tanh rather than a saturated or
    NaN plane.
    """
    model = load(weights)
    model.w = _Counting(model.w)
    w = model.w
    rng = np.random.default_rng(0)
    x = preprocess(rng.random((1, 3, size, size), dtype=np.float32))
    y = model.forward(x)
    never = sorted(set(w) - set(w.reads))
    # `gap_fc.weight` and `gmp_fc.weight` are each read TWICE, and that is
    # upstream's own `forward`: once as the `Linear` over the pooled vector that
    # produces `gap_logit`/`gmp_logit`, and once as the per-channel scale on the
    # full map that the 1x1 conv reduces. Any OTHER tensor read twice would mean
    # the graph applies a weight somewhere it should not, which is why the list is
    # printed at all rather than left as a count.
    EXPECTED_TWICE = {"gap_fc.weight", "gmp_fc.weight"}
    twice = sorted(k for k, n in w.reads.items() if n > 1)
    print(f"weights: {len(w)} tensors, {sum(v.size for v in w.values())} values")
    print(f"never read: {never if never else 'none'}")
    print(f"read more than once: {twice if twice else 'none'}" +
          ("" if set(twice) <= EXPECTED_TWICE else "   <-- UNEXPECTED"))
    assert not never, "a tensor in the file is not part of the graph"
    assert y.shape == x.shape, f"{x.shape} -> {y.shape}: not shape-preserving"
    assert not np.isnan(y).any(), "NaN in the output"
    print(f"self-test: {x.shape} -> {y.shape}, mean {float(y.mean()):.6f}, "
          f"min {float(y.min()):.6f}, max {float(y.max()):.6f}")
    out = (y[0].transpose(1, 2, 0) * 0.5 + 0.5).clip(0, 1)
    print(f"as an image: mean {float(out.mean()):.4f}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="../models/nightenh-lol.safetensors")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--size", type=int, default=64, help="the self-test's input size")
    ap.add_argument("--fixture", default=None)
    ap.add_argument("--image", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--native", action="store_true",
                    help="run at the input's own size instead of 512x512")
    args = ap.parse_args()

    if args.check or not (args.fixture or args.image):
        self_test(args.weights, args.size)
        # The shape rule the engine's padding exists for, checked where it can
        # be: every multiple of 4 in a range round-trips through the two strided
        # convs and the two 2x upsamples, and the arithmetic says a size that is
        # not one cannot - 33 -> 17 -> 9 -> 18 -> 36 - so a non-multiple is a
        # padding decision for the caller and not a case this tool runs.
        model = load(args.weights)
        for size in range(8, 68, 4):
            got = model.forward(np.zeros((1, 3, size, size), dtype=np.float32)).shape[-1]
            assert got == size, (size, got)
        print("shape rule: every multiple of 4 from 8 to 64 round-trips exactly")
        return 0
    model = load(args.weights)
    if args.image:
        img = read_png(args.image)
        size = (img.shape[1], img.shape[2]) if args.native else (512, 512)
        inp = _bilinear_resize(img, size[0], size[1]) if size != (img.shape[1], img.shape[2]) else img
        y = model.forward(preprocess(inp[None]))
        out = (y[0].transpose(1, 2, 0) * 0.5 + 0.5).clip(0, 1)
        if not args.native and size != (img.shape[1], img.shape[2]):
            out = out.transpose(2, 0, 1)
            out = _bilinear_resize(out, img.shape[1], img.shape[2]).transpose(1, 2, 0)
        if args.out:
            write_png(args.out, (out * 255.0 + 0.5).astype(np.uint8))
    if args.fixture:
        os.makedirs(args.fixture, exist_ok=True)
        rng = np.random.default_rng(1234)
        # Sizes are multiples of 4, because that is the graph's requirement (see
        # the shape rule above) and a fixture outside it is not something the
        # engine should be asked to match. `rect-44x84` is deliberately not a
        # multiple of 64 in either axis: the engine pads to a multiple of 64 by
        # default, so a rectangular fixture is what proves the pad-and-crop path
        # and not just the aligned case.
        xs = {"tiny-16": (1, 3, 16, 16), "small-64": (1, 3, 64, 64), "rect-44x84": (1, 3, 44, 84)}
        for tag, shape in xs.items():
            x = preprocess(rng.random(shape, dtype=np.float32))
            np.save(os.path.join(args.fixture, f"input-{tag}.npy"), x)
            np.save(os.path.join(args.fixture, f"output-{tag}.npy"), model.forward(x))
        print(f"fixture: {args.fixture}")
    return 0


def read_png(path):
    """Minimal PNG reader for the RGB8/RGBA8/grayscale8 cases this tool sees."""
    import zlib
    with open(path, "rb") as f:
        data = f.read()
    assert data[:8] == b"\x89PNG\r\n\x1a\n", path
    pos, idat, meta = 8, [], None
    while pos < len(data):
        ln = struct.unpack(">I", data[pos:pos + 4])[0]
        typ = data[pos + 4:pos + 8]
        chunk = data[pos + 8:pos + 8 + ln]
        if typ == b"IHDR":
            w, h, depth, ctype = struct.unpack(">IIBB", chunk[:10])
            assert depth == 8, "only 8-bit PNGs"
            meta = (w, h, ctype)
        elif typ == b"IDAT":
            idat.append(chunk)
        elif typ == b"IEND":
            break
        pos += 12 + ln
    w, h, ctype = meta
    ch = {0: 1, 2: 3, 4: 2, 6: 4}[ctype]
    raw = zlib.decompress(b"".join(idat))
    stride = w * ch
    out = np.zeros((h, stride), dtype=np.uint8)
    prev = np.zeros(stride, dtype=np.uint8)
    p = 0
    for y in range(h):
        ft = raw[p]; p += 1
        line = np.frombuffer(raw[p:p + stride], dtype=np.uint8).astype(np.int32); p += stride
        cur = line.copy()
        if ft == 1:
            for i in range(ch, stride):
                cur[i] = (cur[i] + cur[i - ch]) & 0xFF
        elif ft == 2:
            cur = (cur + prev) & 0xFF
        elif ft == 3:
            for i in range(stride):
                a = cur[i - ch] if i >= ch else 0
                cur[i] = (cur[i] + ((a + int(prev[i])) >> 1)) & 0xFF
        elif ft == 4:
            for i in range(stride):
                a = int(cur[i - ch]) if i >= ch else 0
                b = int(prev[i]); cc = int(prev[i - ch]) if i >= ch else 0
                pa, pb, pc = abs(b - cc), abs(a - cc), abs(a + b - 2 * cc)
                pr = a if (pa <= pb and pa <= pc) else (b if pb <= pc else cc)
                cur[i] = (cur[i] + pr) & 0xFF
        out[y] = cur.astype(np.uint8)
        prev = cur.astype(np.uint8)
    px = out.reshape(h, w, ch)
    if ch == 1:
        rgb = np.repeat(px, 3, axis=2)
    elif ch == 2:
        rgb = np.repeat(px[:, :, :1], 3, axis=2)
    else:
        rgb = px[:, :, :3]
    return rgb.transpose(2, 0, 1).astype(np.float32) / 255.0


def write_png(path, rgb_u8):
    """Minimal PNG writer, `rgb_u8` being [h][w][3] uint8."""
    import zlib
    h, w, _ = rgb_u8.shape
    raw = b"".join(b"\x00" + rgb_u8[y].tobytes() for y in range(h))

    def chunk(typ, payload):
        c = struct.pack(">I", len(payload)) + typ + payload
        return c + struct.pack(">I", zlib.crc32(typ + payload) & 0xFFFFFFFF)

    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)))
        f.write(chunk(b"IDAT", zlib.compress(raw, 6)))
        f.write(chunk(b"IEND", b""))


if __name__ == "__main__":
    sys.exit(main())
