#!/usr/bin/env python3
"""The same generator in torch, for timing - a benchmark, not a reference.

`tools/reference.py` is the specification: a torch-free numpy transcription of
upstream's `ResnetGenerator`, written to be checkable by reading. This file is the
other half of that arrangement - the SAME weights, the SAME block order and the
SAME numeric conventions, but running on torch's operators so that "how fast is
this network on PyTorch CPU / PyTorch CUDA" is a measurement rather than a guess.
It is deliberately NOT a third reference implementation: every operator here is a
call to torch, and the weight names come from `reference.read_safetensors`, so a
disagreement with `reference.py` is torch's arithmetic and not a second reading of
upstream.

Two conventions from `reference.py` that must not be simplified, and are the only
places this file departs from a naive module tree:

  * `nn.InstanceNorm2d` is BIASED (var over H*W, divisor H*W) and is what
    DownBlock and ResnetBlock use; `torch.var` is UNBIASED (divisor n - 1) and is
    what adaILN and ILN use, over [2,3] for the instance half and [1,2,3] for the
    layer half. Both are spelled out here rather than left to a module's default.
  * the padding split: the 7x7s and the strided 3x3s are reflect-padded
    (`ReflectionPad2d`, which EXCLUDES the edge sample), the 3x3s inside the blocks
    are zero-padded. A single padding mode everywhere would still produce a
    plausible image.

    python3 tools/torch_bench.py --weights ../models/nightenh-lol.safetensors \
        --size 512 --device cpu --device cuda
"""
import argparse
import os
import resource
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reference  # read_safetensors, preprocess


def build_torch():
    """Import torch, or explain why this benchmark cannot run."""
    try:
        import torch
    except ImportError:
        raise SystemExit("no torch on this interpreter: use one that has it "
                         "(python3.11 here) or skip this benchmark")
    return torch


class Net:
    """The generator, held as a plain dict of tensors plus the forward pass.

    Not `nn.Module`: the weights arrive from `read_safetensors` as numpy arrays
    with upstream's own names, so wrapping each in a Module would mean
    re-deriving the name mapping - the thing `reference.py` already got right. A
    dict and an explicit forward keeps this file about the arithmetic.
    """

    def __init__(self, tensors, meta, torch, device):
        # `np.ascontiguousarray(..., dtype=np.float32)` from `read_safetensors`
        # gives a read-only view of the mmap-less buffer; torch warns that a
        # non-writable tensor is undefined if written to. Nothing here writes to
        # one, but the copy makes the intent explicit rather than relying on that.
        self.t = {k: torch.from_numpy(np.array(v, copy=True)).to(device)
                  for k, v in tensors.items()}
        self.n_res = int(meta["n_res"])
        self.torch = torch
        self.device = device

    # --- primitives, in reference.py's order ---

    def conv(self, x, w, b, stride=1, pad_mode=None, pad=0):
        """Conv2d with an explicit padding mode.

        `pad_mode` is None for zero padding (torch's own `padding=`), "reflect"
        for `ReflectionPad2d(pad)` followed by a padding-free conv.
        """
        F = self.torch.nn.functional
        if pad_mode == "reflect":
            x = F.pad(x, (pad, pad, pad, pad), mode="reflect")
            return F.conv2d(x, w, b, stride=stride)
        return F.conv2d(x, w, b, stride=stride, padding=pad)

    def instance_norm_biased(self, x):
        """`nn.InstanceNorm2d(affine=False, track_running_stats=False)`."""
        n = x.shape[2] * x.shape[3]
        mu = x.mean(dim=(2, 3), keepdim=True)
        var = ((x - mu) ** 2).sum(dim=(2, 3), keepdim=True) / n
        return (x - mu) / (var + 1e-5).sqrt()

    def instance_norm_var(self, x):
        """`torch.var(x, [2, 3])` - UNBIASED, the divisor is n - 1."""
        n = x.shape[2] * x.shape[3]
        mu = x.mean(dim=(2, 3), keepdim=True)
        var = ((x - mu) ** 2).sum(dim=(2, 3), keepdim=True) / (n - 1)
        return (x - mu) / (var + 1e-5).sqrt()

    def layer_norm_var(self, x):
        """`torch.var(x, [1, 2, 3])` - over CHANNELS and space, unbiased."""
        n = x.shape[1] * x.shape[2] * x.shape[3]
        mu = x.mean(dim=(1, 2, 3), keepdim=True)
        var = ((x - mu) ** 2).sum(dim=(1, 2, 3), keepdim=True) / (n - 1)
        return (x - mu) / (var + 1e-5).sqrt()

    def adailn_gb(self, x, rho, gamma, beta):
        """adaILN with an external gamma/beta - the ResnetAdaILNBlock's form."""
        out_i = self.instance_norm_var(x)
        out_l = self.layer_norm_var(x)
        return rho * out_i + (1.0 - rho) * out_l * gamma + beta

    def linear(self, x, w, b=None):
        y = x @ w.t()
        return y if b is None else y + b

    # --- the graph ---

    def forward(self, x):
        t = self.t
        F = self.torch.nn.functional
        cat = self.torch.cat

        h = self.conv(x, t["DownBlock.1.weight"], None, pad_mode="reflect", pad=3)
        d1 = self.conv(h, t["DownBlock.5.weight"], None, stride=2, pad_mode="reflect", pad=1)
        d1 = self.instance_norm_biased(d1).relu()
        d2 = self.conv(d1, t["DownBlock.9.weight"], None, stride=2, pad_mode="reflect", pad=1)
        d2 = self.instance_norm_biased(d2).relu()

        body = d2
        for i in range(self.n_res):
            p = f"DownBlock.{12 + i}"
            o = self.conv(body, t[f"{p}.conv_block.1.weight"], None, pad=1)
            o = self.instance_norm_biased(o).relu()
            o = self.conv(o, t[f"{p}.conv_block.5.weight"], None, pad=1)
            o = self.instance_norm_biased(o)
            body = body + o

        gap = body.mean(dim=(2, 3), keepdim=True)
        gmp = body.amax(dim=(2, 3), keepdim=True)
        gap_w = self.linear(gap.reshape(1, -1), t["gap_fc.weight"]).reshape(-1, 1, 1)
        gmp_w = self.linear(gmp.reshape(1, -1), t["gmp_fc.weight"]).reshape(-1, 1, 1)
        cat_all = cat((gap, gmp), dim=1)
        feat = self.conv(cat_all, t["conv1x1.weight"], t["conv1x1.bias"]).relu()
        feat = feat.mean(dim=(2, 3)).reshape(1, -1)
        feat = self.linear(feat, t["FC.0.weight"]).relu()
        feat = self.linear(feat, t["FC.2.weight"])
        gamma = self.linear(feat, t["gamma.weight"]).reshape(-1, 1, 1)
        beta = self.linear(feat, t["beta.weight"]).reshape(-1, 1, 1)
        del gap_w, gmp_w

        x_up = body
        for i in range(self.n_res):
            p = f"UpBlock1_{i + 1}"
            o = self.conv(x_up, t[f"{p}.conv1.weight"], None, pad=1)
            o = self.adailn_gb(o, t[f"{p}.norm1.rho"], gamma, beta).relu()
            o = self.conv(o, t[f"{p}.conv2.weight"], None, pad=1)
            o = self.adailn_gb(o, t[f"{p}.norm2.rho"], gamma, beta)
            x_up = x_up + o

        up = F.interpolate(x_up, scale_factor=2, mode="nearest")
        up = self.conv(up, t["UpBlock2.2.weight"], None, pad_mode="reflect", pad=1)
        up = self.adailn_gb(up, t["UpBlock2.3.rho"], t["UpBlock2.3.gamma"], t["UpBlock2.3.beta"]).relu()
        up = F.interpolate(up, scale_factor=2, mode="nearest")
        up = self.conv(up, t["UpBlock2.7.weight"], None, pad_mode="reflect", pad=1)
        up = self.adailn_gb(up, t["UpBlock2.8.rho"], t["UpBlock2.8.gamma"], t["UpBlock2.8.beta"]).relu()
        out = self.conv(up, t["UpBlock2.11.weight"], None, pad_mode="reflect", pad=3)
        return (out + x).tanh()


def peak_rss_mb():
    """Peak RSS of this process in MiB - the host-side figure the engine reports."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def run(args):
    torch = build_torch()
    if args.threads:
        torch.set_num_threads(args.threads)
    tensors, meta = reference.read_safetensors(args.weights)
    rng = np.random.default_rng(int(meta.get("seed", 0)) if meta.get("seed") else 1234)
    x_np = reference.preprocess(
        rng.random((1, 3, args.size, args.size), dtype=np.float32))

    for device in args.device:
        if device == "cuda" and not torch.cuda.is_available():
            print(f"{device:5} skipped: no CUDA device")
            continue
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        net = Net(tensors, meta, torch, device)
        x = torch.from_numpy(x_np).to(device)
        with torch.no_grad():
            # Warm up, then time: the first call pays for cuDNN algorithm
            # selection and for the allocator, which is not a property of the
            # network.
            for _ in range(args.warmup):
                y = net.forward(x)
            if device == "cuda":
                torch.cuda.synchronize()
            times = []
            for _ in range(args.reps):
                t0 = time.perf_counter()
                y = net.forward(x)
                if device == "cuda":
                    torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)
        times.sort()
        best, med = times[0], times[len(times) // 2]
        rss = peak_rss_mb()
        if device == "cuda":
            dev = torch.cuda.max_memory_allocated() / (1024.0 ** 2)
            dev_res = torch.cuda.max_memory_reserved() / (1024.0 ** 2)
            mem = f"host peak RSS {rss:7.1f} MB, device peak {dev:7.1f} MB (reserved {dev_res:.1f})"
        else:
            mem = f"host peak RSS {rss:7.1f} MB"
        print(f"{device:5} {args.size}x{args.size}  best {best:7.3f} s  median {med:7.3f} s  {mem}")
        print(f"      output mean {float(y.mean()):.6f} min {float(y.min()):.6f} max {float(y.max()):.6f}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="../models/nightenh-lol.safetensors")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--device", action="append", default=None,
                    choices=["cpu", "cuda"])
    ap.add_argument("--threads", type=int, default=0,
                    help="torch.set_num_threads for CPU runs (0 = leave torch's default)")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    args = ap.parse_args()
    if args.device is None:
        args.device = ["cpu", "cuda"]
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
