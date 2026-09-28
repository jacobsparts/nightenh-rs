#!/usr/bin/env python3
"""Check `tools/reference.py` against the REAL upstream `networks.py`.

Every engine in this family compares its two backends against a hand-written
transcription of the upstream model, and that transcription is the oracle - but a
transcription is only an oracle once something independent has confirmed it. The
fixtures the Rust test compares against are written BY the transcription, so a
mistake in it is invisible to every comparison in the tree: the engine and the
fixtures agree, because they are the same mistake.

This tool is that missing check, and it needs only what is already on the machine:
the released `.pt` (the converter's own input) and upstream's `networks.py`. It
loads both, runs the real module and the transcription over the same input, and
reports the worst difference PER STAGE - the first stage that diverges is the whole
diagnosis, and it is the thing that reading the two sources cannot be trusted for.

    /usr/local/bin/python3.11 tools/check_upstream.py \
        --networks ~/night-enhancement/networks.py \
        --weights ../models/LOL_params_0900000.pt

`.pt` and `.safetensors` are both accepted. What the per-stage table does NOT check
is the converter: given a `.safetensors`, both implementations are fed the SAME
file, so a bad conversion agrees with itself stage by stage and passes. That is why
`--ref` exists - it takes the released `.pt` alongside the `.safetensors` and
reports the worst difference PER TENSOR, which is `tools/convert.py`'s own check.
Run it whenever the transcription is edited or a checkpoint is converted.

Exit status is 0 when every stage agrees within the tolerance, and 1 otherwise, so
this can be wired into a test script when the upstream source is available.

Nothing about this runs in a normal `cargo test`: it needs torch, the upstream
source and a 42 MB checkpoint. It is the check to run whenever the transcription is
EDITED or a checkpoint is converted, and `reference.py`'s `LOG` lists every decision
in it that could have gone the other way.
"""
import argparse
import importlib.util
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import reference  # noqa: E402  (path set above; must not shadow a stdlib name)

# The stages both implementations are captured at, and where each lives upstream.
# The names are the trace dict's keys (see `Generator.forward`), so a stage added
# on one side and not the other shows up as a missing row rather than silently.
STAGES = ["DownBlock", "conv1x1", "FC", "gamma", "beta", "UpBlock2", "output"]


def fail(msg):
    print(f"check_upstream: {msg}", file=sys.stderr)
    return 1


def load_upstream(path):
    """Import `networks.py` by path, under its own module name."""
    if not os.path.exists(path):
        raise SystemExit(f"check_upstream: no such file: {path}\n"
                         "  get it with:\n"
                         "    git clone --depth 1 https://github.com/jinyeying/night-enhancement")
    spec = importlib.util.spec_from_file_location("up_networks", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "ResnetGenerator"):
        raise SystemExit(f"check_upstream: {path} defines no ResnetGenerator")
    return mod


def n_res_of(w):
    """The ResnetBlock count, from the key names rather than from an argument."""
    idx = [int(k.split(".")[1]) for k in w if k.startswith("DownBlock.") and "conv_block" in k]
    return max(idx) - 12 + 1


def load_state_ptr(path):
    """`{name: [..] f32}` from a released `.pt`, plus the two shapes the
    transcription needs and that a `.pt` does not carry: `n_res` and `ngf`."""
    import torch
    obj = torch.load(path, map_location="cpu", weights_only=False)
    sd = obj["genA2B"] if isinstance(obj, dict) and "genA2B" in obj else obj
    w = {k: v.detach().to(torch.float32).numpy() for k, v in sd.items() if hasattr(v, "numpy")}
    return w, {"n_res": str(n_res_of(w)), "ngf": str(w["DownBlock.1.weight"].shape[0])}


def load_weights(path):
    if path.endswith(".safetensors"):
        w, meta = reference.read_safetensors(path)
        return w, meta
    return load_state_ptr(path)


def upstream_stages(net, x, n_res, ngf):
    """Run upstream's module, hooking every boundary, and return {name: ndarray}."""
    import torch
    got = {}
    hooks = []

    def at(name):
        def f(_m, _i, o):
            got[name] = o.detach().cpu().numpy()
        return f

    # `net.relu` is the ReLU AFTER `conv1x1`, so hooking it gives the
    # post-activation feature map - which is the tensor the transcription calls
    # `conv1x1` and the decoder starts from.
    targets = [("DownBlock", net.DownBlock), ("conv1x1", net.relu), ("FC", net.FC),
               ("gamma", net.gamma), ("beta", net.beta), ("UpBlock2", net.UpBlock2)]
    for i in range(n_res):
        targets.append((f"UpBlock1_{i + 1}", getattr(net, f"UpBlock1_{i + 1}")))
    for name, m in targets:
        hooks.append(m.register_forward_hook(at(name)))
    with torch.no_grad():
        out = net(x)[0]
    for h in hooks:
        h.remove()
    got["output"] = out.detach().cpu().numpy()
    return got


def our_stages(w, meta, x):
    """Run the transcription, with `trace` capturing the same boundaries."""
    model = reference.Generator(w, meta)
    trace = {}
    trace["output"] = model.forward(x, trace=trace)
    return trace


def worst(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.shape != b.shape:
        return float("inf"), -1.0
    d = np.abs(a - b)
    return float(d.max()), float(d.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--networks", required=True,
                    help="path to upstream's networks.py")
    ap.add_argument("--weights", default="../models/LOL_params_0900000.pt",
                    help="released .pt (the converter's input) or a .safetensors")
    ap.add_argument("--ref", default=None,
                    help="a second checkpoint (the released .pt) to compare the "
                         "weights against, tensor by tensor - the CONVERTER check, "
                         "which the per-stage table cannot make")
    ap.add_argument("--size", type=int, default=128,
                    help="input size; a multiple of 4 (see reference.py's shape rule)")
    ap.add_argument("--tol", type=float, default=2e-4,
                    help="worst allowed per-stage difference, in the tensor's own units")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    if args.size % 4:
        return fail(f"--size {args.size} is not a multiple of 4")
    up = load_upstream(args.networks)
    w, meta = load_weights(args.weights)
    n_res, ngf = int(meta["n_res"]), int(meta["ngf"])
    print(f"weights: {args.weights} ({len(w)} tensors, n_res {n_res}, ngf {ngf})")

    if args.ref:
        ref_w, _ = load_weights(args.ref)
        only_here = sorted(set(w) - set(ref_w))
        only_there = sorted(set(ref_w) - set(w))
        rows = []
        for k in sorted(set(w) & set(ref_w)):
            if w[k].shape != ref_w[k].shape:
                rows.append((float("inf"), k))
                continue
            rows.append((float(np.abs(w[k].astype(np.float64)
                                     - ref_w[k].astype(np.float64)).max()), k))
        rows.sort(reverse=True)
        strict = not (only_here or only_there) and all(d == 0.0 for d, _ in rows)
        print(f"\nweights vs {args.ref}: {len(rows)} tensors in common, "
              f"worst |d| {rows[0][0]:.3e} ({rows[0][1]})" if rows else "\nno tensors in common")
        for d, k in rows[:5]:
            print(f"    {d:12.3e}  {k}")
        if only_here:
            print(f"    only in {args.weights}: {only_here[:3]}")
        if only_there:
            print(f"    only in {args.ref}: {only_there[:3]}")
        verdict = ("bit-exact" if strict else
                   "NOT bit-exact - this is tools/convert.py, not the network")
        print(f"    --> {verdict}")
        if not strict:
            return 1

    import torch
    net = up.ResnetGenerator(input_nc=3, output_nc=3, ngf=ngf, n_blocks=n_res, img_size=512)
    missing, unexpected = net.load_state_dict(
        {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in w.items()}, strict=False)
    missing = [m for m in missing if not m.endswith("num_batches_tracked")]
    if missing or unexpected:
        return fail(f"the weight file does not fit the architecture: "
                    f"{len(missing)} missing ({missing[:3]}), "
                    f"{len(unexpected)} unexpected ({unexpected[:3]})")
    net.eval()

    # Varied input, not zeros: on a constant input both implementations agree
    # trivially at every stage, and normalisation makes that worse rather than
    # better (a constant has zero variance, so every stage is the same number).
    rng = np.random.default_rng(args.seed)
    x = reference.preprocess(rng.random((1, 3, args.size, args.size), dtype=np.float32))

    a = upstream_stages(net, torch.from_numpy(x), n_res, ngf)
    b = our_stages(w, meta, x)

    print(f"{'stage':<14} {'values':>9} {'worst |d|':>12} {'mean |d|':>12}")
    bad = []
    for name in STAGES + [f"UpBlock1_{i + 1}" for i in range(n_res)]:
        if name not in a or name not in b:
            print(f"{name:<14} MISSING on {'upstream' if name not in a else 'reference.py'}")
            bad.append(name)
            continue
        d, m = worst(a[name], b[name])
        flag = "" if d <= args.tol else "   <-- DIVERGES"
        print(f"{name:<14} {a[name].size:9d} {d:12.3e} {m:12.3e}{flag}")
        if d > args.tol:
            bad.append(name)

    if bad:
        print(f"\nFAIL: {len(bad)} stage(s) differ by more than {args.tol:.1e}: {', '.join(bad)}")
        print("The FIRST row above is where the two implementations part; fix reference.py")
        print("(and src/model.rs, which mirrors it) before looking at any later row -")
        print("everything downstream of a divergence is the divergence, propagated.")
        return 1
    print(f"\nOK: every stage agrees within {args.tol:.1e} at {args.size}x{args.size}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
