#!/usr/bin/env python3
"""Convert a night-enhancement `.pt` checkpoint into the `.safetensors` this engine loads.

*Unsupervised Night Image Enhancement: When Layer Decomposition Meets Light-Effects
Suppression* (ECCV 2022), Yeying Jin, Wenhan Yang, Robby Tan -
https://github.com/jinyeying/night-enhancement. The checkpoints are the authors'
work and are redistributed as a format conversion, not covered by this
repository's copyright; see the README's attribution section.

    python3 tools/convert.py LOL_params_0900000.pt lol.safetensors
    python3 tools/convert.py delighteffects_params_0600000.pt delighteffects.safetensors

No torch, and no assumption about the container: the two released `.pt` files are
on OPPOSITE SIDES of torch's 1.6 format change. The LOL one is a legacy (pre-1.6)
pickle stream, decoded here by measurement rather than by description, so it is
worth writing down exactly what its bytes are:

  * five concatenated pickle streams - (1) the 15-byte magic pickle that torch
    legacy files open with, whose value is the integer 119547037146038801333356,
    (2) a bare `1001`, (3) a `sys_info` dict, (4) the state dict,
    (5) the list of 212 storage keys - ending at a fixed offset;
  * then the storage payloads in that key order, each **preceded by its element
    count as a little-endian int64** (`torch._legacy_save` writes `storage.size()`
    and then the bytes, one storage at a time) and followed by `numel * 4` bytes
    of little-endian f32. Every storage's extent still comes from the `numel` in
    its persistent id, and the prefix is redundant with it - so the prefix is
    CHECKED rather than used.

    This is the one thing about the container worth being emphatic about, because
    getting it wrong is invisible: reading the payloads as packed back to back
    leaves every storage after the first 8*index bytes too early, which shifts
    each tensor a fraction of its own length while leaving the count and every
    shape correct. The file still parses, still loads, and is wrong everywhere.
    Two independent signals catch a reader that has drifted: the offsets must end
    exactly at EOF (reading as if packed leaves the sum of the 212 prefixes
    unread), and the prefix must equal the `numel` in the persistent id that names
    the storage. The prefix is therefore CHECKED rather than used.

The de-light-effects checkpoint is a ZIP (`archive/data.pkl` plus one
`archive/data/<key>` per storage, the keys being the ones its persistent ids
name), and `_load_zip_state_dict` reads it with the same unpickler. Which reader
runs is decided by the file's first two bytes.

What these files do NOT let us copy blindly, and what is checked below:

  * **Only one generator is released for inference.** The state dict holds six
    entries - `genA2B`, `genB2A` (the two halves of the cycle) and four
    discriminators (`disGA`, `disGB`, `disLA`, `disLB`). Inference is `genA2B`
    alone: 44 tensors, 10.59 M values. `genB2A` has the SAME 44 parameter names
    with DIFFERENT values - it is a second trained network, not a copy of the
    first, which is worth stating because the cycle formulation invites the
    assumption that it is one (measured on the LOL checkpoint: same key set, all
    44 tensors differ). The discriminators are training-only. A converter that
    copied the whole state dict would quadruple the file for nothing.
  * **`torch.var` is UNBIASED and `nn.InstanceNorm2d` is BIASED.** The generator
    mixes them: `DownBlock`/`ResnetBlock` normalise with `InstanceNorm2d` (divide
    by H*W), while `adaILN` and `ILN` compute their statistics with `torch.var`
    over `[2, 3]` and `[1, 2, 3]`, which divides by `n - 1`. The two differ in the
    5th significant figure on a 512x512 map but not on a 2x2 test crop, so the
    engine's kernels take the divisor and the reference reproduces both - the
    converter only has to keep the tensors, but the asymmetry is recorded in the
    header so the Rust cannot silently unify them.
  * **`n_res` is not in the weights.** LOL uses 4 ResnetBlocks and delight-effects
    uses 6, and neither the tensor shapes nor the file name is unambiguous about
    which. The count is read from the deepest `DownBlock.<i>.` index and written
    into `__metadata__`, and `--n-res` can override it.

The `__metadata__` map is the architecture the Rust side reads back, so a
converted file states what it is instead of relying on its name.
"""
import argparse
import collections
import json
import os
import pickle
import struct
import sys

import numpy as np

# torch legacy containers open with this integer, written as its own pickle.
MAGIC = 119547037146038801333356

# The two released inference checkpoints, by the name they are published under.
# `n_res` is the ResnetBlock count; the file names carry the iteration count
# rather than the architecture.
#
# BOTH are 4, and that is a measurement rather than an assumption: the two files
# have identical `genA2B` key sets and identical tensor shapes (44 tensors,
# 10.589 M values, checked tensor by tensor), and the de-light-effects one has
# `DownBlock.12..15` with `conv_block` exactly as the LOL one does. `n_res=6`
# would be the only difference the architecture could carry, and the weights do
# not carry it - so a run that trusted the name over the file would build a plan
# whose parameter lookups could not all resolve. Upstream's own default is a
# separate question from what it released.
KNOWN = {
    "LOL_params_0900000": ("lol", 4),
    "delighteffects_params_0600000": ("delight-effects", 4),
}


class _Unpickler(pickle.Unpickler):
    """A pickler for a torch state dict that never imports torch.

    `_rebuild_tensor_v2(storage, offset, size, stride, ...)` is reduced to the
    four things a converter needs, and the storage itself - fetched through
    `persistent_load` - to its `(key, numel)`.
    """

    def find_class(self, module, name):
        if module == "torch._utils" and name in ("_rebuild_tensor_v2", "_rebuild_tensor"):
            return self._rebuild_tensor
        if module == "torch" and name.endswith("Storage"):
            return lambda *a, **k: ("storage_cls", name)
        if module == "collections" and name == "OrderedDict":
            return collections.OrderedDict
        raise pickle.UnpicklingError(f"unexpected global {module}.{name}")

    @staticmethod
    def _rebuild_tensor(storage, storage_offset, size, stride, *rest):
        return {"storage": storage, "offset": int(storage_offset),
                "size": tuple(int(s) for s in size),
                "stride": tuple(int(s) for s in stride)}

    def persistent_load(self, pid):
        # ('storage', StorageClass, key, location, numel). The discriminators do
        # not have to agree with the generator on the tuple's width, so this
        # takes the two fields it needs by position rather than unpacking all
        # five - a converter that reads a whole multi-model state dict cannot
        # assume one shape.
        if not (isinstance(pid, tuple) and pid and pid[0] == "storage"):
            raise pickle.UnpicklingError(f"unexpected persistent id {pid!r}")
        if len(pid) < 5:
            raise pickle.UnpicklingError(f"short persistent id {pid!r}")
        return ("storage_ref", str(pid[2]), int(pid[4]))


def load_state_dict(path):
    """Read a `.pt` into ({name: tensor_dict}, {storage key: raw bytes}).

    Two containers are accepted, and which one a file uses is decided by its
    first two bytes rather than by its name: torch changed the format in 1.6 and
    the two released night-enhancement checkpoints are on opposite sides of that
    change. The LOL one is the legacy stream described below; the de-light-effects
    one is a zip archive whose `data.pkl` holds the same `_rebuild_tensor_v2`
    graph, so `_Unpickler` is reused for both and only where the BYTES come from
    differs.

    Getting this wrong is silent rather than obvious: the legacy reader on a zip
    file fails with `persistent IDs in protocol 0 must be ASCII strings`, which
    names neither the container nor the checkpoint.
    """
    with open(path, "rb") as f:
        if f.read(2) == b"PK":
            return _load_zip_state_dict(path)
    return _load_legacy_state_dict(path)


def _load_legacy_state_dict(path):
    """Read a pre-1.6 `.pt`: five pickles, then the storage payloads.

    The five pickles are read in order; the payload offset that follows the last
    one is where the storages begin, and each storage's length is the `numel` from
    its persistent id times four.
    """
    with open(path, "rb") as f:
        u = _Unpickler(f)
        magic = u.load()
        if magic != MAGIC:
            raise SystemExit(f"{path}: not a torch legacy file (magic {magic!r})")
        u.load()                       # a bare protocol version, 1001
        sys_info = u.load()            # {protocol_version, little_endian, type_sizes}
        state = u.load()
        keys = u.load()
        payload = f.tell()

        if not sys_info.get("little_endian", True):
            raise SystemExit(f"{path}: big-endian storages are not supported")

        sizes = {}
        for entry in state.values():
            if not isinstance(entry, dict):
                continue
            for t in entry.values():
                if isinstance(t, dict) and "storage" in t:
                    ref, numel = t["storage"][1], t["storage"][2]
                    sizes[ref] = numel
        missing = [k for k in keys if k not in sizes]
        if missing:
            raise SystemExit(f"{path}: storage keys with no tensor: {missing[:4]}")

        storages = {}
        offset = payload
        for k in keys:
            # The per-storage int64 length prefix, checked against the persistent
            # id's `numel`. The bytes come from the prefix's extent only after the
            # two agree.
            f.seek(offset)
            head = f.read(8)
            if len(head) != 8:
                raise SystemExit(f"{path}: no length prefix for storage {k} at {offset}")
            (declared,) = struct.unpack("<q", head)
            if declared != sizes[k]:
                raise SystemExit(
                    f"{path}: storage {k} is prefixed with {declared}, its persistent "
                    f"id says {sizes[k]} - not a torch legacy file?")
            n = sizes[k] * 4
            raw = f.read(n)
            if len(raw) != n:
                raise SystemExit(f"{path}: truncated storage {k} at {offset}")
            storages[k] = raw
            offset += 8 + n
        # Any tail left over is a storage whose persistent id understates a view
        # over a larger buffer: not readable from the header alone, and nothing in
        # `genA2B` lives in it. Honouring the prefixes makes this 0 for the
        # released LOL checkpoint.
        slack = os.path.getsize(path) - offset

    return state, storages, slack, payload


def _load_zip_state_dict(path):
    """Read a torch >= 1.6 `.pt`: a zip holding `data.pkl` and one file per storage.

    The archive holds exactly one pickle, and the storages live beside it under
    `data/`, named by the key in their persistent id - so `_Unpickler` needs no
    change and the `keys` pickle has no counterpart here. `numel * 4` is still the
    length, because a night-enhancement checkpoint is float32 throughout.

    The pickled object is used as-is when it is already the state dict, and one
    level down when it is a training dict carrying the state dict under a key,
    which is what `torch.save({'model': ...})` produces.
    """
    import zipfile

    with zipfile.ZipFile(path) as z:
        pkls = [n for n in z.namelist() if n.endswith("data.pkl")]
        if len(pkls) != 1:
            raise SystemExit(f"{path}: expected one data.pkl, found {pkls[:4]}")
        prefix = pkls[0][: -len("data.pkl")]
        with z.open(pkls[0]) as fh:
            obj = _Unpickler(fh).load()

        # The archive's pickle can be the state dict itself or a dict with one
        # level of wrapper; the caller below expects the SAME object shape the
        # legacy path returns, i.e. a dict keyed by model name (`genA2B`,
        # `genB2A`, the four discriminators) whose values are the tensors - NOT
        # the generator's tensors directly, because the checks below read
        # `state["genB2A"]` to show the two cycle halves are separately trained.
        if any(isinstance(v, dict) and "storage" in v for v in obj.values()):
            state = {"genA2B": obj}
        else:
            state = obj if any(
                isinstance(v, dict) and any(isinstance(t, dict) and "storage" in t for t in v.values())
                for v in obj.values()) else {"genA2B": next(
                    v for v in obj.values() if isinstance(v, dict))}

        # The same TWO-LEVEL walk the legacy path uses, and it has to be two
        # levels: a tensor is `state[model][name]`, so a loop over the outer dict
        # alone finds no storage and under-reports the sizes, which surfaces
        # much later as "storage <key> is missing".
        sizes = {}
        for model in state.values():
            if not isinstance(model, dict):
                continue
            for t in model.values():
                if isinstance(t, dict) and "storage" in t:
                    ref, numel = t["storage"][1], t["storage"][2]
                    sizes[ref] = numel

        storages = {}
        for k, numel in sizes.items():
            member = f"{prefix}data/{k}"
            try:
                raw = z.read(member)
            except KeyError:
                raise SystemExit(f"{path}: {member} is named by a tensor but is not in the archive")
            n = numel * 4
            if len(raw) < n:
                raise SystemExit(f"{path}: {member} holds {len(raw)} bytes, {n} needed")
            storages[k] = raw[:n]

    return state, storages, 0, 0


def materialise(entry, storages):
    """Return the C-contiguous array a saved tensor describes.

    The saved tensor may be a view with an arbitrary stride; numpy's
    `as_strided` + copy walks it, and the C-order copy is what gets written.
    """
    ref, numel = entry["storage"][1], entry["storage"][2]
    raw = storages.get(ref)
    if raw is None:
        raise SystemExit(f"storage {ref} is missing")
    st = np.frombuffer(raw, dtype="<f4")
    offset, shape, stride = entry["offset"], entry["size"], entry["stride"]
    if shape == () or stride == ():
        return np.ascontiguousarray(st[offset:offset + 1].reshape(shape or ()))
    view = np.lib.stride_tricks.as_strided(
        st[offset:], shape=shape,
        strides=tuple(s * 4 for s in stride), writeable=False)
    return np.ascontiguousarray(view)


def n_res_from_weights(state):
    """The ResnetBlock count, from the weights.

    The generator numbers its submodules in DECLARATION order, so the indices are
    not consecutive and the largest one is not the count: for the LOL checkpoint
    the initial 7x7 conv is `DownBlock.1`, the two stride-2 stages are
    `DownBlock.5` and `DownBlock.9`, and the ResnetBlocks are `DownBlock.12..15`.
    A ResnetBlock is the only thing whose name contains `conv_block`, so counting
    those is unambiguous where the index range is not.
    """
    idx = sorted({int(k.split(".")[1]) for k in state
                  if k.startswith("DownBlock.") and k.count(".") >= 2
                  and k.split(".")[1].isdigit()
                  and "conv_block" in k})
    if not idx:
        raise SystemExit("no DownBlock.<i>.conv_block tensors - not a night-enhancement generator?")
    if idx != list(range(idx[0], idx[0] + len(idx))):
        raise SystemExit(f"ResnetBlock indices are not consecutive: {idx}")
    return len(idx)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--variant", default=None, help="lol or delight-effects (default: from the file name)")
    ap.add_argument("--n-res", type=int, default=None, help="ResnetBlock count (default: from the weights)")
    ap.add_argument("--ngf", type=int, default=64)
    ap.add_argument("--img-size", type=int, default=512,
                    help="the size the model was trained at; recorded, not required")
    ap.add_argument("--all-generators", action="store_true",
                    help="also write genB2A (an exact duplicate; off by default)")
    args = ap.parse_args()

    state, storages, slack, payload = load_state_dict(args.src)
    groups = [k for k, v in state.items() if isinstance(v, dict) and v
              and all(isinstance(t, dict) for t in v.values())]
    if "genA2B" not in state:
        raise SystemExit(f"{args.src}: no genA2B - not a night-enhancement checkpoint?")
    gen = {k: v for k, v in state["genA2B"].items() if isinstance(v, dict)}

    # genB2A is the other half of the cycle and is never run at inference. It is
    # documented as a duplicate; check it rather than trust the documentation.
    if "genB2A" in state:
        other = {k: v for k, v in state["genB2A"].items() if isinstance(v, dict)}
        if set(other) != set(gen):
            print(f"note: genB2A has {len(set(other) ^ set(gen))} parameter names "
                  f"genA2B does not", file=sys.stderr)
        else:
            # Same names, and deliberately NOT the same values: the two directions
            # of the cycle are trained separately. Read one tensor to make the
            # distinction visible rather than asserting a duplicate that is not.
            k = sorted(gen)[0]
            same = np.array_equal(materialise(gen[k], storages),
                                  materialise(other[k], storages))
            print(f"genB2A: {len(other)} tensors, same names as genA2B, "
                  f"`{k}` identical: {same} (expected False - separately trained)",
                  file=sys.stderr)
    if args.all_generators:
        for name, tensors in state.items():
            if name.startswith("gen") and name != "genA2B":
                for k, v in tensors.items():
                    if isinstance(v, dict):
                        gen[f"{name}/{k}"] = v

    n_res_w = n_res_from_weights(gen)
    n_res = args.n_res or n_res_w
    if n_res != n_res_w:
        print(f"warning: --n-res {n_res} but the weights have {n_res_w}", file=sys.stderr)

    variant = args.variant
    if variant is None:
        base = os.path.splitext(os.path.basename(args.src))[0]
        variant = KNOWN.get(base, (None, None))[0] or "unknown"
        if variant != "unknown":
            kv = KNOWN[base][1]
            if kv != n_res:
                print(f"warning: {base} is published with n_res={kv}, weights have {n_res}",
                      file=sys.stderr)
    # The generator's parameter names, which is what a converted file is
    # allowed to contain. `gamma`/`beta` are top-level here because the CAM
    # chain's two Linear layers take the name of the attribute they are
    # assigned to, not of a module they live in.
    KNOWN_PARAMS = ("DownBlock.", "UpBlock1_", "UpBlock2.", "conv1x1.",
                    "FC.", "gap_fc.", "gmp_fc.", "gamma.", "beta.")
    for name in gen:
        if not name.startswith(KNOWN_PARAMS):
            print(f"warning: unexpected tensor `{name}`", file=sys.stderr)

    metadata = {
        "format": "pt",
        "arch": "nightenh",
        "variant": str(variant),
        "n_res": str(n_res),
        "ngf": str(args.ngf),
        "img_size": str(args.img_size),
        "in_nc": "3",
        "out_nc": "3",
        "norm": "instance_biased+adailn_unbiased",
        "layout": "torch_nchw_conv_kernel",
    }

    tensors = []
    for name, entry in gen.items():
        arr = materialise(entry, storages).astype("<f4", copy=False)
        tensors.append((name, arr))
    tensors.sort(key=lambda kv: kv[0])

    offset = 0
    header = {}
    for name, arr in tensors:
        nbytes = arr.size * 4
        header[name] = {"dtype": "F32", "shape": list(arr.shape),
                        "data_offsets": [offset, offset + nbytes]}
        offset += (nbytes + 7) & ~7      # 8-byte alignment for the f32 mmap path
    header["__metadata__"] = metadata
    hjson = json.dumps(header, separators=(",", ":")).encode()
    header_bytes = 8 + len(hjson)
    pad = (-header_bytes) & 7

    with open(args.dst, "wb") as f:
        f.write(struct.pack("<Q", len(hjson) + pad))
        f.write(hjson)
        f.write(b" " * pad)
        written = 0
        for name, arr in tensors:
            want = header[name]["data_offsets"][0]
            if written < want:
                f.write(b"\0" * (want - written))
                written = want
            raw = arr.astype("<f4", copy=False).tobytes()
            f.write(raw)
            written += len(raw)

    total = sum(arr.size for _, arr in tensors)
    print(f"{args.dst}: {len(tensors)} tensors, {total} values ({total / 1e6:.2f} M), "
          f"variant {variant}, n_res {n_res}, ngf {args.ngf}")
    if slack:
        print(f"{args.src}: {slack} unread trailing bytes (a view past its id's numel)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
