# nightenh-rs

One of the [lightgpu inference engines](https://github.com/jacobsparts/lightgpu).

Two trained models for night photography in one self-contained binary: one
corrects a very dark photo (lifts shadows, recovers colour and detail), and one
reduces glare and bloom from light sources - what the authors call light-effects
suppression. No Python, PyTorch, ONNX Runtime, or CUDA toolkit needed.

```sh
nightenh -m nightenh-lol.safetensors -i dark.png -o lit.png
```

* Both backends in one executable: a pure-Rust CPU path and a CUDA path with
  hand-written kernels. The GPU is used when the CUDA driver can be brought up
  and the CPU path when it cannot, so one binary covers a machine with no NVIDIA
  driver at all; `--device cpu` selects the CPU path explicitly and `--gpu`
  refuses to fall back.
* 1.60 MiB binary, statically linked except `libc` and `libgcc_s`;
  `libcuda.so.1` is `dlopen`ed, so no driver is required on disk.
* One architecture, two released checkpoints, and `-m` is the whole choice.

Both backends reproduce the upstream PyTorch implementation's output to within
floating-point rounding precision.

## Download

Prebuilt binary and the converted checkpoints are attached to the
[release](https://github.com/jacobsparts/nightenh-rs/releases).

| asset | what it is |
|---|---|
| `nightenh-linux-x86_64` | the engine: x86-64 Linux with glibc >= 2.34 (Ubuntu 22.04+, Debian 12+, RHEL 9+); the GPU path needs a compute capability 6.1+ GPU, and `--device cpu` runs the pure-Rust path anywhere |
| `nightenh-lol.safetensors` | the LOL checkpoint: trained for the low-light case |
| `nightenh-delighteffects.safetensors` | the de-light-effects checkpoint: for reflections and glare on an already-lit scene |

```sh
chmod +x nightenh-linux-x86_64
./nightenh-linux-x86_64 -m nightenh-lol.safetensors -i dark.png -o lit.png
```

## Models

Two checkpoints of the same architecture, differing only in what they were
trained on: `-m` is the whole choice, and the engine cannot tell the wrong one.

| checkpoint | what it does |
|---|---|
| `nightenh-lol` | low-light enhancement: corrects a very dark photo |
| `nightenh-delighteffects` | light-effects suppression: reduces glare from light sources (for already-lit scenes, not darkness) |

Both are converted from the authors' released `.pt` files:

```sh
python3 tools/convert.py LOL_params_0900000.pt lol.safetensors
python3 tools/convert.py delighteffects_params_0600000.pt delighteffects.safetensors
```

## Usage

```sh
nightenh -m nightenh-lol.safetensors -i dark.png -o lit.png
nightenh -m nightenh-lol.safetensors -i dark.png -o lit.png --device cpu
nightenh -m nightenh-lol.safetensors -i dark.png -o lit.png --native
cat dark.png | nightenh -m nightenh-lol.safetensors > lit.png
```

```
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
```

Input is 8-bit RGB or greyscale PNG and output is 8-bit RGB. Leaving `-i` out
reads the PNG from stdin and leaving `-o` out writes it to stdout - both are whole
files, read and written as bytes, so a pipe works. The progress lines go to
stderr and are silenced by `-q`.

## Sizes

Upstream's inference path resizes the input to 512x512, runs the model and
resizes back, and the default here is the same; `--native` runs at the input's
own size instead, each axis padded up to a multiple of 64 and cropped back.
Memory and time at each size (24-core AVX2 host, GTX 1080):

| model runs at | memory a pass needs | wall clock, GPU | wall clock, CPU |
|---|---|---|---|
| 512x512 (the default) | ~236 MiB | 4.75 s | 1.90 s |
| 1024x1024 | ~821 MiB | 12.47 s | 9.48 s |
| 2048x2048 | ~3161 MiB | 50.99 s | 45.81 s |

## Licence and attribution

The Rust and CUDA code in this repository is licensed under the MIT license; see
[LICENSE](LICENSE).

This is an independent reimplementation of the generator in
*Unsupervised Night Image Enhancement: When Layer Decomposition Meets
Light-Effects Suppression* (ECCV 2022) by Yeying Jin, Wenhan Yang and Robby T.
Tan, released at
[jinyeying/night-enhancement](https://github.com/jinyeying/night-enhancement).
The upstream repository carries an MIT license file, and its README states that
the code **and the models** are licensed under MIT **for academic and other
non-commercial uses**, with separate commercial licensing available from the
authors - so the two converted checkpoints attached to the release are
redistributed here under those same non-commercial terms and are **not** covered
by this repository's MIT license. See
[MODEL_LICENSE-NIGHTENH.txt](MODEL_LICENSE-NIGHTENH.txt) for the upstream text and
for exactly which files it covers.

`tools/reference.py` is a transcription of upstream's `networks.py` written
against the released model, and `tools/convert.py` reads the authors' released
`.pt` files; both are derived works under the same terms. The original `.pt`
files are not redistributed here.

