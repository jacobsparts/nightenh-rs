//! Compiles this engine's kernels into two modules: the shared `lightgpu`
//! toolkit's `cuda/kernels.cu` (TOOLKIT_KERNELS) and this project's own
//! `cuda/nightenh.cu` (PROJECT_KERNELS). Each compiles to its own fatbin with its
//! own `--entries` list and `src/cuda.rs` loads them as separate modules, so
//! neither can shadow a name in the other.
//!
//! A kernel missing from its list is PRUNED from the fatbin and fails at launch
//! rather than at build time, so both lists are checked against the source they
//! are compiled from before nvcc runs: a typo, or a kernel moved between files,
//! fails the build.

/// Generic ops from the shared toolkit. Exactly the ones `exec_gpu` launches: an
/// entry here is a kernel embedded in the binary, so a name no op calls is wasted
/// bytes.
///
/// The generator this engine runs is unusual in how LITTLE of the toolkit it can
/// use, and the absences are worth recording because each one is a deliberate
/// new kernel in `cuda/nightenh.cu` rather than an oversight:
///
/// * every convolution in the network is reflection-padded (`ReflectionPad2d(1)`
///   around each 3x3, `ReflectionPad2d(3)` around the two 7x7s), and the toolkit's
///   convs zero-pad. `lg_conv3x3s1p1` is therefore not merely unused here, it is
///   WRONG at the border, which is why `ne_conv3x3_refl` and `ne_conv7x7_refl`
///   exist instead of a pad kernel feeding a toolkit conv: doing it in one kernel
///   keeps the halo read in registers rather than materialising a padded copy of
///   the whole plane per layer.
/// * `lg_upsample2x_nearest` IS used: its contract is `out[o] = in[o / 2]`, which
///   is exactly PyTorch's exact-2x nearest, so this step needs no new kernel.
const TOOLKIT_KERNELS: &[&str] = &[
    // Elementwise, on whole planes.
    "lg_relu",
    "lg_add",
    // The EIGHT zero-padded 3x3s: the four ResnetBlocks and the four adaILN
    // blocks are `nn.Conv2d(3, 3, 1, padding=1)` with a ZERO border, which is
    // what this kernel is, while the six reflection-padded convs use this
    // project's own `ne_*_refl` kernels. The network mixes the two, so a build
    // that dropped this name would launch a pruned kernel and fail at the first
    // resnet block rather than at build time.
    "lg_conv3x3s1p1",
    // The CAM chain's two 1x1 stages: the 512 -> 256 reduction and the
    // 1-channel attention weight per channel.
    "lg_conv1x1",
    // The global average pool that feeds the CAM, and (via `lg_channel_mean`'s
    // twin below) the two halves of the adaILN normalisation.
    "lg_channel_mean",
    // The nearest 2x upsample in the decoder.
    "lg_upsample2x_nearest",
    // EXPERIMENTAL, like `nex_conv3x3s1p1_x8` below: no op launches this one, and
    // it is embedded so the question "what would an im2col + GEMM convolution
    // reach here" stays askable. It exists because a measurement, not a theory,
    // said so: with `torch.backends.cudnn.enabled = False` - no library this
    // project may not link - torch convolves at 5,195 GFLOP/s where the best
    // kernel in this tree reaches 607, and the profiler says it is `im2col_kernel`
    // (25%) followed by a tiled SGEMM (73%) at 6.26 TFLOP/s. Every kernel here is
    // a DIRECT convolution with no operand staged through shared memory, and the
    // toolkit's own `lg_f32_gemm_tiled` is the structure torch uses. This entry
    // exists so `tests/reference.rs`'s `gpu_tiled_gemm_on_the_im2col_shape` can
    // measure it: a kernel missing from this list is pruned from the fatbin and
    // fails at LAUNCH, not at build, which is exactly what happened the first time
    // that test ran.
    "lg_f32_gemm_tiled",
    // The CAM's three Linear layers (gap/gmp weights, then FC.0 and FC.2) are all
    // [out][in] over a 1 x C row, which is what `lg_linear` is.
    "lg_linear",
];

/// This project's own kernels, in `cuda/nightenh.cu`.
const PROJECT_KERNELS: &[&str] = &[
    // Reflection-padded convolutions: the two 7x7s (stride 1 and stride 2) and
    // the 3x3s (stride 1 and stride 2).
    "ne_conv3x3_refl",
    "ne_conv7x7_refl",
    "ne_down3x3_refl",
    "ne_down7x7_refl",
    // EXPERIMENTAL, and the one entry here no op launches: a zero-padded 3x3 that
    // hoists each input value into registers and reuses it across all nine taps,
    // which is the same reuse argument the CPU kernels make with row blocking. It
    // is embedded because `tests/reference.rs`'s `gpu_3x3_kernel_throughput`
    // measures it against the two kernels that ARE used, and the question it
    // answers - how much headroom is left in the convs that are 75% of this
    // device's time - is worth keeping askable. It is not wired into `exec_gpu`
    // because its loop nest is rotated, so it rounds differently from the twin.
    "nex_conv3x3s1p1_x8",
    // EXPERIMENTAL, the other entry no op launches: a 128x64 f32 GEMM that stages
    // BOTH operands in shared memory and double-buffers the k-tiles. It exists
    // because a disassembly named the difference. The kernel cuBLAS ran for the
    // im2col GEMM on this card is `sgemm_128x128x8_NN_vec` at REG:128 SHARED:16912
    // with 128-bit shared loads, while the toolkit's `lg_f32_gemm_tiled` is
    // REG:54 SHARED:512 - one single-buffered strip, weights read from GLOBAL per
    // column slot, addresses recomputed every k-step. `tests/reference.rs`'s
    // `gpu_gemm_double_buffered` measures this against that kernel on the same
    // shapes, and checks it against a triple loop with a TOLERANCE (its 8-wide
    // k-step and shared staging associate the sum differently, so unlike the
    // CPU kernels it is not bit-identical to anything).
    "nex_f32_gemm_db",
    // InstanceNorm2d, biased: the DownBlock/ResnetBlock normalisation.
    "ne_instance_norm",
    // The adaptive instance-layer norm, one kernel for both the four
    // ResnetAdaILNBlocks (with the CAM's gamma/beta) and the two UpBlock2 stages
    // (with their own): the arithmetic is the same and only the operand
    // addresses differ.
    "ne_adailn",
    // The per-channel affine at the end of the ILN (`* gamma + beta`), as its own
    // kernel so the ILN's two normals stay in one place.
    "ne_channel_affine",
    // The global MAX pool feeding the CAM's second weight, paired with
    // `lg_channel_mean` for the first.
    "ne_channel_max",
    // The elementwise multiply of a per-channel weight by a whole plane - the
    // `weight * gap` and `weight * gmp` terms of the CAM, and the `rho * inst`
    // term of the ILN. `lg_mul` is a plane-times-plane op and has no broadcast
    // form, and `lg_channel_affine` has no multiply-without-add.
    "ne_channel_mul",
    // The residual add with the INPUT plane at the end (`tanh(out + input)`) and
    // the tanh itself, fused because the alternative is a whole extra plane.
    "ne_tanh_add",
];

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=cuda/nightenh.cu");

    // `cargo build --no-default-features` is the pure-Rust CPU build: it must not
    // need nvcc, and `src/cuda.rs` (which includes the fatbins) is not compiled at
    // all, so the env vars it would embed are not needed.
    if std::env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }

    let toolkit = lightgpu_build::toolkit_kernels_cu()
        .expect("locate lightgpu's cuda/kernels.cu (set LA_GPU_DIR to override)");
    let toolkit = toolkit.to_string_lossy().into_owned();

    for k in TOOLKIT_KERNELS {
        assert!(
            lightgpu_build::known_kernel(k),
            "unknown kernel `{k}`: not defined by the toolkit - did it move into cuda/nightenh.cu?"
        );
    }
    let src = std::fs::read_to_string("cuda/nightenh.cu").expect("read cuda/nightenh.cu");
    let defined = lightgpu_build::kernel_names_in(&src);
    for k in PROJECT_KERNELS {
        assert!(
            defined.iter().any(|d| d == k),
            "`{k}` is not defined in cuda/nightenh.cu (it has {})",
            defined.join(", ")
        );
    }
    // The other direction matters just as much: a kernel defined but NOT listed is
    // pruned from the fatbin by `--entries`, and then it fails at LAUNCH rather
    // than at build time. Checking both directions makes the list and the source
    // agree rather than merely overlap.
    for d in &defined {
        assert!(
            PROJECT_KERNELS.contains(&d.as_str()),
            "cuda/nightenh.cu defines `{d}`, which PROJECT_KERNELS does not list - \
             it would be pruned from the fatbin and fail at launch"
        );
    }

    lightgpu_build::fatbin_modules(&[
        lightgpu_build::Source {
            path: &toolkit,
            out_name: "nightenh_toolkit.fatbin",
            entries: Some(TOOLKIT_KERNELS),
        },
        lightgpu_build::Source {
            path: "cuda/nightenh.cu",
            out_name: "nightenh_project.fatbin",
            entries: Some(PROJECT_KERNELS),
        },
    ]);
}
