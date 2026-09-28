//! Night image enhancement on the lightgpu toolkit: the model plan, a CPU
//! executor and a GPU one.
//!
//! The engine is organised around one artifact: a [`model::Plan`], the
//! straight-line sequence of ops a *fixed input shape* runs. Building it resolves
//! every parameter name and records every buffer's size and live range in ONE
//! arena. Both backends then walk that same list, so `--device cpu` and
//! `--device gpu` cannot drift apart by more than floating point, and a new op is
//! added in three places that the compiler checks against each other: the [`Op`]
//! enum, `exec_cpu`, `exec_gpu`.
//!
//! [`Op`]: model::Op
//!
//! What this engine reproduces
//! --------------------------
//!
//! One feed-forward generator: `networks.py`'s `ResnetGenerator` from
//! [night-enhancement](https://github.com/jinyeying/night-enhancement) (ECCV 2022,
//! "Unsupervised Night Image Enhancement: When Layer Decomposition Meets
//! Light-Effects Suppression"). The repository's other parts - the vendored
//! Deep-Image-Prior `net/`, the discriminators, the Matlab decomposition stages -
//! are training-time or a different stage of the paper, and none of them run here.
//!
//! Two of the generator's properties shape the whole crate and are worth stating
//! before the code:
//!
//! * Every convolution is REFLECTION-padded (`ReflectionPad2d(3)` around the two
//!   7x7s, `(1)` around every 3x3). The toolkit's convs zero-pad, which is wrong
//!   at the border rather than merely different, so this engine carries reflect
//!   convolutions of its own and does not use `lg_conv3x3s1p1` at all.
//! * The network mixes the two variance conventions: `nn.InstanceNorm2d` divides
//!   by `H*W` and `torch.var` divides by `H*W - 1`, and both appear in one
//!   forward pass. [`model::Norm::divisor`] is where that lives, and the kernels
//!   take the divisor as an argument so it cannot be forgotten.
//!
//! The generator is exactly shape-preserving when both spatial dimensions are
//! multiples of 4, and only then; see [`image::pad_to_multiple`].

pub mod config;
#[cfg(feature = "cuda")]
pub mod cuda;
pub mod exec_cpu;
#[cfg(feature = "cuda")]
pub mod exec_gpu;
pub mod host;
pub mod image;
pub mod memguard;
pub mod model;
pub mod weights;

/// The crate's error type: a message, because every failure here is something a
/// person has to read and act on.
#[derive(Debug)]
pub struct Error(pub String);

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

impl From<String> for Error {
    fn from(s: String) -> Error {
        Error(s)
    }
}

impl From<&str> for Error {
    fn from(s: &str) -> Error {
        Error(s.to_string())
    }
}

/// The embedded kernel modules, one per `cuda/*.cu` (see `build.rs`).
///
/// Two modules rather than one: the shared toolkit's kernels and this engine's
/// own, each compiled with its own `--entries` list, so neither can shadow a name
/// in the other and a consumer of this crate cannot accidentally resolve a
/// kernel from the toolkit's set that this engine does not use.
#[cfg(feature = "cuda")]
pub fn modules() -> Vec<(&'static str, &'static [u8])> {
    vec![
        ("nightenh_toolkit", include_bytes!(concat!(env!("OUT_DIR"), "/nightenh_toolkit.fatbin"))),
        ("nightenh_project", include_bytes!(concat!(env!("OUT_DIR"), "/nightenh_project.fatbin"))),
    ]
}
