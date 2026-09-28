//! Parameter access: the checkpoint's tensors by the name the graph asks for,
//! and the few layout facts the kernels depend on.
//!
//! Torch stores a conv kernel as `[c_out][c_in][kh][kw]` and a Linear as
//! `[out][in]`, and BOTH are already the layout the kernels here read: NCHW
//! input, `c_in` contiguous inside a tap, `ky, kx, ci` accumulation order (the
//! toolkit's convention, documented in `lightgpu/cuda/CONVENTIONS.md`). So unlike
//! the engines that convert Flax checkpoints - where every conv is `[kh][kw][ci][co]`
//! and needs a real transpose - there is no rearrangement here at all, and the
//! only thing this file does is hand out borrowed slices under the names the
//! graph uses. A transpose added "for symmetry with the other engines" would be
//! the bug.
//!
//! The architecture comes from the FILE, not from a flag: the converter writes
//! `n_res` and `ngf` into `__metadata__` after counting the model's own
//! `conv_block` modules, and a name that does not carry them (`n_res` is not a
//! weight shape - 4 for LOL and 6 for delight-effects) cannot be guessed from the
//! tensors or the file name. A missing key is an error here rather than a
//! default, because a default would silently load the wrong architecture.

use lightgpu::safetensors::{File, TensorInfo};

use crate::Error;

pub struct Weights {
    file: File,
    /// The ResnetBlock count, from the file's metadata.
    pub n_res: usize,
    /// The first convolution's channel width (`ngf` upstream), 64 for both
    /// released checkpoints.
    pub ngf: usize,
    /// `lol` or `delight-effects` - the checkpoint's own label, carried through
    /// from the converter. Reported by `--version`-style output and by the plan
    /// line; nothing branches on it.
    pub variant: String,
}

impl Weights {
    pub fn open(path: &str) -> Result<Weights, Error> {
        let file = File::open(path)?;
        let n_res = file.metadata_usize("n_res")?;
        let ngf = file.metadata_usize("ngf")?;
        let variant = file.metadata_get("variant").unwrap_or("unknown").to_string();
        let arch = file.metadata_get("arch").unwrap_or("");
        if arch != "nightenh" {
            return Err(Error(format!(
                "{path}: arch is `{arch}`, not `nightenh` - is this a converted \
                 night-enhancement checkpoint? (tools/convert.py writes the arch)"
            )));
        }
        if n_res != 4 && n_res != 6 {
            return Err(Error(format!(
                "{path}: n_res {n_res}; only 4 (LOL) and 6 (delight-effects) have \
                 been released, and the ResnetBlock indices differ between them"
            )));
        }
        let w = Weights { file, n_res, ngf, variant };
        w.check_present()?;
        Ok(w)
    }

    /// Every parameter the graph reads, in graph order. Used for the presence
    /// check and by the tests, so the list exists once.
    ///
    /// The ResnetBlock indices are `12 + i` for `i < n_res`, which is a fact the
    /// converter asserts (their `conv_block` names must be consecutive) rather
    /// than something this file can verify from the weights alone - so `open`
    /// rejects any `n_res` outside the two that have been released.
    pub fn expected_names(&self) -> Vec<String> {
        let mut v: Vec<String> = vec![
            "DownBlock.1.weight".into(),
            "DownBlock.5.weight".into(),
            "DownBlock.9.weight".into(),
        ];
        for i in 0..self.n_res {
            v.push(format!("DownBlock.{}.conv_block.1.weight", 12 + i));
            v.push(format!("DownBlock.{}.conv_block.5.weight", 12 + i));
        }
        v.push("gap_fc.weight".into());
        v.push("gmp_fc.weight".into());
        v.push("conv1x1.weight".into());
        v.push("conv1x1.bias".into());
        v.push("FC.0.weight".into());
        v.push("FC.2.weight".into());
        v.push("gamma.weight".into());
        v.push("beta.weight".into());
        for i in 1..=self.n_res {
            let p = format!("UpBlock1_{i}");
            v.push(format!("{p}.conv1.weight"));
            v.push(format!("{p}.conv2.weight"));
            v.push(format!("{p}.norm1.rho"));
            v.push(format!("{p}.norm2.rho"));
        }
        // The two ILN stages: `.2` is the 128-channel one and `.7` the
        // 64-channel one, each followed by a `.3` / `.8` group holding rho,
        // gamma and beta. The index pairs are upstream's, taken from the
        // checkpoint rather than guessed - see `tools/convert.py`.
        v.push("UpBlock2.2.weight".into());
        v.push("UpBlock2.3.rho".into());
        v.push("UpBlock2.3.gamma".into());
        v.push("UpBlock2.3.beta".into());
        v.push("UpBlock2.7.weight".into());
        v.push("UpBlock2.8.rho".into());
        v.push("UpBlock2.8.gamma".into());
        v.push("UpBlock2.8.beta".into());
        v.push("UpBlock2.11.weight".into());
        v
    }

    /// Every parameter the graph reads must be present, so a truncated or
    /// renamed checkpoint fails here, once, with the full list - rather than
    /// part-way through a run on the first name the graph happens to ask for.
    fn check_present(&self) -> Result<(), Error> {
        let missing: Vec<String> = self.expected_names()
            .into_iter()
            .filter(|n| !self.file.contains(n))
            .collect();
        if !missing.is_empty() {
            return Err(Error(format!(
                "checkpoint is missing {} of the {} parameters the graph reads: {}",
                missing.len(),
                self.expected_names().len(),
                missing.join(", ")
            )));
        }
        Ok(())
    }

    pub fn f32(&self, name: &str) -> Result<&[f32], Error> {
        Ok(self.file.f32(name)?)
    }

    pub fn info(&self, name: &str) -> Result<&TensorInfo, Error> {
        Ok(self.file.info(name)?)
    }

    pub fn shape(&self, name: &str) -> Result<&[usize], Error> {
        Ok(self.file.shape(name)?)
    }

    pub fn metadata_get(&self, key: &str) -> Option<&str> {
        self.file.metadata_get(key)
    }
}
