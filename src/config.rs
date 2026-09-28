//! What a checkpoint has to say for itself.
//!
//! The architecture is not a flag: the converter writes `n_res`, `ngf` and the
//! variant into the safetensors header after reading the model's own
//! `conv_block` modules, so a file that does not carry them is refused rather
//! than guessed at. This module is the one place that reads those keys, so
//! `main.rs`, the tests and any future tool agree on what a checkpoint is.

use crate::weights::Weights;
use crate::Error;

/// A validated checkpoint: what the file says it is, plus the sizes the plan
/// builder needs.
#[derive(Debug, Clone, PartialEq)]
pub struct Variant {
    pub n_res: usize,
    pub ngf: usize,
    pub variant: String,
    /// The channel count the width above was measured against - 64 in both
    /// released checkpoints, and not derived from `ngf` because a checkpoint with
    /// a different input width would be a different network.
    pub in_nc: usize,
}

impl Variant {
    pub fn from_weights(w: &Weights) -> Result<Variant, Error> {
        let in_nc = w.metadata_get("in_nc").unwrap_or("3").parse::<usize>()
            .map_err(|e| Error(format!("in_nc in the checkpoint header: {e}")))?;
        if in_nc != 3 {
            return Err(Error(format!(
                "in_nc {in_nc}: this engine handles three-channel images"
            )));
        }
        Ok(Variant { n_res: w.n_res, ngf: w.ngf, variant: w.variant.clone(), in_nc })
    }

    /// True for the three-stage checkpoints... which do not exist here: both
    /// released generators are two-stage (down to 1/4, back up), so this returns
    /// false for every file that can be loaded.
    ///
    /// It is kept as a NAMED predicate rather than deleted, because the thing it
    /// guards is real: a third stage would change the arena's shape and every
    /// buffer's length, and `model::build` would have to size from it. A future
    /// checkpoint must say which it is explicitly rather than have the count
    /// inferred by accident from a key set that both forms share.
    pub fn is_three_stage(&self) -> bool {
        false
    }
}
