//! The host side of a plan: the input plane, the arena (or as little of it as a
//! backend needs), and the reference crop.
//!
//! A GPU run holds only the INPUT in host memory unless a dump or the verify walk
//! needs the whole arena, which is what [`Host::arena_len`] is for: sizing the
//! host arena from the plan unconditionally would make a GPU run allocate
//! gigabytes it never touches. A CPU run holds the whole arena, obviously.

use crate::image::Image;
use crate::model::Plan;

pub struct Host {
    pub plan: Plan,
    /// The input, already preprocessed to the model's range.
    pub input: Vec<f32>,
    /// The arena. Full length on the CPU, or just long enough for the input on a
    /// GPU-only run.
    pub arena: Vec<f32>,
    /// The crop the pipeline applies to the model's output: `[top, bottom, left,
    /// right]` of the PAD, so the result comes back at the caller's size.
    pub crop: [usize; 4],
}

impl Host {
    /// The arena the host must actually hold: the whole thing when the host
    /// computes with it, otherwise only enough for the input.
    pub fn arena_len(plan: &Plan, full: bool) -> usize {
        if full {
            plan.arena_len
        } else {
            plan.input_len
        }
    }

    pub fn new(plan: Plan, input: Vec<f32>, full_arena: bool, crop: [usize; 4]) -> Host {
        let arena_len = Self::arena_len(&plan, full_arena);
        Host { plan, input, arena: vec![0.0; arena_len], crop }
    }

    /// `(x - 0.5) / 0.5` on an 8-bit image, laid out `[c][h][w]` for the arena.
    pub fn preprocess(img: &Image) -> Vec<f32> {
        img.data.iter().map(|v| (v - 0.5) / 0.5).collect()
    }

    /// The inverse, for turning the model's output back into an image.
    pub fn postprocess(v: &[f32]) -> Vec<f32> {
        v.iter().map(|x| x * 0.5 + 0.5).collect()
    }
}
