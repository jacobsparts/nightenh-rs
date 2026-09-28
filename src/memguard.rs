//! The CPU backend's pre-allocation guard.
//!
//! Every engine in this family that allocates a host arena checks first, because
//! the failure mode is not an allocation error: a machine that is short of RAM
//! does not fail, it swaps, and a run that swaps takes hours instead of seconds
//! and looks like a hang rather than a refusal. So the arithmetic is done from
//! `/proc/meminfo` BEFORE anything is allocated, and the refusal quotes the
//! numbers it used.
//!
//! What is counted: the arena (the plan's own packed size), the weights (the
//! resident, transposed copies the kernels read), a scratch estimate, and a
//! margin on top - because a check that passes exactly at the limit still swaps,
//! since other things on the machine are also allocating.

use crate::Error;

/// The margin over the plan's own total, as a percentage. 5% is what the sibling
/// engines use and is enough to cover the allocator's own overhead, a dump's
/// buffers and the run's temporary host copies without making the check
/// meaningless on a machine that is genuinely full.
pub const SLACK_PCT: usize = 5;

/// One mebibyte, for the refusal text. Named rather than written as `1 << 20` at
/// each use so the conversion cannot drift between the lines of one message.
const MIB: f64 = (1 << 20) as f64;

/// Bytes available for a large allocation, from `MemAvailable`.
///
/// `MemAvailable` rather than `MemFree`: the kernel's own estimate includes
/// reclaimable page cache, and using `MemFree` would refuse runs that would
/// actually have fitted on every machine that has been up for a while.
pub fn available_bytes() -> Result<usize, Error> {
    let text = std::fs::read_to_string("/proc/meminfo")
        .map_err(|e| Error(format!("read /proc/meminfo: {e}")))?;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("MemAvailable:") {
            let kb: usize = rest
                .trim()
                .split_whitespace()
                .next()
                .and_then(|s| s.parse().ok())
                .ok_or_else(|| Error("MemAvailable is not a number".into()))?;
            return Ok(kb * 1024);
        }
    }
    Err(Error("no MemAvailable line in /proc/meminfo".into()))
}

/// The total a CPU run needs, and the three parts of it.
pub fn need_bytes(arena_len: usize, weights_bytes: usize, scratch_bytes: usize) -> usize {
    let base = arena_len * 4 + weights_bytes + scratch_bytes;
    base + base / 100 * SLACK_PCT
}

/// Refuse before allocating if the machine cannot hold the run.
///
/// `what` names the pass (its padded size), so the message says which run was
/// refused rather than only how much it wanted.
pub fn check(what: &str, arena_len: usize, weights_bytes: usize, scratch_bytes: usize)
    -> Result<(), Error>
{
    let need = need_bytes(arena_len, weights_bytes, scratch_bytes);
    let have = available_bytes()?;
    if need > have {
        return Err(Error(format!(
            "not enough memory for a {what} pass on the CPU\n\
             nightenh: it needs {:.0} MiB (arena {} + weights {} + scratch {}, plus {}% slack)\n\
             nightenh: {:.0} MiB is available now\n\
             nightenh: a smaller image, a shorter checkpoint or a freer machine is what fits",
            need as f64 / MIB,
            arena_len as f64 * 4.0 / MIB,
            weights_bytes as f64 / MIB,
            scratch_bytes as f64 / MIB,
            SLACK_PCT,
            have as f64 / MIB,
        )));
    }
    Ok(())
}

/// The scratch a CPU run needs beyond the arena: the resize's intermediate when
/// the pipeline resizes in and out, plus one image's worth for the crop.
///
/// It is an estimate and is labelled as one - unlike the arena, which is the
/// plan's own packed size. The 5% slack above covers the case where this
/// under-counts something small.
pub fn scratch_bytes(c: usize, h: usize, w: usize) -> usize {
    4 * c * h * w * 2
}
