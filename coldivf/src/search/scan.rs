//! The scan, which estimates the distance to every vector in a cluster from its 1-bit code and keeps the best
//! candidates.

use super::candidates::{Candidate, Candidates, before};
use crate::kernels::{self, Tier};
use crate::quantization::rabitq::{BLOCK, QueryLut, block_bytes, block_sums, est, lane_factors};

/// Scores one block for the query and writes the lanes that rank before `bound` into `out`, in lane order. Only the
/// first `count` lanes hold vectors, and lane j has internal id `first_id + j`. Returns how many it wrote. This is the
/// portable path, and the SIMD paths must match it bit for bit.
pub(crate) fn scan_block_scalar(
    block: &[u8],
    query: &QueryLut,
    first_id: u32,
    count: usize,
    bound: Candidate,
    out: &mut [Candidate; BLOCK],
) -> usize {
    let d_pad = query.lut.len() / 4;
    let mut sums = [0u32; BLOCK];
    block_sums(block, &query.lut, d_pad, &mut sums);
    let mut n = 0;

    for (lane, &s) in sums.iter().enumerate().take(count) {
        let c = Candidate {
            score: est(s, &lane_factors(block, d_pad, lane), &query.k),
            id: first_id + lane as u32,
        };
        if before(c, bound) {
            out[n] = c;
            n += 1;
        }
    }

    n
}

/// Scans one cluster's blocks and offers each vector that ranks before the current bound to `candidates`. The cluster
/// holds `count` vectors, with internal ids from `first_id`.
pub(crate) fn scan_list(
    tier: Tier,
    blocks: &[u8],
    count: usize,
    first_id: u32,
    query: &QueryLut,
    candidates: &mut Candidates,
    survivors: &mut [Candidate; BLOCK],
) {
    let size = block_bytes(query.lut.len() / 4);
    assert_eq!(blocks.len(), count.div_ceil(BLOCK) * size);

    for (b, block) in blocks.chunks_exact(size).enumerate() {
        let lanes = (count - BLOCK * b).min(BLOCK);
        let first = first_id + (BLOCK * b) as u32;
        let n = kernels::scan_block(tier, block, query, first, lanes, candidates.bound(), survivors);
        for &c in &survivors[..n] {
            candidates.offer(c);
        }
    }
}

#[cfg(test)]
#[path = "../../tests/unit/search/scan.rs"]
mod tests;
