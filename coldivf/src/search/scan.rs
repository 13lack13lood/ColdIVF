//! The scan, which estimates the distance to every vector in a cluster from its 1-bit code and keeps the best
//! candidates.

use super::View;
use super::candidates::{Candidate, Candidates, before};
use crate::kernels::{self, Tier};
use crate::quantization::rabitq::{BLOCK, QueryConsts, QueryLut, block_bytes, block_sums, est, lane_factors};
use crate::quantization::sq4::QueryS32;
use crate::{Anchor, LeafId};

/// Scores one block for the query and writes the lanes that rank before `bound` into `out`, in lane order. Only the
/// first `count` lanes hold vectors, and lane j has internal id `first_id + j`. Each lane's estimate adds the query's
/// t2, or `t2[j]` when `t2` is given. Returns how many it wrote. This is the portable path, and the single instruction,
/// multiple data (SIMD) paths must match it bit for bit.
pub(crate) fn scan_block_scalar(
    block: &[u8],
    query: &QueryLut,
    t2: Option<&[f32; BLOCK]>,
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
        let k = QueryConsts {
            t2: t2.map_or(query.k.t2, |t2| t2[lane]),
            ..query.k
        };
        let c = Candidate {
            score: est(s, &lane_factors(block, d_pad, lane), &k),
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
        let n = kernels::scan_block(tier, block, query, None, first, lanes, candidates.bound(), survivors);
        for &c in &survivors[..n] {
            candidates.offer(c);
        }
    }
}

/// Scans cluster `leaf` of `view`, whose blocks are `blocks`, like `scan_list`. For codes measured from the cluster's
/// center, it first sets the query's t and t2 from its distance to that center, from `q32`. Every search path scans
/// its clusters here.
pub(crate) fn scan_leaf(
    view: &View,
    leaf: LeafId,
    blocks: &[u8],
    query: &mut QueryLut,
    q32: &QueryS32,
    candidates: &mut Candidates,
    survivors: &mut [Candidate; BLOCK],
) {
    let list = view.lists[leaf as usize];

    if list.count == 0 {
        return;
    }

    if view.anchor == Anchor::Cluster {
        query.set_anchor(view.centers.distance2(view.tier, leaf, q32));
    }

    scan_list(
        view.tier,
        blocks,
        list.count,
        list.first_id,
        query,
        candidates,
        survivors,
    );
}

/// Scans blocks of `count` codes whose vectors may each add their own t2, and offers every vector that ranks before
/// the current bound to `candidates`. Vector j has internal id `first_id + j` and adds `t2(j)` in place of the query's
/// t2, eg. the squared distance to the cluster center it was measured from. So each estimate has the bits that a scan
/// of the same code with that t2 gives.
#[allow(clippy::too_many_arguments)]
pub(crate) fn scan_each(
    tier: Tier,
    blocks: &[u8],
    count: usize,
    first_id: u32,
    query: &QueryLut,
    t2: impl Fn(usize) -> f32,
    candidates: &mut Candidates,
    survivors: &mut [Candidate; BLOCK],
) {
    let size = block_bytes(query.lut.len() / 4);
    assert_eq!(blocks.len(), count.div_ceil(BLOCK) * size);
    let mut lanes_t2 = [0f32; BLOCK];

    for (b, block) in blocks.chunks_exact(size).enumerate() {
        let lanes = (count - BLOCK * b).min(BLOCK);
        let first = first_id + (BLOCK * b) as u32;
        for (lane, x) in lanes_t2.iter_mut().enumerate().take(lanes) {
            *x = t2(BLOCK * b + lane);
        }
        let n = kernels::scan_block(
            tier,
            block,
            query,
            Some(&lanes_t2),
            first,
            lanes,
            candidates.bound(),
            survivors,
        );
        for &c in &survivors[..n] {
            candidates.offer(c);
        }
    }
}

#[cfg(test)]
#[path = "../../tests/unit/search/scan.rs"]
mod tests;
