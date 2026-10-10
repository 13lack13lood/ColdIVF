//! The scan, which estimates the distance to every vector in a cluster from its 1-bit code and keeps the best
//! candidates. A vector that a change hid, as of the search's snapshot of the changes, never becomes a candidate, so
//! it takes no place among the rows that the search rescores.

use std::cell::Cell;

use super::View;
use super::candidates::{Candidate, Candidates, before};
use crate::kernels::{self, Tier};
use crate::quantization::rabitq::{
    BLOCK, QueryConsts, QueryLut, block_bytes, block_sums, code_bytes, est, lane_factors, plane_sum,
};
use crate::quantization::sq4::QueryS32;
use crate::update::delta::Marks;
use crate::update::view::Group;
use crate::update::{Delta, Seen};
use crate::{Anchor, LeafId};

/// The rows of a data file that a search skips: those that a change before `next` hid.
#[derive(Clone, Copy)]
pub(crate) struct Hides<'a> {
    pub(crate) marks: Option<&'a Marks>,
    pub(crate) next: u64,
}

impl<'a> Hides<'a> {
    /// Nothing hidden, eg. in an index without changes.
    pub(crate) const NONE: Hides<'static> = Hides { marks: None, next: 0 };

    /// The rows that `delta` hid.
    pub(crate) fn of(delta: &'a Delta) -> Hides<'a> {
        Hides {
            marks: delta.marks(),
            next: delta.next(),
        }
    }

    /// Whether row `id` stays visible. Only a row whose bit some change set costs more than one bit test.
    #[inline]
    pub(crate) fn keeps(&self, id: u32) -> bool {
        self.marks.is_none_or(|m| !m.hidden(id, self.next))
    }
}

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

/// Scans one cluster's blocks and offers each vector that ranks before the current bound, and that `hides` keeps, to
/// `candidates`. The cluster holds `count` vectors, with internal ids from `first_id`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn scan_list(
    tier: Tier,
    blocks: &[u8],
    count: usize,
    first_id: u32,
    query: &QueryLut,
    hides: Hides,
    candidates: &mut Candidates,
    survivors: &mut [Candidate; BLOCK],
) {
    let size = block_bytes(query.lut.len() / 4);
    assert_eq!(blocks.len(), count.div_ceil(BLOCK) * size);

    for (b, block) in blocks.chunks_exact(size).enumerate() {
        let lanes = (count - BLOCK * b).min(BLOCK);
        let first = first_id + (BLOCK * b) as u32;
        let n = kernels::scan_block(tier, block, query, None, first, lanes, candidates.bound(), survivors);
        for &c in survivors[..n].iter().filter(|c| hides.keeps(c.id)) {
            candidates.offer(c);
        }
    }
}

/// The most pending inserts of a group that a scan scores one by one, from their unpacked codes and the query's bit
/// planes, instead of with a block of 32 lanes that they would leave mostly empty. That reads far fewer bytes and gives
/// the same sums.
pub(crate) const FEW: usize = 6;

/// Scans the pending inserts of `group` and offers each one that ranks before the current bound, and that `seen` still
/// shows, to `candidates`. An insert's internal id is `first`, the data file's count, plus its place. The kernel gets
/// lane numbers as ids, which are at most the real ones, so its test against the bound lets through every lane that
/// the real id would, and `offer` makes the exact test. A group of `FEW` or fewer is scored one by one with the
/// query's bit planes, which `make_planes` must have made.
#[allow(clippy::too_many_arguments)]
pub(crate) fn scan_group(
    tier: Tier,
    group: Group,
    first: u32,
    query: &QueryLut,
    delta: &Delta,
    seen: Seen,
    candidates: &mut Candidates,
    survivors: &mut [Candidate; BLOCK],
) {
    if group.len() <= FEW {
        let (codes, factors) = group.lanes();
        let cb = code_bytes(query.lut.len() / 4);

        for (j, (code, f)) in codes.chunks_exact(cb).zip(factors).enumerate() {
            let c = Candidate {
                score: est(plane_sum(code, &query.planes), f, &query.k),
                id: first + group.place(j),
            };
            if before(c, candidates.bound()) && delta.insert_visible(group.place(j) as usize, seen) {
                candidates.offer(c);
            }
        }

        return;
    }

    let size = block_bytes(query.lut.len() / 4);

    for (b, block) in group.blocks().chunks_exact(size).enumerate() {
        let lanes = (group.len() - BLOCK * b).min(BLOCK);
        let n = kernels::scan_block(tier, block, query, None, 0, lanes, candidates.bound(), survivors);
        for c in &survivors[..n] {
            let place = group.place(BLOCK * b + c.id as usize);
            if delta.insert_visible(place as usize, seen) {
                candidates.offer(Candidate {
                    score: c.score,
                    id: first + place,
                });
            }
        }
    }
}

/// Scans cluster `leaf` of `view`, whose blocks are `blocks`, like `scan_list`. For codes measured from the cluster's
/// center, it first sets the query's t and t2 from its distance to that center, from `q32`.
pub(crate) fn scan_leaf(
    view: &View,
    leaf: LeafId,
    blocks: &[u8],
    query: &mut QueryLut,
    q32: &QueryS32,
    candidates: &mut Candidates,
    survivors: &mut [Candidate; BLOCK],
) {
    scan_probed(view, leaf, blocks, query, q32, None, candidates, survivors);
}

/// The pending inserts that a search scans with the clusters it probes, and with those that the route ranks next, as of
/// the snapshot `delta`, which it saw as `seen`. Their internal ids start at `first`, the data file's count.
#[derive(Clone, Copy)]
pub(crate) struct Pending<'a> {
    pub(crate) delta: &'a Delta,
    pub(crate) seen: Seen,
    pub(crate) first: u32,
    /// The vectors and the inserts of the clusters scanned so far, and the inserts of the clusters past them whose
    /// inserts alone were scanned.
    pub(crate) lanes: &'a Cell<[u64; 3]>,
}

impl Pending<'_> {
    /// Asks the CPU to bring every cache line of cluster `leaf`'s inserts into its caches, those that a scan reads, so
    /// that their memory comes in before the scan needs it, eg. while it scans the cluster before.
    #[inline]
    pub(crate) fn touch(&self, leaf: LeafId) {
        for g in self.delta.groups(leaf) {
            match g.len() <= FEW {
                true => {
                    let (codes, factors) = g.lanes();
                    kernels::prefetch(codes);
                    kernels::prefetch(factors);
                }
                false => kernels::prefetch(g.blocks()),
            }
        }
    }
}

/// The same, then the cluster's inserts in `pending`, which are measured from the same center, so the query's
/// distance to it serves both. A cluster whose own blocks hold no vector may still hold inserts. Every search path
/// scans its clusters here.
#[allow(clippy::too_many_arguments)]
pub(crate) fn scan_probed(
    view: &View,
    leaf: LeafId,
    blocks: &[u8],
    query: &mut QueryLut,
    q32: &QueryS32,
    pending: Option<Pending>,
    candidates: &mut Candidates,
    survivors: &mut [Candidate; BLOCK],
) {
    let list = view.lists[leaf as usize];
    let inserts = pending.filter(|p| p.delta.groups(leaf).next().is_some());

    if list.count == 0 && inserts.is_none() {
        return;
    }

    if view.anchor == Anchor::Cluster {
        query.set_anchor(view.centers.distance2(view.tier, leaf, q32));
    }

    if list.count > 0 {
        scan_list(
            view.tier,
            blocks,
            list.count,
            list.first_id,
            query,
            view.hides,
            candidates,
            survivors,
        );
    }

    if let Some(p) = pending {
        let mut lanes = p.lanes.get();
        lanes[0] += list.count as u64;
        p.lanes.set(lanes);
    }

    if let Some(p) = inserts {
        scan_groups(view.tier, leaf, query, p, 1, candidates, survivors);
    }
}

/// Scans only the pending inserts of cluster `leaf`, in `pending`, eg. of a cluster that the route ranks just past the
/// ones a search reads, whose file vectors it skips. For codes measured from the cluster's center, it first sets the
/// query's t and t2 from its distance to that center, from `q32`, when the cluster holds an insert.
pub(crate) fn scan_inserts(
    view: &View,
    leaf: LeafId,
    query: &mut QueryLut,
    q32: &QueryS32,
    pending: Pending,
    candidates: &mut Candidates,
    survivors: &mut [Candidate; BLOCK],
) {
    if pending.delta.groups(leaf).next().is_none() {
        return;
    }

    if view.anchor == Anchor::Cluster {
        query.set_anchor(view.centers.distance2(view.tier, leaf, q32));
    }

    scan_groups(view.tier, leaf, query, pending, 2, candidates, survivors);
}

/// Scans every group of cluster `leaf`'s pending inserts with the query as it is set, and counts them in lane counter
/// `counter` of `p`. It first makes the query's bit planes when a group is small enough to score one by one.
fn scan_groups(
    tier: Tier,
    leaf: LeafId,
    query: &mut QueryLut,
    p: Pending,
    counter: usize,
    candidates: &mut Candidates,
    survivors: &mut [Candidate; BLOCK],
) {
    if p.delta.groups(leaf).any(|g| g.len() <= FEW) {
        query.make_planes();
    }

    let mut lanes = p.lanes.get();

    for g in p.delta.groups(leaf) {
        lanes[counter] += g.len() as u64;
        scan_group(tier, g, p.first, query, p.delta, p.seen, candidates, survivors);
    }

    p.lanes.set(lanes);
}

#[cfg(test)]
#[path = "../../tests/unit/search/scan.rs"]
mod tests;
