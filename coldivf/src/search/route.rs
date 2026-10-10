//! Routing, which picks the clusters a query reads.

use super::candidates::{Candidate, Candidates, order};
use crate::LeafId;
use crate::kernels::{Tier, center_distance2, sq4_dots};
use crate::quantization::sq4::{CenterFactors, QueryS8, QueryS32, route_score};

/// The 4-bit codes of every cluster center, back to back, and the factors stored with them.
#[derive(Clone, Copy)]
pub(crate) struct Centers<'a> {
    pub(crate) codes: &'a [u8],
    pub(crate) factors: &'a [CenterFactors],
    /// Each center's `decoded_norm2`, for codes measured from the clusters' centers. It is empty with the mean anchor.
    pub(crate) norms: &'a [f64],
}

impl<'a> Centers<'a> {
    /// The codes and the factors of cluster `leaf`'s center.
    pub(crate) fn center(&self, leaf: LeafId) -> (&'a [u8], &'a CenterFactors) {
        let (i, size) = (leaf as usize, self.codes.len() / self.factors.len());
        (&self.codes[i * size..(i + 1) * size], &self.factors[i])
    }

    /// ‖q − ĉ‖² for the center ĉ that cluster `leaf`'s codes decode to, from the query's fixed point `q`, on `tier`. It
    /// needs `norms`, so only codes measured from the clusters' centers ask for it.
    pub(crate) fn distance2(&self, tier: Tier, leaf: LeafId, q: &QueryS32) -> f64 {
        let (codes, f) = self.center(leaf);
        center_distance2(tier, codes, f, self.norms[leaf as usize], q)
    }
}

/// Scores every cluster center and leaves the `nprobe` nearest in `routes`, nearest first. Ties go to the lower cluster
/// id. `t2` is T² = ‖q − μ‖², the squared length of the query's offset from the mean, and `dots` has one slot per
/// cluster.
pub(crate) fn route(
    tier: Tier,
    centers: &Centers,
    q8: &QueryS8,
    t2: f32,
    nprobe: usize,
    dots: &mut [i32],
    routes: &mut Vec<Candidate>,
) {
    assert_eq!(dots.len(), centers.factors.len());

    sq4_dots(tier, centers.codes, &q8.q8, dots);

    routes.clear();
    routes.extend(
        centers
            .factors
            .iter()
            .zip(dots.iter())
            .enumerate()
            .map(|(i, (f, &dot))| Candidate {
                score: route_score(t2, f, q8, dot),
                id: i as u32,
            }),
    );

    let nprobe = nprobe.min(routes.len());

    if nprobe == 0 {
        routes.clear();
        return;
    }

    routes.select_nth_unstable_by(nprobe - 1, order);
    routes.truncate(nprobe);
    routes.sort_unstable_by(order);
}

/// The cluster centers that a search scores at a time, so that none of its buffers grows with the cluster count.
pub(crate) const CHUNK: usize = 1024;

/// Scores every cluster center, `CHUNK` at a time, and leaves the `nprobe` nearest in `routes`, nearest first, exactly
/// as `route` does. `dots` has room for `CHUNK` dot products, and `best`, which keeps the nearest as the scores come,
/// for `nprobe` of them, so neither depends on the cluster count.
#[allow(clippy::too_many_arguments)]
pub(crate) fn route_chunked(
    tier: Tier,
    centers: &Centers,
    q8: &QueryS8,
    t2: f32,
    nprobe: usize,
    dots: &mut [i32],
    best: &mut Candidates,
    routes: &mut Vec<Candidate>,
) {
    let size = q8.q8.len() / 2;
    best.reset(nprobe);

    for (c, codes) in centers.codes.chunks(CHUNK * size).enumerate() {
        let dots = &mut dots[..codes.len() / size];
        sq4_dots(tier, codes, &q8.q8, dots);
        for (i, &dot) in dots.iter().enumerate() {
            let id = c * CHUNK + i;
            best.push(Candidate {
                score: route_score(t2, &centers.factors[id], q8, dot),
                id: id as u32,
            });
        }
    }

    routes.clear();
    routes.extend_from_slice(best.as_slice());
    routes.sort_unstable_by(order);
}

/// How many of `routes`, nearest first, lie within `reach`. A cluster is within reach when its distance from the query
/// is at most (1 + reach) times the nearest one's, and the nearest is always within reach. Each score estimates a
/// squared distance, so the test compares it with (1 + reach)² times the nearest score, and a nearest score below 0
/// counts as 0. `None` keeps every route.
pub(crate) fn within_reach(routes: &[Candidate], reach: Option<f32>) -> usize {
    reach_of(routes.iter().map(|r| r.score), reach)
}

/// The same, for the scores of routes nearest first.
pub(crate) fn reach_of(mut scores: impl ExactSizeIterator<Item = f32>, reach: Option<f32>) -> usize {
    let len = scores.len();
    let (Some(reach), Some(first)) = (reach, scores.next()) else {
        return len;
    };
    let limit = f64::from(first.max(0.0)) * (1.0 + f64::from(reach)).powi(2);
    1 + scores.take_while(|&s| f64::from(s) <= limit).count()
}

#[cfg(test)]
#[path = "../../tests/unit/search/route.rs"]
mod tests;
