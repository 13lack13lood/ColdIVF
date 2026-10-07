//! Routing, which picks the clusters a query reads.

use super::candidates::{Candidate, order};
use crate::kernels::{Tier, sq4_dots};
use crate::quantization::sq4::{CenterFactors, QueryS8, route_score};

/// The 4-bit codes of every cluster center, back to back, and the factors stored with them.
pub(crate) struct Centers<'a> {
    pub(crate) codes: &'a [u8],
    pub(crate) factors: &'a [CenterFactors],
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

#[cfg(test)]
#[path = "../../tests/unit/search/route.rs"]
mod tests;
