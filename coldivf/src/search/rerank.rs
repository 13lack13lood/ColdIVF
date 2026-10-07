//! The rerank, which rescores the best candidates with their 8-bit rows and returns the top `k` as hits.

use super::candidates::{Candidate, order};
use crate::quantization::sq8;
use crate::{Hit, Metric};

/// Rescores `candidates` with their rows and writes the best `k` into `hits`, by squared distance and then internal id.
/// `row(id)` returns a vector's row, and `offset` is the query's offset q − μ from the index's mean μ. L2 hits report
/// the squared distance d². Cosine hits report the similarity 1 − d²/2, where d² is between the normalized vectors.
pub(crate) fn rerank<'r>(
    candidates: &[Candidate],
    row: impl Fn(u32) -> &'r [u8],
    offset: &[f32],
    k: usize,
    metric: Metric,
    rescored: &mut Vec<Candidate>,
    hits: &mut Vec<Hit>,
) {
    let dim = offset.len();
    let norm2 = offset.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>() as f32;

    rescored.clear();
    rescored.extend(candidates.iter().map(|c| Candidate {
        score: sq8::distance(row(c.id), offset, norm2),
        id: c.id,
    }));

    if k == 0 {
        rescored.clear();
    } else if rescored.len() > k {
        rescored.select_nth_unstable_by(k - 1, order);
        rescored.truncate(k);
    }

    rescored.sort_unstable_by(order);

    hits.clear();
    hits.extend(rescored.iter().map(|c| Hit {
        id: sq8::id(row(c.id), dim),
        score: match metric {
            Metric::L2 => c.score,
            Metric::Cosine => 1.0 - c.score / 2.0,
        },
    }));
}

#[cfg(test)]
#[path = "../../tests/unit/search/rerank.rs"]
mod tests;
