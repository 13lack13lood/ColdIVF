//! Unit tests for the rerank.

use super::*;
use crate::quantization::SplitMix64;
use crate::quantization::testutil::{gaussian, mean};

/// Rows for `n` random vectors, with external ids from 1000, and the query's offset q − μ.
fn rows_and_offset(rng: &mut SplitMix64, n: usize, dim: usize) -> (Vec<u8>, Vec<f32>) {
    let data = gaussian(rng, n, dim, |_| 1.0);
    let mu = mean(&data, dim);
    let rb = sq8::row_bytes(dim);
    let mut rows = vec![0u8; n * rb];
    for (i, (o, row)) in data.chunks_exact(dim).zip(rows.chunks_exact_mut(rb)).enumerate() {
        sq8::write_row(o, &mu, 1000 + i as u64, row);
    }

    let q = gaussian(rng, 1, dim, |_| 1.0);
    let offset = q.iter().zip(&mu).map(|(&x, &m)| x - m).collect();
    (rows, offset)
}

#[test]
fn keeps_the_best_k_by_distance_then_id() {
    let mut rng = SplitMix64::new(5);
    let (n, dim) = (60, 200);
    let (mut rows, offset) = rows_and_offset(&mut rng, n, dim);
    let rb = sq8::row_bytes(dim);
    // Row 9 repeats row 4 with its own id, so their distances tie and internal id 4 must come first.
    let copy = rows[4 * rb..5 * rb - 8].to_vec();
    rows[9 * rb..10 * rb - 8].copy_from_slice(&copy);

    let row = |id: u32| &rows[id as usize * rb..(id as usize + 1) * rb];
    let candidates: Vec<Candidate> = (0..n as u32)
        .step_by(2)
        .chain([9])
        .map(|id| Candidate { score: 0.0, id })
        .collect();
    let norm2 = offset.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>() as f32;
    let mut want: Vec<Candidate> = candidates
        .iter()
        .map(|c| Candidate {
            score: sq8::distance(row(c.id), &offset, norm2),
            id: c.id,
        })
        .collect();
    want.sort_unstable_by(order);
    let (p4, p9) = (want.iter().position(|c| c.id == 4), want.iter().position(|c| c.id == 9));
    assert!(p4 < p9, "the tie between rows 4 and 9 must go to 4");

    let (mut rescored, mut hits) = (Vec::new(), Vec::new());
    for k in [0, 1, 5, candidates.len(), candidates.len() + 3] {
        rerank(&candidates, row, &offset, k, Metric::L2, &mut rescored, &mut hits);
        let got: Vec<(u64, u32)> = hits.iter().map(|h| (h.id, h.score.to_bits())).collect();
        let expected: Vec<(u64, u32)> = want
            .iter()
            .take(k)
            .map(|c| (1000 + u64::from(c.id), c.score.to_bits()))
            .collect();
        assert_eq!(got, expected, "k {k}");
    }
}

#[test]
fn cosine_reports_the_similarity() {
    let mut rng = SplitMix64::new(6);
    let (rows, offset) = rows_and_offset(&mut rng, 10, 128);
    let rb = sq8::row_bytes(128);
    let row = |id: u32| &rows[id as usize * rb..(id as usize + 1) * rb];
    let candidates: Vec<Candidate> = (0..10).map(|id| Candidate { score: 0.0, id }).collect();

    let (mut rescored, mut l2, mut cosine) = (Vec::new(), Vec::new(), Vec::new());
    rerank(&candidates, row, &offset, 10, Metric::L2, &mut rescored, &mut l2);
    rerank(
        &candidates,
        row,
        &offset,
        10,
        Metric::Cosine,
        &mut rescored,
        &mut cosine,
    );
    for (a, b) in l2.iter().zip(&cosine) {
        assert_eq!(a.id, b.id);
        assert_eq!(b.score, 1.0 - a.score / 2.0);
    }
}
