//! k-means, which finds the cluster centers when a build doesn't import them. Each round gives every vector to its
//! nearest center, then moves each center to the mean of its vectors. The distances come from matrix products of a
//! block of vectors with a block of centers, run on the build's threads.

use std::cmp::Reverse;
use std::io::Error;

use faer::linalg::matmul::matmul;
use faer::{Accum, MatMut, MatRef, Par};
use rayon::ThreadPool;
use rayon::prelude::*;

use crate::quantization::SplitMix64;
use crate::search::{LIMIT, invalid, normalize};
use crate::{Metric, VectorSource};

/// Vectors in one matrix product.
const ROWS: usize = 20_000;

/// Centers in one matrix product.
const CENTERS: usize = 1024;

/// How far a split pushes the two copies of a center apart, as a share of each value.
const SPLIT: f32 = 1.0 / 1024.0;

/// Vectors that k-means reads a block at a time into its own buffer, eg. a source's rows or 8-bit rows decoded to
/// f32.
pub(crate) trait Rows {
    fn count(&self) -> usize;
    fn dim(&self) -> usize;
    /// Writes the rows `ids`, which are in increasing order, one after another into `out`, as k-means clusters them. A
    /// vector that can't be clustered fails with `InvalidInput`, and a damaged one with `InvalidData`.
    fn load(&self, ids: &[u32], out: &mut [f32], pool: &ThreadPool) -> Result<(), Error>;
}

/// The rows of a source, normalized for cosine. A value that isn't finite, a zero vector with cosine, or a vector too
/// long to cluster fails with `InvalidInput`, and the first such vector decides the error, whatever the threads do.
pub(crate) struct Source<'a> {
    pub(crate) src: &'a dyn VectorSource,
    pub(crate) metric: Metric,
}

impl Rows for Source<'_> {
    fn count(&self) -> usize {
        self.src.count()
    }

    fn dim(&self) -> usize {
        self.src.dim()
    }

    fn load(&self, ids: &[u32], out: &mut [f32], pool: &ThreadPool) -> Result<(), Error> {
        let slices: Vec<&[f32]> = ids.iter().map(|&r| self.src.rows(r as usize, 1)).collect();
        let (dim, metric) = (self.dim(), self.metric);
        let mut faults = vec![None; slices.len()];
        pool.install(|| {
            out.par_chunks_mut(dim)
                .zip(faults.par_iter_mut())
                .zip(&slices)
                .for_each(|((row, fault), x)| *fault = copy_row(x, metric, row).err());
        });

        match faults.iter().find_map(|f| *f) {
            Some(message) => Err(invalid(message)),
            None => Ok(()),
        }
    }
}

/// k-means settings.
#[derive(Clone, Copy, Debug)]
pub(crate) struct KMeans {
    /// How many centers to find.
    pub(crate) k: usize,
    /// Rounds of assigning the vectors and moving the centers.
    pub(crate) rounds: usize,
    /// The most vectors to train on. With more vectors than this, it trains on that many picked at random.
    pub(crate) sample: usize,
    /// Keeps every center at length 1.
    pub(crate) spherical: bool,
    pub(crate) seed: u64,
}

/// Trains `params.k` centers on the vectors of `src`, normalized for cosine, and returns them back to back. The first
/// centers are training vectors picked at random. A center left without vectors after a round takes half of the
/// largest cluster.
pub(crate) fn train(
    params: &KMeans,
    src: &dyn VectorSource,
    metric: Metric,
    pool: &ThreadPool,
) -> Result<Vec<f32>, Error> {
    train_rows(params, &Source { src, metric }, ROWS, pool)
}

/// The same for any rows, reading at most `chunk` of them at a time.
pub(crate) fn train_rows(params: &KMeans, src: &dyn Rows, chunk: usize, pool: &ThreadPool) -> Result<Vec<f32>, Error> {
    let (n, dim, k, chunk) = (src.count(), src.dim(), params.k, chunk.max(1));
    let mut rng = SplitMix64::new(params.seed);
    let set = pick(n, params.sample.min(n), &mut rng);

    if k == 0 || k > set.len() {
        return Err(invalid("k-means needs at least one vector for each center"));
    }

    let first: Vec<u32> = pick(set.len(), k, &mut rng).iter().map(|&i| set[i as usize]).collect();
    let mut scratch = Scratch::new(set.len().min(chunk), dim, k, 1);
    scratch.load(src, &first, pool)?;
    let mut centers = scratch.rows.clone();
    let (mut sums, mut counts, mut norms) = (vec![0f64; k * dim], vec![0usize; k], vec![0f32; k]);

    for _ in 0..params.rounds {
        squared_norms(&centers, dim, &mut norms);
        sums.fill(0.0);
        counts.fill(0);
        for chunk in set.chunks(chunk) {
            scratch.load(src, chunk, pool)?;
            scratch.nearest(&centers, &norms, dim, pool);
            for (row, &c) in scratch.rows.chunks_exact(dim).zip(&scratch.best) {
                counts[c as usize] += 1;
                let sum = &mut sums[c as usize * dim..(c as usize + 1) * dim];
                sum.iter_mut().zip(row).for_each(|(s, &x)| *s += f64::from(x));
            }
        }
        update(&mut centers, &sums, &counts, dim, params.spherical);
        split(&mut centers, &mut counts, dim, params.spherical);
    }

    Ok(centers)
}

/// The nearest of `centers` to each vector of `src`, normalized for cosine. Ties go to the lower center.
pub(crate) fn assign(
    centers: &[f32],
    src: &dyn VectorSource,
    metric: Metric,
    pool: &ThreadPool,
) -> Result<Vec<u32>, Error> {
    Ok(nearest(centers, src, metric, 1, pool)?.0)
}

/// The same for any rows, reading at most `chunk` of them at a time.
pub(crate) fn assign_rows(centers: &[f32], src: &dyn Rows, chunk: usize, pool: &ThreadPool) -> Result<Vec<u32>, Error> {
    Ok(nearest_rows(centers, src, 1, chunk, pool)?.0)
}

/// The `m` nearest of `centers` to each vector of `src`, normalized for cosine, nearest first, and their scores
/// ‖c‖² − 2⟨x, c⟩ for the vector x and the center c. A vector's scores rank its centers like their distances do. `m`
/// is at most the number of centers. Ties go to the lower center.
pub(crate) fn nearest(
    centers: &[f32],
    src: &dyn VectorSource,
    metric: Metric,
    m: usize,
    pool: &ThreadPool,
) -> Result<(Vec<u32>, Vec<f32>), Error> {
    nearest_rows(centers, &Source { src, metric }, m, ROWS, pool)
}

/// The same for any rows, reading at most `chunk` of them at a time.
pub(crate) fn nearest_rows(
    centers: &[f32],
    src: &dyn Rows,
    m: usize,
    chunk: usize,
    pool: &ThreadPool,
) -> Result<(Vec<u32>, Vec<f32>), Error> {
    let n = src.count();
    let (mut best, mut scores) = (vec![0u32; n * m], vec![0f32; n * m]);
    scan_nearest(centers, src, m, chunk, pool, |start, _, ids, out| {
        best[start * m..start * m + ids.len()].copy_from_slice(ids);
        scores[start * m..start * m + out.len()].copy_from_slice(out);
    })?;

    Ok((best, scores))
}

/// Finds the `m` nearest of `centers` to each row of `src`, nearest first, at most `chunk` rows at a time, like
/// `nearest`. For each chunk it calls `visit` with the chunk's first row, its rows as k-means read them, and their
/// nearest centers and scores, `m` per row.
pub(crate) fn scan_nearest(
    centers: &[f32],
    src: &dyn Rows,
    m: usize,
    chunk: usize,
    pool: &ThreadPool,
    mut visit: impl FnMut(usize, &[f32], &[u32], &[f32]),
) -> Result<(), Error> {
    let (n, dim, chunk) = (src.count(), src.dim(), chunk.max(1));
    let k = centers.len() / dim;
    let mut norms = vec![0f32; k];
    squared_norms(centers, dim, &mut norms);
    let mut scratch = Scratch::new(n.min(chunk), dim, k, m);
    let mut ids = Vec::with_capacity(n.min(chunk));

    for start in (0..n).step_by(chunk) {
        let count = chunk.min(n - start);
        ids.clear();
        ids.extend(start as u32..(start + count) as u32);
        scratch.load(src, &ids, pool)?;
        scratch.nearest(centers, &norms, dim, pool);
        visit(
            start,
            &scratch.rows,
            &scratch.best[..count * m],
            &scratch.scores[..count * m],
        );
    }

    Ok(())
}

/// The buffers for one block of vectors.
struct Scratch {
    rows: Vec<f32>,
    /// How many nearest centers to keep for each vector.
    m: usize,
    /// Each vector's `m` nearest centers so far, nearest first, and their scores ‖c‖² − 2⟨x, c⟩, for the vector x and
    /// the center c.
    best: Vec<u32>,
    scores: Vec<f32>,
    /// The dot products of the vectors with one block of centers.
    dots: Vec<f32>,
}

impl Scratch {
    fn new(rows: usize, dim: usize, k: usize, m: usize) -> Scratch {
        Scratch {
            rows: Vec::with_capacity(rows * dim),
            m,
            best: vec![0; rows * m],
            scores: vec![0.0; rows * m],
            dots: vec![0.0; rows * k.min(CENTERS)],
        }
    }

    /// Reads the rows `ids` of `src` into `self.rows`.
    fn load(&mut self, src: &dyn Rows, ids: &[u32], pool: &ThreadPool) -> Result<(), Error> {
        self.rows.resize(ids.len() * src.dim(), 0.0);
        src.load(ids, &mut self.rows, pool)
    }

    /// Finds the `self.m` nearest of `centers` to each vector in `self.rows`, where `norms` holds each center's
    /// squared length. The squared distance between a vector x and a center c is ‖x‖² + ‖c‖² − 2⟨x, c⟩, where ⟨x, c⟩
    /// is their dot product. ‖x‖² is the same for every center, so the rest decides.
    fn nearest(&mut self, centers: &[f32], norms: &[f32], dim: usize, pool: &ThreadPool) {
        let (count, m) = (self.rows.len() / dim, self.m);
        let (best, scores) = (&mut self.best[..count * m], &mut self.scores[..count * m]);
        best.fill(0);
        scores.fill(f32::INFINITY);

        if count == 0 {
            return;
        }

        let lhs = MatRef::from_row_major_slice(&self.rows, count, dim);

        for (b, block) in centers.chunks(CENTERS * dim).enumerate() {
            let (kb, base) = (block.len() / dim, b * CENTERS);
            let dots = &mut self.dots[..count * kb];
            let rhs = MatRef::from_column_major_slice(block, dim, kb);
            pool.install(|| {
                let out = MatMut::from_row_major_slice_mut(&mut *dots, count, kb);
                matmul(out, Accum::Replace, lhs, rhs, 1.0, Par::rayon(0));
                let norms = &norms[base..base + kb];
                dots.par_chunks(kb)
                    .zip(best.par_chunks_mut(m).zip(scores.par_chunks_mut(m)))
                    .for_each(|(dots, (best, scores))| {
                        for (j, (&dot, &norm)) in dots.iter().zip(norms).enumerate() {
                            let s = norm - 2.0 * dot;
                            if s < scores[m - 1] {
                                keep(best, scores, s, (base + j) as u32);
                            }
                        }
                    });
            });
        }
    }
}

/// Puts the center `c`, with score `s`, among one vector's nearest centers so far, `best` and `scores`, which are in
/// order. The farthest drops out, and a center already there with the same score stays ahead of `c`.
fn keep(best: &mut [u32], scores: &mut [f32], s: f32, c: u32) {
    let mut i = best.len() - 1;

    while i > 0 && s < scores[i - 1] {
        (best[i], scores[i]) = (best[i - 1], scores[i - 1]);
        i -= 1;
    }

    (best[i], scores[i]) = (c, s);
}

/// `m` different numbers below `n`, picked at random and sorted. All of them when `m` is `n`.
pub(crate) fn pick(n: usize, m: usize, rng: &mut SplitMix64) -> Vec<u32> {
    let mut all: Vec<u32> = (0..n as u32).collect();

    if m < n {
        for i in 0..m {
            let j = i + (rng.next_u64() % (n - i) as u64) as usize;
            all.swap(i, j);
        }
        all.truncate(m);
        all.sort_unstable();
    }

    all
}

/// Copies `x` into `row`, normalized for cosine, or says why it can't be clustered.
fn copy_row(x: &[f32], metric: Metric, row: &mut [f32]) -> Result<(), &'static str> {
    if !x.iter().all(|v| v.is_finite()) {
        return Err("a vector has a value that isn't finite");
    }

    row.copy_from_slice(x);

    if metric == Metric::Cosine && !normalize(row) {
        return Err("a zero vector has no direction for cosine");
    }

    if row.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>() > LIMIT {
        return Err("a vector is too long to cluster");
    }

    Ok(())
}

fn squared_norms(centers: &[f32], dim: usize, out: &mut [f32]) {
    for (c, norm) in centers.chunks_exact(dim).zip(out) {
        *norm = c.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>() as f32;
    }
}

/// Moves each center that has vectors to their mean, at length 1 if `spherical` is set.
pub(crate) fn update(centers: &mut [f32], sums: &[f64], counts: &[usize], dim: usize, spherical: bool) {
    let parts = centers.chunks_exact_mut(dim).zip(sums.chunks_exact(dim)).zip(counts);

    for ((center, sum), &count) in parts.filter(|(_, count)| **count > 0) {
        center
            .iter_mut()
            .zip(sum)
            .for_each(|(c, &s)| *c = (s / count as f64) as f32);
        if spherical {
            normalize(center);
        }
    }
}

/// Gives each center without vectors half of the largest cluster. It copies that cluster's center and pushes the two
/// copies apart by a small share of each value, in opposite directions. Ties go to the lower cluster.
fn split(centers: &mut [f32], counts: &mut [usize], dim: usize, spherical: bool) {
    for empty in 0..counts.len() {
        if counts[empty] > 0 {
            continue;
        }
        let big = (0..counts.len()).max_by_key(|&c| (counts[c], Reverse(c))).unwrap_or(0);
        if counts[big] < 2 {
            return;
        }
        centers.copy_within(big * dim..(big + 1) * dim, empty * dim);
        for j in 0..dim {
            let push = if j % 2 == 0 { SPLIT } else { -SPLIT };
            centers[empty * dim + j] *= 1.0 + push;
            centers[big * dim + j] *= 1.0 - push;
        }
        if spherical {
            normalize(&mut centers[empty * dim..(empty + 1) * dim]);
            normalize(&mut centers[big * dim..(big + 1) * dim]);
        }
        counts[empty] = counts[big] / 2;
        counts[big] -= counts[empty];
    }
}

#[cfg(test)]
#[path = "../../tests/unit/build/kmeans.rs"]
mod tests;
