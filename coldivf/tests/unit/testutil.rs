//! Random data and exact reference math for the unit tests.

use super::SplitMix64;
use crate::{BuildParams, Clusters, LeafOrder, Metric, VectorSource};

impl SplitMix64 {
    /// A uniform value in [0, 1).
    pub(crate) fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// A standard normal value, by the Box-Muller transform.
    pub(crate) fn normal(&mut self) -> f64 {
        let u1 = 1.0 - self.uniform();
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

/// `n` random vectors of `dim` values, row-major. Axis k has standard deviation `sigma(k)`.
pub(crate) fn gaussian(rng: &mut SplitMix64, n: usize, dim: usize, sigma: impl Fn(usize) -> f64) -> Vec<f32> {
    (0..n * dim).map(|i| (rng.normal() * sigma(i % dim)) as f32).collect()
}

/// The mean of the row-major vectors in `rows`, each `dim` values long.
pub(crate) fn mean(rows: &[f32], dim: usize) -> Vec<f32> {
    let n = rows.len() / dim;
    let mut sum = vec![0f64; dim];
    for row in rows.chunks_exact(dim) {
        for (s, &v) in sum.iter_mut().zip(row) {
            *s += f64::from(v);
        }
    }
    sum.iter().map(|s| (s / n as f64) as f32).collect()
}

pub(crate) fn dot(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b).map(|(&x, &y)| f64::from(x) * f64::from(y)).sum()
}

pub(crate) fn dist2(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| (f64::from(x) - f64::from(y)).powi(2))
        .sum()
}

/// Rows held in RAM, as a `VectorSource`.
pub(crate) struct Rows {
    pub(crate) dim: usize,
    pub(crate) data: Vec<f32>,
}

impl VectorSource for Rows {
    fn count(&self) -> usize {
        self.data.len() / self.dim
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn rows(&self, start: usize, rows: usize) -> &[f32] {
        &self.data[start * self.dim..(start + rows) * self.dim]
    }
}

/// `n` vectors around `nlist` random centers. Each center's values have standard deviation 1, and each vector's
/// offset from its center has standard deviation `spread`. The clustering puts each vector in its nearest center's
/// cluster, and all clusters in one group.
pub(crate) fn clustered(rng: &mut SplitMix64, n: usize, dim: usize, nlist: usize, spread: f64) -> (Rows, Clusters) {
    let centroids = gaussian(rng, nlist, dim, |_| 1.0);
    let mut data = Vec::with_capacity(n * dim);
    for _ in 0..n {
        let c = (rng.next_u64() % nlist as u64) as usize;
        data.extend(
            centroids[c * dim..(c + 1) * dim]
                .iter()
                .map(|&x| x + (rng.normal() * spread) as f32),
        );
    }

    let assignment = data.chunks_exact(dim).map(|v| nearest(&centroids, dim, v)).collect();
    let clusters = Clusters {
        dim,
        parents: mean(&centroids, dim),
        parent_of: vec![0; nlist],
        centroids,
        assignment,
    };
    (Rows { dim, data }, clusters)
}

/// The row of `rows` nearest to `v`, by exact squared distance. Ties go to the lower row.
pub(crate) fn nearest(rows: &[f32], dim: usize, v: &[f32]) -> u32 {
    let mut best = (f64::INFINITY, 0u32);
    for (i, r) in rows.chunks_exact(dim).enumerate() {
        let d = dist2(r, v);
        if d < best.0 {
            best = (d, i as u32);
        }
    }

    best.1
}

/// Build settings for the tests, with the given metric.
pub(crate) fn params(metric: Metric) -> BuildParams {
    BuildParams {
        metric,
        vectors_per_list: 160,
        parents: 1,
        order: LeafOrder::Id,
        page_size: 4096,
        threads: 1,
        seed: 7,
    }
}

/// Shuffles `v` in place.
pub(crate) fn shuffle<T>(v: &mut [T], rng: &mut SplitMix64) {
    for i in (1..v.len()).rev() {
        v.swap(i, (rng.next_u64() % (i as u64 + 1)) as usize);
    }
}
