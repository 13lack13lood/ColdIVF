//! Random data and exact reference math for the unit tests.

use super::SplitMix64;

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
