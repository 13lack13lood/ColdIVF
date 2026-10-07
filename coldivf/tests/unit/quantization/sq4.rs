//! Unit tests for the 4-bit cluster center codes and the center scan.

use super::*;
use crate::kernels::{Tier, sq4_dots};
use crate::quantization::SplitMix64;
use crate::quantization::testutil::{dot, gaussian};

/// The codes of `n` random centers back to back, and their factors.
fn random_centers(rng: &mut SplitMix64, n: usize, dim: usize) -> (Vec<u8>, Vec<CenterFactors>) {
    let cb = code_bytes(padded_dim(dim));
    let zs = gaussian(rng, n, dim, |_| 1.0);
    let mut codes = vec![0u8; n * cb];
    let factors = zs
        .chunks_exact(dim)
        .zip(codes.chunks_exact_mut(cb))
        .map(|(z, c)| encode(z, c))
        .collect();
    (codes, factors)
}

#[test]
fn simd_matches_scalar_exactly() {
    let best = Tier::best();
    #[cfg(target_arch = "aarch64")]
    assert!(
        !best.is_scalar(),
        "every 64-bit Arm CPU this runs on should have dotprod"
    );
    let mut rng = SplitMix64::new(1);
    for dim in [128, 200, 768, 1536] {
        let n = 103;
        let (mut codes, _) = random_centers(&mut rng, n, dim);
        // The last center has every code at 15, the largest products.
        let cb = code_bytes(padded_dim(dim));
        codes[(n - 1) * cb..].fill(0xff);
        let mut query = QueryS8::new(dim);
        query.prepare(&gaussian(&mut rng, 1, dim, |_| 1.0));
        query.q8[..dim].iter_mut().step_by(7).for_each(|q| *q = -127);
        let (mut scalar, mut simd) = (vec![0i32; n], vec![0i32; n]);
        sq4_dots(Tier::SCALAR, &codes, &query.q8, &mut scalar);
        sq4_dots(best, &codes, &query.q8, &mut simd);
        assert_eq!(scalar, simd, "dim {dim}");
    }
}

#[test]
fn codes_follow_the_chunk_layout() {
    let dim = 128;
    // Codes 0 to 15 that differ between neighboring dimensions, between runs of 16 and between chunks.
    let k = |i: usize| ((7 * i + i / 16 + 3) % 16) as u8;
    let z: Vec<f32> = (0..dim).map(|i| f32::from(k(i))).collect();
    let mut codes = vec![0u8; code_bytes(padded_dim(dim))];
    let f = encode(&z, &mut codes);
    assert_eq!((f.lo, f.step), (0.0, 1.0));
    for c in 0..2 {
        for b in 0..16 {
            assert_eq!(codes[32 * c + b], k(64 * c + b) | (k(64 * c + 16 + b) << 4));
            assert_eq!(codes[32 * c + 16 + b], k(64 * c + 32 + b) | (k(64 * c + 48 + b) << 4));
        }
    }
    assert!((0..dim).all(|i| code_at(&codes, i) == k(i)));
}

#[test]
fn each_value_is_within_half_a_step() {
    let mut rng = SplitMix64::new(2);
    // At 200 dimensions the codes have padding, which must stay 0.
    for dim in [200, 768] {
        let zs = gaussian(&mut rng, 20, dim, |_| 1.0);
        let mut codes = vec![0u8; code_bytes(padded_dim(dim))];
        for z in zs.chunks_exact(dim) {
            let f = encode(z, &mut codes);
            for (i, &v) in z.iter().enumerate() {
                let decoded = f64::from(f.lo) + f64::from(f.step) * f64::from(code_at(&codes, i));
                assert!(
                    (f64::from(v) - decoded).abs() <= f64::from(f.step) * (0.5 + 1e-4),
                    "dimension {i}"
                );
            }
            assert!((dim..padded_dim(dim)).all(|i| code_at(&codes, i) == 0));
        }
    }
}

#[test]
fn route_score_matches_the_decoded_dot_product() {
    let mut rng = SplitMix64::new(3);
    for dim in [200, 768] {
        let (codes, factors) = random_centers(&mut rng, 30, dim);
        let y = gaussian(&mut rng, 1, dim, |_| 1.0);
        let t2 = dot(&y, &y) as f32;
        let mut query = QueryS8::new(dim);
        query.prepare(&y);
        let mut dots = vec![0i32; factors.len()];
        sq4_dots(Tier::best(), &codes, &query.q8, &mut dots);
        let cb = code_bytes(padded_dim(dim));
        for ((f, &d), center) in factors.iter().zip(&dots).zip(codes.chunks_exact(cb)) {
            let ip: f64 = (0..dim)
                .map(|i| {
                    let z = f64::from(f.lo) + f64::from(f.step) * f64::from(code_at(center, i));
                    f64::from(query.delta) * f64::from(query.q8[i]) * z
                })
                .sum();
            let expected = f64::from(t2) + f64::from(f.norm2) - 2.0 * ip;
            let got = f64::from(route_score(t2, f, &query, d));
            assert!(
                (got - expected).abs() <= 1e-5 * (f64::from(t2) + f64::from(f.norm2)),
                "dim {dim}"
            );
        }
    }
}

#[test]
fn a_center_with_zero_step_works() {
    let dim = 256;
    let y = gaussian(&mut SplitMix64::new(4), 1, dim, |_| 1.0);
    let t2 = dot(&y, &y) as f32;
    let mut query = QueryS8::new(dim);
    query.prepare(&y);
    let mut codes = vec![0u8; code_bytes(padded_dim(dim))];
    for value in [0.25f32, 0.0] {
        let f = encode(&vec![value; dim], &mut codes);
        assert_eq!(f.step, 0.0);
        assert!(codes.iter().all(|&c| c == 0));
        let mut d = [0i32];
        sq4_dots(Tier::best(), &codes, &query.q8, &mut d);
        let score = route_score(t2, &f, &query, d[0]);
        assert_eq!(score, t2 + f.norm2 - 2.0 * (f.lo * query.s_q));
    }
}
