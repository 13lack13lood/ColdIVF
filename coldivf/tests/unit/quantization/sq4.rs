//! Unit tests for the 4-bit cluster center codes and the center scan.

use super::*;
use crate::kernels::{Tier, center_distance2, sq4_dots};
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
    #[cfg(target_arch = "aarch64")]
    assert!(
        !Tier::best().is_scalar(),
        "every 64-bit Arm CPU this runs on should have dotprod"
    );
    let mut rng = SplitMix64::new(1);
    // 64, 150 and 300 dimensions pad to 1, 3 and 5 chunks of 64, so a kernel that takes two chunks a step ends on one.
    // 1 to 5 centers leave every remainder of the kernels that score four centers a pass.
    for dim in [64, 128, 150, 200, 300, 768, 1536] {
        for n in [1, 2, 3, 4, 5, 103] {
            let (mut codes, _) = random_centers(&mut rng, n, dim);
            // The last center has every code at 15, the largest products.
            let cb = code_bytes(padded_dim(dim));
            codes[(n - 1) * cb..].fill(0xff);
            let mut query = QueryS8::new(dim);
            query.prepare(&gaussian(&mut rng, 1, dim, |_| 1.0));
            // The largest values a byte holds, −128 included, which the fixed-point digits can take.
            query.q8[..dim].iter_mut().step_by(7).for_each(|q| *q = -128);
            query.q8[3..dim].iter_mut().step_by(11).for_each(|q| *q = 127);
            let mut scalar = vec![0i32; n];
            sq4_dots(Tier::SCALAR, &codes, &query.q8, &mut scalar);
            for tier in Tier::simd() {
                let mut simd = vec![0i32; n];
                sq4_dots(tier, &codes, &query.q8, &mut simd);
                assert_eq!(scalar, simd, "{} at dim {dim} with {n} centers", tier.name());
            }
        }
    }
}

#[test]
fn the_largest_products_add_up_exactly_on_every_tier() {
    // Every code at 15 against every query value at −128 or 127, so each product is as large as it gets, and the
    // whole dot product is 15·d_pad·q.
    for d_pad in [768, 1536] {
        let codes = vec![0xffu8; 2 * code_bytes(d_pad)];
        for q in [-128i8, 127] {
            let q8 = vec![q; d_pad];
            for tier in Tier::supported() {
                let mut dots = [0i32; 2];
                sq4_dots(tier, &codes, &q8, &mut dots);
                assert_eq!(
                    dots,
                    [15 * d_pad as i32 * i32::from(q); 2],
                    "{} at d_pad {d_pad}",
                    tier.name()
                );
            }
        }
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

#[test]
fn each_fixed_point_value_rebuilds_from_its_digits_within_the_bound() {
    let mut rng = SplitMix64::new(5);
    for dim in [200, 768] {
        let d_pad = padded_dim(dim);
        // With 1.995, 2^(s + 1)·1.995 falls between the largest value the digits hold and 2^31, and 1.9999999 meets
        // the bound nearly exactly. 2^−40 and 3e38 need s far above 31 and far below 0.
        for top in [1.0f32, 1.995, 1.999_999_9, 2f32.powi(-40), 3e38] {
            let mut y: Vec<f32> = gaussian(&mut rng, 1, dim, |_| 0.3)
                .iter()
                .map(|&v| v.clamp(-0.9, 0.9) * top)
                .collect();
            (y[7], y[9]) = (top, -top);
            let mut q = QueryS32::new(dim);
            q.prepare(&y, 0.0);
            let planes: Vec<&[i8]> = q.planes().chunks_exact(d_pad).collect();
            let mut sum = 0i64;
            for i in 0..d_pad {
                let fixed = planes.iter().rev().fold(0i64, |y, p| 256 * y + i64::from(p[i]));
                sum += fixed;
                let want = y.get(i).map_or(0.0, |&v| f64::from(v));
                assert!(
                    (fixed as f64 * q.unit - want).abs() <= f64::from(top) / FIXED_MAX,
                    "value {i} of {top} at dim {dim}: {} against {want}",
                    fixed as f64 * q.unit
                );
            }
            assert_eq!(q.sum, sum as f64 * q.unit, "{top} at dim {dim}");
        }
    }
}

#[test]
fn the_distance_to_a_decoded_center_is_exact_on_every_tier() {
    let mut rng = SplitMix64::new(6);
    // 150 dimensions pad to 3 chunks of 64, so a kernel that takes two chunks a step ends on one.
    for dim in [150, 200, 768] {
        let cb = code_bytes(padded_dim(dim));
        let (mut codes, factors) = random_centers(&mut rng, 20, dim);
        // The last center has every code at 15, the largest products, and its padding at 0 as `encode` leaves it.
        codes[19 * cb..].fill(0xff);
        for i in dim..padded_dim(dim) {
            let (byte, shift) = position(i);
            codes[19 * cb + byte] &= !(0x0f << shift);
        }
        let norms = decoded_norms(&codes, &factors, dim);
        let y = gaussian(&mut rng, 1, dim, |_| 1.0);
        let mut q = QueryS32::new(dim);
        q.prepare(&y, dot(&y, &y));
        for ((center, f), &norm2) in codes.chunks_exact(cb).zip(&factors).zip(&norms) {
            let z = |i: usize| f64::from(f.lo) + f64::from(f.step) * f64::from(code_at(center, i));
            let exact: f64 = (0..dim).map(|i| (f64::from(y[i]) - z(i)).powi(2)).sum();
            assert!((norm2 - (0..dim).map(|i| z(i).powi(2)).sum::<f64>()).abs() <= 1e-12 * norm2);
            let scalar = center_distance2(Tier::SCALAR, center, f, norm2, &q);
            assert!(
                (scalar - exact).abs() <= 1e-9 * exact,
                "dim {dim}: {scalar} against {exact}"
            );
            for tier in Tier::simd() {
                let simd = center_distance2(tier, center, f, norm2, &q);
                assert_eq!(simd.to_bits(), scalar.to_bits(), "{} at dim {dim}", tier.name());
            }
        }
    }
}

#[test]
fn every_digit_gives_the_same_dot_products_on_every_tier() {
    // With lo 0, step 1, a unit of 1 and nothing else in the query, the distance is 10^14 − 2·Σ_t 256^t·dots[t]. Every
    // term is an integer below 2^53, so the distance is exact and shows any difference in the dot products. The digits
    // run through −128 and 127, the largest products, and random bytes, in every padded dimension.
    let mut rng = SplitMix64::new(7);
    let f = CenterFactors {
        lo: 0.0,
        step: 1.0,
        norm2: 0.0,
    };

    for dim in [150, 768] {
        let (d_pad, cb) = (padded_dim(dim), code_bytes(padded_dim(dim)));
        let mut codes = vec![0xffu8; 2 * cb];
        codes[cb..].iter_mut().for_each(|c| *c = rng.next_u64() as u8);
        let mut q = QueryS32::new(dim);
        (q.unit, q.sum, q.t2) = (1.0, 0.0, 0.0);
        for (t, plane) in q.planes.chunks_exact_mut(d_pad).enumerate() {
            for (i, digit) in plane.iter_mut().enumerate() {
                *digit = match (i + t) % 4 {
                    0 => 127,
                    1 => rng.next_u64() as i8,
                    _ => -128,
                };
            }
        }
        for center in codes.chunks_exact(cb) {
            let scalar = center_distance2(Tier::SCALAR, center, &f, 1e14, &q);
            for tier in Tier::simd() {
                let simd = center_distance2(tier, center, &f, 1e14, &q);
                assert_eq!(simd.to_bits(), scalar.to_bits(), "{} at dim {dim}", tier.name());
            }
        }
    }
}
