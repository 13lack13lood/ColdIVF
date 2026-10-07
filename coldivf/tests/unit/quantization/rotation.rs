//! Unit tests for the rotation.

use super::*;
use crate::quantization::testutil::{dot, gaussian};

fn rotate(rotation: &Rotation, x: &[f32]) -> Vec<f32> {
    let (mut out, mut tmp) = (vec![0f32; x.len()], vec![0f32; x.len()]);
    rotation.apply(x, &mut out, &mut tmp);
    out
}

/// The plain definition, built from the dimension and the stored sign bytes alone. Each round negates the
/// coordinates whose bit is set, transforms and then scales each block, then shuffles.
fn reference(rotation: &Rotation, x: &[f32]) -> Vec<f32> {
    let dim = rotation.dim();
    let mut sizes = Vec::new();
    let mut left = dim;
    while left > 0 {
        let size = 1usize << (usize::BITS - 1 - left.leading_zeros());
        sizes.push(size);
        left -= size;
    }
    let stride = (7..)
        .step_by(2)
        .find(|&k: &usize| (2..=k).all(|f| !k.is_multiple_of(f) || !dim.is_multiple_of(f)))
        .unwrap();
    let mut v = x.to_vec();
    for round in rotation.signs().chunks_exact(dim.div_ceil(8)) {
        for (j, x) in v.iter_mut().enumerate() {
            if (round[j / 8] >> (j % 8)) & 1 == 1 {
                *x = -*x;
            }
        }
        let mut start = 0;
        for &size in &sizes {
            let block = &mut v[start..start + size];
            let mut h = 1;
            while h < size {
                for pair in block.chunks_exact_mut(2 * h) {
                    let (lo, hi) = pair.split_at_mut(h);
                    for (a, b) in lo.iter_mut().zip(hi) {
                        (*a, *b) = (*a + *b, *a - *b);
                    }
                }
                h *= 2;
            }
            let scale = (1.0 / (size as f64).sqrt()) as f32;
            block.iter_mut().for_each(|x| *x *= scale);
            start += size;
        }
        v = (0..dim).map(|j| v[(j * stride) % dim]).collect();
    }
    v
}

#[test]
fn matches_the_plain_definition_bit_for_bit() {
    let mut rng = SplitMix64::new(12);
    // 200 has a block of 8, 784 a block of 16 and 1007 blocks of 4, 2 and 1. 784 and 896 have stride 9, the rest 7.
    for dim in [1, 2, 4, 8, 128, 200, 768, 784, 896, 1007, 1536] {
        let rotation = Rotation::new(dim, 13);
        for x in gaussian(&mut rng, 3, dim, |_| 1.0).chunks_exact(dim) {
            let same = rotate(&rotation, x)
                .iter()
                .zip(reference(&rotation, x))
                .all(|(a, b)| a.to_bits() == b.to_bits());
            assert!(same, "dim {dim}");
        }
    }
}

#[test]
fn keeps_lengths_and_dot_products() {
    let mut rng = SplitMix64::new(1);
    for dim in [128, 768, 784, 1000, 1536] {
        let rotation = Rotation::new(dim, 7);
        let rows = gaussian(&mut rng, 20, dim, |_| 1.0);
        for pair in rows.chunks_exact(2 * dim) {
            let (x, y) = pair.split_at(dim);
            let (px, py) = (rotate(&rotation, x), rotate(&rotation, y));
            let (nx, ny) = (dot(x, x).sqrt(), dot(y, y).sqrt());
            assert!(
                (dot(&px, &px).sqrt() - nx).abs() <= 1e-5 * nx,
                "a length changed at dim {dim}"
            );
            assert!(
                (dot(&px, &py) - dot(x, y)).abs() <= 1e-5 * nx * ny,
                "a dot product changed at dim {dim}"
            );
        }
    }
}

#[test]
fn same_seed_gives_the_same_rotation() {
    assert_eq!(Rotation::new(768, 42), Rotation::new(768, 42));
    assert_ne!(Rotation::new(768, 42).signs(), Rotation::new(768, 43).signs());
    let rotation = Rotation::new(1000, 5);
    assert_eq!(Rotation::from_signs(1000, rotation.signs()).unwrap(), rotation);
    assert!(Rotation::from_signs(1000, &rotation.signs()[1..]).is_err());
    let x = gaussian(&mut SplitMix64::new(9), 1, 1000, |_| 1.0);
    let (a, b) = (rotate(&rotation, &x), rotate(&Rotation::new(1000, 5), &x));
    assert!(a.iter().zip(&b).all(|(p, q)| p.to_bits() == q.to_bits()));
}

#[test]
fn blocks_and_stride_follow_the_dimension() {
    let sizes = |dim| blocks(dim).iter().map(|b| b.1).collect::<Vec<_>>();
    assert_eq!(sizes(768), [512, 256]);
    assert_eq!(sizes(1000), [512, 256, 128, 64, 32, 8]);
    assert_eq!(stride(768), 7);
    // 896 = 7·128 rules out 7. 6720 = 3·5·7·64 also rules out 9.
    assert_eq!(stride(896), 9);
    assert_eq!(stride(6720), 11);
}

#[test]
fn mixes_every_basis_vector() {
    for dim in [768, 1536] {
        let rotation = Rotation::new(dim, 0x5EED);
        let (mut e, mut out, mut tmp) = (vec![0f32; dim], vec![0f32; dim], vec![0f32; dim]);
        let (mut fourth_sum, mut largest) = (0f64, 0f64);
        for i in 0..dim {
            e.fill(0.0);
            e[i] = 1.0;
            rotation.apply(&e, &mut out, &mut tmp);
            let zeros = out.iter().filter(|&&x| x == 0.0).count();
            assert_eq!(zeros, 0, "basis vector {i} misses {zeros} of {dim} outputs");
            fourth_sum += dim as f64 * out.iter().map(|&x| f64::from(x).powi(4)).sum::<f64>();
            largest = out.iter().fold(largest, |m, &x| m.max(f64::from(x).abs()));
        }
        // A random unit vector has d·Σx⁴ ≈ 3, and its largest coordinate is a few times 1/√d.
        let mean_fourth = fourth_sum / dim as f64;
        assert!(
            (2.9..=3.1).contains(&mean_fourth),
            "the mean of d·Σx⁴ is {mean_fourth} at dim {dim}"
        );
        assert!(
            largest * (dim as f64).sqrt() <= 7.0,
            "the largest coordinate is {largest} at dim {dim}"
        );
    }
}
