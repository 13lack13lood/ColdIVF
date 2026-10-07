//! Unit tests for the 8-bit rows.

use super::*;
use crate::quantization::SplitMix64;
use crate::quantization::testutil::{dot, gaussian, mean};

#[test]
fn row_sizes_and_rows_per_page() {
    assert_eq!(row_bytes(768), 784);
    assert_eq!(center_row_bytes(768), 776);
    assert_eq!(rows_per_page(768, 4096), 5);
    assert_eq!(rows_per_page(768, 16384), 20);
    assert_eq!(rows_per_page(1536, 4096), 2);
}

#[test]
fn each_value_is_within_half_a_step() {
    let dim = 768;
    let rows = gaussian(&mut SplitMix64::new(1), 50, dim, |k| 1.0 + k as f64 / 100.0);
    let mu = mean(&rows, dim);
    let mut row = vec![0u8; row_bytes(dim)];
    for o in rows.chunks_exact(dim) {
        write_row(o, &mu, 0, &mut row);
        let s = f64::from(scale(&row, dim));
        for (i, (&x, &m)) in o.iter().zip(&mu).enumerate() {
            let off = f64::from(x - m);
            assert!(
                (off - s * f64::from(code(&row, i))).abs() <= s * (0.5 + 1e-4),
                "dimension {i}"
            );
        }
    }
}

#[test]
fn a_code_is_never_minus_128() {
    let dim = 128;
    let mut o = vec![0.5f32; dim];
    o[3] = -2.0;
    let mut row = vec![0u8; row_bytes(dim)];
    write_row(&o, &vec![0.0; dim], 0, &mut row);
    assert_eq!(code(&row, 3), -127);
    let rows = gaussian(&mut SplitMix64::new(2), 100, dim, |_| 1.0);
    for o in rows.chunks_exact(dim) {
        write_row(o, &vec![0.0; dim], 0, &mut row);
        assert!((0..dim).all(|i| code(&row, i) >= -127));
    }
}

#[test]
fn distance_matches_the_decoded_copy() {
    let mut rng = SplitMix64::new(3);
    for dim in [128, 200, 768] {
        let rows = gaussian(&mut rng, 20, dim, |_| 1.0);
        let mu = mean(&rows, dim);
        let q = gaussian(&mut rng, 1, dim, |_| 1.0);
        let q_off: Vec<f32> = q.iter().zip(&mu).map(|(&x, &m)| x - m).collect();
        let q_norm2 = dot(&q_off, &q_off) as f32;
        let mut row = vec![0u8; row_bytes(dim)];
        for o in rows.chunks_exact(dim) {
            write_row(o, &mu, 7, &mut row);
            let s = f64::from(scale(&row, dim));
            let exact: f64 = (0..dim)
                .map(|i| (f64::from(q_off[i]) - s * f64::from(code(&row, i))).powi(2))
                .sum();
            let got = f64::from(distance(&row, &q_off, q_norm2));
            assert!(
                (got - exact).abs() <= 1e-4 * exact,
                "{got} against {exact} at dim {dim}"
            );
        }
    }
}

#[test]
fn a_vector_at_the_mean_gets_zero_codes() {
    let dim = 128;
    let mu = gaussian(&mut SplitMix64::new(4), 1, dim, |_| 1.0);
    let mut row = vec![0u8; row_bytes(dim)];
    write_row(&mu, &mu, 0, &mut row);
    assert!(row[..dim].iter().all(|&c| c == 0));
    assert_eq!((scale(&row, dim), norm2(&row, dim)), (0.0, 0.0));
    let q_off = vec![0.5f32; dim];
    assert_eq!(distance(&row, &q_off, 32.0), 32.0);
}

#[test]
fn rows_keep_their_id_and_center_rows_have_none() {
    let dim = 768;
    let o = gaussian(&mut SplitMix64::new(5), 1, dim, |_| 1.0);
    let mu = vec![0.0; dim];
    let mut row = vec![0u8; row_bytes(dim)];
    write_row(&o, &mu, 0xDEAD_BEEF_0123_4567, &mut row);
    assert_eq!(id(&row, dim), 0xDEAD_BEEF_0123_4567);
    let mut center = vec![0u8; center_row_bytes(dim)];
    write_center_row(&o, &mu, &mut center);
    assert_eq!(center[..], row[..center_row_bytes(dim)]);
}

#[test]
fn rows_follow_the_file_layout() {
    assert_eq!((row_bytes(130), center_row_bytes(130)), (152, 144));
    let mut rng = SplitMix64::new(6);
    // Dimensions that leave the f32 fields unaligned and need padding at the end.
    for dim in [129, 130, 131] {
        let o = gaussian(&mut rng, 1, dim, |_| 1.0);
        let mu = gaussian(&mut rng, 1, dim, |_| 0.5);
        let id = 0x0102_0304_0506_0708u64 + dim as u64;
        let off: Vec<f32> = o.iter().zip(&mu).map(|(&x, &m)| x - m).collect();
        let scale = off.iter().fold(0f32, |m, &x| m.max(x.abs())) / 127.0;
        let codes: Vec<i8> = off
            .iter()
            .map(|&x| (x / scale).round().clamp(-127.0, 127.0) as i8)
            .collect();
        let sum_sq: f64 = codes.iter().map(|&c| f64::from(c).powi(2)).sum();
        let n2 = (f64::from(scale).powi(2) * sum_sq) as f32;
        let mut want: Vec<u8> = codes.iter().map(|&c| c as u8).collect();
        want.extend(scale.to_le_bytes());
        want.extend(n2.to_le_bytes());
        want.extend(id.to_le_bytes());
        want.resize(row_bytes(dim), 0);
        let mut row = vec![0xAAu8; row_bytes(dim)];
        write_row(&o, &mu, id, &mut row);
        assert_eq!(row, want, "dim {dim}");
        let mut center = vec![0xAAu8; center_row_bytes(dim)];
        write_center_row(&o, &mu, &mut center);
        want.truncate(dim + 8);
        want.resize(center_row_bytes(dim), 0);
        assert_eq!(center, want, "center row at dim {dim}");
    }
}
