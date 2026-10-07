//! Unit tests for the 1-bit codes.

use super::*;
use crate::quantization::SplitMix64;
use crate::quantization::testutil::{dist2, gaussian, mean};

/// The codes of `rows`, back to back, and their factors.
fn encode_all(rotation: &Rotation, mean: &[f32], rows: &[f32]) -> (Vec<u8>, Vec<Factors>) {
    let dim = rotation.dim();
    let cb = code_bytes(padded_dim(dim));
    let mut encoder = Encoder::new(rotation, mean);
    let mut codes = vec![0u8; rows.len() / dim * cb];
    let factors = rows
        .chunks_exact(dim)
        .zip(codes.chunks_exact_mut(cb))
        .map(|(o, bits)| encoder.encode(o, bits));
    let factors = factors.collect();
    (codes, factors)
}

fn prepare(rotation: &Rotation, mean: &[f32], q: &[f32]) -> QueryLut {
    let dim = rotation.dim();
    let (mut y, mut tmp) = (vec![0f32; dim], vec![0f32; dim]);
    let t = rotate_offset(rotation, q, mean, &mut y, &mut tmp);
    let mut query = QueryLut::new(dim);
    query.prepare(&y, t);
    query
}

#[test]
fn pack_and_unpack_round_trip() {
    let mut rng = SplitMix64::new(3);
    let d_pad = 768;
    let cb = code_bytes(d_pad);
    for n in [32, 13] {
        let codes: Vec<u8> = (0..n * cb).map(|_| rng.next_u64() as u8).collect();
        let factors: Vec<Factors> = (0..n)
            .map(|_| Factors {
                f_add: rng.uniform() as f32,
                f_rescale: -(rng.uniform() as f32),
                f_pop: (rng.next_u64() % 768) as f32,
                f_err: rng.uniform() as f32,
            })
            .collect();
        let mut block = vec![0u8; block_bytes(d_pad)];
        pack_block(&codes, &factors, d_pad, &mut block);
        let mut bits = vec![0u8; cb];
        for lane in 0..BLOCK {
            let f = unpack_lane(&block, d_pad, lane, &mut bits);
            assert_eq!(f, lane_factors(&block, d_pad, lane));
            if lane < n {
                assert_eq!(bits, codes[lane * cb..(lane + 1) * cb], "lane {lane} of {n}");
                assert_eq!(f, factors[lane]);
            } else {
                assert!(bits.iter().all(|&b| b == 0), "unused lane {lane} has bits");
                assert_eq!(f, Factors::EMPTY);
            }
        }
    }
}

#[test]
fn per_vector_and_block_paths_agree_bit_for_bit() {
    let mut rng = SplitMix64::new(4);
    for dim in [200, 768] {
        let d_pad = padded_dim(dim);
        let cb = code_bytes(d_pad);
        let rows = gaussian(&mut rng, 45, dim, |_| 1.0);
        let mu = mean(&rows, dim);
        let rotation = Rotation::new(dim, 11);
        let (codes, factors) = encode_all(&rotation, &mu, &rows);
        let query = prepare(&rotation, &mu, &gaussian(&mut rng, 1, dim, |_| 1.0));
        assert!(query.lut.iter().all(|&e| e <= 60));
        let mut block = vec![0u8; block_bytes(d_pad)];
        let mut sums = [0u32; BLOCK];
        for (start, fs) in (0..factors.len()).step_by(BLOCK).zip(factors.chunks(BLOCK)) {
            pack_block(&codes[start * cb..(start + fs.len()) * cb], fs, d_pad, &mut block);
            block_sums(&block, &query.lut, d_pad, &mut sums);
            for (lane, &s) in sums.iter().enumerate() {
                let packed = est(s, &lane_factors(&block, d_pad, lane), &query.k);
                if lane >= fs.len() {
                    assert_eq!(packed, f32::INFINITY, "an unused lane ranks");
                    continue;
                }
                let i = start + lane;
                let direct = code_sum(&codes[i * cb..(i + 1) * cb], &query.qu);
                assert_eq!(s, direct, "vector {i} at dim {dim}");
                assert_eq!(packed.to_bits(), est(direct, &fs[lane], &query.k).to_bits());
            }
        }
    }
}

/// Checks est against the exact distance on Gaussian data with standard deviation `sigma(k)` on axis k.
fn check_error(dim: usize, sigma: impl Fn(usize) -> f64, name: &str) {
    let mut rng = SplitMix64::new(5);
    let rows = gaussian(&mut rng, 1000, dim, &sigma);
    let queries = gaussian(&mut rng, 40, dim, &sigma);
    let mu = mean(&rows, dim);
    let rotation = Rotation::new(dim, 6);
    let cb = code_bytes(padded_dim(dim));
    let (codes, factors) = encode_all(&rotation, &mu, &rows);
    let (mut sum, mut sum_sq, mut covered, mut pairs) = (0f64, 0f64, 0usize, 0usize);
    for q in queries.chunks_exact(dim) {
        let query = prepare(&rotation, &mu, q);
        for (i, (o, f)) in rows.chunks_exact(dim).zip(&factors).enumerate() {
            let e = est(code_sum(&codes[i * cb..(i + 1) * cb], &query.qu), f, &query.k);
            let err = f64::from(e) - dist2(q, o);
            sum += err;
            sum_sq += err * err;
            covered += usize::from(err.abs() <= f64::from(f.f_err) * f64::from(query.k.t));
            pairs += 1;
        }
    }
    let (bias, rms) = (sum / pairs as f64, (sum_sq / pairs as f64).sqrt());
    let coverage = covered as f64 / pairs as f64;
    assert!(
        bias.abs() <= 0.1 * rms,
        "{name} at dim {dim}: the mean error is {bias}, against an rms error of {rms}"
    );
    assert!(
        (0.92..=0.96).contains(&coverage),
        "{name} at dim {dim}: the error bar covers {coverage} of pairs"
    );
}

#[test]
fn estimates_are_unbiased_and_error_bars_cover_95_percent() {
    // At 200 dimensions d is well below d_pad = 256, so mixing the two up fails here.
    for dim in [200, 768] {
        check_error(dim, |_| 1.0, "isotropic");
        // Most of the variance sits in the first axes, so this fails without a good rotation.
        check_error(dim, |k| (k as f64 + 1.0).powf(-0.75), "anisotropic");
    }
}

/// Asserts that `got` is within `rel` of `want`, relative to `scale`.
fn close(got: f64, want: f64, scale: f64, rel: f64, what: &str) {
    assert!((got - want).abs() <= rel * scale, "{what}: {got} against {want}");
}

#[test]
fn factors_and_query_follow_the_formulas() {
    let mut rng = SplitMix64::new(13);
    for dim in [200, 768] {
        let d = dim as f64;
        let rows = gaussian(&mut rng, 10, dim, |k| 1.0 + (k % 7) as f64);
        let mu = mean(&rows, dim);
        let rotation = Rotation::new(dim, 14);
        let (codes, factors) = encode_all(&rotation, &mu, &rows);
        let cb = code_bytes(padded_dim(dim));
        let (mut r, mut tmp) = (vec![0f32; dim], vec![0f32; dim]);
        for (i, (o, f)) in rows.chunks_exact(dim).zip(&factors).enumerate() {
            let l = rotate_offset(&rotation, o, &mu, &mut r, &mut tmp);
            let bits = &codes[i * cb..(i + 1) * cb];
            let w: Vec<f64> = r.iter().map(|&x| f64::from(x) / l).collect();
            let alpha = w.iter().map(|x| x.abs()).sum::<f64>() / d.sqrt();
            let f_err = 2.0 * l * 1.9 * ((1.0 - alpha * alpha) / (alpha * alpha * (d - 1.0))).sqrt();
            close(f64::from(f.f_add), l * l, l * l, 1e-6, "f_add");
            close(
                f64::from(f.f_rescale),
                -2.0 * l / alpha,
                2.0 * l / alpha,
                1e-6,
                "f_rescale",
            );
            close(f64::from(f.f_err), f_err, f_err, 1e-6, "f_err");
            assert_eq!(f.f_pop, w.iter().filter(|&&x| x > 0.0).count() as f32);
            for j in 0..padded_dim(dim) {
                let bit = (bits[j / 8] >> (j % 8)) & 1 == 1;
                assert_eq!(bit, j < dim && w[j] > 0.0, "bit {j} at dim {dim}");
            }
        }
        let q = gaussian(&mut rng, 1, dim, |_| 1.0);
        let t = rotate_offset(&rotation, &q, &mu, &mut r, &mut tmp);
        let mut query = QueryLut::new(dim);
        query.prepare(&r, t);
        let y: Vec<f64> = r.iter().map(|&x| f64::from(x) / t).collect();
        let v_l = y.iter().copied().fold(f64::INFINITY, f64::min);
        let delta = (y.iter().copied().fold(f64::NEG_INFINITY, f64::max) - v_l) / 15.0;
        for (j, &yj) in y.iter().enumerate() {
            let qu = f64::from(query.qu[j]);
            assert_eq!(
                qu,
                ((yj - v_l) / delta).round().clamp(0.0, 15.0),
                "q_u {j} at dim {dim}"
            );
            assert!((yj - (v_l + delta * qu)).abs() <= delta * (0.5 + 1e-9));
        }
        assert!(query.qu[dim..].iter().all(|&qu| qu == 0));
        assert_eq!(query.qu[..dim].iter().min(), Some(&0));
        assert_eq!(query.qu[..dim].iter().max(), Some(&15));
        for (g, entries) in query.lut.chunks(16).enumerate() {
            for (j, &e) in entries.iter().enumerate() {
                let want: u8 = (0..4).filter(|&b| (j >> b) & 1 == 1).map(|b| query.qu[4 * g + b]).sum();
                assert_eq!(e, want, "entry {j} of group {g}");
            }
        }
        let s_y: f64 = query.qu[..dim].iter().map(|&qu| v_l + delta * f64::from(qu)).sum();
        let k = &query.k;
        close(
            f64::from(k.k_s),
            2.0 * delta * t / d.sqrt(),
            2.0 * delta * t / d.sqrt(),
            1e-6,
            "k_s",
        );
        close(
            f64::from(k.k_p),
            2.0 * v_l * t / d.sqrt(),
            (2.0 * v_l * t / d.sqrt()).abs(),
            1e-6,
            "k_p",
        );
        close(
            f64::from(k.k_c),
            -t * s_y / d.sqrt(),
            (t * s_y / d.sqrt()).abs(),
            1e-6,
            "k_c",
        );
        close(f64::from(k.t), t, t, 1e-7, "T");
        close(f64::from(k.t2), t * t, t * t, 1e-7, "T²");
        for (i, f) in factors.iter().enumerate() {
            let s = code_sum(&codes[i * cb..(i + 1) * cb], &query.qu);
            let ip = f64::from(k.k_s) * f64::from(s) + f64::from(k.k_p) * f64::from(f.f_pop) + f64::from(k.k_c);
            let want = f64::from(f.f_add) + f64::from(k.t2) + f64::from(f.f_rescale) * ip;
            let scale = f64::from(f.f_add) + f64::from(k.t2) + (f64::from(f.f_rescale) * ip).abs();
            close(f64::from(est(s, f, k)), want, scale, 1e-6, "est");
        }
    }
}

#[test]
fn blocks_follow_the_file_layout() {
    let d_pad = 768;
    assert_eq!(block_bytes(d_pad), 3584);
    let cb = code_bytes(d_pad);
    let mut rng = SplitMix64::new(15);
    let n = 20;
    let codes: Vec<u8> = (0..n * cb).map(|_| rng.next_u64() as u8).collect();
    let factors: Vec<Factors> = (0..n)
        .map(|v| {
            let v = v as f32;
            Factors {
                f_add: 1.0 + v,
                f_rescale: -2.0 - v,
                f_pop: 3.0 + v,
                f_err: 0.5 + v,
            }
        })
        .collect();
    let mut block = vec![0u8; block_bytes(d_pad)];
    pack_block(&codes, &factors, d_pad, &mut block);
    // Bit t of vector v's group g is the bit of dimension 4g + t.
    let group_of = |v: usize, g: usize| -> u8 {
        if v >= n {
            return 0;
        }
        (0..4)
            .map(|t| ((codes[v * cb + (4 * g + t) / 8] >> ((4 * g + t) % 8)) & 1) << t)
            .sum()
    };
    for g in 0..d_pad / 4 {
        for i in 0..16 {
            assert_eq!(
                block[16 * g + i],
                group_of(i, g) | (group_of(16 + i, g) << 4),
                "byte {i} of group {g}"
            );
        }
    }
    for lane in 0..BLOCK {
        let want = factors.get(lane).copied().unwrap_or(Factors::EMPTY);
        for (k, value) in [want.f_add, want.f_rescale, want.f_pop, want.f_err]
            .into_iter()
            .enumerate()
        {
            let at = 4 * d_pad + 4 * (BLOCK * k + lane);
            let got = f32::from_le_bytes(block[at..at + 4].try_into().unwrap());
            assert_eq!(got.to_bits(), value.to_bits(), "factor {k} of lane {lane}");
        }
    }
}

#[test]
fn a_vector_that_underflows_after_rotation_stays_finite() {
    // L is the smallest positive f32, so P·r rounds to zero and α would be 0 without its clamp.
    let dim = 768;
    let mu = vec![0f32; dim];
    let mut o = vec![0f32; dim];
    o[0] = f32::from_bits(1);
    let rotation = Rotation::new(dim, 16);
    let (codes, factors) = encode_all(&rotation, &mu, &o);
    let f = factors[0];
    assert!(f.f_rescale.is_finite() && f.f_err.is_finite());
    let mut rng = SplitMix64::new(17);
    for q in gaussian(&mut rng, 5, dim, |_| 1.0).chunks_exact(dim) {
        let query = prepare(&rotation, &mu, q);
        assert_eq!(est(code_sum(&codes, &query.qu), &f, &query.k), query.k.t2);
    }
}

#[test]
fn a_vector_at_the_mean_is_exact() {
    let dim = 256;
    let mut rng = SplitMix64::new(7);
    let mu = gaussian(&mut rng, 1, dim, |_| 1.0);
    let rotation = Rotation::new(dim, 8);
    let (codes, factors) = encode_all(&rotation, &mu, &mu);
    assert!(codes.iter().all(|&b| b == 0));
    let query = prepare(&rotation, &mu, &gaussian(&mut rng, 1, dim, |_| 1.0));
    assert_eq!(est(code_sum(&codes, &query.qu), &factors[0], &query.k), query.k.t2);
}

#[test]
fn a_query_at_the_mean_gets_f_add() {
    let dim = 256;
    let mut rng = SplitMix64::new(9);
    let rows = gaussian(&mut rng, 5, dim, |_| 1.0);
    let mu = mean(&rows, dim);
    let rotation = Rotation::new(dim, 10);
    let (codes, factors) = encode_all(&rotation, &mu, &rows);
    let query = prepare(&rotation, &mu, &mu);
    assert_eq!(query.k, QueryConsts::default());
    let cb = code_bytes(padded_dim(dim));
    for (i, f) in factors.iter().enumerate() {
        assert_eq!(
            est(code_sum(&codes[i * cb..(i + 1) * cb], &query.qu), f, &query.k),
            f.f_add
        );
    }
}
