//! Unit tests for k-means.

use std::io::ErrorKind;

use super::*;
use crate::build::threads;
use crate::quantization::testutil::{Rows, clustered, dist2, dot, gaussian};
use crate::search::normalize;

fn settings(k: usize, rounds: usize) -> KMeans {
    KMeans {
        k,
        rounds,
        sample: usize::MAX,
        spherical: false,
        seed: 7,
    }
}

/// The nearest of `centers` to `x` by exact squared distance. Ties go to the lower center.
fn brute_nearest(x: &[f32], centers: &[f32], dim: usize) -> (usize, f64) {
    centers
        .chunks_exact(dim)
        .map(|c| dist2(x, c))
        .enumerate()
        .fold(
            (0, f64::INFINITY),
            |best, (i, d)| if d < best.1 { (i, d) } else { best },
        )
}

/// The sum of the squared distances from each vector to its nearest center.
fn inertia(rows: &[f32], centers: &[f32], dim: usize) -> f64 {
    rows.chunks_exact(dim).map(|x| brute_nearest(x, centers, dim).1).sum()
}

#[test]
fn every_vector_goes_to_its_nearest_center() {
    let mut rng = SplitMix64::new(1);
    let pool = threads(3).unwrap();
    // 1500 centers take two blocks of products, and 20,500 vectors take two blocks of vectors.
    let cases = [(300, 1500), (20_500, 3)];

    for (n, k) in cases {
        let dim = 128;
        let rows = Rows {
            dim,
            data: gaussian(&mut rng, n, dim, |_| 1.0),
        };
        let centers = gaussian(&mut rng, k, dim, |_| 1.0);
        let got = assign(&centers, &rows, Metric::L2, &pool).unwrap();
        for (x, &c) in rows.data.chunks_exact(dim).zip(&got) {
            let (best, d) = brute_nearest(x, &centers, dim);
            let mine = dist2(x, &centers[c as usize * dim..(c as usize + 1) * dim]);
            assert!(
                c as usize == best || mine <= d * (1.0 + 1e-5),
                "{n} vectors, {k} centers"
            );
        }
    }
}

#[test]
fn nearest_keeps_the_m_nearest_centers_in_order() {
    let mut rng = SplitMix64::new(12);
    let (dim, n, k, m) = (24, 700, 1100, 5);
    let rows = gaussian(&mut rng, n, dim, |_| 1.0);
    let centers = gaussian(&mut rng, k, dim, |_| 1.0);
    let src = Rows {
        dim,
        data: rows.clone(),
    };
    let (best, scores) = nearest(&centers, &src, Metric::L2, m, &threads(3).unwrap()).unwrap();
    assert_eq!((best.len(), scores.len()), (n * m, n * m));

    for (r, x) in rows.chunks_exact(dim).enumerate() {
        let mut by: Vec<(f64, u32)> = centers
            .chunks_exact(dim)
            .enumerate()
            .map(|(c, y)| (dist2(x, y), c as u32))
            .collect();
        by.sort_by(|a, b| a.0.total_cmp(&b.0));
        let (ids, s) = (&best[r * m..(r + 1) * m], &scores[r * m..(r + 1) * m]);
        assert!(s.windows(2).all(|w| w[0] <= w[1]), "row {r}: {s:?}");
        for (j, &c) in ids.iter().enumerate() {
            let d = dist2(x, &centers[c as usize * dim..(c as usize + 1) * dim]);
            assert!(
                (d - by[j].0).abs() <= 1e-3 * by[j].0,
                "row {r}, place {j}: {c} at {d}, not {:?}",
                by[j]
            );
            let norm = dot(x, x);
            assert!(
                (f64::from(s[j]) + norm - d).abs() <= 1e-3 * d.max(1.0),
                "row {r}, place {j}"
            );
        }
    }
}

#[test]
fn nearest_keeps_the_lower_of_tied_centers_first() {
    let src = Rows {
        dim: 2,
        data: vec![0.0, 0.0],
    };
    let centers = [3.0, 0.0, 1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 3.0];
    let (best, scores) = nearest(&centers, &src, Metric::L2, 4, &threads(1).unwrap()).unwrap();
    assert_eq!(best, [1, 2, 3, 0]);
    assert_eq!(scores, [1.0, 1.0, 1.0, 9.0]);
}

#[test]
fn a_tie_goes_to_the_lower_center() {
    let mut rng = SplitMix64::new(2);
    let dim = 128;
    let mut centers = gaussian(&mut rng, 3, dim, |_| 1.0);
    centers.copy_within(0..dim, 2 * dim);
    let rows = Rows {
        dim,
        data: centers[..dim].iter().map(|&x| x + 0.01).collect(),
    };

    assert_eq!(assign(&centers, &rows, Metric::L2, &threads(2).unwrap()).unwrap(), [0]);
}

/// A plain Lloyd's k-means in f64 on `rows`, started from the rows `first`.
fn reference(rows: &[f32], dim: usize, first: &[u32], rounds: usize) -> Vec<f32> {
    let k = first.len();
    let mut centers: Vec<f32> = first
        .iter()
        .flat_map(|&r| rows[r as usize * dim..(r as usize + 1) * dim].to_vec())
        .collect();

    for _ in 0..rounds {
        let (mut sums, mut counts) = (vec![0f64; k * dim], vec![0usize; k]);
        for x in rows.chunks_exact(dim) {
            let c = brute_nearest(x, &centers, dim).0;
            counts[c] += 1;
            sums[c * dim..(c + 1) * dim]
                .iter_mut()
                .zip(x)
                .for_each(|(s, &v)| *s += f64::from(v));
        }
        assert!(counts.iter().all(|&c| c > 0), "the test data left a cluster empty");
        for ((center, sum), &count) in centers.chunks_exact_mut(dim).zip(sums.chunks_exact(dim)).zip(&counts) {
            center
                .iter_mut()
                .zip(sum)
                .for_each(|(c, &s)| *c = (s / count as f64) as f32);
        }
    }

    centers
}

#[test]
fn training_matches_a_plain_lloyd_s_k_means() {
    let pool = threads(4).unwrap();
    // 20,500 vectors take two blocks of vectors in each round. Their 20 clusters overlap, so no center goes empty.
    let cases = [(2000, 12, 12, 0.3, vec![1, 5, 20]), (20_500, 20, 5, 1.0, vec![2])];

    for (n, nlist, k, spread, rounds) in cases {
        let (rows, _) = clustered(&mut SplitMix64::new(3), n, 128, nlist, spread);
        let first = pick(n, k, &mut SplitMix64::new(7));
        for rounds in rounds {
            let got = train(&settings(k, rounds), &rows, Metric::L2, &pool).unwrap();
            let off = worst(&got, &reference(&rows.data, 128, &first, rounds));
            assert!(off < 1e-4, "{n} vectors after {rounds} rounds are off by {off}");
        }
    }
}

/// The largest difference between two sets of centers.
fn worst(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max)
}

#[test]
fn a_sample_trains_on_exactly_its_rows() {
    // The rows are sorted by cluster, so training on the first rows instead of a random sample would miss most of them.
    let dim = 128;
    let (rows, clusters) = clustered(&mut SplitMix64::new(16), 3000, dim, 10, 0.3);
    let mut order: Vec<usize> = (0..3000).collect();
    order.sort_by_key(|&r| clusters.assignment[r]);
    let sorted: Vec<f32> = order
        .iter()
        .flat_map(|&r| rows.data[r * dim..(r + 1) * dim].to_vec())
        .collect();
    let rows = Rows { dim, data: sorted };
    let sampled = KMeans {
        sample: 400,
        ..settings(10, 5)
    };

    let got = train(&sampled, &rows, Metric::L2, &threads(3).unwrap()).unwrap();
    let mut rng = SplitMix64::new(7);
    let set = pick(3000, 400, &mut rng);
    let first = pick(400, 10, &mut rng);
    let train_rows: Vec<f32> = set
        .iter()
        .flat_map(|&r| rows.data[r as usize * dim..(r as usize + 1) * dim].to_vec())
        .collect();
    let off = worst(&got, &reference(&train_rows, dim, &first, 5));
    assert!(off < 1e-4, "the sample's centers are off by {off}");
}

#[test]
fn cosine_clusters_the_normalized_vectors() {
    let mut rng = SplitMix64::new(17);
    let dim = 128;
    let (mut rows, _) = clustered(&mut rng, 800, dim, 8, 0.5);
    // Lengths from 0.1 to 10 would move the clusters if the vectors weren't normalized first.
    for row in rows.data.chunks_exact_mut(dim) {
        let scale = (0.1 + 9.9 * rng.uniform()) as f32;
        row.iter_mut().for_each(|x| *x *= scale);
    }
    let mut normalized = rows.data.clone();
    normalized.chunks_exact_mut(dim).for_each(|row| assert!(normalize(row)));
    let pool = threads(2).unwrap();

    let cosine = train(&settings(8, 5), &rows, Metric::Cosine, &pool).unwrap();
    let l2 = train(&settings(8, 5), &Rows { dim, data: normalized }, Metric::L2, &pool).unwrap();
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&cosine), bits(&l2));
}

#[test]
fn a_center_left_empty_takes_part_of_the_largest_cluster() {
    // Two copies of one vector, then a tight cluster elsewhere. When both copies start as centers, every vector of the
    // copies goes to the first, so the second is left empty until it splits the tight cluster.
    let (dim, mut rng) = (128, SplitMix64::new(18));
    let p = gaussian(&mut rng, 1, dim, |_| 1.0);
    let q = gaussian(&mut rng, 1, dim, |_| 1.0);
    let mut data = [p.clone(), p].concat();
    for _ in 0..50 {
        data.extend(q.iter().map(|&x| x + (rng.normal() * 0.01) as f32));
    }
    let rows = Rows { dim, data };
    let seed = (0..)
        .find(|&s| {
            let first = pick(52, 3, &mut SplitMix64::new(s));
            first.contains(&0) && first.contains(&1)
        })
        .unwrap();
    let one_round = KMeans { seed, ..settings(3, 1) };
    let pool = threads(2).unwrap();

    let centers = train(&one_round, &rows, Metric::L2, &pool).unwrap();
    let mut counts = [0; 3];
    assign(&centers, &rows, Metric::L2, &pool)
        .unwrap()
        .iter()
        .for_each(|&c| counts[c as usize] += 1);
    assert!(counts.iter().all(|&c| c > 0), "a center has no vectors: {counts:?}");
}

#[test]
fn the_first_bad_vector_decides_the_error() {
    // Row 5 has a value that isn't finite and row 100 is a zero vector, so cosine fails on row 5 on any thread count.
    let dim = 128;
    let mut data = gaussian(&mut SplitMix64::new(19), 300, dim, |_| 1.0);
    data[5 * dim + 3] = f32::INFINITY;
    data[100 * dim..101 * dim].fill(0.0);
    let rows = Rows { dim, data };

    for threads in [1, 4] {
        let pool = crate::build::threads(threads).unwrap();
        let err = train(&settings(4, 2), &rows, Metric::Cosine, &pool).unwrap_err();
        assert_eq!(err.to_string(), "a vector has a value that isn't finite");
    }
}

#[test]
fn more_rounds_never_raise_the_total_distance() {
    let (rows, _) = clustered(&mut SplitMix64::new(4), 1500, 128, 20, 1.0);
    let pool = threads(2).unwrap();
    let mut last = f64::INFINITY;

    for rounds in 0..8 {
        let centers = train(&settings(15, rounds), &rows, Metric::L2, &pool).unwrap();
        let total = inertia(&rows.data, &centers, 128);
        assert!(
            total <= last * (1.0 + 1e-4),
            "round {rounds} raised it from {last} to {total}"
        );
        last = total;
    }
}

#[test]
fn spherical_centers_have_length_1() {
    let dim = 128;
    let (rows, _) = clustered(&mut SplitMix64::new(5), 1000, dim, 8, 0.5);
    let spherical = KMeans {
        spherical: true,
        ..settings(8, 10)
    };
    let centers = train(&spherical, &rows, Metric::Cosine, &threads(2).unwrap()).unwrap();

    for c in centers.chunks_exact(dim) {
        let norm = c.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "a center has length {norm}");
    }
}

#[test]
fn the_same_seed_gives_the_same_centers_on_any_thread_count() {
    let (rows, _) = clustered(&mut SplitMix64::new(6), 3000, 200, 30, 1.0);
    let k = settings(25, 6);
    let one = train(&k, &rows, Metric::L2, &threads(1).unwrap()).unwrap();
    let four = train(&k, &rows, Metric::L2, &threads(4).unwrap()).unwrap();
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&one), bits(&four));

    let other = train(&KMeans { seed: 8, ..k }, &rows, Metric::L2, &threads(1).unwrap()).unwrap();
    assert_ne!(bits(&one), bits(&other));
}

#[test]
fn a_sample_has_that_many_different_rows_in_order() {
    let mut rng = SplitMix64::new(9);

    for (n, m) in [(10, 10), (1000, 1), (1000, 300), (5, 0)] {
        let picked = pick(n, m, &mut rng);
        assert_eq!(picked.len(), m);
        assert!(picked.windows(2).all(|w| w[0] < w[1]) && picked.iter().all(|&r| (r as usize) < n));
    }

    // Training on a sample still needs a vector for each center.
    let (rows, _) = clustered(&mut SplitMix64::new(10), 500, 128, 5, 0.5);
    let small = KMeans {
        sample: 4,
        ..settings(5, 3)
    };
    let err = train(&small, &rows, Metric::L2, &threads(1).unwrap()).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    let enough = KMeans { sample: 50, ..small };
    assert_eq!(
        train(&enough, &rows, Metric::L2, &threads(1).unwrap()).unwrap().len(),
        5 * 128
    );
}

#[test]
fn an_empty_cluster_takes_half_of_the_largest() {
    let dim = 4;
    let mut centers = vec![1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, -1.0, -2.0, -3.0, -4.0];
    let mut counts = vec![5, 0, 9];
    split(&mut centers, &mut counts, dim, false);

    let (up, down) = (1.0 + SPLIT, 1.0 - SPLIT);
    assert_eq!(counts, [5, 4, 5]);
    assert_eq!(&centers[4..8], &[-up, -2.0 * down, -3.0 * up, -4.0 * down]);
    assert_eq!(&centers[8..12], &[-down, -2.0 * up, -3.0 * down, -4.0 * up]);

    // A tie goes to the lower cluster, and every empty cluster gets a split.
    let mut counts = vec![7, 0, 7];
    split(&mut centers, &mut counts, dim, false);
    assert_eq!(counts, [4, 3, 7]);
    let mut counts = vec![10, 0, 0];
    split(&mut centers, &mut counts, dim, false);
    assert_eq!(counts, [3, 5, 2]);

    // Clusters of one vector can't be split, so nothing changes.
    let (mut counts, before) = (vec![1, 0, 1], centers.clone());
    split(&mut centers, &mut counts, dim, false);
    assert_eq!((counts, &centers), (vec![1, 0, 1], &before));

    // With spherical centers, both copies keep length 1.
    let mut centers = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let mut counts = vec![4, 0];
    split(&mut centers, &mut counts, dim, true);
    for c in centers.chunks_exact(dim) {
        let norm = c.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6, "a copy has length {norm}");
    }
}

#[test]
fn bad_input_fails_with_invalid_input() {
    let pool = threads(1).unwrap();
    let dim = 128;
    let good = gaussian(&mut SplitMix64::new(11), 20, dim, |_| 1.0);
    let kind = |data: Vec<f32>, k: usize, metric: Metric| {
        let rows = Rows { dim, data };
        train(&settings(k, 2), &rows, metric, &pool).unwrap_err().kind()
    };

    assert_eq!(kind(good.clone(), 0, Metric::L2), ErrorKind::InvalidInput);
    assert_eq!(kind(good.clone(), 21, Metric::L2), ErrorKind::InvalidInput);
    let mut nan = good.clone();
    nan[300] = f32::NAN;
    assert_eq!(kind(nan, 20, Metric::L2), ErrorKind::InvalidInput);
    let mut zero = good.clone();
    zero[..dim].fill(0.0);
    assert_eq!(kind(zero, 20, Metric::Cosine), ErrorKind::InvalidInput);
    let mut huge = good;
    huge[5] = 1e38;
    assert_eq!(kind(huge, 20, Metric::L2), ErrorKind::InvalidInput);
}
