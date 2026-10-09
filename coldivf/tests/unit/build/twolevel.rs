//! Unit tests for two-level clustering.

use super::*;
use crate::build::threads;
use crate::quantization::SplitMix64;
use crate::quantization::testutil::{Rows, clustered, dist2, params, temp_file, temps_of};

#[test]
fn two_level_clustering_puts_each_vector_in_the_nearest_cluster_of_its_nearest_groups() {
    let (dim, n) = (128, 6000);
    let (rows, _) = clustered(&mut SplitMix64::new(9), n, dim, 40, 1.0);
    let settings = BuildParams {
        vectors_per_list: 100,
        parents: 4,
        ..params(Metric::L2)
    };
    let out = temp_file("twolevel.coldivf");
    let one = cluster(&settings, &rows, &out, &threads(1).unwrap()).unwrap();
    let four = cluster(&settings, &rows, &out, &threads(4).unwrap()).unwrap();
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&one.centroids), bits(&four.centroids));
    assert_eq!((&one.assignment, &one.parent_of), (&four.assignment, &four.parent_of));
    assert!(temps_of(&out).is_empty(), "the groups file was left");

    // 60 clusters' worth of vectors make at least the square root of 60 groups, and each group's clusters hold about
    // 100 vectors, so rounding up in each group adds at most a cluster per group.
    let (nlist, groups) = (one.centroids.len() / dim, one.parents.len() / dim);
    assert_eq!(groups, 7);
    assert!((60..=60 + groups).contains(&nlist), "{nlist} clusters");
    let row = |data: &[f32], i: usize| data[i * dim..(i + 1) * dim].to_vec();
    let mut moved = 0;

    for (r, x) in rows.data.chunks_exact(dim).enumerate() {
        let mut by_group: Vec<(f64, usize)> = (0..groups).map(|g| (dist2(x, &row(&one.parents, g)), g)).collect();
        by_group.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        let near: Vec<usize> = by_group.iter().take(NEAR_GROUPS).map(|&(_, g)| g).collect();
        let (c, last) = (one.assignment[r] as usize, by_group[NEAR_GROUPS - 1].0);
        let g = one.parent_of[c] as usize;
        assert!(
            near.contains(&g) || dist2(x, &row(&one.parents, g)) <= last * (1.0 + 1e-5),
            "row {r} is in cluster {c}"
        );
        let (best_d, best) = (0..nlist)
            .filter(|&i| near.contains(&(one.parent_of[i] as usize)))
            .map(|i| (dist2(x, &row(&one.centroids, i)), i))
            .min_by(|a, b| a.0.total_cmp(&b.0))
            .unwrap();
        let mine = dist2(x, &row(&one.centroids, c));
        assert!(c == best || mine <= best_d * (1.0 + 1e-5), "row {r} is in cluster {c}");
        moved += usize::from(g != near[0]);
    }

    assert!(moved > 0, "no vector left its nearest group");
}

#[test]
fn the_groups_file_holds_each_group_s_vectors_in_input_order() {
    let (rows, _) = clustered(&mut SplitMix64::new(3), 2000, 24, 5, 0.5);
    let out = temp_file("groups.coldivf");
    let group_of = |r: usize| (r * 7 + r / 3) % 5;
    let mut data = Vec::new();

    {
        let groups = Groups::write(&rows, group_of, 6, &out).unwrap();
        assert_eq!(temps_of(&out).len(), 1);
        for g in 0..6 {
            groups.read(g, &mut data).unwrap();
            let want: Vec<f32> = (0..2000)
                .filter(|&r| group_of(r) == g)
                .flat_map(|r| rows.data[r * 24..(r + 1) * 24].to_vec())
                .collect();
            assert_eq!(data, want, "group {g}");
        }
    }

    assert!(temps_of(&out).is_empty(), "the groups file was left");
}

#[test]
fn a_pass_in_runs_gives_what_one_run_gives() {
    let (dim, n) = (128, 3000);
    let (rows, _) = clustered(&mut SplitMix64::new(5), n, dim, 30, 0.8);
    let settings = BuildParams {
        vectors_per_list: 50,
        parents: 6,
        ..params(Metric::Cosine)
    };
    let pool = threads(2).unwrap();
    let start = cluster(&settings, &rows, &temp_file("runs.coldivf"), &pool).unwrap();
    let k = start.parents.len() / dim;
    let near = Near {
        groups: kmeans::nearest(&start.parents, &rows, Metric::Cosine, 3.min(k), &pool)
            .unwrap()
            .0,
        m: 3.min(k),
    };
    let copy = |c: &Clusters| Clusters {
        dim,
        centroids: c.centroids.clone(),
        assignment: vec![0; n],
        parents: c.parents.clone(),
        parent_of: c.parent_of.clone(),
    };
    let (mut whole, mut runs) = (copy(&start), copy(&start));
    reassign(&mut whole, &rows, &near, Metric::Cosine, &pool, n).unwrap();
    reassign(&mut runs, &rows, &near, Metric::Cosine, &pool, 7).unwrap();
    assert_eq!(whole.assignment, runs.assignment);

    recenter(&mut whole, &rows, Metric::Cosine, n);
    recenter(&mut runs, &rows, Metric::Cosine, 7);
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&whole.centroids), bits(&runs.centroids));
}

#[test]
fn a_vector_between_two_groups_goes_to_the_lower_cluster() {
    // Row 0 sits halfway between cluster 1, in group 0, and cluster 0, in group 1, and group 0 is scored first.
    let rows = Rows {
        dim: 2,
        data: vec![0.0, 0.0, -2.0, 0.0, 2.0, 0.0],
    };
    let mut clusters = Clusters {
        dim: 2,
        centroids: vec![1.0, 0.0, -1.0, 0.0],
        assignment: vec![1, 1, 1],
        parents: vec![-1.0, 0.0, 1.0, 0.0],
        parent_of: vec![1, 0],
    };
    let near = Near {
        groups: vec![0, 1, 0, 1, 1, 0],
        m: 2,
    };
    reassign(&mut clusters, &rows, &near, Metric::L2, &threads(1).unwrap(), 2).unwrap();
    assert_eq!(clusters.assignment, [0, 1, 0]);

    // With only its own group to pick from, a vector stays there.
    let near = Near {
        groups: vec![0, 0, 0],
        m: 1,
    };
    reassign(&mut clusters, &rows, &near, Metric::L2, &threads(1).unwrap(), 2).unwrap();
    assert_eq!(clusters.assignment, [1, 1, 1]);
}

#[test]
fn recentering_moves_each_center_to_the_mean_of_its_vectors() {
    let rows = Rows {
        dim: 2,
        data: vec![3.0, 4.0, 0.0, 2.0, 10.0, 10.0],
    };
    let clusters = |assignment: Vec<u32>| Clusters {
        dim: 2,
        centroids: vec![9.0, 9.0, 5.0, 5.0, -7.0, 1.0],
        assignment,
        parents: vec![0.0, 0.0],
        parent_of: vec![0, 0, 0],
    };
    let mut l2 = clusters(vec![0, 0, 1]);
    recenter(&mut l2, &rows, Metric::L2, 2);
    assert_eq!(l2.centroids, [1.5, 3.0, 10.0, 10.0, -7.0, 1.0]);

    // Cosine averages the vectors at length 1: (0.6, 0.8) and (0, 1).
    let mut cosine = clusters(vec![0, 0, 1]);
    recenter(&mut cosine, &rows, Metric::Cosine, 2);
    let want = [0.3, 0.9, 0.5f32.sqrt(), 0.5f32.sqrt(), -7.0, 1.0];
    assert!(
        cosine.centroids.iter().zip(want).all(|(&x, y)| (x - y).abs() < 1e-6),
        "{:?}",
        cosine.centroids
    );
}

#[test]
fn refining_lowers_the_total_distance_to_the_centers() {
    let (dim, n) = (128, 4000);
    let (rows, _) = clustered(&mut SplitMix64::new(13), n, dim, 50, 1.0);
    let settings = BuildParams {
        vectors_per_list: 80,
        parents: 4,
        ..params(Metric::L2)
    };
    let (pool, out) = (threads(2).unwrap(), temp_file("refine.coldivf"));
    let refined = cluster(&settings, &rows, &out, &pool).unwrap();

    // The same steps without the refining rounds.
    let k = refined.parents.len() / dim;
    let m = NEAR_GROUPS.min(k);
    let near = Near {
        groups: kmeans::nearest(&refined.parents, &rows, Metric::L2, m, &pool)
            .unwrap()
            .0,
        m,
    };
    let (centroids, parent_of) = train_groups(&settings, &rows, &near, k, &out, &pool).unwrap();
    let mut placed = Clusters {
        dim,
        centroids,
        assignment: vec![0; n],
        parents: refined.parents.clone(),
        parent_of,
    };
    reassign(&mut placed, &rows, &near, Metric::L2, &pool, n).unwrap();
    let total = |c: &Clusters| -> f64 {
        rows.data
            .chunks_exact(dim)
            .zip(&c.assignment)
            .map(|(x, &a)| dist2(x, &c.centroids[a as usize * dim..(a as usize + 1) * dim]))
            .sum()
    };
    assert!(
        total(&refined) < total(&placed),
        "{} vs {}",
        total(&refined),
        total(&placed)
    );
}
