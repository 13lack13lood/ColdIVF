//! Unit tests for the cluster order.

use super::*;
use crate::quantization::SplitMix64;
use crate::quantization::testutil::{dist2, gaussian, mean, nearest};

/// `nlist` random clusters, each in the group of its nearest group center.
fn grouped(rng: &mut SplitMix64, nlist: usize, nparents: usize, dim: usize) -> Clusters {
    let centroids = gaussian(rng, nlist, dim, |_| 1.0);
    let parents = gaussian(rng, nparents, dim, |_| 1.0);
    let parent_of = centroids.chunks_exact(dim).map(|c| nearest(&parents, dim, c)).collect();

    Clusters {
        dim,
        centroids,
        assignment: Vec::new(),
        parents,
        parent_of,
    }
}

/// The greedy chain over the rows `ids` of `points`, written the plain way.
fn greedy(points: &[f32], dim: usize, start: &[f32], ids: &[u32]) -> Vec<u32> {
    let row = |id: u32| &points[id as usize * dim..(id as usize + 1) * dim];
    let (mut left, mut out, mut from) = (ids.to_vec(), Vec::new(), start.to_vec());

    while !left.is_empty() {
        left.sort_by(|&a, &b| dist2(row(a), &from).total_cmp(&dist2(row(b), &from)).then(a.cmp(&b)));
        let id = left.remove(0);
        out.push(id);
        from = row(id).to_vec();
    }

    out
}

#[test]
fn id_order_keeps_the_cluster_ids() {
    let clusters = grouped(&mut SplitMix64::new(1), 20, 3, 16);
    let mean = mean(&clusters.centroids, 16);

    assert_eq!(
        leaf_order(LeafOrder::Id, &clusters, &mean),
        (0..20).collect::<Vec<u32>>()
    );
}

#[test]
fn chained_order_chains_the_groups_then_the_clusters_in_each() {
    let mut rng = SplitMix64::new(2);
    for (nlist, nparents) in [(60, 5), (40, 12), (7, 1)] {
        let dim = 16;
        let clusters = grouped(&mut rng, nlist, nparents, dim);
        let mean = mean(&clusters.centroids, dim);
        let order = leaf_order(LeafOrder::Chained, &clusters, &mean);
        assert_eq!(
            order,
            leaf_order(LeafOrder::Chained, &clusters, &mean),
            "the same input gives the same order"
        );

        let groups = greedy(&clusters.parents, dim, &mean, &(0..nparents as u32).collect::<Vec<_>>());
        let mut want = Vec::new();
        let mut last = mean.clone();
        for g in groups {
            let members: Vec<u32> = (0..nlist as u32)
                .filter(|&c| clusters.parent_of[c as usize] == g)
                .collect();
            let placed = greedy(&clusters.centroids, dim, &last, &members);
            if let Some(&c) = placed.last() {
                last = clusters.centroids[c as usize * dim..(c as usize + 1) * dim].to_vec();
            }
            want.extend(placed);
        }

        assert_eq!(order, want, "{nlist} clusters in {nparents} groups");
    }
}

#[test]
fn a_distance_tie_goes_to_the_lower_id() {
    // Rows 0 and 1 are both at distance 1 from the start, and row 1 is listed first.
    let points = [1.0f32, 0.0, 0.0, 1.0, 3.0, 3.0];
    assert_eq!(chain(&points, 2, &[0.0, 0.0], &[1, 0, 2]), [0, 1, 2]);
}

#[test]
fn an_empty_group_between_two_others_is_skipped() {
    let mut rng = SplitMix64::new(3);
    let dim = 16;
    let mut clusters = grouped(&mut rng, 30, 3, dim);
    // Group 1 gets no clusters.
    clusters
        .parent_of
        .iter_mut()
        .for_each(|g| *g = if *g == 1 { 2 } else { *g });
    let mean = mean(&clusters.centroids, dim);

    let groups = greedy(&clusters.parents, dim, &mean, &[0, 1, 2]);
    let mut want = Vec::new();
    let mut last = mean.clone();
    for g in groups {
        let members: Vec<u32> = (0..30).filter(|&c| clusters.parent_of[c as usize] == g).collect();
        let placed = greedy(&clusters.centroids, dim, &last, &members);
        if let Some(&c) = placed.last() {
            last = clusters.centroids[c as usize * dim..(c as usize + 1) * dim].to_vec();
        }
        want.extend(placed);
    }

    assert_eq!(leaf_order(LeafOrder::Chained, &clusters, &mean), want);
}
