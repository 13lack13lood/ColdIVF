//! The order of the clusters in an index file. With the chained order, clusters that queries read together sit next to
//! each other, so a query's reads touch fewer stretches of the file.

use crate::{Clusters, LeafOrder};

/// The cluster id at each file position.
pub(crate) fn leaf_order(order: LeafOrder, clusters: &Clusters, mean: &[f32]) -> Vec<u32> {
    let nlist = clusters.centroids.len() / clusters.dim;

    match order {
        LeafOrder::Id => (0..nlist as u32).collect(),
        LeafOrder::Chained => chained(clusters, mean),
    }
}

/// Chains the groups from the one nearest the mean, each time taking the nearest group not yet placed. Inside each
/// group it chains the clusters the same way, from the one nearest the last cluster placed.
fn chained(clusters: &Clusters, mean: &[f32]) -> Vec<u32> {
    let dim = clusters.dim;
    let nparents = clusters.parents.len() / dim;
    let mut members = vec![Vec::new(); nparents];

    for (c, &g) in clusters.parent_of.iter().enumerate() {
        members[g as usize].push(c as u32);
    }

    let groups: Vec<u32> = (0..nparents as u32).collect();
    let mut last = mean;
    let mut out = Vec::with_capacity(clusters.parent_of.len());

    for g in chain(&clusters.parents, dim, mean, &groups) {
        let placed = chain(&clusters.centroids, dim, last, &members[g as usize]);
        if let Some(&c) = placed.last() {
            last = &clusters.centroids[c as usize * dim..(c as usize + 1) * dim];
        }
        out.extend(placed);
    }

    out
}

/// Orders the rows `ids` of `points` greedily. The first is the one nearest `start`, and each next one is the nearest
/// to the one before that isn't placed yet. Ties go to the lower id.
fn chain(points: &[f32], dim: usize, start: &[f32], ids: &[u32]) -> Vec<u32> {
    let row = |id: u32| &points[id as usize * dim..(id as usize + 1) * dim];
    let mut left = ids.to_vec();
    let mut out = Vec::with_capacity(ids.len());
    let mut from = start;

    while !left.is_empty() {
        let mut best = 0;
        let mut best_d = dist2(row(left[0]), from);
        for (i, &id) in left.iter().enumerate().skip(1) {
            let d = dist2(row(id), from);
            if d < best_d || (d == best_d && id < left[best]) {
                (best, best_d) = (i, d);
            }
        }
        let id = left.swap_remove(best);
        out.push(id);
        from = row(id);
    }

    out
}

/// The exact squared distance, summed in f64 in a fixed order so every machine picks the same chain.
fn dist2(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| (f64::from(x) - f64::from(y)).powi(2))
        .sum()
}

#[cfg(test)]
#[path = "../../tests/unit/build/order.rs"]
mod tests;
