//! Unit tests for the ground-truth tool.

use coldivf_bench::data::Ibin;

use super::*;

/// A folder of its own in the system temporary folder, which tests may write to.
fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("coldivf-bench/{}-gt-{name}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// The `k` nearest of the first `n` base rows to each query, from a full sort of every (distance, id) pair.
fn sorted(base: &Fbin, queries: &[&[f32]], metric: Metric, n: usize, k: usize) -> Vec<Nearest> {
    let nearest = |query: &[f32]| {
        let mut all: Nearest = (0..n)
            .map(|id| (distance(metric, query, base.row(id)), id as u32))
            .collect();
        all.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        all.truncate(k);
        all
    };
    queries.iter().map(|&q| nearest(q)).collect()
}

#[test]
fn one_pass_matches_a_full_sort_at_every_snapshot() {
    let dir = scratch("scan");
    // Small whole numbers make many exact ties, and the second half of the base repeats the first, so the order of
    // tied ids decides the truth. 20 values a row leave a tail past the 8-value chunks.
    let (rows, dim, nq, k) = (10_000, 20, 12, 100);
    let mut seed = 1u64;
    let mut next = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((seed >> 33) % 5) as f32 - 2.0
    };
    let half: Vec<f32> = (0..rows / 2 * dim).map(|_| next()).collect();
    let queries: Vec<f32> = (0..nq * dim).map(|_| next()).collect();
    write_fbin(&dir.join("base.fbin"), &[&half[..], &half[..]].concat(), dim).unwrap();
    let base = Fbin::open(&dir.join("base.fbin")).unwrap();
    let queries: Vec<&[f32]> = queries.chunks(dim).collect();
    // 3000 and 7001 aren't multiples of the tile, so the scan stops partway through a tile at each, and 7001 cuts a
    // pair of copies in two.
    let sizes = [3000, 7001, rows];

    for metric in [Metric::L2, Metric::Cosine] {
        let mut seen = Vec::new();
        scan(&base, &queries, metric, k, 3, &sizes, |size, best| {
            seen.push((size, best.to_vec()))
        });
        assert_eq!(seen.iter().map(|s| s.0).collect::<Vec<_>>(), sizes);

        for (size, best) in &seen {
            assert_eq!(
                best,
                &sorted(&base, &queries, metric, *size, k),
                "{metric:?}, n = {size}"
            );
        }
    }

    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn the_ids_and_their_distances_go_to_files_side_by_side() {
    let dir = scratch("write");
    let (path, dists) = (
        dir.join("t-euclidean-3.gt.ibin"),
        dir.join("t-euclidean-3.gt.dist.fbin"),
    );
    let best = vec![vec![(0.5, 2), (1.5, 0)], vec![(-1.0, 1), (2.0, 2)]];
    // An old distances file of another shape is replaced.
    write_fbin(&dists, &[9.0; 3], 1).unwrap();
    write(&path, &best, 2).unwrap();
    let (ids, values) = (Ibin::open(&path).unwrap(), Fbin::open(&dists).unwrap());
    assert_eq!((ids.len(), ids.row(0), ids.row(1)), (2, &[2, 0][..], &[1, 2][..]));
    assert_eq!(
        (values.len(), values.row(0), values.row(1)),
        (2, &[0.5, 1.5][..], &[-1.0, 2.0][..])
    );

    // A write whose ids fail, here since a folder holds their name, leaves no distances file, not even the old one.
    let blocked = dir.join("blocked.gt.ibin");
    std::fs::create_dir_all(&blocked).unwrap();
    write_fbin(&dist_path(&blocked), &[9.0; 4], 2).unwrap();
    assert!(write(&blocked, &best, 2).is_err());
    assert!(!dist_path(&blocked).exists());
    std::fs::remove_dir_all(&dir).unwrap();
}
