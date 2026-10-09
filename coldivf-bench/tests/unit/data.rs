//! Unit tests for the dataset files.

use std::io::ErrorKind;

use super::*;

/// A path in a folder of the system temporary folder, which tests may write to.
fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join("coldivf-bench");
    std::fs::create_dir_all(&dir).unwrap();
    dir.join(format!("{}-{name}", std::process::id()))
}

#[test]
fn vectors_and_ids_round_trip_through_their_files() {
    let (fbin, ibin) = (scratch("rt.fbin"), scratch("rt.ibin"));
    let rows: Vec<f32> = (0..30).map(|i| i as f32 * 0.5).collect();
    write_fbin(&fbin, &rows, 3).unwrap();
    write_ibin(&ibin, &[4, 5, 6, 7], 2).unwrap();

    let vectors = Fbin::open(&fbin).unwrap();
    assert_eq!((vectors.len(), vectors.dim()), (10, 3));
    assert_eq!(vectors.row(9), &[13.5, 14.0, 14.5]);
    let prefix = vectors.prefix(4);
    assert_eq!((prefix.count(), prefix.dim()), (4, 3));
    assert_eq!(prefix.rows(2, 2), &rows[6..12]);
    let ids = Ibin::open(&ibin).unwrap();
    assert_eq!((ids.len(), ids.width(), ids.row(1)), (2, 2, &[6, 7][..]));
    std::fs::remove_file(&fbin).unwrap();
    std::fs::remove_file(&ibin).unwrap();
}

#[test]
fn a_file_with_the_wrong_length_or_header_fails_with_invalid_data() {
    let path = scratch("bad.fbin");
    let mut bytes = Vec::new();
    bytes.extend(2i32.to_le_bytes());
    bytes.extend(4i32.to_le_bytes());
    bytes.extend([0u8; 4 * 7]);

    let mut long = bytes.clone();
    long.extend([0u8; 4 * 2]);
    let mut negative = bytes.clone();
    negative[..4].copy_from_slice(&(-1i32).to_le_bytes());
    // Two rows of width 0 take no bytes after the header, so only the width check can catch this one.
    let mut no_width = bytes[..8].to_vec();
    no_width[4..8].copy_from_slice(&0i32.to_le_bytes());

    for (what, bytes) in [
        ("short", &bytes[..]),
        ("header only", &bytes[..6]),
        ("long", &long[..]),
        ("negative count", &negative[..]),
        ("zero width", &no_width[..]),
    ] {
        std::fs::write(&path, bytes).unwrap();
        assert_eq!(
            Fbin::open(&path).err().unwrap().kind(),
            ErrorKind::InvalidData,
            "{what}"
        );
    }

    std::fs::remove_file(&path).unwrap();

    for (rows, width) in [(&[1, 2, 3][..], 2), (&[][..], 0)] {
        assert_eq!(
            write_ibin(&path, rows, width).unwrap_err().kind(),
            ErrorKind::InvalidInput,
            "{rows:?} at width {width}"
        );
    }
}

#[test]
fn a_dataset_opens_with_its_full_or_its_subset_truth() {
    let (root, derived) = (scratch("root"), scratch("derived"));
    std::fs::create_dir_all(&root).unwrap();
    std::fs::create_dir_all(&derived).unwrap();
    let base: Vec<f32> = (0..40).map(|i| i as f32).collect();
    write_fbin(&root.join("t-euclidean.base.fbin"), &base, 4).unwrap();
    write_fbin(&root.join("t-euclidean.query.fbin"), &base[..12], 4).unwrap();
    write_ibin(&root.join("t-euclidean.gt.ibin"), &[9, 8, 7, 6, 5, 4], 2).unwrap();
    write_ibin(&derived.join("t-euclidean-6.gt.ibin"), &[5, 4, 3, 2, 1, 0], 2).unwrap();

    let open = |n| Dataset::open_in(&root, &derived, "t-euclidean", n);
    let full = open(None).unwrap();
    assert_eq!((full.n, full.queries.len(), full.truth.row(2)), (10, 3, &[5, 4][..]));
    assert_eq!(full.metric, Metric::L2);
    // n at or past the base count still takes the full truth.
    assert_eq!(open(Some(99)).unwrap().truth.row(0), &[9, 8]);
    let subset = open(Some(6)).unwrap();
    assert_eq!((subset.n, subset.truth.row(0)), (6, &[5, 4][..]));

    // The queries must have the base's width, and the truth one row per query.
    write_fbin(&root.join("t-euclidean.query.fbin"), &base[..15], 5).unwrap();
    assert_eq!(open(None).err().unwrap().kind(), ErrorKind::InvalidData);
    write_fbin(&root.join("t-euclidean.query.fbin"), &base[..12], 4).unwrap();
    write_ibin(&root.join("t-euclidean.gt.ibin"), &[9, 8, 7, 6], 2).unwrap();
    assert_eq!(open(None).err().unwrap().kind(), ErrorKind::InvalidData);
    std::fs::remove_dir_all(&root).unwrap();
    std::fs::remove_dir_all(&derived).unwrap();
}

#[test]
fn a_dataset_s_metric_comes_from_its_name() {
    assert_eq!(metric_of("sift-128-euclidean"), Metric::L2);
    assert_eq!(metric_of("cohere-768-cosine"), Metric::Cosine);
}

#[test]
fn the_whole_base_s_truth_can_be_the_one_gt_writes() {
    let (root, derived) = (scratch("fallback-root"), scratch("fallback-derived"));
    std::fs::create_dir_all(&root).unwrap();
    std::fs::create_dir_all(&derived).unwrap();
    let base: Vec<f32> = (0..40).map(|i| i as f32).collect();
    write_fbin(&root.join("t-euclidean.base.fbin"), &base, 4).unwrap();
    write_fbin(&root.join("t-euclidean.query.fbin"), &base[..12], 4).unwrap();
    write_ibin(&derived.join("t-euclidean-10.gt.ibin"), &[1, 2, 3, 4, 5, 6], 2).unwrap();
    write_fbin(&derived.join("t-euclidean-10.gt.dist.fbin"), &[0.0; 6], 2).unwrap();

    // Without a ground truth in the data folder, the whole base takes the one in the derived folder, and its
    // distances.
    let open = || Dataset::open_in(&root, &derived, "t-euclidean", None).unwrap();
    let made = open();
    assert_eq!(
        (made.n, made.truth.row(1), made.dists.is_some()),
        (10, &[3, 4][..], true)
    );
    // One in the data folder comes first.
    write_ibin(&root.join("t-euclidean.gt.ibin"), &[9, 8, 7, 6, 5, 4], 2).unwrap();
    let shipped = open();
    assert_eq!((shipped.truth.row(1), shipped.dists.is_some()), (&[7, 6][..], false));
    std::fs::remove_dir_all(&root).unwrap();
    std::fs::remove_dir_all(&derived).unwrap();
}

#[test]
fn a_truth_s_distances_sit_next_to_its_ids() {
    assert_eq!(
        dist_path(Path::new("data/cohere-768-cosine-50000.gt.ibin")),
        Path::new("data/cohere-768-cosine-50000.gt.dist.fbin")
    );
    assert_eq!(
        dist_path(Path::new("data/truth.ibin")),
        Path::new("data/truth.dist.fbin")
    );
}

#[test]
fn recall_counts_ties_with_or_without_the_distances_file() {
    let (root, derived) = (scratch("ties-root"), scratch("ties-derived"));
    std::fs::create_dir_all(&root).unwrap();
    // The query is the origin, so each row's distance is its squared length: 1, 4, 9, 9, 9 and 16. Row 4 is a copy of
    // row 2, the 3rd nearest, and row 3 is another row as far.
    let rows = [
        [1, 0, 0, 0],
        [0, 2, 0, 0],
        [0, 0, 3, 0],
        [0, 0, 0, 3],
        [0, 0, 3, 0],
        [4, 0, 0, 0],
    ];
    let base: Vec<f32> = rows.iter().flatten().map(|&x| x as f32).collect();
    write_fbin(&root.join("t-euclidean.base.fbin"), &base, 4).unwrap();
    write_fbin(&root.join("t-euclidean.query.fbin"), &[0.0; 4], 4).unwrap();
    write_ibin(&root.join("t-euclidean.gt.ibin"), &[0, 1, 2, 3, 4, 5], 6).unwrap();
    let dists = root.join("t-euclidean.gt.dist.fbin");
    let open = || Dataset::open_in(&root, &derived, "t-euclidean", None);

    // Without the distances file, with distances, and with similarity scores, which shrink along the row. The ties
    // come from the base rows each time, and the file only checks the truth.
    for (what, sign) in [("no file", 0.0), ("distances", 1.0), ("scores", -1.0)] {
        if sign != 0.0 {
            let values: Vec<f32> = [1.0, 4.0, 9.0, 9.0, 9.0, 16.0].iter().map(|v| sign * v).collect();
            write_fbin(&dists, &values, 6).unwrap();
        }
        let ties = open().unwrap();
        assert_eq!(ties.dists.is_some(), sign != 0.0, "{what}");
        let recall = |found: &[u64], k: usize| ties.recall(0, found, k).unwrap();
        assert_eq!(ties.kth(0, 3).unwrap(), 9.0, "{what}");
        // A copy of the 3rd nearest ties it, and so does another row as far.
        assert_eq!((recall(&[0, 1, 4], 3), recall(&[0, 1, 3], 3)), (1.0, 1.0), "{what}");
        // A farther row, an id found twice and an id past the base don't count.
        for found in [[0, 1, 5], [0, 4, 4], [0, 1, 99]] {
            assert_eq!(recall(&found, 3), 2.0 / 3.0, "{what}, {found:?}");
        }
        // At k = 2 the 2nd nearest is at 4, so row 2 at 9 doesn't tie.
        assert_eq!(recall(&[0, 2], 2), 0.5, "{what}");
    }

    // Distances that the base doesn't give, and a file of the wrong shape, fail.
    write_fbin(&dists, &[1.0, 4.0, 10.0, 9.0, 9.0, 16.0], 6).unwrap();
    let wrong = open().unwrap();
    assert_eq!(
        wrong.recall(0, &[0, 1, 4], 3).unwrap_err().kind(),
        ErrorKind::InvalidData
    );
    write_fbin(&dists, &[1.0, 4.0, 9.0], 3).unwrap();
    assert_eq!(open().err().unwrap().kind(), ErrorKind::InvalidData);
    std::fs::remove_dir_all(&root).unwrap();
}

#[test]
fn distances_are_the_squared_length_or_minus_the_inner_product() {
    // 11 values leave a tail past the 8-value chunks. Halves of small whole numbers add up exactly in any order.
    let q: Vec<f32> = (0..11).map(|i| i as f32).collect();
    let x: Vec<f32> = (0..11).map(|i| (10 - i) as f32 * 0.5).collect();
    let squares: f32 = q.iter().zip(&x).map(|(a, b)| (a - b) * (a - b)).sum();
    let dot: f32 = q.iter().zip(&x).map(|(a, b)| a * b).sum();
    assert_eq!(distance(Metric::L2, &q, &x), squares);
    assert_eq!(distance(Metric::Cosine, &q, &x), -dot);
}

#[test]
fn splits_take_the_even_or_the_odd_positions() {
    assert_eq!(split(5, "tune"), Some(vec![0, 2, 4]));
    assert_eq!(split(5, "test"), Some(vec![1, 3]));
    assert_eq!(split(3, "all"), Some(vec![0, 1, 2]));
    assert_eq!(split(0, "test"), Some(vec![]));
    assert_eq!(split(4, "rest"), None);
}

#[test]
fn tune_and_test_take_the_first_thousand_queries_unless_asked() {
    // eg. wiki20m's 5,000 queries and glove's 10,000, then a dataset of 500.
    assert_eq!(split_count("tune", None, 5000), 1000);
    assert_eq!(split_count("test", None, 10_000), 1000);
    assert_eq!(split_count("test", None, 500), 500);
    // All takes every query, and --queries sets the head of any split, up to the dataset's count.
    assert_eq!(split_count("all", None, 5000), 5000);
    assert_eq!(split_count("tune", Some(2000), 5000), 2000);
    assert_eq!(split_count("all", Some(300), 5000), 300);
    assert_eq!(split_count("test", Some(9000), 5000), 5000);
}

/// A clustering of 5 vectors in 3 clusters of 4 values, in 2 groups when `grouped` is set.
fn small_clusters(grouped: bool) -> Clusters {
    let (parents, parent_of) = match grouped {
        true => (vec![0.5, -0.5, 1.5, 2.0, 3.0, -4.0, 0.0, 1.0], vec![1, 0, 1]),
        false => (Vec::new(), Vec::new()),
    };
    Clusters {
        dim: 4,
        centroids: (0..12).map(|i| i as f32 * 0.25 - 1.0).collect(),
        assignment: vec![2, 0, 1, 2, 0],
        parents,
        parent_of,
    }
}

#[test]
fn clusters_round_trip_through_their_files() {
    let stem = scratch("clusters/round-trip");
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    let saved = small_clusters(true);
    save_clusters(&stem, &saved).unwrap();
    let files = cluster_files(&stem);
    let names: Vec<String> = files
        .iter()
        .map(|f| f.file_name().unwrap().to_string_lossy().into_owned())
        .collect();
    assert_eq!(
        names,
        ["centroids.fbin", "assign.ibin", "parents.fbin", "parent_of.ibin"].map(|end| format!("round-trip.{end}"))
    );
    assert!(files.iter().all(|f| f.exists()));

    let loaded = load_clusters(&stem, 5, 4).unwrap();
    assert_eq!(loaded.dim, 4);
    assert_eq!(bits(&loaded.centroids), bits(&saved.centroids));
    assert_eq!(loaded.assignment, saved.assignment);
    assert_eq!(bits(&loaded.parents), bits(&saved.parents));
    assert_eq!(loaded.parent_of, saved.parent_of);
    // A build of fewer vectors takes the first ones' clusters.
    assert_eq!(load_clusters(&stem, 3, 4).unwrap().assignment, [2, 0, 1]);

    // Clusters without groups leave out the group files, so old ones go, and they load without groups.
    save_clusters(&stem, &small_clusters(false)).unwrap();
    assert!(!files[2].exists() && !files[3].exists());
    let bare = load_clusters(&stem, 5, 4).unwrap();
    assert!(bare.parents.is_empty() && bare.parent_of.is_empty());
    assert_eq!(bare.assignment, saved.assignment);
    std::fs::remove_dir_all(stem.parent().unwrap()).unwrap();
}

#[test]
fn cluster_files_of_the_wrong_shape_fail() {
    let stem = scratch("bad-clusters/c");
    let [_, assign, _, parent_of] = cluster_files(&stem);
    let kind = |n: usize, dim: usize| load_clusters(&stem, n, dim).err().unwrap().kind();
    assert_eq!(kind(5, 4), ErrorKind::NotFound);
    save_clusters(&stem, &small_clusters(true)).unwrap();
    load_clusters(&stem, 5, 4).unwrap();

    // Centers of another dimension, and more vectors than the assignment covers.
    assert_eq!(kind(5, 3), ErrorKind::InvalidData);
    assert_eq!(kind(6, 4), ErrorKind::InvalidData);
    // One group file without the other, too few groups, and a group below 0.
    std::fs::remove_file(&parent_of).unwrap();
    assert_eq!(kind(5, 4), ErrorKind::NotFound);
    write_ibin(&parent_of, &[0, 1], 1).unwrap();
    assert_eq!(kind(5, 4), ErrorKind::InvalidData);
    write_ibin(&parent_of, &[0, -1, 1], 1).unwrap();
    assert_eq!(kind(5, 4), ErrorKind::InvalidData);
    write_ibin(&parent_of, &[0, 1, 1], 1).unwrap();
    load_clusters(&stem, 5, 4).unwrap();
    // A cluster below 0, and two ids to a row.
    write_ibin(&assign, &[0, 1, -2, 0, 0], 1).unwrap();
    assert_eq!(kind(5, 4), ErrorKind::InvalidData);
    write_ibin(&assign, &[0; 10], 2).unwrap();
    assert_eq!(kind(5, 4), ErrorKind::InvalidData);

    // An id that an .ibin file can't hold doesn't save.
    let mut huge = small_clusters(false);
    huge.assignment[0] = u32::MAX;
    assert_eq!(save_clusters(&stem, &huge).unwrap_err().kind(), ErrorKind::InvalidInput);
    std::fs::remove_dir_all(stem.parent().unwrap()).unwrap();
}
