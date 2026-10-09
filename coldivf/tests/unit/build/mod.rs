//! Unit tests for building index files.

use std::io::ErrorKind;

use super::*;
use crate::format::{read_header, verify};
use crate::quantization::SplitMix64;
use crate::quantization::testutil::{Rows, clustered, dist2, gaussian, params, temp_file, temps_of};
use crate::{Clustering, Metric};

#[test]
fn a_built_file_checks_out() {
    let (rows, clusters) = clustered(&mut SplitMix64::new(1), 300, 128, 6, 0.5);
    let out = temp_file("built.coldivf");
    build_with_clusters(&params(Metric::L2), &rows, &clusters, &out).unwrap();

    let bytes = fs::read(&out).unwrap();
    let layout = read_header(&bytes, bytes.len()).unwrap();
    verify(&bytes, &layout).unwrap();
    assert_eq!((layout.dim, layout.count, layout.nlist), (128, 300, 6));
    assert!(temps_of(&out).is_empty(), "a temporary file is still there");
}

#[test]
fn the_same_input_gives_the_same_bytes_and_a_rebuild_replaces_the_file() {
    let (rows, clusters) = clustered(&mut SplitMix64::new(2), 200, 128, 4, 0.5);
    let (a, b) = (temp_file("same-a.coldivf"), temp_file("same-b.coldivf"));
    build_with_clusters(&params(Metric::Cosine), &rows, &clusters, &a).unwrap();
    build_with_clusters(&params(Metric::Cosine), &rows, &clusters, &b).unwrap();
    assert_eq!(fs::read(&a).unwrap(), fs::read(&b).unwrap());

    let other_seed = BuildParams {
        seed: 99,
        ..params(Metric::Cosine)
    };
    build_with_clusters(&other_seed, &rows, &clusters, &a).unwrap();
    let bytes = fs::read(&a).unwrap();
    assert_eq!(read_header(&bytes, bytes.len()).unwrap().seed, 99);
}

#[test]
fn a_rejected_build_writes_nothing() {
    let (rows, clusters) = clustered(&mut SplitMix64::new(3), 50, 128, 2, 0.5);
    let out = temp_file("rejected.coldivf");
    let odd_pages = BuildParams {
        page_size: 5000,
        ..params(Metric::L2)
    };

    let err = build_with_clusters(&odd_pages, &rows, &clusters, &out).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert!(!out.exists() && temps_of(&out).is_empty());
}

#[test]
fn a_failed_rename_leaves_no_temporary_file() {
    let (rows, clusters) = clustered(&mut SplitMix64::new(4), 50, 128, 2, 0.5);
    // A folder where the index should go makes the rename fail after the temporary file is written.
    let out = temp_file("taken.coldivf");
    fs::create_dir_all(out.join("inside")).unwrap();

    assert!(build_with_clusters(&params(Metric::L2), &rows, &clusters, &out).is_err());
    assert!(temps_of(&out).is_empty(), "a temporary file is still there");
    fs::remove_dir_all(&*out).unwrap();
}

#[test]
fn every_build_gets_its_own_temporary_file() {
    let out = temp_file("unique.coldivf");
    let ((_, a), (_, b)) = (create_temp(&out).unwrap(), create_temp(&out).unwrap());
    assert_ne!(a, b);

    fs::remove_file(a).unwrap();
    fs::remove_file(b).unwrap();
}

#[test]
fn the_builder_finds_clusters_that_search_well() {
    let mut rng = SplitMix64::new(5);
    let (dim, n) = (128, 3000);
    let (rows, _) = clustered(&mut rng, n, dim, 20, 1.0);
    let settings = BuildParams {
        vectors_per_list: 150,
        parents: 4,
        ..params(Metric::L2)
    };
    let some = crate::SearchParams {
        k: 10,
        nprobe: 5,
        rerank: 100,
        reach: None,
    };
    // Queries land near the data, and a quarter of the clusters holds most of their true neighbors.
    let queries: Vec<(Vec<f32>, Vec<u64>)> = (0..20)
        .map(|_| {
            let r = (rng.next_u64() % n as u64) as usize;
            let q: Vec<f32> = rows.data[r * dim..(r + 1) * dim]
                .iter()
                .map(|&x| x + (rng.normal() * 0.3) as f32)
                .collect();
            let mut exact: Vec<(f64, u64)> = rows
                .data
                .chunks_exact(dim)
                .enumerate()
                .map(|(i, o)| (dist2(&q, o), i as u64))
                .collect();
            exact.sort_by(|a, b| a.0.total_cmp(&b.0));
            (q, exact.iter().take(10).map(|e| e.1).collect())
        })
        .collect();

    for clustering in [Clustering::Flat, Clustering::TwoLevel] {
        let out = temp_file(&format!("kmeans-{clustering:?}.coldivf"));
        crate::Builder::new(settings.clone())
            .clustering(clustering)
            .build(&rows, &out)
            .unwrap();
        let bytes = fs::read(&out).unwrap();
        let layout = read_header(&bytes, bytes.len()).unwrap();
        // Two-level clustering rounds up in each group, which adds at most a cluster per group.
        assert!(
            layout.count == 3000 && (20..=24).contains(&layout.nlist),
            "{clustering:?}"
        );
        assert_eq!(layout.nparents, 4, "{clustering:?}");
        let index = crate::Index::open(&out).unwrap();
        index.verify().unwrap();
        let mut ctx = crate::SearchCtx::new(crate::Limits {
            max_k: 10,
            max_nprobe: 20,
            max_rerank: 100,
        });
        let found: usize = queries
            .iter()
            .map(|(q, truth)| {
                let hits = index.search(q, &some, &mut ctx).unwrap();
                hits.iter().filter(|h| truth.contains(&h.id)).count()
            })
            .sum();
        let recall = found as f64 / 200.0;
        assert!(
            recall >= 0.9,
            "{clustering:?} found {recall} of the true 10 nearest neighbors"
        );
    }
}

#[test]
fn a_build_gives_the_same_bytes_on_any_thread_count() {
    // k-means and the encode both give the same bytes for any number of threads. Large clusters keep k-means quick.
    let (rows, _) = clustered(&mut SplitMix64::new(6), 6000, 128, 30, 1.0);

    for metric in [Metric::L2, Metric::Cosine] {
        let builds: Vec<Vec<u8>> = [1, 3, 8]
            .into_iter()
            .map(|threads| {
                let settings = BuildParams {
                    threads,
                    vectors_per_list: 1000,
                    ..params(metric)
                };
                let out = temp_file(&format!("threads-{threads}.coldivf"));
                crate::Builder::new(settings).build(&rows, &out).unwrap();
                fs::read(&out).unwrap()
            })
            .collect();
        assert!(builds.windows(2).all(|w| w[0] == w[1]), "{metric:?}");
    }
}

#[test]
fn the_groups_never_outnumber_the_clusters() {
    let (rows, _) = clustered(&mut SplitMix64::new(7), 400, 128, 4, 0.5);
    let settings = BuildParams {
        vectors_per_list: 100,
        parents: 50,
        ..params(Metric::L2)
    };
    let clusters = cluster(&settings, &rows, &threads(2).unwrap()).unwrap();

    assert_eq!(clusters.centroids.len(), 4 * 128);
    assert_eq!(clusters.parents.len(), 4 * 128);
    assert!(clusters.parent_of.iter().all(|&g| g < 4) && clusters.assignment.iter().all(|&c| c < 4));
}

#[test]
fn bad_build_settings_fail_with_invalid_input() {
    let (rows, _) = clustered(&mut SplitMix64::new(8), 100, 128, 2, 0.5);
    let empty = Rows {
        dim: 128,
        data: Vec::new(),
    };
    let out = temp_file("bad-settings.coldivf");
    let cases = [
        (
            BuildParams {
                vectors_per_list: 0,
                ..params(Metric::L2)
            },
            &rows,
        ),
        (
            BuildParams {
                parents: 0,
                ..params(Metric::L2)
            },
            &rows,
        ),
        (params(Metric::L2), &empty),
    ];

    for (settings, src) in cases {
        let err = crate::Builder::new(settings).build(src, &out).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidInput);
        assert!(!out.exists() && temps_of(&out).is_empty());
    }
}

#[test]
fn clustering_runs_k_means_with_the_documented_settings() {
    let (rows, _) = clustered(&mut SplitMix64::new(9), 1000, 128, 10, 1.0);
    let settings = BuildParams {
        vectors_per_list: 120,
        parents: 3,
        ..params(Metric::Cosine)
    };
    let pool = threads(2).unwrap();
    let clusters = cluster(&settings, &rows, &pool).unwrap();

    // 1000 vectors at 120 each make 9 clusters, from 20 rounds on every vector. The groups use the next seed.
    let leaves = KMeans {
        k: 9,
        rounds: 20,
        sample: usize::MAX,
        spherical: false,
        seed: settings.seed,
    };
    let centroids = kmeans::train(&leaves, &rows, Metric::Cosine, &pool).unwrap();
    let assignment = kmeans::assign(&centroids, &rows, Metric::Cosine, &pool).unwrap();
    let centers = Slice {
        dim: 128,
        data: &centroids,
    };
    let groups = KMeans {
        k: 3,
        seed: settings.seed + 1,
        ..leaves
    };
    let parents = kmeans::train(&groups, &centers, Metric::L2, &pool).unwrap();
    let parent_of = kmeans::assign(&parents, &centers, Metric::L2, &pool).unwrap();
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&clusters.centroids), bits(&centroids));
    assert_eq!(bits(&clusters.parents), bits(&parents));
    assert_eq!((clusters.assignment, clusters.parent_of), (assignment, parent_of));
}

#[test]
fn building_from_the_clusters_a_build_finds_writes_the_same_file() {
    let (rows, _) = clustered(&mut SplitMix64::new(11), 2000, 128, 10, 1.0);
    let settings = BuildParams {
        vectors_per_list: 200,
        parents: 3,
        order: crate::LeafOrder::Chained,
        ..params(Metric::Cosine)
    };

    for clustering in [Clustering::Flat, Clustering::TwoLevel] {
        let builder = crate::Builder::new(settings.clone()).clustering(clustering);
        let (built, again) = (
            temp_file(&format!("found-{clustering:?}.coldivf")),
            temp_file(&format!("found-again-{clustering:?}.coldivf")),
        );
        builder.build(&rows, &built).unwrap();
        let clusters = builder.cluster(&rows, &again).unwrap();
        // Clustering alone writes no index and leaves nothing next to where it would go.
        assert!(!again.exists() && temps_of(&again).is_empty(), "{clustering:?}");
        builder.build_with_clusters(&rows, &clusters, &again).unwrap();
        assert_eq!(fs::read(&built).unwrap(), fs::read(&again).unwrap(), "{clustering:?}");
    }
}

#[test]
fn grouping_finds_the_groups_a_flat_build_finds() {
    let (rows, _) = clustered(&mut SplitMix64::new(12), 1000, 128, 10, 1.0);
    let settings = BuildParams {
        vectors_per_list: 100,
        parents: 3,
        ..params(Metric::L2)
    };
    let flat = cluster(&settings, &rows, &threads(2).unwrap()).unwrap();
    let mut bare = Clusters {
        dim: flat.dim,
        centroids: flat.centroids.clone(),
        assignment: flat.assignment.clone(),
        parents: Vec::new(),
        parent_of: Vec::new(),
    };
    crate::Builder::new(settings.clone()).group(&mut bare).unwrap();

    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&bare.parents), bits(&flat.parents));
    assert_eq!(bare.parent_of, flat.parent_of);
    assert_eq!(
        (bits(&bare.centroids), &bare.assignment),
        (bits(&flat.centroids), &flat.assignment)
    );

    // More groups than clusters make one group per cluster.
    let many = BuildParams {
        parents: 50,
        ..settings
    };
    crate::Builder::new(many).group(&mut bare).unwrap();
    assert_eq!(bare.parents.len(), 10 * 128);
    assert!(bare.parent_of.iter().all(|&g| g < 10));
}

#[test]
fn grouping_needs_whole_rows_of_finite_centers_and_a_group() {
    let (_, clusters) = clustered(&mut SplitMix64::new(13), 100, 128, 4, 0.5);
    let bare = |dim: usize, centroids: &[f32]| Clusters {
        dim,
        centroids: centroids.to_vec(),
        assignment: clusters.assignment.clone(),
        parents: Vec::new(),
        parent_of: Vec::new(),
    };
    let mut infinite = clusters.centroids.clone();
    infinite[200] = f32::INFINITY;
    let no_groups = BuildParams {
        parents: 0,
        ..params(Metric::L2)
    };
    let cases = [
        (bare(128, &[]), params(Metric::L2)),
        (bare(128, &clusters.centroids[1..]), params(Metric::L2)),
        (bare(0, &clusters.centroids), params(Metric::L2)),
        (bare(128, &infinite), params(Metric::L2)),
        (bare(128, &clusters.centroids), no_groups),
    ];

    for (i, (mut c, settings)) in cases.into_iter().enumerate() {
        let err = crate::Builder::new(settings).group(&mut c).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidInput, "case {i}");
        assert!(c.parents.is_empty() && c.parent_of.is_empty(), "case {i}");
    }
}

/// A source that counts the rows read from it.
struct Counted {
    dim: usize,
    count: usize,
    data: Vec<f32>,
    reads: std::cell::Cell<usize>,
}

impl VectorSource for Counted {
    fn count(&self) -> usize {
        self.count
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn rows(&self, start: usize, rows: usize) -> &[f32] {
        self.reads.set(self.reads.get() + rows);
        &self.data[start * self.dim..(start + rows) * self.dim]
    }
}

#[test]
fn bad_settings_fail_before_any_clustering() {
    let counted = |dim: usize, count: usize| Counted {
        dim,
        count,
        data: gaussian(&mut SplitMix64::new(10), count, dim, |_| 1.0),
        reads: std::cell::Cell::new(0),
    };
    let out = temp_file("early.coldivf");
    let odd_pages = BuildParams {
        page_size: 5000,
        ..params(Metric::L2)
    };
    let cases = [
        (counted(0, 10), params(Metric::L2)),
        (counted(0, 10), params(Metric::Cosine)),
        (counted(64, 500), params(Metric::L2)),
        (counted(128, 500), odd_pages),
    ];

    for (src, settings) in cases {
        let err = crate::Builder::new(settings).build(&src, &out).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidInput, "{} dimensions", src.dim);
        assert_eq!(
            src.reads.get(),
            0,
            "a bad setting was found only after reading the vectors"
        );
    }
}

#[test]
fn a_source_read_in_order_asks_for_each_next_stretch() {
    struct Logged {
        rows: Rows,
        asked: std::cell::RefCell<Vec<(usize, usize)>>,
    }

    impl VectorSource for Logged {
        fn count(&self) -> usize {
            self.rows.count()
        }

        fn dim(&self) -> usize {
            self.rows.dim()
        }

        fn rows(&self, start: usize, rows: usize) -> &[f32] {
            self.asked.borrow_mut().push((start, rows));
            self.rows.rows(start, rows)
        }
    }

    let (rows, _) = clustered(&mut SplitMix64::new(2), 10, 128, 2, 0.5);
    let logged = Logged {
        rows,
        asked: Default::default(),
    };
    let ahead = Ahead {
        src: &logged,
        stretch: 3,
        reads: ReadAhead::new(),
    };

    // Read a row at a time, it asks for the first stretch as it starts and for each next one as it reaches the one
    // before, never past the end, and hands out the source's rows.
    for r in 0..10 {
        assert_eq!(ahead.rows(r, 1), logged.rows.rows(r, 1));
    }
    let hints = [(0, 3), (3, 3), (6, 3), (9, 1)];
    let rows: Vec<(usize, usize)> = (0..10).map(|r| (r, 1)).collect();
    let mut want: Vec<(usize, usize)> = Vec::new();
    for r in 0..10 {
        if r == 0 {
            want.push(hints[0]);
        }
        if r % 3 == 0 && r + 3 < 10 {
            want.push(hints[r / 3 + 1]);
        }
        want.push(rows[r]);
    }
    assert_eq!(*logged.asked.borrow(), want);

    // A run asks for the stretch after each one that starts inside it.
    logged.asked.borrow_mut().clear();
    assert_eq!(ahead.rows(2, 5), logged.rows.rows(2, 5));
    assert_eq!(*logged.asked.borrow(), [(6, 3), (9, 1), (2, 5)]);
}
