//! Unit tests for searches over several collections.

use std::io::ErrorKind;

use super::*;
use crate::build::write;
use crate::quantization::SplitMix64;
use crate::quantization::testutil::{Rows, TempFile, clustered, gaussian, group, params, temp_file};
use crate::search::inmem::InMemory;
use crate::{Anchor, BuildParams, Collections, Limits};

/// `count` index files of `n` vectors each, from different data, with their in-memory indexes and their vectors. The
/// files take the cluster anchor and the mean anchor in turn, so a set of them searches both together.
fn files(name: &str, metric: Metric, count: usize, n: usize, dim: usize) -> (Vec<InMemory>, Vec<TempFile>, Vec<Rows>) {
    let mut rng = SplitMix64::new(41);
    let (mut memory, mut paths, mut data) = (Vec::new(), Vec::new(), Vec::new());

    for i in 0..count {
        let (rows, mut clusters) = clustered(&mut rng, n, dim, 12, 1.0);
        group(&mut clusters, 3);
        let p = BuildParams {
            anchor: [Anchor::Cluster, Anchor::Mean][i % 2],
            ..params(metric)
        };
        let index = InMemory::build(&p, &rows, &clusters).unwrap();
        let path = temp_file(&format!("multi-{name}-{i}.coldivf"));
        write(&index, &path).unwrap();
        memory.push(index);
        paths.push(path);
        data.push(rows);
    }

    (memory, paths, data)
}

fn refs(paths: &[TempFile]) -> Vec<&Path> {
    paths.iter().map(|p| p.as_ref()).collect()
}

fn ctx() -> SearchCtx {
    SearchCtx::new(Limits {
        max_k: 20,
        max_nprobe: 64,
        max_rerank: 2000,
    })
}

fn bits(hits: &[(usize, Hit)]) -> Vec<(usize, u64, u32)> {
    hits.iter().map(|(f, h)| (*f, h.id, h.score.to_bits())).collect()
}

#[test]
fn one_collection_searches_like_its_index() {
    let dim = 128;

    for metric in [Metric::L2, Metric::Cosine] {
        let (_, paths, rows) = files("one", metric, 1, 600, 128);
        let set = Collections::open(&[&paths[0]]).unwrap();
        let index = Index::open(&paths[0]).unwrap();
        let (mut a, mut b, mut out) = (ctx(), ctx(), Vec::new());
        // Queries on the data, so the reach keeps fewer clusters than nprobe. The cosine files route by centers of the
        // data before it was normalized, which sit about as far from every query, so only L2 is sure to cut.
        let near = rows[0].data[..8 * dim].iter().map(|&x| x + 0.1);
        let queries: Vec<f32> = near
            .chain(gaussian(&mut SplitMix64::new(42), 4, 128, |_| 1.0))
            .collect();
        let mut leaves = Vec::new();
        index.route(&queries[..dim], 5, Some(0.2), &mut b, &mut leaves).unwrap();
        assert!(metric == Metric::Cosine || leaves.len() < 5);
        for q in queries.chunks_exact(dim) {
            for reach in [None, Some(0.2)] {
                let sp = SearchParams {
                    k: 10,
                    nprobe: 5,
                    rerank: 40,
                    reach,
                };
                set.search(q, &sp, &mut a, &mut out).unwrap();
                let want: Vec<(usize, Hit)> = index.search(q, &sp, &mut b).unwrap().iter().map(|&h| (0, h)).collect();
                assert_eq!(bits(&out), bits(&want), "{metric:?}, {reach:?}");
            }
        }
    }
}

#[test]
fn reading_everything_finds_the_nearest_rows_of_every_collection() {
    let dim = 128;

    for metric in [Metric::L2, Metric::Cosine] {
        let (memory, paths, _) = files("all", metric, 3, 400, 128);
        let set = Collections::open(&refs(&paths)).unwrap();
        let (mut a, mut b, mut out) = (ctx(), ctx(), Vec::new());
        // Every cluster of every collection and every vector rescored.
        let all = SearchParams {
            k: 20,
            nprobe: 36,
            rerank: 1200,
            reach: None,
        };
        let each = SearchParams {
            nprobe: 12,
            rerank: 400,
            ..all
        };
        for q in gaussian(&mut SplitMix64::new(43), 5, 128, |_| 1.0).chunks_exact(dim) {
            set.search(q, &all, &mut a, &mut out).unwrap();
            // The k nearest of the collections' own k nearest, with ties to the lower collection.
            let mut want: Vec<(usize, Hit)> = memory
                .iter()
                .enumerate()
                .flat_map(|(f, m)| {
                    m.search(q, &each, &mut b)
                        .unwrap()
                        .iter()
                        .map(|&h| (f, h))
                        .collect::<Vec<_>>()
                })
                .collect();
            want.sort_by(|x, y| {
                match metric {
                    Metric::L2 => x.1.score.total_cmp(&y.1.score),
                    Metric::Cosine => y.1.score.total_cmp(&x.1.score),
                }
                .then(x.0.cmp(&y.0))
            });
            want.truncate(20);
            assert_eq!(bits(&out), bits(&want), "{metric:?}");
        }
    }
}

#[test]
fn the_same_file_twice_gives_every_hit_twice() {
    let (_, paths, _) = files("twice", Metric::L2, 1, 600, 128);
    let set = Collections::open(&[&paths[0], &paths[0]]).unwrap();
    let (mut a, mut out) = (ctx(), Vec::new());
    let sp = SearchParams {
        k: 10,
        nprobe: 10,
        rerank: 40,
        reach: None,
    };

    set.search(
        &gaussian(&mut SplitMix64::new(44), 1, 128, |_| 1.0),
        &sp,
        &mut a,
        &mut out,
    )
    .unwrap();

    for pair in out.as_chunks::<2>().0 {
        assert_eq!((pair[0].0, pair[1].0, pair[0].1.id), (0, 1, pair[1].1.id));
    }
}

#[test]
fn both_copies_of_a_file_named_twice_move_with_its_writer_s_commits() {
    let (_, paths, data) = files("twice-commit", Metric::L2, 1, 600, 128);
    let mut writer = crate::Writer::open(&paths[0]).unwrap();
    writer.compact().unwrap();
    let set = Collections::open(&[&paths[0], &paths[0]]).unwrap();
    let q = &data[0].data[..128];
    let sp = SearchParams {
        k: 20,
        nprobe: 12,
        rerank: 200,
        reach: None,
    };
    let (mut c, mut out) = (ctx(), Vec::new());
    let mut found = |id: u64| {
        set.search(q, &sp, &mut c, &mut out).unwrap();
        let mut copies: Vec<usize> = out.iter().filter(|(_, h)| h.id == id).map(|(f, _)| *f).collect();
        copies.sort_unstable();
        copies
    };
    writer.insert(700_000, q).unwrap();
    assert_eq!(found(700_000), [0, 1]);
    writer.compact().unwrap();
    writer.insert(700_001, q).unwrap();
    writer.delete(700_000).unwrap();
    assert_eq!(found(700_001), [0, 1]);
    assert_eq!(found(700_000), [] as [usize; 0]);
    // Both copies share one cell, and each has reads of its own.
    assert!(Arc::ptr_eq(&set.indexes[0].shared, &set.indexes[1].shared));
    assert_eq!(c.reads.len(), 2);
}

#[test]
fn warm_and_cold_collections_search_alike() {
    let dim = 128;
    let (_, paths, _) = files("warm", Metric::L2, 2, 500, 128);
    let set = Collections::open(&refs(&paths)).unwrap();
    let (mut a, mut cold, mut warm) = (ctx(), Vec::new(), Vec::new());
    let sp = SearchParams {
        k: 10,
        nprobe: 8,
        rerank: 60,
        reach: None,
    };

    for q in gaussian(&mut SplitMix64::new(45), 6, 128, |_| 1.0).chunks_exact(dim) {
        set.indexes.iter().for_each(|i| i.set_warm(false));
        set.search(q, &sp, &mut a, &mut cold).unwrap();
        // The first file takes the cluster anchor and the second the mean.
        for (f, what) in [(1, "the mean file warm"), (0, "both warm")] {
            set.indexes[f].set_warm(true);
            set.search(q, &sp, &mut a, &mut warm).unwrap();
            assert_eq!(bits(&warm), bits(&cold), "{what}");
        }
    }
}

#[test]
fn collections_must_share_a_shape() {
    let (_, l2, _) = files("shape-l2", Metric::L2, 1, 300, 128);
    let (_, cosine, _) = files("shape-cosine", Metric::Cosine, 1, 300, 128);
    let (_, wide, _) = files("shape-wide", Metric::L2, 1, 300, 200);
    let kind = |paths: &[&Path]| Collections::open(paths).map(|_| ()).unwrap_err().kind();

    assert_eq!(kind(&[]), ErrorKind::InvalidInput);
    assert_eq!(kind(&[&l2[0], &cosine[0]]), ErrorKind::InvalidInput);
    assert_eq!(kind(&[&l2[0], &wide[0]]), ErrorKind::InvalidInput);
    let set = Collections::open(&[&l2[0]]).unwrap();
    let too_deep = SearchParams {
        k: 10,
        nprobe: 4,
        rerank: 5000,
        reach: None,
    };
    let err = set
        .search(&[0.5; 128], &too_deep, &mut ctx(), &mut Vec::new())
        .unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidInput);
}

#[test]
fn a_context_used_before_with_other_indexes_still_searches_right() {
    let (_, paths, _) = files("reused", Metric::L2, 2, 500, 128);
    let (_, others, _) = files("reused-other", Metric::L2, 1, 300, 128);
    let refs = refs(&paths);
    let sp = SearchParams {
        k: 10,
        nprobe: 8,
        rerank: 60,
        reach: None,
    };
    let q = gaussian(&mut SplitMix64::new(46), 1, 128, |_| 1.0);
    let (mut fresh, mut used, mut want, mut got) = (ctx(), ctx(), Vec::new(), Vec::new());
    Collections::open(&refs)
        .unwrap()
        .search(&q, &sp, &mut fresh, &mut want)
        .unwrap();

    // The other index's reads come first in the context, and closing it moves the rest when new reads are made.
    let other = Index::open(&others[0]).unwrap();
    other.search(&q, &sp, &mut used).unwrap();
    drop(other);
    Collections::open(&refs)
        .unwrap()
        .search(&q, &sp, &mut used, &mut got)
        .unwrap();
    assert_eq!(bits(&got), bits(&want));
}

/// The hits of a search of `set`, as `bits` gives them.
fn hits_of(set: &Collections, q: &[f32], sp: &SearchParams, ctx: &mut SearchCtx) -> Vec<(usize, u64, u32)> {
    let mut out = Vec::new();
    set.search(q, sp, ctx, &mut out).unwrap();
    bits(&out)
}

#[test]
fn the_nearest_clusters_and_best_candidates_of_any_collection_win() {
    let dim = 128;
    let (_, paths, rows) = files("pick", Metric::L2, 2, 400, 128);
    let set = Collections::open(&refs(&paths)).unwrap();
    let mut c = ctx();
    let sp = SearchParams {
        k: 10,
        nprobe: 6,
        rerank: 30,
        reach: None,
    };

    for q in gaussian(&mut SplitMix64::new(47), 6, 128, |_| 1.0).chunks_exact(dim) {
        hits_of(&set, q, &sp, &mut c);
        let parts = &c.parts[..2];
        // Each collection reads the first of its own routes, and none it left out beats one that another kept.
        assert_eq!(parts.iter().map(|p| p.leaves.len()).sum::<usize>(), 6);
        for (f, part) in parts.iter().enumerate() {
            let mut own = Vec::new();
            set.indexes[f]
                .route(q, part.leaves.len(), None, &mut ctx(), &mut own)
                .unwrap();
            assert_eq!(part.leaves, own);
        }
        let kept = parts
            .iter()
            .filter_map(|p| p.s.routes.get(p.leaves.len().wrapping_sub(1)));
        let mut left = parts.iter().filter_map(|p| p.s.routes.get(p.leaves.len()));
        let worst_kept = kept.map(|r| r.score).fold(f32::MIN, f32::max);
        assert!(left.all(|r| r.score >= worst_kept));
        // The rows of the best 30 candidates overall are read, wherever they are.
        assert_eq!(parts.iter().map(|p| p.chosen.len()).sum::<usize>(), 30);
        let mut all: Vec<Candidate> = parts.iter().flat_map(|p| p.s.candidates.as_slice().to_vec()).collect();
        all.sort_unstable_by(order);
        let worst_chosen = parts
            .iter()
            .flat_map(|p| &p.chosen)
            .map(|c| c.score)
            .fold(f32::MIN, f32::max);
        assert!(all[30..].iter().all(|c| c.score >= worst_chosen));
    }

    // A reach counts from the nearest cluster of any collection, so a query on one collection's data reads none of
    // the other's far clusters.
    let near = SearchParams { reach: Some(0.1), ..sp };
    let q = &rows[0].data[..dim];
    hits_of(&set, q, &near, &mut c);
    let mut own = Vec::new();
    set.indexes[0].route(q, 6, Some(0.1), &mut ctx(), &mut own).unwrap();
    assert!(c.parts[1].leaves.is_empty() && c.parts[0].leaves == own && own.len() < 6);
}

#[test]
fn a_context_shared_with_other_searches_keeps_each_collection_s_reads_apart() {
    let dim = 128;
    let (_, paths, _) = files("slots", Metric::L2, 2, 500, 128);
    let (_, others, _) = files("slots-other", Metric::L2, 2, 300, 128);
    let (_, wide, _) = files("slots-wide", Metric::L2, 2, 300, 200);
    let (set, wide_set) = (
        Collections::open(&refs(&paths)).unwrap(),
        Collections::open(&refs(&wide)).unwrap(),
    );
    let sp = SearchParams {
        k: 10,
        nprobe: 8,
        rerank: 60,
        reach: None,
    };
    let q = gaussian(&mut SplitMix64::new(48), 1, dim, |_| 1.0);
    let wide_q = gaussian(&mut SplitMix64::new(49), 1, 200, |_| 1.0);
    let (want, wide_want) = (
        hits_of(&set, &q, &sp, &mut ctx()),
        hits_of(&wide_set, &wide_q, &sp, &mut ctx()),
    );

    // Another index searches first, then one collection alone, then the other index closes.
    let mut c = ctx();
    let other = Index::open(&others[0]).unwrap();
    other.search(&q, &sp, &mut c).unwrap();
    set.indexes[0].search(&q, &sp, &mut c).unwrap();
    drop(other);
    assert_eq!(hits_of(&set, &q, &sp, &mut c), want);

    // The set searches, an index closes, another searches, and the set searches again.
    let mut c = ctx();
    let other = Index::open(&others[0]).unwrap();
    other.search(&q, &sp, &mut c).unwrap();
    assert_eq!(hits_of(&set, &q, &sp, &mut c), want);
    drop(other);
    Index::open(&others[1]).unwrap().search(&q, &sp, &mut c).unwrap();
    assert_eq!(hits_of(&set, &q, &sp, &mut c), want);

    // A set of another dimension, then this set again.
    assert_eq!(hits_of(&wide_set, &wide_q, &sp, &mut c), wide_want);
    assert_eq!(hits_of(&set, &q, &sp, &mut c), want);
}

#[cfg(feature = "bench")]
#[test]
fn a_search_of_several_collections_traces_its_own_stages_and_reads() {
    let (_, paths, _) = files("trace", Metric::L2, 2, 500, 128);
    let set = Collections::open(&refs(&paths)).unwrap();
    let q = gaussian(&mut SplitMix64::new(50), 1, 128, |_| 1.0);
    let sp = SearchParams {
        k: 10,
        nprobe: 8,
        rerank: 60,
        reach: None,
    };
    let (zero, page) = (std::time::Duration::ZERO, set.indexes[0].inner().read_page());
    let mut c = ctx();
    // A search of one index leaves its trace first, and the set's search must start a new one.
    Index::open(&paths[0]).unwrap().search(&q, &sp, &mut c).unwrap();

    for warm in [false, true] {
        set.indexes[1].set_warm(warm);
        let start = std::time::Instant::now();
        set.search(&q, &sp, &mut c, &mut Vec::new()).unwrap();
        let took = start.elapsed();
        let t = c.trace();
        assert!(
            t.tables + t.buffers + t.route + t.submit + t.wave + t.scan + t.rows + t.rerank <= took,
            "{t:?}"
        );
        // The set's first search loads its tables and makes its parts and its read buffers.
        assert!(warm || (t.tables > zero && t.buffers > zero), "{t:?}");
        assert!(t.route > zero && t.submit > zero && t.scan > zero && t.rows > zero && t.rerank > zero);
        // On macOS the clusters arrive while the scan waits for them.
        assert!(cfg!(not(target_os = "macos")) || t.wave > zero, "{t:?}");
        assert!(t.row_ranges > 0 && t.row_bytes.is_multiple_of(page) && t.row_bytes >= t.row_ranges * page);

        // It scanned every collection's clusters but read only the cold ones', the same bytes that a fetch of them
        // from each index alone reads.
        let (mut scanned, mut read) = (0, (0, 0));
        for (f, index) in set.indexes.iter().enumerate() {
            let leaves = c.parts[f].leaves.clone();
            let inner = index.inner();
            let lists = &inner.tables().unwrap().lists;
            scanned += leaves.iter().map(|&l| lists[l as usize].count).sum::<usize>();
            if f == 0 || !warm {
                let mut alone = ctx();
                index.fetch(&leaves, &mut alone).unwrap();
                read = (
                    read.0 + alone.trace().cluster_bytes,
                    read.1 + alone.trace().cluster_ranges,
                );
            }
        }
        assert_eq!(
            (t.scanned, t.cluster_bytes, t.cluster_ranges),
            (scanned, read.0, read.1)
        );
    }
}

#[test]
fn a_context_keeps_a_part_per_collection_whatever_their_cluster_counts() {
    let (dim, mut rng) = (128, SplitMix64::new(47));
    let paths: Vec<TempFile> = [8, 12, 16]
        .into_iter()
        .map(|nlist| {
            let (rows, mut clusters) = clustered(&mut rng, 400, dim, nlist, 1.0);
            group(&mut clusters, 3);
            let path = temp_file(&format!("multi-counts-{nlist}.coldivf"));
            write(&InMemory::build(&params(Metric::L2), &rows, &clusters).unwrap(), &path).unwrap();
            path
        })
        .collect();
    let sp = SearchParams {
        k: 10,
        nprobe: 6,
        rerank: 60,
        reach: None,
    };
    let q = gaussian(&mut SplitMix64::new(48), 1, dim, |_| 1.0);
    let mut shared = ctx();

    // Every maintenance run changes a file's cluster count, so one context meets sets of other counts in turn.
    for round in 0..2 {
        for [a, b] in [[0, 1], [2, 0], [1, 2]] {
            let set = Collections::open(&[&paths[a], &paths[b]]).unwrap();
            let want = hits_of(&set, &q, &sp, &mut ctx());
            assert_eq!(
                hits_of(&set, &q, &sp, &mut shared),
                want,
                "round {round}, files {a} and {b}"
            );
        }
    }

    assert_eq!(shared.parts.len(), 2, "a part per collection of a set");
}
