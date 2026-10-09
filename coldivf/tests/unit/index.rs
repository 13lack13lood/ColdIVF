//! Unit tests for opening and searching index files.

use std::io::ErrorKind;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use super::*;
use crate::format::Layout;
use crate::io::pool::PoolWave;
use crate::io::portable::PortableWave;
use crate::io::{Memory, NewWave, Range, Wave};
use crate::kernels::Tier;
use crate::quantization::SplitMix64;
use crate::quantization::testutil::{
    EndsEarly, Faulty, Rows, Shuffled, TempFile, clustered, gaussian, group, params, shuffle, temp_file, written,
};
use crate::search::inmem::InMemory;
use crate::{Anchor, BuildParams, Index, LeafOrder, Limits, Metric};

fn ctx() -> SearchCtx {
    SearchCtx::new(Limits {
        max_k: 50,
        max_nprobe: 64,
        max_rerank: 200,
    })
}

fn bits(hits: &[Hit]) -> Vec<(u64, u32)> {
    hits.iter().map(|h| (h.id, h.score.to_bits())).collect()
}

fn public(inner: Inner) -> Index {
    Index::of(inner)
}

/// Every read backend, by name.
fn waves() -> Vec<(&'static str, NewWave)> {
    vec![
        ("portable", || Box::<PortableWave>::default()),
        ("pool", || Box::<PoolWave>::default()),
        ("shuffled", || Box::<Shuffled>::default()),
        #[cfg(target_os = "macos")]
        ("rdadvise", || Box::<crate::io::macos::RdAdviseWave>::default()),
        #[cfg(target_os = "linux")]
        ("uring", || Box::<crate::io::linux::UringWave>::default()),
        #[cfg(target_os = "linux")]
        ("fadvise", || Box::<crate::io::linux::FadviseWave>::default()),
    ]
}

/// Both anchors, the default first.
const ANCHORS: [Anchor; 2] = [Anchor::Cluster, Anchor::Mean];

#[test]
fn cold_and_mapped_searches_give_exactly_the_in_memory_hits() {
    let mut rng = SplitMix64::new(2);
    let cases = [
        (Metric::L2, LeafOrder::Id, 128, 4096),
        (Metric::L2, LeafOrder::Chained, 200, 4096),
        (Metric::Cosine, LeafOrder::Chained, 768, 4096),
        (Metric::L2, LeafOrder::Chained, 768, 16384),
        (Metric::L2, LeafOrder::Chained, 3000, 4096),
    ];
    let settings = [
        SearchParams {
            k: 10,
            nprobe: 3,
            rerank: 30,
            reach: None,
        },
        SearchParams {
            k: 50,
            nprobe: 12,
            rerank: 200,
            reach: None,
        },
    ];

    for ((i, (metric, order, dim, page_size)), anchor) in
        cases.into_iter().enumerate().flat_map(|c| ANCHORS.map(|a| (c, a)))
    {
        let queries = gaussian(&mut rng, 4, dim, |_| 1.0);
        for tier in Tier::supported() {
            let params = BuildParams {
                order,
                page_size,
                anchor,
                ..params(metric)
            };
            let (index, file, path) = written(&format!("exact-{i}-{anchor:?}.coldivf"), &params, 400, dim, 12);
            // Each file opens with the anchor it was built with, and checks out.
            assert_eq!(file.layout().anchor, anchor);
            file.verify().unwrap();
            let (index, file) = (index.with_tier(tier), file.with_tier(tier));
            let colds: Vec<(&str, Index)> = waves()
                .into_iter()
                .map(|(name, wave)| {
                    (
                        name,
                        public(Inner::open(&path).unwrap().with_tier(tier).with_wave(wave)),
                    )
                })
                .collect();
            let (mut a, mut b, mut c) = (ctx(), ctx(), ctx());
            let (mut routes_a, mut routes_b) = (Vec::new(), Vec::new());
            for q in queries.chunks_exact(dim) {
                for sp in &settings {
                    let want = bits(index.search(q, sp, &mut a).unwrap());
                    assert_eq!(
                        bits(file.search_mapped(q, sp, &mut b).unwrap()),
                        want,
                        "case {i} with the {anchor:?} anchor on {tier:?}"
                    );
                    for (name, cold) in &colds {
                        let got = bits(cold.search(q, sp, &mut c).unwrap());
                        assert_eq!(
                            got, want,
                            "case {i} with the {anchor:?} anchor on {tier:?} with the {name} reader"
                        );
                    }
                }
                index.route(q, 5, None, &mut a, &mut routes_a).unwrap();
                file.route(q, 5, None, &mut b, &mut routes_b).unwrap();
                assert_eq!(routes_a, routes_b);
            }
        }
    }
}

#[test]
fn prepare_then_search_works() {
    let (index, file, _path) = written("prepare.coldivf", &params(Metric::L2), 100, 128, 12);
    let file = public(file);
    let (mut a, mut b) = (ctx(), ctx());
    file.prepare(&mut b).unwrap();
    let q = vec![0.5f32; 128];
    let sp = SearchParams {
        k: 5,
        nprobe: 4,
        rerank: 20,
        reach: None,
    };

    assert_eq!(
        bits(file.search(&q, &sp, &mut b).unwrap()),
        bits(index.search(&q, &sp, &mut a).unwrap())
    );
}

#[test]
fn a_damaged_file_fails_with_invalid_data() {
    let (_, file, path) = written("broken.coldivf", &params(Metric::L2), 100, 128, 12);
    let (bytes, layout) = file.parts();
    let bytes = bytes.to_vec();
    let extents = layout.span(Part::Extents);
    drop(file);

    let reopen = |b: &[u8]| {
        std::fs::write(&path, b).unwrap();
        Inner::open(&path).map(|_| ())
    };
    let mut flipped = bytes.clone();
    flipped[40] ^= 1;
    assert_eq!(reopen(&flipped).unwrap_err().kind(), ErrorKind::InvalidData);
    assert_eq!(
        reopen(&bytes[..bytes.len() - 10]).unwrap_err().kind(),
        ErrorKind::InvalidData
    );
    assert_eq!(reopen(b"not an index").unwrap_err().kind(), ErrorKind::InvalidData);

    // Damaged extents open, since opening doesn't read them, but the first search finds them.
    let mut misnumbered = bytes.clone();
    misnumbered[extents.offset + 16 + 12] ^= 1;
    std::fs::write(&path, &misnumbered).unwrap();
    let file = public(Inner::open(&path).unwrap());
    let sp = SearchParams {
        k: 5,
        nprobe: 4,
        rerank: 20,
        reach: None,
    };
    let mapped = file.inner().search_mapped(&[0.5; 128], &sp, &mut ctx()).map(|_| ());
    assert_eq!(mapped.unwrap_err().kind(), ErrorKind::InvalidData);
    let cold = file.search(&[0.5; 128], &sp, &mut ctx()).map(|_| ());
    assert_eq!(cold.unwrap_err().kind(), ErrorKind::InvalidData);
}

#[test]
fn the_public_index_builds_opens_and_searches() {
    let mut rng = SplitMix64::new(3);
    let (rows, mut clusters) = clustered(&mut rng, 300, 128, 8, 0.5);
    group(&mut clusters, 2);
    let params = BuildParams {
        order: LeafOrder::Chained,
        ..params(Metric::L2)
    };
    let index = InMemory::build(&params, &rows, &clusters).unwrap();
    let path = temp_file("public.coldivf");
    crate::Builder::new(params)
        .build_with_clusters(&rows, &clusters, &path)
        .unwrap();

    let file = crate::Index::open(&path).unwrap();
    let shared = file.clone();
    let (mut a, mut b) = (ctx(), ctx());
    file.prepare(&mut b).unwrap();
    let sp = SearchParams {
        k: 10,
        nprobe: 4,
        rerank: 40,
        reach: None,
    };
    let q = &rows.data[..128];
    assert_eq!(
        bits(shared.search(q, &sp, &mut b).unwrap()),
        bits(index.search(q, &sp, &mut a).unwrap())
    );

    fn shareable<T: Send + Sync + Clone>(_: &T) {}
    fn sendable<T: Send>(_: &T) {}
    shareable(&file);
    sendable(&b);
}

/// An index of 3000 vectors in 64 clusters, with 30 random queries.
fn sample(name: &str) -> (InMemory, Index, TempFile, Vec<Vec<f32>>) {
    let dim = 128;
    let (index, file, path) = written(name, &params(Metric::L2), 3000, dim, 64);
    let queries = gaussian(&mut SplitMix64::new(4), 30, dim, |_| 1.0);
    (
        index,
        public(file),
        path,
        queries.chunks_exact(dim).map(<[f32]>::to_vec).collect(),
    )
}

#[test]
fn a_fetch_of_the_routed_clusters_searches_like_search() {
    let (_, index, _path, queries) = sample("fetched.coldivf");
    let sp = SearchParams {
        k: 10,
        nprobe: 16,
        rerank: 40,
        reach: None,
    };
    let (mut a, mut b) = (ctx(), ctx());
    let mut rng = SplitMix64::new(5);
    let mut leaves = Vec::new();

    for q in &queries {
        let want = bits(index.search(q, &sp, &mut a).unwrap());
        // The leaves may come in any order, and repeats are read once.
        index.route(q, sp.nprobe, None, &mut b, &mut leaves).unwrap();
        leaves.extend_from_within(..5);
        shuffle(&mut leaves, &mut rng);
        let fetched = index.fetch(&leaves, &mut b).unwrap();
        assert_eq!(bits(fetched.search(q, sp.k, sp.rerank).unwrap()), want);
    }
}

#[test]
fn rows_prefetched_for_a_guess_never_change_the_hits() {
    let (_, index, _path, queries) = sample("guessed.coldivf");
    let sp = SearchParams {
        k: 10,
        nprobe: 16,
        rerank: 40,
        reach: None,
    };
    let (mut a, mut b) = (ctx(), ctx());
    let mut rng = SplitMix64::new(6);
    let mut leaves = Vec::new();

    for q in &queries {
        let want = bits(index.search(q, &sp, &mut a).unwrap());
        let noise = gaussian(&mut rng, 1, 128, |_| 0.3);
        let near: Vec<f32> = q.iter().zip(&noise).map(|(x, e)| x + e).collect();
        let far = gaussian(&mut rng, 1, 128, |_| 1.0);
        let guesses: [&[(&[f32], usize)]; 5] = [
            &[],
            &[(&near, sp.rerank)],
            &[(&near, 0)],
            &[(&far, 200)],
            &[(&far, 30), (q, sp.rerank), (&near, 100)],
        ];
        index.route(q, sp.nprobe, None, &mut b, &mut leaves).unwrap();
        for (g, guess) in guesses.iter().enumerate() {
            let mut fetched = index.fetch(&leaves, &mut b).unwrap();
            for &(e, n) in guess.iter() {
                fetched.prefetch_rows(e, n).unwrap();
            }
            assert_eq!(bits(fetched.search(q, sp.k, sp.rerank).unwrap()), want, "guess {g}");
        }
    }
}

#[test]
fn a_fetch_ranks_only_its_clusters() {
    let (memory, index, _path, queries) = sample("only.coldivf");
    let mut b = ctx();
    let leaves = [3u32, 10, 11, 40, 63];
    let ids_of = |leaves: &[u32]| -> Vec<u64> {
        let lists = leaves.iter().map(|&l| memory.lists[l as usize]);
        lists
            .flat_map(|l| (l.first_id..l.first_id + l.count as u32).map(|i| memory.ids[i as usize]))
            .collect()
    };
    let allowed = ids_of(&leaves);

    for q in &queries {
        let hits = index.fetch(&leaves, &mut b).unwrap().search(q, 50, 200).unwrap();
        assert_eq!(hits.len(), 50.min(allowed.len()));
        assert!(hits.iter().all(|h| allowed.contains(&h.id)));

        // Fetching every cluster is a search of all of them, and fetching none finds nothing.
        let all: Vec<u32> = (0..64).collect();
        let every = bits(index.fetch(&all, &mut b).unwrap().search(q, 20, 100).unwrap());
        let sp = SearchParams {
            k: 20,
            nprobe: 64,
            rerank: 100,
            reach: None,
        };
        assert_eq!(every, bits(memory.search(q, &sp, &mut ctx()).unwrap()));
        assert!(index.fetch(&[], &mut b).unwrap().search(q, 10, 40).unwrap().is_empty());
    }
}

#[test]
fn dropping_a_fetch_at_any_point_is_safe() {
    let (_, index, _path, queries) = sample("dropped.coldivf");
    let sp = SearchParams {
        k: 10,
        nprobe: 16,
        rerank: 40,
        reach: None,
    };
    let (mut a, mut b) = (ctx(), ctx());
    let mut leaves = Vec::new();

    for q in queries.iter().take(5) {
        let want = bits(index.search(q, &sp, &mut a).unwrap());
        index.route(q, sp.nprobe, None, &mut b, &mut leaves).unwrap();
        let _ = index.fetch(&leaves, &mut b).unwrap();
        index.fetch(&leaves, &mut b).unwrap().prefetch_rows(q, 10).unwrap();
        assert_eq!(bits(index.search(q, &sp, &mut b).unwrap()), want);
    }
}

#[test]
fn bad_fetch_arguments_fail_with_invalid_input() {
    let (_, index, _path, queries) = sample("bad-fetch.coldivf");
    let q = &queries[0];
    let mut b = ctx();
    let kind = |r: Result<(), Error>| r.unwrap_err().kind();

    let too_many: Vec<u32> = (0..65).map(|i| i % 64).collect();
    assert_eq!(
        kind(index.fetch(&too_many, &mut b).map(|_| ())),
        ErrorKind::InvalidInput
    );
    assert_eq!(kind(index.fetch(&[2, 64], &mut b).map(|_| ())), ErrorKind::InvalidInput);

    let mut fetched = index.fetch(&[1, 2, 3], &mut b).unwrap();
    assert_eq!(kind(fetched.prefetch_rows(q, 201)), ErrorKind::InvalidInput);
    assert_eq!(kind(fetched.prefetch_rows(&q[1..], 10)), ErrorKind::InvalidInput);
    assert_eq!(
        kind(fetched.prefetch_rows(&[f32::NAN; 128], 10)),
        ErrorKind::InvalidInput
    );
    fetched.prefetch_rows(q, 200).unwrap();
    let search = |k, rerank, b: &mut SearchCtx| index.fetch(&[1], b).unwrap().search(q, k, rerank).map(|_| ());
    assert_eq!(kind(search(51, 10, &mut b)), ErrorKind::InvalidInput);
    assert_eq!(kind(search(10, 201, &mut b)), ErrorKind::InvalidInput);

    let wide = SearchParams {
        k: 10,
        nprobe: 65,
        rerank: 40,
        reach: None,
    };
    assert_eq!(
        kind(index.search(q, &wide, &mut b).map(|_| ())),
        ErrorKind::InvalidInput
    );
}

#[test]
fn a_failed_read_fails_the_search_instead_of_dropping_clusters() {
    let (memory, file, path) = written("faulty.coldivf", &params(Metric::L2), 3000, 128, 64);
    let q = &gaussian(&mut SplitMix64::new(7), 1, 128, |_| 1.0)[..];
    let sp = SearchParams {
        k: 10,
        nprobe: 16,
        rerank: 40,
        reach: None,
    };
    let want = bits(memory.search(q, &sp, &mut ctx()).unwrap());
    drop(file);
    let open = |wave: NewWave| public(Inner::open(&path).unwrap().with_wave(wave));
    let mut leaves = Vec::new();

    // The first read of the clusters fails. Every later use of that fetch fails too, and a new search starts over.
    let index = open(|| Box::<Faulty<0, 1>>::default());
    let mut b = ctx();
    index.route(q, sp.nprobe, None, &mut b, &mut leaves).unwrap();
    let mut fetched = index.fetch(&leaves, &mut b).unwrap();
    assert!(fetched.prefetch_rows(q, 40).is_err());
    assert!(fetched.prefetch_rows(q, 40).is_err());
    assert!(fetched.search(q, sp.k, sp.rerank).is_err());
    assert_eq!(bits(index.search(q, &sp, &mut b).unwrap()), want);

    // The early row wave fails to start, so the search reads all of its rows itself.
    let index = open(|| Box::<Faulty<2, 0>>::default());
    let mut fetched = index.fetch(&leaves, &mut b).unwrap();
    assert!(fetched.prefetch_rows(q, 40).is_err());
    assert_eq!(bits(fetched.search(q, sp.k, sp.rerank).unwrap()), want);

    // The first read of the row wave fails, and the next search starts over.
    let index = open(|| Box::<Faulty<0, 2>>::default());
    let mut b = ctx();
    assert!(index.search(q, &sp, &mut b).is_err());
    assert_eq!(bits(index.search(q, &sp, &mut b).unwrap()), want);

    // The early row wave starts, but its first read fails while the search waits for it.
    let index = open(|| Box::<Faulty<0, 2>>::default());
    let mut b = ctx();
    let mut fetched = index.fetch(&leaves, &mut b).unwrap();
    fetched.prefetch_rows(q, 40).unwrap();
    assert!(fetched.search(q, sp.k, sp.rerank).is_err());
    assert_eq!(bits(index.search(q, &sp, &mut b).unwrap()), want);

    // A wave that hands over too few ranges fails, whether it reads clusters or rows.
    let short = [
        open(|| Box::<EndsEarly<1>>::default()),
        open(|| Box::<EndsEarly<2>>::default()),
    ];
    for (i, index) in short.iter().enumerate() {
        assert!(index.search(q, &sp, &mut ctx()).is_err(), "wave {}", i + 1);
    }
}

#[test]
fn an_open_index_keeps_searching_its_own_file_after_a_rebuild() {
    let (memory, file, path) = written("rebuilt-open.coldivf", &params(Metric::L2), 400, 128, 12);
    let open = public(file);
    let q = &gaussian(&mut SplitMix64::new(8), 1, 128, |_| 1.0)[..];
    let sp = SearchParams {
        k: 10,
        nprobe: 4,
        rerank: 40,
        reach: None,
    };
    let want = bits(memory.search(q, &sp, &mut ctx()).unwrap());

    // A rebuild with other vectors renames a new file over the path.
    let (rows, mut clusters) = clustered(&mut SplitMix64::new(9), 300, 128, 8, 0.5);
    group(&mut clusters, 2);
    crate::Builder::new(params(Metric::L2))
        .build_with_clusters(&rows, &clusters, &path)
        .unwrap();

    // A context that has never read the index still reads the old file.
    assert_eq!(bits(open.search(q, &sp, &mut ctx()).unwrap()), want);
}

/// The file of an index with 3000 vectors in 64 clusters, after `damage` changes its bytes. The library's sorts only
/// check that an order is total for more than 20 items, so the tests need more clusters than that.
fn damaged(name: &str, damage: impl Fn(&mut [u8], &Layout)) -> (Index, TempFile) {
    let (_, file, path) = written(name, &params(Metric::L2), 3000, 128, 64);
    let (bytes, layout) = file.parts();
    let (mut bytes, layout) = (bytes.to_vec(), layout.clone());
    drop(file);
    damage(&mut bytes, &layout);
    std::fs::write(&path, &bytes).unwrap();
    (public(Inner::open(&path).unwrap()), path)
}

#[test]
fn a_damaged_row_fails_the_search_with_invalid_data() {
    // The n2 of internal id 0 becomes NaN. Rows sit after the 128 codes and the f32 scale.
    let (index, _path) = damaged("nan-row.coldivf", |bytes, layout| {
        let at = layout.span(Part::Rows).offset + 128 + 4;
        bytes[at..at + 4].copy_from_slice(&f32::NAN.to_le_bytes());
    });
    // Every cluster is read and every vector rescored, so the damaged row is always reached.
    let all = SearchParams {
        k: 50,
        nprobe: 64,
        rerank: 3000,
        reach: None,
    };
    let mut ctx = SearchCtx::new(Limits {
        max_k: 50,
        max_nprobe: 64,
        max_rerank: 3000,
    });

    let mapped = index.inner().search_mapped(&[0.5; 128], &all, &mut ctx).map(|_| ());
    assert_eq!(mapped.unwrap_err().kind(), ErrorKind::InvalidData);
    let cold = index.search(&[0.5; 128], &all, &mut ctx).map(|_| ());
    assert_eq!(cold.unwrap_err().kind(), ErrorKind::InvalidData);
}

#[test]
fn damaged_cluster_factors_never_panic() {
    // Cluster 5's minimum becomes −f32::MAX and its step f32::MAX, so its route score is NaN.
    let (index, _path) = damaged("nan-route.coldivf", |bytes, layout| {
        let arrays = layout.span(Part::Centers).offset
            + layout.nparents * sq8::center_row_bytes(128)
            + layout.nlist * crate::quantization::sq4::code_bytes(128);
        let lo = arrays + 4 * 5;
        let step = arrays + 4 * (layout.nlist + 5);
        bytes[lo..lo + 4].copy_from_slice(&(-f32::MAX).to_le_bytes());
        bytes[step..step + 4].copy_from_slice(&f32::MAX.to_le_bytes());
    });
    let mut ctx = ctx();
    let mut routes = Vec::new();

    let dim = 128;
    for q in gaussian(&mut SplitMix64::new(4), 30, dim, |_| 1.0).chunks_exact(dim) {
        let sp = SearchParams {
            k: 50,
            nprobe: 64,
            rerank: 200,
            reach: None,
        };
        let _ = index.inner().search_mapped(q, &sp, &mut ctx);
        let _ = index.search(q, &sp, &mut ctx);
        let _ = index.route(q, 64, None, &mut ctx, &mut routes);
    }
}

#[test]
fn verify_reads_each_section_in_pieces() {
    let (_, file, path) = written("pieces.coldivf", &params(Metric::L2), 3000, 128, 64);
    let (bytes, layout) = file.parts();
    let (bytes, layout) = (bytes.to_vec(), layout.clone());
    drop(file);

    // Pieces of 1000 bytes split every section, and a flipped byte near any section's end is still found.
    for size in [1000, 4096, PIECE] {
        Inner::open(&path).unwrap().verify_in(size).unwrap();
        for part in crate::format::CHECKED {
            let mut damaged = bytes.clone();
            damaged[layout.span(part).end() - 3] ^= 1;
            std::fs::write(&path, &damaged).unwrap();
            let err = Inner::open(&path).and_then(|f| f.verify_in(size)).unwrap_err();
            assert_eq!(err.kind(), ErrorKind::InvalidData, "{part:?} in pieces of {size}");
            std::fs::write(&path, &bytes).unwrap();
        }
    }
}

#[test]
fn a_warm_search_gives_exactly_the_cold_hits() {
    let mut rng = SplitMix64::new(20);
    let cases = [
        (Metric::L2, LeafOrder::Chained, 128, 4096),
        (Metric::Cosine, LeafOrder::Chained, 768, 4096),
        (Metric::L2, LeafOrder::Id, 200, 16384),
    ];
    let sp = SearchParams {
        k: 10,
        nprobe: 6,
        rerank: 40,
        reach: None,
    };

    for ((i, (metric, order, dim, page_size)), anchor) in
        cases.into_iter().enumerate().flat_map(|c| ANCHORS.map(|a| (c, a)))
    {
        let params = BuildParams {
            order,
            page_size,
            anchor,
            ..params(metric)
        };
        let (memory, _, path) = written(&format!("warm-{i}-{anchor:?}.coldivf"), &params, 400, dim, 12);
        let queries = gaussian(&mut rng, 5, dim, |_| 1.0);
        for (name, wave) in waves() {
            let index = public(Inner::open(&path).unwrap().with_wave(wave));
            let (mut a, mut b) = (ctx(), ctx());
            for q in queries.chunks_exact(dim) {
                let want = bits(memory.search(q, &sp, &mut a).unwrap());
                index.set_warm(false);
                assert_eq!(
                    bits(index.search(q, &sp, &mut b).unwrap()),
                    want,
                    "case {i} cold with the {anchor:?} anchor, {name}"
                );
                index.set_warm(true);
                assert_eq!(
                    bits(index.search(q, &sp, &mut b).unwrap()),
                    want,
                    "case {i} warm with the {anchor:?} anchor, {name}"
                );
                // A fetch from before stays harmless, even one that ended without waiting for its prefetched rows.
                let mut fetched = index.fetch(&[0, 1], &mut b).unwrap();
                fetched.prefetch_rows(q, 40).unwrap();
                fetched.search(q, 0, 0).unwrap();
                let got = bits(index.search(q, &sp, &mut b).unwrap());
                assert_eq!(
                    got, want,
                    "case {i} warm after a fetch with the {anchor:?} anchor, {name}"
                );
            }
        }
    }
}

thread_local! {
    /// What a `Recorded` backend was asked to do, in order.
    static CALLS: std::cell::RefCell<Vec<String>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// A backend for tests that reads like the portable one and records every call but its reads of ranges.
#[derive(Default)]
struct Recorded(PortableWave);

impl Recorded {
    fn note(call: String) {
        CALLS.with_borrow_mut(|c| c.push(call));
    }

    /// The calls so far, which it then forgets.
    fn calls() -> Vec<String> {
        CALLS.with_borrow_mut(std::mem::take)
    }
}

impl Wave for Recorded {
    fn page(&self) -> usize {
        4096
    }

    fn ready(&mut self, _file: &std::fs::File, arenas: &[Memory], _most: usize) {
        Recorded::note(format!("ready {}", arenas.len()));
    }

    fn prepare(&mut self) {
        Recorded::note("prepare".into());
    }

    fn submit(&mut self, lane: usize, file: &std::fs::File, ranges: &[Range], arena: Memory) -> Result<(), Error> {
        Recorded::note(format!("submit {lane}"));
        self.0.submit(lane, file, ranges, arena)
    }

    fn submit_cached(
        &mut self,
        lane: usize,
        file: &std::fs::File,
        ranges: &[Range],
        arena: Memory,
    ) -> Result<(), Error> {
        Recorded::note(format!("cached {lane}"));
        self.0.submit(lane, file, ranges, arena)
    }

    fn next(
        &mut self,
        lane: usize,
        file: &std::fs::File,
        ranges: &[Range],
        arena: Memory,
    ) -> Result<Option<usize>, Error> {
        self.0.next(lane, file, ranges, arena)
    }
}

#[test]
fn prepare_readies_the_backend_and_warm_searches_read_rows_through_the_cache() {
    let (_, file, _path) = written("recorded.coldivf", &params(Metric::L2), 400, 128, 12);
    let index = public(file.with_wave(|| Box::<Recorded>::default()));
    let q = gaussian(&mut SplitMix64::new(24), 1, 128, |_| 1.0);
    let sp = SearchParams {
        k: 10,
        nprobe: 6,
        rerank: 40,
        reach: None,
    };
    let mut c = ctx();
    Recorded::calls();

    // Before the tables load, prepare readies the backend with the two row arenas. It asks for no setup that waits,
    // since no search has gone through the reads yet.
    index.prepare(&mut c).unwrap();
    assert_eq!(Recorded::calls(), ["ready 2"]);
    // The first search loads the tables, names the cluster arena too, and reads the clusters and the rows directly.
    index.search(&q, &sp, &mut c).unwrap();
    assert_eq!(Recorded::calls(), ["ready 3", "submit 0", "submit 0"]);
    // After a search, prepare asks for the setup that waits, eg. registering the arenas.
    index.prepare(&mut c).unwrap();
    assert_eq!(Recorded::calls(), ["prepare"]);
    // A warm search scans in place and reads its rows through the cache.
    index.set_warm(true);
    index.search(&q, &sp, &mut c).unwrap();
    assert_eq!(Recorded::calls(), ["cached 0"]);
    index.set_warm(false);
    index.search(&q, &sp, &mut c).unwrap();
    assert_eq!(Recorded::calls(), ["submit 0", "submit 0"]);

    // A context that prepare readies once the tables are loaded gets all of its arenas at once, and still no setup
    // that waits before its first search.
    index.prepare(&mut ctx()).unwrap();
    assert_eq!(Recorded::calls(), ["ready 2", "ready 3"]);
}

#[test]
fn a_warm_search_scans_in_place_and_checks_its_arguments() {
    let (_, file, _path) = written("warm-path.coldivf", &params(Metric::L2), 400, 128, 12);
    // This backend fails its second wave. A cold search reads the clusters and then the rows, and a warm one only
    // the rows.
    let index = public(file.with_wave(|| Box::<Faulty<2, 0>>::default()));
    let q = gaussian(&mut SplitMix64::new(23), 1, 128, |_| 1.0);
    let sp = SearchParams {
        k: 10,
        nprobe: 6,
        rerank: 40,
        reach: None,
    };
    assert!(index.search(&q, &sp, &mut ctx()).is_err());
    index.set_warm(true);
    assert!(index.search(&q, &sp, &mut ctx()).is_ok());

    let bad = [
        (&q[..127], sp),
        (&q[..], SearchParams { nprobe: 65, ..sp }),
        (
            &q[..],
            SearchParams {
                reach: Some(f32::NAN),
                ..sp
            },
        ),
    ];
    let mut c = ctx();
    for (query, sp) in bad {
        let err = index.search(query, &sp, &mut c).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidInput, "{sp:?}");
    }
}

/// Runs warm-ups of `index` until one leaves it warm. A memory-pressure warning may clear the bit in between.
fn warm(index: &Index) {
    let start = std::time::Instant::now();

    while !index.is_warm() {
        assert!(start.elapsed().as_secs() < 30, "no warm-up set the bit");
        index.warm_up().unwrap();
        while index.inner().warming.load(Ordering::Acquire) > 0 {
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
    }
}

#[test]
fn a_warm_up_sets_the_warm_bit_and_set_warm_clears_it() {
    let (_, file, _path) = written("warm-up.coldivf", &params(Metric::L2), 3000, 128, 64);
    let index = public(file);
    assert!(!index.is_warm());

    for _ in 0..2 {
        index.warm_up().unwrap();
        index.warm_up().unwrap();
        warm(&index);
        index.set_warm(false);
        assert!(!index.is_warm());
    }

    // The warm-up started the watch that clears the bit when memory runs short.
    assert!(index.inner().pressure.get().is_some());
}

#[test]
fn setting_the_warm_bit_starts_the_pressure_watch() {
    let (_, file, _path) = written("warm-watch.coldivf", &params(Metric::L2), 400, 128, 12);
    let index = public(file);

    index.set_warm(false);
    assert!(index.inner().pressure.get().is_none());
    index.set_warm(true);
    assert!(index.inner().pressure.get().is_some());
}

#[test]
fn a_warm_up_stops_at_a_clear_or_once_the_index_closes() {
    let (_, file, _path) = written("warm-stop.coldivf", &params(Metric::L2), 3000, 128, 64);
    let index = Arc::new(file);
    let codes = index.layout.span(Part::Codes).len;

    // A clear since the mark stops it before it reads anything, and the bit stays clear.
    let mark = index.warmth.mark();
    index.set_warm(false);
    assert_eq!(warm_once(&Arc::downgrade(&index), mark).unwrap(), 0);
    assert!(!index.is_warm());

    // Without one, it reads the whole codes section and sets the bit, and on a warm index it reads nothing.
    let mark = index.warmth.mark();
    assert_eq!(warm_once(&Arc::downgrade(&index), mark).unwrap(), codes);
    assert!(index.is_warm());
    assert_eq!(warm_once(&Arc::downgrade(&index), index.warmth.mark()).unwrap(), 0);

    // Once nothing holds the index, it reads nothing either.
    index.set_warm(false);
    let (weak, mark) = (Arc::downgrade(&index), index.warmth.mark());
    drop(index);
    assert_eq!(warm_once(&weak, mark).unwrap(), 0);

    // Dropping an index while its warm-up thread runs is safe.
    let (_, file, _path) = written("warm-closed.coldivf", &params(Metric::L2), 400, 128, 12);
    let index = public(file);
    index.warm_up().unwrap();
    drop(index);
}

#[test]
fn a_warm_up_warms_the_open_file_after_a_rebuild_renames_another_over_it() {
    let (memory, file, path) = written("warm-replaced.coldivf", &params(Metric::L2), 400, 128, 12);
    let index = public(file);
    let (rows, mut clusters) = clustered(&mut SplitMix64::new(21), 300, 128, 8, 0.5);
    group(&mut clusters, 2);
    crate::Builder::new(params(Metric::L2))
        .build_with_clusters(&rows, &clusters, &path)
        .unwrap();

    warm(&index);
    let q = gaussian(&mut SplitMix64::new(24), 1, 128, |_| 1.0);
    let sp = SearchParams {
        k: 10,
        nprobe: 6,
        rerank: 40,
        reach: None,
    };
    let want = bits(memory.search(&q, &sp, &mut ctx()).unwrap());
    assert_eq!(bits(index.search(&q, &sp, &mut ctx()).unwrap()), want);
}

#[test]
fn every_search_path_reads_the_same_clusters_within_reach() {
    let (memory, file, _path) = written("reach.coldivf", &params(Metric::L2), 3000, 128, 64);
    let index = public(file);
    let (mut a, mut b, dim) = (ctx(), ctx(), 128);
    let mut leaves = Vec::new();

    for q in gaussian(&mut SplitMix64::new(22), 10, dim, |_| 1.0).chunks_exact(dim) {
        for reach in [Some(0.0), Some(0.15), Some(0.6)] {
            let sp = SearchParams {
                k: 10,
                nprobe: 24,
                rerank: 40,
                reach,
            };
            let want = bits(memory.search(q, &sp, &mut a).unwrap());
            assert_eq!(
                bits(index.inner().search_mapped(q, &sp, &mut b).unwrap()),
                want,
                "mapped, {reach:?}"
            );
            index.set_warm(false);
            assert_eq!(bits(index.search(q, &sp, &mut b).unwrap()), want, "cold, {reach:?}");
            index.set_warm(true);
            assert_eq!(bits(index.search(q, &sp, &mut b).unwrap()), want, "warm, {reach:?}");
            // Routing with the same reach, then fetching and ranking those clusters, is the same search.
            index.route(q, sp.nprobe, reach, &mut b, &mut leaves).unwrap();
            assert!(reach != Some(0.0) || leaves.len() < sp.nprobe);
            let fetched = index.fetch(&leaves, &mut b).unwrap();
            assert_eq!(
                bits(fetched.search(q, sp.k, sp.rerank).unwrap()),
                want,
                "fetched, {reach:?}"
            );
        }
    }
}

/// The trace of the search that `search` runs on `c`, after checking that it gave `want` and that its stages fit in the
/// time it took.
#[cfg(feature = "bench")]
fn traced(
    c: &mut SearchCtx,
    want: &[(u64, u32)],
    search: impl FnOnce(&mut SearchCtx) -> Vec<(u64, u32)>,
) -> crate::Trace {
    let start = std::time::Instant::now();
    let got = search(c);
    let took = start.elapsed();
    assert_eq!(got, want);
    let t = c.trace();
    let stages = t.tables + t.buffers + t.route + t.submit + t.wave + t.scan + t.rows + t.rerank;
    assert!(stages <= took, "{t:?} in {took:?}");
    t
}

#[cfg(feature = "bench")]
#[test]
fn every_search_traces_its_stages_and_its_reads() {
    use crate::Backend;
    use std::time::Duration;

    let (memory, _, path) = written("trace.coldivf", &params(Metric::L2), 3000, 128, 64);
    let q = gaussian(&mut SplitMix64::new(25), 1, 128, |_| 1.0);
    let sp = SearchParams {
        k: 10,
        nprobe: 8,
        rerank: 40,
        reach: None,
    };
    let want = bits(memory.search(&q, &sp, &mut ctx()).unwrap());
    let probed = |index: &Index| {
        let mut leaves = Vec::new();
        index.route(&q, sp.nprobe, None, &mut ctx(), &mut leaves).unwrap();
        let inner = index.inner();
        let lists = &inner.tables().unwrap().lists;
        leaves.iter().map(|&l| lists[l as usize].count).sum::<usize>()
    };
    let zero = Duration::ZERO;
    // On macOS the backend that opening picks rounds its reads out to 16 KiB, and the other two to 4 KiB.
    let best = Index::open(&path).unwrap().inner().read_page();

    for (backend, page) in [(Backend::Best, best), (Backend::Pool, 4096), (Backend::Portable, 4096)] {
        let index = Index::open_with(&path, backend).unwrap();
        assert_eq!(index.inner().read_page(), page, "{backend:?}");
        let mut c = ctx();

        // The first search of an index loads its tables, then reads its clusters and its rows in two waves of whole
        // pages. `prepare` makes the scratch and the row buffers first, as in a cold trial. The buffer for the clusters
        // waits for the tables, so the search makes it, and counts the time in `buffers`, not in `submit`.
        index.prepare(&mut c).unwrap();
        let t = traced(&mut c, &want, |c| bits(index.search(&q, &sp, c).unwrap()));
        let stages = [t.tables, t.route, t.submit, t.scan, t.rows, t.rerank];
        assert!(stages.iter().all(|&d| d > zero), "{backend:?} {t:?}");
        assert!(t.buffers >= Duration::from_micros(1), "{backend:?} {t:?}");
        assert!(t.cluster_ranges > 0 && t.cluster_ranges <= 8 && t.cluster_bytes >= t.cluster_ranges * page);
        assert!(t.row_ranges > 0 && t.row_bytes >= t.row_ranges * page);
        assert!(
            t.cluster_bytes.is_multiple_of(page) && t.row_bytes.is_multiple_of(page),
            "{backend:?} {t:?}"
        );
        assert_eq!(t.scanned, probed(&index));

        // The mapped control starts a new trace on the same context. It reads through page faults, so it counts its
        // route in the scan and its rows in the rerank, and it reads nothing in waves.
        let t = traced(&mut c, &want, |c| bits(index.search_mapped(&q, &sp, c).unwrap()));
        assert!(t.scan > zero && t.rerank > zero, "{backend:?} mapped {t:?}");
        assert_eq!((t.route, t.submit, t.wave, t.rows), (zero, zero, zero, zero));
        assert_eq!(
            (t.cluster_bytes, t.cluster_ranges, t.row_bytes, t.row_ranges),
            (0, 0, 0, 0)
        );
        assert_eq!(t.scanned, probed(&index));

        // A later cold search makes no buffers, so its submit only plans and starts the reads. The pool waits for every
        // read inside its submit, and the other backends wait as the scan asks for each range.
        let t = traced(&mut c, &want, |c| bits(index.search(&q, &sp, c).unwrap()));
        if page == 4096 && backend != Backend::Portable {
            assert!(t.wave < t.submit, "{backend:?} {t:?}");
        } else {
            assert!(t.wave > zero, "{backend:?} {t:?}");
        }

        // A warm search right after a cold one on the same context reads no clusters, and its trace says so.
        index.set_warm(true);
        let t = traced(&mut c, &want, |c| bits(index.search(&q, &sp, c).unwrap()));
        index.set_warm(false);
        assert_eq!(
            (t.route, t.submit, t.wave),
            (zero, zero, zero),
            "{backend:?} warm {t:?}"
        );
        assert_eq!((t.cluster_bytes, t.cluster_ranges), (0, 0));
        assert!(t.scan > zero && t.rows > zero && t.rerank > zero && t.row_ranges > 0);
        assert_eq!(t.scanned, probed(&index));

        // A route alone starts a new trace as well, with only the tables, the buffers and the route in it.
        index.route(&q, sp.nprobe, None, &mut c, &mut Vec::new()).unwrap();
        let t = c.trace();
        assert!(t.route > zero, "{backend:?} route {t:?}");
        assert_eq!(
            (t.scan, t.rows, t.rerank, t.scanned, t.row_bytes),
            (zero, zero, zero, 0, 0)
        );

        // On a new context, a mapped search, a warm one and a route each make the buffers they need, and count the time
        // in `buffers`.
        let mapped = traced(&mut ctx(), &want, |c| bits(index.search_mapped(&q, &sp, c).unwrap()));
        index.set_warm(true);
        let warm = traced(&mut ctx(), &want, |c| bits(index.search(&q, &sp, c).unwrap()));
        index.set_warm(false);
        let (mut fresh, mut leaves) = (ctx(), Vec::new());
        index.route(&q, sp.nprobe, None, &mut fresh, &mut leaves).unwrap();
        let route = fresh.trace();
        assert!(
            mapped.buffers > zero && warm.buffers > zero && route.buffers > zero,
            "{backend:?} {mapped:?} {warm:?} {route:?}"
        );

        // So do a fetch, which makes the read buffers, and then the search of what it fetched, which makes the scratch.
        let mut other = ctx();
        let fetched = index.fetch(&leaves, &mut other).unwrap();
        let reads = fetched.ctx.trace().buffers;
        assert_eq!(bits(fetched.search(&q, sp.k, sp.rerank).unwrap()), want);
        let t = other.trace();
        assert!(reads > zero && t.buffers > reads, "{backend:?} fetched {t:?}");
    }
}

#[test]
fn a_mapped_search_reads_nothing_through_the_cold_handle() {
    let (memory, mut file, _path) = written("mapped-only.coldivf", &params(Metric::L2), 400, 128, 12);
    // Every read through a handle to an empty file fails, so the search must read the tables through the map too.
    let empty = temp_file("mapped-only-empty.bin");
    std::fs::write(&empty, b"").unwrap();
    file.cold.file = File::open(&empty).unwrap();
    let q = gaussian(&mut SplitMix64::new(28), 1, 128, |_| 1.0);
    let sp = SearchParams {
        k: 10,
        nprobe: 4,
        rerank: 40,
        reach: None,
    };

    assert_eq!(
        bits(file.search_mapped(&q, &sp, &mut ctx()).unwrap()),
        bits(memory.search(&q, &sp, &mut ctx()).unwrap())
    );
}

#[test]
fn opening_reads_one_file_even_if_a_rebuild_renames_another_over_it_midway() {
    let dim = 128;
    let (rows, mut clusters) = clustered(&mut SplitMix64::new(26), 400, dim, 12, 0.5);
    group(&mut clusters, 3);
    let memory = InMemory::build(&params(Metric::L2), &rows, &clusters).unwrap();
    let path = temp_file("renamed-midway.coldivf");
    crate::build::write(&memory, &path).unwrap();
    let opening = Opening::start(&path).unwrap();

    // The rebuild has other vectors in clusters of the same sizes, so the new file is just as long.
    let negated = Rows {
        dim,
        data: rows.data.iter().map(|x| -x).collect(),
    };
    let rebuilt = InMemory::build(&params(Metric::L2), &negated, &clusters).unwrap();
    let len = std::fs::metadata(&path).unwrap().len();
    crate::build::write(&rebuilt, &path).unwrap();
    assert_eq!(std::fs::metadata(&path).unwrap().len(), len);

    // The header, the map and every wave come from the file that opening started on.
    let index = public(opening.finish(&path).unwrap());
    let sp = SearchParams {
        k: 10,
        nprobe: 4,
        rerank: 40,
        reach: None,
    };

    for q in gaussian(&mut SplitMix64::new(27), 5, dim, |_| 1.0).chunks_exact(dim) {
        let want = bits(memory.search(q, &sp, &mut ctx()).unwrap());
        assert_ne!(bits(rebuilt.search(q, &sp, &mut ctx()).unwrap()), want);
        assert_eq!(
            bits(index.inner().search_mapped(q, &sp, &mut ctx()).unwrap()),
            want,
            "mapped"
        );
        assert_eq!(bits(index.search(q, &sp, &mut ctx()).unwrap()), want, "cold");
    }
}

#[cfg(unix)]
#[test]
fn a_path_that_isn_t_a_regular_file_fails_with_invalid_data() {
    let kind = |path: &Path| Inner::open(path).map(|_| ()).unwrap_err().kind();

    assert_eq!(kind(&std::env::temp_dir()), ErrorKind::InvalidData);
    assert_eq!(kind(Path::new("/dev/null")), ErrorKind::InvalidData);
    assert_eq!(kind(Path::new("/dev/zero")), ErrorKind::InvalidData);
}

#[test]
fn prepare_makes_the_context_s_buffers_for_the_index() {
    let (_, file, _path) = written("prepare-buffers.coldivf", &params(Metric::L2), 3000, 128, 64);
    let index = public(file);
    let mut c = ctx();

    // Before the tables load it makes the scratch and the reads, but not the cluster arena, which is sized from them.
    index.prepare(&mut c).unwrap();
    assert!(index.inner().tables.get().is_none());
    assert_eq!((c.scratch.len(), c.reads.len()), (1, 1));
    assert!(c.scratch[0].fits(128) && !c.reads[0].sized());

    // Once a route has loaded them, it makes the cluster arena too, and a search then needs no more.
    index.route(&[0.5; 128], 4, None, &mut ctx(), &mut Vec::new()).unwrap();
    index.prepare(&mut c).unwrap();
    assert_eq!((c.scratch.len(), c.reads.len()), (1, 1));
    assert!(c.reads[0].sized());
    let sp = SearchParams {
        k: 10,
        nprobe: 16,
        rerank: 40,
        reach: None,
    };
    index.search(&[0.5; 128], &sp, &mut c).unwrap();
    index.prepare(&mut c).unwrap();
    assert_eq!((c.scratch.len(), c.reads.len()), (1, 1));
}

/// An index of 3000 vectors in clusters of 50, made by a build and a first maintenance run, so it has a root, and its
/// vectors.
fn rooted(name: &str) -> (TempFile, crate::quantization::testutil::Rows) {
    use crate::maintain::tests::{build_params, sample, settings};

    let (path, rows) = sample(name, &build_params(Metric::L2, Anchor::Cluster), 3000, 30, 70);
    crate::Writer::open(&path)
        .unwrap()
        .maintain(&settings(), crate::RunKind::Fold)
        .unwrap();
    (path, rows)
}

#[test]
fn every_open_handle_of_an_index_moves_with_its_writer_s_commits() {
    let (path, rows) = rooted("follows.coldivf");
    let (a, b) = (Index::open(&path).unwrap(), Index::open(&path).unwrap());
    let set = crate::Collections::open(&[&*path]).unwrap();
    let mut writer = crate::Writer::open(&path).unwrap();
    let epoch = a.inner().layout().epoch;
    let v: Vec<f32> = rows.data[..128].iter().map(|&x| x + 0.2).collect();
    writer.insert(90_000, &v).unwrap();
    writer.compact().unwrap();

    // Every handle of the path shares one cell, which the commit moved, so each sees the fold without opening again.
    for index in [&a, &b, &set.indexes[0], &writer.index] {
        assert!(Arc::ptr_eq(&index.shared, &a.shared));
        assert_eq!(index.inner().layout().epoch, epoch + 1);
        assert_eq!(index.inner().changes().read().len(), 0);
    }

    let sp = SearchParams {
        k: 1,
        nprobe: 60,
        rerank: 200,
        reach: None,
    };
    assert_eq!(a.search(&v, &sp, &mut ctx()).unwrap()[0].id, 90_000);
    let mut out = Vec::new();
    set.search(&v, &sp, &mut ctx(), &mut out).unwrap();
    assert_eq!(out[0].1.id, 90_000);
}

#[test]
fn a_refresh_moves_to_a_generation_that_another_writer_committed() {
    let (path, rows) = rooted("refreshed.coldivf");
    let index = Index::open(&path).unwrap();
    let before = index.inner().located().generation;
    assert!(!index.refresh().unwrap());

    // A build at the path commits through the root alone, as another process's writer would, so the cell stays.
    let other = crate::quantization::testutil::Rows {
        dim: 128,
        data: rows.data.iter().map(|&x| x + 5.0).collect(),
    };
    crate::Builder::new(crate::maintain::tests::build_params(Metric::L2, Anchor::Cluster))
        .build(&other, &path)
        .unwrap();
    assert_eq!(index.inner().located().generation, before);
    let sp = SearchParams {
        k: 1,
        nprobe: 60,
        rerank: 200,
        reach: None,
    };
    // The new vectors lie far from the old ones, so a query on one of them finds it only in the new generation.
    let q = &other.data[7 * 128..8 * 128];
    assert!(index.search(q, &sp, &mut ctx()).unwrap()[0].score > 100.0);
    assert!(index.refresh().unwrap());
    assert_eq!(index.inner().located().generation, before + 1);
    let hit = index.search(q, &sp, &mut ctx()).unwrap()[0];
    assert!(hit.id == 7 && hit.score < 0.1, "{hit:?}");
    assert!(!index.refresh().unwrap());

    // `prepare` moves too, once the root's files change, and a context's reads follow.
    crate::Builder::new(crate::maintain::tests::build_params(Metric::L2, Anchor::Cluster))
        .build(&rows, &path)
        .unwrap();
    let mut c = ctx();
    index.prepare(&mut c).unwrap();
    assert_eq!(index.inner().located().generation, before + 2);
    assert_eq!(index.search(&rows.data[..128], &sp, &mut c).unwrap()[0].id, 0);
}

#[test]
fn a_refresh_moves_to_a_file_rebuilt_before_the_first_commit() {
    let (_, file, path) = written("refresh-legacy.coldivf", &params(Metric::L2), 600, 128, 6);
    drop(file);
    let index = Index::open(&path).unwrap();
    let check = index.inner().layout().check;
    assert!(!index.refresh().unwrap());
    let (rows, mut clusters) = clustered(&mut SplitMix64::new(71), 600, 128, 6, 0.5);
    group(&mut clusters, 3);
    crate::build::write(&InMemory::build(&params(Metric::L2), &rows, &clusters).unwrap(), &path).unwrap();
    assert!(index.refresh().unwrap());
    assert_ne!(index.inner().layout().check, check);
}

#[test]
fn a_writer_frees_an_old_generation_once_no_search_or_fetch_holds_it() {
    let (path, rows) = rooted("reclaim.coldivf");
    let index = Index::open(&path).unwrap();
    let mut writer = crate::Writer::open(&path).unwrap();
    let mut c = ctx();
    let old = Arc::downgrade(&index.shared.load_full());
    let mut leaves = Vec::new();
    index.route(&rows.data[..128], 8, None, &mut c, &mut leaves).unwrap();
    let fetched = index.fetch(&leaves, &mut c).unwrap();
    writer.compact().unwrap();

    // The fetch holds the old generation, so the cell keeps it, and the fetch still searches it, without an insert
    // that goes to the new generation.
    let v: Vec<f32> = rows.data[..128].iter().map(|&x| x + 0.2).collect();
    writer.insert(90_000, &v).unwrap();
    assert_eq!(index.shared.retired(), 1);
    assert!(fetched.search(&v, 10, 40).unwrap().iter().all(|h| h.id != 90_000));
    assert_eq!(index.search(&v, &everything(), &mut ctx()).unwrap()[0].id, 90_000);
    // Once nothing holds it, the writer's next change frees it.
    assert!(old.upgrade().is_some());
    writer.delete(5).unwrap();
    assert_eq!(index.shared.retired(), 0);
    assert!(old.upgrade().is_none());
}

/// Every cluster read and every candidate rescored, within `ctx`'s limits.
fn everything() -> SearchParams {
    SearchParams {
        k: 10,
        nprobe: 64,
        rerank: 200,
        reach: None,
    }
}
