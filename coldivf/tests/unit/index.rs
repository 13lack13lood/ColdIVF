//! Unit tests for opening and searching index files.

use std::io::ErrorKind;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use super::*;
use crate::format::Layout;
use crate::io::NewWave;
use crate::io::pool::PoolWave;
use crate::io::portable::PortableWave;
use crate::kernels::Tier;
use crate::quantization::SplitMix64;
use crate::quantization::testutil::{
    EndsEarly, Faulty, Shuffled, TempFile, clustered, gaussian, group, params, shuffle, temp_file, written,
};
use crate::search::inmem::InMemory;
use crate::{BuildParams, Index, LeafOrder, Limits, Metric};

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
    Index { inner: Arc::new(inner) }
}

/// Every read backend, by name.
fn waves() -> Vec<(&'static str, NewWave)> {
    vec![
        ("portable", || Box::<PortableWave>::default()),
        ("pool", || Box::<PoolWave>::default()),
        ("shuffled", || Box::<Shuffled>::default()),
        #[cfg(target_os = "macos")]
        ("rdadvise", || Box::<crate::io::macos::RdAdviseWave>::default()),
    ]
}

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

    for (i, (metric, order, dim, page_size)) in cases.into_iter().enumerate() {
        let queries = gaussian(&mut rng, 4, dim, |_| 1.0);
        for tier in [Tier::SCALAR, Tier::best()] {
            let params = BuildParams {
                order,
                page_size,
                ..params(metric)
            };
            let (index, file, path) = written(&format!("exact-{i}.coldivf"), &params, 400, dim, 12);
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
                        "case {i} on {tier:?}"
                    );
                    for (name, cold) in &colds {
                        let got = bits(cold.search(q, sp, &mut c).unwrap());
                        assert_eq!(got, want, "case {i} on {tier:?} with the {name} reader");
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
    let mapped = file.inner.search_mapped(&[0.5; 128], &sp, &mut ctx()).map(|_| ());
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
    let (memory, file, path) = written("rebuilt.coldivf", &params(Metric::L2), 400, 128, 12);
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

    let mapped = index.inner.search_mapped(&[0.5; 128], &all, &mut ctx).map(|_| ());
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
        let _ = index.inner.search_mapped(q, &sp, &mut ctx);
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

    for (i, (metric, order, dim, page_size)) in cases.into_iter().enumerate() {
        let params = BuildParams {
            order,
            page_size,
            ..params(metric)
        };
        let (memory, _, path) = written(&format!("warm-{i}.coldivf"), &params, 400, dim, 12);
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
                    "case {i} cold, {name}"
                );
                index.set_warm(true);
                assert_eq!(
                    bits(index.search(q, &sp, &mut b).unwrap()),
                    want,
                    "case {i} warm, {name}"
                );
                // A fetch from before stays harmless, even one that ended without waiting for its prefetched rows.
                let mut fetched = index.fetch(&[0, 1], &mut b).unwrap();
                fetched.prefetch_rows(q, 40).unwrap();
                fetched.search(q, 0, 0).unwrap();
                let got = bits(index.search(q, &sp, &mut b).unwrap());
                assert_eq!(got, want, "case {i} warm after a fetch, {name}");
            }
        }
    }
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
        while index.inner.warming.load(Ordering::Acquire) > 0 {
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
    assert!(index.inner.pressure.get().is_some());
}

#[test]
fn setting_the_warm_bit_starts_the_pressure_watch() {
    let (_, file, _path) = written("warm-watch.coldivf", &params(Metric::L2), 400, 128, 12);
    let index = public(file);

    index.set_warm(false);
    assert!(index.inner.pressure.get().is_none());
    index.set_warm(true);
    assert!(index.inner.pressure.get().is_some());
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
                bits(index.inner.search_mapped(q, &sp, &mut b).unwrap()),
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
