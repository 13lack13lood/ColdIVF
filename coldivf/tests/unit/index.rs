//! Unit tests for opening and searching index files.

use std::io::ErrorKind;
use std::sync::Arc;

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
        },
        SearchParams {
            k: 50,
            nprobe: 12,
            rerank: 200,
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
                index.route(q, 5, &mut a, &mut routes_a).unwrap();
                file.route(q, 5, &mut b, &mut routes_b).unwrap();
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
    };
    let (mut a, mut b) = (ctx(), ctx());
    let mut rng = SplitMix64::new(5);
    let mut leaves = Vec::new();

    for q in &queries {
        let want = bits(index.search(q, &sp, &mut a).unwrap());
        // The leaves may come in any order, and repeats are read once.
        index.route(q, sp.nprobe, &mut b, &mut leaves).unwrap();
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
        index.route(q, sp.nprobe, &mut b, &mut leaves).unwrap();
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
    };
    let (mut a, mut b) = (ctx(), ctx());
    let mut leaves = Vec::new();

    for q in queries.iter().take(5) {
        let want = bits(index.search(q, &sp, &mut a).unwrap());
        index.route(q, sp.nprobe, &mut b, &mut leaves).unwrap();
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
    };
    let want = bits(memory.search(q, &sp, &mut ctx()).unwrap());
    drop(file);
    let open = |wave: NewWave| public(Inner::open(&path).unwrap().with_wave(wave));
    let mut leaves = Vec::new();

    // The first read of the clusters fails. Every later use of that fetch fails too, and a new search starts over.
    let index = open(|| Box::<Faulty<0, 1>>::default());
    let mut b = ctx();
    index.route(q, sp.nprobe, &mut b, &mut leaves).unwrap();
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
        };
        let _ = index.inner.search_mapped(q, &sp, &mut ctx);
        let _ = index.search(q, &sp, &mut ctx);
        let _ = index.route(q, 64, &mut ctx, &mut routes);
    }
}
