//! Unit tests for opening and searching index files.

use std::io::ErrorKind;

use super::*;
use crate::build::write;
use crate::format::Layout;
use crate::kernels::Tier;
use crate::quantization::SplitMix64;
use crate::quantization::testutil::{TempFile, clustered, gaussian, group, params, temp_file};
use crate::search::inmem::InMemory;
use crate::{BuildParams, LeafOrder, Limits, Metric};

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

/// An in-memory index, written to a file and opened again.
fn written(name: &str, params: &BuildParams, n: usize, dim: usize, nlist: usize) -> (InMemory, Inner, TempFile) {
    let mut rng = SplitMix64::new(1);
    let (rows, mut clusters) = clustered(&mut rng, n, dim, nlist, 0.5);
    group(&mut clusters, 3);
    let index = InMemory::build(params, &rows, &clusters).unwrap();

    let path = temp_file(name);
    write(&index, &path).unwrap();
    let file = Inner::open(&path).unwrap();
    (index, file, path)
}

#[test]
fn a_file_search_gives_exactly_the_in_memory_hits() {
    let mut rng = SplitMix64::new(2);
    let cases = [
        (Metric::L2, LeafOrder::Id, 128, 4096),
        (Metric::L2, LeafOrder::Chained, 200, 4096),
        (Metric::Cosine, LeafOrder::Chained, 768, 4096),
        (Metric::L2, LeafOrder::Chained, 768, 16384),
        (Metric::L2, LeafOrder::Chained, 3000, 4096),
    ];

    for (i, (metric, order, dim, page_size)) in cases.into_iter().enumerate() {
        let queries = gaussian(&mut rng, 4, dim, |_| 1.0);
        for tier in [Tier::SCALAR, Tier::best()] {
            let params = BuildParams {
                order,
                page_size,
                ..params(metric)
            };
            let (index, file, _path) = written(&format!("exact-{i}.coldivf"), &params, 400, dim, 12);
            let (index, file) = (index.with_tier(tier), file.with_tier(tier));
            let (mut a, mut b) = (ctx(), ctx());
            let (mut routes_a, mut routes_b) = (Vec::new(), Vec::new());
            for q in queries.chunks_exact(dim) {
                for sp in [
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
                ] {
                    let want = bits(index.search(q, &sp, &mut a).unwrap());
                    assert_eq!(bits(file.search(q, &sp, &mut b).unwrap()), want, "case {i} on {tier:?}");
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
    let (index, file, path) = written("prepare.coldivf", &params(Metric::L2), 100, 128, 12);
    file.prepare().unwrap();
    let q = vec![0.5f32; 128];
    let sp = SearchParams {
        k: 5,
        nprobe: 4,
        rerank: 20,
    };
    let (mut a, mut b) = (ctx(), ctx());

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
    let file = Inner::open(&path).unwrap();
    let sp = SearchParams {
        k: 5,
        nprobe: 4,
        rerank: 20,
    };
    let err = file.search(&[0.5; 128], &sp, &mut ctx()).map(|_| ()).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidData);
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
    shareable(&file);
}

/// The file of an index with 3000 vectors in 64 clusters, after `damage` changes its bytes. The library's sorts only
/// check that an order is total for more than 20 items, so the tests need more clusters than that.
fn damaged(name: &str, damage: impl Fn(&mut [u8], &Layout)) -> (Inner, TempFile) {
    let (_, file, path) = written(name, &params(Metric::L2), 3000, 128, 64);
    let (bytes, layout) = file.parts();
    let (mut bytes, layout) = (bytes.to_vec(), layout.clone());
    drop(file);
    damage(&mut bytes, &layout);
    std::fs::write(&path, &bytes).unwrap();
    (Inner::open(&path).unwrap(), path)
}

#[test]
fn a_damaged_row_fails_the_search_with_invalid_data() {
    // The n2 of internal id 0 becomes NaN. Rows sit after the 128 codes and the f32 scale.
    let (file, _path) = damaged("nan-row.coldivf", |bytes, layout| {
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

    let err = file.search(&[0.5; 128], &all, &mut ctx).map(|_| ()).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidData);
}

#[test]
fn damaged_cluster_factors_never_panic() {
    // Cluster 5's minimum becomes −f32::MAX and its step f32::MAX, so its route score is NaN.
    let (file, _path) = damaged("nan-route.coldivf", |bytes, layout| {
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
        let _ = file.search(q, &sp, &mut ctx);
        let _ = file.route(q, 64, &mut ctx, &mut routes);
    }
}
