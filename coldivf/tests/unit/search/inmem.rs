//! Unit tests for the in-memory index.

use std::io::ErrorKind;

use super::*;
use crate::build::order::leaf_order;
use crate::quantization::SplitMix64;
use crate::quantization::padded_dim;
use crate::quantization::rabitq::{BLOCK, block_bytes, lane_factors};
use crate::quantization::testutil::{Rows, clustered, dist2, gaussian, group, mean, params};
use crate::search::candidates::{Candidate, order};
use crate::search::scan::scan_leaf;
use crate::search::{List, Scratch, normalize, prepare_query};
use crate::{Anchor, LeafOrder, Limits, SearchCtx, SearchParams};

fn ctx(max: usize) -> SearchCtx {
    SearchCtx::new(Limits {
        max_k: max,
        max_nprobe: max,
        max_rerank: max,
    })
}

/// The exact 8-bit ranking: every vector rescored with its row, by squared distance and then internal id.
fn exact_8_bit(index: &InMemory, query: &[f32], k: usize) -> Vec<Hit> {
    let mut q = query.to_vec();
    if index.metric == Metric::Cosine {
        assert!(normalize(&mut q));
    }
    let offset: Vec<f32> = q.iter().zip(&index.mean).map(|(&x, &m)| x - m).collect();
    let norm2 = offset.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>() as f32;

    let data = index.in_place();
    let mut all: Vec<Candidate> = (0..index.ids.len() as u32)
        .map(|id| Candidate {
            score: sq8::distance(data.rows.row(id), &offset, norm2),
            id,
        })
        .collect();
    all.sort_unstable_by(order);

    all.iter()
        .take(k)
        .map(|c| Hit {
            id: sq8::id(data.rows.row(c.id), index.dim),
            score: if index.metric == Metric::Cosine {
                1.0 - c.score / 2.0
            } else {
                c.score
            },
        })
        .collect()
}

fn bits(hits: &[Hit]) -> Vec<(u64, u32)> {
    hits.iter().map(|h| (h.id, h.score.to_bits())).collect()
}

#[test]
fn reading_every_cluster_and_rescoring_every_vector_is_exact() {
    let mut rng = SplitMix64::new(1);
    for (dim, metric) in [(200, Metric::L2), (768, Metric::L2), (200, Metric::Cosine)] {
        let (rows, clusters) = clustered(&mut rng, 300, dim, 8, 0.5);
        let index = InMemory::build(&params(metric), &rows, &clusters).unwrap();
        let mut ctx = ctx(300);
        let all = SearchParams {
            k: 10,
            nprobe: 8,
            rerank: 300,
            reach: None,
        };
        for q in gaussian(&mut rng, 5, dim, |_| 1.0).chunks_exact(dim) {
            let hits = index.search(q, &all, &mut ctx).unwrap();
            assert_eq!(bits(hits), bits(&exact_8_bit(&index, q, 10)), "dim {dim}, {metric:?}");
        }
    }
}

#[test]
fn every_tier_gives_the_same_hits() {
    let mut rng = SplitMix64::new(2);
    let (rows, clusters) = clustered(&mut rng, 500, 256, 12, 0.5);
    let best = InMemory::build(&params(Metric::L2), &rows, &clusters).unwrap();
    let scalar = InMemory::build(&params(Metric::L2), &rows, &clusters)
        .unwrap()
        .with_tier(Tier::SCALAR);
    let (mut a, mut b) = (ctx(64), ctx(64));
    let some = SearchParams {
        k: 10,
        nprobe: 4,
        rerank: 30,
        reach: None,
    };
    let dim = 256;
    for q in gaussian(&mut rng, 10, dim, |_| 1.0).chunks_exact(dim) {
        let hits = bits(best.search(q, &some, &mut a).unwrap());
        assert_eq!(hits, bits(scalar.search(q, &some, &mut b).unwrap()));
    }
}

#[test]
fn finds_most_true_neighbors_on_clustered_data() {
    let mut rng = SplitMix64::new(3);
    let (dim, n, nlist, spread) = (128, 2000, 16, 1.0);
    let (rows, clusters) = clustered(&mut rng, n, dim, nlist, spread);
    let index = InMemory::build(&params(Metric::L2), &rows, &clusters).unwrap();
    let mut ctx = ctx(200);
    let some = SearchParams {
        k: 10,
        nprobe: 8,
        rerank: 200,
        reach: None,
    };

    // Queries land near the data's clusters, the way real queries land near the documents they match.
    let mut found = 0;
    for _ in 0..20 {
        let c = (rng.next_u64() % nlist as u64) as usize;
        let center = &clusters.centroids[c * dim..(c + 1) * dim];
        let q: Vec<f32> = center.iter().map(|&x| x + (rng.normal() * spread) as f32).collect();
        let mut exact: Vec<(f64, u64)> = rows
            .data
            .chunks_exact(dim)
            .enumerate()
            .map(|(i, o)| (dist2(&q, o), i as u64))
            .collect();
        exact.sort_by(|a, b| a.0.total_cmp(&b.0));
        let truth: Vec<u64> = exact.iter().take(10).map(|e| e.1).collect();
        found += index
            .search(&q, &some, &mut ctx)
            .unwrap()
            .iter()
            .filter(|h| truth.contains(&h.id))
            .count();
    }

    let recall = found as f64 / 200.0;
    assert!(recall >= 0.9, "found {recall} of the true 10 nearest neighbors");
}

#[test]
fn estimates_cover_the_true_distances() {
    // Clusters of isotropic data and queries near them. For either anchor, t measures the query from what the codes
    // are measured from, with the query's own 4-bit error added for a cluster's center, and the error bar f_err·t
    // holds for ~95% of pairs.
    let mut rng = SplitMix64::new(8);
    let (dim, n, nlist) = (256, 1200, 8);
    let (rows, clusters) = clustered(&mut rng, n, dim, nlist, 1.0);
    let mut queries = Vec::with_capacity(10 * dim);

    for _ in 0..10 {
        let r = (rng.next_u64() % n as u64) as usize;
        let near = rows.data[r * dim..(r + 1) * dim]
            .iter()
            .map(|&x| x + (rng.normal() * 0.5) as f32);
        queries.extend(near);
    }

    let (d_pad, size) = (padded_dim(dim), block_bytes(padded_dim(dim)));
    let limits = Limits {
        max_k: 10,
        max_nprobe: nlist,
        max_rerank: n,
    };

    for anchor in [Anchor::Mean, Anchor::Cluster] {
        let p = BuildParams {
            anchor,
            ..params(Metric::L2)
        };
        let index = InMemory::build(&p, &rows, &clusters).unwrap();
        let (view, mut s) = (index.view(), Scratch::new(dim, nlist, &limits));
        let (mut sum, mut sum_sq, mut covered, mut pairs) = (0f64, 0f64, 0usize, 0usize);
        for q in queries.chunks_exact(dim) {
            prepare_query(&view, q, &mut s).unwrap();
            for (leaf, list) in index.lists.iter().enumerate() {
                let blocks = &index.blocks[list.offset..list.offset + list.count.div_ceil(BLOCK) * size];
                s.candidates.reset(n);
                scan_leaf(
                    &view,
                    leaf as LeafId,
                    blocks,
                    &mut s.lut,
                    &s.q32,
                    &mut s.candidates,
                    &mut s.survivors,
                );
                for c in s.candidates.as_slice() {
                    let (j, r) = ((c.id - list.first_id) as usize, index.ids[c.id as usize] as usize);
                    let f = lane_factors(&blocks[j / BLOCK * size..(j / BLOCK + 1) * size], d_pad, j % BLOCK);
                    let err = f64::from(c.score) - dist2(q, &rows.data[r * dim..(r + 1) * dim]);
                    sum += err;
                    sum_sq += err * err;
                    covered += usize::from(err.abs() <= f64::from(f.f_err) * f64::from(s.lut.k.t));
                    pairs += 1;
                }
            }
        }
        assert_eq!(pairs, 10 * n, "{anchor:?}");
        let (bias, rms) = (sum / pairs as f64, (sum_sq / pairs as f64).sqrt());
        let coverage = covered as f64 / pairs as f64;
        assert!(
            bias.abs() <= 0.1 * rms,
            "{anchor:?}: the mean error is {bias}, against an rms error of {rms}"
        );
        assert!(
            (0.92..=0.96).contains(&coverage),
            "{anchor:?}: the error bar covers {coverage} of pairs"
        );
    }
}

#[test]
fn a_search_can_return_nothing() {
    let mut rng = SplitMix64::new(4);
    let (rows, clusters) = clustered(&mut rng, 50, 128, 4, 0.5);
    let index = InMemory::build(&params(Metric::L2), &rows, &clusters).unwrap();
    let mut ctx = ctx(10);
    let q = &rows.data[..128];
    for params in [
        SearchParams {
            k: 0,
            nprobe: 4,
            rerank: 10,
            reach: None,
        },
        SearchParams {
            k: 5,
            nprobe: 0,
            rerank: 10,
            reach: None,
        },
    ] {
        assert!(index.search(q, &params, &mut ctx).unwrap().is_empty());
    }
}

#[test]
fn clusters_without_vectors_are_skipped() {
    let mut rng = SplitMix64::new(5);
    let (rows, mut clusters) = clustered(&mut rng, 100, 128, 4, 0.5);
    // Clusters 1 and 3 get no vectors.
    clusters
        .assignment
        .iter_mut()
        .for_each(|c| *c = if *c % 2 == 0 { *c } else { *c - 1 });
    let index = InMemory::build(&params(Metric::L2), &rows, &clusters).unwrap();
    assert_eq!((index.lists[1].count, index.lists[3].count), (0, 0));

    let mut ctx = ctx(100);
    let all = SearchParams {
        k: 10,
        nprobe: 4,
        rerank: 100,
        reach: None,
    };
    let q = &rows.data[..128];
    assert_eq!(
        bits(index.search(q, &all, &mut ctx).unwrap()),
        bits(&exact_8_bit(&index, q, 10))
    );
}

#[test]
fn lists_follow_the_file_order() {
    let size = 1000;
    let in_id_order = [
        List {
            offset: 0,
            count: 2,
            first_id: 0,
        },
        List {
            offset: 1000,
            count: 0,
            first_id: 2,
        },
        List {
            offset: 1000,
            count: 3,
            first_id: 2,
        },
        List {
            offset: 2000,
            count: 0,
            first_id: 5,
        },
    ];
    assert_eq!(lists(&[2, 0, 2, 2, 0], &[0, 1, 2, 3], size), in_id_order);

    // Clusters 2, 0, 3, 1 at file positions 0 to 3.
    let reordered = [
        List {
            offset: 0,
            count: 3,
            first_id: 0,
        },
        List {
            offset: 1000,
            count: 2,
            first_id: 3,
        },
        List {
            offset: 2000,
            count: 0,
            first_id: 5,
        },
        List {
            offset: 2000,
            count: 0,
            first_id: 5,
        },
    ];
    assert_eq!(lists(&[2, 0, 2, 2, 0], &[2, 0, 3, 1], size), reordered);
}

#[test]
fn bad_input_is_rejected_with_invalid_input() {
    let mut rng = SplitMix64::new(6);
    let (rows, clusters) = clustered(&mut rng, 40, 128, 3, 0.5);
    let index = InMemory::build(&params(Metric::Cosine), &rows, &clusters).unwrap();
    let mut ctx = ctx(10);
    let ok = SearchParams {
        k: 5,
        nprobe: 3,
        rerank: 10,
        reach: None,
    };
    let kind = |r: Result<&[Hit], Error>| r.map(|_| ()).unwrap_err().kind();

    assert_eq!(kind(index.search(&[1.0; 127], &ok, &mut ctx)), ErrorKind::InvalidInput);
    assert_eq!(kind(index.search(&[0.0; 128], &ok, &mut ctx)), ErrorKind::InvalidInput);
    let mut nan = vec![1.0f32; 128];
    nan[5] = f32::NAN;
    assert_eq!(kind(index.search(&nan, &ok, &mut ctx)), ErrorKind::InvalidInput);
    let too_many = SearchParams { k: 11, ..ok };
    assert_eq!(
        kind(index.search(&[1.0; 128], &too_many, &mut ctx)),
        ErrorKind::InvalidInput
    );

    let build = |rows: &Rows, clusters: &Clusters| InMemory::build(&params(Metric::Cosine), rows, clusters).map(|_| ());
    let wrong_id = Clusters {
        assignment: vec![3; 40],
        ..clone_clusters(&clusters)
    };
    let short = Clusters {
        assignment: vec![0; 39],
        ..clone_clusters(&clusters)
    };
    let other_dim = Clusters {
        dim: 64,
        ..clone_clusters(&clusters)
    };
    for bad in [wrong_id, short, other_dim] {
        assert_eq!(build(&rows, &bad).unwrap_err().kind(), ErrorKind::InvalidInput);
    }
    let zero = Rows {
        dim: 128,
        data: vec![0.0; 40 * 128],
    };
    assert_eq!(build(&zero, &clusters).unwrap_err().kind(), ErrorKind::InvalidInput);
}

fn clone_clusters(c: &Clusters) -> Clusters {
    Clusters {
        dim: c.dim,
        centroids: c.centroids.clone(),
        assignment: c.assignment.clone(),
        parents: c.parents.clone(),
        parent_of: c.parent_of.clone(),
    }
}

#[cfg(feature = "inmem")]
#[test]
fn the_public_index_gives_the_same_hits() {
    let mut rng = SplitMix64::new(7);
    let (rows, clusters) = clustered(&mut rng, 200, 128, 6, 0.5);
    let inner = InMemory::build(&params(Metric::L2), &rows, &clusters).unwrap();
    let public = crate::InMemoryIndex::build(&params(Metric::L2), &rows, &clusters).unwrap();
    let (mut a, mut b) = (ctx(20), ctx(20));
    let some = SearchParams {
        k: 10,
        nprobe: 3,
        rerank: 20,
        reach: None,
    };
    let q = &rows.data[..128];
    assert_eq!(
        bits(inner.search(q, &some, &mut a).unwrap()),
        bits(public.search(q, &some, &mut b).unwrap())
    );
}

#[test]
fn more_search_checks_reject_with_invalid_input_and_rerank_below_k_still_gives_k() {
    let mut rng = SplitMix64::new(9);
    let (rows, clusters) = clustered(&mut rng, 40, 128, 3, 0.5);
    let index = InMemory::build(&params(Metric::L2), &rows, &clusters).unwrap();
    let mut ctx = ctx(10);
    let ok = SearchParams {
        k: 5,
        nprobe: 3,
        rerank: 10,
        reach: None,
    };
    let kind = |r: Result<&[Hit], Error>| r.map(|_| ()).unwrap_err().kind();

    for bad in [SearchParams { rerank: 11, ..ok }, SearchParams { nprobe: 11, ..ok }] {
        assert_eq!(kind(index.search(&[1.0; 128], &bad, &mut ctx)), ErrorKind::InvalidInput);
    }
    // Finite values whose squared distance from the mean overflows an f32.
    assert_eq!(kind(index.search(&[1e18; 128], &ok, &mut ctx)), ErrorKind::InvalidInput);

    let few = SearchParams { rerank: 2, ..ok };
    assert_eq!(index.search(&rows.data[..128], &few, &mut ctx).unwrap().len(), 5);
}

#[test]
fn every_build_check_rejects_with_invalid_input() {
    let mut rng = SplitMix64::new(10);
    let (rows, clusters) = clustered(&mut rng, 40, 128, 3, 0.5);
    let kind = |rows: &Rows, c: &Clusters, p: &BuildParams| InMemory::build(p, rows, c).map(|_| ()).unwrap_err().kind();
    let l2 = params(Metric::L2);

    let mut bad_clusters = Vec::new();
    let mut c = clone_clusters(&clusters);
    c.parent_of[0] = 1;
    bad_clusters.push(c);
    let mut c = clone_clusters(&clusters);
    c.parent_of.pop();
    bad_clusters.push(c);
    bad_clusters.push(Clusters {
        parents: Vec::new(),
        ..clone_clusters(&clusters)
    });
    bad_clusters.push(Clusters {
        parents: vec![0.5; 64],
        ..clone_clusters(&clusters)
    });
    let mut c = clone_clusters(&clusters);
    c.centroids.push(0.5);
    bad_clusters.push(c);
    let mut c = clone_clusters(&clusters);
    c.centroids[7] = f32::NAN;
    bad_clusters.push(c);
    let mut c = clone_clusters(&clusters);
    c.parents[7] = f32::NAN;
    bad_clusters.push(c);
    for (i, c) in bad_clusters.iter().enumerate() {
        assert_eq!(kind(&rows, c, &l2), ErrorKind::InvalidInput, "case {i}");
    }

    let mut nan = Rows {
        dim: 128,
        data: rows.data.clone(),
    };
    nan.data[5] = f32::NAN;
    let mut far = Rows {
        dim: 128,
        data: rows.data.clone(),
    };
    far.data[..128].fill(1e19);
    for bad in [nan, far] {
        assert_eq!(kind(&bad, &clusters, &l2), ErrorKind::InvalidInput);
    }

    for page_size in [0, 4096 + 1, 2 << 20] {
        assert_eq!(
            kind(
                &rows,
                &clusters,
                &BuildParams {
                    page_size,
                    ..params(Metric::L2)
                }
            ),
            ErrorKind::InvalidInput
        );
    }

    // Dimensions just outside 128 to 4080. The page size is large enough that only the dimension is wrong.
    for dim in [127, 4081] {
        let (rows, clusters) = clustered(&mut rng, 10, dim, 2, 0.5);
        let p = BuildParams {
            page_size: 8192,
            ..params(Metric::L2)
        };
        assert_eq!(kind(&rows, &clusters, &p), ErrorKind::InvalidInput, "dim {dim}");
    }
}

#[test]
fn the_anchor_is_the_mean_of_the_vectors() {
    let mut rng = SplitMix64::new(11);
    let (mut rows, clusters) = clustered(&mut rng, 100, 128, 3, 0.5);
    rows.data.iter_mut().for_each(|x| *x += 5.0);

    let index = InMemory::build(&mean_anchor(Metric::L2), &rows, &clusters).unwrap();
    assert_eq!(index.mean, mean(&rows.data, 128));

    let mut normalized = rows.data.clone();
    let dim = 128;
    normalized.chunks_exact_mut(dim).for_each(|v| assert!(normalize(v)));
    let index = InMemory::build(&mean_anchor(Metric::Cosine), &rows, &clusters).unwrap();
    assert_eq!(index.mean, mean(&normalized, 128));
}

/// The tests' build settings, with the mean anchor.
fn mean_anchor(metric: Metric) -> BuildParams {
    BuildParams {
        anchor: Anchor::Mean,
        ..params(metric)
    }
}

#[test]
fn chained_and_id_order_find_the_same_hits() {
    let mut rng = SplitMix64::new(12);
    let (rows, mut clusters) = clustered(&mut rng, 600, 128, 16, 0.5);
    group(&mut clusters, 4);
    let by_id = InMemory::build(&params(Metric::L2), &rows, &clusters).unwrap();
    let chained = InMemory::build(
        &BuildParams {
            order: LeafOrder::Chained,
            ..params(Metric::L2)
        },
        &rows,
        &clusters,
    )
    .unwrap();

    let order = leaf_order(LeafOrder::Chained, &clusters, &chained.mean);
    assert_ne!(
        order,
        (0..16).collect::<Vec<u32>>(),
        "the test needs an order that isn't the identity"
    );
    for (p, &c) in order.iter().enumerate() {
        assert_eq!(chained.parent_of[p], clusters.parent_of[c as usize] as u16);
        assert_eq!(
            chained.lists[p].count,
            clusters.assignment.iter().filter(|&&a| a == c).count()
        );
    }

    let (mut a, mut b) = (ctx(64), ctx(64));
    let dim = 128;
    for q in gaussian(&mut rng, 6, dim, |_| 1.0).chunks_exact(dim) {
        for nprobe in [5, 16] {
            let sp = SearchParams {
                k: 10,
                nprobe,
                rerank: 40,
                reach: None,
            };
            assert_eq!(
                bits(by_id.search(q, &sp, &mut a).unwrap()),
                bits(chained.search(q, &sp, &mut b).unwrap())
            );
        }
    }
}

#[test]
fn a_build_reads_every_chunk_and_keeps_every_id() {
    let mut rng = SplitMix64::new(13);
    let (dim, n) = (128, 4100);
    let (rows, clusters) = clustered(&mut rng, n, dim, 3, 0.5);
    let index = InMemory::build(&params(Metric::L2), &rows, &clusters).unwrap();
    let data = index.in_place();

    let mut ids = index.ids.clone();
    ids.sort_unstable();
    assert_eq!(ids, (0..n as u64).collect::<Vec<_>>());

    // Each row decodes to its own input vector, within half a step, so no row came from the wrong chunk.
    for id in 0..n as u32 {
        let (row, ext) = (data.rows.row(id), index.ids[id as usize] as usize);
        assert_eq!(sq8::id(row, dim), ext as u64);
        let scale = f64::from(sq8::scale(row, dim));
        for (i, (&x, &m)) in rows.data[ext * dim..(ext + 1) * dim]
            .iter()
            .zip(&index.mean)
            .enumerate()
        {
            let decoded = scale * f64::from(sq8::code(row, i));
            assert!((f64::from(x - m) - decoded).abs() <= scale * (0.5 + 1e-4), "row {id}");
        }
    }
}

#[test]
fn one_ctx_serves_indexes_of_different_shapes() {
    let mut rng = SplitMix64::new(14);
    let shapes = [(128, 12), (200, 20), (128, 20)];
    let indexes: Vec<InMemory> = shapes
        .iter()
        .map(|&(dim, nlist)| {
            let (rows, clusters) = clustered(&mut rng, 300, dim, nlist, 0.5);
            InMemory::build(&params(Metric::L2), &rows, &clusters).unwrap()
        })
        .collect();
    let sp = SearchParams {
        k: 10,
        nprobe: 4,
        rerank: 30,
        reach: None,
    };
    let mut shared = ctx(64);

    for round in 0..3 {
        for index in &indexes {
            let q = gaussian(&mut rng, 1, index.dim, |_| 1.0);
            let want = bits(index.search(&q, &sp, &mut ctx(64)).unwrap());
            assert_eq!(bits(index.search(&q, &sp, &mut shared).unwrap()), want, "round {round}");
        }
    }
    assert_eq!(shared.scratch.len(), 2, "one set of buffers per dimension");
}

#[test]
fn route_lists_the_nearest_clusters_first() {
    let mut rng = SplitMix64::new(15);
    let (rows, clusters) = clustered(&mut rng, 300, 128, 12, 0.5);
    let index = InMemory::build(&params(Metric::L2), &rows, &clusters).unwrap();
    let mut ctx = ctx(64);
    let (mut full, mut some) = (Vec::new(), Vec::new());

    let dim = 128;
    for q in gaussian(&mut rng, 4, dim, |_| 1.0).chunks_exact(dim) {
        index.route(q, 12, None, &mut ctx, &mut full).unwrap();
        let mut sorted = full.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..12).collect::<Vec<u32>>());
        for n in [1, 3, 7] {
            index.route(q, n, None, &mut ctx, &mut some).unwrap();
            assert_eq!(some, full[..n]);
        }
    }
}

#[test]
fn every_batch_split_gives_the_same_index() {
    // 20,000 vectors fill 5 chunks, which one thread encodes in 3 batches, two threads in 2 and eight threads in 1.
    let mut rng = SplitMix64::new(14);
    let (dim, n) = (128, 20_000);
    let (mut rows, clusters) = clustered(&mut rng, n, dim, 20, 0.5);
    // A far last row moves the mean, so a mean of only the first chunks would show.
    rows.data[(n - 1) * dim..].iter_mut().for_each(|x| *x += 50.0);
    let build = |threads| {
        let p = BuildParams {
            threads,
            ..params(Metric::L2)
        };
        InMemory::build(&p, &rows, &clusters).unwrap()
    };
    let one = build(1);
    let parts =
        |index: &InMemory| -> Vec<Vec<u8>> { crate::format::encode(index).parts.iter().map(|p| p.to_vec()).collect() };

    let want = parts(&one);
    for threads in [2, 8] {
        assert!(parts(&build(threads)) == want, "{threads} threads built another index");
    }

    let exact = mean(&rows.data, dim);
    assert!(one.mean.iter().zip(&exact).all(|(a, b)| (a - b).abs() < 1e-5));
    let data = one.in_place();
    let mut ids = one.ids.clone();
    ids.sort_unstable();
    assert_eq!(ids, (0..n as u64).collect::<Vec<_>>());
    for id in 0..n as u32 {
        assert_eq!(sq8::id(data.rows.row(id), dim), one.ids[id as usize], "row {id}");
    }
}

#[test]
fn a_bad_vector_in_a_later_batch_fails_the_build() {
    // Row 19,000 is in the third batch of a one-thread build, and too far from the mean to encode.
    let (dim, n) = (128, 20_000);
    let (mut rows, clusters) = clustered(&mut SplitMix64::new(15), n, dim, 4, 0.5);
    rows.data[19_000 * dim..19_001 * dim].fill(1e19);

    let err = InMemory::build(&params(Metric::L2), &rows, &clusters)
        .map(|_| ())
        .unwrap_err();
    assert_eq!(err.to_string(), "a vector is too far from the mean to encode");
}

#[test]
fn a_reach_reads_only_the_clusters_within_it() {
    let mut rng = SplitMix64::new(16);
    let (dim, n, nlist) = (128, 2000, 32);
    let (rows, clusters) = clustered(&mut rng, n, dim, nlist, 1.0);
    let index = InMemory::build(&params(Metric::L2), &rows, &clusters).unwrap();
    let (mut a, mut b) = (ctx(64), ctx(64));
    let mut routes = Vec::new();

    for q in gaussian(&mut rng, 10, dim, |_| 1.0).chunks_exact(dim) {
        for reach in [0.0, 0.1, 0.3, 1.0] {
            let reached = SearchParams {
                k: 10,
                nprobe: 20,
                rerank: 40,
                reach: Some(reach),
            };
            // The same search with exactly the clusters in reach.
            index.route(q, 20, None, &mut a, &mut routes).unwrap();
            let s = crate::search::scratch_for(&mut a.scratch, dim, nlist, &a.limits);
            let count = crate::search::route::within_reach(&s.routes, Some(reach));
            let fixed = SearchParams {
                nprobe: count,
                reach: None,
                ..reached
            };
            assert_eq!(
                bits(index.search(q, &reached, &mut a).unwrap()),
                bits(index.search(q, &fixed, &mut b).unwrap())
            );
        }
    }

    let bad = SearchParams {
        k: 10,
        nprobe: 20,
        rerank: 40,
        reach: Some(f32::NAN),
    };
    assert_eq!(
        index
            .search(&rows.data[..dim], &bad, &mut a)
            .map(|_| ())
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidInput
    );
}

#[test]
fn the_cluster_anchor_finds_at_least_as_many_true_neighbors_as_the_mean_anchor() {
    // Every cluster is read, so only the 1-bit estimates decide which candidates get rescored.
    let mut rng = SplitMix64::new(17);
    let (dim, n, nlist, spread) = (128, 3000, 30, 0.5);
    let (rows, clusters) = clustered(&mut rng, n, dim, nlist, spread);
    let by_cluster = InMemory::build(&params(Metric::L2), &rows, &clusters).unwrap();
    let by_mean = InMemory::build(&mean_anchor(Metric::L2), &rows, &clusters).unwrap();
    assert_eq!((by_cluster.anchor, by_mean.anchor), (Anchor::Cluster, Anchor::Mean));
    let queries: Vec<(Vec<f32>, Vec<u64>)> = (0..50)
        .map(|_| {
            let r = (rng.next_u64() % n as u64) as usize;
            let q: Vec<f32> = rows.data[r * dim..(r + 1) * dim]
                .iter()
                .map(|&x| x + (rng.normal() * spread) as f32)
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

    for rerank in [10, 20, 40] {
        let sp = SearchParams {
            k: 10,
            nprobe: nlist,
            rerank,
            reach: None,
        };
        let recall = |index: &InMemory| {
            let mut c = ctx(64);
            let found: usize = queries
                .iter()
                .map(|(q, truth)| {
                    let hits = index.search(q, &sp, &mut c).unwrap();
                    hits.iter().filter(|h| truth.contains(&h.id)).count()
                })
                .sum();
            found as f64 / (10 * queries.len()) as f64
        };
        let (cluster, mean) = (recall(&by_cluster), recall(&by_mean));
        assert!(
            cluster >= mean,
            "rescoring {rerank}, the cluster anchor finds {cluster} and the mean anchor {mean}"
        );
    }
}
