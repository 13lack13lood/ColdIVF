//! Unit tests for cold reads.

use super::*;
use crate::io::NewWave;
use crate::io::pool::PoolWave;
use crate::quantization::SplitMix64;
use crate::quantization::testutil::{Shuffled, TempFile, clustered, gaussian, group, params, temp_file, written};
use crate::search::inmem::InMemory;
use crate::{Index, Limits, Metric, SearchCtx, SearchParams};

/// Where each of a context's read buffers sits, and how much room it has.
fn buffers(ctx: &SearchCtx) -> Vec<(usize, usize)> {
    let batches = ctx.reads.iter().flat_map(|r| [&r.clusters, &r.early, &r.late]);
    let mut all: Vec<(usize, usize)> = batches
        .flat_map(|b| {
            [
                (b.keys.as_ptr() as usize, b.keys.capacity()),
                (b.spans.as_ptr() as usize, b.spans.capacity()),
                (b.at.as_ptr() as usize, b.at.capacity()),
                (b.ranges.as_ptr() as usize, b.ranges.capacity()),
                (b.arena.as_ptr() as usize, b.arena.len()),
            ]
        })
        .collect();
    all.push((ctx.leaves.as_ptr() as usize, ctx.leaves.capacity()));
    all.push((ctx.hits.as_ptr() as usize, ctx.hits.capacity()));
    all
}

/// The backends to try, where `None` keeps the one that opening picked.
fn waves() -> [Option<NewWave>; 3] {
    [
        None,
        Some(|| Box::<PoolWave>::default()),
        Some(|| Box::<Shuffled>::default()),
    ]
}

fn open(path: &TempFile, wave: Option<NewWave>) -> Index {
    let inner = Inner::open(path).unwrap();
    Index::of(wave.map_or(inner, |w| Inner::open(path).unwrap().with_wave(w)))
}

#[test]
fn queries_reuse_every_read_buffer() {
    let (_, file, _path) = written("buffers.coldivf", &params(Metric::L2), 3000, 128, 64);
    let index = Index::of(file);
    let dim = 128;
    let queries = gaussian(&mut SplitMix64::new(1), 20, dim, |_| 1.0);
    // The second context keeps more hits than candidates to rescore.
    let limits = [(50, 64, 200), (50, 64, 10)];

    for (max_k, max_nprobe, max_rerank) in limits {
        let mut ctx = SearchCtx::new(Limits {
            max_k,
            max_nprobe,
            max_rerank,
        });
        // An empty fetch makes the reads, so the buffers are all there before the first search.
        index.fetch(&[], &mut ctx).unwrap();
        let first = buffers(&ctx);
        let most = SearchParams {
            k: max_k,
            nprobe: max_nprobe,
            rerank: max_rerank,
            reach: None,
        };
        // A fetch of up to `max_nprobe` leaves, repeats included, fits too.
        let leaves: Vec<u32> = (0..64).map(|i| i % 40).collect();
        for q in queries.chunks_exact(dim) {
            index.search(q, &most, &mut ctx).unwrap();
            let mut fetched = index.fetch(&leaves, &mut ctx).unwrap();
            fetched.prefetch_rows(q, max_rerank).unwrap();
            fetched.search(q, max_k, max_rerank).unwrap();
            assert_eq!(buffers(&ctx), first);
        }
    }
}

/// An index of 7500 vectors in 64 clusters, where clusters 0, 16, 32 and 48 hold 1500 vectors each and the other 60
/// share the rest, written to the test file `name`.
fn skewed(name: &str) -> (InMemory, TempFile) {
    let (rows, mut clusters) = clustered(&mut SplitMix64::new(3), 7500, 128, 64, 0.5);
    let big = [0u32, 16, 32, 48];
    let small: Vec<u32> = (0..64).filter(|c| !big.contains(c)).collect();
    clusters.assignment = (0..7500)
        .map(|r| if r < 6000 { big[r / 1500] } else { small[r % 60] })
        .collect();
    group(&mut clusters, 3);
    let index = InMemory::build(&params(Metric::L2), &rows, &clusters).unwrap();

    let path = temp_file(name);
    crate::build::write(&index, &path).unwrap();
    (index, path)
}

#[test]
fn the_arena_holds_the_largest_clusters_one_fetch_may_read() {
    let (memory, path) = skewed("largest.coldivf");
    let big = [0u32, 16, 32, 48];
    let ids: Vec<u64> = big
        .iter()
        .map(|&l| memory.lists[l as usize])
        .flat_map(|l| (l.first_id..l.first_id + l.count as u32).map(|i| memory.ids[i as usize]))
        .collect();
    assert_eq!(ids.len(), 6000);
    let q = &gaussian(&mut SplitMix64::new(2), 1, 128, |_| 1.0)[..];

    for wave in waves() {
        let index = open(&path, wave);
        let mut ctx = SearchCtx::new(Limits {
            max_k: 10,
            max_nprobe: 4,
            max_rerank: 40,
        });
        let hits = index.fetch(&big, &mut ctx).unwrap().search(q, 10, 40).unwrap();
        assert_eq!(hits.len(), 10);
        assert!(hits.iter().all(|h| ids.contains(&h.id)));
    }
}

#[test]
fn the_row_arenas_hold_the_most_rows_one_search_may_rescore() {
    // 11,200 vectors at 128 dimensions fill 400 row pages of 28 rows. Candidates 8 pages apart never share or touch a
    // read page, so each takes the most arena room that a row page can.
    let (memory, file, path) = written("rows.coldivf", &params(Metric::L2), 11_200, 128, 64);
    let (bytes, layout) = file.parts();
    let section = RowSection::new(layout);
    let limits = Limits {
        max_k: 50,
        max_nprobe: 4,
        max_rerank: 10,
    };
    let candidates: Vec<Candidate> = (0..50)
        .map(|i| Candidate {
            score: 0.0,
            id: (i * 8 * section.per_page) as u32,
        })
        .collect();
    let handle = File::open(&path).unwrap();
    let owner = Arc::new(Inner::open(&path).unwrap());
    let all: Vec<Box<dyn Wave>> = vec![
        Box::<PoolWave>::default(),
        Box::<Shuffled>::default(),
        #[cfg(target_os = "macos")]
        Box::<crate::io::macos::RdAdviseWave>::default(),
        #[cfg(target_os = "linux")]
        Box::<crate::io::linux::UringWave>::default(),
        #[cfg(target_os = "linux")]
        Box::<crate::io::linux::FadviseWave>::default(),
    ];

    for wave in all {
        let mut reads = Reads::new(&owner, wave, layout, bytes.len(), owner.tables().unwrap(), &limits);
        // Once through the late wave alone, then again with every row read early, then through the cache, as a warm
        // search reads them.
        for (early, warm) in [(false, false), (true, false), (false, true)] {
            reads.fetch(&handle, &memory.lists, &[]).unwrap();
            if early {
                reads.prefetch(&handle, &candidates).unwrap();
            }
            reads.read_rows(&handle, &candidates, warm).unwrap();
            for c in &candidates {
                let page = (c.id as usize / section.per_page) * section.page_size;
                let at = section.offset + page + (c.id as usize % section.per_page) * section.row_bytes;
                assert_eq!(reads.row(c.id), &bytes[at..at + section.row_bytes]);
            }
        }
    }
}

#[test]
fn the_late_wave_skips_the_rows_the_early_wave_read() {
    let (_, file, _path) = written("skip.coldivf", &params(Metric::L2), 3000, 128, 64);
    let index = Index::of(file);
    let mut ctx = SearchCtx::new(Limits {
        max_k: 10,
        max_nprobe: 16,
        max_rerank: 40,
    });
    let q = &gaussian(&mut SplitMix64::new(4), 1, 128, |_| 1.0)[..];
    let mut leaves = Vec::new();

    index.route(q, 16, None, &mut ctx, &mut leaves).unwrap();
    let mut fetched = index.fetch(&leaves, &mut ctx).unwrap();
    fetched.prefetch_rows(q, 40).unwrap();
    fetched.search(q, 10, 40).unwrap();
    let reads = &ctx.reads[0];
    assert!(!reads.early.keys.is_empty() && reads.late.ranges.is_empty());
}

#[test]
fn a_row_page_takes_the_fewest_read_pages_its_offsets_allow() {
    let section = |offset, page_size| RowSection {
        count: 1000,
        offset,
        row_bytes: 784,
        per_page: page_size / 784,
        page_size,
    };
    let cases = [
        (4096, 16384, 16384),
        (4096, 4096, 4096),
        (16384, 4096, 16384),
        (8192, 16384, 16384),
        (12288, 16384, 32768),
        (20480, 8192, 24576),
    ];

    for (page_size, page, room) in cases {
        assert_eq!(section(16384 * 3, page_size).room(page), room, "{page_size} in {page}");
        // Every row page fits the room, wherever it falls.
        for p in 0..64 {
            let start = 16384 * 3 + p * page_size;
            let rounded = (start + page_size).div_ceil(page) * page - start / page * page;
            assert!(rounded <= room, "page {p} of {page_size} in {page}");
        }
    }

    // A section off the read pages' boundaries falls back to the general bound.
    assert_eq!(section(4096, 4096).room(16384), 4096 + 2 * 16384);
}

#[test]
fn a_context_drops_the_reads_of_closed_indexes() {
    let open = |name: &str, dim| {
        let (_, file, path) = written(name, &params(Metric::L2), 400, dim, 12);
        (Index::of(file), path)
    };
    let search = |index: &Index, ctx: &mut SearchCtx| {
        let dim = index.inner().parts().1.dim;
        let sp = SearchParams {
            k: 5,
            nprobe: 4,
            rerank: 20,
            reach: None,
        };
        index.search(&vec![0.5; dim], &sp, ctx).unwrap();
    };
    let ((a, _pa), (b, _pb), (c, _pc)) = (open("closed-a", 128), open("closed-b", 200), open("closed-c", 128));
    let mut ctx = SearchCtx::new(Limits {
        max_k: 10,
        max_nprobe: 12,
        max_rerank: 40,
    });
    search(&a, &mut ctx);
    search(&b, &mut ctx);
    search(&a, &mut ctx);
    assert_eq!(ctx.reads.len(), 2);

    // Closing A frees nothing until another index needs new reads.
    drop(a);
    search(&b, &mut ctx);
    assert_eq!(ctx.reads.len(), 2);
    search(&c, &mut ctx);
    let owners: Vec<u64> = ctx.reads.iter().map(|r| r.cell).collect();
    assert_eq!(owners, [b.inner().cell(), c.inner().cell()]);
}

#[test]
fn a_commit_plans_a_context_s_reads_again_in_place() {
    use crate::maintain::tests::{build_params, churn, sample, settings};

    let (path, rows) = sample(
        "replan.coldivf",
        &build_params(Metric::L2, crate::Anchor::Cluster),
        3000,
        30,
        72,
    );
    let index = Index::open(&path).unwrap();
    let mut writer = crate::Writer::open(&path).unwrap();
    let mut ctx = SearchCtx::new(Limits {
        max_k: 10,
        max_nprobe: 20,
        max_rerank: 40,
    });
    let sp = SearchParams {
        k: 10,
        nprobe: 20,
        rerank: 40,
        reach: None,
    };
    let q = &rows.data[..128];
    index.search(q, &sp, &mut ctx).unwrap();
    let (first, generation) = (buffers(&ctx), ctx.reads[0].generation());

    // A fold after a few changes: the reads move to the new generation, in the same slot, with the same buffers.
    churn(&mut writer, &rows, 73);
    writer.maintain(&settings(), crate::RunKind::Fold).unwrap();
    let hits: Vec<u64> = index.search(q, &sp, &mut ctx).unwrap().iter().map(|h| h.id).collect();
    assert_eq!((ctx.reads.len(), buffers(&ctx)), (1, first));
    assert_ne!(ctx.reads[0].generation(), generation);
    assert_eq!(ctx.reads[0].generation(), index.inner().uid());
    // And they read the new file, as a context made now does.
    let mut fresh = SearchCtx::new(ctx.limits);
    let want: Vec<u64> = index.search(q, &sp, &mut fresh).unwrap().iter().map(|h| h.id).collect();
    assert_eq!(hits, want);
}
