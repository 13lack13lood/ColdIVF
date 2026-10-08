//! Checks that searches allocate nothing once a search context is warm. A counting allocator counts every allocation
//! in the process while a check runs, so this runs as its own program.

use std::alloc::{GlobalAlloc, Layout, System};
use std::process::ExitCode;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use coldivf::{
    BuildParams, Builder, Collections, Index, LeafOrder, Limits, Metric, SearchCtx, SearchParams, VectorSource, Writer,
};

struct Counting;

static ON: AtomicBool = AtomicBool::new(false);
static COUNT: AtomicUsize = AtomicUsize::new(0);

fn note() {
    if ON.load(Ordering::Relaxed) {
        COUNT.fetch_add(1, Ordering::Relaxed);
    }
}

// SAFETY: every call goes straight to the system allocator with the caller's arguments. Counting touches only atomics.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        note();
        // SAFETY: the caller keeps the promises `GlobalAlloc::alloc` asks of it.
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        note();
        // SAFETY: as for `alloc`.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        note();
        // SAFETY: `ptr` came from this allocator with `layout`, as the caller promises.
        unsafe { System.realloc(ptr, layout, new_size) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: as for `realloc`.
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

/// How many allocations `f` makes.
fn count(f: impl FnOnce()) -> usize {
    COUNT.store(0, Ordering::SeqCst);
    ON.store(true, Ordering::SeqCst);
    f();
    ON.store(false, Ordering::SeqCst);
    COUNT.load(Ordering::SeqCst)
}

/// Rows held in memory.
struct Rows {
    dim: usize,
    data: Vec<f32>,
}

impl VectorSource for Rows {
    fn count(&self) -> usize {
        self.data.len() / self.dim
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn rows(&self, start: usize, rows: usize) -> &[f32] {
        &self.data[start * self.dim..(start + rows) * self.dim]
    }
}

/// Random values in [−1, 1), from a SplitMix64 stream.
fn random(state: &mut u64, n: usize) -> Vec<f32> {
    (0..n)
        .map(|_| {
            *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = *state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            ((z ^ (z >> 31)) >> 40) as f32 / (1u64 << 23) as f32 - 1.0
        })
        .collect()
}

/// The index file, deleted when dropped, even when a check panics.
struct TempFile(std::path::PathBuf);

impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// Runs every search call with `ctx` at the settings `sp`, and returns each call's name and allocation count.
fn checks(index: &Index, ctx: &mut SearchCtx, q: &[f32], sp: &SearchParams) -> Vec<(&'static str, usize)> {
    let (k, nprobe, rerank, reach) = (sp.k, sp.nprobe, sp.rerank, sp.reach);
    let mut leaves = Vec::with_capacity(nprobe);
    index.route(q, nprobe, reach, ctx, &mut leaves).unwrap();
    vec![
        ("search", count(|| _ = index.search(q, sp, ctx).unwrap())),
        (
            "route",
            count(|| index.route(q, nprobe, reach, ctx, &mut leaves).unwrap()),
        ),
        ("prepare", count(|| index.prepare(ctx).unwrap())),
        (
            "fetch, prefetch_rows and search",
            count(|| {
                let mut fetched = index.fetch(&leaves, ctx).unwrap();
                fetched.prefetch_rows(q, rerank).unwrap();
                _ = fetched.search(q, k, rerank).unwrap();
            }),
        ),
    ]
}

fn main() -> ExitCode {
    // The counter must see an allocation, or a check that finds none proves nothing.
    assert_eq!(count(|| drop(std::hint::black_box(vec![0u8; 8]))), 1);

    let (dim, n) = (128, 4000);
    let mut state = 1;
    let data = random(&mut state, n * dim);
    let rows = Rows { dim, data };
    let file = TempFile(std::env::temp_dir().join(format!("coldivf-alloc-{}.coldivf", std::process::id())));
    let params = BuildParams {
        metric: Metric::L2,
        vectors_per_list: 100,
        parents: 4,
        order: LeafOrder::Chained,
        page_size: 4096,
        threads: 2,
        seed: 7,
    };
    Builder::new(params.clone()).build(&rows, &file.0).unwrap();
    let index = Index::open(&file.0).unwrap();
    let limits = Limits {
        max_k: 10,
        max_nprobe: 40,
        max_rerank: 60,
    };
    let queries = random(&mut state, 8 * dim);
    let mut failed = false;

    // A context warmed at the smallest settings must still allocate nothing at its limits, since it sizes every
    // buffer from them. A context warmed at the limits is checked the same way, and so is a search with a reach.
    for (k, nprobe, rerank, reach) in [(1, 1, 1, None), (10, 40, 60, None), (10, 40, 60, Some(0.2))] {
        let mut ctx = SearchCtx::new(limits);
        let small = SearchParams {
            k,
            nprobe,
            rerank,
            reach,
        };
        let sp = SearchParams {
            k: 10,
            nprobe: 40,
            rerank: 60,
            reach,
        };
        checks(&index, &mut ctx, &queries[..dim], &small);
        for q in queries.chunks_exact(dim).skip(1) {
            for (name, allocations) in checks(&index, &mut ctx, q, &sp) {
                if allocations > 0 {
                    eprintln!("alloc: {name} made {allocations} allocations, warmed up at {small:?}");
                    failed = true;
                }
            }
        }
    }

    // A search over several collections keeps its parts in the context too, even for a context first used at the
    // smallest settings that then takes turns between two sets of collections of other shapes.
    let other = TempFile(std::env::temp_dir().join(format!("coldivf-alloc-{}-2.coldivf", std::process::id())));
    let third = TempFile(std::env::temp_dir().join(format!("coldivf-alloc-{}-3.coldivf", std::process::id())));
    let more = Rows {
        dim,
        data: random(&mut state, n * dim),
    };
    Builder::new(params.clone()).build(&more, &other.0).unwrap();
    let finer = BuildParams {
        vectors_per_list: 50,
        ..params
    };
    Builder::new(finer).build(&more, &third.0).unwrap();
    let sets = [
        Collections::open(&[&file.0, &other.0]).unwrap(),
        Collections::open(&[&third.0, &file.0]).unwrap(),
    ];
    let mut ctx = SearchCtx::new(limits);
    let mut out = Vec::with_capacity(10);
    let small = SearchParams {
        k: 1,
        nprobe: 1,
        rerank: 1,
        reach: None,
    };
    let sp = SearchParams {
        k: 10,
        nprobe: 40,
        rerank: 60,
        reach: Some(0.3),
    };
    sets.iter()
        .for_each(|set| set.search(&queries[..dim], &small, &mut ctx, &mut out).unwrap());

    for q in queries.chunks_exact(dim).skip(1) {
        for set in &sets {
            let allocations = count(|| set.search(q, &sp, &mut ctx, &mut out).unwrap());
            if allocations > 0 {
                eprintln!("alloc: a search of two collections made {allocations} allocations");
                failed = true;
            }
        }
    }

    // Inserts are scanned and deletes filtered in place, so they allocate nothing either.
    let _log = TempFile(std::path::PathBuf::from(format!("{}.delta", file.0.display())));
    let mut writer = Writer::open(&file.0).unwrap();

    for (i, v) in random(&mut state, 50 * dim).chunks_exact(dim).enumerate() {
        writer.insert(100_000 + i as u64, v).unwrap();
    }

    (0..20).for_each(|id| writer.delete(id).unwrap());
    let sp = SearchParams {
        k: 10,
        nprobe: 40,
        rerank: 60,
        reach: None,
    };
    let mut ctx = SearchCtx::new(limits);
    _ = index.search(&queries[..dim], &sp, &mut ctx).unwrap();

    for q in queries.chunks_exact(dim).skip(1) {
        let allocations = count(|| _ = index.search(q, &sp, &mut ctx).unwrap());
        if allocations > 0 {
            eprintln!("alloc: a search with inserts and deletes made {allocations} allocations");
            failed = true;
        }
    }

    // A warm index scans in place and reads only rows, which must not allocate either, with or without a reach.
    index.set_warm(true);

    for reach in [None, Some(0.2)] {
        let sp = SearchParams {
            k: 10,
            nprobe: 40,
            rerank: 60,
            reach,
        };
        let mut ctx = SearchCtx::new(limits);
        _ = index.search(&queries[..dim], &sp, &mut ctx).unwrap();
        for q in queries.chunks_exact(dim).skip(1) {
            let allocations = count(|| _ = index.search(q, &sp, &mut ctx).unwrap());
            if allocations > 0 {
                eprintln!("alloc: a warm search made {allocations} allocations with reach {reach:?}");
                failed = true;
            }
        }
    }

    if failed {
        return ExitCode::FAILURE;
    }

    println!("alloc: no search allocated after warming up");
    ExitCode::SUCCESS
}
