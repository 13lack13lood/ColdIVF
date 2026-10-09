//! Recall, warm latency and throughput over a grid of settings. The index stays in memory, so every search scans its
//! clusters in place, which gives the same hits as a cold search.
//!
//! warm --index PATH --dataset NAME [--n N] [--nprobe 128,160,192,256] [--rerank 29,35,39] [--reach R] [--k 10]
//!      [--queries 1000] [--split all|tune|test] [--mode whole|t1] [--threads N] [--passes 3] [--seconds 10]
//!      [--warmup 1000] [--results results/warm.jsonl] [--label TEXT]
//!
//! `--split tune` takes the even positions among the dataset's first `--queries`, 1,000 unless it says otherwise, as
//! the benchmark driver splits them, `test` the odd ones, and `all` every one of the first `--queries`, all of them
//! unless it says otherwise. Each setting first runs `--warmup` searches untimed, one pass over the split's queries,
//! which gives the recall and brings their rows into the cache, and then more over them in turn. Then it times
//! `--passes` passes over them at one thread, and reports the 50th, 90th and 99th percentile latencies (p50, p90 and
//! p99) and the mean, each the median of the passes' values. With `--threads N` it runs N threads instead, each with
//! its own search context, and counts the queries per second (QPS) over at least `--seconds`. It does that `--passes`
//! times and reports the median. `warmup` records how many searches ran untimed.
//!
//! `--mode` sets what is in memory, and each line records it:
//! - `whole`: the whole index. It reads the file through before the first setting.
//! - `t1`: every cluster's 1-bit codes, which the warm-up reads, and the cluster tables, which the first search loads.
//!   Before each timed query it drops the rows from the cache, so each query reads the rows it rescores from the disk.
//!   It can't run with `--threads`.
//!
//! `resident_share` is the share of those pages in the cache when a setting's timing starts. In `t1`,
//! `rows_cached_pages` adds up the rows' pages still cached right before each timed query, so 0 means every query read
//! its rows from the disk.
//!
//! Recall@1, @10 and @100 come from the first 1, 10 and 100 hits, for those up to `--k` and the ground truth's width.
//! They count ties, and `ties` says so. A hit counts if its id is among the true k nearest, or if it is at most as far
//! from the query as the true k-th neighbor. That distance is measured from the base like the hit's, so a copy of the
//! k-th neighbor ties it exactly. When the distances file `<stem>.gt.dist.fbin` sits next to the ground truth, its
//! values check that the ground truth belongs to the base, and `truth_checked` is true. `recall10_p05` and
//! `recall10_p10` are the 5th and 10th percentiles of the queries' recall@10, by nearest rank. `recall` and
//! `share_at_0_9` are the recall at `--k` by ids alone, and the share of queries where it is at least 0.9.
//!
//! `backend` and `read_paths` say what read the rows, i.e. the read backend of each search context and how it is set
//! up, eg. on Linux whether a context got an io_uring ring. `cache_hits_per_query` is the timed searches' ranges that
//! came from the backend's own copies of pages rather than from the file.

use std::fs::File;
use std::io::Read;
use std::ops::Range;
use std::path::Path;
use std::sync::Barrier;
use std::time::{Duration, Instant};

use coldivf::{Index, Limits, SearchCtx, SearchParams};
use coldivf_bench::args::{Args, fail};
use coldivf_bench::coldcache::{Mapped, page_size};
use coldivf_bench::data::{Dataset, split, split_count};
use coldivf_bench::metrics::{IndexFile, Json, append, interactive, percentile, provenance, read_paths, recall, state};

/// The k of each recall@k a line reports, where `--k` and the ground truth's width allow it.
const LEVELS: [usize; 3] = [1, 10, 100];

fn main() {
    let args = Args::parse();
    let name = args.need("dataset");
    let n = args
        .str("n")
        .map(|v| v.parse().unwrap_or_else(|_| fail("--n must be a number")));
    let dataset = Dataset::open(name, n).unwrap_or_else(|e| fail(&format!("opening {name}: {e}")));
    let path = args.need("index");
    let id = IndexFile::read(Path::new(path)).unwrap_or_else(|e| fail(&format!("reading {path}: {e}")));
    let index = Index::open(Path::new(path)).unwrap_or_else(|e| fail(&format!("opening {path}: {e}")));
    let nprobes = args.list("nprobe", &[128, 160, 192, 256]);
    let reranks = args.list("rerank", &[29, 35, 39]);
    let reach = args
        .str("reach")
        .map(|v| v.parse().unwrap_or_else(|_| fail("--reach must be a number")));
    let k = args.get("k", 10);
    let part = args.str("split").unwrap_or("all");
    let asked = args
        .str("queries")
        .map(|v| v.parse().unwrap_or_else(|_| fail("--queries must be a number")));
    let nq = split_count(part, asked, dataset.queries.len());
    let queries = split(nq, part).unwrap_or_else(|| fail(&format!("--split must be tune, test or all, not {part}")));
    let warmup = args.get("warmup", 1000usize);
    let mode = args.str("mode").unwrap_or("whole");
    let threads = args.str("threads").map(|v| match v.parse::<usize>() {
        Ok(t) if t > 0 => t,
        _ => fail("--threads must be a whole number above 0"),
    });
    let (passes, seconds) = (args.get("passes", 3), args.get("seconds", 10.0f64));
    let results = args.str("results").unwrap_or("results/warm.jsonl");

    if mode != "whole" && mode != "t1" {
        fail(&format!("--mode must be whole or t1, not {mode}"));
    }

    if mode == "t1" && threads.is_some() {
        fail("--mode t1 drops the rows from the cache before each query, so it can't run with --threads");
    }

    if passes == 0 || seconds.is_nan() || seconds <= 0.0 || queries.is_empty() {
        fail("--passes and --seconds must be above 0, and the split must hold a query");
    }

    let limits = Limits {
        max_k: k,
        max_nprobe: nprobes.iter().copied().max().unwrap_or(1),
        max_rerank: reranks.iter().copied().max().unwrap_or(1).max(k),
    };
    let mut ctx = SearchCtx::new(limits);
    // Every check and drop of the cache goes through this one mapping, since a new mapping can make another process
    // read the file's first pages back.
    let file = Mapped::open(Path::new(path)).unwrap_or_else(|e| fail(&format!("mapping {path}: {e}")));
    let all = 0..file.len();
    let (resident, rows) = match mode {
        "whole" => (vec![all], None),
        _ => (vec![id.tables.clone(), id.codes.clone()], Some(id.rows.clone())),
    };

    if mode == "whole" {
        read_through(Path::new(path));
    }

    // A system that can't drop pages from the cache fails here, before the warm-up.
    if let Some(rows) = &rows {
        file.evict_range(rows.clone())
            .unwrap_or_else(|e| fail(&format!("dropping the rows from the cache: {e}")));
    }

    warm(&index);
    interactive();
    let vectors: Vec<&[f32]> = queries.iter().map(|&q| dataset.queries.row(q)).collect();

    for &nprobe in &nprobes {
        for &rerank in &reranks {
            let params = SearchParams {
                k,
                nprobe,
                rerank,
                reach,
            };
            let (mut found, mut scanned) = (Vec::with_capacity(vectors.len()), 0);

            for vector in &vectors {
                let hits = index
                    .search(vector, &params, &mut ctx)
                    .unwrap_or_else(|e| fail(&format!("searching: {e}")));
                found.push(hits.iter().map(|h| h.id).collect::<Vec<u64>>());
                scanned += ctx.trace().scanned;
            }

            // The rest of the warm-up goes over the queries again in turn, untimed.
            for i in vectors.len()..warmup {
                index
                    .search(vectors[i % vectors.len()], &params, &mut ctx)
                    .unwrap_or_else(|e| fail(&format!("searching: {e}")));
            }

            let r = Recalls::of(&dataset, &queries, &found, k);
            let scanned = scanned as f64 / vectors.len() as f64;
            let line = state(provenance(Json::new()))
                .str("label", args.str("label").unwrap_or(""))
                .str("index", path)
                .str("dataset", name)
                .int("n", dataset.n as u64)
                .int("k", k as u64)
                .int("nprobe", nprobe as u64)
                .int("rerank", rerank as u64)
                .num("reach", reach.map_or(f64::NAN, f64::from))
                .str("split", part)
                .int("queries", vectors.len() as u64)
                .str("mode", mode)
                .num("resident_share", share(&file, &resident))
                .num("recall", r.ids)
                .num("share_at_0_9", r.share)
                .bool("ties", true)
                .bool("truth_checked", dataset.dists.is_some())
                .num("recall1", r.levels[0])
                .num("recall10", r.levels[1])
                .num("recall100", r.levels[2])
                .num("recall10_p05", r.p05)
                .num("recall10_p10", r.p10)
                .num("scanned", scanned)
                .int("warmup", warmup.max(vectors.len()) as u64)
                .int("passes", passes as u64);
            let (line, timing) = match threads {
                None => {
                    let (mut rows_cached, mut hits) = (0, 0);
                    let stats = latency(passes, vectors.len(), |i| {
                        if let Some(rows) = &rows {
                            file.evict_range(rows.clone())
                                .unwrap_or_else(|e| fail(&format!("dropping the rows from the cache: {e}")));
                            rows_cached += file
                                .resident(rows.clone())
                                .unwrap_or_else(|e| fail(&format!("checking the cache: {e}")));
                        }

                        let start = Instant::now();
                        index
                            .search(vectors[i], &params, &mut ctx)
                            .unwrap_or_else(|e| fail(&format!("searching: {e}")));
                        let took = start.elapsed();
                        hits += ctx.trace().cache_hits;
                        took
                    });
                    let [p50, p90, p99, mean] = medians(&stats);
                    let line = read_paths(line, &[ctx.trace().read_path])
                        .num("cache_hits_per_query", hits as f64 / (passes * vectors.len()) as f64)
                        .int("threads", 1)
                        .num("p50_us", p50)
                        .num("p90_us", p90)
                        .num("p99_us", p99)
                        .num("mean_us", mean)
                        .nums("p50_us_passes", &stats.iter().map(|s| s[0]).collect::<Vec<_>>());
                    let line = match rows {
                        Some(_) => line.int("rows_cached_pages", rows_cached as u64),
                        None => line.null("rows_cached_pages"),
                    };
                    (line, format!("p50 {p50:.0} us, p90 {p90:.0} us, p99 {p99:.0} us"))
                }
                Some(t) => {
                    let make = || SearchCtx::new(limits);
                    let search = |ctx: &mut SearchCtx, i: usize| {
                        index
                            .search(vectors[i], &params, ctx)
                            .unwrap_or_else(|e| fail(&format!("searching: {e}")));
                    };
                    let time = Duration::from_secs_f64(seconds);
                    let mut paths = Vec::new();
                    let runs: Vec<f64> = (0..passes)
                        .map(|_| {
                            let (done, seconds, ctxs) = throughput(t, time, vectors.len(), make, search);
                            // Each pass has contexts of its own, so the line counts the last pass's.
                            paths = ctxs.iter().map(|c| c.trace().read_path).collect();
                            done as f64 / seconds
                        })
                        .collect();
                    let mut sorted = runs.clone();
                    sorted.sort_by(f64::total_cmp);
                    let qps = percentile(&sorted, 50.0);
                    let line = read_paths(line, &paths)
                        .null("cache_hits_per_query")
                        .int("threads", t as u64)
                        .num("seconds", seconds)
                        .num("qps", qps)
                        .nums("qps_runs", &runs);
                    (line, format!("{qps:.0} queries per second on {t} threads"))
                }
            };
            eprintln!(
                "warm: nprobe {nprobe:4} rerank {rerank:3}: recall {:.4}, {:.1}% of queries at 0.9, recall@10 {:.4} \
                 with ties, {timing}, {scanned:.0} vectors scanned",
                r.ids,
                r.share * 100.0,
                r.levels[1]
            );
            let line = id.fields(line).line();
            append(Path::new(results), &line).unwrap_or_else(|e| fail(&format!("writing {results}: {e}")));
        }
    }
}

/// The recall of one setting's hits.
struct Recalls {
    /// The mean recall@k for each k of `LEVELS`, or NaN where `--k` or the ground truth's width falls short of it.
    levels: [f64; 3],
    /// The 5th and 10th percentiles of the queries' recall@10, or NaN without it.
    p05: f64,
    p10: f64,
    /// The mean recall at `--k` by ids alone, and the share of queries where it is at least 0.9.
    ids: f64,
    share: f64,
}

impl Recalls {
    /// The recall of `found`, where `found[i]` holds the ids that query `queries[i]` found, nearest first.
    fn of(dataset: &Dataset, queries: &[usize], found: &[Vec<u64>], k: usize) -> Recalls {
        let at = |level: usize| -> Vec<f64> {
            if level > k || level > dataset.truth.width() {
                return Vec::new();
            }

            let judge = |(&q, ids): (&usize, &Vec<u64>)| {
                dataset
                    .recall(q, ids, level)
                    .unwrap_or_else(|e| fail(&format!("judging the hits of query {q}: {e}")))
            };
            queries.iter().zip(found).map(judge).collect()
        };
        let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
        let levels = LEVELS.map(at);
        let mut ten = levels[1].clone();
        ten.sort_by(f64::total_cmp);
        let ids: Vec<f64> = queries
            .iter()
            .zip(found)
            .map(|(&q, f)| recall(f, dataset.truth.row(q), k))
            .collect();
        Recalls {
            levels: levels.map(|v| mean(&v)),
            p05: percentile(&ten, 5.0),
            p10: percentile(&ten, 10.0),
            ids: mean(&ids),
            share: ids.iter().filter(|&&r| r >= 0.9).count() as f64 / ids.len() as f64,
        }
    }
}

/// Times `passes` passes over `count` queries at one thread, where `search(i)` runs query i and returns how long its
/// search took. Each pass gives its p50, p90, p99 and mean latency in microseconds.
fn latency(passes: usize, count: usize, mut search: impl FnMut(usize) -> Duration) -> Vec<[f64; 4]> {
    let mut stats = Vec::with_capacity(passes);

    for _ in 0..passes {
        let mut times: Vec<f64> = (0..count).map(|i| search(i).as_secs_f64() * 1e6).collect();
        times.sort_by(f64::total_cmp);
        let mean = times.iter().sum::<f64>() / count as f64;
        stats.push([
            percentile(&times, 50.0),
            percentile(&times, 90.0),
            percentile(&times, 99.0),
            mean,
        ]);
    }

    stats
}

/// The median of each of the four values over the passes.
fn medians(stats: &[[f64; 4]]) -> [f64; 4] {
    std::array::from_fn(|i| {
        let mut column: Vec<f64> = stats.iter().map(|s| s[i]).collect();
        column.sort_by(f64::total_cmp);
        percentile(&column, 50.0)
    })
}

/// Runs `threads` threads for at least `time`, each with its own context from `make`. Each calls `search(ctx, i)` once
/// before the timer starts, then over and over, for query positions i that start at its own place among `count` and
/// count up. Returns how many searches finished in the timed part, the seconds from its start to the last one, and
/// each thread's context.
fn throughput<C: Send>(
    threads: usize,
    time: Duration,
    count: usize,
    make: impl Fn() -> C + Sync,
    search: impl Fn(&mut C, usize) + Sync,
) -> (u64, f64, Vec<C>) {
    let barrier = Barrier::new(threads + 1);

    std::thread::scope(|scope| {
        let workers: Vec<_> = (0..threads)
            .map(|t| {
                let (barrier, make, search) = (&barrier, &make, &search);
                scope.spawn(move || {
                    interactive();
                    let (mut ctx, first) = (make(), t * count / threads);
                    // A first search makes the context's buffers before the timer.
                    search(&mut ctx, first);
                    barrier.wait();
                    let (start, mut done) = (Instant::now(), 0);

                    while start.elapsed() < time {
                        search(&mut ctx, (first + done) % count);
                        done += 1;
                    }

                    (done as u64, ctx)
                })
            })
            .collect();
        barrier.wait();
        let start = Instant::now();
        let (mut done, mut ctxs) = (0, Vec::with_capacity(threads));

        for w in workers {
            let (n, ctx) = w.join().unwrap_or_else(|_| fail("a search thread panicked"));
            done += n;
            ctxs.push(ctx);
        }

        (done, start.elapsed().as_secs_f64(), ctxs)
    })
}

/// The share of the pages of `ranges` of `file` that are in the cache, or NaN where that can't be measured, eg. on a
/// system without a way to ask.
fn share(file: &Mapped, ranges: &[Range<usize>]) -> f64 {
    let page = page_size();
    let (mut cached, mut pages) = (0, 0);

    for range in ranges.iter().filter(|r| r.start < r.end.min(file.len())) {
        pages += range.end.min(file.len()).div_ceil(page) - range.start / page;

        match file.resident(range.clone()) {
            Ok(n) => cached += n,
            Err(_) => return f64::NAN,
        }
    }

    cached as f64 / pages as f64
}

/// Reads the whole file at `path` with plain reads, which leaves it in the cache.
fn read_through(path: &Path) {
    let mut file = File::open(path).unwrap_or_else(|e| fail(&format!("opening {}: {e}", path.display())));
    let mut buf = vec![0u8; 1 << 20];

    loop {
        let read = file
            .read(&mut buf)
            .unwrap_or_else(|e| fail(&format!("reading {}: {e}", path.display())));

        if read == 0 {
            return;
        }
    }
}

/// Starts a warm-up and waits for it, or after a minute marks the index warm anyway, since the hits are the same.
fn warm(index: &Index) {
    index
        .warm_up()
        .unwrap_or_else(|e| fail(&format!("warming the index: {e}")));
    let start = Instant::now();

    while !index.is_warm() && start.elapsed() < Duration::from_secs(60) {
        std::thread::sleep(Duration::from_millis(20));
    }

    if !index.is_warm() {
        eprintln!("warm: the warm-up didn't finish within a minute, so the first searches fault their pages in");
        index.set_warm(true);
    }
}

#[cfg(test)]
#[path = "../../tests/unit/bin/warm.rs"]
mod tests;
