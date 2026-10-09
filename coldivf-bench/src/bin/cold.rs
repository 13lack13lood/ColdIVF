//! Cold trials. Each trial is a fresh process. It maps the index file once, empties the file's cache through that
//! mapping, and waits until nothing reads the file back. Then it times opening the index, preparing it and searching
//! one query, and prints one JSON line. With `--per-trial n` it then searches n - 1 more queries without emptying the
//! cache, to show the queries after the first. The parent runs the trials one after another, adds where the numbers
//! came from, and appends the lines to the results file.
//!
//! cold --index PATH --dataset NAME [--n N] [--nprobe 128] [--rerank 29] [--reach R] [--k 10] [--trials 100]
//!      [--first 0] [--variant idle,prewarm] [--mode wave,pool,portable,mapped,blocking] [--gap-ms 0,30]
//!      [--per-trial 1] [--results results/cold.jsonl] [--label TEXT]
//!
//! The variants set the CPU's state at the start of the timer. A thread that wakes from idle runs its first few
//! milliseconds on an efficiency core, and a performance core reaches full clock only after ~100 ms of work. So
//! `idle` sleeps first, and `prewarm` spins for 150 ms first. `open_p_share` and `open_ghz` show the state the open
//! ran in. The modes are the read paths:
//! - `wave`: the backend that `Index::open` picks, eg. `F_RDADVISE` read waves on macOS.
//! - `pool` and `portable`: the thread-pool and the one-read-at-a-time backends.
//! - `mapped`: page faults through the memory map, the lazy-map control. It skips `prepare`, which would start reading
//!   the cluster tables, so they fault in too.
//! - `blocking`: one `read()` of the whole file through a handle opened before the timer, then a warm search.
//!
//! `--mode`, `--variant` and `--gap-ms` each take a list, and `--trials` counts the trials of each combination. The
//! trials take the combinations in turn, so a change in the machine's load during a run reaches all of them alike.
//!
//! The stage fields come from the library's trace. In `wave`, `pool` and `portable` the route is in `route_us`.
//! `cluster_wave_us` is `submit_us` plus `wave_us`, the time the cluster reads held up the search. Compare it across
//! backends rather than `wave_us`, since the pool waits for its reads in `submit_us`. A mapped search counts the route
//! and the clusters' page faults in `scan_us`, and the rows' faults in `rerank_us`. It reads nothing in waves, so its
//! `cluster_bytes`, `row_bytes` and their ranges are 0, and `disk_bytes` holds what it read. A warm search, as in
//! `blocking`, scans the clusters in place, so it counts the route in `scan_us` and reads only rows.
//!
//! `ctx_us` is the time to make the empty search context, before the timer. Its buffers for the index are made inside
//! the timer. `prepare` makes the scratch and the row buffers and readies the read backend, eg. its io_uring ring on
//! Linux, but the cluster buffer is sized from the cluster tables, which aren't loaded yet, so the first search makes
//! that one. It records the buffers it makes in `buffers_us`. In `mapped`, which has no `prepare` and no read buffers,
//! that is the scratch.
//!
//! `backend` and `read_paths` say what read the waves and how it was set up, eg. on Linux whether the context got an
//! io_uring ring and reads with `O_DIRECT`. A mapped search reads no waves, so it leaves them empty. `cache_hits` is
//! the search's ranges that came from the backend's own copies of pages rather than from the file.
//!
//! The gap is the time right before the search, where an app runs its embedding model. The variant sleeps or spins
//! through it, and no time or counter includes it. With a gap longer than the read of the cluster tables that
//! `prepare` starts, `tables_us` is only their loading, and `post_embedding_us` is the time after the embedding.
//! Without a gap both also hold that read. `post_embedding_us` is the sum of `tables_us`, `route_us`,
//! `cluster_wave_us`, `scan_us`, `rows_us` and `rerank_us`, so it leaves out `buffers_us`.
//!
//! Each line records `pages_resident_at_start`, the cached pages of the header and the cluster tables right before the
//! timer, and the parent's summary leaves out the trials with any. The counters, eg. `energy_nj` and `disk_bytes`,
//! cover the open and the first search on the first line, and one search on each later line. Each line also records
//! `syscall_open_us`, the cheapest of three plain opens of the file after the timed part, since on a Mac with an
//! endpoint-security agent every open can cost hundreds of microseconds.

use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::process::Command;
use std::time::{Duration, Instant};

use coldivf::{Backend, Index, Limits, SearchCtx, SearchParams, Trace};
use coldivf_bench::args::{Args, fail};
use coldivf_bench::coldcache::{Delta, Mapped, Usage, make_cold};
use coldivf_bench::data::Dataset;
use coldivf_bench::metrics::{IndexFile, Json, append, interactive, percentile, provenance, read_paths, recall, state};

const MODES: [&str; 5] = ["wave", "pool", "portable", "mapped", "blocking"];
const VARIANTS: [&str; 2] = ["idle", "prewarm"];

/// How much of the file `Index::open` reads first.
const FIRST_READ: usize = 16384;

fn main() {
    let args = Args::parse();

    if args.flag("one") {
        trial(&args);
        return;
    }

    let trials = args.get("trials", 100);
    let results = args.str("results").unwrap_or("results/cold.jsonl").to_string();
    let exe = std::env::current_exe().unwrap_or_else(|e| fail(&format!("finding this program: {e}")));
    let tail = provenance(Json::new()).line();
    let (modes, variants) = (
        args.list("mode", &["wave".to_string()]),
        args.list("variant", &["idle".to_string()]),
    );
    let gaps = args.list("gap-ms", &[0u64]);
    let mut combos = Vec::new();

    for mode in &modes {
        for variant in &variants {
            for &gap in &gaps {
                combos.push((mode.as_str(), variant.as_str(), gap));
            }
        }
    }

    if let Some(&(mode, variant, _)) = combos
        .iter()
        .find(|(m, v, _)| !MODES.contains(m) || !VARIANTS.contains(v))
    {
        fail(&format!("unknown mode {mode} or variant {variant}"));
    }

    let mut summaries: Vec<Summary> = combos.iter().map(|_| Summary::default()).collect();

    for i in 0..trials * combos.len() {
        let ((mode, variant, gap), round) = (combos[i % combos.len()], i / combos.len());
        let summary = &mut summaries[i % combos.len()];
        let out = Command::new(&exe)
            .args(args.forward())
            .args([
                "--one",
                "--trial",
                &round.to_string(),
                "--mode",
                mode,
                "--variant",
                variant,
            ])
            .args(["--gap-ms", &gap.to_string()])
            .output()
            .unwrap_or_else(|e| fail(&format!("starting a trial: {e}")));
        if !out.status.success() {
            eprintln!(
                "cold: trial {round} ({mode}, {variant}, gap {gap} ms) failed: {}",
                String::from_utf8_lossy(&out.stderr).trim()
            );
            summary.failed += 1;
            continue;
        }
        for line in String::from_utf8_lossy(&out.stdout)
            .lines()
            .filter(|l| l.starts_with('{'))
        {
            if field(line, "seq") == Some(0.0) {
                if field(line, "pages_resident_at_start") == Some(0.0) {
                    summary.totals.extend(field(line, "total_us"));
                    summary.posts.extend(field(line, "post_embedding_us"));
                } else {
                    summary.cached += 1;
                }
            }
            append(Path::new(&results), &merge(line, &tail)).unwrap_or_else(|e| fail(&format!("writing: {e}")));
        }
        if (i + 1) % 10 == 0 {
            eprintln!("cold: {} of {} trials", i + 1, trials * combos.len());
        }
    }

    for ((mode, variant, gap), mut s) in combos.into_iter().zip(summaries) {
        s.totals.sort_by(f64::total_cmp);
        s.posts.sort_by(f64::total_cmp);
        eprintln!(
            "cold: {mode}, {variant}, gap {gap} ms, {} trials ({} failed, {} left out for starting with cached pages): \
             total p50 {:.0} us, p99 {:.0} us; after the embedding p50 {:.0} us, p99 {:.0} us",
            s.totals.len(),
            s.failed,
            s.cached,
            percentile(&s.totals, 50.0),
            percentile(&s.totals, 99.0),
            percentile(&s.posts, 50.0),
            percentile(&s.posts, 99.0)
        );
    }
}

/// The first queries' times of one mode, variant and gap, and the trials that failed or started with cached pages.
#[derive(Default)]
struct Summary {
    totals: Vec<f64>,
    posts: Vec<f64>,
    failed: usize,
    cached: usize,
}

/// One search of a trial.
struct Run {
    search_us: f64,
    ids: Vec<u64>,
    trace: Trace,
    used: Delta,
}

/// One trial, in its own process.
fn trial(args: &Args) {
    // The trial runs at the class of work a user waits on, whatever started it.
    interactive();
    let name = args.need("dataset");
    let n = args
        .str("n")
        .map(|v| v.parse().unwrap_or_else(|_| fail("--n must be a number")));
    let dataset = Dataset::open(name, n).unwrap_or_else(|e| fail(&format!("opening {name}: {e}")));
    let path = Path::new(args.need("index"));
    let (k, nprobe, rerank) = (args.get("k", 10), args.get("nprobe", 128), args.get("rerank", 29));
    let reach = args
        .str("reach")
        .map(|v| v.parse().unwrap_or_else(|_| fail("--reach must be a number")));
    let params = SearchParams {
        k,
        nprobe,
        rerank,
        reach,
    };
    let (mode, variant) = (
        args.str("mode").unwrap_or("wave"),
        args.str("variant").unwrap_or("idle"),
    );
    let gap = Duration::from_millis(args.get("gap-ms", 0));

    if !MODES.contains(&mode) || !VARIANTS.contains(&variant) {
        fail(&format!(
            "--mode must be wave, pool, portable, mapped or blocking, and --variant idle or prewarm, not {mode} and \
             {variant}"
        ));
    }

    let (index_no, per) = (args.get::<usize>("trial", 0), args.get("per-trial", 1).max(1));
    let first = args.get::<usize>("first", 0);
    let nq = dataset.queries.len();
    let queries: Vec<usize> = (0..per).map(|j| (first + index_no * per + j) % nq).collect();
    let vectors: Vec<Vec<f32>> = queries.iter().map(|&q| dataset.queries.row(q).to_vec()).collect();

    // The empty context is made outside the timer, as an app makes it once. Its buffers for the index are made inside
    // the timer, by `prepare` and the first search.
    let made = Instant::now();
    let mut ctx = SearchCtx::new(Limits {
        max_k: k,
        max_nprobe: nprobe,
        max_rerank: rerank.max(k),
    });
    let ctx_us = us(made.elapsed());

    // The header is read and the file mapped before the eviction, and every check after it goes through that one
    // mapping. The blocking control's handle and buffer are made here too. So the control pays for one open, the
    // index's own, and its buffer's pages, filled now, don't count in its footprint.
    let id = IndexFile::read(path).unwrap_or_else(|e| fail(&format!("reading {}: {e}", path.display())));
    let file = Mapped::open(path).unwrap_or_else(|e| fail(&format!("mapping {}: {e}", path.display())));
    let mut plain = (mode == "blocking")
        .then(|| File::open(path).unwrap_or_else(|e| fail(&format!("opening {}: {e}", path.display()))));
    let mut buf = vec![1u8; if plain.is_some() { 1 << 20 } else { 0 }];
    make_cold(&[&file], Duration::from_millis(500))
        .unwrap_or_else(|e| fail(&format!("emptying the cache of {}: {e}", path.display())));
    pause(
        variant,
        Duration::from_millis(if variant == "prewarm" { 150 } else { 200 }),
    );
    let cached = [0..FIRST_READ, id.tables.clone()]
        .into_iter()
        .map(|range| {
            file.resident(range)
                .unwrap_or_else(|e| fail(&format!("checking the cache: {e}")))
        })
        .sum::<usize>();

    let before = Usage::settled();
    let start = Instant::now();
    let index = match mode {
        "pool" => Index::open_with(path, Backend::Pool),
        "portable" => Index::open_with(path, Backend::Portable),
        _ => Index::open(path),
    }
    .unwrap_or_else(|e| fail(&format!("opening {}: {e}", path.display())));
    let open_us = us(start.elapsed());
    let opened = Usage::now().since(&before);
    let read_us = plain.as_mut().map_or(0.0, |f| read_all(f, &mut buf, &index));
    let prepared = Instant::now();

    // The lazy-map control skips `prepare`, which would start reading the cluster tables, so they fault in too.
    if mode != "mapped" {
        index
            .prepare(&mut ctx)
            .unwrap_or_else(|e| fail(&format!("preparing: {e}")));
    }

    let prepare_us = us(prepared.elapsed());
    // A gap splits the counters' window in two, and `setup` holds the part before it.
    let (paused, mut setup, mut base) = (Instant::now(), Delta::default(), before);

    if !gap.is_zero() {
        let now = Usage::settled();
        setup = now.since(&before);
        pause(variant, gap);
        base = Usage::settled();
    }

    let gap_us = if gap.is_zero() { 0.0 } else { us(paused.elapsed()) };
    let mut runs = Vec::with_capacity(per);

    for vector in &vectors {
        let searched = Instant::now();
        let hits = match mode {
            "mapped" => index.search_mapped(vector, &params, &mut ctx),
            _ => index.search(vector, &params, &mut ctx),
        }
        .unwrap_or_else(|e| fail(&format!("searching: {e}")));
        let search_us = us(searched.elapsed());
        let ids: Vec<u64> = hits.iter().map(|h| h.id).collect();
        let trace = ctx.trace();
        let after = Usage::settled();
        runs.push(Run {
            search_us,
            ids,
            trace,
            used: after.since(&base) + std::mem::take(&mut setup),
        });
        base = after;
    }

    // What one plain open of the file costs this process, eg. with an endpoint-security agent screening every open.
    // It is measured after the timed part, so it can't warm anything the timed open needs.
    let opens: Vec<f64> = (0..3)
        .map(|_| {
            let start = Instant::now();
            drop(File::open(path).unwrap_or_else(|e| fail(&format!("opening {}: {e}", path.display()))));
            us(start.elapsed())
        })
        .collect();
    let syscall = opens.iter().copied().fold(f64::INFINITY, f64::min);
    let tail = id.fields(state(Json::new())).line();

    for (seq, (&q, run)) in queries.iter().zip(&runs).enumerate() {
        let (t, used) = (run.trace, run.used);
        let post = us(t.tables + t.route + t.submit + t.wave + t.scan + t.rows + t.rerank);
        let total_us = match seq {
            0 => open_us + read_us + prepare_us + run.search_us,
            _ => run.search_us,
        };
        let line = Json::new()
            .str("label", args.str("label").unwrap_or(""))
            .str("index", &path.display().to_string())
            .str("dataset", name)
            .int("n", dataset.n as u64)
            .str("mode", mode)
            .str("variant", variant)
            .int("gap_ms", gap.as_millis() as u64)
            .int("trial", index_no as u64)
            .int("seq", seq as u64)
            .int("query", q as u64)
            .int("k", k as u64)
            .int("nprobe", nprobe as u64)
            .int("rerank", rerank as u64)
            .num("reach", reach.map_or(f64::NAN, f64::from))
            .num("recall", recall(&run.ids, dataset.truth.row(q), k))
            .int("pages_resident_at_start", cached as u64)
            .num("ctx_us", ctx_us)
            .num("open_us", if seq == 0 { open_us } else { 0.0 })
            .num("read_us", if seq == 0 { read_us } else { 0.0 })
            .num("prepare_us", if seq == 0 { prepare_us } else { 0.0 })
            .num("gap_us", if seq == 0 { gap_us } else { 0.0 })
            .num("search_us", run.search_us)
            .num("total_us", total_us)
            .num("tables_us", us(t.tables))
            .num("buffers_us", us(t.buffers))
            .num("route_us", us(t.route))
            .num("submit_us", us(t.submit))
            .num("wave_us", us(t.wave))
            .num("cluster_wave_us", us(t.submit + t.wave))
            .num("scan_us", us(t.scan))
            .num("rows_us", us(t.rows))
            .num("rerank_us", us(t.rerank))
            .num("post_embedding_us", post)
            .int("cluster_bytes", t.cluster_bytes as u64)
            .int("cluster_ranges", t.cluster_ranges as u64)
            .int("row_bytes", t.row_bytes as u64)
            .int("row_ranges", t.row_ranges as u64)
            .int("cache_hits", t.cache_hits as u64);
        let line = read_paths(line, &[t.read_path])
            .num("footprint_delta", used.footprint as f64)
            .num("p_share", used.p_share())
            .num("ghz", used.ghz())
            .num("open_p_share", if seq == 0 { opened.p_share() } else { f64::NAN })
            .num("open_ghz", if seq == 0 { opened.ghz() } else { f64::NAN })
            .int("energy_nj", used.energy_nj)
            .int("pageins", used.pageins)
            .int("disk_bytes", used.bytes_read)
            .num("syscall_open_us", syscall);
        println!("{}", merge(&line.line(), &tail));
    }
}

/// Reads the whole index file with plain reads through `file`, which fills the cache, then marks the index warm, as an
/// app that loads its index at open would.
fn read_all(file: &mut File, buf: &mut [u8], index: &Index) -> f64 {
    let start = Instant::now();

    while file.read(buf).unwrap_or_else(|e| fail(&format!("reading: {e}"))) > 0 {}

    index.set_warm(true);
    us(start.elapsed())
}

/// Waits `time` the way `variant` sets the CPU's state. `idle` sleeps, so the next step starts on a thread that wakes
/// from idle, and `prewarm` spins, so it starts on a core at full clock.
fn pause(variant: &str, time: Duration) {
    if variant == "prewarm" {
        spin(time);
    } else {
        std::thread::sleep(time);
    }
}

/// Keeps this core busy for `time`, so it runs at full clock when the timer starts.
fn spin(time: Duration) {
    let start = Instant::now();
    let mut x = 0u64;

    while start.elapsed() < time {
        x = std::hint::black_box(x.wrapping_mul(6364136223846793005).wrapping_add(1));
    }
}

/// `time` in microseconds.
fn us(time: Duration) -> f64 {
    time.as_secs_f64() * 1e6
}

/// The number in `line`'s field `key`, a JSON line this program wrote.
fn field(line: &str, key: &str) -> Option<f64> {
    let at = line.find(&format!("\"{key}\":"))? + key.len() + 3;
    let end = line[at..].find([',', '}']).map_or(line.len(), |e| at + e);
    line[at..end].parse().ok()
}

/// `line` with the fields of `tail`, another JSON object, added at its end.
fn merge(line: &str, tail: &str) -> String {
    let inner = tail.trim().trim_start_matches('{').trim_end_matches('}');
    format!("{},{inner}}}", line.trim_end().trim_end_matches('}'))
}
