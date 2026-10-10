//! Cold trials. Each trial is a fresh process. It maps the index's files once, empties their cache through those
//! mappings, and waits until nothing reads them back. Then it times opening the index, preparing it and searching one
//! query, and prints one JSON line. With `--per-trial n` it then searches n - 1 more queries without emptying the
//! cache, to show the queries after the first. The parent runs the trials one after another, adds where the numbers
//! came from, and appends the lines to the results file.
//!
//! cold --index PATH --dataset NAME [--n N] [--nprobe 128] [--rerank 29] [--reach R] [--k 10] [--trials 100]
//!      [--first 0] [--variant idle,prewarm] [--mode wave,pool,portable,mapped,blocking] [--gap-ms 0,30]
//!      [--resident 0,0.5] [--per-trial 1] [--pool 1,3,18] [--seed 1] [--c1] [--dummy PATH] [--warmup-query Q]
//!      [--derived DIR] [--results results/cold.jsonl] [--label TEXT]
//! cold --manifest FILE --dataset NAME [the same settings, in mode wave and without --c1 or --dummy]
//!
//! The variants set the CPU's state at the start of the timer. A thread that wakes from idle runs its first few
//! milliseconds on an efficiency core, and a performance core reaches full clock only after ~100 ms of work. So `idle`
//! sleeps first, and `prewarm` spins for 150 ms first. `open_p_share` and `open_ghz` show the state the open ran in.
//! The modes are the read paths:
//! - `wave`: the backend that `Index::open` picks, eg. `F_RDADVISE` read waves on macOS.
//! - `pool` and `portable`: the thread-pool and the one-read-at-a-time backends.
//! - `mapped`: page faults through the memory map, the lazy-map control. It skips `prepare`, which would start reading
//!   the cluster tables, so they fault in too.
//! - `blocking`: one `read()` of the whole file through a handle opened before the timer, then a warm search.
//!
//! `--mode`, `--variant`, `--gap-ms` and `--resident` each take a list, and `--trials` counts the trials of each
//! combination. The trials run in rounds of one trial of every combination, so a change in the machine's load during a
//! run reaches all of them alike. The combination that runs right before a trial can change its time, eg. a trial
//! right after a blocking read of the whole file loads its cluster tables faster. So the rounds come in blocks whose
//! orders form a Williams design. Over a block every combination runs at each place in its round equally often, and
//! right after each other combination within a round equally often. A block has as many rounds as there are
//! combinations, or twice as many for an odd count, and its rounds run in a random order drawn from `--seed`. A line
//! records its place in its round in `order`, the combination that ran right before it in `predecessor`, and the seed
//! in `order_seed`. Every trial searches queries of its own, the next `--per-trial` positions from `--first`, wrapping
//! around the dataset's queries, since a trial that reads the pages the trial before it just read can come out faster,
//! although none of them shows as cached. `--pool` lists the query positions that the trials take in turn instead of
//! every query, eg. to time a few queries many times, so each comes back once the pool wraps around.
//!
//! `--resident` takes shares of the clusters, eg. `0,0.25,0.5,0.75,1`, for trials of one index in fresh processes, in
//! any mode but `blocking`. After it empties the cache, a trial reads that share of the clusters into it, both their
//! 1-bit codes and their rows, picked at random from `--seed` and the trial's number. The header, the rotation and the
//! cluster tables stay out of the cache. So the trial shows what a cold search costs when part of the index is still
//! cached, eg. after memory pressure took the rest. `resident_clusters` is how many clusters it read, `pages_loaded`
//! the pages they take, `pages_loaded_cached` how many of those were still cached right before the timer, and
//! `load_us` how long the reads took. Its `pages_resident_at_start` then counts only the other cached pages, so the
//! summary still leaves out a trial with any.
//!
//! With `--c1` the trials measure the disk-resident steady state (C1) instead of a fresh process's first query (C0).
//! One process per combination opens the index, searches one query that no trial searches, which loads the cluster
//! tables, and keeps them. It then serves that combination's trials, one per line of its input, eg.
//! `--trial 3 --queries 7,8`, until its input ends. Before each trial it drops the codes, the rows and the ids from the
//! cache, and the timer covers the search alone. A C1 trial's `pages_resident_at_start` counts the pages of those
//! sections still cached when its timer starts, and the parent leaves out the trials with more than 0.5% of them
//! cached. Its `footprint_before` is the process's footprint right before the search, which holds the state the
//! process keeps, and the line it prints once it is ready gives its footprint after the warm-up.
//!
//! `--dummy PATH` names a small index of the same settings, eg. of 50K vectors. A C0 trial opens it and searches it
//! once, with a context of its own, before it empties the cache, as the other systems search a dummy index before their
//! timers. So the timed search finds its code and its threads warm. `--warmup-query` is the query of that search and of
//! the search a C1 process warms its index with. The parent picks the first position after every trial's queries, and a
//! trial run on its own takes the dataset's last query.
//!
//! After its queries, every trial searches its first query again, with what that search read still cached.
//! `resident_rerun_us` is the time of that search, so the cold time minus it is about the wait on reads.
//! `rerun_disk_bytes` shows what the rerun still read from the disk, which is 0 when everything was cached.
//!
//! `--manifest FILE` names K index files that a trial searches together as collections. Each line of the file holds a
//! file's path, the id of its first vector in their union and its vector count, which must be the count in the file's
//! header. The files must cover the union's ids from 0 on, one after another, and the hits count with their ids in the
//! union, against the dataset's truth for that many vectors. A trial drops all K files from the cache, then times
//! `Collections::open`, which starts every header read before it waits for any, plus the first search, which loads
//! every file's cluster tables. `collections` is K.
//!
//! `header_bytes` and `table_bytes` are what the open and the loading of the cluster tables read of the index files, in
//! whole pages. The open reads each file's first 16 KiB, which hold its header and, up to ~2,800 dimensions, its
//! rotation, and it reads the rotation again where the rotation runs past them. The cluster tables are the clusters'
//! 4-bit codes and the extents. The codes follow the centers of the clusters' groups in the centers section, which a
//! search doesn't read. So a cold search's `disk_bytes` is about the sum of the two plus `cluster_bytes`, `row_bytes`
//! and the change log, if there is one.
//!
//! `recall` is the recall at `--k` by ids alone. `recall1`, `recall10` and `recall100` count ties, as the warm harness
//! does. A hit counts if its id is among the true k nearest, or if it is at most as far from the query as the true k-th
//! neighbor, with both distances measured from the base. The truth is the dataset's own for its whole base, and
//! otherwise the one for that many vectors in the folder of derived data. `--derived DIR` takes the truth for that many
//! vectors from `DIR`, even for the whole base, eg. one with a distances file. `truth_checked` says whether a distances
//! file checked that the truth belongs to the base. `ids` holds the hits' ids, nearest first.
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
//! that is the scratch. Collections have no `prepare` either, so their first search makes every buffer.
//!
//! `backend` and `read_paths` say what read the waves and how it was set up, eg. on Linux whether the context got an
//! io_uring ring and reads with `O_DIRECT`. A mapped search reads no waves and a search of collections doesn't say, so
//! they leave them empty. `cache_hits` is the search's ranges that came from the backend's own copies of pages rather
//! than from the file.
//!
//! The gap is the time from the query text to its vector, where an app runs its embedding model. The text arrives as
//! the timer starts, and the vector arrives `--gap-ms` later. The open and `prepare` run at once, and the search starts
//! once both the vector is there and `prepare` is done. A trial waits for the vector as the other systems' harness
//! does. `idle` sleeps until 5 ms before the vector, since a sleep can end late, and spins the rest, and `prewarm`
//! spins all of it. `gap_us` is the time the search waited for the vector, `late_us` how long after the vector's
//! arrival the search started, and `from_vector_us` the time from the vector's arrival to the hits, which is what an
//! app waits for. With a gap longer than the read of the cluster tables that `prepare` starts, `tables_us` is only
//! their loading, and `post_embedding_us` is the time after the embedding. Without a gap both also hold that read.
//! `post_embedding_us` is the sum of `tables_us`, `route_us`, `cluster_wave_us`, `scan_us`, `rows_us` and `rerank_us`,
//! so it leaves out `buffers_us`.
//!
//! Each line records `pages_resident_at_start`, the cached pages right before the timer, out of `pages_checked`. In a
//! C0 trial those are every page of each index file and of the files beside it that a reader may read, eg. its change
//! log `.delta`, and the parent's summary leaves out the trials with any. The counters, eg. `energy_nj` and
//! `disk_bytes`, cover the open, the wait for the vector and the first search on the first line, and one search on
//! each later line. `footprint_before` is the process's footprint right before them, and `peak_footprint` its highest
//! footprint by their end. Each line also records `syscall_open_us`, the cheapest of three plain opens of the file
//! after the timed part, since on a Mac with an endpoint-security agent every open can cost hundreds of microseconds.

use std::fs::File;
use std::io::{BufRead, BufReader, Error, ErrorKind, Read, Seek, SeekFrom, Write};
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::time::{Duration, Instant, SystemTime};

use coldivf::{Backend, Collections, Hit, Index, Limits, SearchCtx, SearchParams, Trace, data_file};
use coldivf_bench::args::{Args, fail};
use coldivf_bench::coldcache::{Delta, Mapped, Usage, make_cold, page_size};
use coldivf_bench::data::{Dataset, data_root, derived_root};
use coldivf_bench::metrics::{
    IndexFile, Json, append, interactive, percentile, provenance, read_paths, recall, state, utc,
};

const MODES: [&str; 5] = ["wave", "pool", "portable", "mapped", "blocking"];
const VARIANTS: [&str; 2] = ["idle", "prewarm"];

/// The k of each recall@k with ties that a line reports, where `--k` and the truth's width allow it.
const LEVELS: [usize; 3] = [1, 10, 100];

/// How long before the vector arrives an idle wait for it stops sleeping and spins, since a sleep can end late.
const SPIN_TAIL: Duration = Duration::from_millis(5);

/// How long a trial watches its files after dropping them, for anything that reads them back.
const COLD_WAIT: Duration = Duration::from_millis(500);

/// The share of a C1 trial's bulk pages that may still be cached when its timer starts, as the benchmark driver
/// allows.
const C1_CACHED: f64 = 0.005;

/// How much of an index file its open reads first, the header and, up to ~2,800 dimensions, the rotation.
const FIRST_READ: usize = 16384;

fn main() {
    let args = Args::parse();

    match (args.flag("one"), args.flag("c1")) {
        (true, true) => serve(&args),
        (true, false) => trial(&args),
        (false, _) => run(&args),
    }
}

/// Runs every trial of every combination and appends their lines to the results file.
fn run(args: &Args) {
    if args.str("queries").is_some() || args.str("trial").is_some() {
        fail("--queries and --trial name the queries of one trial, so they go with --one");
    }

    let c1 = args.flag("c1");
    let (trials, seed) = (args.get("trials", 100usize), args.get("seed", 1u64));
    let results = args.str("results").unwrap_or("results/cold.jsonl").to_string();
    let exe = std::env::current_exe().unwrap_or_else(|e| fail(&format!("finding this program: {e}")));
    let tail = provenance(Json::new()).line();
    let combos = combos(args);
    let (dataset, _, files) = sources(args);
    // A file that holds other vectors than the run says fails here, before any trial.
    headers(&files);
    let nq = dataset.queries.len();
    let (first, per) = (args.get("first", 0usize), args.get("per-trial", 1usize).max(1));
    let pool = args.list::<usize>("pool", &[]);

    if pool.iter().any(|&q| q >= nq) {
        fail(&format!(
            "--pool must list positions among the {nq} queries of {}",
            dataset.name
        ));
    }

    // The warm-up searches a query that no trial searches, unless the trials take every query.
    let warmup = args.get("warmup-query", spare(&pool, first + trials * combos.len() * per, nq));
    let mut servers = Vec::new();

    if c1 {
        for combo in &combos {
            let server = Server::start(&exe, args, combo, warmup);
            servers.push(server.unwrap_or_else(|e| fail(&format!("starting C1 for {}: {e}", combo.label()))));
        }
    }

    let allowed = if c1 { C1_CACHED } else { 0.0 };
    let mut summaries: Vec<Summary> = combos.iter().map(|_| Summary::default()).collect();
    let (mut previous, total): (Option<String>, _) = (None, trials * combos.len());

    for (slot, &(round, order, c)) in schedule(trials, combos.len(), seed).iter().enumerate() {
        let (combo, queries) = (&combos[c], pooled(&pool, first, slot, per, nq));
        let lines = match servers.get_mut(c) {
            Some(server) => server.trial(round, &queries),
            None => fresh(&exe, args, combo, round, &queries, warmup),
        };
        let summary = &mut summaries[c];

        match lines {
            Ok(lines) => {
                for line in lines {
                    if field(&line, "seq") == Some(0.0) {
                        summary.add(&line, order, allowed);
                    }

                    let place = Json::new().int("order", order as u64).int("order_seed", seed);
                    let place = match &previous {
                        Some(label) => place.str("predecessor", label),
                        None => place.null("predecessor"),
                    };
                    let line = merge(&merge(&line, &place.line()), &tail);
                    append(Path::new(&results), &line).unwrap_or_else(|e| fail(&format!("writing: {e}")));
                }
            }
            Err(e) => {
                eprintln!("cold: trial {round} ({}) failed: {e}", combo.label());
                summary.failed += 1;
            }
        }

        previous = Some(combo.label());

        if (slot + 1) % 10 == 0 {
            eprintln!("cold: {} of {total} trials", slot + 1);
        }
    }

    // Dropping a C1 process's handle ends its input, and so the process.
    drop(servers);

    for (combo, summary) in combos.iter().zip(summaries) {
        summary.report(combo, combos.len() > 1);
    }
}

/// One mode, variant, gap and share of resident clusters of a run.
#[derive(Clone, Debug, PartialEq)]
struct Combo {
    mode: String,
    variant: String,
    /// The gap in milliseconds.
    gap: f64,
    /// The share of the clusters cached at the timer, when the run names one with `--resident`.
    resident: Option<f64>,
}

impl Combo {
    /// Its name in messages and in the `predecessor` of the next line, eg. "wave idle 0", or "wave idle 0 r0.25" with
    /// a quarter of the clusters cached.
    fn label(&self) -> String {
        let label = format!("{} {} {}", self.mode, self.variant, self.gap);

        match self.resident {
            Some(share) => format!("{label} r{share}"),
            None => label,
        }
    }

    /// The arguments that give a trial process this combination and the warm-up query.
    fn args(&self, warmup: usize) -> Vec<String> {
        let mut out = [
            "--mode",
            &self.mode,
            "--variant",
            &self.variant,
            "--gap-ms",
            &self.gap.to_string(),
            "--warmup-query",
            &warmup.to_string(),
        ]
        .map(str::to_string)
        .to_vec();

        if let Some(share) = self.resident {
            out.extend(["--resident".to_string(), share.to_string()]);
        }

        out
    }
}

/// Every combination of the modes, variants, gaps and resident shares that the arguments list, once each goes with
/// the rest of them.
fn combos(args: &Args) -> Vec<Combo> {
    let modes = args.list("mode", &["wave".to_string()]);
    let variants = args.list("variant", &["idle".to_string()]);
    let gaps = args.list("gap-ms", &[0.0f64]);
    let shares: Vec<Option<f64>> = match args.str("resident") {
        Some(_) => args.list::<f64>("resident", &[]).into_iter().map(Some).collect(),
        None => vec![None],
    };
    let mut out = Vec::new();

    for mode in &modes {
        for variant in &variants {
            for &gap in &gaps {
                for &resident in &shares {
                    check(args, mode, variant, gap);
                    check_resident(args, mode, resident);
                    out.push(Combo {
                        mode: mode.clone(),
                        variant: variant.clone(),
                        gap,
                        resident,
                    });
                }
            }
        }
    }

    out
}

/// Fails unless the mode, the variant and the gap in milliseconds go with the rest of the arguments.
fn check(args: &Args, mode: &str, variant: &str, gap: f64) {
    if !MODES.contains(&mode) || !VARIANTS.contains(&variant) {
        fail(&format!(
            "--mode must be wave, pool, portable, mapped or blocking, and --variant idle or prewarm, not {mode} and \
             {variant}"
        ));
    }

    if !(gap.is_finite() && gap >= 0.0) {
        fail("--gap-ms must be a number of milliseconds, at least 0");
    }

    if args.str("manifest").is_some() && (mode != "wave" || args.flag("c1") || args.str("dummy").is_some()) {
        fail("--manifest searches collections, which run in mode wave, without --c1 and without --dummy");
    }

    if args.flag("c1") && (mode == "blocking" || gap > 0.0 || args.str("dummy").is_some()) {
        fail("--c1 searches an index it keeps open, so it runs no blocking mode, no gap and no --dummy");
    }
}

/// Fails unless the share of resident clusters goes with the mode and the rest of the arguments. It is a share from 0
/// to 1, for C0 trials of one index, since a blocking read caches the whole file anyway.
fn check_resident(args: &Args, mode: &str, resident: Option<f64>) {
    let Some(share) = resident else {
        return;
    };

    if !(0.0..=1.0).contains(&share) {
        fail("--resident must list shares of the clusters, from 0 to 1");
    }

    if mode == "blocking" || args.flag("c1") || args.str("manifest").is_some() {
        fail("--resident caches clusters of one index for C0 trials, so it runs no blocking mode, --c1 or --manifest");
    }
}

/// The order of a run's trials, `trials` rounds of one trial of each of `combos` combinations, as each trial's round,
/// its place in the round and its combination. The rounds come in blocks of the orders that `balanced` gives, and the
/// rounds of each block run in a random order drawn from `seed`.
fn schedule(trials: usize, combos: usize, seed: u64) -> Vec<(usize, usize, usize)> {
    let (orders, mut rng) = (balanced(combos), Mix(seed));
    let (mut out, mut left) = (Vec::with_capacity(trials * combos), Vec::new());

    for round in 0..trials {
        if left.is_empty() {
            left = (0..orders.len()).collect();
            rng.shuffle(&mut left);
        }

        let order = &orders[left.pop().unwrap_or(0)];
        out.extend(order.iter().enumerate().map(|(place, &c)| (round, place, c)));
    }

    out
}

/// The orders of a block of rounds of `n` combinations, which form a Williams design. Over the block every combination
/// runs at each place equally often, and right after each other combination equally often. The first order runs 0, 1,
/// n - 1, 2, n - 2 and so on, so for an even n its steps from one combination to the next, 1, -2, 3, -4 and so on,
/// counted around the n combinations, all differ. Each other order starts one combination later than the one before,
/// and for an odd n every order also runs backwards.
fn balanced(n: usize) -> Vec<Vec<usize>> {
    let first: Vec<usize> = (0..n)
        .map(|j| match j {
            0 => 0,
            j if !j.is_multiple_of(2) => j.div_ceil(2),
            j => n - j / 2,
        })
        .collect();
    let mut orders: Vec<Vec<usize>> = (0..n).map(|i| first.iter().map(|&c| (c + i) % n).collect()).collect();

    if !n.is_multiple_of(2) {
        let backwards: Vec<Vec<usize>> = orders.iter().map(|o| o.iter().rev().copied().collect()).collect();
        orders.extend(backwards);
    }

    orders
}

/// A small generator of random numbers from a seed, SplitMix64.
struct Mix(u64);

impl Mix {
    fn draw(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let z = (self.0 ^ (self.0 >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        let z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Puts `items` in a random order.
    fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            let j = (self.draw() % (i as u64 + 1)) as usize;
            items.swap(i, j);
        }
    }
}

/// The queries of the trial at place `slot` of a run, `per` positions after those of the trial before it, from
/// `first` on and wrapping around the `nq` queries. So two trials share a query only in a run of more than `nq`.
fn trial_queries(first: usize, slot: usize, per: usize, nq: usize) -> Vec<usize> {
    (0..per).map(|j| (first + slot * per + j) % nq).collect()
}

/// The queries of the trial at place `slot`, as `trial_queries` gives them, but taken from `pool` in turn when it lists
/// any, instead of from all `nq` queries.
fn pooled(pool: &[usize], first: usize, slot: usize, per: usize, nq: usize) -> Vec<usize> {
    match pool.is_empty() {
        true => trial_queries(first, slot, per, nq),
        false => trial_queries(first, slot, per, pool.len())
            .into_iter()
            .map(|i| pool[i])
            .collect(),
    }
}

/// A query position that no trial searches, of `nq`. Without a pool it is position `next`, the first after the trials'
/// queries. With one it is the first position after the pool's highest that the pool doesn't hold.
fn spare(pool: &[usize], next: usize, nq: usize) -> usize {
    match pool.iter().max() {
        None => next % nq,
        Some(&top) => (1..=nq)
            .map(|j| (top + j) % nq)
            .find(|q| !pool.contains(q))
            .unwrap_or(0),
    }
}

/// The positions as a comma-separated list, eg. "7,8".
fn list(positions: &[usize]) -> String {
    positions.iter().map(usize::to_string).collect::<Vec<_>>().join(",")
}

/// Runs one C0 trial in a fresh process and returns its lines.
fn fresh(
    exe: &Path,
    args: &Args,
    combo: &Combo,
    round: usize,
    queries: &[usize],
    warmup: usize,
) -> Result<Vec<String>, String> {
    let out = Command::new(exe)
        .args(args.forward())
        .args(["--one", "--trial", &round.to_string(), "--queries", &list(queries)])
        .args(combo.args(warmup))
        .output()
        .map_err(|e| format!("starting it: {e}"))?;

    if !out.status.success() {
        return Err(String::from_utf8_lossy(&out.stderr).trim().to_string());
    }

    let lines = String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter(|l| l.starts_with('{'))
        .map(str::to_string)
        .collect();
    Ok(lines)
}

/// A C1 process that keeps the index open and serves the trials of one combination.
struct Server {
    child: Child,
    input: ChildStdin,
    output: BufReader<ChildStdout>,
}

impl Server {
    /// Starts the process and waits until it has warmed its index.
    fn start(exe: &Path, args: &Args, combo: &Combo, warmup: usize) -> Result<Server, String> {
        let mut child = Command::new(exe)
            .args(args.forward())
            .args(["--one", "--c1"])
            .args(combo.args(warmup))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .map_err(|e| format!("starting it: {e}"))?;
        let (Some(input), Some(output)) = (child.stdin.take(), child.stdout.take()) else {
            return Err("it has no pipes".to_string());
        };
        let mut server = Server {
            child,
            input,
            output: BufReader::new(output),
        };
        let ready = server.line()?;

        if field(&ready, "warmup_us").is_none() {
            return Err(format!("it said {ready} instead of that it is ready"));
        }

        Ok(server)
    }

    /// The next JSON line the process prints.
    fn line(&mut self) -> Result<String, String> {
        let mut line = String::new();

        loop {
            line.clear();

            match self.output.read_line(&mut line) {
                Ok(0) => return Err("the C1 process ended".to_string()),
                Ok(_) if line.starts_with('{') => return Ok(line.trim_end().to_string()),
                Ok(_) => {}
                Err(e) => return Err(format!("reading from the C1 process: {e}")),
            }
        }
    }

    /// Runs one trial and returns its lines, one for each query.
    fn trial(&mut self, round: usize, queries: &[usize]) -> Result<Vec<String>, String> {
        writeln!(self.input, "--trial {round} --queries {}", list(queries))
            .and_then(|()| self.input.flush())
            .map_err(|e| format!("writing to the C1 process: {e}"))?;
        queries.iter().map(|_| self.line()).collect()
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = writeln!(self.input, "stop").and_then(|()| self.input.flush());
        let _ = self.child.wait();
    }
}

/// The first queries' times of one combination, and the trials that failed or started with cached pages.
#[derive(Default)]
struct Summary {
    totals: Vec<f64>,
    posts: Vec<f64>,
    arrivals: Vec<f64>,
    /// The totals of the trials that ran first in their round, and of the others.
    firsts: Vec<f64>,
    others: Vec<f64>,
    failed: usize,
    cached: usize,
}

impl Summary {
    /// Adds the first query's line of a trial that ran at `order` in its round. A trial that started with more than
    /// `allowed` of its checked pages cached is only counted.
    fn add(&mut self, line: &str, order: usize, allowed: f64) {
        let cached = field(line, "pages_resident_at_start").unwrap_or(0.0);

        if cached > allowed * field(line, "pages_checked").unwrap_or(0.0) {
            self.cached += 1;
            return;
        }

        let total = field(line, "total_us");
        self.totals.extend(total);
        self.posts.extend(field(line, "post_embedding_us"));
        self.arrivals.extend(field(line, "from_vector_us"));

        match order {
            0 => self.firsts.extend(total),
            _ => self.others.extend(total),
        }
    }

    /// Prints the summary of `combo`, with the times by place in the round when a run has `several` combinations.
    fn report(mut self, combo: &Combo, several: bool) {
        for v in [
            &mut self.totals,
            &mut self.posts,
            &mut self.arrivals,
            &mut self.firsts,
            &mut self.others,
        ] {
            v.sort_by(f64::total_cmp);
        }

        let p = |v: &[f64], at: f64| percentile(v, at);
        let mut text = format!(
            "cold: {}, {} trials ({} failed, {} left out for starting with cached pages): total p50 {:.0} us, \
             p99 {:.0} us; after the embedding p50 {:.0} us, p99 {:.0} us",
            combo.label(),
            self.totals.len(),
            self.failed,
            self.cached,
            p(&self.totals, 50.0),
            p(&self.totals, 99.0),
            p(&self.posts, 50.0),
            p(&self.posts, 99.0)
        );

        if combo.gap > 0.0 {
            let (p50, p99) = (p(&self.arrivals, 50.0), p(&self.arrivals, 99.0));
            text += &format!("; from the vector p50 {p50:.0} us, p99 {p99:.0} us");
        }

        if several {
            text += &format!(
                "; first in their round p50 {:.0} us over {}, later p50 {:.0} us over {}",
                p(&self.firsts, 50.0),
                self.firsts.len(),
                p(&self.others, 50.0),
                self.others.len()
            );
        }

        eprintln!("{text}");
    }
}

/// One index file, the id that its first vector has in the union of a set of collections, 0 for a file alone, and how
/// many vectors it must hold.
#[derive(Debug, PartialEq)]
struct Collection {
    path: PathBuf,
    offset: u64,
    count: u64,
}

/// The dataset, the index's name for the lines, and the index's files. Those are the one file of `--index`, which must
/// hold every vector the dataset's truth covers, or the files that `--manifest` lists, whose union the truth then
/// covers.
fn sources(args: &Args) -> (Dataset, String, Vec<Collection>) {
    let name = args.need("dataset");
    let n = args
        .str("n")
        .map(|v| v.parse().unwrap_or_else(|_| fail("--n must be a number")));
    let (source, listed, n) = match (args.str("index"), args.str("manifest")) {
        (Some(index), None) => (index.to_string(), None, n),
        (None, Some(manifest)) => {
            let text = std::fs::read_to_string(manifest).unwrap_or_else(|e| fail(&format!("reading {manifest}: {e}")));
            let dir = Path::new(manifest).parent().unwrap_or(Path::new("."));
            let (files, union) =
                parse_manifest(&text, dir).unwrap_or_else(|e| fail(&format!("reading {manifest}: {e}")));

            if n.is_some_and(|n| n != union) {
                fail(&format!(
                    "--n must be the {union} vectors that the manifest's files hold, or left out"
                ));
            }

            (manifest.to_string(), Some(files), Some(union))
        }
        _ => fail("name the index with one of --index and --manifest"),
    };
    let dataset = match args.str("derived") {
        Some(dir) => Dataset::open_derived(&data_root(), Path::new(dir), name, n),
        None => Dataset::open_in(&data_root(), &derived_root(), name, n),
    }
    .unwrap_or_else(|e| fail(&format!("opening {name}: {e}")));

    if n.is_some_and(|n| n != dataset.n) {
        fail(&format!("{name} has only {} vectors", dataset.n));
    }

    let files = listed.unwrap_or_else(|| {
        vec![Collection {
            path: PathBuf::from(&source),
            offset: 0,
            count: dataset.n as u64,
        }]
    });
    (dataset, source, files)
}

/// The files a manifest lists and the size of their union. Each line holds a file's path, the id of its first vector
/// in the union and its vector count, and the files must cover the union's ids from 0 on, one after another. A
/// relative path is taken from the manifest's folder `dir`.
fn parse_manifest(text: &str, dir: &Path) -> Result<(Vec<Collection>, usize), String> {
    let (mut files, mut next) = (Vec::new(), 0usize);

    for (i, line) in text.lines().enumerate().filter(|(_, l)| !l.trim().is_empty()) {
        let [path, offset, count] = line.split_whitespace().collect::<Vec<_>>()[..] else {
            return Err(format!("line {} should hold a path, a first id and a count", i + 1));
        };
        let number = |v: &str| {
            v.parse::<usize>()
                .map_err(|_| format!("line {} has a bad number {v}", i + 1))
        };
        let (offset, count) = (number(offset)?, number(count)?);

        if offset != next {
            return Err(format!(
                "line {} starts at id {offset}, but the files before it end at {next}",
                i + 1
            ));
        }

        files.push(Collection {
            path: dir.join(path),
            offset: offset as u64,
            count: count as u64,
        });
        next += count;
    }

    if files.is_empty() {
        return Err("the manifest lists no files".to_string());
    }

    Ok((files, next))
}

/// The headers of the index files, read through a plain open and read. It fails unless each file holds as many vectors
/// as the run says it does.
fn headers(files: &[Collection]) -> Vec<IndexFile> {
    let ids: Vec<IndexFile> = files
        .iter()
        .map(|f| IndexFile::read(&f.path).unwrap_or_else(|e| fail(&format!("reading {}: {e}", f.path.display()))))
        .collect();
    check_counts(files, &ids).unwrap_or_else(|e| fail(&e));
    ids
}

/// Fails unless each of `files` holds the vectors that it must, by the count in its header in `ids`.
fn check_counts(files: &[Collection], ids: &[IndexFile]) -> Result<(), String> {
    match files.iter().zip(ids).find(|(f, id)| f.count != id.count) {
        Some((f, id)) => Err(format!(
            "{} holds {} vectors, not the {} that the manifest or the dataset gives it",
            f.path.display(),
            id.count,
            f.count
        )),
        None => Ok(()),
    }
}

/// What a trial process searches and how, from its arguments.
struct Setup {
    dataset: Dataset,
    /// The index file or the manifest, as the lines name it.
    source: String,
    files: Vec<Collection>,
    /// Whether the files are searched together as collections, as `--manifest` asks.
    many: bool,
    params: SearchParams,
    limits: Limits,
    mode: String,
    variant: String,
    gap: Duration,
    gap_ms: f64,
    /// The share of the clusters that a C0 trial caches before its timer, if any.
    resident: Option<f64>,
    label: String,
}

impl Setup {
    fn from(args: &Args) -> Setup {
        let (dataset, source, files) = sources(args);
        let (k, nprobe, rerank) = (args.get("k", 10), args.get("nprobe", 128), args.get("rerank", 29));
        let reach = args
            .str("reach")
            .map(|v| v.parse().unwrap_or_else(|_| fail("--reach must be a number")));
        let (mode, variant) = (
            args.str("mode").unwrap_or("wave").to_string(),
            args.str("variant").unwrap_or("idle").to_string(),
        );
        let gap_ms = args.get("gap-ms", 0.0f64);
        let share = |v: &str| {
            v.parse()
                .unwrap_or_else(|_| fail("--resident must be a share of the clusters"))
        };
        let resident = args.str("resident").map(share);
        check(args, &mode, &variant, gap_ms);
        check_resident(args, &mode, resident);
        Setup {
            dataset,
            source,
            files,
            many: args.str("manifest").is_some(),
            params: SearchParams {
                k,
                nprobe,
                rerank,
                reach,
            },
            limits: Limits {
                max_k: k,
                max_nprobe: nprobe,
                max_rerank: rerank.max(k),
            },
            mode,
            variant,
            gap: Duration::from_secs_f64(gap_ms / 1000.0),
            gap_ms,
            resident,
            label: args.str("label").unwrap_or("").to_string(),
        }
    }

    /// The trial's queries, the positions that `--queries` lists, or else `--per-trial` positions from `--first` plus
    /// the trial number times that many.
    fn queries(&self, args: &Args) -> Vec<usize> {
        let nq = self.dataset.queries.len();
        let listed = args.list::<usize>("queries", &[]);
        let queries = match listed.is_empty() {
            true => {
                let (first, trial) = (args.get("first", 0usize), args.get("trial", 0usize));
                let per = args.get("per-trial", 1usize).max(1);
                (0..per).map(|j| (first + trial * per + j) % nq).collect()
            }
            false => listed,
        };

        if let Some(q) = queries.iter().find(|&&q| q >= nq) {
            fail(&format!("query {q} is past the {nq} queries of {}", self.dataset.name));
        }

        queries
    }

    /// The position of the query that warms a dummy or a C1 process's index, the dataset's last unless
    /// `--warmup-query` names one.
    fn warmup(&self, args: &Args) -> usize {
        let nq = self.dataset.queries.len();
        let q = args.get("warmup-query", nq - 1);

        if q >= nq {
            fail(&format!(
                "--warmup-query {q} is past the {nq} queries of {}",
                self.dataset.name
            ));
        }

        q
    }

    /// Opens the index the way the mode says, or the files `paths` as collections.
    fn open(&self, paths: &[&Path]) -> Result<Target, Error> {
        match self.many {
            true => Collections::open(paths).map(Target::Many),
            false => open_index(paths[0], &self.mode).map(Target::One),
        }
    }

    /// The regime that the trial's lines name, which is C0, or intent with a gap.
    fn regime(&self) -> &'static str {
        if self.gap.is_zero() { "c0" } else { "intent" }
    }
}

/// Opens the index at `path` with the read backend of `mode`.
fn open_index(path: &Path, mode: &str) -> Result<Index, Error> {
    match mode {
        "pool" => Index::open_with(path, Backend::Pool),
        "portable" => Index::open_with(path, Backend::Portable),
        _ => Index::open(path),
    }
}

/// An open index, or several searched together.
enum Target {
    One(Index),
    Many(Collections),
}

/// One search's time, when it ended and the ids it found, nearest first.
struct Found {
    took: Duration,
    ended: Instant,
    ids: Vec<u64>,
}

/// Searches `query` the way the trial's mode does. The hits of collections come in `merged`, by collection, and the
/// ids they get are their ids in the union.
fn search(
    target: &Target,
    s: &Setup,
    query: &[f32],
    ctx: &mut SearchCtx,
    merged: &mut Vec<(usize, Hit)>,
) -> Result<Found, Error> {
    let start = Instant::now();

    match target {
        Target::One(index) => {
            let hits = match s.mode.as_str() {
                "mapped" => index.search_mapped(query, &s.params, ctx)?,
                _ => index.search(query, &s.params, ctx)?,
            };
            let ended = Instant::now();
            Ok(Found {
                took: ended - start,
                ended,
                ids: hits.iter().map(|h| h.id).collect(),
            })
        }
        Target::Many(sets) => {
            sets.search(query, &s.params, ctx, merged)?;
            let ended = Instant::now();
            Ok(Found {
                took: ended - start,
                ended,
                ids: union_ids(merged, &s.files),
            })
        }
    }
}

/// The ids in the collections' union of hits given by collection, i.e. each id plus its collection's first id.
fn union_ids(hits: &[(usize, Hit)], files: &[Collection]) -> Vec<u64> {
    hits.iter().map(|&(f, hit)| files[f].offset + hit.id).collect()
}

/// One search of a trial, with what it found, its trace and the counters over it. `footprint` is the process's
/// footprint right before it, and `peak` its highest footprint by the search's end.
struct Run {
    found: Found,
    trace: Trace,
    used: Delta,
    footprint: u64,
    peak: u64,
}

/// One C0 trial, in its own process.
fn trial(args: &Args) {
    // The trial runs at the class of work a user waits on, whatever started it.
    interactive();
    let s = Setup::from(args);
    let queries = s.queries(args);
    let vectors: Vec<Vec<f32>> = queries.iter().map(|&q| s.dataset.queries.row(q).to_vec()).collect();
    let warmup = s.warmup(args);

    // The empty context is made outside the timer, as an app makes it once. Its buffers for the index are made inside
    // the timer, by `prepare` and the first search.
    let made = Instant::now();
    let mut ctx = SearchCtx::new(s.limits);
    let ctx_us = us(made.elapsed());
    let mut merged = Vec::with_capacity(s.params.k);
    let dummy = args.str("dummy").map(|d| {
        (
            d.to_string(),
            search_dummy(Path::new(d), &s, s.dataset.queries.row(warmup)),
        )
    });

    // The headers are read and the files mapped before the eviction, and every check after it goes through those
    // mappings. The blocking control's handle and buffer are made here too. So the control pays for one open, the
    // index's own, and its buffer's pages, filled now, don't count in its footprint.
    let ids = headers(&s.files);
    let layouts: Vec<Layout> = s
        .files
        .iter()
        .map(|f| Layout::read(&f.path).unwrap_or_else(|e| fail(&format!("reading {}: {e}", f.path.display()))))
        .collect();
    let round = args.get("trial", 0usize);
    // A trial that caches part of the index reads where each cluster's codes and rows are before the eviction too.
    let parts = match s.resident {
        Some(_) => layouts[0]
            .parts(&s.files[0].path)
            .unwrap_or_else(|e| fail(&format!("reading the extents of {}: {e}", s.source))),
        None => Vec::new(),
    };
    let watched = Watched::map(&s.files);
    let mut plain = (s.mode == "blocking").then(|| {
        let path = watched.first();
        File::open(path).unwrap_or_else(|e| fail(&format!("opening {}: {e}", path.display())))
    });
    let mut buf = vec![1u8; if plain.is_some() { 1 << 20 } else { 0 }];
    // The handle that caches clusters is opened before the eviction too, since an open can make another process read
    // the file's first page back.
    let mut reader = s.resident.map(|_| {
        let path = watched.first();
        let file = File::open(path).unwrap_or_else(|e| fail(&format!("opening {}: {e}", path.display())));
        no_read_ahead(&file)
            .unwrap_or_else(|e| fail(&format!("turning off reading ahead for {}: {e}", path.display())));
        file
    });
    evict(&watched.all());
    let loaded = match (s.resident, reader.as_mut()) {
        (Some(share), Some(file)) => {
            let picked = pick(parts.len(), share, args.get("seed", 1u64), round);
            load(file, &parts, &picked).unwrap_or_else(|e| fail(&format!("caching clusters of {}: {e}", s.source)))
        }
        _ => Loaded::default(),
    };
    pause(
        &s.variant,
        Duration::from_millis(if s.variant == "prewarm" { 150 } else { 200 }),
    );
    // The pages the trial cached on purpose don't count as cached at the start.
    let (all, checked) = watched.cached();
    let loaded_cached = watched.cached_in(&loaded.pages);
    let cached = (all - loaded_cached.min(all), checked);
    let paths: Vec<&Path> = s.files.iter().map(|f| f.path.as_path()).collect();

    let before = Usage::settled();
    let start = Instant::now();
    let target = s
        .open(&paths)
        .unwrap_or_else(|e| fail(&format!("opening {}: {e}", s.source)));
    let open_us = us(start.elapsed());
    let opened = Usage::now().since(&before);
    let read_us = match (plain.as_mut(), &target) {
        (Some(file), Target::One(index)) => read_all(file, &mut buf, index),
        _ => 0.0,
    };
    let prepared = Instant::now();

    // The lazy-map control skips `prepare`, which would start reading the cluster tables, so they fault in too.
    if let Target::One(index) = &target
        && s.mode != "mapped"
    {
        index
            .prepare(&mut ctx)
            .unwrap_or_else(|e| fail(&format!("preparing: {e}")));
    }

    let prepare_us = us(prepared.elapsed());
    let waited = wait_for_vector(start, s.gap, &s.variant);
    let ready = start + s.gap;
    let (mut base, mut runs) = (before, Vec::with_capacity(vectors.len()));

    for vector in &vectors {
        let found =
            search(&target, &s, vector, &mut ctx, &mut merged).unwrap_or_else(|e| fail(&format!("searching: {e}")));
        let trace = ctx.trace();
        let after = Usage::settled();
        runs.push(Run {
            found,
            trace,
            used: after.since(&base),
            footprint: base.footprint,
            peak: after.peak_footprint,
        });
        base = after;
    }

    let (rerun_us, rerun_bytes) = rerun(&target, &s, &vectors[0], &mut ctx, &mut merged);
    let syscall = cheapest_open(watched.first());
    let tail = identity(&ids, state(Json::new())).line();
    let (header_bytes, table_bytes) = open_reads(&layouts, page_size());
    let loaded_pages: usize = loaded.pages.iter().map(|r| pages(r, watched.0[0].1.len())).sum();

    for (seq, (&q, run)) in queries.iter().zip(&runs).enumerate() {
        let (first, search_us) = (seq == 0, us(run.found.took));
        let at = |v: f64| if first { v } else { 0.0 };
        let started = run.found.ended - run.found.took;
        let line = head(&s, s.regime(), round, seq, q, run, cached)
            .num("ctx_us", ctx_us)
            .num("open_us", at(open_us))
            .num("read_us", at(read_us))
            .num("prepare_us", at(prepare_us))
            .num("gap_us", at(us(waited)))
            .num("late_us", at(us(started.saturating_duration_since(ready))))
            .num("search_us", search_us)
            .num(
                "total_us",
                if first {
                    open_us + read_us + prepare_us + search_us
                } else {
                    search_us
                },
            )
            .num(
                "from_vector_us",
                if first {
                    us(run.found.ended.saturating_duration_since(ready))
                } else {
                    search_us
                },
            );
        let line = stages(line, run)
            .num("open_p_share", if first { opened.p_share() } else { f64::NAN })
            .num("open_ghz", if first { opened.ghz() } else { f64::NAN })
            .int("header_bytes", if first { header_bytes } else { 0 })
            .int("table_bytes", if first { table_bytes } else { 0 })
            .num("resident_rerun_us", if first { rerun_us } else { f64::NAN });
        let line = match first {
            true => line.int("rerun_disk_bytes", rerun_bytes),
            false => line.null("rerun_disk_bytes"),
        };
        let line = match &dummy {
            Some((path, took)) => line.str("dummy", path).num("dummy_us", *took),
            None => line.null("dummy").null("dummy_us"),
        };
        let line = line
            .int("warmup_query", warmup as u64)
            .num("syscall_open_us", syscall)
            .num("resident_share", s.resident.unwrap_or(0.0))
            .int("resident_clusters", loaded.clusters as u64)
            .int("pages_loaded", loaded_pages as u64)
            .int("pages_loaded_cached", loaded_cached as u64)
            .num("load_us", us(loaded.took));
        println!("{}", merge(&line.line(), &tail));
    }
}

/// One cluster's parts in an index file, its 1-bit codes and its rows.
#[derive(Clone, Debug, PartialEq)]
struct Parts {
    codes: Range<usize>,
    rows: Range<usize>,
}

/// What a trial read into the cache before its timer, on purpose.
#[derive(Default)]
struct Loaded {
    clusters: usize,
    /// The bytes it read of the data file, in whole pages of the cache, sorted and apart from each other.
    pages: Vec<Range<usize>>,
    took: Duration,
}

/// The clusters, of `n`, that the trial with the number `trial` caches, `share` of them picked at random from `seed`,
/// in increasing order.
fn pick(n: usize, share: f64, seed: u64, trial: usize) -> Vec<usize> {
    let mut all: Vec<usize> = (0..n).collect();
    Mix(seed ^ (trial as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)).shuffle(&mut all);
    all.truncate(((share * n as f64).round() as usize).min(n));
    all.sort_unstable();
    all
}

/// Reads the codes and the rows of the clusters `picked` of `parts` through `file`, a plain handle on the data file
/// that reads nothing ahead, in whole pages of the cache. So the reads leave those pages in the cache, and only them.
fn load(file: &mut File, parts: &[Parts], picked: &[usize]) -> Result<Loaded, Error> {
    let start = Instant::now();
    let page = page_size();
    let whole = |r: &Range<usize>| r.start / page * page..r.end.div_ceil(page) * page;
    let spans = picked
        .iter()
        .flat_map(|&c| [&parts[c].codes, &parts[c].rows])
        .filter(|r| r.start < r.end)
        .map(whole)
        .collect();
    let ranges = joined(spans);
    let mut buf = vec![0u8; 1 << 20];

    for r in &ranges {
        file.seek(SeekFrom::Start(r.start as u64))?;
        let mut left = r.len();

        while left > 0 {
            let piece = left.min(buf.len());

            match file.read(&mut buf[..piece])? {
                0 => break,
                n => left -= n,
            }
        }
    }

    Ok(Loaded {
        clusters: picked.len(),
        pages: ranges,
        took: start.elapsed(),
    })
}

/// Turns off reading ahead for `file`, so its reads cache only the pages they read.
fn no_read_ahead(file: &File) -> Result<(), Error> {
    #[cfg(target_os = "macos")]
    {
        use std::os::fd::AsRawFd;

        // SAFETY: F_RDAHEAD takes an integer and touches no memory.
        if unsafe { libc::fcntl(file.as_raw_fd(), libc::F_RDAHEAD, 0) } == -1 {
            return Err(Error::last_os_error());
        }
    }

    #[cfg(target_os = "linux")]
    {
        use std::os::fd::AsRawFd;

        // SAFETY: posix_fadvise takes only integers and touches no memory.
        let rc = unsafe { libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_RANDOM) };

        if rc != 0 {
            return Err(Error::from_raw_os_error(rc));
        }
    }

    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    let _ = file;

    Ok(())
}

/// `ranges` sorted by their start, with those that touch or overlap joined into one.
fn joined(mut ranges: Vec<Range<usize>>) -> Vec<Range<usize>> {
    ranges.sort_by_key(|r| r.start);
    let mut out: Vec<Range<usize>> = Vec::with_capacity(ranges.len());

    for r in ranges {
        match out.last_mut() {
            Some(last) if r.start <= last.end => last.end = last.end.max(r.end),
            _ => out.push(r),
        }
    }

    out
}

/// A C1 process. It opens the index, warms it with one search and then serves trials, one for each line of its input,
/// until the input ends or says stop.
fn serve(args: &Args) {
    // The process runs at the class of work a user waits on, whatever started it.
    interactive();
    let s = Setup::from(args);
    let warmup = s.warmup(args);
    let made = Instant::now();
    let mut ctx = SearchCtx::new(s.limits);
    let ctx_us = us(made.elapsed());
    let mut merged = Vec::new();
    let path = s.files[0].path.as_path();
    let id = headers(&s.files).remove(0);
    let data = data_file(path).unwrap_or_else(|e| fail(&format!("finding the data file of {}: {e}", path.display())));
    let file = Mapped::open(&data).unwrap_or_else(|e| fail(&format!("mapping {}: {e}", data.display())));
    // The codes, the rows and the ids come after the cluster tables, and every trial drops them. The header, the
    // rotation and the tables stay as the process keeps them.
    let bulk = id.codes.start..file.len();
    let target = s
        .open(&[path])
        .unwrap_or_else(|e| fail(&format!("opening {}: {e}", path.display())));
    let warmed = Instant::now();

    if let Target::One(index) = &target
        && s.mode != "mapped"
    {
        index
            .prepare(&mut ctx)
            .unwrap_or_else(|e| fail(&format!("preparing: {e}")));
    }

    search(&target, &s, s.dataset.queries.row(warmup), &mut ctx, &mut merged)
        .unwrap_or_else(|e| fail(&format!("warming the index: {e}")));
    let held = Usage::now();
    let ready = Json::new()
        .str("event", "ready")
        .int("warmup_query", warmup as u64)
        .num("warmup_us", us(warmed.elapsed()))
        .num("warmup_tables_us", us(ctx.trace().tables))
        .num("ctx_us", ctx_us)
        .int("footprint", held.footprint)
        .int("peak_footprint", held.peak_footprint);
    println!("{}", ready.line());
    let nq = s.dataset.queries.len();

    for line in std::io::stdin().lock().lines() {
        let line = line.unwrap_or_else(|e| fail(&format!("reading a command: {e}")));
        let words: Vec<String> = line.split_whitespace().map(str::to_string).collect();

        match words.first().map(String::as_str) {
            None => continue,
            Some("stop") => break,
            Some(_) => {}
        }

        let (round, queries) = command(&Args::from(words), nq).unwrap_or_else(|e| fail(&e));
        let vectors: Vec<&[f32]> = queries.iter().map(|&q| s.dataset.queries.row(q)).collect();
        file.evict_range(bulk.clone())
            .unwrap_or_else(|e| fail(&format!("dropping the bulk of {} from the cache: {e}", data.display())));
        pause(
            &s.variant,
            Duration::from_millis(if s.variant == "prewarm" { 150 } else { 200 }),
        );
        let cached = residency(&file, &bulk).unwrap_or_else(|e| fail(&format!("checking the cache: {e}")));
        let mut base = Usage::settled();
        let mut runs = Vec::with_capacity(vectors.len());

        for vector in &vectors {
            let found =
                search(&target, &s, vector, &mut ctx, &mut merged).unwrap_or_else(|e| fail(&format!("searching: {e}")));
            let trace = ctx.trace();
            let after = Usage::settled();
            runs.push(Run {
                found,
                trace,
                used: after.since(&base),
                footprint: base.footprint,
                peak: after.peak_footprint,
            });
            base = after;
        }

        let (rerun_us, rerun_bytes) = rerun(&target, &s, vectors[0], &mut ctx, &mut merged);
        let tail = id.fields(state(Json::new())).line();

        for (seq, (&q, run)) in queries.iter().zip(&runs).enumerate() {
            let (first, search_us) = (seq == 0, us(run.found.took));
            let line = head(&s, "c1", round, seq, q, run, cached)
                .num("ctx_us", ctx_us)
                .null("open_us")
                .null("read_us")
                .null("prepare_us")
                .null("gap_us")
                .null("late_us")
                .num("search_us", search_us)
                .num("total_us", search_us)
                .num("from_vector_us", search_us);
            let line = stages(line, run)
                .null("open_p_share")
                .null("open_ghz")
                .int("header_bytes", 0)
                .int("table_bytes", 0)
                .num("resident_rerun_us", if first { rerun_us } else { f64::NAN });
            let line = match first {
                true => line.int("rerun_disk_bytes", rerun_bytes),
                false => line.null("rerun_disk_bytes"),
            };
            let line = line
                .null("dummy")
                .null("dummy_us")
                .int("warmup_query", warmup as u64)
                .null("syscall_open_us");
            println!("{}", merge(&line.line(), &tail));
        }

        std::io::stdout()
            .flush()
            .unwrap_or_else(|e| fail(&format!("writing: {e}")));
    }
}

/// The trial number and the query positions of a C1 command, eg. `--trial 3 --queries 7,8`, for a dataset of `nq`
/// queries.
fn command(cmd: &Args, nq: usize) -> Result<(usize, Vec<usize>), String> {
    let queries = cmd.list::<usize>("queries", &[]);

    if queries.is_empty() {
        return Err("a C1 command needs --queries, the positions of its trial's queries".to_string());
    }

    if let Some(q) = queries.iter().find(|&&q| q >= nq) {
        return Err(format!("query {q} is past the dataset's {nq} queries"));
    }

    Ok((cmd.get("trial", 0), queries))
}

/// The first fields of a search's line: what its trial searched and how, the query, what the search found, and how
/// many of the `cached` pages checked were cached right before the timer.
fn head(s: &Setup, regime: &str, trial: usize, seq: usize, q: usize, run: &Run, cached: (usize, usize)) -> Json {
    let r = judge(&s.dataset, q, &run.found.ids, s.params.k);
    let ids: Vec<f64> = run.found.ids.iter().map(|&id| id as f64).collect();
    Json::new()
        .str("label", &s.label)
        .str("index", &s.source)
        .str("dataset", &s.dataset.name)
        .int("n", s.dataset.n as u64)
        .int("collections", s.files.len() as u64)
        .str("regime", regime)
        .str("mode", &s.mode)
        .str("variant", &s.variant)
        .num("gap_ms", s.gap_ms)
        .int("trial", trial as u64)
        .int("seq", seq as u64)
        .int("query", q as u64)
        .int("k", s.params.k as u64)
        .int("nprobe", s.params.nprobe as u64)
        .int("rerank", s.params.rerank as u64)
        .num("reach", s.params.reach.map_or(f64::NAN, f64::from))
        .num("recall", r.ids)
        .num("recall1", r.levels[0])
        .num("recall10", r.levels[1])
        .num("recall100", r.levels[2])
        .bool("ties", true)
        .bool("truth_checked", s.dataset.dists.is_some())
        .nums("ids", &ids)
        .int("pages_resident_at_start", cached.0 as u64)
        .int("pages_checked", cached.1 as u64)
}

/// The stage, read and counter fields of a search's line.
fn stages(json: Json, run: &Run) -> Json {
    let (t, used) = (&run.trace, &run.used);
    let post = us(t.tables + t.route + t.submit + t.wave + t.scan + t.rows + t.rerank);
    let json = json
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
    read_paths(json, &[t.read_path])
        .num("footprint_delta", used.footprint as f64)
        .int("footprint_before", run.footprint)
        .int("peak_footprint", run.peak)
        .num("p_share", used.p_share())
        .num("ghz", used.ghz())
        .int("energy_nj", used.energy_nj)
        .int("pageins", used.pageins)
        .int("disk_bytes", used.bytes_read)
}

/// A query's recall by ids alone at the search's k, and with ties at each k of `LEVELS`, or NaN where the search's k
/// or the truth's width falls short of it.
struct Recalls {
    ids: f64,
    levels: [f64; 3],
}

/// The recall of `ids`, the hits of query `q` nearest first, at `k`.
fn judge(dataset: &Dataset, q: usize, ids: &[u64], k: usize) -> Recalls {
    let levels = LEVELS.map(|level| match level <= k && level <= dataset.truth.width() {
        true => dataset
            .recall(q, ids, level)
            .unwrap_or_else(|e| fail(&format!("judging the hits of query {q}: {e}"))),
        false => f64::NAN,
    });
    Recalls {
        ids: recall(ids, dataset.truth.row(q), k),
        levels,
    }
}

/// The mappings that a trial drops the cache and checks it through. They are each index file's data file, then the
/// files beside it that a reader may read, so the first is the first index file's data file.
struct Watched(Vec<(PathBuf, Mapped)>);

impl Watched {
    fn map(files: &[Collection]) -> Watched {
        let map = |path: PathBuf| {
            let mapped = Mapped::open(&path).unwrap_or_else(|e| fail(&format!("mapping {}: {e}", path.display())));
            (path, mapped)
        };
        Watched(files.iter().flat_map(|f| reader_files(&f.path)).map(map).collect())
    }

    /// The first index file's data file.
    fn first(&self) -> &Path {
        &self.0[0].0
    }

    fn all(&self) -> Vec<&Mapped> {
        self.0.iter().map(|(_, m)| m).collect()
    }

    /// How many of the files' pages are cached, and how many pages they have.
    fn cached(&self) -> (usize, usize) {
        self.0.iter().fold((0, 0), |(cached, checked), (path, map)| {
            let (c, n) = residency(map, &(0..map.len()))
                .unwrap_or_else(|e| fail(&format!("checking the cache of {}: {e}", path.display())));
            (cached + c, checked + n)
        })
    }

    /// How many pages of `ranges` of the first index file's data file are cached. The ranges must be apart from each
    /// other, so no page counts twice.
    fn cached_in(&self, ranges: &[Range<usize>]) -> usize {
        let (path, map) = &self.0[0];
        let cached = |r: &Range<usize>| {
            residency(map, r).unwrap_or_else(|e| fail(&format!("checking the cache of {}: {e}", path.display())))
        };
        ranges.iter().map(|r| cached(r).0).sum()
    }
}

/// How many pages of the bytes `range` of the file `map` are cached, and how many pages the range touches.
fn residency(map: &Mapped, range: &Range<usize>) -> Result<(usize, usize), Error> {
    Ok((map.resident(range.clone())?, pages(range, map.len())))
}

/// The files that a reader of the index at `path` may read, which a trial drops from the cache. They are its data
/// file first, then the change log and the writers' lock beside it, and once maintenance has run, every other file of
/// its folder. Their names start from where the index really is, as the library's do, so a symbolic link to the index
/// file leads to the files beside the file it names.
fn reader_files(path: &Path) -> Vec<PathBuf> {
    let data = data_file(path).unwrap_or_else(|e| fail(&format!("finding the data file of {}: {e}", path.display())));
    let at = real_path(path);
    let real = |p: &Path| std::fs::canonicalize(p).unwrap_or_else(|_| p.to_path_buf());
    let beside = |suffix: &str| {
        let mut name = at.as_os_str().to_owned();
        name.push(suffix);
        PathBuf::from(name)
    };
    let mut out = vec![data.clone()];
    let folder = std::fs::read_dir(beside(".d"))
        .map(|entries| entries.flatten().map(|e| e.path()).collect::<Vec<_>>())
        .unwrap_or_default();

    for file in [beside(".delta"), beside(".lock")].into_iter().chain(folder) {
        if file.is_file() && !out.iter().any(|o| real(o) == real(&file)) {
            out.push(file);
        }
    }

    out
}

/// Where the index at `path` really is, as the library finds it. It follows symbolic links, even to a target that is
/// gone, then makes the path canonical, or only its folder when the file is gone, eg. after the first maintenance run.
fn real_path(path: &Path) -> PathBuf {
    let mut at = path.to_path_buf();

    for _ in 0..40 {
        match std::fs::read_link(&at) {
            Ok(target) => at = at.parent().map_or(target.clone(), |dir| dir.join(&target)),
            Err(_) => break,
        }
    }

    if let Ok(real) = std::fs::canonicalize(&at) {
        return real;
    }

    let dir = at
        .parent()
        .filter(|d| !d.as_os_str().is_empty())
        .and_then(|d| std::fs::canonicalize(d).ok());

    match (dir, at.file_name()) {
        (Some(dir), Some(name)) => dir.join(name),
        _ => at,
    }
}

/// The pages of `range` of a file of `len` bytes.
fn pages(range: &Range<usize>, len: usize) -> usize {
    let (page, end) = (page_size(), range.end.min(len));

    if range.start >= end {
        0
    } else {
        end.div_ceil(page) - range.start / page
    }
}

/// Drops `files` from the cache until they stay out. It tries again a few times when something keeps reading them
/// back, eg. right after many files were mapped at once.
fn evict(files: &[&Mapped]) {
    let mut tries = 0;

    loop {
        match make_cold(files, COLD_WAIT) {
            Ok(()) => return,
            Err(e) if e.kind() == ErrorKind::TimedOut && tries < 2 => tries += 1,
            Err(e) => fail(&format!("emptying the cache of the index's files: {e}")),
        }
    }
}

/// Where the parts of an index file sit, from the file's header. Those are the parts that a cold search reads before
/// its first wave, and the codes and the rows that its waves read.
#[derive(Clone, Debug, PartialEq)]
struct Layout {
    /// The file's size.
    len: usize,
    dim: usize,
    /// The dimension padded to a multiple of 64, which sizes a block of codes.
    padded: usize,
    /// How many groups the clusters fall in. The groups' centers start the centers section.
    groups: usize,
    /// The size of a page of rows, and how many rows a page holds.
    page_size: usize,
    per_page: usize,
    rotation: Range<usize>,
    centers: Range<usize>,
    extents: Range<usize>,
    codes: Range<usize>,
    rows: Range<usize>,
}

impl Layout {
    /// Reads the header of the data file of the index at `path`. A file that isn't an index of format version 1 fails
    /// with `InvalidData`.
    fn read(path: &Path) -> Result<Layout, Error> {
        let data = data_file(path)?;
        let mut file = File::open(&data)?;
        let mut head = [0u8; 324];
        let damaged = || {
            Error::new(
                ErrorKind::InvalidData,
                format!("{} isn't an index file", data.display()),
            )
        };
        file.read_exact(&mut head).map_err(|_| damaged())?;
        let len = usize::try_from(file.metadata()?.len()).map_err(|_| damaged())?;
        Layout::parse(&head, len).ok_or_else(damaged)
    }

    /// The layout in `head`, the 324 bytes of the header of version 1 of a file of `len` bytes, or None for any other
    /// header. The dimension and the padded dimension are the u32s at bytes 16 and 20, the group count the u32 at
    /// byte 36, and the page size of the rows and the rows per page the u32s at bytes 48 and 64. The table of sections
    /// starts at byte 80, with 24 bytes for each, an offset and a length first. The rotation, the centers, the
    /// extents, the codes and the rows are the second to the sixth.
    fn parse(head: &[u8; 324], len: usize) -> Option<Layout> {
        let u32_at = |at: usize| head[at..].first_chunk().map_or(0, |b| u32::from_le_bytes(*b)) as usize;
        let u64_at = |at: usize| head[at..].first_chunk().map_or(0, |b| u64::from_le_bytes(*b)) as usize;
        let section = |i: usize| {
            let at = 80 + 24 * i;
            u64_at(at)..u64_at(at).saturating_add(u64_at(at + 8))
        };

        if head[..8] != *b"COLDIVF\0" || u32_at(8) != 1 {
            return None;
        }

        Some(Layout {
            len,
            dim: u32_at(16),
            padded: u32_at(20),
            groups: u32_at(36),
            page_size: u32_at(48),
            per_page: u32_at(64),
            rotation: section(1),
            centers: section(2),
            extents: section(3),
            codes: section(4),
            rows: section(5),
        })
    }

    /// Each cluster's codes and rows in the data file of the index at `path`, from the file's extents.
    fn parts(&self, path: &Path) -> Result<Vec<Parts>, Error> {
        let mut file = File::open(data_file(path)?)?;
        let mut extents = vec![0u8; self.extents.len()];
        file.seek(SeekFrom::Start(self.extents.start as u64))?;
        file.read_exact(&mut extents)?;
        Ok(self.parts_of(&extents))
    }

    /// Each cluster's codes and rows from `extents`, the bytes of the extents section. It holds 16 bytes per cluster,
    /// in cluster order. They are where the cluster's blocks start, counted from the start of the codes, as a u64, then
    /// its vector count and the id of its first vector, as u32s. A block of 32 vectors' codes takes 4·p + 512 bytes,
    /// where p is the padded dimension, and the rows from a cluster's first id on fill `per_page` rows to a page.
    fn parts_of(&self, extents: &[u8]) -> Vec<Parts> {
        let (block, per) = (4 * self.padded + 16 * 32, self.per_page.max(1));
        let part = |e: &[u8; 16]| {
            let offset = u64::from_le_bytes(e[..8].try_into().unwrap_or_default()) as usize;
            let count = u32::from_le_bytes(e[8..12].try_into().unwrap_or_default()) as usize;
            let first = u32::from_le_bytes(e[12..].try_into().unwrap_or_default()) as usize;
            let (codes, pages) = (self.codes.start + offset, first / per..(first + count).div_ceil(per));
            let page = |p: usize| self.rows.start + p * self.page_size;
            Parts {
                codes: codes..codes + count.div_ceil(32) * block,
                rows: if count == 0 {
                    page(0)..page(0)
                } else {
                    page(pages.start)..page(pages.end)
                },
            }
        };
        extents.as_chunks::<16>().0.iter().map(part).collect()
    }

    /// What the open and the loading of the cluster tables read, in whole pages of `page` bytes. The open reads the
    /// first 16 KiB, and the rotation where it runs past them. The tables run from the clusters' codes to the end of
    /// the extents, and the codes start after the groups' centers, which take g·r bytes, where g is the group count and
    /// r is d + 8 rounded up to a multiple of 8, for d dimensions.
    fn reads(&self, page: usize) -> (u64, u64) {
        let whole = |r: Range<usize>| match r.start < r.end {
            true => (r.end.div_ceil(page) - r.start / page) as u64 * page as u64,
            false => 0,
        };
        let mut header = whole(0..FIRST_READ.min(self.len));

        if self.rotation.end > FIRST_READ {
            header += whole(self.rotation.clone());
        }

        let codes = self.centers.start + self.groups * (self.dim + 8).div_ceil(8) * 8;
        (header, whole(codes..self.extents.end))
    }
}

/// What opening the files of `layouts` and loading their cluster tables read in all, in whole pages of `page` bytes.
fn open_reads(layouts: &[Layout], page: usize) -> (u64, u64) {
    layouts
        .iter()
        .map(|l| l.reads(page))
        .fold((0, 0), |(header, tables), (h, t)| (header + h, tables + t))
}

/// Adds the fields that tie a line to its index files. For one file they are its own, and for several they are their
/// total size, their newest write, their highest epoch and a hash of their header checksums in the manifest's order.
fn identity(ids: &[IndexFile], json: Json) -> Json {
    if let [one] = ids {
        return one.fields(json);
    }

    let newest = ids.iter().map(|f| f.modified).max().unwrap_or(SystemTime::UNIX_EPOCH);
    let hash = ids.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, f| {
        (h ^ u64::from(f.crc)).wrapping_mul(0x0100_0000_01b3)
    });
    json.int("index_bytes", ids.iter().map(|f| f.bytes).sum())
        .str("index_mtime", &utc(newest))
        .int("index_epoch", ids.iter().map(|f| f.epoch).max().unwrap_or(0))
        .int("index_crc", hash)
}

/// Opens the dummy index at `path`, a small index of the same settings, and searches it once with `query` the way the
/// trial's mode searches, so the timed search finds its code and its threads warm. It has a context of its own, which
/// it drops after. Returns how long that took, in microseconds.
fn search_dummy(path: &Path, s: &Setup, query: &[f32]) -> f64 {
    let start = Instant::now();
    let index = open_index(path, &s.mode).unwrap_or_else(|e| fail(&format!("opening {}: {e}", path.display())));
    let mut ctx = SearchCtx::new(s.limits);
    let searched = match s.mode.as_str() {
        "mapped" => index.search_mapped(query, &s.params, &mut ctx).map(drop),
        "blocking" => data_file(path)
            .and_then(|data| std::io::copy(&mut File::open(data)?, &mut std::io::sink()))
            .and_then(|_| {
                index.set_warm(true);
                index.search(query, &s.params, &mut ctx).map(drop)
            }),
        _ => index
            .prepare(&mut ctx)
            .and_then(|()| index.search(query, &s.params, &mut ctx).map(drop)),
    };
    searched.unwrap_or_else(|e| fail(&format!("searching the dummy {}: {e}", path.display())));
    us(start.elapsed())
}

/// Searches the trial's first query again after the others, with the pages that its first search read still cached,
/// for the split of its cold time into computing and waiting on reads. Returns that search's time in microseconds and
/// the bytes it still read from the disk.
fn rerun(target: &Target, s: &Setup, query: &[f32], ctx: &mut SearchCtx, merged: &mut Vec<(usize, Hit)>) -> (f64, u64) {
    let before = Usage::now();
    let found = search(target, s, query, ctx, merged).unwrap_or_else(|e| fail(&format!("searching again: {e}")));
    (us(found.took), Usage::now().since(&before).bytes_read)
}

/// What one plain open of the file costs this process, eg. with an endpoint-security agent screening every open. It
/// is measured after the timed part, so it can't warm anything the timed open needs.
fn cheapest_open(path: &Path) -> f64 {
    (0..3)
        .map(|_| {
            let start = Instant::now();
            drop(File::open(path).unwrap_or_else(|e| fail(&format!("opening {}: {e}", path.display()))));
            us(start.elapsed())
        })
        .fold(f64::INFINITY, f64::min)
}

/// Waits for the vector, which arrives `gap` after `start`, when the query text arrived and the timer started, and
/// returns how long it waited. It sleeps for the part of the wait that `asleep` gives and spins the rest.
fn wait_for_vector(start: Instant, gap: Duration, variant: &str) -> Duration {
    let (now, ready) = (Instant::now(), start + gap);

    if now >= ready {
        return Duration::ZERO;
    }

    std::thread::sleep(asleep(ready - now, variant));
    spin(ready.saturating_duration_since(Instant::now()));
    now.elapsed()
}

/// How much of a wait of `left` for the vector to sleep through, as the other systems' harness waits. `idle` sleeps
/// until 5 ms before the vector, since a sleep can end late, and `prewarm` spins all of it.
fn asleep(left: Duration, variant: &str) -> Duration {
    match variant {
        "prewarm" => Duration::ZERO,
        _ => left.saturating_sub(SPIN_TAIL),
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

#[cfg(test)]
#[path = "../../tests/unit/bin/cold.rs"]
mod tests;
