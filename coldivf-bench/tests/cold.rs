//! Checks of the cold harness end to end, on a small dataset in a folder of the build's own: the order of a run's
//! trials and their queries, the dummy and the rerun, C1, a gap, the files beside an index and a set of collections.
//! They check what the lines say, not the times, which a test machine can't hold steady.
#![cfg(any(target_os = "macos", target_os = "linux"))]

use std::fs::File;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::{Mutex, MutexGuard, OnceLock};

use coldivf::{Metric, Writer};
use coldivf_bench::coldcache::page_size;
use coldivf_bench::data::{distance, write_fbin, write_ibin};
use coldivf_bench::metrics::IndexFile;

/// The dataset's dimension, size and query count, and the width of its truth.
const DIM: usize = 128;
const ROWS: usize = 2400;
const QUERIES: usize = 30;
const WIDTH: usize = 20;

/// The dataset's name. Its truth covers every row, and its halves get datasets of their own for the collections.
const NAME: &str = "tiny-cosine";

/// The folder every test shares, with the dataset and the index files, made once.
static FOLDER: OnceLock<PathBuf> = OnceLock::new();

/// Held while a test runs trials, so no other test's trial reads the same files meanwhile and keeps them cached.
static TRIALS: Mutex<()> = Mutex::new(());

/// `n` vectors of unit length around 12 centers, from `seed`.
fn unit_rows(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 40) as f32 / (1u64 << 24) as f32 - 0.5
    };
    let centers: Vec<f32> = (0..12 * DIM).map(|_| 4.0 * next()).collect();
    let mut rows = Vec::with_capacity(n * DIM);

    for i in 0..n {
        let row: Vec<f32> = (0..DIM).map(|d| centers[(i % 12) * DIM + d] + next()).collect();
        let norm = row.iter().map(|x| x * x).sum::<f32>().sqrt();
        rows.extend(row.iter().map(|x| x / norm));
    }

    rows
}

/// Runs `build_index` on the dataset `name` of the folder, into `out`, and writes the file out to the disk, since a
/// page that isn't written yet can't leave the cache.
fn build(dir: &Path, name: &str, out: &str, more: &[&str]) {
    let small = ["--vectors-per-list", "100", "--parents", "3", "--threads", "2"];
    let done = Command::new(env!("CARGO_BIN_EXE_build_index"))
        .env("COLDIVF_DATA", dir)
        .args(["--dataset", name, "--out"])
        .arg(dir.join(out))
        .args(small)
        .arg("--results")
        .arg(dir.join("build.jsonl"))
        .args(more)
        .output()
        .unwrap();
    assert!(done.status.success(), "{}", String::from_utf8_lossy(&done.stderr));
    File::open(dir.join(out)).unwrap().sync_all().unwrap();
}

/// The shared folder. It holds the dataset with its exact truth, an index of all of it, a dummy of its first 600 rows,
/// and an index of each half with a manifest of the two. The folder `derived` holds the same truth for all the rows,
/// with its distances.
fn folder() -> &'static Path {
    FOLDER.get_or_init(|| {
        let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("cold-harness");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("derived")).unwrap();
        let (base, queries) = (unit_rows(ROWS, 5), unit_rows(QUERIES, 9));
        let (mut truth, mut dists) = (Vec::with_capacity(QUERIES * WIDTH), Vec::with_capacity(QUERIES * WIDTH));

        for q in queries.chunks(DIM) {
            let mut ranked: Vec<(f32, usize)> = base
                .chunks(DIM)
                .enumerate()
                .map(|(i, row)| (distance(Metric::Cosine, q, row), i))
                .collect();
            ranked.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            truth.extend(ranked[..WIDTH].iter().map(|&(_, i)| i as i32));
            dists.extend(ranked[..WIDTH].iter().map(|&(d, _)| d));
        }

        write_fbin(&dir.join(format!("{NAME}.base.fbin")), &base, DIM).unwrap();
        write_fbin(&dir.join(format!("{NAME}.query.fbin")), &queries, DIM).unwrap();
        write_ibin(&dir.join(format!("{NAME}.gt.ibin")), &truth, WIDTH).unwrap();
        let derived = dir.join("derived");
        write_ibin(&derived.join(format!("{NAME}-{ROWS}.gt.ibin")), &truth, WIDTH).unwrap();
        write_fbin(&derived.join(format!("{NAME}-{ROWS}.gt.dist.fbin")), &dists, WIDTH).unwrap();
        let half = ROWS / 2 * DIM;
        write_fbin(&dir.join("tinya-cosine.base.fbin"), &base[..half], DIM).unwrap();
        write_fbin(&dir.join("tinyb-cosine.base.fbin"), &base[half..], DIM).unwrap();
        build(&dir, NAME, "all.coldivf", &[]);
        build(&dir, NAME, "dummy.coldivf", &["--n", "600"]);
        build(&dir, "tinya-cosine", "a.coldivf", &[]);
        build(&dir, "tinyb-cosine", "b.coldivf", &[]);
        let manifest = format!("a.coldivf 0 {}\nb.coldivf {} {}\n", ROWS / 2, ROWS / 2, ROWS / 2);
        std::fs::write(dir.join("halves.manifest"), manifest).unwrap();
        dir
    })
}

/// Runs `cold` with the dataset's settings and `args`, writing its lines to `results`.
fn run_cold(results: &str, args: &[&str]) -> Output {
    let dir = folder();
    let out = dir.join(results);
    let _ = std::fs::remove_file(&out);
    Command::new(env!("CARGO_BIN_EXE_cold"))
        .env("COLDIVF_DATA", dir)
        .current_dir(dir)
        .args(["--dataset", NAME, "--nprobe", "24", "--rerank", "300", "--results"])
        .arg(&out)
        .args(args)
        .output()
        .unwrap()
}

/// Runs a whole `cold` run with `args` on the dataset and returns the lines it wrote to `results`.
fn cold(results: &str, args: &[&str]) -> Vec<String> {
    let done = run_cold(results, args);
    assert!(done.status.success(), "{}", String::from_utf8_lossy(&done.stderr));
    std::fs::read_to_string(folder().join(results))
        .unwrap()
        .lines()
        .map(str::to_string)
        .collect()
}

/// The lock that keeps the tests' trials apart.
fn alone() -> MutexGuard<'static, ()> {
    TRIALS.lock().unwrap_or_else(|e| e.into_inner())
}

/// The raw value of `line`'s field `key`, a JSON line the harness wrote, up to the next comma outside a list.
fn raw<'a>(line: &'a str, key: &str) -> &'a str {
    let at = line
        .find(&format!("\"{key}\":"))
        .unwrap_or_else(|| panic!("no {key} in {line}"))
        + key.len()
        + 3;
    let rest = &line[at..];
    let end = match rest.starts_with('[') {
        true => rest.find(']').unwrap() + 1,
        false => rest.find([',', '}']).unwrap(),
    };
    &rest[..end]
}

/// The number in `line`'s field `key`, or None for null.
fn num(line: &str, key: &str) -> Option<f64> {
    match raw(line, key) {
        "null" => None,
        v => Some(v.parse().unwrap_or_else(|_| panic!("{key} is {v}"))),
    }
}

/// The text in `line`'s field `key`, or None for null.
fn text(line: &str, key: &str) -> Option<String> {
    let v = raw(line, key);
    (v != "null").then(|| v.trim_matches('"').to_string())
}

/// The ids in `line`.
fn ids(line: &str) -> Vec<u64> {
    let v = raw(line, "ids").trim_matches(['[', ']']);
    v.split(',')
        .filter(|x| !x.is_empty())
        .map(|x| x.parse().unwrap())
        .collect()
}

/// The lines of the first queries of their trials.
fn firsts(lines: &[String]) -> Vec<&String> {
    lines.iter().filter(|l| num(l, "seq") == Some(0.0)).collect()
}

/// The pages of the file at `path`, or of its bytes from `from` on.
fn pages(path: &Path, from: usize) -> f64 {
    let (len, page) = (std::fs::metadata(path).unwrap().len() as usize, page_size());
    (len.div_ceil(page) - from / page) as f64
}

/// Checks the counters of a first query's line, which only macOS fills: the reads of a trial that started with
/// nothing cached reach the disk, and the footprints are there.
fn counted(line: &str) {
    if cfg!(target_os = "macos") {
        let get = |key: &str| num(line, key).unwrap();
        let read = get("header_bytes") + get("table_bytes") + get("cluster_bytes") + get("row_bytes");
        assert!(get("disk_bytes") >= read, "{line}");
        assert!(get("footprint_before") > 0.0, "{line}");
        assert!(get("peak_footprint") >= get("footprint_before"), "{line}");
    }
}

#[test]
fn a_run_takes_its_combinations_in_turn_and_gives_each_trial_queries_of_its_own() {
    let _alone = alone();
    let dummy = folder().join("dummy.coldivf").display().to_string();
    let args = [
        "--index",
        "all.coldivf",
        "--trials",
        "2",
        "--variant",
        "idle,prewarm",
        "--per-trial",
        "2",
        "--dummy",
        &dummy,
        "--seed",
        "5",
    ];
    let lines = cold("rotate.jsonl", &args);
    assert_eq!(lines.len(), 8);
    let first = firsts(&lines);
    let variants: Vec<String> = first.iter().map(|l| text(l, "variant").unwrap()).collect();
    let orders: Vec<f64> = first.iter().map(|l| num(l, "order").unwrap()).collect();
    // Each round runs both variants, and the second runs them the other way round.
    assert_ne!(variants[0], variants[1]);
    assert_eq!((&variants[2], &variants[3]), (&variants[1], &variants[0]));
    assert_eq!(orders, [0.0, 1.0, 0.0, 1.0]);
    assert_eq!(text(first[0], "predecessor"), None);
    assert_eq!(text(first[2], "predecessor"), Some(format!("wave {} 0", variants[1])));
    assert!(lines.iter().all(|l| num(l, "order_seed") == Some(5.0)));

    // Every trial reads its own queries, two each, and the warm-up query comes after all of them.
    let queries: Vec<f64> = lines.iter().map(|l| num(l, "query").unwrap()).collect();
    assert_eq!(queries, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
    assert!(lines.iter().all(|l| num(l, "warmup_query") == Some(8.0)));

    for line in &first {
        assert_eq!(text(line, "regime").as_deref(), Some("c0"), "{line}");
        assert_eq!(text(line, "dummy"), Some(dummy.clone()), "{line}");
        assert!(num(line, "dummy_us").unwrap() > 0.0, "{line}");
        assert!(num(line, "resident_rerun_us").unwrap() > 0.0, "{line}");
        assert!(num(line, "rerun_disk_bytes").is_some(), "{line}");
        // The open reads the first 16 KiB, and the tables take at least a page.
        assert_eq!(num(line, "header_bytes"), Some(16384.0), "{line}");
        assert!(num(line, "table_bytes").unwrap() >= page_size() as f64, "{line}");
        assert_eq!(num(line, "collections"), Some(1.0));
        // The check before the timer covers every page of the index, which has no files beside it.
        assert_eq!(
            num(line, "pages_checked"),
            Some(pages(&folder().join("all.coldivf"), 0))
        );
        assert_eq!(num(line, "pages_resident_at_start"), Some(0.0), "{line}");
        counted(line);
    }

    // Every query finds its 10 nearest with every cluster read, with ties counted and by ids, and the line has the ids.
    for line in &lines {
        assert_eq!(ids(line).len(), 10, "{line}");
        assert!(
            num(line, "recall10").unwrap() >= 0.9 && num(line, "recall").unwrap() >= 0.9,
            "{line}"
        );
        assert_eq!(text(line, "ties").as_deref(), Some("true"));
        // The dataset's own truth has no distances file beside it.
        assert_eq!(text(line, "truth_checked").as_deref(), Some("false"));
    }

    // Only a trial's first search reran.
    assert!(
        lines
            .iter()
            .filter(|l| num(l, "seq") == Some(1.0))
            .all(|l| num(l, "resident_rerun_us").is_none())
    );
}

#[test]
fn a_c1_run_keeps_the_index_open_and_times_the_search_alone() {
    let _alone = alone();
    let args = [
        "--index",
        "all.coldivf",
        "--c1",
        "--trials",
        "3",
        "--variant",
        "prewarm",
        "--per-trial",
        "2",
    ];
    let lines = cold("c1.jsonl", &args);
    assert_eq!(lines.len(), 6);
    let queries: Vec<f64> = lines.iter().map(|l| num(l, "query").unwrap()).collect();
    assert_eq!(queries, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    let index = folder().join("all.coldivf");
    let codes = IndexFile::read(&index).unwrap().codes.start;

    for line in firsts(&lines) {
        assert_eq!(text(line, "regime").as_deref(), Some("c1"), "{line}");
        assert_eq!((num(line, "open_us"), num(line, "prepare_us")), (None, None), "{line}");
        assert_eq!(num(line, "total_us"), num(line, "search_us"));
        // The process loaded the cluster tables with its warm-up, so no trial loads them.
        assert!(num(line, "tables_us").unwrap() < 5.0, "{line}");
        // Every trial drops the bulk sections, from the codes to the end, which the warm-up read, and nothing reads
        // them back before the timer.
        assert_eq!(num(line, "pages_checked"), Some(pages(&index, codes)), "{line}");
        assert_eq!(num(line, "pages_resident_at_start"), Some(0.0), "{line}");
        assert!(num(line, "resident_rerun_us").unwrap() > 0.0, "{line}");
        assert_eq!(num(line, "warmup_query"), Some(6.0));
        counted(line);
    }
}

#[test]
fn a_set_of_collections_is_judged_by_its_ids_in_the_union() {
    let _alone = alone();
    let args = ["--manifest", "halves.manifest", "--trials", "2", "--derived", "derived"];
    let lines = cold("halves.jsonl", &args);
    assert_eq!(lines.len(), 2);
    let dir = folder();

    for line in &lines {
        assert_eq!(
            (num(line, "collections"), num(line, "n")),
            (Some(2.0), Some(ROWS as f64)),
            "{line}"
        );
        assert!(ids(line).iter().all(|&id| id < ROWS as u64), "{line}");
        // Both halves hold some of the nearest, so a search finds them only with their ids in the union.
        assert!(num(line, "recall10").unwrap() >= 0.9, "{line}");
        // The truth for all the rows comes from the folder asked for, with its distances.
        assert_eq!(text(line, "truth_checked").as_deref(), Some("true"), "{line}");
        // The open read both headers, and the first search both files' cluster tables.
        assert_eq!(num(line, "header_bytes"), Some(2.0 * 16384.0), "{line}");
        assert!(num(line, "table_bytes").unwrap() >= 2.0 * page_size() as f64, "{line}");
        assert!(num(line, "tables_us").unwrap() > 0.0, "{line}");
        assert_eq!(text(line, "index").as_deref(), Some("halves.manifest"));
        // Both files were dropped from the cache and checked whole.
        let both = pages(&dir.join("a.coldivf"), 0) + pages(&dir.join("b.coldivf"), 0);
        assert_eq!(num(line, "pages_checked"), Some(both), "{line}");
        assert_eq!(num(line, "pages_resident_at_start"), Some(0.0), "{line}");
        counted(line);
    }

    let truth = std::fs::read(dir.join(format!("{NAME}.gt.ibin"))).unwrap();
    let nearest = |q: usize| i32::from_le_bytes(truth[8 + q * WIDTH * 4..12 + q * WIDTH * 4].try_into().unwrap());
    let halves: Vec<bool> = (0..QUERIES).map(|q| nearest(q) < ROWS as i32 / 2).collect();
    assert!(
        halves.contains(&true) && halves.contains(&false),
        "the queries' nearest rows are all in one half"
    );

    // A manifest whose counts are wrong fails, even when they add up to the dataset's size.
    let wrong = format!("a.coldivf 0 {}\nb.coldivf {} {}\n", ROWS / 3, ROWS / 3, ROWS - ROWS / 3);
    std::fs::write(dir.join("wrong.manifest"), wrong).unwrap();
    let done = run_cold("wrong.jsonl", &["--manifest", "wrong.manifest", "--trials", "1"]);
    let said = String::from_utf8_lossy(&done.stderr);
    assert!(
        !done.status.success() && said.contains("a.coldivf holds 1200 vectors, not the 800"),
        "{said}"
    );
}

#[test]
fn a_trial_drops_and_checks_the_change_log_beside_the_index() {
    let _alone = alone();
    let dir = folder();
    // A copy of the index with a log of one insert and its delete, and the writers' lock.
    let (index, log, lock) = (
        dir.join("side.coldivf"),
        dir.join("side.coldivf.delta"),
        dir.join("side.coldivf.lock"),
    );
    std::fs::copy(dir.join("all.coldivf"), &index).unwrap();
    let mut writer = Writer::open(&index).unwrap();
    writer.insert(ROWS as u64, &unit_rows(1, 11)).unwrap();
    writer.delete(ROWS as u64).unwrap();
    writer.sync().unwrap();
    drop(writer);
    assert!(std::fs::metadata(&log).unwrap().len() > 0 && lock.exists());

    // Reading the log puts it in the cache, which the trial must empty before its timer.
    let _ = std::fs::read(&log).unwrap();
    let lines = cold("side.jsonl", &["--index", "side.coldivf", "--trials", "1"]);
    let want = pages(&index, 0) + pages(&log, 0) + pages(&lock, 0);

    for line in firsts(&lines) {
        assert_eq!(num(line, "pages_checked"), Some(want), "{line}");
        assert_eq!(num(line, "pages_resident_at_start"), Some(0.0), "{line}");
        // The deleted insert is no hit.
        assert!(num(line, "recall10").unwrap() >= 0.9, "{line}");
    }
}

#[test]
fn a_gap_starts_the_search_when_the_vector_arrives() {
    let _alone = alone();
    let args = [
        "--index",
        "all.coldivf",
        "--trials",
        "2",
        "--variant",
        "prewarm",
        "--gap-ms",
        "40",
    ];
    let lines = cold("gap.jsonl", &args);

    for line in &lines {
        assert_eq!(text(line, "regime").as_deref(), Some("intent"), "{line}");
        let get = |key: &str| num(line, key).unwrap();
        // The search waited for the vector, which came 40 ms after the open started.
        let before = get("open_us") + get("read_us") + get("prepare_us");
        assert!(before + get("gap_us") >= 39_000.0 && get("gap_us") > 30_000.0, "{line}");
        assert!(get("late_us") >= 0.0 && get("late_us") < 10_000.0, "{line}");
        assert!(get("from_vector_us") >= get("search_us"), "{line}");
        assert_eq!(get("total_us"), before + get("search_us"));
    }

    // The search starts a moment after the vector arrives, which the clock shows in at least one of the trials.
    assert!(lines.iter().any(|l| num(l, "late_us").unwrap() > 0.0), "{lines:?}");
}

#[test]
fn a_trial_caches_its_share_of_the_clusters_and_counts_only_the_other_pages() {
    let _alone = alone();
    let lines = cold(
        "resident.jsonl",
        &["--index", "all.coldivf", "--trials", "2", "--resident", "0,0.5,1"],
    );
    let firsts = firsts(&lines);
    assert_eq!(firsts.len(), 6);
    let nlist = IndexFile::read(&folder().join("all.coldivf")).unwrap().nlist as f64;
    let disk = |share: f64| -> Vec<f64> {
        let at = firsts.iter().filter(|l| num(l, "resident_share") == Some(share));
        at.map(|l| num(l, "disk_bytes").unwrap()).collect()
    };

    for line in &firsts {
        let share = num(line, "resident_share").unwrap();
        let (loaded, cached) = (
            num(line, "pages_loaded").unwrap(),
            num(line, "pages_loaded_cached").unwrap(),
        );
        assert_eq!(num(line, "resident_clusters"), Some((share * nlist).round()), "{line}");
        // Nothing the trial didn't load was cached at its timer, and what it loaded still was.
        assert_eq!(num(line, "pages_resident_at_start"), Some(0.0), "{line}");
        assert_eq!(cached, loaded, "{line}");
        assert_eq!(loaded > 0.0, share > 0.0, "{line}");
        // The combination before it names its share too.
        assert!(text(line, "predecessor").is_none_or(|p| p.contains(" r")), "{line}");
    }

    // With every cluster cached, the waves read nothing from the disk, so the trial reads less than with none.
    if cfg!(target_os = "macos") {
        let (none, all) = (disk(0.0), disk(1.0));
        assert!(
            all.iter().fold(0.0f64, |a, &b| a.max(b)) < none.iter().fold(f64::MAX, |a, &b| a.min(b)),
            "{none:?} {all:?}"
        );
    }
}

#[test]
fn a_pool_gives_its_queries_to_the_trials_in_turn() {
    let _alone = alone();
    let lines = cold(
        "pool.jsonl",
        &["--index", "all.coldivf", "--trials", "4", "--pool", "7,2,9"],
    );
    let queries: Vec<f64> = firsts(&lines).iter().map(|l| num(l, "query").unwrap()).collect();
    assert_eq!(queries, [7.0, 2.0, 9.0, 7.0]);
    // The warm-up query is one the pool doesn't hold.
    assert!(
        firsts(&lines).iter().all(|l| num(l, "warmup_query") == Some(10.0)),
        "{lines:?}"
    );
}
