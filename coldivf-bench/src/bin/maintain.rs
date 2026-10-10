//! Replays a stream of inserts, deletes and upserts on an index file and runs maintenance after each round. Each round
//! appends one JSON line with what its run did, what searches read after it, and the same measures on a fresh build of
//! the same vectors.
//!
//! maintain --dataset NAME --index PATH --label TEXT [--base 100000] [--base-from random|first] [--rounds 10]
//!          [--insert N] [--delete SHARE] [--upsert SHARE] [--pattern uniform|drift|novel|file] [--stream FILE]
//!          [--seed 1] [--settings key=value,...] [--config FILE] [--kind auto|fold|minor|major] [--queries 1000]
//!          [--pseudo 5000] [--hot 1000] [--fresh every|none|LIST] [--fixed NPROBE:RERANK] [--rerank 39]
//!          [--calibrate on|off] [--threads 4] [--results results/maintain/<label>.jsonl] [--data DIR]
//!          [--vectors-per-list 160] [--parents 128] [--anchor cluster|mean] [--page-size 4096]
//!          [--clustering flat|two-level] [--build-threads 0] [--build-seed 7] [--topics 200] [--hot-topics 10]
//!          [--topic-sample 100000] [--upsert-pool 100000]
//!
//! The stream. `--base` vectors form the base, `--base-from random` draws them at random and `first` takes the
//! dataset's first rows. Each of `--rounds` rounds then deletes a share `--delete` of the live vectors, gives a share
//! `--upsert` of the rest a new vector, and inserts `--insert` rows, by default enough for the base to grow by two
//! thirds over the rounds. An upsert's new vector is the nearest held-out row not used yet, among `--upsert-pool`
//! held-out rows. `--pattern` sets where the inserts come from:
//! - `uniform`: random held-out rows.
//! - `drift`: held-out rows of topics the base holds too. k-means over `--topic-sample` held-out rows finds `--topics`
//!   clusters, and the inserts are the held-out rows nearest `--hot-topics` of them.
//! - `novel`: rows of topics the base lacks. The topics are found the same way among all the rows, and the base is
//!   drawn from the rows outside them.
//! - `file`: the stream file `--stream`, whose format `coldivf_bench::stream` describes. Its first round is the base.
//!
//! The queries. The first `--queries` of the dataset's queries split in two: `tune` takes the even positions and `test`
//! the odd ones. `--pseudo` held-out rows form `pseudo`, and for drift and novel `--hot` held-out rows of the topics
//! form `hot`. Each query keeps its 64 nearest live vectors, exactly, as the rounds change them.
//!
//! The settings. `--config` reads a TOML file of maintenance settings, eg. `[minor]` then `change = 0.25`, and
//! `--settings` sets any after it by its path, eg. `minor.change=0.25,major.inserts=0.5,major.sample=64`. Every path of
//! `MaintainParams` is a setting, and `none` turns off a setting that can be off, eg. `major.inserts=none`. `--kind`
//! picks the run kind.
//!
//! Each round applies its changes through a `Writer`, runs `Writer::maintain`, and measures the new file and, at the
//! rounds `--fresh` names, a fresh flat build of the same vectors, made with the build settings. Round 0 measures the
//! base. For each query set:
//! - `at95`: the read cost at a mean recall@10 of 0.95. Searches with the rescore depth `--rerank` run over a grid of
//!   cluster counts until the recall passes 0.95, and the vectors scanned, the bytes of 1-bit codes and 8-bit rows
//!   read, and their ranges are interpolated between the grid's points either side. Searches go through `route`,
//!   `fetch` and `Fetched::search`, whose trace counts what a cold search reads whether or not the pages are in the
//!   cache, and after the first searches the file is.
//! - `fixed`: the same measures at the setting `--fixed`, or else at round 0's calibrated setting, or without one at
//!   round 0's first grid point at the target on `tune`.
//! - `calibrated`: the same at the setting that `calibrate` picks on the pseudo-queries for a mean recall@10 of 0.95.
//!
//! A round's summary compares the bytes of 1-bit codes read at the target with the fresh build's, since that is what
//! the clustering decides. The bytes of rows depend on the rows' order in the file too.
//!
//! Recall counts a hit outside the true 10 nearest when it is at most as far as the true 10th. A line's `usage` holds
//! what varies from one run of the same replay to the next: the times, CPU, memory and disk counters, eg. the run's
//! highest memory footprint above the one it started from, which macOS reports, and where and when the line was made.
//!
//! A rerun with the same results file resumes. It replays the stream's bookkeeping up to the first round without a
//! line, then goes on from there. The index file's epoch counts its runs, so a round whose run finished but whose line
//! wasn't written is measured without running again. Its run's report comes from the file that each run saves next to
//! the index, and a kill between the run and that save loses it, which the line marks with `"run":{"lost":true}`. A
//! round cut short before its run applies its changes again, which is safe, since an insert replaces any vector of
//! its id and a delete of a vector that is gone does nothing.

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Error, ErrorKind, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use coldivf::{
    Anchor, BuildParams, Builder, Calibration, Clustering, Index, LeafOrder, Limits, MaintainParams, RunKind,
    RunReport, SearchCtx, SearchParams, Target, Writer, calibrate,
};
use coldivf_bench::args::{Args, fail};
use coldivf_bench::coldcache::Usage;
use coldivf_bench::data::{Fbin, Ibin, data_root, metric_of, write_ibin};
use coldivf_bench::metrics::{IndexFile, Json, append, provenance, state, utc};
use coldivf_bench::stream::{self, Change, Made, Pattern, Replay, Rng};
use coldivf_bench::truth::{K, Truth};

/// The mean recall@10 that the read cost is measured at.
const TARGET: f64 = 0.95;

/// The nearest live vectors that each query keeps.
const KEEP: usize = 64;

/// The cluster counts that the read cost tries, cut at the index's cluster count. From 2 on, each is at most 1.5
/// times the one before.
const NPROBES: [usize; 38] = [
    1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, 176, 192, 224,
    256, 320, 384, 512, 768, 1024, 1536, 2048, 3072, 4096,
];

/// The most clusters and rescored candidates that any setting uses, which a calibrated setting stays within.
const MAX_NPROBE: usize = 4096;
const MAX_RERANK: usize = 2048;

fn main() {
    let config = Config::from(&Args::parse());

    if let Err(e) = replay(&config, None) {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

/// A replay's settings, from the command line.
struct Config {
    dataset: String,
    /// The folder that holds the dataset's files.
    data: PathBuf,
    index: PathBuf,
    label: String,
    results: PathBuf,
    source: Source,
    /// The dataset's queries that the real query sets take, from the first.
    queries: usize,
    params: MaintainParams,
    kind: RunKind,
    fresh: Fresh,
    fixed: Option<SearchParams>,
    rerank: usize,
    calibrate: bool,
    threads: usize,
    build: BuildParams,
    clustering: Clustering,
}

/// Where the stream comes from.
enum Source {
    Make(stream::Params),
    /// A stream file, and the held-out rows to draw at random as pseudo-queries.
    File {
        path: PathBuf,
        pseudo: usize,
        seed: u64,
    },
}

impl Source {
    /// The stream's settings as text, which a resumed replay must match.
    fn text(&self) -> String {
        match self {
            Source::Make(p) => format!(
                "pattern={},base={},base_from={},rounds={},insert={},delete={},upsert={},pseudo={},hot={},topics={},\
                 hot_topics={},topic_sample={},upsert_pool={},seed={}",
                pattern_name(p.pattern),
                p.base,
                if p.first { "first" } else { "random" },
                p.rounds,
                p.insert,
                p.delete,
                p.upsert,
                p.pseudo,
                p.hot,
                p.topics,
                p.hot_topics,
                p.sample,
                p.upsert_pool,
                p.seed
            ),
            Source::File { path, pseudo, seed } => {
                format!("file={},pseudo={pseudo},seed={seed}", path.display())
            }
        }
    }
}

fn pattern_name(p: Pattern) -> &'static str {
    match p {
        Pattern::Uniform => "uniform",
        Pattern::Drift => "drift",
        Pattern::Novel => "novel",
    }
}

/// The rounds that get a fresh build to compare with.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Fresh {
    Every,
    Never,
    At(Vec<usize>),
}

impl Fresh {
    fn at(&self, round: usize) -> bool {
        match self {
            Fresh::Every => true,
            Fresh::Never => false,
            Fresh::At(rounds) => rounds.contains(&round),
        }
    }
}

impl Config {
    /// The settings that `args` give. A bad one prints why and exits.
    fn from(args: &Args) -> Config {
        let dataset = args.need("dataset").to_string();
        let label = args.need("label").to_string();

        if label.is_empty() || label.contains(['/', '\\']) {
            fail("--label must be a name for the results file, without a slash");
        }

        let results = args.str("results").map_or_else(
            || PathBuf::from(format!("results/maintain/{label}.jsonl")),
            PathBuf::from,
        );
        let threads = args.get("threads", default_threads());
        let seed = args.get("seed", 1u64);
        let pseudo = args.get("pseudo", 5000);
        let source = match args.str("pattern").unwrap_or("uniform") {
            "file" => Source::File {
                path: PathBuf::from(args.need("stream")),
                pseudo,
                seed,
            },
            name => {
                let pattern = match name {
                    "uniform" => Pattern::Uniform,
                    "drift" => Pattern::Drift,
                    "novel" => Pattern::Novel,
                    _ => fail(&format!("--pattern must be uniform, drift, novel or file, not {name}")),
                };
                let base = args.get("base", 100_000usize);
                let rounds = args.get("rounds", 10usize);
                let first = match args.str("base-from").unwrap_or("random") {
                    "random" => false,
                    "first" => true,
                    other => fail(&format!("--base-from must be random or first, not {other}")),
                };
                Source::Make(stream::Params {
                    pattern,
                    base,
                    first,
                    rounds,
                    insert: args.get("insert", (base * 2 / 3).checked_div(rounds).unwrap_or(0)),
                    delete: args.get("delete", 0.0),
                    upsert: args.get("upsert", 0.0),
                    pseudo,
                    hot: args.get("hot", 1000),
                    topics: args.get("topics", 200),
                    hot_topics: args.get("hot-topics", 10),
                    sample: args.get("topic-sample", 100_000),
                    upsert_pool: args.get("upsert-pool", 100_000),
                    seed,
                    threads,
                })
            }
        };
        let mut params = MaintainParams::default();

        if let Some(path) = args.str("config") {
            let text = std::fs::read_to_string(path).unwrap_or_else(|e| fail(&format!("reading {path}: {e}")));
            let pairs = toml(&text).unwrap_or_else(|e| fail(&format!("{path}: {e}")));
            apply(&mut params, &pairs).unwrap_or_else(|e| fail(&format!("{path}: {e}")));
        }

        if let Some(list) = args.str("settings") {
            let pairs = pairs(list).unwrap_or_else(|e| fail(&format!("--settings: {e}")));
            apply(&mut params, &pairs).unwrap_or_else(|e| fail(&format!("--settings: {e}")));
        }

        let kind = match args.str("kind").unwrap_or("auto") {
            "auto" => RunKind::Auto,
            "fold" => RunKind::Fold,
            "minor" => RunKind::Minor,
            "major" => RunKind::Major,
            other => fail(&format!("--kind must be auto, fold, minor or major, not {other}")),
        };
        let fresh = match args.str("fresh").unwrap_or("every") {
            "every" => Fresh::Every,
            "none" => Fresh::Never,
            _ => Fresh::At(args.list("fresh", &[])),
        };
        let fixed = args
            .str("fixed")
            .map(|v| match v.split_once(':').map(|(a, b)| (a.parse(), b.parse())) {
                Some((Ok(nprobe), Ok(rerank))) if nprobe > 0 && rerank >= K => SearchParams {
                    k: K,
                    nprobe: usize::min(nprobe, MAX_NPROBE),
                    rerank: usize::min(rerank, MAX_RERANK),
                    reach: None,
                },
                _ => fail("--fixed must be NPROBE:RERANK, with NPROBE above 0 and RERANK at least 10"),
            });
        let rerank = args.get("rerank", 39usize).clamp(K, MAX_RERANK);
        let calibrate = match args.str("calibrate").unwrap_or("on") {
            "on" => true,
            "off" => false,
            other => fail(&format!("--calibrate must be on or off, not {other}")),
        };
        let build = BuildParams {
            metric: metric_of(&dataset),
            vectors_per_list: args.get("vectors-per-list", 160),
            parents: args.get("parents", 128),
            order: match args.str("order").unwrap_or("chained") {
                "chained" => LeafOrder::Chained,
                "id" => LeafOrder::Id,
                other => fail(&format!("--order must be chained or id, not {other}")),
            },
            anchor: match args.str("anchor").unwrap_or("cluster") {
                "cluster" => Anchor::Cluster,
                "mean" => Anchor::Mean,
                other => fail(&format!("--anchor must be cluster or mean, not {other}")),
            },
            page_size: args.get("page-size", 4096),
            threads: args.get("build-threads", 0),
            seed: args.get("build-seed", 7),
        };
        let clustering = match args.str("clustering").unwrap_or("flat") {
            "flat" => Clustering::Flat,
            "two-level" => Clustering::TwoLevel,
            other => fail(&format!("--clustering must be flat or two-level, not {other}")),
        };

        Config {
            data: args.str("data").map_or_else(data_root, PathBuf::from),
            index: PathBuf::from(args.need("index")),
            dataset,
            label,
            results,
            source,
            queries: args.get("queries", 1000),
            params,
            kind,
            fresh,
            fixed,
            rerank,
            calibrate,
            threads: threads.max(1),
            build,
            clustering,
        }
    }
}

/// `RAYON_NUM_THREADS` when it is set, and otherwise 4.
fn default_threads() -> usize {
    std::env::var("RAYON_NUM_THREADS")
        .ok()
        .and_then(|n| n.parse().ok())
        .filter(|&n| n > 0)
        .unwrap_or(4)
}

/// A maintenance setting's value.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Value {
    Count(usize),
    Real(f32),
    Flag(bool),
    Order(LeafOrder),
    Seed(u64),
    /// A share or limit that `None` turns off.
    Optional(Option<f32>),
}

impl Value {
    fn text(&self) -> String {
        match *self {
            Value::Count(n) => n.to_string(),
            Value::Real(x) | Value::Optional(Some(x)) => x.to_string(),
            Value::Flag(b) => b.to_string(),
            Value::Order(LeafOrder::Chained) => "chained".to_string(),
            Value::Order(LeafOrder::Id) => "id".to_string(),
            Value::Seed(n) => n.to_string(),
            Value::Optional(None) => "none".to_string(),
        }
    }

    fn json(&self, json: Json, key: &str) -> Json {
        match *self {
            Value::Count(n) => json.int(key, n as u64),
            Value::Seed(n) => json.int(key, n),
            // The value's shortest text, so 0.1 stays 0.1 rather than the f32's exact value.
            Value::Real(x) | Value::Optional(Some(x)) => json.num(key, x.to_string().parse().unwrap_or(f64::NAN)),
            Value::Flag(b) => json.bool(key, b),
            Value::Order(_) => json.str(key, &self.text()),
            Value::Optional(None) => json.null(key),
        }
    }
}

/// Every maintenance setting by its path, with its value in `p`.
fn settings(p: &MaintainParams) -> Vec<(&'static str, Value)> {
    let (minor, major) = (&p.minor, &p.major);
    vec![
        ("vectors_per_list", Value::Count(p.vectors_per_list)),
        ("rounds", Value::Count(p.rounds)),
        ("parents", Value::Count(p.parents)),
        ("order", Value::Order(p.order)),
        ("threads", Value::Count(p.threads)),
        ("seed", Value::Seed(p.seed)),
        ("minor.change", Value::Real(minor.change)),
        ("minor.shrink", Value::Real(minor.shrink)),
        ("minor.hot_list", Value::Real(minor.hot_list)),
        ("minor.list_loss", Value::Real(minor.list_loss)),
        ("minor.region_lists", Value::Count(minor.region_lists)),
        ("minor.split", Value::Real(minor.split)),
        ("minor.merge", Value::Real(minor.merge)),
        ("minor.ring", Value::Count(minor.ring)),
        ("minor.pool", Value::Optional(minor.pool)),
        ("minor.global_pass", Value::Optional(minor.global_pass)),
        ("minor.refresh", Value::Flag(minor.refresh)),
        ("major.inserts", Value::Optional(major.inserts)),
        ("major.deletes", Value::Optional(major.deletes)),
        ("major.indicator", Value::Optional(major.indicator)),
        ("major.beta", Value::Real(major.beta)),
        ("major.sample", Value::Count(major.sample)),
    ]
}

/// Every setting of `p` as text, `path=value` joined by commas.
fn settings_text(p: &MaintainParams) -> String {
    let all: Vec<String> = settings(p)
        .iter()
        .map(|(path, v)| format!("{path}={}", v.text()))
        .collect();
    all.join(",")
}

/// Sets the setting of `p` at `path` to the value that `text` gives.
fn set(p: &mut MaintainParams, path: &str, text: &str) -> Result<(), String> {
    let bad = || format!("{path} can't be {text}");
    let count = || text.parse::<usize>().map_err(|_| bad());
    let real = || match text.parse::<f32>() {
        Ok(x) if x.is_finite() => Ok(x),
        _ => Err(bad()),
    };
    let optional = || match text {
        "none" | "off" | "false" => Ok(None),
        _ => real().map(Some),
    };
    let (minor, major) = (&mut p.minor, &mut p.major);

    match path {
        "vectors_per_list" => p.vectors_per_list = count()?,
        "rounds" => p.rounds = count()?,
        "parents" => p.parents = count()?,
        "order" => {
            p.order = match text {
                "chained" => LeafOrder::Chained,
                "id" => LeafOrder::Id,
                _ => return Err(bad()),
            }
        }
        "threads" => p.threads = count()?,
        "seed" => p.seed = text.parse().map_err(|_| bad())?,
        "minor.change" => minor.change = real()?,
        "minor.shrink" => minor.shrink = real()?,
        "minor.hot_list" => minor.hot_list = real()?,
        "minor.list_loss" => minor.list_loss = real()?,
        "minor.region_lists" => minor.region_lists = count()?,
        "minor.split" => minor.split = real()?,
        "minor.merge" => minor.merge = real()?,
        "minor.ring" => minor.ring = count()?,
        "minor.pool" => minor.pool = optional()?,
        "minor.global_pass" => minor.global_pass = optional()?,
        "minor.refresh" => minor.refresh = text.parse().map_err(|_| bad())?,
        "major.inserts" => major.inserts = optional()?,
        "major.deletes" => major.deletes = optional()?,
        "major.indicator" => major.indicator = optional()?,
        "major.beta" => major.beta = real()?,
        "major.sample" => major.sample = count()?,
        _ => return Err(format!("there is no setting {path}")),
    }

    Ok(())
}

/// Sets each (path, value) pair of `pairs` in turn.
fn apply(p: &mut MaintainParams, pairs: &[(String, String)]) -> Result<(), String> {
    pairs.iter().try_for_each(|(path, text)| set(p, path, text))
}

/// The pairs of a `key=value,...` list.
fn pairs(list: &str) -> Result<Vec<(String, String)>, String> {
    list.split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|s| match s.split_once('=') {
            Some((k, v)) => Ok((k.trim().to_string(), v.trim().to_string())),
            None => Err(format!("{s} isn't key=value")),
        })
        .collect()
}

/// The settings in TOML text, as (path, value) pairs in order. It reads `key = value` lines, `[section]` headers that
/// put `section.` before the keys under them, and `#` comments. A value is a number, `true` or `false`, or a quoted
/// string, eg. "none".
fn toml(text: &str) -> Result<Vec<(String, String)>, String> {
    let mut section = String::new();
    let mut out = Vec::new();

    for (n, line) in text.lines().enumerate() {
        let line = uncomment(line).trim();

        if line.is_empty() {
            continue;
        }

        if let Some(name) = line.strip_prefix('[').and_then(|l| l.strip_suffix(']')) {
            section = name.trim().to_string();
            continue;
        }

        let Some((key, value)) = line.split_once('=') else {
            return Err(format!("line {} isn't key = value", n + 1));
        };
        let (key, value) = (key.trim(), value.trim());
        let value = match value.strip_prefix('"').and_then(|v| v.strip_suffix('"')) {
            Some(inner) => inner,
            None => value,
        };
        let path = match section.is_empty() {
            true => key.to_string(),
            false => format!("{section}.{key}"),
        };
        out.push((path, value.to_string()));
    }

    Ok(out)
}

/// `line` up to a `#` that isn't inside a quoted string.
fn uncomment(line: &str) -> &str {
    let mut quoted = false;

    for (i, c) in line.char_indices() {
        match c {
            '"' => quoted = !quoted,
            '#' if !quoted => return &line[..i],
            _ => {}
        }
    }

    line
}

/// Where a test stops a replay, as a kill would: after round r applies its changes, or after its run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Stop {
    Applied(usize),
    Ran(usize),
}

/// Replays the stream of `cfg` on its index file, from where its results file ends, and appends a line per round.
fn replay(cfg: &Config, stop: Option<Stop>) -> Result<(), Error> {
    let data = Fbin::open(&cfg.data.join(format!("{}.base.fbin", cfg.dataset)))?;
    let metric = metric_of(&cfg.dataset);
    let earlier = Earlier::read(&cfg.results)?;
    let start = earlier.lines.len();

    if start == 0 {
        clear(&cfg.index)?;
    } else {
        earlier.check(&cfg.source.text(), &settings_text(&cfg.params))?;
        remove_temps(&cfg.index)?;
    }

    let made = stream_of(cfg, &data, metric, start == 0)?;
    let rounds = made.stream.rounds.len();
    let mut replay = Replay::new(&made.stream.base)?;

    for round in made.stream.rounds.iter().take(start.saturating_sub(1)) {
        replay.apply(round)?;
    }

    let queries = Queries::load(cfg, &data, &made)?;
    eprintln!(
        "maintain: {} queries in {} sets, finding their {KEEP} nearest of {} live vectors",
        queries.count,
        queries.sets.len(),
        replay.len()
    );
    let mut truth = Truth::new(
        &data,
        metric,
        queries.vectors.clone(),
        KEEP,
        &replay.live(),
        cfg.threads,
    );
    let (base_epoch, mut fixed) = match start {
        0 => {
            let cost = build_base(cfg, &data, &made.stream.base)?;
            let file = IndexFile::read(&cfg.index)?;
            let judge = Judge {
                data: &data,
                replay: &replay,
            };
            let mut fixed = cfg.fixed;
            let base = Subject::new(&cfg.index, None)?.measure(cfg, &queries, &truth, &judge, &mut fixed)?;
            let fixed = fixed.ok_or_else(|| Error::other("round 0 found no setting to fix"))?;
            let line = Line {
                round: 0,
                rounds,
                base_epoch: file.epoch,
                fixed,
                changes: [made.stream.base.len(), 0, 0, replay.len()],
                ran: None,
                file: &file,
                maintained: &base,
                fresh: None,
                usage: Json::new().object("base", &cost.json()),
            };
            write_line(cfg, line)?;
            (file.epoch, Some(fixed))
        }
        _ => (earlier.base_epoch()?, Some(earlier.fixed()?)),
    };
    let mut writer = Writer::open(&cfg.index)?;

    for r in start.max(1)..=rounds {
        let round = &made.stream.rounds[r - 1];
        let changes = replay.apply(round)?;
        let epoch = IndexFile::read(&cfg.index)?.epoch;
        let target = base_epoch + r as u64;
        let (ran, applied) = if epoch == target {
            (Ran::load(&cfg.index, r), None)
        } else if epoch + 1 == target {
            let ((), applied) = measured(|| put(&mut writer, &data, &changes))?;

            if stop == Some(Stop::Applied(r)) {
                return Ok(());
            }

            let (report, cost) = measured(|| writer.maintain(&cfg.params, cfg.kind))?;
            let ran = Ran { report, cost };
            ran.save(&cfg.index, r)?;

            if stop == Some(Stop::Ran(r)) {
                return Ok(());
            }

            (Some(ran), Some(applied))
        } else {
            return Err(Error::new(
                ErrorKind::InvalidData,
                format!(
                    "{} has had {} runs since its base, but its results end before round {r}",
                    cfg.index.display(),
                    epoch.saturating_sub(base_epoch)
                ),
            ));
        };
        let at = Instant::now();
        truth.update(&data, &changes, &replay.live(), cfg.threads);
        let judge = Judge {
            data: &data,
            replay: &replay,
        };
        let file = IndexFile::read(&cfg.index)?;
        let maintained = Subject::new(&cfg.index, None)?.measure(cfg, &queries, &truth, &judge, &mut fixed)?;
        let (fresh, fresh_cost) = match cfg.fresh.at(r) {
            true => {
                let (subject, cost) = build_fresh(cfg, &data, &replay, r)?;
                let m = subject.measure(cfg, &queries, &truth, &judge, &mut fixed)?;
                drop(subject);
                remove_fresh(&cfg.index)?;
                (Some(m), Some(cost))
            }
            false => (None, None),
        };
        let mut usage = Json::new();

        if let Some(applied) = applied {
            usage = usage.object("apply", &applied.json());
        }

        if let Some(ran) = &ran {
            usage = usage
                .object("run", &ran.cost.json())
                .num("run_cpu_s", ran.report.cpu.as_secs_f64())
                .num("run_wall_s", ran.report.wall.as_secs_f64());
        }

        if let Some(cost) = fresh_cost {
            usage = usage.object("fresh", &cost.json());
        }

        let deleted = changes.iter().filter(|c| matches!(c, Change::Delete { .. })).count();
        let line = Line {
            round: r,
            rounds,
            base_epoch,
            fixed: fixed.unwrap_or(SearchParams {
                k: K,
                nprobe: 1,
                rerank: K,
                reach: None,
            }),
            changes: [round.inserts.len(), deleted, round.upserts.len(), replay.len()],
            ran: ran.as_ref(),
            file: &file,
            maintained: &maintained,
            fresh: fresh.as_ref(),
            usage: usage.num("measure_s", at.elapsed().as_secs_f64()),
        };
        write_line(cfg, line)?;
    }

    Ok(())
}

/// Applies `changes` through `writer`, then syncs them.
fn put(writer: &mut Writer, data: &Fbin, changes: &[Change]) -> Result<(), Error> {
    for change in changes {
        match *change {
            Change::Put { id, row } => writer.insert(id, data.row(row as usize))?,
            Change::Delete { id } => writer.delete(id)?,
        }
    }

    writer.sync()
}

/// The stream of `cfg`, made or read. A made stream and its query rows are saved next to the index when `save` is
/// set, and read back from there on a resume, or made again when they aren't there.
fn stream_of(cfg: &Config, data: &Fbin, metric: coldivf::Metric, save: bool) -> Result<Made, Error> {
    let paths = ["stream", "pseudo.ibin", "hot.ibin"].map(|e| side(&cfg.index, e));
    let [stream_path, pseudo_path, hot_path] = &paths;
    let made = match &cfg.source {
        Source::Make(_) if !save && paths.iter().all(|p| p.exists()) => Made {
            stream: stream::read(stream_path)?,
            pseudo: rows(&Ibin::open(pseudo_path)?),
            hot: rows(&Ibin::open(hot_path)?),
        },
        Source::Make(p) => {
            eprintln!("maintain: making the stream, {}", cfg.source.text());
            let made = stream::make(data, metric, p)?;
            stream::write(stream_path, &made.stream)?;
            let ids = |rows: &[u32]| rows.iter().map(|&r| r as i32).collect::<Vec<i32>>();
            write_ibin(pseudo_path, &ids(&made.pseudo), 1)?;
            write_ibin(hot_path, &ids(&made.hot), 1)?;
            made
        }
        Source::File { path, pseudo, seed } => {
            let stream = stream::read(path)?;
            let mut used = vec![false; data.len()];
            let named = stream.rounds.iter().flat_map(|r| {
                let upserts = r.upserts.iter().map(|&(_, row)| row);
                r.inserts.iter().copied().chain(upserts)
            });

            for row in stream.base.iter().copied().chain(named) {
                match used.get_mut(row as usize) {
                    Some(u) => *u = true,
                    None => {
                        return Err(Error::new(
                            ErrorKind::InvalidData,
                            "the stream names a row past the dataset",
                        ));
                    }
                }
            }

            let mut free: Vec<u32> = (0..data.len() as u32).filter(|&r| !used[r as usize]).collect();
            Rng::new(*seed).shuffle(&mut free);
            free.truncate(*pseudo);
            Made {
                stream,
                pseudo: free,
                hot: Vec::new(),
            }
        }
    };

    if made
        .stream
        .base
        .iter()
        .chain(&made.pseudo)
        .chain(&made.hot)
        .any(|&r| r as usize >= data.len())
    {
        return Err(Error::new(
            ErrorKind::InvalidData,
            "the stream names a row past the dataset",
        ));
    }

    Ok(made)
}

/// The rows of an `.ibin` file of one row id per line.
fn rows(ibin: &Ibin) -> Vec<u32> {
    (0..ibin.len()).map(|i| ibin.row(i)[0] as u32).collect()
}

/// `path` with `.ext` added to its name.
fn side(path: &Path, ext: &str) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(".");
    name.push(ext);
    PathBuf::from(name)
}

/// Removes the index file at `path`, its folder of generations and every file that a replay keeps next to it, but not
/// its lock file.
fn clear(path: &Path) -> Result<(), Error> {
    let files = ["delta", "report", "report.new", "base.fbin"].map(|e| side(path, e));

    for p in std::iter::once(path.to_path_buf()).chain(files) {
        remove(&p)?;
    }

    match std::fs::remove_dir_all(side(path, "d")) {
        Err(e) if e.kind() != ErrorKind::NotFound => return Err(e),
        _ => {}
    }

    remove_fresh(path)?;
    remove_temps(path)
}

/// Removes the fresh build next to the index file at `path`, and its files.
fn remove_fresh(path: &Path) -> Result<(), Error> {
    for p in [side(path, "fresh"), side(path, "fresh.fbin"), side(path, "fresh.delta")] {
        remove(&p)?;
    }

    Ok(())
}

/// Removes the temporary files that a build or a run killed partway leaves next to the index at `path` or its fresh
/// build, eg. `index.coldivf.123.0.tmp`.
fn remove_temps(path: &Path) -> Result<(), Error> {
    let Some(dir) = path
        .parent()
        .map(|d| if d.as_os_str().is_empty() { Path::new(".") } else { d })
    else {
        return Ok(());
    };
    let names: Vec<String> = [path.to_path_buf(), side(path, "fresh")]
        .iter()
        .filter_map(|p| p.file_name().map(|n| format!("{}.", n.to_string_lossy())))
        .collect();

    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().into_owned();
        let temp = names.iter().any(|stem| {
            name.strip_prefix(stem.as_str())
                .and_then(|rest| rest.strip_suffix(".tmp").or_else(|| rest.strip_suffix(".spill")))
                .is_some_and(|mid| {
                    mid.split('.').count() == 2
                        && mid
                            .split('.')
                            .all(|p| !p.is_empty() && p.bytes().all(|b| b.is_ascii_digit()))
                })
        });

        if temp {
            remove(&entry.path())?;
        }
    }

    Ok(())
}

fn remove(path: &Path) -> Result<(), Error> {
    match std::fs::remove_file(path) {
        Err(e) if e.kind() != ErrorKind::NotFound => Err(e),
        _ => Ok(()),
    }
}

/// Builds the base index of `cfg` from the dataset rows `base`. Rows that aren't the dataset's first ones in order are
/// copied to a file next to the index first, so that the base's vectors get the ids 0 to n − 1.
fn build_base(cfg: &Config, data: &Fbin, base: &[u32]) -> Result<Cost, Error> {
    let prefix = base.iter().enumerate().all(|(i, &r)| r as usize == i);
    let copy = side(&cfg.index, "base.fbin");

    if !prefix {
        write_rows(&copy, data, base.iter().copied())?;
    }

    let rows = match prefix {
        true => None,
        false => Some(Fbin::open(&copy)?),
    };
    let src = rows.as_ref().unwrap_or(data).prefix(base.len());
    eprintln!("maintain: building the base of {} vectors", base.len());
    let builder = Builder::new(cfg.build.clone()).clustering(cfg.clustering);
    let ((), cost) = measured(|| builder.build(&src, &cfg.index))?;
    drop(rows);
    remove(&copy)?;
    Ok(cost)
}

/// Builds a fresh flat index of the live vectors of `replay` next to the index of `cfg`, with the build settings and
/// the seed of round `round`. Returns it and what the build used.
fn build_fresh(cfg: &Config, data: &Fbin, replay: &Replay, round: usize) -> Result<(Subject, Cost), Error> {
    let (copy, out) = (side(&cfg.index, "fresh.fbin"), side(&cfg.index, "fresh"));
    let live = replay.live();
    write_rows(&copy, data, live.iter().map(|&(_, row)| row))?;
    let rows = Fbin::open(&copy)?;
    let params = BuildParams {
        seed: cfg.build.seed.wrapping_add(round as u64),
        ..cfg.build.clone()
    };
    let builder = Builder::new(params).clustering(Clustering::Flat);
    let ((), cost) = measured(|| builder.build(&rows.prefix(live.len()), &out))?;
    drop(rows);
    remove(&copy)?;
    Ok((
        Subject::new(&out, Some(live.iter().map(|&(id, _)| id).collect()))?,
        cost,
    ))
}

/// Writes the vectors of the dataset rows `rows` to the `.fbin` file `path`, one after another.
fn write_rows(path: &Path, data: &Fbin, rows: impl ExactSizeIterator<Item = u32>) -> Result<(), Error> {
    let count = i32::try_from(rows.len()).map_err(|_| Error::other("too many rows for an .fbin file"))?;
    let mut out = BufWriter::with_capacity(1 << 20, File::create(path)?);
    out.write_all(&count.to_le_bytes())?;
    out.write_all(&(data.dim() as i32).to_le_bytes())?;
    let mut bytes = Vec::with_capacity(4 * data.dim());

    for row in rows {
        bytes.clear();
        bytes.extend(data.row(row as usize).iter().flat_map(|x| x.to_le_bytes()));
        out.write_all(&bytes)?;
    }

    out.into_inner().map_err(|e| e.into_error())?.flush()
}

/// The query vectors, and the query sets that measure reads, as positions among them.
struct Queries {
    vectors: Vec<f32>,
    count: usize,
    sets: Vec<(&'static str, Vec<usize>)>,
}

impl Queries {
    /// The first `cfg.queries` of the dataset's queries, then the pseudo-queries and the hot queries of `made`.
    fn load(cfg: &Config, data: &Fbin, made: &Made) -> Result<Queries, Error> {
        let real = Fbin::open(&cfg.data.join(format!("{}.query.fbin", cfg.dataset)))?;

        if real.dim() != data.dim() {
            return Err(Error::new(
                ErrorKind::InvalidData,
                "the queries and the base differ in width",
            ));
        }

        let nq = cfg.queries.min(real.len());
        let mut vectors = Vec::with_capacity((nq + made.pseudo.len() + made.hot.len()) * data.dim());
        (0..nq).for_each(|q| vectors.extend_from_slice(real.row(q)));

        for &r in made.pseudo.iter().chain(&made.hot) {
            vectors.extend_from_slice(data.row(r as usize));
        }

        let (p, h) = (nq + made.pseudo.len(), nq + made.pseudo.len() + made.hot.len());
        let sets: Vec<(&'static str, Vec<usize>)> = vec![
            ("tune", (0..nq).step_by(2).collect()),
            ("test", (1..nq).step_by(2).collect()),
            ("pseudo", (nq..p).collect()),
            ("hot", (p..h).collect()),
        ];
        Ok(Queries {
            vectors,
            count: h,
            sets: sets.into_iter().filter(|(_, s)| !s.is_empty()).collect(),
        })
    }

    fn set(&self, name: &str) -> Option<&[usize]> {
        self.sets.iter().find(|(n, _)| *n == name).map(|(_, s)| &s[..])
    }
}

/// Gives each live vector's vector by index id, to judge hits by their distance.
struct Judge<'a> {
    data: &'a Fbin,
    replay: &'a Replay,
}

impl<'a> Judge<'a> {
    fn vector(&self, id: u64) -> Option<&'a [f32]> {
        self.replay.row(id).map(|r| self.data.row(r as usize))
    }
}

/// What searches over a query set read with one setting, as means over the queries.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Point {
    nprobe: f64,
    recall: f64,
    scanned: f64,
    t1_bytes: f64,
    t1_ranges: f64,
    t2_bytes: f64,
    t2_ranges: f64,
}

impl Point {
    /// The point a share `f` of the way from `self` to `to`.
    fn toward(&self, to: &Point, f: f64) -> Point {
        let at = |a: f64, b: f64| a + f * (b - a);
        Point {
            nprobe: at(self.nprobe, to.nprobe),
            recall: at(self.recall, to.recall),
            scanned: at(self.scanned, to.scanned),
            t1_bytes: at(self.t1_bytes, to.t1_bytes),
            t1_ranges: at(self.t1_ranges, to.t1_ranges),
            t2_bytes: at(self.t2_bytes, to.t2_bytes),
            t2_ranges: at(self.t2_ranges, to.t2_ranges),
        }
    }

    fn json(&self) -> Json {
        Json::new()
            .num("nprobe", self.nprobe)
            .num("recall", self.recall)
            .num("scanned", self.scanned)
            .num("t1_bytes", self.t1_bytes)
            .num("t2_bytes", self.t2_bytes)
            .num("t1_ranges", self.t1_ranges)
            .num("t2_ranges", self.t2_ranges)
    }
}

/// Where the mean recall first reaches `TARGET` along a grid, from `crossing`.
struct Crossing {
    /// The point interpolated there, or None when even the grid's last point falls short.
    at: Option<Point>,
    /// Every point evaluated, in grid order.
    points: Vec<Point>,
}

/// Finds where the mean recall first reaches `TARGET` along a grid of `len` points that `eval` evaluates. It starts
/// at the point `hint` and goes up while the recall falls short, or down while it doesn't, and interpolates between
/// the two points either side of the target. Before the grid's first point it takes a point of zeros.
fn crossing(len: usize, hint: usize, mut eval: impl FnMut(usize) -> Result<Point, Error>) -> Result<Crossing, Error> {
    let mut seen: Vec<(usize, Point)> = Vec::new();
    let mut at = |i: usize, seen: &mut Vec<(usize, Point)>| -> Result<Point, Error> {
        let p = eval(i)?;
        seen.push((i, p));
        Ok(p)
    };
    let mut i = hint.min(len.saturating_sub(1));
    let mut p = at(i, &mut seen)?;
    let (lo, hi) = if p.recall >= TARGET {
        loop {
            if i == 0 {
                break (None, Some(p));
            }

            let q = at(i - 1, &mut seen)?;

            if q.recall < TARGET {
                break (Some((i - 1, q)), Some(p));
            }

            (i, p) = (i - 1, q);
        }
    } else {
        loop {
            if i + 1 >= len {
                break (Some((i, p)), None);
            }

            let q = at(i + 1, &mut seen)?;

            if q.recall >= TARGET {
                break (Some((i, p)), Some(q));
            }

            (i, p) = (i + 1, q);
        }
    };
    seen.sort_by_key(|&(i, _)| i);
    let at = hi.map(|hi| {
        let lo = lo.map_or(Point::default(), |(_, p)| p);
        lo.toward(&hi, (TARGET - lo.recall) / (hi.recall - lo.recall))
    });
    Ok(Crossing {
        at,
        points: seen.into_iter().map(|(_, p)| p).collect(),
    })
}

/// The grid of cluster counts for an index of `nlist` clusters: the counts of `NPROBES` below it, then `nlist`.
fn grid(nlist: usize) -> Vec<usize> {
    let mut g: Vec<usize> = NPROBES.iter().copied().filter(|&n| n < nlist).collect();
    g.push(nlist.clamp(1, MAX_NPROBE));
    g.dedup();
    g
}

/// An index to measure: the maintained file, or a fresh build whose ids are places among the live vectors.
struct Subject {
    index: Index,
    file: IndexFile,
    /// For a fresh build, the index id of the live vector at each of its ids.
    ids: Option<Vec<u64>>,
    /// The other way, each index id's id in a fresh build, or `u64::MAX`.
    place: HashMap<u64, u64>,
}

/// One query set's measures on one index.
struct SetMeasure {
    name: &'static str,
    queries: usize,
    crossing: Crossing,
    fixed: Point,
    calibrated: Option<Point>,
}

/// An index's measures.
struct Measured {
    lists: usize,
    count: u64,
    bytes: u64,
    calibration: Option<Calibration>,
    sets: Vec<SetMeasure>,
}

impl Subject {
    fn new(path: &Path, ids: Option<Vec<u64>>) -> Result<Subject, Error> {
        let place = ids
            .iter()
            .flatten()
            .enumerate()
            .map(|(i, &id)| (id, i as u64))
            .collect();
        Ok(Subject {
            index: Index::open(path)?,
            file: IndexFile::read(path)?,
            ids,
            place,
        })
    }

    /// The measures of every query set. With no `fixed` setting yet, it takes the calibrated one, or else the grid's
    /// first point at the target on `tune`, and keeps it in `fixed`.
    fn measure(
        &self,
        cfg: &Config,
        queries: &Queries,
        truth: &Truth,
        judge: &Judge,
        fixed: &mut Option<SearchParams>,
    ) -> Result<Measured, Error> {
        let nlist = self.file.nlist as usize;
        let g = grid(nlist);
        let mut ctx = SearchCtx::new(Limits {
            max_k: K,
            max_nprobe: MAX_NPROBE,
            max_rerank: MAX_RERANK,
        });
        let mut crossings = Vec::with_capacity(queries.sets.len());

        // The search starts at the grid's first count of at least 1/32 of the clusters, which is near the target at
        // the default rerank, so the points it evaluates depend on the index alone.
        let start = g.partition_point(|&n| n * 32 < nlist);

        for (_, set) in &queries.sets {
            let c = crossing(g.len(), start, |i| {
                let s = SearchParams {
                    k: K,
                    nprobe: g[i],
                    rerank: cfg.rerank,
                    reach: None,
                };
                self.evaluate(&s, set, truth, judge, &mut ctx)
            })?;
            crossings.push(c);
        }

        let calibration = match (cfg.calibrate, queries.set("pseudo")) {
            (true, Some(set)) if set.len() >= 2 => self.calibrate(set, truth),
            _ => None,
        };
        let tune = queries.sets.iter().position(|(n, _)| *n == "tune").unwrap_or(0);
        let from_grid = crossings.get(tune).and_then(|c| {
            let nprobe = c
                .points
                .iter()
                .find(|p| p.recall >= TARGET)
                .map(|p| p.nprobe as usize)?;
            Some(SearchParams {
                k: K,
                nprobe,
                rerank: cfg.rerank,
                reach: None,
            })
        });
        let calibrated = calibration.map(|c| SearchParams {
            k: K,
            nprobe: c.nprobe,
            rerank: c.rerank.max(K),
            reach: c.reach,
        });
        let chosen = *fixed.get_or_insert_with(|| {
            calibrated.or(from_grid).unwrap_or(SearchParams {
                k: K,
                nprobe: nlist.min(MAX_NPROBE),
                rerank: cfg.rerank,
                reach: None,
            })
        });
        let chosen = SearchParams {
            nprobe: chosen.nprobe.min(nlist),
            ..chosen
        };
        let mut sets = Vec::with_capacity(queries.sets.len());

        for ((name, set), crossing) in queries.sets.iter().zip(crossings) {
            let fixed = self.evaluate(&chosen, set, truth, judge, &mut ctx)?;
            let calibrated = match calibrated {
                Some(s) => Some(self.evaluate(
                    &SearchParams {
                        nprobe: s.nprobe.min(nlist),
                        ..s
                    },
                    set,
                    truth,
                    judge,
                    &mut ctx,
                )?),
                None => None,
            };
            sets.push(SetMeasure {
                name,
                queries: set.len(),
                crossing,
                fixed,
                calibrated,
            });
        }

        Ok(Measured {
            lists: nlist,
            count: self.file.count,
            bytes: self.file.bytes,
            calibration,
            sets,
        })
    }

    /// The setting that `calibrate` picks on the queries `set`, or None when none meets the target.
    fn calibrate(&self, set: &[usize], truth: &Truth) -> Option<Calibration> {
        let mut vectors = Vec::new();
        let mut ids = Vec::with_capacity(set.len() * K);

        for &q in set {
            vectors.extend_from_slice(truth.query(q));
            let near = truth.nearest(q);
            let place = |id: u64| match self.ids {
                Some(_) => self.place.get(&id).copied().unwrap_or(u64::MAX),
                None => id,
            };
            ids.extend((0..K).map(|i| near.get(i).map_or(u32::MAX, |&(_, id)| place(id) as u32)));
        }

        let target = Target::MeanRecall {
            recall_at_10: TARGET as f32,
        };

        match calibrate(&self.index, &vectors, &ids, target) {
            Ok(c) => Some(c),
            Err(e) => {
                eprintln!("maintain: calibration found no setting: {e}");
                None
            }
        }
    }

    /// Searches every query of `set` with `s`, and averages its recall and what it read.
    fn evaluate(
        &self,
        s: &SearchParams,
        set: &[usize],
        truth: &Truth,
        judge: &Judge,
        ctx: &mut SearchCtx,
    ) -> Result<Point, Error> {
        let (mut sum, mut leaves, mut found) = (Point::default(), Vec::new(), Vec::new());

        for &q in set {
            let query = truth.query(q);
            self.index.route(query, s.nprobe, s.reach, ctx, &mut leaves)?;
            let hits = self.index.fetch(&leaves, ctx)?.search(query, s.k, s.rerank)?;
            found.clear();
            found.extend(hits.iter().map(|h| match &self.ids {
                Some(ids) => ids.get(h.id as usize).copied().unwrap_or(u64::MAX),
                None => h.id,
            }));
            let t = ctx.trace();
            sum.recall += truth.recall(q, &found, |id| judge.vector(id));
            sum.scanned += t.scanned as f64;
            sum.t1_bytes += t.cluster_bytes as f64;
            sum.t1_ranges += t.cluster_ranges as f64;
            sum.t2_bytes += t.row_bytes as f64;
            sum.t2_ranges += t.row_ranges as f64;
        }

        let n = set.len().max(1) as f64;
        Ok(Point {
            nprobe: s.nprobe as f64,
            recall: sum.recall / n,
            scanned: sum.scanned / n,
            t1_bytes: sum.t1_bytes / n,
            t1_ranges: sum.t1_ranges / n,
            t2_bytes: sum.t2_bytes / n,
            t2_ranges: sum.t2_ranges / n,
        })
    }
}

impl Measured {
    fn json(&self) -> Json {
        let mut json = Json::new()
            .int("lists", self.lists as u64)
            .int("count", self.count)
            .int("bytes", self.bytes);
        json = match self.calibration {
            Some(c) => json.object("calibration", &setting(c.nprobe, c.rerank, c.reach)),
            None => json.null("calibration"),
        };

        for s in &self.sets {
            let mut set = Json::new().int("queries", s.queries as u64);
            set = match &s.crossing.at {
                Some(p) => set.object("at95", &p.json()),
                None => set.null("at95"),
            };
            set = set.object("fixed", &s.fixed.json());
            set = match &s.calibrated {
                Some(p) => set.object("calibrated", &p.json()),
                None => set.null("calibrated"),
            };
            let grid: Vec<Json> = s.crossing.points.iter().map(Point::json).collect();
            json = json.object(s.name, &set.objects("grid", &grid));
        }

        json
    }

    /// The bytes of 1-bit codes read at the target on the set `name`.
    fn cluster_bytes_at(&self, name: &str) -> Option<f64> {
        let s = self.sets.iter().find(|s| s.name == name)?;
        s.crossing.at.map(|p| p.t1_bytes)
    }
}

fn setting(nprobe: usize, rerank: usize, reach: Option<f32>) -> Json {
    let json = Json::new().int("nprobe", nprobe as u64).int("rerank", rerank as u64);
    match reach {
        Some(r) => json.num("reach", f64::from(r)),
        None => json.null("reach"),
    }
}

/// What a piece of work used of the machine.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Cost {
    wall_ns: u64,
    cpu_ns: u64,
    disk_read: u64,
    disk_written: u64,
    /// The process's memory footprint when the work started, and the highest it reached during it.
    footprint: u64,
    peak: u64,
}

impl Cost {
    fn json(&self) -> Json {
        Json::new()
            .num("wall_s", self.wall_ns as f64 / 1e9)
            .num("cpu_s", self.cpu_ns as f64 / 1e9)
            .int("disk_read", self.disk_read)
            .int("disk_written", self.disk_written)
            .int("footprint", self.footprint)
            .int("peak", self.peak)
            .int("peak_above", self.peak.saturating_sub(self.footprint))
    }
}

/// Runs `work` and measures what it used. A thread reads the process's memory footprint every millisecond for the
/// highest it reaches. macOS reports the footprint and the counters, and elsewhere they are 0.
fn measured<T>(work: impl FnOnce() -> Result<T, Error>) -> Result<(T, Cost), Error> {
    let stop = Arc::new(AtomicBool::new(false));
    let flag = Arc::clone(&stop);
    let sampler = std::thread::spawn(move || {
        let mut most = 0;

        while !flag.load(Ordering::Relaxed) {
            most = u64::max(most, Usage::now().footprint);
            std::thread::sleep(Duration::from_millis(1));
        }

        most
    });
    let (before, start) = (Usage::now(), Instant::now());
    let out = work();
    let (wall, after) = (start.elapsed(), Usage::now());
    stop.store(true, Ordering::Relaxed);
    let sampled = sampler.join().unwrap_or(0);
    let used = after.since(&before);
    let cost = Cost {
        wall_ns: wall.as_nanos() as u64,
        cpu_ns: used.cpu_ns,
        disk_read: used.bytes_read,
        disk_written: after.bytes_written.saturating_sub(before.bytes_written),
        footprint: before.footprint,
        peak: sampled.max(after.footprint).max(before.footprint),
    };
    Ok((out?, cost))
}

/// A finished run: what `Writer::maintain` reported and what it used. It is saved next to the index, so a round whose
/// line wasn't written gets it back.
struct Ran {
    report: RunReport,
    cost: Cost,
}

impl Ran {
    /// The run of round `round` as text, `key=value` pairs.
    fn text(&self, round: usize) -> String {
        let r = &self.report;
        let c = &self.cost;
        let numbers = [
            ("round", round as u64),
            ("regions_due", r.regions_due as u64),
            ("reclustered", r.reclustered as u64),
            ("moved", r.moved as u64),
            ("reencoded", r.reencoded as u64),
            ("copied", r.copied as u64),
            ("count", r.count as u64),
            ("lists", r.lists as u64),
            ("regions", r.regions as u64),
            ("distances", r.distances),
            ("bytes_read", r.bytes_read),
            ("bytes_written", r.bytes_written),
            ("cpu_ns", r.cpu.as_nanos() as u64),
            ("wall_ns", r.wall.as_nanos() as u64),
            ("cost_wall_ns", c.wall_ns),
            ("cost_cpu_ns", c.cpu_ns),
            ("cost_disk_read", c.disk_read),
            ("cost_disk_written", c.disk_written),
            ("cost_footprint", c.footprint),
            ("cost_peak", c.peak),
        ];
        let mut text = format!("kind={}", kind_name(r.kind));

        for (key, value) in numbers {
            text.push_str(&format!(" {key}={value}"));
        }

        text
    }

    /// The run in `text`, and its round, or None when the text isn't one.
    fn parse(text: &str) -> Option<(usize, Ran)> {
        let fields: HashMap<&str, &str> = text.split_whitespace().filter_map(|p| p.split_once('=')).collect();
        let n = |key: &str| fields.get(key).and_then(|v| v.parse::<u64>().ok());
        let kind = match *fields.get("kind")? {
            "auto" => RunKind::Auto,
            "fold" => RunKind::Fold,
            "minor" => RunKind::Minor,
            "major" => RunKind::Major,
            _ => return None,
        };
        let report = RunReport {
            kind,
            regions_due: n("regions_due")? as usize,
            reclustered: n("reclustered")? as usize,
            moved: n("moved")? as usize,
            reencoded: n("reencoded")? as usize,
            copied: n("copied")? as usize,
            count: n("count")? as usize,
            lists: n("lists")? as usize,
            regions: n("regions")? as usize,
            distances: n("distances")?,
            bytes_read: n("bytes_read")?,
            bytes_written: n("bytes_written")?,
            cpu: Duration::from_nanos(n("cpu_ns")?),
            wall: Duration::from_nanos(n("wall_ns")?),
            indicator: None,
        };
        let cost = Cost {
            wall_ns: n("cost_wall_ns")?,
            cpu_ns: n("cost_cpu_ns")?,
            disk_read: n("cost_disk_read")?,
            disk_written: n("cost_disk_written")?,
            footprint: n("cost_footprint")?,
            peak: n("cost_peak")?,
        };
        Some((n("round")? as usize, Ran { report, cost }))
    }

    /// Saves the run of round `round` next to the index at `index`, through a temporary file and a rename.
    fn save(&self, index: &Path, round: usize) -> Result<(), Error> {
        let (path, tmp) = (side(index, "report"), side(index, "report.new"));
        std::fs::write(&tmp, self.text(round))?;
        std::fs::rename(&tmp, path)
    }

    /// The run of round `round` saved next to the index at `index`, if that is the one saved.
    fn load(index: &Path, round: usize) -> Option<Ran> {
        let text = std::fs::read_to_string(side(index, "report")).ok()?;
        Ran::parse(&text).filter(|&(r, _)| r == round).map(|(_, ran)| ran)
    }

    fn json(&self) -> Json {
        let r = &self.report;
        Json::new()
            .str("kind", kind_name(r.kind))
            .int("regions_due", r.regions_due as u64)
            .int("reclustered", r.reclustered as u64)
            .int("moved", r.moved as u64)
            .int("reencoded", r.reencoded as u64)
            .int("copied", r.copied as u64)
            .int("count", r.count as u64)
            .int("lists", r.lists as u64)
            .int("regions", r.regions as u64)
            .int("distances", r.distances)
            .int("bytes_read", r.bytes_read)
            .int("bytes_written", r.bytes_written)
    }
}

fn kind_name(kind: RunKind) -> &'static str {
    match kind {
        RunKind::Auto => "auto",
        RunKind::Fold => "fold",
        RunKind::Minor => "minor",
        RunKind::Major => "major",
    }
}

/// One round's line.
struct Line<'a> {
    round: usize,
    rounds: usize,
    base_epoch: u64,
    fixed: SearchParams,
    /// The round's inserts, deletes and upserts, and the live vectors after them.
    changes: [usize; 4],
    ran: Option<&'a Ran>,
    file: &'a IndexFile,
    maintained: &'a Measured,
    fresh: Option<&'a Measured>,
    usage: Json,
}

/// Appends `line` to the results file of `cfg`, and prints a summary.
fn write_line(cfg: &Config, line: Line) -> Result<(), Error> {
    eprintln!("{}", summary(&line));
    let mut json = Json::new()
        .str("label", &cfg.label)
        .str("dataset", &cfg.dataset)
        .str("index", &cfg.index.display().to_string())
        .int("round", line.round as u64)
        .int("rounds", line.rounds as u64)
        .str("stream", &cfg.source.text())
        .str("settings_text", &settings_text(&cfg.params))
        .str("kind", kind_name(cfg.kind))
        .int("base_epoch", line.base_epoch)
        .str("fixed_setting", &fixed_text(&line.fixed))
        .int("rerank", cfg.rerank as u64);
    let mut set = Json::new();

    for (path, value) in settings(&cfg.params) {
        set = value.json(set, path);
    }

    let [inserted, deleted, upserted, live] = line.changes.map(|n| n as u64);
    json = json.object("settings", &set).object(
        "changes",
        &Json::new()
            .int("inserted", inserted)
            .int("deleted", deleted)
            .int("upserted", upserted)
            .int("live", live),
    );
    json = match line.ran {
        Some(ran) => json.object("run", &ran.json()),
        None if line.round == 0 => json.null("run"),
        None => json.object("run", &Json::new().bool("lost", true)),
    };
    let file = Json::new()
        .int("bytes", line.file.bytes)
        .int("epoch", line.file.epoch)
        .int("crc", u64::from(line.file.crc))
        .int("lists", u64::from(line.file.nlist))
        .int("count", line.file.count);
    json = json
        .object("file", &file)
        .object(
            "fixed",
            &setting(line.fixed.nprobe, line.fixed.rerank, line.fixed.reach),
        )
        .object("maintained", &line.maintained.json());
    json = match line.fresh {
        Some(f) => json.object("fresh", &f.json()),
        None => json.null("fresh"),
    };
    let process = Usage::now();
    let usage = state(provenance(line.usage))
        .str("index_mtime", &utc(line.file.modified))
        .num("process_cpu_s", (process.user_ns + process.system_ns) as f64 / 1e9)
        .int("process_peak", process.peak_footprint);
    append(&cfg.results, &json.object("usage", &usage).line())
}

/// A one-line summary of a round, eg. its run and how its read cost compares with the fresh build's.
fn summary(line: &Line) -> String {
    let run = match line.ran {
        Some(ran) => {
            let r = &ran.report;
            format!(
                "{} run, {} regions due, {} of {} codes measured again, {:.1} s, {:.0} MB above its start",
                kind_name(r.kind),
                r.regions_due,
                r.reencoded,
                r.count,
                r.wall.as_secs_f64(),
                ran.cost.peak.saturating_sub(ran.cost.footprint) as f64 / 1e6
            )
        }
        None if line.round == 0 => "the base".to_string(),
        None => "a run whose report was lost".to_string(),
    };
    let sets: Vec<String> = line
        .maintained
        .sets
        .iter()
        .map(|s| {
            let ratio = match (
                line.maintained.cluster_bytes_at(s.name),
                line.fresh.and_then(|f| f.cluster_bytes_at(s.name)),
            ) {
                (Some(m), Some(f)) => format!(" ({:.3}x fresh)", m / f),
                _ => String::new(),
            };
            let bytes = line
                .maintained
                .cluster_bytes_at(s.name)
                .map_or("-".to_string(), |b| format!("{:.0} KB", b / 1e3));
            format!("{} {bytes}{ratio}", s.name)
        })
        .collect();
    format!(
        "maintain: round {} of {}, {} live: {run}; cluster bytes at recall 0.95: {}",
        line.round,
        line.rounds,
        line.changes[3],
        sets.join(", ")
    )
}

/// A fixed setting as text, `nprobe:rerank:reach`, with `none` for no reach.
fn fixed_text(s: &SearchParams) -> String {
    let reach = s.reach.map_or("none".to_string(), |r| r.to_string());
    format!("{}:{}:{reach}", s.nprobe, s.rerank)
}

/// The setting in a fixed setting's text.
fn parse_fixed(text: &str) -> Option<SearchParams> {
    let mut parts = text.split(':');
    let (nprobe, rerank, reach) = (parts.next()?.parse().ok()?, parts.next()?.parse().ok()?, parts.next()?);
    let reach = match reach {
        "none" => None,
        r => Some(r.parse().ok()?),
    };
    Some(SearchParams {
        k: K,
        nprobe,
        rerank,
        reach,
    })
}

/// The lines that an earlier replay with the same results file wrote.
#[derive(Debug)]
struct Earlier {
    lines: Vec<String>,
}

impl Earlier {
    /// Reads the results file at `path`, which may not be there. A last line cut short, eg. by a kill, is cut off the
    /// file. Lines that don't count the rounds from 0 fail with `InvalidData`.
    fn read(path: &Path) -> Result<Earlier, Error> {
        let text = match std::fs::read_to_string(path) {
            Ok(text) => text,
            Err(e) if e.kind() == ErrorKind::NotFound => String::new(),
            Err(e) => return Err(e),
        };
        let mut lines: Vec<String> = text.lines().map(str::to_string).collect();

        if lines.last().is_some_and(|l| !l.ends_with('}')) || (!text.is_empty() && !text.ends_with('\n')) {
            lines.pop();
            // One truncation cuts the line off where it starts, so a kill can't leave less of the file, as it could
            // while the lines before it were written again.
            let body = text.strip_suffix('\n').unwrap_or(&text);
            let kept = body.rfind('\n').map_or(0, |at| at + 1);
            OpenOptions::new().write(true).open(path)?.set_len(kept as u64)?;
        }

        for (i, line) in lines.iter().enumerate() {
            if field(line, "round") != Some(&i.to_string()[..]) {
                return Err(Error::new(
                    ErrorKind::InvalidData,
                    format!("line {} of {} isn't round {i}", i + 1, path.display()),
                ));
            }
        }

        Ok(Earlier { lines })
    }

    /// Checks that the earlier replay used the stream and the settings `stream` and `settings`.
    fn check(&self, stream: &str, settings: &str) -> Result<(), Error> {
        let first = &self.lines[0];

        for (key, want) in [("stream", stream), ("settings_text", settings)] {
            if field(first, key) != Some(want) {
                return Err(Error::new(
                    ErrorKind::InvalidInput,
                    format!(
                        "the results file was started with the {key} {}, not {want}, so it can't resume",
                        field(first, key).unwrap_or("")
                    ),
                ));
            }
        }

        Ok(())
    }

    fn base_epoch(&self) -> Result<u64, Error> {
        field(&self.lines[0], "base_epoch")
            .and_then(|v| v.parse().ok())
            .ok_or_else(|| {
                Error::new(
                    ErrorKind::InvalidData,
                    "the results file's first line has no base_epoch",
                )
            })
    }

    fn fixed(&self) -> Result<SearchParams, Error> {
        field(&self.lines[0], "fixed_setting")
            .and_then(parse_fixed)
            .ok_or_else(|| {
                Error::new(
                    ErrorKind::InvalidData,
                    "the results file's first line has no fixed setting",
                )
            })
    }
}

/// The value of the first field `key` of the JSON line `line`, a string's text without its quotes. It reads only
/// what this program writes, whose strings hold no quotes.
fn field<'a>(line: &'a str, key: &str) -> Option<&'a str> {
    let at = line.find(&format!("\"{key}\":"))? + key.len() + 3;
    let rest = &line[at..];

    match rest.strip_prefix('"') {
        Some(s) => s.find('"').map(|end| &s[..end]),
        None => Some(&rest[..rest.find([',', '}']).unwrap_or(rest.len())]),
    }
}

#[cfg(test)]
#[path = "../../tests/unit/bin/maintain.rs"]
mod tests;
