//! The build tool, and a quick search check of one index.
//!
//! `diskann3 build` builds an index of the first rows of a base file and prints one JSON line with its times, the
//! process's counters and the index's meta. `diskann3 search` searches the first queries of a query file with one
//! reader and prints one JSON line with the latency and, given a truth file, recall by ids.

use std::collections::HashMap;
use std::io::{Error, ErrorKind};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::str::FromStr;
use std::time::Instant;

use diskann3::builder::{self, Spec};
use diskann3::log::Log;
use diskann3::mac_reader::ReadPool;
use diskann3::search::{Index, Options, Reader};
use diskann3::usage::Usage;
use serde_json::{Map, Value, json};

const USAGE: &str = "usage:
  diskann3 build --data BASE.fbin --rows N --out DIR [--degree 64] [--list 100] [--pq-bytes 96] [--alpha A]
                 [--saturate false] [--metric l2] [--threads 1] [--seed 1] [--ram-gb 7] [--build-pq-bytes 0]
  diskann3 search --index DIR --queries QUERIES.fbin [--truth TRUTH.ibin] [--count 100] [--k 10] [--list 100]
                  [--beam 4] [--reader pool|advise|stock] [--cache-nodes 0] [--threads 1]";

const BUILD_FLAGS: &[&str] = &[
    "data",
    "rows",
    "out",
    "degree",
    "list",
    "pq-bytes",
    "alpha",
    "saturate",
    "metric",
    "threads",
    "seed",
    "ram-gb",
    "build-pq-bytes",
];
const SEARCH_FLAGS: &[&str] = &[
    "index",
    "queries",
    "truth",
    "count",
    "k",
    "list",
    "beam",
    "reader",
    "cache-nodes",
    "threads",
];

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.first().map(String::as_str) {
        Some("build") => Flags::parse(&args[1..], BUILD_FLAGS).and_then(|f| build(&f)),
        Some("search") => Flags::parse(&args[1..], SEARCH_FLAGS).and_then(|f| search(&f)),
        _ => Err(Error::new(ErrorKind::InvalidInput, USAGE)),
    };

    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("diskann3: {e}");
            ExitCode::FAILURE
        }
    }
}

/// Flags given as `--name value` pairs.
struct Flags(HashMap<String, String>);

impl Flags {
    fn parse(args: &[String], allowed: &[&str]) -> Result<Flags, Error> {
        let mut out = HashMap::new();

        for pair in args.chunks(2) {
            let name = pair[0].strip_prefix("--").filter(|n| allowed.contains(n));
            let (Some(name), [_, value]) = (name, pair) else {
                return Err(Error::new(
                    ErrorKind::InvalidInput,
                    format!("bad flag {}\n{USAGE}", pair[0]),
                ));
            };
            out.insert(name.to_string(), value.clone());
        }

        Ok(Flags(out))
    }

    /// The flag's value, or `default` when it isn't given.
    fn get<T: FromStr>(&self, name: &str, default: Option<T>) -> Result<T, Error> {
        match (self.0.get(name), default) {
            (Some(v), _) => v
                .parse()
                .map_err(|_| Error::new(ErrorKind::InvalidInput, format!("--{name} can't be {v}"))),
            (None, Some(d)) => Ok(d),
            (None, None) => Err(Error::new(
                ErrorKind::InvalidInput,
                format!("--{name} is required\n{USAGE}"),
            )),
        }
    }

    fn path(&self, name: &str) -> Result<PathBuf, Error> {
        self.get::<String>(name, None).map(PathBuf::from)
    }
}

fn build(f: &Flags) -> Result<(), Error> {
    let threads = f.get("threads", Some(1))?;
    let spec = Spec {
        base: f.path("data")?,
        rows: f.get("rows", None)?,
        out: f.path("out")?,
        degree: f.get("degree", Some(64))?,
        list: f.get("list", Some(100))?,
        pq_bytes: f.get("pq-bytes", Some(96))?,
        alpha: f.0.get("alpha").map(|_| f.get("alpha", None)).transpose()?,
        saturate: f.get("saturate", Some(false))?,
        metric: builder::metric(&f.get::<String>("metric", Some("l2".into()))?)?,
        threads,
        seed: f.get("seed", Some(1))?,
        ram_gb: f.get("ram-gb", Some(7.0))?,
        build_pq_bytes: f.get("build-pq-bytes", Some(0))?,
    };
    builder::check(&spec)?;
    std::fs::create_dir_all(&spec.out)?;
    Log::install(tracing::Level::INFO)?;
    let before = Usage::now();
    let start = Instant::now();
    let meta = builder::build(&spec)?;
    let seconds = start.elapsed().as_secs_f64();
    let after = Usage::now();
    let mut files = Map::new();

    for path in builder::files(&spec.out)?
        .iter()
        .map(PathBuf::from)
        .chain([spec.out.join("meta.json")])
    {
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();
        files.insert(name, json!(std::fs::metadata(&path)?.len()));
    }

    let stages: Vec<Value> = diskann3::log::stages()
        .into_iter()
        .map(|(s, t)| json!([s, t]))
        .collect();
    let report = json!({
        "event": "build",
        "seconds": seconds,
        "cpu_seconds": after.cpu_ns.saturating_sub(before.cpu_ns) as f64 / 1e9,
        "peak_footprint": after.peak_footprint,
        "bytes_read": after.bytes_read.saturating_sub(before.bytes_read),
        "bytes_written": after.bytes_written.saturating_sub(before.bytes_written),
        "stages": stages,
        "files": files,
        "meta": meta,
    });
    println!("{report}");
    Ok(())
}

fn search(f: &Flags) -> Result<(), Error> {
    let dir = f.path("index")?;
    let meta: Value = serde_json::from_str(&std::fs::read_to_string(dir.join("meta.json"))?).map_err(Error::other)?;
    let metric = builder::metric(meta["metric"].as_str().unwrap_or("l2"))?;
    let reader = match f.get::<String>("reader", Some("pool".into()))?.as_str() {
        "pool" => Reader::Pool,
        "advise" => Reader::Advise,
        "stock" => Reader::Stock,
        other => return Err(Error::new(ErrorKind::InvalidInput, format!("unknown reader {other}"))),
    };
    let (k, list, beam): (usize, usize, usize) = (
        f.get("k", Some(10))?,
        f.get("list", Some(100))?,
        f.get("beam", Some(4))?,
    );
    let threads: usize = f.get("threads", Some(1))?;
    let options = Options {
        reader,
        metric,
        threads,
        cache_nodes: f.get("cache-nodes", Some(0))?,
    };
    let (rows, width, queries) = table(&f.path("queries")?)?;
    let count = f.get("count", Some(100usize))?.min(rows);
    ReadPool::global().reserve(beam * threads)?;
    let index = Index::open(&builder::prefix(&dir)?, &options, ReadPool::global())?;
    let mut times = Vec::with_capacity(count);
    let mut found = Vec::with_capacity(count);
    let (mut ios, mut bytes) = (0u64, 0u64);

    for q in 0..count {
        let start = Instant::now();
        let hit = index.search(&queries[q * width..(q + 1) * width], k, list, beam)?;
        times.push(start.elapsed().as_secs_f64() * 1e6);
        ios += hit.stats.ios as u64;
        bytes += hit.stats.reads.bytes;
        found.push(hit.ids);
    }

    let recall = match f.0.get("truth") {
        Some(path) => Some(recall(Path::new(path), &found, k)?),
        None => None,
    };
    let mut sorted = times.clone();
    sorted.sort_by(f64::total_cmp);
    let n = count.max(1) as f64;
    let report = json!({
        "event": "search",
        "reader": reader.name(),
        "queries": count,
        "k": k,
        "list": list,
        "beam": beam,
        "mean_us": times.iter().sum::<f64>() / n,
        "p50_us": sorted.get(count / 2),
        "first_us": times.first(),
        "ios_per_query": ios as f64 / n,
        "bytes_per_query": bytes as f64 / n,
        "recall_ids": recall,
        "open": {"pq_us": index.opened.pq_us, "reader_us": index.opened.reader_us,
                 "searcher_us": index.opened.searcher_us, "bytes": index.opened.bytes},
    });
    println!("{report}");
    Ok(())
}

/// The rows, the width and the values of an .fbin or .ibin file, read whole. The values keep their bits, so an
/// .ibin's ids come back as floats to reinterpret.
fn table(path: &Path) -> Result<(usize, usize, Vec<f32>), Error> {
    let bytes = std::fs::read(path)?;
    let word = |i: usize| u32::from_le_bytes([bytes[i], bytes[i + 1], bytes[i + 2], bytes[i + 3]]);

    if bytes.len() < 8 || bytes.len() != 8 + 4 * word(0) as usize * word(4) as usize {
        return Err(Error::new(
            ErrorKind::InvalidData,
            format!("{} isn't a table of 4-byte values", path.display()),
        ));
    }

    let values = bytes[8..]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_bits(u32::from_le_bytes(*c)));
    Ok((word(0) as usize, word(4) as usize, values.collect()))
}

/// The share of each query's first `k` true ids that its search found, over all queries.
fn recall(truth: &Path, found: &[Vec<u32>], k: usize) -> Result<f64, Error> {
    let (rows, width, values) = table(truth)?;

    if rows < found.len() || width < k {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "the truth file has too few rows or columns",
        ));
    }

    let mut hits = 0usize;

    for (q, ids) in found.iter().enumerate() {
        let truth: Vec<u32> = values[q * width..q * width + k].iter().map(|v| v.to_bits()).collect();
        hits += ids.iter().take(k).filter(|id| truth.contains(id)).count();
    }

    Ok(hits as f64 / (found.len().max(1) * k) as f64)
}
