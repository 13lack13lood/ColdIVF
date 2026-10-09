//! Building a DiskANN disk index from the first rows of a base file.
//!
//! An index is a folder with DiskANN's three files, named with the prefix "index". `index_disk.index` holds the graph
//! and the full vectors in 4 KiB sectors, `index_pq_pivots.bin` the product quantization (PQ) codebooks and
//! `index_pq_compressed.bin` every vector's PQ code. A `meta.json` beside them records how it was built.

use std::io::{Error, ErrorKind};
use std::path::{Path, PathBuf};
use std::time::Instant;

use diskann::graph::config::{Builder, MaxDegree};
use diskann::utils::ONE;
use diskann_disk::QuantizationType;
use diskann_disk::build::builder::build::DiskIndexBuilder;
use diskann_disk::disk_index_build_parameter::{DiskIndexBuildParameters, MemoryBudget, NumPQChunks};
use diskann_disk::storage::DiskIndexWriter;
use diskann_providers::model::IndexConfiguration;
use diskann_providers::storage::{get_compressed_pq_file, get_disk_index_file, get_pq_pivot_file};
use diskann_vector::distance::Metric;
use serde_json::{Value, json};

use crate::mac_reader::SECTOR;
use crate::prefix::PrefixStorage;
use crate::search::Data;

/// The file name prefix of every index file.
pub const PREFIX: &str = "index";

/// How to build an index.
#[derive(Clone, Debug)]
pub struct Spec {
    /// The base file, of which rows 0 to `rows` are indexed. Row i gets id i.
    pub base: PathBuf,
    pub rows: usize,
    /// The folder the index goes to.
    pub out: PathBuf,
    /// The graph's degree after pruning, and the search list size while building.
    pub degree: usize,
    pub list: usize,
    /// Bytes per vector of the PQ codes that searches keep in memory.
    pub pq_bytes: usize,
    /// The pruning parameter, or DiskANN's default when `None`.
    pub alpha: Option<f32>,
    /// Whether pruned lists are filled back up to the degree.
    pub saturate: bool,
    pub metric: Metric,
    pub threads: usize,
    pub seed: u64,
    /// The memory DiskANN may plan for. A build that needs more is split into overlapping shards.
    pub ram_gb: f64,
}

/// The prefix of the index files in folder `dir`.
pub fn prefix(dir: &Path) -> Result<String, Error> {
    dir.join(PREFIX)
        .to_str()
        .map(str::to_string)
        .ok_or_else(|| Error::new(ErrorKind::InvalidInput, format!("{} isn't valid UTF-8", dir.display())))
}

/// The index files of the folder `dir`, in the order the searcher reads them at open.
pub fn files(dir: &Path) -> Result<[String; 3], Error> {
    let p = prefix(dir)?;
    Ok([
        get_pq_pivot_file(&p),
        get_compressed_pq_file(&p),
        get_disk_index_file(&p),
    ])
}

/// The metric named "l2", "ip", "cosine" or "cosine-normalized". The last assumes unit vectors.
pub fn metric(name: &str) -> Result<Metric, Error> {
    match name {
        "l2" => Ok(Metric::L2),
        "ip" => Ok(Metric::InnerProduct),
        "cosine" => Ok(Metric::Cosine),
        "cosine-normalized" => Ok(Metric::CosineNormalized),
        _ => Err(Error::new(ErrorKind::InvalidInput, format!("unknown metric {name}"))),
    }
}

/// The name `metric` takes for a metric.
pub fn metric_name(metric: Metric) -> &'static str {
    match metric {
        Metric::L2 => "l2",
        Metric::InnerProduct => "ip",
        Metric::Cosine => "cosine",
        Metric::CosineNormalized => "cosine-normalized",
    }
}

/// Checks a spec before any work starts, and returns the base file's width.
pub fn check(spec: &Spec) -> Result<usize, Error> {
    let (count, width) = crate::prefix::header(&spec.base)?;
    let bad = |msg: String| Err(Error::new(ErrorKind::InvalidInput, msg));

    if spec.rows == 0 || spec.rows > count {
        return bad(format!(
            "{} has {count} rows, so it has no prefix of {}",
            spec.base.display(),
            spec.rows
        ));
    }

    if spec.degree == 0 || spec.list == 0 || spec.threads == 0 {
        return bad("the degree, the build list size and the threads must be positive".into());
    }

    if spec.pq_bytes == 0 || spec.pq_bytes > width {
        return bad(format!(
            "PQ codes take 1 to {width} bytes per vector, not {}",
            spec.pq_bytes
        ));
    }

    if spec.ram_gb.is_nan() || spec.ram_gb <= 0.0 {
        return bad(format!(
            "the build's memory budget must be positive, not {} GiB",
            spec.ram_gb
        ));
    }

    Ok(width)
}

/// Builds the index and writes its `meta.json`. It returns the meta's contents.
pub fn build(spec: &Spec) -> Result<Value, Error> {
    let width = check(spec)?;
    let start = Instant::now();
    let storage = PrefixStorage::new(&spec.base, spec.rows)?;
    let invalid = |e: String| Error::new(ErrorKind::InvalidInput, e);
    let params = DiskIndexBuildParameters::new(
        MemoryBudget::try_from_gb(spec.ram_gb).map_err(|e| invalid(e.to_string()))?,
        QuantizationType::FP,
        NumPQChunks::new_with(spec.pq_bytes, width).map_err(|e| invalid(e.to_string()))?,
    );
    let config = Builder::new_with(
        spec.degree,
        MaxDegree::default_slack(),
        spec.list,
        spec.metric.into(),
        |b| {
            if let Some(alpha) = spec.alpha {
                b.alpha(alpha);
            }

            b.saturate_after_prune(spec.saturate);
        },
    )
    .build()
    .map_err(|e| invalid(e.to_string()))?;
    let alpha = config.alpha();
    let index_config = IndexConfiguration::new(spec.metric, width, spec.rows, ONE, spec.threads, config)
        .with_pseudo_rng_from_seed(spec.seed);
    let writer =
        DiskIndexWriter::new(storage.name().to_string(), prefix(&spec.out)?, None, SECTOR).map_err(crate::ann)?;
    let mut builder =
        DiskIndexBuilder::<Data, PrefixStorage>::new(&storage, params, index_config, writer).map_err(crate::ann)?;
    builder.build().map_err(crate::ann)?;

    let meta = json!({
        "format": "diskann3/1",
        "diskann": crate::DISKANN_VERSION,
        "prefix": PREFIX,
        "rows": spec.rows,
        "dim": width,
        "metric": metric_name(spec.metric),
        "degree": spec.degree,
        "build_list": spec.list,
        "pq_bytes": spec.pq_bytes,
        "alpha": alpha,
        "saturate": spec.saturate,
        "seed": spec.seed,
        "build_ram_gb": spec.ram_gb,
        "build_seconds": start.elapsed().as_secs_f64(),
    });
    let text = serde_json::to_string_pretty(&meta).map_err(Error::other)?;
    std::fs::write(spec.out.join("meta.json"), text + "\n")?;
    Ok(meta)
}
