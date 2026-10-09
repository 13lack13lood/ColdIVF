//! Builds an index file from the first `n` vectors of a dataset, with ColdIVF's own k-means or with clusters saved
//! earlier. It records the clustering and the encoding apart, each with its time and disk reads, the file's size and
//! header checksum, which later results name the file by, and on macOS the process's peak memory and disk traffic.
//!
//! build_index --dataset cohere-768-cosine [--n N] --out PATH [--vectors-per-list 160] [--parents 128]
//!       [--order chained|id] [--anchor cluster|mean] [--page-size 4096] [--threads 4] [--seed 7]
//!       [--clustering flat|two-level] [--clusters STEM] [--export-clusters STEM] [--results results/build.jsonl]
//!
//! `--clusters STEM` builds from the clusters saved in `STEM.centroids.fbin`, a cluster center per row, and
//! `STEM.assign.ibin`, each vector's cluster, of which the first `n` count. The groups come from `STEM.parents.fbin`
//! and `STEM.parent_of.ibin` when both are there. Otherwise k-means over the cluster centers finds `--parents` of
//! them, as a flat build does, and that counts as clustering. `--export-clusters STEM` saves the clusters and groups
//! that the build uses to those four files before it encodes, and that time counts as neither.

use std::path::{Path, PathBuf};
use std::time::Instant;

use coldivf::{Anchor, BuildParams, Builder, Clustering, Clusters, LeafOrder};
use coldivf_bench::args::{Args, fail};
use coldivf_bench::coldcache::Usage;
use coldivf_bench::data::{Fbin, data_root, load_clusters, metric_of, save_clusters};
use coldivf_bench::metrics::{IndexFile, Json, append, provenance, state};

fn main() {
    let args = Args::parse();
    let name = args.need("dataset");
    let base_path = data_root().join(format!("{name}.base.fbin"));
    let base = Fbin::open(&base_path).unwrap_or_else(|e| fail(&format!("opening {}: {e}", base_path.display())));
    let n = args.get("n", base.len()).min(base.len());
    let out = PathBuf::from(args.need("out"));
    let order = match args.str("order").unwrap_or("chained") {
        "chained" => LeafOrder::Chained,
        "id" => LeafOrder::Id,
        other => fail(&format!("--order must be chained or id, not {other}")),
    };
    let anchor = match args.str("anchor").unwrap_or("cluster") {
        "cluster" => Anchor::Cluster,
        "mean" => Anchor::Mean,
        other => fail(&format!("--anchor must be cluster or mean, not {other}")),
    };
    let clustering = match args.str("clustering").unwrap_or("flat") {
        "flat" => Clustering::Flat,
        "two-level" => Clustering::TwoLevel,
        other => fail(&format!("--clustering must be flat or two-level, not {other}")),
    };
    let (imported, exported) = (args.str("clusters"), args.str("export-clusters"));

    if imported.is_some() && args.str("clustering").is_some() {
        fail("--clusters builds from saved clusters, so it takes no --clustering");
    }

    let params = BuildParams {
        metric: metric_of(name),
        vectors_per_list: args.get("vectors-per-list", 160),
        parents: args.get("parents", 128),
        order,
        anchor,
        page_size: args.get("page-size", 4096),
        threads: args.get("threads", 4),
        seed: args.get("seed", 7),
    };
    let builder = Builder::new(params.clone()).clustering(clustering);
    let src = base.prefix(n);
    let (start, before) = (Instant::now(), Usage::now());
    let clusters = match imported {
        Some(stem) => saved(&builder, Path::new(stem), n, base.dim()),
        None => builder
            .cluster(&src, &out)
            .unwrap_or_else(|e| fail(&format!("clustering for {}: {e}", out.display()))),
    };
    let (cluster_seconds, clustered) = (start.elapsed().as_secs_f64(), Usage::now());
    let export_seconds = exported.map(|stem| {
        let at = Instant::now();
        save_clusters(Path::new(stem), &clusters)
            .unwrap_or_else(|e| fail(&format!("saving the clusters to {stem}: {e}")));
        at.elapsed().as_secs_f64()
    });
    let (encoding, encode_before) = (Instant::now(), Usage::now());

    builder
        .build_with_clusters(&src, &clusters, &out)
        .unwrap_or_else(|e| fail(&format!("building {}: {e}", out.display())));
    let (encode_seconds, after) = (encoding.elapsed().as_secs_f64(), Usage::now());
    let (cluster_used, encode_used) = (clustered.since(&before), after.since(&encode_before));
    let seconds = cluster_seconds + encode_seconds;
    let id = IndexFile::read(&out).unwrap_or_else(|e| fail(&format!("reading {}: {e}", out.display())));
    let bytes = id.bytes;
    let input = (n * base.dim() * 4) as u64;
    eprintln!(
        "build: {n} x {} in {seconds:.1} s, {cluster_seconds:.1} s clustering and {encode_seconds:.1} s encoding, \
         {bytes} bytes ({:.0} per vector, {:.1}x smaller than the {input} input bytes), peak footprint {:.2} GB, to {}",
        base.dim(),
        bytes as f64 / n.max(1) as f64,
        input as f64 / bytes.max(1) as f64,
        after.peak_footprint as f64 / 1e9,
        out.display()
    );
    let dim = clusters.dim.max(1);
    let line = state(provenance(Json::new()))
        .str("dataset", name)
        .int("n", n as u64)
        .str("out", &out.display().to_string())
        .int("vectors_per_list", params.vectors_per_list as u64)
        .int("parents", params.parents as u64)
        .str("order", args.str("order").unwrap_or("chained"))
        .str("anchor", args.str("anchor").unwrap_or("cluster"))
        .str(
            "clustering",
            imported.map_or(args.str("clustering").unwrap_or("flat"), |_| "imported"),
        )
        .int("page_size", params.page_size as u64)
        .int("threads", params.threads as u64)
        .int("seed", params.seed);
    let line = text(line, "clusters", imported);
    let line = text(line, "export_clusters", exported)
        .int("nlist", (clusters.centroids.len() / dim) as u64)
        .int("nparents", (clusters.parents.len() / dim) as u64)
        .num("seconds", seconds)
        .num("cluster_seconds", cluster_seconds)
        .num("encode_seconds", encode_seconds)
        .num("export_seconds", export_seconds.unwrap_or(f64::NAN))
        .int("bytes", bytes)
        .int("input_bytes", input)
        .int("peak_footprint", after.peak_footprint)
        .num("cpu_seconds", (cluster_used + encode_used).cpu_ns as f64 / 1e9)
        .num("cluster_cpu_seconds", cluster_used.cpu_ns as f64 / 1e9)
        .num("encode_cpu_seconds", encode_used.cpu_ns as f64 / 1e9)
        .int("disk_read", after.bytes_read.saturating_sub(before.bytes_read))
        .int("cluster_disk_read", cluster_used.bytes_read)
        .int("encode_disk_read", encode_used.bytes_read)
        .int("disk_written", after.bytes_written.saturating_sub(before.bytes_written));
    let line = id.fields(line).line();
    let results = args.str("results").unwrap_or("results/build.jsonl");
    append(Path::new(results), &line).unwrap_or_else(|e| fail(&format!("writing {results}: {e}")));
}

/// The clusters saved under `stem` for the first `n` vectors of `dim` dimensions. When the files hold no groups,
/// `builder` finds them.
fn saved(builder: &Builder, stem: &Path, n: usize, dim: usize) -> Clusters {
    let mut clusters = load_clusters(stem, n, dim)
        .unwrap_or_else(|e| fail(&format!("reading the clusters at {}: {e}", stem.display())));

    if clusters.parents.is_empty() {
        builder
            .group(&mut clusters)
            .unwrap_or_else(|e| fail(&format!("grouping the clusters at {}: {e}", stem.display())));
    }

    clusters
}

/// `json` with the text `value` under `key`, or a null without one.
fn text(json: Json, key: &str, value: Option<&str>) -> Json {
    match value {
        Some(v) => json.str(key, v),
        None => json.null(key),
    }
}
