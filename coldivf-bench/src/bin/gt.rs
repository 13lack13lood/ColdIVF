//! Exact ground truth for the first `n` base vectors of a dataset: each query's 100 nearest neighbors and their
//! distances, by a plain scan on several threads. Each thread takes a share of the queries and scans the base in tiles
//! small enough to stay in the CPU's cache while every one of its queries visits them. Ties go to the lower id. It
//! reads only the base and the queries, so it also makes the first ground truth of a new dataset.
//!
//! gt --dataset cohere-768-cosine [--n N] [--k 100] [--threads 8] [--queries 1000] [--out PATH]
//!    [--snapshots 1000000,2000000,5000000]
//!
//! The ids go to `--out`, by default `<name>-<n>.gt.ibin` in the folder of derived data, nearest first. Their distances
//! go next to them, as `<stem>.gt.dist.fbin`. A distance is the squared Euclidean distance, or minus the inner product
//! for cosine. `--snapshots` takes prefix sizes up to `n` and writes the ground truth of each in the same pass, to
//! `<name>-<size>.gt.ibin` in the folder of derived data and its distances file, as soon as the scan passes it.

use std::io::{Error, ErrorKind};
use std::path::{Path, PathBuf};
use std::time::Instant;

use coldivf::Metric;
use coldivf_bench::args::{Args, fail};
use coldivf_bench::data::{Fbin, data_root, dist_path, distance, metric_of, subset_truth, write_fbin, write_ibin};

/// Base rows in one tile.
const TILE: usize = 2048;

fn main() {
    let args = Args::parse();
    let name = args.need("dataset");
    let open = |kind: &str| {
        let path = data_root().join(format!("{name}.{kind}.fbin"));
        Fbin::open(&path).unwrap_or_else(|e| fail(&format!("opening {}: {e}", path.display())))
    };
    let (base, queries) = (open("base"), open("query"));
    let n = args.get("n", base.len()).min(base.len());
    let k = args.get("k", 100);
    let threads = args.get("threads", 8).max(1);
    let nq = args.get("queries", queries.len()).min(queries.len());
    let out = args.str("out").map_or_else(|| subset_truth(name, n), PathBuf::from);
    let mut sizes: Vec<usize> = args.list("snapshots", &[]);
    sizes.push(n);
    sizes.sort_unstable();
    sizes.dedup();

    if queries.dim() != base.dim() {
        fail("the queries and the base vectors have different widths");
    }

    if k == 0 || nq == 0 {
        fail("--k and --queries must be at least 1");
    }

    if sizes[sizes.len() - 1] > n {
        fail(&format!("--snapshots can't be more than n, which is {n}"));
    }

    if sizes[0] < k {
        fail("n and every snapshot must hold at least --k rows");
    }

    let vectors: Vec<&[f32]> = (0..nq).map(|q| queries.row(q)).collect();
    let start = Instant::now();
    scan(&base, &vectors, metric_of(name), k, threads, &sizes, |size, best| {
        let path = if size == n {
            out.clone()
        } else {
            subset_truth(name, size)
        };
        write(&path, best, k).unwrap_or_else(|e| fail(&format!("writing {}: {e}", path.display())));
        eprintln!(
            "gt: {nq} queries x {size} rows, top {k}, after {:.1} s, to {} and {}",
            start.elapsed().as_secs_f64(),
            path.display(),
            dist_path(&path).display()
        );
    });
}

/// Each query's nearest rows, as (distance, id) pairs, nearest first.
type Nearest = Vec<(f32, u32)>;

/// Finds the `k` nearest base rows of each query in one pass over the base. At each of `sizes`, which must be sorted,
/// it hands `done` the nearest of the first `size` rows, one list per query.
fn scan(
    base: &Fbin,
    queries: &[&[f32]],
    metric: Metric,
    k: usize,
    threads: usize,
    sizes: &[usize],
    mut done: impl FnMut(usize, &[Nearest]),
) {
    let mut best: Vec<Nearest> = vec![Vec::with_capacity(k + 1); queries.len()];
    let (per, mut from) = (queries.len().div_ceil(threads).max(1), 0);

    for &size in sizes {
        std::thread::scope(|scope| {
            for (group, mine) in best.chunks_mut(per).zip(queries.chunks(per)) {
                scope.spawn(move || {
                    for start in (from..size).step_by(TILE) {
                        for (b, &query) in group.iter_mut().zip(mine) {
                            for id in start..(start + TILE).min(size) {
                                offer(b, (distance(metric, query, base.row(id)), id as u32), k);
                            }
                        }
                    }
                });
            }
        });
        done(size, &best);
        from = size;
    }
}

/// Keeps the `k` smallest (distance, id) pairs in `best`, sorted.
fn offer(best: &mut Nearest, c: (f32, u32), k: usize) {
    if best.len() == k && !before(c, best[k - 1]) {
        return;
    }

    let at = best.partition_point(|&b| before(b, c));
    best.insert(at, c);
    best.truncate(k);
}

fn before(a: (f32, u32), b: (f32, u32)) -> bool {
    a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)).is_lt()
}

/// Writes the ids of `best`, `k` to a query, to `path`, and their distances next to it. It removes an old distances
/// file first, so a write that fails never leaves one that belongs to other ids.
fn write(path: &Path, best: &[Nearest], k: usize) -> Result<(), Error> {
    let dists = dist_path(path);

    if let Err(e) = std::fs::remove_file(&dists)
        && e.kind() != ErrorKind::NotFound
    {
        return Err(e);
    }

    let ids: Vec<i32> = best.iter().flatten().map(|&(_, id)| id as i32).collect();
    let values: Vec<f32> = best.iter().flatten().map(|&(d, _)| d).collect();
    write_ibin(path, &ids, k)?;
    write_fbin(&dists, &values, k)
}

#[cfg(test)]
#[path = "../../tests/unit/bin/gt.rs"]
mod tests;
