//! Exact ground truth for the first `n` base vectors of a dataset: each query's 100 nearest neighbors and their
//! distances, by a plain scan on several threads. Each thread takes a share of the queries and scans the base in tiles
//! small enough to stay in the CPU's cache while every one of its queries visits them. Ties go to the lower id. It
//! reads only the base and the queries, so it also makes the first ground truth of a new dataset.
//!
//! gt --dataset cohere-768-cosine [--n N] [--k 100] [--threads 8] [--queries 1000] [--out PATH]
//!    [--snapshots 1000000,2000000,5000000]
//! gt --dataset cohere10m-768-cosine --stream FILE --states s0=0,s1=1,s1i=1i --out-prefix PATH [--k 100]
//!    [--threads 8] [--queries 1000]
//!
//! The ids go to `--out`, by default `<name>-<n>.gt.ibin` in the folder of derived data, nearest first. Their distances
//! go next to them, as `<stem>.gt.dist.fbin`. A distance is the squared Euclidean distance, or minus the inner product
//! for cosine. `--snapshots` takes prefix sizes up to `n` and writes the ground truth of each in the same pass, to
//! `<name>-<size>.gt.ibin` in the folder of derived data and its distances file, as soon as the scan passes it.
//!
//! With `--stream` it writes the truth of each state of a stream file instead, all in one pass over the rows the
//! stream names: the exact top `k` over the rows the state holds, to `<out-prefix>-<state>.gt.ibin` and its distances
//! file. A state `name=R` holds the base and the first R rounds with their deletes, and `name=Ri` the same rounds
//! without their deletes. The stream's base must be rows 0 to n in order, and each insert a row of its own past it.

use std::io::{Error, ErrorKind};
use std::path::{Path, PathBuf};
use std::time::Instant;

use coldivf::Metric;
use coldivf_bench::args::{Args, fail};
use coldivf_bench::data::{Fbin, data_root, dist_path, distance, metric_of, subset_truth, write_fbin, write_ibin};
use coldivf_bench::stream::{self, Stream};

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

    if let Some(file) = args.str("stream") {
        let vectors: Vec<&[f32]> = (0..nq).map(|q| queries.row(q)).collect();
        let states = parse_states(args.need("states")).unwrap_or_else(|e| fail(&e.to_string()));
        let prefix = args.need("out-prefix");
        let stream = stream::read(Path::new(file)).unwrap_or_else(|e| fail(&format!("reading {file}: {e}")));
        let start = Instant::now();
        let result = truths(&base, &vectors, metric_of(name), k, threads, &stream, &states)
            .and_then(|best| write_states(prefix, &states, &best, k));

        if let Err(e) = result {
            fail(&e.to_string());
        }

        eprintln!(
            "gt: {nq} queries, {} states of {file}, top {k}, in {:.1} s, to {prefix}-<state>.gt.ibin",
            states.len(),
            start.elapsed().as_secs_f64()
        );
        return;
    }
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

/// A state of a stream: its name, how many of the stream's rounds it holds, and whether it takes their deletes.
#[derive(Clone, Debug, PartialEq, Eq)]
struct State {
    name: String,
    rounds: usize,
    deletes: bool,
}

/// The states of `--states`, eg. "s0=0,s1=1,s1i=1i".
fn parse_states(text: &str) -> Result<Vec<State>, Error> {
    let bad = |item: &str| invalid(&format!("--states takes name=R or name=Ri, not {item}"));
    let mut out: Vec<State> = Vec::new();

    for item in text.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        let (name, value) = item.split_once('=').ok_or_else(|| bad(item))?;
        let (rounds, deletes) = value.strip_suffix('i').map_or((value, true), |r| (r, false));
        let rounds = rounds.parse().map_err(|_| bad(item))?;

        if name.is_empty() || !name.chars().all(|c| c.is_ascii_alphanumeric()) || out.iter().any(|s| s.name == name) {
            return Err(bad(item));
        }

        out.push(State {
            name: name.to_string(),
            rounds,
            deletes,
        });
    }

    if out.is_empty() || out.len() > 16 {
        return Err(invalid("--states takes 1 to 16 states"));
    }

    Ok(out)
}

/// Every row the stream names, sorted, and for each one the states that hold it, one bit per state.
fn masks(stream: &Stream, states: &[State]) -> Result<(Vec<u32>, Vec<u16>), Error> {
    let n = stream.base.len();

    if stream.base.iter().enumerate().any(|(i, &row)| row as usize != i) {
        return Err(invalid("the stream's base must be rows 0 to n in order"));
    }

    let mut rows = stream.base.clone();

    for round in &stream.rounds {
        if !round.upserts.is_empty() {
            return Err(invalid("the stream has upserts, which a state's truth doesn't take"));
        }

        rows.extend(&round.inserts);
    }

    let named = rows.len();
    rows.sort_unstable();
    rows.dedup();

    // Every insert is a row of its own past the base, so the base is the first n of the sorted rows.
    if rows.len() != named {
        return Err(invalid("the stream must insert each row once, and no row of its base"));
    }

    let place = |row: u64| u32::try_from(row).ok().and_then(|r| rows.binary_search(&r).ok());
    let mut out = vec![0u16; rows.len()];

    for (bit, state) in states.iter().enumerate() {
        if state.rounds > stream.rounds.len() {
            return Err(invalid(&format!(
                "state {} holds {} rounds of a stream of {}",
                state.name,
                state.rounds,
                stream.rounds.len()
            )));
        }

        let mut live = vec![false; rows.len()];
        live[..n].fill(true);

        for round in &stream.rounds[..state.rounds] {
            if state.deletes {
                for &id in &round.deletes {
                    let at = place(id)
                        .filter(|&at| live[at])
                        .ok_or_else(|| invalid(&format!("the stream deletes {id}, which isn't live")))?;
                    live[at] = false;
                }
            }

            for &row in &round.inserts {
                if let Some(at) = place(u64::from(row)) {
                    live[at] = true;
                }
            }
        }

        for (mask, _) in out.iter_mut().zip(&live).filter(|(_, l)| **l) {
            *mask |= 1 << bit;
        }
    }

    Ok((rows, out))
}

/// The truth of each state, one list per state and query, from one pass over the rows the stream names. Each row's
/// distance is measured once and offered to every state that holds it.
fn truths(
    base: &Fbin,
    queries: &[&[f32]],
    metric: Metric,
    k: usize,
    threads: usize,
    stream: &Stream,
    states: &[State],
) -> Result<Vec<Vec<Nearest>>, Error> {
    let (rows, masks) = masks(stream, states)?;

    if rows.last().is_some_and(|&last| last as usize >= base.len()) {
        return Err(invalid(&format!(
            "the stream names rows up to {}, but the base holds {}",
            rows[rows.len() - 1],
            base.len()
        )));
    }

    for (bit, state) in states.iter().enumerate() {
        if masks.iter().filter(|&&m| m & (1 << bit) != 0).count() < k {
            return Err(invalid(&format!("state {} holds fewer than {k} vectors", state.name)));
        }
    }

    Ok(scan_states(
        base,
        queries,
        metric,
        k,
        threads,
        &rows,
        &masks,
        states.len(),
    ))
}

/// Each query's `k` nearest rows for each state, nearest first with ties to the lower row, of the rows `rows`, where
/// `masks` says which states hold each one. Each thread takes a share of the queries and scans the rows in tiles, as
/// `scan` does.
#[allow(clippy::too_many_arguments)]
fn scan_states(
    base: &Fbin,
    queries: &[&[f32]],
    metric: Metric,
    k: usize,
    threads: usize,
    rows: &[u32],
    masks: &[u16],
    states: usize,
) -> Vec<Vec<Nearest>> {
    let mut best: Vec<Vec<Nearest>> = vec![vec![Vec::with_capacity(k + 1); states]; queries.len()];
    let per = queries.len().div_ceil(threads).max(1);

    std::thread::scope(|scope| {
        for (group, mine) in best.chunks_mut(per).zip(queries.chunks(per)) {
            scope.spawn(move || {
                for start in (0..masks.len()).step_by(TILE) {
                    let end = (start + TILE).min(masks.len());

                    for (lists, &query) in group.iter_mut().zip(mine) {
                        for (offset, &mask) in masks[start..end].iter().enumerate() {
                            if mask == 0 {
                                continue;
                            }

                            let row = rows[start + offset];
                            let c = (distance(metric, query, base.row(row as usize)), row);
                            let mut left = mask;

                            while left != 0 {
                                offer(&mut lists[left.trailing_zeros() as usize], c, k);
                                left &= left - 1;
                            }
                        }
                    }
                }
            });
        }
    });

    (0..states)
        .map(|s| best.iter().map(|lists| lists[s].clone()).collect())
        .collect()
}

/// Writes each state's truth to `<prefix>-<state>.gt.ibin` and its distances file.
fn write_states(prefix: &str, states: &[State], best: &[Vec<Nearest>], k: usize) -> Result<(), Error> {
    for (state, lists) in states.iter().zip(best) {
        write(&PathBuf::from(format!("{prefix}-{}.gt.ibin", state.name)), lists, k)?;
    }

    Ok(())
}

fn invalid(message: &str) -> Error {
    Error::new(ErrorKind::InvalidInput, message.to_string())
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
