//! Exact nearest neighbors over the live vectors, kept up to date as a stream changes them, to judge searches by. Each
//! query keeps its nearest live vectors, a few dozen of them. A round drops the ones it deletes or replaces and offers
//! the ones it puts, and a query left with fewer than 10 is searched again over every live vector.

use std::collections::{HashMap, HashSet};

use coldivf::Metric;

use crate::data::{Fbin, distance};
use crate::stream::Change;

/// The nearest neighbors that a search should find, and so the fewest that a query may keep before it is searched
/// again.
pub const K: usize = 10;

/// Candidates in one tile of an exact search, which each query of a thread visits while they stay in the cache.
const TILE: usize = 2048;

/// A distance and an id, ranked by the distance, then by the id.
pub type Near = (f32, u64);

fn before(a: Near, b: Near) -> bool {
    a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)).is_lt()
}

/// Keeps the `k` first pairs of `best`, in order, with `c` offered.
fn offer(best: &mut Vec<Near>, c: Near, k: usize) {
    if best.len() >= k && best.last().is_none_or(|&last| !before(c, last)) {
        return;
    }

    let at = best.partition_point(|&b| before(b, c));
    best.insert(at, c);
    best.truncate(k);
}

/// The `k` nearest of `candidates` to each row-major query of `queries`, as (distance, key) pairs, nearest first. A
/// candidate is a key and the row of `data` that holds its vector, and `metric` measures the distance, as
/// `data::distance` does. Each of `threads` threads takes a share of the queries.
pub fn nearest(
    data: &Fbin,
    metric: Metric,
    queries: &[f32],
    candidates: &[(u64, u32)],
    k: usize,
    threads: usize,
) -> Vec<Vec<Near>> {
    let dim = data.dim();
    let mut best: Vec<Vec<Near>> = vec![Vec::with_capacity(k + 1); queries.len() / dim];
    let per = best.len().div_ceil(threads.max(1)).max(1);

    std::thread::scope(|scope| {
        for (group, mine) in best.chunks_mut(per).zip(queries.chunks(per * dim)) {
            scope.spawn(move || {
                for tile in candidates.chunks(TILE) {
                    for (b, query) in group.iter_mut().zip(mine.chunks_exact(dim)) {
                        for &(key, row) in tile {
                            offer(b, (distance(metric, query, data.row(row as usize)), key), k);
                        }
                    }
                }
            });
        }
    });

    best
}

/// Each query's nearest live vectors, by index id.
pub struct Truth {
    metric: Metric,
    dim: usize,
    keep: usize,
    queries: Vec<f32>,
    /// Each query's nearest live vectors, nearest first. They are exactly its nearest, as many as it holds.
    best: Vec<Vec<Near>>,
    /// Whether a query's list holds every live vector, so that any new one belongs in it.
    whole: Vec<bool>,
    /// The queries searched again over every live vector, since the first search.
    pub redone: usize,
}

impl Truth {
    /// Searches each row-major query of `queries` over `live`, every live vector as its id and its row of `data`, and
    /// keeps its `keep` nearest, at least 10.
    pub fn new(
        data: &Fbin,
        metric: Metric,
        queries: Vec<f32>,
        keep: usize,
        live: &[(u64, u32)],
        threads: usize,
    ) -> Truth {
        let keep = keep.max(K);
        let best = nearest(data, metric, &queries, live, keep, threads);
        Truth {
            metric,
            dim: data.dim(),
            keep,
            whole: vec![live.len() <= keep; best.len()],
            queries,
            best,
            redone: 0,
        }
    }

    /// Applies `changes` in order, where `live` holds every live vector after them, as in `new`.
    pub fn update(&mut self, data: &Fbin, changes: &[Change], live: &[(u64, u32)], threads: usize) {
        // Where each changed id ends up: its last change's row, or gone.
        let mut last: HashMap<u64, Option<u32>> = HashMap::new();

        for change in changes {
            match *change {
                Change::Put { id, row } => last.insert(id, Some(row)),
                Change::Delete { id } => last.insert(id, None),
            };
        }

        let changed: HashSet<u64> = last.keys().copied().collect();
        let mut puts: Vec<(u64, u32)> = last.iter().filter_map(|(&id, &row)| Some((id, row?))).collect();
        puts.sort_unstable();
        let (dim, metric, keep) = (self.dim, self.metric, self.keep);
        let per = self.best.len().div_ceil(threads.max(1)).max(1);
        let parts = self.best.chunks_mut(per).zip(self.whole.chunks_mut(per));

        std::thread::scope(|scope| {
            for ((best, whole), mine) in parts.zip(self.queries.chunks(per * dim)) {
                let (changed, puts) = (&changed, &puts);
                scope.spawn(move || {
                    for b in best.iter_mut() {
                        b.retain(|&(_, id)| !changed.contains(&id));
                    }

                    for tile in puts.chunks(TILE) {
                        let lists = best.iter_mut().zip(whole.iter_mut());

                        for ((b, w), query) in lists.zip(mine.chunks_exact(dim)) {
                            for &(id, row) in tile {
                                let c = (distance(metric, query, data.row(row as usize)), id);
                                offer_live(b, w, c, keep);
                            }
                        }
                    }
                });
            }
        });

        let short: Vec<usize> = (0..self.best.len())
            .filter(|&q| self.best[q].len() < K && !self.whole[q])
            .collect();

        if short.is_empty() {
            return;
        }

        let queries: Vec<f32> = short
            .iter()
            .flat_map(|&q| self.queries[q * dim..(q + 1) * dim].iter().copied())
            .collect();

        for (q, b) in short.iter().zip(nearest(data, metric, &queries, live, keep, threads)) {
            self.best[*q] = b;
            self.whole[*q] = live.len() <= keep;
        }

        self.redone += short.len();
    }

    /// How many queries there are.
    pub fn len(&self) -> usize {
        self.best.len()
    }

    pub fn is_empty(&self) -> bool {
        self.best.is_empty()
    }

    /// Query `q`'s vector.
    pub fn query(&self, q: usize) -> &[f32] {
        &self.queries[q * self.dim..(q + 1) * self.dim]
    }

    /// Query `q`'s nearest live vectors, nearest first, at least 10 of them unless fewer are live.
    pub fn nearest(&self, q: usize) -> &[Near] {
        &self.best[q]
    }

    /// Query `q`'s recall@10 of `found`, its hits nearest first, mapped to index ids. It is the share of its true 10
    /// nearest that the first 10 hits hold. A hit outside them still counts when it is at most as far as the true 10th,
    /// eg. a copy of it, measured from the vector that `vector` gives for its id. An id found twice counts once.
    pub fn recall<'a>(&self, q: usize, found: &[u64], vector: impl Fn(u64) -> Option<&'a [f32]>) -> f64 {
        let truth = &self.best[q][..K.min(self.best[q].len())];
        let Some(&(kth, _)) = truth.last() else {
            return f64::NAN;
        };
        let found = &found[..K.min(found.len())];
        let hits = found
            .iter()
            .enumerate()
            .filter(|&(i, id)| !found[..i].contains(id))
            .filter(|&(_, &id)| {
                truth.iter().any(|&(_, t)| t == id)
                    || vector(id).is_some_and(|v| distance(self.metric, self.query(q), v) <= kth)
            })
            .count();
        hits as f64 / K as f64
    }
}

/// Offers `c` to a list that holds exactly a query's nearest live vectors. It goes in only where the list stays exact:
/// when the list holds every live vector, or when `c` comes before its last entry.
fn offer_live(best: &mut Vec<Near>, whole: &mut bool, c: Near, keep: usize) {
    if !*whole && best.last().is_none_or(|&last| !before(c, last)) {
        return;
    }

    let at = best.partition_point(|&b| before(b, c));
    best.insert(at, c);

    if best.len() > keep {
        best.truncate(keep);
        *whole = false;
    }
}

#[cfg(test)]
#[path = "../tests/unit/truth.rs"]
mod tests;
