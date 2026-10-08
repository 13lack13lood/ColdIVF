//! Calibration, which picks the cheapest search settings that meet a recall target on sample queries, with 90%
//! confidence. It measures every setting of a grid on each query, orders the settings by how well they did on half of
//! the queries, then tests them in that order on the other half and stops at the first that fails. Testing in a fixed
//! order keeps the chance that any passed setting misses the target at 10% or less, however many settings there are.
//! The cheapest of the settings that passed wins, except that a setting with a reach beats the cheapest fixed one only
//! when that one costs at least 1.1 times as much. A setting's cost is the bytes that a cold search with it reads,
//! counting each rescored row as one page of the index's read backend, since each row is a read of its own.

use std::collections::HashMap;
use std::io::{Error, ErrorKind};

use super::candidates::{Candidate, order};
use super::route::{route, within_reach};
use super::scan::scan_list;
use super::{InPlace, Scratch, View, check_search, invalid, prepare_query};
use crate::index::Inner;
use crate::quantization::rabitq::{BLOCK, block_bytes};
use crate::quantization::{padded_dim, sq8};
use crate::{Calibration, Limits, SearchParams, Target};

/// The chance that the chosen setting misses the target, at most.
const DELTA: f64 = 0.1;

/// How many true nearest neighbors each query comes with. A search's recall is the share of them that it finds.
const TRUTH: usize = 10;

/// The cluster counts that the grid tries, cut at the index's cluster count.
const NPROBES: [usize; 34] = [
    1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, 176, 192, 224,
    256, 320, 384, 512, 768, 1024,
];

/// The rescore depths that the grid tries.
const RERANKS: [usize; 21] = [
    10, 12, 14, 16, 18, 20, 24, 29, 32, 35, 39, 44, 48, 56, 64, 80, 96, 128, 160, 192, 256,
];

/// The caps on the clusters that a setting with a reach reads, as multiples of the cluster count of the cheapest fixed
/// setting that met the target on the first half.
const CAPS: [f64; 3] = [1.0, 1.25, 1.5];

/// The settings with a reach try only the rescore depths up to this many places either side of that fixed setting's.
const NEAR: usize = 2;

/// How much cheaper a setting with a reach must be to win over the cheapest fixed one.
const GAIN: f64 = 1.1;

/// The reaches that the grid tries, 0.01 to 0.6 in steps of 0.01. Where the needed reach usually falls, the clusters
/// within reach grow several times over from one 0.05 to the next, so coarser steps would lose most of the gain.
fn reaches() -> impl Iterator<Item = f32> {
    (1..=60).map(|i| i as f32 / 100.0)
}

/// Picks the cheapest setting that meets `target` on the row-major `queries`, whose true 10 nearest neighbors are in
/// `truth`, 10 to a query.
pub(crate) fn calibrate(index: &Inner, queries: &[f32], truth: &[u32], target: Target) -> Result<Calibration, Error> {
    let tables = index.tables()?;
    let (view, data) = (index.view(tables), index.in_place());
    let dim = view.dim;
    let nq = queries.len() / dim;

    if !queries.len().is_multiple_of(dim) {
        return Err(invalid("the queries' length isn't a multiple of the index's dimension"));
    }

    if nq < 2 || truth.len() != nq * TRUTH {
        return Err(invalid(
            "calibration needs at least 2 queries and 10 true neighbors for each",
        ));
    }

    let alpha = match target {
        Target::MeanRecall { recall_at_10: t } if t > 0.0 && t < 1.0 => 1.0 - f64::from(t),
        Target::MeanRecall { .. } => return Err(invalid("a mean target's recall must be above 0 and below 1")),
        Target::QueryShare { recall_at_10: t, share } if t > 0.0 && t <= 1.0 && share > 0.0 && share < 1.0 => {
            1.0 - f64::from(share)
        }
        Target::QueryShare { .. } => {
            return Err(invalid(
                "a share target's recall must be above 0 and at most 1, and its share above 0 and below 1",
            ));
        }
    };

    // The fixed settings first. The cheapest that meets the target on the first half sets the caps and the rescore
    // depths of the settings with a reach.
    let nlist = view.lists.len();
    let mut nprobes: Vec<usize> = NPROBES.iter().copied().filter(|&n| n < nlist).collect();
    nprobes.push(nlist.min(NPROBES[NPROBES.len() - 1]));
    let fixed: Vec<Calibration> = nprobes
        .iter()
        .flat_map(|&nprobe| {
            RERANKS.iter().map(move |&rerank| Calibration {
                nprobe,
                rerank,
                reach: None,
            })
        })
        .collect();
    let mut eval = Evaluator::new(&view, &data, nlist, index.read_page());
    let fixed_runs = eval.run_all(queries, truth, &fixed)?;
    let cheapest = (0..fixed.len())
        .filter(|&s| fixed_runs.mean(s, 0, |r| loss(target, r)) <= alpha)
        .min_by(|&a, &b| fixed_runs.cost(a, 0).total_cmp(&fixed_runs.cost(b, 0)));
    let base = cheapest.map_or(fixed[fixed.len() - 1], |s| fixed[s]);
    let reached = reach_grid(&base, nlist);
    let reached_runs = eval.run_all(queries, truth, &reached)?;
    let settings: Vec<Setting> = (0..fixed.len())
        .map(|s| (fixed[s], &fixed_runs, s))
        .chain((0..reached.len()).map(|s| (reached[s], &reached_runs, s)))
        .collect();

    select(&settings, target, alpha)
        .ok_or_else(|| invalid("no setting meets the target with 90% confidence on these queries"))
}

/// The settings with a reach that the grid tries, given `base`, the cheapest fixed setting that met the target on the
/// first half. Each reads at most `base.nprobe` times one of the caps, cut at `nlist`, and rescores to a depth near
/// `base.rerank`.
fn reach_grid(base: &Calibration, nlist: usize) -> Vec<Calibration> {
    let at = RERANKS.partition_point(|&r| r < base.rerank);
    let reranks = &RERANKS[at.saturating_sub(NEAR)..(at + NEAR + 1).min(RERANKS.len())];
    let mut caps: Vec<usize> = CAPS
        .iter()
        .map(|&c| ((base.nprobe as f64 * c).ceil() as usize).min(nlist))
        .collect();
    caps.dedup();

    caps.into_iter()
        .flat_map(|cap| {
            reaches().flat_map(move |reach| {
                reranks.iter().map(move |&rerank| Calibration {
                    nprobe: cap,
                    rerank,
                    reach: Some(reach),
                })
            })
        })
        .collect()
}

/// One setting of the grid, with the runs that hold its recall and cost and its place in them.
type Setting<'a> = (Calibration, &'a Runs, usize);

/// Tests `settings` in order of their mean loss on the first half, and among equal losses costliest first, since the
/// costlier setting is likelier to pass. It stops at the first that fails on the second half. Of those that passed,
/// the cheapest over every query wins, but a setting with a reach must cost at least 1.1 times less than the cheapest
/// fixed one.
fn select(settings: &[Setting], target: Target, alpha: f64) -> Option<Calibration> {
    let keys: Vec<(f64, f64)> = settings
        .iter()
        .map(|&(_, runs, s)| (runs.mean(s, 0, |r| loss(target, r)), runs.cost(s, 0)))
        .collect();
    let mut sequence: Vec<usize> = (0..settings.len()).collect();
    sequence.sort_by(|&a, &b| {
        let ((loss_a, cost_a), (loss_b, cost_b)) = (keys[a], keys[b]);
        loss_a
            .total_cmp(&loss_b)
            .then(cost_b.total_cmp(&cost_a))
            .then(a.cmp(&b))
    });
    let mut passed = Vec::new();

    for i in sequence {
        let (_, runs, s) = settings[i];
        let losses: Vec<f64> = runs.half(s, 1).map(|r| loss(target, r)).collect();
        if p_value(target, &losses, alpha) > DELTA {
            break;
        }
        passed.push(i);
    }

    let cost = |i: usize| settings[i].1.cost(settings[i].2, 2);
    let best = |reach: bool| {
        passed
            .iter()
            .copied()
            .filter(|&i| settings[i].0.reach.is_some() == reach)
            .min_by(|&a, &b| cost(a).total_cmp(&cost(b)))
    };

    match (best(false), best(true)) {
        (Some(f), Some(r)) if cost(r) * GAIN <= cost(f) => Some(settings[r].0),
        (Some(f), _) => Some(settings[f].0),
        (None, r) => r.map(|r| settings[r].0),
    }
}

/// A query's loss under `target`, for a search that found `recall` of its true neighbors. A setting meets the target
/// when the mean loss over all queries is at most 1 minus the target's recall or share. For a mean target the loss is
/// the share of true neighbors missed, and for a share target it is 1 when the query falls short and 0 otherwise.
fn loss(target: Target, recall: f32) -> f64 {
    match target {
        Target::MeanRecall { .. } => 1.0 - f64::from(recall),
        Target::QueryShare { recall_at_10, .. } => f64::from(u8::from(recall < recall_at_10)),
    }
}

/// How likely `losses` this low would be if the setting's true mean loss were `alpha` or more.
fn p_value(target: Target, losses: &[f64], alpha: f64) -> f64 {
    match target {
        Target::MeanRecall { .. } => bounded_p(losses, alpha),
        Target::QueryShare { .. } => binomial_p(losses, alpha),
    }
}

/// Every setting's recall and cost on every query, in query order.
struct Runs {
    settings: usize,
    recall: Vec<f32>,
    /// The bytes that the setting reads for the query.
    bytes: Vec<f64>,
}

impl Runs {
    /// The recalls of setting `s` on the even queries for half 0 and the odd ones for half 1.
    fn half(&self, s: usize, half: usize) -> impl Iterator<Item = f32> + '_ {
        self.recall
            .chunks_exact(self.settings)
            .skip(half)
            .step_by(2)
            .map(move |r| r[s])
    }

    /// The mean of `f` over setting `s`'s recalls on half `half`.
    fn mean(&self, s: usize, half: usize, f: impl Fn(f32) -> f64) -> f64 {
        let (sum, n) = self.half(s, half).fold((0.0, 0), |(sum, n), r| (sum + f(r), n + 1));
        sum / f64::from(n.max(1))
    }

    /// The mean bytes that setting `s` reads, on half 0, half 1, or every query for any other `half`.
    fn cost(&self, s: usize, half: usize) -> f64 {
        let rows = self.bytes.chunks_exact(self.settings).enumerate();
        let picked = rows.filter(|(q, _)| half > 1 || q % 2 == half).map(|(_, b)| b[s]);
        let (sum, n) = picked.fold((0.0, 0), |(sum, n), b| (sum + b, n + 1));
        sum / f64::from(n.max(1))
    }
}

/// Runs the real search pipeline of an index once per query and reads off the recall of many settings.
struct Evaluator<'a> {
    view: &'a View<'a>,
    data: &'a InPlace<'a>,
    s: Scratch,
    /// The bytes that rescoring one candidate costs, one page of the index's read backend.
    row_cost: f64,
    /// The candidates after each cluster count that a setting reads, best first.
    snapshots: HashMap<usize, Vec<Candidate>>,
    /// The query's distance to each candidate's 8-bit row, kept for this query.
    exact: HashMap<u32, f32>,
}

impl<'a> Evaluator<'a> {
    fn new(view: &'a View<'a>, data: &'a InPlace<'a>, nlist: usize, row_cost: usize) -> Evaluator<'a> {
        let limits = Limits {
            max_k: TRUTH,
            max_nprobe: nlist,
            max_rerank: RERANKS[RERANKS.len() - 1],
        };
        Evaluator {
            view,
            data,
            s: Scratch::new(view.dim, nlist, &limits),
            row_cost: row_cost as f64,
            snapshots: HashMap::new(),
            exact: HashMap::new(),
        }
    }

    fn run_all(&mut self, queries: &[f32], truth: &[u32], settings: &[Calibration]) -> Result<Runs, Error> {
        let n = settings.len();
        let mut runs = Runs {
            settings: n,
            recall: Vec::with_capacity(n * truth.len() / TRUTH),
            bytes: Vec::with_capacity(n * truth.len() / TRUTH),
        };

        for (q, t) in queries.chunks_exact(self.view.dim).zip(truth.as_chunks::<TRUTH>().0) {
            self.run(q, t, settings, &mut runs)?;
        }

        Ok(runs)
    }

    /// Searches `query` once, reading clusters nearest first, and appends each setting's recall and cost to `runs`.
    fn run(&mut self, query: &[f32], truth: &[u32], settings: &[Calibration], runs: &mut Runs) -> Result<(), Error> {
        let (view, s) = (self.view, &mut self.s);
        let most = settings.iter().map(|c| c.nprobe).max().unwrap_or(1);
        let params = SearchParams {
            k: TRUTH,
            nprobe: most,
            rerank: RERANKS[RERANKS.len() - 1],
            reach: None,
        };
        let limits = Limits {
            max_k: TRUTH,
            max_nprobe: most,
            max_rerank: params.rerank,
        };
        check_search(query, view.dim, &params, &limits)?;

        prepare_query(view, query, s)?;
        route(
            view.tier,
            &view.centers,
            &s.q8,
            s.lut.k.t2,
            most,
            &mut s.dots,
            &mut s.routes,
        );

        // Each setting reads the first clusters of the route, so one scan in route order serves them all.
        let reads: Vec<usize> = settings
            .iter()
            .map(|c| within_reach(&s.routes[..c.nprobe.min(s.routes.len())], c.reach))
            .collect();
        let mut points = reads.clone();
        points.sort_unstable();
        points.dedup();
        s.candidates.reset(params.rerank);
        self.snapshots.clear();
        let size = block_bytes(padded_dim(view.dim));
        let (mut scanned, mut bytes) = (0, vec![0f64; s.routes.len() + 1]);

        for &point in &points {
            while scanned < point {
                let list = view.lists[s.routes[scanned].id as usize];
                let len = list.count.div_ceil(BLOCK) * size;
                scan_list(
                    view.tier,
                    &self.data.blocks[list.offset..list.offset + len],
                    list.count,
                    list.first_id,
                    &s.lut,
                    &mut s.candidates,
                    &mut s.survivors,
                );
                bytes[scanned + 1] = bytes[scanned] + len as f64;
                scanned += 1;
            }
            let mut snapshot = s.candidates.as_slice().to_vec();
            snapshot.sort_unstable_by(order);
            self.snapshots.insert(point, snapshot);
        }

        let norm2 = s.offset.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>() as f32;
        let base = runs.recall.len();
        runs.recall.resize(base + settings.len(), 0.0);
        runs.bytes.resize(base + settings.len(), 0.0);

        // Settings that read the same clusters differ only in how deep they rescore, so one walk down the candidates,
        // best estimate first, keeps the 10 nearest so far and reads off every depth on the way.
        for &point in &points {
            let mut depths: Vec<(usize, usize)> = (0..settings.len())
                .filter(|&i| reads[i] == point)
                .map(|i| (settings[i].rerank.max(TRUTH), i))
                .collect();
            depths.sort_unstable();
            let (mut best, mut next): (Vec<(Candidate, u64)>, usize) = (Vec::with_capacity(TRUTH + 1), 0);
            for (seen, cand) in self.snapshots[&point].iter().enumerate() {
                let row = self.data.rows.row(cand.id);
                let score = *self
                    .exact
                    .entry(cand.id)
                    .or_insert_with(|| sq8::distance(row, &s.offset, norm2));
                if score.is_nan() {
                    return Err(Error::new(ErrorKind::InvalidData, "a vector's 8-bit row is damaged"));
                }
                let c = Candidate { score, id: cand.id };
                let at = best.partition_point(|b| order(&b.0, &c).is_lt());
                if at < TRUTH {
                    best.insert(at, (c, sq8::id(row, view.dim)));
                    best.truncate(TRUTH);
                }
                while next < depths.len() && depths[next].0 <= seen + 1 {
                    let found = best
                        .iter()
                        .filter(|(_, id)| truth.iter().any(|&t| u64::from(t) == *id))
                        .count();
                    runs.recall[base + depths[next].1] = found as f32 / TRUTH as f32;
                    next += 1;
                }
            }
            // A depth past the candidates there are sees them all.
            let found = best
                .iter()
                .filter(|(_, id)| truth.iter().any(|&t| u64::from(t) == *id))
                .count();
            for &(_, i) in &depths[next..] {
                runs.recall[base + i] = found as f32 / TRUTH as f32;
            }
            for &(depth, i) in &depths {
                runs.bytes[base + i] = bytes[point] + depth as f64 * self.row_cost;
            }
        }

        self.exact.clear();
        Ok(())
    }
}

/// How likely losses with a mean this low would be if the setting's true mean loss were `alpha` or more, for losses
/// each from 0 to 1. It takes the smaller of two standard bounds on that chance. Hoeffding's comes from how far the
/// mean falls below `alpha`. Bentkus's is e (about 2.718) times the chance that at most the summed loss, rounded up,
/// of `losses.len()` trials succeed, each with chance `alpha`.
fn bounded_p(losses: &[f64], alpha: f64) -> f64 {
    let n = losses.len() as f64;
    let risk = losses.iter().sum::<f64>() / n;

    if risk >= alpha {
        return 1.0;
    }

    let hoeffding = (-n * kl(risk, alpha)).exp();
    let bentkus = std::f64::consts::E * binomial_cdf((n * risk).ceil() as usize, losses.len(), alpha);
    hoeffding.min(bentkus).min(1.0)
}

/// How likely this few failures among `losses`, each 0 or 1, would be if each query failed with chance `alpha`.
fn binomial_p(losses: &[f64], alpha: f64) -> f64 {
    let failures = losses.iter().filter(|&&l| l > 0.0).count();
    binomial_cdf(failures, losses.len(), alpha)
}

/// The chance that at most `x` of `n` trials succeed, when each succeeds with chance `p` on its own.
fn binomial_cdf(x: usize, n: usize, p: f64) -> f64 {
    let (lp, lq) = (p.ln(), (1.0 - p).ln());
    let ln_n = libm::lgamma(n as f64 + 1.0);
    let sum: f64 = (0..=x.min(n))
        .map(|k| {
            let (k, m) = (k as f64, (n - k) as f64);
            (ln_n - libm::lgamma(k + 1.0) - libm::lgamma(m + 1.0) + k * lp + m * lq).exp()
        })
        .sum();
    sum.min(1.0)
}

/// How far chance `a` of a yes-or-no outcome is from chance `b`, measured as the Kullback-Leibler divergence
/// a·ln(a/b) + (1 − a)·ln((1 − a)/(1 − b)), where ln is the natural log.
fn kl(a: f64, b: f64) -> f64 {
    let term = |x: f64, y: f64| if x == 0.0 { 0.0 } else { x * (x / y).ln() };
    term(a, b) + term(1.0 - a, 1.0 - b)
}

#[cfg(test)]
#[path = "../../tests/unit/search/calibrate.rs"]
mod tests;
