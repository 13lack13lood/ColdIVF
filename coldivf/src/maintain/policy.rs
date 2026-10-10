//! When maintenance runs. Runs start from what changes cost searches: inserts that wait in memory, vectors that
//! changes hid, and the extra scanning they cause, or from an idle index with half as much to do. Regions that are due
//! only steer what a run clusters again, so a level alone never starts one. A minimum interval and a write budget over
//! the last 24 hours hold a due run back, and when the budget does so while inserts pile up, maintenance is deferred.
//!
//! Each finished run adds a line to the folder's `runs` file: when it ended, how long it took, what it wrote and
//! measured again, its processor time, and, when runs measure it, the read-cost indicator after it. The indicator
//! compares the bytes that a query reads for 95% recall with what a fresh clustering of as many vectors would read.

use std::fs;
use std::io::Error;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::build::rewrite::Gate;
use crate::format::{self, Layout, Part, read_tables};
use crate::io;
use crate::quantization::rabitq::{BLOCK, block_bytes, code_bytes};
use crate::quantization::{padded_dim, rotate_offset, sq4, sq8};
use crate::root;
use crate::update::view::REBUILD;
use crate::{MaintainParams, RunKind, RunReport, StartParams};

/// The most lines that the `runs` file keeps.
const KEPT: usize = 64;

/// The sample rows that the indicator takes as queries.
const SAMPLES: usize = 500;

/// The nearest rows of each sample that the indicator's recall counts.
const NEAREST: usize = 10;

/// The recall that the indicator's bytes per query reach.
const RECALL: f64 = 0.95;

/// One finished run, as the `runs` file records it.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct Ran {
    /// When it ended, in milliseconds since 1970, and how long it took.
    pub(crate) ended_ms: u64,
    pub(crate) wall_ms: u64,
    /// What it wrote, its threads' processor time in microseconds, the codes it measured again, and whether it was a
    /// major run.
    pub(crate) bytes_written: u64,
    pub(crate) cpu_us: u64,
    pub(crate) reencoded: u64,
    pub(crate) major: bool,
    /// The vector count after it, and when runs measure them, the bytes per query for 95% recall and the indicator.
    pub(crate) count: u64,
    pub(crate) cost: Option<f64>,
    pub(crate) indicator: Option<f32>,
}

impl Ran {
    /// What `report` says of a run that ended at `ended_ms`.
    pub(crate) fn of(report: &RunReport, ended_ms: u64, cost: Option<f64>) -> Ran {
        Ran {
            ended_ms,
            wall_ms: report.wall.as_millis() as u64,
            bytes_written: report.bytes_written,
            cpu_us: report.cpu.as_micros() as u64,
            reencoded: report.reencoded as u64,
            major: report.kind == RunKind::Major,
            count: report.count as u64,
            cost,
            indicator: report.indicator,
        }
    }

    fn line(&self) -> String {
        let opt = |x: Option<String>| x.unwrap_or_else(|| "-".to_string());
        format!(
            "{} {} {} {} {} {} {} {} {}\n",
            self.ended_ms,
            self.wall_ms,
            self.bytes_written,
            self.cpu_us,
            self.reencoded,
            u8::from(self.major),
            self.count,
            opt(self.cost.map(|c| c.to_string())),
            opt(self.indicator.map(|i| i.to_string())),
        )
    }

    fn parse(line: &str) -> Option<Ran> {
        let w: Vec<&str> = line.split_whitespace().collect();
        let n = |i: usize| w.get(i)?.parse::<u64>().ok();
        let opt = |i: usize| w.get(i).filter(|s| **s != "-").map(|s| s.parse::<f64>().ok());

        if w.len() != 9 {
            return None;
        }

        Some(Ran {
            ended_ms: n(0)?,
            wall_ms: n(1)?,
            bytes_written: n(2)?,
            cpu_us: n(3)?,
            reencoded: n(4)?,
            major: n(5)? == 1,
            count: n(6)?,
            cost: match opt(7) {
                None => None,
                Some(c) => Some(c?),
            },
            indicator: match opt(8) {
                None => None,
                Some(i) => Some(i? as f32),
            },
        })
    }
}

/// The `runs` file of the index whose folder is `dir`.
fn runs_path(dir: &Path) -> PathBuf {
    dir.join("runs")
}

/// The runs that the index at `path` recorded, oldest first. A line that a crash cut short counts as no run.
pub(crate) fn runs(path: &Path) -> Vec<Ran> {
    let text = fs::read_to_string(runs_path(&root::folder(path))).unwrap_or_default();
    text.lines().filter_map(Ran::parse).collect()
}

/// Adds `ran` to the runs of the index at `path`, keeping the last `KEPT`. It is best effort, since a run that goes
/// unrecorded only lets the next one start sooner.
pub(crate) fn record(path: &Path, ran: &Ran) {
    let mut all = runs(path);
    all.push(*ran);
    let start = all.len().saturating_sub(KEPT);
    let text: String = all[start..].iter().map(Ran::line).collect();
    let _ = fs::write(runs_path(&root::folder(path)), text);
}

/// Milliseconds since 1970 now.
pub(crate) fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as u64)
}

/// What the runs of the last 24 hours before `now_ms` wrote, measured again and used of the processor.
pub(crate) fn last_day(runs: &[Ran], now_ms: u64) -> (u64, u64, Duration) {
    let day = runs.iter().filter(|r| r.ended_ms + 86_400_000 > now_ms);
    day.fold((0, 0, Duration::ZERO), |(b, e, c), r| {
        (
            b + r.bytes_written,
            e + r.reencoded,
            c + Duration::from_micros(r.cpu_us),
        )
    })
}

/// The inserts that start a run: `pending_share` of the `count` vectors, or as many as the memory budget holds, if
/// fewer, as `pending_bytes` counts them for vectors of `dim` dimensions in `nlist` clusters. At least 1.
pub(crate) fn soft_limit(start: &StartParams, count: usize, dim: usize, nlist: usize) -> usize {
    // Shares are taken in f32, the type they are given in, as the regions' due marks take theirs.
    let by_share = ((start.pending_share * count as f32) as usize).max(1);
    let fits = |n: usize| pending_bytes(n, nlist, dim) <= start.ram_budget as f64;

    if fits(by_share) {
        return by_share;
    }

    // The bytes grow with the inserts, so halving finds the most that fit.
    let (mut lo, mut hi) = (0, by_share);

    while lo + 1 < hi {
        let mid = lo + (hi - lo) / 2;
        match fits(mid) {
            true => lo = mid,
            false => hi = mid,
        }
    }

    lo.max(1)
}

/// The bytes of memory that `pending` inserts of `dim` dimensions take, in an index of `nlist` clusters, when they
/// land in its clusters at random. Each one holds its row, its code and factors, its sequence number, cluster and
/// stamp, and its id's entry, and the sorted view holds its code and factors again, unpacked. The sorted view also
/// holds every cluster's inserts in whole blocks of 32 codes, each with the places of its lanes, so a cluster with one
/// insert takes as many bytes there as one with 32. The inserts since the view's last rebuild, up to `REBUILD` of them,
/// take a block each at most.
pub(crate) fn pending_bytes(pending: usize, nlist: usize, dim: usize) -> f64 {
    let d_pad = padded_dim(dim);
    let each = sq8::row_bytes(dim) + 2 * (code_bytes(d_pad) + 16) + 8 + 4 + 8 + 16;
    let block = (block_bytes(d_pad) + 4 * BLOCK) as f64;
    let nlist = nlist.max(1) as f64;
    let blocks = nlist * blocks_per_cluster(pending as f64 / nlist) + pending.min(REBUILD) as f64;
    pending as f64 * each as f64 + blocks * block
}

/// The blocks of 32 that a cluster's inserts fill on average, when the cluster holds `mean` of them on average and its
/// count follows a Poisson distribution, as inserts that land in clusters at random give. It is the sum over k ≥ 0 of
/// the chance that the cluster holds more than 32k. Far above 32 a cluster fills about `mean`/32 blocks and half a
/// block more.
fn blocks_per_cluster(mean: f64) -> f64 {
    if mean >= 64.0 {
        return mean / BLOCK as f64 + 0.5;
    }

    let (mut chance, mut below, mut blocks) = ((-mean).exp(), 0.0, 0.0);

    // `chance` is the chance of a count of exactly i, and `below` of i or fewer.
    for i in 0..=(mean + 10.0 * mean.sqrt()) as usize + BLOCK {
        below += chance;
        if i % BLOCK == 0 {
            blocks += (1.0 - below).max(0.0);
        }
        chance *= mean / (i + 1) as f64;
    }

    blocks
}

/// What the index looks like to the policy now.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Now {
    /// The vectors that searches see, the rows of the data file, the inserts waiting in the log and the rows that
    /// changes hid.
    pub(crate) count: usize,
    pub(crate) rows: usize,
    pub(crate) pending: usize,
    pub(crate) deleted: usize,
    pub(crate) regions_due: usize,
    pub(crate) major_due: bool,
    /// The dimension, and the clusters that the pending inserts wait in.
    pub(crate) dim: usize,
    pub(crate) nlist: usize,
    /// The data file's bytes, which a run writes about as many of.
    pub(crate) file_bytes: u64,
    /// The share more that searches scanned, over the searches counted since the last run, and how many those were.
    pub(crate) excess: f64,
    pub(crate) queries: u64,
    /// How long no search ran, and the time, in milliseconds since 1970.
    pub(crate) idle: Duration,
    pub(crate) now_ms: u64,
}

/// What the policy says.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Decision {
    /// The run to start, if one is due and nothing holds it back.
    pub(crate) start: Option<RunKind>,
    /// Whether a run is due.
    pub(crate) due: bool,
    /// Whether the write budget holds a due run back while the inserts reach twice what starts one.
    pub(crate) deferred: bool,
}

/// Whether a run is due for `at`, with the settings `params` and the past `runs`, and whether its interval and the
/// write budget let it start now.
pub(crate) fn decide(params: &MaintainParams, at: &Now, runs: &[Ran]) -> Decision {
    let s = &params.start;
    let soft = soft_limit(s, at.count, at.dim, at.nlist);
    let excess = at.queries >= s.excess_queries as u64 && at.excess as f32 >= s.read_excess;
    let idle = s
        .idle
        .is_some_and(|i| at.idle >= i && (2 * at.pending >= soft || at.regions_due > 0));
    let dead = at.deleted as f32 >= s.dead_share * at.rows.max(1) as f32;
    let major = at.major_due || indicator_due(params, runs);
    let due = at.pending >= soft || dead || excess || idle || major;

    if !due {
        return Decision {
            start: None,
            due,
            deferred: false,
        };
    }

    // The interval runs from the end of the last run, and grows with the run's length.
    let waits = runs.last().is_some_and(|last| {
        let gap = (s.min_interval.as_millis() as u64).max(10 * last.wall_ms);
        at.now_ms < last.ended_ms + gap
    });
    let (written, _, _) = last_day(runs, at.now_ms);
    let budget = f64::from(s.write_budget) * at.file_bytes as f64;
    let over = (written + at.file_bytes) as f64 > budget;
    let start = (!waits && !over).then_some(if major { RunKind::Major } else { RunKind::Auto });

    Decision {
        start,
        due,
        deferred: over && at.pending >= 2 * soft,
    }
}

/// Whether the read-cost indicator calls for a major run: it is turned on, and it stayed over its threshold after the
/// last two runs, which were minor ones.
fn indicator_due(params: &MaintainParams, runs: &[Ran]) -> bool {
    let Some(threshold) = params.major.indicator else {
        return false;
    };
    let last: Vec<&Ran> = runs.iter().rev().take(2).collect();
    let over = |r: &&Ran| !r.major && r.indicator.is_some_and(|i| i > threshold);
    last.len() == 2 && last.iter().all(over)
}

/// The read-cost indicator of the data file at `data`, after a run of `kind`, against the `runs` before it: the bytes
/// of codes that a query reads for 95% mean recall of its 10 nearest rows, with exact distances inside the clusters it
/// reads, over the same after the last major run, grown by the vector count to the power `beta`. The queries are 500
/// rows of the file spread over it, each with its true nearest rows among the others by their 8-bit copies. Every read
/// of the file waits for `gate`. Returns the bytes per query and the indicator, which is 1 with no major run before.
pub(crate) fn indicator(
    data: &Path,
    kind: RunKind,
    runs: &[Ran],
    beta: f32,
    gate: &dyn Gate,
) -> Result<(f64, f32), Error> {
    let layout = root::header_of(data)?;
    let cost = bytes_for_recall(data, &layout, gate)?;
    let base = runs.iter().rev().find(|r| r.major && r.cost.is_some());
    let indicator = match (kind, base) {
        (RunKind::Major, _) | (_, None) => 1.0,
        (_, Some(b)) => {
            let grown = (layout.count as f64 / b.count.max(1) as f64).powf(f64::from(beta));
            cost / (b.cost.unwrap_or(cost) * grown).max(f64::MIN_POSITIVE)
        }
    };
    Ok((cost, indicator as f32))
}

/// The bytes of codes that a sample query of the data file at `data`, whose header says `layout`, reads for 95% mean
/// recall, reading clusters nearest center first. It reads the rows a few megabytes at a time, and holds only each
/// sample's nearest rows so far.
fn bytes_for_recall(data: &Path, layout: &Layout, gate: &dyn Gate) -> Result<f64, Error> {
    let file = fs::File::open(data)?;
    let read = |offset: usize, len: usize| -> Result<Vec<u8>, Error> {
        let mut bytes = vec![0u8; len];
        gate.read(len)?;
        io::read_at(&file, &mut bytes, offset)?;
        Ok(bytes)
    };
    let span = |part: Part| layout.span(part);
    let tables = read_tables(
        read(format::tables_span(layout).offset, format::tables_span(layout).len)?,
        layout,
    )?;
    let (rotation, mean) =
        format::read_rotation(&read(span(Part::Rotation).offset, span(Part::Rotation).len)?, layout)?;
    let (dim, n, page) = (layout.dim, layout.count, layout.page_size);
    let (rb, per_page) = (sq8::row_bytes(dim), page / sq8::row_bytes(dim));
    let at_row = |r: usize| span(Part::Rows).offset + r / per_page * page + r % per_page * rb;
    let wanted = SAMPLES.min(n);
    let samples: Vec<usize> = (0..wanted).map(|i| i * n / wanted.max(1)).collect();

    // Each sample's offset from the mean, its squared length and its nearest rows so far, by their decoded copies.
    let mut queries = Vec::with_capacity(samples.len());

    for &s in &samples {
        let (mut q, row) = (vec![0f32; dim], read(at_row(s), rb)?);
        sq8::decode(&row, &mean, &mut q);
        let off: Vec<f32> = q.iter().zip(&mean).map(|(&x, &m)| x - m).collect();
        let norm2: f32 = off.iter().map(|x| x * x).sum();
        queries.push((s, q, off, norm2, Vec::<(f32, usize)>::with_capacity(NEAREST + 1)));
    }

    let pages_at_once = ((4 << 20) / page).max(1);

    for first in (0..n.div_ceil(per_page)).step_by(pages_at_once) {
        let pages = pages_at_once.min(n.div_ceil(per_page) - first);
        let bytes = read(span(Part::Rows).offset + first * page, pages * page)?;
        for r in first * per_page..((first + pages) * per_page).min(n) {
            let row = &bytes[(r / per_page - first) * page + r % per_page * rb..][..rb];
            for (_, _, off, norm2, near) in queries.iter_mut().filter(|q| q.0 != r) {
                let d = sq8::distance(row, off, *norm2);
                if near.len() < NEAREST || d < near[near.len() - 1].0 {
                    let at = near.partition_point(|x| x.0 <= d);
                    near.insert(at, (d, r));
                    near.truncate(NEAREST);
                }
            }
        }
    }

    // The clusters nearest each query first, by the distance to their decoded centers, and the bytes read once each is
    // read. A nearest row is found once its cluster is.
    let ccb = sq4::code_bytes(padded_dim(dim));
    let centers: Vec<Vec<f32>> = (0..tables.lists.len())
        .map(|j| {
            let mut c = vec![0f32; dim];
            sq4::decode(&tables.codes[j * ccb..(j + 1) * ccb], &tables.factors[j], &mut c);
            c
        })
        .collect();
    let size = block_bytes(padded_dim(dim));
    let list_of = |r: usize| tables.lists.partition_point(|l| l.first_id as usize + l.count <= r);
    let (mut z, mut tmp) = (vec![0f32; dim], vec![0f32; dim]);
    let mut covers: Vec<Vec<f64>> = Vec::with_capacity(queries.len());

    for (_, q, _, _, near) in queries.iter().filter(|q| !q.4.is_empty()) {
        rotate_offset(&rotation, q, &mean, &mut z, &mut tmp);
        let mut order: Vec<(f64, usize)> = centers
            .iter()
            .enumerate()
            .map(|(j, c)| (z.iter().zip(c).map(|(&a, &b)| f64::from(a - b).powi(2)).sum(), j))
            .collect();
        order.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        let (mut at, mut total) = (vec![0f64; centers.len()], 0f64);

        for &(_, j) in &order {
            total += (tables.lists[j].count.div_ceil(BLOCK) * size) as f64;
            at[j] = total;
        }

        covers.push(near.iter().map(|&(_, r)| at[list_of(r)]).collect());
    }

    // The least bytes at which the mean recall over the samples reaches 95%.
    let mut all: Vec<f64> = covers.iter().flatten().copied().collect();
    all.sort_unstable_by(f64::total_cmp);
    let recall = |b: f64| {
        let found: f64 = covers
            .iter()
            .map(|c| c.iter().filter(|&&x| x <= b).count() as f64 / c.len() as f64)
            .sum();
        found / covers.len().max(1) as f64
    };
    let at = all.partition_point(|&b| recall(b) < RECALL);
    Ok(all.get(at).copied().unwrap_or(0.0))
}

#[cfg(test)]
#[path = "../../tests/unit/maintain/policy.rs"]
mod tests;
