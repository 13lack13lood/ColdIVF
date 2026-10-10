//! Minor runs, which cluster again the regions that are due. For each due region, most changed first, a run reads the
//! rows of its clusters' vectors and runs k-means over them for about `vectors_per_list` vectors per cluster. Then it
//! fixes the boundaries exactly, once for every due region:
//! 1. Each vector of a due region moves to its nearest cluster of the whole file.
//! 2. The vectors of the `ring` clusters outside its region nearest each new cluster move to their nearest new cluster
//!    when it is nearer than their own.
//!
//! A region that grows past `split` times the clusters per region splits in two, and one that falls under `merge`
//! times that joins its nearest region. Each new cluster's center is its vectors' mean. Clusters outside the due
//! regions keep their centers even when they gained or lost vectors, unless `refresh` is set. Inserts far from every
//! center may form a region of their own, and a global pass may move every vector, as `MinorParams` says.

use std::io::Error;

use super::regions::Regions;
use super::rows::{self, Held, Streamed};
use super::{CHUNK, Ctx, Ordered, order, trailer};
use crate::build::Slice;
use crate::build::kmeans::{self, KMeans};
use crate::build::rewrite::{GONE, Plan, PlanCenter, PlanGroups, PlanList};
use crate::quantization::{SplitMix64, sq8};
use crate::{Metric, RunReport};

/// How many passes over the regions a run makes to merge the small ones.
const MERGES: usize = 4;

/// The most file rows that set the distance a pooled insert must pass.
const POOL_SAMPLE: usize = 20_000;

/// What a minor run has to do.
pub(crate) struct Work {
    /// The due regions, most changed first.
    pub(crate) due: Vec<u32>,
    /// The inserts far from every center, as vector ids, to cluster on their own, or none.
    pub(crate) far: Vec<u32>,
    /// Whether a global pass is due.
    pub(crate) pass: bool,
    /// Whether `refresh` is set and some cluster gained or lost vectors.
    pub(crate) refresh: bool,
}

impl Work {
    /// What a minor run of `ctx` with the regions `regions` has to do.
    pub(crate) fn of(ctx: &Ctx, regions: &Regions, report: &mut RunReport) -> Result<Work, Error> {
        let (params, live) = (&ctx.params.minor, ctx.live);
        let changed = live.added.iter().zip(&live.deleted).any(|(&a, &d)| a + d > 0);
        let base = regions.major_count.max(1) as f32;

        Ok(Work {
            due: regions.due(&live.sizes(), params),
            far: match params.pool {
                Some(share) => far_inserts(ctx, share, report)?,
                None => Vec::new(),
            },
            pass: params
                .global_pass
                .is_some_and(|f| regions.pass_changes as f32 >= f * base),
            refresh: params.refresh && changed,
        })
    }

    /// Whether there is nothing to do but fold.
    pub(crate) fn is_empty(&self) -> bool {
        self.due.is_empty() && self.far.is_empty() && !self.pass && !self.refresh
    }
}

/// The visible inserts whose squared distance to their cluster's center, from their decoded rows, is over the `share`
/// quantile of the file rows' distances to theirs, measured on a sample. None unless there are `vectors_per_list` of
/// them.
fn far_inserts(ctx: &Ctx, share: f32, report: &mut RunReport) -> Result<Vec<u32>, Error> {
    let (dim, old, live) = (ctx.dim(), ctx.old, ctx.live);
    let rows: Vec<u32> = (0..old.layout().count as u32)
        .filter(|&r| live.rows[r as usize])
        .collect();
    let mut rng = SplitMix64::new(ctx.seed ^ 0x706F_6F6C);
    let picked: Vec<u32> = kmeans::pick(rows.len(), POOL_SAMPLE.min(rows.len()), &mut rng)
        .iter()
        .map(|&i| rows[i as usize])
        .collect();
    let mut bytes = Vec::new();
    report.bytes_read += rows::read_rows(old, ctx.gate, ctx.delta, &picked, &mut bytes)?;
    let (rb, lists) = (sq8::row_bytes(dim), &ctx.tables.lists);
    let mut o = vec![0f32; dim];
    let mut far_of = |row: &[u8], leaf: usize| {
        sq8::decode(row, old.mean(), &mut o);
        dist2(&o, &ctx.centers[leaf * dim..(leaf + 1) * dim])
    };
    let mut sample: Vec<f64> = picked
        .iter()
        .zip(bytes.chunks_exact(rb))
        .map(|(&r, row)| far_of(row, lists.partition_point(|l| l.first_id + l.count as u32 <= r)))
        .collect();

    if sample.is_empty() {
        return Ok(Vec::new());
    }

    sample.sort_by(f64::total_cmp);
    let at = ((f64::from(share) * sample.len() as f64) as usize).min(sample.len() - 1);
    let (count, limit) = (old.layout().count as u32, sample[at]);
    let far: Vec<u32> = live
        .inserts
        .iter()
        .filter(|&&i| far_of(ctx.delta.row(i as usize), ctx.delta.leaf(i as usize) as usize) > limit)
        .map(|&i| count + i)
        .collect();

    match far.len() >= ctx.params.vectors_per_list {
        true => Ok(far),
        false => Ok(Vec::new()),
    }
}

/// The clusters while a minor run plans them. A cluster's id is its place in these arrays. The old clusters come first,
/// in file order, and each new one goes after them.
pub(crate) struct Drafts {
    dim: usize,
    /// The old cluster whose center each one keeps, or `None` for a new one.
    pub(crate) keep: Vec<Option<u32>>,
    /// Each one's center as it decodes, one after another, in the vectors' space.
    pub(crate) centers: Vec<f32>,
    pub(crate) region: Vec<u32>,
    pub(crate) members: Vec<Vec<u32>>,
    pub(crate) alive: Vec<bool>,
    /// Each one's count when its region was last clustered.
    pub(crate) last: Vec<u32>,
    /// Whether a vector moved into or out of each one.
    pub(crate) touched: Vec<bool>,
    /// The cluster of each vector by vector id, or `GONE` for a vector that drops out, and its place among the
    /// cluster's members.
    pub(crate) list_of: Vec<u32>,
    place: Vec<u32>,
}

impl Drafts {
    /// The old clusters with the vectors that stay in each.
    pub(crate) fn of(ctx: &Ctx, regions: &Regions) -> Drafts {
        let members = ctx.members_of_old();
        let n = ctx.old.layout().count + ctx.seen.inserts();
        let (mut list_of, mut place) = (vec![GONE; n], vec![0u32; n]);

        for (l, m) in members.iter().enumerate() {
            for (k, &v) in m.iter().enumerate() {
                (list_of[v as usize], place[v as usize]) = (l as u32, k as u32);
            }
        }

        Drafts {
            dim: ctx.dim(),
            keep: (0..members.len() as u32).map(Some).collect(),
            centers: ctx.centers.to_vec(),
            region: regions.of.clone(),
            alive: vec![true; members.len()],
            touched: vec![false; members.len()],
            members,
            last: regions.list_last.clone(),
            list_of,
            place,
        }
    }

    pub(crate) fn center(&self, l: usize) -> &[f32] {
        &self.centers[l * self.dim..(l + 1) * self.dim]
    }

    /// Adds a new cluster without vectors, with the decoded center `center`, to region `region`.
    fn add(&mut self, center: &[f32], region: u32) -> usize {
        self.keep.push(None);
        self.centers.extend_from_slice(center);
        self.region.push(region);
        self.members.push(Vec::new());
        self.alive.push(true);
        self.touched.push(true);
        self.last.push(0);
        self.keep.len() - 1
    }

    /// Moves vector `v` to cluster `l`.
    pub(crate) fn put(&mut self, v: u32, l: usize) {
        let (vi, from) = (v as usize, self.list_of[v as usize]);

        if from == l as u32 {
            return;
        }

        if from != GONE {
            let (from, at) = (from as usize, self.place[vi] as usize);
            let last = self.members[from].pop().unwrap_or(v);
            if last != v {
                self.members[from][at] = last;
                self.place[last as usize] = at as u32;
            }
            self.touched[from] = true;
        }

        self.place[vi] = self.members[l].len() as u32;
        self.members[l].push(v);
        self.list_of[vi] = l as u32;
        self.touched[l] = true;
    }

    /// The live clusters of region `r`.
    fn of_region(&self, r: u32) -> Vec<usize> {
        (0..self.keep.len())
            .filter(|&l| self.alive[l] && self.region[l] == r)
            .collect()
    }

    /// The live clusters that `pick` takes, and their centers one after another.
    fn centers_of(&self, pick: impl Fn(usize) -> bool) -> (Vec<usize>, Vec<f32>) {
        let ids: Vec<usize> = (0..self.keep.len()).filter(|&l| self.alive[l] && pick(l)).collect();
        let centers = ids.iter().flat_map(|&l| self.center(l).iter().copied()).collect();
        (ids, centers)
    }

    /// Drops the clusters left without vectors.
    fn drop_empty(&mut self) {
        for l in 0..self.keep.len() {
            if self.members[l].is_empty() {
                self.alive[l] = false;
            }
        }
    }
}

/// The plan of a minor run that does `work`.
pub(crate) fn plan(ctx: &Ctx, regions: Regions, work: &Work, report: &mut RunReport) -> Result<Plan, Error> {
    let mut d = Drafts::of(ctx, &regions);
    // A cluster that lost every vector drops out before any vector could move to it.
    d.drop_empty();
    let (mut last, mut changes) = (regions.last.clone(), regions.changes.clone());
    let mut rng = SplitMix64::new(ctx.seed);
    // The vectors that the boundary fix places, and the regions clustered again.
    let (mut moving, mut fresh) = (Vec::new(), Vec::new());

    if !work.far.is_empty() {
        let q = last.len() as u32;
        last.push(0);
        changes.push(0);
        cluster(ctx, &mut d, &work.far, q, rng.next_u64(), report)?;
        moving.extend_from_slice(&work.far);
        fresh.push(q);
    }

    for &p in &work.due {
        let lists = d.of_region(p);
        let mut members: Vec<u32> = lists.iter().flat_map(|&l| d.members[l].iter().copied()).collect();
        members.sort_unstable();
        lists.iter().for_each(|&l| d.alive[l] = false);
        report.reclustered += members.len();
        cluster(ctx, &mut d, &members, p, rng.next_u64(), report)?;
        moving.extend(members);
        fresh.push(p);
        let mine = d.of_region(p);
        let split = f64::from(ctx.params.minor.split) * f64::from(regions.per_region);

        if ctx.params.minor.split > 0.0 && mine.len() as f64 > split && mine.len() >= 2 {
            let centers: Vec<f32> = mine.iter().flat_map(|&l| d.center(l).iter().copied()).collect();
            let halves = two_means(&centers, ctx.dim(), rng.next_u64(), ctx)?;
            report.distances += (2 * mine.len() * (ctx.params.rounds + 1)) as u64;
            if halves.contains(&0) && halves.contains(&1) {
                let q = last.len() as u32;
                last.push(0);
                changes.push(0);
                let half: Vec<usize> = mine
                    .iter()
                    .zip(&halves)
                    .filter(|&(_, &h)| h == 1)
                    .map(|(&l, _)| l)
                    .collect();
                half.iter().for_each(|&l| d.region[l] = q);
                fresh.push(q);
            }
        }
    }

    moving.sort_unstable();
    moving.dedup();
    nearest_of_all(ctx, &mut d, &moving, report)?;
    d.drop_empty();

    if ctx.params.minor.ring > 0 && !moving.is_empty() {
        ring(ctx, &mut d, report)?;
        d.drop_empty();
    }

    fresh
        .iter()
        .for_each(|&r| clustered(&mut d, r, &mut last, &mut changes));
    merge(ctx, &mut d, regions.per_region, &mut last, &mut changes);
    let mut regions = regions;

    if work.pass {
        global_pass(ctx, &mut d, report)?;
        regions.pass_changes = 0;
    }

    if ctx.params.minor.refresh {
        for j in 0..ctx.tables.lists.len() {
            let changed = d.touched[j] || ctx.live.added[j] + ctx.live.deleted[j] > 0;
            if d.alive[j] && changed {
                d.keep[j] = None;
            }
        }
    }

    let lists: Vec<usize> = (0..d.keep.len()).filter(|&l| d.alive[l]).collect();
    let new: Vec<usize> = lists.iter().copied().filter(|&l| d.keep[l].is_none()).collect();
    let means = means(ctx, &d, &new, report)?;
    let mut mean_of = vec![usize::MAX; d.keep.len()];
    new.iter().enumerate().for_each(|(i, &l)| mean_of[l] = i);
    let dim = ctx.dim();
    let mean = |l: usize| &means[mean_of[l] * dim..(mean_of[l] + 1) * dim];
    let mut centers = Vec::with_capacity(lists.len() * dim);

    for &l in &lists {
        match d.keep[l] {
            Some(_) => centers.extend_from_slice(d.center(l)),
            None => centers.extend(ctx.as_stored(mean(l))?),
        }
    }

    let Ordered {
        file,
        parent_of,
        parents,
        distances,
    } = order(ctx, centers)?;
    report.distances += distances;
    let mut new_of = vec![GONE; d.keep.len()];
    file.iter()
        .enumerate()
        .for_each(|(pos, &i)| new_of[lists[i as usize]] = pos as u32);
    let plan_lists = file
        .iter()
        .map(|&i| {
            let l = lists[i as usize];
            let center = match d.keep[l] {
                Some(j) => PlanCenter::Keep(j),
                None => PlanCenter::New(mean(l).to_vec()),
            };
            PlanList {
                center,
                group: parent_of[i as usize] as u16,
            }
        })
        .collect();
    let new_lists: Vec<(u32, u32)> = file
        .iter()
        .map(|&i| (d.region[lists[i as usize]], d.last[lists[i as usize]]))
        .collect();
    let count = ctx.old.layout().count;
    let place = |v: usize| match d.list_of[v] {
        GONE => GONE,
        l => new_of[l as usize],
    };

    Ok(Plan {
        lists: plan_lists,
        groups: PlanGroups::New(parents),
        file_rows: (0..count).map(place).collect(),
        delta_rows: (0..ctx.seen.inserts()).map(|i| place(count + i)).collect(),
        trailer: Some(trailer(&regions, &last, &changes, &new_lists)),
    })
}

/// Clusters the vectors `members`, in increasing order, with k-means over their rows into new clusters of region
/// `region`, about `vectors_per_list` vectors each. The vectors stay where they are until the boundary fix.
fn cluster(
    ctx: &Ctx,
    d: &mut Drafts,
    members: &[u32],
    region: u32,
    seed: u64,
    report: &mut RunReport,
) -> Result<(), Error> {
    let (dim, params) = (ctx.dim(), ctx.params);

    if members.is_empty() {
        return Ok(());
    }

    let mut rows = Vec::new();
    report.bytes_read += rows::read_rows(ctx.old, ctx.gate, ctx.delta, members, &mut rows)?;
    let held = Held {
        rows,
        dim,
        mean: ctx.old.mean(),
    };
    let k = ((members.len() as f64 / params.vectors_per_list as f64).round() as usize).clamp(1, members.len());
    let kmeans = KMeans {
        k,
        rounds: params.rounds,
        sample: usize::MAX,
        spherical: false,
        seed,
    };
    let trained = kmeans::train_rows(&kmeans, &held, CHUNK, ctx.pool)?;
    report.distances += (members.len() * k * params.rounds) as u64;

    for c in trained.chunks_exact(dim) {
        d.add(&ctx.as_stored(c)?, region);
    }

    Ok(())
}

/// Moves each of the vectors `moving`, in increasing order, to its nearest live cluster of the whole file, reading
/// their rows a block at a time.
fn nearest_of_all(ctx: &Ctx, d: &mut Drafts, moving: &[u32], report: &mut RunReport) -> Result<(), Error> {
    if moving.is_empty() {
        return Ok(());
    }

    let (ids, centers) = d.centers_of(|_| true);
    let src = Streamed::new(ctx.old, ctx.gate, ctx.delta, moving);
    let best = kmeans::nearest_rows(&centers, &src, 1, CHUNK, ctx.pool)?.0;
    report.bytes_read += src.read.get();
    report.distances += (moving.len() * ids.len()) as u64;
    moving.iter().zip(&best).for_each(|(&v, &b)| d.put(v, ids[b as usize]));
    Ok(())
}

/// Moves the vectors of the clusters in the rings of the new clusters to their nearest new cluster when it is nearer
/// than their own. A new cluster's ring is the `ring` live clusters outside its region nearest it. The vectors of new
/// clusters sit in their nearest cluster already, so it reads the rows of the rings' kept clusters only, once, in
/// increasing order, and compares exact distances in f64.
fn ring(ctx: &Ctx, d: &mut Drafts, report: &mut RunReport) -> Result<(), Error> {
    let dim = ctx.dim();
    let (new, new_centers) = d.centers_of(|l| d.keep[l].is_none());
    let mut regions: Vec<u32> = new.iter().map(|&l| d.region[l]).collect();
    regions.sort_unstable();
    regions.dedup();
    let mut lists = Vec::new();

    for r in regions {
        let (outside, outside_centers) = d.centers_of(|l| d.region[l] != r);
        let mine: Vec<f32> = new
            .iter()
            .filter(|&&l| d.region[l] == r)
            .flat_map(|&l| d.center(l).iter().copied())
            .collect();
        if outside.is_empty() {
            continue;
        }
        let m = ctx.params.minor.ring.min(outside.len());
        let near = kmeans::nearest(&outside_centers, &Slice { dim, data: &mine }, Metric::L2, m, ctx.pool)?.0;
        report.distances += (mine.len() / dim * outside.len()) as u64;
        lists.extend(
            near.iter()
                .map(|&i| outside[i as usize])
                .filter(|&l| d.keep[l].is_some()),
        );
    }

    lists.sort_unstable();
    lists.dedup();
    let mut vectors: Vec<u32> = lists.iter().flat_map(|&l| d.members[l].iter().copied()).collect();
    vectors.sort_unstable();

    if vectors.is_empty() {
        return Ok(());
    }

    let src = Streamed::new(ctx.old, ctx.gate, ctx.delta, &vectors);
    let mut moves = Vec::new();
    kmeans::scan_nearest(&new_centers, &src, 1, CHUNK, ctx.pool, |start, decoded, best, _| {
        for (k, (x, &b)) in decoded.chunks_exact(dim).zip(best).enumerate() {
            let v = vectors[start + k];
            let (own, to) = (d.list_of[v as usize] as usize, new[b as usize]);
            if dist2(x, d.center(to)) < dist2(x, d.center(own)) {
                moves.push((v, to));
            }
        }
    })?;
    report.bytes_read += src.read.get();
    report.distances += (vectors.len() * new.len()) as u64;
    moves.into_iter().for_each(|(v, to)| d.put(v, to));
    Ok(())
}

/// Moves every vector to its nearest live cluster and makes every cluster new, so its center becomes its vectors'
/// mean. It reads every row once, in order.
fn global_pass(ctx: &Ctx, d: &mut Drafts, report: &mut RunReport) -> Result<(), Error> {
    let all = ctx.live_ids();
    let (ids, centers) = d.centers_of(|_| true);
    let src = Streamed::new(ctx.old, ctx.gate, ctx.delta, &all);
    let best = kmeans::nearest_rows(&centers, &src, 1, CHUNK, ctx.pool)?.0;
    report.bytes_read += src.read.get();
    report.distances += (all.len() * ids.len()) as u64;
    all.iter().zip(&best).for_each(|(&v, &b)| d.put(v, ids[b as usize]));
    d.drop_empty();
    ids.iter().for_each(|&l| d.keep[l] = None);
    Ok(())
}

/// Marks region `r` as clustered now, so its count and each of its clusters' counts become the ones to measure changes
/// against, and its changes start again from 0.
fn clustered(d: &mut Drafts, r: u32, last: &mut [u64], changes: &mut [u64]) {
    let mine = d.of_region(r);
    last[r as usize] = mine.iter().map(|&l| d.members[l].len() as u64).sum();
    changes[r as usize] = 0;
    mine.iter().for_each(|&l| d.last[l] = d.members[l].len() as u32);
}

/// Merges each region with fewer than `merge` times `per_region` clusters into the region with the nearest mean
/// center, adding its counts to that region's.
fn merge(ctx: &Ctx, d: &mut Drafts, per_region: f32, last: &mut [u64], changes: &mut [u64]) {
    let (dim, limit) = (ctx.dim(), f64::from(ctx.params.minor.merge) * f64::from(per_region));

    if ctx.params.minor.merge <= 0.0 {
        return;
    }

    for _ in 0..MERGES {
        let nregions = last.len();
        let (mut sizes, mut sums) = (vec![0usize; nregions], vec![0f64; nregions * dim]);

        for l in (0..d.keep.len()).filter(|&l| d.alive[l]) {
            let r = d.region[l] as usize;
            sizes[r] += 1;
            let sum = &mut sums[r * dim..(r + 1) * dim];
            sum.iter_mut().zip(d.center(l)).for_each(|(s, &x)| *s += f64::from(x));
        }

        let mut present: Vec<usize> = (0..nregions).filter(|&r| sizes[r] > 0).collect();
        let small: Vec<usize> = present.iter().copied().filter(|&r| (sizes[r] as f64) < limit).collect();

        if small.is_empty() || present.len() < 2 {
            return;
        }

        let mean = |r: usize, i: usize| sums[r * dim + i] / sizes[r] as f64;

        for p in small {
            present.retain(|&r| r != p);
            let gap = |q: usize| (0..dim).map(|i| (mean(p, i) - mean(q, i)).powi(2)).sum::<f64>();
            let Some(&q) = present
                .iter()
                .min_by(|&&a, &&b| gap(a).total_cmp(&gap(b)).then(a.cmp(&b)))
            else {
                break;
            };
            let moving: Vec<usize> = (0..d.keep.len())
                .filter(|&l| d.alive[l] && d.region[l] as usize == p)
                .collect();
            moving.iter().for_each(|&l| d.region[l] = q as u32);
            (last[q], changes[q]) = (last[q] + last[p], changes[q] + changes[p]);
        }
    }
}

/// Each of the clusters `new`'s mean of its vectors' decoded rows, one after another.
fn means(ctx: &Ctx, d: &Drafts, new: &[usize], report: &mut RunReport) -> Result<Vec<f32>, Error> {
    let dim = ctx.dim();
    let mut slot = vec![u32::MAX; d.keep.len()];
    new.iter().enumerate().for_each(|(i, &l)| slot[l] = i as u32);
    let (mut ids, mut group) = (Vec::new(), Vec::new());

    for (v, &l) in d.list_of.iter().enumerate() {
        if l != GONE && slot[l as usize] != u32::MAX {
            ids.push(v as u32);
            group.push(slot[l as usize]);
        }
    }

    let (sums, read) = rows::sums(ctx.old, ctx.gate, ctx.delta, &ids, &group, new.len(), ctx.pool)?;
    report.bytes_read += read;
    let counts: Vec<usize> = new.iter().map(|&l| d.members[l].len()).collect();
    let means = sums.chunks_exact(dim).zip(&counts);
    Ok(means
        .flat_map(|(s, &n)| s.iter().map(move |&x| (x / n as f64) as f32))
        .collect())
}

/// Splits the centers `centers` in two with 2-means. Returns each one's half, 0 or 1.
fn two_means(centers: &[f32], dim: usize, seed: u64, ctx: &Ctx) -> Result<Vec<u32>, Error> {
    let src = Slice { dim, data: centers };
    let kmeans = KMeans {
        k: 2,
        rounds: ctx.params.rounds,
        sample: usize::MAX,
        spherical: false,
        seed,
    };
    let halves = kmeans::train(&kmeans, &src, Metric::L2, ctx.pool)?;
    kmeans::assign(&halves, &src, Metric::L2, ctx.pool)
}

/// The exact squared distance in f64.
fn dist2(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| (f64::from(x) - f64::from(y)).powi(2))
        .sum()
}

#[cfg(test)]
#[path = "../../tests/unit/maintain/minor.rs"]
mod tests;
