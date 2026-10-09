//! Regions, the sets of nearby clusters that a minor run clusters again together, and the counts that say when each
//! one is due. A file without a trailer, eg. a fresh build's, gets its regions at its first run, from k-means over its
//! clusters' centers.

use std::io::Error;

use rayon::ThreadPool;

use super::Live;
use crate::MinorParams;
use crate::build::Slice;
use crate::build::kmeans::{self, KMeans};
use crate::format::Trailer;
use crate::{MaintainParams, Metric};

/// The regions of an old file's clusters and the counts that say when each is due, with a run's own changes added.
pub(crate) struct Regions {
    /// The region of each old cluster.
    pub(crate) of: Vec<u32>,
    /// Each region's vector count when it was last clustered, and its inserts plus deletes since, this run's included.
    pub(crate) last: Vec<u64>,
    pub(crate) changes: Vec<u64>,
    /// Each old cluster's count when its region was last clustered.
    pub(crate) list_last: Vec<u32>,
    /// Clusters per region when the regions were made.
    pub(crate) per_region: f32,
    /// The count at the last major run, and the inserts and the deletes since, this run's included. An insert that
    /// replaced a vector counts as an insert only.
    pub(crate) major_count: u64,
    pub(crate) major_inserts: u64,
    pub(crate) major_deletes: u64,
    /// The inserts plus deletes since the last major run or global pass, this run's included.
    pub(crate) pass_changes: u64,
}

impl Regions {
    /// The regions of the old file's trailer, with the changes of `live` added.
    pub(crate) fn carried(t: Trailer, live: &Live) -> Regions {
        let mut regions = Regions {
            of: t.region_of,
            last: t.region_last,
            changes: t.region_changes,
            list_last: t.list_last,
            per_region: t.per_region,
            major_count: t.major_count,
            major_inserts: t.major_inserts,
            major_deletes: t.major_deletes,
            pass_changes: t.pass_changes,
        };
        regions.add(live);
        regions
    }

    /// Regions for a file that has none: k-means over the clusters' decoded `centers` into `regions_for(nlist)`
    /// regions. Each region's count when last clustered is its clusters' count in the file, so the changes since the
    /// build count, and they are added.
    pub(crate) fn made(
        centers: &[f32],
        dim: usize,
        counts: &[usize],
        live: &Live,
        params: &MaintainParams,
        seed: u64,
        pool: &ThreadPool,
    ) -> Result<(Regions, u64), Error> {
        let (of, per_region, distances) = cluster(centers, dim, params, seed, pool)?;
        let nregions = of.iter().map(|&r| r as usize + 1).max().unwrap_or(1);
        let mut last = vec![0u64; nregions];
        counts.iter().zip(&of).for_each(|(&n, &r)| last[r as usize] += n as u64);
        let mut regions = Regions {
            of,
            changes: vec![0; nregions],
            last,
            list_last: counts.iter().map(|&n| n as u32).collect(),
            per_region,
            major_count: counts.iter().sum::<usize>() as u64,
            major_inserts: 0,
            major_deletes: 0,
            pass_changes: 0,
        };
        regions.add(live);
        Ok((regions, distances))
    }

    /// Adds the inserts and deletes that `live` folds in to the counts of their regions, the major run's and the
    /// global pass's.
    fn add(&mut self, live: &Live) {
        for (j, &r) in self.of.iter().enumerate() {
            let (added, deleted) = (u64::from(live.added[j]), u64::from(live.deleted[j]));
            self.changes[r as usize] += added + deleted;
            self.pass_changes += added + deleted;
            self.major_inserts += added;
            self.major_deletes += deleted - u64::from(live.replaced[j]);
        }
    }

    /// The regions that are due as `params` says, most changed first, i.e. by their changes over their count when last
    /// clustered. `sizes` is each old cluster's vector count now.
    pub(crate) fn due(&self, sizes: &[usize], params: &MinorParams) -> Vec<u32> {
        let nregions = self.last.len();
        let mut now = vec![0u64; nregions];
        let mut due = vec![false; nregions];
        let total: usize = sizes.iter().sum();
        // The shares are compared in f32, the type they are given in, so that eg. 100 changes of 1000 reach 0.1.
        let mean = total as f32 / sizes.len().max(1) as f32;

        for (j, (&n, &r)) in sizes.iter().zip(&self.of).enumerate() {
            now[r as usize] += n as u64;
            let (hot, lost) = (
                n as f32 > params.hot_list * mean,
                self.list_last[j] > 0 && n as f32 <= (1.0 - params.list_loss) * self.list_last[j] as f32,
            );
            if (params.hot_list > 0.0 && hot) || (params.list_loss > 0.0 && lost) {
                due[r as usize] = true;
            }
        }

        for r in 0..nregions {
            let (last, changes) = (self.last[r] as f32, self.changes[r] as f32);
            let changed = params.change > 0.0 && changes >= params.change * last.max(1.0);
            let shrunk = params.shrink > 0.0 && last > 0.0 && now[r] as f32 <= (1.0 - params.shrink) * last;
            due[r] |= changed || shrunk;
            due[r] &= now[r] > 0;
        }

        let mut out: Vec<u32> = (0..nregions as u32).filter(|&r| due[r as usize]).collect();
        let ratio = |r: u32| self.changes[r as usize] as f64 / (self.last[r as usize] as f64).max(1.0);
        out.sort_by(|&a, &b| ratio(b).total_cmp(&ratio(a)).then(a.cmp(&b)));
        out
    }
}

/// How many regions `nlist` clusters make, `params.minor.region_lists` clusters each, or the square root of `nlist`
/// with 0.
pub(crate) fn regions_for(nlist: usize, params: &MaintainParams) -> usize {
    let per = match params.minor.region_lists {
        0 => (nlist as f64).sqrt(),
        n => n as f64,
    };
    ((nlist as f64 / per.max(1.0)).round() as usize).clamp(1, nlist.max(1))
}

/// The region of each of the clusters whose decoded centers are `centers`, from k-means into `regions_for` regions,
/// numbered in order with no empty one, and the clusters per region. Returns the distances k-means computed too.
pub(crate) fn cluster(
    centers: &[f32],
    dim: usize,
    params: &MaintainParams,
    seed: u64,
    pool: &ThreadPool,
) -> Result<(Vec<u32>, f32, u64), Error> {
    let nlist = centers.len() / dim;
    let k = regions_for(nlist, params);
    let src = Slice { dim, data: centers };
    let kmeans = KMeans {
        k,
        rounds: params.rounds,
        sample: usize::MAX,
        spherical: false,
        seed,
    };
    let of = match k {
        1 => vec![0; nlist],
        _ => {
            let regions = kmeans::train(&kmeans, &src, Metric::L2, pool)?;
            kmeans::assign(&regions, &src, Metric::L2, pool)?
        }
    };
    let of = renumber(&of);
    let nregions = of.iter().map(|&r| r as usize + 1).max().unwrap_or(1);
    let distances = (nlist * k * (params.rounds + 1)) as u64;
    Ok((of, nlist as f32 / nregions as f32, distances))
}

/// The same labels numbered 0, 1, 2 and so on in order of their value, so no number is left without a member.
pub(crate) fn renumber(labels: &[u32]) -> Vec<u32> {
    let mut used: Vec<u32> = labels.to_vec();
    used.sort_unstable();
    used.dedup();
    labels.iter().map(|l| used.partition_point(|u| u < l) as u32).collect()
}

#[cfg(test)]
#[path = "../../tests/unit/maintain/regions.rs"]
mod tests;
