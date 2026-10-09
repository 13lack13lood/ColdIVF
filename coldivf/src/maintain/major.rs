//! Major runs, which cluster every vector again with k-means over the whole file, as a fresh build would. The training
//! set is every vector, read from the file again each round, or `major.sample` vectors per cluster held as 8-bit rows.
//! One more pass then gives every vector to its nearest center, and each cluster's center becomes its vectors' mean.
//! The regions are made again from the new centers.

use std::io::Error;

use super::regions;
use super::rows::{self, Held, Streamed};
use super::{CHUNK, Ctx, Ordered, order};
use crate::RunReport;
use crate::build::kmeans::{self, KMeans};
use crate::build::rewrite::{GONE, Plan, PlanCenter, PlanGroups, PlanList};
use crate::format::Trailer;
use crate::quantization::SplitMix64;
use crate::search::invalid;

/// The plan of a major run. It makes new regions, so the old ones only count toward the decision to run it.
pub(crate) fn plan(ctx: &Ctx, report: &mut RunReport) -> Result<Plan, Error> {
    let (dim, params) = (ctx.dim(), ctx.params);
    let ids = ctx.live_ids();
    let n = ids.len();

    if n == 0 {
        return Err(invalid("every vector was deleted, so there is nothing to write"));
    }

    let k = ((n as f64 / params.vectors_per_list as f64).round() as usize).clamp(1, n);
    let mut rng = SplitMix64::new(ctx.seed);
    let per = params.major.sample;
    let kmeans = KMeans {
        k,
        rounds: params.rounds,
        sample: usize::MAX,
        spherical: false,
        seed: rng.next_u64(),
    };
    let all = Streamed::new(ctx.old, ctx.delta, &ids);
    let (centers, trained) = match per > 0 && per.saturating_mul(k) < n {
        true => {
            let picked = kmeans::pick(n, per * k, &mut rng);
            let sample: Vec<u32> = picked.iter().map(|&i| ids[i as usize]).collect();
            let mut rows = Vec::new();
            report.bytes_read += rows::read_rows(ctx.old, ctx.delta, &sample, &mut rows)?;
            let held = Held {
                rows,
                dim,
                mean: ctx.old.mean(),
            };
            (kmeans::train_rows(&kmeans, &held, CHUNK, ctx.pool)?, sample.len())
        }
        false => (kmeans::train_rows(&kmeans, &all, CHUNK, ctx.pool)?, n),
    };
    report.distances += (trained * k * params.rounds) as u64;
    report.reclustered = n;
    let (mut assigned, mut sums, mut counts) = (vec![0u32; n], vec![0f64; k * dim], vec![0usize; k]);
    kmeans::scan_nearest(&centers, &all, 1, CHUNK, ctx.pool, |start, decoded, best, _| {
        for (i, (x, &c)) in decoded.chunks_exact(dim).zip(best).enumerate() {
            assigned[start + i] = c;
            counts[c as usize] += 1;
            let sum = &mut sums[c as usize * dim..(c as usize + 1) * dim];
            sum.iter_mut().zip(x).for_each(|(s, &v)| *s += f64::from(v));
        }
    })?;
    report.distances += (n * k) as u64;
    report.bytes_read += all.read.get();
    let kept: Vec<usize> = (0..k).filter(|&c| counts[c] > 0).collect();
    let means: Vec<f32> = kept
        .iter()
        .flat_map(|&c| {
            let n = counts[c] as f64;
            sums[c * dim..(c + 1) * dim].iter().map(move |&s| (s / n) as f32)
        })
        .collect();
    let mut stored = Vec::with_capacity(means.len());

    for m in means.chunks_exact(dim) {
        stored.extend(ctx.as_stored(m)?);
    }

    let (region_of, per_region, distances) = regions::cluster(&stored, dim, params, rng.next_u64(), ctx.pool)?;
    report.distances += distances;
    let Ordered {
        file,
        parent_of,
        parents,
        distances,
    } = order(ctx, stored)?;
    report.distances += distances;
    let mut new_of = vec![GONE; k];
    file.iter()
        .enumerate()
        .for_each(|(pos, &i)| new_of[kept[i as usize]] = pos as u32);
    let lists = file
        .iter()
        .map(|&i| PlanList {
            center: PlanCenter::New(means[i as usize * dim..(i as usize + 1) * dim].to_vec()),
            group: parent_of[i as usize] as u16,
        })
        .collect();
    let count = ctx.old.layout().count;
    let (mut file_rows, mut delta_rows) = (vec![GONE; count], vec![GONE; ctx.seen.inserts()]);

    for (&v, &c) in ids.iter().zip(&assigned) {
        match (v as usize).checked_sub(count) {
            Some(i) => delta_rows[i] = new_of[c as usize],
            None => file_rows[v as usize] = new_of[c as usize],
        }
    }

    let nregions = region_of.iter().map(|&r| r as usize + 1).max().unwrap_or(1);
    let mut region_last = vec![0u64; nregions];
    let list_last: Vec<u32> = file.iter().map(|&i| counts[kept[i as usize]] as u32).collect();
    let in_file: Vec<u32> = file.iter().map(|&i| region_of[i as usize]).collect();
    in_file
        .iter()
        .zip(&list_last)
        .for_each(|(&r, &c)| region_last[r as usize] += u64::from(c));

    Ok(Plan {
        lists,
        groups: PlanGroups::New(parents),
        file_rows,
        delta_rows,
        trailer: Some(Trailer {
            region_of: in_file,
            list_last,
            region_last,
            region_changes: vec![0; nregions],
            per_region,
            major_count: n as u64,
            major_inserts: 0,
            major_deletes: 0,
            pass_changes: 0,
        }),
    })
}

#[cfg(test)]
#[path = "../../tests/unit/maintain/major.rs"]
mod tests;
