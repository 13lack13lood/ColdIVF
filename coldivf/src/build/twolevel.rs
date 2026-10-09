//! Two-level clustering, for more vectors than flat k-means can cluster in reasonable time. It finds groups of
//! clusters first, then the clusters inside each group, then lets each vector move to a nearer cluster of a nearby
//! group. It holds a few numbers per vector, not the vectors, and reads them in order or a group at a time.

use std::fs::File;
use std::io::Error;
use std::path::{Path, PathBuf};

use rayon::ThreadPool;

use super::kmeans::{self, KMeans};
use super::stream::write_at;
use super::{ROUNDS, Slice, side_file};
use crate::search::inmem::check_settings;
use crate::search::{invalid, normalize};
use crate::{BuildParams, Clusters, Metric, VectorSource};

/// Rounds of moving the centers and the vectors after every vector is first placed.
const REFINE_ROUNDS: usize = 2;

/// The groups nearest to a vector, whose clusters it may join.
const NEAR_GROUPS: usize = 4;

/// Vectors per group that the group k-means trains on.
const GROUP_SAMPLE: usize = 256;

/// The bytes of vectors that a pass over all of them copies from the source at a time.
const PASS_BYTES: usize = 256 << 20;

/// The bytes of one group's vectors that gather before they go to the groups file, and that a read of a group takes
/// at a time.
const GROUP_BUFFER: usize = 256 << 10;

/// Finds the groups first, then the clusters inside each group, then refines the clusters with all the vectors. It runs
/// k-means over a sample of the vectors for at least `params.parents` groups, and as many as the square root of the
/// cluster count, then over each group's vectors, those nearest its center, for about `params.vectors_per_list`
/// vectors per cluster. Then each vector joins its nearest cluster among those of its `NEAR_GROUPS` nearest groups, and
/// each of `REFINE_ROUNDS` rounds moves every center to the mean of its vectors and does that again. It copies the
/// vectors into a file next to `out`, a group after another, for the k-means of each group, and reads them in order
/// for the rest. Every setting is checked first.
pub(crate) fn cluster(
    params: &BuildParams,
    src: &dyn VectorSource,
    out: &Path,
    pool: &ThreadPool,
) -> Result<Clusters, Error> {
    let (n, dim) = (src.count(), src.dim());
    check_settings(n, dim, params.page_size)?;

    if n == 0 || params.vectors_per_list == 0 || params.parents == 0 {
        return Err(invalid(
            "a build needs vectors, and at least one vector per cluster and one group",
        ));
    }

    let nlist = n.div_ceil(params.vectors_per_list);
    let k = params.parents.max(nlist.isqrt()).min(nlist);

    if k > usize::from(u16::MAX) {
        return Err(invalid("an index holds at most 65535 groups"));
    }

    let groups = KMeans {
        k,
        rounds: ROUNDS,
        sample: GROUP_SAMPLE * k,
        spherical: false,
        seed: params.seed.wrapping_add(1),
    };
    let parents = kmeans::train(&groups, src, params.metric, pool)?;
    let m = NEAR_GROUPS.min(k);
    let near = Near {
        groups: kmeans::nearest(&parents, src, params.metric, m, pool)?.0,
        m,
    };
    let (centroids, parent_of) = train_groups(params, src, &near, k, out, pool)?;
    let mut clusters = Clusters {
        dim,
        centroids,
        assignment: vec![0; n],
        parents,
        parent_of,
    };
    let per_pass = (PASS_BYTES / (4 * dim)).max(1);
    reassign(&mut clusters, src, &near, params.metric, pool, per_pass)?;

    for _ in 0..REFINE_ROUNDS {
        recenter(&mut clusters, src, params.metric, per_pass);
        reassign(&mut clusters, src, &near, params.metric, pool, per_pass)?;
    }

    Ok(clusters)
}

/// Each vector's nearest groups, `m` per vector, nearest first.
struct Near {
    groups: Vec<u32>,
    m: usize,
}

impl Near {
    fn of(&self, r: usize) -> &[u32] {
        &self.groups[r * self.m..(r + 1) * self.m]
    }
}

/// Runs k-means on the vectors of each of the `k` groups, those nearest its center, for about
/// `params.vectors_per_list` vectors per cluster. Returns the cluster centers and each cluster's group.
fn train_groups(
    params: &BuildParams,
    src: &dyn VectorSource,
    near: &Near,
    k: usize,
    out: &Path,
    pool: &ThreadPool,
) -> Result<(Vec<f32>, Vec<u32>), Error> {
    let dim = src.dim();
    let groups = Groups::write(src, |r| near.of(r)[0] as usize, k, out)?;
    let (mut centroids, mut parent_of, mut data) = (Vec::new(), Vec::new(), Vec::new());

    for g in 0..k {
        groups.read(g, &mut data)?;
        if data.is_empty() {
            continue;
        }
        let leaves = KMeans {
            k: (data.len() / dim).div_ceil(params.vectors_per_list),
            rounds: ROUNDS,
            sample: usize::MAX,
            spherical: false,
            seed: params.seed.wrapping_add(2 + g as u64),
        };
        centroids.extend(kmeans::train(
            &leaves,
            &Slice { dim, data: &data },
            params.metric,
            pool,
        )?);
        parent_of.extend(std::iter::repeat_n(g as u32, leaves.k));
    }

    Ok((centroids, parent_of))
}

/// Every vector of a source in a temporary file next to the index file, a group after another and in input order
/// inside each group. Dropping it removes the file.
struct Groups {
    file: File,
    path: PathBuf,
    dim: usize,
    /// The row where each group's vectors start in the file, then the row count.
    starts: Vec<usize>,
}

impl Groups {
    /// Copies every vector of `src` into the run of its group, `group_of(r)` for row r, out of `k` groups. It reads
    /// `src` in order.
    fn write(src: &dyn VectorSource, group_of: impl Fn(usize) -> usize, k: usize, out: &Path) -> Result<Groups, Error> {
        let (n, dim) = (src.count(), src.dim());
        let mut starts = vec![0usize; k + 1];
        (0..n).for_each(|r| starts[group_of(r) + 1] += 1);

        for g in 0..k {
            starts[g + 1] += starts[g];
        }

        let (file, path) = side_file(out, "groups")?;
        let groups = Groups {
            file,
            path,
            dim,
            starts,
        };
        groups.file.set_len((n * dim * 4) as u64)?;
        let (mut buffers, mut next) = (vec![Vec::new(); k], groups.starts[..k].to_vec());

        for r in 0..n {
            let g = group_of(r);
            let buffer = &mut buffers[g];
            src.rows(r, 1)
                .iter()
                .for_each(|x| buffer.extend_from_slice(&x.to_le_bytes()));
            if buffer.len() >= GROUP_BUFFER {
                groups.flush(buffer, &mut next[g])?;
            }
        }

        for (buffer, next) in buffers.iter_mut().zip(&mut next) {
            groups.flush(buffer, next)?;
        }

        Ok(groups)
    }

    /// Writes `buffer`, whole rows of one group, at the row `next` of the file, and moves `next` past them.
    fn flush(&self, buffer: &mut Vec<u8>, next: &mut usize) -> Result<(), Error> {
        write_at(&self.file, buffer, *next * self.dim * 4)?;
        *next += buffer.len() / (self.dim * 4);
        buffer.clear();
        Ok(())
    }

    /// Reads group `g`'s vectors into `data`.
    fn read(&self, g: usize, data: &mut Vec<f32>) -> Result<(), Error> {
        let row = self.dim * 4;
        let (start, end) = (self.starts[g] * row, self.starts[g + 1] * row);
        let mut bytes = vec![0u8; GROUP_BUFFER.min(end - start)];
        data.clear();

        for at in (start..end).step_by(GROUP_BUFFER) {
            let piece = &mut bytes[..GROUP_BUFFER.min(end - at)];
            crate::io::read_at(&self.file, piece, at)?;
            data.extend(piece.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)));
        }

        Ok(())
    }
}

impl Drop for Groups {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

/// Moves each vector to its nearest cluster among those of its nearest groups. It copies `per_pass` rows of `src` at a
/// time, in order, and scores each run's vectors a group at a time. Ties go to the lower cluster.
fn reassign(
    clusters: &mut Clusters,
    src: &dyn VectorSource,
    near: &Near,
    metric: Metric,
    pool: &ThreadPool,
    per_pass: usize,
) -> Result<(), Error> {
    let (n, dim) = (src.count(), clusters.dim);
    let mut in_group: Vec<Vec<u32>> = vec![Vec::new(); clusters.parents.len() / dim];

    for (c, &g) in clusters.parent_of.iter().enumerate() {
        in_group[g as usize].push(c as u32);
    }

    let centers: Vec<Vec<f32>> = in_group
        .iter()
        .map(|ids| {
            ids.iter()
                .flat_map(|&c| &clusters.centroids[c as usize * dim..(c as usize + 1) * dim])
                .copied()
                .collect()
        })
        .collect();
    let mut rows_of: Vec<Vec<u32>> = vec![Vec::new(); in_group.len()];
    let (mut copy, mut best) = (Vec::new(), Vec::new());

    for start in (0..n).step_by(per_pass) {
        let end = (start + per_pass).min(n);
        // The copy reads the source in order, however the groups pick the vectors out of it.
        copy.clear();
        copy.extend_from_slice(src.rows(start, end - start));
        let run = Slice { dim, data: &copy };
        rows_of.iter_mut().for_each(Vec::clear);
        for r in start..end {
            near.of(r)
                .iter()
                .for_each(|&g| rows_of[g as usize].push((r - start) as u32));
        }
        best.clear();
        best.resize(end - start, f32::INFINITY);
        for (g, rows) in rows_of.iter().enumerate() {
            if rows.is_empty() || in_group[g].is_empty() {
                continue;
            }
            let (found, scores) = kmeans::nearest(&centers[g], &Subset { src: &run, rows }, metric, 1, pool)?;
            for ((&i, &f), &s) in rows.iter().zip(&found).zip(&scores) {
                let (r, c) = (start + i as usize, in_group[g][f as usize]);
                let b = &mut best[i as usize];
                if s < *b || (s == *b && c < clusters.assignment[r]) {
                    (*b, clusters.assignment[r]) = (s, c);
                }
            }
        }
    }

    Ok(())
}

/// Moves each cluster's center to the mean of its vectors, normalized for cosine, reading `per_pass` rows of `src` at a
/// time. A cluster without vectors keeps its center. It sums the vectors in order, so the centers are the same for any
/// number of threads.
fn recenter(clusters: &mut Clusters, src: &dyn VectorSource, metric: Metric, per_pass: usize) {
    let (n, dim) = (src.count(), clusters.dim);
    let nlist = clusters.centroids.len() / dim;
    let (mut sums, mut counts) = (vec![0f64; nlist * dim], vec![0usize; nlist]);
    let mut x = vec![0f32; dim];

    for start in (0..n).step_by(per_pass) {
        let end = (start + per_pass).min(n);
        for (row, &c) in src
            .rows(start, end - start)
            .chunks_exact(dim)
            .zip(&clusters.assignment[start..end])
        {
            x.copy_from_slice(row);
            if metric == Metric::Cosine {
                normalize(&mut x);
            }
            counts[c as usize] += 1;
            let sum = &mut sums[c as usize * dim..(c as usize + 1) * dim];
            sum.iter_mut().zip(&x).for_each(|(s, &v)| *s += f64::from(v));
        }
    }

    kmeans::update(&mut clusters.centroids, &sums, &counts, dim, false);
}

/// Some rows of a source, by row number, eg. one group's vectors. k-means only ever asks it for one row at a time.
struct Subset<'a> {
    src: &'a dyn VectorSource,
    rows: &'a [u32],
}

impl VectorSource for Subset<'_> {
    fn count(&self) -> usize {
        self.rows.len()
    }

    fn dim(&self) -> usize {
        self.src.dim()
    }

    fn rows(&self, start: usize, rows: usize) -> &[f32] {
        assert_eq!(rows, 1, "a subset hands out one row at a time");
        self.src.rows(self.rows[start] as usize, 1)
    }
}

#[cfg(test)]
#[path = "../../tests/unit/build/twolevel.rs"]
mod tests;
