//! The whole index in memory. It holds the same bytes as the sections of an index file, so a builder writes them out
//! as they are, and a search of the file gives exactly its results.

use std::io::Error;

use super::route::Centers;
use super::{LIMIT, List, RowPages, View, invalid, normalize, run, run_route};
use crate::build::order::leaf_order;
use crate::kernels::Tier;
use crate::quantization::rabitq::{BLOCK, Encoder, Factors, block_bytes, code_bytes, pack_block};
use crate::quantization::rotation::Rotation;
use crate::quantization::sq4::{self, CenterFactors};
use crate::quantization::{padded_dim, rotate_offset, sq8};
use crate::{BuildParams, Clusters, Hit, LeafId, Metric, SearchCtx, SearchParams, VectorSource};

/// The smallest dimension an index takes.
pub(crate) const MIN_DIM: usize = 128;

/// The largest dimension. Up to here a block's lookup sums fit in 16 bits and a row fits in a 4 KiB page.
pub(crate) const MAX_DIM: usize = 4080;

/// Rows read from the source at a time.
const CHUNK: usize = 4096;

/// The largest page size.
const MAX_PAGE: usize = 1 << 20;

pub(crate) struct InMemory {
    pub(crate) dim: usize,
    pub(crate) metric: Metric,
    pub(crate) seed: u64,
    pub(crate) page_size: usize,
    pub(crate) mean: Vec<f32>,
    pub(crate) rotation: Rotation,
    /// The cluster group centers as 8-bit rows without ids.
    pub(crate) parents: Vec<u8>,
    /// The 4-bit codes of the cluster centers, back to back, in file order.
    pub(crate) centers: Vec<u8>,
    pub(crate) center_factors: Vec<CenterFactors>,
    /// Each cluster's group, in file order.
    pub(crate) parent_of: Vec<u16>,
    pub(crate) lists: Vec<List>,
    /// Every cluster's blocks, in file order.
    pub(crate) blocks: Vec<u8>,
    /// The 8-bit rows by internal id, in pages.
    pub(crate) rows: Vec<u8>,
    /// The id given at build time of each internal id.
    pub(crate) ids: Vec<u64>,
    tier: Tier,
}

impl InMemory {
    /// Builds the index from `src` and an imported clustering. It uses `params.metric`, `params.order`,
    /// `params.page_size` and `params.seed`. The vectors in each cluster keep their input order, and a vector's id is
    /// its input row.
    pub(crate) fn build(params: &BuildParams, src: &dyn VectorSource, clusters: &Clusters) -> Result<InMemory, Error> {
        let (n, dim) = (src.count(), src.dim());
        let (nlist, nparents) = check_input(n, dim, clusters)?;
        let metric = params.metric;
        let rb = sq8::row_bytes(dim);

        if !params.page_size.is_multiple_of(4096) || params.page_size < rb || params.page_size > MAX_PAGE {
            return Err(invalid(
                "the page size must be a multiple of 4096, at most 1 MiB, that holds a row",
            ));
        }

        let mut sum = vec![0f64; dim];

        for_each_row(src, metric, |_, o| {
            sum.iter_mut().zip(o).for_each(|(s, &x)| *s += f64::from(x));
            Ok(())
        })?;

        let mean: Vec<f32> = sum
            .iter()
            .map(|&s| if n == 0 { 0.0 } else { (s / n as f64) as f32 })
            .collect();
        let rotation = Rotation::new(dim, params.seed);
        let order = leaf_order(params.order, clusters, &mean);
        let mut position = vec![0u32; nlist];
        order
            .iter()
            .enumerate()
            .for_each(|(p, &c)| position[c as usize] = p as u32);

        let d_pad = padded_dim(dim);
        let size = block_bytes(d_pad);
        let lists = lists(&clusters.assignment, &order, size);
        let (cb, per_page) = (code_bytes(d_pad), params.page_size / rb);
        let mut codes = vec![0u8; n * cb];
        let mut factors = vec![Factors::EMPTY; n];
        let mut rows = vec![0u8; n.div_ceil(per_page) * params.page_size];
        let mut ids = vec![0u64; n];
        let mut next: Vec<u32> = lists.iter().map(|l| l.first_id).collect();
        let mut encoder = Encoder::new(&rotation, &mean);

        for_each_row(src, metric, |r, o| {
            let p = position[clusters.assignment[r] as usize] as usize;
            let id = next[p] as usize;
            next[p] += 1;
            let f = encoder.encode(o, &mut codes[id * cb..(id + 1) * cb]);
            let at = (id / per_page) * params.page_size + (id % per_page) * rb;
            let row = &mut rows[at..at + rb];
            sq8::write_row(o, &mean, r as u64, row);
            if !(fits(f.f_add) && f.f_rescale.is_finite() && f.f_err.is_finite() && fits(sq8::norm2(row, dim))) {
                return Err(invalid("a vector is too far from the mean to encode"));
            }
            factors[id] = f;
            ids[id] = r as u64;
            Ok(())
        })?;

        let mut blocks = vec![0u8; lists.last().map_or(0, |l| l.offset + l.count.div_ceil(BLOCK) * size)];

        for list in &lists {
            for b in 0..list.count.div_ceil(BLOCK) {
                let first = list.first_id as usize + BLOCK * b;
                let last = (first + BLOCK).min(list.first_id as usize + list.count);
                let at = list.offset + b * size;
                pack_block(
                    &codes[first * cb..last * cb],
                    &factors[first..last],
                    d_pad,
                    &mut blocks[at..at + size],
                );
            }
        }

        let ccb = sq4::code_bytes(d_pad);
        let mut centers = vec![0u8; nlist * ccb];
        let (mut z, mut tmp) = (vec![0f32; dim], vec![0f32; dim]);
        let mut center_factors = Vec::with_capacity(nlist);

        for (&c, codes) in order.iter().zip(centers.chunks_exact_mut(ccb)) {
            let centroid = &clusters.centroids[c as usize * dim..(c as usize + 1) * dim];
            rotate_offset(&rotation, centroid, &mean, &mut z, &mut tmp);
            let f = sq4::encode(&z, codes);
            if !(fits(f.norm2) && f.lo.is_finite() && f.step.is_finite()) {
                return Err(invalid("a cluster center is too far from the mean to encode"));
            }
            center_factors.push(f);
        }

        let crb = sq8::center_row_bytes(dim);
        let mut parents = vec![0u8; nparents * crb];

        for (g, row) in clusters.parents.chunks_exact(dim).zip(parents.chunks_exact_mut(crb)) {
            sq8::write_center_row(g, &mean, row);
            if !fits(sq8::norm2(row, dim)) {
                return Err(invalid("a group center is too far from the mean to encode"));
            }
        }

        Ok(InMemory {
            dim,
            metric,
            seed: params.seed,
            page_size: params.page_size,
            mean,
            rotation,
            parents,
            centers,
            center_factors,
            parent_of: order.iter().map(|&c| clusters.parent_of[c as usize] as u16).collect(),
            lists,
            blocks,
            rows,
            ids,
            tier: Tier::best(),
        })
    }

    /// What a search reads.
    pub(crate) fn view(&self) -> View<'_> {
        let row_bytes = sq8::row_bytes(self.dim);
        View {
            dim: self.dim,
            metric: self.metric,
            mean: &self.mean,
            rotation: &self.rotation,
            centers: Centers {
                codes: &self.centers,
                factors: &self.center_factors,
            },
            lists: &self.lists,
            blocks: &self.blocks,
            rows: RowPages {
                bytes: &self.rows,
                row_bytes,
                per_page: self.page_size / row_bytes,
                page_size: self.page_size,
            },
            tier: self.tier,
        }
    }

    /// Searches for the `params.k` nearest vectors. The hits live in `ctx`.
    pub(crate) fn search<'a>(
        &self,
        query: &[f32],
        params: &SearchParams,
        ctx: &'a mut SearchCtx,
    ) -> Result<&'a [Hit], Error> {
        run(&self.view(), query, params, ctx)
    }

    /// Writes the `nprobe` clusters nearest the query into `out`, nearest first, by file position.
    pub(crate) fn route(
        &self,
        query: &[f32],
        nprobe: usize,
        ctx: &mut SearchCtx,
        out: &mut Vec<LeafId>,
    ) -> Result<(), Error> {
        run_route(&self.view(), query, nprobe, ctx, out)
    }

    /// The same index with another kernel tier, so tests can check that every tier gives the same results.
    #[cfg(test)]
    pub(crate) fn with_tier(mut self, tier: Tier) -> InMemory {
        self.tier = tier;
        self
    }
}

/// Checks the input and the clustering, and returns the number of clusters and of cluster groups.
fn check_input(n: usize, dim: usize, clusters: &Clusters) -> Result<(usize, usize), Error> {
    if !(MIN_DIM..=MAX_DIM).contains(&dim) {
        return Err(invalid("the dimension must be 128 to 4080"));
    }

    if clusters.dim != dim {
        return Err(invalid("the clusters' dimension doesn't match the vectors"));
    }

    let nlist = clusters.centroids.len() / dim;
    let nparents = clusters.parents.len() / dim;

    if nlist == 0 || nlist * dim != clusters.centroids.len() || nlist > u32::MAX as usize {
        return Err(invalid(
            "the cluster centers must be a whole number of rows, at least one",
        ));
    }

    if nparents == 0 || nparents * dim != clusters.parents.len() || nparents > usize::from(u16::MAX) {
        return Err(invalid("the group centers must be a whole number of rows, 1 to 65535"));
    }

    if clusters.assignment.len() != n || clusters.assignment.iter().any(|&c| c as usize >= nlist) {
        return Err(invalid("every vector needs the id of an existing cluster"));
    }

    if clusters.parent_of.len() != nlist || clusters.parent_of.iter().any(|&g| g as usize >= nparents) {
        return Err(invalid("every cluster needs the id of an existing group"));
    }

    if n > u32::MAX as usize - BLOCK {
        return Err(invalid("an index holds at most 2^32 − 33 vectors"));
    }

    if !clusters
        .centroids
        .iter()
        .chain(&clusters.parents)
        .all(|x| x.is_finite())
    {
        return Err(invalid("a cluster or group center has a value that isn't finite"));
    }

    Ok((nlist, nparents))
}

/// Calls `f` with each input row, in order, normalized for cosine. It fails on a value that isn't finite, on a zero
/// vector with cosine, or when `f` fails.
fn for_each_row(
    src: &dyn VectorSource,
    metric: Metric,
    mut f: impl FnMut(usize, &[f32]) -> Result<(), Error>,
) -> Result<(), Error> {
    let (n, dim) = (src.count(), src.dim());
    let mut row = vec![0f32; dim];

    for start in (0..n).step_by(CHUNK) {
        let chunk = src.rows(start, CHUNK.min(n - start));
        for (i, o) in chunk.chunks_exact(dim).enumerate() {
            if !o.iter().all(|x| x.is_finite()) {
                return Err(invalid("a vector has a value that isn't finite"));
            }
            row.copy_from_slice(o);
            if metric == Metric::Cosine && !normalize(&mut row) {
                return Err(invalid("a zero vector has no direction for cosine"));
            }
            f(start + i, &row)?;
        }
    }

    Ok(())
}

/// Whether a squared length, eg. f_add or a row's n2, leaves a search room to add another one.
fn fits(norm2: f32) -> bool {
    f64::from(norm2) <= LIMIT
}

/// Each cluster's place in file order, where `order` gives the cluster id at each position. A cluster's blocks follow
/// the ones before it, `size` bytes each, and its first internal id follows their vectors.
fn lists(assignment: &[u32], order: &[u32], size: usize) -> Vec<List> {
    let mut counts = vec![0usize; order.len()];

    for &c in assignment {
        counts[c as usize] += 1;
    }

    let (mut offset, mut first_id) = (0, 0u32);

    order
        .iter()
        .map(|&c| {
            let count = counts[c as usize];
            let list = List {
                offset,
                count,
                first_id,
            };
            offset += count.div_ceil(BLOCK) * size;
            first_id += count as u32;
            list
        })
        .collect()
}

#[cfg(test)]
#[path = "../../tests/unit/search/inmem.rs"]
mod tests;
