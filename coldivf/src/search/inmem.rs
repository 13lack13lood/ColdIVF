//! The whole index in RAM, built from the same codes as an index file. Other searches are checked against it.

use std::io::Error;

use super::rerank::rerank;
use super::route::{Centers, route};
use super::scan::scan_list;
use super::{check_search, invalid, normalize, scratch_for};
use crate::kernels::Tier;
use crate::quantization::rabitq::{BLOCK, Encoder, Factors, block_bytes, code_bytes, pack_block};
use crate::quantization::rotation::Rotation;
use crate::quantization::sq4::{self, CenterFactors};
use crate::quantization::{padded_dim, rotate_offset, sq8};
use crate::{BuildParams, Clusters, Hit, Metric, SearchCtx, SearchParams, VectorSource};

/// The largest dimension. Up to here a block's lookup sums fit in 16 bits and a row fits in a 4 KiB page.
pub(crate) const MAX_DIM: usize = 4080;

/// Rows read from the source at a time.
const CHUNK: usize = 4096;

/// One cluster's place in the index.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct List {
    /// Its first block.
    pub(crate) block: usize,
    /// Its vectors, which have internal ids `first_id..first_id + count`.
    pub(crate) count: usize,
    pub(crate) first_id: u32,
}

pub(crate) struct InMemory {
    pub(crate) dim: usize,
    pub(crate) metric: Metric,
    pub(crate) mean: Vec<f32>,
    pub(crate) rotation: Rotation,
    /// The 4-bit codes of the cluster centers, back to back.
    pub(crate) centers: Vec<u8>,
    pub(crate) center_factors: Vec<CenterFactors>,
    pub(crate) lists: Vec<List>,
    /// Every cluster's blocks, one cluster after another.
    pub(crate) blocks: Vec<u8>,
    /// The 8-bit rows, by internal id.
    pub(crate) rows: Vec<u8>,
    tier: Tier,
}

impl InMemory {
    /// Builds the index from `src` and an imported clustering, with `params.metric` and `params.seed`. The clusters keep
    /// their id order, and the vectors in each cluster keep their input order. A vector's id is its input row.
    pub(crate) fn build(params: &BuildParams, src: &dyn VectorSource, clusters: &Clusters) -> Result<InMemory, Error> {
        let (n, dim) = (src.count(), src.dim());
        let nlist = check_clusters(n, dim, clusters)?;
        let metric = params.metric;

        let mut sum = vec![0f64; dim];

        for_each_row(src, metric, |_, o| {
            sum.iter_mut().zip(o).for_each(|(s, &x)| *s += f64::from(x));
        })?;

        let mean: Vec<f32> = sum
            .iter()
            .map(|&s| if n == 0 { 0.0 } else { (s / n as f64) as f32 })
            .collect();
        let rotation = Rotation::new(dim, params.seed);

        let lists = lists(&clusters.assignment, nlist);
        let d_pad = padded_dim(dim);
        let (cb, rb) = (code_bytes(d_pad), sq8::row_bytes(dim));
        let mut codes = vec![0u8; n * cb];
        let mut factors = vec![Factors::EMPTY; n];
        let mut rows = vec![0u8; n * rb];
        let mut next: Vec<u32> = lists.iter().map(|l| l.first_id).collect();
        let mut encoder = Encoder::new(&rotation, &mean);

        for_each_row(src, metric, |r, o| {
            let list = clusters.assignment[r] as usize;
            let id = next[list] as usize;
            next[list] += 1;
            factors[id] = encoder.encode(o, &mut codes[id * cb..(id + 1) * cb]);
            sq8::write_row(o, &mean, r as u64, &mut rows[id * rb..(id + 1) * rb]);
        })?;

        let size = block_bytes(d_pad);
        let total = lists.last().map_or(0, |l| l.block + l.count.div_ceil(BLOCK));
        let mut blocks = vec![0u8; total * size];

        for list in &lists {
            for b in 0..list.count.div_ceil(BLOCK) {
                let first = list.first_id as usize + BLOCK * b;
                let last = (first + BLOCK).min(list.first_id as usize + list.count);
                let out = &mut blocks[(list.block + b) * size..(list.block + b + 1) * size];
                pack_block(&codes[first * cb..last * cb], &factors[first..last], d_pad, out);
            }
        }

        let ccb = sq4::code_bytes(d_pad);
        let mut centers = vec![0u8; nlist * ccb];
        let (mut z, mut tmp) = (vec![0f32; dim], vec![0f32; dim]);
        let mut center_factors = Vec::with_capacity(nlist);

        for (c, codes) in clusters.centroids.chunks_exact(dim).zip(centers.chunks_exact_mut(ccb)) {
            rotate_offset(&rotation, c, &mean, &mut z, &mut tmp);
            center_factors.push(sq4::encode(&z, codes));
        }

        Ok(InMemory {
            dim,
            metric,
            mean,
            rotation,
            centers,
            center_factors,
            lists,
            blocks,
            rows,
            tier: Tier::best(),
        })
    }

    /// Searches for the `params.k` nearest vectors. The hits live in `ctx`.
    pub(crate) fn search<'a>(
        &self,
        query: &[f32],
        params: &SearchParams,
        ctx: &'a mut SearchCtx,
    ) -> Result<&'a [Hit], Error> {
        check_search(query, self.dim, params, &ctx.limits)?;

        let SearchCtx { limits, scratch, hits } = ctx;
        let s = scratch_for(scratch, self.dim, self.lists.len(), limits);
        s.query.copy_from_slice(query);

        if self.metric == Metric::Cosine && !normalize(&mut s.query) {
            return Err(invalid("a zero query has no direction for cosine"));
        }

        let mut t2 = 0f64;

        for ((o, &q), &m) in s.offset.iter_mut().zip(&s.query).zip(&self.mean) {
            *o = q - m;
            t2 += f64::from(*o).powi(2);
        }

        self.rotation.apply(&s.offset, &mut s.rotated, &mut s.tmp);
        s.lut.prepare(&s.rotated, t2.sqrt());
        s.q8.prepare(&s.rotated);

        let centers = Centers {
            codes: &self.centers,
            factors: &self.center_factors,
        };
        route(
            self.tier,
            &centers,
            &s.q8,
            s.lut.k.t2,
            params.nprobe,
            &mut s.dots,
            &mut s.routes,
        );

        s.candidates.reset(params.rerank.max(params.k));
        let size = block_bytes(padded_dim(self.dim));

        for r in &s.routes {
            let list = self.lists[r.id as usize];
            let blocks = &self.blocks[list.block * size..(list.block + list.count.div_ceil(BLOCK)) * size];
            scan_list(
                self.tier,
                blocks,
                list.count,
                list.first_id,
                &s.lut,
                &mut s.candidates,
                &mut s.survivors,
            );
        }

        let rb = sq8::row_bytes(self.dim);
        let row = |id: u32| &self.rows[id as usize * rb..(id as usize + 1) * rb];
        rerank(
            s.candidates.as_slice(),
            row,
            &s.offset,
            params.k,
            self.metric,
            &mut s.rescored,
            hits,
        );

        Ok(&hits[..])
    }

    /// The same index with another kernel tier, so tests can check that every tier gives the same results.
    #[cfg(test)]
    pub(crate) fn with_tier(mut self, tier: Tier) -> InMemory {
        self.tier = tier;
        self
    }
}

/// Checks the input and the clustering, and returns the number of clusters.
fn check_clusters(n: usize, dim: usize, clusters: &Clusters) -> Result<usize, Error> {
    if !(2..=MAX_DIM).contains(&dim) {
        return Err(invalid("the dimension must be 2 to 4080"));
    }

    if clusters.dim != dim {
        return Err(invalid("the clusters' dimension doesn't match the vectors"));
    }

    let nlist = clusters.centroids.len() / dim;

    if nlist == 0 || nlist * dim != clusters.centroids.len() {
        return Err(invalid(
            "the cluster centers must be a whole number of rows, at least one",
        ));
    }

    if clusters.assignment.len() != n || clusters.assignment.iter().any(|&c| c as usize >= nlist) {
        return Err(invalid("every vector needs the id of an existing cluster"));
    }

    if n > u32::MAX as usize - BLOCK {
        return Err(invalid("an index holds at most 2^32 − 33 vectors"));
    }

    if !clusters.centroids.iter().all(|x| x.is_finite()) {
        return Err(invalid("a cluster center has a value that isn't finite"));
    }

    Ok(nlist)
}

/// Calls `f` with each input row, in order, normalized for cosine. It fails on a value that isn't finite, or on a zero
/// vector with cosine.
fn for_each_row(src: &dyn VectorSource, metric: Metric, mut f: impl FnMut(usize, &[f32])) -> Result<(), Error> {
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
            f(start + i, &row);
        }
    }

    Ok(())
}

/// Each cluster's place in cluster id order, with its first block, its vector count and its first internal id.
fn lists(assignment: &[u32], nlist: usize) -> Vec<List> {
    let mut counts = vec![0usize; nlist];

    for &c in assignment {
        counts[c as usize] += 1;
    }

    let (mut block, mut first_id) = (0, 0u32);

    counts
        .iter()
        .map(|&count| {
            let list = List { block, count, first_id };
            block += count.div_ceil(BLOCK);
            first_id += count as u32;
            list
        })
        .collect()
}

#[cfg(test)]
#[path = "../../tests/unit/search/inmem.rs"]
mod tests;
