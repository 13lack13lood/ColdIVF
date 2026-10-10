//! The whole index in memory. It holds the same bytes as the sections of an index file, so a builder writes them out
//! as they are, and a search of the file gives exactly its results.

use std::io::Error;

use rayon::ThreadPool;
use rayon::prelude::*;

use super::route::Centers;
use super::{InPlace, LIMIT, List, RowPages, View, invalid, normalize, run, run_route};
use crate::build::order::leaf_order;
use crate::build::threads;
use crate::format::Layout;
use crate::kernels::Tier;
use crate::quantization::rabitq::{BLOCK, Encoder, Factors, block_bytes, code_bytes, pack_block};
use crate::quantization::rotation::Rotation;
use crate::quantization::sq4::{self, CenterFactors};
use crate::quantization::{padded_dim, rotate_offset, sq8};
use crate::{Anchor, BuildParams, Clusters, Hit, LeafId, Metric, SearchCtx, SearchParams, VectorSource};

/// The smallest dimension an index takes.
pub(crate) const MIN_DIM: usize = 128;

/// The largest dimension. Up to here a block's lookup sums fit in 16 bits and a row fits in a 4 KiB page.
pub(crate) const MAX_DIM: usize = 4080;

/// Rows read from the source at a time.
pub(crate) const CHUNK: usize = 4096;

/// Chunks encoded at once for each thread.
pub(crate) const BATCH: usize = 2;

/// The largest page size.
const MAX_PAGE: usize = 1 << 20;

pub(crate) struct InMemory {
    pub(crate) dim: usize,
    pub(crate) metric: Metric,
    /// What the 1-bit codes are measured from.
    pub(crate) anchor: Anchor,
    pub(crate) seed: u64,
    /// How many compactions the index has had.
    pub(crate) epoch: u64,
    pub(crate) page_size: usize,
    pub(crate) mean: Vec<f32>,
    pub(crate) rotation: Rotation,
    /// The cluster group centers as 8-bit rows without ids.
    pub(crate) parents: Vec<u8>,
    /// The 4-bit codes of the cluster centers, back to back, in file order.
    pub(crate) centers: Vec<u8>,
    pub(crate) center_factors: Vec<CenterFactors>,
    /// Each center's squared length as its codes decode, for codes measured from the clusters' centers. It is empty
    /// with the mean anchor.
    pub(crate) center_norms: Vec<f64>,
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
    /// `params.anchor`, `params.page_size`, `params.seed` and `params.threads`, and gives the same index for any number
    /// of threads. The vectors in each cluster keep their input order, and a vector's id is its input row.
    pub(crate) fn build(params: &BuildParams, src: &dyn VectorSource, clusters: &Clusters) -> Result<InMemory, Error> {
        Self::build_ids(params, src, clusters, None)
    }

    /// The same, with `ids[r]` as the id of input row r instead of r itself.
    pub(crate) fn build_ids(
        params: &BuildParams,
        src: &dyn VectorSource,
        clusters: &Clusters,
        ids: Option<&[u64]>,
    ) -> Result<InMemory, Error> {
        let pool = threads(params.threads)?;
        let (mut index, id_of) = InMemory::frame(params, src, clusters, ids, &pool)?;
        let (n, dim) = (src.count(), src.dim());
        let d_pad = padded_dim(dim);
        let Encoded { codes, factors, rows } = Encoding::new(&index, ids, &id_of).run(&chunks(src), &pool)?;
        index.blocks = pack(&index.lists, &codes, &factors, d_pad, &pool);
        index.rows = rows;
        index.ids = vec![0u64; n];
        let external = |r: usize| ids.map_or(r as u64, |ids| ids[r]);
        id_of
            .iter()
            .enumerate()
            .for_each(|(r, &id)| index.ids[id as usize] = external(r));
        Ok(index)
    }

    /// The index without its blocks, rows and ids, which take memory in proportion to the vectors, and each input
    /// row's internal id. It reads the vectors once, for their mean. A build then encodes the vectors into the frame,
    /// in memory or a part at a time.
    pub(crate) fn frame(
        params: &BuildParams,
        src: &dyn VectorSource,
        clusters: &Clusters,
        ids: Option<&[u64]>,
        pool: &ThreadPool,
    ) -> Result<(InMemory, Vec<u32>), Error> {
        let (n, dim) = (src.count(), src.dim());

        if ids.is_some_and(|ids| ids.len() != n) {
            return Err(invalid("every vector needs an id"));
        }

        check_settings(n, dim, params.page_size)?;
        let (nlist, nparents) = check_input(n, dim, clusters)?;
        let metric = params.metric;
        let mean = mean(&chunks(src), dim, metric, pool)?;
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
        // The vectors of each cluster keep their input order.
        let mut next: Vec<u32> = lists.iter().map(|l| l.first_id).collect();
        let id_of: Vec<u32> = clusters
            .assignment
            .iter()
            .map(|&c| {
                let p = position[c as usize] as usize;
                next[p] += 1;
                next[p] - 1
            })
            .collect();

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

        let center_norms = match params.anchor {
            Anchor::Mean => Vec::new(),
            Anchor::Cluster => sq4::decoded_norms(&centers, &center_factors, dim),
        };

        let crb = sq8::center_row_bytes(dim);
        let mut parents = vec![0u8; nparents * crb];

        for (g, row) in clusters.parents.chunks_exact(dim).zip(parents.chunks_exact_mut(crb)) {
            sq8::write_center_row(g, &mean, row);
            if !fits(sq8::norm2(row, dim)) {
                return Err(invalid("a group center is too far from the mean to encode"));
            }
        }

        let index = InMemory {
            dim,
            metric,
            anchor: params.anchor,
            seed: params.seed,
            epoch: 0,
            page_size: params.page_size,
            mean,
            rotation,
            parents,
            centers,
            center_factors,
            center_norms,
            parent_of: order.iter().map(|&c| clusters.parent_of[c as usize] as u16).collect(),
            lists,
            blocks: Vec::new(),
            rows: Vec::new(),
            ids: Vec::new(),
            tier: Tier::best(),
        };

        Ok((index, id_of))
    }

    /// An index without blocks, rows or ids, with the settings and epoch of `layout`, the mean `mean`, the rotation
    /// `rotation` and tables as a file stores them. A rewrite lays out its new file's small sections and header from
    /// it.
    pub(crate) fn with_tables(layout: &Layout, mean: Vec<f32>, rotation: Rotation, tables: Stored) -> InMemory {
        let center_norms = match layout.anchor {
            Anchor::Mean => Vec::new(),
            Anchor::Cluster => sq4::decoded_norms(&tables.centers, &tables.center_factors, layout.dim),
        };

        InMemory {
            dim: layout.dim,
            metric: layout.metric,
            anchor: layout.anchor,
            seed: layout.seed,
            epoch: layout.epoch,
            page_size: layout.page_size,
            mean,
            rotation,
            parents: tables.parents,
            centers: tables.centers,
            center_factors: tables.center_factors,
            center_norms,
            parent_of: tables.parent_of,
            lists: tables.lists,
            blocks: Vec::new(),
            rows: Vec::new(),
            ids: Vec::new(),
            tier: Tier::best(),
        }
    }

    /// What a search reads from the tables.
    pub(crate) fn view(&self) -> View<'_> {
        View {
            hides: super::scan::Hides::NONE,
            dim: self.dim,
            metric: self.metric,
            anchor: self.anchor,
            mean: &self.mean,
            rotation: &self.rotation,
            centers: Centers {
                codes: &self.centers,
                factors: &self.center_factors,
                norms: &self.center_norms,
            },
            lists: &self.lists,
            tier: self.tier,
        }
    }

    /// The blocks and rows.
    pub(crate) fn in_place(&self) -> InPlace<'_> {
        let row_bytes = sq8::row_bytes(self.dim);
        InPlace {
            blocks: &self.blocks,
            rows: RowPages {
                bytes: &self.rows,
                row_bytes,
                per_page: self.page_size / row_bytes,
                page_size: self.page_size,
            },
        }
    }

    /// Searches for the `params.k` nearest vectors. The hits live in `ctx`.
    pub(crate) fn search<'a>(
        &self,
        query: &[f32],
        params: &SearchParams,
        ctx: &'a mut SearchCtx,
    ) -> Result<&'a [Hit], Error> {
        run(&self.view(), &self.in_place(), query, params, ctx)
    }

    /// Writes the `nprobe` clusters nearest the query into `out`, nearest first, by file position, keeping only those
    /// within `reach`.
    pub(crate) fn route(
        &self,
        query: &[f32],
        nprobe: usize,
        reach: Option<f32>,
        ctx: &mut SearchCtx,
        out: &mut Vec<LeafId>,
    ) -> Result<(), Error> {
        run_route(&self.view(), query, nprobe, reach, nprobe, ctx, out)
    }

    /// The same index with another kernel tier, so tests can check that every tier gives the same results.
    #[cfg(test)]
    pub(crate) fn with_tier(mut self, tier: Tier) -> InMemory {
        self.tier = tier;
        self
    }
}

/// An index's tables as a file stores them, in file order: the group centers as 8-bit rows without ids, each cluster's
/// 4-bit center codes, back to back, and their factors, each cluster's group and each cluster's place.
pub(crate) struct Stored {
    pub(crate) parents: Vec<u8>,
    pub(crate) centers: Vec<u8>,
    pub(crate) center_factors: Vec<CenterFactors>,
    pub(crate) parent_of: Vec<u16>,
    pub(crate) lists: Vec<List>,
}

/// Checks the dimension, the vector count and the page size, the settings that need no clustering.
pub(crate) fn check_settings(n: usize, dim: usize, page_size: usize) -> Result<(), Error> {
    if !(MIN_DIM..=MAX_DIM).contains(&dim) {
        return Err(invalid("the dimension must be 128 to 4080"));
    }

    if n > u32::MAX as usize - BLOCK {
        return Err(invalid("an index holds at most 2^32 − 33 vectors"));
    }

    if !page_size.is_multiple_of(4096) || page_size < sq8::row_bytes(dim) || page_size > MAX_PAGE {
        return Err(invalid(
            "the page size must be a multiple of 4096, at most 1 MiB, that holds a row",
        ));
    }

    Ok(())
}

/// Checks the clustering of `n` vectors of `dim` values, and returns the number of clusters and of cluster groups.
fn check_input(n: usize, dim: usize, clusters: &Clusters) -> Result<(usize, usize), Error> {
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

/// Copies the input vector `x` into `o`, normalized for cosine. It fails on a value that isn't finite, or on a zero
/// vector with cosine.
fn load(x: &[f32], metric: Metric, o: &mut [f32]) -> Result<(), Error> {
    if !x.iter().all(|v| v.is_finite()) {
        return Err(invalid("a vector has a value that isn't finite"));
    }

    o.copy_from_slice(x);

    if metric == Metric::Cosine && !normalize(o) {
        return Err(invalid("a zero vector has no direction for cosine"));
    }

    Ok(())
}

/// The mean of the input vectors, normalized for cosine. Each chunk is summed on its own and the sums are added in
/// order, so the mean is the same for any number of threads.
fn mean(chunks: &[&[f32]], dim: usize, metric: Metric, pool: &ThreadPool) -> Result<Vec<f32>, Error> {
    let sums: Vec<Result<Vec<f64>, Error>> = pool.install(|| {
        chunks
            .par_iter()
            .map(|chunk| {
                let (mut sum, mut o) = (vec![0f64; dim], vec![0f32; dim]);
                for x in chunk.chunks_exact(dim) {
                    load(x, metric, &mut o)?;
                    sum.iter_mut().zip(&o).for_each(|(s, &v)| *s += f64::from(v));
                }
                Ok(sum)
            })
            .collect()
    });
    let (mut total, mut n) = (vec![0f64; dim], 0usize);

    for (chunk, sum) in chunks.iter().zip(sums) {
        total.iter_mut().zip(sum?).for_each(|(t, s)| *t += s);
        n += chunk.len() / dim;
    }

    Ok(total
        .iter()
        .map(|&t| if n == 0 { 0.0 } else { (t / n as f64) as f32 })
        .collect())
}

/// The rows of `src` in chunks of `CHUNK`, as borrowed slices, so a source in a mapped file is never copied.
pub(crate) fn chunks(src: &dyn VectorSource) -> Vec<&[f32]> {
    let n = src.count();
    (0..n)
        .step_by(CHUNK)
        .map(|start| src.rows(start, CHUNK.min(n - start)))
        .collect()
}

/// What encoding a vector needs.
pub(crate) struct Encoding<'a> {
    dim: usize,
    metric: Metric,
    anchor: Anchor,
    mean: &'a [f32],
    rotation: &'a Rotation,
    page_size: usize,
    /// The id of each input row, when it isn't the row's number.
    ids: Option<&'a [u64]>,
    /// The internal id of each input row, which tells its cluster.
    id_of: &'a [u32],
    lists: &'a [List],
    /// Each cluster's rotated offset P(ĉ − μ) as its codes decode, one after another in file order, for codes measured
    /// from the clusters' centers. A build computes them once, which takes the cluster count times the dimension in
    /// f32, eg. 19 MB for 6250 clusters of 768 dimensions at 1M vectors and 192 MB at 10M. With the mean anchor it is
    /// empty.
    centers: Vec<f32>,
}

/// Vectors encoded into their 1-bit codes, their factors and their 8-bit rows. A chunk's are in input order, with its
/// rows back to back. An index's are by internal id, with its rows in pages.
pub(crate) struct Encoded {
    pub(crate) codes: Vec<u8>,
    pub(crate) factors: Vec<Factors>,
    pub(crate) rows: Vec<u8>,
}

impl<'a> Encoding<'a> {
    /// The encoding of the vectors of `index`'s frame, whose input row r has internal id `id_of[r]`, with `ids[r]` as
    /// its id when given.
    pub(crate) fn new(index: &'a InMemory, ids: Option<&'a [u64]>, id_of: &'a [u32]) -> Encoding<'a> {
        let centers = match index.anchor {
            Anchor::Mean => Vec::new(),
            Anchor::Cluster => decoded_centers(index),
        };

        Encoding {
            dim: index.dim,
            metric: index.metric,
            anchor: index.anchor,
            mean: &index.mean,
            rotation: &index.rotation,
            page_size: index.page_size,
            ids,
            id_of,
            lists: &index.lists,
            centers,
        }
    }

    /// Encodes every input vector into its 1-bit code, its factors and its 8-bit row, and puts each at its internal
    /// id. The threads encode a batch of chunks at a time, which this thread then puts in place in input order, so only
    /// a few encoded chunks ever wait.
    fn run(&self, chunks: &[&[f32]], pool: &ThreadPool) -> Result<Encoded, Error> {
        let (n, id_of) = (self.id_of.len(), self.id_of);
        let (cb, rb) = (code_bytes(padded_dim(self.dim)), sq8::row_bytes(self.dim));
        let per_page = self.page_size / rb;
        let mut codes = vec![0u8; n * cb];
        let mut factors = vec![Factors::EMPTY; n];
        let mut rows = vec![0u8; n.div_ceil(per_page) * self.page_size];
        let batch = BATCH * pool.current_num_threads();

        for (b, group) in chunks.chunks(batch).enumerate() {
            let encoded: Vec<Result<Encoded, Error>> = pool.install(|| {
                group
                    .par_iter()
                    .enumerate()
                    .map(|(i, chunk)| self.chunk(chunk, (b * batch + i) * CHUNK))
                    .collect()
            });
            for (i, e) in encoded.into_iter().enumerate() {
                let e = e?;
                let first = (b * batch + i) * CHUNK;
                for (j, f) in e.factors.iter().enumerate() {
                    let id = id_of[first + j] as usize;
                    codes[id * cb..(id + 1) * cb].copy_from_slice(&e.codes[j * cb..(j + 1) * cb]);
                    factors[id] = *f;
                    let at = (id / per_page) * self.page_size + (id % per_page) * rb;
                    rows[at..at + rb].copy_from_slice(&e.rows[j * rb..(j + 1) * rb]);
                }
            }
        }

        Ok(Encoded { codes, factors, rows })
    }

    /// Encodes the vectors of one chunk, whose first input row is `first`.
    pub(crate) fn chunk(&self, chunk: &[f32], first: usize) -> Result<Encoded, Error> {
        let (dim, count) = (self.dim, chunk.len() / self.dim);
        let (cb, rb) = (code_bytes(padded_dim(dim)), sq8::row_bytes(dim));
        let mut e = Encoded {
            codes: vec![0u8; count * cb],
            factors: Vec::with_capacity(count),
            rows: vec![0u8; count * rb],
        };
        let mut encoder = Encoder::new(self.rotation, self.mean);
        let mut o = vec![0f32; dim];
        let outputs = e.codes.chunks_exact_mut(cb).zip(e.rows.chunks_exact_mut(rb));

        for (i, (x, (code, row))) in chunk.chunks_exact(dim).zip(outputs).enumerate() {
            load(x, self.metric, &mut o)?;
            let f = match self.anchor {
                Anchor::Mean => encoder.encode(&o, code),
                Anchor::Cluster => encoder.encode_from(&o, self.center_of(first + i), code),
            };
            let id = self.ids.map_or((first + i) as u64, |ids| ids[first + i]);
            sq8::write_row(&o, self.mean, id, row);
            if !(fits(f.f_add.abs()) && f.f_rescale.is_finite() && f.f_err.is_finite() && fits(sq8::norm2(row, dim))) {
                return Err(invalid("a vector is too far from the mean to encode"));
            }
            e.factors.push(f);
        }

        Ok(e)
    }

    /// The decoded center of the cluster that holds input row `r`.
    fn center_of(&self, r: usize) -> &[f32] {
        let id = self.id_of[r] as usize;
        let leaf = self.lists.partition_point(|l| l.first_id as usize + l.count <= id);
        &self.centers[leaf * self.dim..(leaf + 1) * self.dim]
    }
}

/// Every cluster's rotated offset P(ĉ − μ) as its codes decode, one after another in file order.
fn decoded_centers(index: &InMemory) -> Vec<f32> {
    let (dim, size) = (index.dim, sq4::code_bytes(padded_dim(index.dim)));
    let mut centers = vec![0f32; index.center_factors.len() * dim];
    let codes = index.centers.chunks_exact(size).zip(&index.center_factors);
    codes
        .zip(centers.chunks_exact_mut(dim))
        .for_each(|((codes, f), out)| sq4::decode(codes, f, out));
    centers
}

/// Packs each cluster's codes into its blocks, a cluster at a time on the pool's threads. The clusters' blocks follow
/// one another.
fn pack(lists: &[List], codes: &[u8], factors: &[Factors], d_pad: usize, pool: &ThreadPool) -> Vec<u8> {
    let (cb, size) = (code_bytes(d_pad), block_bytes(d_pad));
    let mut blocks = vec![0u8; lists.last().map_or(0, |l| l.offset + l.count.div_ceil(BLOCK) * size)];
    let mut parts = Vec::with_capacity(lists.len());
    let mut rest = &mut blocks[..];

    for list in lists {
        let (mine, tail) = std::mem::take(&mut rest).split_at_mut(list.count.div_ceil(BLOCK) * size);
        parts.push(mine);
        rest = tail;
    }

    pool.install(|| {
        parts.into_par_iter().zip(lists).for_each(|(out, list)| {
            for (b, block) in out.chunks_exact_mut(size).enumerate() {
                let first = list.first_id as usize + BLOCK * b;
                let last = (first + BLOCK).min(list.first_id as usize + list.count);
                pack_block(&codes[first * cb..last * cb], &factors[first..last], d_pad, block);
            }
        })
    });

    blocks
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

    place(order.iter().map(|&c| counts[c as usize]), size)
}

/// The places of clusters of these vector counts, in file order, laid out as `lists` lays them out.
pub(crate) fn place(counts: impl Iterator<Item = usize>, size: usize) -> Vec<List> {
    let (mut offset, mut first_id) = (0, 0u32);

    counts
        .map(|count| {
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
