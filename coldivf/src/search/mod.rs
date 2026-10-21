//! Search, which picks the clusters to read, scans their 1-bit codes and rescores the best candidates with their 8-bit
//! rows. A query runs in three steps:
//! 1. Route: score every cluster center with its 4-bit code and keep the `nprobe` nearest, or those of them within the
//!    reach.
//! 2. Scan: estimate the distance to every vector in those clusters, 32 at a time, and keep the best candidates. Codes
//!    measured from the clusters' centers first take the query's distance to each center. The pending inserts of the
//!    clusters ranked next, up to `PENDING_DEPTH` of `nprobe`, are scanned too.
//! 3. Rerank: rescore the candidates with their 8-bit rows and return the top `k`.
//!
//! Candidates and results are ordered by score and then internal id, so they never depend on the order of the scan.

pub(crate) mod calibrate;
pub(crate) mod candidates;
pub(crate) mod cold;
pub(crate) mod inmem;
pub(crate) mod rerank;
pub(crate) mod route;
pub(crate) mod scan;

use std::io::{Error, ErrorKind};

use crate::kernels::Tier;
use crate::quantization::padded_dim;
use crate::quantization::rabitq::{BLOCK, QueryLut, block_bytes};
use crate::quantization::rotation::Rotation;
use crate::quantization::sq4::{QueryS8, QueryS32};
use crate::update::view::ClusterGroups;
use crate::{Anchor, Hit, LeafId, Limits, Metric, SearchCtx, SearchParams};
use candidates::{Candidate, Candidates};
use rerank::rerank;
use route::{CHUNK, Centers, route_chunked, within_reach};
pub(crate) use scan::Pending;
use scan::{Hides, scan_inserts, scan_probed};

/// The largest squared length a query's or a vector's offset from the mean may have, a quarter of the largest f32, so
/// adding two of them can't overflow.
pub(crate) const LIMIT: f64 = f32::MAX as f64 / 4.0;

/// The share of `nprobe`, as a fraction, of the clusters nearest the query that a search scans the pending inserts
/// of, though it reads the file's vectors of only `nprobe` of them. An insert joins its nearest cluster, which a query
/// near the insert may rank just past the clusters it reads. Scanning the inserts of the next clusters finds most of
/// those, for far less than a scan of every insert. A few more clusters' inserts cost less than the higher `nprobe`
/// that the same recall would need otherwise, but many more cost more than it: on Cohere 1M with 100K pending inserts,
/// uniform or drifting, 5/4 of `nprobe` reached recall@10 0.90 and 0.95 within 3% of the fastest share tried, from 1 to
/// 4 times `nprobe`.
pub(crate) const PENDING_DEPTH: (usize, usize) = (5, 4);

/// The number of clusters a search routes to: `nprobe`, or `PENDING_DEPTH` of it, rounded up, when the index has
/// pending inserts.
pub(crate) fn depth(nprobe: usize, pending: bool) -> usize {
    let (num, den) = PENDING_DEPTH;
    match pending {
        true => nprobe.saturating_mul(num).div_ceil(den),
        false => nprobe,
    }
}

/// One cluster's place in an index.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct List {
    /// Where its blocks start, in bytes from the start of all the blocks.
    pub(crate) offset: usize,
    /// Its vectors, which have internal ids `first_id..first_id + count`.
    pub(crate) count: usize,
    pub(crate) first_id: u32,
}

/// The 8-bit rows, `per_page` to a page so that no row crosses a page boundary.
pub(crate) struct RowPages<'a> {
    pub(crate) bytes: &'a [u8],
    pub(crate) row_bytes: usize,
    pub(crate) per_page: usize,
    pub(crate) page_size: usize,
}

impl<'a> RowPages<'a> {
    /// The row of internal id `id`.
    pub(crate) fn row(&self, id: u32) -> &'a [u8] {
        let id = id as usize;
        let at = (id / self.per_page) * self.page_size + (id % self.per_page) * self.row_bytes;
        &self.bytes[at..at + self.row_bytes]
    }
}

/// What a search reads from an index's tables. They are the same whether the index is in RAM, in a mapped file or read
/// cold.
pub(crate) struct View<'a> {
    pub(crate) dim: usize,
    pub(crate) metric: Metric,
    /// What the 1-bit codes are measured from.
    pub(crate) anchor: Anchor,
    pub(crate) mean: &'a [f32],
    pub(crate) rotation: &'a Rotation,
    pub(crate) centers: Centers<'a>,
    pub(crate) lists: &'a [List],
    pub(crate) tier: Tier,
    /// The rows that the search skips, as of its snapshot of the changes.
    pub(crate) hides: Hides<'a>,
}

/// Every cluster's blocks and every row, read in place from RAM or a mapped file.
pub(crate) struct InPlace<'a> {
    pub(crate) blocks: &'a [u8],
    pub(crate) rows: RowPages<'a>,
}

/// Searches `view` for the `params.k` nearest vectors, reading the blocks and rows from `data`. The hits live in `ctx`.
pub(crate) fn run<'a>(
    view: &View,
    data: &InPlace,
    query: &[f32],
    params: &SearchParams,
    ctx: &'a mut SearchCtx,
) -> Result<&'a [Hit], Error> {
    check_search(query, view.dim, params, &ctx.limits)?;

    let SearchCtx {
        limits, scratch, hits, ..
    } = ctx;
    let s = scratch_for(scratch, view.dim, limits);
    prepare_query(view, query, s)?;
    scan_in_place(view, data.blocks, params, s, None);
    let row = |id: u32| data.rows.row(id);
    rerank(
        s.candidates.as_slice(),
        row,
        &s.offset,
        params.k,
        view.metric,
        &mut s.rescored,
        hits,
    )?;

    Ok(&hits[..])
}

/// Routes the query that `s` holds and scans the `params.nprobe` nearest clusters, or those of them within
/// `params.reach`, reading their blocks in place from `blocks`. With `pending`, it then scans the inserts of the
/// clusters ranked after those, up to the `depth` nearest. It leaves the best `max(params.rerank, params.k)`
/// candidates in `s`, and the clusters whose blocks it read in `s.routes`.
pub(crate) fn scan_in_place(
    view: &View,
    blocks: &[u8],
    params: &SearchParams,
    s: &mut Scratch,
    pending: Option<Pending>,
) {
    route_chunked(
        view.tier,
        &view.centers,
        &s.q8,
        s.lut.mean.t2,
        depth(params.nprobe, pending.is_some()),
        &mut s.chunk,
        &mut s.best,
        &mut s.routes,
    );
    let read = within_reach(&s.routes[..params.nprobe.min(s.routes.len())], params.reach);
    s.candidates.reset(params.rerank.max(params.k));
    let Scratch {
        routes,
        lut,
        q32,
        candidates,
        survivors,
        ..
    } = s;
    let leaves = routes[..read].iter().map(|r| r.id);
    scan_lists(view, blocks, leaves, lut, q32, pending, candidates, survivors);

    if let Some(p) = pending {
        let next = routes[read..].iter().map(|r| r.id);
        scan_next(view, next, lut, q32, p, candidates, survivors);
    }

    routes.truncate(read);
}

/// Scans the clusters `leaves`, reading their blocks in place from `blocks`, with their inserts in `pending`, and
/// offers their vectors to `candidates`. While it scans one cluster, the memory of the next one's inserts starts
/// coming into the cache.
#[allow(clippy::too_many_arguments)]
pub(crate) fn scan_lists(
    view: &View,
    blocks: &[u8],
    leaves: impl Iterator<Item = LeafId>,
    lut: &mut QueryLut,
    q32: &QueryS32,
    pending: Option<Pending>,
    candidates: &mut Candidates,
    survivors: &mut [Candidate; BLOCK],
) {
    let size = block_bytes(padded_dim(view.dim));
    let mut leaves = leaves.peekable();
    // The inserts of the cluster that comes next, which the touch before its scan looked up.
    let mut ahead: Option<ClusterGroups> = None;

    while let Some(leaf) = leaves.next() {
        let here = pending.map(|p| (p, ahead.take().unwrap_or_else(|| p.of(leaf))));
        if let (Some(p), Some(&next)) = (pending, leaves.peek()) {
            ahead = Some(p.touch(next));
        }
        let list = view.lists[leaf as usize];
        let blocks = &blocks[list.offset..list.offset + list.count.div_ceil(BLOCK) * size];
        scan_probed(view, leaf, blocks, lut, q32, here, candidates, survivors);
    }
}

/// Scans only the pending inserts in `pending` of the clusters `leaves`, eg. those that the route ranks past the ones
/// a search reads, and offers them to `candidates`. While it scans one cluster's inserts, the memory of the next one's
/// starts coming into the cache.
pub(crate) fn scan_next(
    view: &View,
    leaves: impl Iterator<Item = LeafId>,
    lut: &mut QueryLut,
    q32: &QueryS32,
    pending: Pending,
    candidates: &mut Candidates,
    survivors: &mut [Candidate; BLOCK],
) {
    let mut leaves = leaves.peekable();
    let mut ahead: Option<ClusterGroups> = None;

    while let Some(leaf) = leaves.next() {
        let here = ahead.take().unwrap_or_else(|| pending.of(leaf));
        if let Some(&next) = leaves.peek() {
            ahead = Some(pending.touch(next));
        }
        scan_inserts(view, leaf, lut, q32, pending, here, candidates, survivors);
    }
}

/// Writes the `nprobe` clusters nearest the query into `out`, nearest first, keeping only those within `reach`, and the
/// clusters ranked after those, up to the `deep` nearest, into the context's `next`, which then belongs to no index
/// until the caller names one.
pub(crate) fn run_route(
    view: &View,
    query: &[f32],
    nprobe: usize,
    reach: Option<f32>,
    deep: usize,
    ctx: &mut SearchCtx,
    out: &mut Vec<LeafId>,
) -> Result<(), Error> {
    let params = SearchParams {
        k: 0,
        nprobe,
        rerank: 0,
        reach,
    };
    check_search(query, view.dim, &params, &ctx.limits)?;

    let SearchCtx {
        limits,
        scratch,
        next,
        next_of,
        ..
    } = ctx;
    *next_of = None;
    let s = scratch_for(scratch, view.dim, limits);
    prepare_route(view, query, s)?;
    route_chunked(
        view.tier,
        &view.centers,
        &s.q8,
        s.lut.mean.t2,
        deep.max(nprobe),
        &mut s.chunk,
        &mut s.best,
        &mut s.routes,
    );
    let read = within_reach(&s.routes[..nprobe.min(s.routes.len())], reach);
    out.clear();
    out.extend(s.routes[..read].iter().map(|r| r.id));
    next.clear();
    next.extend(s.routes[read..].iter().map(|r| r.id));

    Ok(())
}

/// Prepares the query for a search of `view`. It normalizes the query for cosine, then writes its offset q − μ, that
/// offset rotated and its two quantized forms into `s`. For codes measured from the clusters' centers it also writes
/// the query's fixed-point form.
pub(crate) fn prepare_query(view: &View, query: &[f32], s: &mut Scratch) -> Result<(), Error> {
    let t2 = prepare_route(view, query, s)?;

    if view.anchor == Anchor::Cluster {
        s.q32.prepare(&s.rotated, t2);
    }

    Ok(())
}

/// The part of `prepare_query` that routing needs, all but the fixed-point form. Returns T² = ‖q − μ‖².
fn prepare_route(view: &View, query: &[f32], s: &mut Scratch) -> Result<f64, Error> {
    s.query.copy_from_slice(query);

    if view.metric == Metric::Cosine && !normalize(&mut s.query) {
        return Err(invalid("a zero query has no direction for cosine"));
    }

    let mut t2 = 0f64;

    for ((o, &q), &m) in s.offset.iter_mut().zip(&s.query).zip(view.mean) {
        *o = q - m;
        t2 += f64::from(*o).powi(2);
    }

    // An estimate adds T², or the squared distance to a cluster's center, to a vector's f_add, and a route score adds
    // T² to a center's ‖z‖², so T² needs room to spare.
    if t2 > LIMIT {
        return Err(invalid("the query is too far from the index's mean"));
    }

    view.rotation.apply(&s.offset, &mut s.rotated, &mut s.tmp);
    s.lut.prepare(&s.rotated, t2.sqrt());
    s.q8.prepare(&s.rotated);
    Ok(t2)
}

/// The buffers of one search, sized for one index's dimension and a context's limits. Routing scores the clusters a
/// chunk at a time, so no buffer of a search grows with the cluster count, and a `SearchCtx` keeps one scratch per
/// dimension, so a search allocates nothing once it has one.
pub(crate) struct Scratch {
    dim: usize,
    /// The query, normalized for cosine.
    pub(crate) query: Vec<f32>,
    /// The query's offset q − μ from the index's mean μ.
    pub(crate) offset: Vec<f32>,
    /// That offset rotated by the index's rotation P, i.e. P(q − μ).
    pub(crate) rotated: Vec<f32>,
    pub(crate) tmp: Vec<f32>,
    pub(crate) lut: QueryLut,
    pub(crate) q8: QueryS8,
    /// The offset rotated in 32-bit fixed point, for the query's distance to the clusters' centers.
    pub(crate) q32: QueryS32,
    /// One dot product per cluster, for a router that scores every cluster at once, eg. calibration's.
    pub(crate) dots: Vec<i32>,
    /// The dot products of one chunk of clusters, and the nearest clusters so far, for a search's routing.
    pub(crate) chunk: Vec<i32>,
    pub(crate) best: Candidates,
    /// The clusters to read, nearest first.
    pub(crate) routes: Vec<Candidate>,
    pub(crate) candidates: Candidates,
    pub(crate) survivors: [Candidate; BLOCK],
    pub(crate) rescored: Vec<Candidate>,
}

impl Scratch {
    /// Whether this scratch fits an index with this dimension.
    pub(crate) fn fits(&self, dim: usize) -> bool {
        self.dim == dim
    }

    pub(crate) fn dim(&self) -> usize {
        self.dim
    }

    /// Buffers for a search of `dim` dimensions within `limits`. Calibration's router scores all `nlist` clusters at
    /// once, so it keeps that many dot products and routes, where a search's scratch has 0.
    pub(crate) fn new(dim: usize, nlist: usize, limits: &Limits) -> Scratch {
        let keep = limits.max_rerank.max(limits.max_k);
        Scratch {
            dim,
            query: vec![0.0; dim],
            offset: vec![0.0; dim],
            rotated: vec![0.0; dim],
            tmp: vec![0.0; dim],
            lut: QueryLut::new(dim),
            q8: QueryS8::new(dim),
            q32: QueryS32::new(dim),
            dots: vec![0; nlist],
            chunk: vec![0; CHUNK],
            best: Candidates::new(depth(limits.max_nprobe, true)),
            routes: Vec::with_capacity(nlist.max(depth(limits.max_nprobe, true))),
            candidates: Candidates::new(keep),
            survivors: [Candidate::NONE; BLOCK],
            rescored: Vec::with_capacity(keep),
        }
    }
}

/// The scratch in `slots` for an index with this dimension, made first if there isn't one yet. It fits an index of any
/// cluster count, which every maintenance run changes.
pub(crate) fn scratch_for<'a>(slots: &'a mut Vec<Scratch>, dim: usize, limits: &Limits) -> &'a mut Scratch {
    let i = match slots.iter().position(|s| s.dim == dim) {
        Some(i) => i,
        None => {
            slots.push(Scratch::new(dim, 0, limits));
            slots.len() - 1
        }
    };

    &mut slots[i]
}

/// Checks a search's arguments, so a bad one fails with `InvalidInput` instead of a panic.
pub(crate) fn check_search(query: &[f32], dim: usize, params: &SearchParams, limits: &Limits) -> Result<(), Error> {
    if query.len() != dim {
        return Err(invalid("the query's length doesn't match the index's dimension"));
    }

    if params.k > limits.max_k || params.nprobe > limits.max_nprobe || params.rerank > limits.max_rerank {
        return Err(invalid("the search settings exceed the limits of the search context"));
    }

    if !query.iter().all(|x| x.is_finite()) {
        return Err(invalid("the query has a value that isn't finite"));
    }

    if params.reach.is_some_and(|r| !(r.is_finite() && r >= 0.0)) {
        return Err(invalid("the reach must be finite and at least 0"));
    }

    Ok(())
}

/// Scales `v` to length 1 for cosine. Returns false for a zero vector, which has no direction.
pub(crate) fn normalize(v: &mut [f32]) -> bool {
    let norm = v.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>().sqrt();

    if norm == 0.0 {
        return false;
    }

    v.iter_mut().for_each(|x| *x = (f64::from(*x) / norm) as f32);
    true
}

pub(crate) fn invalid(message: &str) -> Error {
    Error::new(ErrorKind::InvalidInput, message)
}
