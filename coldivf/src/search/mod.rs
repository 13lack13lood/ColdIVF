//! Search, which picks the clusters to read, scans their 1-bit codes and rescores the best candidates with their 8-bit
//! rows. A query runs in three steps:
//! 1. Route: score every cluster center with its 4-bit code and keep the `nprobe` nearest.
//! 2. Scan: estimate the distance to every vector in those clusters, 32 at a time, and keep the best candidates.
//! 3. Rerank: rescore the candidates with their 8-bit rows and return the top `k`.
//!
//! Candidates and results are ordered by score and then internal id, so they never depend on the order of the scan.

pub(crate) mod candidates;
pub(crate) mod inmem;
pub(crate) mod rerank;
pub(crate) mod route;
pub(crate) mod scan;

use std::io::{Error, ErrorKind};

use crate::kernels::Tier;
use crate::quantization::padded_dim;
use crate::quantization::rabitq::{BLOCK, QueryLut, block_bytes};
use crate::quantization::rotation::Rotation;
use crate::quantization::sq4::QueryS8;
use crate::{Hit, LeafId, Limits, Metric, SearchCtx, SearchParams};
use candidates::{Candidate, Candidates};
use rerank::rerank;
use route::{Centers, route};
use scan::scan_list;

/// The largest squared length a query's or a vector's offset from the mean may have, a quarter of the largest f32, so
/// adding two of them can't overflow.
pub(crate) const LIMIT: f64 = f32::MAX as f64 / 4.0;

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

/// What a search reads from an index. The bytes are the same whether the index is in RAM or in a mapped file.
pub(crate) struct View<'a> {
    pub(crate) dim: usize,
    pub(crate) metric: Metric,
    pub(crate) mean: &'a [f32],
    pub(crate) rotation: &'a Rotation,
    pub(crate) centers: Centers<'a>,
    pub(crate) lists: &'a [List],
    /// Every cluster's blocks.
    pub(crate) blocks: &'a [u8],
    pub(crate) rows: RowPages<'a>,
    pub(crate) tier: Tier,
}

/// Searches `view` for the `params.k` nearest vectors. The hits live in `ctx`.
pub(crate) fn run<'a>(
    view: &View,
    query: &[f32],
    params: &SearchParams,
    ctx: &'a mut SearchCtx,
) -> Result<&'a [Hit], Error> {
    check_search(query, view.dim, params, &ctx.limits)?;

    let SearchCtx { limits, scratch, hits } = ctx;
    let s = scratch_for(scratch, view.dim, view.lists.len(), limits);
    prepare_query(view, query, s)?;
    route(
        view.tier,
        &view.centers,
        &s.q8,
        s.lut.k.t2,
        params.nprobe,
        &mut s.dots,
        &mut s.routes,
    );
    s.candidates.reset(params.rerank.max(params.k));
    let size = block_bytes(padded_dim(view.dim));

    for r in &s.routes {
        let list = view.lists[r.id as usize];
        let blocks = &view.blocks[list.offset..list.offset + list.count.div_ceil(BLOCK) * size];
        scan_list(
            view.tier,
            blocks,
            list.count,
            list.first_id,
            &s.lut,
            &mut s.candidates,
            &mut s.survivors,
        );
    }

    let row = |id: u32| view.rows.row(id);
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

/// Writes the `nprobe` clusters nearest the query into `out`, nearest first.
pub(crate) fn run_route(
    view: &View,
    query: &[f32],
    nprobe: usize,
    ctx: &mut SearchCtx,
    out: &mut Vec<LeafId>,
) -> Result<(), Error> {
    let params = SearchParams {
        k: 0,
        nprobe,
        rerank: 0,
    };
    check_search(query, view.dim, &params, &ctx.limits)?;

    let SearchCtx { limits, scratch, .. } = ctx;
    let s = scratch_for(scratch, view.dim, view.lists.len(), limits);
    prepare_query(view, query, s)?;
    route(
        view.tier,
        &view.centers,
        &s.q8,
        s.lut.k.t2,
        nprobe,
        &mut s.dots,
        &mut s.routes,
    );
    out.clear();
    out.extend(s.routes.iter().map(|r| r.id));

    Ok(())
}

/// Normalizes the query for cosine, then writes its offset q − μ, that offset rotated and its two quantized forms into
/// `s`.
fn prepare_query(view: &View, query: &[f32], s: &mut Scratch) -> Result<(), Error> {
    s.query.copy_from_slice(query);

    if view.metric == Metric::Cosine && !normalize(&mut s.query) {
        return Err(invalid("a zero query has no direction for cosine"));
    }

    let mut t2 = 0f64;

    for ((o, &q), &m) in s.offset.iter_mut().zip(&s.query).zip(view.mean) {
        *o = q - m;
        t2 += f64::from(*o).powi(2);
    }

    // An estimate adds T² to a vector's f_add, and a route score adds it to a center's ‖z‖², so T² needs room to spare.
    if t2 > LIMIT {
        return Err(invalid("the query is too far from the index's mean"));
    }

    view.rotation.apply(&s.offset, &mut s.rotated, &mut s.tmp);
    s.lut.prepare(&s.rotated, t2.sqrt());
    s.q8.prepare(&s.rotated);
    Ok(())
}

/// The buffers of one search, sized for one index's dimension and cluster count. A `SearchCtx` keeps one per shape and
/// reuses it, so a search allocates nothing after the first with each shape.
pub(crate) struct Scratch {
    dim: usize,
    nlist: usize,
    /// The query, normalized for cosine.
    pub(crate) query: Vec<f32>,
    /// The query's offset q − μ from the index's mean μ.
    pub(crate) offset: Vec<f32>,
    /// That offset rotated by the index's rotation P, i.e. P(q − μ).
    pub(crate) rotated: Vec<f32>,
    pub(crate) tmp: Vec<f32>,
    pub(crate) lut: QueryLut,
    pub(crate) q8: QueryS8,
    /// One dot product per cluster.
    pub(crate) dots: Vec<i32>,
    /// The clusters to read, nearest first.
    pub(crate) routes: Vec<Candidate>,
    pub(crate) candidates: Candidates,
    pub(crate) survivors: [Candidate; BLOCK],
    pub(crate) rescored: Vec<Candidate>,
}

impl Scratch {
    fn new(dim: usize, nlist: usize, limits: &Limits) -> Scratch {
        let keep = limits.max_rerank.max(limits.max_k);
        Scratch {
            dim,
            nlist,
            query: vec![0.0; dim],
            offset: vec![0.0; dim],
            rotated: vec![0.0; dim],
            tmp: vec![0.0; dim],
            lut: QueryLut::new(dim),
            q8: QueryS8::new(dim),
            dots: vec![0; nlist],
            routes: Vec::with_capacity(nlist),
            candidates: Candidates::new(keep),
            survivors: [Candidate::NONE; BLOCK],
            rescored: Vec::with_capacity(keep),
        }
    }
}

/// The scratch in `slots` for an index with this dimension and cluster count, made first if there isn't one yet.
pub(crate) fn scratch_for<'a>(
    slots: &'a mut Vec<Scratch>,
    dim: usize,
    nlist: usize,
    limits: &Limits,
) -> &'a mut Scratch {
    let i = match slots.iter().position(|s| s.dim == dim && s.nlist == nlist) {
        Some(i) => i,
        None => {
            slots.push(Scratch::new(dim, nlist, limits));
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
