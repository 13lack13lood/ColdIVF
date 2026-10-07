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

use crate::quantization::rabitq::{BLOCK, QueryLut};
use crate::quantization::sq4::QueryS8;
use crate::{Limits, SearchParams};
use candidates::{Candidate, Candidates};

/// The buffers of one search, sized for one index's dimension and cluster count. A `SearchCtx` keeps one and reuses
/// it, so a search allocates nothing.
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

/// The scratch in `slot`, made first if the slot is empty or sized for another index's shape.
pub(crate) fn scratch_for<'a>(
    slot: &'a mut Option<Scratch>,
    dim: usize,
    nlist: usize,
    limits: &Limits,
) -> &'a mut Scratch {
    if !matches!(slot, Some(s) if s.dim == dim && s.nlist == nlist) {
        *slot = Some(Scratch::new(dim, nlist, limits));
    }

    slot.as_mut().expect("the slot was just filled")
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
