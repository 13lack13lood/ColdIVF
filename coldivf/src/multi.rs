//! Searches over several index files at once. One search shares its batches of reads across them. It sends one hint
//! for every file's cluster tables, reads the nearest clusters of any file in one wave, then reads the rows of the best
//! candidates overall in one more.

use std::fs::File;
use std::io::Error;
use std::path::Path;

use crate::format::FIRST_READ;
use crate::io;
use crate::search::candidates::{Candidate, order};
use crate::search::rerank::rerank_live;
use crate::search::route::{reach_of, route};
use crate::search::{Scratch, invalid, prepare_query, scan_lists};
use crate::update::Seen;
use crate::{Hit, Index, LeafId, Limits, Metric, SearchCtx, SearchParams};

/// One collection's part of a search over several, kept in the search context.
pub(crate) struct Part {
    s: Scratch,
    /// The collection's clusters among the nearest overall.
    leaves: Vec<LeafId>,
    /// The collection's candidates among the best overall.
    chosen: Vec<Candidate>,
    hits: Vec<Hit>,
    /// How many of `hits` the merge has taken.
    taken: usize,
    /// Where the collection's reads sit in the context.
    slot: usize,
    /// Whether the collection was warm when the search began, which decides how it reads its clusters.
    warm: bool,
    /// What the search saw of the collection's inserts and deletes when it scanned them.
    seen: Seen,
}

impl Part {
    fn new(dim: usize, nlist: usize, limits: &Limits) -> Part {
        Part {
            s: Scratch::new(dim, nlist, limits),
            leaves: Vec::with_capacity(limits.max_nprobe),
            chosen: Vec::with_capacity(limits.max_rerank.max(limits.max_k)),
            hits: Vec::with_capacity(limits.max_k),
            taken: 0,
            slot: 0,
            warm: false,
            seen: Seen::default(),
        }
    }
}

/// Opens every file of `paths`. It first hints every file's first page, so the header reads of all of them overlap.
pub(crate) fn open(paths: &[&Path]) -> Result<Vec<Index>, Error> {
    if paths.is_empty() {
        return Err(invalid("a set of collections needs at least one file"));
    }

    for path in paths {
        io::will_need(&File::open(path)?, 0, FIRST_READ);
    }

    let indexes = paths.iter().map(|p| Index::open(p)).collect::<Result<Vec<_>, _>>()?;
    let shape = |i: &Index| (i.inner.dim(), i.inner.metric());

    if indexes.iter().any(|i| shape(i) != shape(&indexes[0])) {
        return Err(invalid("the collections must share a dimension and a metric"));
    }

    Ok(indexes)
}

/// Searches `indexes` together for the `params.k` nearest vectors overall, and writes them into `out` as (collection,
/// hit), nearest first. Ties go to the lower collection.
pub(crate) fn search(
    indexes: &[Index],
    query: &[f32],
    params: &SearchParams,
    ctx: &mut SearchCtx,
    out: &mut Vec<(usize, Hit)>,
) -> Result<(), Error> {
    indexes[0].inner.check(query, params, &ctx.limits)?;

    // One hint for every collection's cluster tables, which then load one by one. Every collection's reads exist
    // before any slot is kept, since making reads drops those of closed indexes and moves the rest.
    indexes.iter().for_each(|i| i.inner.prepare());

    for index in indexes {
        index.inner.reads(index.inner.tables()?, ctx);
    }

    let keep = params.rerank.max(params.k);
    let limits = ctx.limits;
    ctx.picks.clear();
    ctx.picks
        .reserve(indexes.len() * limits.max_nprobe.max(limits.max_rerank).max(limits.max_k));

    for (f, index) in indexes.iter().enumerate() {
        let tables = index.inner.tables()?;
        let view = index.inner.view(tables);
        let slot = index.inner.reads(tables, ctx);
        // The parts past the collections are spares, so a context that searches other sets keeps a part for each
        // shape it has seen.
        let parts = &mut ctx.parts;
        match (f..parts.len()).find(|&j| parts[j].s.fits(view.dim, view.lists.len())) {
            Some(j) => parts.swap(f, j),
            None => {
                parts.push(Part::new(view.dim, view.lists.len(), &limits));
                let last = parts.len() - 1;
                parts.swap(f, last);
            }
        }
        let part = &mut parts[f];
        (part.slot, part.warm) = (slot, index.inner.is_warm());
        let s = &mut part.s;
        // Each collection measures the query against its own mean and rotation.
        prepare_query(&view, query, s)?;
        route(
            view.tier,
            &view.centers,
            &s.q8,
            s.lut.k.t2,
            params.nprobe,
            &mut s.dots,
            &mut s.routes,
        );
        ctx.picks.extend(s.routes.iter().map(|&r| (r, f)));
    }

    // The nearest clusters overall, within reach of the nearest one. Route scores estimate squared distances, so they
    // compare across collections.
    let picks = &mut ctx.picks;
    let n = params.nprobe.min(picks.len());

    if n < picks.len() {
        picks.select_nth_unstable_by(n, by_collection);
    }

    picks.truncate(n);
    picks.sort_unstable_by(by_collection);
    picks.truncate(reach_of(picks.iter().map(|p| p.0.score), params.reach));
    ctx.parts[..indexes.len()].iter_mut().for_each(|p| p.leaves.clear());

    for &(c, f) in picks.iter() {
        ctx.parts[f].leaves.push(c.id);
    }

    // One wave of cluster reads across every cold collection, then each scans its clusters as they arrive. A warm one
    // scans in place.
    for (f, index) in indexes.iter().enumerate() {
        if !ctx.parts[f].warm {
            let SearchCtx { parts, reads, .. } = ctx;
            index.inner.fetch_into(&parts[f].leaves, &mut reads[parts[f].slot])?;
        }
    }

    for (f, index) in indexes.iter().enumerate() {
        let tables = index.inner.tables()?;
        let view = index.inner.view(tables);
        let SearchCtx { parts, reads, .. } = ctx;
        let part = &mut parts[f];
        part.s.candidates.reset(keep);
        if part.warm {
            let Scratch {
                lut,
                candidates,
                survivors,
                ..
            } = &mut part.s;
            let leaves = part.leaves.iter().copied();
            scan_lists(&view, index.inner.in_place().blocks, leaves, lut, candidates, survivors);
            reads[part.slot].forget();
        } else {
            reads[part.slot].scan(index.inner.cold_file(), &view, &mut part.s)?;
        }
        part.seen = index
            .inner
            .scan_delta(&index.inner.changes().read(), &view, &mut part.s);
    }

    // The best candidates overall, then one wave for their rows.
    ctx.picks.clear();

    for (f, part) in ctx.parts[..indexes.len()].iter().enumerate() {
        ctx.picks.extend(part.s.candidates.as_slice().iter().map(|&c| (c, f)));
    }

    let best = &mut ctx.picks;

    if keep < best.len() {
        best.select_nth_unstable_by(keep, by_collection);
        best.truncate(keep);
    }

    ctx.parts[..indexes.len()].iter_mut().for_each(|p| p.chosen.clear());

    for &(c, f) in ctx.picks.iter() {
        ctx.parts[f].chosen.push(c);
    }

    for (f, index) in indexes.iter().enumerate() {
        let SearchCtx { parts, reads, .. } = ctx;
        let part = &parts[f];
        reads[part.slot].start_rows(index.inner.cold_file(), &part.chosen)?;
    }

    // Each collection rescores its share, and the hits merge by distance.
    for (f, index) in indexes.iter().enumerate() {
        let view = index.inner.view(index.inner.tables()?);
        let SearchCtx { parts, reads, .. } = ctx;
        let slot = parts[f].slot;
        let (part, r) = (&mut parts[f], &mut reads[slot]);
        r.finish_rows(index.inner.cold_file())?;
        // Each collection's lock is taken alone and only for its rescore, which reads nothing from the disk.
        let (delta, seen) = (index.inner.changes().read(), part.seen);
        let row = |id: u32| index.inner.row_of(&delta, id, |id| r.row(id));
        let live = |id: u32, row: &[u8]| index.inner.is_live(&delta, id, row, seen);
        let s = &mut part.s;
        let (offset, rescored) = (&s.offset, &mut s.rescored);
        rerank_live(
            &part.chosen,
            row,
            live,
            offset,
            params.k,
            view.metric,
            rescored,
            &mut part.hits,
        )?;
    }

    merge(
        &mut ctx.parts[..indexes.len()],
        indexes[0].inner.metric(),
        params.k,
        out,
    );
    Ok(())
}

/// Orders routes and candidates of several collections by score, then internal id, then collection.
fn by_collection(a: &(Candidate, usize), b: &(Candidate, usize)) -> std::cmp::Ordering {
    order(&a.0, &b.0).then(a.1.cmp(&b.1))
}

/// Merges the collections' hits, each list nearest first, into the `k` nearest overall. Ties go to the lower
/// collection, so a collection's hits keep their order.
fn merge(parts: &mut [Part], metric: Metric, k: usize, out: &mut Vec<(usize, Hit)>) {
    let nearer = |a: &Hit, b: &Hit| match metric {
        Metric::L2 => a.score < b.score,
        Metric::Cosine => a.score > b.score,
    };
    out.clear();
    parts.iter_mut().for_each(|p| p.taken = 0);

    while out.len() < k {
        let heads = parts
            .iter()
            .enumerate()
            .filter_map(|(f, p)| p.hits.get(p.taken).map(|&h| (f, h)));
        let Some((f, h)) = heads.reduce(|best, next| if nearer(&next.1, &best.1) { next } else { best }) else {
            break;
        };
        parts[f].taken += 1;
        out.push((f, h));
    }
}

#[cfg(test)]
#[path = "../tests/unit/multi.rs"]
mod tests;
