//! Searches over several index files at once. One search shares its batches of reads across them. It sends one hint
//! for every file's cluster tables, reads the nearest clusters of any file in one wave, then reads the rows of the best
//! candidates overall in one more.

use std::io::Error;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use crate::index::{Inner, Opening, Shared};
use crate::search::candidates::{Candidate, order};
use crate::search::rerank::rerank_live;
use crate::search::route::{reach_of, route_chunked};
use crate::search::{Scratch, invalid, prepare_query, scan_lists};
use crate::trace::{Clock, Trace};
use crate::update::{Seen, real_path};
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
    fn new(dim: usize, limits: &Limits) -> Part {
        Part {
            s: Scratch::new(dim, 0, limits),
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

/// Opens every file of `paths`. It opens all of them and starts every header read before it waits for any.
pub(crate) fn open(paths: &[&Path]) -> Result<Vec<Index>, Error> {
    if paths.is_empty() {
        return Err(invalid("a set of collections needs at least one file"));
    }

    let opening = paths.iter().map(|p| Opening::start(p)).collect::<Result<Vec<_>, _>>()?;
    let mut indexes: Vec<Index> = Vec::with_capacity(paths.len());

    for (o, p) in opening.into_iter().zip(paths) {
        let (real, inner) = (real_path(p), o.finish(p)?);
        // Each collection of a set has reads of its own in a search context, which are keyed by the shared cell, so a
        // path that the set holds twice gets a cell of its own the second time.
        let shared = Shared::adopt(&real, inner);
        let twice = indexes.iter().any(|i| Arc::ptr_eq(&i.shared, &shared));
        let shared = match twice {
            true => Shared::alone(&real, Inner::open(p)?, None),
            false => shared,
        };
        indexes.push(Index { shared });
    }
    let shape = |i: &Index| (i.inner().dim(), i.inner().metric());

    if indexes.iter().any(|i| shape(i) != shape(&indexes[0])) {
        return Err(invalid("the collections must share a dimension and a metric"));
    }

    Ok(indexes)
}

/// Searches `indexes` together for the `params.k` nearest vectors overall, and writes them into `out` as (collection,
/// hit), nearest first. Ties go to the lower collection. It loads each collection's current generation once, and
/// searches those generations whatever commits meanwhile.
pub(crate) fn search(
    indexes: &[Index],
    query: &[f32],
    params: &SearchParams,
    ctx: &mut SearchCtx,
    out: &mut Vec<(usize, Hit)>,
) -> Result<(), Error> {
    // The context keeps the list's room, so the loads allocate nothing once a search of as many collections ran.
    let mut loaded = std::mem::take(&mut ctx.loaded);
    loaded.extend(indexes.iter().map(|i| i.shared.load_full()));
    let result = search_loaded(&loaded, query, params, ctx, out);
    loaded.clear();
    ctx.loaded = loaded;
    result
}

/// The same, for the loaded generations `indexes`.
fn search_loaded(
    indexes: &[Arc<Inner>],
    query: &[f32],
    params: &SearchParams,
    ctx: &mut SearchCtx,
    out: &mut Vec<(usize, Hit)>,
) -> Result<(), Error> {
    indexes[0].check(query, params, &ctx.limits)?;

    ctx.trace = Trace::default();
    let mut clock = Clock::start();
    // One hint for every collection's cluster tables, which then load one by one.
    indexes.iter().for_each(|i| i.prepare());

    for index in indexes {
        index.tables()?;
    }

    ctx.trace.tables = clock.lap();

    // Every collection's reads exist before any slot is kept, since making reads drops those of closed indexes and
    // moves the rest.
    for index in indexes {
        index.reads(index.tables()?, ctx);
    }

    let keep = params.rerank.max(params.k);
    let limits = ctx.limits;
    ctx.picks.clear();
    ctx.picks
        .reserve(indexes.len() * limits.max_nprobe.max(limits.max_rerank).max(limits.max_k));

    for (f, index) in indexes.iter().enumerate() {
        let tables = index.tables()?;
        let (dim, nlist) = (index.dim(), tables.lists.len());
        let slot = index.reads(tables, ctx);
        // The parts past the collections are spares, so a context that searches other sets keeps as many parts of
        // each dimension as the most collections of it that one search had. Each part fits itself to the collection's
        // cluster count, which every maintenance run changes.
        let parts = &mut ctx.parts;
        match (f..parts.len()).find(|&j| parts[j].s.dim() == dim) {
            Some(j) => parts.swap(f, j),
            None => {
                parts.push(Part::new(dim, &limits));
                let last = parts.len() - 1;
                parts.swap(f, last);
            }
        }
        parts[f].s.fit(nlist);
        (parts[f].slot, parts[f].warm) = (slot, index.is_warm());
    }

    ctx.trace.buffers = clock.lap();

    for (f, index) in indexes.iter().enumerate() {
        let view = index.view(index.tables()?);
        let s = &mut ctx.parts[f].s;
        // Each collection measures the query against its own mean and rotation.
        prepare_query(&view, query, s)?;
        route_chunked(
            view.tier,
            &view.centers,
            &s.q8,
            s.lut.mean.t2,
            params.nprobe,
            &mut s.chunk,
            &mut s.best,
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

    ctx.trace.route = clock.lap();

    // One wave of cluster reads across every cold collection, then each scans its clusters as they arrive. A warm one
    // scans in place.
    for (f, index) in indexes.iter().enumerate() {
        if !ctx.parts[f].warm {
            let SearchCtx {
                parts, reads, trace, ..
            } = ctx;
            let r = &mut reads[parts[f].slot];
            index.fetch_into(&parts[f].leaves, r)?;
            let (bytes, ranges) = r.cluster_traffic();
            trace.cluster_bytes += bytes;
            trace.cluster_ranges += ranges;
        }
    }

    ctx.trace.submit = clock.lap();

    for (f, index) in indexes.iter().enumerate() {
        let tables = index.tables()?;
        let view = index.view(tables);
        let SearchCtx {
            parts, reads, trace, ..
        } = ctx;
        let part = &mut parts[f];
        part.s.candidates.reset(keep);
        trace.scanned += part.leaves.iter().map(|&l| view.lists[l as usize].count).sum::<usize>();
        if part.warm {
            let Scratch {
                lut,
                q32,
                candidates,
                survivors,
                ..
            } = &mut part.s;
            let (blocks, leaves) = (index.in_place().blocks, part.leaves.iter().copied());
            scan_lists(&view, blocks, leaves, lut, q32, candidates, survivors);
            reads[part.slot].forget();
        } else {
            let r = &mut reads[part.slot];
            r.scan(index.cold_file(), &view, &mut part.s)?;
            trace.wave += r.waited();
        }
        part.seen = index.scan_delta(&index.changes().read(), &view, &mut part.s);
    }

    ctx.trace.scan = clock.lap().saturating_sub(ctx.trace.wave);

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
        let SearchCtx {
            parts, reads, trace, ..
        } = ctx;
        let r = &mut reads[parts[f].slot];
        r.start_rows(index.cold_file(), &parts[f].chosen, parts[f].warm)?;
        let (bytes, ranges) = r.row_traffic();
        trace.row_bytes += bytes;
        trace.row_ranges += ranges;
    }

    let (mut rows, mut rerank) = (clock.lap(), Duration::ZERO);

    // Each collection rescores its share, and the hits merge by distance.
    for (f, index) in indexes.iter().enumerate() {
        let view = index.view(index.tables()?);
        let SearchCtx { parts, reads, .. } = ctx;
        let slot = parts[f].slot;
        let (part, r) = (&mut parts[f], &mut reads[slot]);
        r.finish_rows(index.cold_file())?;
        rows += clock.lap();
        // Each collection's lock is taken alone and only for its rescore, which reads nothing from the disk.
        let (delta, seen) = (index.changes().read(), part.seen);
        let row = |id: u32| index.row_of(&delta, id, |id| r.row(id));
        let live = |id: u32, row: &[u8]| index.is_live(&delta, id, row, seen);
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
        rerank += clock.lap();
    }

    merge(&mut ctx.parts[..indexes.len()], indexes[0].metric(), params.k, out);
    (ctx.trace.rows, ctx.trace.rerank) = (rows, rerank + clock.lap());
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
