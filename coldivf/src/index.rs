//! An index file opened for search. Opening reads the first 16 KiB, maps the whole file without loading it and opens a
//! second handle for cold reads, which every later read of the file goes through. It also reads the file's change log,
//! if it has one. The cluster tables load on the first search.

use std::fs::File;
use std::io::{Error, ErrorKind};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock, Weak};

use crate::build::Slice;
use crate::format::{self, FIRST_READ, Layout, Part, Tables};
use crate::io::map::Map;
use crate::io::{self, Cold, Pressure, Span, Warmth};
use crate::kernels::Tier;
use crate::quantization::rabitq::{BLOCK, Encoder, Factors, code_bytes};
use crate::quantization::rotation::Rotation;
use crate::quantization::{padded_dim, sq8};
use crate::search::cold::{Reads, reads_for};
use crate::search::inmem::InMemory;
use crate::search::rerank::rerank_live;
use crate::search::route::Centers;
use crate::search::scan::scan_list;
use crate::search::{
    InPlace, LIMIT, RowPages, Scratch, View, check_search, invalid, normalize, prepare_query, run_route, scan_in_place,
    scratch_for,
};
use crate::update::{self, Changes, Delta, Seen};
use crate::{BuildParams, Clusters, Hit, LeafId, LeafOrder, Limits, Metric, SearchCtx, SearchParams};

/// The most bytes that `verify` and a warm-up read at once.
const PIECE: usize = 1 << 22;

pub(crate) struct Inner {
    cold: Cold,
    /// The inserts and deletes since the file was built, shared with every other open index of the file.
    changes: Arc<Changes>,
    map: Map,
    len: usize,
    layout: Layout,
    mean: Vec<f32>,
    rotation: Rotation,
    tables: OnceLock<Tables>,
    tier: Tier,
    warmth: Arc<Warmth>,
    /// The `warm_up` calls that the running warm-up hasn't seen yet, or 0 when none runs.
    warming: AtomicUsize,
    /// Clears the warm bit when memory runs short, from the first time the bit may be set.
    pressure: OnceLock<Pressure>,
}

impl Inner {
    /// Opens `path`. A damaged file fails with `InvalidData`.
    pub(crate) fn open(path: &Path) -> Result<Inner, Error> {
        let file = File::open(path)?;
        let len = usize::try_from(file.metadata()?.len()).map_err(|_| damaged("the file is too large"))?;
        let mut first = vec![0u8; FIRST_READ.min(len)];
        io::read_at(&file, &mut first, 0)?;
        let layout = format::read_header(&first, len)?;

        // Above 2806 dimensions the rotation runs past the first read.
        let span = layout.span(Part::Rotation);
        let rotation_bytes = match first.get(span.range()) {
            Some(bytes) => bytes.to_vec(),
            None => read_span(&file, span)?,
        };
        let (rotation, mean) = format::read_rotation(&rotation_bytes, &layout)?;
        let map = Map::new(&file)?;

        if map.bytes().len() != len {
            return Err(damaged("the file changed while it was being opened"));
        }

        Ok(Inner {
            cold: io::open_cold(path, &first)?,
            changes: update::changes_for(path, &layout)?,
            map,
            len,
            layout,
            mean,
            rotation,
            tables: OnceLock::new(),
            tier: Tier::best(),
            warmth: Arc::default(),
            warming: AtomicUsize::new(0),
            pressure: OnceLock::new(),
        })
    }

    /// Asks the operating system (OS) to start reading the cluster tables, so the read overlaps the embedding model.
    /// Once the tables are loaded, it does nothing.
    pub(crate) fn prepare(&self) {
        if self.tables.get().is_none() {
            let span = format::tables_span(&self.layout);
            io::will_need(&self.cold.file, span.offset, span.len);
        }
    }

    /// Checks a search's arguments against this index and `limits`.
    pub(crate) fn check(&self, query: &[f32], params: &SearchParams, limits: &Limits) -> Result<(), Error> {
        check_search(query, self.layout.dim, params, limits)
    }

    /// Searches through the memory map, which leaves the reads to page faults. Tests check the cold search against it.
    pub(crate) fn search_mapped<'a>(
        &self,
        query: &[f32],
        params: &SearchParams,
        ctx: &'a mut SearchCtx,
    ) -> Result<&'a [Hit], Error> {
        self.check(query, params, &ctx.limits)?;

        let (tables, data) = (self.tables()?, self.in_place());
        let view = self.view(tables);
        let SearchCtx {
            limits, scratch, hits, ..
        } = ctx;
        let s = scratch_for(scratch, view.dim, view.lists.len(), limits);
        prepare_query(&view, query, s)?;
        scan_in_place(&view, data.blocks, params, s);
        let delta = self.changes.read();
        let seen = self.scan_delta(&delta, &view, s);
        let row = |id: u32| self.row_of(&delta, id, |id| data.rows.row(id));
        let live = |id: u32, row: &[u8]| self.is_live(&delta, id, row, seen);
        let (offset, rescored) = (&s.offset, &mut s.rescored);
        rerank_live(
            s.candidates.as_slice(),
            row,
            live,
            offset,
            params.k,
            view.metric,
            rescored,
            hits,
        )?;

        Ok(&hits[..])
    }

    pub(crate) fn changes(&self) -> &Changes {
        &self.changes
    }

    pub(crate) fn layout(&self) -> &Layout {
        &self.layout
    }

    /// Scans the inserted vectors for the query in `s`, as one more cluster, with internal ids from the file's count.
    /// Returns what it saw, so the rescore can tell which rows were visible then.
    pub(crate) fn scan_delta(&self, delta: &Delta, view: &View, s: &mut Scratch) -> Seen {
        if delta.len() > 0 {
            let first = self.layout.count as u32;
            let (lut, candidates, survivors) = (&s.lut, &mut s.candidates, &mut s.survivors);
            scan_list(
                view.tier,
                delta.blocks(),
                delta.len(),
                first,
                lut,
                candidates,
                survivors,
            );
        }

        delta.seen()
    }

    /// The row of internal id `id`, from the file through `in_file` or from the inserted vectors.
    pub(crate) fn row_of<'a>(&self, delta: &'a Delta, id: u32, in_file: impl Fn(u32) -> &'a [u8]) -> &'a [u8] {
        match (id as usize).checked_sub(self.layout.count) {
            Some(i) => delta.row(i),
            None => in_file(id),
        }
    }

    /// Whether the candidate with internal id `id` and row `row` was visible when `seen` was taken.
    pub(crate) fn is_live(&self, delta: &Delta, id: u32, row: &[u8], seen: Seen) -> bool {
        let seq = (id as usize).checked_sub(self.layout.count).map_or(0, |i| delta.seq(i));
        delta.hides_nothing() || delta.is_live(sq8::id(row, self.layout.dim), seq, seen)
    }

    /// Writes the `nprobe` clusters nearest the query into `out`, nearest first, keeping only those within `reach`.
    pub(crate) fn route(
        &self,
        query: &[f32],
        nprobe: usize,
        reach: Option<f32>,
        ctx: &mut SearchCtx,
        out: &mut Vec<LeafId>,
    ) -> Result<(), Error> {
        run_route(&self.view(self.tables()?), query, nprobe, reach, ctx, out)
    }

    /// Starts reading `leaves` in one wave, through `ctx`'s reads of this index, which the first fetch makes. Returns
    /// the reads' slot in `ctx`.
    pub(crate) fn fetch(self: &Arc<Inner>, leaves: &[LeafId], ctx: &mut SearchCtx) -> Result<usize, Error> {
        let tables = self.tables()?;

        if leaves.len() > ctx.limits.max_nprobe {
            return Err(invalid("more leaves than the search context's max_nprobe"));
        }

        if leaves.iter().any(|&l| l as usize >= tables.lists.len()) {
            return Err(invalid("a leaf is past the index's clusters"));
        }

        let slot = self.reads(tables, ctx);
        ctx.reads[slot].fetch(&self.cold.file, &tables.lists, leaves)?;
        Ok(slot)
    }

    /// Starts reading `leaves`, each under the cluster count, in one wave through `reads`.
    pub(crate) fn fetch_into(&self, leaves: &[LeafId], reads: &mut Reads) -> Result<(), Error> {
        reads.fetch(&self.cold.file, &self.tables()?.lists, leaves)
    }

    pub(crate) fn cold_file(&self) -> &File {
        &self.cold.file
    }

    pub(crate) fn dim(&self) -> usize {
        self.layout.dim
    }

    pub(crate) fn metric(&self) -> Metric {
        self.layout.metric
    }

    /// The slot in `ctx` of its reads of this index, made the first time.
    pub(crate) fn reads(self: &Arc<Inner>, tables: &Tables, ctx: &mut SearchCtx) -> usize {
        let SearchCtx { limits, reads, .. } = ctx;
        let make = || Reads::new(self, (self.cold.wave)(), &self.layout, self.len, &tables.lists, limits);
        reads_for(reads, self, make)
    }

    /// Searches a warm index. It scans the clusters in place through the map, then reads the rows to rescore in a
    /// wave, which costs little for rows already in the cache. The hits live in `ctx`.
    pub(crate) fn search_warm<'a>(
        self: &Arc<Inner>,
        query: &[f32],
        params: &SearchParams,
        ctx: &'a mut SearchCtx,
    ) -> Result<&'a [Hit], Error> {
        self.check(query, params, &ctx.limits)?;

        let tables = self.tables()?;
        let slot = self.reads(tables, ctx);
        let (view, data) = (self.view(tables), self.in_place());
        let SearchCtx {
            limits,
            scratch,
            reads,
            hits,
            ..
        } = ctx;
        let s = scratch_for(scratch, view.dim, view.lists.len(), limits);
        prepare_query(&view, query, s)?;
        scan_in_place(&view, data.blocks, params, s);
        let seen = self.scan_delta(&self.changes.read(), &view, s);
        let r = &mut reads[slot];
        r.forget();
        r.read_rows(&self.cold.file, s.candidates.as_slice())?;
        // The row wave runs without the lock, so a writer never waits for a read.
        let delta = self.changes.read();
        let row = |id: u32| self.row_of(&delta, id, |id| r.row(id));
        let live = |id: u32, row: &[u8]| self.is_live(&delta, id, row, seen);
        let (offset, rescored) = (&s.offset, &mut s.rescored);
        rerank_live(
            s.candidates.as_slice(),
            row,
            live,
            offset,
            params.k,
            view.metric,
            rescored,
            hits,
        )?;

        Ok(&hits[..])
    }

    pub(crate) fn is_warm(&self) -> bool {
        self.warmth.is_warm()
    }

    pub(crate) fn set_warm(&self, warm: bool) {
        if warm {
            self.watch_pressure();
        }

        self.warmth.set(warm);
    }

    /// Starts the watch that clears the warm bit when memory runs short, unless it has started.
    fn watch_pressure(&self) {
        self.pressure.get_or_init(|| Pressure::watch(Arc::clone(&self.warmth)));
    }

    /// Starts reading every cluster's blocks into the cache on a background thread, unless the index is warm. A call
    /// while a warm-up runs makes it run again once it ends, if the index isn't warm by then. The thread stops once
    /// the index closes.
    pub(crate) fn warm_up(self: &Arc<Inner>) -> Result<(), Error> {
        self.watch_pressure();

        if self.is_warm() || self.warming.fetch_add(1, Ordering::AcqRel) > 0 {
            return Ok(());
        }

        let index = Arc::downgrade(self);
        let started = std::thread::Builder::new().name("coldivf-warm".into()).spawn(move || {
            io::background();
            keep_warming(&index);
        });

        if let Err(e) = started {
            self.warming.store(0, Ordering::Release);
            return Err(e);
        }

        Ok(())
    }

    /// Encodes `vector` with the id `id` for an insert. Returns the cluster it is nearest to, its 1-bit code and
    /// factors, and its 8-bit row. A vector that doesn't fit the index fails with `InvalidInput`.
    pub(crate) fn encode_insert(
        &self,
        id: u64,
        vector: &[f32],
        ctx: &mut SearchCtx,
    ) -> Result<(u32, Vec<u8>, Factors, Vec<u8>), Error> {
        let (dim, count) = (self.layout.dim, self.layout.count);

        if vector.len() != dim {
            return Err(invalid("the vector's length doesn't match the index's dimension"));
        }

        if !vector.iter().all(|x| x.is_finite()) {
            return Err(invalid("a vector has a value that isn't finite"));
        }

        if count + self.changes.read().len() >= u32::MAX as usize - BLOCK {
            return Err(invalid("the index is full, so compact it first"));
        }

        let mut o = vector.to_vec();

        if self.layout.metric == Metric::Cosine && !normalize(&mut o) {
            return Err(invalid("a zero vector has no direction for cosine"));
        }

        let offset2: f64 = o
            .iter()
            .zip(&self.mean)
            .map(|(&x, &m)| (f64::from(x) - f64::from(m)).powi(2))
            .sum();

        if offset2 > LIMIT {
            return Err(invalid("a vector is too far from the mean to encode"));
        }

        let mut leaf = Vec::with_capacity(1);
        self.route(&o, 1, None, ctx, &mut leaf)?;
        let d_pad = padded_dim(dim);
        let mut code = vec![0u8; code_bytes(d_pad)];
        let factors = Encoder::new(&self.rotation, &self.mean).encode(&o, &mut code);
        let mut row = vec![0u8; sq8::row_bytes(dim)];
        sq8::write_row(&o, &self.mean, id, &mut row);
        let fits = |x: f32| f64::from(x) <= LIMIT;

        if !(fits(factors.f_add)
            && factors.f_rescale.is_finite()
            && factors.f_err.is_finite()
            && fits(sq8::norm2(&row, dim)))
        {
            return Err(invalid("a vector is too far from the mean to encode"));
        }

        Ok((leaf.first().copied().unwrap_or(0), code, factors, row))
    }

    /// This index rebuilt with every visible row of the file and every visible insert, decoded from their 8-bit rows,
    /// in the same clusters. Clusters and groups left without vectors drop out, and the epoch goes up by one.
    pub(crate) fn compacted(&self) -> Result<InMemory, Error> {
        let (tables, data) = (self.tables()?, self.in_place());
        let (dim, count) = (self.layout.dim, self.layout.count);
        let delta = self.changes.read();
        let seen = delta.seen();
        let decode = |row: &[u8], out: &mut Vec<f32>| {
            let scale = sq8::scale(row, dim);
            out.extend((0..dim).map(|i| self.mean[i] + scale * f32::from(sq8::code(row, i))));
        };
        let (mut vectors, mut ids, mut leaves) = (Vec::new(), Vec::new(), Vec::new());

        for (leaf, list) in tables.lists.iter().enumerate() {
            for id in list.first_id..list.first_id + list.count as u32 {
                let row = data.rows.row(id);
                if self.is_live(&delta, id, row, seen) {
                    decode(row, &mut vectors);
                    ids.push(sq8::id(row, dim));
                    leaves.push(leaf as u32);
                }
            }
        }

        for i in 0..delta.len() {
            let row = delta.row(i);
            if self.is_live(&delta, (count + i) as u32, row, seen) {
                decode(row, &mut vectors);
                ids.push(sq8::id(row, dim));
                leaves.push(delta.leaf(i));
            }
        }

        if ids.is_empty() {
            return Err(invalid("every vector was deleted, so there is nothing to compact"));
        }

        let clusters = regroup(dim, &vectors, &leaves, &tables.parent_of);
        let params = BuildParams {
            metric: self.layout.metric,
            vectors_per_list: 1,
            parents: 1,
            order: LeafOrder::Id,
            page_size: self.layout.page_size,
            threads: 0,
            seed: self.layout.seed,
        };
        let rows = Slice { dim, data: &vectors };
        let mut index = InMemory::build_ids(&params, &rows, &clusters, Some(&ids))?;
        index.epoch = self.layout.epoch + 1;
        Ok(index)
    }

    /// Scores the fetched clusters for `query` and starts reading the rows of the best `n`, using the reads in `slot`.
    pub(crate) fn prefetch_rows(&self, query: &[f32], n: usize, ctx: &mut SearchCtx, slot: usize) -> Result<(), Error> {
        let params = SearchParams {
            k: 0,
            nprobe: 0,
            rerank: n,
            reach: None,
        };
        self.check(query, &params, &ctx.limits)?;

        let view = self.view(self.tables()?);
        let SearchCtx {
            limits, scratch, reads, ..
        } = ctx;
        let s = scratch_for(scratch, view.dim, view.lists.len(), limits);
        prepare_query(&view, query, s)?;
        s.candidates.reset(n);
        let r = &mut reads[slot];
        r.scan(&self.cold.file, &view, s)?;
        self.scan_delta(&self.changes.read(), &view, s);
        r.prefetch(&self.cold.file, s.candidates.as_slice())
    }

    /// Ranks the fetched clusters for `query`, with the reads in `slot`, and returns the `params.k` nearest. The hits
    /// live in `ctx`.
    pub(crate) fn search_fetched<'a>(
        &self,
        query: &[f32],
        params: &SearchParams,
        ctx: &'a mut SearchCtx,
        slot: usize,
    ) -> Result<&'a [Hit], Error> {
        self.check(query, params, &ctx.limits)?;

        let view = self.view(self.tables()?);
        let SearchCtx {
            limits,
            scratch,
            reads,
            hits,
            ..
        } = ctx;
        let s = scratch_for(scratch, view.dim, view.lists.len(), limits);
        prepare_query(&view, query, s)?;
        s.candidates.reset(params.rerank.max(params.k));
        let r = &mut reads[slot];
        r.scan(&self.cold.file, &view, s)?;
        let seen = self.scan_delta(&self.changes.read(), &view, s);
        r.read_rows(&self.cold.file, s.candidates.as_slice())?;
        // The row wave runs without the lock, so a writer never waits for a read.
        let delta = self.changes.read();
        let row = |id: u32| self.row_of(&delta, id, |id| r.row(id));
        let live = |id: u32, row: &[u8]| self.is_live(&delta, id, row, seen);
        let (offset, rescored) = (&s.offset, &mut s.rescored);
        rerank_live(
            s.candidates.as_slice(),
            row,
            live,
            offset,
            params.k,
            view.metric,
            rescored,
            hits,
        )?;

        Ok(&hits[..])
    }

    /// Checks every section against its checksum, then the cluster tables. It reads the whole file through the cold
    /// handle a piece at a time, so a failed read gives an error instead of a fault on a mapped page.
    pub(crate) fn verify(&self) -> Result<(), Error> {
        self.verify_in(PIECE)
    }

    /// The same, reading at most `size` bytes at once.
    fn verify_in(&self, size: usize) -> Result<(), Error> {
        let mut buf = vec![0u8; size.min(self.len)];

        for part in format::CHECKED {
            let span = self.layout.span(part);
            let mut crc = crc32fast::Hasher::new();
            for at in (span.offset..span.end()).step_by(size) {
                let piece = &mut buf[..size.min(span.end() - at)];
                io::read_at(&self.cold.file, piece, at)?;
                crc.update(piece);
            }
            format::check_section(&self.layout, part, crc.finalize())?;
        }

        let tables = read_span(&self.cold.file, format::tables_span(&self.layout))?;
        format::read_tables(tables, &self.layout).map(|_| ())
    }

    /// The cluster tables, read and checked the first time. Two threads may both read them, and one copy wins.
    pub(crate) fn tables(&self) -> Result<&Tables, Error> {
        if let Some(tables) = self.tables.get() {
            return Ok(tables);
        }

        let bytes = read_span(&self.cold.file, format::tables_span(&self.layout))?;
        let tables = format::read_tables(bytes, &self.layout)?;

        Ok(self.tables.get_or_init(|| tables))
    }

    /// The page size that the cold reads round out to, eg. 16 KiB with `F_RDADVISE`.
    pub(crate) fn read_page(&self) -> usize {
        (self.cold.wave)().page()
    }

    pub(crate) fn view<'a>(&'a self, tables: &'a Tables) -> View<'a> {
        View {
            dim: self.layout.dim,
            metric: self.layout.metric,
            mean: &self.mean,
            rotation: &self.rotation,
            centers: Centers {
                codes: &tables.codes,
                factors: &tables.factors,
            },
            lists: &tables.lists,
            tier: self.tier,
        }
    }

    pub(crate) fn in_place(&self) -> InPlace<'_> {
        let (bytes, layout) = (self.map.bytes(), &self.layout);
        let row_bytes = sq8::row_bytes(layout.dim);
        InPlace {
            blocks: &bytes[layout.span(Part::Codes).range()],
            rows: RowPages {
                bytes: &bytes[layout.span(Part::Rows).range()],
                row_bytes,
                per_page: layout.page_size / row_bytes,
                page_size: layout.page_size,
            },
        }
    }

    /// The same index with another kernel tier, so tests can check that every tier gives the same results.
    #[cfg(test)]
    pub(crate) fn with_tier(mut self, tier: Tier) -> Inner {
        self.tier = tier;
        self
    }

    /// The same index with another read backend, so tests can check that every backend gives the same results.
    #[cfg(test)]
    pub(crate) fn with_wave(mut self, wave: io::NewWave) -> Inner {
        self.cold.wave = wave;
        self
    }

    /// The whole file, and what its header says, for checks.
    #[cfg(test)]
    pub(crate) fn parts(&self) -> (&[u8], &Layout) {
        (self.map.bytes(), &self.layout)
    }
}

/// Warms `index` until a warm-up ends with no `warm_up` call that it hasn't seen, or the index closes.
fn keep_warming(index: &Weak<Inner>) {
    while let Some(inner) = index.upgrade() {
        let (asked, mark) = (inner.warming.load(Ordering::Acquire), inner.warmth.mark());
        drop(inner);
        // A failed read ends this warm-up and leaves the bit clear.
        let _ = warm_once(index, mark);
        let Some(inner) = index.upgrade() else {
            return;
        };
        // It stops unless a `warm_up` call came in while it ran.
        let stop = inner
            .warming
            .compare_exchange(asked, 0, Ordering::AcqRel, Ordering::Acquire);
        if stop.is_ok() {
            return;
        }
    }
}

/// Reads the codes section of `index` in file order, a piece at a time, through its cold handle in a way that leaves
/// the bytes in the cache, then sets the warm bit. It does nothing for a warm index, and stops before a piece once the
/// index has closed or a clear has come since `mark`. Returns how many bytes it read.
pub(crate) fn warm_once(index: &Weak<Inner>, mark: u64) -> Result<usize, Error> {
    let span = match index.upgrade() {
        Some(inner) if !inner.is_warm() => inner.layout.span(Part::Codes),
        _ => return Ok(0),
    };
    let mut buf = vec![0u8; PIECE.min(span.len)];

    for at in (span.offset..span.end()).step_by(PIECE) {
        let Some(inner) = index.upgrade().filter(|inner| inner.warmth.mark() == mark) else {
            return Ok(at - span.offset);
        };
        io::read_cached(&inner.cold.file, &mut buf[..PIECE.min(span.end() - at)], at)?;
    }

    if let Some(inner) = index.upgrade() {
        inner.warmth.finish(mark);
    }

    Ok(span.len)
}

/// Clusters for `vectors` from their old clusters `leaves`, whose groups are `parent_of`. Each kept cluster's center is
/// the mean of its vectors, and each kept group's center the mean of its clusters' centers. Clusters and groups with
/// nothing in them drop out, and the rest keep their order.
fn regroup(dim: usize, vectors: &[f32], leaves: &[u32], parent_of: &[u16]) -> Clusters {
    let nlist = parent_of.len();
    let (mut sums, mut counts) = (vec![0f64; nlist * dim], vec![0usize; nlist]);

    for (v, &leaf) in vectors.chunks_exact(dim).zip(leaves) {
        counts[leaf as usize] += 1;
        let sum = &mut sums[leaf as usize * dim..(leaf as usize + 1) * dim];
        sum.iter_mut().zip(v).for_each(|(s, &x)| *s += f64::from(x));
    }

    let mut new_leaf = vec![u32::MAX; nlist];
    let mut centroids = Vec::new();
    let mut groups_of_kept = Vec::new();

    for (leaf, &n) in counts.iter().enumerate().filter(|(_, n)| **n > 0) {
        new_leaf[leaf] = (centroids.len() / dim) as u32;
        let sum = &sums[leaf * dim..(leaf + 1) * dim];
        centroids.extend(sum.iter().map(|&s| (s / n as f64) as f32));
        groups_of_kept.push(parent_of[leaf] as usize);
    }

    let ngroups = parent_of.iter().map(|&g| g as usize + 1).max().unwrap_or(1);
    let (mut gsums, mut gcounts) = (vec![0f64; ngroups * dim], vec![0usize; ngroups]);

    for (c, &g) in centroids.chunks_exact(dim).zip(&groups_of_kept) {
        gcounts[g] += 1;
        let sum = &mut gsums[g * dim..(g + 1) * dim];
        sum.iter_mut().zip(c).for_each(|(s, &x)| *s += f64::from(x));
    }

    let mut new_group = vec![u32::MAX; ngroups];
    let mut parents = Vec::new();

    for (g, &n) in gcounts.iter().enumerate().filter(|(_, n)| **n > 0) {
        new_group[g] = (parents.len() / dim) as u32;
        let sum = &gsums[g * dim..(g + 1) * dim];
        parents.extend(sum.iter().map(|&s| (s / n as f64) as f32));
    }

    Clusters {
        dim,
        assignment: leaves.iter().map(|&l| new_leaf[l as usize]).collect(),
        parent_of: groups_of_kept.iter().map(|&g| new_group[g]).collect(),
        centroids,
        parents,
    }
}

fn read_span(file: &File, span: Span) -> Result<Vec<u8>, Error> {
    let mut bytes = vec![0u8; span.len];
    io::read_at(file, &mut bytes, span.offset)?;
    Ok(bytes)
}

fn damaged(message: &str) -> Error {
    Error::new(ErrorKind::InvalidData, message)
}

#[cfg(test)]
#[path = "../tests/unit/index.rs"]
mod tests;
