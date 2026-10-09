//! An index's data file opened for search, as one generation of the index. Opening finds the index's current data
//! file, opens it once, reads the first 16 KiB and maps the whole file without loading it, and every later read of the
//! file goes through that same handle. It also reads the file's change log, if it has one. The cluster tables load on
//! the first search.
//!
//! Every open index of a path in this process shares one cell that holds its current generation. A search loads the
//! cell once and keeps that generation until it returns. A commit or a refresh swaps in a newer generation, and the old
//! one waits in the cell until no search holds it, so a search never frees it.

use std::fs::File;
use std::io::{Error, ErrorKind};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock, PoisonError, Weak};

use arc_swap::{ArcSwap, Guard};

use crate::format::{self, FIRST_READ, Layout, Part, Tables, Trailer};
use crate::io::map::Map;
use crate::io::{self, Cold, NewWave, Pressure, Span, Warmth};
use crate::kernels::Tier;
use crate::quantization::rabitq::{BLOCK, Encoder, Factors, code_bytes};
use crate::quantization::rotation::Rotation;
use crate::quantization::{padded_dim, sq4, sq8};
use crate::root::{self, Located};
use crate::search::cold::{Reads, reads_for};
use crate::search::rerank::rerank_live;
use crate::search::route::Centers;
use crate::search::scan::{scan_each, scan_list};
use crate::search::{
    InPlace, LIMIT, RowPages, Scratch, View, check_search, invalid, normalize, prepare_query, run_route, scan_in_place,
    scratch_for,
};
use crate::trace::{Clock, Trace};
use crate::update::{self, Changes, Delta, Seen};
use crate::{Anchor, Hit, LeafId, Limits, Metric, SearchCtx, SearchParams};

/// The most bytes that `verify` and a warm-up read at once.
const PIECE: usize = 1 << 22;

/// The most times an open starts again because a rebuild or a commit replaced the index's data file while it ran.
const REOPENS: usize = 3;

/// The next number of a generation opened in this process.
static NEXT_UID: AtomicU64 = AtomicU64::new(1);

/// The shared cells of the open indexes of this process.
static SHARED: Mutex<Vec<Weak<Shared>>> = Mutex::new(Vec::new());

/// One index as every open handle of it in this process shares it: the generation that it holds now, and the ones it
/// held before that a search or a fetch may still hold.
pub(crate) struct Shared {
    /// A number that no other cell of this process has, which search contexts key their reads by.
    id: u64,
    me: Weak<Shared>,
    /// The index's path, through symbolic links.
    path: PathBuf,
    current: ArcSwap<Inner>,
    /// The generations swapped out, until nothing else holds them.
    retired: Mutex<Vec<Arc<Inner>>>,
    /// The read backend of every generation, when a benchmark chose one.
    wave: Option<NewWave>,
    /// The files whose change tells that the index may have a newer generation: its two root slots and its data file
    /// from before the first commit. And a stamp of them as the last refresh saw them.
    watched: [PathBuf; 3],
    stamp: AtomicU64,
}

impl Shared {
    /// The cell of the index at `path`: the one that an open handle of this process shares, moved to the generation
    /// that the root names when that is newer, or a new one.
    pub(crate) fn open(path: &Path) -> Result<Arc<Shared>, Error> {
        let real = update::real_path(path);
        let at = root::locate(&real)?;

        // An open reads whatever the log holds past what this process has read, as the open of a new handle would.
        if let Some(shared) = find(&real, |cur| cur.is(&at)) {
            let cur = shared.load();
            update::read_more(&cur.changes, &cur.at.log, &cur.layout, cur.at.version())?;
            drop(cur);
            return Ok(shared);
        }

        Ok(Shared::adopt(&real, Inner::open(&real)?))
    }

    /// The cell for `inner`, a generation of the index at `path`: the one that holds it or an older generation of the
    /// same index, which moves forward to it, or else a new one.
    pub(crate) fn adopt(path: &Path, inner: Inner) -> Arc<Shared> {
        let mut open = SHARED.lock().unwrap_or_else(PoisonError::into_inner);
        open.retain(|s| s.strong_count() > 0);
        let joined = open.iter().filter_map(Weak::upgrade).find(|s| {
            let cur = s.current.load();
            s.path == path && cur.same_index(&inner)
        });

        if let Some(shared) = joined {
            let cur = shared.current.load_full();
            if inner.at.generation > cur.at.generation {
                shared.publish(&cur, Arc::new(inner));
            }
            return shared;
        }

        let shared = Shared::alone(path, inner, None);
        open.push(Arc::downgrade(&shared));
        shared
    }

    /// A cell of its own for `inner`, which no open joins, eg. for another read backend, `wave`, which the generations
    /// it moves to get too.
    pub(crate) fn alone(path: &Path, inner: Inner, wave: Option<NewWave>) -> Arc<Shared> {
        let dir = root::folder(path);
        let watched = [dir.join(root::SLOTS[0]), dir.join(root::SLOTS[1]), path.to_path_buf()];
        let stamp = AtomicU64::new(stamp_of(&watched));
        Arc::new_cyclic(|me| {
            inner.join(inner.uid, me.clone());
            Shared {
                id: inner.uid,
                me: me.clone(),
                path: path.to_path_buf(),
                current: ArcSwap::from_pointee(inner),
                retired: Mutex::new(Vec::new()),
                wave,
                watched,
                stamp,
            }
        })
    }

    /// Whether the files that tell of a newer generation changed since the last refresh. It only reads their metadata,
    /// so it allocates nothing for paths of up to a few hundred bytes.
    pub(crate) fn may_have_moved(&self) -> bool {
        stamp_of(&self.watched) != self.stamp.load(Ordering::Relaxed)
    }

    /// The current generation, which a search keeps until it returns.
    pub(crate) fn load(&self) -> Guard<Arc<Inner>> {
        self.current.load()
    }

    /// The same, as its own reference, eg. for a fetch, which keeps it past the call.
    pub(crate) fn load_full(&self) -> Arc<Inner> {
        self.current.load_full()
    }

    /// Makes `next` current if `from` still is, and keeps `from` until nothing else holds it. Returns whether it did.
    pub(crate) fn publish(&self, from: &Arc<Inner>, next: Arc<Inner>) -> bool {
        next.join(self.id, self.me.clone());
        let prev = self.current.compare_and_swap(from, next);

        if !Arc::ptr_eq(&prev, from) {
            return false;
        }

        let prev = Guard::into_inner(prev);
        self.retired.lock().unwrap_or_else(PoisonError::into_inner).push(prev);
        true
    }

    /// Moves to the generation that the root names, when another process committed a newer one, or to the file at the
    /// path when a rebuild replaced the one before the index's first commit. Returns whether it moved. Then it frees
    /// the old generations that nothing holds.
    pub(crate) fn refresh(&self) -> Result<bool, Error> {
        self.stamp.store(stamp_of(&self.watched), Ordering::Relaxed);
        let cur = self.current.load_full();
        let at = root::locate(&self.path)?;
        let newer = match (&cur.at.root, &at.root) {
            (None, Some(_)) => true,
            (Some(c), Some(r)) => r.index_id == c.index_id && r.generation > c.generation,
            (None, None) => !names(&at.data, &cur.cold.file)?,
            (Some(_), None) => false,
        };
        let moved = newer && {
            let next = Inner::open(&self.path)?;
            let next = match self.wave {
                Some(wave) => next.with_wave(wave),
                None => next,
            };
            self.publish(&cur, Arc::new(next))
        };
        drop(cur);
        self.reclaim();
        Ok(moved)
    }

    /// Frees the generations swapped out that nothing else holds now, on the calling thread.
    pub(crate) fn reclaim(&self) {
        let freed: Vec<Arc<Inner>> = {
            let mut retired = self.retired.lock().unwrap_or_else(PoisonError::into_inner);
            let (free, keep) = std::mem::take(&mut *retired)
                .into_iter()
                .partition(|g| Arc::strong_count(g) == 1);
            *retired = keep;
            free
        };
        drop(freed);
    }

    /// How many generations swapped out wait to be freed.
    #[cfg(test)]
    pub(crate) fn retired(&self) -> usize {
        self.retired.lock().unwrap_or_else(PoisonError::into_inner).len()
    }
}

/// A stamp of the metadata of `files`: each one's length, modification time and inode, or that it is missing.
fn stamp_of(files: &[PathBuf]) -> u64 {
    use std::hash::{Hash, Hasher};

    let mut h = std::hash::DefaultHasher::new();

    for f in files {
        match std::fs::metadata(f) {
            Ok(m) => {
                (m.len(), m.modified().ok()).hash(&mut h);
                #[cfg(unix)]
                std::os::unix::fs::MetadataExt::ino(&m).hash(&mut h);
            }
            Err(_) => 0u8.hash(&mut h),
        }
    }

    h.finish()
}

/// The cell of the index at `path` whose current generation passes `fits`, if an open handle of this process shares
/// one.
fn find(path: &Path, fits: impl Fn(&Inner) -> bool) -> Option<Arc<Shared>> {
    let open = SHARED.lock().unwrap_or_else(PoisonError::into_inner);
    let mut live = open.iter().filter_map(Weak::upgrade);
    live.find(|s| s.path == path && fits(&s.current.load()))
}

pub(crate) struct Inner {
    /// A number that no other generation opened in this process has, and the number of the cell that holds it, which
    /// search contexts key their reads by, and the cell itself.
    uid: u64,
    cell: AtomicU64,
    owner: OnceLock<Weak<Shared>>,
    /// The index's path, and the generation of it that this data file holds.
    path: PathBuf,
    at: Located,
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
    /// Opens the current generation of the index at `path`. A damaged file fails with `InvalidData`.
    pub(crate) fn open(path: &Path) -> Result<Inner, Error> {
        Opening::start(path)?.finish(path)
    }

    /// Opens generation `at` of the index at `path`, eg. the one that a commit is about to make current.
    pub(crate) fn open_at(path: &Path, at: Located) -> Result<Inner, Error> {
        let cold = io::open_cold(&at.data)?;
        let len = usize::try_from(cold.file.metadata()?.len()).map_err(|_| damaged("the file is too large"))?;
        Opening { cold, len, at }.finish_once(path)
    }

    /// The generation of the index that this data file holds.
    pub(crate) fn located(&self) -> &Located {
        &self.at
    }

    /// The number of this generation, which no other opened in this process has.
    pub(crate) fn uid(&self) -> u64 {
        self.uid
    }

    /// The number of the cell that holds this generation, or its own number before a cell holds it.
    pub(crate) fn cell(&self) -> u64 {
        self.cell.load(Ordering::Relaxed)
    }

    /// The cell that holds this generation, if one does.
    pub(crate) fn owner(&self) -> Weak<Shared> {
        self.owner.get().cloned().unwrap_or_default()
    }

    /// Puts this generation in the cell `owner`, numbered `cell`. A generation goes in one cell in its life.
    fn join(&self, cell: u64, owner: Weak<Shared>) {
        if self.owner.set(owner).is_ok() {
            self.cell.store(cell, Ordering::Relaxed);
        }
    }

    /// Whether this is the generation at `at`: the one that the same root names, or before the first commit the same
    /// data file.
    fn is(&self, at: &Located) -> bool {
        match (&self.at.root, &at.root) {
            (Some(a), Some(b)) => (a.index_id, a.generation, &a.data) == (b.index_id, b.generation, &b.data),
            (None, None) => names(&at.data, &self.cold.file).unwrap_or(false),
            _ => false,
        }
    }

    /// Whether `other` is a generation of the same index: one of the same index id, or the generation after the first
    /// commit of an index before it, or before the first commit the same data file.
    fn same_index(&self, other: &Inner) -> bool {
        match (&self.at.root, &other.at.root) {
            (Some(a), Some(b)) => a.index_id == b.index_id,
            (None, Some(_)) => true,
            (None, None) => self.at.data == other.at.data && self.layout.check == other.layout.check,
            (Some(_), None) => false,
        }
    }

    /// The index's path.
    pub(crate) fn path(&self) -> &Path {
        &self.path
    }

    /// Asks the operating system (OS) to start reading the cluster tables, so the read overlaps the embedding model.
    /// Once the tables are loaded, it does nothing.
    pub(crate) fn prepare(&self) {
        if self.tables.get().is_none() {
            let span = format::tables_span(&self.layout);
            io::will_need(&self.cold.file, span.offset, span.len);
        }
    }

    /// Makes `ctx`'s buffers for this index, unless it has them, and readies its read backend. The cluster arena is
    /// sized from the cluster tables, so until they are loaded it waits for the first search. Once a search has gone
    /// through the reads, the backend may also do setup that waits, eg. register the arenas with the kernel.
    pub(crate) fn make_buffers(self: &Arc<Inner>, ctx: &mut SearchCtx) {
        let SearchCtx {
            limits, scratch, reads, ..
        } = ctx;
        scratch_for(scratch, self.layout.dim, self.layout.nlist, limits);
        let slot = match self.tables.get() {
            Some(tables) => self.reads(tables, ctx),
            None => {
                let make = || Reads::ahead(self, (self.cold.wave)(), &self.layout, self.len, limits);
                reads_for(reads, self, limits, make)
            }
        };
        ctx.reads[slot].prepare();
    }

    /// Checks a search's arguments against this index and `limits`.
    pub(crate) fn check(&self, query: &[f32], params: &SearchParams, limits: &Limits) -> Result<(), Error> {
        check_search(query, self.layout.dim, params, limits)
    }

    /// Searches through the memory map, which leaves every read to a page fault, the cluster tables' too if no search
    /// has loaded them yet. Tests check the cold search against it.
    pub(crate) fn search_mapped<'a>(
        &self,
        query: &[f32],
        params: &SearchParams,
        ctx: &'a mut SearchCtx,
    ) -> Result<&'a [Hit], Error> {
        self.check(query, params, &ctx.limits)?;

        ctx.trace = Trace::default();
        let mut clock = Clock::start();
        let (tables, data) = (self.mapped_tables()?, self.in_place());
        ctx.trace.tables = clock.lap();
        let view = self.view(tables);
        let SearchCtx {
            limits,
            scratch,
            hits,
            trace,
            ..
        } = ctx;
        let s = scratch_for(scratch, view.dim, view.lists.len(), limits);
        trace.buffers = clock.lap();
        prepare_query(&view, query, s)?;
        scan_in_place(&view, data.blocks, params, s);
        let delta = self.changes.read();
        let seen = self.scan_delta(&delta, &view, s);
        // The route and the scan in place count as the scan.
        trace.scan = clock.lap();
        trace.scanned = s.routes.iter().map(|r| view.lists[r.id as usize].count).sum();
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
        trace.rerank = clock.lap();

        Ok(&hits[..])
    }

    pub(crate) fn changes(&self) -> &Changes {
        &self.changes
    }

    pub(crate) fn layout(&self) -> &Layout {
        &self.layout
    }

    pub(crate) fn mean(&self) -> &[f32] {
        &self.mean
    }

    pub(crate) fn rotation(&self) -> &Rotation {
        &self.rotation
    }

    /// The file's length in bytes.
    pub(crate) fn file_len(&self) -> usize {
        self.len
    }

    /// What maintenance keeps about the clusters, read and checked through the cold handle, or `None` when the file has
    /// no trailer, eg. a fresh build's. A damaged trailer fails with `InvalidData`.
    pub(crate) fn trailer(&self) -> Result<Option<Trailer>, Error> {
        let at = format::trailer_at(self.layout.span(Part::Ids));

        if self.len <= at {
            return Ok(None);
        }

        let bytes = read_span(
            &self.cold.file,
            Span {
                offset: at,
                len: self.len - at,
            },
        )?;
        format::read_trailer(&bytes, &self.layout).map(Some)
    }

    /// Scans the inserted vectors for the query in `s`, as one more cluster, with internal ids from the file's count.
    /// A code measured from its cluster's center adds the query's squared distance to that center, which the scan
    /// measures once per cluster that holds inserts. Returns what it saw, so the rescore can tell which rows were
    /// visible then.
    pub(crate) fn scan_delta(&self, delta: &Delta, view: &View, s: &mut Scratch) -> Seen {
        if delta.len() == 0 {
            return delta.seen();
        }

        let first = self.layout.count as u32;

        match view.anchor {
            Anchor::Mean => {
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
            Anchor::Cluster => {
                let (q32, inserted) = (&s.q32, &mut s.inserted);
                inserted.clear();
                let t2_of = |&leaf: &u32| view.centers.distance2(view.tier, leaf, q32) as f32;
                inserted.extend(delta.clusters().iter().map(t2_of));
                let t2 = |i: usize| inserted[delta.place(i)];
                let (lut, candidates, survivors) = (&s.lut, &mut s.candidates, &mut s.survivors);
                scan_each(
                    view.tier,
                    delta.blocks(),
                    delta.len(),
                    first,
                    lut,
                    t2,
                    candidates,
                    survivors,
                );
            }
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

    /// Writes the `nprobe` clusters nearest the query into `out`, nearest first, keeping only those within `reach`. It
    /// starts a new trace, which a fetch and the search of what it fetched then add to.
    pub(crate) fn route(
        &self,
        query: &[f32],
        nprobe: usize,
        reach: Option<f32>,
        ctx: &mut SearchCtx,
        out: &mut Vec<LeafId>,
    ) -> Result<(), Error> {
        ctx.trace = Trace::default();
        let mut clock = Clock::start();
        let tables = self.tables()?;
        ctx.trace.tables = clock.lap();
        scratch_for(&mut ctx.scratch, self.layout.dim, tables.lists.len(), &ctx.limits);
        ctx.trace.buffers = clock.lap();
        run_route(&self.view(tables), query, nprobe, reach, ctx, out)?;
        ctx.trace.route = clock.lap();
        Ok(())
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

        let mut clock = Clock::start();
        let slot = self.reads(tables, ctx);
        ctx.trace.buffers += clock.lap();
        let r = &mut ctx.reads[slot];
        r.fetch(&self.cold.file, &tables.lists, leaves)?;
        ctx.trace.submit = clock.lap();
        (ctx.trace.cluster_bytes, ctx.trace.cluster_ranges) = r.cluster_traffic();
        ctx.trace.scanned = leaves.iter().map(|&l| tables.lists[l as usize].count).sum();
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

    /// The slot in `ctx` of its reads of this index, made the first time, with every arena, and planned again for this
    /// generation when they were planned for another.
    pub(crate) fn reads(self: &Arc<Inner>, tables: &Tables, ctx: &mut SearchCtx) -> usize {
        let SearchCtx { limits, reads, .. } = ctx;
        let make = || Reads::new(self, (self.cold.wave)(), &self.layout, self.len, tables, limits);
        let slot = reads_for(reads, self, limits, make);
        reads[slot].size(&self.cold.file, tables, limits);
        slot
    }

    /// Searches a warm index. It scans the clusters in place through the map, then reads the rows to rescore in a
    /// wave through the operating system's cache, which costs little for rows already there. The hits live in `ctx`.
    pub(crate) fn search_warm<'a>(
        self: &Arc<Inner>,
        query: &[f32],
        params: &SearchParams,
        ctx: &'a mut SearchCtx,
    ) -> Result<&'a [Hit], Error> {
        self.check(query, params, &ctx.limits)?;

        let mut clock = Clock::start();
        let tables = self.tables()?;
        ctx.trace = Trace {
            tables: clock.lap(),
            ..Trace::default()
        };
        let slot = self.reads(tables, ctx);
        let (view, data) = (self.view(tables), self.in_place());
        let SearchCtx {
            limits,
            scratch,
            reads,
            hits,
            trace,
            ..
        } = ctx;
        let s = scratch_for(scratch, view.dim, view.lists.len(), limits);
        trace.buffers = clock.lap();
        prepare_query(&view, query, s)?;
        scan_in_place(&view, data.blocks, params, s);
        let seen = self.scan_delta(&self.changes.read(), &view, s);
        // The route and the scan in place count as the scan.
        trace.scan = clock.lap();
        trace.scanned = s.routes.iter().map(|r| view.lists[r.id as usize].count).sum();
        let r = &mut reads[slot];
        r.forget();
        r.read_rows(&self.cold.file, s.candidates.as_slice(), true)?;
        trace.rows = clock.lap();
        (trace.row_bytes, trace.row_ranges) = r.row_traffic();
        (trace.cache_hits, trace.read_path) = (r.hits(), r.read_path());
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
        trace.rerank = clock.lap();

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
    /// factors, and its 8-bit row. With the cluster anchor the code is measured from that cluster's center as it
    /// decodes, the same center the file's vectors are measured from. A vector that doesn't fit the index fails with
    /// `InvalidInput`.
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

        let mut leaves = Vec::with_capacity(1);
        self.route(&o, 1, None, ctx, &mut leaves)?;
        let leaf = leaves.first().copied().unwrap_or(0);
        let mut code = vec![0u8; code_bytes(padded_dim(dim))];
        let mut encoder = Encoder::new(&self.rotation, &self.mean);
        let factors = match self.layout.anchor {
            Anchor::Mean => encoder.encode(&o, &mut code),
            Anchor::Cluster => {
                let (codes, f) = self.view(self.tables()?).centers.center(leaf);
                let mut center = vec![0f32; dim];
                sq4::decode(codes, f, &mut center);
                encoder.encode_from(&o, &center, &mut code)
            }
        };
        let mut row = vec![0u8; sq8::row_bytes(dim)];
        sq8::write_row(&o, &self.mean, id, &mut row);
        let fits = |x: f32| f64::from(x) <= LIMIT;

        if !(fits(factors.f_add.abs())
            && factors.f_rescale.is_finite()
            && factors.f_err.is_finite()
            && fits(sq8::norm2(&row, dim)))
        {
            return Err(invalid("a vector is too far from the mean to encode"));
        }

        Ok((leaf, code, factors, row))
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

        let mut clock = Clock::start();
        let view = self.view(self.tables()?);
        let SearchCtx {
            limits,
            scratch,
            reads,
            hits,
            trace,
            ..
        } = ctx;
        let s = scratch_for(scratch, view.dim, view.lists.len(), limits);
        trace.buffers += clock.lap();
        prepare_query(&view, query, s)?;
        s.candidates.reset(params.rerank.max(params.k));
        let r = &mut reads[slot];
        r.scan(&self.cold.file, &view, s)?;
        let seen = self.scan_delta(&self.changes.read(), &view, s);
        let scanning = clock.lap();
        (trace.wave, trace.scan) = (r.waited(), scanning.saturating_sub(r.waited()));
        r.read_rows(&self.cold.file, s.candidates.as_slice(), false)?;
        trace.rows = clock.lap();
        (trace.row_bytes, trace.row_ranges) = r.row_traffic();
        (trace.cache_hits, trace.read_path) = (r.hits(), r.read_path());
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
        trace.rerank = clock.lap();

        Ok(&hits[..])
    }

    /// Checks every section against its checksum, then the cluster tables, then the trailer if the file has one. It
    /// reads the whole file through the cold handle a piece at a time, so a failed read gives an error instead of a
    /// fault on a mapped page.
    pub(crate) fn verify(&self) -> Result<(), Error> {
        self.verify_in(PIECE)?;
        self.trailer().map(|_| ())
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
        self.load_tables(|span| read_span(&self.cold.file, span))
    }

    /// The same, read through the map the first time, which leaves the read to page faults.
    fn mapped_tables(&self) -> Result<&Tables, Error> {
        self.load_tables(|span| Ok(self.map.bytes()[span.range()].to_vec()))
    }

    /// The cluster tables, read with `read` and checked the first time.
    fn load_tables(&self, read: impl FnOnce(Span) -> Result<Vec<u8>, Error>) -> Result<&Tables, Error> {
        if let Some(tables) = self.tables.get() {
            return Ok(tables);
        }

        let bytes = read(format::tables_span(&self.layout))?;
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
            anchor: self.layout.anchor,
            mean: &self.mean,
            rotation: &self.rotation,
            centers: Centers {
                codes: &tables.codes,
                factors: &tables.factors,
                norms: &tables.norms,
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

    /// The same index with another read backend, so tests can check that every backend gives the same results, and
    /// benchmarks can compare them.
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

/// An index's data file opened, with its header's read started, so a set of files can start all of their reads before
/// any waits.
pub(crate) struct Opening {
    cold: Cold,
    len: usize,
    at: Located,
}

impl Opening {
    /// Opens the current data file of the index at `path`, once, and starts reading its first 16 KiB. A commit may
    /// delete the file that the root named between the lookup and the open, so then it looks again.
    pub(crate) fn start(path: &Path) -> Result<Opening, Error> {
        let mut tries = 0;

        loop {
            let at = root::locate(path)?;
            match io::open_cold(&at.data) {
                Ok(cold) => {
                    let len =
                        usize::try_from(cold.file.metadata()?.len()).map_err(|_| damaged("the file is too large"))?;
                    io::will_need(&cold.file, 0, FIRST_READ.min(len));
                    return Ok(Opening { cold, len, at });
                }
                Err(e) if e.kind() == ErrorKind::NotFound && at.root.is_some() && tries < REOPENS => tries += 1,
                Err(e) => return Err(e),
            }
        }
    }

    /// Reads and checks the header and the rotation, and maps the file, all through the one handle, so it reads one
    /// file even if a rebuild renames another over `path` meanwhile. Then it reads the file's log. Before the first
    /// commit, a rebuild may start the log again for its new file, and a commit may delete the log for its new
    /// generation, before this reads it, which would leave the old file without its changes. So then, and when a
    /// commit deleted a generation's log before this read it, it opens the index again, up to `REOPENS` times.
    pub(crate) fn finish(self, path: &Path) -> Result<Inner, Error> {
        let mut inner = self.finish_once(path);

        for _ in 0..REOPENS {
            let again = match &inner {
                Err(e) => e.kind() == ErrorKind::NotFound,
                Ok(inner) if inner.at.root.is_none() => {
                    let rebuilt = !names(path, &inner.cold.file)? && update::foreign_log(path, &inner.layout)?;
                    rebuilt || root::locate(path)?.root.is_some()
                }
                Ok(_) => false,
            };
            if !again {
                break;
            }
            inner = Opening::start(path).and_then(|o| o.finish_once(path));
        }

        inner
    }

    fn finish_once(self, path: &Path) -> Result<Inner, Error> {
        let Opening { cold, len, at } = self;
        let mut first = vec![0u8; FIRST_READ.min(len)];
        io::read_at(&cold.file, &mut first, 0)?;
        let layout = format::read_header(&first, len)?;
        // Above 2806 dimensions the rotation runs past the first read.
        let span = layout.span(Part::Rotation);
        let rotation_bytes = match first.get(span.range()) {
            Some(bytes) => bytes.to_vec(),
            None => read_span(&cold.file, span)?,
        };
        let (rotation, mean) = format::read_rotation(&rotation_bytes, &layout)?;
        let map = Map::new(&cold.file)?;

        if map.bytes().len() != len {
            return Err(damaged("the file changed while it was being opened"));
        }

        // The root names one data file by its header's checksum, and a generation's log must be there and name it too.
        // A commit may have deleted the log since the lookup, which gives `NotFound`.
        if let Some(root) = &at.root {
            if root.check != layout.check {
                return Err(damaged("the data file isn't the one that the index's root names"));
            }
            if update::log::open_ours(&at.log, &layout, at.version())?.is_none() {
                return match std::fs::exists(&at.log)? {
                    true => Err(damaged("the log doesn't belong to the data file that the root names")),
                    false => Err(Error::new(ErrorKind::NotFound, "the generation's log is gone")),
                };
            }
        }

        let uid = NEXT_UID.fetch_add(1, Ordering::Relaxed);
        Ok(Inner {
            uid,
            cell: AtomicU64::new(uid),
            owner: OnceLock::new(),
            changes: update::changes_for(path, &layout, &at.log, at.version())?,
            path: path.to_path_buf(),
            at,
            cold,
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
}

/// Whether `path` still names the file that `file` has open, by its device and inode. It says yes when `path` can't be
/// looked up.
#[cfg(unix)]
fn names(path: &Path, file: &File) -> Result<bool, Error> {
    use std::os::unix::fs::MetadataExt;

    let opened = file.metadata()?;
    let same = |now: std::fs::Metadata| (now.dev(), now.ino()) == (opened.dev(), opened.ino());
    Ok(std::fs::metadata(path).ok().is_none_or(same))
}

/// Elsewhere it can't tell, so it says yes.
#[cfg(not(unix))]
fn names(_path: &Path, _file: &File) -> Result<bool, Error> {
    Ok(true)
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

/// Brings the codes section of `index` into the operating system's cache in file order, a piece at a time, then sets
/// the warm bit. It does nothing for a warm index, and stops before a piece once the index has closed or a clear has
/// come since `mark`. Returns how many bytes it brought in.
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
        let piece = &mut buf[..PIECE.min(span.end() - at)];
        io::warm(&inner.cold.file, inner.map.bytes(), piece, at)?;
    }

    if let Some(inner) = index.upgrade() {
        inner.warmth.finish(mark);
    }

    Ok(span.len)
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
