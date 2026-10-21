//! An index's data file opened for search, as one generation of the index. Opening finds the index's current data
//! file, opens it once, reads the first 16 KiB and maps the whole file without loading it, and every later read of the
//! file goes through that same handle. It also reads the file's change log, if it has one. The cluster tables load on
//! the first search.
//!
//! Every open index of a path in this process shares one cell that holds its current generation. A search loads the
//! cell once and keeps that generation until it returns. A commit or a refresh swaps in a newer generation, and the old
//! one waits in the cell until no search holds it, so a search never frees it.

use std::cell::Cell;
use std::fs::File;
use std::io::{Error, ErrorKind};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};

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
use crate::search::scan::{Hides, Pending};
use crate::search::{
    InPlace, LIMIT, RowPages, View, check_search, depth, invalid, normalize, prepare_query, run_route, scan_in_place,
    scan_next, scratch_for,
};
use crate::trace::{Clock, Trace};
use crate::update::{self, Changes, Delta};
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
    /// from before the first commit. And a stamp of them as the last refresh, or this process's writer's last commit,
    /// saw them.
    watched: [PathBuf; 3],
    stamp: AtomicU64,
    /// Set while this process's writer commits a generation, which it then makes current itself.
    committing: AtomicBool,
    /// When a search of the index last ran, in any process, which a maintenance job waits on.
    activity: Arc<io::Activity>,
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
        let mut open = update::locked(&SHARED);
        open.retain(|s| s.strong_count() > 0);
        let joined = open.iter().filter_map(Weak::upgrade).find(|s| {
            let cur = s.current.load();
            s.path == path && cur.same_index(&inner)
        });

        // While this process's writer commits, the generation it publishes takes the place of the one before.
        if let Some(shared) = joined {
            let cur = shared.current.load_full();
            if inner.at.generation > cur.at.generation && !shared.committing.load(Ordering::Acquire) {
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
        let activity = Arc::new(io::Activity::open(&dir));
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
                committing: AtomicBool::new(false),
                activity,
            }
        })
    }

    /// Notes that a search of the index runs now, for the maintenance jobs of every process.
    #[inline]
    pub(crate) fn note(&self) {
        self.activity.note();
    }

    /// When a search of the index last ran, which a maintenance job waits on.
    pub(crate) fn activity(&self) -> Arc<io::Activity> {
        Arc::clone(&self.activity)
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
        update::locked(&self.retired).push(prev);
        true
    }

    /// Marks that this process's writer commits a generation, until the returned guard drops. Meanwhile a refresh or
    /// an open of this process leaves the move to the writer, so none of them opens a copy of the new generation.
    pub(crate) fn committing(&self) -> Committing<'_> {
        self.committing.store(true, Ordering::Release);
        Committing(self)
    }

    /// Makes `next`, the generation that this process's writer just committed, current in place of `from`, or in place
    /// of a copy of the same generation that a refresh or an open of this process made current as the commit began.
    /// The writer's own one has its cluster tables loaded and keeps the warm bit. Then it takes the stamp of the files
    /// as the commit left them, so a `prepare` of this process doesn't look for a newer generation. Returns whether it
    /// moved.
    pub(crate) fn publish_own(&self, from: &Arc<Inner>, next: Arc<Inner>) -> bool {
        let cur = self.current.load_full();
        let from = match cur.is(&next.at) {
            true => &cur,
            false => from,
        };
        let moved = self.publish(from, next);
        self.stamp.store(stamp_of(&self.watched), Ordering::Relaxed);
        moved
    }

    /// Moves to the generation that the root names, when another process committed a newer one, or to the file at the
    /// path when a rebuild replaced the one before the index's first commit, as `follow` does. Then it frees the old
    /// generations that nothing holds. Returns whether it moved.
    pub(crate) fn refresh(&self) -> Result<bool, Error> {
        let moved = self.follow()?;
        self.reclaim();
        Ok(moved)
    }

    /// The same, but it frees nothing. While this process's writer commits, it does nothing, since the writer then
    /// makes the new generation current. Returns whether it moved.
    pub(crate) fn follow(&self) -> Result<bool, Error> {
        if self.committing.load(Ordering::Acquire) {
            return Ok(false);
        }

        self.stamp.store(stamp_of(&self.watched), Ordering::Relaxed);
        // The index's first commit makes its folder, and with it the word that every process shares.
        self.activity.map();
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
        Ok(moved)
    }

    /// Frees the generations swapped out that nothing else holds now, on the calling thread.
    pub(crate) fn reclaim(&self) {
        let freed: Vec<Arc<Inner>> = {
            let mut retired = update::locked(&self.retired);
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
        update::locked(&self.retired).len()
    }
}

/// A cell's mark that this process's writer commits, which it clears when it drops, whether the commit failed or not.
pub(crate) struct Committing<'a>(&'a Shared);

impl Drop for Committing<'_> {
    fn drop(&mut self) {
        self.0.committing.store(false, Ordering::Release);
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
    let open = update::locked(&SHARED);
    let mut live = open.iter().filter_map(Weak::upgrade);
    live.find(|s| s.path == path && fits(&s.current.load()))
}

/// What the searches of a generation scanned while it had pending inserts: how many searches, the vectors of the
/// clusters they probed, and the pending inserts of those clusters. Each search adds once, at its end.
#[derive(Default)]
pub(crate) struct Scanned {
    searches: AtomicU64,
    vectors: AtomicU64,
    inserts: AtomicU64,
}

impl Scanned {
    /// Adds a search that scanned `lanes`: the vectors of the clusters it read, their inserts, and the inserts of the
    /// clusters ranked next. Only the first two count toward the excess, which tells inserts that crowd the clusters
    /// that searches read. The inserts of the next clusters grow with every insert alike, which the pending share
    /// bounds.
    pub(crate) fn add(&self, lanes: [u64; 3]) {
        if lanes[..2] != [0, 0] {
            self.searches.fetch_add(1, Ordering::Relaxed);
            self.vectors.fetch_add(lanes[0], Ordering::Relaxed);
            self.inserts.fetch_add(lanes[1], Ordering::Relaxed);
        }
    }

    /// The share more that the searches scanned than they would with every change folded in, when `dead` is the share
    /// of the file's vectors that changes hid, and the searches that it counts.
    pub(crate) fn excess(&self, dead: f64) -> (f64, u64) {
        let (vectors, inserts) = (
            self.vectors.load(Ordering::Relaxed),
            self.inserts.load(Ordering::Relaxed),
        );
        let excess = inserts as f64 / vectors.max(1) as f64 + dead;
        (excess, self.searches.load(Ordering::Relaxed))
    }
}

/// Every pending insert that a search counted in `lanes`, those of the clusters it read and of those ranked next.
pub(crate) fn inserts(lanes: &Cell<[u64; 3]>) -> usize {
    let [_, read, next] = lanes.get();
    (read + next) as usize
}

/// An insert as `encode_insert` measures it: the cluster it is nearest to, its 1-bit code and factors and its 8-bit
/// row, and the cluster, code and factors that it gets in the next generation, when a job writes one.
pub(crate) struct Encoded {
    pub(crate) leaf: u32,
    pub(crate) code: Vec<u8>,
    pub(crate) factors: Factors,
    pub(crate) row: Vec<u8>,
    pub(crate) next: Option<(u32, Vec<u8>, Factors)>,
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
    /// What the searches of this generation scanned, for the read-cost excess that makes maintenance due.
    scanned: Scanned,
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
        scratch_for(scratch, self.layout.dim, limits);
        let slot = match self.tables.get() {
            Some(tables) => self.reads(tables, ctx),
            None => {
                let make = || Reads::ahead(self, (self.cold.wave)(), &self.layout, self.len, limits);
                reads_for(reads, self, 0, limits, make)
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
        let delta = self.changes.load();
        let view = self.view_with(tables, &delta);
        let SearchCtx {
            limits,
            scratch,
            hits,
            trace,
            ..
        } = ctx;
        let s = scratch_for(scratch, view.dim, limits);
        trace.buffers = clock.lap();
        prepare_query(&view, query, s)?;
        let lanes = Cell::new([0; 3]);
        scan_in_place(&view, data.blocks, params, s, self.pending(&delta, &lanes));
        self.scanned.add(lanes.get());
        (trace.inserts_scanned, trace.generation) = (inserts(&lanes), self.at.generation);
        let seen = delta.seen();
        // The route and the scan in place count as the scan.
        trace.scan = clock.lap();
        trace.scanned = s.routes.iter().map(|r| view.lists[r.id as usize].count).sum();
        let row = |id: u32| self.row_of(&delta, id, |id| data.rows.row(id));
        let live = |id: u32, _: &[u8]| delta.visible(id, seen);
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

    /// The pending inserts of `delta` that a search scans with the clusters it probes and those ranked next, or `None`
    /// when there are none. The scan counts the vectors and the inserts of the clusters it probes, then the inserts of
    /// those ranked next, in `lanes`.
    pub(crate) fn pending<'a>(&self, delta: &'a Delta, lanes: &'a Cell<[u64; 3]>) -> Option<Pending<'a>> {
        (delta.len() > 0).then(|| Pending {
            delta,
            seen: delta.seen(),
            first: self.layout.count as u32,
            lanes,
        })
    }

    /// What the searches of this generation scanned.
    pub(crate) fn scanned(&self) -> &Scanned {
        &self.scanned
    }

    /// The row of internal id `id`, from the file through `in_file` or from the inserted vectors.
    pub(crate) fn row_of<'a>(&self, delta: &'a Delta, id: u32, in_file: impl Fn(u32) -> &'a [u8]) -> &'a [u8] {
        match (id as usize).checked_sub(self.layout.count) {
            Some(i) => delta.row(i),
            None => in_file(id),
        }
    }

    /// Writes the `nprobe` clusters nearest the query into `out`, nearest first, keeping only those within `reach`.
    /// When the index has pending inserts, `ctx` keeps the clusters ranked after those, up to the `depth` nearest,
    /// whose inserts a search of what is fetched scans too. It starts a new trace, which a fetch and the search of what
    /// it fetched then add to.
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
        scratch_for(&mut ctx.scratch, self.layout.dim, &ctx.limits);
        ctx.trace.buffers = clock.lap();
        let deep = depth(nprobe, self.changes.load().len() > 0);
        run_route(&self.view(tables), query, nprobe, reach, deep, ctx, out)?;
        ctx.next_of = Some((self.cell(), self.uid()));
        ctx.trace.route = clock.lap();
        Ok(())
    }

    /// The cluster nearest `vector` among `centers`, the centers of another generation of the index, eg. the next one
    /// while a job writes it. Every generation has the same mean and rotation, so the vector routes as it would there.
    pub(crate) fn route_among(&self, vector: &[f32], centers: Centers, ctx: &mut SearchCtx) -> Result<u32, Error> {
        let view = View {
            centers,
            ..self.view(self.tables()?)
        };
        let mut leaves = Vec::with_capacity(1);
        run_route(&view, vector, 1, None, 1, ctx, &mut leaves)?;
        Ok(leaves.first().copied().unwrap_or(0))
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
        self.reads_of(tables, ctx, 0)
    }

    /// The same, for the index's copy `copy` in a set of collections that names it more than once, which has reads of
    /// its own.
    pub(crate) fn reads_of(self: &Arc<Inner>, tables: &Tables, ctx: &mut SearchCtx, copy: usize) -> usize {
        let SearchCtx { limits, reads, .. } = ctx;
        let make = || Reads::new(self, (self.cold.wave)(), &self.layout, self.len, tables, limits);
        let slot = reads_for(reads, self, copy, limits, make);
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
        let delta = self.changes.load();
        let (view, data) = (self.view_with(tables, &delta), self.in_place());
        let SearchCtx {
            limits,
            scratch,
            reads,
            hits,
            trace,
            ..
        } = ctx;
        let s = scratch_for(scratch, view.dim, limits);
        trace.buffers = clock.lap();
        prepare_query(&view, query, s)?;
        let lanes = Cell::new([0; 3]);
        scan_in_place(&view, data.blocks, params, s, self.pending(&delta, &lanes));
        self.scanned.add(lanes.get());
        (trace.inserts_scanned, trace.generation) = (inserts(&lanes), self.at.generation);
        let seen = delta.seen();
        // The route and the scan in place count as the scan.
        trace.scan = clock.lap();
        trace.scanned = s.routes.iter().map(|r| view.lists[r.id as usize].count).sum();
        let r = &mut reads[slot];
        r.forget();
        r.read_rows(&self.cold.file, s.candidates.as_slice(), true)?;
        trace.rows = clock.lap();
        (trace.row_bytes, trace.row_ranges) = r.row_traffic();
        (trace.cache_hits, trace.read_path) = (r.hits(), r.read_path());
        let row = |id: u32| self.row_of(&delta, id, |id| r.row(id));
        let live = |id: u32, _: &[u8]| delta.visible(id, seen);
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
    /// decodes, the same center the file's vectors are measured from. With `next`, the centers of the generation that
    /// a job writes, it also routes and encodes the vector there, sharing the rotation, unless it doesn't fit there. A
    /// vector that doesn't fit the index fails with `InvalidInput`.
    pub(crate) fn encode_insert(
        &self,
        id: u64,
        vector: &[f32],
        ctx: &mut SearchCtx,
        next: Option<Centers>,
    ) -> Result<Encoded, Error> {
        let (dim, count) = (self.layout.dim, self.layout.count);

        if vector.len() != dim {
            return Err(invalid("the vector's length doesn't match the index's dimension"));
        }

        if !vector.iter().all(|x| x.is_finite()) {
            return Err(invalid("a vector has a value that isn't finite"));
        }

        if count + self.changes.load().len() >= u32::MAX as usize - BLOCK {
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
        let fits = |f: &Factors| f64::from(f.f_add.abs()) <= LIMIT && f.f_rescale.is_finite() && f.f_err.is_finite();

        if !(fits(&factors) && f64::from(sq8::norm2(&row, dim)) <= LIMIT) {
            return Err(invalid("a vector is too far from the mean to encode"));
        }

        let next = match next {
            None => None,
            Some(centers) => {
                let list = self.route_among(&o, centers, ctx)?;
                let (mut code, mut center) = (code.clone(), vec![0f32; dim]);
                let factors = match self.layout.anchor {
                    Anchor::Mean => factors,
                    Anchor::Cluster => {
                        let (codes, f) = centers.center(list);
                        sq4::decode(codes, f, &mut center);
                        encoder.again_from(&center, &mut code)
                    }
                };
                fits(&factors).then_some((list, code, factors))
            }
        };

        Ok(Encoded {
            leaf,
            code,
            factors,
            row,
            next,
        })
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

        let delta = self.changes.load();
        let view = self.view_with(self.tables()?, &delta);
        let SearchCtx {
            limits, scratch, reads, ..
        } = ctx;
        let s = scratch_for(scratch, view.dim, limits);
        prepare_query(&view, query, s)?;
        s.candidates.reset(n);
        let r = &mut reads[slot];
        // A guess at the query scans what the search will scan again, so it counts nothing.
        r.scan(&self.cold.file, &view, s, self.pending(&delta, &Cell::new([0; 3])))?;
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
        let delta = self.changes.load();
        let view = self.view_with(self.tables()?, &delta);
        let SearchCtx {
            limits,
            scratch,
            reads,
            hits,
            trace,
            next,
            next_of,
            ..
        } = ctx;
        let s = scratch_for(scratch, view.dim, limits);
        trace.buffers += clock.lap();
        prepare_query(&view, query, s)?;
        s.candidates.reset(params.rerank.max(params.k));
        let r = &mut reads[slot];
        let lanes = Cell::new([0; 3]);
        let pending = self.pending(&delta, &lanes);
        r.scan(&self.cold.file, &view, s, pending)?;

        // The inserts of the clusters that the last route of this generation ranked next, unless they were fetched.
        if let Some(p) = pending
            && *next_of == Some((self.cell(), self.uid()))
        {
            let fetched = r.probed();
            let leaves = next.iter().copied().filter(|l| fetched.binary_search(l).is_err());
            scan_next(
                &view,
                leaves,
                &mut s.lut,
                &s.q32,
                p,
                &mut s.candidates,
                &mut s.survivors,
            );
        }

        self.scanned.add(lanes.get());
        (trace.inserts_scanned, trace.generation) = (inserts(&lanes), self.at.generation);
        let seen = delta.seen();
        let scanning = clock.lap();
        (trace.wave, trace.scan) = (r.waited(), scanning.saturating_sub(r.waited()));
        r.read_rows(&self.cold.file, s.candidates.as_slice(), false)?;
        trace.rows = clock.lap();
        (trace.row_bytes, trace.row_ranges) = r.row_traffic();
        (trace.cache_hits, trace.read_path) = (r.hits(), r.read_path());
        let row = |id: u32| self.row_of(&delta, id, |id| r.row(id));
        let live = |id: u32, _: &[u8]| delta.visible(id, seen);
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

    /// What a search reads of this index without its changes.
    pub(crate) fn view<'a>(&'a self, tables: &'a Tables) -> View<'a> {
        View {
            hides: Hides::NONE,
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

    /// The same, skipping the rows that `delta` hid.
    pub(crate) fn view_with<'a>(&'a self, tables: &'a Tables, delta: &'a Delta) -> View<'a> {
        View {
            hides: Hides::of(delta),
            ..self.view(tables)
        }
    }

    /// Whether the system holds each of the rows section's pages in memory, by the file's page size.
    pub(crate) fn resident_rows(&self) -> Vec<bool> {
        let (rows, page) = (self.in_place().rows.bytes, self.layout.page_size);
        let held = io::resident(rows);
        // The system's pages may be smaller or larger than the file's, and need not start where the section does.
        let (system, lead) = (io::page_size(), rows.as_ptr() as usize % io::page_size());
        let on = |byte: usize| (lead + byte) / system;
        (0..rows.len().div_ceil(page))
            .map(|p| (on(p * page)..=on(((p + 1) * page).min(rows.len()) - 1)).any(|s| held.get(s) == Some(&true)))
            .collect()
    }

    /// Touches a byte of each page of the codes section's map, so a scan in place of them takes no page fault for a
    /// page that the system holds in memory.
    pub(crate) fn prefault_codes(&self) {
        let codes = self.in_place().blocks;
        codes.iter().step_by(4096).for_each(|b| _ = std::hint::black_box(*b));
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
            changes: update::changes_for(path, &layout, &at.data, &at.log, at.version())?,
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
            scanned: Scanned::default(),
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
