#![allow(unused_variables, dead_code, clippy::ptr_arg)]

mod build;
mod format;
mod index;
mod io;
mod kernels;
mod maintain;
mod multi;
mod quantization;
mod root;
mod search;
mod trace;
mod update;

use std::io::Error;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

#[cfg(feature = "bench")]
pub use trace::{ReadPath, Trace};

/// How the index measures the distance between vectors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Metric {
    L2,
    Cosine,
}

/// Reads the input vectors a chunk of rows at a time (eg. from a memory-mapped `.fbin`). A build never holds the
/// whole dataset in RAM.
pub trait VectorSource {
    fn count(&self) -> usize;
    fn dim(&self) -> usize;
    /// Rows `start..start + rows`, row-major.
    fn rows(&self, start: usize, rows: usize) -> &[f32];
}

/// The order of the clusters in the file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LeafOrder {
    /// Clusters that queries usually read together sit next to each other on disk. This is the default.
    Chained,
    /// Cluster id order. It is the baseline for measuring what the chained order saves.
    Id,
}

/// What each vector's 1-bit code is measured from. The nearer that point is to the vector, the closer the vector's
/// distance estimate.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Anchor {
    /// The center of the vector's cluster. It sits nearer to the vector than the mean does, so a search needs fewer
    /// rescored candidates for the same recall. Each search also measures the query's distance to the center of every
    /// cluster it scans, and of every cluster that holds inserts until a compaction. This is the default.
    #[default]
    Cluster,
    /// The mean of all the vectors. It is the baseline for measuring what the cluster anchor gains.
    Mean,
}

// TODO: Add defaults, possibly a set per platform, eg. macOS, Linux and Windows.
/// Build settings.
#[derive(Clone, Debug)]
pub struct BuildParams {
    pub metric: Metric,
    /// Average vectors per cluster, ~160. The number of clusters is the vector count divided by this.
    pub vectors_per_list: usize,
    /// Groups of clusters used to order the file, 64 to 128.
    pub parents: usize,
    pub order: LeafOrder,
    pub anchor: Anchor,
    /// Page size for packing the precise vector copies that rescoring reads. 4096 lets one file work on both
    /// 16 KiB-page Macs and 4 KiB-page PCs.
    pub page_size: usize,
    /// Threads for clustering and encoding. 0 takes `RAYON_NUM_THREADS` when it is set, and otherwise one thread per
    /// logical CPU. The file is the same for any number.
    pub threads: usize,
    pub seed: u64,
}

/// A clustering for [`Builder::build_with_clusters`], eg. one computed elsewhere with faiss, or the one
/// [`Builder::cluster`] finds.
pub struct Clusters {
    /// The vector dimension, eg. 768.
    pub dim: usize,
    /// Every cluster's center, one after another, so cluster i is `centroids[i * dim..(i + 1) * dim]`. Its length is
    /// the number of clusters times `dim`, eg. 6250 × 768 at 1M vectors.
    pub centroids: Vec<f32>,
    /// The cluster of each input vector, in input order. `assignment[r] = i` puts row r in cluster i. It has one
    /// entry per vector, eg. 1M.
    pub assignment: Vec<u32>,
    /// Every group's center, laid out like `centroids`. A group is a set of nearby clusters, and the groups decide the
    /// order of the clusters in the file. Its length is the number of groups times `dim`, eg. 128 × 768.
    pub parents: Vec<f32>,
    /// The group of each cluster. `parent_of[i] = g` puts cluster i in group g. It has one entry per cluster, eg. 6250
    /// at 1M vectors.
    pub parent_of: Vec<u32>,
}

/// How a build finds its clusters.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Clustering {
    /// k-means over every vector for the clusters, then k-means over the cluster centers for the groups. Its cost grows
    /// with the square of the vector count, so it suits up to a few million vectors. This is the default.
    Flat,
    /// k-means over a sample for the groups first, then k-means inside each group for its clusters, then a few rounds
    /// over all the vectors, where each vector may join the clusters of its few nearest groups. It costs about the
    /// square root of the cluster count per vector, so it suits tens of millions of vectors, and it holds a few numbers
    /// per vector, not the vectors. Its recall for the same number of vectors scanned is a little below flat's.
    TwoLevel,
}

pub struct Builder {
    params: BuildParams,
    clustering: Clustering,
}

impl Builder {
    pub fn new(params: BuildParams) -> Builder {
        Builder {
            params,
            clustering: Clustering::Flat,
        }
    }

    /// Chooses how `build` finds its clusters.
    pub fn clustering(mut self, clustering: Clustering) -> Builder {
        self.clustering = clustering;
        self
    }

    /// Clusters, orders, encodes and writes `src` to `out`. k-means finds clusters of about `vectors_per_list` vectors,
    /// and the groups that order them, as `Builder::clustering` says. It encodes and writes the vectors a part at a
    /// time, so the build holds a bounded amount of memory for any number of vectors, plus 8 bytes per vector and the
    /// cluster centers, or 20 bytes per vector while two-level clustering runs. The cluster anchor holds the centers a
    /// second time, as they decode. It writes a temporary file and a spill file next to `out`, and two-level clustering
    /// a copy of the vectors there too, then syncs the temporary file and renames it over `out`.
    pub fn build(&self, src: &dyn VectorSource, out: &Path) -> Result<(), Error> {
        build::build(&self.params, self.clustering, src, out)
    }

    /// The same, with an imported clustering. The clusters must hold their groups, so for clusters made elsewhere, eg.
    /// with faiss, which come without them, call [`Builder::group`] first.
    pub fn build_with_clusters(&self, src: &dyn VectorSource, clusters: &Clusters, out: &Path) -> Result<(), Error> {
        build::build_with_clusters(&self.params, src, clusters, out)
    }

    /// Finds the clusters and groups that `build` would build `out` from, without building it, eg. to save them or to
    /// build several files from one clustering. [`Builder::build_with_clusters`] with them writes the same file as
    /// `build`. Two-level clustering writes its temporary copy of the vectors next to `out`, as `build` does.
    pub fn cluster(&self, src: &dyn VectorSource, out: &Path) -> Result<Clusters, Error> {
        build::find_clusters(&self.params, self.clustering, src, out)
    }

    /// Finds `params.parents` groups for `clusters`, at most one per cluster, with k-means over their centers as a flat
    /// build does, and puts them in `clusters.parents` and `clusters.parent_of`. It is for clusters made elsewhere, eg.
    /// with faiss, which come without groups.
    pub fn group(&self, clusters: &mut Clusters) -> Result<(), Error> {
        build::group(&self.params, clusters)
    }
}

/// An open collection, stored as one file, or once maintenance has run, as the current generation of the files in its
/// folder. It is cheap to clone and safe to share across threads. Every open index of the same path in this process
/// shares one current generation, so a commit by this process's writer reaches them all at once.
#[derive(Clone)]
pub struct Index {
    shared: Arc<index::Shared>,
}

impl Index {
    /// Reads the header and the rotation, the first 16 KiB up to 2806 dimensions, and maps the file without loading it.
    /// It also reads the file's change log, if it has one, which takes time and memory in proportion to the inserts
    /// since the last maintenance run. Nothing else is read until a query. An index already open in this process
    /// shares its generation instead. Don't rewrite the file in place while it is open, eg. by copying over it. The
    /// index reads it a little at a time, so it would pair the old cluster tables with the new vectors. A rebuild that
    /// renames a new file over it is safe.
    pub fn open(path: &Path) -> Result<Index, Error> {
        Ok(Index {
            shared: index::Shared::open(path)?,
        })
    }

    /// The current generation, which a call keeps until it returns.
    pub(crate) fn inner(&self) -> arc_swap::Guard<Arc<index::Inner>> {
        self.shared.load()
    }

    /// An index of its own for the generation `inner` of the index at its path, which no other open shares.
    #[cfg(test)]
    pub(crate) fn of(inner: index::Inner) -> Index {
        let path = inner.path().to_path_buf();
        Index {
            shared: index::Shared::alone(&path, inner, None),
        }
    }

    /// Moves to the index's newest generation, which another process's writer may have committed since, and frees the
    /// generations that this process held before and no search holds now. Returns whether it moved. A search that
    /// runs meanwhile finishes on the generation it started on, and so does a [`Fetched`] made before. A writer of this
    /// process moves every open index of the path itself, and [`Index::prepare`] moves too.
    pub fn refresh(&self) -> Result<bool, Error> {
        self.shared.refresh()
    }

    /// Call this when the query text arrives. It moves to the index's newest generation, as `refresh` does, then starts
    /// reading the cluster tables, so that read overlaps the embedding model, and it makes `ctx`'s buffers for this
    /// index ahead of the search and readies its reads, eg. an io_uring ring on Linux. The buffer for the clusters is
    /// sized from the tables, so until a search has loaded them, the first search still makes that one. Once a search
    /// has gone through `ctx`'s reads of the index, it also does the setup that waits, so no search pays for it, eg. on
    /// Linux 5.13 and later it registers the buffers with the kernel where they fit the locked-memory limit. A failed
    /// move leaves the index on the generation it has.
    pub fn prepare(&self, ctx: &mut SearchCtx) -> Result<(), Error> {
        if self.shared.may_have_moved() {
            let _ = self.shared.refresh();
        }

        let inner = self.shared.load();
        inner.prepare();
        inner.make_buffers(ctx);
        Ok(())
    }

    /// Searches with the final query embedding. It picks the nearest clusters, reads them in one batch, scores
    /// their vectors, then reads and rescores the best candidates in a second small batch. On a warm index it scores
    /// the clusters in place instead of reading them. The hits live in `ctx`, so a query allocates nothing.
    pub fn search<'a>(&self, query: &[f32], params: &SearchParams, ctx: &'a mut SearchCtx) -> Result<&'a [Hit], Error> {
        let inner = self.shared.load();

        if inner.is_warm() {
            return inner.search_warm(query, params, ctx);
        }

        inner.check(query, params, &ctx.limits)?;

        let mut leaves = std::mem::take(&mut ctx.leaves);
        let slot = inner
            .route(query, params.nprobe, params.reach, ctx, &mut leaves)
            .and_then(|()| inner.fetch(&leaves, ctx));
        ctx.leaves = leaves;
        inner.search_fetched(query, params, ctx, slot?)
    }

    /// Writes the positions of the clusters `query` would read into `out`, nearest first. They are the `nprobe`
    /// nearest, or with a `reach`, those of them within it, as in [`SearchParams`]. The positions belong to the
    /// index's current generation, and `ctx` keeps which one, so a fetch with `ctx` after a commit fails.
    pub fn route(
        &self,
        query: &[f32],
        nprobe: usize,
        reach: Option<f32>,
        ctx: &mut SearchCtx,
        out: &mut Vec<LeafId>,
    ) -> Result<(), Error> {
        let inner = self.shared.load();
        inner.route(query, nprobe, reach, ctx, out)?;
        ctx.routed = Some((inner.cell(), inner.uid()));
        Ok(())
    }

    /// Starts reading `leaves` in one batch. On macOS and Linux it returns while the reads are still running. On other
    /// systems, and on Mac drives that refuse read hints, eg. FAT32 (File Allocation Table), it returns once they are
    /// done. The handle then ranks those clusters, and only those, for one or more queries, along with any vectors
    /// inserted since the last maintenance run, all of the generation that the index had at the fetch. [`Index::search`]
    /// is `route` with the same `nprobe` and `reach`, then `fetch`, then [`Fetched::search`]. On a warm index it gives
    /// the same hits, but scores the clusters in place. When `ctx`'s last route of this index was of an older
    /// generation, its positions would name other clusters, so the fetch fails with `InvalidInput`, and the caller
    /// routes again.
    pub fn fetch<'a>(&'a self, leaves: &[LeafId], ctx: &'a mut SearchCtx) -> Result<Fetched<'a>, Error> {
        let inner = self.shared.load_full();

        if ctx
            .routed
            .is_some_and(|(cell, uid)| cell == inner.cell() && uid != inner.uid())
        {
            return Err(search::invalid(
                "the index moved to a newer generation since the route, so route again",
            ));
        }

        let slot = inner.fetch(leaves, ctx)?;
        Ok(Fetched {
            index: self,
            inner,
            ctx,
            slot,
        })
    }

    /// Reads every cluster's blocks into the operating system's cache on a background thread, in file order, and
    /// returns at once. The thread runs at low priority on macOS and Linux. When it is done, the index is warm, and
    /// searches score the clusters in place. A clear while it runs stops it, and so does a failed read, which nothing
    /// reports. Either way the index stays cold. A call while a warm-up runs makes it run again if the index isn't warm
    /// when it ends, and a call once the index is warm does nothing. On Mac drives that refuse read hints, eg. FAT32,
    /// it can't fill the cache, so it stops at once.
    pub fn warm_up(&self) -> Result<(), Error> {
        self.shared.load().warm_up()
    }

    /// Whether every cluster's blocks are known to be in memory. A warm-up sets it, and so does `set_warm(true)`. A
    /// macOS memory-pressure warning clears it, and so does `set_warm(false)`.
    pub fn is_warm(&self) -> bool {
        self.shared.load().is_warm()
    }

    /// Sets whether the index is warm, eg. clears it when the app knows the cache was emptied. A warm index with blocks
    /// that aren't in memory still gives the same results, but its searches wait for each page.
    pub fn set_warm(&self, warm: bool) {
        self.shared.load().set_warm(warm);
    }

    /// Reads the whole file and checks every section against its checksum, then checks the cluster tables. A damaged
    /// file fails with `InvalidData`. Searches skip the checksums, since they read only part of the file, but they
    /// check the cluster tables when they first load them.
    pub fn verify(&self) -> Result<(), Error> {
        self.shared.load().verify()
    }
}

/// The whole index in memory, built from the same codes as an index file. Benchmarks compare against it, and a search
/// of a file built the same way returns exactly its results.
#[cfg(feature = "inmem")]
pub struct InMemoryIndex {
    inner: search::inmem::InMemory,
}

#[cfg(feature = "inmem")]
impl InMemoryIndex {
    /// Builds the index from `src` and an imported clustering. It uses `params.metric`, `params.order`,
    /// `params.anchor`, `params.page_size`, `params.seed` and `params.threads`.
    pub fn build(params: &BuildParams, src: &dyn VectorSource, clusters: &Clusters) -> Result<InMemoryIndex, Error> {
        Ok(InMemoryIndex {
            inner: search::inmem::InMemory::build(params, src, clusters)?,
        })
    }

    /// Searches like [`Index::search`]. The hits live in `ctx`.
    pub fn search<'a>(&self, query: &[f32], params: &SearchParams, ctx: &'a mut SearchCtx) -> Result<&'a [Hit], Error> {
        self.inner.search(query, params, ctx)
    }
}

/// Per-query settings. Reading more clusters and rescoring more candidates finds more of a query's true nearest
/// neighbors, but takes longer. [`calibrate`] picks the cheapest values that meet a [`Target`].
#[derive(Clone, Copy, Debug)]
pub struct SearchParams {
    /// How many results to return.
    pub k: usize,
    /// Clusters to read. At 1M vectors, 160 to 192 finds ~95% of each query's true 10 nearest neighbors.
    pub nprobe: usize,
    /// Candidates to rescore with the more precise copy of each vector. 29 to 39 gives that same ~95%.
    pub rerank: usize,
    /// When set, a search reads only the clusters whose distance from the query is at most (1 + reach) times the
    /// nearest one's, up to `nprobe` of them, so easy queries read fewer. `None` reads exactly `nprobe`.
    pub reach: Option<f32>,
}

/// One result, with the id given at build time and its score.
#[derive(Clone, Copy, Debug)]
pub struct Hit {
    pub id: u64,
    pub score: f32,
}

/// The largest request a `SearchCtx` must handle. It sizes its buffers from these.
#[derive(Clone, Copy, Debug)]
pub struct Limits {
    pub max_k: usize,
    /// Also the most clusters one `fetch` may read.
    pub max_nprobe: usize,
    /// Also the most candidates whose rows one `prefetch_rows` may read.
    pub max_rerank: usize,
}

/// Per-thread scratch space, reused across queries. It keeps one set of buffers per dimension, fitted to each index's
/// cluster count with room for the largest it has seen, and one set of read buffers per index. A search over several
/// collections also keeps a set of buffers for each of them, which later searches reuse for collections of the same
/// dimension. Each is made on the first search that needs it, so run one search with each index and each set of
/// collections before timing. For one index, [`Index::prepare`] makes its buffers and its read buffers ahead of its
/// search instead, apart from the buffer for the clusters, which waits until a search has loaded the index's cluster
/// tables. On Linux the read buffers of an index also hold an io_uring ring and a second handle on its file, two file
/// descriptors, and up to 16 MiB of copies of pages read twice. A context keeps them, and so keeps the file open, until
/// it drops, or until it makes the read buffers of another index after this one has closed.
pub struct SearchCtx {
    limits: Limits,
    scratch: Vec<search::Scratch>,
    reads: Vec<search::cold::Reads>,
    /// The clusters that a search routes to.
    leaves: Vec<LeafId>,
    /// The cell and the generation of the index that the last `Index::route` routed, so a fetch of another generation
    /// fails.
    routed: Option<(u64, u64)>,
    hits: Vec<Hit>,
    /// Each collection's part of a search over several, and the routes and candidates of all of them.
    parts: Vec<multi::Part>,
    /// The generations of the collections that a search over several loaded, kept only while it runs.
    loaded: Vec<Arc<index::Inner>>,
    picks: Vec<(search::candidates::Candidate, usize)>,
    /// Where the last search's time went, filled in only with the `bench` feature.
    trace: trace::Trace,
}

impl SearchCtx {
    pub fn new(limits: Limits) -> SearchCtx {
        SearchCtx {
            limits,
            scratch: Vec::new(),
            reads: Vec::new(),
            leaves: Vec::with_capacity(limits.max_nprobe),
            routed: None,
            hits: Vec::with_capacity(limits.max_k),
            parts: Vec::new(),
            loaded: Vec::new(),
            picks: Vec::new(),
            trace: trace::Trace::default(),
        }
    }

    /// Where the last search on this context spent its time, and what it read.
    #[cfg(feature = "bench")]
    pub fn trace(&self) -> Trace {
        self.trace
    }
}

/// The read backends that a benchmark can choose instead of the one `Index::open` picks.
#[cfg(feature = "bench")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    /// The one `Index::open` picks.
    Best,
    /// Up to 16 threads of positioned reads, whose submit waits for every read.
    Pool,
    /// One positioned read at a time, in order.
    Portable,
}

#[cfg(feature = "bench")]
impl Index {
    /// Opens `path` like [`Index::open`], but reads with `backend`.
    pub fn open_with(path: &Path, backend: Backend) -> Result<Index, Error> {
        let inner = index::Inner::open(path)?;
        let wave: Option<io::NewWave> = match backend {
            Backend::Best => None,
            Backend::Pool => Some(|| Box::<io::pool::PoolWave>::default()),
            Backend::Portable => Some(|| Box::<io::portable::PortableWave>::default()),
        };
        let inner = match wave {
            Some(wave) => inner.with_wave(wave),
            None => inner,
        };
        let path = update::real_path(path);
        Ok(Index {
            shared: index::Shared::alone(&path, inner, wave),
        })
    }

    /// Searches through the memory map instead of read waves, which leaves every read to a page fault. That includes
    /// the cluster tables, unless a search has loaded them already, so a cold control skips [`Index::prepare`], which
    /// starts reading them. It gives the same hits as [`Index::search`].
    pub fn search_mapped<'a>(
        &self,
        query: &[f32],
        params: &SearchParams,
        ctx: &'a mut SearchCtx,
    ) -> Result<&'a [Hit], Error> {
        self.shared.load().search_mapped(query, params, ctx)
    }
}

/// A cluster's position in the index file, as returned by [`Index::route`]. It is the cluster's row in [`Clusters`]
/// only with [`LeafOrder::Id`].
pub type LeafId = u32;

/// Clusters being read in one batch. Rank them for any number of queries. Nothing outside them is ever read, though the
/// vectors inserted since the last maintenance run are ranked with them. It keeps the generation that the index had at
/// the fetch, so a commit meanwhile changes nothing it ranks. Use it to read once and rank several versions of a query,
/// or to start reading from a guessed query before the final one is known. Dropping it cancels or finishes the pending
/// reads.
pub struct Fetched<'a> {
    index: &'a Index,
    inner: Arc<index::Inner>,
    ctx: &'a mut SearchCtx,
    /// Where the index's reads sit in `ctx`.
    slot: usize,
}

impl<'a> Fetched<'a> {
    /// Starts reading the precise copies of the `n` best candidates for `query`. It ranks candidates by their quick
    /// 1-bit estimate, and waits for the clusters to arrive first.
    pub fn prefetch_rows(&mut self, query: &[f32], n: usize) -> Result<(), Error> {
        self.inner.prefetch_rows(query, n, self.ctx, self.slot)
    }

    /// Ranks the fetched clusters for `query`. It scores every vector, reads any precise copies not read yet,
    /// rescores the best `rerank` candidates and returns the top `k`.
    pub fn search(self, query: &[f32], k: usize, rerank: usize) -> Result<&'a [Hit], Error> {
        let params = SearchParams {
            k,
            nprobe: 0,
            rerank,
            reach: None,
        };
        self.inner.search_fetched(query, &params, self.ctx, self.slot)
    }
}

/// Searches several collection files together in a few batched reads, instead of one cold query per file.
pub struct Collections {
    indexes: Vec<Index>,
}

impl Collections {
    /// Opens every file of `paths`. It first asks the operating system to start reading every header at once, which
    /// macOS and Linux do. The collections must share a dimension and a metric.
    pub fn open(paths: &[&Path]) -> Result<Collections, Error> {
        Ok(Collections {
            indexes: multi::open(paths)?,
        })
    }

    /// Writes the top `k` across all collections into `out`, as (collection index, hit), nearest first. `nprobe`,
    /// `rerank` and `reach` count across all of them. The nearest clusters of any collection are read in one batch,
    /// then the rows of the best candidates of any collection in one more. On macOS each batch reads every collection
    /// at once. Elsewhere, and on Mac drives that refuse read hints, each collection's reads run in turn.
    pub fn search(
        &self,
        query: &[f32],
        params: &SearchParams,
        ctx: &mut SearchCtx,
        out: &mut Vec<(usize, Hit)>,
    ) -> Result<(), Error> {
        multi::search(&self.indexes, query, params, ctx, out)
    }
}

/// Inserts and deletes for one collection. They go to a crash-safe log next to the file. The next search of any open
/// index of the file in this process sees them, and so does any index opened later, in any process. `maintain` folds
/// them into a new file and clusters again what drifted, and `compact` only folds them.
pub struct Writer {
    path: PathBuf,
    index: Index,
    log: update::Log,
    /// The writer's lock, the root and the current generation.
    root: root::Owner,
    /// For routing each insert to its nearest cluster.
    ctx: SearchCtx,
}

impl Writer {
    /// Opens the index at `path` for changes, with its log and its lock file, eg. `index.coldivf.delta` and
    /// `index.coldivf.lock` next to it, or the log in its folder `index.coldivf.d` once maintenance has run. Only one
    /// writer may hold an index at a time, in any process, and a second one fails with `WouldBlock`. A path through a
    /// symbolic link names the file it leads to. The open deletes what crashes and failed runs left in the folder.
    pub fn open(path: &Path) -> Result<Writer, Error> {
        // The lock comes first, so no other writer can commit between the open and the log's replay.
        let path = update::real_path(path);
        let root = root::Owner::open(&path)?;
        let index = Index::open(&path)?;
        let inner = index.inner();
        let at = inner.located();

        if at != root.current() {
            return Err(Error::other("the index changed while the writer opened it"));
        }

        let log = update::Log::open(&at.log, at.version(), inner.layout(), inner.changes(), root.lock())?;
        drop(inner);
        root.clean();
        let limits = Limits {
            max_k: 1,
            max_nprobe: 1,
            max_rerank: 1,
        };
        Ok(Writer {
            path,
            index,
            log,
            root,
            ctx: SearchCtx::new(limits),
        })
    }

    /// Adds `vector` with the id `id`, replacing any earlier vector with that id. With the cluster anchor its code is
    /// measured from the center of its nearest cluster, so it scores as it would in the file. Until a compaction, each
    /// search then also measures the query's distance to the center of every cluster that holds inserts, and each of
    /// those costs about as much as scanning 10 inserts. Every open index of the file in this process sees it from its
    /// next search, and indexes opened later read it from the log. It survives the process dying, and
    /// [`Writer::sync`] makes it survive a power loss too.
    pub fn insert(&mut self, id: u64, vector: &[f32]) -> Result<(), Error> {
        self.index.shared.reclaim();
        let inner = self.index.inner();
        let (leaf, code, factors, row) = inner.encode_insert(id, vector, &mut self.ctx)?;
        update::insert(&mut self.log, inner.changes(), id, leaf, &code, factors, &row)
    }

    /// Removes the vector with the id `id`, if there is one, the same way. Until a compaction, a deleted vector still
    /// takes a place among the candidates that a search rescores.
    pub fn delete(&mut self, id: u64) -> Result<(), Error> {
        self.index.shared.reclaim();
        update::delete(&mut self.log, self.index.inner().changes(), id)
    }

    /// Makes every change so far survive a power loss.
    pub fn sync(&mut self) -> Result<(), Error> {
        self.log.sync()
    }

    /// Rewrites the index file with every change folded in, then starts the log again. It keeps every cluster and its
    /// center, so it copies every vector's codes. It is [`Writer::maintain`] with the default settings and
    /// [`RunKind::Fold`].
    pub fn compact(&mut self) -> Result<(), Error> {
        self.maintain(&MaintainParams::default(), RunKind::Fold).map(|_| ())
    }

    /// Runs one maintenance run of `kind` now, with `params`. It folds every change into a new data file, clustering
    /// again what `kind` says, and commits it as the index's next generation, so it blocks the writer until the commit.
    /// The first run makes the folder `name.d` next to the index file, which then holds every generation's files and
    /// the root that names the current one, and deletes the index file and its log. It copies every vector's 8-bit
    /// copy, and the codes of every vector that keeps its cluster's center. With the cluster anchor it measures every
    /// other code again from its vector's 8-bit copy, since the file keeps no other. Its memory doesn't grow with the
    /// vector count, apart from a few bytes per vector and the cluster centers, eg. at most a few hundred MB for 1M
    /// vectors of 768 dimensions. Searches and opens of the index go on while it runs. At the commit every open index
    /// of the path in this process moves to the new generation, and one in another process does at its next
    /// [`Index::refresh`] or [`Index::prepare`]. A damaged trailer, which holds only the regions and the counts that say
    /// when runs are due, is made again as for a fresh build. A run that fails before its commit leaves the index as it
    /// was. One that fails while it writes the root leaves unknown whether it committed, so the writer then refuses
    /// changes until it is opened again, which finds the generation that is current.
    pub fn maintain(&mut self, params: &MaintainParams, kind: RunKind) -> Result<RunReport, Error> {
        self.log.check()?;
        let old = self.index.shared.load_full();
        let (next, file) = self.root.next()?;
        let (snapshot, report) = maintain::run(&old, &file, &next.data, params, kind)?;
        drop(file);
        let (new, log) = maintain::commit(&mut self.root, next, &self.path, old.layout(), &snapshot, &mut self.log)?;
        // A refresh of this process may have moved the cell to the new generation first, which is as good.
        self.index.shared.publish(&old, new);
        self.log = log;
        drop(old);
        self.index.shared.reclaim();
        Ok(report)
    }

    /// What maintenance with `params` would do now. It reads the file's ids when a change hid any vector.
    pub fn status(&self, params: &MaintainParams) -> Result<Status, Error> {
        maintain::status(&self.index.inner(), params)
    }
}

/// Maintenance settings, i.e. every threshold that decides what a maintenance run does.
#[derive(Clone, Debug)]
pub struct MaintainParams {
    /// Average vectors per cluster that clustering again aims for, as in `BuildParams`, ~160.
    pub vectors_per_list: usize,
    pub minor: MinorParams,
    pub major: MajorParams,
    /// Rounds of every k-means.
    pub rounds: usize,
    /// Groups of clusters that order the file, 64 to 128, found again by every run that clusters. 0 keeps the old
    /// file's number.
    pub parents: usize,
    pub order: LeafOrder,
    /// Threads for k-means and encoding. 0 takes `RAYON_NUM_THREADS` when it is set, and otherwise 2. The file is the
    /// same for any number.
    pub threads: usize,
    /// Seeds every k-means of a run. 0 takes the file's seed plus its epoch, so each run of a file draws other starting
    /// centers, and a repeat of the same run draws the same ones.
    pub seed: u64,
}

impl Default for MaintainParams {
    fn default() -> MaintainParams {
        MaintainParams {
            vectors_per_list: 160,
            minor: MinorParams::default(),
            major: MajorParams::default(),
            rounds: 20,
            parents: 0,
            order: LeafOrder::Chained,
            threads: 0,
            seed: 0,
        }
    }
}

/// When a minor run clusters a region again, and how. A region is a set of nearby clusters, ~80 at 1M vectors, that a
/// minor run clusters again together. Regions are made from the clusters' centers at the first run, and they last
/// across runs.
#[derive(Clone, Debug)]
pub struct MinorParams {
    /// A region is due once its inserts plus deletes since it was last clustered reach this share of its count then.
    pub change: f32,
    /// Or once its count falls to 1 − `shrink` of its count then.
    pub shrink: f32,
    /// Or once one of its clusters holds over this many times the mean cluster size.
    pub hot_list: f32,
    /// Or once one of its clusters has lost this share of its vectors since the region was last clustered.
    pub list_loss: f32,
    /// Clusters per region when regions are made, 0 for the square root of the cluster count.
    pub region_lists: usize,
    /// A region splits in two once it holds over `split` times the clusters per region it was made with.
    pub split: f32,
    /// A region joins its nearest region once it holds under `merge` times that many.
    pub merge: f32,
    /// Once the due regions are clustered again, the vectors of the `ring` clusters outside its region nearest each
    /// new cluster move to their nearest new cluster when it is nearer than their own.
    pub ring: usize,
    /// When set, the inserts farther from their cluster's center than this share of the file's vectors are from theirs
    /// are clustered on their own in a new region, once there are `vectors_per_list` of them.
    pub pool: Option<f32>,
    /// When set, a minor run also moves every vector to its nearest cluster and every center to its vectors' mean, once
    /// the inserts plus deletes since the last major run or such pass reach this share of the count at the last major
    /// run. It measures every code again.
    pub global_pass: Option<f32>,
    /// Moves the center of every cluster that gained or lost vectors to its vectors' mean, which measures every code of
    /// the cluster again.
    pub refresh: bool,
}

impl Default for MinorParams {
    fn default() -> MinorParams {
        MinorParams {
            change: 0.10,
            // TODO: Confirm the values below on replayed streams of inserts, deletes and upserts.
            shrink: 0.20,
            hot_list: 8.0,
            list_loss: 0.5,
            region_lists: 0,
            split: 2.0,
            merge: 0.4,
            ring: 16,
            pool: None,
            global_pass: None,
            refresh: false,
        }
    }
}

/// When a major run clusters every vector again, and how. `None` turns a trigger off.
#[derive(Clone, Debug)]
pub struct MajorParams {
    /// Once the inserts since the last major run reach this share of the vector count then. An insert that replaces a
    /// vector counts as one insert, so without deletes this is the growth since the last major run.
    pub inserts: Option<f32>,
    /// Once the deletes since the last major run reach this share of the vector count then.
    pub deletes: Option<f32>,
    /// Once the read-cost indicator stays above this for two minor runs in a row. The indicator divides the bytes a
    /// query reads for 95% recall by what a fresh clustering of as many vectors would read. Runs don't measure it, so
    /// this never fires.
    pub indicator: Option<f32>,
    /// The indicator's growth exponent β, where a fresh clustering's bytes per query grow as the vector count to the
    /// power β.
    pub beta: f32,
    /// Vectors per cluster that its k-means trains on, 0 for all of them. Every vector is assigned afterwards. With 0
    /// each round reads every vector from the file again. A sample is held in memory as 8-bit copies instead, about the
    /// dimension plus 16 bytes per vector.
    pub sample: usize,
}

impl Default for MajorParams {
    fn default() -> MajorParams {
        MajorParams {
            inserts: Some(0.25),
            // TODO: Confirm on replayed streams with deletes and upserts.
            deletes: None,
            indicator: None,
            beta: 0.30,
            sample: 0,
        }
    }
}

/// Which run [`Writer::maintain`] does.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RunKind {
    /// A major run when its trigger holds, else a minor run when a region is due, else a fold.
    Auto,
    /// Folds the changes in and clusters nothing again. This is the default.
    #[default]
    Fold,
    /// Clusters again the regions that are due. With none due, no far inserts to pool and no global pass or refresh to
    /// do, it folds only.
    Minor,
    /// Clusters every vector again.
    Major,
}

/// What a maintenance run did.
#[derive(Clone, Debug, Default)]
pub struct RunReport {
    /// The run it did. `Auto` becomes `Major`, `Minor` or `Fold`, and so does a minor run with nothing to do.
    pub kind: RunKind,
    pub regions_due: usize,
    /// Vectors in the regions it clustered again, before the boundary fix.
    pub reclustered: usize,
    /// Vectors that changed cluster. Every vector of a cluster it made counts.
    pub moved: usize,
    /// Codes measured again from their vectors' 8-bit copies, and codes copied.
    pub reencoded: usize,
    pub copied: usize,
    /// Vectors, clusters and regions in the new file.
    pub count: usize,
    pub lists: usize,
    pub regions: usize,
    /// Vector-to-center distances that its k-means and boundary fix computed.
    pub distances: u64,
    pub bytes_read: u64,
    pub bytes_written: u64,
    /// Processor time of the run's threads, and the time it took.
    pub cpu: Duration,
    pub wall: Duration,
}

/// What maintenance would do now.
#[derive(Clone, Debug, Default)]
pub struct Status {
    /// Inserts waiting in the log, and vectors of the file that a change hid.
    pub pending: usize,
    pub deleted: usize,
    pub regions_due: usize,
    pub major_due: bool,
}

/// The accuracy setting. It is the share of each query's true 10 nearest neighbors that a search must find, eg.
/// finding 9 of the true 10 is 0.9.
#[derive(Clone, Copy, Debug)]
pub enum Target {
    /// On average over queries, at least this share of the true 10 nearest neighbors is found. 0.95 means 9.5 of 10.
    MeanRecall { recall_at_10: f32 },
    /// At least `share` of queries each find at least `recall_at_10` of their true 10 nearest neighbors, eg. 95% of
    /// queries find at least 9 of 10.
    QueryShare { recall_at_10: f32, share: f32 },
}

/// The search settings that `calibrate` picks. Use them in [`SearchParams`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Calibration {
    pub nprobe: usize,
    pub rerank: usize,
    pub reach: Option<f32>,
}

/// The data file that holds the index at `path` now. It is `path` itself until the index's first maintenance run, and
/// a file in the index's folder `name.d` from then on.
pub fn data_file(path: &Path) -> Result<PathBuf, Error> {
    Ok(root::locate(&update::real_path(path))?.data)
}

/// Picks the cheapest `nprobe`, `rerank` and `reach` that meet `target` with 90% confidence. It measures them on
/// sample `queries` (row-major, not part of the index), half of which order the settings and half of which test them.
/// `truth` holds each query's true 10 nearest neighbors, from an exact search. A setting's cost is the bytes that a
/// cold search with it reads, counting each rescored row as one page of the read backend, eg. 16 KiB on macOS. A
/// setting with a reach wins only when the cheapest fixed one costs at least 1.1 times as much. It tries up to 4096
/// clusters and 2048 rescored candidates, so a [`SearchCtx`] with those limits runs any setting it picks. A target
/// that no setting meets with that confidence fails with `InvalidInput`, eg. with too few queries for a strict one.
pub fn calibrate(index: &Index, queries: &[f32], truth: &[u32], target: Target) -> Result<Calibration, Error> {
    search::calibrate::calibrate(&index.inner(), queries, truth, target)
}
