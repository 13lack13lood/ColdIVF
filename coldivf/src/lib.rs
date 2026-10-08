#![allow(unused_variables, dead_code, clippy::ptr_arg)]

mod build;
mod format;
mod index;
mod io;
mod kernels;
mod multi;
mod quantization;
mod search;
mod update;

use std::io::Error;
use std::path::{Path, PathBuf};
use std::sync::Arc;

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
    /// Page size for packing the precise vector copies that rescoring reads. 4096 lets one file work on both
    /// 16 KiB-page Macs and 4 KiB-page PCs.
    pub page_size: usize,
    /// Threads for clustering and encoding. 0 takes `RAYON_NUM_THREADS` when it is set, and otherwise one thread per
    /// logical CPU. The file is the same for any number.
    pub threads: usize,
    pub seed: u64,
}

/// A clustering computed elsewhere (eg. with faiss), used instead of the built-in k-means.
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

pub struct Builder {
    params: BuildParams,
}

impl Builder {
    pub fn new(params: BuildParams) -> Builder {
        Builder { params }
    }

    /// Clusters, orders, encodes and writes `src` to `out`. k-means finds clusters of about `vectors_per_list` vectors,
    /// and more k-means over their centers finds the groups. It writes a temporary file, syncs it, then renames it.
    pub fn build(&self, src: &dyn VectorSource, out: &Path) -> Result<(), Error> {
        build::build(&self.params, src, out)
    }

    /// The same, with an imported clustering.
    pub fn build_with_clusters(&self, src: &dyn VectorSource, clusters: &Clusters, out: &Path) -> Result<(), Error> {
        build::build_with_clusters(&self.params, src, clusters, out)
    }
}

/// An open collection, stored as one file. It is cheap to clone and safe to share across threads.
#[derive(Clone)]
pub struct Index {
    inner: Arc<index::Inner>,
}

impl Index {
    /// Reads the header and the rotation, the first 16 KiB up to 2806 dimensions, and maps the file without loading it.
    /// It also reads the file's change log, if it has one, which takes time and memory in proportion to the inserts
    /// since the last compaction. Nothing else is read until a query. Don't rewrite the file in place while it is open,
    /// eg. by copying over it. The index reads it a little at a time, so it would pair the old cluster tables with the
    /// new vectors. A rebuild that renames a new file over it is safe.
    pub fn open(path: &Path) -> Result<Index, Error> {
        Ok(Index {
            inner: Arc::new(index::Inner::open(path)?),
        })
    }

    /// Call this when the query text arrives. It starts reading the cluster centers, so that read overlaps the
    /// embedding model. Once a search has loaded them, it does nothing.
    pub fn prepare(&self, ctx: &mut SearchCtx) -> Result<(), Error> {
        self.inner.prepare();
        Ok(())
    }

    /// Searches with the final query embedding. It picks the nearest clusters, reads them in one batch, scores
    /// their vectors, then reads and rescores the best candidates in a second small batch. On a warm index it scores
    /// the clusters in place instead of reading them. The hits live in `ctx`, so a query allocates nothing.
    pub fn search<'a>(&self, query: &[f32], params: &SearchParams, ctx: &'a mut SearchCtx) -> Result<&'a [Hit], Error> {
        if self.inner.is_warm() {
            return self.inner.search_warm(query, params, ctx);
        }

        self.inner.check(query, params, &ctx.limits)?;

        let mut leaves = std::mem::take(&mut ctx.leaves);
        let slot = self
            .inner
            .route(query, params.nprobe, params.reach, ctx, &mut leaves)
            .and_then(|()| self.inner.fetch(&leaves, ctx));
        ctx.leaves = leaves;
        self.inner.search_fetched(query, params, ctx, slot?)
    }

    /// Writes the positions of the clusters `query` would read into `out`, nearest first. They are the `nprobe`
    /// nearest, or with a `reach`, those of them within it, as in [`SearchParams`].
    pub fn route(
        &self,
        query: &[f32],
        nprobe: usize,
        reach: Option<f32>,
        ctx: &mut SearchCtx,
        out: &mut Vec<LeafId>,
    ) -> Result<(), Error> {
        self.inner.route(query, nprobe, reach, ctx, out)
    }

    /// Starts reading `leaves` in one batch. On macOS it returns while the reads are still running. On other systems,
    /// and on Mac drives that refuse read hints, eg. FAT32 (File Allocation Table), it returns once they are done. The
    /// handle then ranks those clusters, and only those, for one or more queries, along with any vectors inserted since
    /// the last compaction. [`Index::search`] is `route` with the same `nprobe` and `reach`, then `fetch`, then
    /// [`Fetched::search`]. On a warm index it gives the same hits, but scores the clusters in place.
    pub fn fetch<'a>(&'a self, leaves: &[LeafId], ctx: &'a mut SearchCtx) -> Result<Fetched<'a>, Error> {
        let slot = self.inner.fetch(leaves, ctx)?;
        Ok(Fetched { index: self, ctx, slot })
    }

    /// Reads every cluster's blocks into the operating system's cache on a background thread, in file order, and
    /// returns at once. The thread runs at low priority on macOS and Linux. When it is done, the index is warm, and
    /// searches score the clusters in place. A clear while it runs stops it, and so does a failed read, which nothing
    /// reports. Either way the index stays cold. A call while a warm-up runs makes it run again if the index isn't warm
    /// when it ends, and a call once the index is warm does nothing. On Mac drives that refuse read hints, eg. FAT32,
    /// it can't fill the cache, so it stops at once.
    pub fn warm_up(&self) -> Result<(), Error> {
        self.inner.warm_up()
    }

    /// Whether every cluster's blocks are known to be in memory. A warm-up sets it, and so does `set_warm(true)`. A
    /// macOS memory-pressure warning clears it, and so does `set_warm(false)`.
    pub fn is_warm(&self) -> bool {
        self.inner.is_warm()
    }

    /// Sets whether the index is warm, eg. clears it when the app knows the cache was emptied. A warm index with blocks
    /// that aren't in memory still gives the same results, but its searches wait for each page.
    pub fn set_warm(&self, warm: bool) {
        self.inner.set_warm(warm);
    }

    /// Reads the whole file and checks every section against its checksum, then checks the cluster tables. A damaged
    /// file fails with `InvalidData`. Searches skip the checksums, since they read only part of the file, but they
    /// check the cluster tables when they first load them.
    pub fn verify(&self) -> Result<(), Error> {
        self.inner.verify()
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
    /// `params.page_size`, `params.seed` and `params.threads`.
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

/// Per-thread scratch space, reused across queries. It keeps one set of buffers per index shape, i.e. per dimension and
/// cluster count, and one set of read buffers per index. A search over several collections also keeps a set of
/// buffers for each of them, which later searches reuse for collections of the same shape. Each is made on the first
/// search that needs it, so run one search with each index and each set of collections before timing.
pub struct SearchCtx {
    limits: Limits,
    scratch: Vec<search::Scratch>,
    reads: Vec<search::cold::Reads>,
    /// The clusters that a search routes to.
    leaves: Vec<LeafId>,
    hits: Vec<Hit>,
    /// Each collection's part of a search over several, and the routes and candidates of all of them.
    parts: Vec<multi::Part>,
    picks: Vec<(search::candidates::Candidate, usize)>,
}

impl SearchCtx {
    pub fn new(limits: Limits) -> SearchCtx {
        SearchCtx {
            limits,
            scratch: Vec::new(),
            reads: Vec::new(),
            leaves: Vec::with_capacity(limits.max_nprobe),
            hits: Vec::with_capacity(limits.max_k),
            parts: Vec::new(),
            picks: Vec::new(),
        }
    }
}

/// A cluster's position in the index file, as returned by [`Index::route`]. It is the cluster's row in [`Clusters`]
/// only with [`LeafOrder::Id`].
pub type LeafId = u32;

/// Clusters being read in one batch. Rank them for any number of queries. Nothing outside them is ever read, though the
/// vectors inserted since the last compaction are ranked with them. Use it to read once and rank several versions of a
/// query, or to start reading from a guessed query before the final one is known. Dropping it cancels or finishes the
/// pending reads.
pub struct Fetched<'a> {
    index: &'a Index,
    ctx: &'a mut SearchCtx,
    /// Where the index's reads sit in `ctx`.
    slot: usize,
}

impl<'a> Fetched<'a> {
    /// Starts reading the precise copies of the `n` best candidates for `query`. It ranks candidates by their quick
    /// 1-bit estimate, and waits for the clusters to arrive first.
    pub fn prefetch_rows(&mut self, query: &[f32], n: usize) -> Result<(), Error> {
        self.index.inner.prefetch_rows(query, n, self.ctx, self.slot)
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
        self.index.inner.search_fetched(query, &params, self.ctx, self.slot)
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
/// index of the file in this process sees them, and so does any index opened later, in any process. `compact` folds
/// them into a new file.
pub struct Writer {
    path: PathBuf,
    index: Index,
    log: update::Log,
    /// For routing each insert to its nearest cluster.
    ctx: SearchCtx,
}

impl Writer {
    /// Opens the index at `path` for changes, with its log and its lock file next to it, eg. `index.coldivf.delta` and
    /// `index.coldivf.lock`. Only one writer may hold an index at a time, in any process, and a second one fails with
    /// `WouldBlock`. A path through a symbolic link names the file it leads to.
    pub fn open(path: &Path) -> Result<Writer, Error> {
        // The lock comes first, so no other writer can compact the file between the open and the log's replay.
        let path = update::real_path(path);
        let lock = update::lock(&path)?;
        let index = Index::open(&path)?;
        let log = update::Log::open(&path, index.inner.layout(), index.inner.changes(), lock)?;
        let limits = Limits {
            max_k: 1,
            max_nprobe: 1,
            max_rerank: 1,
        };
        Ok(Writer {
            path,
            index,
            log,
            ctx: SearchCtx::new(limits),
        })
    }

    /// Adds `vector` with the id `id`, replacing any earlier vector with that id. Every open index of the file in
    /// this process sees it from its next search, and indexes opened later read it from the log. It survives the
    /// process dying, and [`Writer::sync`] makes it survive a power loss too.
    pub fn insert(&mut self, id: u64, vector: &[f32]) -> Result<(), Error> {
        let inner = &self.index.inner;
        let (leaf, code, factors, row) = inner.encode_insert(id, vector, &mut self.ctx)?;
        update::insert(&mut self.log, inner.changes(), id, leaf, &code, factors, &row)
    }

    /// Removes the vector with the id `id`, if there is one, the same way. Until a compaction, a deleted vector still
    /// takes a place among the candidates that a search rescores.
    pub fn delete(&mut self, id: u64) -> Result<(), Error> {
        update::delete(&mut self.log, self.index.inner.changes(), id)
    }

    /// Makes every change so far survive a power loss.
    pub fn sync(&mut self) -> Result<(), Error> {
        self.log.sync()
    }

    /// Rebuilds the index file with every change folded in, then empties the log. It keeps the clusters and decodes
    /// every vector from its 8-bit copy, so it holds them all in memory as f32 while it builds the new index there
    /// too, eg. about 4.2 GB at its peak for 1M vectors of 768 dimensions. Indexes opened before keep their view until
    /// they open the file again.
    pub fn compact(&mut self) -> Result<(), Error> {
        let compacted = self.index.inner.compacted()?;
        build::write(&compacted, &self.path)?;
        // Until the log starts again, it names the old file, so a crash here loses nothing and replays nothing twice.
        let index = Index::open(&self.path)?;
        self.log.restart(index.inner.layout(), index.inner.changes())?;
        self.index = index;
        Ok(())
    }
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

/// Picks the cheapest `nprobe`, `rerank` and `reach` that meet `target` with 90% confidence. It measures them on
/// sample `queries` (row-major, not part of the index), half of which order the settings and half of which test them.
/// `truth` holds each query's true 10 nearest neighbors, from an exact search. A setting's cost is the bytes that a
/// cold search with it reads, counting each rescored row as one page of the read backend, eg. 16 KiB on macOS. A
/// setting with a reach wins only when the cheapest fixed one costs at least 1.1 times as much. A target that no
/// setting meets with that confidence fails with `InvalidInput`, eg. with too few queries for a strict one.
pub fn calibrate(index: &Index, queries: &[f32], truth: &[u32], target: Target) -> Result<Calibration, Error> {
    search::calibrate::calibrate(&index.inner, queries, truth, target)
}
