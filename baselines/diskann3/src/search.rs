//! Opening a built index with one of the readers, and searching it.
//!
//! An open loads the product quantization (PQ) data, builds the reader's factory and creates DiskANN's searcher, which
//! reads the index header. A search then walks the graph a beam at a time and reranks with the full vectors it read.

use std::io::{Error, ErrorKind};
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Instant;

use diskann_disk::data_model::{AdHoc, CachingStrategy};
use diskann_disk::search::provider::aligned_file_reader::traits::AlignedReaderFactory;
use diskann_disk::search::provider::disk_provider::{DiskIndexSearcher, SearchResult};
use diskann_disk::search::provider::disk_vertex_provider_factory::DiskVertexProviderFactory;
use diskann_disk::search::search_mode::SearchMode;
use diskann_disk::storage::disk_index_reader::DiskIndexReader;
use diskann_providers::storage::{FileStorageProvider, get_compressed_pq_file, get_disk_index_file, get_pq_pivot_file};
use diskann_vector::distance::Metric;
use tokio::runtime::Runtime;

use crate::mac_reader::{AdviseFactory, Counters, PoolFactory, ReadPool, StockFactory, Tally};

/// Indexes hold 32-bit float vectors with no data beside them.
pub type Data = AdHoc<f32>;

/// The most nodes one search may read. DiskANN stops a search early past it, so it is set out of reach.
const IO_LIMIT: usize = u32::MAX as usize;

/// The stock reader serves only as a control that loads the index at open, of at most this many vectors at one
/// thread.
pub const STOCK_MAX_POINTS: usize = 1_000_000;

/// Which reader serves the sector reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reader {
    /// The sectors of a beam at once on the read pool, through a handle that skips the page cache.
    Pool,
    /// `F_RDADVISE` for every sector of a beam, then a read of each, through the same kind of handle.
    Advise,
    /// DiskANN's own macOS reader, which copies the whole index file into memory at open.
    Stock,
}

impl Reader {
    /// The reader with code 0, 1 or 2, in the order above.
    pub fn from_code(code: u32) -> Option<Reader> {
        match code {
            0 => Some(Reader::Pool),
            1 => Some(Reader::Advise),
            2 => Some(Reader::Stock),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Reader::Pool => "pool",
            Reader::Advise => "advise",
            Reader::Stock => "stock",
        }
    }
}

/// How to open an index.
#[derive(Clone, Copy, Debug)]
pub struct Options {
    pub reader: Reader,
    pub metric: Metric,
    /// The threads DiskANN may plan its searcher for.
    pub threads: usize,
    /// The nodes nearest the graph's entry point to keep in memory, found by a breadth-first walk at open. 0 keeps
    /// none.
    pub cache_nodes: usize,
}

/// How long each step of an open took, in microseconds, and what it read of the index file.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct OpenStats {
    /// Reading the PQ codebooks and codes.
    pub pq_us: f64,
    /// Building the reader's factory, which opens the index file, or copies it for the stock reader, and caches the
    /// nodes asked for.
    pub reader_us: f64,
    /// Creating the searcher, which reads the header.
    pub searcher_us: f64,
    /// The bytes of the index file read through the reader, plus the whole file for the stock reader. The PQ files
    /// aren't counted.
    pub bytes: u64,
}

/// What one search did.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Stats {
    /// Nodes expanded, nodes loaded, distance computations and results found. The reader's calls count the beams.
    pub hops: u32,
    pub ios: u32,
    pub cmps: u32,
    pub results: u32,
    /// DiskANN's own times in microseconds, for loading nodes, for the query's PQ table and for the whole search.
    pub io_us: f64,
    pub pq_us: f64,
    pub total_us: f64,
    /// What the reader read.
    pub reads: Tally,
}

/// The ids and distances of a search, nearest first, with what it did.
#[derive(Clone, Debug, Default)]
pub struct Found {
    pub ids: Vec<u32>,
    pub distances: Vec<f32>,
    pub stats: Stats,
}

/// DiskANN's searcher over the readers of factory `F`.
type DiskSearcher<F> = DiskIndexSearcher<Data, DiskVertexProviderFactory<Data, F>>;

enum Searcher {
    Pool(DiskSearcher<PoolFactory>),
    Advise(DiskSearcher<AdviseFactory>),
    Stock(DiskSearcher<StockFactory>),
}

/// An open index.
pub struct Index {
    searcher: Searcher,
    counters: Arc<Counters>,
    /// The pool reader's prefill switch, which the other readers don't have.
    prefill: Option<Arc<AtomicBool>>,
    dim: usize,
    points: usize,
    pub reader: Reader,
    pub opened: OpenStats,
}

/// A runtime made ahead of an open, so the open itself doesn't make one.
static NEXT_RUNTIME: Mutex<Option<Runtime>> = Mutex::new(None);

/// Makes the single-thread runtime the next open takes.
pub fn prepare_runtime() -> Result<(), Error> {
    let runtime = tokio::runtime::Builder::new_current_thread().build()?;
    *NEXT_RUNTIME.lock().unwrap_or_else(PoisonError::into_inner) = Some(runtime);
    Ok(())
}

fn take_runtime() -> Option<Runtime> {
    NEXT_RUNTIME.lock().unwrap_or_else(PoisonError::into_inner).take()
}

impl Index {
    /// Opens the index whose files start with `prefix`, eg. "<folder>/index". The pool reader reads on `pool`.
    pub fn open(prefix: &str, options: &Options, pool: &Arc<ReadPool>) -> Result<Index, Error> {
        let (pivots, codes, disk) = (
            get_pq_pivot_file(prefix),
            get_compressed_pq_file(prefix),
            get_disk_index_file(prefix),
        );
        let counters = Arc::new(Counters::default());
        let t0 = Instant::now();
        let pq = DiskIndexReader::new(pivots, codes, &FileStorageProvider).map_err(crate::ann)?;
        let t1 = Instant::now();

        if options.reader == Reader::Stock && (pq.get_num_points() > STOCK_MAX_POINTS || options.threads > 1) {
            return Err(Error::new(
                ErrorKind::Unsupported,
                "the stock reader runs only as a control of at most 1M vectors at one thread",
            ));
        }

        let disk = Path::new(&disk);
        let runtime = take_runtime();

        let (searcher, t2, loaded, prefill) = match options.reader {
            Reader::Pool => {
                let factory = PoolFactory::new(disk, Arc::clone(pool), Arc::clone(&counters))?;
                let prefill = factory.prefill();
                let (s, t) = searcher(factory, options, &pq, runtime)?;
                (Searcher::Pool(s), t, 0, Some(prefill))
            }
            Reader::Advise => {
                let factory = AdviseFactory::new(disk, Arc::clone(&counters))?;
                let (s, t) = searcher(factory, options, &pq, runtime)?;
                (Searcher::Advise(s), t, 0, None)
            }
            Reader::Stock => {
                let factory = StockFactory::new(disk, Arc::clone(&counters))?;
                let loaded = factory.loaded();
                let (s, t) = searcher(factory, options, &pq, runtime)?;
                (Searcher::Stock(s), t, loaded, None)
            }
        };

        let t3 = Instant::now();
        let us = |a: Instant, b: Instant| b.duration_since(a).as_secs_f64() * 1e6;
        let opened = OpenStats {
            pq_us: us(t0, t1),
            reader_us: us(t1, t2),
            searcher_us: us(t2, t3),
            bytes: counters.tally().bytes + loaded,
        };
        Ok(Index {
            searcher,
            counters,
            prefill,
            dim: pq.get_pq_data().get_dim(),
            points: pq.get_num_points(),
            reader: options.reader,
            opened,
        })
    }

    /// Turns the pool reader's prefill on or off. While it is on, every read first asks for its sectors with
    /// `F_RDADVISE`, which leaves them in the page cache, so a later search of the same query reads them from there.
    /// It returns whether this index's reader has the switch, which only the pool reader does.
    pub fn set_prefill(&self, on: bool) -> bool {
        match &self.prefill {
            Some(flag) => {
                flag.store(on, Ordering::Relaxed);
                true
            }
            None => false,
        }
    }

    /// The width of the vectors.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// The vector count.
    pub fn points(&self) -> usize {
        self.points
    }

    /// The `k` nearest vectors to `query`, with search list size `list` and beam width `beam`.
    pub fn search(&self, query: &[f32], k: usize, list: usize, beam: usize) -> Result<Found, Error> {
        if query.len() != self.dim {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                format!("the query has {} values, not {}", query.len(), self.dim),
            ));
        }

        if k == 0 || list < k || beam == 0 {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                format!(
                    "k {k}, list size {list} and beam width {beam} \
                                                                    need 0 < k <= list and a beam of at least 1"
                ),
            ));
        }

        let before = self.counters.tally();
        let (k32, list32) = (k as u32, list as u32);
        let mode = SearchMode::graph();
        let result = match &self.searcher {
            Searcher::Pool(s) => s.search(query, k32, list32, Some(beam), mode),
            Searcher::Advise(s) => s.search(query, k32, list32, Some(beam), mode),
            Searcher::Stock(s) => s.search(query, k32, list32, Some(beam), mode),
        }
        .map_err(crate::ann)?;

        Ok(found(result, self.counters.tally().since(before)))
    }

    /// The `k` nearest vectors to each query of `queries`, row after row, on `threads` threads. Row i of the result
    /// holds query i's ids, with -1 past the ids found.
    pub fn search_batch(
        &self,
        queries: &[f32],
        k: usize,
        list: usize,
        beam: usize,
        threads: usize,
    ) -> Result<Vec<i64>, Error> {
        if self.dim == 0 || !queries.len().is_multiple_of(self.dim) {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "the queries aren't whole rows of the index's width",
            ));
        }

        let count = queries.len() / self.dim;
        let mut out = vec![-1i64; count * k];
        let next = AtomicUsize::new(0);
        let failed: Mutex<Option<Error>> = Mutex::new(None);
        let rows: Vec<Mutex<&mut [i64]>> = out.chunks_mut(k.max(1)).map(Mutex::new).collect();

        std::thread::scope(|scope| {
            // The threads run at the quality of service (QoS) class a new thread starts with, as the threads of every
            // other library's batch call do.
            for _ in 0..threads.max(1).min(count.max(1)) {
                scope.spawn(|| {
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);

                        if i >= count {
                            break;
                        }

                        match self.search(&queries[i * self.dim..(i + 1) * self.dim], k, list, beam) {
                            Ok(found) => {
                                let mut row = rows[i].lock().unwrap_or_else(PoisonError::into_inner);

                                for (slot, id) in row.iter_mut().zip(&found.ids) {
                                    *slot = *id as i64;
                                }
                            }
                            Err(e) => {
                                failed.lock().unwrap_or_else(PoisonError::into_inner).get_or_insert(e);
                                break;
                            }
                        }
                    }
                });
            }
        });

        drop(rows);

        match failed.into_inner().unwrap_or_else(PoisonError::into_inner) {
            Some(e) => Err(e),
            None => Ok(out),
        }
    }
}

/// Creates DiskANN's searcher over `factory`, and returns it with the moment the factory was ready.
fn searcher<F: AlignedReaderFactory>(
    factory: F,
    options: &Options,
    pq: &DiskIndexReader,
    runtime: Option<Runtime>,
) -> Result<(DiskSearcher<F>, Instant), Error> {
    let caching = match options.cache_nodes {
        0 => CachingStrategy::None,
        n => CachingStrategy::StaticCacheWithBfsNodes(n),
    };
    let vertices = DiskVertexProviderFactory::new(factory, caching).map_err(crate::ann)?;
    let ready = Instant::now();
    let searcher = DiskIndexSearcher::new(options.threads.max(1), IO_LIMIT, pq, vertices, options.metric, runtime)
        .map_err(crate::ann)?;
    Ok((searcher, ready))
}

fn found(result: SearchResult<()>, reads: Tally) -> Found {
    let q = &result.stats.query_statistics;
    let n = (result.stats.result_count as usize).min(result.results.len());
    Found {
        ids: result.results[..n].iter().map(|r| r.vertex_id).collect(),
        distances: result.results[..n].iter().map(|r| r.distance).collect(),
        stats: Stats {
            hops: q.search_hops,
            ios: q.total_io_operations,
            cmps: result.stats.cmps,
            results: result.stats.result_count,
            io_us: q.io_time_us as f64,
            pq_us: q.query_pq_preprocess_time_us as f64,
            total_us: q.total_execution_time_us as f64,
            reads,
        },
    }
}
