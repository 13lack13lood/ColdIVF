//! Where a search's time goes, for benchmarks. Without the `bench` feature no clock is read, so a search pays nothing
//! for it.

use std::time::Duration;
#[cfg(feature = "bench")]
use std::time::Instant;

/// Where the last search on a context spent its time, and what it read. Only the `bench` feature fills it in. Every
/// search starts it afresh, and so does a route, which a fetch and the search of what it fetched then add to. A warm
/// search scans in place, so it counts the route in `scan` and reads only rows. A mapped search reads through page
/// faults, so it counts the route and the clusters' faults in `scan` and the rows' faults in `rerank`, and it reads no
/// bytes in waves.
#[derive(Clone, Copy, Debug, Default)]
pub struct Trace {
    /// Reading and checking the cluster tables, which only the first search of an index does.
    pub tables: Duration,
    /// Making the context's buffers for the index, which a search does only when the context lacks them.
    /// `Index::prepare` makes them ahead and readies the read backend, but the buffer for the clusters is sized from
    /// the cluster tables, so until those are loaded, the first search of the index on a context makes it here.
    pub buffers: Duration,
    /// Scoring the cluster centers and choosing the clusters to read.
    pub route: Duration,
    /// Planning the cluster reads and starting them. The thread pool's submit also waits for them, so with the pool
    /// this holds the whole wave.
    pub submit: Duration,
    /// Waiting for the cluster reads to arrive. The macOS backend and the portable one wait here, but the thread pool
    /// has every read in by the end of its submit, so with the pool this is about 0. The pool reads on Linux and
    /// Windows, on Mac drives that refuse read hints, and with `Backend::Pool`. On every backend, `submit` plus `wave`
    /// is the time the cluster wave held up the search.
    pub wave: Duration,
    /// Scanning the clusters' 1-bit codes and the inserted vectors.
    pub scan: Duration,
    /// Planning the row reads, starting them and waiting for them.
    pub rows: Duration,
    /// Rescoring the candidates with their rows.
    pub rerank: Duration,
    /// The vectors in the clusters it scanned, not counting inserted ones.
    pub scanned: usize,
    /// The inserts since the last maintenance run that it scanned, those of the clusters it scanned and of the clusters
    /// ranked next, and the generation of the index that it searched, which each maintenance run's commit moves on by
    /// one. Over several collections it adds up the inserts and keeps the last collection's generation.
    pub inserts_scanned: usize,
    pub generation: u64,
    /// The cluster reads, in bytes and in ranges, once rounded out to pages.
    pub cluster_bytes: usize,
    pub cluster_ranges: usize,
    /// The final query's row reads, the same way.
    pub row_bytes: usize,
    pub row_ranges: usize,
    /// The ranges of the search's waves that came from the backend's own copies of pages it read before, eg. the Linux
    /// backend's extent cache, instead of from the file.
    pub cache_hits: usize,
    /// The read backend of the context's reads of the index, as set up at the end of the search. A mapped search reads
    /// no waves, so it leaves this empty.
    pub read_path: ReadPath,
}

/// A search context's read backend for an index, and how it is set up.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ReadPath {
    /// The backend: "io_uring", "fadvise" (read hints, then each range in order), "rdadvise" (the macOS read hints,
    /// then each range in order), "pool" (a thread pool) or "portable" (one read at a time).
    pub backend: &'static str,
    /// The io_uring ring's size, or 0 for other backends.
    pub ring_entries: u32,
    /// Whether io_uring's cold waves read through a handle that skips the operating system's cache (`O_DIRECT`). A warm
    /// search's row waves read through the cache either way.
    pub direct: bool,
    /// Whether io_uring knows the files by registered slots, and how many arenas it has registered.
    pub fixed_files: bool,
    pub registered: usize,
}

/// A stopwatch for the stages of a search. Without the `bench` feature it reads no clock, and every lap is zero.
pub(crate) struct Clock {
    #[cfg(feature = "bench")]
    at: Instant,
}

impl Clock {
    pub(crate) fn start() -> Clock {
        Clock {
            #[cfg(feature = "bench")]
            at: Instant::now(),
        }
    }

    /// The time since the last lap, or since the start.
    pub(crate) fn lap(&mut self) -> Duration {
        #[cfg(feature = "bench")]
        {
            let now = Instant::now();
            let lap = now - self.at;
            self.at = now;
            lap
        }

        #[cfg(not(feature = "bench"))]
        Duration::ZERO
    }
}
