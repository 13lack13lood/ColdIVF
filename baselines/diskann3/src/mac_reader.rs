//! Readers that serve DiskANN's sector reads on macOS without loading the whole index file.
//!
//! DiskANN reads a beam of nodes at a time, one 4 KiB sector per node, and hands every sector of a beam to its reader
//! in one call. Both readers here read through one handle per index with `F_NOCACHE` set, so their plain reads skip
//! the page cache.
//! - `PoolReader` reads the sectors of a beam at once, the first on the calling thread and the rest on a pool of
//!   threads. Its factory's prefill switch makes it ask for each sector with `F_RDADVISE` first, which brings the
//!   sectors into the page cache, eg. for a rerun of a query that reads them from there.
//! - `AdviseReader` first starts every read of a beam with `F_RDADVISE`, which has the kernel read them all at once
//!   into the page cache, then reads each sector in order.
//!
//! `StockReader` wraps DiskANN's own macOS reader, which copies the whole file into memory. It is built once per index
//! and shared by every search, as a control that loads the index at open.

use std::collections::VecDeque;
use std::fs::File;
use std::io::Error;
use std::os::fd::AsRawFd;
use std::os::unix::fs::FileExt;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock, PoisonError};
use std::thread::JoinHandle;
use std::time::Instant;

use diskann::ANNResult;
use diskann_disk::search::provider::aligned_file_reader::traits::{AlignedFileReader, AlignedReaderFactory};
use diskann_disk::search::provider::aligned_file_reader::{
    A1, AlignedRead, Alignment, StorageProviderAlignedFileReader,
};
use diskann_providers::storage::FileStorageProvider;
use diskann_quantization::num::PowerOfTwo;

/// The sector size of a DiskANN index, and the alignment of every read the two readers make.
pub const SECTOR: usize = 4096;

/// One sector of alignment for the readers' offsets, lengths and buffers. DiskANN allocates its sector buffers with
/// the alignment a reader asks for.
#[derive(Debug, Clone, Copy)]
pub struct A4096;

impl Alignment for A4096 {
    const VALUE: PowerOfTwo = PowerOfTwo::V4096;
}

/// What the readers of one index have read, added up over every search.
#[derive(Debug, Default)]
pub struct Counters {
    reads: AtomicU64,
    calls: AtomicU64,
    bytes: AtomicU64,
    nanos: AtomicU64,
}

/// A snapshot of `Counters`. `reads` counts sectors, `calls` the reader calls, one per beam, and `nanos` the time spent
/// inside them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Tally {
    pub reads: u64,
    pub calls: u64,
    pub bytes: u64,
    pub nanos: u64,
}

impl Counters {
    pub fn tally(&self) -> Tally {
        Tally {
            reads: self.reads.load(Ordering::Relaxed),
            calls: self.calls.load(Ordering::Relaxed),
            bytes: self.bytes.load(Ordering::Relaxed),
            nanos: self.nanos.load(Ordering::Relaxed),
        }
    }

    fn add(&self, reads: usize, bytes: usize, start: Instant) {
        self.reads.fetch_add(reads as u64, Ordering::Relaxed);
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.bytes.fetch_add(bytes as u64, Ordering::Relaxed);
        self.nanos
            .fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
}

impl Tally {
    /// What was read from `before` to `self`.
    pub fn since(self, before: Tally) -> Tally {
        Tally {
            reads: self.reads.saturating_sub(before.reads),
            calls: self.calls.saturating_sub(before.calls),
            bytes: self.bytes.saturating_sub(before.bytes),
            nanos: self.nanos.saturating_sub(before.nanos),
        }
    }
}

/// Opens `path` for reading with `F_NOCACHE` set, so reads that miss the page cache don't fill it.
pub fn open_uncached(path: &Path) -> Result<File, Error> {
    let file = File::open(path)?;
    // SAFETY: F_NOCACHE takes an integer and touches no memory.
    if unsafe { libc::fcntl(file.as_raw_fd(), libc::F_NOCACHE, 1) } == -1 {
        return Err(Error::last_os_error());
    }

    Ok(file)
}

/// Asks the kernel to start reading `len` bytes of `file` from `offset` into the page cache. A range past the end of
/// the file gets `EFBIG`, which is harmless.
pub fn advise(file: &File, offset: u64, len: usize) -> Result<(), Error> {
    let ra = libc::radvisory {
        ra_offset: offset as libc::off_t,
        ra_count: len.min(i32::MAX as usize) as libc::c_int,
    };
    // SAFETY: `ra` outlives the call, and F_RDADVISE only reads it.
    if unsafe { libc::fcntl(file.as_raw_fd(), libc::F_RDADVISE, &ra as *const libc::radvisory) } == -1 {
        let e = Error::last_os_error();

        if e.raw_os_error() != Some(libc::EFBIG) {
            return Err(e);
        }
    }

    Ok(())
}

/// Threads that read sectors for `PoolReader`. One pool serves every index of a process, and it only grows. Its
/// threads run at the quality of service (QoS) class a new thread starts with, as a library's own threads do.
pub struct ReadPool {
    shared: Arc<Shared>,
    threads: Mutex<Vec<JoinHandle<()>>>,
    /// How many threads run, read on every call without the lock.
    running: AtomicUsize,
}

struct Shared {
    queue: Mutex<Queue>,
    /// Wakes a thread for each job queued, and every thread when the pool stops.
    ready: Condvar,
}

#[derive(Default)]
struct Queue {
    jobs: VecDeque<Job>,
    stop: bool,
}

/// One sector read for a pool thread. Its buffer belongs to a `ReadPool::read` call, which waits for every job of its
/// batch before it returns.
struct Job {
    file: Arc<File>,
    buf: *mut u8,
    len: usize,
    offset: u64,
    batch: Arc<Batch>,
}

// SAFETY: only the one thread that runs a job writes its buffer, and the `read` call that owns the buffer touches no
// part of it until the job's batch is done.
unsafe impl Send for Job {}

/// The jobs of one `read` call. The thread that finishes the last one wakes the caller.
struct Batch {
    left: AtomicUsize,
    failed: Mutex<Option<Error>>,
    lock: Mutex<()>,
    done: Condvar,
}

impl Batch {
    fn new(jobs: usize) -> Batch {
        Batch {
            left: AtomicUsize::new(jobs),
            failed: Mutex::new(None),
            lock: Mutex::new(()),
            done: Condvar::new(),
        }
    }

    fn finish(&self, result: Result<(), Error>) {
        if let Err(e) = result {
            lock(&self.failed).get_or_insert(e);
        }

        if self.left.fetch_sub(1, Ordering::AcqRel) == 1 {
            let _guard = lock(&self.lock);
            self.done.notify_all();
        }
    }

    /// Waits until every job is done, and returns the first error.
    fn wait(&self) -> Result<(), Error> {
        if self.left.load(Ordering::Acquire) != 0 {
            let mut guard = lock(&self.lock);

            while self.left.load(Ordering::Acquire) != 0 {
                guard = self.done.wait(guard).unwrap_or_else(PoisonError::into_inner);
            }
        }

        match lock(&self.failed).take() {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

impl Default for ReadPool {
    fn default() -> Self {
        ReadPool {
            shared: Arc::new(Shared {
                queue: Mutex::new(Queue::default()),
                ready: Condvar::new(),
            }),
            threads: Mutex::new(Vec::new()),
            running: AtomicUsize::new(0),
        }
    }
}

impl ReadPool {
    /// The process's pool, which starts with no threads.
    pub fn global() -> &'static Arc<ReadPool> {
        static POOL: OnceLock<Arc<ReadPool>> = OnceLock::new();
        POOL.get_or_init(|| Arc::new(ReadPool::default()))
    }

    /// How many threads run.
    pub fn threads(&self) -> usize {
        self.running.load(Ordering::Acquire)
    }

    /// Starts threads until at least `want` run, and returns how many run.
    pub fn reserve(&self, want: usize) -> Result<usize, Error> {
        let mut threads = lock(&self.threads);

        while threads.len() < want {
            let shared = Arc::clone(&self.shared);
            let handle = std::thread::Builder::new()
                .name("diskann3-read".into())
                .spawn(move || work(&shared))?;
            threads.push(handle);
            self.running.store(threads.len(), Ordering::Release);
        }

        Ok(threads.len())
    }

    /// Reads every request from `file`. The first is read on the calling thread while the pool reads the rest, so a
    /// call with one request, or a pool with no threads, reads on the calling thread alone.
    pub fn read(&self, file: &Arc<File>, requests: &mut [AlignedRead<'_, u8, A4096>]) -> Result<(), Error> {
        let Some((first, rest)) = requests.split_first_mut() else {
            return Ok(());
        };

        if rest.is_empty() || self.threads() == 0 {
            read_sector(file, first)?;
            return rest.iter_mut().try_for_each(|r| read_sector(file, r));
        }

        let batch = Arc::new(Batch::new(rest.len()));
        {
            let mut queue = lock(&self.shared.queue);

            for r in rest.iter_mut() {
                let offset = r.offset();
                let buf = r.aligned_buf_mut();
                queue.jobs.push_back(Job {
                    file: Arc::clone(file),
                    buf: buf.as_mut_ptr(),
                    len: buf.len(),
                    offset,
                    batch: Arc::clone(&batch),
                });
            }
        }

        for _ in 0..rest.len() {
            self.shared.ready.notify_one();
        }

        let mine = read_sector(file, first);
        let theirs = batch.wait();
        mine.and(theirs)
    }
}

impl Drop for ReadPool {
    fn drop(&mut self) {
        lock(&self.shared.queue).stop = true;
        self.shared.ready.notify_all();

        for handle in lock(&self.threads).drain(..) {
            let _ = handle.join();
        }
    }
}

/// A pool thread's loop. It takes jobs until the pool stops and its queue is empty.
fn work(shared: &Shared) {
    loop {
        let job = {
            let mut queue = lock(&shared.queue);

            loop {
                if let Some(job) = queue.jobs.pop_front() {
                    break job;
                }

                if queue.stop {
                    return;
                }

                queue = shared.ready.wait(queue).unwrap_or_else(PoisonError::into_inner);
            }
        };

        let result = catch_unwind(AssertUnwindSafe(|| {
            // SAFETY: the job's `read` call keeps the buffer alive and untouched until the batch is done, and the
            // slice ends before `finish` tells it so.
            let buf = unsafe { std::slice::from_raw_parts_mut(job.buf, job.len) };
            job.file.read_exact_at(buf, job.offset)
        }));
        job.batch
            .finish(result.unwrap_or_else(|_| Err(Error::other("a read thread panicked"))));
    }
}

fn read_sector<A: Alignment>(file: &File, request: &mut AlignedRead<'_, u8, A>) -> Result<(), Error> {
    let offset = request.offset();
    file.read_exact_at(request.aligned_buf_mut(), offset)
}

/// Builds `PoolReader`s over one shared handle of the index file.
pub struct PoolFactory {
    file: Arc<File>,
    pool: Arc<ReadPool>,
    counters: Arc<Counters>,
    prefill: Arc<AtomicBool>,
}

impl PoolFactory {
    /// Opens `path` without the page cache, adding every read to `counters`. A beam of W sectors is read all at once
    /// when `pool` has W - 1 threads free, since the calling thread reads one sector itself.
    pub fn new(path: &Path, pool: Arc<ReadPool>, counters: Arc<Counters>) -> Result<PoolFactory, Error> {
        Ok(PoolFactory {
            file: Arc::new(open_uncached(path)?),
            pool,
            counters,
            prefill: Arc::new(AtomicBool::new(false)),
        })
    }

    /// The prefill switch of every reader this factory builds, off at first. While it is on, a read first asks for
    /// each of its sectors with `F_RDADVISE`, which leaves them in the page cache.
    pub fn prefill(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.prefill)
    }
}

impl AlignedReaderFactory for PoolFactory {
    type AlignedReaderType = PoolReader;

    fn build(&self) -> ANNResult<PoolReader> {
        Ok(PoolReader {
            file: Arc::clone(&self.file),
            pool: Arc::clone(&self.pool),
            counters: Arc::clone(&self.counters),
            prefill: Arc::clone(&self.prefill),
        })
    }
}

/// Reads the sectors of a beam at once on a pool of threads.
pub struct PoolReader {
    file: Arc<File>,
    pool: Arc<ReadPool>,
    counters: Arc<Counters>,
    prefill: Arc<AtomicBool>,
}

impl AlignedFileReader for PoolReader {
    type Alignment = A4096;

    fn read(&mut self, requests: &mut [AlignedRead<u8, A4096>]) -> ANNResult<()> {
        let start = Instant::now();
        let bytes = requests.iter().map(|r| r.aligned_buf().len()).sum();

        if self.prefill.load(Ordering::Relaxed) {
            for r in requests.iter() {
                advise(&self.file, r.offset(), r.aligned_buf().len())?;
            }
        }

        self.pool.read(&self.file, requests)?;
        self.counters.add(requests.len(), bytes, start);
        Ok(())
    }
}

/// Builds `AdviseReader`s over one shared handle of the index file.
pub struct AdviseFactory {
    file: Arc<File>,
    counters: Arc<Counters>,
}

impl AdviseFactory {
    /// Opens `path` without the page cache, adding every read to `counters`.
    pub fn new(path: &Path, counters: Arc<Counters>) -> Result<AdviseFactory, Error> {
        Ok(AdviseFactory {
            file: Arc::new(open_uncached(path)?),
            counters,
        })
    }
}

impl AlignedReaderFactory for AdviseFactory {
    type AlignedReaderType = AdviseReader;

    fn build(&self) -> ANNResult<AdviseReader> {
        Ok(AdviseReader {
            file: Arc::clone(&self.file),
            counters: Arc::clone(&self.counters),
        })
    }
}

/// Starts every read of a beam with `F_RDADVISE`, then reads each sector in order.
pub struct AdviseReader {
    file: Arc<File>,
    counters: Arc<Counters>,
}

impl AlignedFileReader for AdviseReader {
    type Alignment = A4096;

    fn read(&mut self, requests: &mut [AlignedRead<u8, A4096>]) -> ANNResult<()> {
        let start = Instant::now();
        let mut bytes = 0;

        for r in requests.iter() {
            advise(&self.file, r.offset(), r.aligned_buf().len())?;
        }

        for r in requests.iter_mut() {
            bytes += r.aligned_buf().len();
            read_sector(&self.file, r)?;
        }

        self.counters.add(requests.len(), bytes, start);
        Ok(())
    }
}

/// DiskANN's macOS reader, built once from the whole index file and shared by every reader the factory builds.
pub struct StockFactory {
    reader: Arc<Mutex<StorageProviderAlignedFileReader>>,
    counters: Arc<Counters>,
    loaded: u64,
}

impl StockFactory {
    /// Reads all of `path` into memory.
    pub fn new(path: &Path, counters: Arc<Counters>) -> Result<StockFactory, Error> {
        let name = path
            .to_str()
            .ok_or_else(|| Error::other(format!("{} isn't valid UTF-8", path.display())))?;
        let reader = StorageProviderAlignedFileReader::new(&FileStorageProvider, name).map_err(crate::ann)?;
        Ok(StockFactory {
            reader: Arc::new(Mutex::new(reader)),
            counters,
            loaded: std::fs::metadata(path)?.len(),
        })
    }

    /// The bytes copied into memory, the whole file.
    pub fn loaded(&self) -> u64 {
        self.loaded
    }
}

impl AlignedReaderFactory for StockFactory {
    type AlignedReaderType = StockReader;

    fn build(&self) -> ANNResult<StockReader> {
        Ok(StockReader {
            reader: Arc::clone(&self.reader),
            counters: Arc::clone(&self.counters),
        })
    }
}

/// Copies sectors out of the shared copy of the file.
pub struct StockReader {
    reader: Arc<Mutex<StorageProviderAlignedFileReader>>,
    counters: Arc<Counters>,
}

impl AlignedFileReader for StockReader {
    type Alignment = A1;

    fn read(&mut self, requests: &mut [AlignedRead<u8, A1>]) -> ANNResult<()> {
        let start = Instant::now();
        let bytes = requests.iter().map(|r| r.aligned_buf().len()).sum();
        lock(&self.reader).read(requests)?;
        self.counters.add(requests.len(), bytes, start);
        Ok(())
    }
}

#[cfg(test)]
#[path = "../tests/unit/mac_reader.rs"]
mod tests;
