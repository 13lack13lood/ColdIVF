//! A backend that reads a wave's ranges on up to 16 threads at once, with positioned reads. It works on every system,
//! and macOS falls back to it where `F_RDADVISE` doesn't work, eg. on FAT32 (File Allocation Table) or some network
//! drives. It starts a thread per range of a wave, up to 16, as waves need them, and keeps them waiting for the next
//! wave. A wave finishes within `submit`, so every range is in when `submit` returns. On Windows the reads through one
//! handle still run one at a time.

use std::fs::File;
use std::io::Error;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::thread::JoinHandle;

use super::{Cursor, LANES, Memory, Range, Wave, read_at};
use crate::trace::ReadPath;

/// The most threads one backend starts.
pub(crate) const THREADS: usize = 16;

#[derive(Default)]
pub(crate) struct PoolWave {
    shared: Arc<Shared>,
    threads: Vec<JoinHandle<()>>,
    lanes: [Cursor; LANES],
}

/// What the threads share with `submit`.
#[derive(Default)]
struct Shared {
    state: Mutex<State>,
    /// Wakes the threads when there are jobs, or when they should stop.
    start: Condvar,
    /// Wakes `submit` when the last job is done.
    done: Condvar,
}

#[derive(Default)]
struct State {
    /// Each thread's job, and its result once the job is done.
    jobs: [Option<Job>; THREADS],
    results: [Option<Result<(), Error>>; THREADS],
    /// How many jobs aren't done yet.
    left: usize,
    stop: bool,
}

/// A run of consecutive ranges for one thread to read, and the part of the arena they land in, which starts at arena
/// byte `base`.
#[derive(Clone, Copy)]
struct Job {
    file: *const File,
    ranges: *const Range,
    count: usize,
    arena: *mut u8,
    len: usize,
    base: usize,
}

// SAFETY: a job's pointers are only used while `submit` waits for the job, and `submit` keeps the file, the ranges and
// the job's part of the arena borrowed for it alone until then.
unsafe impl Send for Job {}

impl PoolWave {
    /// How many threads it has started.
    #[cfg(test)]
    pub(crate) fn started(&self) -> usize {
        self.threads.len()
    }

    /// Starts threads until `want` of them run, up to 16, and returns how many run. A thread the system refuses is
    /// tried again on the next wave.
    fn spawn(&mut self, want: usize) -> usize {
        while self.threads.len() < want.min(THREADS) {
            let (shared, t) = (Arc::clone(&self.shared), self.threads.len());
            let started = std::thread::Builder::new()
                .name("coldivf-read".into())
                .spawn(move || work(&shared, t));
            match started {
                Ok(handle) => self.threads.push(handle),
                Err(_) => break,
            }
        }

        self.threads.len()
    }
}

impl Wave for PoolWave {
    fn page(&self) -> usize {
        4096
    }

    /// Reads every range. The ranges must be packed in order in the arena, as `coalesce` lays them out.
    fn submit(&mut self, lane: usize, file: &File, ranges: &[Range], arena: Memory) -> Result<(), Error> {
        self.lanes[lane].start(0);
        let threads = self.spawn(ranges.len()).min(ranges.len());
        // SAFETY: this backend's waves are done when `submit` returns, so no read of either lane is in flight, and the
        // caller leaves the lane's memory to this wave until it is done.
        let arena = unsafe { arena.slice_mut(0, arena.len()) };

        if threads == 0 {
            ranges
                .iter()
                .try_for_each(|r| read_at(file, &mut arena[r.at..r.at + r.len], r.offset))?;
            self.lanes[lane].start(ranges.len());
            return Ok(());
        }

        // Each thread reads a run of consecutive ranges into its own part of the arena.
        let mut jobs = [None; THREADS];
        let (mut rest, mut base) = (arena, 0);

        for (t, job) in jobs.iter_mut().enumerate().take(threads) {
            let group = &ranges[t * ranges.len() / threads..(t + 1) * ranges.len() / threads];
            let end = group.last().map_or(base, |r| r.at + r.len);
            let (mine, tail) = std::mem::take(&mut rest).split_at_mut(end - base);
            *job = Some(Job {
                file,
                ranges: group.as_ptr(),
                count: group.len(),
                arena: mine.as_mut_ptr(),
                len: mine.len(),
                base,
            });
            (rest, base) = (tail, end);
        }

        let mut state = lock(&self.shared.state);
        state.jobs = jobs;
        state.left = threads;
        self.shared.start.notify_all();

        while state.left > 0 {
            state = self.shared.done.wait(state).unwrap_or_else(PoisonError::into_inner);
        }

        let mut failed = None;

        for result in state.results.iter_mut().take(threads) {
            if let Some(Err(e)) = result.take() {
                failed.get_or_insert(e);
            }
        }

        drop(state);

        if let Some(e) = failed {
            return Err(e);
        }

        self.lanes[lane].start(ranges.len());
        Ok(())
    }

    fn next(&mut self, lane: usize, _file: &File, _ranges: &[Range], _arena: Memory) -> Result<Option<usize>, Error> {
        Ok(self.lanes[lane].take())
    }

    fn read_path(&self) -> ReadPath {
        ReadPath {
            backend: "pool",
            ..ReadPath::default()
        }
    }
}

impl Drop for PoolWave {
    fn drop(&mut self) {
        lock(&self.shared.state).stop = true;
        self.shared.start.notify_all();
        self.threads.drain(..).for_each(|h| {
            let _ = h.join();
        });
    }
}

/// Thread t's loop. It waits for a job, reads it and reports the result, until the backend stops.
fn work(shared: &Shared, t: usize) {
    let mut state = lock(&shared.state);

    while !state.stop {
        let Some(job) = state.jobs[t].take() else {
            state = shared.start.wait(state).unwrap_or_else(PoisonError::into_inner);
            continue;
        };
        drop(state);
        let result = catch_unwind(AssertUnwindSafe(|| run(job)));
        state = lock(&shared.state);
        state.results[t] = Some(result.unwrap_or_else(|_| Err(Error::other("a read thread panicked"))));
        state.left -= 1;
        if state.left == 0 {
            shared.done.notify_one();
        }
    }
}

/// Reads a job's ranges.
fn run(job: Job) -> Result<(), Error> {
    // SAFETY: `submit` keeps the file, the ranges and the job's part of the arena alive, and touches none of them,
    // until the job's result is in.
    let (file, ranges, arena) = unsafe {
        (
            &*job.file,
            std::slice::from_raw_parts(job.ranges, job.count),
            std::slice::from_raw_parts_mut(job.arena, job.len),
        )
    };
    let at = |r: &Range| r.at - job.base..r.at - job.base + r.len;
    ranges
        .iter()
        .try_for_each(|r| read_at(file, &mut arena[at(r)], r.offset))
}

fn lock(state: &Mutex<State>) -> MutexGuard<'_, State> {
    state.lock().unwrap_or_else(PoisonError::into_inner)
}
