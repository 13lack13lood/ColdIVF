//! Inserts and deletes. Each change goes to the log of the index's current data file, then into the changes that every
//! search of that file in this process reads. A search loads them once, as one snapshot, without a lock, and keeps it
//! until it returns, so it sees the changes as they were when it began. Inserted vectors wait in memory, grouped by
//! the cluster they were routed to, and a search scans those of the clusters it probes and of the clusters ranked
//! next, up to 5/4 of `nprobe`, until a maintenance run folds them into a new data file. With the cluster anchor each
//! one is measured from the center of its cluster.
//!
//! Every change gets the next sequence number, and the rows of the data file count as number 0. An insert hides the
//! visible version of its id, and a delete hides it too. A hidden row or insert keeps the sequence number of the change
//! that hid it, so a search judges each one as of its snapshot, even after later changes. Rows that a change hid never
//! reach a search's candidates, so they take no place among the rows it rescores.

pub(crate) mod delta;
pub(crate) mod log;
pub(crate) mod view;

use std::fs::{File, OpenOptions, TryLockError};
use std::io::{Error, ErrorKind, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError, Weak};

use arc_swap::{ArcSwap, Guard};

use crate::format::Layout;
use crate::quantization::rabitq::Factors;
pub(crate) use delta::{Applier, Delta, Seen};
pub(crate) use log::{Log, Record, Version};

/// The changes of one data file, shared by every open index and writer of that file in this process.
pub(crate) struct Changes {
    /// What searches see: the latest snapshot.
    current: ArcSwap<Delta>,
    /// What builds the next snapshot, which only writers and opens take.
    applier: Mutex<Applier>,
    /// The snapshots swapped out, until no search holds them, so a search never frees one.
    retired: Mutex<Vec<Arc<Delta>>>,
}

impl Changes {
    fn new(layout: &Layout, version: Version, data: &Path) -> Changes {
        let mut applier = Applier::new(layout, version, data);
        Changes {
            current: ArcSwap::from_pointee(applier.snapshot()),
            applier: Mutex::new(applier),
            retired: Mutex::new(Vec::new()),
        }
    }

    /// The latest snapshot, which a search keeps until it returns.
    pub(crate) fn load(&self) -> Guard<Arc<Delta>> {
        self.current.load()
    }

    /// The same, as its own reference, eg. for a maintenance run or a search over several collections.
    pub(crate) fn load_full(&self) -> Arc<Delta> {
        self.current.load_full()
    }

    /// The applier, held until the guard drops.
    pub(crate) fn applier(&self) -> MutexGuard<'_, Applier> {
        locked(&self.applier)
    }

    /// Makes what `applier` built the snapshot that searches see, and frees the old snapshots that no search holds.
    pub(crate) fn publish(&self, applier: &mut Applier) {
        let old = self.current.swap(Arc::new(applier.snapshot()));
        let mut retired = locked(&self.retired);
        retired.push(old);
        retired.retain(|d| Arc::strong_count(d) > 1);
    }

    /// The sequence number of the next change, and where the visible version of `id` is, as an internal id, for a
    /// record of a change of `id`.
    pub(crate) fn hint(&self, id: u64) -> Result<(u64, u32), Error> {
        let mut a = self.applier();
        let at = a.locate(id)?;
        Ok((a.next(), at))
    }

    /// Applies `record`, which ends at byte `end` of the log, and publishes it, unless an open of the file in this
    /// process read it from the log first.
    pub(crate) fn apply(&self, record: &[u8], end: usize) -> Result<(), Error> {
        let mut a = self.applier();

        if a.log_len() < end {
            a.apply(record, false)?;
            a.set_log_len(end);
            self.publish(&mut a);
        }

        Ok(())
    }
}

/// Takes `mutex`, whatever a panic left in it while another thread held it. Every mutex of the crate outside the read
/// backends goes through here, so tests can count the locks that each thread takes, eg. to check that searches take
/// none.
pub(crate) fn locked<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    #[cfg(test)]
    LOCKS.with(|n| n.set(n.get() + 1));

    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

#[cfg(test)]
thread_local! {
    /// How many locks `locked` took on this thread.
    pub(crate) static LOCKS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// The changes of an open generation, by the index's path, the data file's header and the version of its log, which
/// names the generation, so each generation gets its own, even one whose data file is the same as another's, as two
/// builds of the same vectors write it.
struct Opened {
    path: PathBuf,
    layout: Layout,
    version: Version,
    changes: Weak<Changes>,
}

/// The changes of every open generation.
static OPEN: Mutex<Vec<Opened>> = Mutex::new(Vec::new());

/// Where the index at `path` really is, following symbolic links, so every way of naming it shares one log and one
/// lock. It follows a link whose target is gone too, since an index's first commit deletes its original data file and
/// keeps its files next to where that was.
pub(crate) fn real_path(path: &Path) -> PathBuf {
    let mut at = path.to_path_buf();

    for _ in 0..40 {
        match std::fs::read_link(&at) {
            Ok(target) => at = at.parent().map_or(target.clone(), |dir| dir.join(&target)),
            Err(_) => break,
        }
    }

    if let Ok(real) = std::fs::canonicalize(&at) {
        return real;
    }

    // The file is gone, eg. after the first commit, so only its folder can be made canonical.
    let real_dir = at
        .parent()
        .filter(|d| !d.as_os_str().is_empty())
        .and_then(|d| std::fs::canonicalize(d).ok());

    match (real_dir, at.file_name()) {
        (Some(dir), Some(name)) => dir.join(name),
        _ => at,
    }
}

/// The version 1 log of the index at `path`, its log from before its first commit.
pub(crate) fn log_path(path: &Path) -> PathBuf {
    with_suffix(path, ".delta")
}

/// The file that a writer of the index at `path` locks. It is a file of its own, since on Windows nothing else can
/// read a locked file.
pub(crate) fn lock_path(path: &Path) -> PathBuf {
    with_suffix(path, ".lock")
}

pub(crate) fn with_suffix(path: &Path, suffix: &str) -> PathBuf {
    let mut name = path.as_os_str().to_os_string();
    name.push(suffix);
    PathBuf::from(name)
}

/// Takes the writer's lock of the index at `path`, so no other writer, in this process or another, can change it until
/// the returned file closes. When another writer holds it, it fails with `WouldBlock`.
pub(crate) fn lock(path: &Path) -> Result<File, Error> {
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(lock_path(path))?;

    match file.try_lock() {
        Ok(()) => Ok(file),
        Err(TryLockError::WouldBlock) => Err(Error::new(ErrorKind::WouldBlock, "another writer has this index open")),
        Err(TryLockError::Error(e)) => Err(e),
    }
}

/// The changes of the data file `data`, whose header says `layout`, of the index at `path`, with its log `log` of
/// `version`, shared with every other open index of that generation in this process. Each open reads whatever the log
/// holds past what the changes already have, eg. changes that another process made since an earlier open.
pub(crate) fn changes_for(
    path: &Path,
    layout: &Layout,
    data: &Path,
    log: &Path,
    version: Version,
) -> Result<Arc<Changes>, Error> {
    let key = real_path(path);
    let changes = {
        let mut open = locked(&OPEN);
        open.retain(|o| o.changes.strong_count() > 0);
        let found = open
            .iter()
            .find(|o| o.path == key && o.layout == *layout && o.version == version)
            .and_then(|o| o.changes.upgrade());
        found.unwrap_or_else(|| {
            let changes = Arc::new(Changes::new(layout, version, data));
            open.push(Opened {
                path: key,
                layout: layout.clone(),
                version,
                changes: Arc::downgrade(&changes),
            });
            changes
        })
    };

    read_more(&changes, log, layout, version)?;
    Ok(changes)
}

/// Brings `changes` up to date with the log `log`, of `version`, of the data file whose header says `layout`, eg. with
/// what another process's writer appended since. It takes the applier only when the log holds more than the changes
/// have read, and searches never wait for it.
pub(crate) fn read_more(changes: &Changes, log: &Path, layout: &Layout, version: Version) -> Result<(), Error> {
    let read = changes.load().log_len() as u64;
    let behind = log::open_ours(log, layout, version)?.is_some_and(|(_, len)| len > read);

    if behind {
        catch_up(changes, &mut changes.applier(), log, layout, version)?;
    }

    Ok(())
}

/// Whether the index at `path` has a version 1 log that wasn't written for the data file whose header says `layout`,
/// eg. one that a rebuild started again for its new file. A log too short to hold a header counts as another file's,
/// and a missing one as none, since nothing removes a log before the index's first commit.
pub(crate) fn foreign_log(path: &Path, layout: &Layout) -> Result<bool, Error> {
    let log = log_path(&real_path(path));

    match std::fs::exists(&log)? {
        true => Ok(log::open_ours(&log, layout, Version::One)?.is_none()),
        false => Ok(false),
    }
}

/// Applies the records that the log at `path`, of `version`, holds past what `applier` has read, then publishes them
/// to `changes` as one snapshot, so a replay packs each block once. Returns whether the log was written for this file.
/// A missing log, or one written for another file, holds nothing for it, and so does one shorter than what the applier
/// read. A version 1 log ends at its first record that is cut short, damaged or doesn't fit this file, as a crash in
/// the middle of a write leaves it. A version 2 log ends only at a damaged last frame, and any other damage fails with
/// `InvalidData`.
pub(crate) fn catch_up(
    changes: &Changes,
    applier: &mut Applier,
    path: &Path,
    layout: &Layout,
    version: Version,
) -> Result<bool, Error> {
    let Some((mut file, len)) = log::open_ours(path, layout, version)? else {
        return Ok(false);
    };
    let start = applier.log_len().max(version.header_len());

    if len < start as u64 {
        return Ok(true);
    }

    let mut bytes = Vec::new();
    file.seek(SeekFrom::Start(start as u64))?;
    file.read_to_end(&mut bytes)?;
    let replayed = applier.replay(&bytes, start);
    changes.publish(applier);
    replayed.map(|()| true)
}

/// Logs an insert, then makes it visible to every search of `changes`. Returns its sequence number.
pub(crate) fn insert(
    log: &mut Log,
    changes: &Changes,
    id: u64,
    leaf: u32,
    code: &[u8],
    factors: Factors,
    row: &[u8],
) -> Result<u64, Error> {
    let (seq, hides) = changes.hint(id)?;
    let record = Record::Insert {
        seq,
        id,
        list: leaf,
        hides,
        code,
        factors,
        row,
    };
    log.write(changes, &log::encode(&record, log.version()))?;
    Ok(seq)
}

/// Logs a delete, then hides the id from every search of `changes`.
pub(crate) fn delete(log: &mut Log, changes: &Changes, id: u64) -> Result<(), Error> {
    let (seq, hides) = changes.hint(id)?;
    let record = Record::Delete { seq, id, hides };
    log.write(changes, &log::encode(&record, log.version()))
}

#[cfg(test)]
#[path = "../../tests/unit/update/mod.rs"]
mod tests;

#[cfg(test)]
#[path = "../../tests/perf/delta.rs"]
mod perf;
