//! Inserts and deletes. Each change goes to the log of the index's current data file, then into the changes that every
//! search of that file in this process reads. Inserted vectors are scanned with every query, apart from the clusters,
//! until a maintenance run folds them into a new data file. With the cluster anchor each one is measured from the
//! center of the cluster it is nearest to, so a search also measures the query's distance to every cluster that holds
//! inserts, once each. A delete hides an id from then on.
//!
//! Every change gets the next sequence number, and the rows of the data file count as number 0. An insert hides the
//! earlier versions of its id, and a delete hides every version so far. Each id keeps the changes that hid its rows,
//! so a search can tell what was visible when it scanned, even after later changes.

pub(crate) mod log;

use std::collections::HashMap;
use std::fs::{File, OpenOptions, TryLockError};
use std::io::{Error, ErrorKind, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, PoisonError, RwLock, RwLockReadGuard, RwLockWriteGuard, Weak};

use crate::format::Layout;
use crate::quantization::rabitq::{BLOCK, Factors, block_bytes, code_bytes, pack_block};
use crate::quantization::{padded_dim, sq8};
pub(crate) use log::{Log, NONE, Record, Version};

/// How many vectors were inserted, and the next change's sequence number, when a search scanned the changes.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Seen {
    len: usize,
    next: u64,
}

impl Seen {
    /// How many vectors were inserted, visible or not.
    pub(crate) fn inserts(&self) -> usize {
        self.len
    }

    /// The sequence number of the first change it doesn't see.
    pub(crate) fn next(&self) -> u64 {
        self.next
    }
}

/// The changes to one data file since it was written.
pub(crate) struct Delta {
    dim: usize,
    /// The file's cluster count, which every insert's cluster must be under.
    nlist: usize,
    /// The log that the changes come from, which says how its records read.
    version: Version,
    /// The inserted vectors' 1-bit codes and factors, and the same packed in blocks of 32 for the scan.
    codes: Vec<u8>,
    factors: Vec<Factors>,
    blocks: Vec<u8>,
    /// Their 8-bit rows, back to back, each with its id.
    rows: Vec<u8>,
    /// Each one's sequence number, and the cluster it is nearest to.
    seqs: Vec<u64>,
    leaves: Vec<u32>,
    /// The clusters that hold inserts, in the order of their first, and each insert's place in that list. `place_of`
    /// finds a cluster's place.
    clusters: Vec<u32>,
    places: Vec<u32>,
    place_of: HashMap<u32, u32>,
    /// For each id that changes hid rows of, those changes in order. Each has its sequence number and the first
    /// sequence number it left visible.
    hide: HashMap<u64, Vec<(u64, u64)>>,
    /// The next change's sequence number.
    next: u64,
    /// How many bytes of the log the changes came from, or 0 before a header for this file was read.
    log_len: usize,
}

impl Delta {
    pub(crate) fn new(layout: &Layout, version: Version) -> Delta {
        Delta {
            dim: layout.dim,
            nlist: layout.nlist,
            version,
            codes: Vec::new(),
            factors: Vec::new(),
            blocks: Vec::new(),
            rows: Vec::new(),
            seqs: Vec::new(),
            leaves: Vec::new(),
            clusters: Vec::new(),
            places: Vec::new(),
            place_of: HashMap::new(),
            hide: HashMap::new(),
            next: version.base(),
            log_len: 0,
        }
    }

    /// How many vectors were inserted, visible or not.
    pub(crate) fn len(&self) -> usize {
        self.seqs.len()
    }

    pub(crate) fn blocks(&self) -> &[u8] {
        &self.blocks
    }

    /// Inserted vector i's 8-bit row.
    pub(crate) fn row(&self, i: usize) -> &[u8] {
        let rb = sq8::row_bytes(self.dim);
        &self.rows[i * rb..(i + 1) * rb]
    }

    pub(crate) fn seq(&self, i: usize) -> u64 {
        self.seqs[i]
    }

    pub(crate) fn leaf(&self, i: usize) -> u32 {
        self.leaves[i]
    }

    /// Inserted vector i's 1-bit code and factors.
    pub(crate) fn lane(&self, i: usize) -> (&[u8], Factors) {
        let cb = code_bytes(padded_dim(self.dim));
        (&self.codes[i * cb..(i + 1) * cb], self.factors[i])
    }

    /// How many bytes of the log the changes came from, which is where a change made next starts.
    pub(crate) fn log_len(&self) -> usize {
        self.log_len
    }

    pub(crate) fn set_log_len(&mut self, len: usize) {
        self.log_len = len;
    }

    /// The log that the changes come from.
    pub(crate) fn version(&self) -> Version {
        self.version
    }

    /// The sequence number that the next change gets.
    pub(crate) fn next(&self) -> u64 {
        self.next
    }

    /// The clusters that hold inserts, in the order of their first insert.
    pub(crate) fn clusters(&self) -> &[u32] {
        &self.clusters
    }

    /// The place of inserted vector i's cluster in `clusters`.
    pub(crate) fn place(&self, i: usize) -> usize {
        self.places[i] as usize
    }

    /// What a search that scans the inserted vectors now sees.
    pub(crate) fn seen(&self) -> Seen {
        Seen {
            len: self.len(),
            next: self.next,
        }
    }

    /// Whether the row with id `id` and sequence number `seq` was visible when `seen` was taken.
    pub(crate) fn is_live(&self, id: u64, seq: u64, seen: Seen) -> bool {
        let Some(changes) = self.hide.get(&id) else {
            return true;
        };
        let last = changes.iter().rev().find(|&&(at, _)| at < seen.next);
        last.is_none_or(|&(_, first)| seq >= first)
    }

    /// Whether no change has hidden anything, so every row is visible.
    pub(crate) fn hides_nothing(&self) -> bool {
        self.hide.is_empty()
    }

    fn insert(&mut self, id: u64, leaf: u32, code: &[u8], factors: Factors, row: &[u8]) {
        let (seq, d_pad) = (self.next, padded_dim(self.dim));
        let (cb, size) = (code_bytes(d_pad), block_bytes(d_pad));
        self.next += 1;
        self.codes.extend_from_slice(code);
        self.factors.push(factors);
        self.rows.extend_from_slice(row);
        self.seqs.push(seq);
        self.leaves.push(leaf);
        let place = *self.place_of.entry(leaf).or_insert_with(|| {
            self.clusters.push(leaf);
            (self.clusters.len() - 1) as u32
        });
        self.places.push(place);
        self.hide.entry(id).or_default().push((seq, seq));
        // Pack the block that the new vector joins again.
        let b = (self.len() - 1) / BLOCK;
        self.blocks.resize((b + 1) * size, 0);
        let (first, last) = (b * BLOCK, self.len());
        let out = &mut self.blocks[b * size..(b + 1) * size];
        pack_block(
            &self.codes[first * cb..last * cb],
            &self.factors[first..last],
            d_pad,
            out,
        );
    }

    fn delete(&mut self, id: u64) {
        let seq = self.next;
        self.next += 1;
        self.hide.entry(id).or_default().push((seq, seq + 1));
    }

    /// Applies one record of the log. A record that doesn't fit this file fails with `InvalidData`.
    pub(crate) fn apply(&mut self, record: &[u8]) -> Result<(), Error> {
        match log::decode(record, self.version, self.dim, self.nlist, self.next)? {
            Record::Insert {
                id,
                list,
                code,
                factors,
                row,
                ..
            } => self.insert(id, list, code, factors, row),
            Record::Delete { id, .. } => self.delete(id),
        }

        Ok(())
    }

    /// Applies the records in `bytes`, which start at byte `at` of the log. A version 1 log ends at its first record
    /// that is cut short, damaged or doesn't fit this file, as a crash in the middle of a write leaves it. A version 2
    /// log ends only at a damaged last frame, and any other damage fails with `InvalidData`.
    fn apply_all(&mut self, bytes: &[u8], at: usize) -> Result<(), Error> {
        let (records, _) = log::records(bytes, self.version)?;
        let mut used = 0;

        for record in records {
            if let Err(e) = self.apply(record) {
                self.log_len = at + used;
                return match self.version {
                    Version::One => Ok(()),
                    Version::Two { .. } => Err(e),
                };
            }
            used += 8 + record.len();
        }

        self.log_len = at + used;
        Ok(())
    }
}

/// The changes of one data file, shared by every open index and writer of that file in this process.
pub(crate) struct Changes {
    delta: RwLock<Delta>,
}

impl Changes {
    pub(crate) fn read(&self) -> RwLockReadGuard<'_, Delta> {
        self.delta.read().unwrap_or_else(PoisonError::into_inner)
    }

    pub(crate) fn write(&self) -> RwLockWriteGuard<'_, Delta> {
        self.delta.write().unwrap_or_else(PoisonError::into_inner)
    }
}

/// The open files' changes, by the index's path and the data file's header, so each data file gets its own.
static OPEN: Mutex<Vec<(PathBuf, Layout, Weak<Changes>)>> = Mutex::new(Vec::new());

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

/// The changes of the data file whose header says `layout`, of the index at `path`, with its log `log` of `version`,
/// shared with every other open index of that file in this process. Each open reads whatever the log holds past what
/// the changes already have, eg. changes that another process made since an earlier open. It takes the changes' write
/// lock only when the log holds more than they have read.
pub(crate) fn changes_for(path: &Path, layout: &Layout, log: &Path, version: Version) -> Result<Arc<Changes>, Error> {
    let key = real_path(path);
    let changes = {
        let mut open = OPEN.lock().unwrap_or_else(PoisonError::into_inner);
        open.retain(|(_, _, c)| c.strong_count() > 0);
        let found = open
            .iter()
            .find(|(p, l, _)| *p == key && l == layout)
            .and_then(|(_, _, c)| c.upgrade());
        found.unwrap_or_else(|| {
            let changes = Arc::new(Changes {
                delta: RwLock::new(Delta::new(layout, version)),
            });
            open.push((key, layout.clone(), Arc::downgrade(&changes)));
            changes
        })
    };

    read_more(&changes, log, layout, version)?;
    Ok(changes)
}

/// Brings `changes` up to date with the log `log`, of `version`, of the data file whose header says `layout`, eg. with
/// what another process's writer appended since. It takes the changes' write lock only when the log holds more than
/// they have read, since a maintenance run holds the read lock while it runs, and std's lock queues every later reader
/// behind a writer that waits, so a write lock taken for nothing would hold up every search of the file.
pub(crate) fn read_more(changes: &Changes, log: &Path, layout: &Layout, version: Version) -> Result<(), Error> {
    let read = changes.read().log_len as u64;
    let behind = log::open_ours(log, layout, version)?.is_some_and(|(_, len)| len > read);

    if behind {
        catch_up(&mut changes.write(), log, layout, version)?;
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

/// Applies the records that the log at `path`, of `version`, holds past what `delta` has read. Returns whether the log
/// was written for this file. A missing log, or one written for another file, holds nothing for it, and so does one
/// shorter than what `delta` read.
pub(crate) fn catch_up(delta: &mut Delta, path: &Path, layout: &Layout, version: Version) -> Result<bool, Error> {
    let Some((mut file, len)) = log::open_ours(path, layout, version)? else {
        return Ok(false);
    };
    let start = delta.log_len.max(version.header_len());

    if len >= start as u64 {
        let mut bytes = Vec::new();
        file.seek(SeekFrom::Start(start as u64))?;
        file.read_to_end(&mut bytes)?;
        delta.apply_all(&bytes, start)?;
    }

    Ok(true)
}

/// Logs an insert, then makes it visible to every search of `changes`.
pub(crate) fn insert(
    log: &mut Log,
    changes: &Changes,
    id: u64,
    leaf: u32,
    code: &[u8],
    factors: Factors,
    row: &[u8],
) -> Result<(), Error> {
    let record = Record::Insert {
        seq: changes.read().next,
        id,
        list: leaf,
        hides: NONE,
        code,
        factors,
        row,
    };
    log.write(changes, &log::encode(&record, log.version()))
}

/// Logs a delete, then hides the id from every search of `changes`.
pub(crate) fn delete(log: &mut Log, changes: &Changes, id: u64) -> Result<(), Error> {
    let record = Record::Delete {
        seq: changes.read().next,
        id,
        hides: NONE,
    };
    log.write(changes, &log::encode(&record, log.version()))
}

#[cfg(test)]
#[path = "../../tests/unit/update/mod.rs"]
mod tests;
