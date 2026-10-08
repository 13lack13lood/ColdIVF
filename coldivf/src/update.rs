//! Inserts and deletes. Each change goes to a log next to the index file, then into the changes that every search of
//! that file in this process reads. Inserted vectors are scanned with every query, apart from the clusters, until a
//! compaction rebuilds the file with them. A delete hides an id from then on.
//!
//! Every change gets the next sequence number, and the rows of the file count as number 0. An insert hides the earlier
//! versions of its id, and a delete hides every version so far. Each id keeps the changes that hid its rows, so a
//! search can tell what was visible when it scanned, even after later changes.
//!
//! A log starts with the epoch and the header checksum of the index file it was written for. An index file's epoch
//! counts its compactions, and its header checksum tells it apart from any other build. So a log from before the last
//! compaction, or one left next to a file built again, is ignored.

use std::collections::HashMap;
use std::fs::{File, OpenOptions, TryLockError};
use std::io::{Error, ErrorKind, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, PoisonError, RwLock, RwLockReadGuard, RwLockWriteGuard, Weak};

use crate::format::Layout;
use crate::quantization::rabitq::{BLOCK, Factors, block_bytes, code_bytes, pack_block};
use crate::quantization::{padded_dim, read_f32, sq8};

/// The first bytes of a log.
const MAGIC: [u8; 8] = *b"COLDLOG\0";
const VERSION: u32 = 1;

/// The size of the log's header, which holds the magic, the version, the epoch and the header checksum of the index
/// file it belongs to, and a checksum of those.
pub(crate) const HEADER_BYTES: usize = 8 + 4 + 8 + 4 + 4;

/// The kinds of log records.
const INSERT: u8 = 1;
const DELETE: u8 = 2;

/// How many vectors were inserted, and the next change's sequence number, when a search scanned the changes.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Seen {
    len: usize,
    next: u64,
}

/// The changes to one index file since it was built.
pub(crate) struct Delta {
    dim: usize,
    /// The file's cluster count, which every insert's cluster must be under.
    nlist: usize,
    /// The inserted vectors' 1-bit codes and factors, and the same packed in blocks of 32 for the scan.
    codes: Vec<u8>,
    factors: Vec<Factors>,
    blocks: Vec<u8>,
    /// Their 8-bit rows, back to back, each with its id.
    rows: Vec<u8>,
    /// Each one's sequence number, and the cluster it is nearest to.
    seqs: Vec<u64>,
    leaves: Vec<u32>,
    /// For each id that changes hid rows of, those changes in order. Each has its sequence number and the first
    /// sequence number it left visible.
    hide: HashMap<u64, Vec<(u64, u64)>>,
    /// The next change's sequence number.
    next: u64,
    /// How many bytes of the log the changes came from, or 0 before a header for this file was read.
    log_len: usize,
}

impl Delta {
    pub(crate) fn new(layout: &Layout) -> Delta {
        Delta {
            dim: layout.dim,
            nlist: layout.nlist,
            codes: Vec::new(),
            factors: Vec::new(),
            blocks: Vec::new(),
            rows: Vec::new(),
            seqs: Vec::new(),
            leaves: Vec::new(),
            hide: HashMap::new(),
            next: 1,
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
    fn apply(&mut self, record: &[u8]) -> Result<(), Error> {
        let d_pad = padded_dim(self.dim);
        let (cb, rb) = (code_bytes(d_pad), sq8::row_bytes(self.dim));
        let id = u64::from_le_bytes(record.get(1..9).ok_or_else(cut)?.try_into().map_err(|_| cut())?);

        match record[0] {
            INSERT if record.len() == 1 + 8 + 4 + cb + 16 + rb => {
                let leaf = u32::from_le_bytes(record[9..13].try_into().map_err(|_| cut())?);
                if leaf as usize >= self.nlist {
                    return Err(cut());
                }
                let code = &record[13..13 + cb];
                let at = 13 + cb;
                let factors = Factors {
                    f_add: read_f32(record, at),
                    f_rescale: read_f32(record, at + 4),
                    f_pop: read_f32(record, at + 8),
                    f_err: read_f32(record, at + 12),
                };
                self.insert(id, leaf, code, factors, &record[at + 16..]);
            }
            DELETE if record.len() == 1 + 8 => self.delete(id),
            _ => return Err(cut()),
        }

        Ok(())
    }

    /// Applies the records in `bytes`, which start at byte `at` of the log, up to the first one that is cut short,
    /// damaged or doesn't fit this file, as a crash in the middle of a write leaves the log.
    fn apply_all(&mut self, bytes: &[u8], at: usize) {
        let mut used = 0;

        while let Some(record) = next_record(bytes, used) {
            if self.apply(record).is_err() {
                break;
            }
            used += 8 + record.len();
        }

        self.log_len = at + used;
    }
}

fn cut() -> Error {
    Error::new(ErrorKind::InvalidData, "a log record is damaged")
}

/// The changes of one index file, shared by every open index and writer of that file in this process.
pub(crate) struct Changes {
    delta: RwLock<Delta>,
}

impl Changes {
    pub(crate) fn read(&self) -> RwLockReadGuard<'_, Delta> {
        self.delta.read().unwrap_or_else(PoisonError::into_inner)
    }

    fn write(&self) -> RwLockWriteGuard<'_, Delta> {
        self.delta.write().unwrap_or_else(PoisonError::into_inner)
    }
}

/// The open files' changes, by path and header, so a rebuilt file at the same path gets its own.
static OPEN: Mutex<Vec<(PathBuf, Layout, Weak<Changes>)>> = Mutex::new(Vec::new());

/// Where the index file at `path` really is, following symbolic links, so every way of naming it shares one log and
/// one lock.
pub(crate) fn real_path(path: &Path) -> PathBuf {
    std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
}

/// The log of the index file at `path`.
pub(crate) fn log_path(path: &Path) -> PathBuf {
    with_suffix(path, ".delta")
}

/// The file that a writer of the index file at `path` locks. It is a file of its own, since on Windows nothing else
/// can read a locked file.
pub(crate) fn lock_path(path: &Path) -> PathBuf {
    with_suffix(path, ".lock")
}

fn with_suffix(path: &Path, suffix: &str) -> PathBuf {
    let mut name = path.as_os_str().to_os_string();
    name.push(suffix);
    PathBuf::from(name)
}

/// Takes the writer's lock of the index file at `path`, so no other writer, in this process or another, can change it
/// until the returned file closes. When another writer holds it, it fails with `WouldBlock`.
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

/// The changes of the index file at `path`, whose header says `layout`, shared with every other open index of that
/// file in this process. Each open reads whatever the log holds past what the changes already have, eg. changes that
/// another process made since an earlier open.
pub(crate) fn changes_for(path: &Path, layout: &Layout) -> Result<Arc<Changes>, Error> {
    let key = real_path(path);
    let log = log_path(&key);
    let changes = {
        let mut open = OPEN.lock().unwrap_or_else(PoisonError::into_inner);
        open.retain(|(_, _, c)| c.strong_count() > 0);
        let found = open
            .iter()
            .find(|(p, l, _)| *p == key && l == layout)
            .and_then(|(_, _, c)| c.upgrade());
        found.unwrap_or_else(|| {
            let changes = Arc::new(Changes {
                delta: RwLock::new(Delta::new(layout)),
            });
            open.push((key, layout.clone(), Arc::downgrade(&changes)));
            changes
        })
    };

    catch_up(&mut changes.write(), &log, layout)?;
    Ok(changes)
}

/// Applies the records that the log at `path` holds past what `delta` has read. Returns whether the log was written for
/// this file. A missing log, or one written for another file, holds nothing for it, and so does one shorter than what
/// `delta` read.
fn catch_up(delta: &mut Delta, path: &Path, layout: &Layout) -> Result<bool, Error> {
    let mut file = match File::open(path) {
        Ok(file) => file,
        Err(e) if e.kind() == ErrorKind::NotFound => return Ok(false),
        Err(e) => return Err(e),
    };
    let mut head = [0u8; HEADER_BYTES];

    match file.read_exact(&mut head) {
        Ok(()) => {}
        Err(e) if e.kind() == ErrorKind::UnexpectedEof => return Ok(false),
        Err(e) => return Err(e),
    }

    let start = delta.log_len.max(HEADER_BYTES);

    if head != header(layout) {
        return Ok(false);
    }

    if file.metadata()?.len() >= start as u64 {
        let mut bytes = Vec::new();
        file.seek(SeekFrom::Start(start as u64))?;
        file.read_to_end(&mut bytes)?;
        delta.apply_all(&bytes, start);
    }

    Ok(true)
}

/// The record that starts at `at` in `bytes`, if it is whole and its checksum matches.
fn next_record(bytes: &[u8], at: usize) -> Option<&[u8]> {
    let len = u32::from_le_bytes(bytes.get(at..at + 4)?.try_into().ok()?) as usize;
    let crc = u32::from_le_bytes(bytes.get(at + 4..at + 8)?.try_into().ok()?);
    let record = bytes.get(at + 8..at.checked_add(8 + len)?)?;

    (len > 0 && crc32fast::hash(record) == crc).then_some(record)
}

/// The log header for the index file whose header says `layout`.
fn header(layout: &Layout) -> [u8; HEADER_BYTES] {
    let mut h = [0u8; HEADER_BYTES];
    h[..8].copy_from_slice(&MAGIC);
    h[8..12].copy_from_slice(&VERSION.to_le_bytes());
    h[12..20].copy_from_slice(&layout.epoch.to_le_bytes());
    h[20..24].copy_from_slice(&layout.check.to_le_bytes());
    let crc = crc32fast::hash(&h[..24]);
    h[24..].copy_from_slice(&crc.to_le_bytes());
    h
}

/// A record framed for the log: its length, its checksum, then its bytes.
fn frame(record: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(8 + record.len());
    out.extend((record.len() as u32).to_le_bytes());
    out.extend(crc32fast::hash(record).to_le_bytes());
    out.extend_from_slice(record);
    out
}

/// The log of one index file, open for appending, with its writer's lock held.
pub(crate) struct Log {
    file: File,
    _lock: File,
    /// How many bytes its whole records take, which is where the next one goes.
    len: usize,
    /// Set when a failed write left bytes that couldn't be cut off, so no record may follow them.
    broken: bool,
}

impl Log {
    /// Opens the log of the index file at `index_path`, whose header says `layout`, for a writer that holds `lock`. It
    /// first brings `changes` up to date with the log. Then it cuts off a damaged tail, or starts the log again when
    /// it is missing or belongs to another file. A log that no longer holds what `changes` read from it fails with
    /// `InvalidData`.
    pub(crate) fn open(index_path: &Path, layout: &Layout, changes: &Changes, lock: File) -> Result<Log, Error> {
        let path = log_path(index_path);
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)?;
        let mut delta = changes.write();
        let ours = catch_up(&mut delta, &path, layout)?;
        let len = file.metadata()?.len();

        if (!ours && delta.log_len > 0) || len < delta.log_len as u64 {
            return Err(Error::new(
                ErrorKind::InvalidData,
                "the log no longer holds the changes read from it",
            ));
        }

        if !ours {
            file.set_len(0)?;
            file.write_all(&header(layout))?;
            file.sync_data()?;
            delta.log_len = HEADER_BYTES;
        } else if len > delta.log_len as u64 {
            file.set_len(delta.log_len as u64)?;
            file.sync_data()?;
        }

        file.seek(SeekFrom::Start(delta.log_len as u64))?;
        Ok(Log {
            file,
            _lock: lock,
            len: delta.log_len,
            broken: false,
        })
    }

    /// Appends `record` in one write, so the process dying can't leave it half there in the operating system's cache.
    /// When the write fails, it cuts off whatever part of the record landed. Returns where the record ends.
    fn append(&mut self, record: &[u8]) -> Result<usize, Error> {
        if self.broken {
            return Err(Error::other(
                "an earlier write to the log failed, so open the writer again",
            ));
        }

        if let Err(e) = self.file.write_all(&frame(record)) {
            let back = self.len as u64;
            let cut = self
                .file
                .set_len(back)
                .and_then(|()| self.file.seek(SeekFrom::Start(back)));
            self.broken = cut.is_err();
            return Err(e);
        }

        self.len += 8 + record.len();
        Ok(self.len)
    }

    pub(crate) fn sync(&mut self) -> Result<(), Error> {
        self.file.sync_data()
    }

    /// Starts the log again for the rebuilt index file whose header says `layout`, and whose changes are `changes`.
    pub(crate) fn restart(&mut self, layout: &Layout, changes: &Changes) -> Result<(), Error> {
        let mut delta = changes.write();
        self.broken = true;
        self.file.set_len(0)?;
        self.file.seek(SeekFrom::Start(0))?;
        self.file.write_all(&header(layout))?;
        self.file.sync_data()?;
        (self.len, self.broken, delta.log_len) = (HEADER_BYTES, false, HEADER_BYTES);
        Ok(())
    }

    /// Appends `record`, then applies it to `changes`, unless an open of the file in this process read it from the log
    /// first.
    fn write(&mut self, changes: &Changes, record: &[u8]) -> Result<(), Error> {
        let end = self.append(record)?;
        let mut delta = changes.write();

        if delta.log_len < end {
            delta.apply(record)?;
            delta.log_len = end;
        }

        Ok(())
    }
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
    let mut record = Vec::with_capacity(1 + 8 + 4 + code.len() + 16 + row.len());
    record.push(INSERT);
    record.extend(id.to_le_bytes());
    record.extend(leaf.to_le_bytes());
    record.extend_from_slice(code);
    [factors.f_add, factors.f_rescale, factors.f_pop, factors.f_err]
        .iter()
        .for_each(|f| record.extend(f.to_le_bytes()));
    record.extend_from_slice(row);
    log.write(changes, &record)
}

/// Logs a delete, then hides the id from every search of `changes`.
pub(crate) fn delete(log: &mut Log, changes: &Changes, id: u64) -> Result<(), Error> {
    let mut record = vec![DELETE];
    record.extend(id.to_le_bytes());
    log.write(changes, &record)
}

#[cfg(test)]
#[path = "../tests/unit/update.rs"]
mod tests;
