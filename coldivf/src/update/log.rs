//! The change log of a data file. Each change goes to the end of the log as one record, framed by its length and its
//! checksum, so a crash in the middle of an append leaves at most a damaged last frame.
//!
//! There are two versions:
//! - Version 1 is `name.delta`, the log of an index before its first commit. Its header names its data file by epoch
//!   and header checksum, and its records are numbered by their place, from 1.
//! - Version 2 is the log of a generation. Its header names the index, the generation, the data file's header checksum
//!   and the sequence number of its first record. Each record carries its own sequence number and hints for that
//!   generation, eg. the cluster that an insert was routed to, and the row or insert that its change hides, which the
//!   writer looked up. The first logs of this version left that hint unset, which its record's kind tells.
//!
//! In a version 2 log only a damaged last frame counts as a torn write, which readers ignore and a writer cuts off. A
//! damaged frame that a whole one follows, or a whole frame whose record doesn't fit the data file, fails with
//! `InvalidData`, and nothing is cut off. A frame whose length is a record's, or zeros as a power cut leaves them, ends
//! where its length says, so a whole frame must start there. A frame of any other length has a damaged length, and a
//! whole frame anywhere after it counts. A version 1 log ends at its first record that is cut short, damaged or doesn't
//! fit, as it always has.

use std::fs::{File, OpenOptions};
use std::io::{Error, ErrorKind, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use super::Changes;
use crate::format::Layout;
use crate::io;
use crate::quantization::rabitq::{Factors, code_bytes};
use crate::quantization::{padded_dim, read_f32, sq8};

/// The first bytes of a log.
const MAGIC: [u8; 8] = *b"COLDLOG\0";

/// The size of a version 1 header: the magic, the version, the epoch and the header checksum of the data file it
/// belongs to, and a checksum of those.
pub(crate) const HEADER_BYTES: usize = 8 + 4 + 8 + 4 + 4;

/// The size of a version 2 header: the magic, the version, the index id, the generation, the data file's header
/// checksum, the sequence number of the first record, and a checksum of those.
pub(crate) const HEADER_2_BYTES: usize = 8 + 4 + 8 + 8 + 4 + 8 + 4;

/// The kinds of records.
pub(crate) const INSERT: u8 = 1;
pub(crate) const DELETE: u8 = 2;

/// Set in the kind of a version 2 record whose hint of what its change hides is the writer's lookup, so a hint that
/// names nothing means the id had no visible version. The first version 2 logs wrote records without it, whose hints
/// name nothing whatever the change hides, so a replay looks their ids up.
pub(crate) const LOOKED_UP: u8 = 0x80;

/// A hint that names nothing, eg. the row of a delete of an id that no vector has.
pub(crate) const NONE: u32 = u32::MAX;

/// The bytes before a version 2 record's payload: the kind, the sequence number, the id, the generation of the hints,
/// the cluster and the row that the change hides.
const HEAD_2: usize = 1 + 8 + 8 + 8 + 4 + 4;

/// Which log a data file has.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Version {
    /// `name.delta`, the log of an index before its first commit.
    One,
    /// The log of generation `generation` of the index `index_id`, whose first record has the sequence number `base`.
    Two { index_id: u64, generation: u64, base: u64 },
}

impl Version {
    /// The sequence number of the first change.
    pub(crate) fn base(&self) -> u64 {
        match *self {
            Version::One => 1,
            Version::Two { base, .. } => base,
        }
    }

    pub(crate) fn header_len(&self) -> usize {
        match self {
            Version::One => HEADER_BYTES,
            Version::Two { .. } => HEADER_2_BYTES,
        }
    }

    /// The header of this log for the data file whose header says `layout`.
    pub(crate) fn header(&self, layout: &Layout) -> Vec<u8> {
        let mut h = MAGIC.to_vec();

        match *self {
            Version::One => {
                h.extend(1u32.to_le_bytes());
                h.extend(layout.epoch.to_le_bytes());
                h.extend(layout.check.to_le_bytes());
            }
            Version::Two {
                index_id,
                generation,
                base,
            } => {
                h.extend(2u32.to_le_bytes());
                h.extend(index_id.to_le_bytes());
                h.extend(generation.to_le_bytes());
                h.extend(layout.check.to_le_bytes());
                h.extend(base.to_le_bytes());
            }
        }

        let crc = crc32fast::hash(&h);
        h.extend(crc.to_le_bytes());
        h
    }
}

/// What a version 2 header says, for a log read without its data file, eg. to find the newest whole generation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Header2 {
    pub(crate) index_id: u64,
    pub(crate) generation: u64,
    pub(crate) check: u32,
    pub(crate) base: u64,
}

/// Reads a version 2 header from the start of `bytes`, or `None` when it isn't one.
pub(crate) fn read_header_2(bytes: &[u8]) -> Option<Header2> {
    let h = bytes.get(..HEADER_2_BYTES)?;
    let u64_at = |at: usize| u64::from_le_bytes(h[at..at + 8].try_into().unwrap_or_default());
    let u32_at = |at: usize| u32::from_le_bytes(h[at..at + 4].try_into().unwrap_or_default());
    let whole = h[..8] == MAGIC && u32_at(8) == 2 && crc32fast::hash(&h[..HEADER_2_BYTES - 4]) == u32_at(40);

    whole.then(|| Header2 {
        index_id: u64_at(12),
        generation: u64_at(20),
        check: u32_at(28),
        base: u64_at(32),
    })
}

/// One change, as a record says it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Record<'a> {
    /// An insert of the vector `id`, routed to cluster `list`, with its 1-bit code, its factors and its 8-bit row.
    /// `hides` is the row or the insert that it replaces, by internal id, or `NONE`.
    Insert {
        seq: u64,
        id: u64,
        list: u32,
        hides: u32,
        code: &'a [u8],
        factors: Factors,
        row: &'a [u8],
    },
    /// A delete of the vector `id`, which hides `hides` the same way.
    Delete { seq: u64, id: u64, hides: u32 },
}

impl Record<'_> {
    pub(crate) fn seq(&self) -> u64 {
        match *self {
            Record::Insert { seq, .. } | Record::Delete { seq, .. } => seq,
        }
    }

    pub(crate) fn id(&self) -> u64 {
        match *self {
            Record::Insert { id, .. } | Record::Delete { id, .. } => id,
        }
    }
}

/// The bytes of `record` in a log of `version`. A version 1 record keeps only the kind, the id and an insert's cluster
/// and payload.
pub(crate) fn encode(record: &Record, version: Version) -> Vec<u8> {
    let mut out = Vec::new();
    let (kind, seq, id, list, hides) = match *record {
        Record::Insert {
            seq, id, list, hides, ..
        } => (INSERT, seq, id, list, hides),
        Record::Delete { seq, id, hides } => (DELETE, seq, id, NONE, hides),
    };
    out.push(kind);

    match version {
        Version::One => {
            out.extend(id.to_le_bytes());
            if kind == INSERT {
                out.extend(list.to_le_bytes());
            }
        }
        Version::Two { generation, .. } => {
            out[0] |= LOOKED_UP;
            out.extend(seq.to_le_bytes());
            out.extend(id.to_le_bytes());
            out.extend(generation.to_le_bytes());
            out.extend(list.to_le_bytes());
            out.extend(hides.to_le_bytes());
        }
    }

    if let Record::Insert { code, factors, row, .. } = *record {
        out.extend_from_slice(code);
        [factors.f_add, factors.f_rescale, factors.f_pop, factors.f_err]
            .iter()
            .for_each(|f| out.extend(f.to_le_bytes()));
        out.extend_from_slice(row);
    }

    out
}

/// The change that `bytes` records, in a log of `version` for a data file of `dim` dimensions and `nlist` clusters.
/// `seq` is the sequence number that the change must have, which a version 1 record takes. A record that doesn't fit
/// fails with `InvalidData`: one cut short, one for a cluster that the file lacks, and in a version 2 log one whose
/// sequence number isn't `seq` or whose hints were computed for another generation.
pub(crate) fn decode(bytes: &[u8], version: Version, dim: usize, nlist: usize, seq: u64) -> Result<Record<'_>, Error> {
    let (cb, rb) = (code_bytes(padded_dim(dim)), sq8::row_bytes(dim));
    let u64_at = |at: usize| {
        bytes
            .get(at..at + 8)
            .map_or(0, |b| u64::from_le_bytes(b.try_into().unwrap_or_default()))
    };
    let u32_at = |at: usize| {
        bytes
            .get(at..at + 4)
            .map_or(0, |b| u32::from_le_bytes(b.try_into().unwrap_or_default()))
    };
    // Only called for a record whose length was checked.
    let payload = |at: usize| {
        let f = at + cb;
        (
            &bytes[at..f],
            Factors {
                f_add: read_f32(bytes, f),
                f_rescale: read_f32(bytes, f + 4),
                f_pop: read_f32(bytes, f + 8),
                f_err: read_f32(bytes, f + 12),
            },
            &bytes[f + 16..],
        )
    };
    let record = match (version, bytes.first()) {
        (Version::One, Some(&INSERT)) if bytes.len() == 13 + cb + 16 + rb => {
            let (code, factors, row) = payload(13);
            Record::Insert {
                seq,
                id: u64_at(1),
                list: u32_at(9),
                hides: NONE,
                code,
                factors,
                row,
            }
        }
        (Version::One, Some(&DELETE)) if bytes.len() == 9 => Record::Delete {
            seq,
            id: u64_at(1),
            hides: NONE,
        },
        (Version::Two { generation, .. }, Some(&kind))
            if bytes.len() >= HEAD_2 && u64_at(17) == generation && u64_at(1) == seq =>
        {
            match kind & !LOOKED_UP {
                INSERT if bytes.len() == HEAD_2 + cb + 16 + rb => {
                    let (code, factors, row) = payload(HEAD_2);
                    Record::Insert {
                        seq,
                        id: u64_at(9),
                        list: u32_at(25),
                        hides: u32_at(29),
                        code,
                        factors,
                        row,
                    }
                }
                DELETE if bytes.len() == HEAD_2 => Record::Delete {
                    seq,
                    id: u64_at(9),
                    hides: u32_at(29),
                },
                _ => return Err(damaged()),
            }
        }
        _ => return Err(damaged()),
    };

    if let Record::Insert { list, .. } = record
        && list as usize >= nlist
    {
        return Err(damaged());
    }

    Ok(record)
}

/// Whether `bytes`, a record of a log of `version`, says that its hint of what its change hides is the writer's
/// lookup, which a version 2 record of a later log does.
pub(crate) fn looked_up(bytes: &[u8], version: Version) -> bool {
    version != Version::One && bytes.first().is_some_and(|&kind| kind & LOOKED_UP != 0)
}

/// The sequence number that a version 2 record says it has, or `None` for a record too short to hold one.
pub(crate) fn seq_of(bytes: &[u8]) -> Option<u64> {
    Some(u64::from_le_bytes(bytes.get(1..9)?.try_into().ok()?))
}

/// The id whose vector `bytes`, a record of a log of `version`, changes, or `None` for a record too short to hold one.
pub(crate) fn id_of(bytes: &[u8], version: Version) -> Option<u64> {
    let at = match version {
        Version::One => 1,
        Version::Two { .. } => 9,
    };
    Some(u64::from_le_bytes(bytes.get(at..at + 8)?.try_into().ok()?))
}

pub(crate) fn damaged() -> Error {
    Error::new(ErrorKind::InvalidData, "a log record is damaged")
}

/// A record framed for the log: its length, its checksum, then its bytes.
pub(crate) fn frame(record: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(8 + record.len());
    out.extend((record.len() as u32).to_le_bytes());
    out.extend(crc32fast::hash(record).to_le_bytes());
    out.extend_from_slice(record);
    out
}

/// The record that starts at `at` in `bytes`, if its frame is whole and its checksum matches.
pub(crate) fn next_record(bytes: &[u8], at: usize) -> Option<&[u8]> {
    let len = u32::from_le_bytes(bytes.get(at..at + 4)?.try_into().ok()?) as usize;
    let crc = u32::from_le_bytes(bytes.get(at + 4..at + 8)?.try_into().ok()?);
    let record = bytes.get(at + 8..at.checked_add(8 + len)?)?;

    (len > 0 && crc32fast::hash(record) == crc).then_some(record)
}

/// The whole frames' records in `bytes`, part of a log of `version` for a data file of `dim` dimensions, and where they
/// end. A version 1 log ends at its first frame that is cut short or damaged. A version 2 log ends only at a damaged
/// last frame, and a damaged frame that a whole one follows fails with `InvalidData`.
pub(crate) fn records(bytes: &[u8], version: Version, dim: usize) -> Result<(Vec<&[u8]>, usize), Error> {
    let (mut out, mut at) = (Vec::new(), 0);

    while at < bytes.len() {
        let Some(record) = next_record(bytes, at) else {
            if version != Version::One && followed(bytes, at, dim) {
                return Err(damaged());
            }
            break;
        };
        out.push(record);
        at += 8 + record.len();
    }

    Ok((out, at))
}

/// Whether a whole frame of a record of a version 2 log for a data file of `dim` dimensions comes after the frame at
/// `at` of `bytes`, which isn't whole, so that frame isn't a torn last write. A frame whose length is a record's, or 0
/// as zeros that a power cut left give it, ends where its length says, and the next frame must start there. Any other
/// length is damaged, so the frame may end anywhere, and a whole frame anywhere after it counts.
fn followed(bytes: &[u8], at: usize, dim: usize) -> bool {
    let insert = HEAD_2 + code_bytes(padded_dim(dim)) + 16 + sq8::row_bytes(dim);
    let length = |p: usize| {
        bytes
            .get(p..p + 4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap_or_default()) as usize)
    };
    // Only frames of a record's length count, so the search past a damaged length computes few checksums.
    let whole =
        |p: usize| length(p).is_some_and(|len| len == HEAD_2 || len == insert) && next_record(bytes, p).is_some();

    match length(at) {
        None => false,
        Some(len) if len == 0 || len == HEAD_2 || len == insert => whole(at + 8 + len),
        Some(_) => (at + 1..bytes.len()).any(whole),
    }
}

/// Opens the log at `path` when it was written for the data file whose header says `layout`, as `version` says, and
/// returns it read past its header, with its length. A missing log, one shorter than a header and one written for
/// another file give `None`.
pub(crate) fn open_ours(path: &Path, layout: &Layout, version: Version) -> Result<Option<(File, u64)>, Error> {
    let mut file = match File::open(path) {
        Ok(file) => file,
        Err(e) if e.kind() == ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e),
    };
    let mut head = vec![0u8; version.header_len()];

    match file.read_exact(&mut head) {
        Ok(()) => {}
        Err(e) if e.kind() == ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(e),
    }

    if head != version.header(layout) {
        return Ok(None);
    }

    let len = file.metadata()?.len();
    Ok(Some((file, len)))
}

/// The log of one data file, open for appending, with its writer's lock held.
pub(crate) struct Log {
    pub(super) file: File,
    path: PathBuf,
    version: Version,
    /// The dimension of the data file's vectors, which sets the length of each kind of record.
    dim: usize,
    _lock: Arc<File>,
    /// How many bytes its whole records take, which is where the next one goes.
    len: usize,
    /// Set when a failed write left bytes that couldn't be cut off, so no record may follow them.
    broken: bool,
    /// Set when a commit failed in a way that leaves unknown which generation is current, so no record may go to this
    /// log until the writer opens again.
    poisoned: bool,
}

impl Log {
    /// Opens the log at `path`, of `version`, of the data file whose header says `layout`, for a writer that holds
    /// `lock`. It first brings `changes` up to date with the log. Then it cuts off a torn tail. A version 1 log that is
    /// missing or belongs to another file starts again, and a version 2 one fails with `InvalidData`, since the root
    /// names it. A log that no longer holds what `changes` read from it fails with `InvalidData` too.
    pub(crate) fn open(
        path: &Path,
        version: Version,
        layout: &Layout,
        changes: &Changes,
        lock: Arc<File>,
    ) -> Result<Log, Error> {
        let exists = std::fs::exists(path)?;

        if version != Version::One && !exists {
            return Err(Error::new(
                ErrorKind::InvalidData,
                "the log of the current generation is missing",
            ));
        }

        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)?;

        if !exists {
            io::sync_parent(path)?;
        }

        let mut a = changes.applier();
        let ours = super::catch_up(changes, &mut a, path, layout, version)?;
        let len = file.metadata()?.len();

        if (!ours && a.log_len() > 0) || len < a.log_len() as u64 {
            return Err(Error::new(
                ErrorKind::InvalidData,
                "the log no longer holds the changes read from it",
            ));
        }

        if !ours && version != Version::One {
            return Err(Error::new(
                ErrorKind::InvalidData,
                "the log was written for another data file",
            ));
        }

        if !ours {
            file.set_len(0)?;
            file.write_all(&version.header(layout))?;
            io::sync_file(&file, path)?;
            a.set_log_len(version.header_len());
            changes.publish(&mut a);
        } else if len > a.log_len() as u64 {
            file.set_len(a.log_len() as u64)?;
            io::sync_file(&file, path)?;
        }

        file.seek(SeekFrom::Start(a.log_len() as u64))?;
        Ok(Log {
            file,
            path: path.to_path_buf(),
            version,
            dim: layout.dim,
            _lock: lock,
            len: a.log_len(),
            broken: false,
            poisoned: false,
        })
    }

    /// Makes the new log at `path`, of `version`, for the data file whose header says `layout`, with its header and the
    /// framed `records` in one write, then syncs it fully. Its folder is left to the caller to sync. A file already at
    /// `path` fails with `AlreadyExists`.
    pub(crate) fn create(
        path: &Path,
        version: Version,
        layout: &Layout,
        records: &[Vec<u8>],
        lock: Arc<File>,
    ) -> Result<Log, Error> {
        let mut file = io::create_new(path)?;
        let mut bytes = version.header(layout);
        records.iter().for_each(|r| bytes.extend(frame(r)));
        io::write_all_at(&file, &bytes, 0, path)?;
        io::sync_file(&file, path)?;
        file.seek(SeekFrom::Start(bytes.len() as u64))?;
        Ok(Log {
            file,
            path: path.to_path_buf(),
            version,
            dim: layout.dim,
            _lock: lock,
            len: bytes.len(),
            broken: false,
            poisoned: false,
        })
    }

    pub(crate) fn version(&self) -> Version {
        self.version
    }

    pub(crate) fn path(&self) -> &Path {
        &self.path
    }

    /// How many bytes its whole records take.
    pub(crate) fn len(&self) -> usize {
        self.len
    }

    /// Appends `record` in one write, so the process dying can't leave it half there in the operating system's cache.
    /// When the write fails, it cuts off whatever part of the record landed. Returns where the record ends.
    pub(crate) fn append(&mut self, record: &[u8]) -> Result<usize, Error> {
        self.check()?;

        if let Err(e) = io::write_all_at(&self.file, &frame(record), self.len as u64, &self.path) {
            let back = self.len as u64;
            let cut = self
                .file
                .set_len(back)
                .and_then(|()| self.file.seek(SeekFrom::Start(back)).map(|_| ()));
            self.broken = cut.is_err();
            return Err(e);
        }

        self.len += 8 + record.len();
        Ok(self.len)
    }

    /// Fails when the log takes no more records.
    pub(crate) fn check(&self) -> Result<(), Error> {
        if self.poisoned {
            return Err(Error::other(
                "a commit failed partway, so open the writer again, which finds the generation that is current",
            ));
        }

        if self.broken {
            return Err(Error::other(
                "an earlier write to the log failed, so open the writer again",
            ));
        }

        Ok(())
    }

    /// Makes every record so far survive a power loss.
    pub(crate) fn sync(&mut self) -> Result<(), Error> {
        io::sync_file(&self.file, &self.path)
    }

    /// Refuses every record from now on, after a commit whose outcome is unknown.
    pub(crate) fn poison(&mut self) {
        self.poisoned = true;
    }

    /// The records after byte `at` of the log, which a snapshot of the changes took as their end. `at` must be where a
    /// record starts.
    pub(crate) fn records_after(&mut self, at: usize) -> Result<Vec<Vec<u8>>, Error> {
        let mut bytes = vec![0u8; self.len.saturating_sub(at)];
        let read = self
            .file
            .seek(SeekFrom::Start(at as u64))
            .and_then(|_| self.file.read_exact(&mut bytes));
        // Appends go where the cursor is, so it goes back to the end whatever the read did.
        self.file.seek(SeekFrom::Start(self.len as u64))?;
        read?;
        let (records, used) = records(&bytes, self.version, self.dim)?;

        if used != bytes.len() {
            return Err(damaged());
        }

        Ok(records.into_iter().map(<[u8]>::to_vec).collect())
    }

    /// Appends `record`, then applies it to `changes`, unless an open of the file in this process read it from the log
    /// first.
    pub(crate) fn write(&mut self, changes: &Changes, record: &[u8]) -> Result<(), Error> {
        let end = self.append(record)?;
        changes.apply(record, end)
    }
}

#[cfg(test)]
#[path = "../../tests/unit/update/log.rs"]
mod tests;
