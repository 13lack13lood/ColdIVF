//! What searches see of a data file's changes: one snapshot of every change up to a sequence number, which a search
//! loads without a lock and keeps until it returns. The writer, or an open that reads what another process logged,
//! builds the next snapshot from the last one and swaps it in. Pending inserts sit in pieces of 32 in place order,
//! which snapshots share, and by cluster in groups for the scan. A change that hides a row of the data file or an
//! insert stamps it with its sequence number, in memory that every snapshot shares, so each search judges a row as of
//! its own snapshot.

use std::collections::HashMap;
use std::fs::File;
use std::io::Error;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

use super::log::{self, NONE, Record, Version};
use super::view::{ClusterGroups, Groups};
use crate::format::{Layout, Part};
use crate::io::{self, Span};
use crate::quantization::rabitq::{BLOCK, Factors, code_bytes};
use crate::quantization::{padded_dim, sq8};

/// A stamp that hides nothing.
const NEVER: u64 = u64::MAX;

/// Up to 32 pending inserts in place order. Snapshots share pieces, and a piece that gains an insert while a snapshot
/// holds it is copied, but every copy shares the stamps that hide its inserts.
struct Piece {
    codes: Vec<u8>,
    factors: Vec<Factors>,
    rows: Vec<u8>,
    seqs: Vec<u64>,
    lists: Vec<u32>,
    hidden: Arc<[AtomicU64; BLOCK]>,
}

impl Piece {
    fn new() -> Piece {
        Piece {
            codes: Vec::new(),
            factors: Vec::new(),
            rows: Vec::new(),
            seqs: Vec::new(),
            lists: Vec::new(),
            hidden: Arc::new(std::array::from_fn(|_| AtomicU64::new(NEVER))),
        }
    }

    fn len(&self) -> usize {
        self.seqs.len()
    }

    /// A copy with one more insert, which shares the stamps.
    fn with(&self, code: &[u8], factors: Factors, row: &[u8], seq: u64, list: u32) -> Piece {
        let mut p = Piece {
            codes: [&self.codes[..], code].concat(),
            factors: self.factors.clone(),
            rows: [&self.rows[..], row].concat(),
            seqs: self.seqs.clone(),
            lists: self.lists.clone(),
            hidden: Arc::clone(&self.hidden),
        };
        p.factors.push(factors);
        p.seqs.push(seq);
        p.lists.push(list);
        p
    }

    /// Adds one more insert in place.
    fn push(&mut self, code: &[u8], factors: Factors, row: &[u8], seq: u64, list: u32) {
        self.codes.extend_from_slice(code);
        self.factors.push(factors);
        self.rows.extend_from_slice(row);
        self.seqs.push(seq);
        self.lists.push(list);
    }
}

/// The rows of the data file that changes hid: a bit for each row that a change may have hidden, and each such row's
/// stamp, the hiding change's sequence number times 2, plus 1 when an insert of the same id hid it.
pub(crate) struct Marks {
    bits: Box<[AtomicU64]>,
    at: Box<[AtomicU64]>,
}

impl Marks {
    fn new(count: usize) -> Marks {
        Marks {
            bits: (0..count.div_ceil(64)).map(|_| AtomicU64::new(0)).collect(),
            at: (0..count).map(|_| AtomicU64::new(NEVER)).collect(),
        }
    }

    /// The stamp of row `row`, or `NEVER` for one that no change hid.
    #[inline]
    fn stamp(&self, row: u32) -> u64 {
        let (word, bit) = (row as usize / 64, row % 64);

        match self.bits.get(word) {
            Some(w) if w.load(Ordering::Acquire) >> bit & 1 == 1 => self.at[row as usize].load(Ordering::Acquire),
            _ => NEVER,
        }
    }

    /// Whether a change before sequence number `next` hid row `row`.
    #[inline]
    pub(crate) fn hidden(&self, row: u32, next: u64) -> bool {
        let s = self.stamp(row);
        s != NEVER && s >> 1 < next
    }

    /// Whether an insert of the same id before `next` hid row `row`.
    fn replaced(&self, row: u32, next: u64) -> bool {
        self.hidden(row, next) && self.stamp(row) & 1 == 1
    }

    /// Hides row `row` from sequence number `seq` on. The stamp lands before the bit, so a search that sees the bit
    /// sees the stamp.
    fn hide(&self, row: u32, seq: u64, by_insert: bool) {
        self.at[row as usize].store(seq * 2 + u64::from(by_insert), Ordering::Release);
        self.bits[row as usize / 64].fetch_or(1 << (row % 64), Ordering::Release);
    }
}

/// What a search that scans the inserted vectors sees: how many there were, visible or not, and the sequence number
/// of the first change it doesn't see.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Seen {
    len: usize,
    next: u64,
}

impl Seen {
    /// What a search of a snapshot with `inserts` inserts, and `next` as the sequence number of the first change it
    /// doesn't hold, sees, eg. a maintenance job's snapshot as its resume file records it.
    pub(crate) fn at(inserts: usize, next: u64) -> Seen {
        Seen { len: inserts, next }
    }

    /// How many vectors were inserted, visible or not.
    pub(crate) fn inserts(&self) -> usize {
        self.len
    }

    /// The sequence number of the first change it doesn't see.
    pub(crate) fn next(&self) -> u64 {
        self.next
    }
}

/// One snapshot of a data file's changes: every change before `next`. It never changes, apart from the stamps of later
/// changes, which a search of it ignores.
pub(crate) struct Delta {
    dim: usize,
    /// The data file's rows, whose count is where the internal ids of inserts start.
    count: usize,
    next: u64,
    len: usize,
    /// How many bytes of the log the changes came from, which is where a change made next starts.
    log_len: usize,
    /// How many changes hid a row or an insert, and how many of them an insert.
    hides: usize,
    insert_hides: usize,
    /// The pieces that the last rebuild of the groups sealed, then those since.
    sealed: Arc<Vec<Arc<Piece>>>,
    recent: Vec<Arc<Piece>>,
    groups: Groups,
    marks: Option<Arc<Marks>>,
}

impl Delta {
    /// How many vectors were inserted, visible or not.
    pub(crate) fn len(&self) -> usize {
        self.len
    }

    fn piece(&self, i: usize) -> (&Piece, usize) {
        piece_of(&self.sealed, &self.recent, i)
    }

    /// Inserted vector i's 8-bit row.
    pub(crate) fn row(&self, i: usize) -> &[u8] {
        let ((piece, j), rb) = (self.piece(i), sq8::row_bytes(self.dim));
        &piece.rows[j * rb..(j + 1) * rb]
    }

    pub(crate) fn seq(&self, i: usize) -> u64 {
        let (piece, j) = self.piece(i);
        piece.seqs[j]
    }

    /// The cluster that inserted vector i was routed to.
    pub(crate) fn leaf(&self, i: usize) -> u32 {
        let (piece, j) = self.piece(i);
        piece.lists[j]
    }

    /// Inserted vector i's 1-bit code and factors.
    pub(crate) fn lane(&self, i: usize) -> (&[u8], Factors) {
        let ((piece, j), cb) = (self.piece(i), code_bytes(padded_dim(self.dim)));
        (&piece.codes[j * cb..(j + 1) * cb], piece.factors[j])
    }

    /// How many bytes of the log the changes came from.
    pub(crate) fn log_len(&self) -> usize {
        self.log_len
    }

    /// The sequence number of the first change it doesn't hold.
    pub(crate) fn next(&self) -> u64 {
        self.next
    }

    /// What a search of this snapshot sees.
    pub(crate) fn seen(&self) -> Seen {
        Seen {
            len: self.len,
            next: self.next,
        }
    }

    /// The groups of cluster `list`'s inserts, for a scan of the cluster.
    pub(crate) fn groups(&self, list: u32) -> ClusterGroups<'_> {
        self.groups.of(list)
    }

    /// The bytes of the sorted view of the inserts, and of the groups of those since its last rebuild.
    #[cfg(test)]
    pub(crate) fn view_bytes(&self) -> (usize, usize) {
        self.groups.bytes()
    }

    /// The rows of the data file that changes hid, if any did.
    pub(crate) fn marks(&self) -> Option<&Marks> {
        self.marks.as_deref()
    }

    /// Whether no change has hidden anything, so every row and every insert is visible.
    pub(crate) fn hides_nothing(&self) -> bool {
        self.hides == 0
    }

    /// Whether inserted vector i was visible as of `seen`. When no change before this snapshot's end hid an insert,
    /// none before an earlier end did either, so it needn't read the stamp.
    #[inline]
    pub(crate) fn insert_visible(&self, i: usize, seen: Seen) -> bool {
        if i >= seen.len {
            return false;
        }

        if self.insert_hides == 0 && seen.next <= self.next {
            return true;
        }

        let (piece, j) = self.piece(i);
        piece.hidden[j].load(Ordering::Acquire) >= seen.next
    }

    /// Whether row `row` of the data file was visible as of `seen`.
    pub(crate) fn row_visible(&self, row: u32, seen: Seen) -> bool {
        self.marks.as_ref().is_none_or(|m| !m.hidden(row, seen.next))
    }

    /// Whether an insert of the same id hid row `row` of the data file as of `seen`.
    pub(crate) fn row_replaced(&self, row: u32, seen: Seen) -> bool {
        self.marks.as_ref().is_some_and(|m| m.replaced(row, seen.next))
    }

    /// Whether the candidate with internal id `id` was visible as of `seen`.
    pub(crate) fn visible(&self, id: u32, seen: Seen) -> bool {
        match (id as usize).checked_sub(self.count) {
            Some(i) => self.insert_visible(i, seen),
            None => self.row_visible(id, seen),
        }
    }
}

/// Insert i's piece among `sealed` and then `recent`, and its lane in the piece.
fn piece_of<'a>(sealed: &'a [Arc<Piece>], recent: &'a [Arc<Piece>], i: usize) -> (&'a Piece, usize) {
    let p = i / BLOCK;
    let piece = match sealed.get(p) {
        Some(piece) => piece,
        None => &recent[p - sealed.len()],
    };
    (piece, i % BLOCK)
}

/// The ids of a data file's rows, by row, and its rows in the order of their ids, which a change that names a row by
/// its id looks up.
struct Ids {
    by_row: Vec<u64>,
    sorted: Vec<u32>,
}

/// What builds a data file's snapshots: the writer, as it logs each change, or an open that reads the records another
/// process logged. Only one applies at a time, and no search touches it.
pub(crate) struct Applier {
    version: Version,
    dim: usize,
    nlist: usize,
    count: usize,
    /// The data file, opened the first time a change checks one of its ids, and where its ids are, which it reads
    /// whole the first time a change must find a row by its id.
    data: PathBuf,
    file: Option<File>,
    ids_at: Span,
    ids: Option<Ids>,
    /// The latest insert of each id that has one, by place.
    latest: HashMap<u64, u32>,
    /// The next snapshot as it builds: everything but the groups, which `lanes` holds back during a replay.
    next: u64,
    len: usize,
    log_len: usize,
    hides: usize,
    insert_hides: usize,
    sealed: Arc<Vec<Arc<Piece>>>,
    recent: Vec<Arc<Piece>>,
    groups: Groups,
    /// The lanes of the inserts that a replay applied and the groups don't hold yet, as places.
    held: Vec<u32>,
    marks: Arc<OnceLock<Arc<Marks>>>,
}

impl Applier {
    /// No change yet to the data file at `data`, whose header says `layout`, with its log of `version`.
    pub(crate) fn new(layout: &Layout, version: Version, data: &Path) -> Applier {
        Applier {
            version,
            dim: layout.dim,
            nlist: layout.nlist,
            count: layout.count,
            data: data.to_path_buf(),
            file: None,
            ids_at: layout.span(Part::Ids),
            ids: None,
            latest: HashMap::new(),
            next: version.base(),
            len: 0,
            log_len: 0,
            hides: 0,
            insert_hides: 0,
            sealed: Arc::new(Vec::new()),
            recent: Vec::new(),
            groups: Groups::new(layout.nlist, layout.dim),
            held: Vec::new(),
            marks: Arc::new(OnceLock::new()),
        }
    }

    pub(crate) fn version(&self) -> Version {
        self.version
    }

    pub(crate) fn log_len(&self) -> usize {
        self.log_len
    }

    pub(crate) fn set_log_len(&mut self, len: usize) {
        self.log_len = len;
    }

    /// The sequence number that the next change gets.
    pub(crate) fn next(&self) -> u64 {
        self.next
    }

    /// The snapshot of every change applied so far. During a replay it first packs the held inserts into groups.
    pub(crate) fn snapshot(&mut self) -> Delta {
        if !self.held.is_empty() {
            let (held, cb) = (std::mem::take(&mut self.held), code_bytes(padded_dim(self.dim)));
            let (sealed, recent) = (&self.sealed, &self.recent);
            let lanes = held.iter().map(|&i| {
                let (piece, j) = piece_of(sealed, recent, i as usize);
                (piece.lists[j], &piece.codes[j * cb..(j + 1) * cb], piece.factors[j], i)
            });
            let groups = self.groups.with_many(lanes);
            self.groups = groups;
        }

        self.seal();
        Delta {
            dim: self.dim,
            count: self.count,
            next: self.next,
            len: self.len,
            log_len: self.log_len,
            hides: self.hides,
            insert_hides: self.insert_hides,
            sealed: Arc::clone(&self.sealed),
            recent: self.recent.clone(),
            groups: self.groups.clone(),
            marks: self.marks.get().cloned(),
        }
    }

    /// Moves the full recent pieces among the sealed ones once there are `REBUILD` inserts' worth of them.
    fn seal(&mut self) {
        let full = self.recent.iter().take_while(|p| p.len() == BLOCK).count();

        if full * BLOCK >= super::view::REBUILD {
            let mut sealed = self.sealed.as_ref().clone();
            sealed.extend(self.recent.drain(..full));
            self.sealed = Arc::new(sealed);
        }
    }

    fn piece(&self, i: usize) -> (&Piece, usize) {
        piece_of(&self.sealed, &self.recent, i)
    }

    /// Where the visible version of `id` is, as an internal id: an insert's, or a row's of the data file, or `NONE`.
    /// It reads the data file's ids the first time it looks for a row.
    pub(crate) fn locate(&mut self, id: u64) -> Result<u32, Error> {
        if let Some(&i) = self.latest.get(&id) {
            let (piece, j) = self.piece(i as usize);
            if piece.hidden[j].load(Ordering::Relaxed) == NEVER {
                return Ok(self.count as u32 + i);
            }
            return Ok(NONE);
        }

        let ids = self.ids()?;
        let at = ids.sorted.partition_point(|&r| ids.by_row[r as usize] < id);
        let row = ids.sorted.get(at).copied().filter(|&r| ids.by_row[r as usize] == id);
        let hidden = |r: u32| self.marks.get().is_some_and(|m| m.stamp(r) != NEVER);

        Ok(row.filter(|&r| !hidden(r)).unwrap_or(NONE))
    }

    /// The data file, opened the first time.
    fn file(&mut self) -> Result<&File, Error> {
        if self.file.is_none() {
            self.file = Some(File::open(&self.data)?);
        }

        Ok(self.file.as_ref().expect("the file was just opened"))
    }

    /// The id of row `row` of the data file.
    fn id_of(&mut self, row: u32) -> Result<u64, Error> {
        if let Some(ids) = &self.ids {
            return Ok(ids.by_row[row as usize]);
        }

        let mut id = [0u8; 8];
        let at = self.ids_at.offset + 8 * row as usize;
        io::read_at(self.file()?, &mut id, at)?;
        Ok(u64::from_le_bytes(id))
    }

    /// The data file's ids, read the first time.
    fn ids(&mut self) -> Result<&Ids, Error> {
        if self.ids.is_none() {
            let (mut bytes, at) = (vec![0u8; 8 * self.count], self.ids_at.offset);
            io::read_at(self.file()?, &mut bytes, at)?;
            let by_row: Vec<u64> = bytes
                .as_chunks::<8>()
                .0
                .iter()
                .map(|b| u64::from_le_bytes(*b))
                .collect();
            let mut sorted: Vec<u32> = (0..self.count as u32).collect();
            sorted.sort_unstable_by_key(|&r| by_row[r as usize]);
            self.ids = Some(Ids { by_row, sorted });
        }

        Ok(self.ids.as_ref().expect("the ids were just read"))
    }

    /// Whether `hint`, a record's guess at where the visible version of `id` is, names it now. A guess of an insert
    /// is checked by the id in its row, and one of a row of the data file by the file's ids.
    fn checks(&mut self, id: u64, hint: u32) -> Result<bool, Error> {
        match (hint as usize).checked_sub(self.count) {
            Some(i) if i < self.len => {
                let (piece, j) = self.piece(i);
                let rb = sq8::row_bytes(self.dim);
                let fits = sq8::id(&piece.rows[j * rb..(j + 1) * rb], self.dim) == id;
                Ok(fits && piece.hidden[j].load(Ordering::Relaxed) == NEVER)
            }
            Some(_) => Ok(false),
            None => {
                let hidden = self.marks.get().is_some_and(|m| m.stamp(hint) != NEVER);
                Ok(!hidden && self.id_of(hint)? == id)
            }
        }
    }

    /// Hides `at`, the visible version of an id as an internal id, from sequence number `seq` on, for an insert of the
    /// same id when `by_insert` is set.
    fn hide(&mut self, at: u32, seq: u64, by_insert: bool) {
        if at == NONE {
            return;
        }

        self.hides += 1;

        match (at as usize).checked_sub(self.count) {
            Some(i) => {
                let (piece, j) = self.piece(i);
                piece.hidden[j].store(seq, Ordering::Release);
                self.insert_hides += 1;
            }
            None => {
                let count = self.count;
                let marks = self.marks.get_or_init(|| Arc::new(Marks::new(count)));
                marks.hide(at, seq, by_insert);
            }
        }
    }

    /// Applies one record of the log. With `hold`, the inserts wait for `snapshot` to join the groups, so a replay of
    /// many packs each block once. Where a record's hint names the visible version of its id, it goes by the hint, and
    /// otherwise it looks the id up. A record that doesn't fit the file fails with `InvalidData`.
    pub(crate) fn apply(&mut self, bytes: &[u8], hold: bool) -> Result<(), Error> {
        let record = log::decode(bytes, self.version, self.dim, self.nlist, self.next)?;
        let (id, hint) = match record {
            Record::Insert { id, hides, .. } | Record::Delete { id, hides, .. } => (id, hides),
        };
        // A hint that names nothing stands when it is the writer's lookup. Version 1 records and the first version 2
        // ones have no lookup.
        let at = if hint == NONE && log::looked_up(bytes, self.version) {
            NONE
        } else if hint != NONE && self.checks(id, hint)? {
            hint
        } else {
            self.locate(id)?
        };
        let seq = self.next;

        match record {
            Record::Insert {
                code,
                factors,
                row,
                list,
                ..
            } => {
                self.hide(at, seq, true);
                self.push(id, list, code, factors, row, hold);
            }
            Record::Delete { .. } => self.hide(at, seq, false),
        }

        self.next += 1;
        Ok(())
    }

    /// Applies the records in `bytes`, which start at byte `at` of the log, holding their inserts for `snapshot`, so a
    /// replay packs each block once. A version 1 log ends at its first record that is cut short, damaged or doesn't
    /// fit this file, as a crash in the middle of a write leaves it. A version 2 log ends only at a damaged last frame,
    /// and any other damage fails with `InvalidData`. Either way it keeps the records before the end or the damage.
    pub(crate) fn replay(&mut self, bytes: &[u8], at: usize) -> Result<(), Error> {
        let (records, _) = log::records(bytes, self.version, self.dim)?;
        let mut used = 0;

        for record in records {
            if let Err(e) = self.apply(record, true) {
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

    /// Adds an insert at the next place. The last piece takes it in place when no snapshot holds that piece, eg. in a
    /// replay, which takes none until it ends, and otherwise a copy of it does, so each snapshot keeps the piece it saw.
    fn push(&mut self, id: u64, list: u32, code: &[u8], factors: Factors, row: &[u8], hold: bool) {
        let place = self.len as u32;
        let seq = self.next;
        let last = match self.recent.last() {
            Some(p) if p.len() < BLOCK => self.recent.pop(),
            _ => None,
        };
        let mut piece = last.unwrap_or_else(|| Arc::new(Piece::new()));
        match Arc::get_mut(&mut piece) {
            Some(p) => p.push(code, factors, row, seq, list),
            None => piece = Arc::new(piece.with(code, factors, row, seq, list)),
        }
        self.recent.push(piece);
        self.len += 1;
        self.latest.insert(id, place);

        match hold {
            true => self.held.push(place),
            false => self.groups = self.groups.with(list, code, factors, place),
        }
    }
}

#[cfg(test)]
#[path = "../../tests/unit/update/delta.rs"]
mod tests;
