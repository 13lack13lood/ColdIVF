//! Cold search, which reads only what a query needs. A fetch reads the chosen clusters' blocks in one wave. A search
//! scans each range as it arrives, then reads its best candidates' rows in a second, small wave.

use std::fs::File;
use std::io::Error;
use std::ptr;
use std::sync::{Arc, Weak};

use super::candidates::Candidate;
use super::scan::scan_list;
use super::{List, Scratch, View};
use crate::format::{Layout, Part};
use crate::index::Inner;
use crate::io::{Range, Span, Wave, arena_bound, coalesce};
use crate::quantization::rabitq::{BLOCK, block_bytes};
use crate::quantization::{padded_dim, sq8};
use crate::{LeafId, Limits};

/// The backend lane of the cluster wave and the late row wave, which never overlap.
const MAIN: usize = 0;

/// The backend lane of the early row wave.
const EARLY: usize = 1;

/// One search context's reads of one index. Its arenas are sized once from the context's limits, so a query allocates
/// nothing.
pub(crate) struct Reads {
    /// The index it reads. Holding it keeps the index's allocation alive, so no later index can take its address.
    owner: Weak<Inner>,
    wave: Box<dyn Wave>,
    page: usize,
    file_len: usize,
    /// Where the codes section starts.
    codes: usize,
    /// The size of one block of 32 codes.
    block: usize,
    rows: RowSection,
    /// The fetched clusters' blocks, one span per cluster that has vectors.
    clusters: Batch,
    /// Whether every range of `clusters` is in.
    clusters_in: bool,
    /// Whether a read of `clusters` failed, which leaves them incomplete.
    failed: bool,
    /// The row pages read early for a guess at the query, and whether that wave may still be arriving.
    early: Batch,
    early_pending: bool,
    /// Whether the last `start_rows` needs some of the early rows.
    early_used: bool,
    /// The row pages that the final query needs and the early wave didn't read.
    late: Batch,
}

impl Reads {
    /// Reads of the index `owner`, with the file layout `layout` and the clusters `lists`, through `wave`.
    pub(crate) fn new(
        owner: &Arc<Inner>,
        wave: Box<dyn Wave>,
        layout: &Layout,
        file_len: usize,
        lists: &[List],
        limits: &Limits,
    ) -> Reads {
        let page = wave.page();
        let block = block_bytes(padded_dim(layout.dim));
        let rows = RowSection::new(layout);
        let keep = limits.max_rerank.max(limits.max_k);
        // The arena must hold the largest clusters that one fetch may read.
        let mut lens: Vec<usize> = lists.iter().map(|l| l.count.div_ceil(BLOCK) * block).collect();
        let most = limits.max_nprobe.min(lens.len());

        if most < lens.len() {
            lens.select_nth_unstable_by(most, |a, b| b.cmp(a));
        }

        let row_arena = keep * rows.room(page);
        Reads {
            owner: Arc::downgrade(owner),
            wave,
            page,
            file_len,
            codes: layout.span(Part::Codes).offset,
            block,
            rows,
            clusters: Batch::new(limits.max_nprobe, arena_bound(lens[..most].iter().copied(), page)),
            clusters_in: true,
            failed: false,
            early: Batch::new(keep, row_arena),
            early_pending: false,
            early_used: false,
            late: Batch::new(keep, row_arena),
        }
    }

    /// Starts reading the blocks of `leaves`, each of which must be under the cluster count. A cluster without vectors
    /// needs no read.
    pub(crate) fn fetch(&mut self, file: &File, lists: &[List], leaves: &[LeafId]) -> Result<(), Error> {
        (self.clusters_in, self.failed, self.early_pending) = (false, true, false);
        self.early.clear();
        self.late.clear();
        let c = &mut self.clusters;
        c.keys.clear();
        c.keys.extend_from_slice(leaves);
        c.keys.sort_unstable();
        c.keys.dedup();
        c.keys.retain(|&l| lists[l as usize].count > 0);
        c.spans.clear();
        c.spans.extend(c.keys.iter().map(|&l| {
            let list = lists[l as usize];
            Span {
                offset: self.codes + list.offset,
                len: list.count.div_ceil(BLOCK) * self.block,
            }
        }));
        coalesce(&c.spans, self.page, self.file_len, &mut c.ranges, &mut c.at);
        self.wave.submit(MAIN, file, &c.ranges, &mut c.arena)?;
        (self.clusters_in, self.failed) = (c.ranges.is_empty(), false);
        Ok(())
    }

    /// Forgets every fetch and row wave, eg. before a search that scans its clusters in place and reads only rows.
    pub(crate) fn forget(&mut self) {
        (self.clusters_in, self.failed, self.early_pending) = (true, false, false);
        self.clusters.clear();
        self.early.clear();
        self.late.clear();
    }

    /// Scans every fetched cluster for the query in `s`. While the fetch is still arriving, it scans each range as it
    /// comes in.
    pub(crate) fn scan(&mut self, file: &File, view: &View, s: &mut Scratch) -> Result<(), Error> {
        if self.failed {
            return Err(Error::other("an earlier read of these clusters failed"));
        }

        let c = &mut self.clusters;

        if self.clusters_in {
            (0..c.ranges.len()).for_each(|i| scan_range(c, i, view, s));
            return Ok(());
        }

        self.failed = true;
        let mut arrived = 0;

        while let Some(i) = self.wave.next(MAIN, file, &c.ranges, &mut c.arena)? {
            scan_range(c, i, view, s);
            arrived += 1;
        }

        if arrived != c.ranges.len() {
            return Err(lost());
        }

        (self.clusters_in, self.failed) = (true, false);
        Ok(())
    }

    /// Starts reading the rows of `candidates`, eg. the best candidates for a guess at the query.
    pub(crate) fn prefetch(&mut self, file: &File, candidates: &[Candidate]) -> Result<(), Error> {
        self.early_pending = false;
        let e = &mut self.early;
        e.keys.clear();
        e.keys.extend(self.rows.pages(candidates));
        e.keys.sort_unstable();
        e.keys.dedup();
        self.rows.plan(e, self.page, self.file_len);

        if let Err(err) = self.wave.submit(EARLY, file, &e.ranges, &mut e.arena) {
            e.clear();
            return Err(err);
        }

        self.early_pending = true;
        Ok(())
    }

    /// Reads the rows of `candidates`. It starts reading the rows that the early wave didn't before it waits for the
    /// early rows, and it skips the wait when none of them are needed.
    pub(crate) fn read_rows(&mut self, file: &File, candidates: &[Candidate]) -> Result<(), Error> {
        self.start_rows(file, candidates)?;
        self.finish_rows(file)
    }

    /// Starts reading the rows of `candidates` that the early wave didn't, so other reads can start before the wait.
    pub(crate) fn start_rows(&mut self, file: &File, candidates: &[Candidate]) -> Result<(), Error> {
        let (e, l) = (&mut self.early, &mut self.late);
        l.keys.clear();
        l.keys.extend(self.rows.pages(candidates));
        l.keys.sort_unstable();
        l.keys.dedup();
        let needed = l.keys.len();
        l.keys.retain(|p| e.keys.binary_search(p).is_err());
        self.rows.plan(l, self.page, self.file_len);
        self.early_used = l.keys.len() < needed;
        self.wave.submit(MAIN, file, &l.ranges, &mut l.arena)
    }

    /// Waits for the rows that `start_rows` started, and for the early rows when some of them are needed.
    pub(crate) fn finish_rows(&mut self, file: &File) -> Result<(), Error> {
        if self.early_used && self.early_pending {
            drain(&mut *self.wave, EARLY, file, &mut self.early)?;
        }

        self.early_pending = false;
        drain(&mut *self.wave, MAIN, file, &mut self.late)
    }

    /// The row of internal id `id`, which one of the row waves read.
    pub(crate) fn row(&self, id: u32) -> &[u8] {
        let page = self.rows.page_of(id);
        let (b, j) = match self.early.keys.binary_search(&page) {
            Ok(j) => (&self.early, j),
            Err(_) => {
                let j = self.late.keys.binary_search(&page);
                (
                    &self.late,
                    j.expect("the late wave reads every page that the early one didn't"),
                )
            }
        };
        let at = b.at[j] + (id as usize % self.rows.per_page) * self.rows.row_bytes;
        &b.arena[at..at + self.rows.row_bytes]
    }
}

/// The slot in `slots` of the reads of `index`, made with `make` if there isn't one yet. Making one first drops the
/// reads of indexes that have closed.
pub(crate) fn reads_for(slots: &mut Vec<Reads>, index: &Arc<Inner>, make: impl FnOnce() -> Reads) -> usize {
    if let Some(i) = slots.iter().position(|r| ptr::eq(r.owner.as_ptr(), Arc::as_ptr(index))) {
        return i;
    }

    slots.retain(|r| r.owner.strong_count() > 0);
    slots.push(make());
    slots.len() - 1
}

/// One wave's plan. Span j is the cluster or row page `keys[j]`, and its bytes land at `arena[at[j]..]`.
struct Batch {
    keys: Vec<u32>,
    spans: Vec<Span>,
    at: Vec<usize>,
    ranges: Vec<Range>,
    arena: Vec<u8>,
}

impl Batch {
    fn new(spans: usize, bytes: usize) -> Batch {
        Batch {
            keys: Vec::with_capacity(spans),
            spans: Vec::with_capacity(spans),
            at: Vec::with_capacity(spans),
            ranges: Vec::with_capacity(spans),
            arena: vec![0; bytes],
        }
    }

    fn clear(&mut self) {
        self.keys.clear();
        self.spans.clear();
        self.at.clear();
        self.ranges.clear();
    }
}

/// Where the rows section starts, and how it packs rows into pages.
#[derive(Clone, Copy)]
struct RowSection {
    /// How many rows the file holds. Candidates past them are inserts, whose rows are in memory.
    count: usize,
    offset: usize,
    row_bytes: usize,
    per_page: usize,
    page_size: usize,
}

impl RowSection {
    fn new(layout: &Layout) -> RowSection {
        RowSection {
            count: layout.count,
            offset: layout.span(Part::Rows).offset,
            row_bytes: sq8::row_bytes(layout.dim),
            per_page: sq8::rows_per_page(layout.dim, layout.page_size),
            page_size: layout.page_size,
        }
    }

    fn page_of(&self, id: u32) -> u32 {
        id / self.per_page as u32
    }

    /// The pages of the rows of `candidates` that the file holds.
    fn pages<'a>(&'a self, candidates: &'a [Candidate]) -> impl Iterator<Item = u32> + 'a {
        let file = candidates.iter().filter(|c| (c.id as usize) < self.count);
        file.map(|c| self.page_of(c.id))
    }

    /// The most arena bytes that one row page takes once its read rounds out to `page` boundaries. Row pages sit at
    /// multiples of their size from the start of the section, so when the section starts on a multiple of `page`, a
    /// row page starts at most page − g into a read page, where g is the largest power of 2 that divides both sizes.
    fn room(&self, page: usize) -> usize {
        if !page.is_power_of_two() || !self.offset.is_multiple_of(page) {
            return self.page_size + 2 * page;
        }

        let g = (self.page_size & self.page_size.wrapping_neg()).min(page);
        (page - g + self.page_size).div_ceil(page) * page
    }

    /// Lays out the pages in `b.keys` as spans and plans their reads.
    fn plan(&self, b: &mut Batch, page: usize, file_len: usize) {
        b.spans.clear();
        b.spans.extend(b.keys.iter().map(|&p| Span {
            offset: self.offset + p as usize * self.page_size,
            len: self.page_size,
        }));
        coalesce(&b.spans, page, file_len, &mut b.ranges, &mut b.at);
    }
}

/// Scans the clusters that range i of `c` read.
fn scan_range(c: &Batch, i: usize, view: &View, s: &mut Scratch) {
    let end = c.ranges.get(i + 1).map_or(c.spans.len(), |r| r.tag);

    for j in c.ranges[i].tag..end {
        let list = view.lists[c.keys[j] as usize];
        let blocks = &c.arena[c.at[j]..c.at[j] + c.spans[j].len];
        scan_list(
            view.tier,
            blocks,
            list.count,
            list.first_id,
            &s.lut,
            &mut s.candidates,
            &mut s.survivors,
        );
    }
}

/// Waits for every range of the lane's last submit, which read `b`.
fn drain(wave: &mut dyn Wave, lane: usize, file: &File, b: &mut Batch) -> Result<(), Error> {
    let mut arrived = 0;

    while wave.next(lane, file, &b.ranges, &mut b.arena)?.is_some() {
        arrived += 1;
    }

    if arrived != b.ranges.len() {
        return Err(lost());
    }

    Ok(())
}

fn lost() -> Error {
    Error::other("a read wave ended before all of its ranges arrived")
}

#[cfg(test)]
#[path = "../../tests/unit/search/cold.rs"]
mod tests;
