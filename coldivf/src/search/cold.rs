//! Cold search, which reads only what a query needs. A fetch reads the chosen clusters' blocks in one wave. A search
//! scans each range as it arrives, then reads its best candidates' rows in a second, small wave.

use std::fs::File;
use std::io::Error;
use std::sync::{Arc, Weak};
use std::time::Duration;

use super::candidates::Candidate;
use super::scan::{Pending, scan_probed};
use super::{List, Scratch, View};
use crate::format::{Layout, Part, Tables};
use crate::index::{Inner, Shared};
use crate::io::{Arena, Range, Span, Wave, arena_bound, coalesce};
use crate::quantization::rabitq::{BLOCK, block_bytes};
use crate::quantization::{padded_dim, sq8};
use crate::trace::{Clock, ReadPath};
use crate::{LeafId, Limits};

/// The backend lane of the cluster wave and the late row wave, which never overlap.
const MAIN: usize = 0;

/// The backend lane of the early row wave.
const EARLY: usize = 1;

/// One search context's reads of one index. Its arenas are sized once from the context's limits, so a query allocates
/// nothing. The cluster arena also needs the cluster tables, so reads made before they load make it later, in `size`.
/// When the index moves to a new generation, the reads are planned again in place, and keep their arenas and their
/// backend's threads. Only a cluster arena that the new generation's largest clusters outgrow grows, with room to
/// spare.
pub(crate) struct Reads {
    /// The cell of the index it reads, held weakly so it tells when the index closed, and the cell's number, which
    /// keys the reads in a context with the copy, the index's place among the collections of a set that name the same
    /// one, or 0. Numbers are never used twice, unlike addresses.
    owner: Weak<Shared>,
    cell: u64,
    copy: usize,
    /// The number of the generation that the reads are planned for, and of the one that the cluster arena was sized
    /// for, or 0 before the cluster tables of any loaded.
    generation: u64,
    sized_for: u64,
    /// It comes before the batches, so it drops first and waits for any reads still landing in their arenas.
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
    /// Every fetched cluster, once each, with or without vectors in the file, whose pending inserts a scan reads.
    probed: Vec<LeafId>,
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
    /// How long the last scan waited for its clusters to arrive, measured only with the `bench` feature.
    waited: Duration,
    /// How many searches read their rows through these reads, and the backend's hits when the last search began.
    searches: usize,
    hits_at: usize,
}

impl Reads {
    /// Reads of the index `owner`, with the file layout `layout` and the cluster tables `tables`, through `wave`.
    pub(crate) fn new(
        owner: &Arc<Inner>,
        wave: Box<dyn Wave>,
        layout: &Layout,
        file_len: usize,
        tables: &Tables,
        limits: &Limits,
    ) -> Reads {
        let mut reads = Reads::ahead(owner, wave, layout, file_len, limits);
        reads.size(owner.cold_file(), tables, limits);
        reads
    }

    /// The same, made before the cluster tables load, eg. by `Index::prepare` of a cold index. It readies the backend
    /// for the file and makes every arena but the cluster arena, which `size` makes once the tables are loaded.
    pub(crate) fn ahead(
        owner: &Arc<Inner>,
        wave: Box<dyn Wave>,
        layout: &Layout,
        file_len: usize,
        limits: &Limits,
    ) -> Reads {
        let page = wave.page();
        let rows = RowSection::new(layout);
        let keep = limits.max_rerank.max(limits.max_k);
        let row_arena = keep * rows.room(page);
        let mut reads = Reads {
            owner: owner.owner(),
            cell: owner.cell(),
            copy: 0,
            generation: owner.uid(),
            sized_for: 0,
            wave,
            page,
            file_len,
            codes: layout.span(Part::Codes).offset,
            block: block_bytes(padded_dim(layout.dim)),
            rows,
            clusters: Batch::new(limits.max_nprobe, 0),
            probed: Vec::with_capacity(limits.max_nprobe),
            clusters_in: true,
            failed: false,
            early: Batch::new(keep, row_arena),
            early_pending: false,
            early_used: false,
            late: Batch::new(keep, row_arena),
            waited: Duration::ZERO,
            searches: 0,
            hits_at: 0,
        };
        let Reads { wave, early, late, .. } = &mut reads;
        wave.ready(owner.cold_file(), &[early.arena.memory(), late.arena.memory()], keep);
        reads
    }

    /// Makes the cluster arena for the generation that the reads are planned for, unless it is made. It holds the
    /// largest clusters of `tables`, those that one fetch may read, and a quarter more, and the backend may then
    /// register it along with the others. For a later generation it keeps the arena that it has when the largest
    /// clusters fit, and makes one half again as large as they need when they don't.
    pub(crate) fn size(&mut self, file: &File, tables: &Tables, limits: &Limits) {
        if self.sized_for == self.generation {
            return;
        }

        let most = limits.max_nprobe.min(tables.largest.len());
        let lens = tables.largest[..most].iter().map(|&b| b as usize * self.block);
        let bound = arena_bound(lens, self.page);

        if self.sized_for == 0 {
            self.clusters.arena = Arena::new(bound + bound / 4);
        } else if bound > self.clusters.arena.len() {
            self.clusters.arena = Arena::new(bound + bound / 2);
        }

        self.sized_for = self.generation;
        let keep = limits.max_rerank.max(limits.max_k);
        let Reads {
            wave,
            clusters,
            early,
            late,
            ..
        } = self;
        let arenas = [clusters.arena.memory(), early.arena.memory(), late.arena.memory()];
        wave.ready(file, &arenas, limits.max_nprobe.max(keep));
    }

    /// Does the backend's setup that may wait, eg. registering the arenas with the kernel, once a search has read its
    /// rows through these reads, so a process that searches once never pays for it.
    pub(crate) fn prepare(&mut self) {
        if self.sized() && self.searches > 0 {
            self.wave.prepare();
        }
    }

    /// Whether the cluster arena is made for the generation that the reads are planned for.
    pub(crate) fn sized(&self) -> bool {
        self.sized_for == self.generation
    }

    /// The number of the generation that the reads are planned for.
    pub(crate) fn generation(&self) -> u64 {
        self.generation
    }

    /// Plans the reads again in place for `owner`, a newer generation of the index they read, with the file layout
    /// `layout`. It forgets every wave and keeps every arena. The cluster arena waits for `size`, which checks it
    /// against the new generation's clusters. A row arena grows only if the new file packs its rows in larger pieces
    /// of read pages, which a new generation never does.
    pub(crate) fn replan(&mut self, owner: &Inner, layout: &Layout, file_len: usize, limits: &Limits) {
        self.forget();
        (self.generation, self.file_len) = (owner.uid(), file_len);
        self.codes = layout.span(Part::Codes).offset;
        self.block = block_bytes(padded_dim(layout.dim));
        self.rows = RowSection::new(layout);
        let keep = limits.max_rerank.max(limits.max_k);
        let row_arena = keep * self.rows.room(self.page);

        if row_arena > self.early.arena.len() {
            (self.early.arena, self.late.arena) = (Arena::new(row_arena), Arena::new(row_arena));
        }

        let Reads { wave, early, late, .. } = self;
        wave.ready(owner.cold_file(), &[early.arena.memory(), late.arena.memory()], keep);
    }

    /// The backend and how it is set up.
    pub(crate) fn read_path(&self) -> ReadPath {
        self.wave.read_path()
    }

    /// The ranges that came from the backend's own copies of pages since the last search began. A backend that set
    /// itself up again since counts from 0.
    pub(crate) fn hits(&self) -> usize {
        self.wave.hits().saturating_sub(self.hits_at)
    }

    /// Starts reading the blocks of `leaves`, each of which must be under the cluster count. A cluster without vectors
    /// needs no read.
    pub(crate) fn fetch(&mut self, file: &File, lists: &[List], leaves: &[LeafId]) -> Result<(), Error> {
        (self.clusters_in, self.failed, self.early_pending) = (false, true, false);
        self.hits_at = self.wave.hits();
        self.early.clear();
        self.late.clear();
        let c = &mut self.clusters;
        c.keys.clear();
        c.keys.extend_from_slice(leaves);
        c.keys.sort_unstable();
        c.keys.dedup();
        self.probed.clear();
        self.probed.extend_from_slice(&c.keys);
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
        self.wave.submit(MAIN, file, &c.ranges, c.arena.memory())?;
        (self.clusters_in, self.failed) = (c.ranges.is_empty(), false);
        Ok(())
    }

    /// Forgets every fetch and row wave, eg. before a search that scans its clusters in place and reads only rows.
    pub(crate) fn forget(&mut self) {
        (self.clusters_in, self.failed, self.early_pending) = (true, false, false);
        self.hits_at = self.wave.hits();
        self.clusters.clear();
        self.probed.clear();
        self.early.clear();
        self.late.clear();
    }

    /// Every fetched cluster, once each, in increasing order.
    pub(crate) fn probed(&self) -> &[LeafId] {
        &self.probed
    }

    /// Scans every fetched cluster for the query in `s`, with its inserts in `pending`. While the fetch is still
    /// arriving, it scans each range as it comes in. The inserts of a fetched cluster whose own blocks hold no vector
    /// come last, since the fetch reads nothing of it.
    pub(crate) fn scan(
        &mut self,
        file: &File,
        view: &View,
        s: &mut Scratch,
        pending: Option<Pending>,
    ) -> Result<(), Error> {
        if self.failed {
            return Err(Error::other("an earlier read of these clusters failed"));
        }

        let c = &mut self.clusters;
        let unread = |probed: &[LeafId], s: &mut Scratch| {
            for &leaf in probed.iter().filter(|&&l| view.lists[l as usize].count == 0) {
                let (lut, q32, candidates, survivors) = (&mut s.lut, &s.q32, &mut s.candidates, &mut s.survivors);
                scan_probed(view, leaf, &[], lut, q32, pending, candidates, survivors);
            }
        };

        if self.clusters_in {
            self.waited = Duration::ZERO;
            (0..c.ranges.len()).for_each(|i| scan_range(c, i, view, s, pending));
            unread(&self.probed, s);
            return Ok(());
        }

        self.failed = true;
        let (mut arrived, mut clock) = (0, Clock::start());
        self.waited = Duration::ZERO;

        while let Some(i) = self.wave.next(MAIN, file, &c.ranges, c.arena.memory())? {
            self.waited += clock.lap();
            scan_range(c, i, view, s, pending);
            clock.lap();
            arrived += 1;
        }

        if arrived != c.ranges.len() {
            return Err(lost());
        }

        unread(&self.probed, s);
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

        if let Err(err) = self.wave.submit(EARLY, file, &e.ranges, e.arena.memory()) {
            e.clear();
            return Err(err);
        }

        self.early_pending = true;
        Ok(())
    }

    /// Reads the rows of `candidates`. It starts reading the rows that the early wave didn't before it waits for the
    /// early rows, and it skips the wait when none of them are needed. With `warm`, eg. for a warm index, the rows are
    /// likely in the operating system's cache, so they are read through it.
    pub(crate) fn read_rows(&mut self, file: &File, candidates: &[Candidate], warm: bool) -> Result<(), Error> {
        self.start_rows(file, candidates, warm)?;
        self.finish_rows(file)
    }

    /// Starts reading the rows of `candidates` that the early wave didn't, so other reads can start before the wait.
    pub(crate) fn start_rows(&mut self, file: &File, candidates: &[Candidate], warm: bool) -> Result<(), Error> {
        let (e, l) = (&mut self.early, &mut self.late);
        l.keys.clear();
        l.keys.extend(self.rows.pages(candidates));
        l.keys.sort_unstable();
        l.keys.dedup();
        let needed = l.keys.len();
        l.keys.retain(|p| e.keys.binary_search(p).is_err());
        self.rows.plan(l, self.page, self.file_len);
        self.early_used = l.keys.len() < needed;

        match warm {
            true => self.wave.submit_cached(MAIN, file, &l.ranges, l.arena.memory()),
            false => self.wave.submit(MAIN, file, &l.ranges, l.arena.memory()),
        }
    }

    /// Waits for the rows that `start_rows` started, and for the early rows when some of them are needed.
    pub(crate) fn finish_rows(&mut self, file: &File) -> Result<(), Error> {
        if self.early_used && self.early_pending {
            drain(&mut *self.wave, EARLY, file, &mut self.early)?;
        }

        self.early_pending = false;
        drain(&mut *self.wave, MAIN, file, &mut self.late)?;
        self.searches += 1;
        Ok(())
    }

    /// How long the last scan waited for its clusters to arrive.
    pub(crate) fn waited(&self) -> Duration {
        self.waited
    }

    /// The bytes and the ranges of the last cluster wave.
    pub(crate) fn cluster_traffic(&self) -> (usize, usize) {
        traffic(&self.clusters)
    }

    /// The bytes and the ranges of the last row wave that the final query needed.
    pub(crate) fn row_traffic(&self) -> (usize, usize) {
        traffic(&self.late)
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
        b.arena.part(at, self.rows.row_bytes)
    }
}

/// The slot in `slots` of the reads of `index` for its copy `copy` in a set of collections, made with `make` if there
/// isn't one yet, and planned again in place if it was planned for another generation of the same index. Making one
/// first drops the reads of indexes that have closed.
pub(crate) fn reads_for(
    slots: &mut Vec<Reads>,
    index: &Arc<Inner>,
    copy: usize,
    limits: &Limits,
    make: impl FnOnce() -> Reads,
) -> usize {
    if let Some(i) = slots.iter().position(|r| r.cell == index.cell() && r.copy == copy) {
        if slots[i].generation != index.uid() {
            slots[i].replan(index, index.layout(), index.file_len(), limits);
        }
        return i;
    }

    slots.retain(|r| r.owner.strong_count() > 0);
    slots.push(Reads { copy, ..make() });
    slots.len() - 1
}

/// The bytes and the ranges of `batch`'s wave.
fn traffic(batch: &Batch) -> (usize, usize) {
    (batch.ranges.iter().map(|r| r.len).sum(), batch.ranges.len())
}

/// One wave's plan. Span j is the cluster or row page `keys[j]`, and its bytes land at `arena[at[j]..]`.
struct Batch {
    keys: Vec<u32>,
    spans: Vec<Span>,
    at: Vec<usize>,
    ranges: Vec<Range>,
    arena: Arena,
}

impl Batch {
    fn new(spans: usize, bytes: usize) -> Batch {
        Batch {
            keys: Vec::with_capacity(spans),
            spans: Vec::with_capacity(spans),
            at: Vec::with_capacity(spans),
            ranges: Vec::with_capacity(spans),
            arena: Arena::new(bytes),
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

/// Scans the clusters that range i of `c` read, with their inserts in `pending`.
fn scan_range(c: &Batch, i: usize, view: &View, s: &mut Scratch, pending: Option<Pending>) {
    let end = c.ranges.get(i + 1).map_or(c.spans.len(), |r| r.tag);

    for j in c.ranges[i].tag..end {
        if let (Some(p), Some(&next)) = (pending, c.keys.get(j + 1)) {
            p.touch(next);
        }
        let blocks = c.arena.part(c.at[j], c.spans[j].len);
        scan_probed(
            view,
            c.keys[j],
            blocks,
            &mut s.lut,
            &s.q32,
            pending,
            &mut s.candidates,
            &mut s.survivors,
        );
    }
}

/// Waits for every range of the lane's last submit, which read `b`.
fn drain(wave: &mut dyn Wave, lane: usize, file: &File, b: &mut Batch) -> Result<(), Error> {
    let mut arrived = 0;

    while wave.next(lane, file, &b.ranges, b.arena.memory())?.is_some() {
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
