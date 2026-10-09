//! Rewriting an index file as a plan says, from the open file and its changes as of a snapshot. A plan names the new
//! file's clusters in file order, each with the center of an old cluster or a new one, and the new cluster of every
//! vector that stays. Every 8-bit row is copied byte for byte. A vector's 1-bit code and factors are copied too when it
//! stays in a cluster whose center codes are copied, or always with the mean anchor. Every other code is measured
//! again from the vector's row, against its new cluster's center as the center decodes, since the file keeps no other
//! copy of the vector. So nothing is quantized twice, and each section is the one that a build of the same clusters,
//! centers, rows and codes writes.
//!
//! It goes over the new clusters once, in file order, a window of clusters at a time. For each window it gathers the
//! rows, writes them and their ids, then copies or measures again each code and writes the blocks. Then it writes the
//! small sections and the trailer, and the header last. It reads the old rows and codes through the cold handle, each
//! read covering whole pages of the rows or whole clusters of the codes, and keeps the checksum of each. Before it
//! writes the header it reads what no pass read of the sections it copied from, and the checksums must add up to the
//! old header's, so a damaged old file fails with `InvalidData` and leaves no new file.
//!
//! The rows go out a window at a time. When most vectors move, eg. in a major run, a window's rows lie in many places
//! of the old file, so each read covers only pages that the window needs, apart from short gaps. A page with rows for
//! several windows is read once for each. The disk holds only the old file and the new one.

use std::fs::{self, File};
use std::io::Error;
use std::path::{Path, PathBuf};

use rayon::ThreadPool;
use rayon::prelude::*;

use super::create_temp;
use super::stream::{Section, write_at};
use crate::Anchor;
use crate::LeafId;
use crate::format::{self, Layout, Part, Tables, Trailer};
use crate::index::Inner;
use crate::io::{self, Span};
use crate::quantization::rabitq::{BLOCK, Encoder, Factors, block_bytes, code_bytes, pack_block, unpack_lane};
use crate::quantization::rotation::Rotation;
use crate::quantization::{padded_dim, rotate_offset, sq4, sq8};
use crate::search::inmem::{InMemory, Stored, place};
use crate::search::{LIMIT, invalid};
use crate::update::{Delta, Seen};

/// The new cluster of a vector that drops out.
pub(crate) const GONE: u32 = u32::MAX;

/// The most bytes of rows that one window gathers at once, unless one cluster alone holds more.
const WINDOW: usize = 16 << 20;

/// The most bytes that one read of the old file takes.
const READ: usize = 4 << 20;

/// The most bytes that no vector needs between two pages that some do, which still share one read. When most vectors
/// move, a window takes a few rows from each of many old clusters, so a larger gap mostly reads rows that nothing
/// needs.
const GAP: usize = 32 << 10;

/// What a rewrite writes: the new clusters in file order, the groups that order them and the new cluster of every
/// vector. It is a pure function of the old file, the changes as of a snapshot and a seed, so it can be computed again.
pub(crate) struct Plan {
    /// The new clusters in file order.
    pub(crate) lists: Vec<PlanList>,
    pub(crate) groups: PlanGroups,
    /// The new cluster of each row of the old file, by internal id, or `GONE` for a row that drops out.
    pub(crate) file_rows: Vec<u32>,
    /// The new cluster of each insert of the snapshot, the same way.
    pub(crate) delta_rows: Vec<u32>,
    /// What maintenance keeps about the new clusters, written after the ids, if anything.
    pub(crate) trailer: Option<Trailer>,
}

pub(crate) struct PlanList {
    pub(crate) center: PlanCenter,
    /// Its group, as a place in the plan's groups.
    pub(crate) group: u16,
}

pub(crate) enum PlanCenter {
    /// The old cluster at this file position keeps its 4-bit center codes and factors, so the codes measured from it
    /// are copied.
    Keep(LeafId),
    /// A new center, which the rewrite encodes to 4 bits. Every member's code is measured again from it as it decodes.
    New(Vec<f32>),
}

/// The groups that order the new file.
pub(crate) enum PlanGroups {
    /// The old file's groups at these places, whose 8-bit rows are copied, in this order.
    Keep(Vec<u16>),
    /// New group centers, one after another, which the rewrite encodes as 8-bit rows.
    New(Vec<f32>),
}

/// What a rewrite did.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Rewritten {
    pub(crate) count: usize,
    pub(crate) nlist: usize,
    /// Codes copied, and codes measured again from their rows.
    pub(crate) copied: usize,
    pub(crate) reencoded: usize,
    /// Bytes read from the old file, and bytes written.
    pub(crate) bytes_read: u64,
    pub(crate) bytes_written: u64,
}

/// Writes the file that `plan` describes to a temporary file next to `out`, from the open index `old` and its changes
/// `delta` as of `seen`. Returns the temporary file's path and what it did. A plan that doesn't fit the file fails with
/// `InvalidInput`, and a damaged input section with `InvalidData`. Either way no temporary file remains.
pub(crate) fn rewrite(
    old: &Inner,
    delta: &Delta,
    seen: Seen,
    plan: &Plan,
    out: &Path,
    pool: &ThreadPool,
) -> Result<(PathBuf, Rewritten), Error> {
    rewrite_with(old, delta, seen, plan, out, pool, WINDOW)
}

/// The same, with windows of at most `window` bytes of rows, unless one cluster alone holds more.
pub(crate) fn rewrite_with(
    old: &Inner,
    delta: &Delta,
    seen: Seen,
    plan: &Plan,
    out: &Path,
    pool: &ThreadPool,
    window: usize,
) -> Result<(PathBuf, Rewritten), Error> {
    let src = Old::new(old, delta, seen, window)?;
    check(plan, &src)?;
    let members = Members::of(plan, src.count);
    let (file, tmp) = create_temp(out)?;
    let result = write(&src, plan, &members, &file, &tmp, pool);
    drop(file);

    match result {
        Ok(done) => Ok((tmp, done)),
        Err(e) => {
            let _ = fs::remove_file(&tmp);
            Err(e)
        }
    }
}

/// The same, into `file`, the empty file at `path`, which it syncs fully at the end. On an error the caller deletes it.
pub(crate) fn rewrite_into(
    old: &Inner,
    delta: &Delta,
    seen: Seen,
    plan: &Plan,
    file: &File,
    path: &Path,
    pool: &ThreadPool,
) -> Result<Rewritten, Error> {
    let src = Old::new(old, delta, seen, WINDOW)?;
    check(plan, &src)?;
    let members = Members::of(plan, src.count);
    write(&src, plan, &members, file, path, pool)
}

/// The old file and its changes as of a snapshot, with the sizes that the rewrite needs.
struct Old<'a> {
    inner: &'a Inner,
    layout: &'a Layout,
    tables: &'a Tables,
    delta: &'a Delta,
    /// The old file's vectors, and the inserts of the snapshot.
    count: usize,
    inserts: usize,
    dim: usize,
    d_pad: usize,
    /// The bytes of a code, a block of 32 codes and a row.
    code: usize,
    block: usize,
    row: usize,
    per_page: usize,
    /// The most rows that a window of a pass holds.
    window: usize,
}

impl<'a> Old<'a> {
    fn new(inner: &'a Inner, delta: &'a Delta, seen: Seen, window: usize) -> Result<Old<'a>, Error> {
        let layout = inner.layout();
        let (dim, d_pad) = (layout.dim, padded_dim(layout.dim));

        if seen.inserts() > delta.len() {
            return Err(invalid("the snapshot holds more inserts than the changes"));
        }

        Ok(Old {
            inner,
            layout,
            tables: inner.tables()?,
            delta,
            count: layout.count,
            inserts: seen.inserts(),
            dim,
            d_pad,
            code: code_bytes(d_pad),
            block: block_bytes(d_pad),
            row: sq8::row_bytes(dim),
            per_page: layout.page_size / sq8::row_bytes(dim),
            window: (window / sq8::row_bytes(dim)).max(1),
        })
    }

    fn file(&self) -> &File {
        self.inner.cold_file()
    }

    /// The old cluster that file row `r` sits in.
    fn list_of(&self, r: usize) -> usize {
        let lists = &self.tables.lists;
        lists.partition_point(|l| l.first_id as usize + l.count <= r)
    }

    /// Whether vector `v` keeps its code in a new cluster with the center `center`. With the mean anchor every code
    /// stays, and with the cluster anchor only a code measured from that same center does.
    fn keeps(&self, v: u32, center: &PlanCenter) -> bool {
        let (v, j) = match (self.layout.anchor, center) {
            (Anchor::Mean, _) => return true,
            (Anchor::Cluster, PlanCenter::New(_)) => return false,
            (Anchor::Cluster, PlanCenter::Keep(j)) => (v as usize, *j as usize),
        };

        match v.checked_sub(self.count) {
            Some(i) => self.delta.leaf(i) as usize == j,
            None => {
                let list = self.tables.lists[j];
                (list.first_id as usize..list.first_id as usize + list.count).contains(&v)
            }
        }
    }
}

/// Checks that `plan` fits the old file and the snapshot.
fn check(plan: &Plan, src: &Old) -> Result<(), Error> {
    let nlist = plan.lists.len();
    let fits = |&l: &u32| l == GONE || (l as usize) < nlist;

    if plan.file_rows.len() != src.count || plan.delta_rows.len() != src.inserts {
        return Err(invalid(
            "the plan doesn't place every vector of the file and the snapshot",
        ));
    }

    // This comes before the clusters, since a plan that places no vector has none, eg. a fold of an emptied file.
    if plan.file_rows.iter().chain(&plan.delta_rows).all(|&l| l == GONE) {
        return Err(invalid("every vector was deleted, so there is nothing to write"));
    }

    if nlist == 0 || nlist > u32::MAX as usize - 1 || !plan.file_rows.iter().chain(&plan.delta_rows).all(fits) {
        return Err(invalid("the plan places a vector in a cluster it lacks"));
    }

    let ngroups = match &plan.groups {
        PlanGroups::Keep(old) if old.iter().all(|&g| usize::from(g) < src.layout.nparents) => old.len(),
        PlanGroups::New(centers) if centers.len().is_multiple_of(src.dim) && centers.iter().all(|x| x.is_finite()) => {
            centers.len() / src.dim
        }
        _ => return Err(invalid("the plan's groups don't fit the file")),
    };

    if ngroups == 0 || ngroups > usize::from(u16::MAX) {
        return Err(invalid("a file holds 1 to 65535 groups"));
    }

    for list in &plan.lists {
        let center_ok = match &list.center {
            PlanCenter::Keep(j) => (*j as usize) < src.layout.nlist,
            PlanCenter::New(c) => c.len() == src.dim && c.iter().all(|x| x.is_finite()),
        };
        if !center_ok || usize::from(list.group) >= ngroups {
            return Err(invalid(
                "a cluster of the plan has a center or a group that doesn't fit",
            ));
        }
    }

    if plan
        .trailer
        .as_ref()
        .is_some_and(|t| t.region_of.len() != nlist || t.list_last.len() != nlist)
    {
        return Err(invalid("the plan's trailer doesn't match its clusters"));
    }

    Ok(())
}

/// The members of every new cluster, as vector ids in the order that the new file stores them. A vector id is a file
/// row's internal id, or the old file's count plus an insert's place. Each cluster holds its old file rows by internal
/// id, then its inserts by sequence number.
struct Members {
    order: Vec<u32>,
    /// Where each new cluster's members start in `order`, then where the last one's end.
    starts: Vec<usize>,
}

impl Members {
    fn of(plan: &Plan, count: usize) -> Members {
        let mut starts = vec![0usize; plan.lists.len() + 1];
        let placed = plan.file_rows.iter().chain(&plan.delta_rows).filter(|&&l| l != GONE);

        for &l in placed {
            starts[l as usize + 1] += 1;
        }

        for l in 0..plan.lists.len() {
            starts[l + 1] += starts[l];
        }

        let mut next = starts.clone();
        let mut order = vec![0u32; starts[plan.lists.len()]];
        let ids = (0..count as u32).chain((0..plan.delta_rows.len() as u32).map(|i| count as u32 + i));

        for (v, &l) in ids.zip(plan.file_rows.iter().chain(&plan.delta_rows)) {
            if l != GONE {
                order[next[l as usize]] = v;
                next[l as usize] += 1;
            }
        }

        Members { order, starts }
    }

    /// The new clusters in runs of at most `most` vectors, unless one cluster alone holds more.
    fn windows(&self, most: usize) -> Vec<std::ops::Range<usize>> {
        let mut out: Vec<std::ops::Range<usize>> = Vec::new();

        for l in 0..self.starts.len() - 1 {
            match out.last_mut() {
                Some(w) if self.starts[l + 1] - self.starts[w.start] <= most => w.end = l + 1,
                _ => out.push(l..l + 1),
            }
        }

        out
    }

    fn of_list(&self, l: usize) -> &[u32] {
        &self.order[self.starts[l]..self.starts[l + 1]]
    }
}

/// Writes every section of the new file into `file`, the file at `path`, then checks the old file's checksums, then
/// writes the header and syncs the file fully.
fn write(
    src: &Old,
    plan: &Plan,
    members: &Members,
    file: &File,
    path: &Path,
    pool: &ThreadPool,
) -> Result<Rewritten, Error> {
    let layout = src.layout;
    let n = members.order.len();
    let mut done = Rewritten {
        count: n,
        nlist: plan.lists.len(),
        ..Rewritten::default()
    };
    let small = read_small(src, &mut done)?;
    let frame = frame(src, plan, members, &small)?;
    let mut sections = format::small_sections(&frame);
    // The rotation section is copied as the old file stores it.
    sections[0] = small.rotation;
    let codes_len = frame
        .lists
        .last()
        .map_or(0, |l| l.offset + l.count.div_ceil(BLOCK) * src.block);
    let lens = [
        0,
        sections[0].len(),
        sections[1].len(),
        sections[2].len(),
        codes_len,
        format::rows_len(n, src.dim, layout.page_size),
        format::ids_len(n),
    ];
    let spans = format::layout(lens);
    let trailer = plan.trailer.as_ref().map(format::encode_trailer);
    let end = match &trailer {
        Some(t) => format::trailer_at(spans[6]) + t.len(),
        None => spans[6].end(),
    };
    file.set_len(end as u64)?;
    let mut crcs = [0u32; 7];
    let mut pieces = (Pieces::rows(layout), Pieces::codes(layout, &src.tables.lists));
    (crcs[4], crcs[5], crcs[6]) = write_lists(src, plan, members, &frame, file, &spans, &mut pieces, &mut done, pool)?;

    for (i, bytes) in sections.iter().enumerate() {
        write_at(file, bytes, spans[i + 1].offset)?;
        crcs[i + 1] = crc32fast::hash(bytes);
    }

    if let Some(t) = &trailer {
        write_at(file, t, format::trailer_at(spans[6]))?;
    }

    let (mut rows, mut codes) = pieces;
    rows.finish(src, Part::Rows, &mut done)?;

    // Without a copied code, the new file holds nothing of the old codes section.
    if done.copied > 0 {
        codes.finish(src, Part::Codes, &mut done)?;
    }
    // The header goes last, once every other section is on the disk, so a half-written file never opens.
    file.sync_data()?;
    write_at(file, &format::header(&frame, n, &spans, &crcs), 0)?;
    io::sync_file(file, path)?;
    done.bytes_written = end as u64;
    Ok(done)
}

/// The old file's small sections, each checked against the old header.
struct Small {
    rotation: Vec<u8>,
    centers: Vec<u8>,
}

fn read_small(src: &Old, done: &mut Rewritten) -> Result<Small, Error> {
    let mut read = |part: Part| -> Result<Vec<u8>, Error> {
        let span = src.layout.span(part);
        let mut bytes = vec![0u8; span.len];
        io::read_at(src.file(), &mut bytes, span.offset)?;
        done.bytes_read += span.len as u64;
        format::check_section(src.layout, part, crc32fast::hash(&bytes))?;
        Ok(bytes)
    };
    // The extents are only checked. The tables the old file loaded hold them already.
    read(Part::Extents)?;

    Ok(Small {
        rotation: read(Part::Rotation)?,
        centers: read(Part::Centers)?,
    })
}

/// The new file's small tables as an index without blocks, rows or ids, with the epoch one past the old file's.
fn frame(src: &Old, plan: &Plan, members: &Members, small: &Small) -> Result<InMemory, Error> {
    let (dim, layout, tables) = (src.dim, src.layout, src.tables);
    let (mean, rotation) = (src.inner.mean(), src.inner.rotation());
    let crb = sq8::center_row_bytes(dim);
    let parents = match &plan.groups {
        PlanGroups::Keep(old) => old
            .iter()
            .flat_map(|&g| &small.centers[usize::from(g) * crb..(usize::from(g) + 1) * crb])
            .copied()
            .collect(),
        PlanGroups::New(centers) => {
            let mut rows = vec![0u8; centers.len() / dim * crb];
            for (g, row) in centers.chunks_exact(dim).zip(rows.chunks_exact_mut(crb)) {
                sq8::write_center_row(g, mean, row);
                if !fits(sq8::norm2(row, dim)) {
                    return Err(invalid("a group center is too far from the mean to encode"));
                }
            }
            rows
        }
    };
    let ccb = sq4::code_bytes(src.d_pad);
    let mut centers = Vec::with_capacity(plan.lists.len() * ccb);
    let mut center_factors = Vec::with_capacity(plan.lists.len());
    let (mut z, mut tmp, mut codes) = (vec![0f32; dim], vec![0f32; dim], vec![0u8; ccb]);

    for list in &plan.lists {
        match &list.center {
            PlanCenter::Keep(j) => {
                let j = *j as usize;
                centers.extend_from_slice(&tables.codes[j * ccb..(j + 1) * ccb]);
                center_factors.push(tables.factors[j]);
            }
            PlanCenter::New(c) => {
                rotate_offset(rotation, c, mean, &mut z, &mut tmp);
                let f = sq4::encode(&z, &mut codes);
                if !(fits(f.norm2) && f.lo.is_finite() && f.step.is_finite()) {
                    return Err(invalid("a cluster center is too far from the mean to encode"));
                }
                centers.extend_from_slice(&codes);
                center_factors.push(f);
            }
        }
    }

    let counts = (0..plan.lists.len()).map(|l| members.of_list(l).len());
    let stored = Stored {
        parents,
        centers,
        center_factors,
        parent_of: plan.lists.iter().map(|l| l.group).collect(),
        lists: place(counts, src.block),
    };
    let mut frame = InMemory::with_tables(layout, mean.to_vec(), rotation.clone(), stored);
    frame.epoch = layout.epoch + 1;
    Ok(frame)
}

/// Writes the codes, rows and ids sections in the new order, a window of clusters at a time. Each window takes its rows
/// from the old file and from the snapshot, then copies each code that stays, from whole old clusters or the snapshot,
/// and measures the others again from the rows on `pool`. Returns the three sections' checksums.
#[allow(clippy::too_many_arguments)]
fn write_lists(
    src: &Old,
    plan: &Plan,
    members: &Members,
    frame: &InMemory,
    file: &File,
    spans: &[Span; 7],
    pieces: &mut (Pieces, Pieces),
    done: &mut Rewritten,
    pool: &ThreadPool,
) -> Result<(u32, u32, u32), Error> {
    let (rb, cb, page_size) = (src.row, src.code, src.layout.page_size);
    let (mut t1, mut t2, mut ids) = (
        Section::new(file, spans[4].offset),
        Section::new(file, spans[5].offset),
        Section::new(file, spans[6].offset),
    );
    let (mut page, mut in_page) = (vec![0u8; page_size], 0);
    let (mut rows, mut wanted) = (Vec::new(), Vec::new());
    let (mut codes, mut factors, mut keep, mut copies, mut block) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());

    for window in members.windows(src.window) {
        let first = members.starts[window.start];
        let slice = &members.order[first..members.starts[window.end]];
        rows.resize(slice.len() * rb, 0);
        wanted.clear();
        for (slot, &v) in slice.iter().enumerate() {
            match (v as usize).checked_sub(src.count) {
                Some(i) => rows[slot * rb..(slot + 1) * rb].copy_from_slice(src.delta.row(i)),
                None => wanted.push((v, slot as u32)),
            }
        }
        wanted.sort_unstable();
        gather_rows(src, &wanted, &mut rows, &mut pieces.0, done)?;
        for row in rows.chunks_exact(rb) {
            page[in_page * rb..(in_page + 1) * rb].copy_from_slice(row);
            in_page += 1;
            if in_page == src.per_page {
                t2.write(&page)?;
                page.fill(0);
                in_page = 0;
            }
            ids.write(&sq8::id(row, src.dim).to_le_bytes())?;
        }

        codes.resize(slice.len() * cb, 0);
        factors.resize(slice.len(), Factors::EMPTY);
        keep.clear();
        copies.clear();
        for l in window.clone() {
            let center = &plan.lists[l].center;
            for (k, &v) in members.of_list(l).iter().enumerate() {
                let slot = members.starts[l] - first + k;
                let stays = src.keeps(v, center);
                keep.push(stays);
                match (stays, (v as usize).checked_sub(src.count)) {
                    (true, Some(i)) => {
                        let (code, f) = src.delta.lane(i);
                        codes[slot * cb..(slot + 1) * cb].copy_from_slice(code);
                        factors[slot] = f;
                    }
                    (true, None) => copies.push((src.list_of(v as usize) as u32, v, slot as u32)),
                    (false, _) => {}
                }
            }
        }
        copies.sort_unstable();
        copy_lanes(src, &copies, &mut codes, &mut factors, &mut block, &mut pieces.1, done)?;
        let moved = keep.iter().filter(|&&k| !k).count();
        (done.copied, done.reencoded) = (done.copied + slice.len() - moved, done.reencoded + moved);
        let blocks = pack_window(
            src,
            frame,
            members,
            window.clone(),
            &codes,
            &factors,
            &keep,
            &rows,
            pool,
        )?;
        blocks.iter().try_for_each(|b| t1.write(b))?;
    }

    if in_page > 0 {
        t2.write(&page)?;
    }

    // The ids section ends in the tombstones, one bit per vector, all clear in a new file.
    ids.write(&vec![0u8; 8 * members.order.len().div_ceil(64)])?;
    Ok((t1.finish()?, t2.finish()?, ids.finish()?))
}

/// Copies the old file's rows `wanted`, as (internal id, slot) pairs sorted by internal id, into their slots of `out`.
/// It reads runs of whole pages, each run at most `READ` bytes, and keeps each page's checksum in `pieces`.
fn gather_rows(
    src: &Old,
    wanted: &[(u32, u32)],
    out: &mut [u8],
    pieces: &mut Pieces,
    done: &mut Rewritten,
) -> Result<(), Error> {
    let (rb, per_page, page_size) = (src.row, src.per_page, src.layout.page_size);
    let section = src.layout.span(Part::Rows).offset;
    let (most, gap) = ((READ / page_size).max(1), GAP / page_size);
    let mut buf = Vec::new();
    let mut i = 0;

    while i < wanted.len() {
        let first = wanted[i].0 as usize / per_page;
        let mut j = i;
        let mut last = first;
        while j < wanted.len() {
            let p = wanted[j].0 as usize / per_page;
            if p > last + gap + 1 || p - first >= most {
                break;
            }
            last = p;
            j += 1;
        }
        buf.resize((last + 1 - first) * page_size, 0);
        io::read_at(src.file(), &mut buf, section + first * page_size)?;
        done.bytes_read += buf.len() as u64;
        for (k, page) in buf.chunks_exact(page_size).enumerate() {
            pieces.record(first + k, page);
        }
        for &(id, slot) in &wanted[i..j] {
            let (id, slot) = (id as usize, slot as usize);
            let at = (id / per_page - first) * page_size + (id % per_page) * rb;
            out[slot * rb..(slot + 1) * rb].copy_from_slice(&buf[at..at + rb]);
        }
        i = j;
    }

    Ok(())
}

/// Copies the codes `copies`, as (old cluster, internal id, slot) sorted by old cluster, from their old clusters'
/// blocks into their slots. It reads each old cluster whole, which keeps its checksum in `pieces`.
fn copy_lanes(
    src: &Old,
    copies: &[(u32, u32, u32)],
    codes: &mut [u8],
    factors: &mut [Factors],
    buf: &mut Vec<u8>,
    pieces: &mut Pieces,
    done: &mut Rewritten,
) -> Result<(), Error> {
    let (cb, size) = (src.code, src.block);
    let section = src.layout.span(Part::Codes).offset;
    let mut i = 0;

    while i < copies.len() {
        let j0 = copies[i].0 as usize;
        let list = src.tables.lists[j0];
        buf.resize(list.count.div_ceil(BLOCK) * size, 0);
        io::read_at(src.file(), buf, section + list.offset)?;
        done.bytes_read += buf.len() as u64;
        pieces.record_list(j0, buf);
        while i < copies.len() && copies[i].0 as usize == j0 {
            let (_, r, slot) = copies[i];
            let k = r as usize - list.first_id as usize;
            let (slot, block) = (slot as usize, &buf[k / BLOCK * size..(k / BLOCK + 1) * size]);
            factors[slot] = unpack_lane(block, src.d_pad, k % BLOCK, &mut codes[slot * cb..(slot + 1) * cb]);
            i += 1;
        }
    }

    Ok(())
}

/// Packs the blocks of the new clusters `lists` on `pool`, one cluster per task. A code whose slot `keep` marks is
/// taken from `codes` and `factors`, and every other one is measured again from its row in `rows`, against its
/// cluster's center as the center decodes. Returns each cluster's blocks, in order.
#[allow(clippy::too_many_arguments)]
fn pack_window(
    src: &Old,
    frame: &InMemory,
    members: &Members,
    lists: std::ops::Range<usize>,
    codes: &[u8],
    factors: &[Factors],
    keep: &[bool],
    rows: &[u8],
    pool: &ThreadPool,
) -> Result<Vec<Vec<u8>>, Error> {
    let (cb, rb, dim, d_pad, size) = (src.code, src.row, src.dim, src.d_pad, src.block);
    let first = members.starts[lists.start];
    let (mean, rotation) = (src.inner.mean(), src.inner.rotation());
    let ccb = sq4::code_bytes(d_pad);
    let state = || Recode::new(rotation, mean, dim);

    pool.install(|| {
        lists
            .into_par_iter()
            .map_init(state, |r, l| {
                let (a, b) = (members.starts[l] - first, members.starts[l + 1] - first);
                let mut lane_codes = codes[a * cb..b * cb].to_vec();
                let mut lane_factors = factors[a..b].to_vec();
                let center = (&frame.centers[l * ccb..(l + 1) * ccb], &frame.center_factors[l]);
                let mut decoded = false;
                for k in 0..b - a {
                    if keep[a + k] {
                        continue;
                    }
                    if !decoded {
                        sq4::decode(center.0, center.1, &mut r.center);
                        decoded = true;
                    }
                    let row = &rows[(a + k) * rb..(a + k + 1) * rb];
                    lane_factors[k] = r.encode(src, row, &mut lane_codes[k * cb..(k + 1) * cb])?;
                }
                let mut out = vec![0u8; (b - a).div_ceil(BLOCK) * size];
                for (i, blk) in out.chunks_exact_mut(size).enumerate() {
                    let (s, e) = (i * BLOCK, ((i + 1) * BLOCK).min(b - a));
                    pack_block(&lane_codes[s * cb..e * cb], &lane_factors[s..e], d_pad, blk);
                }
                Ok(out)
            })
            .collect()
    })
}

/// One thread's state for measuring codes again from rows.
struct Recode<'a> {
    encoder: Encoder<'a>,
    mean: &'a [f32],
    /// The decoded row, and the cluster's center P(ĉ − μ) as its codes decode, where P is the rotation, ĉ the center
    /// and μ the mean.
    o: Vec<f32>,
    center: Vec<f32>,
}

impl<'a> Recode<'a> {
    fn new(rotation: &'a Rotation, mean: &'a [f32], dim: usize) -> Recode<'a> {
        Recode {
            encoder: Encoder::new(rotation, mean),
            mean,
            o: vec![0f32; dim],
            center: vec![0f32; dim],
        }
    }

    /// Measures the code of the vector that `row` stores from `self.center`, or from the mean with the mean anchor,
    /// into `bits`, and returns its factors.
    fn encode(&mut self, src: &Old, row: &[u8], bits: &mut [u8]) -> Result<Factors, Error> {
        sq8::decode(row, self.mean, &mut self.o);

        if !self.o.iter().all(|x| x.is_finite()) {
            return Err(damaged_row());
        }

        let f = match src.layout.anchor {
            Anchor::Mean => self.encoder.encode(&self.o, bits),
            Anchor::Cluster => self.encoder.encode_from(&self.o, &self.center, bits),
        };

        // A vector whose code fails the checks a build makes means a damaged row or center.
        if !(fits(f.f_add.abs()) && f.f_rescale.is_finite() && f.f_err.is_finite() && fits(sq8::norm2(row, src.dim))) {
            return Err(damaged_row());
        }

        Ok(f)
    }
}

/// The checksums of the pieces of an old section that the rewrite has read. Once every pass is done, `finish` reads the
/// rest and checks that they add up to the old header's checksum.
struct Pieces {
    /// Where each piece starts in the file, then where the last one ends.
    bounds: Vec<usize>,
    crcs: Vec<Option<u32>>,
    /// The piece of each old cluster's blocks, for the codes section, or `None` for a cluster without vectors.
    lists: Vec<Option<usize>>,
}

impl Pieces {
    /// The rows section, a piece per page.
    fn rows(layout: &Layout) -> Pieces {
        let span = layout.span(Part::Rows);
        let bounds: Vec<usize> = (span.offset..=span.end()).step_by(layout.page_size).collect();
        Pieces {
            crcs: vec![None; bounds.len() - 1],
            bounds,
            lists: Vec::new(),
        }
    }

    /// The codes section, a piece per old cluster with vectors, and one for each stretch between them.
    fn codes(layout: &Layout, lists: &[crate::search::List]) -> Pieces {
        let span = layout.span(Part::Codes);
        let size = block_bytes(padded_dim(layout.dim));
        let (mut bounds, mut pieces) = (vec![span.offset], Vec::with_capacity(lists.len()));

        for list in lists {
            let (start, len) = (span.offset + list.offset, list.count.div_ceil(BLOCK) * size);
            if len == 0 {
                pieces.push(None);
                continue;
            }
            if start > *bounds.last().unwrap_or(&start) {
                bounds.push(start);
            }
            pieces.push(Some(bounds.len() - 1));
            bounds.push(start + len);
        }

        if span.end() > *bounds.last().unwrap_or(&span.end()) {
            bounds.push(span.end());
        }

        Pieces {
            crcs: vec![None; bounds.len() - 1],
            bounds,
            lists: pieces,
        }
    }

    /// Keeps the checksum of piece `piece`, whose bytes are `bytes`.
    fn record(&mut self, piece: usize, bytes: &[u8]) {
        if self.crcs[piece].is_none() {
            self.crcs[piece] = Some(crc32fast::hash(bytes));
        }
    }

    /// The same for the piece of old cluster `list`.
    fn record_list(&mut self, list: usize, bytes: &[u8]) {
        if let Some(piece) = self.lists[list] {
            self.record(piece, bytes);
        }
    }

    /// Reads the pieces that no pass read, in runs of at most `READ` bytes, then checks that the checksums of all of
    /// them, in file order, give the old header's checksum of section `part`.
    fn finish(&mut self, src: &Old, part: Part, done: &mut Rewritten) -> Result<(), Error> {
        let mut buf = Vec::new();
        let mut p = 0;

        while p < self.crcs.len() {
            if self.crcs[p].is_some() {
                p += 1;
                continue;
            }
            let mut q = p + 1;
            while q < self.crcs.len() && self.crcs[q].is_none() && self.bounds[q + 1] - self.bounds[p] <= READ {
                q += 1;
            }
            buf.resize(self.bounds[q] - self.bounds[p], 0);
            io::read_at(src.file(), &mut buf, self.bounds[p])?;
            done.bytes_read += buf.len() as u64;
            for k in p..q {
                let bytes = &buf[self.bounds[k] - self.bounds[p]..self.bounds[k + 1] - self.bounds[p]];
                self.crcs[k] = Some(crc32fast::hash(bytes));
            }
            p = q;
        }

        let mut crc = crc32fast::Hasher::new();

        for (k, piece) in self.crcs.iter().enumerate() {
            let len = (self.bounds[k + 1] - self.bounds[k]) as u64;
            crc.combine(&crc32fast::Hasher::new_with_initial_len(piece.unwrap_or(0), len));
        }

        format::check_section(src.layout, part, crc.finalize())
    }
}

/// Whether a squared length, eg. f_add or a row's n2, leaves a search room to add another one.
fn fits(norm2: f32) -> bool {
    f64::from(norm2) <= LIMIT
}

fn damaged_row() -> Error {
    Error::new(std::io::ErrorKind::InvalidData, "a vector's 8-bit row is damaged")
}

#[cfg(test)]
#[path = "../../tests/unit/build/rewrite.rs"]
mod tests;
