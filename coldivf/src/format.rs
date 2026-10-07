//! The index file, version 1. Every number is little-endian.
//!
//! | Section | What It Holds |
//! |---|---|
//! | Header | The sizes and settings, and where each section starts, with a checksum. It is the first 4 KiB |
//! | Rotation | The rotation's sign bits and the mean, right after the header |
//! | Centers | The group centers as 8-bit rows, then each cluster's 4-bit codes, factors and group |
//! | Extents | Where each cluster's blocks start in the codes section, with its vector count and first internal id |
//! | Codes | Each cluster's blocks of 1-bit codes, in file order |
//! | Rows | The 8-bit rows by internal id, in pages |
//! | Ids | The id given at build time of each internal id, then one deleted bit per vector, all 0 at build |
//!
//! The sections after the rotation start on 16 KiB boundaries, and the extents follow the centers, so one read gets
//! both. Up to 2806 dimensions the header and the rotation fit in the first 16 KiB. The header holds a 32-bit cyclic
//! redundancy check (CRC32) of each section. Opening checks the header's and the rotation's, and `verify` checks all of
//! them.

use std::borrow::Cow;
use std::io::{Error, ErrorKind};
use std::mem::size_of;
use std::ops::Range;

use zerocopy::byteorder::little_endian::{F32, U32, U64};
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout, Unaligned};

use crate::Metric;
use crate::quantization::rabitq::{BLOCK, EPS0, block_bytes};
use crate::quantization::rotation::{ROUNDS, Rotation};
use crate::quantization::sq4::{self, CenterFactors};
use crate::quantization::{padded_dim, read_f32, round_up, sq8};
use crate::search::List;
use crate::search::inmem::{InMemory, MAX_DIM, MIN_DIM};

const MAGIC: [u8; 8] = *b"COLDIVF\0";
const VERSION: u32 = 1;

/// A writer that used the other byte order would have stored it as 0x04030201.
const ENDIAN: u32 = 0x0102_0304;

/// The header's size.
pub(crate) const HEADER_BYTES: usize = 4096;

/// How much of the file opening reads first.
pub(crate) const FIRST_READ: usize = 16384;

/// Every section after the rotation starts on a multiple of this.
const ALIGN: usize = 16384;

/// Each cluster's blocks start on a multiple of this.
const LINE: usize = 128;

/// The kind of codes this version writes in each coded section, and what the 1-bit codes are measured against.
const CODES_KIND: u8 = 1;
const ROWS_KIND: u8 = 1;
const CENTERS_KIND: u8 = 1;
const ANCHOR_MEAN: u8 = 0;

/// The sections, in the order of the header's table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Part {
    Header,
    Rotation,
    Centers,
    Extents,
    Codes,
    Rows,
    Ids,
}

/// One entry of the header's section table. Its kind is its place in the table plus 1.
#[derive(Clone, Copy, FromBytes, IntoBytes, Immutable, KnownLayout, Unaligned)]
#[repr(C)]
struct Entry {
    offset: U64,
    len: U64,
    kind: U32,
    crc: U32,
}

/// The start of the header. The rest of its 4 KiB is zero.
#[derive(Clone, Copy, FromBytes, IntoBytes, Immutable, KnownLayout, Unaligned)]
#[repr(C)]
struct Header {
    magic: [u8; 8],
    version: U32,
    endian: U32,
    dim: U32,
    d_pad: U32,
    count: U64,
    nlist: U32,
    nparents: U32,
    metric: u8,
    anchor: u8,
    codes_kind: u8,
    rows_kind: u8,
    centers_kind: u8,
    reserved: [u8; 3],
    page_size: U32,
    line_size: U32,
    block: U32,
    row_bytes: U32,
    rows_per_page: U32,
    seed: U64,
    eps0: F32,
    sections: [Entry; 7],
    model_id: [u8; 64],
    epoch: U64,
    /// The CRC32 of every byte of the header before it.
    crc: U32,
}

/// One cluster's entry in the extents section.
#[derive(Clone, Copy, FromBytes, IntoBytes, Immutable, KnownLayout, Unaligned)]
#[repr(C)]
struct Extent {
    /// Where its blocks start, in bytes from the start of the codes section.
    offset: U64,
    count: U32,
    first_id: U32,
}

const _: () = assert!(size_of::<Header>() == 324 && size_of::<Entry>() == 24 && size_of::<Extent>() == 16);

/// Where a section sits in the file.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Span {
    pub(crate) offset: usize,
    pub(crate) len: usize,
}

impl Span {
    pub(crate) fn end(&self) -> usize {
        self.offset + self.len
    }

    pub(crate) fn range(&self) -> Range<usize> {
        self.offset..self.end()
    }
}

/// What a checked header says.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Layout {
    pub(crate) dim: usize,
    pub(crate) count: usize,
    pub(crate) nlist: usize,
    pub(crate) nparents: usize,
    pub(crate) metric: Metric,
    pub(crate) page_size: usize,
    pub(crate) seed: u64,
    spans: [Span; 7],
    crcs: [u32; 7],
}

impl Layout {
    pub(crate) fn span(&self, part: Part) -> Span {
        self.spans[part as usize]
    }
}

/// The cluster tables from the centers and extents sections, checked before any kernel uses them.
pub(crate) struct Tables {
    /// Where the clusters' 4-bit codes sit in the file.
    pub(crate) codes: Span,
    pub(crate) factors: Vec<CenterFactors>,
    pub(crate) parent_of: Vec<u16>,
    pub(crate) lists: Vec<List>,
}

/// An index's sections as bytes, with where each one goes. The codes and rows sections are the index's own buffers.
pub(crate) struct Encoded<'a> {
    pub(crate) spans: [Span; 7],
    pub(crate) parts: [Cow<'a, [u8]>; 7],
}

/// Lays out `index` as a file.
pub(crate) fn encode(index: &InMemory) -> Encoded<'_> {
    let rotation = encode_rotation(&index.rotation, &index.mean);
    let centers = encode_centers(index);
    let extents: Vec<u8> = index
        .lists
        .iter()
        .flat_map(|l| {
            let e = Extent {
                offset: U64::new(l.offset as u64),
                count: U32::new(l.count as u32),
                first_id: U32::new(l.first_id),
            };
            e.as_bytes().to_vec()
        })
        .collect();
    let mut ids: Vec<u8> = index.ids.iter().flat_map(|id| id.to_le_bytes()).collect();
    ids.resize(ids_len(index.ids.len()), 0);

    let mut parts = [
        Cow::Owned(Vec::new()),
        Cow::Owned(rotation),
        Cow::Owned(centers),
        Cow::Owned(extents),
        Cow::Borrowed(&index.blocks[..]),
        Cow::Borrowed(&index.rows[..]),
        Cow::Owned(ids),
    ];
    let mut spans = [Span::default(); 7];
    spans[0] = Span {
        offset: 0,
        len: HEADER_BYTES,
    };
    spans[1] = Span {
        offset: HEADER_BYTES,
        len: parts[1].len(),
    };

    for i in 2..7 {
        spans[i] = Span {
            offset: round_up(spans[i - 1].end(), ALIGN),
            len: parts[i].len(),
        };
    }

    let mut crcs = [0u32; 7];
    crcs.iter_mut()
        .zip(&parts)
        .skip(1)
        .for_each(|(c, p)| *c = crc32fast::hash(p));
    parts[0] = Cow::Owned(encode_header(index, &spans, &crcs));

    Encoded { spans, parts }
}

fn encode_header(index: &InMemory, spans: &[Span; 7], crcs: &[u32; 7]) -> Vec<u8> {
    let dim = index.dim;
    let rb = sq8::row_bytes(dim);
    let sections = std::array::from_fn(|i| Entry {
        offset: U64::new(spans[i].offset as u64),
        len: U64::new(spans[i].len as u64),
        kind: U32::new(i as u32 + 1),
        crc: U32::new(crcs[i]),
    });

    let mut header = Header {
        magic: MAGIC,
        version: U32::new(VERSION),
        endian: U32::new(ENDIAN),
        dim: U32::new(dim as u32),
        d_pad: U32::new(padded_dim(dim) as u32),
        count: U64::new(index.ids.len() as u64),
        nlist: U32::new(index.lists.len() as u32),
        nparents: U32::new((index.parents.len() / sq8::center_row_bytes(dim)) as u32),
        metric: match index.metric {
            Metric::L2 => 0,
            Metric::Cosine => 1,
        },
        anchor: ANCHOR_MEAN,
        codes_kind: CODES_KIND,
        rows_kind: ROWS_KIND,
        centers_kind: CENTERS_KIND,
        reserved: [0; 3],
        page_size: U32::new(index.page_size as u32),
        line_size: U32::new(LINE as u32),
        block: U32::new(BLOCK as u32),
        row_bytes: U32::new(rb as u32),
        rows_per_page: U32::new((index.page_size / rb) as u32),
        seed: U64::new(index.seed),
        eps0: F32::new(EPS0),
        sections,
        model_id: [0; 64],
        epoch: U64::new(0),
        crc: U32::new(0),
    };
    header.crc = U32::new(crc32fast::hash(&header.as_bytes()[..size_of::<Header>() - 4]));

    let mut out = vec![0u8; HEADER_BYTES];
    out[..size_of::<Header>()].copy_from_slice(header.as_bytes());
    out
}

/// The rotation section: the round count and the dimension as u32, the sign bits padded to 4 bytes, then the mean.
fn encode_rotation(rotation: &Rotation, mean: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(rotation_len(mean.len()));
    out.extend((ROUNDS as u32).to_le_bytes());
    out.extend((mean.len() as u32).to_le_bytes());
    out.extend(rotation.signs());
    out.resize(round_up(out.len(), 4), 0);
    mean.iter().for_each(|m| out.extend(m.to_le_bytes()));
    out
}

/// The centers section: the group rows, the 4-bit codes, then the minimums, the steps, the squared lengths and the
/// groups as arrays with one value per cluster.
fn encode_centers(index: &InMemory) -> Vec<u8> {
    let mut out = Vec::with_capacity(index.parents.len() + index.centers.len() + 14 * index.lists.len());
    out.extend(&index.parents);
    out.extend(&index.centers);

    let arrays: [fn(&CenterFactors) -> f32; 3] = [|f| f.lo, |f| f.step, |f| f.norm2];

    for get in arrays {
        index
            .center_factors
            .iter()
            .for_each(|f| out.extend(get(f).to_le_bytes()));
    }

    index.parent_of.iter().for_each(|g| out.extend(g.to_le_bytes()));
    out
}

fn rotation_len(dim: usize) -> usize {
    8 + round_up(ROUNDS * dim.div_ceil(8), 4) + 4 * dim
}

fn centers_len(dim: usize, nlist: usize, nparents: usize) -> usize {
    nparents * sq8::center_row_bytes(dim) + nlist * (sq4::code_bytes(padded_dim(dim)) + 14)
}

fn rows_len(count: usize, dim: usize, page_size: usize) -> usize {
    count.div_ceil(page_size / sq8::row_bytes(dim)) * page_size
}

fn ids_len(count: usize) -> usize {
    8 * count + 8 * count.div_ceil(64)
}

/// Checks the header at the start of `first`, part of a file of `file_len` bytes, and returns what it says. A damaged
/// or unknown header fails with `InvalidData`.
pub(crate) fn read_header(first: &[u8], file_len: usize) -> Result<Layout, Error> {
    let (h, _) = Header::read_from_prefix(first).map_err(|_| damaged("the file is too short for its header"))?;

    if h.magic != MAGIC {
        return Err(damaged("the file isn't an index"));
    }

    if h.version.get() != VERSION || h.endian.get() != ENDIAN {
        return Err(damaged("the file has an unknown version or byte order"));
    }

    if crc32fast::hash(&h.as_bytes()[..size_of::<Header>() - 4]) != h.crc.get() {
        return Err(damaged("the header's checksum doesn't match"));
    }

    let dim = h.dim.get() as usize;
    let (count, nlist, nparents) = (
        h.count.get() as usize,
        h.nlist.get() as usize,
        h.nparents.get() as usize,
    );
    let page_size = h.page_size.get() as usize;
    let rb = sq8::row_bytes(dim);
    let settings_ok = (MIN_DIM..=MAX_DIM).contains(&dim)
        && h.d_pad.get() as usize == padded_dim(dim)
        && h.block.get() as usize == BLOCK
        && h.line_size.get() as usize == LINE
        && h.row_bytes.get() as usize == rb
        && page_size.is_multiple_of(4096)
        && page_size >= rb
        && h.rows_per_page.get() as usize == page_size / rb
        && (h.anchor, h.codes_kind, h.rows_kind, h.centers_kind) == (ANCHOR_MEAN, CODES_KIND, ROWS_KIND, CENTERS_KIND)
        && nlist >= 1
        && (1..=usize::from(u16::MAX)).contains(&nparents)
        && count <= u32::MAX as usize - BLOCK;

    if !settings_ok {
        return Err(damaged("the header's settings are out of range"));
    }

    let metric = match h.metric {
        0 => Metric::L2,
        1 => Metric::Cosine,
        _ => return Err(damaged("the header names an unknown metric")),
    };

    let expected = [
        Some(HEADER_BYTES),
        Some(rotation_len(dim)),
        Some(centers_len(dim, nlist, nparents)),
        Some(16 * nlist),
        None,
        Some(rows_len(count, dim, page_size)),
        Some(ids_len(count)),
    ];
    let mut spans = [Span::default(); 7];
    let mut crcs = [0u32; 7];
    let mut end = 0;

    for (i, e) in h.sections.iter().enumerate() {
        let (offset, len) = (e.offset.get() as usize, e.len.get() as usize);
        let fits = offset.checked_add(len).is_some_and(|last| last <= file_len);
        let placed = match i {
            0 => offset == 0,
            1 => offset == HEADER_BYTES,
            _ => offset.is_multiple_of(ALIGN) && offset >= end,
        };
        let sized = expected[i].map_or(len.is_multiple_of(LINE), |want| len == want);
        if e.kind.get() as usize != i + 1 || !fits || !placed || !sized {
            return Err(damaged("a section is missing, misplaced or the wrong size"));
        }
        spans[i] = Span { offset, len };
        crcs[i] = e.crc.get();
        end = offset + len;
    }

    Ok(Layout {
        dim,
        count,
        nlist,
        nparents,
        metric,
        page_size,
        seed: h.seed.get(),
        spans,
        crcs,
    })
}

/// Checks the rotation section and returns the rotation and the mean.
pub(crate) fn read_rotation(bytes: &[u8], layout: &Layout) -> Result<(Rotation, Vec<f32>), Error> {
    check_crc(bytes, layout, Part::Rotation)?;

    let dim = layout.dim;
    let u32_at = |at: usize| u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]]);

    if u32_at(0) as usize != ROUNDS || u32_at(4) as usize != dim {
        return Err(damaged("the rotation's settings don't match the header"));
    }

    let signs = &bytes[8..8 + ROUNDS * dim.div_ceil(8)];
    let start = 8 + round_up(signs.len(), 4);
    let mean: Vec<f32> = (0..dim).map(|i| read_f32(bytes, start + 4 * i)).collect();

    if !mean.iter().all(|m| m.is_finite()) {
        return Err(damaged("the mean has a value that isn't finite"));
    }

    Ok((Rotation::from_signs(dim, signs)?, mean))
}

/// Reads the centers and extents sections from the whole file `file` and checks them.
pub(crate) fn read_tables(file: &[u8], layout: &Layout) -> Result<Tables, Error> {
    let span = layout.span(Part::Centers);
    let centers = &file[span.range()];
    let nlist = layout.nlist;
    let codes_start = layout.nparents * sq8::center_row_bytes(layout.dim);
    let codes_len = nlist * sq4::code_bytes(padded_dim(layout.dim));
    let arrays = codes_start + codes_len;
    let f32_at = |array: usize, i: usize| read_f32(centers, arrays + 4 * (array * nlist + i));

    let factors: Vec<CenterFactors> = (0..nlist)
        .map(|i| CenterFactors {
            lo: f32_at(0, i),
            step: f32_at(1, i),
            norm2: f32_at(2, i),
        })
        .collect();
    let groups = arrays + 12 * nlist;
    let parent_of: Vec<u16> = (0..nlist)
        .map(|i| u16::from_le_bytes([centers[groups + 2 * i], centers[groups + 2 * i + 1]]))
        .collect();

    let factors_ok = factors
        .iter()
        .all(|f| f.lo.is_finite() && f.step.is_finite() && f.step >= 0.0 && f.norm2.is_finite() && f.norm2 >= 0.0);

    if !factors_ok || parent_of.iter().any(|&g| usize::from(g) >= layout.nparents) {
        return Err(damaged("a cluster's factors or group are out of range"));
    }

    Ok(Tables {
        codes: Span {
            offset: span.offset + codes_start,
            len: codes_len,
        },
        factors,
        parent_of,
        lists: check_extents(&file[layout.span(Part::Extents).range()], layout)?,
    })
}

/// Reads the extents and checks them before any kernel uses them. Each cluster's blocks must start on a 128-byte line,
/// after the blocks before them and inside the codes section. The first ids must count up the vectors, and the counts
/// must add up to the index's.
pub(crate) fn check_extents(bytes: &[u8], layout: &Layout) -> Result<Vec<List>, Error> {
    let codes_len = layout.span(Part::Codes).len;
    let size = block_bytes(padded_dim(layout.dim));
    let (mut lists, mut next_id, mut free) = (Vec::with_capacity(layout.nlist), 0usize, 0usize);

    for raw in bytes.as_chunks::<16>().0 {
        let e = Extent::read_from_bytes(raw).map_err(|_| damaged("an extent is cut short"))?;
        let (offset, count, first_id) = (e.offset.get() as usize, e.count.get() as usize, e.first_id.get());
        let end = count
            .div_ceil(BLOCK)
            .checked_mul(size)
            .and_then(|len| offset.checked_add(len))
            .filter(|&end| end <= codes_len);
        if !offset.is_multiple_of(LINE) || offset < free || first_id as usize != next_id {
            return Err(damaged(
                "a cluster's blocks are out of order, off their line or misnumbered",
            ));
        }
        free = end.ok_or_else(|| damaged("a cluster's blocks run past the codes section"))?;
        next_id += count;
        lists.push(List {
            offset,
            count,
            first_id,
        });
    }

    if next_id != layout.count {
        return Err(damaged("the clusters' vectors don't add up to the index's"));
    }

    Ok(lists)
}

/// Checks every section's checksum in the whole file `file`.
pub(crate) fn verify(file: &[u8], layout: &Layout) -> Result<(), Error> {
    for part in [
        Part::Rotation,
        Part::Centers,
        Part::Extents,
        Part::Codes,
        Part::Rows,
        Part::Ids,
    ] {
        check_crc(&file[layout.span(part).range()], layout, part)?;
    }

    Ok(())
}

fn check_crc(bytes: &[u8], layout: &Layout, part: Part) -> Result<(), Error> {
    if crc32fast::hash(bytes) != layout.crcs[part as usize] {
        return Err(damaged("a section's checksum doesn't match"));
    }

    Ok(())
}

fn damaged(message: &str) -> Error {
    Error::new(ErrorKind::InvalidData, message)
}

#[cfg(test)]
#[path = "../tests/unit/format.rs"]
mod tests;
