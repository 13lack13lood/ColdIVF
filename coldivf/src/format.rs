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
//! them. The header's anchor says whether the 1-bit codes are measured from the mean or from each vector's cluster
//! center. The sections are laid out the same way for both.
//!
//! A file that maintenance wrote also has a trailer, on the next 16 KiB boundary after the ids. It holds what
//! maintenance keeps between runs: each cluster's region, a set of nearby clusters that it clusters again together, and
//! the counts that tell when to do that. It has its own magic, version and checksum. Every section ends before it, so
//! a reader that doesn't know it opens, verifies and searches the file the same way.
//!
//! An index that maintenance has changed also has a root, two slot files of 4 KiB in its folder. Each slot names a
//! generation of the index, i.e. a data file and its log, with a checksum of its own. The valid slot with the higher
//! generation is the current one.

use std::borrow::Cow;
use std::io::{Error, ErrorKind};
use std::mem::size_of;

use zerocopy::byteorder::little_endian::{F32, U32, U64};
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout, Unaligned};

use crate::io::Span;
use crate::quantization::rabitq::{BLOCK, EPS0, block_bytes};
use crate::quantization::rotation::{ROUNDS, Rotation};
use crate::quantization::sq4::{self, CenterFactors};
use crate::quantization::{padded_dim, read_f32, round_up, sq8};
use crate::search::List;
use crate::search::inmem::{InMemory, MAX_DIM, MIN_DIM};
use crate::{Anchor, Metric};

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

/// The kind of codes this version writes in each coded section.
const CODES_KIND: u8 = 1;
const ROWS_KIND: u8 = 1;
const CENTERS_KIND: u8 = 1;

/// What the 1-bit codes are measured from, the mean or each vector's cluster center.
const ANCHOR_MEAN: u8 = 0;
const ANCHOR_CLUSTER: u8 = 1;

/// The first bytes of the trailer.
const TRAILER_MAGIC: [u8; 8] = *b"COLDMNT\0";
const TRAILER_VERSION: u32 = 1;

/// The first bytes of a root slot.
const ROOT_MAGIC: [u8; 8] = *b"COLDROOT";
const ROOT_VERSION: u32 = 1;

/// A root slot's size, one block, so that the drive writes it whole or not at all as often as it can.
pub(crate) const ROOT_BYTES: usize = 4096;

/// The longest file name that a root slot holds.
pub(crate) const NAME_BYTES: usize = 64;

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

/// The start of the trailer. The cluster arrays follow, then the region arrays, then a CRC32 of every byte before it.
#[derive(Clone, Copy, FromBytes, IntoBytes, Immutable, KnownLayout, Unaligned)]
#[repr(C)]
struct TrailerHead {
    magic: [u8; 8],
    version: U32,
    nlist: U32,
    nregions: U32,
    per_region: F32,
    major_count: U64,
    major_inserts: U64,
    major_deletes: U64,
    pass_changes: U64,
}

/// A root slot. The rest of its 4 KiB is zero.
#[derive(Clone, Copy, FromBytes, IntoBytes, Immutable, KnownLayout, Unaligned)]
#[repr(C)]
struct RootSlot {
    magic: [u8; 8],
    version: U32,
    index_id: U64,
    generation: U64,
    check: U32,
    base: U64,
    writer: U64,
    data_len: u8,
    log_len: u8,
    data: [u8; NAME_BYTES],
    log: [u8; NAME_BYTES],
    /// The CRC32 of every byte of the slot before it.
    crc: U32,
}

const _: () = assert!(size_of::<Header>() == 324 && size_of::<Entry>() == 24 && size_of::<Extent>() == 16);
const _: () = assert!(size_of::<TrailerHead>() == 56 && size_of::<RootSlot>() == 182);

/// What a checked header says.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Layout {
    pub(crate) dim: usize,
    pub(crate) count: usize,
    pub(crate) nlist: usize,
    pub(crate) nparents: usize,
    pub(crate) metric: Metric,
    pub(crate) anchor: Anchor,
    pub(crate) page_size: usize,
    pub(crate) seed: u64,
    /// How many compactions the file has had.
    pub(crate) epoch: u64,
    /// The header's checksum, which tells this file apart from any other, so a log written for another file is
    /// ignored.
    pub(crate) check: u32,
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
    /// The clusters' 4-bit codes.
    pub(crate) codes: Vec<u8>,
    pub(crate) factors: Vec<CenterFactors>,
    /// Each center's squared length as its codes decode, for codes measured from the clusters' centers. It is empty
    /// with the mean anchor.
    pub(crate) norms: Vec<f64>,
    pub(crate) parent_of: Vec<u16>,
    pub(crate) lists: Vec<List>,
    /// Each cluster's count of blocks, largest first, which sizes the buffers that a fetch reads clusters into.
    pub(crate) largest: Vec<u32>,
}

/// What maintenance keeps about an index's clusters between runs, from the file's trailer.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Trailer {
    /// The region of each cluster, in file order.
    pub(crate) region_of: Vec<u32>,
    /// Each cluster's vector count when its region was last clustered.
    pub(crate) list_last: Vec<u32>,
    /// Each region's vector count when it was last clustered, and its inserts plus deletes since.
    pub(crate) region_last: Vec<u64>,
    pub(crate) region_changes: Vec<u64>,
    /// Clusters per region when the regions were made.
    pub(crate) per_region: f32,
    /// The vector count at the last major run, which clustered every vector again, and the inserts and the deletes
    /// since. An insert that replaced a vector counts as an insert only.
    pub(crate) major_count: u64,
    pub(crate) major_inserts: u64,
    pub(crate) major_deletes: u64,
    /// The inserts plus deletes since the last major run or global pass, which moved every vector to its nearest
    /// cluster.
    pub(crate) pass_changes: u64,
}

/// What a root slot says: the index's current generation and the files that hold it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Root {
    /// The index's id. Its first commit draws it at random, and it stays the same for the life of the index.
    pub(crate) index_id: u64,
    pub(crate) generation: u64,
    /// The names of the generation's data file and log, in the index's folder.
    pub(crate) data: String,
    pub(crate) log: String,
    /// The data file's header checksum.
    pub(crate) check: u32,
    /// The sequence number of the log's first record.
    pub(crate) base: u64,
    /// The id of the writer that wrote the slot.
    pub(crate) writer: u64,
}

/// An index's sections as bytes, with where each one goes. The codes and rows sections are the index's own buffers.
pub(crate) struct Encoded<'a> {
    pub(crate) spans: [Span; 7],
    pub(crate) parts: [Cow<'a, [u8]>; 7],
}

/// Lays out `index` as a file.
pub(crate) fn encode(index: &InMemory) -> Encoded<'_> {
    let [rotation, centers, extents] = small_sections(index);
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
    let spans = layout(std::array::from_fn(|i| parts[i].len()));
    let mut crcs = [0u32; 7];
    crcs.iter_mut()
        .zip(&parts)
        .skip(1)
        .for_each(|(c, p)| *c = crc32fast::hash(p));
    parts[0] = Cow::Owned(header(index, index.ids.len(), &spans, &crcs));

    Encoded { spans, parts }
}

/// The small sections of `index`: the rotation, the centers and the extents. They don't need its codes, rows or ids.
pub(crate) fn small_sections(index: &InMemory) -> [Vec<u8>; 3] {
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
    [
        encode_rotation(&index.rotation, &index.mean),
        encode_centers(index),
        extents,
    ]
}

/// Where each section goes, given every section's length. The header's comes first and is ignored. The header sits at
/// the start, the rotation right after it, and every later section on the next 16 KiB boundary.
pub(crate) fn layout(lens: [usize; 7]) -> [Span; 7] {
    let mut spans = [Span::default(); 7];
    spans[0] = Span {
        offset: 0,
        len: HEADER_BYTES,
    };
    spans[1] = Span {
        offset: HEADER_BYTES,
        len: lens[1],
    };

    for i in 2..7 {
        spans[i] = Span {
            offset: round_up(spans[i - 1].end(), ALIGN),
            len: lens[i],
        };
    }

    spans
}

/// The header of a file of `index`'s tables and `count` vectors, with these spans and section checksums.
pub(crate) fn header(index: &InMemory, count: usize, spans: &[Span; 7], crcs: &[u32; 7]) -> Vec<u8> {
    encode_header(index, count, spans, crcs)
}

fn encode_header(index: &InMemory, count: usize, spans: &[Span; 7], crcs: &[u32; 7]) -> Vec<u8> {
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
        count: U64::new(count as u64),
        nlist: U32::new(index.lists.len() as u32),
        nparents: U32::new((index.parents.len() / sq8::center_row_bytes(dim)) as u32),
        metric: match index.metric {
            Metric::L2 => 0,
            Metric::Cosine => 1,
        },
        anchor: match index.anchor {
            Anchor::Mean => ANCHOR_MEAN,
            Anchor::Cluster => ANCHOR_CLUSTER,
        },
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
        epoch: U64::new(index.epoch),
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

pub(crate) fn rows_len(count: usize, dim: usize, page_size: usize) -> usize {
    count.div_ceil(page_size / sq8::row_bytes(dim)) * page_size
}

pub(crate) fn ids_len(count: usize) -> usize {
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
        && (h.codes_kind, h.rows_kind, h.centers_kind) == (CODES_KIND, ROWS_KIND, CENTERS_KIND)
        && h.eps0.get() == EPS0
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
    let anchor = match h.anchor {
        ANCHOR_MEAN => Anchor::Mean,
        ANCHOR_CLUSTER => Anchor::Cluster,
        _ => return Err(damaged("the header names an unknown anchor")),
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
        anchor,
        page_size,
        seed: h.seed.get(),
        epoch: h.epoch.get(),
        check: h.crc.get(),
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

/// Where the cluster tables sit, from the clusters' 4-bit codes after the group rows in the centers section to the end
/// of the extents.
pub(crate) fn tables_span(layout: &Layout) -> Span {
    let start = layout.span(Part::Centers).offset + layout.nparents * sq8::center_row_bytes(layout.dim);
    Span {
        offset: start,
        len: layout.span(Part::Extents).end() - start,
    }
}

/// Checks the cluster tables in `bytes`, read from `tables_span`, and keeps the 4-bit codes from them. With the cluster
/// anchor it also sums each center's codes once for the squared length they decode to.
pub(crate) fn read_tables(mut bytes: Vec<u8>, layout: &Layout) -> Result<Tables, Error> {
    let nlist = layout.nlist;
    let codes_len = nlist * sq4::code_bytes(padded_dim(layout.dim));
    let f32_at = |array: usize, i: usize| read_f32(&bytes, codes_len + 4 * (array * nlist + i));

    let factors: Vec<CenterFactors> = (0..nlist)
        .map(|i| CenterFactors {
            lo: f32_at(0, i),
            step: f32_at(1, i),
            norm2: f32_at(2, i),
        })
        .collect();
    let groups = codes_len + 12 * nlist;
    let parent_of: Vec<u16> = (0..nlist)
        .map(|i| u16::from_le_bytes([bytes[groups + 2 * i], bytes[groups + 2 * i + 1]]))
        .collect();

    let factors_ok = factors
        .iter()
        .all(|f| f.lo.is_finite() && f.step.is_finite() && f.step >= 0.0 && f.norm2.is_finite() && f.norm2 >= 0.0);

    if !factors_ok || parent_of.iter().any(|&g| usize::from(g) >= layout.nparents) {
        return Err(damaged("a cluster's factors or group are out of range"));
    }

    let extents = layout.span(Part::Extents).offset - tables_span(layout).offset;
    let lists = check_extents(&bytes[extents..], layout)?;
    bytes.truncate(codes_len);
    bytes.shrink_to_fit();

    let norms = match layout.anchor {
        Anchor::Mean => Vec::new(),
        Anchor::Cluster => sq4::decoded_norms(&bytes, &factors, layout.dim),
    };

    let mut largest: Vec<u32> = lists.iter().map(|l| l.count.div_ceil(BLOCK) as u32).collect();
    largest.sort_unstable_by(|a, b| b.cmp(a));

    Ok(Tables {
        codes: bytes,
        factors,
        norms,
        parent_of,
        lists,
        largest,
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

/// Where the trailer starts in a file whose ids section is `ids`, the first 16 KiB boundary after it. A file that ends
/// there has no trailer.
pub(crate) fn trailer_at(ids: Span) -> usize {
    round_up(ids.end(), ALIGN)
}

/// The trailer's bytes.
pub(crate) fn encode_trailer(t: &Trailer) -> Vec<u8> {
    let head = TrailerHead {
        magic: TRAILER_MAGIC,
        version: U32::new(TRAILER_VERSION),
        nlist: U32::new(t.region_of.len() as u32),
        nregions: U32::new(t.region_last.len() as u32),
        per_region: F32::new(t.per_region),
        major_count: U64::new(t.major_count),
        major_inserts: U64::new(t.major_inserts),
        major_deletes: U64::new(t.major_deletes),
        pass_changes: U64::new(t.pass_changes),
    };
    let mut out = head.as_bytes().to_vec();
    t.region_of
        .iter()
        .chain(&t.list_last)
        .for_each(|x| out.extend(x.to_le_bytes()));
    t.region_last
        .iter()
        .chain(&t.region_changes)
        .for_each(|x| out.extend(x.to_le_bytes()));
    let crc = crc32fast::hash(&out);
    out.extend(crc.to_le_bytes());
    out
}

/// Checks the trailer in `bytes`, everything from `trailer_at` to the end of a file whose header says `layout`,
/// and returns what it says. A damaged trailer, or one written for another number of clusters, fails with
/// `InvalidData`.
pub(crate) fn read_trailer(bytes: &[u8], layout: &Layout) -> Result<Trailer, Error> {
    let (h, rest) = TrailerHead::read_from_prefix(bytes).map_err(|_| damaged("the trailer is cut short"))?;
    let (nlist, nregions) = (h.nlist.get() as usize, h.nregions.get() as usize);

    if h.magic != TRAILER_MAGIC || h.version.get() != TRAILER_VERSION {
        return Err(damaged("the trailer has an unknown magic or version"));
    }

    if nlist != layout.nlist || (8 * nlist as u64) + (16 * nregions as u64) + 4 != rest.len() as u64 {
        return Err(damaged("the trailer doesn't match the file's clusters or its own size"));
    }

    let (body, crc) = bytes.split_at(bytes.len() - 4);

    if crc32fast::hash(body) != u32::from_le_bytes([crc[0], crc[1], crc[2], crc[3]]) {
        return Err(damaged("the trailer's checksum doesn't match"));
    }

    let u32s = |from: usize, n: usize| -> Vec<u32> {
        let bytes = &rest[from..from + 4 * n];
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| u32::from_le_bytes(*b))
            .collect()
    };
    let u64s = |from: usize, n: usize| -> Vec<u64> {
        let bytes = &rest[from..from + 8 * n];
        bytes
            .as_chunks::<8>()
            .0
            .iter()
            .map(|b| u64::from_le_bytes(*b))
            .collect()
    };
    let region_of = u32s(0, nlist);
    let per_region = h.per_region.get();

    if region_of.iter().any(|&r| r as usize >= nregions) || !(per_region.is_finite() && per_region > 0.0) {
        return Err(damaged("a cluster's region or the regions' size is out of range"));
    }

    Ok(Trailer {
        region_of,
        list_last: u32s(4 * nlist, nlist),
        region_last: u64s(8 * nlist, nregions),
        region_changes: u64s(8 * nlist + 8 * nregions, nregions),
        per_region,
        major_count: h.major_count.get(),
        major_inserts: h.major_inserts.get(),
        major_deletes: h.major_deletes.get(),
        pass_changes: h.pass_changes.get(),
    })
}

/// The bytes of a root slot that says `root`. Each file name must be ASCII and at most `NAME_BYTES` long.
pub(crate) fn encode_root(root: &Root) -> Vec<u8> {
    let name = |s: &str| {
        assert!(
            s.is_ascii() && s.len() <= NAME_BYTES,
            "a root slot's file name is too long"
        );
        let mut out = [0u8; NAME_BYTES];
        out[..s.len()].copy_from_slice(s.as_bytes());
        out
    };
    let mut slot = RootSlot {
        magic: ROOT_MAGIC,
        version: U32::new(ROOT_VERSION),
        index_id: U64::new(root.index_id),
        generation: U64::new(root.generation),
        check: U32::new(root.check),
        base: U64::new(root.base),
        writer: U64::new(root.writer),
        data_len: root.data.len() as u8,
        log_len: root.log.len() as u8,
        data: name(&root.data),
        log: name(&root.log),
        crc: U32::new(0),
    };
    slot.crc = U32::new(crc32fast::hash(&slot.as_bytes()[..size_of::<RootSlot>() - 4]));
    let mut out = vec![0u8; ROOT_BYTES];
    out[..size_of::<RootSlot>()].copy_from_slice(slot.as_bytes());
    out
}

/// Checks the root slot in `bytes` and returns what it says. A damaged slot, eg. one that a crash tore, fails with
/// `InvalidData`.
pub(crate) fn read_root(bytes: &[u8]) -> Result<Root, Error> {
    let (s, _) = RootSlot::read_from_prefix(bytes).map_err(|_| damaged("the root slot is cut short"))?;

    if s.magic != ROOT_MAGIC || s.version.get() != ROOT_VERSION {
        return Err(damaged("the root slot has an unknown magic or version"));
    }

    if crc32fast::hash(&s.as_bytes()[..size_of::<RootSlot>() - 4]) != s.crc.get() {
        return Err(damaged("the root slot's checksum doesn't match"));
    }

    let name = |bytes: &[u8], len: u8| {
        let bytes = bytes.get(..usize::from(len)).filter(|b| !b.is_empty() && b.is_ascii());
        let names_a_file = |s: &&str| !s.contains(['/', '\\']) && *s != "." && *s != "..";
        let name = bytes.and_then(|b| std::str::from_utf8(b).ok()).filter(names_a_file);
        name.map(str::to_owned)
            .ok_or_else(|| damaged("the root slot names no file"))
    };

    Ok(Root {
        index_id: s.index_id.get(),
        generation: s.generation.get(),
        data: name(&s.data, s.data_len)?,
        log: name(&s.log, s.log_len)?,
        check: s.check.get(),
        base: s.base.get(),
        writer: s.writer.get(),
    })
}

/// The sections that have checksums, every one after the header.
pub(crate) const CHECKED: [Part; 6] = [
    Part::Rotation,
    Part::Centers,
    Part::Extents,
    Part::Codes,
    Part::Rows,
    Part::Ids,
];

/// Checks every section's checksum in the whole file `file`.
pub(crate) fn verify(file: &[u8], layout: &Layout) -> Result<(), Error> {
    for part in CHECKED {
        check_crc(&file[layout.span(part).range()], layout, part)?;
    }

    Ok(())
}

fn check_crc(bytes: &[u8], layout: &Layout, part: Part) -> Result<(), Error> {
    check_section(layout, part, crc32fast::hash(bytes))
}

/// Checks that `crc`, the checksum of section `part`, matches the one in the header.
pub(crate) fn check_section(layout: &Layout, part: Part, crc: u32) -> Result<(), Error> {
    if crc != layout.crcs[part as usize] {
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
