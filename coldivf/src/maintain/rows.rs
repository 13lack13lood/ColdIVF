//! The vectors that maintenance clusters. The index keeps each vector only as its 8-bit row, so maintenance reads the
//! rows of the old file and of the changes, and decodes them to f32 a block at a time for k-means.

use std::cell::Cell;
use std::io::{Error, ErrorKind};

use rayon::ThreadPool;
use rayon::prelude::*;

use crate::build::kmeans::Rows;
use crate::format::Part;
use crate::index::Inner;
use crate::io;
use crate::quantization::sq8;
use crate::update::Delta;

/// The most bytes that one read of the old file takes.
const READ: usize = 4 << 20;

/// The most bytes that no vector needs between two pages that some do, which still share one read. The rows read here
/// are a few clusters spread over the file, so a larger gap mostly reads rows that nothing needs.
const GAP: usize = 32 << 10;

/// The vectors that `sums` decodes at a time.
const SUMS: usize = 8192;

/// The vectors that k-means reads, as their 8-bit rows held in memory back to back.
pub(crate) struct Held<'a> {
    pub(crate) rows: Vec<u8>,
    pub(crate) dim: usize,
    pub(crate) mean: &'a [f32],
}

impl Rows for Held<'_> {
    fn count(&self) -> usize {
        self.rows.len() / sq8::row_bytes(self.dim)
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn load(&self, ids: &[u32], out: &mut [f32], pool: &ThreadPool) -> Result<(), Error> {
        let rb = sq8::row_bytes(self.dim);
        decode_all(
            ids.iter().map(|&i| &self.rows[i as usize * rb..(i as usize + 1) * rb]),
            self.mean,
            out,
            pool,
        )
    }
}

/// The vectors `ids` of an old file and its changes, which k-means reads from the file each time, so they take no
/// memory between reads. Its row i is vector `ids[i]`, and `ids` is in increasing order. It counts the bytes it reads.
pub(crate) struct Streamed<'a> {
    old: &'a Inner,
    delta: &'a Delta,
    ids: &'a [u32],
    pub(crate) read: Cell<u64>,
}

impl<'a> Streamed<'a> {
    pub(crate) fn new(old: &'a Inner, delta: &'a Delta, ids: &'a [u32]) -> Streamed<'a> {
        Streamed {
            old,
            delta,
            ids,
            read: Cell::new(0),
        }
    }
}

impl Rows for Streamed<'_> {
    fn count(&self) -> usize {
        self.ids.len()
    }

    fn dim(&self) -> usize {
        self.old.dim()
    }

    fn load(&self, ids: &[u32], out: &mut [f32], pool: &ThreadPool) -> Result<(), Error> {
        let rb = sq8::row_bytes(self.dim());
        let vectors: Vec<u32> = ids.iter().map(|&i| self.ids[i as usize]).collect();
        let mut rows = Vec::new();
        let read = read_rows(self.old, self.delta, &vectors, &mut rows)?;
        self.read.set(self.read.get() + read);
        decode_all(rows.chunks_exact(rb), self.old.mean(), out, pool)
    }
}

/// Decodes `rows` one after another into `out` on the pool's threads. A row that decodes to a value that isn't finite
/// is damaged and fails with `InvalidData`.
fn decode_all<'r>(
    rows: impl Iterator<Item = &'r [u8]>,
    mean: &[f32],
    out: &mut [f32],
    pool: &ThreadPool,
) -> Result<(), Error> {
    let rows: Vec<&[u8]> = rows.collect();
    let dim = mean.len();
    let ok = pool.install(|| {
        out.par_chunks_mut(dim).zip(&rows).all(|(o, row)| {
            sq8::decode(row, mean, o);
            o.iter().all(|x| x.is_finite())
        })
    });

    if !ok {
        return Err(damaged_row());
    }

    Ok(())
}

/// Reads the 8-bit rows of the vectors `ids`, in increasing order, into `out`, one after another. A vector id below the
/// old file's count is a file row's internal id, and one above it is the file's count plus an insert's place. It reads
/// the old file through the cold handle, in runs of whole pages of at most `READ` bytes. Returns the bytes it read.
pub(crate) fn read_rows(old: &Inner, delta: &Delta, ids: &[u32], out: &mut Vec<u8>) -> Result<u64, Error> {
    let layout = old.layout();
    let (rb, page_size, count) = (sq8::row_bytes(layout.dim), layout.page_size, layout.count);
    let per_page = page_size / rb;
    let section = layout.span(Part::Rows).offset;
    let (most, gap) = ((READ / page_size).max(1), GAP / page_size);
    let (mut buf, mut read) = (Vec::new(), 0u64);
    out.resize(ids.len() * rb, 0);
    let file_ids = ids.partition_point(|&v| (v as usize) < count);
    let mut i = 0;

    while i < file_ids {
        let first = ids[i] as usize / per_page;
        let (mut j, mut last) = (i, first);
        while j < file_ids {
            let p = ids[j] as usize / per_page;
            if p > last + gap + 1 || p - first >= most {
                break;
            }
            last = p;
            j += 1;
        }
        buf.resize((last + 1 - first) * page_size, 0);
        io::read_at(old.cold_file(), &mut buf, section + first * page_size)?;
        read += buf.len() as u64;
        for (k, &v) in ids[i..j].iter().enumerate() {
            let v = v as usize;
            let at = (v / per_page - first) * page_size + (v % per_page) * rb;
            out[(i + k) * rb..(i + k + 1) * rb].copy_from_slice(&buf[at..at + rb]);
        }
        i = j;
    }

    for (k, &v) in ids[file_ids..].iter().enumerate() {
        let at = (file_ids + k) * rb;
        out[at..at + rb].copy_from_slice(delta.row(v as usize - count));
    }

    Ok(read)
}

/// The sums, in f64, of the decoded rows of the vectors `ids`, in increasing order, by their groups `group`, one group
/// after another, for the groups' means. It reads `SUMS` vectors at a time and adds each group's rows in the order of
/// `ids`, so the sums are the same for any number of threads. Returns the sums and the bytes read.
pub(crate) fn sums(
    old: &Inner,
    delta: &Delta,
    ids: &[u32],
    group: &[u32],
    groups: usize,
    pool: &ThreadPool,
) -> Result<(Vec<f64>, u64), Error> {
    let (dim, mean) = (old.dim(), old.mean());
    let rb = sq8::row_bytes(dim);
    let mut out = vec![0f64; groups * dim];
    let (mut rows, mut decoded, mut read) = (Vec::new(), Vec::new(), 0u64);

    for (ids, group) in ids.chunks(SUMS).zip(group.chunks(SUMS)) {
        read += read_rows(old, delta, ids, &mut rows)?;
        decoded.resize(ids.len() * dim, 0.0);
        decode_all(rows.chunks_exact(rb), mean, &mut decoded, pool)?;
        for (o, &g) in decoded.chunks_exact(dim).zip(group) {
            let sum = &mut out[g as usize * dim..(g as usize + 1) * dim];
            sum.iter_mut().zip(o).for_each(|(s, &x)| *s += f64::from(x));
        }
    }

    Ok((out, read))
}

pub(crate) fn damaged_row() -> Error {
    Error::new(ErrorKind::InvalidData, "a vector's 8-bit row is damaged")
}
