//! Writing an index file whose encoded vectors don't fit in memory. It reads the vectors twice, once for their mean and
//! once to encode them, and sends each encoded vector to a temporary spill file, tagged with its internal id. A part is
//! a run of whole clusters in file order, so it covers a run of internal ids, and each part's vectors are kept together
//! in the spill. Then it writes the codes, the rows and the ids a part at a time, in order, so the build holds one part
//! and a few small tables, whatever the number of vectors. The file has exactly the bytes of an in-memory build.

use std::fs::File;
use std::io::{Error, Write};
use std::path::{Path, PathBuf};

use rayon::prelude::*;

use super::{create_temp, side_file, threads};
use crate::format;
use crate::quantization::rabitq::{BLOCK, Factors, block_bytes, code_bytes, pack_block};
use crate::quantization::{padded_dim, read_f32, sq8};
use crate::search::inmem::{BATCH, Encoded, Encoding, InMemory, chunks};
use crate::{BuildParams, Clusters, VectorSource};

/// The most bytes of encoded vectors that one part holds while it is written out.
const PART_BYTES: usize = 256 << 20;

/// The bytes of spill that each part gathers before they go to the file.
const SPILL_BUFFER: usize = 512 << 10;

/// Builds the index from `src` and an imported clustering and writes it to `out`, a part at a time. Like `write`, it
/// writes a temporary file next to `out`, with the header last, syncs it and puts it in place as `root::publish` says.
pub(crate) fn write_streaming(
    params: &BuildParams,
    src: &dyn VectorSource,
    clusters: &Clusters,
    out: &Path,
) -> Result<(), Error> {
    write_parts(params, src, clusters, out, PART_BYTES)
}

/// The same, with parts of at most `part_bytes` bytes of encoded vectors, unless one cluster alone holds more.
pub(crate) fn write_parts(
    params: &BuildParams,
    src: &dyn VectorSource,
    clusters: &Clusters,
    out: &Path,
    part_bytes: usize,
) -> Result<(), Error> {
    let pool = threads(params.threads)?;
    let (frame, id_of) = InMemory::frame(params, src, clusters, None, &pool)?;
    let n = src.count();
    let shape = Shape::of(&frame);
    let parts = parts(&frame, part_bytes / shape.record);
    let spill = Spill::create(out)?;
    let lists_end = frame
        .lists
        .last()
        .map_or(0, |l| l.offset + l.count.div_ceil(BLOCK) * shape.block);
    let small = format::small_sections(&frame);
    let lens = [
        0,
        small[0].len(),
        small[1].len(),
        small[2].len(),
        lists_end,
        format::rows_len(n, shape.dim, shape.page_size),
        format::ids_len(n),
    ];
    let spans = format::layout(lens);
    let (file, tmp) = create_temp(out)?;
    let result = (|| {
        let chunks = spill.fill(&frame, src, &id_of, &parts, &shape, &pool)?;
        file.set_len(spans[6].end() as u64)?;
        let mut crcs = [0u32; 7];
        (crcs[4], crcs[5], crcs[6]) = assemble(&file, &spans, &frame, &parts, &chunks, &spill, &shape, n)?;

        for (i, bytes) in small.iter().enumerate() {
            write_at(&file, bytes, spans[i + 1].offset)?;
            crcs[i + 1] = crc32fast::hash(bytes);
        }

        // The header goes last, once every other section is on the disk, so a half-written file never opens.
        file.sync_data()?;
        write_at(&file, &format::header(&frame, n, &spans, &crcs), 0)?;
        file.sync_all()?;
        crate::root::publish(out, &tmp)
    })();

    if result.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }

    result
}

/// The sizes that encoding and writing need.
struct Shape {
    dim: usize,
    d_pad: usize,
    page_size: usize,
    /// The bytes of one vector's 1-bit code, of a block of 32 codes, and of an 8-bit row.
    code: usize,
    block: usize,
    row: usize,
    /// The bytes of one vector in the spill: its internal id, code, four factors and row.
    record: usize,
}

impl Shape {
    fn of(frame: &InMemory) -> Shape {
        let (dim, d_pad) = (frame.dim, padded_dim(frame.dim));
        let (code, row) = (code_bytes(d_pad), sq8::row_bytes(dim));
        Shape {
            dim,
            d_pad,
            page_size: frame.page_size,
            code,
            block: block_bytes(d_pad),
            row,
            record: 4 + code + 16 + row,
        }
    }
}

/// A run of whole clusters in file order, which covers the internal ids `first_id..end_id`.
struct Part {
    first_list: usize,
    end_list: usize,
    first_id: usize,
    end_id: usize,
}

/// Splits the clusters, in file order, into parts of at most `most` vectors each, unless one cluster alone holds more.
fn parts(frame: &InMemory, most: usize) -> Vec<Part> {
    let most = most.max(1);
    let mut parts: Vec<Part> = Vec::new();

    for (i, list) in frame.lists.iter().enumerate() {
        let end_id = list.first_id as usize + list.count;
        match parts.last_mut() {
            Some(p) if end_id - p.first_id <= most => (p.end_list, p.end_id) = (i + 1, end_id),
            _ => parts.push(Part {
                first_list: i,
                end_list: i + 1,
                first_id: list.first_id as usize,
                end_id,
            }),
        }
    }

    parts
}

/// The spill file, next to the index file, removed when dropped.
struct Spill {
    file: File,
    path: PathBuf,
}

impl Spill {
    fn create(out: &Path) -> Result<Spill, Error> {
        let (file, path) = side_file(out, "spill")?;
        Ok(Spill { file, path })
    }

    /// Encodes every vector of `src` and writes it to the spill in its part's runs. The threads encode a batch of
    /// chunks at a time, and this thread sorts the results into the parts' buffers. Returns where each part's runs
    /// are, as offsets and lengths in the spill.
    fn fill(
        &self,
        frame: &InMemory,
        src: &dyn VectorSource,
        id_of: &[u32],
        parts: &[Part],
        shape: &Shape,
        pool: &rayon::ThreadPool,
    ) -> Result<Vec<Vec<(u64, usize)>>, Error> {
        let encoding = Encoding::new(frame, None, id_of);
        let chunks = chunks(src);
        let batch = BATCH * pool.current_num_threads();
        let mut buffers: Vec<Vec<u8>> = parts.iter().map(|_| Vec::new()).collect();
        let mut runs: Vec<Vec<(u64, usize)>> = parts.iter().map(|_| Vec::new()).collect();
        let (mut writer, mut at) = (&self.file, 0u64);
        let mut first = 0;

        for group in chunks.chunks(batch) {
            let starts: Vec<usize> = group
                .iter()
                .scan(first, |next, c| {
                    let start = *next;
                    *next += c.len() / shape.dim;
                    Some(start)
                })
                .collect();
            let encoded: Vec<Result<Encoded, Error>> = pool.install(|| {
                group
                    .par_iter()
                    .zip(&starts)
                    .map(|(chunk, &start)| encoding.chunk(chunk, start))
                    .collect()
            });
            for (e, &start) in encoded.into_iter().zip(&starts) {
                let e = e?;
                for (j, f) in e.factors.iter().enumerate() {
                    let id = id_of[start + j] as usize;
                    let p = parts.partition_point(|p| p.end_id <= id);
                    let buffer = &mut buffers[p];
                    buffer.extend((id as u32).to_le_bytes());
                    buffer.extend_from_slice(&e.codes[j * shape.code..(j + 1) * shape.code]);
                    [f.f_add, f.f_rescale, f.f_pop, f.f_err]
                        .iter()
                        .for_each(|x| buffer.extend(x.to_le_bytes()));
                    buffer.extend_from_slice(&e.rows[j * shape.row..(j + 1) * shape.row]);
                    if buffer.len() >= SPILL_BUFFER {
                        writer.write_all(buffer)?;
                        runs[p].push((at, buffer.len()));
                        at += buffer.len() as u64;
                        buffer.clear();
                    }
                }
                first = start + e.factors.len();
            }
        }

        for (p, buffer) in buffers.iter().enumerate().filter(|(_, b)| !b.is_empty()) {
            writer.write_all(buffer)?;
            runs[p].push((at, buffer.len()));
            at += buffer.len() as u64;
        }

        writer.flush()?;
        Ok(runs)
    }
}

impl Drop for Spill {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

/// Writes the codes, rows and ids sections of the file a part at a time, from the spill, and returns their checksums.
#[allow(clippy::too_many_arguments)]
fn assemble(
    file: &File,
    spans: &[crate::io::Span; 7],
    frame: &InMemory,
    parts: &[Part],
    runs: &[Vec<(u64, usize)>],
    spill: &Spill,
    shape: &Shape,
    n: usize,
) -> Result<(u32, u32, u32), Error> {
    let largest = parts.iter().map(|p| p.end_id - p.first_id).max().unwrap_or(0);
    let (mut codes, mut factors) = (vec![0u8; largest * shape.code], vec![Factors::EMPTY; largest]);
    let mut rows = vec![0u8; largest * shape.row];
    let mut read = Vec::new();
    let (mut t1, mut t2, mut ids) = (
        Section::new(file, spans[4].offset),
        Section::new(file, spans[5].offset),
        Section::new(file, spans[6].offset),
    );
    let per_page = shape.page_size / shape.row;
    let (mut page, mut in_page) = (vec![0u8; shape.page_size], 0);
    let mut block = vec![0u8; shape.block];

    for (part, runs) in parts.iter().zip(runs) {
        for &(at, len) in runs {
            read.resize(len, 0);
            crate::io::read_at(&spill.file, &mut read, at as usize)?;
            for record in read.chunks_exact(shape.record) {
                let i = u32::from_le_bytes([record[0], record[1], record[2], record[3]]) as usize - part.first_id;
                let (code, rest) = record[4..].split_at(shape.code);
                codes[i * shape.code..(i + 1) * shape.code].copy_from_slice(code);
                factors[i] = Factors {
                    f_add: read_f32(rest, 0),
                    f_rescale: read_f32(rest, 4),
                    f_pop: read_f32(rest, 8),
                    f_err: read_f32(rest, 12),
                };
                rows[i * shape.row..(i + 1) * shape.row].copy_from_slice(&rest[16..]);
            }
        }
        for list in &frame.lists[part.first_list..part.end_list] {
            let first = list.first_id as usize - part.first_id;
            for start in (first..first + list.count).step_by(BLOCK) {
                let end = (start + BLOCK).min(first + list.count);
                block.fill(0);
                pack_block(
                    &codes[start * shape.code..end * shape.code],
                    &factors[start..end],
                    shape.d_pad,
                    &mut block,
                );
                t1.write(&block)?;
            }
        }
        for row in rows[..(part.end_id - part.first_id) * shape.row].chunks_exact(shape.row) {
            page[in_page * shape.row..(in_page + 1) * shape.row].copy_from_slice(row);
            in_page += 1;
            if in_page == per_page {
                t2.write(&page)?;
                page.fill(0);
                in_page = 0;
            }
            ids.write(&sq8::id(row, shape.dim).to_le_bytes())?;
        }
    }

    if in_page > 0 {
        t2.write(&page)?;
    }

    // The ids section ends in the tombstones, one bit per vector, all clear at build time.
    ids.write(&vec![0u8; 8 * n.div_ceil(64)])?;
    Ok((t1.finish()?, t2.finish()?, ids.finish()?))
}

/// Where a section's bytes go: a file, or a maintenance job's gated writes to one.
pub(crate) trait Sink: Sync {
    /// Writes all of `bytes` at `offset`.
    fn write_at(&self, bytes: &[u8], offset: usize) -> Result<(), Error>;

    /// The same, for bytes that should stay in the system's cache of file pages once written, eg. what searches of a
    /// warm index read.
    fn write_kept_at(&self, bytes: &[u8], offset: usize) -> Result<(), Error> {
        self.write_at(bytes, offset)
    }
}

impl Sink for File {
    fn write_at(&self, bytes: &[u8], offset: usize) -> Result<(), Error> {
        write_at(self, bytes, offset)
    }
}

/// How far a section got: the offset it reached in the file, and the checksum and length of what it wrote.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Reached {
    pub(crate) at: usize,
    pub(crate) crc: u32,
    pub(crate) len: u64,
}

/// A section of the file written in order through a buffer, with a running checksum.
pub(crate) struct Section<'a, S: Sink + ?Sized = File> {
    out: &'a S,
    at: usize,
    len: u64,
    buffer: Vec<u8>,
    /// The parts of the buffer that `write_kept` wrote, in order.
    kept: Vec<std::ops::Range<usize>>,
    crc: crc32fast::Hasher,
}

impl<'a, S: Sink + ?Sized> Section<'a, S> {
    pub(crate) fn new(out: &'a S, at: usize) -> Section<'a, S> {
        Section::resume(out, Reached { at, crc: 0, len: 0 })
    }

    /// The section as `reached` left it, eg. in an earlier try of the same write.
    pub(crate) fn resume(out: &'a S, reached: Reached) -> Section<'a, S> {
        Section {
            out,
            at: reached.at,
            len: reached.len,
            buffer: Vec::with_capacity(SPILL_BUFFER),
            kept: Vec::new(),
            crc: crc32fast::Hasher::new_with_initial_len(reached.crc, reached.len),
        }
    }

    /// Writes what the buffer holds and returns how far the section got.
    pub(crate) fn reached(&mut self) -> Result<Reached, Error> {
        self.flush()?;
        Ok(Reached {
            at: self.at,
            crc: self.crc.clone().finalize(),
            len: self.len,
        })
    }

    pub(crate) fn write(&mut self, bytes: &[u8]) -> Result<(), Error> {
        self.crc.update(bytes);
        self.len += bytes.len() as u64;
        self.buffer.extend_from_slice(bytes);

        if self.buffer.len() >= SPILL_BUFFER {
            self.flush()?;
        }

        Ok(())
    }

    /// The same, for bytes that should stay in the system's cache once written, which go out through `write_kept_at`.
    pub(crate) fn write_kept(&mut self, bytes: &[u8]) -> Result<(), Error> {
        let start = self.buffer.len();
        match self.kept.last_mut() {
            Some(last) if last.end == start => last.end += bytes.len(),
            _ => self.kept.push(start..start + bytes.len()),
        }
        self.write(bytes)
    }

    fn flush(&mut self) -> Result<(), Error> {
        let mut done = 0;

        for kept in std::mem::take(&mut self.kept) {
            if kept.start > done {
                self.out.write_at(&self.buffer[done..kept.start], self.at + done)?;
            }
            self.out
                .write_kept_at(&self.buffer[kept.clone()], self.at + kept.start)?;
            done = kept.end;
        }

        if done < self.buffer.len() {
            self.out.write_at(&self.buffer[done..], self.at + done)?;
        }

        self.at += self.buffer.len();
        self.buffer.clear();
        Ok(())
    }

    /// Writes what is left and returns the section's checksum.
    pub(crate) fn finish(mut self) -> Result<u32, Error> {
        self.flush()?;
        Ok(self.crc.finalize())
    }
}

/// Writes all of `bytes` at `offset` of `file`.
pub(crate) fn write_at(file: &File, bytes: &[u8], offset: usize) -> Result<(), Error> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt;
        file.write_all_at(bytes, offset as u64)
    }

    #[cfg(windows)]
    {
        use std::os::windows::fs::FileExt;
        let mut done = 0;

        while done < bytes.len() {
            let n = file.seek_write(&bytes[done..], (offset + done) as u64)?;
            if n == 0 {
                return Err(Error::new(std::io::ErrorKind::WriteZero, "the file took no more bytes"));
            }
            done += n;
        }

        Ok(())
    }
}

#[cfg(test)]
#[path = "../../tests/unit/build/stream.rs"]
mod tests;
