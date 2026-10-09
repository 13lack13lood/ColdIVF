//! A storage provider for DiskANN's builder that serves the first rows of a base file as a file of their own.
//!
//! A base file holds a row count and a row width as two little-endian 32-bit integers, then the rows as 32-bit
//! floats, and DiskANN's builder reads its input in the same layout. The provider answers for one made-up path with a
//! header that gives the prefix's row count, followed by the base file's own bytes, so the prefix is never copied.
//! Every other path goes to the file system.

use std::fs::File;
use std::io::{BufReader, BufWriter, Error, ErrorKind, Read, Seek, SeekFrom};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

use diskann_providers::storage::{FileStorageProvider, StorageReadProvider, StorageWriteProvider};

/// The bytes of a base file's header.
pub const HEADER: u64 = 8;

/// How much a prefix reader reads at a time when asked for less.
const CHUNK: usize = 1 << 20;

/// The row count and the row width in the header of the base file at `path`, checked against the file's size.
pub fn header(path: &Path) -> Result<(usize, usize), Error> {
    let file = File::open(path)?;
    let mut head = [0u8; HEADER as usize];
    file.read_exact_at(&mut head, 0)?;
    let count = u32::from_le_bytes([head[0], head[1], head[2], head[3]]) as u64;
    let width = u32::from_le_bytes([head[4], head[5], head[6], head[7]]) as u64;

    if width == 0 || file.metadata()?.len() != HEADER + count * width * 4 {
        return Err(Error::new(
            ErrorKind::InvalidData,
            format!(
                "{} doesn't hold the {count} rows of width {width} its header says",
                path.display()
            ),
        ));
    }

    Ok((count as usize, width as usize))
}

/// The file system, plus one read-only file that holds rows 0 to `rows` of a base file.
pub struct PrefixStorage {
    name: String,
    base: PathBuf,
    rows: u32,
    width: u32,
}

impl PrefixStorage {
    /// A provider whose file `name()` holds the first `rows` rows of the base file at `base`.
    pub fn new(base: &Path, rows: usize) -> Result<PrefixStorage, Error> {
        let (count, width) = header(base)?;

        if rows == 0 || rows > count {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                format!("{} has {count} rows, so it has no prefix of {rows}", base.display()),
            ));
        }

        Ok(PrefixStorage {
            name: format!("{}#rows={rows}", base.display()),
            base: base.to_path_buf(),
            rows: rows as u32,
            width: width as u32,
        })
    }

    /// The path the prefix answers to. No file has it.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The prefix's length in bytes, its header included.
    pub fn len(&self) -> u64 {
        HEADER + self.rows as u64 * self.width as u64 * 4
    }

    pub fn is_empty(&self) -> bool {
        self.rows == 0
    }

    fn open_prefix(&self) -> Result<PrefixReader, Error> {
        let mut head = [0u8; HEADER as usize];
        head[..4].copy_from_slice(&self.rows.to_le_bytes());
        head[4..].copy_from_slice(&self.width.to_le_bytes());
        Ok(PrefixReader {
            file: File::open(&self.base)?,
            head,
            len: self.len(),
            pos: 0,
            buf: Vec::new(),
            buf_at: 0,
        })
    }

    fn refuse(&self, item: &str) -> Result<(), Error> {
        if item == self.name {
            return Err(Error::new(
                ErrorKind::PermissionDenied,
                format!("{item} is a read-only prefix"),
            ));
        }

        Ok(())
    }
}

impl StorageReadProvider for PrefixStorage {
    type Reader = Reader;

    fn open_reader(&self, item: &str) -> Result<Reader, Error> {
        if item == self.name {
            return Ok(Reader::Prefix(self.open_prefix()?));
        }

        Ok(Reader::File(FileStorageProvider.open_reader(item)?))
    }

    fn get_length(&self, item: &str) -> Result<u64, Error> {
        if item == self.name {
            return Ok(self.len());
        }

        FileStorageProvider.get_length(item)
    }

    fn exists(&self, item: &str) -> bool {
        item == self.name || FileStorageProvider.exists(item)
    }
}

impl StorageWriteProvider for PrefixStorage {
    type Writer = BufWriter<File>;

    fn open_writer(&self, item: &str) -> Result<BufWriter<File>, Error> {
        self.refuse(item)?;
        FileStorageProvider.open_writer(item)
    }

    fn create_for_write(&self, item: &str) -> Result<BufWriter<File>, Error> {
        self.refuse(item)?;
        FileStorageProvider.create_for_write(item)
    }

    fn delete(&self, item: &str) -> Result<(), Error> {
        self.refuse(item)?;
        FileStorageProvider.delete(item)
    }
}

/// A reader of a real file or of the prefix.
pub enum Reader {
    File(BufReader<File>),
    Prefix(PrefixReader),
}

impl Read for Reader {
    fn read(&mut self, out: &mut [u8]) -> Result<usize, Error> {
        match self {
            Reader::File(r) => r.read(out),
            Reader::Prefix(r) => r.read(out),
        }
    }
}

impl Seek for Reader {
    fn seek(&mut self, to: SeekFrom) -> Result<u64, Error> {
        match self {
            Reader::File(r) => r.seek(to),
            Reader::Prefix(r) => r.seek(to),
        }
    }
}

/// Reads the made-up header, then the base file's bytes from the same offsets, up to the prefix's end. It reads the
/// base with positioned reads, so a seek moves only its own position.
pub struct PrefixReader {
    file: File,
    head: [u8; HEADER as usize],
    len: u64,
    pos: u64,
    /// The last chunk read from the base, which starts at byte `buf_at`.
    buf: Vec<u8>,
    buf_at: u64,
}

impl Read for PrefixReader {
    fn read(&mut self, out: &mut [u8]) -> Result<usize, Error> {
        if self.pos >= self.len || out.is_empty() {
            return Ok(0);
        }

        if self.pos < HEADER {
            let n = out.len().min((HEADER - self.pos) as usize);
            out[..n].copy_from_slice(&self.head[self.pos as usize..self.pos as usize + n]);
            self.pos += n as u64;
            return Ok(n);
        }

        let want = out.len().min((self.len - self.pos) as usize);
        let buffered = self.pos >= self.buf_at && self.pos < self.buf_at + self.buf.len() as u64;

        if !buffered && want >= CHUNK {
            let n = self.file.read_at(&mut out[..want], self.pos)?;
            self.pos += n as u64;
            return Ok(n);
        }

        if !buffered {
            let size = CHUNK.min((self.len - self.pos) as usize);
            self.buf.resize(size, 0);
            self.file.read_exact_at(&mut self.buf, self.pos)?;
            self.buf_at = self.pos;
        }

        let start = (self.pos - self.buf_at) as usize;
        let n = want.min(self.buf.len() - start);
        out[..n].copy_from_slice(&self.buf[start..start + n]);
        self.pos += n as u64;
        Ok(n)
    }
}

impl Seek for PrefixReader {
    fn seek(&mut self, to: SeekFrom) -> Result<u64, Error> {
        let target = match to {
            SeekFrom::Start(p) => Some(p),
            SeekFrom::End(d) => self.len.checked_add_signed(d),
            SeekFrom::Current(d) => self.pos.checked_add_signed(d),
        };

        self.pos = target.ok_or_else(|| Error::new(ErrorKind::InvalidInput, "seek before the start of the prefix"))?;
        Ok(self.pos)
    }
}

#[cfg(test)]
#[path = "../tests/unit/prefix.rs"]
mod tests;
