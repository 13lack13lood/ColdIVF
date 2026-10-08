//! Reading index files, through the memory map or through the read waves that fetch what a cold query needs. A wave is
//! one batch of reads. Its backend starts them together, so they overlap on the disk, then hands over each range as it
//! arrives. Besides the kernels, this is the only module with `unsafe` code.

#[cfg(target_os = "macos")]
pub(crate) mod macos;
pub(crate) mod map;
pub(crate) mod pool;
pub(crate) mod portable;

use std::fs::File;
use std::io::{Error, ErrorKind};
use std::path::Path;

/// A run of bytes in a file, eg. a section or one cluster's blocks.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Span {
    pub(crate) offset: usize,
    pub(crate) len: usize,
}

impl Span {
    pub(crate) fn end(&self) -> usize {
        self.offset + self.len
    }

    pub(crate) fn range(&self) -> std::ops::Range<usize> {
        self.offset..self.end()
    }
}

/// One read of a wave. It starts on a page boundary and ends on one or at the end of the file. It covers the spans from
/// `spans[tag]` up to the next range's `tag`, and its bytes land in `arena[at..at + len]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Range {
    pub(crate) offset: usize,
    pub(crate) len: usize,
    pub(crate) tag: usize,
    pub(crate) at: usize,
}

impl Range {
    pub(crate) fn end(&self) -> usize {
        self.offset + self.len
    }
}

/// Plans the reads for `spans`, which must be sorted by offset and inside a file of `file_len` bytes. It rounds each
/// span out to `page` boundaries, cuts it at the end of the file, and merges ranges that touch or overlap. The ranges
/// go into `ranges` in order, packed one after another in the arena, and where each span's bytes land goes into `at`.
pub(crate) fn coalesce(spans: &[Span], page: usize, file_len: usize, ranges: &mut Vec<Range>, at: &mut Vec<usize>) {
    ranges.clear();
    at.clear();
    let mut next = 0;

    for (tag, s) in spans.iter().enumerate() {
        let start = s.offset / page * page;
        let end = s.end().div_ceil(page).saturating_mul(page).min(file_len);
        let r = match ranges.last_mut() {
            Some(r) if start <= r.end() => {
                let grow = end.saturating_sub(r.end());
                r.len += grow;
                next += grow;
                *r
            }
            _ => {
                let r = Range {
                    offset: start,
                    len: end - start,
                    tag,
                    at: next,
                };
                ranges.push(r);
                next += r.len;
                r
            }
        };
        at.push(r.at + (s.offset - r.offset));
    }
}

/// The most arena bytes that `coalesce` packs spans of these lengths into. Rounding out adds less than two pages to a
/// span, and merging only saves bytes.
pub(crate) fn arena_bound(lens: impl Iterator<Item = usize>, page: usize) -> usize {
    lens.map(|len| len + 2 * page).sum()
}

/// How many waves a backend keeps going at once, one per lane, so a search can start one wave before it waits for
/// another.
pub(crate) const LANES: usize = 2;

/// One search context's state for a read backend. Its waves read through the index's cold handle, which every context
/// shares.
pub(crate) trait Wave: Send {
    /// The page size that its reads round out to.
    fn page(&self) -> usize;

    /// Starts reading `ranges` from `file` into `arena` on lane `lane`, and may return before the reads finish. A lane
    /// holds one wave, so the ranges of its earlier submit that `next` hasn't returned are dropped.
    fn submit(&mut self, lane: usize, file: &File, ranges: &[Range], arena: &mut [u8]) -> Result<(), Error>;

    /// Waits until one more range of the lane's last submit is in `arena` and returns its index, or `None` once every
    /// range has been returned. It returns each range once, in any order. `ranges` and `arena` must be that submit's.
    fn next(&mut self, lane: usize, file: &File, ranges: &[Range], arena: &mut [u8]) -> Result<Option<usize>, Error>;
}

/// Makes a backend's state for one search context.
pub(crate) type NewWave = fn() -> Box<dyn Wave>;

/// An index file's handle for cold reads, and the backend that reads through it.
pub(crate) struct Cold {
    pub(crate) file: File,
    pub(crate) wave: NewWave,
}

/// Opens `path` again for cold reads, with the fastest backend that works on its file system. `first` is the start of
/// the file as opening read it. The header in it holds a checksum of every section, so if a rebuild renamed a new file
/// over `path` in between, the check fails instead of mixing two files' bytes.
pub(crate) fn open_cold(path: &Path, first: &[u8]) -> Result<Cold, Error> {
    let cold = File::open(path)?;
    let mut again = vec![0u8; first.len()];
    read_at(&cold, &mut again, 0)?;

    if again != first {
        return Err(Error::new(
            ErrorKind::InvalidData,
            "the file changed while it was being opened",
        ));
    }

    #[cfg(target_os = "macos")]
    {
        macos::uncache(&cold)?;

        if macos::takes_hints(&cold)? {
            return Ok(Cold {
                file: cold,
                wave: || Box::<macos::RdAdviseWave>::default(),
            });
        }
    }

    Ok(Cold {
        file: cold,
        wave: || Box::<pool::PoolWave>::default(),
    })
}

/// Reads exactly `buf.len()` bytes from `offset` of `file`. On Unix it leaves the file position alone. On Windows it
/// moves the handle's cursor, which nothing here uses.
pub(crate) fn read_at(file: &File, buf: &mut [u8], offset: usize) -> Result<(), Error> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt;
        file.read_exact_at(buf, offset as u64)
    }

    #[cfg(windows)]
    {
        use std::os::windows::fs::FileExt;
        let mut done = 0;

        while done < buf.len() {
            let n = file.seek_read(&mut buf[done..], (offset + done) as u64)?;
            if n == 0 {
                return Err(Error::new(
                    ErrorKind::UnexpectedEof,
                    "the file ended before the read did",
                ));
            }
            done += n;
        }

        Ok(())
    }
}

/// Tells the operating system (OS) that `len` bytes from `offset` of `file` will be read soon, so it can start reading
/// them into its cache. A refused hint changes nothing, so errors are ignored.
pub(crate) fn will_need(file: &File, offset: usize, len: usize) {
    #[cfg(target_os = "macos")]
    let _ = macos::advise(file, offset, len);

    #[cfg(target_os = "linux")]
    {
        use std::os::fd::AsRawFd;
        // SAFETY: posix_fadvise takes only integers and touches no memory.
        unsafe {
            libc::posix_fadvise(
                file.as_raw_fd(),
                offset as libc::off_t,
                len as libc::off_t,
                libc::POSIX_FADV_WILLNEED,
            )
        };
    }

    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    let _ = (file, offset, len);
}

/// The next range of a lane's last submit to hand over, and how many it has.
#[derive(Clone, Copy, Default)]
pub(crate) struct Cursor {
    next: usize,
    count: usize,
}

impl Cursor {
    pub(crate) fn start(&mut self, count: usize) {
        (self.next, self.count) = (0, count);
    }

    /// The next range to hand over, which then counts as handed over.
    pub(crate) fn take(&mut self) -> Option<usize> {
        if self.next == self.count {
            return None;
        }

        self.next += 1;
        Some(self.next - 1)
    }

    fn peek(&self) -> Option<usize> {
        (self.next < self.count).then_some(self.next)
    }
}

/// Hands over the ranges of each lane's submit in order, reading each one when asked for it.
#[derive(Default)]
pub(crate) struct InOrder {
    lanes: [Cursor; LANES],
}

impl InOrder {
    pub(crate) fn start(&mut self, lane: usize, count: usize) {
        self.lanes[lane].start(count);
    }

    /// Reads the lane's next range. If the read fails, the next call tries it again.
    pub(crate) fn read(
        &mut self,
        lane: usize,
        file: &File,
        ranges: &[Range],
        arena: &mut [u8],
    ) -> Result<Option<usize>, Error> {
        let Some(i) = self.lanes[lane].peek() else {
            return Ok(None);
        };
        let r = ranges[i];
        read_at(file, &mut arena[r.at..r.at + r.len], r.offset)?;
        Ok(self.lanes[lane].take())
    }
}

#[cfg(test)]
#[path = "../../tests/unit/io/mod.rs"]
mod tests;
