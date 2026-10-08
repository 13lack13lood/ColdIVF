//! The macOS backend. The index's cold handle has `F_NOCACHE`, so its reads skip the operating system's cache. A wave
//! first tells the kernel about every range with `F_RDADVISE`, which starts them all at once, then reads each range in
//! order.

use std::fs::File;
use std::io::Error;
use std::os::fd::AsRawFd;

use super::{InOrder, Range, Wave};

/// The page size of Apple silicon, which the reads round out to.
const PAGE: usize = 16384;

/// Turns off caching for `file`, the index's cold handle.
pub(crate) fn uncache(file: &File) -> Result<(), Error> {
    // SAFETY: F_NOCACHE takes an integer and touches no memory.
    if unsafe { libc::fcntl(file.as_raw_fd(), libc::F_NOCACHE, 1) } == -1 {
        return Err(Error::last_os_error());
    }

    Ok(())
}

/// Whether the file system takes `F_RDADVISE` hints, tried once on the first page of `file`. FAT32 (File Allocation
/// Table) and some network drives don't, so there the caller falls back to the thread pool.
pub(crate) fn takes_hints(file: &File) -> Result<bool, Error> {
    match advise(file, 0, PAGE) {
        Ok(()) => Ok(true),
        Err(e) if refused(&e) => Ok(false),
        Err(e) => Err(e),
    }
}

/// Whether `F_RDADVISE` failed because the file system doesn't support it.
pub(crate) fn refused(e: &Error) -> bool {
    matches!(e.raw_os_error(), Some(libc::ENOTTY | libc::ENOTSUP | libc::EINVAL))
}

#[derive(Default)]
pub(crate) struct RdAdviseWave(InOrder);

impl Wave for RdAdviseWave {
    fn page(&self) -> usize {
        PAGE
    }

    fn submit(&mut self, lane: usize, file: &File, ranges: &[Range], _arena: &mut [u8]) -> Result<(), Error> {
        self.0.start(lane, 0);
        ranges
            .iter()
            .filter(|r| r.len > 0)
            .try_for_each(|r| advise(file, r.offset, r.len))?;
        self.0.start(lane, ranges.len());
        Ok(())
    }

    fn next(&mut self, lane: usize, file: &File, ranges: &[Range], arena: &mut [u8]) -> Result<Option<usize>, Error> {
        self.0.read(lane, file, ranges, arena)
    }
}

/// Asks the kernel to start reading `len` bytes of `file` from `offset`. A range that runs past the end of the file
/// gets `EFBIG`, which is harmless.
pub(crate) fn advise(file: &File, offset: usize, len: usize) -> Result<(), Error> {
    let ra = libc::radvisory {
        ra_offset: offset as libc::off_t,
        ra_count: len.min(i32::MAX as usize) as libc::c_int,
    };
    // SAFETY: `ra` outlives the call, and F_RDADVISE only reads it.
    let failed = unsafe { libc::fcntl(file.as_raw_fd(), libc::F_RDADVISE, &ra as *const libc::radvisory) } == -1;

    if failed {
        let e = Error::last_os_error();
        if e.raw_os_error() != Some(libc::EFBIG) {
            return Err(e);
        }
    }

    Ok(())
}
