//! Memory maps of index files.

use std::fs::File;
use std::io::Error;

/// A whole index file, mapped read-only.
pub(crate) struct Map(memmap2::Mmap);

impl Map {
    pub(crate) fn new(file: &File) -> Result<Map, Error> {
        // SAFETY: an index file is never changed in place. A build writes a temporary file and renames it over the old
        // one, so a mapping keeps the bytes it was made from.
        let map = unsafe { memmap2::Mmap::map(file)? };
        Ok(Map(map))
    }

    pub(crate) fn bytes(&self) -> &[u8] {
        &self.0
    }

    /// Asks the operating system (OS) to start reading `len` bytes from `offset` in the background. Only Unix systems
    /// take the hint, and a refused hint changes nothing, so errors are ignored.
    pub(crate) fn will_need(&self, offset: usize, len: usize) {
        #[cfg(unix)]
        let _ = self.0.advise_range(memmap2::Advice::WillNeed, offset, len);
        #[cfg(not(unix))]
        let _ = (offset, len);
    }
}
