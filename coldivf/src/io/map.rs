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
}
