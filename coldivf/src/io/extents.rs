//! The extent cache, which holds pages of one index file that a search context read past the operating system's
//! cache, so a later wave can take them from memory instead of the device.

/// A fixed number of whole pages of one file, each in the one slot that its page number hashes to. A page takes its
/// slot the second time it is read, so pages read once don't push out the ones that come back. A range comes from the
/// cache only when every one of its pages is there.
pub(crate) struct ExtentCache {
    page: usize,
    /// How many whole pages the file holds.
    pages: usize,
    /// The page number plus 1 that each slot holds, or 0 for an empty slot.
    keys: Vec<u64>,
    bytes: Vec<u8>,
    /// The page number plus 1 of the last page read but not kept, for each of twice as many entries as slots.
    seen: Vec<u64>,
}

impl ExtentCache {
    /// A cache of up to `slots` pages of `page` bytes, for a file of `file_len` bytes. It rounds `slots` down to a
    /// power of 2 that the file's whole pages can fill. With no slots it keeps nothing.
    pub(crate) fn new(slots: usize, page: usize, file_len: usize) -> ExtentCache {
        let pages = file_len / page;
        let slots = match slots.min(pages) {
            0 => 0,
            n => 1 << n.ilog2(),
        };
        ExtentCache {
            page,
            pages,
            keys: vec![0; slots],
            bytes: vec![0; slots * page],
            seen: vec![0; 2 * slots],
        }
    }

    /// Copies the bytes from `offset` into `out` if every page they cover is here, and says whether it did. Both must
    /// fall on page boundaries, so a range that doesn't, eg. the end of the file, never comes from here.
    pub(crate) fn get(&self, offset: usize, out: &mut [u8]) -> bool {
        if self.keys.is_empty() || out.is_empty() {
            return false;
        }

        if !offset.is_multiple_of(self.page) || !out.len().is_multiple_of(self.page) {
            return false;
        }

        let first = offset / self.page;
        let pages = first..first + out.len() / self.page;

        if !pages.clone().all(|p| self.keys[self.slot(p)] == key(p)) {
            return false;
        }

        for (p, chunk) in pages.zip(out.chunks_exact_mut(self.page)) {
            let at = self.slot(p) * self.page;
            chunk.copy_from_slice(&self.bytes[at..at + self.page]);
        }

        true
    }

    /// Offers the bytes just read from `offset`. Each whole page of them inside the file is kept the second time it is
    /// offered, in place of whatever its slot held. The first time it is only noted.
    pub(crate) fn offer(&mut self, offset: usize, bytes: &[u8]) {
        if self.keys.is_empty() {
            return;
        }

        let first = offset.div_ceil(self.page);
        let Some(whole) = bytes.get(first * self.page - offset..) else {
            return;
        };

        for (p, chunk) in (first..self.pages).zip(whole.chunks_exact(self.page)) {
            let slot = self.slot(p);

            if self.keys[slot] == key(p) {
                continue;
            }

            let seen = &mut self.seen[mix(p, 1) as usize & (2 * self.keys.len() - 1)];

            if *seen == key(p) {
                self.keys[slot] = key(p);
                self.bytes[slot * self.page..(slot + 1) * self.page].copy_from_slice(chunk);
            } else {
                *seen = key(p);
            }
        }
    }

    /// The slot that page `p` goes in.
    fn slot(&self, p: usize) -> usize {
        mix(p, 0) as usize & (self.keys.len() - 1)
    }
}

/// What a slot or a noted entry holds for page `p`, which is never 0.
fn key(p: usize) -> u64 {
    p as u64 + 1
}

/// A hash of page number `p`, by SplitMix64's finalizer, where `salt` picks one of several.
fn mix(p: usize, salt: u64) -> u64 {
    let mut z = (p as u64).wrapping_add(salt.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[cfg(test)]
#[path = "../../tests/unit/io/extents.rs"]
mod tests;
