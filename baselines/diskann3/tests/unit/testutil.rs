//! Helpers the unit tests share.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use diskann_quantization::alloc::{AlignedAllocator, Poly};
use diskann_quantization::num::PowerOfTwo;

/// A folder of its own under the system's temporary folder, removed when dropped.
pub(crate) struct Scratch(PathBuf);

impl Scratch {
    pub(crate) fn new(label: &str) -> Scratch {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("diskann3-{label}-{}-{n}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        Scratch(dir)
    }

    pub(crate) fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// The byte at `offset` of a file from `sectors_file`, which differs for nearby offsets and for nearby sectors.
pub(crate) fn byte_at(offset: u64) -> u8 {
    (offset.wrapping_mul(31) ^ (offset >> 12).wrapping_mul(7)) as u8
}

/// Writes a file of `sectors` 4 KiB sectors whose bytes follow `byte_at`.
pub(crate) fn sectors_file(path: &Path, sectors: usize) {
    let bytes: Vec<u8> = (0..sectors as u64 * 4096).map(byte_at).collect();
    std::fs::write(path, bytes).unwrap();
}

/// A zeroed buffer of `len` bytes aligned to 4 KiB, as DiskANN allocates its sector buffers.
pub(crate) fn aligned(len: usize) -> Poly<[u8], AlignedAllocator> {
    Poly::broadcast(0u8, len, AlignedAllocator::new(PowerOfTwo::V4096)).unwrap()
}

/// Writes a base file of `rows` rows of width `width`, whose value j of row i is i·width + j.
pub(crate) fn base_file(path: &Path, rows: usize, width: usize) {
    let mut bytes = Vec::with_capacity(8 + rows * width * 4);
    bytes.extend_from_slice(&(rows as u32).to_le_bytes());
    bytes.extend_from_slice(&(width as u32).to_le_bytes());

    for v in 0..rows * width {
        bytes.extend_from_slice(&(v as f32).to_le_bytes());
    }

    std::fs::write(path, bytes).unwrap();
}
