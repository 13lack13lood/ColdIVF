//! Checks on Linux that dropping a file from the cache works, through `posix_fadvise(DONTNEED)`, and that `mincore`
//! sees it. The file sits in the build's target folder, since the system's temporary folder may be a tmpfs, whose pages
//! never leave memory.
#![cfg(target_os = "linux")]

use std::io::Write;
use std::path::PathBuf;
use std::time::Duration;

use coldivf_bench::coldcache::{Mapped, make_cold, page_size};

/// The largest block of pages that the kernel caches a file in, a folio of 2 MiB on x86-64 and on Arm with 4 KiB
/// pages. A drop of part of a file keeps the folios that reach past the part, so the test drops whole ones.
const FOLIO: usize = 2 << 20;

/// A file of `bytes` bytes of 7s, written a MiB at a time and not yet written out to the disk.
fn scratch(name: &str, bytes: usize) -> PathBuf {
    let path = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("coldivf-bench-{}-{name}", std::process::id()));
    let (mut file, chunk) = (std::fs::File::create(&path).unwrap(), vec![7u8; 1 << 20]);

    for at in (0..bytes).step_by(chunk.len()) {
        file.write_all(&chunk[..chunk.len().min(bytes - at)]).unwrap();
    }

    path
}

#[test]
fn a_file_drops_out_of_the_cache_a_range_at_a_time_or_whole() {
    let path = scratch("evict.bin", 4 * FOLIO + 100);
    let _ = std::fs::read(&path).unwrap();
    let (file, page) = (Mapped::open(&path).unwrap(), page_size());
    assert!(file.residency().unwrap() > 0.9);

    // The pages were just written, so dropping them writes them out first. The range rounds out to whole pages, which
    // here are the folio from 2 MiB to 4 MiB, and the pages on either side stay.
    let (start, end) = (FOLIO, 2 * FOLIO);
    file.evict_range(start + 1..end - 5).unwrap();
    assert_eq!(file.resident(start..end).unwrap(), 0);
    assert_eq!(file.resident(start - page..start).unwrap(), 1);
    assert_eq!(file.resident(end..end + page).unwrap(), 1);

    // The whole file goes, its last part page too, and a read brings it back.
    make_cold(&[&file], Duration::from_millis(100)).unwrap();
    assert_eq!(file.residency().unwrap(), 0.0);
    let _ = std::fs::read(&path).unwrap();
    assert!(file.residency().unwrap() > 0.9);
    std::fs::remove_file(&path).unwrap();
}
