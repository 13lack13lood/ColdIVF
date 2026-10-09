//! Unit tests for the cold-cache tools, which are written for macOS.

use std::io::Write;
use std::path::PathBuf;

use super::*;
use crate::testutil::counters;

/// A file of `bytes` bytes of 7s, in a folder of the system temporary folder, which tests may write to. It is written a
/// MiB at a time, since the allocator may keep a large freed buffer, and give it back during a later footprint check.
fn scratch(name: &str, bytes: usize) -> PathBuf {
    let dir = std::env::temp_dir().join("coldivf-bench");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(format!("{}-{name}", std::process::id()));
    let (mut file, chunk) = (std::fs::File::create(&path).unwrap(), vec![7u8; 1 << 20]);

    for at in (0..bytes).step_by(chunk.len()) {
        file.write_all(&chunk[..chunk.len().min(bytes - at)]).unwrap();
    }

    path
}

/// Spins for `time` and returns what the process used meanwhile.
fn spin(time: Duration) -> Delta {
    let (before, start) = (Usage::now(), Instant::now());
    let mut x = [0u64; 8];

    while start.elapsed() < time {
        x.iter_mut().for_each(|v| *v = std::hint::black_box(v.wrapping_add(1)));
    }

    Usage::now().since(&before)
}

/// Asks the kernel which pages of `file` are cached for `time`, which is nearly all system time.
fn ask_the_kernel(file: &Mapped, time: Duration) -> Delta {
    let (before, start) = (Usage::now(), Instant::now());

    while start.elapsed() < time {
        file.resident(0..file.len()).unwrap();
    }

    Usage::now().since(&before)
}

#[test]
fn making_a_file_cold_empties_its_cached_pages() {
    let _lock = counters();
    let path = scratch("evict.bin", 4 << 20);
    let _ = std::fs::read(&path).unwrap();
    let file = Mapped::open(&path).unwrap();

    assert!(file.residency().unwrap() > 0.9);
    // A new mapping can make another process read the file's first pages back, up to ~300 ms after the first
    // eviction. So `make_cold` must drop them again once they come back, and the file must still be cold well after it
    // returns. It waits a second for them here, twice as long as the harnesses do, so a read-back slowed by load still
    // lands before it returns.
    make_cold(&[&file], Duration::from_secs(1)).unwrap();
    std::thread::sleep(Duration::from_millis(400));
    assert_eq!(file.residency().unwrap(), 0.0);

    // A range counts the pages it touches, rounded out to whole pages.
    let (page, map) = (page_size(), file.map.as_ref().unwrap());
    std::hint::black_box(map[5 * page + 3]);
    assert_eq!(file.resident(0..file.len()).unwrap(), 1);
    assert_eq!(file.resident(5 * page + 1..5 * page + 2).unwrap(), 1);
    assert_eq!(file.resident(4 * page + 1..5 * page + 1).unwrap(), 1);
    assert_eq!(file.resident(0..5 * page).unwrap(), 0);
    assert_eq!(file.resident(6 * page..usize::MAX).unwrap(), 0);
    std::fs::remove_file(&path).unwrap();
}

#[test]
fn dropping_a_range_of_a_file_leaves_the_rest_cached() {
    let path = scratch("range.bin", 4 << 20);
    let _ = std::fs::read(&path).unwrap();
    let (file, page) = (Mapped::open(&path).unwrap(), page_size());
    assert!(file.residency().unwrap() > 0.9);

    // The range is past the first MiB, which another process may read back after a new mapping. It rounds out to the
    // pages from 2 MiB to a page past 3 MiB.
    let (start, end) = (2 << 20, 3 << 20);
    file.evict_range(start + 1..end + 5).unwrap();
    assert_eq!(file.resident(start..end + page).unwrap(), 0);
    assert_eq!(file.resident(start - page..start).unwrap(), 1);
    assert_eq!(file.resident(end + page..end + 2 * page).unwrap(), 1);
    // A range past the end drops nothing.
    file.evict_range(4 << 20..usize::MAX).unwrap();
    assert_eq!(file.resident(end + page..end + 2 * page).unwrap(), 1);
    std::fs::remove_file(&path).unwrap();
}

#[test]
fn cpu_time_comes_in_nanoseconds() {
    let _lock = counters();
    let path = scratch("kernel.bin", 16 << 20);
    let file = Mapped::open(&path).unwrap();

    // 100 ms of spinning, or of system calls, must read as about 100 ms, not 100 ms in clock ticks. The bound above is
    // loose, since every thread of the process adds to the count.
    for (what, used) in [
        ("user", spin(Duration::from_millis(100))),
        ("system", ask_the_kernel(&file, Duration::from_millis(100))),
    ] {
        let ms = used.cpu_ns as f64 / 1e6;
        assert!((80.0..1000.0).contains(&ms), "{what}: {ms} ms");
    }

    std::fs::remove_file(&path).unwrap();
}

#[cfg(target_arch = "aarch64")]
#[test]
fn cycles_and_performance_core_time_add_up() {
    let _lock = counters();
    let path = scratch("p-cores.bin", 16 << 20);
    let file = Mapped::open(&path).unwrap();
    crate::metrics::interactive();

    // A user-interactive thread runs on the performance cores whenever it can, so one of a few tries should be mostly
    // P-core time, both in user code and in the kernel. P-core time is never more than the whole.
    for what in ["user", "system"] {
        let mut tries: Vec<Delta> = Vec::new();

        while tries.len() < 5 && !tries.iter().any(|d| d.p_share() > 0.5) {
            tries.push(match what {
                "user" => spin(Duration::from_millis(100)),
                _ => ask_the_kernel(&file, Duration::from_millis(100)),
            });
        }

        let shares: Vec<f64> = tries.iter().map(Delta::p_share).collect();
        assert!(
            shares.iter().any(|&s| s > 0.5) && shares.iter().all(|&s| s <= 1.01),
            "{what}: {shares:?}"
        );
        // The cycles over the CPU time are the clock the work ran at.
        assert!(
            tries.iter().all(|d| (0.5..6.0).contains(&d.ghz())),
            "{what}: {:?} GHz",
            tries.iter().map(Delta::ghz).collect::<Vec<_>>()
        );
    }

    std::fs::remove_file(&path).unwrap();
}

#[test]
fn the_footprint_counts_memory_written_but_not_a_cached_file() {
    let _lock = counters();
    let path = scratch("footprint.bin", 32 << 20);
    // Memory of its own map, which goes back to the system when it's dropped, unlike a large heap block that the
    // allocator may keep.
    let before = Usage::now();
    let mut memory = memmap2::MmapMut::map_anon(48 << 20).unwrap();
    memory.iter_mut().step_by(4096).for_each(|b| *b = 1);
    let grown = Usage::now().since(&before).footprint;
    drop(memory);
    let freed = Usage::now().since(&before).footprint;

    // Reading every page of a mapped file caches it, but its pages are the file's, not the process's.
    let file = Mapped::open(&path).unwrap();
    let map = file.map.as_ref().unwrap();
    let sum: u64 = map.iter().step_by(page_size()).map(|&b| u64::from(b)).sum();
    assert_eq!(sum, 7 * (32 << 20) as u64 / page_size() as u64);
    let mapped = Usage::now().since(&before).footprint;

    let mib = |bytes: i64| bytes as f64 / (1 << 20) as f64;
    assert!((40.0..56.0).contains(&mib(grown)), "{} MiB", mib(grown));
    assert!(mib(freed).abs() < 8.0, "{} MiB", mib(freed));
    assert!(mib(mapped).abs() < 8.0, "{} MiB", mib(mapped));
    std::fs::remove_file(&path).unwrap();
}

#[test]
fn reading_a_cold_file_counts_disk_reads_and_page_ins() {
    let _lock = counters();
    let path = scratch("disk.bin", 4 << 20);
    let file = Mapped::open(&path).unwrap();
    make_cold(&[&file], Duration::from_millis(500)).unwrap();

    // Every page of the file now comes from the disk, as a page fault through the mapping.
    let before = Usage::now();
    let map = file.map.as_ref().unwrap();
    let sum: u64 = map.iter().step_by(page_size()).map(|&b| u64::from(b)).sum();
    assert_eq!(sum, 7 * (4 << 20) as u64 / page_size() as u64);
    let used = Usage::now().since(&before);

    assert!(used.pageins > 0, "{used:?}");
    assert!(used.bytes_read >= 2 << 20, "{used:?}");
    std::fs::remove_file(&path).unwrap();
}

#[test]
fn two_windows_add_up_field_by_field() {
    let (a, b) = (
        Delta {
            cpu_ns: 1,
            p_cpu_ns: 2,
            cycles: 3,
            instructions: 4,
            energy_nj: 5,
            pageins: 6,
            bytes_read: 7,
            footprint: -8,
        },
        Delta {
            cpu_ns: 10,
            p_cpu_ns: 20,
            cycles: 30,
            instructions: 40,
            energy_nj: 50,
            pageins: 60,
            bytes_read: 70,
            footprint: 80,
        },
    );
    let sum = a + b;
    assert_eq!(
        (sum.cpu_ns, sum.p_cpu_ns, sum.cycles, sum.instructions),
        (11, 22, 33, 44)
    );
    assert_eq!(
        (sum.energy_nj, sum.pageins, sum.bytes_read, sum.footprint),
        (55, 66, 77, 72)
    );
}
