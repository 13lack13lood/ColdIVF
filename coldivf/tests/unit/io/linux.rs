//! Unit tests for the Linux backend. Their files sit in the build's target folder, since the system's temporary folder
//! may be a tmpfs, which takes no direct reads before Linux 6.6. With COLDIVF_EXPECT_URING=1 every backend must get its
//! ring, eg. when the tests run one at a time, so a backend that quietly fell back to `posix_fadvise` waves fails them.

use std::os::fd::AsRawFd;

use super::*;
use crate::io::map::Map;
use crate::io::portable::PortableWave;
use crate::io::tests::random_spans;
use crate::io::{Arena, Cold, LANES, Memory, coalesce, open_cold};
use crate::quantization::SplitMix64;
use crate::quantization::testutil::{TempFile, temp_file};

/// A test file of `len` random bytes, with a cold handle on it.
fn random_file(name: &str, len: usize) -> (Vec<u8>, Cold, TempFile) {
    let mut rng = SplitMix64::new(2);
    let bytes: Vec<u8> = (0..len).map(|_| rng.next_u64() as u8).collect();
    let path = temp_file(name);
    std::fs::write(&path, &bytes).unwrap();
    let cold = open_cold(&path).unwrap();
    (bytes, cold, path)
}

/// Whether this system gives io_uring rings at all.
fn rings_work() -> bool {
    IoUring::new(8).is_ok()
}

/// Whether every backend gets its ring, so a test can insist on io_uring. Before Linux 5.12 rings count against
/// RLIMIT_MEMLOCK, so under a low limit some of the backends that tests make at once fall back to `posix_fadvise`.
fn rings_always() -> bool {
    IoUring::new(8).is_ok_and(|r| r.params().is_feature_native_workers() || memlock_limit() == usize::MAX)
}

/// Whether the tests must find every backend on io_uring, as COLDIVF_EXPECT_URING=1 says, or since every backend can
/// get a ring.
fn rings_expected() -> bool {
    std::env::var("COLDIVF_EXPECT_URING").is_ok_and(|v| v == "1") || rings_always()
}

/// Whether the kernel registers files and arenas without stopping the ring, as from Linux 5.13.
fn registers_freely() -> bool {
    IoUring::new(8).is_ok_and(|r| r.params().is_feature_resource_tagging())
}

/// Checks that `wave` read through io_uring, with direct reads still on when `direct` says, or that it fell back to
/// `posix_fadvise` waves where the system may not give it a ring.
fn check_state(wave: &UringWave, direct: bool, what: &str) {
    match wave.state() {
        State::Ring { direct: on, reads, .. } => assert!(on == direct && reads > 0, "{what}: {:?}", wave.state()),
        State::Fadvise => assert!(!rings_expected(), "{what} fell back to fadvise waves"),
        State::Unset => panic!("{what} never set itself up"),
    }
}

/// Checks what `wave`'s queue handed the kernel: a read of the direct handle exactly when `direct` says some go out,
/// every read by its registered slot when the files are registered, and reads into registered arenas only once some
/// are registered.
fn check_sent(wave: &mut UringWave, direct: bool, what: &str) {
    let path = wave.read_path();
    let sent = wave.sent();

    if let State::Ring { registered, reads, .. } = wave.state() {
        assert_eq!(path.backend, "io_uring", "{what}");
        assert_eq!(sent.direct + sent.cold, reads, "{what}: {sent:?}");
        assert_eq!(sent.direct > 0, direct, "{what}: {sent:?}");
        assert_eq!(
            sent.fixed_files,
            if path.fixed_files { reads } else { 0 },
            "{what}: {sent:?}"
        );
        assert!(registered > 0 || sent.fixed_buffers == 0, "{what}: {sent:?}");
    }
}

/// Whether the file system of `file` takes direct reads.
fn direct_works(file: &File) -> bool {
    open_direct(file, &Setup::default()).is_ok_and(|direct| dio_align(&direct).is_some())
}

/// Drops the file of `cold` from the cache, as far as the kernel lets it go.
fn evict(cold: &Cold) {
    cold.file.sync_all().unwrap();
    // SAFETY: posix_fadvise takes only integers and touches no memory.
    unsafe { libc::posix_fadvise(cold.file.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED) };
}

/// How many bytes of the pages of the mapped file `map` are in the cache.
fn resident(map: &Map) -> usize {
    let (bytes, page) = (map.bytes(), page_size());
    let mut vec = vec![0u8; bytes.len().div_ceil(page)];
    // SAFETY: the range is the whole live mapping, and `vec` has one byte per page of it, as mincore writes.
    let rc = unsafe { libc::mincore(bytes.as_ptr() as *mut libc::c_void, bytes.len(), vec.as_mut_ptr()) };
    assert_eq!(rc, 0);
    vec.iter().filter(|&&v| v & 1 == 1).count() * page
}

/// Submits `ranges` on the lane and waits for each, checking that each arrives once.
fn read_all(wave: &mut dyn Wave, lane: usize, file: &File, ranges: &[Range], arena: &mut [u8]) {
    let arena = Memory::of(arena);
    wave.submit(lane, file, ranges, arena).unwrap();
    drain_all(wave, lane, file, ranges, arena);
}

/// Waits for each range of the lane's last submit, checking that each arrives once.
fn drain_all(wave: &mut dyn Wave, lane: usize, file: &File, ranges: &[Range], arena: Memory) {
    let mut seen = vec![false; ranges.len()];

    while let Some(i) = wave.next(lane, file, ranges, arena).unwrap() {
        assert!(!seen[i], "range {i} arrived twice");
        seen[i] = true;
    }

    assert!(seen.iter().all(|&s| s), "a range never arrived");
}

/// The bytes of an arena that holds any wave of `random_spans`.
const SIZE: usize = 60 * (70_000 + 2 * PAGE);

/// Two arenas for `compare`, a byte longer than it needs. A test makes them before its backend, so they outlast it.
fn arenas() -> [Arena; 2] {
    [Arena::aligned(SIZE + 1, PAGE), Arena::aligned(SIZE + 1, PAGE)]
}

/// Reads 20 waves of random ranges of the file in `cold` through `wave` and through the portable backend, into
/// `arenas` from `skew` bytes past their start, and checks that both give the file's bytes. With `ready` it first
/// readies `wave` with those arenas.
fn compare(
    wave: &mut dyn Wave,
    ready: bool,
    bytes: &[u8],
    cold: &Cold,
    arenas: &mut [Arena; 2],
    skew: usize,
    what: &str,
) {
    let len = bytes.len();
    let mut rng = SplitMix64::new(3);
    let (mut ranges, mut at) = (Vec::new(), Vec::new());
    let mut portable = PortableWave::default();
    let mut plain = vec![0u8; SIZE];
    let [a, b] = arenas;
    let arenas = [&mut a[skew..skew + SIZE], &mut b[skew..skew + SIZE]];

    if ready {
        wave.ready(&cold.file, &[Memory::of(arenas[0]), Memory::of(arenas[1])], 64);
    }

    for round in 0..20 {
        let spans = random_spans(&mut rng, len, 1 + round * 7 % 60);
        coalesce(&spans, PAGE, len, &mut ranges, &mut at);
        let lane = round % LANES;
        let arena = &mut *arenas[lane];
        arena.fill(0);
        read_all(wave, lane, &cold.file, &ranges, arena);
        read_all(&mut portable, lane, &cold.file, &ranges, &mut plain);
        for r in &ranges {
            assert_eq!(
                &arena[r.at..r.at + r.len],
                &plain[r.at..r.at + r.len],
                "{what}, round {round}"
            );
        }
        for (s, &a) in spans.iter().zip(&at) {
            assert_eq!(&arena[a..a + s.len], &bytes[s.range()], "{what}, round {round}");
        }
    }
}

#[test]
fn the_ring_reads_what_the_portable_backend_reads() {
    let (bytes, cold, _path) = random_file("uring.bin", (1 << 20) + 12_345);
    let setups = [
        ("default", Setup::default()),
        (
            "buffered",
            Setup {
                direct: false,
                ..Setup::default()
            },
        ),
        (
            "unregistered",
            Setup {
                budget: Some(0),
                ..Setup::default()
            },
        ),
        (
            "registered files",
            Setup {
                register: Some(true),
                ..Setup::default()
            },
        ),
        (
            "a ring of 4",
            Setup {
                entries: Some(&[4]),
                ..Setup::default()
            },
        ),
    ];
    let works = direct_works(&cold.file);

    for (name, setup) in setups {
        for (skew, ready) in [(0, true), (0, false), (1, true)] {
            let what = format!("{name}, skew {skew}");
            let mut arenas = arenas();
            let mut wave = UringWave::with(setup);
            compare(&mut wave, ready, &bytes, &cold, &mut arenas, skew, &what);
            check_state(&wave, setup.direct && works, &what);
            // An arena a byte past a page takes no direct reads, so they all go through the cache.
            check_sent(&mut wave, setup.direct && works && skew == 0, &what);

            if setup.register == Some(true) {
                assert!(wave.read_path().fixed_files || wave.state() == State::Fadvise, "{what}");
            }
        }
    }

    eprintln!(
        "linux: rings {}, direct reads {works}, alignment {:?}, RLIMIT_MEMLOCK {} bytes, rings of {:?}, registering {}",
        rings_work(),
        dio_align(&cold.file),
        memlock_limit(),
        ring_sizes(),
        if registers_freely() {
            "freely"
        } else {
            "only when a test asks"
        }
    );
}

#[test]
fn cold_waves_skip_the_page_cache_and_cached_waves_fill_it() {
    let (bytes, cold, _path) = random_file("cache.bin", 1 << 20);
    let map = Map::new(&cold.file).unwrap();
    let ranges: Vec<Range> = (0..32)
        .map(|i| Range {
            offset: i * 4 * PAGE,
            len: 2 * PAGE,
            tag: i,
            at: i * 2 * PAGE,
        })
        .collect();
    let mut arena = Arena::aligned(64 * PAGE, PAGE);
    let mut wave = UringWave::default();
    let check = |arena: &[u8], what: &str| {
        for r in &ranges {
            assert_eq!(&arena[r.at..r.at + r.len], &bytes[r.offset..r.end()], "{what}");
        }
    };

    // A cold wave's direct reads leave the file's pages out of the cache.
    evict(&cold);
    let before = resident(&map);
    read_all(&mut wave, 0, &cold.file, &ranges, &mut arena);
    check(&arena, "direct");
    let direct = matches!(wave.state(), State::Ring { direct: true, .. });

    if direct {
        assert!(
            resident(&map) <= before,
            "{} bytes came into the cache",
            resident(&map) - before
        );
        assert_eq!(wave.sent().direct, ranges.len());
    } else {
        check_state(&wave, false, "cold wave");
        eprintln!("linux: no direct reads here, so the cold wave read through the cache");
    }

    // A cached wave of the same ranges reads them through the cache, which then holds them.
    evict(&cold);
    arena.fill(0);
    let memory = arena.memory();
    wave.submit_cached(0, &cold.file, &ranges, memory).unwrap();
    drain_all(&mut wave, 0, &cold.file, &ranges, memory);
    check(&arena, "cached");

    if direct {
        assert!(
            resident(&map) >= ranges.len() * 2 * PAGE,
            "{} bytes in the cache",
            resident(&map)
        );
        assert_eq!(wave.sent().cold, ranges.len());
    }
}

#[test]
fn a_file_system_without_direct_reads_gets_buffered_reads() {
    let (bytes, cold, _path) = random_file("buffered.bin", 300_000);
    // Opening the direct handle fails with EINVAL, as on a tmpfs before Linux 6.6.
    let setup = Setup {
        direct_error: Some(libc::EINVAL),
        ..Setup::default()
    };
    let mut arenas = arenas();
    let mut wave = UringWave::with(setup);
    compare(&mut wave, true, &bytes, &cold, &mut arenas, 0, "without direct reads");
    check_state(&wave, false, "without direct reads");
    check_sent(&mut wave, false, "without direct reads");
}

#[test]
fn a_direct_read_the_kernel_turns_down_falls_back_to_a_buffered_one() {
    let (bytes, cold, _path) = random_file("refused.bin", 300_000);
    // The arena starts a byte past a page and the check is off, so the kernel turns the direct reads down with EINVAL.
    // A file system that reads them through its cache instead leaves direct reads on.
    let setup = Setup {
        unchecked: true,
        ..Setup::default()
    };
    let mut arenas = arenas();
    let mut wave = UringWave::with(setup);
    compare(
        &mut wave,
        true,
        &bytes,
        &cold,
        &mut arenas,
        1,
        "after the kernel's EINVAL",
    );

    if let State::Ring { direct: true, .. } = wave.state() {
        eprintln!("linux: the file system took direct reads off their boundaries");
    } else {
        check_state(&wave, false, "after the kernel's EINVAL");
    }
}

#[test]
fn prepare_registers_the_arenas_where_the_kernel_and_the_budget_let_it() {
    let (bytes, cold, _path) = random_file("pinned.bin", 300_000);
    let ranges: Vec<Range> = (0..8)
        .map(|i| Range {
            offset: i * 3 * PAGE,
            len: 2 * PAGE,
            tag: i,
            at: i * 2 * PAGE,
        })
        .collect();

    // Whether the backend registers, and the budget. Waves never register, and `prepare` registers both arenas
    // where the backend registers and they fit the budget, unless the kernel turns that down with ENOMEM, eg. under a
    // low RLIMIT_MEMLOCK.
    for (register, budget) in [
        (Some(true), usize::MAX),
        (Some(true), 0),
        (Some(false), usize::MAX),
        (None, usize::MAX),
    ] {
        let what = format!("register {register:?}, budget {budget}");
        let mut arenas = [Arena::aligned(16 * PAGE, PAGE), Arena::aligned(16 * PAGE, PAGE)];
        let setup = Setup {
            register,
            budget: Some(budget),
            ..Setup::default()
        };
        let mut wave = UringWave::with(setup);
        let [a, b] = &mut arenas;
        wave.ready(&cold.file, &[a.memory(), b.memory()], 8);

        for round in 0..4 {
            if round == 2 {
                wave.prepare();
            }

            let arena = &mut arenas[round % 2];
            arena.fill(0);
            read_all(&mut wave, round % 2, &cold.file, &ranges, arena);
            for r in &ranges {
                assert_eq!(
                    &arena[r.at..r.at + r.len],
                    &bytes[r.offset..r.end()],
                    "{what}, round {round}"
                );
            }
            if round == 1 {
                assert_eq!(wave.sent().fixed_buffers, 0, "{what}");
            }
        }

        let State::Ring { registered, .. } = wave.state() else {
            check_state(&wave, true, &what);
            continue;
        };
        let wanted = register.unwrap_or_else(registers_freely) && budget > 0;

        match (wanted, memlock_limit() == usize::MAX) {
            (true, true) => {
                assert_eq!(registered, 2, "{what}");
                assert_eq!(wave.sent().fixed_buffers, 2 * ranges.len(), "{what}");
            }
            (true, false) => eprintln!(
                "linux: {registered} arenas registered under {} bytes of memlock",
                memlock_limit()
            ),
            (false, _) => assert_eq!((registered, wave.sent().fixed_buffers), (0, 0), "{what}"),
        }

        check_sent(&mut wave, direct_works(&cold.file), &what);
    }
}

#[test]
fn a_kernel_that_blocks_io_uring_gets_fadvise_waves() {
    for errno in [libc::EPERM, libc::ENOSYS, libc::EACCES] {
        assert!(blocked(&Error::from_raw_os_error(errno)), "{errno}");
    }
    for errno in [libc::ENOMEM, libc::EINVAL, libc::EMFILE] {
        assert!(!blocked(&Error::from_raw_os_error(errno)), "{errno}");
    }

    static MEMO: AtomicBool = AtomicBool::new(false);
    static NOT_BLOCKED: AtomicBool = AtomicBool::new(false);
    let (bytes, cold, _path) = random_file("fadvise.bin", 300_000);
    let blocked_before = BLOCKED.load(Ordering::Relaxed);

    // EPERM says that the system blocks io_uring, so the backend reads with fadvise waves and the memo keeps it.
    let setup = Setup {
        ring_error: Some(libc::EPERM),
        blocked: &MEMO,
        ..Setup::default()
    };
    let mut arenas = arenas();
    let mut wave = UringWave::with(setup);
    compare(&mut wave, true, &bytes, &cold, &mut arenas, 0, "blocked");
    assert_eq!(wave.state(), State::Fadvise);
    assert_eq!(wave.read_path().backend, "fadvise");
    assert!(MEMO.load(Ordering::Relaxed));

    // A later backend with the same memo doesn't try io_uring, even where the system gives rings.
    let setup = Setup {
        blocked: &MEMO,
        ..Setup::default()
    };
    let mut wave = UringWave::with(setup);
    compare(&mut wave, true, &bytes, &cold, &mut arenas, 0, "remembered");
    assert_eq!(wave.state(), State::Fadvise);

    // ENOMEM at every size only says that no ring fits now, so nothing is remembered.
    let setup = Setup {
        ring_error: Some(libc::ENOMEM),
        blocked: &NOT_BLOCKED,
        ..Setup::default()
    };
    let mut wave = UringWave::with(setup);
    compare(&mut wave, true, &bytes, &cold, &mut arenas, 0, "short of memory");
    assert_eq!(wave.state(), State::Fadvise);
    assert!(!NOT_BLOCKED.load(Ordering::Relaxed));
    // A test's errors don't block io_uring for the rest of the process.
    assert_eq!(BLOCKED.load(Ordering::Relaxed), blocked_before);

    let mut fadvise = FadviseWave::default();
    compare(&mut fadvise, false, &bytes, &cold, &mut arenas, 1, "fadvise");
}

#[test]
fn a_backend_sets_itself_up_again_for_another_handle() {
    let (bytes, cold, path) = random_file("handles.bin", 300_000);
    let other = File::open(&path).unwrap();
    let mut arena = Arena::aligned(1 << 16, PAGE);
    let mut wave = UringWave::default();
    let ranges = [Range {
        offset: 8192,
        len: 3 * PAGE,
        tag: 0,
        at: 0,
    }];

    wave.ready(&cold.file, &[arena.memory()], 4);
    assert_ne!(wave.state(), State::Unset);
    read_all(&mut wave, 0, &other, &ranges, &mut arena);
    assert_eq!(&arena[..3 * PAGE], &bytes[8192..8192 + 3 * PAGE]);

    if let Mode::Ring(ring) = &wave.mode {
        assert_eq!(ring.fd, other.as_raw_fd());
    }

    // Readying it again for the same handle, eg. with more arenas, keeps its ring.
    let reads = wave.state();
    wave.ready(&other, &[arena.memory()], 4);
    assert_eq!(wave.state(), reads);
}

#[test]
fn a_ring_another_process_set_up_is_left_alone() {
    let (bytes, cold, _path) = random_file("forked.bin", 300_000);
    let mut arena = Arena::aligned(1 << 16, PAGE);
    let mut wave = UringWave::default();
    let ranges = [Range {
        offset: 8192,
        len: 3 * PAGE,
        tag: 0,
        at: 0,
    }];
    read_all(&mut wave, 0, &cold.file, &ranges, &mut arena);

    // As in the child of a fork, the ring is another process's. The backend drops it without waiting on it, which in
    // this test leaks it, and sets up a ring of its own for the next wave.
    let Mode::Ring(ring) = &mut wave.mode else {
        check_state(&wave, true, "before the fork");
        return;
    };
    ring.pid = ring.pid.wrapping_add(1);
    arena.fill(0);
    read_all(&mut wave, 0, &cold.file, &ranges, &mut arena);
    assert_eq!(&arena[..3 * PAGE], &bytes[8192..8192 + 3 * PAGE]);
    let Mode::Ring(ring) = &wave.mode else {
        panic!("no ring after the fork: {:?}", wave.state());
    };
    assert_eq!(ring.pid, std::process::id());
    assert_eq!(ring.engine.counts().0, 1);
}

#[test]
fn warming_brings_the_mapped_pages_into_memory() {
    let (_, cold, path) = random_file("warm.bin", 1 << 20);
    let map = Map::new(&cold.file).unwrap();

    evict(&cold);
    let mut buf = vec![0u8; 1 << 18];
    crate::io::warm(&cold.file, map.bytes(), &mut buf, 1 << 18).unwrap();
    assert!(resident(&map) >= 1 << 18, "{} bytes resident", resident(&map));

    // From Linux 5.14 the kernel faults the pages in itself, and before it says so.
    match populate(&map.bytes()[1..]) {
        Ok(true) => assert_eq!(resident(&map), 1 << 20),
        Ok(false) => assert!(NO_POPULATE.load(Ordering::Relaxed)),
        Err(e) => panic!("{e}"),
    }

    drop(path);
}

#[test]
fn a_kernel_that_gives_no_direct_read_alignment_reads_whole_pages() {
    // statx fails before Linux 4.11 and answers without STATX_DIOALIGN before Linux 6.1, eg. on Linux 5.10. Both give
    // whole pages.
    let page = Some(Align {
        offset: PAGE,
        memory: PAGE,
    });
    // SAFETY: statx is plain old data, for which all zeros is a valid value.
    let mut stx: libc::statx = unsafe { std::mem::zeroed() };
    stx.stx_mask = libc::STATX_BASIC_STATS;
    assert_eq!(align_from(-1, &stx), page);
    assert_eq!(align_from(0, &stx), page);

    // From Linux 6.1 the answer holds the alignment. One of 0, past a page or not a power of two takes no direct reads.
    stx.stx_mask |= libc::STATX_DIOALIGN;
    for (offset, memory, want) in [
        (512, 512, Some((512, 512))),
        (4096, 512, Some((4096, 512))),
        (0, 0, None),
        (8192, 512, None),
        (512, 8192, None),
        (768, 512, None),
    ] {
        (stx.stx_dio_offset_align, stx.stx_dio_mem_align) = (offset, memory);
        let want = want.map(|(offset, memory)| Align { offset, memory });
        assert_eq!(align_from(0, &stx), want, "{offset} and {memory}");
    }

    // Asked without the question, this kernel answers as one before Linux 6.1 does.
    let (bytes, cold, _path) = random_file("pages.bin", 300_000);

    if let Ok(direct) = open_direct(&cold.file, &Setup::default()) {
        assert_eq!(dio_align_asking(&direct, false), page);
        eprintln!(
            "linux: direct reads here align to {:?}, and to whole pages without the question",
            dio_align(&direct)
        );
    }

    // So a backend set up that way reads whole pages directly, and arenas off a page through the cache.
    let setup = Setup {
        ask_alignment: false,
        ..Setup::default()
    };
    let works = direct_works(&cold.file);

    for skew in [0, 1] {
        let what = format!("whole pages, skew {skew}");
        let mut arenas = arenas();
        let mut wave = UringWave::with(setup);
        compare(&mut wave, true, &bytes, &cold, &mut arenas, skew, &what);
        check_state(&wave, works, &what);
        check_sent(&mut wave, works && skew == 0, &what);
    }
}

#[test]
fn a_kernel_without_populate_read_warms_through_read_hints() {
    // Linux before 5.14, eg. Linux 5.10, turns MADV_POPULATE_READ down with EINVAL, as every kernel does an advice it
    // doesn't know. So an advice that no kernel knows stands in for it, with a memo of this test's own.
    const UNKNOWN: libc::c_int = libc::c_int::MAX;
    static REFUSED: AtomicBool = AtomicBool::new(false);
    let (bytes, cold, _path) = random_file("hinted.bin", 1 << 20);
    let map = Map::new(&cold.file).unwrap();
    let range = (1 << 18)..(1 << 19);

    evict(&cold);
    let before = resident(&map);
    assert!(!populate_with(&map.bytes()[range.clone()], UNKNOWN, &REFUSED).unwrap());
    assert!(REFUSED.load(Ordering::Relaxed));
    assert!(resident(&map) <= before, "{} bytes came in", resident(&map) - before);

    // Once turned down, it doesn't ask again, even with an advice that the kernel takes.
    assert!(!populate_with(&map.bytes()[range.clone()], libc::MADV_WILLNEED, &REFUSED).unwrap());

    // So the warm-up reads the bytes through the cache after a WILLNEED hint, as `io::warm` does next, and they stay in
    // the cache.
    let mut buf = vec![0u8; range.len()];
    crate::io::read_cached(&cold.file, &mut buf, range.start).unwrap();
    assert_eq!(buf, &bytes[range.clone()]);
    assert!(resident(&map) >= range.len(), "{} bytes resident", resident(&map));

    // Whether this kernel takes MADV_POPULATE_READ, asked with a memo of its own, which says the path warm-ups take.
    static ASKED: AtomicBool = AtomicBool::new(false);
    let takes = populate_with(&map.bytes()[..PAGE], libc::MADV_POPULATE_READ, &ASKED).unwrap();
    assert_eq!(takes, !ASKED.load(Ordering::Relaxed));
    eprintln!(
        "linux: warm-ups here {}",
        if takes {
            "fault the pages in with MADV_POPULATE_READ"
        } else {
            "read through the cache after a WILLNEED hint, since the kernel lacks MADV_POPULATE_READ"
        }
    );
}
