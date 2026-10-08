//! Unit tests for read planning and the read backends.

use std::collections::BTreeSet;
use std::io::ErrorKind;

use super::pool::PoolWave;
use super::portable::PortableWave;
use super::*;
use crate::quantization::SplitMix64;
use crate::quantization::testutil::{Shuffled, temp_file};

/// `n` random spans inside a file of `file_len` bytes, sorted by offset. Some are empty, a quarter start where another
/// ends, and many overlap.
fn random_spans(rng: &mut SplitMix64, file_len: usize, n: usize) -> Vec<Span> {
    let mut spans: Vec<Span> = Vec::with_capacity(n);

    for _ in 0..n {
        let offset = match spans.last() {
            Some(s) if rng.next_u64().is_multiple_of(4) => s.end(),
            _ => (rng.next_u64() % file_len as u64) as usize,
        };
        let len = match rng.next_u64() % 4 {
            0 => 0,
            1 => rng.next_u64() % 64,
            2 => rng.next_u64() % 20_000,
            _ => rng.next_u64() % 70_000,
        } as usize;
        spans.push(Span {
            offset,
            len: len.min(file_len - offset),
        });
    }

    spans.sort_by_key(|s| s.offset);
    spans
}

/// The pages that the spans cover once rounded out to `page` boundaries.
fn pages_of(spans: impl Iterator<Item = (usize, usize)>, page: usize) -> BTreeSet<usize> {
    spans
        .flat_map(|(start, end)| start / page..end.div_ceil(page))
        .collect()
}

#[test]
fn coalescing_covers_every_span_with_sorted_aligned_disjoint_ranges() {
    let mut rng = SplitMix64::new(1);
    let (mut ranges, mut at) = (Vec::new(), Vec::new());

    for round in 0..300 {
        let page = [4096, 8192, 16384][round % 3];
        let file_len = page * 40 + (rng.next_u64() % page as u64) as usize;
        let spans = random_spans(&mut rng, file_len, 1 + round % 50);
        coalesce(&spans, page, file_len, &mut ranges, &mut at);

        assert_eq!(at.len(), spans.len());
        assert_eq!(ranges[0].tag, 0);
        for (i, r) in ranges.iter().enumerate() {
            assert!(r.offset.is_multiple_of(page), "round {round}");
            assert!(r.end().is_multiple_of(page) || r.end() == file_len);
            assert!(r.end() <= file_len);
            let next = ranges.get(i + 1);
            // Ranges that touched would have been merged.
            assert!(next.is_none_or(|n| r.end() < n.offset && r.tag < n.tag));
            assert!(next.is_none_or(|n| n.at == r.at + r.len));
            let members = r.tag..next.map_or(spans.len(), |n| n.tag);
            for j in members {
                let s = spans[j];
                assert!(
                    r.offset <= s.offset && s.end() <= r.end(),
                    "span {j} is outside its range"
                );
                assert_eq!(at[j], r.at + s.offset - r.offset);
            }
        }

        // The ranges read exactly the pages that the spans touch, and their bytes fit the bound.
        let wanted = pages_of(spans.iter().map(|s| (s.offset, s.end())), page);
        assert_eq!(pages_of(ranges.iter().map(|r| (r.offset, r.end())), page), wanted);
        let used = ranges.last().map_or(0, |r| r.at + r.len);
        assert!(used <= arena_bound(spans.iter().map(|s| s.len), page));
    }
}

#[test]
fn coalescing_nothing_plans_nothing() {
    let (mut ranges, mut at) = (
        vec![Range {
            offset: 0,
            len: 1,
            tag: 0,
            at: 0,
        }],
        vec![3],
    );
    coalesce(&[], 4096, 1 << 20, &mut ranges, &mut at);

    assert!(ranges.is_empty() && at.is_empty());
}

/// A test file of `len` random bytes, with a handle for cold reads.
fn random_file(name: &str, len: usize) -> (Vec<u8>, Cold, crate::quantization::testutil::TempFile) {
    let mut rng = SplitMix64::new(2);
    let bytes: Vec<u8> = (0..len).map(|_| rng.next_u64() as u8).collect();
    let path = temp_file(name);
    std::fs::write(&path, &bytes).unwrap();
    let cold = open_cold(&path, &bytes[..len.min(16384)]).unwrap();
    (bytes, cold, path)
}

/// Every backend, the one that opening picked first.
fn backends(cold: &Cold) -> Vec<Box<dyn Wave>> {
    vec![
        (cold.wave)(),
        Box::<PortableWave>::default(),
        Box::<PoolWave>::default(),
        Box::<Shuffled>::default(),
        #[cfg(target_os = "macos")]
        Box::<macos::RdAdviseWave>::default(),
    ]
}

/// Waits for each range of the lane's last submit, checking that each arrives once.
fn drain_all(wave: &mut dyn Wave, lane: usize, file: &File, ranges: &[Range], arena: &mut [u8]) {
    let mut seen = vec![false; ranges.len()];

    while let Some(i) = wave.next(lane, file, ranges, arena).unwrap() {
        assert!(!seen[i], "range {i} arrived twice");
        seen[i] = true;
    }

    assert!(seen.iter().all(|&s| s), "a range never arrived");
}

/// Submits `ranges` on the lane and waits for each.
fn read_all(wave: &mut dyn Wave, lane: usize, file: &File, ranges: &[Range], arena: &mut [u8]) {
    wave.submit(lane, file, ranges, arena).unwrap();
    drain_all(wave, lane, file, ranges, arena);
}

#[test]
fn every_backend_reads_the_file_s_bytes() {
    let len = (1 << 20) + 12_345;
    let (bytes, cold, _path) = random_file("waves.bin", len);
    let mut rng = SplitMix64::new(3);
    let (mut ranges, mut at) = (Vec::new(), Vec::new());

    for mut wave in backends(&cold) {
        for round in 0..20 {
            let spans = random_spans(&mut rng, len, 1 + round * 7 % 60);
            coalesce(&spans, wave.page(), len, &mut ranges, &mut at);
            let mut arena = vec![0u8; arena_bound(spans.iter().map(|s| s.len), wave.page())];
            read_all(&mut *wave, round % LANES, &cold.file, &ranges, &mut arena);
            for r in &ranges {
                assert_eq!(&arena[r.at..r.at + r.len], &bytes[r.offset..r.end()]);
            }
            for (s, &a) in spans.iter().zip(&at) {
                assert_eq!(&arena[a..a + s.len], &bytes[s.range()]);
            }
        }
    }
}

#[test]
fn a_submit_drops_the_ranges_still_pending_on_its_lane() {
    let len = 1 << 18;
    let (bytes, cold, _path) = random_file("resubmit.bin", len);
    let span = |offset, len| Span { offset, len };
    let (first, second) = (
        [span(0, 100), span(50_000, 9000), span(200_000, 10)],
        [span(4096, 30_000), span(150_000, 1)],
    );
    let (mut ranges, mut at) = (Vec::new(), Vec::new());

    for mut wave in backends(&cold) {
        let mut arena = vec![0u8; 1 << 17];
        coalesce(&first, wave.page(), len, &mut ranges, &mut at);
        wave.submit(0, &cold.file, &ranges, &mut arena).unwrap();
        wave.next(0, &cold.file, &ranges, &mut arena).unwrap();
        coalesce(&second, wave.page(), len, &mut ranges, &mut at);
        read_all(&mut *wave, 0, &cold.file, &ranges, &mut arena);
        for (s, &a) in second.iter().zip(&at) {
            assert_eq!(&arena[a..a + s.len], &bytes[s.range()]);
        }
    }
}

#[test]
fn the_two_lanes_keep_their_waves_apart() {
    let len = 1 << 18;
    let (bytes, cold, _path) = random_file("lanes.bin", len);
    let span = |offset, len| Span { offset, len };
    let lanes = [
        [span(0, 100), span(50_000, 9000), span(200_000, 10)],
        [span(4096, 30_000), span(120_000, 40_000), span(250_000, 1)],
    ];

    for mut wave in backends(&cold) {
        let mut plans: Vec<(Vec<Range>, Vec<usize>, Vec<u8>)> = Vec::new();
        for (lane, spans) in lanes.iter().enumerate() {
            let (mut ranges, mut at) = (Vec::new(), Vec::new());
            coalesce(spans, wave.page(), len, &mut ranges, &mut at);
            let mut arena = vec![0u8; arena_bound(spans.iter().map(|s| s.len), wave.page())];
            wave.submit(lane, &cold.file, &ranges, &mut arena).unwrap();
            plans.push((ranges, at, arena));
        }
        // Lane 1 is drained before lane 0, the reverse of the submits.
        for lane in [1, 0] {
            let (ranges, at, arena) = &mut plans[lane];
            drain_all(&mut *wave, lane, &cold.file, ranges, arena);
            for (s, &a) in lanes[lane].iter().zip(at.iter()) {
                assert_eq!(&arena[a..a + s.len], &bytes[s.range()]);
            }
        }
    }
}

#[test]
fn an_empty_span_on_a_page_boundary_reads_nothing() {
    let len = 1 << 18;
    let (bytes, cold, _path) = random_file("empty.bin", len);
    let spans = [
        Span { offset: 10, len: 100 },
        Span { offset: 65_536, len: 0 },
        Span {
            offset: 200_000,
            len: 10,
        },
    ];
    let (mut ranges, mut at) = (Vec::new(), Vec::new());

    for mut wave in backends(&cold) {
        coalesce(&spans, wave.page(), len, &mut ranges, &mut at);
        assert_eq!(ranges.iter().filter(|r| r.len == 0).count(), 1);
        let mut arena = vec![0u8; arena_bound(spans.iter().map(|s| s.len), wave.page())];
        read_all(&mut *wave, 0, &cold.file, &ranges, &mut arena);
        assert_eq!(&arena[at[2]..at[2] + 10], &bytes[200_000..200_010]);
    }
}

#[test]
fn a_read_that_fails_fails_its_wave_on_every_backend() {
    // The second range runs past the end of the 10,000-byte file.
    let (_, cold, _path) = random_file("failing.bin", 10_000);
    let ranges = [
        Range {
            offset: 0,
            len: 4096,
            tag: 0,
            at: 0,
        },
        Range {
            offset: 8192,
            len: 8192,
            tag: 1,
            at: 4096,
        },
    ];

    for mut wave in backends(&cold) {
        let mut arena = vec![0u8; 12_288];
        let mut failed = wave.submit(0, &cold.file, &ranges, &mut arena).is_err();
        while !failed {
            match wave.next(0, &cold.file, &ranges, &mut arena) {
                Ok(Some(_)) => {}
                Ok(None) => break,
                Err(_) => failed = true,
            }
        }
        assert!(failed, "a backend with page {} read past the end", wave.page());
    }
}

#[test]
fn reads_past_the_end_fail_and_hints_never_do() {
    let (_, cold, _path) = random_file("short.bin", 10_000);
    let mut buf = [0u8; 100];

    assert_eq!(
        read_at(&cold.file, &mut buf, 9950).unwrap_err().kind(),
        ErrorKind::UnexpectedEof
    );
    will_need(&cold.file, 0, 1 << 30);
    will_need(&cold.file, 1 << 40, 4096);
}

#[test]
fn opening_the_cold_handle_checks_that_it_is_the_same_file() {
    let (bytes, _, path) = random_file("same.bin", 20_000);

    // A start that differs in one byte, as a rebuild renamed over the path would give, fails the check.
    let mut other = bytes[..16384].to_vec();
    other[16383] ^= 1;
    assert_eq!(open_cold(&path, &other).err().unwrap().kind(), ErrorKind::InvalidData);
    assert!(open_cold(&path, &bytes[..16384]).is_ok());
}

#[cfg(target_os = "macos")]
#[test]
fn macos_falls_back_only_when_the_file_system_refuses_the_hint() {
    for errno in [libc::ENOTTY, libc::ENOTSUP, libc::EINVAL] {
        assert!(macos::refused(&Error::from_raw_os_error(errno)));
    }
    for errno in [libc::EIO, libc::EBADF, libc::EFBIG] {
        assert!(!macos::refused(&Error::from_raw_os_error(errno)));
    }

    // The temporary folder is on the Apple File System (APFS), which takes the hint, even for a range past the end of
    // the file, so opening picks the F_RDADVISE backend there.
    let (_, cold, _path) = random_file("hint.bin", 100_000);
    assert!(macos::takes_hints(&cold.file).unwrap());
    macos::advise(&cold.file, 90_000, 1 << 20).unwrap();
    macos::advise(&cold.file, 1 << 30, 16384).unwrap();
    assert_eq!((cold.wave)().page(), 16384);
}

#[test]
fn a_warm_up_that_a_clear_interrupted_leaves_the_bit_clear() {
    let warmth = Warmth::default();
    let mark = warmth.mark();
    warmth.finish(mark);
    assert!(warmth.is_warm());

    // A clear between the start and the end of a warm-up wins.
    let mark = warmth.mark();
    warmth.set(false);
    warmth.finish(mark);
    assert!(!warmth.is_warm());
    warmth.set(true);
    assert!(warmth.is_warm());
}

#[test]
fn a_clear_racing_a_finish_always_wins() {
    use std::hint::spin_loop;
    use std::sync::Arc;

    let warmth = Arc::new(Warmth::default());
    // Rounds start and end on spins rather than locks, so the two calls of a round run close together.
    let (go, done) = (Arc::new(AtomicU64::new(0)), Arc::new(AtomicU64::new(0)));
    let rounds = 20_000;
    let (w, g, d) = (Arc::clone(&warmth), Arc::clone(&go), Arc::clone(&done));
    let clearer = std::thread::spawn(move || {
        for round in 1..=rounds {
            while g.load(Ordering::Acquire) != round {
                spin_loop();
            }
            (0..round % 8).for_each(|_| spin_loop());
            w.set(false);
            d.store(round, Ordering::Release);
        }
    });

    // Each clear comes after the warm-up's mark, so the bit must end clear whichever call runs first.
    for round in 1..=rounds {
        let mark = warmth.mark();
        go.store(round, Ordering::Release);
        (0..round / 8 % 8).for_each(|_| spin_loop());
        warmth.finish(mark);
        while done.load(Ordering::Acquire) != round {
            spin_loop();
        }
        assert!(!warmth.is_warm(), "round {round}");
    }

    clearer.join().unwrap();
}

#[cfg(target_os = "macos")]
#[test]
fn a_memory_pressure_warning_clears_the_warm_bit() {
    let warmth = std::sync::Arc::new(Warmth::default());
    warmth.set(true);
    let mut context = std::sync::Arc::clone(&warmth);
    // SAFETY: the handler reads its context as an `Arc<Warmth>`, which `context` is, and it outlives the call.
    unsafe { macos::on_pressure(&mut context as *mut std::sync::Arc<Warmth> as *mut std::ffi::c_void) };
    assert!(!warmth.is_warm());

    // A watch starts and stops cleanly, and frees its context.
    for _ in 0..10 {
        let pressure = Pressure::watch(std::sync::Arc::clone(&warmth));
        assert!(pressure._watch.is_some());
    }
    let start = std::time::Instant::now();
    while std::sync::Arc::strong_count(&warmth) > 2 {
        assert!(start.elapsed().as_secs() < 10, "a watch never freed its context");
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
}

#[test]
fn the_pool_starts_only_the_threads_a_wave_needs() {
    let (_, cold, _path) = random_file("threads.bin", 100_000);
    let mut wave = PoolWave::default();
    let mut arena = vec![0u8; 3 * 4096];

    wave.submit(0, &cold.file, &[], &mut arena).unwrap();
    assert_eq!(wave.started(), 0);
    let ranges: Vec<Range> = (0..3)
        .map(|i| Range {
            offset: i * 8192,
            len: 4096,
            tag: i,
            at: i * 4096,
        })
        .collect();
    wave.submit(0, &cold.file, &ranges, &mut arena).unwrap();
    assert_eq!(wave.started(), 3);
}
