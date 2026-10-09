//! Unit tests for read waves over a kernel queue, through a queue that reads from memory.

use std::io::ErrorKind;

use super::*;
use crate::io::tests::random_spans;
use crate::io::{Arena, Memory, Span, arena_bound, coalesce};
use crate::quantization::SplitMix64;
use crate::quantization::testutil::shuffle;

/// What `Fake` writes past the end of the file for a direct read that runs past it.
const JUNK: u8 = 0x5A;

/// The order in which a fake queue's reads complete.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Order {
    /// Each entry completes a random share of the reads it holds, at least as many as it waits for, in a random order.
    Random,
    /// Each entry completes only as many as it waits for, the oldest first.
    Oldest,
    /// Each entry completes only as many as it waits for, the newest first.
    Newest,
}

/// A queue for tests that reads from a copy of the file in memory. It turns down a direct read off `align` with EINVAL,
/// and a read that its registered buffer doesn't hold with EFAULT, as the kernel does. A direct read that runs past the
/// end of the file writes junk on to its full length, as a device writes whole blocks.
struct Fake {
    file: Vec<u8>,
    rng: SplitMix64,
    order: Order,
    /// How many reads it queues before the kernel must take them, and how many completions it has room for.
    slots: usize,
    room: usize,
    /// The most bytes that one read brings, if any.
    short: Option<usize>,
    align: Align,
    /// The memory that reads may land in, each an address and a length.
    memory: Vec<(usize, usize)>,
    registered: Vec<(usize, usize)>,
    /// The error number that registering fails with, if any.
    refuse: Option<i32>,
    /// Whether every entry fails, so the kernel takes nothing.
    broken: bool,
    queued: Vec<Read>,
    held: Vec<Read>,
    done: Vec<(u64, i32)>,
    /// Every read that ran, in order.
    ran: Vec<Read>,
}

impl Fake {
    fn new(file: &[u8], order: Order, slots: usize, room: usize) -> Fake {
        Fake {
            file: file.to_vec(),
            rng: SplitMix64::new(5),
            order,
            slots,
            room,
            short: None,
            align: Align {
                offset: 4096,
                memory: 4096,
            },
            memory: Vec::new(),
            registered: Vec::new(),
            refuse: None,
            broken: false,
            queued: Vec::new(),
            held: Vec::new(),
            done: Vec::new(),
            ran: Vec::new(),
        }
    }

    /// Runs `read` and returns its result.
    fn run(&mut self, read: Read) -> i32 {
        let (len, offset) = (read.len as usize, read.offset as usize);
        let lands = |&(base, size): &(usize, usize)| base <= read.addr && read.addr + len <= base + size;
        assert!(
            self.memory.iter().any(lands),
            "a read lands outside every arena: {read:?}"
        );
        self.ran.push(read);
        let a = self.align;
        let aligned = offset.is_multiple_of(a.offset) && len.is_multiple_of(a.offset);

        if read.direct && !(aligned && read.addr.is_multiple_of(a.memory)) {
            return -22;
        }

        if let Some(b) = read.buffer
            && !self.registered.get(b as usize).is_some_and(lands)
        {
            return -14;
        }

        let start = offset.min(self.file.len());
        let n = len.min(self.short.unwrap_or(usize::MAX)).min(self.file.len() - start);
        let to = std::ptr::with_exposed_provenance_mut::<u8>(read.addr);
        // SAFETY: the read lies in memory that the test keeps until the read completes, as just checked. It writes
        // through the read's address, as the kernel does, and makes no slice of the memory, so it can't clash with
        // one that the engine holds.
        unsafe { std::ptr::copy_nonoverlapping(self.file[start..].as_ptr(), to, n) };

        if read.direct && start + n == self.file.len() {
            // SAFETY: the same memory, past what the read brought.
            unsafe { std::ptr::write_bytes(to.add(n), JUNK, len - n) };
        }

        n as i32
    }
}

impl Queue for Fake {
    fn push(&mut self, read: &Read) -> bool {
        if self.queued.len() == self.slots {
            return false;
        }

        self.queued.push(*read);
        true
    }

    fn enter(&mut self, want: usize) -> Result<(), Error> {
        if self.broken {
            return Err(Error::from_raw_os_error(9));
        }

        self.held.append(&mut self.queued);
        let need = want.saturating_sub(self.done.len());
        assert!(
            need <= self.held.len(),
            "the kernel would wait for completions that never come"
        );
        let finished: Vec<Read> = match self.order {
            Order::Random => {
                shuffle(&mut self.held, &mut self.rng);
                let some = (self.rng.next_u64() % (self.held.len() as u64 + 1)) as usize;
                self.held.drain(..some.max(need)).collect()
            }
            Order::Oldest => self.held.drain(..need).collect(),
            Order::Newest => {
                let at = self.held.len() - need;
                self.held.drain(at..).rev().collect()
            }
        };

        for read in finished {
            let res = self.run(read);
            self.done.push((read.tag, res));
        }

        Ok(())
    }

    fn reap(&mut self, out: &mut [(u64, i32)]) -> usize {
        let n = out.len().min(self.done.len());
        out[..n].copy_from_slice(&self.done[..n]);
        self.done.drain(..n);
        n
    }

    fn room(&self) -> usize {
        self.room
    }

    fn unsent(&mut self) -> usize {
        self.queued.len()
    }

    fn register(&mut self, arenas: &[(usize, usize)]) -> Result<(), Error> {
        if let Some(errno) = self.refuse {
            return Err(Error::from_raw_os_error(errno));
        }

        self.registered = arenas.to_vec();
        Ok(())
    }
}

impl Drop for Fake {
    fn drop(&mut self) {
        // The engine must wait for every read the kernel took before the queue goes away, as each may still land.
        if !std::thread::panicking() {
            assert!(
                self.held.is_empty(),
                "the queue went away with reads in the kernel's hands"
            );
        }
    }
}

/// `len` random bytes.
fn random_bytes(len: usize, seed: u64) -> Vec<u8> {
    let mut rng = SplitMix64::new(seed);
    (0..len).map(|_| rng.next_u64() as u8).collect()
}

/// Settings for an engine over a file of `file_len` bytes, with direct reads and an extent cache when `direct` says.
fn settings(direct: bool, file_len: usize) -> Settings {
    Settings {
        direct,
        align: Align {
            offset: 4096,
            memory: 4096,
        },
        file_len,
        read_page: 4096,
        cache_pages: if direct { 256 } else { 0 },
        budget: usize::MAX,
        registers: true,
        page: 4096,
        pinned: Box::leak(Box::new(Pinned::new())),
        unchecked: false,
    }
}

/// Where `arena` is, as an address and a length.
fn place(arena: &[u8]) -> (usize, usize) {
    (arena.as_ptr().expose_provenance(), arena.len())
}

/// An engine over `fake`, whose reads may land in `memory`, each an address and a length.
fn engine(mut fake: Fake, s: Settings, memory: &[(usize, usize)]) -> Engine<Fake> {
    fake.memory = memory.to_vec();
    Engine::new(fake, s)
}

/// Takes every range of the lane's wave, checking that each arrives once.
fn take_all(e: &mut Engine<Fake>, lane: usize, count: usize, arena: &mut [u8]) {
    let (mut seen, arena) = (vec![false; count], Memory::of(arena));

    while let Some(i) = e.next(lane, arena).unwrap() {
        assert!(!seen[i], "range {i} arrived twice");
        seen[i] = true;
    }

    assert!(seen.iter().all(|&s| s), "a range never arrived");
}

/// `n` ranges of `len` bytes, `stride` bytes apart in the file from `first`, packed in the arena one after another.
fn packed(n: usize, first: usize, stride: usize, len: usize) -> Vec<Range> {
    (0..n)
        .map(|i| Range {
            offset: first + i * stride,
            len,
            tag: i,
            at: i * len,
        })
        .collect()
}

/// Checks that each of `ranges` holds the file's bytes in `arena`.
fn check(file: &[u8], ranges: &[Range], arena: &[u8], what: &str) {
    for (i, r) in ranges.iter().enumerate() {
        assert_eq!(
            &arena[r.at..r.at + r.len],
            &file[r.offset..r.end()],
            "{what}, range {i}"
        );
    }
}

#[test]
fn every_range_arrives_once_with_the_file_s_bytes() {
    let len = (1 << 20) + 12_345;
    let file = random_bytes(len, 1);
    let mut rng = SplitMix64::new(3);
    let (mut ranges, mut at) = (Vec::new(), Vec::new());
    let size = arena_bound((0..60).map(|_| 70_000), 4096);
    // Direct reads or not, an arena on a page boundary or a byte past one, the completion order, the queue's slots and
    // room, and how many bytes a read brings at most.
    let cases = [
        (true, 0, Order::Random, 256, 512, None),
        (true, 0, Order::Random, 3, 5, None),
        (true, 0, Order::Oldest, 4, 4, Some(5000)),
        (true, 1, Order::Random, 16, 32, None),
        (true, 0, Order::Newest, 7, 9, Some(4096)),
        (false, 0, Order::Random, 8, 16, Some(777)),
        (false, 1, Order::Newest, 2, 3, None),
    ];

    for (case, (direct, skew, order, slots, room, short)) in cases.into_iter().enumerate() {
        let mut big = Arena::aligned(size + skew, 4096);
        let arena = &mut big[skew..];
        let mut fake = Fake::new(&file, order, slots, room);
        fake.short = short;
        let mut e = engine(fake, settings(direct, len), &[place(arena)]);
        for round in 0..20 {
            let spans = random_spans(&mut rng, len, 1 + round * 7 % 60);
            coalesce(&spans, 4096, len, &mut ranges, &mut at);
            let lane = round % LANES;
            arena.fill(0);
            e.submit(lane, &ranges, Memory::of(arena)).unwrap();
            take_all(&mut e, lane, ranges.len(), arena);
            for (s, &a) in spans.iter().zip(&at) {
                assert_eq!(&arena[a..a + s.len], &file[s.range()], "case {case}, round {round}");
            }
        }
        // Reads on a page boundary went direct, and the others through the cache.
        let ran = &e.queue().ran;
        assert_eq!(ran.iter().any(|r| r.direct), direct && skew == 0, "case {case}");
    }
}

#[test]
fn the_two_lanes_keep_their_waves_apart() {
    let len = 1 << 18;
    let file = random_bytes(len, 2);
    let span = |offset, len| Span { offset, len };
    let lanes = [
        [span(0, 100), span(50_000, 9000), span(200_000, 10)],
        [span(4096, 30_000), span(120_000, 40_000), span(250_000, 1)],
    ];

    for order in [Order::Random, Order::Oldest, Order::Newest] {
        let mut arenas = [Arena::aligned(1 << 17, 4096), Arena::aligned(1 << 17, 4096)];
        // Room for 3 completions, so lane 1's reads wait for lane 0's to make room.
        let fake = Fake::new(&file, order, 2, 3);
        let mut e = engine(fake, settings(true, len), &[place(&arenas[0]), place(&arenas[1])]);
        let mut plans = Vec::new();
        for (lane, spans) in lanes.iter().enumerate() {
            let (mut ranges, mut at) = (Vec::new(), Vec::new());
            coalesce(spans, 4096, len, &mut ranges, &mut at);
            e.submit(lane, &ranges, Memory::of(&mut arenas[lane])).unwrap();
            plans.push((ranges, at));
        }
        // Lane 1 is taken before lane 0, the reverse of the submits.
        for lane in [1, 0] {
            let (ranges, at) = &plans[lane];
            take_all(&mut e, lane, ranges.len(), &mut arenas[lane]);
            for (s, &a) in lanes[lane].iter().zip(at) {
                assert_eq!(&arenas[lane][a..a + s.len], &file[s.range()], "{order:?}, lane {lane}");
            }
        }
    }
}

#[test]
fn a_lane_s_new_wave_waits_for_the_reads_of_the_wave_it_drops() {
    let len = 1 << 18;
    let file = random_bytes(len, 3);
    // Both waves land in the same places of the arena, from different parts of the file.
    let (first, second) = (packed(8, 0, 8192, 4096), packed(8, 131_072, 4096, 4096));

    for order in [Order::Oldest, Order::Newest, Order::Random] {
        let mut arena = Arena::aligned(8 * 4096, 4096);
        let mut e = engine(Fake::new(&file, order, 64, 64), settings(true, len), &[place(&arena)]);
        e.submit(0, &first, Memory::of(&mut arena)).unwrap();
        assert!(e.next(0, Memory::of(&mut arena)).unwrap().is_some());
        e.submit(0, &second, Memory::of(&mut arena)).unwrap();
        take_all(&mut e, 0, second.len(), &mut arena);
        check(&file, &second, &arena, &format!("{order:?}"));
    }
}

#[test]
fn a_short_read_goes_on_from_where_it_stopped() {
    let len = 1 << 18;
    let file = random_bytes(len, 4);
    let ranges = packed(6, 4096, 3 * 4096, 2 * 4096);

    // A cut that keeps direct reads on block boundaries, and one that sends the rest through the cache.
    for (short, direct) in [(4096, true), (1000, false)] {
        let mut arena = Arena::aligned(6 * 2 * 4096, 4096);
        let mut fake = Fake::new(&file, Order::Random, 8, 16);
        fake.short = Some(short);
        let mut e = engine(fake, settings(true, len), &[place(&arena)]);
        e.submit(1, &ranges, Memory::of(&mut arena)).unwrap();
        take_all(&mut e, 1, ranges.len(), &mut arena);
        check(&file, &ranges, &arena, &format!("cut to {short}"));
        let ran = &e.queue().ran;
        assert!(ran.len() >= 2 * ranges.len(), "{short}");
        assert_eq!(ran.iter().all(|r| r.direct), direct, "{short}");
    }
}

#[test]
fn a_read_past_the_end_of_the_file_fails_its_wave() {
    let file = random_bytes(10_000, 5);
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

    for direct in [false, true] {
        let mut arena = Arena::aligned(12_288, 4096);
        let mut e = engine(
            Fake::new(&file, Order::Random, 8, 8),
            settings(direct, 10_000),
            &[place(&arena)],
        );
        e.submit(0, &ranges, Memory::of(&mut arena)).unwrap();
        let mut failed = None;
        while failed.is_none() {
            match e.next(0, Memory::of(&mut arena)) {
                Ok(Some(_)) => {}
                Ok(None) => break,
                Err(err) => failed = Some(err.kind()),
            }
        }
        assert_eq!(failed, Some(ErrorKind::UnexpectedEof), "direct {direct}");
    }
}

#[test]
fn a_direct_read_the_kernel_turns_down_goes_through_the_cache_instead() {
    let len = 300_000;
    let file = random_bytes(len, 6);
    let ranges = packed(10, 0, 16_384, 8192);
    let mut big = Arena::aligned(10 * 8192 + 1, 4096);
    // Off the page boundary, and with the check skipped, so the kernel turns the direct reads down.
    let arena = &mut big[1..];
    let s = Settings {
        unchecked: true,
        ..settings(true, len)
    };
    let mut e = engine(Fake::new(&file, Order::Random, 4, 8), s, &[place(arena)]);
    assert!(e.direct());

    e.submit(0, &ranges, Memory::of(arena)).unwrap();
    take_all(&mut e, 0, ranges.len(), arena);
    check(&file, &ranges, arena, "after the fallback");
    assert!(!e.direct());
    let ran = &e.queue().ran;
    assert!(ran.iter().any(|r| r.direct) && ran.last().is_some_and(|r| !r.direct));
}

#[test]
fn only_the_range_that_ends_the_file_and_the_arena_rounds_up() {
    let len = 10_000;
    let file = random_bytes(len, 7);
    let range = |offset, len, at| Range {
        offset,
        len,
        tag: 0,
        at,
    };

    // The file's last range is the arena's last too, so it reads whole blocks, and the bytes past the file land in the
    // arena's spare room.
    let mut arena = Arena::aligned(8192, 4096);
    let ranges = [range(0, 4096, 0), range(8192, len - 8192, 4096)];
    let mut e = engine(
        Fake::new(&file, Order::Random, 8, 8),
        settings(true, len),
        &[place(&arena)],
    );
    e.submit(0, &ranges, Memory::of(&mut arena)).unwrap();
    take_all(&mut e, 0, 2, &mut arena);
    check(&file, &ranges, &arena, "rounded");
    let last = e.queue().ran.iter().find(|r| r.offset == 8192).copied().unwrap();
    assert!(last.direct && last.len == 4096);
    assert!(arena[4096 + len - 8192..].iter().all(|&b| b == JUNK));

    // Here a later range sits where the extra bytes would land, so the last range of the file goes through the cache.
    // Reads complete newest first, so a rounded read would land on that range after it arrived.
    let mut arena = Arena::aligned(8192, 4096);
    let ranges = [range(8192, len - 8192, 0), range(0, 4096, 2048)];
    let mut e = engine(
        Fake::new(&file, Order::Newest, 8, 8),
        settings(true, len),
        &[place(&arena)],
    );
    e.submit(0, &ranges, Memory::of(&mut arena)).unwrap();
    take_all(&mut e, 0, 2, &mut arena);
    check(&file, &ranges, &arena, "not last");
    assert!(e.queue().ran.iter().all(|r| !r.direct));

    // Without spare room past it, the last range goes through the cache too.
    let mut arena = Arena::aligned(4096 + len - 8192, 4096);
    let ranges = [range(0, 4096, 0), range(8192, len - 8192, 4096)];
    let mut e = engine(
        Fake::new(&file, Order::Random, 8, 8),
        settings(true, len),
        &[place(&arena)],
    );
    e.submit(0, &ranges, Memory::of(&mut arena)).unwrap();
    take_all(&mut e, 0, 2, &mut arena);
    check(&file, &ranges, &arena, "no room");
    assert!(e.queue().ran.iter().any(|r| r.offset == 8192 && !r.direct));
}

#[test]
fn arenas_register_only_when_asked_and_only_within_the_budget() {
    let len = 1 << 18;
    let file = random_bytes(len, 8);
    let ranges = packed(4, 0, 16_384, 8192);

    // Whether the engine registers at all, the budget, the error that registering fails with, and whether the arena
    // is registered once asked to.
    for (allowed, budget, refuse, registers) in [
        (true, usize::MAX, None, true),
        (true, 40_000, None, false),
        (true, usize::MAX, Some(12), false),
        (false, usize::MAX, None, false),
    ] {
        let what = format!("{allowed} {budget} {refuse:?}");
        let mut arena = Arena::aligned(65_536, 4096);
        let s = settings(true, len);
        let pinned = s.pinned;
        let mut fake = Fake::new(&file, Order::Random, 8, 16);
        fake.refuse = refuse;
        // Without the extent cache, so every wave reads all its ranges.
        let s = Settings {
            budget,
            cache_pages: 0,
            registers: allowed,
            ..s
        };
        let mut e = engine(fake, s, &[place(&arena)]);
        e.keep(&[place(&arena)], 16);
        // Waves never register, however many there are, so no search pays for it.
        for _ in 0..2 {
            arena.fill(0);
            e.submit(0, &ranges, Memory::of(&mut arena)).unwrap();
            take_all(&mut e, 0, ranges.len(), &mut arena);
            check(&file, &ranges, &arena, &what);
        }
        assert_eq!((e.registered(), pinned.held()), (0, 0), "{what}");
        // Asked between waves, it registers once, and the next wave reads into the registered arena.
        e.register();
        e.register();
        assert_eq!(e.registered(), usize::from(registers), "{what}");
        arena.fill(0);
        e.submit(0, &ranges, Memory::of(&mut arena)).unwrap();
        take_all(&mut e, 0, ranges.len(), &mut arena);
        check(&file, &ranges, &arena, &what);
        let fixed = e.queue().ran.iter().filter(|r| r.buffer == Some(0)).count();
        assert_eq!(fixed, if registers { ranges.len() } else { 0 }, "{what}");
        assert_eq!(pinned.held(), if registers { 65_536 } else { 0 }, "{what}");
        drop(e);
        assert_eq!(pinned.held(), 0, "{what}");
    }

    // An arena past 64 MiB stays plain memory, since registering would fault in and pin all of it.
    let big = (64 << 20) + 4096;
    let s = settings(true, len);
    let pinned = s.pinned;
    let mut e = engine(Fake::new(&file, Order::Random, 8, 16), s, &[]);
    e.keep(&[(1 << 32, big), (2 << 32, 8192)], 16);
    e.register();
    assert_eq!((e.registered(), pinned.held()), (1, 8192));
}

#[test]
fn the_budget_counts_every_engine_and_waits_for_reads_to_finish() {
    let len = 1 << 18;
    let file = random_bytes(len, 9);
    let ranges = packed(4, 0, 16_384, 8192);
    let mut arenas = [Arena::aligned(65_536, 4096), Arena::aligned(65_536, 4096)];
    let s = Settings {
        budget: 100_000,
        ..settings(true, len)
    };
    let pinned = s.pinned;
    // Two engines share one count, so only the first arena fits.
    let [a, b] = &mut arenas;
    let mut first = engine(Fake::new(&file, Order::Random, 8, 16), Settings { ..s }, &[place(a)]);
    let mut second = engine(Fake::new(&file, Order::Newest, 8, 16), Settings { ..s }, &[place(b)]);
    first.keep(&[place(a)], 16);
    second.keep(&[place(b)], 16);
    first.submit(0, &ranges, Memory::of(a)).unwrap();
    take_all(&mut first, 0, ranges.len(), a);
    first.register();
    assert_eq!((first.registered(), pinned.held()), (1, 65_536));
    second.submit(0, &ranges, Memory::of(b)).unwrap();
    take_all(&mut second, 0, ranges.len(), b);
    second.register();
    assert_eq!(second.registered(), 0);
    drop(first);
    assert_eq!(pinned.held(), 0);

    // Registering waits for a moment with no reads out, and a wave left on the other lane holds it off.
    let mut c = Arena::aligned(65_536, 4096);
    let mut third = engine(
        Fake::new(&file, Order::Newest, 8, 16),
        Settings { ..s },
        &[place(&c), place(b)],
    );
    third.keep(&[place(&c)], 16);
    third.submit(1, &ranges, Memory::of(b)).unwrap();
    third.register();
    assert_eq!(third.registered(), 0);
    take_all(&mut third, 1, ranges.len(), b);
    third.register();
    assert_eq!(third.registered(), 1);
    third.submit(0, &ranges, Memory::of(&mut c)).unwrap();
    take_all(&mut third, 0, ranges.len(), &mut c);
    check(&file, &ranges, &c, "third");
    assert!(
        third
            .queue()
            .ran
            .iter()
            .rev()
            .take(ranges.len())
            .all(|r| r.buffer == Some(0))
    );
}

#[test]
fn pages_read_twice_come_from_the_extent_cache_after() {
    let len = (1 << 20) + 1696;
    let file = random_bytes(len, 10);
    // Four ranges of whole pages, and the file's last range, whose part page is never cached.
    let mut ranges = packed(4, 0, 16_384, 8192);
    ranges.push(Range {
        offset: 1 << 20,
        len: 1696,
        tag: 4,
        at: 4 * 8192,
    });
    let mut arena = Arena::aligned(6 * 8192, 4096);
    let mut e = engine(
        Fake::new(&file, Order::Random, 8, 16),
        settings(true, len),
        &[place(&arena)],
    );

    for wave in 0..3 {
        let (reads, hits) = e.counts();
        arena.fill(0);
        e.submit(0, &ranges, Memory::of(&mut arena)).unwrap();
        take_all(&mut e, 0, ranges.len(), &mut arena);
        check(&file, &ranges, &arena, &format!("wave {wave}"));
        let (reads, hits) = (e.counts().0 - reads, e.counts().1 - hits);
        assert_eq!((reads, hits), if wave < 2 { (5, 0) } else { (1, 4) }, "wave {wave}");
    }

    // Without direct reads, the operating system's cache holds the pages, so the engine keeps none.
    let mut e = engine(
        Fake::new(&file, Order::Random, 8, 16),
        settings(false, len),
        &[place(&arena)],
    );

    for _ in 0..3 {
        e.submit(0, &ranges, Memory::of(&mut arena)).unwrap();
        take_all(&mut e, 0, ranges.len(), &mut arena);
    }

    assert_eq!(e.counts(), (15, 0));
}

#[test]
fn a_cached_wave_reads_through_the_operating_system_s_cache_and_keeps_nothing() {
    let len = 1 << 20;
    let file = random_bytes(len, 14);
    let ranges = packed(4, 0, 16_384, 8192);
    let mut arena = Arena::aligned(4 * 8192, 4096);
    let mut e = engine(
        Fake::new(&file, Order::Random, 8, 16),
        settings(true, len),
        &[place(&arena)],
    );

    // Each cached wave sends every read through the cache, even with direct reads on, and none of its pages enters
    // the extent cache, so a direct wave after two of them still reads every range.
    for wave in 0..3 {
        let ran = e.queue().ran.len();
        arena.fill(0);
        match wave {
            2 => e.submit(0, &ranges, Memory::of(&mut arena)).unwrap(),
            _ => e.submit_cached(0, &ranges, Memory::of(&mut arena)).unwrap(),
        }
        take_all(&mut e, 0, ranges.len(), &mut arena);
        check(&file, &ranges, &arena, &format!("wave {wave}"));
        let reads = &e.queue().ran[ran..];
        assert_eq!(reads.len(), ranges.len(), "wave {wave}");
        assert!(reads.iter().all(|r| r.direct == (wave == 2)), "wave {wave}");
    }

    assert_eq!(e.counts(), (12, 0));
    assert!(e.direct());
}

#[test]
fn a_kernel_release_reads_as_its_version() {
    for (release, before) in [
        ("5.10.269-247.1004.amzn2.x86_64", true),
        ("5.11.0", true),
        ("5.12.0-rc1", false),
        ("6.1.0-13-amd64", false),
        ("4.19.0", true),
        ("10.0", false),
        ("", false),
        ("linux", false),
        ("5", false),
    ] {
        assert_eq!(release_before(release, 5, 12), before, "{release}");
    }
}

#[test]
fn dropping_the_engine_waits_for_the_reads_the_kernel_holds() {
    let len = 1 << 18;
    let file = random_bytes(len, 11);
    let ranges = packed(8, 0, 8192, 4096);
    let mut arenas = [Arena::aligned(8 * 4096, 4096), Arena::aligned(8 * 4096, 4096)];
    // Reads complete only when waited for, so every one is still out when the engine drops. The fake's drop checks
    // that none is left.
    let fake = Fake::new(&file, Order::Newest, 64, 64);
    let mut e = engine(fake, settings(true, len), &[place(&arenas[0]), place(&arenas[1])]);
    e.submit(0, &ranges, Memory::of(&mut arenas[0])).unwrap();
    e.submit(1, &ranges, Memory::of(&mut arenas[1])).unwrap();
    drop(e);

    for arena in &arenas {
        check(&file, &ranges, arena, "after the drop");
    }
}

#[test]
fn reads_the_kernel_never_took_keep_the_lane_but_not_the_drop() {
    let len = 1 << 18;
    let file = random_bytes(len, 12);
    let ranges = packed(4, 0, 8192, 4096);
    let mut arena = Arena::aligned(4 * 4096, 4096);
    let mut fake = Fake::new(&file, Order::Random, 64, 64);
    fake.broken = true;
    let mut e = engine(fake, settings(true, len), &[place(&arena)]);

    // The reads stay queued, so a new wave on the lane would race them, and fails instead. The drop doesn't wait.
    assert!(e.submit(0, &ranges, Memory::of(&mut arena)).is_err());
    assert!(e.submit(0, &ranges, Memory::of(&mut arena)).is_err());
    assert!(e.queue().ran.is_empty());
    drop(e);
}

#[test]
fn a_wave_checks_its_ranges_and_its_arena() {
    let len = 1 << 18;
    let file = random_bytes(len, 13);
    let mut arenas = [Arena::aligned(8192, 4096), Arena::aligned(8192, 4096)];
    let mut e = engine(
        Fake::new(&file, Order::Random, 8, 8),
        settings(true, len),
        &[place(&arenas[0]), place(&arenas[1])],
    );
    let too_long = [Range {
        offset: 0,
        len: 8192,
        tag: 0,
        at: 4096,
    }];
    let kind = |r: Result<(), Error>| r.map_err(|e| e.kind()).unwrap_err();

    assert_eq!(
        kind(e.submit(0, &too_long, Memory::of(&mut arenas[0]))),
        ErrorKind::InvalidInput
    );
    assert!(e.queue().ran.is_empty());
    let ranges = packed(2, 0, 8192, 4096);
    e.submit(0, &ranges, Memory::of(&mut arenas[0])).unwrap();
    let [a, b] = &mut arenas;
    assert_eq!(e.next(0, Memory::of(b)).unwrap_err().kind(), ErrorKind::InvalidInput);
    take_all(&mut e, 0, 2, a);
    // A lane without a wave has nothing to hand over.
    assert_eq!(e.next(1, Memory::of(b)).unwrap(), None);
}

#[test]
fn a_tag_comes_back_as_it_went() {
    for (lane, direct, range, done) in [
        (0, false, 0, 0),
        (1, true, (1 << 30) - 1, u32::MAX as usize),
        (1, false, 7, 4096),
    ] {
        let tag = Tag {
            lane,
            direct,
            range,
            done,
        };
        assert_eq!(Tag::unpack(tag.pack()), tag);
    }
}

#[test]
fn a_direct_read_needs_its_offset_length_and_memory_on_boundaries() {
    let align = Align {
        offset: 512,
        memory: 4096,
    };
    let range = |offset, len, at| Range {
        offset,
        len,
        tag: 0,
        at,
    };
    let len = |r, base, last| direct_len(r, base, 1 << 20, align, last);

    assert_eq!(len(range(4096, 8192, 4096), 1 << 16, false), Some(8192));
    assert_eq!(len(range(512, 1024, 0), 1 << 16, false), Some(1024));
    assert_eq!(len(range(100, 1024, 0), 1 << 16, false), None);
    assert_eq!(len(range(4096, 8192, 512), 1 << 16, false), None);
    assert_eq!(len(range(4096, 8192, 0), (1 << 16) + 8, false), None);
    assert_eq!(len(range(4096, 0, 0), 1 << 16, true), None);
    // A length off the boundaries rounds up only for the last range, and only into room the arena has.
    assert_eq!(len(range(4096, 1000, 0), 1 << 16, false), None);
    assert_eq!(len(range(4096, 1000, 0), 1 << 16, true), Some(1024));
    assert_eq!(direct_len(range(4096, 1000, 4096), 1 << 16, 5000, align, true), None);
    assert_eq!(
        direct_len(range(4096, 1000, 4096), 1 << 16, 5120, align, true),
        Some(1024)
    );
}

#[test]
fn the_pinned_count_never_passes_its_limit() {
    let pinned = Pinned::new();

    assert!(pinned.take(4096, 10_000));
    assert!(pinned.take(4096, 10_000));
    assert!(!pinned.take(4096, 10_000));
    assert!(!pinned.take(usize::MAX, usize::MAX));
    assert_eq!(pinned.held(), 8192);
    pinned.give_back(4096);
    assert!(pinned.take(5000, 10_000));
    assert_eq!(pinned.held(), 9096);

    // Registering pins every page that the memory touches.
    assert_eq!(pinned_span(4096, 4096, 4096), 4096);
    assert_eq!(pinned_span(4097, 4096, 4096), 8192);
    assert_eq!(pinned_span(100, 1, 4096), 4096);
    assert_eq!(pinned_span(100, 0, 4096), 0);
}
