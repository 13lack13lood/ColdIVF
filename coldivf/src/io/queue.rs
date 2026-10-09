//! Read waves over a kernel queue that takes many reads at once and hands back each one as it completes, eg. io_uring
//! on Linux. The engine plans each range's read, sends a whole wave to the queue, takes the completions in the order
//! they come, and reads the rest of any range that came back short. It never hands a lane's arena to a new wave while
//! the kernel may still land reads in it. Nothing here touches the system, so tests run it over a queue of their own.

use std::io::{Error, ErrorKind};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use super::extents::ExtentCache;
use super::{LANES, Memory, Range};

/// The most arenas that one engine registers.
pub(crate) const KEPT: usize = 4;

/// The most bytes that one read asks for. A longer range is read in pieces.
const PIECE: usize = 1 << 30;

/// The largest arena that the engine registers, 64 MiB. Registering faults in and pins every page of an arena, so a
/// larger one, eg. a cluster arena sized for thousands of clusters, stays plain memory that only its reads touch.
const MOST_PINNED: usize = 64 << 20;

/// Linux's numbers for the errors that the engine tells apart. io_uring completions carry them on every system that
/// compiles this.
const EINTR: i32 = 4;
const EAGAIN: i32 = 11;
const EBUSY: i32 = 16;
const EINVAL: i32 = 22;

/// The bit of an arrived range that keeps it out of the extent cache, since it read nothing, came from the cache or
/// went through the operating system's cache.
const NO_OFFER: u32 = 1 << 31;

/// One read for the kernel: `len` bytes from `offset` of the cold handle, or of the direct one, into memory at `addr`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Read {
    pub(crate) direct: bool,
    pub(crate) offset: u64,
    pub(crate) addr: usize,
    pub(crate) len: u32,
    /// The registered arena that holds the memory, if any.
    pub(crate) buffer: Option<u16>,
    pub(crate) tag: u64,
}

/// What the engine needs from the kernel's queue of reads.
pub(crate) trait Queue {
    /// Queues `read`, or returns false while the queue is full, until `enter` hands what it holds to the kernel.
    fn push(&mut self, read: &Read) -> bool;

    /// Hands every queued read to the kernel, then waits until at least `want` completions are in.
    fn enter(&mut self, want: usize) -> Result<(), Error>;

    /// Moves completions that are in, each a read's tag and its result, into `out`, and returns how many. A result is
    /// the bytes read, or a negative error number.
    fn reap(&mut self, out: &mut [(u64, i32)]) -> usize;

    /// How many reads the kernel can have at once without dropping a completion.
    fn room(&self) -> usize;

    /// How many queued reads the kernel hasn't taken yet.
    fn unsent(&mut self) -> usize;

    /// Registers `arenas`, each an address and a length, as buffers 0, 1 and so on, which pins their memory.
    fn register(&mut self, arenas: &[(usize, usize)]) -> Result<(), Error>;
}

/// What direct reads need. The file offset and the length must be multiples of `offset`, and the memory address a
/// multiple of `memory`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Align {
    pub(crate) offset: usize,
    pub(crate) memory: usize,
}

/// How many bytes a direct read of `r` asks for, into an arena at address `base` that is `len` bytes long, or `None`
/// if `r` has to go through the operating system's cache. A length off the boundaries rounds up only for `last`, the
/// range that ends both the file and the arena's ranges, since the device writes whole blocks and the bytes past the
/// file then land in spare room.
pub(crate) fn direct_len(r: Range, base: usize, len: usize, align: Align, last: bool) -> Option<usize> {
    if r.len == 0 || !r.offset.is_multiple_of(align.offset) || !(base + r.at).is_multiple_of(align.memory) {
        return None;
    }

    if r.len.is_multiple_of(align.offset) {
        return Some(r.len);
    }

    let rounded = r.len.next_multiple_of(align.offset);
    (last && r.at + rounded <= len).then_some(rounded)
}

/// The bytes of memory that registered arenas pin, across every engine of a process, so that together they stay within
/// the RLIMIT_MEMLOCK limit.
pub(crate) struct Pinned(AtomicUsize);

impl Pinned {
    pub(crate) const fn new() -> Pinned {
        Pinned(AtomicUsize::new(0))
    }

    /// Counts `bytes` more if the total stays at most `limit`, and says whether it did.
    pub(crate) fn take(&self, bytes: usize, limit: usize) -> bool {
        let mut held = self.0.load(Ordering::Acquire);

        loop {
            let Some(total) = held.checked_add(bytes).filter(|&total| total <= limit) else {
                return false;
            };

            match self
                .0
                .compare_exchange_weak(held, total, Ordering::AcqRel, Ordering::Acquire)
            {
                Ok(_) => return true,
                Err(now) => held = now,
            }
        }
    }

    /// Stops counting `bytes` that `take` counted.
    pub(crate) fn give_back(&self, bytes: usize) {
        self.0.fetch_sub(bytes, Ordering::AcqRel);
    }

    pub(crate) fn held(&self) -> usize {
        self.0.load(Ordering::Acquire)
    }
}

/// Whether the kernel release `release`, eg. "5.10.269-247.1004.amzn2.x86_64", is older than `major`.`minor`. A release
/// that doesn't start with two numbers counts as newer.
pub(crate) fn release_before(release: &str, major: u32, minor: u32) -> bool {
    let mut parts = release
        .split(|c: char| !c.is_ascii_digit())
        .map(|p| p.parse::<u32>().ok());

    match (parts.next().flatten(), parts.next().flatten()) {
        (Some(a), Some(b)) => (a, b) < (major, minor),
        _ => false,
    }
}

/// The bytes of the whole pages of `page` bytes that the `len` bytes at address `base` touch, which is what
/// registering them pins.
pub(crate) fn pinned_span(base: usize, len: usize, page: usize) -> usize {
    if len == 0 {
        return 0;
    }

    (base + len).div_ceil(page) * page - base / page * page
}

/// The tag that a read carries to its completion. Bit 63 holds the lane, bit 62 whether the read went through the
/// direct handle, bits 32 to 61 the range, and bits 0 to 31 how many of the range's bytes earlier reads brought.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Tag {
    pub(crate) lane: usize,
    pub(crate) direct: bool,
    pub(crate) range: usize,
    pub(crate) done: usize,
}

impl Tag {
    pub(crate) fn pack(self) -> u64 {
        ((self.lane as u64) << 63) | (u64::from(self.direct) << 62) | ((self.range as u64) << 32) | self.done as u64
    }

    pub(crate) fn unpack(tag: u64) -> Tag {
        Tag {
            lane: (tag >> 63) as usize,
            direct: (tag >> 62) & 1 == 1,
            range: ((tag >> 32) & 0x3FFF_FFFF) as usize,
            done: (tag & 0xFFFF_FFFF) as usize,
        }
    }
}

/// What an engine is set up with.
pub(crate) struct Settings {
    /// Whether the queue reads through a direct handle, which skips the operating system's cache.
    pub(crate) direct: bool,
    pub(crate) align: Align,
    pub(crate) file_len: usize,
    /// The page size that the ranges round out to, which is also the extent cache's page.
    pub(crate) read_page: usize,
    /// The most pages that the extent cache keeps, or 0 for no cache.
    pub(crate) cache_pages: usize,
    /// The most bytes that the process may pin for registered arenas.
    pub(crate) budget: usize,
    /// Whether the engine registers arenas when asked. Before Linux 5.13 registering stops the ring until a grace
    /// period of the kernel's read-copy-update (RCU) passes, which takes milliseconds.
    pub(crate) registers: bool,
    /// The system's page size, the unit that registered memory is pinned in.
    pub(crate) page: usize,
    pub(crate) pinned: &'static Pinned,
    /// Whether direct reads skip the alignment check, so the kernel turns down the ones off its boundaries.
    pub(crate) unchecked: bool,
}

/// One search context's reads of one index file through a queue. Each lane holds one wave at a time.
pub(crate) struct Engine<Q: Queue> {
    queue: Q,
    /// Whether direct reads still go out. A direct read that the kernel turns down turns them off.
    direct: bool,
    align: Align,
    file_len: usize,
    unchecked: bool,
    cache: ExtentCache,
    lanes: [Lane; LANES],
    /// The reads in the queue or in the kernel's hands, on both lanes.
    in_flight: usize,
    /// The arenas that `keep` named, each an address and a length, then the registered ones, as buffers 0, 1 and so on.
    kept: [(usize, usize); KEPT],
    nkept: usize,
    buffers: [(usize, usize); KEPT],
    nbuffers: usize,
    /// Whether registering was tried, and how many bytes it pinned, out of `budget` for the whole process.
    tried: bool,
    held: usize,
    budget: usize,
    registers: bool,
    page: usize,
    pinned: &'static Pinned,
    /// How many reads went to the queue, and how many ranges came from the cache.
    reads: usize,
    hits: usize,
}

/// One lane's wave: a copy of its ranges, where its arena is, and how far each range has come.
#[derive(Default)]
struct Lane {
    ranges: Vec<Range>,
    base: usize,
    len: usize,
    /// Where the last range in the arena ends.
    end: usize,
    /// The registered buffer that holds the arena, if any.
    buffer: Option<u16>,
    /// Whether the wave reads through the operating system's cache, even where direct reads go out.
    cached: bool,
    /// The reads still to send, from `todo[sent]` on.
    todo: Vec<Piece>,
    sent: usize,
    /// The ranges that are in, in the order they came, some with `NO_OFFER` set, and how many `next` returned.
    arrived: Vec<u32>,
    handed: usize,
    /// The lane's reads in the queue or in the kernel's hands.
    in_flight: usize,
    failed: Option<Error>,
    /// Whether the wave was dropped, so its completions only count down.
    dropped: bool,
}

/// The part of range `range` from byte `done` on, which a read still has to bring.
#[derive(Clone, Copy)]
struct Piece {
    range: u32,
    done: u32,
}

impl<Q: Queue> Engine<Q> {
    pub(crate) fn new(queue: Q, s: Settings) -> Engine<Q> {
        Engine {
            queue,
            direct: s.direct,
            align: s.align,
            file_len: s.file_len,
            unchecked: s.unchecked,
            cache: ExtentCache::new(s.cache_pages, s.read_page, s.file_len),
            lanes: Default::default(),
            in_flight: 0,
            kept: [(0, 0); KEPT],
            nkept: 0,
            buffers: [(0, 0); KEPT],
            nbuffers: 0,
            tried: false,
            held: 0,
            budget: s.budget,
            registers: s.registers,
            page: s.page,
            pinned: s.pinned,
            reads: 0,
            hits: 0,
        }
    }

    /// Names arenas, each an address and a length, that stay where they are until the engine drops, so it may
    /// register them. It also makes room for waves of up to `most` ranges, so a wave allocates nothing.
    pub(crate) fn keep(&mut self, arenas: &[(usize, usize)], most: usize) {
        self.nkept = arenas.len().min(KEPT);
        self.kept[..self.nkept].copy_from_slice(&arenas[..self.nkept]);

        for l in &mut self.lanes {
            l.ranges.reserve(most);
            l.todo.reserve(most);
            l.arrived.reserve(most);
        }
    }

    /// Starts reading `ranges` into `arena` on lane `lane`, as `Wave::submit` does. A range whose pages are all in the
    /// extent cache arrives at once.
    pub(crate) fn submit(&mut self, lane: usize, ranges: &[Range], arena: Memory) -> Result<(), Error> {
        self.start(lane, ranges, arena, false)
    }

    /// The same, but every read of the wave goes through the operating system's cache, as for `Wave::submit_cached`.
    pub(crate) fn submit_cached(&mut self, lane: usize, ranges: &[Range], arena: Memory) -> Result<(), Error> {
        self.start(lane, ranges, arena, true)
    }

    fn start(&mut self, lane: usize, ranges: &[Range], arena: Memory, cached: bool) -> Result<(), Error> {
        let fits = |r: &Range| r.len <= u32::MAX as usize && r.at.checked_add(r.len).is_some_and(|e| e <= arena.len());

        if ranges.len() >= 1 << 30 || !ranges.iter().all(fits) {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "a read range doesn't fit its arena",
            ));
        }

        self.drop_lane(lane)?;
        let base = arena.addr();
        let same = |&b: &(usize, usize)| b == (base, arena.len());
        let buffer = self.buffers[..self.nbuffers].iter().position(same).map(|b| b as u16);
        let l = &mut self.lanes[lane];
        l.ranges.extend_from_slice(ranges);
        (l.base, l.len, l.buffer, l.cached) = (base, arena.len(), buffer, cached);
        l.end = ranges.iter().map(|r| r.at + r.len).max().unwrap_or(0);

        for (i, r) in ranges.iter().enumerate() {
            // SAFETY: the lane's last wave was dropped above, which waited for each of its reads, so no read lands in
            // the lane's memory, and the caller gives each lane memory of its own.
            if r.len == 0 || self.cache.get(r.offset, unsafe { arena.slice_mut(r.at, r.len) }) {
                self.hits += usize::from(r.len > 0);
                l.arrived.push(i as u32 | NO_OFFER);
            } else {
                l.todo.push(Piece {
                    range: i as u32,
                    done: 0,
                });
            }
        }

        self.send()
    }

    /// Waits until one more range of the lane's last submit is in its arena, `arena`, and returns its index, or `None`
    /// once every range has been returned. Ranges come in the order their reads complete.
    pub(crate) fn next(&mut self, lane: usize, arena: Memory) -> Result<Option<usize>, Error> {
        let l = &self.lanes[lane];

        if !l.ranges.is_empty() && (arena.addr() != l.base || arena.len() != l.len) {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "a wave's ranges were asked for with another arena than its submit's",
            ));
        }

        loop {
            let l = &mut self.lanes[lane];

            if let Some(e) = l.failed.take() {
                return Err(e);
            }

            if let Some(&came) = l.arrived.get(l.handed) {
                l.handed += 1;
                let i = (came & !NO_OFFER) as usize;

                if came & NO_OFFER == 0 {
                    let r = l.ranges[i];
                    // SAFETY: every read of range i is in, so none lands in its bytes, and the caller reads them only
                    // once this returns.
                    self.cache.offer(r.offset, unsafe { arena.slice(r.at, r.len) });
                }

                return Ok(Some(i));
            }

            if l.handed == l.ranges.len() {
                return Ok(None);
            }

            self.wait()?;
        }
    }

    /// Whether direct reads still go out.
    pub(crate) fn direct(&self) -> bool {
        self.direct
    }

    /// How many arenas are registered.
    pub(crate) fn registered(&self) -> usize {
        self.nbuffers
    }

    /// How many reads went to the queue, and how many ranges came from the extent cache.
    pub(crate) fn counts(&self) -> (usize, usize) {
        (self.reads, self.hits)
    }

    #[cfg(test)]
    pub(crate) fn queue(&mut self) -> &mut Q {
        &mut self.queue
    }

    /// Waits for at least one read of either lane, takes in every completion, then sends what the lanes still need.
    fn wait(&mut self) -> Result<(), Error> {
        if self.in_flight == 0 {
            self.send()?;
        }

        if self.in_flight == 0 {
            return Err(Error::other("a read wave ended before all of its ranges arrived"));
        }

        self.enter(1)?;
        self.take();
        self.send()
    }

    /// Queues the reads that both lanes still need, while the kernel has room for their completions, then hands them
    /// to the kernel.
    fn send(&mut self) -> Result<(), Error> {
        let mut queued = false;

        for lane in 0..LANES {
            while self.in_flight < self.queue.room() {
                let l = &self.lanes[lane];
                let Some(&piece) = l.todo.get(l.sent) else {
                    break;
                };
                let read = self.read(lane, piece);

                // A full queue first hands what it holds to the kernel.
                while !self.queue.push(&read) {
                    self.enter(0)?;
                }

                let l = &mut self.lanes[lane];
                (l.sent, l.in_flight) = (l.sent + 1, l.in_flight + 1);
                (self.in_flight, self.reads, queued) = (self.in_flight + 1, self.reads + 1, true);
            }

            let l = &mut self.lanes[lane];

            if l.sent == l.todo.len() {
                l.todo.clear();
                l.sent = 0;
            }
        }

        if queued {
            self.enter(0)?;
        }

        Ok(())
    }

    /// The read that brings piece `p` of lane `lane`'s wave.
    fn read(&self, lane: usize, p: Piece) -> Read {
        let l = &self.lanes[lane];
        let (r, done) = (l.ranges[p.range as usize], p.done as usize);
        let rest = Range {
            offset: r.offset + done,
            len: r.len - done,
            tag: r.tag,
            at: r.at + done,
        };
        let last = r.end() == self.file_len && r.at + r.len == l.end;
        let direct = match (self.direct && !l.cached, self.unchecked) {
            (false, _) => None,
            (true, true) => Some(rest.len),
            (true, false) => direct_len(rest, l.base, l.len, self.align, last),
        };
        let (direct, len) = direct.map_or((false, rest.len), |len| (true, len));
        let range = p.range as usize;
        Read {
            direct,
            offset: rest.offset as u64,
            addr: l.base + rest.at,
            len: len.min(PIECE) as u32,
            buffer: l.buffer,
            tag: Tag {
                lane,
                direct,
                range,
                done,
            }
            .pack(),
        }
    }

    /// Hands the queued reads to the kernel and waits for `want` completions. When the kernel is short of memory for
    /// reads, or of room for completions, it tries again once it has taken the completions that are in.
    fn enter(&mut self, want: usize) -> Result<(), Error> {
        let mut tries = 0;

        loop {
            match self.queue.enter(want) {
                Ok(()) => return Ok(()),
                Err(e) if transient(&e) && tries < 1000 => {
                    tries += 1;

                    if self.take() > 0 && want > 0 {
                        return Ok(());
                    }

                    std::thread::sleep(Duration::from_micros(100));
                }
                Err(e) => return Err(e),
            }
        }
    }

    /// Takes in every completion that is in, and returns how many there were.
    fn take(&mut self) -> usize {
        let mut came = [(0u64, 0i32); 32];
        let mut total = 0;

        loop {
            let n = self.queue.reap(&mut came);

            for &(tag, res) in &came[..n] {
                self.complete(tag, res);
            }

            total += n;

            if n < came.len() {
                return total;
            }
        }
    }

    /// Takes in the completion of the read tagged `tag`, which returned `res`.
    fn complete(&mut self, tag: u64, res: i32) {
        let t = Tag::unpack(tag);
        self.in_flight -= 1;
        let l = &mut self.lanes[t.lane];
        l.in_flight -= 1;

        if l.dropped {
            return;
        }

        let piece = Piece {
            range: t.range as u32,
            done: t.done as u32,
        };

        match res {
            // An interrupted read, or one the kernel had no room for yet, goes out again.
            r if r == -EINTR || r == -EAGAIN => l.todo.push(piece),
            // The kernel turned a direct read down, eg. for its alignment, so it and every later read go through the
            // operating system's cache.
            r if r == -EINVAL && t.direct => {
                self.direct = false;
                l.todo.push(piece);
            }
            r if r < 0 => {
                if l.failed.is_none() {
                    l.failed = Some(Error::from_raw_os_error(-r));
                }
            }
            0 => {
                if l.failed.is_none() {
                    l.failed = Some(Error::new(
                        ErrorKind::UnexpectedEof,
                        "the file ended before the read did",
                    ));
                }
            }
            n => {
                let done = t.done + n as usize;

                if done >= l.ranges[t.range].len {
                    l.arrived.push(t.range as u32 | if t.direct { 0 } else { NO_OFFER });
                } else {
                    l.todo.push(Piece {
                        range: t.range as u32,
                        done: done as u32,
                    });
                }
            }
        }
    }

    /// Forgets the lane's last wave, then waits for every read of it, since each may still land in its arena. It fails
    /// if the kernel won't take reads that are still queued, since they could land later, and the lane then stays
    /// without a wave.
    fn drop_lane(&mut self, lane: usize) -> Result<(), Error> {
        let l = &mut self.lanes[lane];
        l.ranges.clear();
        l.todo.clear();
        l.arrived.clear();
        (l.sent, l.handed, l.failed, l.dropped) = (0, 0, None, true);

        while self.lanes[lane].in_flight > 0 {
            if let Err(e) = self.queue.enter(1) {
                if self.queue.unsent() > 0 && !transient(&e) {
                    return Err(e);
                }

                std::thread::sleep(Duration::from_millis(1));
            }

            self.take();
        }

        self.lanes[lane].dropped = false;
        Ok(())
    }

    /// Registers the arenas that `keep` named, once, when the engine registers at all and the kernel holds no reads,
    /// eg. between searches. It pins and faults in their every page, so it takes only arenas of up to 64 MiB that fit
    /// the budget, counted across the process. Any failure leaves them to plain reads.
    pub(crate) fn register(&mut self) {
        if !self.registers || self.tried || self.in_flight > 0 || self.nkept == 0 {
            return;
        }

        self.tried = true;
        let (mut chosen, mut n, mut bytes) = ([(0, 0); KEPT], 0, 0);

        for &(base, len) in &self.kept[..self.nkept] {
            let pin = pinned_span(base, len, self.page);

            if len > 0 && len <= MOST_PINNED && self.pinned.take(pin, self.budget) {
                chosen[n] = (base, len);
                (n, bytes) = (n + 1, bytes + pin);
            }
        }

        if n == 0 {
            return;
        }

        match self.queue.register(&chosen[..n]) {
            Ok(()) => (self.buffers, self.nbuffers, self.held) = (chosen, n, bytes),
            Err(_) => self.pinned.give_back(bytes),
        }
    }
}

impl<Q: Queue> Drop for Engine<Q> {
    fn drop(&mut self) {
        // Reads that the kernel holds may still land in arenas, so it waits for them. Reads that it never took go
        // away with the queue.
        for l in &mut self.lanes {
            (l.dropped, l.sent) = (true, 0);
            l.todo.clear();
        }

        while self.in_flight > self.queue.unsent() {
            if self.queue.enter(1).is_err() {
                std::thread::sleep(Duration::from_millis(1));
            }

            self.take();
        }

        self.pinned.give_back(self.held);
    }
}

/// Whether a failed entry into the queue may work once completions are taken, as when the kernel was short of memory
/// for reads (EAGAIN) or of room for completions (EBUSY).
fn transient(e: &Error) -> bool {
    matches!(e.raw_os_error(), Some(EAGAIN | EBUSY))
}

#[cfg(test)]
#[path = "../../tests/unit/io/queue.rs"]
mod tests;
