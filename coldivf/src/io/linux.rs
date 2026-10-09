//! The Linux backend. A search context's waves go through io_uring, which takes a whole wave in one system call and
//! hands back each range as it completes. A cold wave reads through the context's own handle with `O_DIRECT`, which
//! skips the operating system's cache, so the context keeps the pages it reads twice in an extent cache. A warm
//! search's row wave reads through the index's cold handle and the cache instead, since its rows are likely there. On a
//! file system that refuses direct reads every wave goes through the cold handle and the cache. Where the kernel blocks
//! io_uring, eg. through the `kernel.io_uring_disabled` sysctl or a container's seccomp filter, a wave tells the kernel
//! about every range with `posix_fadvise`, which starts them all at once, then reads each one in order, as on macOS.
//!
//! Each context's reads of an index hold a ring with its file descriptor, the direct handle, and an extent cache of up
//! to 16 MiB, of which only the pages it keeps take memory. Before Linux 5.12 each ring also has kernel threads of its
//! own for the reads that wait, and its memory counts against the user's RLIMIT_MEMLOCK limit, often 64 KiB for all of
//! the user's processes. The reads keep the index's file open, even once the index is closed or a rebuild renames
//! another file over it, until the context drops or makes the reads of another index.

use std::ffi::CStr;
use std::fs::File;
use std::io::{Error, ErrorKind};
use std::mem::ManuallyDrop;
use std::os::fd::{AsRawFd, RawFd};
use std::os::unix::fs::OpenOptionsExt;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

use io_uring::{IoUring, Probe, opcode, types};

use super::queue::{Align, Engine, KEPT, Pinned, Queue, Read, Settings, release_before};
use super::{InOrder, Memory, Range, Wave};
use crate::trace::ReadPath;

/// The page size that reads round out to. Direct reads take it as their alignment where the kernel doesn't say, as
/// before Linux 6.1.
const PAGE: usize = 4096;

/// The ring sizes to try, largest first. A ring of 256 entries takes 32 KiB of memory, and one of 64 takes 8 KiB.
const ENTRIES: [u32; 3] = [256, 64, 16];

/// The ring sizes to try where a ring's memory counts against a finite RLIMIT_MEMLOCK limit, as before Linux 5.12.
/// Smaller rings let more contexts have one.
const SMALL_ENTRIES: [u32; 2] = [64, 16];

/// The most pages of an index that a context keeps in its extent cache, 16 MiB.
const CACHED_PAGES: usize = 4096;

/// Set once io_uring's setup fails in a way that says the system blocks it, so later backends don't try again.
static BLOCKED: AtomicBool = AtomicBool::new(false);

/// Set once the kernel turns down MADV_POPULATE_READ, which arrived in Linux 5.14.
static NO_POPULATE: AtomicBool = AtomicBool::new(false);

/// The bytes that the registered arenas of every backend in the process pin.
static PINNED: Pinned = Pinned::new();

/// How a backend sets itself up. Tests change it to reach each fallback.
#[derive(Clone, Copy)]
pub(crate) struct Setup {
    /// The ring sizes to try, largest first, or `None` to choose them from the kernel and the RLIMIT_MEMLOCK limit.
    pub(crate) entries: Option<&'static [u32]>,
    /// Whether cold waves read through a direct handle.
    pub(crate) direct: bool,
    /// The most bytes of arenas that the process may register, or `None` for the RLIMIT_MEMLOCK limit.
    pub(crate) budget: Option<usize>,
    /// Whether to register the files and the arenas with the kernel, or `None` to register them only where that
    /// doesn't stop the ring, from Linux 5.13. Before it, registering waits for every read of the ring to finish and
    /// for a grace period of the kernel's read-copy-update (RCU), which takes milliseconds.
    pub(crate) register: Option<bool>,
    /// An error number that setting up the ring fails with, as on a system that blocks io_uring.
    pub(crate) ring_error: Option<i32>,
    /// An error number that opening the direct handle fails with, as on a file system without direct reads.
    pub(crate) direct_error: Option<i32>,
    /// Whether direct reads skip the alignment check, so the kernel turns down the ones off its boundaries.
    pub(crate) unchecked: bool,
    /// Whether `statx` is asked for the alignment of direct reads. A kernel before Linux 6.1, eg. Linux 5.10, doesn't
    /// know the question, and false asks as on one of those, so direct reads keep to whole pages.
    pub(crate) ask_alignment: bool,
    /// Where a setup failure that says the system blocks io_uring is remembered, so later backends don't try again.
    pub(crate) blocked: &'static AtomicBool,
}

impl Default for Setup {
    fn default() -> Setup {
        Setup {
            entries: None,
            direct: true,
            budget: None,
            register: None,
            ring_error: None,
            direct_error: None,
            unchecked: false,
            ask_alignment: true,
            blocked: &BLOCKED,
        }
    }
}

/// The backend that `Index::open` picks on Linux. It sets itself up when it is readied, or at its first wave through a
/// cold handle it wasn't readied for. Where io_uring can't be had, it reads with `posix_fadvise` waves instead.
#[derive(Default)]
pub(crate) struct UringWave {
    setup: Setup,
    mode: Mode,
    /// The arenas that `ready` named, each an address and a length, and the most ranges that a wave may have.
    kept: [(usize, usize); KEPT],
    nkept: usize,
    most: usize,
}

#[derive(Default)]
enum Mode {
    #[default]
    Unset,
    Ring(Box<Ring>),
    Fadvise(FadviseWave),
}

/// io_uring reads of the cold handle `fd`, set up by the process `pid`, with a ring of `entries`.
struct Ring {
    fd: RawFd,
    pid: u32,
    entries: u32,
    fixed_files: bool,
    /// The engine, which only the process that set the ring up drops.
    engine: ManuallyDrop<Engine<Uring>>,
}

/// What a backend is set up as, for tests to check.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum State {
    Unset,
    /// io_uring, with whether direct reads still go out, how many arenas are registered, how many reads went to the
    /// kernel and how many ranges came from the extent cache.
    Ring {
        direct: bool,
        registered: usize,
        reads: usize,
        hits: usize,
    },
    Fadvise,
}

impl UringWave {
    /// A backend that sets itself up as `setup` says.
    pub(crate) fn with(setup: Setup) -> UringWave {
        UringWave {
            setup,
            ..UringWave::default()
        }
    }

    pub(crate) fn state(&self) -> State {
        match &self.mode {
            Mode::Unset => State::Unset,
            Mode::Ring(ring) => {
                let (reads, hits) = ring.engine.counts();
                State::Ring {
                    direct: ring.engine.direct(),
                    registered: ring.engine.registered(),
                    reads,
                    hits,
                }
            }
            Mode::Fadvise(_) => State::Fadvise,
        }
    }

    /// The reads that the backend's queue has handed to the kernel so far, by kind.
    #[cfg(test)]
    pub(crate) fn sent(&mut self) -> Sent {
        match &mut self.mode {
            Mode::Ring(ring) => ring.engine.queue().sent,
            _ => Sent::default(),
        }
    }

    /// Sets the backend up for the cold handle `file`, with io_uring if it can, and with `posix_fadvise` waves if not.
    fn start(&mut self, file: &File) {
        // An earlier setup waits for its reads as it drops.
        self.mode = Mode::Unset;
        self.mode = match Ring::open(file, &self.setup, &self.kept[..self.nkept], self.most) {
            Ok(ring) => Mode::Ring(Box::new(ring)),
            Err(_) => Mode::Fadvise(FadviseWave::default()),
        };
    }

    /// Drops a ring that another process set up, eg. the parent before a fork, so that the next wave sets up one of
    /// this process's own.
    fn leave_inherited(&mut self) {
        if matches!(&self.mode, Mode::Ring(ring) if ring.pid != std::process::id()) {
            self.mode = Mode::Unset;
        }
    }

    /// Starts a wave, through the cache when `cached` says, after setting the backend up for `file` unless it is.
    fn wave(&mut self, lane: usize, file: &File, ranges: &[Range], arena: Memory, cached: bool) -> Result<(), Error> {
        self.leave_inherited();
        let set_up = match &self.mode {
            Mode::Unset => false,
            Mode::Ring(ring) => ring.fd == file.as_raw_fd(),
            Mode::Fadvise(_) => true,
        };

        if !set_up {
            self.start(file);
        }

        match &mut self.mode {
            Mode::Ring(ring) if cached => ring.engine.submit_cached(lane, ranges, arena),
            Mode::Ring(ring) => ring.engine.submit(lane, ranges, arena),
            Mode::Fadvise(fadvise) => fadvise.submit(lane, file, ranges, arena),
            Mode::Unset => Ok(()),
        }
    }
}

impl Wave for UringWave {
    fn page(&self) -> usize {
        PAGE
    }

    fn ready(&mut self, file: &File, arenas: &[Memory], most: usize) {
        self.leave_inherited();
        self.nkept = arenas.len().min(KEPT);

        for (kept, arena) in self.kept.iter_mut().zip(arenas) {
            *kept = (arena.addr(), arena.len());
        }

        self.most = most;

        match &mut self.mode {
            Mode::Ring(ring) if ring.fd == file.as_raw_fd() => ring.engine.keep(&self.kept[..self.nkept], most),
            _ => self.start(file),
        }
    }

    fn prepare(&mut self) {
        self.leave_inherited();

        if let Mode::Ring(ring) = &mut self.mode {
            ring.engine.register();
        }
    }

    fn submit(&mut self, lane: usize, file: &File, ranges: &[Range], arena: Memory) -> Result<(), Error> {
        self.wave(lane, file, ranges, arena, false)
    }

    fn submit_cached(&mut self, lane: usize, file: &File, ranges: &[Range], arena: Memory) -> Result<(), Error> {
        self.wave(lane, file, ranges, arena, true)
    }

    fn next(&mut self, lane: usize, file: &File, ranges: &[Range], arena: Memory) -> Result<Option<usize>, Error> {
        match &mut self.mode {
            Mode::Ring(ring) => ring.engine.next(lane, arena),
            Mode::Fadvise(fadvise) => fadvise.next(lane, file, ranges, arena),
            Mode::Unset => Ok(None),
        }
    }

    fn read_path(&self) -> ReadPath {
        match &self.mode {
            Mode::Unset => ReadPath::default(),
            Mode::Ring(ring) => ReadPath {
                backend: "io_uring",
                ring_entries: ring.entries,
                direct: ring.engine.direct(),
                fixed_files: ring.fixed_files,
                registered: ring.engine.registered(),
            },
            Mode::Fadvise(fadvise) => fadvise.read_path(),
        }
    }

    fn hits(&self) -> usize {
        match &self.mode {
            Mode::Ring(ring) => ring.engine.counts().1,
            _ => 0,
        }
    }
}

impl Ring {
    /// Sets up io_uring reads of the cold handle `file`. It fails where the system blocks io_uring, where the kernel is
    /// older than Linux 5.6, and where no ring fits RLIMIT_MEMLOCK. `kept` are the arenas that `ready` named.
    fn open(file: &File, setup: &Setup, kept: &[(usize, usize)], most: usize) -> Result<Ring, Error> {
        let ring = new_ring(setup)?;
        let entries = ring.params().sq_entries();
        let registers = setup
            .register
            .unwrap_or_else(|| ring.params().is_feature_resource_tagging());
        let file_len = usize::try_from(file.metadata()?.len()).map_err(|_| Error::from(ErrorKind::InvalidData))?;
        let direct = setup.direct.then(|| open_direct(file, setup).ok()).flatten();
        let align = direct.as_ref().and_then(|d| dio_align_asking(d, setup.ask_alignment));
        let uring = Uring::new(ring, file, direct.filter(|_| align.is_some()), registers);
        let (direct, fixed_files) = (uring.has_direct(), uring.fixed);
        let settings = Settings {
            direct,
            align: align.unwrap_or(Align {
                offset: PAGE,
                memory: PAGE,
            }),
            file_len,
            read_page: PAGE,
            cache_pages: if direct { CACHED_PAGES } else { 0 },
            budget: setup.budget.unwrap_or_else(memlock_limit),
            registers,
            page: page_size(),
            pinned: &PINNED,
            unchecked: setup.unchecked,
        };
        let mut engine = Engine::new(uring, settings);
        engine.keep(kept, most);
        Ok(Ring {
            fd: file.as_raw_fd(),
            pid: std::process::id(),
            entries,
            fixed_files,
            engine: ManuallyDrop::new(engine),
        })
    }
}

impl Drop for Ring {
    fn drop(&mut self) {
        // A ring belongs to the process that set it up. After a fork its memory isn't mapped in the child, and its
        // reads land in the parent's memory, so a child leaves it alone and only its own process waits for its reads.
        if self.pid == std::process::id() {
            // SAFETY: the engine drops here once, and nothing uses it after.
            unsafe { ManuallyDrop::drop(&mut self.engine) };
        }
    }
}

/// A ring of the largest size that the kernel gives. A failure that says the system blocks io_uring is remembered in
/// `setup.blocked`. A child of a fork doesn't get the ring's memory, so it can't touch the ring of its parent.
fn new_ring(setup: &Setup) -> Result<IoUring, Error> {
    if setup.blocked.load(Ordering::Relaxed) {
        return Err(Error::new(ErrorKind::Unsupported, "the system blocks io_uring"));
    }

    let mut short = Error::from(ErrorKind::OutOfMemory);

    for &entries in setup.entries.unwrap_or_else(ring_sizes) {
        let made = match setup.ring_error {
            Some(errno) => Err(Error::from_raw_os_error(errno)),
            None => IoUring::builder().dontfork().build(entries),
        };

        match made {
            Ok(ring) => return usable(ring),
            Err(e) if e.raw_os_error() == Some(libc::ENOMEM) => short = e,
            Err(e) => {
                if blocked(&e) {
                    setup.blocked.store(true, Ordering::Relaxed);
                }

                return Err(e);
            }
        }
    }

    Err(short)
}

/// The ring sizes to try. Where a ring's memory counts against a finite RLIMIT_MEMLOCK limit, they start smaller, so
/// the user's budget fits more of them.
fn ring_sizes() -> &'static [u32] {
    if memlock_limit() < usize::MAX && rings_take_memlock() {
        &SMALL_ENTRIES
    } else {
        &ENTRIES
    }
}

/// Whether a ring's memory counts against RLIMIT_MEMLOCK, as before Linux 5.12, from the kernel's release. A release
/// that doesn't read as a version counts as newer.
fn rings_take_memlock() -> bool {
    static OLD: OnceLock<bool> = OnceLock::new();
    *OLD.get_or_init(|| kernel_release().is_some_and(|r| release_before(&r, 5, 12)))
}

/// The kernel's release, eg. "5.10.269-247.1004.amzn2.x86_64".
fn kernel_release() -> Option<String> {
    // SAFETY: utsname is plain old data, for which all zeros is a valid value.
    let mut name: libc::utsname = unsafe { std::mem::zeroed() };

    // SAFETY: uname writes one utsname into `name`.
    if unsafe { libc::uname(&mut name) } != 0 {
        return None;
    }

    // A C character is signed on x86-64 and unsigned on Arm, so each one is taken as its byte.
    let bytes: Vec<u8> = name.release.iter().map(|c| c.to_ne_bytes()[0]).collect();
    let release = CStr::from_bytes_until_nul(&bytes).ok()?;
    Some(release.to_string_lossy().into_owned())
}

/// `ring`, if its kernel has what the backend needs: reads into plain and registered memory, and a completion queue
/// that never drops a completion. Both came by Linux 5.6.
fn usable(ring: IoUring) -> Result<IoUring, Error> {
    let mut probe = Probe::new();
    let known = ring.submitter().register_probe(&mut probe).is_ok()
        && probe.is_supported(opcode::Read::CODE)
        && probe.is_supported(opcode::ReadFixed::CODE);

    if !known || !ring.params().is_feature_nodrop() {
        return Err(Error::new(
            ErrorKind::Unsupported,
            "io_uring here is older than Linux 5.6",
        ));
    }

    Ok(ring)
}

/// Whether io_uring's setup failed because the system blocks io_uring. The `kernel.io_uring_disabled` sysctl and most
/// seccomp filters give EPERM, a kernel without io_uring gives ENOSYS, and some security modules give EACCES.
pub(crate) fn blocked(e: &Error) -> bool {
    matches!(e.raw_os_error(), Some(libc::EPERM | libc::ENOSYS | libc::EACCES))
}

/// Opens the file of `file` again with `O_DIRECT`, through /proc, so the new handle reads that same file even after a
/// rebuild renames another over its path. A file system without direct reads fails it with EINVAL.
fn open_direct(file: &File, setup: &Setup) -> Result<File, Error> {
    if let Some(errno) = setup.direct_error {
        return Err(Error::from_raw_os_error(errno));
    }

    std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_DIRECT)
        .open(format!("/proc/self/fd/{}", file.as_raw_fd()))
}

/// The alignment that direct reads of `file` need, from `statx` since Linux 6.1, or 4096 bytes where the kernel doesn't
/// say. `None` means that the file takes no direct reads, or needs them on boundaries past a page.
pub(crate) fn dio_align(file: &File) -> Option<Align> {
    dio_align_asking(file, true)
}

/// `dio_align`, asking `statx` for the alignment only when `ask` says. A kernel before Linux 6.1, eg. Linux 5.10,
/// answers without it either way, since it doesn't know STATX_DIOALIGN.
fn dio_align_asking(file: &File, ask: bool) -> Option<Align> {
    // SAFETY: statx is plain old data, for which all zeros is a valid value.
    let mut stx: libc::statx = unsafe { std::mem::zeroed() };
    let mask = if ask {
        libc::STATX_DIOALIGN
    } else {
        libc::STATX_BASIC_STATS
    };
    // SAFETY: with AT_EMPTY_PATH and an empty path, statx describes the open file `file` and writes one statx into
    // `stx`. It is called directly, since the C library's wrapper needs glibc 2.28.
    let rc = unsafe {
        libc::syscall(
            libc::SYS_statx,
            file.as_raw_fd(),
            c"".as_ptr(),
            libc::AT_EMPTY_PATH,
            mask,
            &mut stx as *mut libc::statx,
        )
    };

    align_from(rc, &stx)
}

/// The alignment of direct reads that a `statx` call gives, from its result `rc` and its answer `stx`. It is 4096 bytes
/// where the call failed, as before Linux 4.11, and where the answer lacks STATX_DIOALIGN, as before Linux 6.1. An
/// alignment of 0 says that the file takes no direct reads.
fn align_from(rc: libc::c_long, stx: &libc::statx) -> Option<Align> {
    if rc != 0 || (stx.stx_mask & libc::STATX_DIOALIGN) == 0 {
        return Some(Align {
            offset: PAGE,
            memory: PAGE,
        });
    }

    let (offset, memory) = (stx.stx_dio_offset_align as usize, stx.stx_dio_mem_align as usize);
    let fits = |a: usize| a.is_power_of_two() && a <= PAGE;
    (fits(offset) && fits(memory)).then_some(Align { offset, memory })
}

/// The soft RLIMIT_MEMLOCK limit in bytes, which caps the memory that registered arenas pin.
fn memlock_limit() -> usize {
    let mut limit = libc::rlimit {
        rlim_cur: 0,
        rlim_max: 0,
    };

    // SAFETY: getrlimit writes one rlimit into `limit`.
    if unsafe { libc::getrlimit(libc::RLIMIT_MEMLOCK, &mut limit) } != 0 {
        return 0;
    }

    if limit.rlim_cur == libc::RLIM_INFINITY {
        return usize::MAX;
    }

    usize::try_from(limit.rlim_cur).unwrap_or(usize::MAX)
}

/// The size of a page of memory, eg. 4 KiB on x86-64 and 4, 16 or 64 KiB on Arm.
fn page_size() -> usize {
    // SAFETY: sysconf only reads a setting.
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    usize::try_from(page).unwrap_or(PAGE).max(1)
}

/// io_uring as the engine's queue. It reads the cold handle as file 0 and the direct handle, if any, as file 1. Where
/// the backend registers, both are registered with the kernel, and a registered file stays open with the ring, so the
/// queue closes its own direct handle then, which saves a file descriptor.
struct Uring {
    ring: IoUring,
    cold: RawFd,
    direct: Direct,
    fixed: bool,
    #[cfg(test)]
    sent: Sent,
}

/// The queue's direct handle.
enum Direct {
    None,
    /// Kept open, since the files aren't registered.
    Open(File),
    Registered,
}

/// The reads that a queue handed to the kernel, by kind, for tests.
#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Sent {
    /// Reads of the direct handle, and of the cold handle.
    pub(crate) direct: usize,
    pub(crate) cold: usize,
    /// Reads that named their file by its registered slot, and reads into a registered arena.
    pub(crate) fixed_files: usize,
    pub(crate) fixed_buffers: usize,
}

impl Uring {
    fn new(ring: IoUring, cold: &File, direct: Option<File>, register: bool) -> Uring {
        let fds = [cold.as_raw_fd(), direct.as_ref().map_or(-1, AsRawFd::as_raw_fd)];
        let count = if direct.is_some() { 2 } else { 1 };
        let fixed = register && ring.submitter().register_files(&fds[..count]).is_ok();
        let direct = match direct {
            None => Direct::None,
            Some(_) if fixed => Direct::Registered,
            Some(file) => Direct::Open(file),
        };
        Uring {
            ring,
            cold: cold.as_raw_fd(),
            direct,
            fixed,
            #[cfg(test)]
            sent: Sent::default(),
        }
    }

    fn has_direct(&self) -> bool {
        !matches!(self.direct, Direct::None)
    }
}

impl Queue for Uring {
    fn push(&mut self, read: &Read) -> bool {
        let to = std::ptr::with_exposed_provenance_mut::<u8>(read.addr);
        let (file, fd) = match &self.direct {
            Direct::Open(direct) if read.direct => (types::Fixed(1), direct.as_raw_fd()),
            Direct::Registered if read.direct => (types::Fixed(1), -1),
            _ => (types::Fixed(0), self.cold),
        };
        let entry = match (self.fixed, read.buffer) {
            (true, Some(b)) => opcode::ReadFixed::new(file, to, read.len, b)
                .offset(read.offset)
                .build(),
            (true, None) => opcode::Read::new(file, to, read.len).offset(read.offset).build(),
            (false, Some(b)) => opcode::ReadFixed::new(types::Fd(fd), to, read.len, b)
                .offset(read.offset)
                .build(),
            (false, None) => opcode::Read::new(types::Fd(fd), to, read.len)
                .offset(read.offset)
                .build(),
        };
        // SAFETY: the read's memory lies in an arena that its owner keeps where it is, and leaves to the read, until
        // the engine takes the read's completion, as `Wave::submit` asks. The engine waits for every read that the
        // kernel holds before it gives a lane's arena to a new wave, and before it drops.
        let pushed = unsafe { self.ring.submission().push(&entry.user_data(read.tag)) }.is_ok();

        #[cfg(test)]
        if pushed {
            let s = &mut self.sent;
            (s.direct, s.cold) = if file.0 == 1 {
                (s.direct + 1, s.cold)
            } else {
                (s.direct, s.cold + 1)
            };
            s.fixed_files += usize::from(self.fixed);
            s.fixed_buffers += usize::from(read.buffer.is_some());
        }

        pushed
    }

    fn enter(&mut self, want: usize) -> Result<(), Error> {
        loop {
            match self.ring.submit_and_wait(want) {
                Ok(_) => return Ok(()),
                Err(e) if e.raw_os_error() == Some(libc::EINTR) => {}
                Err(e) => return Err(e),
            }
        }
    }

    fn reap(&mut self, out: &mut [(u64, i32)]) -> usize {
        let mut completions = self.ring.completion();
        let mut n = 0;

        for slot in out.iter_mut() {
            let Some(c) = completions.next() else {
                break;
            };
            *slot = (c.user_data(), c.result());
            n += 1;
        }

        n
    }

    fn room(&self) -> usize {
        self.ring.params().cq_entries() as usize
    }

    fn unsent(&mut self) -> usize {
        self.ring.submission().len()
    }

    fn register(&mut self, arenas: &[(usize, usize)]) -> Result<(), Error> {
        let empty = libc::iovec {
            iov_base: std::ptr::null_mut(),
            iov_len: 0,
        };
        let mut iovecs = [empty; KEPT];
        let count = arenas.len().min(KEPT);

        for (iovec, &(base, len)) in iovecs.iter_mut().zip(arenas) {
            *iovec = libc::iovec {
                iov_base: std::ptr::with_exposed_provenance_mut(base),
                iov_len: len,
            };
        }

        // SAFETY: the engine registers only the arenas that `Wave::ready` named, which their owner keeps where they
        // are, with their lengths, until the backend drops. Dropping it closes the ring, which unregisters them.
        unsafe { self.ring.submitter().register_buffers(&iovecs[..count]) }
    }
}

/// The backend for kernels that block io_uring. Each wave tells the kernel about every range with `posix_fadvise`,
/// which starts them all at once, then reads each range in order, through the cold handle and its cache.
#[derive(Default)]
pub(crate) struct FadviseWave(InOrder);

impl Wave for FadviseWave {
    fn page(&self) -> usize {
        PAGE
    }

    fn submit(&mut self, lane: usize, file: &File, ranges: &[Range], _arena: Memory) -> Result<(), Error> {
        ranges
            .iter()
            .filter(|r| r.len > 0)
            .for_each(|r| will_need(file, r.offset, r.len));
        self.0.start(lane, ranges.len());
        Ok(())
    }

    fn next(&mut self, lane: usize, file: &File, ranges: &[Range], arena: Memory) -> Result<Option<usize>, Error> {
        self.0.read(lane, file, ranges, arena)
    }

    fn read_path(&self) -> ReadPath {
        ReadPath {
            backend: "fadvise",
            ..ReadPath::default()
        }
    }
}

/// Asks the kernel to start reading `len` bytes of `file` from `offset` into its cache. A refused hint changes
/// nothing, so errors are ignored.
pub(crate) fn will_need(file: &File, offset: usize, len: usize) {
    // SAFETY: posix_fadvise takes only integers and touches no memory.
    unsafe {
        libc::posix_fadvise(
            file.as_raw_fd(),
            offset as libc::off_t,
            len as libc::off_t,
            libc::POSIX_FADV_WILLNEED,
        )
    };
}

/// Faults in the pages of the mapped bytes `bytes` for reading, as touching each would, and returns once they are in,
/// with true. Before Linux 5.14 the kernel lacks MADV_POPULATE_READ, so it does nothing and returns false, and a
/// warm-up then reads the bytes through the cache after a `posix_fadvise(WILLNEED)` hint instead.
pub(crate) fn populate(bytes: &[u8]) -> Result<bool, Error> {
    populate_with(bytes, libc::MADV_POPULATE_READ, &NO_POPULATE)
}

/// `populate` with the advice `advice`, which must change no memory, as MADV_POPULATE_READ and MADV_WILLNEED don't.
/// A kernel turns down an advice it doesn't know with EINVAL, as Linux before 5.14 does MADV_POPULATE_READ, and
/// `refused` then remembers it, so later calls don't ask again.
fn populate_with(bytes: &[u8], advice: libc::c_int, refused: &AtomicBool) -> Result<bool, Error> {
    if bytes.is_empty() || refused.load(Ordering::Relaxed) {
        return Ok(false);
    }

    let lead = bytes.as_ptr().addr() % page_size();
    let start = bytes.as_ptr().wrapping_sub(lead);
    // SAFETY: the advice changes no memory, as the callers promise, and one the kernel doesn't know does nothing. The
    // range starts on the page of the first byte and ends with the last, so it covers only pages of the mapping that
    // holds `bytes`.
    let rc = unsafe { libc::madvise(start as *mut libc::c_void, lead + bytes.len(), advice) };

    if rc == 0 {
        return Ok(true);
    }

    let e = Error::last_os_error();

    if e.raw_os_error() == Some(libc::EINVAL) {
        refused.store(true, Ordering::Relaxed);
        return Ok(false);
    }

    Err(e)
}

#[cfg(test)]
#[path = "../../tests/unit/io/linux.rs"]
mod tests;
