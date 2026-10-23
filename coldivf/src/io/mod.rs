//! Reading index files, through the memory map or through the read waves that fetch what a cold query needs. A wave is
//! one batch of reads. Its backend starts them together, so they overlap on the disk, then hands over each range as it
//! arrives. Besides the kernels, this is the only module with `unsafe` code.

#[cfg(any(target_os = "linux", test))]
pub(crate) mod extents;
#[cfg(target_os = "linux")]
pub(crate) mod linux;
#[cfg(target_os = "macos")]
pub(crate) mod macos;
pub(crate) mod map;
pub(crate) mod pool;
pub(crate) mod portable;
#[cfg(any(target_os = "linux", test))]
pub(crate) mod queue;

use std::fs::File;
use std::io::{Error, ErrorKind};
use std::ops::{Deref, DerefMut};
use std::path::Path;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::trace::ReadPath;

/// A run of bytes in a file, eg. a section or one cluster's blocks.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Span {
    pub(crate) offset: usize,
    pub(crate) len: usize,
}

impl Span {
    pub(crate) fn end(&self) -> usize {
        self.offset + self.len
    }

    pub(crate) fn range(&self) -> std::ops::Range<usize> {
        self.offset..self.end()
    }
}

/// One read of a wave. It starts on a page boundary and ends on one or at the end of the file. It covers the spans from
/// `spans[tag]` up to the next range's `tag`, and its bytes land in `arena[at..at + len]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Range {
    pub(crate) offset: usize,
    pub(crate) len: usize,
    pub(crate) tag: usize,
    pub(crate) at: usize,
}

impl Range {
    pub(crate) fn end(&self) -> usize {
        self.offset + self.len
    }
}

/// Plans the reads for `spans`, which must be sorted by offset and inside a file of `file_len` bytes. It rounds each
/// span out to `page` boundaries, cuts it at the end of the file, and merges ranges that touch or overlap. The ranges
/// go into `ranges` in order, packed one after another in the arena, and where each span's bytes land goes into `at`.
pub(crate) fn coalesce(spans: &[Span], page: usize, file_len: usize, ranges: &mut Vec<Range>, at: &mut Vec<usize>) {
    ranges.clear();
    at.clear();
    let mut next = 0;

    for (tag, s) in spans.iter().enumerate() {
        let start = s.offset / page * page;
        let end = s.end().div_ceil(page).saturating_mul(page).min(file_len);
        let r = match ranges.last_mut() {
            Some(r) if start <= r.end() => {
                let grow = end.saturating_sub(r.end());
                r.len += grow;
                next += grow;
                *r
            }
            _ => {
                let r = Range {
                    offset: start,
                    len: end - start,
                    tag,
                    at: next,
                };
                ranges.push(r);
                next += r.len;
                r
            }
        };
        at.push(r.at + (s.offset - r.offset));
    }
}

/// The most arena bytes that `coalesce` packs spans of these lengths into. Rounding out adds less than two pages to a
/// span, and merging only saves bytes.
pub(crate) fn arena_bound(lens: impl Iterator<Item = usize>, page: usize) -> usize {
    lens.map(|len| len + 2 * page).sum()
}

/// The boundary that every arena starts on. Direct reads on Linux need their memory on 4 KiB boundaries, and elsewhere
/// nothing does.
const ARENA_ALIGN: usize = if cfg!(target_os = "linux") { 4096 } else { 1 };

/// A buffer of zeros that waves read into. Its first byte sits on an `ARENA_ALIGN` boundary, and it never changes size,
/// so its bytes stay where they are for as long as it lives. A wave takes it as `memory`, and the bytes of a range are
/// read with `part` once the wave has handed the range over. Its slice, through `Deref`, is only for an arena that no
/// read can still land in, eg. in tests.
pub(crate) struct Arena {
    bytes: Vec<u8>,
    start: usize,
    len: usize,
}

impl Arena {
    pub(crate) fn new(len: usize) -> Arena {
        Arena::aligned(len, ARENA_ALIGN)
    }

    /// An arena of `len` bytes whose first byte sits on an `align` boundary, where `align` is a power of 2. It holds a
    /// little more underneath, of which only the aligned part shows.
    pub(crate) fn aligned(len: usize, align: usize) -> Arena {
        let bytes = vec![0u8; len + align - 1];
        let start = bytes.as_ptr().addr().wrapping_neg() & (align - 1);
        Arena { bytes, start, len }
    }

    /// The arena as memory for a wave. It makes no slice of the bytes, so the reads of a running wave may land in them.
    pub(crate) fn memory(&mut self) -> Memory {
        let start = NonNull::new(self.bytes.as_mut_ptr().wrapping_add(self.start)).unwrap_or(NonNull::dangling());
        Memory { start, len: self.len }
    }

    /// The `len` bytes from `at`, which no read may still land in, eg. a range that a wave's `next` handed over. It
    /// makes a slice of those bytes alone, so reads may still land in the rest of the arena.
    pub(crate) fn part(&self, at: usize, len: usize) -> &[u8] {
        assert!(
            at.checked_add(len).is_some_and(|end| end <= self.len),
            "a part past the arena"
        );
        // SAFETY: the bytes lie in the arena's allocation, which lives as long as `self`, and the caller reads them
        // only once no read can land in them. The pointer comes from `as_ptr`, which makes no slice of the rest.
        unsafe { std::slice::from_raw_parts(self.bytes.as_ptr().add(self.start + at), len) }
    }
}

impl Deref for Arena {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        &self.bytes[self.start..self.start + self.len]
    }
}

impl DerefMut for Arena {
    fn deref_mut(&mut self) -> &mut [u8] {
        &mut self.bytes[self.start..self.start + self.len]
    }
}

/// The memory that a wave reads into, as where it starts and how long it is. The kernel may write to it at any time
/// while the wave runs, so waves take it as this and never as a slice, which would promise Rust that nothing else
/// changes those bytes while the slice lives. A backend makes a slice of it only over bytes that no read can land in.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Memory {
    start: NonNull<u8>,
    len: usize,
}

// SAFETY: a `Memory` is only an address and a length. Which thread may touch its bytes, and when, is up to the wave
// that reads into it, as `Wave::submit` says.
unsafe impl Send for Memory {}

impl Memory {
    /// The memory of `bytes`, eg. a test's buffer. The slice stays borrowed only for this call, so the caller keeps it
    /// where it is and leaves it to the wave's reads, as `Wave::submit` asks.
    pub(crate) fn of(bytes: &mut [u8]) -> Memory {
        Memory {
            start: NonNull::from(&mut *bytes).cast(),
            len: bytes.len(),
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.len
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The address of the first byte, exposed, so a read that the kernel makes may land in the memory.
    pub(crate) fn addr(&self) -> usize {
        self.start.as_ptr().expose_provenance()
    }

    /// The `len` bytes from `at` as a slice to write, eg. for a read that lands in them.
    ///
    /// # Safety
    ///
    /// The memory must still be the buffer it came from. No other read may land in these bytes, and no other slice of
    /// them may live, while this one does.
    pub(crate) unsafe fn slice_mut<'a>(&self, at: usize, len: usize) -> &'a mut [u8] {
        assert!(
            at.checked_add(len).is_some_and(|end| end <= self.len),
            "a slice past the memory"
        );
        // SAFETY: the bytes lie inside the buffer, which still lives, and the caller holds them alone, as above.
        unsafe { std::slice::from_raw_parts_mut(self.start.as_ptr().add(at), len) }
    }

    /// The `len` bytes from `at` as a slice to read.
    ///
    /// # Safety
    ///
    /// The memory must still be the buffer it came from, and no read may land in these bytes while the slice lives.
    pub(crate) unsafe fn slice<'a>(&self, at: usize, len: usize) -> &'a [u8] {
        assert!(
            at.checked_add(len).is_some_and(|end| end <= self.len),
            "a slice past the memory"
        );
        // SAFETY: the bytes lie inside the buffer, which still lives, and nothing writes them, as above.
        unsafe { std::slice::from_raw_parts(self.start.as_ptr().add(at), len) }
    }
}

/// How many waves a backend keeps going at once, one per lane, so a search can start one wave before it waits for
/// another.
pub(crate) const LANES: usize = 2;

/// One search context's state for a read backend. Its waves read through the index's cold handle, which every context
/// shares.
pub(crate) trait Wave: Send {
    /// The page size that its reads round out to.
    fn page(&self) -> usize;

    /// Readies the backend before its first wave, eg. by setting up a kernel queue, so that work doesn't fall on a
    /// search. Its waves will read `file` into `arenas`, each with at most `most` ranges. Each arena stays where it is,
    /// with its length, until the backend drops, so the backend may register it with the kernel. Waves into other
    /// arenas still work. A later call with more arenas, eg. once the cluster tables size the cluster arena, keeps what
    /// the backend set up for the same file.
    fn ready(&mut self, file: &File, arenas: &[Memory], most: usize) {}

    /// Does setup that may wait, between searches, eg. registering the arenas that `ready` named with the kernel.
    /// `Index::prepare` calls it once a search has gone through the context's reads of the index.
    fn prepare(&mut self) {}

    /// Starts reading `ranges` from `file` into `arena` on lane `lane`, and may return before the reads finish. A lane
    /// holds one wave, so the ranges of its earlier submit that `next` hasn't returned are dropped. A range's read may
    /// land in `arena` until `next` returns it, and a dropped range's until the lane's next submit or the backend's
    /// drop. So the caller keeps the arena where it is, gives each lane memory of its own, and reads a range's bytes
    /// only once `next` has returned it.
    fn submit(&mut self, lane: usize, file: &File, ranges: &[Range], arena: Memory) -> Result<(), Error>;

    /// The same, for ranges that are likely in the operating system's cache, eg. the rows of a warm index. A backend
    /// whose waves skip that cache reads these through it.
    fn submit_cached(&mut self, lane: usize, file: &File, ranges: &[Range], arena: Memory) -> Result<(), Error> {
        self.submit(lane, file, ranges, arena)
    }

    /// Waits until one more range of the lane's last submit is in `arena` and returns its index, or `None` once every
    /// range has been returned. It returns each range once, in any order. `ranges` and `arena` must be that submit's.
    fn next(&mut self, lane: usize, file: &File, ranges: &[Range], arena: Memory) -> Result<Option<usize>, Error>;

    /// What the backend is and how it is set up, for benchmarks.
    fn read_path(&self) -> ReadPath {
        ReadPath::default()
    }

    /// How many ranges have come from the backend's own copies of pages so far, rather than from the file.
    fn hits(&self) -> usize {
        0
    }
}

/// Makes a backend's state for one search context.
pub(crate) type NewWave = fn() -> Box<dyn Wave>;

/// An index file's handle for cold reads, and the backend that reads through it.
pub(crate) struct Cold {
    pub(crate) file: File,
    pub(crate) wave: NewWave,
}

/// Opens `path` for cold reads, with the fastest backend that works on its file system. An index reads its header,
/// maps itself and reads every wave through this one handle, so all of them see the same file even if a rebuild renames
/// another over `path`. A path that isn't a regular file, eg. a device, fails with `InvalidData` before any read hint.
/// On macOS the handle skips the cache, and finding whether the file system takes read hints already starts reading
/// the first page. On Linux the handle reads through the cache, and each search context's waves go through io_uring
/// and a direct handle of its own, which skips it, except the row waves of warm searches.
pub(crate) fn open_cold(path: &Path) -> Result<Cold, Error> {
    let file = File::open(path)?;

    if !file.metadata()?.is_file() {
        return Err(Error::new(ErrorKind::InvalidData, "the path isn't a regular file"));
    }

    #[cfg(target_os = "macos")]
    {
        macos::uncache(&file)?;

        if macos::takes_hints(&file)? {
            return Ok(Cold {
                file,
                wave: || Box::<macos::RdAdviseWave>::default(),
            });
        }
    }

    #[cfg(target_os = "linux")]
    let wave: NewWave = || Box::<linux::UringWave>::default();

    #[cfg(not(target_os = "linux"))]
    let wave: NewWave = || Box::<pool::PoolWave>::default();

    Ok(Cold { file, wave })
}

/// Reads exactly `buf.len()` bytes from `offset` of `file`. On Unix it leaves the file position alone. On Windows it
/// moves the handle's cursor, which nothing here uses.
pub(crate) fn read_at(file: &File, buf: &mut [u8], offset: usize) -> Result<(), Error> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt;
        file.read_exact_at(buf, offset as u64)
    }

    #[cfg(windows)]
    {
        use std::os::windows::fs::FileExt;
        let mut done = 0;

        while done < buf.len() {
            let n = file.seek_read(&mut buf[done..], (offset + done) as u64)?;
            if n == 0 {
                return Err(Error::new(
                    ErrorKind::UnexpectedEof,
                    "the file ended before the read did",
                ));
            }
            done += n;
        }

        Ok(())
    }
}

/// Tells the operating system (OS) that `len` bytes from `offset` of `file` will be read soon, so it can start reading
/// them into its cache. A refused hint changes nothing, so errors are ignored.
pub(crate) fn will_need(file: &File, offset: usize, len: usize) {
    #[cfg(target_os = "macos")]
    let _ = macos::advise(file, offset, len);

    #[cfg(target_os = "linux")]
    linux::will_need(file, offset, len);

    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    let _ = (file, offset, len);
}

/// Reads `buf.len()` bytes from `offset` of the cold handle `file` so that they also land in the operating system's
/// cache. The macOS handle skips the cache, but what `F_RDADVISE` reads ahead lands there anyway, so on macOS the read
/// asks for that first, and a file system that refuses the hint fails the read.
pub(crate) fn read_cached(file: &File, buf: &mut [u8], offset: usize) -> Result<(), Error> {
    #[cfg(target_os = "macos")]
    macos::advise(file, offset, buf.len())?;

    #[cfg(not(target_os = "macos"))]
    will_need(file, offset, buf.len());

    read_at(file, buf, offset)
}

/// Brings `buf.len()` bytes from `offset` of an index file into the operating system's cache, and returns once they are
/// in, eg. for a warm-up. `map` is the whole file, mapped. On Linux it faults the mapped pages in, so a scan in place
/// later doesn't stop for them. Before Linux 5.14, and on other systems, it reads them through the cold handle `file`.
pub(crate) fn warm(file: &File, map: &[u8], buf: &mut [u8], offset: usize) -> Result<(), Error> {
    #[cfg(target_os = "linux")]
    {
        if linux::populate(&map[offset..offset + buf.len()])? {
            return Ok(());
        }
    }

    #[cfg(not(target_os = "linux"))]
    let _ = map;

    read_cached(file, buf, offset)
}

/// The next range of a lane's last submit to hand over, and how many it has.
#[derive(Clone, Copy, Default)]
pub(crate) struct Cursor {
    next: usize,
    count: usize,
}

impl Cursor {
    pub(crate) fn start(&mut self, count: usize) {
        (self.next, self.count) = (0, count);
    }

    /// The next range to hand over, which then counts as handed over.
    pub(crate) fn take(&mut self) -> Option<usize> {
        if self.next == self.count {
            return None;
        }

        self.next += 1;
        Some(self.next - 1)
    }

    fn peek(&self) -> Option<usize> {
        (self.next < self.count).then_some(self.next)
    }
}

/// Whether an index's cluster blocks are known to be in the operating system's cache. It also counts the events that
/// cleared it, so a warm-up that one of them interrupted doesn't set it afterwards. Both live in one number, the count
/// times 2 plus 1 when warm, so a warm-up checks the count and sets the bit in one step.
#[derive(Default)]
pub(crate) struct Warmth {
    state: AtomicU64,
}

impl Warmth {
    pub(crate) fn is_warm(&self) -> bool {
        self.state.load(Ordering::Acquire) & 1 == 1
    }

    pub(crate) fn set(&self, warm: bool) {
        if warm {
            self.state.fetch_or(1, Ordering::AcqRel);
        } else {
            // Adding 2 counts the clear before the bit drops, so a warm-up that finishes in between fails its check.
            self.state.fetch_add(2, Ordering::AcqRel);
            self.state.fetch_and(!1, Ordering::AcqRel);
        }
    }

    /// A mark of the clears so far, for `finish`.
    pub(crate) fn mark(&self) -> u64 {
        self.state.load(Ordering::Acquire) >> 1
    }

    /// Sets the bit after a warm-up that started at `mark`, unless something cleared it since.
    pub(crate) fn finish(&self, mark: u64) {
        let _ = self
            .state
            .compare_exchange(mark << 1, mark << 1 | 1, Ordering::AcqRel, Ordering::Acquire);
    }
}

/// A watch that clears an index's warm bit when the system runs short of memory. Only macOS has one so far, so
/// elsewhere it does nothing.
pub(crate) struct Pressure {
    #[cfg(target_os = "macos")]
    _watch: Option<macos::PressureWatch>,
}

impl Pressure {
    pub(crate) fn watch(warmth: std::sync::Arc<Warmth>) -> Pressure {
        #[cfg(target_os = "macos")]
        return Pressure {
            _watch: macos::PressureWatch::new(warmth),
        };

        #[cfg(not(target_os = "macos"))]
        {
            let _ = warmth;
            Pressure {}
        }
    }
}

/// Lowers the calling thread's priority for background work, eg. a warm-up. macOS runs it at utility quality of
/// service (QoS), Linux gives it a nice value of 10, and elsewhere it does nothing.
pub(crate) fn background() {
    #[cfg(target_os = "macos")]
    macos::utility();

    #[cfg(target_os = "linux")]
    // SAFETY: setpriority takes only integers, and on Linux a `who` of 0 means the calling thread.
    unsafe {
        libc::setpriority(libc::PRIO_PROCESS, 0, 10)
    };
}

/// Writes all of `bytes` at `offset` of `file`, the file at `path`. On Unix it leaves the file position alone.
pub(crate) fn write_all_at(file: &File, bytes: &[u8], offset: u64, path: &Path) -> Result<(), Error> {
    #[cfg(test)]
    watch::see(watch::Op::Write, path)?;

    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt;
        let _ = path;
        file.write_all_at(bytes, offset)
    }

    #[cfg(windows)]
    {
        use std::os::windows::fs::FileExt;
        let _ = path;
        let mut done = 0;

        while done < bytes.len() {
            done += file.seek_write(&bytes[done..], offset + done as u64)?;
        }

        Ok(())
    }
}

/// Makes what was written to `file`, the file at `path`, survive a power loss. On macOS a plain sync only reaches the
/// drive, so it asks the drive to write its cache out too (`F_FULLFSYNC`), and falls back to the plain sync on a file
/// system that refuses that. Elsewhere `sync_all` does both.
pub(crate) fn sync_file(file: &File, path: &Path) -> Result<(), Error> {
    #[cfg(test)]
    watch::see(watch::Op::Sync, path)?;
    let _ = path;
    full_sync(file)
}

fn full_sync(file: &File) -> Result<(), Error> {
    #[cfg(target_os = "macos")]
    {
        if macos::full_sync(file)? {
            return Ok(());
        }
    }

    file.sync_all()
}

/// Syncs the folder `dir`, so the files made in it and taken from it since survive a power loss. Windows has no such
/// call, so there it does nothing.
pub(crate) fn sync_folder(dir: &Path) -> Result<(), Error> {
    #[cfg(test)]
    watch::see(watch::Op::SyncFolder, dir)?;

    #[cfg(unix)]
    full_sync(&File::open(dir)?)?;

    #[cfg(not(unix))]
    let _ = dir;

    Ok(())
}

/// Syncs the folder that holds `path`.
pub(crate) fn sync_parent(path: &Path) -> Result<(), Error> {
    let dir = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    sync_folder(dir)
}

/// Creates the file `path`, open to read and write. One already there fails with `AlreadyExists`.
pub(crate) fn create_new(path: &Path) -> Result<File, Error> {
    #[cfg(test)]
    watch::see(watch::Op::Create, path)?;

    std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .open(path)
}

/// Creates the folder `dir` unless it exists. Returns whether it made it.
pub(crate) fn create_folder(dir: &Path) -> Result<bool, Error> {
    #[cfg(test)]
    watch::see(watch::Op::Create, dir)?;

    match std::fs::create_dir(dir) {
        Ok(()) => Ok(true),
        Err(e) if e.kind() == ErrorKind::AlreadyExists && dir.is_dir() => Ok(false),
        Err(e) => Err(e),
    }
}

/// Deletes the file at `path`. One that is gone already counts as deleted.
pub(crate) fn remove(path: &Path) -> Result<(), Error> {
    #[cfg(test)]
    watch::see(watch::Op::Remove, path)?;

    match std::fs::remove_file(path) {
        Err(e) if e.kind() != ErrorKind::NotFound => Err(e),
        _ => Ok(()),
    }
}

/// A monotonic time in nanoseconds that every process of the machine shares, so one process can tell how long ago
/// another searched. It costs no system call. On Linux it reads the clock that the system keeps to within a few
/// milliseconds. The one that macOS keeps so may be as old as the calling thread's last switch, which a busy search
/// loop leaves far behind, so there it reads the exact one.
pub(crate) fn now_ns() -> u64 {
    #[cfg(unix)]
    {
        #[cfg(target_os = "macos")]
        let clock = libc::CLOCK_UPTIME_RAW;
        #[cfg(any(target_os = "linux", target_os = "android"))]
        let clock = libc::CLOCK_MONOTONIC_COARSE;
        #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "android")))]
        let clock = libc::CLOCK_MONOTONIC;
        let mut t = libc::timespec { tv_sec: 0, tv_nsec: 0 };
        // SAFETY: clock_gettime only writes one timespec, through a pointer to a local that outlives the call.
        if unsafe { libc::clock_gettime(clock, &mut t) } == 0 {
            return t.tv_sec as u64 * 1_000_000_000 + t.tv_nsec as u64;
        }
    }

    static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    START.get_or_init(std::time::Instant::now).elapsed().as_nanos() as u64
}

/// When a search of an index last ran, in `now_ns` time, as one 64-bit word in the small file `name.d/activity` that
/// every process with the index open maps, so a maintenance job of any of them sees every process's searches. Until
/// the file can be made and mapped, eg. before the index has its folder, or where the folder can't be written, the
/// word is this process's alone, and a search tries to map it again at most once a second.
pub(crate) struct Activity {
    dir: std::path::PathBuf,
    file: std::path::PathBuf,
    map: std::sync::OnceLock<memmap2::MmapMut>,
    local: AtomicU64,
    tried: AtomicU64,
}

impl Activity {
    /// The word of the index whose folder is `dir`.
    pub(crate) fn open(dir: &Path) -> Activity {
        let activity = Activity {
            dir: dir.to_path_buf(),
            file: dir.join("activity"),
            map: std::sync::OnceLock::new(),
            local: AtomicU64::new(0),
            tried: AtomicU64::new(0),
        };
        activity.map();
        activity
    }

    /// Maps the word, making its file 8 zero bytes long when the folder has none, unless it is mapped already or there
    /// is no folder yet. Returns whether every process shares the word now. It allocates nothing for paths of up to a
    /// few hundred bytes.
    pub(crate) fn map(&self) -> bool {
        let open = || -> Result<memmap2::MmapMut, Error> {
            let file = std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(false)
                .open(&self.file)?;
            if file.metadata()?.len() < 8 {
                file.set_len(8)?;
            }
            // SAFETY: the file only ever holds the word, which every process changes through atomic operations alone,
            // and no process cuts it short.
            unsafe { memmap2::MmapMut::map_mut(&file) }
        };

        if self.map.get().is_none()
            && self.dir.is_dir()
            && let Ok(map) = open()
            && map.len() >= 8
        {
            let _ = self.map.set(map);
            self.forget_earlier_boot();
        }

        self.map.get().is_some()
    }

    /// Clears the shared word if a search stored it before the machine last started. `now_ns` starts again from zero
    /// then, so the word reads as a time to come, which would hold every job back until the clock passed it, and
    /// `note` would never store a later time. A time that a process stored since is never later than one read after
    /// it, and the 1 s of slack covers clocks that cores read a little apart.
    fn forget_earlier_boot(&self) {
        if let Some(word) = self.mapped() {
            let stored = word.load(Ordering::Relaxed);

            if stored > now_ns().saturating_add(1_000_000_000) {
                let _ = word.compare_exchange(stored, 0, Ordering::Relaxed, Ordering::Relaxed);
            }
        }
    }

    /// The shared word, once it is mapped.
    fn mapped(&self) -> Option<&AtomicU64> {
        self.map.get().map(|map| {
            // SAFETY: the map is at least 8 bytes long and starts on a page, so the word is aligned, and it lives as
            // long as `self`. Every access to it is atomic, here and in every other process.
            unsafe { AtomicU64::from_ptr(map.as_ptr() as *mut u64) }
        })
    }

    /// Notes that a search runs now. It stores the time only when the word is over 1 ms old, so it costs one atomic
    /// load almost always.
    #[inline]
    pub(crate) fn note(&self) {
        let now = now_ns();
        let word = match self.mapped() {
            Some(word) => word,
            None => {
                if now.saturating_sub(self.tried.load(Ordering::Relaxed)) > 1_000_000_000 {
                    self.tried.store(now, Ordering::Relaxed);
                    self.map();
                }
                self.mapped().unwrap_or(&self.local)
            }
        };

        if now.saturating_sub(word.load(Ordering::Relaxed)) > 1_000_000 {
            word.store(now, Ordering::Relaxed);
        }
    }

    /// Stores `at` as when a search last ran, in the shared word once it is mapped, eg. a time to come for a test that
    /// holds a job back as long as it likes.
    #[cfg(test)]
    pub(crate) fn set(&self, at: u64) {
        let word = self.mapped().unwrap_or(&self.local);
        word.store(at, Ordering::Relaxed);
        self.local
            .store(at.min(self.local.load(Ordering::Relaxed)), Ordering::Relaxed);
    }

    /// When a search last ran, in `now_ns` time, in any process that shares the word, or in this one before it did.
    pub(crate) fn last(&self) -> u64 {
        let shared = self.mapped().map_or(0, |w| w.load(Ordering::Relaxed));
        shared.max(self.local.load(Ordering::Relaxed))
    }
}

/// Makes the writes to `file`, the file at `path`, reach the drive before any later write does, which ends a burst of
/// writes as cheaply as the system allows. On macOS that is `F_BARRIERFSYNC`, and elsewhere a plain sync of the data.
pub(crate) fn barrier_sync(file: &File, path: &Path) -> Result<(), Error> {
    #[cfg(test)]
    watch::see(watch::Op::Sync, path)?;
    let _ = path;

    #[cfg(target_os = "macos")]
    {
        if macos::barrier_sync(file)? {
            return Ok(());
        }
    }

    file.sync_data()
}

/// The size of the system's pages of memory, eg. 16 KiB on Apple silicon.
pub(crate) fn page_size() -> usize {
    #[cfg(unix)]
    {
        // SAFETY: sysconf takes an integer and touches no memory.
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        if let Ok(page) = usize::try_from(page)
            && page > 0
        {
            return page;
        }
    }

    4096
}

/// Whether the system holds each byte of `bytes`, part of a file's map, in memory, by the system's page that holds it:
/// byte i is on page (i + `bytes` as an address mod `page_size()`) / `page_size()` of the result. Where it can't tell,
/// eg. on Windows, the result is empty.
pub(crate) fn resident(bytes: &[u8]) -> Vec<bool> {
    #[cfg(unix)]
    {
        let page = page_size();
        let start = bytes.as_ptr() as usize / page * page;
        let len = bytes.as_ptr() as usize + bytes.len() - start;
        let mut pages = vec![0 as libc::c_char; len.div_ceil(page)];
        // SAFETY: the range starts at a page boundary and covers only pages of the map that holds `bytes`, which
        // outlives the call, and `pages` has one byte for each of them, which is all that mincore writes.
        let found = unsafe { libc::mincore(start as *mut libc::c_void, len, pages.as_mut_ptr().cast()) } == 0;
        if found {
            return pages.iter().map(|&p| p & 1 == 1).collect();
        }
    }

    let _ = bytes;
    Vec::new()
}

/// A file mapped once, through which a timing check drops the file's pages from the system's cache before a search
/// that must read cold. A new map of a file can make another process, eg. an endpoint-security agent on macOS, read its
/// first pages back in a little later, and one held map sets off nothing.
#[cfg(test)]
pub(crate) struct Evictor {
    map: map::Map,
    #[cfg(target_os = "linux")]
    file: File,
}

#[cfg(test)]
impl Evictor {
    pub(crate) fn open(path: &Path) -> Result<Evictor, Error> {
        let file = File::open(path)?;
        let map = map::Map::new(&file)?;
        Ok(Evictor {
            map,
            #[cfg(target_os = "linux")]
            file,
        })
    }

    /// Drops every page of the file from the system's cache. On macOS `msync` invalidates the map's pages, which drops
    /// them from the cache, and on Linux `posix_fadvise` drops the pages that no map holds. Elsewhere it does nothing.
    pub(crate) fn evict(&self) -> Result<(), Error> {
        let bytes = self.map.bytes();

        #[cfg(target_os = "macos")]
        {
            // SAFETY: the range is the whole live map, which starts on a page, and msync reads none of its bytes.
            let rc = unsafe { libc::msync(bytes.as_ptr() as *mut libc::c_void, bytes.len(), libc::MS_INVALIDATE) };
            if rc != 0 {
                return Err(Error::last_os_error());
            }
        }

        #[cfg(target_os = "linux")]
        {
            use std::os::fd::AsRawFd;

            // SAFETY: posix_fadvise takes only integers and touches no memory.
            let rc = unsafe { libc::posix_fadvise(self.file.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED) };
            if rc != 0 {
                return Err(Error::from_raw_os_error(rc));
            }
        }

        let _ = bytes;
        Ok(())
    }

    /// The share of the file's pages that the system holds in memory, or 0 where it can't tell.
    pub(crate) fn resident(&self) -> f64 {
        let pages = resident(self.map.bytes());
        pages.iter().filter(|&&p| p).count() as f64 / pages.len().max(1) as f64
    }
}

/// Makes the writes to `file` skip the system's cache of file pages where the system allows it, eg. with `F_NOCACHE`
/// on macOS, so a large write pushes no other file's pages out. Elsewhere it does nothing.
pub(crate) fn uncached(file: &File) {
    #[cfg(target_os = "macos")]
    let _ = macos::uncache(file);

    #[cfg(not(target_os = "macos"))]
    let _ = file;
}

/// Sets aside `len` bytes on the disk for `file` and makes it that long, so a write into it never finds the disk full.
/// A file system that can't set space aside just grows the file.
pub(crate) fn preallocate(file: &File, len: u64) -> Result<(), Error> {
    #[cfg(target_os = "macos")]
    macos::preallocate(file, len);

    #[cfg(target_os = "linux")]
    {
        use std::os::fd::AsRawFd;
        // SAFETY: posix_fallocate takes a descriptor and two integers, and touches no memory.
        let _ = unsafe { libc::posix_fallocate(file.as_raw_fd(), 0, len as libc::off_t) };
    }

    file.set_len(len)
}

/// The bytes free for an unprivileged process on the file system that holds `dir`, or `u64::MAX` where that can't be
/// told.
pub(crate) fn free_bytes(dir: &Path) -> u64 {
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;

        let Ok(path) = std::ffi::CString::new(dir.as_os_str().as_bytes()) else {
            return u64::MAX;
        };
        // SAFETY: statvfs only writes the struct, through a pointer to a zeroed local that outlives the call, and
        // reads the path, a valid C string.
        let mut s: libc::statvfs = unsafe { std::mem::zeroed() };
        if unsafe { libc::statvfs(path.as_ptr(), &mut s) } == 0 {
            return (s.f_bavail as u64).saturating_mul(s.f_frsize as u64);
        }
    }

    let _ = dir;
    u64::MAX
}

/// Marks the end of a step of a commit of the index whose folder holds `path`, so a test can take what a crash right
/// then would leave. It does nothing outside tests.
#[inline]
pub(crate) fn step(name: &'static str, path: &Path) {
    #[cfg(test)]
    let _ = watch::see(watch::Op::Step(name), path);
    let _ = (name, path);
}

/// What tests see of the file operations that keep an index crash-safe. A test sets a hook for a folder of its own, and
/// the hook sees every operation on a path inside it, and may fail it, eg. as a full disk would.
#[cfg(test)]
pub(crate) mod watch {
    use std::io::Error;
    use std::path::{Path, PathBuf};
    use std::sync::{Mutex, PoisonError};

    /// One file operation.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(crate) enum Op {
        Create,
        Write,
        Sync,
        SyncFolder,
        Remove,
        /// The end of a step of a commit.
        Step(&'static str),
    }

    type Hook = Box<dyn FnMut(Op, &Path) -> Result<(), Error> + Send>;

    /// Each hook with its folder, as given and as the file system names it, eg. with macOS's `/private` in front of a
    /// temporary folder, since the index's paths may come either way.
    static HOOKS: Mutex<Vec<([PathBuf; 2], Hook)>> = Mutex::new(Vec::new());

    /// Calls the hook of the folder that holds `path`, if a test set one. The hook must not make file operations of
    /// this module itself.
    pub(crate) fn see(op: Op, path: &Path) -> Result<(), Error> {
        let mut hooks = HOOKS.lock().unwrap_or_else(PoisonError::into_inner);

        match hooks
            .iter_mut()
            .find(|(folder, _)| folder.iter().any(|f| path.starts_with(f)))
        {
            Some((_, hook)) => hook(op, path),
            None => Ok(()),
        }
    }

    /// Sets `hook` for every path inside `folder` until the returned guard drops.
    pub(crate) fn set(folder: &Path, hook: impl FnMut(Op, &Path) -> Result<(), Error> + Send + 'static) -> Guard {
        let mut hooks = HOOKS.lock().unwrap_or_else(PoisonError::into_inner);
        let real = std::fs::canonicalize(folder).unwrap_or_else(|_| folder.to_path_buf());
        hooks.push(([folder.to_path_buf(), real], Box::new(hook)));
        Guard(folder.to_path_buf())
    }

    pub(crate) struct Guard(PathBuf);

    impl Drop for Guard {
        fn drop(&mut self) {
            let mut hooks = HOOKS.lock().unwrap_or_else(PoisonError::into_inner);
            hooks.retain(|(folder, _)| folder[0] != self.0);
        }
    }
}

/// The processor time that the calling thread has used so far, or zero where it can't be measured.
pub(crate) fn thread_cpu() -> std::time::Duration {
    #[cfg(unix)]
    {
        let mut t = libc::timespec { tv_sec: 0, tv_nsec: 0 };
        // SAFETY: clock_gettime only writes one timespec, through a pointer to a local that outlives the call.
        if unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut t) } == 0 {
            return std::time::Duration::new(t.tv_sec as u64, t.tv_nsec as u32);
        }
    }

    std::time::Duration::ZERO
}

/// Hands over the ranges of each lane's submit in order, reading each one when asked for it.
#[derive(Default)]
pub(crate) struct InOrder {
    lanes: [Cursor; LANES],
}

impl InOrder {
    pub(crate) fn start(&mut self, lane: usize, count: usize) {
        self.lanes[lane].start(count);
    }

    /// Reads the lane's next range. If the read fails, the next call tries it again.
    pub(crate) fn read(
        &mut self,
        lane: usize,
        file: &File,
        ranges: &[Range],
        arena: Memory,
    ) -> Result<Option<usize>, Error> {
        let Some(i) = self.lanes[lane].peek() else {
            return Ok(None);
        };
        let r = ranges[i];
        // SAFETY: a backend that reads in order reads a range only here, one at a time, and the caller leaves the
        // range's bytes alone until this hands it over.
        read_at(file, unsafe { arena.slice_mut(r.at, r.len) }, r.offset)?;
        Ok(self.lanes[lane].take())
    }
}

#[cfg(test)]
#[path = "../../tests/unit/io/mod.rs"]
mod tests;
