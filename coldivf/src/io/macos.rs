//! The macOS backend. The index's cold handle has `F_NOCACHE`, so its reads skip the operating system's cache. A wave
//! first tells the kernel about every range with `F_RDADVISE`, which starts them all at once, then reads each range in
//! order.

use std::ffi::c_void;
use std::fs::File;
use std::io::Error;
use std::os::fd::AsRawFd;
use std::sync::Arc;

use super::{InOrder, Memory, Range, Warmth, Wave};
use crate::trace::ReadPath;

/// The page size of Apple silicon, which the reads round out to.
const PAGE: usize = 16384;

/// Turns off caching for `file`, the index's cold handle.
pub(crate) fn uncache(file: &File) -> Result<(), Error> {
    // SAFETY: F_NOCACHE takes an integer and touches no memory.
    if unsafe { libc::fcntl(file.as_raw_fd(), libc::F_NOCACHE, 1) } == -1 {
        return Err(Error::last_os_error());
    }

    Ok(())
}

/// Whether the file system takes `F_RDADVISE` hints, tried once on the first page of `file`. FAT32 (File Allocation
/// Table) and some network drives don't, so there the caller falls back to the thread pool.
pub(crate) fn takes_hints(file: &File) -> Result<bool, Error> {
    match advise(file, 0, PAGE) {
        Ok(()) => Ok(true),
        Err(e) if refused(&e) => Ok(false),
        Err(e) => Err(e),
    }
}

/// Whether `F_RDADVISE` failed because the file system doesn't support it.
pub(crate) fn refused(e: &Error) -> bool {
    matches!(e.raw_os_error(), Some(libc::ENOTTY | libc::ENOTSUP | libc::EINVAL))
}

#[derive(Default)]
pub(crate) struct RdAdviseWave(InOrder);

impl Wave for RdAdviseWave {
    fn page(&self) -> usize {
        PAGE
    }

    fn submit(&mut self, lane: usize, file: &File, ranges: &[Range], _arena: Memory) -> Result<(), Error> {
        self.0.start(lane, 0);
        ranges
            .iter()
            .filter(|r| r.len > 0)
            .try_for_each(|r| advise(file, r.offset, r.len))?;
        self.0.start(lane, ranges.len());
        Ok(())
    }

    fn next(&mut self, lane: usize, file: &File, ranges: &[Range], arena: Memory) -> Result<Option<usize>, Error> {
        self.0.read(lane, file, ranges, arena)
    }

    fn read_path(&self) -> ReadPath {
        ReadPath {
            backend: "rdadvise",
            ..ReadPath::default()
        }
    }
}

/// Asks the kernel to start reading `len` bytes of `file` from `offset`. A range that runs past the end of the file
/// gets `EFBIG`, which is harmless.
pub(crate) fn advise(file: &File, offset: usize, len: usize) -> Result<(), Error> {
    let ra = libc::radvisory {
        ra_offset: offset as libc::off_t,
        ra_count: len.min(i32::MAX as usize) as libc::c_int,
    };
    // SAFETY: `ra` outlives the call, and F_RDADVISE only reads it.
    let failed = unsafe { libc::fcntl(file.as_raw_fd(), libc::F_RDADVISE, &ra as *const libc::radvisory) } == -1;

    if failed {
        let e = Error::last_os_error();
        if e.raw_os_error() != Some(libc::EFBIG) {
            return Err(e);
        }
    }

    Ok(())
}

/// Asks the drive to write `file`'s data and its cache to the medium, which survives a power loss, unlike a plain
/// sync. Returns false when the file system refuses it, eg. some network drives, so the caller syncs as usual.
pub(crate) fn full_sync(file: &File) -> Result<bool, Error> {
    // SAFETY: F_FULLFSYNC takes no argument and touches no memory.
    if unsafe { libc::fcntl(file.as_raw_fd(), libc::F_FULLFSYNC) } == -1 {
        let e = Error::last_os_error();
        return match e.raw_os_error() {
            Some(libc::ENOTSUP | libc::EINVAL | libc::ENOTTY) => Ok(false),
            _ => Err(e),
        };
    }

    Ok(true)
}

/// Runs the calling thread at utility quality of service (QoS), which macOS schedules after interactive work.
pub(crate) fn utility() {
    // SAFETY: the call changes only the calling thread's QoS class and touches no memory.
    unsafe { libc::pthread_set_qos_class_self_np(libc::qos_class_t::QOS_CLASS_UTILITY, 0) };
}

/// The memory-pressure levels that clear the warm bit, from libdispatch's source.h.
const WARN: usize = 0x02;
const CRITICAL: usize = 0x04;

// SAFETY: these match libdispatch's declarations in <dispatch/queue.h>, <dispatch/source.h> and <dispatch/object.h>,
// with intptr_t as isize, uintptr_t as usize and every dispatch object as a pointer. libdispatch is part of macOS's
// libSystem, which every program links, and the libc crate doesn't declare it. The source type is really an opaque
// struct, declared here as a byte, which is sound because only its address is used.
unsafe extern "C" {
    static _dispatch_source_type_memorypressure: u8;
    fn dispatch_get_global_queue(identifier: isize, flags: usize) -> *mut c_void;
    fn dispatch_source_create(kind: *const c_void, handle: usize, mask: usize, queue: *mut c_void) -> *mut c_void;
    fn dispatch_set_context(object: *mut c_void, context: *mut c_void);
    fn dispatch_source_set_event_handler_f(source: *mut c_void, handler: unsafe extern "C" fn(*mut c_void));
    fn dispatch_source_set_cancel_handler_f(source: *mut c_void, handler: unsafe extern "C" fn(*mut c_void));
    fn dispatch_resume(object: *mut c_void);
    fn dispatch_source_cancel(source: *mut c_void);
    fn dispatch_release(object: *mut c_void);
}

/// A libdispatch source that clears a warm bit on a memory-pressure warning. Its context is a boxed `Arc<Warmth>`,
/// which the source's cancel handler frees.
pub(crate) struct PressureWatch {
    source: *mut c_void,
}

// SAFETY: libdispatch objects may be used and released from any thread.
unsafe impl Send for PressureWatch {}
// SAFETY: as for Send. The watch has no methods that touch the source while shared.
unsafe impl Sync for PressureWatch {}

impl PressureWatch {
    /// Starts watching, or returns `None` if libdispatch refuses the source.
    pub(crate) fn new(warmth: Arc<Warmth>) -> Option<PressureWatch> {
        // SAFETY: the source type is libdispatch's own static, of which only the address is read, and the queue is a
        // global one, which is never freed.
        let source = unsafe {
            let kind = &raw const _dispatch_source_type_memorypressure as *const c_void;
            let queue = dispatch_get_global_queue(libc::qos_class_t::QOS_CLASS_UTILITY as isize, 0);
            dispatch_source_create(kind, 0, WARN | CRITICAL, queue)
        };

        if source.is_null() {
            return None;
        }

        let context = Box::into_raw(Box::new(warmth)) as *mut c_void;
        // SAFETY: the source is new and not yet resumed, so no handler runs before its context is set. The context
        // lives until the cancel handler frees it, which runs only after the source is cancelled.
        unsafe {
            dispatch_set_context(source, context);
            dispatch_source_set_event_handler_f(source, on_pressure);
            dispatch_source_set_cancel_handler_f(source, on_cancel);
            dispatch_resume(source);
        }

        Some(PressureWatch { source })
    }
}

impl Drop for PressureWatch {
    fn drop(&mut self) {
        // SAFETY: the source is ours and resumed. Cancelling it runs `on_cancel` once, later, on its queue.
        unsafe {
            dispatch_source_cancel(self.source);
            dispatch_release(self.source);
        }
    }
}

/// Clears the warm bit in `context`, a boxed `Arc<Warmth>`.
///
/// # Safety
///
/// `context` must point to a live `Arc<Warmth>`.
pub(crate) unsafe extern "C" fn on_pressure(context: *mut c_void) {
    // SAFETY: the caller promises that `context` points to a live `Arc<Warmth>`. libdispatch passes the one set in
    // `PressureWatch::new`, which lives until `on_cancel`.
    let warmth = unsafe { &*(context as *const Arc<Warmth>) };
    warmth.set(false);
}

/// Frees `context` once the source is cancelled.
///
/// # Safety
///
/// `context` must come from `Box::into_raw` of a `Box<Arc<Warmth>>`, and this must run once for it, after its last
/// event handler.
unsafe extern "C" fn on_cancel(context: *mut c_void) {
    // SAFETY: the caller promises where `context` came from. libdispatch runs this once, after the last event handler.
    drop(unsafe { Box::from_raw(context as *mut Arc<Warmth>) });
}
