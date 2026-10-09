//! Memory maps of index files.

use std::fs::File;
use std::io::Error;
use std::sync::mpsc::Sender;
use std::thread::JoinHandle;

/// A whole index file, mapped read-only.
pub(crate) struct Map(memmap2::Mmap);

impl Map {
    pub(crate) fn new(file: &File) -> Result<Map, Error> {
        // SAFETY: an index file is never changed in place. A build writes a temporary file and renames it over the old
        // one, so a mapping keeps the bytes it was made from.
        let map = unsafe { memmap2::Mmap::map(file)? };
        Ok(Map(map))
    }

    pub(crate) fn bytes(&self) -> &[u8] {
        &self.0
    }
}

/// A thread that asks the system to read memory a build will read next, eg. a stretch of a mapped file, so the reads
/// run beside the build. On macOS a hint returns once the pages are in, so it runs on its own thread. It does nothing
/// on Windows.
pub(crate) struct ReadAhead {
    asks: Option<Sender<(usize, usize)>>,
    thread: Option<JoinHandle<()>>,
    page: usize,
}

impl ReadAhead {
    pub(crate) fn new() -> ReadAhead {
        #[cfg(unix)]
        {
            let (asks, wanted) = std::sync::mpsc::channel::<(usize, usize)>();
            let thread = std::thread::Builder::new()
                .name("coldivf-read-ahead".into())
                .spawn(move || {
                    for (at, len) in wanted {
                        // SAFETY: MADV_WILLNEED only starts reads and changes no memory. The kernel checks the range,
                        // so even memory that is gone by now can only make the call fail.
                        unsafe { libc::madvise(at as *mut libc::c_void, len, libc::MADV_WILLNEED) };
                    }
                })
                .ok();
            // SAFETY: sysconf only reads a setting.
            let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) }.max(1) as usize;
            ReadAhead {
                asks: thread.as_ref().map(|_| asks),
                thread,
                page,
            }
        }

        #[cfg(not(unix))]
        ReadAhead {
            asks: None,
            thread: None,
            page: 1,
        }
    }

    /// Asks for the pages that hold `values`, and returns at once.
    pub(crate) fn ask(&self, values: &[f32]) {
        let Some(asks) = &self.asks else {
            return;
        };

        if values.is_empty() {
            return;
        }

        let start = values.as_ptr() as usize;
        let first = start / self.page * self.page;
        let _ = asks.send((first, start + std::mem::size_of_val(values) - first));
    }
}

impl Drop for ReadAhead {
    fn drop(&mut self) {
        self.asks = None;

        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}
