//! Cold-cache tools. A `Mapped` file can be dropped from the operating system's cache and checked for how much of it is
//! still cached, `make_cold` drops files until they stay out, and `Usage` reads a process's counters, eg. its memory
//! footprint and its time on the performance cores. Dropping and checking work on macOS and Linux, and the counters on
//! macOS. Elsewhere they fail with `Unsupported` or read zeros.

use std::io::{Error, ErrorKind};
use std::ops::{Add, Range};
use std::path::Path;
use std::time::{Duration, Instant};

use memmap2::Mmap;

/// How much of the start of each file `make_cold` watches while it waits.
const HEAD: usize = 1 << 20;

/// A file mapped once, so it can be dropped from the cache and checked many times through the same mapping. On macOS a
/// new mapping of a file can make another process, eg. an endpoint-security agent, read the file's first pages back
/// into the cache up to a few hundred milliseconds later. Dropping and checking pages through a held mapping sets off
/// nothing.
pub struct Mapped {
    /// None for an empty file, which can't be mapped.
    map: Option<Mmap>,
    /// The open file, which Linux drops pages of.
    #[cfg(target_os = "linux")]
    file: std::fs::File,
}

impl Mapped {
    pub fn open(path: &Path) -> Result<Mapped, Error> {
        let file = std::fs::File::open(path)?;

        if file.metadata()?.len() == 0 {
            return Ok(Mapped {
                map: None,
                #[cfg(target_os = "linux")]
                file,
            });
        }

        // SAFETY: `Mapped` never reads the map's bytes. It only passes the map's address range to msync, madvise and
        // mincore.
        let map = unsafe { Mmap::map(&file)? };
        Ok(Mapped {
            map: Some(map),
            #[cfg(target_os = "linux")]
            file,
        })
    }

    pub fn len(&self) -> usize {
        self.map.as_ref().map_or(0, |m| m.len())
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Drops every cached page of the file. On macOS it asks the kernel to invalidate the mapping's pages with `msync`,
    /// which also drops them from the unified buffer cache. On Linux it drops them with `posix_fadvise(DONTNEED)`,
    /// which keeps mapped pages and pages not yet written out. So it first drops this process's own mapping of them
    /// with `madvise` and writes out the file's changes. A page that another process maps stays cached.
    pub fn evict(&self) -> Result<(), Error> {
        self.evict_range(0..self.len())
    }

    /// Drops the cached pages of the bytes `range` of the file, the same way. The range is cut to the file and rounded
    /// out to whole pages. On Linux the cache holds a file in folios, blocks of pages of up to 2 MiB, and a drop keeps
    /// a folio that reaches past the range, so up to a folio at each end of the range may stay cached. Where a caller
    /// needs the whole range out, `resident` tells it what stayed.
    pub fn evict_range(&self, range: Range<usize>) -> Result<(), Error> {
        #[cfg(target_os = "macos")]
        {
            let Some(map) = &self.map else {
                return Ok(());
            };
            let page = page_size();
            let (start, end) = (range.start.min(map.len()) / page * page, range.end.min(map.len()));

            if start >= end {
                return Ok(());
            }

            // SAFETY: the range lies in the live mapping and starts on a page.
            let rc = unsafe {
                libc::msync(
                    map[start..].as_ptr() as *mut libc::c_void,
                    end - start,
                    libc::MS_INVALIDATE,
                )
            };

            if rc != 0 {
                return Err(Error::last_os_error());
            }

            Ok(())
        }

        #[cfg(target_os = "linux")]
        {
            use std::os::fd::AsRawFd;

            let Some(map) = &self.map else {
                return Ok(());
            };
            let page = page_size();
            let (start, end) = (range.start.min(map.len()) / page * page, range.end.min(map.len()));

            if start >= end {
                return Ok(());
            }

            // SAFETY: the range lies in the live mapping and starts on a page. On a shared file mapping, MADV_DONTNEED
            // drops only this process's page table entries, so a later read maps the file's same bytes again.
            let rc = unsafe {
                libc::madvise(
                    map[start..].as_ptr() as *mut libc::c_void,
                    end - start,
                    libc::MADV_DONTNEED,
                )
            };

            if rc != 0 {
                return Err(Error::last_os_error());
            }

            self.file.sync_data()?;
            // The kernel keeps a partial page at the end of the range, so the range ends on a page.
            let len = end.div_ceil(page) * page - start;
            // SAFETY: posix_fadvise takes only integers and touches no memory.
            let rc = unsafe {
                libc::posix_fadvise(
                    self.file.as_raw_fd(),
                    start as libc::off_t,
                    len as libc::off_t,
                    libc::POSIX_FADV_DONTNEED,
                )
            };

            if rc != 0 {
                return Err(Error::from_raw_os_error(rc));
            }

            Ok(())
        }

        #[cfg(not(any(target_os = "macos", target_os = "linux")))]
        {
            let _ = range;
            Err(Error::new(
                ErrorKind::Unsupported,
                "evicting a file is only written for macOS and Linux",
            ))
        }
    }

    /// How many pages of the bytes `range` of the file are in the cache. The range is cut to the file and rounded out
    /// to whole pages. On Linux `mincore` sees the cache only for a file that the process owns or may write, and
    /// otherwise counts only the pages that this process maps.
    pub fn resident(&self, range: Range<usize>) -> Result<usize, Error> {
        #[cfg(unix)]
        {
            let Some(map) = &self.map else {
                return Ok(0);
            };
            let page = page_size();
            let (start, end) = (range.start.min(map.len()) / page * page, range.end.min(map.len()));

            if start >= end {
                return Ok(0);
            }

            let mut vec = vec![0u8; (end - start).div_ceil(page)];
            // SAFETY: the range lies in the live mapping and starts on a page, and `vec` has one byte per page of it,
            // as mincore writes.
            let rc = unsafe {
                libc::mincore(
                    map[start..].as_ptr() as *mut libc::c_void,
                    end - start,
                    vec.as_mut_ptr().cast(),
                )
            };

            if rc != 0 {
                return Err(Error::last_os_error());
            }

            Ok(vec.iter().filter(|&&v| v & 1 == 1).count())
        }

        #[cfg(not(unix))]
        {
            let _ = range;
            Err(Error::new(
                ErrorKind::Unsupported,
                "measuring residency is only written for Unix",
            ))
        }
    }

    /// The share of the file's pages that are in the cache, from 0 to 1.
    pub fn residency(&self) -> Result<f64, Error> {
        let pages = self.len().div_ceil(page_size());

        if pages == 0 {
            return Ok(0.0);
        }

        Ok(self.resident(0..self.len())? as f64 / pages as f64)
    }
}

/// Drops every file of `files` from the cache, then watches the first MiB of each for up to `wait`, since something
/// may read a file's first pages back soon after it was mapped or written. Whenever pages come back it drops the files
/// again, and once none came back for `wait` it checks every page of them. It fails with `TimedOut` if they are still
/// cached after five rounds.
pub fn make_cold(files: &[&Mapped], wait: Duration) -> Result<(), Error> {
    for _ in 0..5 {
        for file in files {
            file.evict()?;
        }

        let start = Instant::now();

        while start.elapsed() < wait && !cached(files, 0..HEAD)? {
            std::thread::sleep(Duration::from_millis(10));
        }

        if !cached(files, 0..usize::MAX)? {
            return Ok(());
        }
    }

    Err(Error::new(
        ErrorKind::TimedOut,
        "the files' pages kept coming back into the cache",
    ))
}

/// Whether any page of the bytes `range` of any file of `files` is cached.
fn cached(files: &[&Mapped], range: Range<usize>) -> Result<bool, Error> {
    for file in files {
        if file.resident(range.clone())? > 0 {
            return Ok(true);
        }
    }

    Ok(false)
}

/// The size of a page of memory, which is what mincore reports on, eg. 16 KiB on Apple silicon.
pub fn page_size() -> usize {
    #[cfg(unix)]
    // SAFETY: sysconf takes no pointers.
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;

    #[cfg(not(unix))]
    let page = 4096;

    page.max(1)
}

/// A process's counters at one moment, from macOS's `proc_pid_rusage`. Times are in nanoseconds.
#[derive(Clone, Copy, Debug, Default)]
pub struct Usage {
    pub user_ns: u64,
    pub system_ns: u64,
    /// The part of the user and system time spent on the performance cores.
    pub p_user_ns: u64,
    pub p_system_ns: u64,
    pub cycles: u64,
    pub instructions: u64,
    pub energy_nj: u64,
    pub pageins: u64,
    pub bytes_read: u64,
    pub bytes_written: u64,
    /// The process's physical footprint, the memory macOS charges it, in bytes, now and at its highest so far.
    pub footprint: u64,
    pub peak_footprint: u64,
}

impl Usage {
    /// The counters now. On other systems they are all zero.
    pub fn now() -> Usage {
        #[cfg(target_os = "macos")]
        {
            // rusage_info_v6: 16 bytes of uuid, then 64-bit counters. The indexes below follow <sys/resource.h>.
            let mut buf = [0u64; 2 + 56];
            // SAFETY: flavor 6 writes one rusage_info_v6, which is 2 + 56 u64s long, into `buf`.
            let rc = unsafe { proc_pid_rusage(std::process::id() as i32, 6, buf.as_mut_ptr()) };

            if rc != 0 {
                return Usage::default();
            }

            let at = |i: usize| buf[2 + i];
            let ns = |ticks: u64| ticks_to_ns(ticks);
            Usage {
                user_ns: ns(at(0)),
                system_ns: ns(at(1)),
                p_user_ns: ns(at(36)),
                p_system_ns: ns(at(37)),
                cycles: at(30),
                instructions: at(29),
                energy_nj: at(40),
                pageins: at(4),
                bytes_read: at(16),
                bytes_written: at(17),
                footprint: at(7),
                peak_footprint: at(28),
            }
        }

        #[cfg(not(target_os = "macos"))]
        Usage::default()
    }

    /// The counters after a 1 ms sleep. macOS adds to the energy count only at a context switch, so a reading right
    /// after busy work would leave that work's energy to a later reading. The sleep is the switch.
    pub fn settled() -> Usage {
        std::thread::sleep(Duration::from_millis(1));
        Usage::now()
    }

    /// The counters from `before` to `self`. The footprint stays a difference too, so it can be below zero.
    pub fn since(&self, before: &Usage) -> Delta {
        Delta {
            cpu_ns: (self.user_ns + self.system_ns).saturating_sub(before.user_ns + before.system_ns),
            p_cpu_ns: (self.p_user_ns + self.p_system_ns).saturating_sub(before.p_user_ns + before.p_system_ns),
            cycles: self.cycles.saturating_sub(before.cycles),
            instructions: self.instructions.saturating_sub(before.instructions),
            energy_nj: self.energy_nj.saturating_sub(before.energy_nj),
            pageins: self.pageins.saturating_sub(before.pageins),
            bytes_read: self.bytes_read.saturating_sub(before.bytes_read),
            footprint: self.footprint as i64 - before.footprint as i64,
        }
    }
}

/// What a process used between two moments.
#[derive(Clone, Copy, Debug, Default)]
pub struct Delta {
    pub cpu_ns: u64,
    pub p_cpu_ns: u64,
    pub cycles: u64,
    pub instructions: u64,
    pub energy_nj: u64,
    pub pageins: u64,
    pub bytes_read: u64,
    pub footprint: i64,
}

impl Delta {
    /// The share of the CPU time on performance cores, or 0 without CPU time.
    pub fn p_share(&self) -> f64 {
        if self.cpu_ns == 0 {
            0.0
        } else {
            self.p_cpu_ns as f64 / self.cpu_ns as f64
        }
    }

    /// Cycles per nanosecond of CPU time, i.e. the clock the work ran at, in GHz.
    pub fn ghz(&self) -> f64 {
        if self.cpu_ns == 0 {
            0.0
        } else {
            self.cycles as f64 / self.cpu_ns as f64
        }
    }
}

/// What a process used over two windows together, eg. before and after a pause that shouldn't count.
impl Add for Delta {
    type Output = Delta;

    fn add(self, other: Delta) -> Delta {
        Delta {
            cpu_ns: self.cpu_ns + other.cpu_ns,
            p_cpu_ns: self.p_cpu_ns + other.p_cpu_ns,
            cycles: self.cycles + other.cycles,
            instructions: self.instructions + other.instructions,
            energy_nj: self.energy_nj + other.energy_nj,
            pageins: self.pageins + other.pageins,
            bytes_read: self.bytes_read + other.bytes_read,
            footprint: self.footprint + other.footprint,
        }
    }
}

// SAFETY: these match <libproc.h> and <mach/mach_time.h>. proc_pid_rusage's rusage_info_t is taken as a pointer to the
// u64 counters it fills, and mach_timebase_info_t as a pointer to its two u32s, the numerator and the denominator.
#[cfg(target_os = "macos")]
unsafe extern "C" {
    fn proc_pid_rusage(pid: i32, flavor: i32, buffer: *mut u64) -> i32;
    fn mach_timebase_info(info: *mut [u32; 2]) -> i32;
}

/// Mach time units to nanoseconds. Apple silicon's units are 125/3 ns, Intel's 1 ns.
#[cfg(target_os = "macos")]
fn ticks_to_ns(ticks: u64) -> u64 {
    let mut base = [0u32; 2];
    // SAFETY: mach_timebase_info writes its numerator and denominator, two u32s, into `base`.
    let rc = unsafe { mach_timebase_info(&mut base) };

    if rc != 0 || base[1] == 0 {
        return ticks;
    }

    (ticks as u128 * base[0] as u128 / base[1] as u128) as u64
}

#[cfg(all(test, target_os = "macos"))]
#[path = "../tests/unit/coldcache.rs"]
mod tests;
