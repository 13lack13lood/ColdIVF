//! The process's own counters from macOS's `proc_pid_rusage`, for the build report.

/// A process's counters at one moment. Times are in nanoseconds and sizes in bytes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Usage {
    pub cpu_ns: u64,
    pub bytes_read: u64,
    pub bytes_written: u64,
    /// The physical footprint, the memory macOS charges the process, now and at its highest so far.
    pub footprint: u64,
    pub peak_footprint: u64,
}

impl Usage {
    /// The counters now, or all zero if they can't be read.
    pub fn now() -> Usage {
        // rusage_info_v6: 16 bytes of uuid, then 64-bit counters. The indexes below follow <sys/resource.h>.
        let mut buf = [0u64; 2 + 56];
        // SAFETY: flavor 6 writes one rusage_info_v6, which is 2 + 56 u64s long, into `buf`.
        let rc = unsafe { proc_pid_rusage(std::process::id() as i32, 6, buf.as_mut_ptr()) };

        if rc != 0 {
            return Usage::default();
        }

        let at = |i: usize| buf[2 + i];
        Usage {
            cpu_ns: ticks_to_ns(at(0) + at(1)),
            bytes_read: at(16),
            bytes_written: at(17),
            footprint: at(7),
            peak_footprint: at(28),
        }
    }
}

// SAFETY: these match <libproc.h> and <mach/mach_time.h>. proc_pid_rusage's rusage_info_t is taken as a pointer to the
// u64 counters it fills, and mach_timebase_info_t as a pointer to its two u32s, the numerator and the denominator.
unsafe extern "C" {
    fn proc_pid_rusage(pid: i32, flavor: i32, buffer: *mut u64) -> i32;
    fn mach_timebase_info(info: *mut [u32; 2]) -> i32;
}

/// Mach time units to nanoseconds. Apple silicon's units are 125/3 ns.
pub fn ticks_to_ns(ticks: u64) -> u64 {
    let mut base = [0u32; 2];
    // SAFETY: mach_timebase_info writes its numerator and denominator, two u32s, into `base`.
    let rc = unsafe { mach_timebase_info(&mut base) };

    if rc != 0 || base[1] == 0 {
        return ticks;
    }

    (ticks as u128 * base[0] as u128 / base[1] as u128) as u64
}

#[cfg(test)]
#[path = "../tests/unit/usage.rs"]
mod tests;
