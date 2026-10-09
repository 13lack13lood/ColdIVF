//! What the unit tests of the library share.

use std::sync::{Mutex, MutexGuard};

/// Held by every test that reads the process's counters, eg. its CPU time, and by every test that keeps several of its
/// threads busy. The counters are the whole process's, so that work would show up in them.
static COUNTERS: Mutex<()> = Mutex::new(());

pub(crate) fn counters() -> MutexGuard<'static, ()> {
    COUNTERS.lock().unwrap_or_else(|e| e.into_inner())
}
