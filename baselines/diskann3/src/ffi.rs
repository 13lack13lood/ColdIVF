//! The C interface that the Python adapter calls through ctypes. Every function catches panics, and a call that fails
//! leaves its message for `diskann3_error` on the calling thread.

use std::cell::RefCell;
use std::ffi::{CStr, c_char};
use std::io::{Error, ErrorKind};
use std::panic::{AssertUnwindSafe, catch_unwind};

use crate::builder;
use crate::mac_reader::ReadPool;
use crate::search::{Index, Options, Reader, prepare_runtime};

thread_local! {
    static ERROR: RefCell<String> = const { RefCell::new(String::new()) };
}

/// Runs `f`, and returns `fail` with its error or panic message kept for `diskann3_error`.
fn guard<T>(fail: T, f: impl FnOnce() -> Result<T, Error>) -> T {
    let message = match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(value)) => return value,
        Ok(Err(e)) => e.to_string(),
        Err(panic) => match (panic.downcast_ref::<&str>(), panic.downcast_ref::<String>()) {
            (Some(s), _) => format!("panic: {s}"),
            (_, Some(s)) => format!("panic: {s}"),
            _ => "panic".to_string(),
        },
    };
    ERROR.with(|e| *e.borrow_mut() = message);
    fail
}

fn invalid(msg: &str) -> Error {
    Error::new(ErrorKind::InvalidInput, msg.to_string())
}

/// What an open read and how long it took, filled by `diskann3_open`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct OpenInfo {
    pub points: u64,
    pub dim: u32,
    pub reader: u32,
    pub pq_us: f64,
    pub reader_us: f64,
    pub searcher_us: f64,
    pub bytes: u64,
}

/// What a search did, filled by `diskann3_search`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct SearchInfo {
    pub hops: u32,
    pub ios: u32,
    pub cmps: u32,
    pub results: u32,
    pub reads: u64,
    pub read_calls: u64,
    pub bytes_requested: u64,
    pub reader_ns: u64,
    pub io_us: f64,
    pub pq_us: f64,
    pub total_us: f64,
}

/// The version of the DiskANN crates, as a C string that lives as long as the library.
#[unsafe(no_mangle)]
pub extern "C" fn diskann3_version() -> *const c_char {
    c"0.60.0".as_ptr()
}

/// Starts read threads until the process's pool runs at least `threads`, and returns how many run, or -1.
#[unsafe(no_mangle)]
pub extern "C" fn diskann3_reserve(threads: u32) -> i64 {
    guard(-1, || Ok(ReadPool::global().reserve(threads as usize)? as i64))
}

/// Makes the runtime that the next open takes, so the open doesn't make one. Returns 0, or -1.
#[unsafe(no_mangle)]
pub extern "C" fn diskann3_prepare() -> i32 {
    guard(-1, || prepare_runtime().map(|()| 0))
}

/// Opens the index whose files start with `prefix` and returns it, or null.
///
/// `reader` is 0 for the pool reader, 1 for the advise reader and 2 for DiskANN's stock reader. `metric` is the name
/// the index was built with. `cache_nodes` nodes nearest the entry point are kept in memory. `info`, when not null,
/// gets the open's sizes and times.
///
/// # Safety
///
/// `prefix` and `metric` must be valid C strings, and `info` null or valid for one write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn diskann3_open(
    prefix: *const c_char,
    reader: u32,
    metric: *const c_char,
    threads: u32,
    cache_nodes: u64,
    info: *mut OpenInfo,
) -> *mut Index {
    guard(std::ptr::null_mut(), || {
        if prefix.is_null() || metric.is_null() {
            return Err(invalid("the prefix and the metric must not be null"));
        }

        // SAFETY: the caller promises that both are valid C strings.
        let (prefix, metric) = unsafe { (CStr::from_ptr(prefix), CStr::from_ptr(metric)) };
        let prefix = prefix.to_str().map_err(|_| invalid("the prefix isn't valid UTF-8"))?;
        let metric = builder::metric(metric.to_str().map_err(|_| invalid("the metric isn't valid UTF-8"))?)?;
        let reader = Reader::from_code(reader).ok_or_else(|| invalid("the reader must be 0, 1 or 2"))?;
        let options = Options {
            reader,
            metric,
            threads: threads as usize,
            cache_nodes: cache_nodes as usize,
        };
        let index = Index::open(prefix, &options, ReadPool::global())?;

        if !info.is_null() {
            let t = index.opened;
            let opened = OpenInfo {
                points: index.points() as u64,
                dim: index.dim() as u32,
                reader: reader as u32,
                pq_us: t.pq_us,
                reader_us: t.reader_us,
                searcher_us: t.searcher_us,
                bytes: t.bytes,
            };
            // SAFETY: the caller promises that a non-null `info` is valid for one write.
            unsafe { info.write(opened) };
        }

        Ok(Box::into_raw(Box::new(index)))
    })
}

/// Searches for the `k` nearest vectors to the `dim` values at `query`. It writes `k` ids to `ids`, with -1 past the
/// ids found, their distances to `distances` when it isn't null, and what the search did to `info` when it isn't
/// null. Returns how many ids it found, or -1.
///
/// # Safety
///
/// `index` must come from `diskann3_open` and not be closed. `query` must hold `dim` floats, `ids` room for `k` ids,
/// and `distances` and `info`, when not null, room for `k` floats and one `SearchInfo`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn diskann3_search(
    index: *const Index,
    query: *const f32,
    dim: u32,
    k: u32,
    list: u32,
    beam: u32,
    ids: *mut i64,
    distances: *mut f32,
    info: *mut SearchInfo,
) -> i64 {
    guard(-1, || {
        if index.is_null() || query.is_null() || ids.is_null() {
            return Err(invalid("the index, the query and the ids must not be null"));
        }

        // SAFETY: the caller promises that `index` is open and that `query` holds `dim` floats.
        let (index, query) = unsafe { (&*index, std::slice::from_raw_parts(query, dim as usize)) };
        let found = index.search(query, k as usize, list as usize, beam as usize)?;
        // SAFETY: the caller promises room for `k` ids.
        let out = unsafe { std::slice::from_raw_parts_mut(ids, k as usize) };
        out.fill(-1);

        for (slot, id) in out.iter_mut().zip(&found.ids) {
            *slot = *id as i64;
        }

        if !distances.is_null() {
            // SAFETY: the caller promises room for `k` floats when `distances` isn't null.
            let out = unsafe { std::slice::from_raw_parts_mut(distances, k as usize) };
            out.fill(f32::INFINITY);
            out.iter_mut().zip(&found.distances).for_each(|(slot, d)| *slot = *d);
        }

        if !info.is_null() {
            let s = found.stats;
            let searched = SearchInfo {
                hops: s.hops,
                ios: s.ios,
                cmps: s.cmps,
                results: s.results,
                reads: s.reads.reads,
                read_calls: s.reads.calls,
                bytes_requested: s.reads.bytes,
                reader_ns: s.reads.nanos,
                io_us: s.io_us,
                pq_us: s.pq_us,
                total_us: s.total_us,
            };
            // SAFETY: the caller promises that a non-null `info` is valid for one write.
            unsafe { info.write(searched) };
        }

        Ok(found.ids.len() as i64)
    })
}

/// Searches `count` queries of `dim` values each, stored row after row at `queries`, on `threads` threads. It writes
/// `k` ids per query to `ids`, with -1 past the ids found. Returns 0, or -1.
///
/// # Safety
///
/// `index` must come from `diskann3_open` and not be closed. `queries` must hold `count` times `dim` floats, and
/// `ids` room for `count` times `k` ids.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn diskann3_search_batch(
    index: *const Index,
    queries: *const f32,
    count: u64,
    dim: u32,
    k: u32,
    list: u32,
    beam: u32,
    threads: u32,
    ids: *mut i64,
) -> i32 {
    guard(-1, || {
        if index.is_null() || queries.is_null() || ids.is_null() {
            return Err(invalid("the index, the queries and the ids must not be null"));
        }

        let values = count as usize * dim as usize;
        // SAFETY: the caller promises that `index` is open and that `queries` holds `count` rows of `dim` floats.
        let (index, queries) = unsafe { (&*index, std::slice::from_raw_parts(queries, values)) };

        if dim as usize != index.dim() {
            return Err(invalid("the queries' width isn't the index's"));
        }

        let found = index.search_batch(queries, k as usize, list as usize, beam as usize, threads as usize)?;
        // SAFETY: the caller promises room for `count` rows of `k` ids.
        unsafe { std::slice::from_raw_parts_mut(ids, found.len()) }.copy_from_slice(&found);
        Ok(0)
    })
}

/// Turns the pool reader's prefill on, for `on` other than 0, or off. While it is on, every read of `index` first asks
/// for its sectors with `F_RDADVISE`, which leaves them in the page cache. Returns 1, or 0 for a reader without the
/// switch, or -1.
///
/// # Safety
///
/// `index` must be null or come from `diskann3_open` and not be closed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn diskann3_prefill(index: *const Index, on: u32) -> i32 {
    guard(-1, || {
        if index.is_null() {
            return Err(invalid("the index must not be null"));
        }

        // SAFETY: the caller promises that a non-null `index` is open.
        Ok(i32::from(unsafe { &*index }.set_prefill(on != 0)))
    })
}

/// Closes an index from `diskann3_open`. Null does nothing.
///
/// # Safety
///
/// `index` must be null or come from `diskann3_open`, and it must not be used again.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn diskann3_close(index: *mut Index) {
    if !index.is_null() {
        guard((), || {
            // SAFETY: the caller promises that `index` came from `diskann3_open`, which made it with Box::into_raw.
            drop(unsafe { Box::from_raw(index) });
            Ok(())
        });
    }
}

/// Copies the calling thread's last error message into `buf` as a C string, cut to `len` bytes with its terminator,
/// and returns the message's full length.
///
/// # Safety
///
/// `buf` must be null or valid for `len` bytes of writes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn diskann3_error(buf: *mut c_char, len: usize) -> usize {
    ERROR.with(|e| {
        let message = e.borrow();

        if !buf.is_null() && len > 0 {
            let n = message.len().min(len - 1);
            // SAFETY: the caller promises `len` bytes at `buf`, and n + 1 <= len.
            unsafe {
                std::ptr::copy_nonoverlapping(message.as_ptr(), buf as *mut u8, n);
                *buf.add(n) = 0;
            }
        }

        message.len()
    })
}

#[cfg(test)]
#[path = "../tests/unit/ffi.rs"]
mod tests;
