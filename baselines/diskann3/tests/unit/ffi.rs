use std::ffi::{CStr, CString};

use super::*;

fn last_error() -> String {
    let mut buf = vec![0 as c_char; 512];
    // SAFETY: `buf` holds 512 bytes.
    unsafe { diskann3_error(buf.as_mut_ptr(), buf.len()) };
    // SAFETY: diskann3_error ends the copy with a terminator inside `buf`.
    unsafe { CStr::from_ptr(buf.as_ptr()) }.to_string_lossy().into_owned()
}

#[test]
fn the_version_names_the_crates() {
    // SAFETY: the version is a static C string.
    let version = unsafe { CStr::from_ptr(diskann3_version()) };
    assert_eq!(version.to_str().unwrap(), crate::DISKANN_VERSION);
}

#[test]
fn a_failed_open_returns_null_and_says_why() {
    let prefix = CString::new("/nonexistent/diskann3/index").unwrap();
    let metric = CString::new("l2").unwrap();
    // SAFETY: both are C strings and `info` is null.
    let index = unsafe { diskann3_open(prefix.as_ptr(), 0, metric.as_ptr(), 1, 0, std::ptr::null_mut()) };
    assert!(index.is_null());
    assert!(!last_error().is_empty());

    let bad = CString::new("hamming").unwrap();
    // SAFETY: as above.
    let index = unsafe { diskann3_open(prefix.as_ptr(), 0, bad.as_ptr(), 1, 0, std::ptr::null_mut()) };
    assert!(index.is_null());
    assert!(last_error().contains("unknown metric"), "{}", last_error());

    // SAFETY: as above.
    let index = unsafe { diskann3_open(prefix.as_ptr(), 7, metric.as_ptr(), 1, 0, std::ptr::null_mut()) };
    assert!(index.is_null());
    assert!(last_error().contains("reader"), "{}", last_error());

    // SAFETY: a null prefix is refused before it is read.
    let index = unsafe { diskann3_open(std::ptr::null(), 0, metric.as_ptr(), 1, 0, std::ptr::null_mut()) };
    assert!(index.is_null());
    assert!(last_error().contains("null"), "{}", last_error());
}

#[test]
fn null_pointers_fail_a_search() {
    let mut ids = [0i64; 4];
    // SAFETY: a null index is refused before anything is read.
    let n = unsafe {
        diskann3_search(
            std::ptr::null(),
            [0f32; 4].as_ptr(),
            4,
            4,
            10,
            2,
            ids.as_mut_ptr(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        )
    };
    assert_eq!(n, -1);
    assert!(last_error().contains("null"));
    // SAFETY: as above.
    let rc = unsafe {
        diskann3_search_batch(
            std::ptr::null(),
            [0f32; 4].as_ptr(),
            1,
            4,
            4,
            10,
            2,
            1,
            ids.as_mut_ptr(),
        )
    };
    assert_eq!(rc, -1);
    // SAFETY: a null index is refused before anything is read.
    assert_eq!(unsafe { diskann3_prefill(std::ptr::null(), 1) }, -1);
    assert!(last_error().contains("null"));
    // SAFETY: closing null does nothing.
    unsafe { diskann3_close(std::ptr::null_mut()) };
}

#[test]
fn the_error_copy_is_cut_to_the_buffer() {
    guard((), || Err(Error::other("0123456789")));
    let mut buf = [1 as c_char; 5];
    // SAFETY: `buf` holds 5 bytes.
    let full = unsafe { diskann3_error(buf.as_mut_ptr(), buf.len()) };
    assert_eq!(full, 10);
    // SAFETY: diskann3_error ends the copy with a terminator inside `buf`.
    assert_eq!(unsafe { CStr::from_ptr(buf.as_ptr()) }.to_str().unwrap(), "0123");
    // SAFETY: a null buffer is only measured.
    assert_eq!(unsafe { diskann3_error(std::ptr::null_mut(), 0) }, 10);
}

#[test]
fn a_panic_becomes_an_error() {
    let value = guard(7, || panic!("boom"));
    assert_eq!(value, 7);
    assert!(last_error().contains("boom"));
}

#[test]
fn the_pool_grows_through_the_interface() {
    assert!(diskann3_reserve(1) >= 1);
    assert_eq!(diskann3_prepare(), 0);
}
