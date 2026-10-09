//! Unit tests for the eviction tool.

use super::*;

/// A folder of its own in the system temporary folder, which tests may write to.
fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("coldivf-bench/{}-evict-{name}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn report(after: Option<usize>, error: Option<&str>) -> Report {
    Report {
        path: PathBuf::from("a/b.coldivf"),
        bytes: Some(40000),
        pages: Some(3),
        before: Some(2),
        after,
        error: error.map(str::to_string),
    }
}

#[test]
fn an_index_file_brings_its_change_log_and_lock_once() {
    let dir = scratch("files");
    let name = |n: &str| dir.join(n);

    for n in ["a.coldivf", "a.coldivf.delta", "a.coldivf.lock", "b.bin"] {
        std::fs::write(name(n), b"x").unwrap();
    }

    let index = [name("a.coldivf"), name("a.coldivf.delta"), name("a.coldivf.lock")];
    assert_eq!(files(&[name("a.coldivf")]), index);
    // A folder lists its files in name order, and a file named twice comes once.
    let all = [&index[..], &[name("b.bin")]].concat();
    assert_eq!(files(std::slice::from_ref(&dir)), all);
    assert_eq!(files(&[name("a.coldivf"), dir.clone(), name("b.bin")]), all);
    // A file with no change log or lock comes alone, and a missing one still comes, so it is reported.
    assert_eq!(files(&[name("b.bin"), name("c.bin")]), [name("b.bin"), name("c.bin")]);
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn a_page_left_in_the_cache_or_any_error_fails_the_run() {
    assert!(cold(&[report(Some(0), None), report(Some(0), None)], &None));
    assert!(cold(&[], &None));
    assert!(!cold(&[report(Some(0), None), report(Some(1), None)], &None));
    assert!(!cold(&[report(None, Some("no such file"))], &None));
    assert!(!cold(
        &[report(Some(0), None)],
        &Some("the pages kept coming back".to_string())
    ));
}

#[test]
fn the_json_holds_each_file_and_the_totals() {
    let reports = [report(Some(0), None), report(Some(1), Some("bad"))];
    let line = summary(&reports, &None).line();
    let file = |after: u32, error: &str| {
        let counts = r#""bytes":40000,"pages":3,"cached_before":2"#;
        format!(r#"{{"path":"a/b.coldivf",{counts},"cached_after":{after},"error":{error}}}"#)
    };
    let expected = format!(
        r#"{{"page_size":{},"files":[{},{}],"pages":6,"cached_before":4,"cached_after":1,"cold":false,"error":null}}"#,
        page_size(),
        file(0, "null"),
        file(1, r#""bad""#)
    );
    assert_eq!(line, expected);

    // A file that couldn't be read has no counts.
    let mut missing = report(None, Some("gone"));
    (missing.bytes, missing.pages, missing.before) = (None, None, None);
    let line = summary(&[missing], &Some("late".to_string())).line();
    assert!(line.contains(r#""bytes":null,"pages":null,"cached_before":null,"cached_after":null,"error":"gone""#));
    assert!(line.ends_with(r#""pages":0,"cached_before":0,"cached_after":0,"cold":false,"error":"late"}"#));
}

#[cfg(target_os = "macos")]
#[test]
fn eviction_leaves_no_page_of_any_file_cached() {
    let dir = scratch("evict");
    let (index, other) = (dir.join("a.coldivf"), dir.join("b.bin"));
    std::fs::write(&index, vec![7u8; 1 << 20]).unwrap();
    std::fs::write(format!("{}.delta", index.display()), vec![7u8; 1 << 16]).unwrap();
    std::fs::write(&other, vec![7u8; 3 << 20]).unwrap();
    let paths = files(&[index, other, dir.join("missing.bin")]);
    paths.iter().for_each(|p| _ = std::fs::read(p));

    let (reports, error) = evict(&paths);
    assert_eq!((reports.len(), &error), (4, &None));
    let read = &reports[..3];
    assert!(read.iter().all(|r| r.before.unwrap() > 0 && r.after == Some(0)));
    assert_eq!(
        read.iter().map(|r| r.bytes.unwrap()).sum::<usize>(),
        (1 << 20) + (1 << 16) + (3 << 20)
    );
    assert!(reports[3].error.is_some() && !cold(&reports, &error));
    assert!(cold(read, &error));
    std::fs::remove_dir_all(&dir).unwrap();
}
