//! Tests of the `coldivf` command.

use std::path::PathBuf;
use std::process::Command;

use coldivf::{BuildParams, Builder, LeafOrder, Metric, VectorSource};

struct Rows {
    dim: usize,
    data: Vec<f32>,
}

impl VectorSource for Rows {
    fn count(&self) -> usize {
        self.data.len() / self.dim
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn rows(&self, start: usize, rows: usize) -> &[f32] {
        &self.data[start * self.dim..(start + rows) * self.dim]
    }
}

/// A test file in the temporary folder, deleted when dropped.
struct TempFile(PathBuf);

impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// Runs `coldivf` with `args`, and returns its exit code and output.
fn run(args: &[&std::ffi::OsStr]) -> (i32, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_coldivf")).args(args).output().unwrap();
    (out.status.code().unwrap(), String::from_utf8(out.stdout).unwrap())
}

#[test]
fn verify_prints_a_line_for_each_file_and_fails_on_a_damaged_one() {
    let dim = 128;
    let data = (0..300 * dim)
        .map(|i| ((i * 7919) % 1000) as f32 / 500.0 - 1.0)
        .collect();
    let file = |name: &str| TempFile(std::env::temp_dir().join(format!("coldivf-cli-{}-{name}", std::process::id())));
    let (good, bad, missing) = (file("good.coldivf"), file("bad.coldivf"), file("missing.coldivf"));
    let params = BuildParams {
        metric: Metric::L2,
        vectors_per_list: 50,
        parents: 2,
        order: LeafOrder::Chained,
        page_size: 4096,
        threads: 1,
        seed: 3,
    };
    Builder::new(params).build(&Rows { dim, data }, &good.0).unwrap();

    // The last byte belongs to the ids section, which only `verify` reads.
    let mut bytes = std::fs::read(&good.0).unwrap();
    *bytes.last_mut().unwrap() ^= 1;
    std::fs::write(&bad.0, &bytes).unwrap();
    let (verify, good_path, bad_path) = ("verify".as_ref(), good.0.as_os_str(), bad.0.as_os_str());

    assert_eq!(run(&[verify, good_path]), (0, format!("{}: ok\n", good.0.display())));
    // A damaged file comes first, and the good one after it still gets its line.
    let (code, out) = run(&[verify, bad_path, good_path]);
    let lines: Vec<&str> = out.lines().collect();
    assert_eq!((code, lines.len()), (1, 2));
    assert!(lines[0].starts_with(&format!("{}: ", bad.0.display())) && !lines[0].ends_with(": ok"));
    assert_eq!(lines[1], format!("{}: ok", good.0.display()));
    let (code, out) = run(&[verify, missing.0.as_os_str()]);
    assert_eq!((code, out.lines().count()), (1, 1));

    // Anything but `verify` and at least one file is a usage error.
    for args in [&[][..], &[verify], &["check".as_ref(), good_path]] {
        assert_eq!(run(args).0, 2, "{args:?}");
    }
}

#[cfg(unix)]
#[test]
fn a_name_that_isnt_utf_8_gets_its_line() {
    use std::os::unix::ffi::OsStrExt;
    let name = std::ffi::OsStr::from_bytes(b"/nowhere/bad-\xff.coldivf");

    let (code, out) = run(&["verify".as_ref(), name]);
    assert_eq!((code, out.lines().count()), (1, 1));
}
