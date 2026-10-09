//! Tests of `build_index` with saved clusters, on a small dataset in a folder of the build's own.

use std::path::{Path, PathBuf};
use std::process::Command;

use coldivf_bench::data::{cluster_files, write_fbin, write_ibin};

/// The tests' vector dimension.
const DIM: usize = 128;

/// A folder for one test, removed when dropped.
struct Scratch(PathBuf);

impl Scratch {
    /// A new folder that holds the dataset `tiny-euclidean`, 2400 vectors around 12 centers.
    fn new(name: &str) -> Scratch {
        let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("build-index-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let mut seed = 5u64;
        let mut next = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5
        };
        let centers: Vec<f32> = (0..12 * DIM).map(|_| 8.0 * next()).collect();
        let rows: Vec<f32> = (0..2400 * DIM)
            .map(|i| centers[(i / DIM % 12) * DIM + i % DIM] + next())
            .collect();
        write_fbin(&dir.join("tiny-euclidean.base.fbin"), &rows, DIM).unwrap();
        Scratch(dir)
    }

    fn path(&self, name: &str) -> String {
        self.0.join(name).display().to_string()
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Runs `build_index` on the scratch dataset with `args`, after settings that keep the build small. Returns its exit
/// code, its result line and what it printed.
fn build(dir: &Scratch, args: &[&str]) -> (i32, String, String) {
    let results = dir.0.join("build.jsonl");
    let _ = std::fs::remove_file(&results);
    let small = [
        "--dataset",
        "tiny-euclidean",
        "--vectors-per-list",
        "100",
        "--parents",
        "3",
        "--threads",
        "2",
    ];
    let out = Command::new(env!("CARGO_BIN_EXE_build_index"))
        .env("COLDIVF_DATA", &dir.0)
        .args(small)
        .arg("--results")
        .arg(&results)
        .args(args)
        .output()
        .unwrap();
    let line = std::fs::read_to_string(&results).unwrap_or_default();
    (
        out.status.code().unwrap(),
        line,
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

/// The group count in the header of the index file at `path`.
fn groups_in(path: &str) -> u32 {
    let bytes = std::fs::read(path).unwrap();
    u32::from_le_bytes(bytes[36..40].try_into().unwrap())
}

#[test]
fn saved_clusters_build_the_same_file_again() {
    let dir = Scratch::new("again");
    let stem = dir.path("clusters/tiny");

    for clustering in ["flat", "two-level"] {
        let (first, again) = (dir.path(&format!("{clustering}.coldivf")), dir.path("again.coldivf"));
        let (code, line, err) = build(
            &dir,
            &["--clustering", clustering, "--out", &first, "--export-clusters", &stem],
        );
        assert_eq!(code, 0, "{err}");
        assert!(
            cluster_files(Path::new(&stem)).iter().all(|f| f.exists()),
            "{clustering}"
        );
        for field in ["\"cluster_seconds\":", "\"encode_seconds\":", "\"export_seconds\":"] {
            assert!(line.contains(field), "{clustering}: {line}");
        }
        assert!(line.contains(&format!("\"export_clusters\":\"{stem}\"")), "{line}");

        let (code, line, err) = build(&dir, &["--out", &again, "--clusters", &stem]);
        assert_eq!(code, 0, "{err}");
        assert!(
            line.contains("\"clustering\":\"imported\"") && line.contains("\"export_seconds\":null"),
            "{line}"
        );
        assert!(line.contains(&format!("\"clusters\":\"{stem}\"")), "{line}");
        assert_eq!(
            std::fs::read(&first).unwrap(),
            std::fs::read(&again).unwrap(),
            "{clustering}"
        );
    }
}

#[test]
fn clusters_without_groups_get_the_groups_a_flat_build_finds() {
    let dir = Scratch::new("groups");
    let (stem, flat, found) = (dir.path("tiny"), dir.path("flat.coldivf"), dir.path("found.coldivf"));
    assert_eq!(build(&dir, &["--out", &flat, "--export-clusters", &stem]).0, 0);
    let [_, _, parents, parent_of] = cluster_files(Path::new(&stem));
    std::fs::remove_file(&parents).unwrap();
    std::fs::remove_file(&parent_of).unwrap();

    let (code, line, err) = build(&dir, &["--out", &found, "--clusters", &stem]);
    assert_eq!(code, 0, "{err}");
    assert!(line.contains("\"nparents\":3"), "{line}");
    assert_eq!(std::fs::read(&flat).unwrap(), std::fs::read(&found).unwrap());

    // Saved groups are taken as they are, eg. a single group, rather than found again.
    write_fbin(&parents, &vec![0.0; DIM], DIM).unwrap();
    write_ibin(&parent_of, &[0; 24], 1).unwrap();
    let one = dir.path("one.coldivf");
    let (code, _, err) = build(&dir, &["--out", &one, "--clusters", &stem]);
    assert_eq!((code, groups_in(&one), groups_in(&flat)), (0, 1, 3), "{err}");
}

#[test]
fn bad_cluster_files_fail_with_a_message() {
    let dir = Scratch::new("bad");
    let (stem, out) = (dir.path("tiny"), dir.path("bad.coldivf"));
    let fails = |args: &[&str], message: &str| {
        let (code, _, err) = build(&dir, args);
        assert!(code == 2 && err.contains(message), "{args:?} exited with {code}: {err}");
        assert!(!Path::new(&out).exists());
    };

    fails(&["--out", &out, "--clusters", &stem], "isn't there");
    assert_eq!(
        build(&dir, &["--out", &dir.path("ok.coldivf"), "--export-clusters", &stem]).0,
        0
    );
    fails(
        &["--out", &out, "--clusters", &stem, "--clustering", "flat"],
        "takes no --clustering",
    );
    // The clusters cover 2400 vectors.
    write_fbin(&dir.0.join("tiny-euclidean.base.fbin"), &vec![0.5; 2500 * DIM], DIM).unwrap();
    fails(&["--out", &out, "--clusters", &stem], "for at least 2500 vectors");
    let [_, assign, parents, parent_of] = cluster_files(Path::new(&stem));
    std::fs::remove_file(&parent_of).unwrap();
    fails(
        &["--out", &out, "--clusters", &stem, "--n", "2400"],
        "one of its two group files",
    );
    // A cluster that isn't there fails in the build.
    std::fs::remove_file(&parents).unwrap();
    write_ibin(&assign, &[99; 2400], 1).unwrap();
    fails(&["--out", &out, "--clusters", &stem, "--n", "2400"], "existing cluster");
}
