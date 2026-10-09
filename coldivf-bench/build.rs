//! Records the git commit, whether the workspace's sources differed from it, and the time, when the harnesses are
//! built. The result lines carry them, so each line names the code that made it, not the code in the folder when it
//! ran.

use std::path::Path;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

/// The workspace's files that go into the harnesses.
const SOURCES: [&str; 5] = ["Cargo.toml", "Cargo.lock", ".cargo", "coldivf", "coldivf-bench"];

fn main() {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
    let root = Path::new(&manifest).parent().unwrap_or(Path::new(".")).to_path_buf();
    // `--no-optional-locks` keeps `git status` from rewriting the index, so a build never takes git's lock from someone
    // else using the repository.
    let git = |args: &[&str]| {
        let out = Command::new("git")
            .arg("-C")
            .arg(&root)
            .arg("--no-optional-locks")
            .args(args)
            .output()
            .ok()?;
        out.status
            .success()
            .then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
    };
    let commit = git(&["rev-parse", "--short", "HEAD"]).unwrap_or_default();
    let status = git(&[&["status", "--porcelain", "--untracked-files=all", "--"][..], &SOURCES].concat());
    // A tree that git can't read counts as changed.
    let dirty = status.is_none_or(|s| !s.is_empty());
    let now = SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs());
    println!("cargo:rustc-env=COLDIVF_COMMIT={commit}");
    println!("cargo:rustc-env=COLDIVF_DIRTY={dirty}");
    println!("cargo:rustc-env=COLDIVF_BUILT={now}");

    // Any change to the sources, a new commit or a switch of branch runs this again.
    for source in SOURCES {
        println!("cargo:rerun-if-changed={}", root.join(source).display());
    }

    if let Some(dir) = git(&["rev-parse", "--absolute-git-dir"]) {
        let branch = git(&["symbolic-ref", "-q", "HEAD"]);

        for name in ["HEAD".to_string(), "packed-refs".to_string()]
            .into_iter()
            .chain(branch)
        {
            let path = Path::new(&dir).join(name);

            if path.exists() {
                println!("cargo:rerun-if-changed={}", path.display());
            }
        }
    }
}
