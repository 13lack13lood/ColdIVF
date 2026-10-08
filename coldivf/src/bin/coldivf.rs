//! The `coldivf` command. `coldivf verify FILE...` checks index files and prints one line for each.

use std::ffi::OsString;
use std::io::{Write, stdout};
use std::path::Path;
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<OsString> = std::env::args_os().skip(1).collect();

    let Some((command, files)) = args.split_first() else {
        return usage();
    };

    if command != "verify" || files.is_empty() {
        return usage();
    }

    let mut out = stdout().lock();
    let mut ok = true;

    for file in files.iter().map(Path::new) {
        let result = coldivf::Index::open(file).and_then(|index| index.verify());
        let line = match &result {
            Ok(()) => writeln!(out, "{}: ok", file.display()),
            Err(e) => writeln!(out, "{}: {e}", file.display()),
        };
        // A reader that stops early, eg. `head`, ends the run.
        if line.is_err() {
            return ExitCode::FAILURE;
        }
        ok &= result.is_ok();
    }

    if ok { ExitCode::SUCCESS } else { ExitCode::FAILURE }
}

fn usage() -> ExitCode {
    eprintln!("usage: coldivf verify FILE...");
    ExitCode::from(2)
}
