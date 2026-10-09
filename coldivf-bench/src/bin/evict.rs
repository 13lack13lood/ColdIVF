//! Drops files, or every file under folders, from the operating system's cache, and prints how much of each was
//! cached before and after. An index file's change log and lock, `<file>.delta` and `<file>.lock`, go with it when they
//! exist. It maps every file once and does all its work through those mappings. It drops them all, waits for anything
//! that reads their first pages back, as a new mapping can make another process do, and drops those again.
//!
//! evict [--json] PATH...
//!
//! It prints the share of each file's pages that was cached before and after. With `--json` it prints one JSON object
//! instead, with each file's size, pages and cached pages before and after, and their totals. It exits with 1 when a
//! file can't be read or dropped, or when any page is still cached at the end.

use std::path::{Path, PathBuf};
use std::time::Duration;

use coldivf_bench::args::fail;
use coldivf_bench::coldcache::{Mapped, make_cold, page_size};
use coldivf_bench::metrics::Json;

/// What happened to one file. A count is None when it couldn't be read.
struct Report {
    path: PathBuf,
    bytes: Option<usize>,
    pages: Option<usize>,
    before: Option<usize>,
    after: Option<usize>,
    error: Option<String>,
}

fn main() {
    let (mut json, mut paths) = (false, Vec::new());

    for arg in std::env::args().skip(1) {
        match arg.as_str() {
            "--json" => json = true,
            flag if flag.starts_with("--") => fail(&format!("unknown flag {flag}")),
            path => paths.push(PathBuf::from(path)),
        }
    }

    let (reports, error) = evict(&files(&paths));

    for report in &reports {
        if let Some(e) = &report.error {
            eprintln!("evict: {}: {e}", report.path.display());
        }
    }

    if let Some(e) = &error {
        eprintln!("evict: {e}");
    }

    if json {
        println!("{}", summary(&reports, &error).line());
    } else {
        for r in reports.iter().filter(|r| r.error.is_none()) {
            let share = |cached: Option<usize>| {
                let pages = r.pages.unwrap_or(0);
                if pages == 0 {
                    0.0
                } else {
                    cached.unwrap_or(0) as f64 / pages as f64
                }
            };
            println!("{:.4} -> {:.4}  {}", share(r.before), share(r.after), r.path.display());
        }
    }

    if !cold(&reports, &error) {
        std::process::exit(1);
    }
}

/// Every file at or under `paths`, in order, each followed by its change log and lock when they exist, and each once.
fn files(paths: &[PathBuf]) -> Vec<PathBuf> {
    let mut found = Vec::new();
    paths.iter().for_each(|p| walk(p, &mut found));
    let mut out: Vec<PathBuf> = Vec::new();

    for path in found {
        let sides = [".delta", ".lock"].map(|end| PathBuf::from(format!("{}{end}", path.display())));

        for p in std::iter::once(path).chain(sides.into_iter().filter(|s| s.is_file())) {
            if !out.contains(&p) {
                out.push(p);
            }
        }
    }

    out
}

/// Every file at or under `path`, each folder's in name order.
fn walk(path: &Path, out: &mut Vec<PathBuf>) {
    match std::fs::read_dir(path) {
        Ok(entries) => {
            let mut paths: Vec<PathBuf> = entries.flatten().map(|e| e.path()).collect();
            paths.sort();
            paths.iter().for_each(|p| walk(p, out));
        }
        Err(_) => out.push(path.to_path_buf()),
    }
}

/// Drops `paths` from the cache, and reports each file's cached pages before and after. The error is that of the drop
/// itself, eg. when the pages kept coming back.
fn evict(paths: &[PathBuf]) -> (Vec<Report>, Option<String>) {
    let mut reports = Vec::new();
    let mut maps = Vec::new();

    for path in paths {
        let mut report = Report {
            path: path.clone(),
            bytes: None,
            pages: None,
            before: None,
            after: None,
            error: None,
        };
        let opened = Mapped::open(path).and_then(|file| Ok((file.resident(0..file.len())?, file)));

        match opened {
            Ok((before, file)) => {
                (report.bytes, report.pages) = (Some(file.len()), Some(file.len().div_ceil(page_size())));
                report.before = Some(before);
                maps.push(Some(file));
            }
            Err(e) => {
                report.error = Some(e.to_string());
                maps.push(None);
            }
        }

        reports.push(report);
    }

    let live: Vec<&Mapped> = maps.iter().flatten().collect();
    let error = make_cold(&live, Duration::from_millis(500))
        .err()
        .map(|e| e.to_string());

    for (report, file) in reports.iter_mut().zip(&maps) {
        if let Some(file) = file {
            match file.resident(0..file.len()) {
                Ok(after) => report.after = Some(after),
                Err(e) => report.error = Some(e.to_string()),
            }
        }
    }

    (reports, error)
}

/// Whether every file was read and dropped, and none has a page left in the cache.
fn cold(reports: &[Report], error: &Option<String>) -> bool {
    error.is_none() && reports.iter().all(|r| r.error.is_none() && r.after == Some(0))
}

/// The JSON object `--json` prints.
fn summary(reports: &[Report], error: &Option<String>) -> Json {
    let count = |n: Option<usize>| n.map_or(f64::NAN, |n| n as f64);
    let total = |get: fn(&Report) -> Option<usize>| reports.iter().filter_map(get).sum::<usize>() as u64;
    let files: Vec<Json> = reports
        .iter()
        .map(|r| {
            let json = Json::new()
                .str("path", &r.path.display().to_string())
                .num("bytes", count(r.bytes))
                .num("pages", count(r.pages))
                .num("cached_before", count(r.before))
                .num("cached_after", count(r.after));
            match &r.error {
                Some(e) => json.str("error", e),
                None => json.null("error"),
            }
        })
        .collect();
    let json = Json::new()
        .int("page_size", page_size() as u64)
        .objects("files", &files)
        .int("pages", total(|r| r.pages))
        .int("cached_before", total(|r| r.before))
        .int("cached_after", total(|r| r.after))
        .bool("cold", cold(reports, error));
    match error {
        Some(e) => json.str("error", e),
        None => json.null("error"),
    }
}

#[cfg(test)]
#[path = "../../tests/unit/bin/evict.rs"]
mod tests;
