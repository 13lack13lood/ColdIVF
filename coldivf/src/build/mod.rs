//! Building index files.

pub(crate) mod order;

use std::fs::{self, File, OpenOptions};
use std::io::{Error, ErrorKind, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::format;
use crate::search::inmem::InMemory;
use crate::search::invalid;
use crate::{BuildParams, Clusters, VectorSource};

/// Builds the index from `src` and an imported clustering, and writes it to `out`.
pub(crate) fn build_with_clusters(
    params: &BuildParams,
    src: &dyn VectorSource,
    clusters: &Clusters,
    out: &Path,
) -> Result<(), Error> {
    let index = InMemory::build(params, src, clusters)?;

    write(&index, out)
}

/// Writes `index` to `out`. It writes a temporary file next to `out`, with the header last, syncs it, then renames it
/// over `out`. A crash can't leave a half-written index under `out`. The other sections reach the disk before the
/// header does, so a half-written temporary file never opens.
pub(crate) fn write(index: &InMemory, out: &Path) -> Result<(), Error> {
    let (file, tmp) = create_temp(out)?;
    let result = write_parts(&format::encode(index), file)
        .and_then(|()| fs::rename(&tmp, out))
        .and_then(|()| sync_dir(out));

    if result.is_err() {
        let _ = fs::remove_file(&tmp);
    }

    result
}

fn write_parts(encoded: &format::Encoded, mut file: File) -> Result<(), Error> {
    file.set_len(encoded.spans[6].end() as u64)?;

    // The gaps between sections stay zero, and so does the header until every other section is on the disk.
    for (span, bytes) in encoded.spans.iter().zip(&encoded.parts).skip(1) {
        file.seek(SeekFrom::Start(span.offset as u64))?;
        file.write_all(bytes)?;
    }

    file.sync_data()?;
    file.seek(SeekFrom::Start(0))?;
    file.write_all(&encoded.parts[0])?;
    file.sync_all()
}

/// Creates a temporary file next to `out` that no other build uses. Its name is `out`'s with the process id, a
/// counter and `.tmp` added, so two builds of the same index never write into one file.
fn create_temp(out: &Path) -> Result<(File, PathBuf), Error> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let name = out
        .file_name()
        .ok_or_else(|| invalid("the output path has no file name"))?;

    loop {
        let mut tmp = name.to_os_string();
        tmp.push(format!(
            ".{}.{}.tmp",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let path = out.with_file_name(tmp);
        match OpenOptions::new().write(true).create_new(true).open(&path) {
            Ok(file) => return Ok((file, path)),
            Err(e) if e.kind() == ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e),
        }
    }
}

/// Syncs the folder that holds `out`, so the rename survives a crash. Windows has no such call, so there a power loss
/// right after the rename may undo it.
fn sync_dir(out: &Path) -> Result<(), Error> {
    #[cfg(unix)]
    {
        let dir = out
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        File::open(dir)?.sync_all()?;
    }

    #[cfg(not(unix))]
    let _ = out;

    Ok(())
}

#[cfg(test)]
#[path = "../../tests/unit/build/mod.rs"]
mod tests;
