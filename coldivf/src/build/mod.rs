//! Building index files.

pub(crate) mod kmeans;
pub(crate) mod order;

use std::fs::{self, File, OpenOptions};
use std::io::{Error, ErrorKind, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use rayon::{ThreadPool, ThreadPoolBuilder};

use crate::format;
use crate::search::inmem::{InMemory, check_settings};
use crate::search::invalid;
use crate::{BuildParams, Clusters, Metric, VectorSource};
use kmeans::KMeans;

/// Rounds of k-means.
const ROUNDS: usize = 20;

/// Clusters `src` with k-means, then builds the index like `build_with_clusters` and writes it to `out`.
pub(crate) fn build(params: &BuildParams, src: &dyn VectorSource, out: &Path) -> Result<(), Error> {
    let clusters = cluster(params, src, &threads(params.threads)?)?;
    build_with_clusters(params, src, &clusters, out)
}

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

/// Finds about `params.vectors_per_list` vectors per cluster with k-means on all of `src`, then `params.parents` groups
/// with k-means on the cluster centers, at most one group per cluster. Every setting is checked first, so a bad one
/// fails before any clustering.
pub(crate) fn cluster(params: &BuildParams, src: &dyn VectorSource, pool: &ThreadPool) -> Result<Clusters, Error> {
    let (n, dim) = (src.count(), src.dim());
    check_settings(n, dim, params.page_size)?;

    if n == 0 || params.vectors_per_list == 0 || params.parents == 0 {
        return Err(invalid(
            "a build needs vectors, and at least one vector per cluster and one group",
        ));
    }

    let nlist = n.div_ceil(params.vectors_per_list);

    if params.parents.min(nlist) > usize::from(u16::MAX) {
        return Err(invalid("an index holds at most 65535 groups"));
    }

    let leaves = KMeans {
        k: nlist,
        rounds: ROUNDS,
        sample: usize::MAX,
        spherical: false,
        seed: params.seed,
    };
    let centroids = kmeans::train(&leaves, src, params.metric, pool)?;
    let assignment = kmeans::assign(&centroids, src, params.metric, pool)?;
    let groups = KMeans {
        k: params.parents.min(leaves.k),
        seed: params.seed.wrapping_add(1),
        ..leaves
    };
    let centers = Slice { dim, data: &centroids };
    let parents = kmeans::train(&groups, &centers, Metric::L2, pool)?;
    let parent_of = kmeans::assign(&parents, &centers, Metric::L2, pool)?;

    Ok(Clusters {
        dim,
        centroids,
        assignment,
        parents,
        parent_of,
    })
}

/// A pool of `threads` threads. With 0 it takes rayon's default, `RAYON_NUM_THREADS` when that is set and otherwise one
/// thread per logical CPU.
pub(crate) fn threads(threads: usize) -> Result<ThreadPool, Error> {
    ThreadPoolBuilder::new()
        .num_threads(threads)
        .thread_name(|i| format!("coldivf-build-{i}"))
        .build()
        .map_err(Error::other)
}

/// Vectors already in memory, eg. the cluster centers.
pub(crate) struct Slice<'a> {
    pub(crate) dim: usize,
    pub(crate) data: &'a [f32],
}

impl VectorSource for Slice<'_> {
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
