//! Building index files.

pub(crate) mod kmeans;
pub(crate) mod order;
pub(crate) mod rewrite;
pub(crate) mod stream;
pub(crate) mod twolevel;

use std::fs::{self, File, OpenOptions};
use std::io::{Error, ErrorKind, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use rayon::{ThreadPool, ThreadPoolBuilder};

use crate::format;
use crate::io::map::ReadAhead;
use crate::search::inmem::{InMemory, check_settings};
use crate::search::invalid;
use crate::{BuildParams, Clustering, Clusters, Metric, VectorSource};
use kmeans::KMeans;

/// Rounds of k-means.
pub(crate) const ROUNDS: usize = 20;

/// Clusters `src` with k-means, then builds the index like `build_with_clusters` and writes it to `out`.
pub(crate) fn build(
    params: &BuildParams,
    clustering: Clustering,
    src: &dyn VectorSource,
    out: &Path,
) -> Result<(), Error> {
    let src = &Ahead::new(src);
    let clusters = clusters_of(params, clustering, src, out)?;
    stream::write_streaming(params, src, &clusters, out)
}

/// The clusters that `build` would build `out` from, found the same way.
pub(crate) fn find_clusters(
    params: &BuildParams,
    clustering: Clustering,
    src: &dyn VectorSource,
    out: &Path,
) -> Result<Clusters, Error> {
    clusters_of(params, clustering, &Ahead::new(src), out)
}

/// Clusters `src` as `clustering` says. Two-level clustering writes its copy of the vectors next to `out`.
fn clusters_of(params: &BuildParams, clustering: Clustering, src: &Ahead, out: &Path) -> Result<Clusters, Error> {
    let pool = threads(params.threads)?;

    match clustering {
        Clustering::Flat => cluster(params, src, &pool),
        Clustering::TwoLevel => twolevel::cluster(params, src, out, &pool),
    }
}

/// Builds the index from `src` and an imported clustering, and writes it to `out`. It encodes and writes the vectors a
/// part at a time, so memory stays bounded whatever their number, and the file is the same as an in-memory build's.
pub(crate) fn build_with_clusters(
    params: &BuildParams,
    src: &dyn VectorSource,
    clusters: &Clusters,
    out: &Path,
) -> Result<(), Error> {
    stream::write_streaming(params, &Ahead::new(src), clusters, out)
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
    let (parents, parent_of) = groups(params, &centroids, dim, pool)?;

    Ok(Clusters {
        dim,
        centroids,
        assignment,
        parents,
        parent_of,
    })
}

/// Replaces the groups of `clusters` with the ones `cluster` would find for their centers. Bad clusters or settings
/// fail before any k-means.
pub(crate) fn group(params: &BuildParams, clusters: &mut Clusters) -> Result<(), Error> {
    let dim = clusters.dim;
    let nlist = clusters.centroids.len().checked_div(dim).unwrap_or(0);

    if nlist == 0 || nlist * dim != clusters.centroids.len() || params.parents == 0 {
        return Err(invalid(
            "grouping needs the cluster centers as whole rows, at least one, and at least one group",
        ));
    }

    if params.parents.min(nlist) > usize::from(u16::MAX) {
        return Err(invalid("an index holds at most 65535 groups"));
    }

    let pool = threads(params.threads)?;
    (clusters.parents, clusters.parent_of) = groups(params, &clusters.centroids, dim, &pool)?;
    Ok(())
}

/// Finds `params.parents` groups of the cluster centers `centroids`, at most one per cluster, with k-means seeded one
/// past the build's seed. Returns the group centers and each cluster's group.
fn groups(
    params: &BuildParams,
    centroids: &[f32],
    dim: usize,
    pool: &ThreadPool,
) -> Result<(Vec<f32>, Vec<u32>), Error> {
    let centers = Slice { dim, data: centroids };
    let groups = KMeans {
        k: params.parents.min(centers.count()),
        rounds: ROUNDS,
        sample: usize::MAX,
        spherical: false,
        seed: params.seed.wrapping_add(1),
    };
    let parents = kmeans::train(&groups, &centers, Metric::L2, pool)?;
    let parent_of = kmeans::assign(&parents, &centers, Metric::L2, pool)?;
    Ok((parents, parent_of))
}

/// The bytes of each stretch of a source that `Ahead` asks the system to read ahead.
const AHEAD_BYTES: usize = 256 << 20;

/// A source that asks for each stretch of `stretch` rows to be read as a build reaches the one before, and for the
/// first as it starts, so a mapped file is read ahead of the passes over it. Its rows are the source's own.
struct Ahead<'a> {
    src: &'a dyn VectorSource,
    stretch: usize,
    reads: ReadAhead,
}

impl<'a> Ahead<'a> {
    fn new(src: &'a dyn VectorSource) -> Ahead<'a> {
        let stretch = (AHEAD_BYTES / (4 * src.dim().max(1))).max(1);
        Ahead {
            src,
            stretch,
            reads: ReadAhead::new(),
        }
    }

    fn ask(&self, start: usize) {
        let n = self.src.count();

        if start < n {
            self.reads.ask(self.src.rows(start, self.stretch.min(n - start)));
        }
    }
}

impl VectorSource for Ahead<'_> {
    fn count(&self) -> usize {
        self.src.count()
    }

    fn dim(&self) -> usize {
        self.src.dim()
    }

    fn rows(&self, start: usize, rows: usize) -> &[f32] {
        let w = self.stretch;
        let mut at = start.div_ceil(w) * w;

        if start == 0 {
            self.ask(0);
        }

        // Each stretch that starts in these rows asks for the next.
        while at < start + rows {
            self.ask(at + w);
            at += w;
        }

        self.src.rows(start, rows)
    }
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

/// Writes `index` to `out`. It writes a temporary file next to `out`, with the header last, syncs it, then puts it in
/// place as `root::publish` says: it renames it over `out`, or commits it as a new generation once the index has a
/// root. A crash can't leave a half-written index under `out`. The other sections reach the disk before the header
/// does, so a half-written temporary file never opens.
pub(crate) fn write(index: &InMemory, out: &Path) -> Result<(), Error> {
    let (file, tmp) = create_temp(out)?;
    let result = write_parts(&format::encode(index), file).and_then(|()| crate::root::publish(out, &tmp));

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
    side_file(out, "tmp")
}

/// Creates a file next to `out` for a build's own use, open to read and write, that no other build uses. Its name is
/// `out`'s with the process id, a counter and `.kind` added.
pub(crate) fn side_file(out: &Path, kind: &str) -> Result<(File, PathBuf), Error> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let name = out
        .file_name()
        .ok_or_else(|| invalid("the output path has no file name"))?;

    loop {
        let mut side = name.to_os_string();
        side.push(format!(
            ".{}.{}.{kind}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let path = out.with_file_name(side);
        match OpenOptions::new().read(true).write(true).create_new(true).open(&path) {
            Ok(file) => {
                // A build writes these files once and reads them back at most once. On macOS they bypass the cache,
                // which would otherwise fill with tens of gigabytes and push other programs' memory out to swap. A
                // drive that refuses the setting just caches as usual.
                #[cfg(target_os = "macos")]
                let _ = crate::io::macos::uncache(&file);
                return Ok((file, path));
            }
            Err(e) if e.kind() == ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e),
        }
    }
}

#[cfg(test)]
#[path = "../../tests/unit/build/mod.rs"]
mod tests;
