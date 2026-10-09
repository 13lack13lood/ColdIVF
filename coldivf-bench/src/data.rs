//! The datasets. Vectors come as `.fbin` files and ground truth as `.ibin` files, both a count, a width, then the rows,
//! and both are mapped, so nothing loads until it's read. The ground truth's distances, when there are any, sit next
//! to its ids as `<stem>.gt.dist.fbin`. A saved clustering is four such files with a shared stem.

use std::fs::File;
use std::io::{BufWriter, Error, ErrorKind, Write};
use std::path::{Path, PathBuf};

use coldivf::{Clusters, Metric, VectorSource};
use memmap2::Mmap;

use crate::metrics::{recall, recall_tied};

/// A mapped file of `count` rows of `width` 4-byte values, after an 8-byte header that holds the two as i32.
struct Table {
    map: Mmap,
    count: usize,
    width: usize,
}

impl Table {
    fn open(path: &Path) -> Result<Table, Error> {
        let file = File::open(path)?;
        // SAFETY: datasets are written once and only read after, so nothing changes the bytes under the map.
        let map = unsafe { Mmap::map(&file)? };
        let field = |at: usize| {
            map.get(at..at + 4)
                .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        };
        let (Some(count), Some(width)) = (field(0), field(4)) else {
            return Err(damaged(path, "is shorter than its header"));
        };

        if count < 0 || width <= 0 {
            return Err(damaged(path, "has a negative count or a width that isn't positive"));
        }

        let (count, width) = (count as usize, width as usize);

        if map.len() != 8 + count * width * 4 {
            return Err(damaged(path, "doesn't hold as many rows as its header says"));
        }

        Ok(Table { map, count, width })
    }

    /// Rows `start..start + n`, back to back.
    fn rows<T: Value>(&self, start: usize, n: usize) -> &[T] {
        assert!(start + n <= self.count, "rows past the end of the file");
        let bytes = &self.map[8 + start * self.width * 4..8 + (start + n) * self.width * 4];
        // SAFETY: the map starts on a page boundary and the rows 8 bytes after it, so they are 4-byte aligned. A
        // `Value` is 4 bytes with that alignment, and every bit pattern is a valid one.
        unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast(), bytes.len() / 4) }
    }
}

/// The values a table holds, f32 and i32. Both are 4 bytes with 4-byte alignment, and every bit pattern is a valid
/// value of either, which `Table::rows` relies on.
trait Value: Copy {}

impl Value for f32 {}
impl Value for i32 {}

fn damaged(path: &Path, what: &str) -> Error {
    Error::new(ErrorKind::InvalidData, format!("{} {what}", path.display()))
}

/// A mapped `.fbin` file of f32 vectors.
pub struct Fbin(Table);

impl Fbin {
    pub fn open(path: &Path) -> Result<Fbin, Error> {
        Table::open(path).map(Fbin)
    }

    pub fn len(&self) -> usize {
        self.0.count
    }

    pub fn is_empty(&self) -> bool {
        self.0.count == 0
    }

    pub fn dim(&self) -> usize {
        self.0.width
    }

    pub fn row(&self, i: usize) -> &[f32] {
        self.0.rows(i, 1)
    }

    /// The first `n` vectors, as a source for a build.
    pub fn prefix(&self, n: usize) -> Prefix<'_> {
        Prefix {
            fbin: self,
            n: n.min(self.len()),
        }
    }
}

/// The first rows of an `.fbin` file.
pub struct Prefix<'a> {
    fbin: &'a Fbin,
    n: usize,
}

impl VectorSource for Prefix<'_> {
    fn count(&self) -> usize {
        self.n
    }

    fn dim(&self) -> usize {
        self.fbin.dim()
    }

    fn rows(&self, start: usize, rows: usize) -> &[f32] {
        assert!(start + rows <= self.n, "rows past the end of the prefix");
        self.fbin.0.rows(start, rows)
    }
}

/// A mapped `.ibin` file of i32 rows, eg. each query's nearest neighbors, nearest first.
pub struct Ibin(Table);

impl Ibin {
    pub fn open(path: &Path) -> Result<Ibin, Error> {
        Table::open(path).map(Ibin)
    }

    pub fn len(&self) -> usize {
        self.0.count
    }

    pub fn is_empty(&self) -> bool {
        self.0.count == 0
    }

    pub fn width(&self) -> usize {
        self.0.width
    }

    pub fn row(&self, i: usize) -> &[i32] {
        self.0.rows(i, 1)
    }
}

/// Writes `rows`, `width` to a row, as an `.ibin` file. It writes a temporary file first and renames it, so a crash
/// never leaves half a file.
pub fn write_ibin(path: &Path, rows: &[i32], width: usize) -> Result<(), Error> {
    write_table(path, rows.iter().map(|x| x.to_le_bytes()), rows.len(), width)
}

/// Writes `rows`, `width` to a row, as an `.fbin` file, the same way.
pub fn write_fbin(path: &Path, rows: &[f32], width: usize) -> Result<(), Error> {
    write_table(path, rows.iter().map(|x| x.to_le_bytes()), rows.len(), width)
}

/// Writes the table through a buffer, so it never holds a second copy of the values.
fn write_table(path: &Path, values: impl Iterator<Item = [u8; 4]>, len: usize, width: usize) -> Result<(), Error> {
    if width == 0 || !len.is_multiple_of(width) {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "the values don't fill whole rows of that width",
        ));
    }

    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }

    let tmp = path.with_extension("tmp");
    let mut out = BufWriter::with_capacity(1 << 20, File::create(&tmp)?);
    out.write_all(&((len / width) as i32).to_le_bytes())?;
    out.write_all(&(width as i32).to_le_bytes())?;

    for v in values {
        out.write_all(&v)?;
    }

    out.into_inner().map_err(|e| e.into_error())?.sync_all()?;
    std::fs::rename(&tmp, path)
}

/// The four files of a clustering saved under `stem`, eg. `data/clusters/c10m` for `data/clusters/c10m.assign.ibin`.
/// They are `<stem>.centroids.fbin` with a cluster center per row, `<stem>.assign.ibin` with each vector's cluster,
/// `<stem>.parents.fbin` with a group center per row and `<stem>.parent_of.ibin` with each cluster's group.
pub fn cluster_files(stem: &Path) -> [PathBuf; 4] {
    ["centroids.fbin", "assign.ibin", "parents.fbin", "parent_of.ibin"].map(|end| {
        let mut name = stem.as_os_str().to_owned();
        name.push(".");
        name.push(end);
        PathBuf::from(name)
    })
}

/// Saves `clusters` under `stem`, to the files that `cluster_files` names. Clusters without groups leave out the two
/// group files and remove any old ones.
pub fn save_clusters(stem: &Path, clusters: &Clusters) -> Result<(), Error> {
    let [centroids, assign, parents, parent_of] = cluster_files(stem);
    write_fbin(&centroids, &clusters.centroids, clusters.dim)?;
    write_ids(&assign, &clusters.assignment)?;

    if clusters.parents.is_empty() && clusters.parent_of.is_empty() {
        for old in [parents, parent_of].iter().filter(|p| p.exists()) {
            std::fs::remove_file(old)?;
        }

        return Ok(());
    }

    write_fbin(&parents, &clusters.parents, clusters.dim)?;
    write_ids(&parent_of, &clusters.parent_of)
}

/// Reads the clustering saved under `stem` for the first `n` vectors of a base of `dim` dimensions. The assignment may
/// cover more vectors, eg. a whole base, and then its first `n` count. Without both group files the groups come back
/// empty. A missing file fails with `NotFound`, and a file of the wrong shape or with an id below 0 with
/// `InvalidData`.
pub fn load_clusters(stem: &Path, n: usize, dim: usize) -> Result<Clusters, Error> {
    let [centroids_path, assign_path, parents_path, parent_of_path] = cluster_files(stem);

    if let Some(missing) = [&centroids_path, &assign_path].iter().find(|p| !p.exists()) {
        return Err(Error::new(
            ErrorKind::NotFound,
            format!("{} isn't there", missing.display()),
        ));
    }

    let centers = Fbin::open(&centroids_path)?;
    let assign = Ibin::open(&assign_path)?;

    if centers.dim() != dim || centers.is_empty() {
        return Err(damaged(
            &centroids_path,
            &format!("should hold cluster centers of {dim} values, at least one"),
        ));
    }

    if assign.width() != 1 || assign.len() < n {
        return Err(damaged(
            &assign_path,
            &format!("should hold one cluster per row for at least {n} vectors"),
        ));
    }

    let assignment = ids(&assign_path, assign.0.rows(0, n))?;
    let centroids = centers.0.rows::<f32>(0, centers.len()).to_vec();
    let (parents, parent_of) = match (parents_path.exists(), parent_of_path.exists()) {
        (false, false) => (Vec::new(), Vec::new()),
        (true, true) => {
            let groups = Fbin::open(&parents_path)?;
            let group_of = Ibin::open(&parent_of_path)?;

            if groups.dim() != dim || groups.is_empty() {
                return Err(damaged(
                    &parents_path,
                    &format!("should hold group centers of {dim} values, at least one"),
                ));
            }

            if group_of.width() != 1 || group_of.len() != centers.len() {
                return Err(damaged(
                    &parent_of_path,
                    "should hold one group per row for each cluster",
                ));
            }

            let parents = groups.0.rows::<f32>(0, groups.len()).to_vec();
            (parents, ids(&parent_of_path, group_of.0.rows(0, group_of.len()))?)
        }
        _ => {
            return Err(Error::new(
                ErrorKind::NotFound,
                format!("{} has only one of its two group files", stem.display()),
            ));
        }
    };

    Ok(Clusters {
        dim,
        centroids,
        assignment,
        parents,
        parent_of,
    })
}

/// Writes `ids` as an `.ibin` file of one id per row. An id past `i32::MAX`, which the file can't hold, fails with
/// `InvalidInput`.
fn write_ids(path: &Path, ids: &[u32]) -> Result<(), Error> {
    if ids.iter().any(|&id| i32::try_from(id).is_err()) {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "an id is too large for an .ibin file",
        ));
    }

    write_table(path, ids.iter().map(|id| id.to_le_bytes()), ids.len(), 1)
}

/// The ids in `values`, read from the file at `path`, which fails with `InvalidData` on one below 0.
fn ids(path: &Path, values: &[i32]) -> Result<Vec<u32>, Error> {
    values
        .iter()
        .map(|&v| u32::try_from(v).map_err(|_| damaged(path, &format!("holds the id {v}, which is below 0"))))
        .collect()
}

/// A dataset: its base vectors, its queries and their true nearest neighbors in the first `n` base vectors.
pub struct Dataset {
    pub name: String,
    pub metric: Metric,
    pub base: Fbin,
    pub queries: Fbin,
    pub truth: Ibin,
    /// The distances that go with the truth's ids, when its distances file is there.
    pub dists: Option<Fbin>,
    /// Where the distances file is, for messages.
    dists_path: PathBuf,
    /// Whether the distances file holds similarity scores, which shrink along a row, eg. inner products. They are
    /// negated to compare with a distance.
    scores: bool,
    /// How many base vectors the truth covers, from the first.
    pub n: usize,
}

impl Dataset {
    /// Opens the dataset `name`, eg. `cohere-768-cosine`, from the data folder. With `n` under the base count, the
    /// truth comes from `<name>-<n>.gt.ibin` in the folder of derived data, which `gt` writes. The truth of the whole
    /// base is `<name>.gt.ibin` in the data folder, or when that isn't there, the one `gt` writes for the whole base.
    pub fn open(name: &str, n: Option<usize>) -> Result<Dataset, Error> {
        Dataset::open_in(&data_root(), &derived_root(), name, n)
    }

    /// Opens it the same way, from the data folder `root` and the folder of derived data `derived`.
    pub fn open_in(root: &Path, derived: &Path, name: &str, n: Option<usize>) -> Result<Dataset, Error> {
        let base = Fbin::open(&root.join(format!("{name}.base.fbin")))?;
        let queries = Fbin::open(&root.join(format!("{name}.query.fbin")))?;
        let n = n.unwrap_or(base.len()).min(base.len());
        let shipped = root.join(format!("{name}.gt.ibin"));
        let path = match n == base.len() && shipped.exists() {
            true => shipped,
            false => subset_truth_in(derived, name, n),
        };
        let truth = Ibin::open(&path)?;
        let dists_path = dist_path(&path);
        let dists = match dists_path.exists() {
            true => Some(Fbin::open(&dists_path)?),
            false => None,
        };
        let metric = metric_of(name);

        if queries.dim() != base.dim() || truth.len() != queries.len() {
            return Err(Error::new(
                ErrorKind::InvalidData,
                "the dataset's files don't match each other",
            ));
        }

        if dists
            .as_ref()
            .is_some_and(|d| d.len() != truth.len() || d.dim() != truth.width())
        {
            return Err(damaged(
                &dists_path,
                "doesn't hold a distance for each id of the ground truth",
            ));
        }

        let scores = dists.as_ref().is_some_and(shrinking);
        Ok(Dataset {
            name: name.to_string(),
            metric,
            base,
            queries,
            truth,
            dists,
            dists_path,
            scores,
            n,
        })
    }

    /// The distance from query `q` to its true k-th nearest neighbor, measured from that neighbor's base row. The
    /// distances file's value must be within 0.1% of it, or within 0.001 of a distance under 1. Otherwise the truth
    /// doesn't belong to this base, which fails with `InvalidData`.
    pub fn kth(&self, q: usize, k: usize) -> Result<f32, Error> {
        let id = self.truth.row(q)[k - 1];

        if id < 0 || id as usize >= self.n {
            return Err(Error::new(
                ErrorKind::InvalidData,
                format!(
                    "the ground truth of query {q} names row {id}, which isn't one of its {} rows",
                    self.n
                ),
            ));
        }

        let d = distance(self.metric, self.queries.row(q), self.base.row(id as usize));

        if let Some(dists) = &self.dists {
            let value = dists.row(q)[k - 1];
            let value = if self.scores { -value } else { value };
            let gap = (value - d).abs();

            if gap.is_nan() || gap > 1e-3 * d.abs().max(1.0) {
                return Err(damaged(
                    &self.dists_path,
                    &format!("puts query {q}'s neighbor {k} at {value}, but the base puts it at {d}"),
                ));
            }
        }

        Ok(d)
    }

    /// Query `q`'s recall@k of `found`, its hits nearest first. It counts the ids among the true k nearest, and ties,
    /// i.e. hits at most as far from the query as the true k-th neighbor. Both distances are measured from the base
    /// rows the same way, so a copy of the k-th neighbor ties it exactly. The distances file isn't needed for that,
    /// and when it is there, it only checks that the truth belongs to this base. A k past the truth's width counts the
    /// ids alone.
    pub fn recall(&self, q: usize, found: &[u64], k: usize) -> Result<f64, Error> {
        let truth = self.truth.row(q);

        if k == 0 || k > truth.len() {
            return Ok(recall(found, truth, k));
        }

        let (kth, query) = (self.kth(q, k)?, self.queries.row(q));
        let tied = |id: u64| id < self.n as u64 && distance(self.metric, query, self.base.row(id as usize)) <= kth;
        Ok(recall_tied(found, truth, k, tied))
    }
}

/// Whether the values of a distances file shrink along its rows, as similarity scores do. It looks at the first 16.
fn shrinking(dists: &Fbin) -> bool {
    let rows: Vec<&[f32]> = (0..dists.len().min(16)).map(|i| dists.row(i)).collect();
    let ends = |row: &[f32]| (row[0], row[row.len() - 1]);
    rows.iter().all(|r| ends(r).0 >= ends(r).1) && rows.iter().any(|r| ends(r).0 > ends(r).1)
}

/// The distance that ranks the dataset: the squared Euclidean distance, or minus the inner product for cosine, whose
/// vectors are unit length. Eight partial sums let the compiler vectorize it, and the same rows always give the same
/// value.
pub fn distance(metric: Metric, q: &[f32], x: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    let (qc, qr) = q.as_chunks::<8>();
    let (xc, xr) = x.as_chunks::<8>();

    match metric {
        Metric::L2 => {
            for (a, b) in qc.iter().zip(xc) {
                (0..8).for_each(|i| acc[i] += (a[i] - b[i]) * (a[i] - b[i]));
            }
            let tail: f32 = qr.iter().zip(xr).map(|(a, b)| (a - b) * (a - b)).sum();
            acc.iter().sum::<f32>() + tail
        }
        Metric::Cosine => {
            for (a, b) in qc.iter().zip(xc) {
                (0..8).for_each(|i| acc[i] += a[i] * b[i]);
            }
            let tail: f32 = qr.iter().zip(xr).map(|(a, b)| a * b).sum();
            -(acc.iter().sum::<f32>() + tail)
        }
    }
}

/// How many of a dataset's first queries the tune and test splits take their positions from unless asked otherwise, as
/// the benchmark driver splits them.
pub const SPLIT_HEAD: usize = 1000;

/// How many of the first queries the split `name` takes its positions from. It is `asked`, eg. from `--queries`, or
/// else the first `SPLIT_HEAD` for tune and test and every query for all, and at most `total`, the dataset's query
/// count.
pub fn split_count(name: &str, asked: Option<usize>, total: usize) -> usize {
    asked
        .unwrap_or(if name == "all" { total } else { SPLIT_HEAD })
        .min(total)
}

/// The positions of the split `name` of `count` queries. `tune` takes the even positions, `test` the odd ones and
/// `all` every one. Any other name gives None.
pub fn split(count: usize, name: &str) -> Option<Vec<usize>> {
    match name {
        "tune" => Some((0..count).step_by(2).collect()),
        "test" => Some((1..count).step_by(2).collect()),
        "all" => Some((0..count).collect()),
        _ => None,
    }
}

/// The distances file that goes with the ground truth at `truth`. It is `<stem>.gt.dist.fbin` for `<stem>.gt.ibin`,
/// and otherwise the file's name without its extension, then `.dist.fbin`.
pub fn dist_path(truth: &Path) -> PathBuf {
    let name = truth
        .file_name()
        .map_or(String::new(), |n| n.to_string_lossy().into_owned());

    match name.strip_suffix(".gt.ibin") {
        Some(stem) => truth.with_file_name(format!("{stem}.gt.dist.fbin")),
        None => truth.with_extension("dist.fbin"),
    }
}

/// How the dataset `name` measures distance, from the end of its name, eg. `-euclidean` or `-cosine`.
pub fn metric_of(name: &str) -> Metric {
    if name.ends_with("-euclidean") {
        Metric::L2
    } else {
        Metric::Cosine
    }
}

/// Where the datasets are: `$COLDIVF_DATA`, or else `data/datasets` in the workspace, eg. a link to the folder that
/// holds them.
pub fn data_root() -> PathBuf {
    std::env::var_os("COLDIVF_DATA").map_or_else(|| derived_root().join("datasets"), PathBuf::from)
}

/// Where derived data goes, eg. subset ground truth and built index files. It is `data` in the workspace.
pub fn derived_root() -> PathBuf {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    manifest.parent().unwrap_or(manifest).join("data")
}

/// The ground truth of the first `n` base vectors of `name`, in the folder of derived data.
pub fn subset_truth(name: &str, n: usize) -> PathBuf {
    subset_truth_in(&derived_root(), name, n)
}

fn subset_truth_in(derived: &Path, name: &str, n: usize) -> PathBuf {
    derived.join(format!("{name}-{n}.gt.ibin"))
}

#[cfg(test)]
#[path = "../tests/unit/data.rs"]
mod tests;
