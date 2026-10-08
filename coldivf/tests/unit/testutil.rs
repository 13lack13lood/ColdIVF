//! Random data, exact reference math, index files and test read backends for the unit tests.

use std::fs::File;
use std::io::Error;

use super::SplitMix64;
use crate::io::portable::PortableWave;
use crate::io::{LANES, Range, Wave, read_at};
use crate::{BuildParams, Clusters, LeafOrder, Metric, VectorSource};

impl SplitMix64 {
    /// A uniform value in [0, 1).
    pub(crate) fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// A standard normal value, by the Box-Muller transform.
    pub(crate) fn normal(&mut self) -> f64 {
        let u1 = 1.0 - self.uniform();
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

/// `n` random vectors of `dim` values, row-major. Axis k has standard deviation `sigma(k)`.
pub(crate) fn gaussian(rng: &mut SplitMix64, n: usize, dim: usize, sigma: impl Fn(usize) -> f64) -> Vec<f32> {
    (0..n * dim).map(|i| (rng.normal() * sigma(i % dim)) as f32).collect()
}

/// The mean of the row-major vectors in `rows`, each `dim` values long.
pub(crate) fn mean(rows: &[f32], dim: usize) -> Vec<f32> {
    let n = rows.len() / dim;
    let mut sum = vec![0f64; dim];
    for row in rows.chunks_exact(dim) {
        for (s, &v) in sum.iter_mut().zip(row) {
            *s += f64::from(v);
        }
    }
    sum.iter().map(|s| (s / n as f64) as f32).collect()
}

pub(crate) fn dot(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b).map(|(&x, &y)| f64::from(x) * f64::from(y)).sum()
}

pub(crate) fn dist2(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| (f64::from(x) - f64::from(y)).powi(2))
        .sum()
}

/// Rows held in RAM, as a `VectorSource`.
pub(crate) struct Rows {
    pub(crate) dim: usize,
    pub(crate) data: Vec<f32>,
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

/// `n` vectors around `nlist` random centers. Each center's values have standard deviation 1, and each vector's
/// offset from its center has standard deviation `spread`. The clustering puts each vector in its nearest center's
/// cluster, and all clusters in one group.
pub(crate) fn clustered(rng: &mut SplitMix64, n: usize, dim: usize, nlist: usize, spread: f64) -> (Rows, Clusters) {
    let centroids = gaussian(rng, nlist, dim, |_| 1.0);
    let mut data = Vec::with_capacity(n * dim);
    for _ in 0..n {
        let c = (rng.next_u64() % nlist as u64) as usize;
        data.extend(
            centroids[c * dim..(c + 1) * dim]
                .iter()
                .map(|&x| x + (rng.normal() * spread) as f32),
        );
    }

    let assignment = data.chunks_exact(dim).map(|v| nearest(&centroids, dim, v)).collect();
    let clusters = Clusters {
        dim,
        parents: mean(&centroids, dim),
        parent_of: vec![0; nlist],
        centroids,
        assignment,
    };
    (Rows { dim, data }, clusters)
}

/// The row of `rows` nearest to `v`, by exact squared distance. Ties go to the lower row.
pub(crate) fn nearest(rows: &[f32], dim: usize, v: &[f32]) -> u32 {
    let mut best = (f64::INFINITY, 0u32);
    for (i, r) in rows.chunks_exact(dim).enumerate() {
        let d = dist2(r, v);
        if d < best.0 {
            best = (d, i as u32);
        }
    }

    best.1
}

/// Build settings for the tests, with the given metric.
pub(crate) fn params(metric: Metric) -> BuildParams {
    BuildParams {
        metric,
        vectors_per_list: 160,
        parents: 1,
        order: LeafOrder::Id,
        page_size: 4096,
        threads: 1,
        seed: 7,
    }
}

/// Shuffles `v` in place.
pub(crate) fn shuffle<T>(v: &mut [T], rng: &mut SplitMix64) {
    for i in (1..v.len()).rev() {
        v.swap(i, (rng.next_u64() % (i as u64 + 1)) as usize);
    }
}

/// Puts `clusters` into `nparents` groups. The group centers are the first `nparents` cluster centers, and each cluster
/// joins the group with the nearest center.
pub(crate) fn group(clusters: &mut Clusters, nparents: usize) {
    let dim = clusters.dim;
    clusters.parents = clusters.centroids[..nparents * dim].to_vec();
    clusters.parent_of = clusters
        .centroids
        .chunks_exact(dim)
        .map(|c| nearest(&clusters.parents, dim, c))
        .collect();
}

/// A test's file in the system's temporary folder. The name holds the process id, so test runs don't clash. Dropping it
/// deletes the file, any temporary files a build left next to it, and its change log and lock file, even when the test
/// fails.
pub(crate) struct TempFile(std::path::PathBuf);

impl std::ops::Deref for TempFile {
    type Target = std::path::Path;

    fn deref(&self) -> &std::path::Path {
        &self.0
    }
}

impl AsRef<std::path::Path> for TempFile {
    fn as_ref(&self) -> &std::path::Path {
        &self.0
    }
}

impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
        temps_of(&self.0).iter().for_each(|t| {
            let _ = std::fs::remove_file(t);
        });
        let _ = std::fs::remove_file(crate::update::log_path(&self.0));
        let _ = std::fs::remove_file(crate::update::lock_path(&self.0));
    }
}

pub(crate) fn temp_file(name: &str) -> TempFile {
    TempFile(std::env::temp_dir().join(format!("coldivf-{}-{name}", std::process::id())))
}

/// The temporary files that builds of `out` left next to it.
pub(crate) fn temps_of(out: &std::path::Path) -> Vec<std::path::PathBuf> {
    let (Some(dir), Some(name)) = (out.parent(), out.file_name()) else {
        return Vec::new();
    };
    let prefix = format!("{}.", name.to_string_lossy());

    std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            let n = p
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default();
            n.starts_with(&prefix) && n.ends_with(".tmp")
        })
        .collect()
}

/// An in-memory index of `n` vectors around `nlist` centers in 3 groups, written to the test file `name` and opened
/// again.
pub(crate) fn written(
    name: &str,
    params: &BuildParams,
    n: usize,
    dim: usize,
    nlist: usize,
) -> (crate::search::inmem::InMemory, crate::index::Inner, TempFile) {
    let mut rng = SplitMix64::new(1);
    let (rows, mut clusters) = clustered(&mut rng, n, dim, nlist, 0.5);
    group(&mut clusters, 3);
    let index = crate::search::inmem::InMemory::build(params, &rows, &clusters).unwrap();

    let path = temp_file(name);
    crate::build::write(&index, &path).unwrap();
    let file = crate::index::Inner::open(&path).unwrap();
    (index, file, path)
}

/// A read backend for tests. Each submit fills its ranges' arena bytes with junk, then each call to `next` reads one
/// range, in a random order, as a backend whose reads complete out of order would. Bytes used before their range
/// arrives are junk.
pub(crate) struct Shuffled {
    orders: [Vec<usize>; LANES],
    rng: SplitMix64,
}

impl Default for Shuffled {
    fn default() -> Shuffled {
        Shuffled {
            orders: Default::default(),
            rng: SplitMix64::new(1),
        }
    }
}

impl Wave for Shuffled {
    fn page(&self) -> usize {
        8192
    }

    fn submit(&mut self, lane: usize, _file: &File, ranges: &[Range], arena: &mut [u8]) -> Result<(), Error> {
        ranges.iter().for_each(|r| arena[r.at..r.at + r.len].fill(0xA5));
        let order = &mut self.orders[lane];
        order.clear();
        order.extend(0..ranges.len());
        shuffle(order, &mut self.rng);
        Ok(())
    }

    fn next(&mut self, lane: usize, file: &File, ranges: &[Range], arena: &mut [u8]) -> Result<Option<usize>, Error> {
        let Some(i) = self.orders[lane].pop() else {
            return Ok(None);
        };
        let r = ranges[i];
        read_at(file, &mut arena[r.at..r.at + r.len], r.offset)?;
        Ok(Some(i))
    }
}

/// A read backend for tests that reads like the portable one, except that submit number `SUBMIT` fails, and so does the
/// first wait for a range of submit number `WAVE`, counting from 1 across both lanes. A 0 fails nothing. A later
/// submit on the same lane ends the failing wave.
#[derive(Default)]
pub(crate) struct Faulty<const SUBMIT: usize, const WAVE: usize> {
    submits: usize,
    /// The lane of submit number `WAVE`, until its first wait fails.
    failing: Option<usize>,
    inner: PortableWave,
}

impl<const SUBMIT: usize, const WAVE: usize> Wave for Faulty<SUBMIT, WAVE> {
    fn page(&self) -> usize {
        4096
    }

    fn submit(&mut self, lane: usize, file: &File, ranges: &[Range], arena: &mut [u8]) -> Result<(), Error> {
        self.submits += 1;

        if self.failing == Some(lane) {
            self.failing = None;
        }

        if self.submits == WAVE {
            self.failing = Some(lane);
        }

        if self.submits == SUBMIT {
            self.inner.submit(lane, file, &[], arena)?;
            return Err(Error::other("a test submit failed"));
        }

        self.inner.submit(lane, file, ranges, arena)
    }

    fn next(&mut self, lane: usize, file: &File, ranges: &[Range], arena: &mut [u8]) -> Result<Option<usize>, Error> {
        if self.failing == Some(lane) {
            self.failing = None;
            return Err(Error::other("a test read failed"));
        }

        self.inner.next(lane, file, ranges, arena)
    }
}

/// A read backend for tests that reads like the portable one, except that submit number `WAVE` hands over one range
/// fewer than it has, as a backend that lost a completion would. A later submit on the same lane ends that wave.
#[derive(Default)]
pub(crate) struct EndsEarly<const WAVE: usize> {
    submits: usize,
    /// The lane of submit number `WAVE`, and how many of its ranges are left to hand over.
    short: Option<(usize, usize)>,
    inner: PortableWave,
}

impl<const WAVE: usize> Wave for EndsEarly<WAVE> {
    fn page(&self) -> usize {
        4096
    }

    fn submit(&mut self, lane: usize, file: &File, ranges: &[Range], arena: &mut [u8]) -> Result<(), Error> {
        self.submits += 1;

        if self.short.is_some_and(|(short, _)| short == lane) {
            self.short = None;
        }

        if self.submits == WAVE {
            self.short = Some((lane, ranges.len().saturating_sub(1)));
        }

        self.inner.submit(lane, file, ranges, arena)
    }

    fn next(&mut self, lane: usize, file: &File, ranges: &[Range], arena: &mut [u8]) -> Result<Option<usize>, Error> {
        match &mut self.short {
            Some((short, 0)) if *short == lane => return Ok(None),
            Some((short, left)) if *short == lane => *left -= 1,
            _ => {}
        }

        self.inner.next(lane, file, ranges, arena)
    }
}

thread_local! {
    /// What a `Hooked` backend runs on its chosen submit, set by a test on the thread that then searches.
    pub(crate) static HOOK: std::cell::RefCell<Option<Box<dyn FnOnce()>>> = const { std::cell::RefCell::new(None) };
}

/// A read backend for tests that reads like the portable one, except that submit number `WAVE`, counting from 1 across
/// both lanes, first runs the hook in `HOOK`, eg. a change that lands in the middle of a search.
#[derive(Default)]
pub(crate) struct Hooked<const WAVE: usize> {
    submits: usize,
    inner: PortableWave,
}

impl<const WAVE: usize> Wave for Hooked<WAVE> {
    fn page(&self) -> usize {
        4096
    }

    fn submit(&mut self, lane: usize, file: &File, ranges: &[Range], arena: &mut [u8]) -> Result<(), Error> {
        self.submits += 1;

        if self.submits == WAVE
            && let Some(hook) = HOOK.with_borrow_mut(Option::take)
        {
            hook();
        }

        self.inner.submit(lane, file, ranges, arena)
    }

    fn next(&mut self, lane: usize, file: &File, ranges: &[Range], arena: &mut [u8]) -> Result<Option<usize>, Error> {
        self.inner.next(lane, file, ranges, arena)
    }
}
