use std::io::{Error, ErrorKind};
use std::path::PathBuf;
use std::sync::{Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant};

use coldivf::{Anchor, BuildParams, Builder, Clustering, LeafOrder, Limits, Metric, SearchCtx, SearchParams};
use pyo3::buffer::PyBuffer;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// An open index. It reads only the header and the rotation when it opens, and the rest as queries need it. Searches
/// may run from several threads at once, since each takes a search context of its own, and they let other Python
/// threads run meanwhile.
#[pyclass(frozen, name = "Index", module = "coldivf")]
struct PyIndex {
    index: coldivf::Index,
    /// The largest request any context of the pool handles, which grows to fit each search.
    limits: Mutex<Limits>,
    /// The contexts that no search holds now, each with the limits it was made for.
    pool: Mutex<Vec<(Limits, SearchCtx)>>,
}

#[pymethods]
impl PyIndex {
    /// Opens the index file at `path`. Each search context fits up to `max_k` results, `max_nprobe` clusters and
    /// `max_rerank` rescored candidates, and a search that asks for more makes a larger one first, so sizing them to
    /// the searches ahead keeps the first of those searches from paying for it.
    #[new]
    #[pyo3(signature = (path, *, max_k = 100, max_nprobe = 256, max_rerank = 256))]
    fn new(py: Python<'_>, path: PathBuf, max_k: usize, max_nprobe: usize, max_rerank: usize) -> PyResult<PyIndex> {
        let index = py.detach(|| coldivf::Index::open(&path)).map_err(error)?;
        let limits = Limits {
            max_k,
            max_nprobe,
            max_rerank: max_rerank.max(max_k),
        };
        Ok(PyIndex {
            index,
            limits: Mutex::new(limits),
            pool: Mutex::new(Vec::new()),
        })
    }

    /// Call this when the query text arrives, before its embedding is ready. It starts reading the cluster tables,
    /// so that read overlaps the embedding model, and makes a search context's buffers ahead of the search.
    fn prepare(&self, py: Python<'_>) -> PyResult<()> {
        let need = *lock(&self.limits);
        let mut slot = self.take(need);
        let done = py.detach(|| self.index.prepare(&mut slot.1));
        self.give(slot);
        done.map_err(error)
    }

    /// The ids and scores of the `k` nearest vectors to `query`, a 1-D float32 array, nearest first. A search reads
    /// the `nprobe` nearest clusters, or with a `reach`, those of them within it of the nearest, and rescores the best
    /// `rerank` candidates with the more precise copy of each vector.
    #[pyo3(signature = (query, k, *, nprobe, rerank, reach = None))]
    fn search(
        &self,
        py: Python<'_>,
        query: &Bound<'_, PyAny>,
        k: usize,
        nprobe: usize,
        rerank: usize,
        reach: Option<f32>,
    ) -> PyResult<(Vec<i64>, Vec<f32>)> {
        // A copy of the query, so the search can run while another thread holds Python's lock.
        let query = vector(py, query)?;
        let params = SearchParams {
            k,
            nprobe,
            rerank,
            reach,
        };
        let need = Limits {
            max_k: k,
            max_nprobe: nprobe,
            max_rerank: rerank.max(k),
        };
        let mut slot = self.take(need);
        let found = py.detach(|| {
            self.index.search(&query, &params, &mut slot.1).map(|hits| {
                let ids = hits.iter().map(|h| h.id as i64).collect::<Vec<_>>();
                let scores = hits.iter().map(|h| h.score).collect::<Vec<_>>();
                (ids, scores)
            })
        });
        self.give(slot);
        found.map_err(error)
    }

    /// Starts reading every cluster into the operating system's cache on a background thread. Once that is done the
    /// index is warm, and searches score the clusters in place. With `wait` it returns once the index is warm or after
    /// `timeout` seconds. It returns whether the index is warm.
    #[pyo3(signature = (wait = false, timeout = 60.0))]
    fn warm_up(&self, py: Python<'_>, wait: bool, timeout: f64) -> PyResult<bool> {
        self.index.warm_up().map_err(error)?;

        if wait {
            let start = Instant::now();
            let limit = Duration::from_secs_f64(timeout.max(0.0));

            while !self.index.is_warm() && start.elapsed() < limit {
                py.detach(|| std::thread::sleep(Duration::from_millis(20)));
                py.check_signals()?;
            }
        }

        Ok(self.index.is_warm())
    }

    /// Whether every cluster is known to be in memory.
    #[getter]
    fn is_warm(&self) -> bool {
        self.index.is_warm()
    }

    /// Marks the index warm or not, eg. not when the app knows the cache was emptied.
    fn set_warm(&self, warm: bool) {
        self.index.set_warm(warm);
    }

    /// Moves to the index's newest generation, which another process's writer may have committed since, and returns
    /// whether it moved.
    fn refresh(&self, py: Python<'_>) -> PyResult<bool> {
        py.detach(|| self.index.refresh()).map_err(error)
    }

    /// Reads the whole file and checks every section against its checksum. A damaged file raises an OSError.
    fn verify(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.index.verify()).map_err(error)
    }
}

impl PyIndex {
    /// A context of the pool that fits `need`, or a new one that fits it and every request before it.
    fn take(&self, need: Limits) -> (Limits, SearchCtx) {
        let popped = lock(&self.pool).pop();

        if let Some(slot) = popped
            && fits(slot.0, need)
        {
            return slot;
        }

        let mut limits = lock(&self.limits);
        *limits = Limits {
            max_k: limits.max_k.max(need.max_k),
            max_nprobe: limits.max_nprobe.max(need.max_nprobe),
            max_rerank: limits.max_rerank.max(need.max_rerank),
        };
        (*limits, SearchCtx::new(*limits))
    }

    fn give(&self, slot: (Limits, SearchCtx)) {
        lock(&self.pool).push(slot);
    }
}

fn fits(have: Limits, need: Limits) -> bool {
    have.max_k >= need.max_k && have.max_nprobe >= need.max_nprobe && have.max_rerank >= need.max_rerank
}

/// A lock that a search which panicked while holding it doesn't spoil, since what it guards stays whole.
fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// A request the index refused, eg. a query of the wrong length, as a ValueError, and other errors as Python's OSError
/// for their kind, eg. FileNotFoundError.
fn error(e: Error) -> PyErr {
    match e.kind() {
        ErrorKind::InvalidInput => PyValueError::new_err(e.to_string()),
        _ => e.into(),
    }
}

/// A float32 array's values in row-major order, copied.
fn vector(py: Python<'_>, array: &Bound<'_, PyAny>) -> PyResult<Vec<f32>> {
    PyBuffer::<f32>::get(array)?.to_vec(py)
}

/// The rows of a build, copied out of the array they came in.
struct Rows {
    data: Vec<f32>,
    dim: usize,
}

impl coldivf::VectorSource for Rows {
    fn count(&self) -> usize {
        self.data.len().checked_div(self.dim).unwrap_or(0)
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn rows(&self, start: usize, rows: usize) -> &[f32] {
        &self.data[start * self.dim..(start + rows) * self.dim]
    }
}

/// Builds an index file at `path` from `vectors`, a 2-D float32 array with one vector per row, whose ids are their
/// row numbers. It copies the array first, so it needs memory for the vectors a second time, and lets other Python
/// threads run while it builds. For more vectors than fit in memory twice, use the build_index binary, which reads
/// them from a file. `metric` is "cosine" or "l2", `order` "chained" or "id", `anchor` "cluster" or "mean", and
/// `clustering` "flat" or "two-level". `threads` 0 takes one per logical CPU.
#[pyfunction]
#[pyo3(signature = (path, vectors, *, metric = "cosine", vectors_per_list = 160, parents = 128, order = "chained",
                    anchor = "cluster", page_size = 4096, clustering = "flat", threads = 0, seed = 7))]
#[allow(clippy::too_many_arguments)]
fn build(
    py: Python<'_>,
    path: PathBuf,
    vectors: &Bound<'_, PyAny>,
    metric: &str,
    vectors_per_list: usize,
    parents: usize,
    order: &str,
    anchor: &str,
    page_size: usize,
    clustering: &str,
    threads: usize,
    seed: u64,
) -> PyResult<()> {
    let buffer = PyBuffer::<f32>::get(vectors)?;

    let &[_, dim] = buffer.shape() else {
        return Err(PyValueError::new_err(format!(
            "vectors must be a 2-D array, not one of {} dimensions",
            buffer.dimensions()
        )));
    };

    let params = BuildParams {
        metric: pick(metric, "metric", &[("cosine", Metric::Cosine), ("l2", Metric::L2)])?,
        vectors_per_list,
        parents,
        order: pick(
            order,
            "order",
            &[("chained", LeafOrder::Chained), ("id", LeafOrder::Id)],
        )?,
        anchor: pick(
            anchor,
            "anchor",
            &[("cluster", Anchor::Cluster), ("mean", Anchor::Mean)],
        )?,
        page_size,
        threads,
        seed,
    };
    let clustering = pick(
        clustering,
        "clustering",
        &[("flat", Clustering::Flat), ("two-level", Clustering::TwoLevel)],
    )?;
    let rows = Rows {
        data: buffer.to_vec(py)?,
        dim,
    };
    let builder = Builder::new(params).clustering(clustering);
    py.detach(|| builder.build(&rows, &path)).map_err(error)
}

/// The choice that `name` names among `choices`, or a ValueError that lists them.
fn pick<T: Copy>(name: &str, what: &str, choices: &[(&str, T)]) -> PyResult<T> {
    match choices.iter().find(|(n, _)| *n == name) {
        Some(&(_, value)) => Ok(value),
        None => {
            let names = choices.iter().map(|(n, _)| *n).collect::<Vec<_>>().join(" or ");
            Err(PyValueError::new_err(format!("{what} must be {names}, not {name}")))
        }
    }
}

// The function takes another name than the module's, since PyO3 makes a Rust module of the function's name, which
// would hide the coldivf crate here.
#[pymodule]
#[pyo3(name = "coldivf")]
fn python_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyIndex>()?;
    m.add_function(wrap_pyfunction!(build, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
