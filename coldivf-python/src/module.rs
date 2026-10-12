use std::io::{Error, ErrorKind};
use std::path::PathBuf;
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError, Weak};
use std::time::{Duration, Instant};

use coldivf::{
    Anchor, BuildParams, Builder, Cancel, Clustering, Done, Job, LeafOrder, Limits, MaintainParams, Metric, RunKind,
    SearchCtx, SearchParams, SyncKind, Throttle,
};
use pyo3::buffer::{PyBuffer, ReadOnlyCell};
use pyo3::exceptions::{PyOverflowError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

/// An open index. It reads only the header and the rotation when it opens, and the rest as queries need it. Searches
/// may run from several threads at once, since each takes a search context of its own, and they let other Python
/// threads run meanwhile.
///
/// `Index(path)` opens the index file at `path`. Each search context fits up to `max_k` results, `max_nprobe`
/// clusters and `max_rerank` rescored candidates, and a search that asks for more makes a larger one first, so sizing
/// them to the searches ahead keeps the first of those searches from paying for it.
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
    // Python shows the class's doc for the constructor, so the doc is there.
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

/// The rows that `insert_many` copies and inserts at a time, and the ids that `delete_many` deletes, between its checks
/// for an interrupt, eg. Ctrl-C.
const PART: usize = 1024;

/// How long a finish that waits for a run sleeps between its checks for an interrupt.
const WAIT: Duration = Duration::from_millis(20);

/// What a run or a finish of a cancelled job raises.
const CANCELLED: &str = "the maintenance job was cancelled";

/// The kinds of maintenance run, by name.
const KINDS: [(&str, RunKind); 4] = [
    ("auto", RunKind::Auto),
    ("fold", RunKind::Fold),
    ("minor", RunKind::Minor),
    ("major", RunKind::Major),
];

/// Inserts and deletes for one index, which go to a crash-safe log next to its file. The next search of any open index
/// of the file in this process sees them, and so does any index opened later, in any process. Only one writer may hold
/// an index at a time, in any process. A with block closes it at the end. Its calls let other Python threads run
/// meanwhile, and several threads may share it, since it takes their calls one at a time.
///
/// `Writer(path)` opens the index at `path` for changes, with its log and its lock file next to it, eg.
/// `index.coldivf.delta` and `index.coldivf.lock`, or the log in its folder `index.coldivf.d` once maintenance has
/// run. A missing index raises FileNotFoundError, and one that another writer holds BlockingIOError.
#[pyclass(frozen, name = "Writer", module = "coldivf")]
struct PyWriter {
    /// The writer, until it closes.
    writer: Mutex<Option<coldivf::Writer>>,
    /// What the writer's last job shares with it, so a close ends the job too.
    job: Mutex<Weak<JobState>>,
}

#[pymethods]
impl PyWriter {
    // Python shows the class's doc for the constructor, so the doc is there.
    #[new]
    fn new(py: Python<'_>, path: PathBuf) -> PyResult<PyWriter> {
        let writer = py.detach(|| coldivf::Writer::open(&path)).map_err(error)?;
        Ok(PyWriter {
            writer: Mutex::new(Some(writer)),
            job: Mutex::new(Weak::new()),
        })
    }

    /// Adds `vector`, a 1-D float32 array, with the id `id`, replacing any earlier vector with that id. Until a
    /// maintenance run folds it in, a search scans it when its cluster is among the `nprobe` nearest the query, or
    /// among the 256 nearest. It survives the process dying, and `sync` makes it survive a power loss too. An array of
    /// another type, eg. float64, raises BufferError, and an object that isn't an array, eg. a list, TypeError. An
    /// array of another shape or length, a value that isn't finite, or a vector of zeros in a cosine index raises
    /// ValueError. Searches return ids as int64, so an id below 0 or of 2^63 or more raises OverflowError. While the
    /// write budget defers maintenance and the inserts that wait reach the hard cap of begin_compact's start settings,
    /// it raises BlockingIOError.
    fn insert(&self, py: Python<'_>, id: i64, vector: &Bound<'_, PyAny>) -> PyResult<()> {
        let buffer = PyBuffer::<f32>::get(vector)?;

        // Any other shape would pass as one vector when its size matches, eg. two rows of half the dimension.
        if buffer.dimensions() != 1 {
            return Err(PyValueError::new_err(format!(
                "vector must be a 1-D array, not one of {} dimensions",
                buffer.dimensions()
            )));
        }

        let (id, vector) = (id_of(id)?, buffer.to_vec(py)?);
        self.with(py, |w| w.insert(id, &vector))
    }

    /// Inserts each row of `vectors`, a 2-D float32 array, with the id at the same place in `ids`, a sequence of ints,
    /// eg. a 1-D integer array, as `insert` does. It copies a part of the rows at a time, so it needs little memory
    /// beyond the array's, unless its rows aren't contiguous, eg. a slice with a step, which it copies whole first. A
    /// shape other than one row per id raises ValueError, and ids that aren't a sequence of ints, eg. a 2-D array,
    /// TypeError. It stops at the first row that fails, with that row's error, and keeps the rows before it.
    fn insert_many(&self, py: Python<'_>, ids: &Bound<'_, PyAny>, vectors: &Bound<'_, PyAny>) -> PyResult<()> {
        let buffer = PyBuffer::<f32>::get(vectors)?;

        let &[rows, dim] = buffer.shape() else {
            return Err(PyValueError::new_err(format!(
                "vectors must be a 2-D array, not one of {} dimensions",
                buffer.dimensions()
            )));
        };

        let ids = ids_of(ids)?;

        if ids.len() != rows {
            return Err(PyValueError::new_err(format!("{} ids for {rows} vectors", ids.len())));
        }

        // Each part is copied, so its inserts run while another thread holds Python's lock. An array whose rows aren't
        // contiguous is copied whole first.
        let whole = match buffer.is_c_contiguous() {
            true => Vec::new(),
            false => buffer.to_vec(py)?,
        };

        for start in (0..rows).step_by(PART) {
            let end = (start + PART).min(rows);
            let (from, to) = (start * dim, end * dim);
            let part: Vec<f32> = match buffer.as_slice(py) {
                Some(cells) => cells[from..to].iter().map(ReadOnlyCell::get).collect(),
                None => whole[from..to].to_vec(),
            };
            let ids = &ids[start..end];
            self.with(py, |w| {
                for (r, &id) in ids.iter().enumerate() {
                    w.insert(id, &part[r * dim..(r + 1) * dim])
                        .map_err(|e| at_row(e, start + r))?;
                }
                Ok(())
            })?;
            py.check_signals()?;
        }

        Ok(())
    }

    /// Removes the vector with the id `id`, if there is one. Searches skip it from then on, and a maintenance run drops
    /// its row from the file. It survives the process dying, as an insert does.
    fn delete(&self, py: Python<'_>, id: i64) -> PyResult<()> {
        let id = id_of(id)?;
        self.with(py, |w| w.delete(id))
    }

    /// Deletes each id of `ids`, a sequence of ints as in insert_many, as `delete` does. It stops at the first that
    /// fails, and keeps the deletes before it.
    fn delete_many(&self, py: Python<'_>, ids: &Bound<'_, PyAny>) -> PyResult<()> {
        let ids = ids_of(ids)?;

        for part in ids.chunks(PART) {
            self.with(py, |w| part.iter().try_for_each(|&id| w.delete(id)))?;
            py.check_signals()?;
        }

        Ok(())
    }

    /// Makes every change so far survive a power loss.
    fn sync(&self, py: Python<'_>) -> PyResult<()> {
        self.with(py, |w| w.sync())
    }

    /// Rewrites the index with every change folded in, then starts the log again, and returns a RunReport. It keeps
    /// every cluster and its center, so it copies every vector's codes. It is `maintain("fold")` with the default
    /// settings.
    fn compact(&self, py: Python<'_>) -> PyResult<PyRunReport> {
        let report = self.with(py, |w| w.maintain(&MaintainParams::default(), RunKind::Fold))?;
        Ok(report.into())
    }

    /// Runs one maintenance run of `kind` now and returns a RunReport. "fold" folds the changes in and clusters nothing
    /// again, "minor" clusters again the regions that are due, "major" clusters every vector again, and "auto" is a
    /// major run when one is due, else a minor run when a region is, else a fold. It writes a new data file and
    /// commits it as the index's next generation, so it blocks the writer until the commit, while searches and other
    /// Python threads go on. At the commit every open index of the file in this process moves to the new generation.
    /// The first run makes the folder `name.d` next to the index file, which then holds every generation's files, and
    /// deletes the index file and its log. `params` are maintenance settings, as in begin_compact, but `start` and
    /// `throttle` change nothing here. While a job of the writer lives it raises BlockingIOError.
    #[pyo3(signature = (kind = "fold", **params))]
    fn maintain(&self, py: Python<'_>, kind: &str, params: Option<&Bound<'_, PyDict>>) -> PyResult<PyRunReport> {
        let kind = pick(kind, "kind", &KINDS)?;
        let params = maintain_params(params)?;
        let report = self.with(py, |w| w.maintain(&params, kind))?;
        Ok(report.into())
    }

    /// Starts a maintenance run as a Job, which `Job.run` runs on the thread that calls it, eg. a thread of its own,
    /// while the writer goes on, or returns None when no run is due, or when the minimum interval or the write budget
    /// holds one back. The job folds in the changes made until now, which it first makes survive a power loss, as
    /// `sync` does. Later changes move to the new generation's log at the commit. It is a run of kind "auto", or
    /// "major" when the read-cost indicator calls for one.
    ///
    /// `params` are maintenance settings, with the names of the fields of the Rust crate's MaintainParams:
    /// `vectors_per_list`, `rounds`, `parents`, `order` ("chained" or "id"), `threads` and `seed`, and a dict each for
    /// `start`, which says when a run is due, `minor`, `major` and `throttle`, which spaces the job's disk work.
    /// Durations are in seconds, and the optional settings take None, eg. `begin_compact(threads=2,
    /// start={"min_interval": 0})`, though the interval stays at least 10 times the last run's length. A setting it
    /// doesn't know, or a value of the wrong type, raises TypeError, a negative count OverflowError, and a value out of
    /// range ValueError. While a job of the writer lives it raises BlockingIOError.
    #[pyo3(signature = (**params))]
    fn begin_compact(slf: &Bound<'_, Self>, params: Option<&Bound<'_, PyDict>>) -> PyResult<Option<PyJob>> {
        let params = maintain_params(params)?;
        let me = slf.get();
        // The writer's lock covers the link to the job, so a close meanwhile can't miss it.
        let state = me.with(slf.py(), |w| {
            let Some(job) = w.begin_compact(&params)? else {
                return Ok(None);
            };
            let state = Arc::new(JobState {
                cancel: job.canceller(),
                stage: Mutex::new(Stage::Ready(Box::new(job))),
                ran: Condvar::new(),
            });
            *lock(&me.job) = Arc::downgrade(&state);
            Ok(Some(state))
        })?;
        Ok(state.map(|state| PyJob {
            writer: slf.clone().unbind(),
            state,
            throttle: Throttle::new(params.throttle),
        }))
    }

    /// What maintenance with `params`, as in begin_compact, would do now, as a Status. It reads the file's ids when a
    /// change hid any vector.
    #[pyo3(signature = (**params))]
    fn status(&self, py: Python<'_>, params: Option<&Bound<'_, PyDict>>) -> PyResult<PyStatus> {
        let params = maintain_params(params)?;
        let status = self.with(py, |w| w.status(&params))?;
        Ok(status.into())
    }

    /// Closes the writer, which ends its job as `Job.cancel` does. Another writer can open the index once the job's
    /// run, if one is under way, stops at its next read or write, so join the thread that runs it first. It doesn't
    /// sync, so the changes since the last sync survive the process dying but not a power loss. Later calls raise
    /// ValueError, and another close does nothing.
    fn close(&self, py: Python<'_>) {
        py.detach(|| {
            drop(lock(&self.writer).take());
            let job = lock(&self.job).upgrade();

            if let Some(job) = job {
                job.end("the writer closed, which cancelled its job");
            }
        });
    }

    fn __enter__(slf: &Bound<'_, Self>) -> PyResult<Py<Self>> {
        // A closed writer refuses a with block, as a closed file does.
        slf.get().with(slf.py(), |_| Ok(()))?;
        Ok(slf.clone().unbind())
    }

    #[pyo3(signature = (*_exception))]
    fn __exit__(&self, py: Python<'_>, _exception: &Bound<'_, PyTuple>) {
        self.close(py);
    }
}

impl PyWriter {
    /// What `f` returns for the writer, or an `InvalidInput` error once the writer is closed.
    fn locked<T>(&self, f: impl FnOnce(&mut coldivf::Writer) -> Result<T, Error>) -> Result<T, Error> {
        match lock(&self.writer).as_mut() {
            Some(writer) => f(writer),
            None => Err(Error::new(ErrorKind::InvalidInput, "the writer is closed")),
        }
    }

    /// The same, while other Python threads run.
    fn with<T: Send>(
        &self,
        py: Python<'_>,
        f: impl FnOnce(&mut coldivf::Writer) -> Result<T, Error> + Send,
    ) -> PyResult<T> {
        py.detach(|| self.locked(f)).map_err(error)
    }
}

/// A maintenance run that `Writer.begin_compact` started. `run` plans and writes the next generation's data file on
/// the thread that calls it, eg. a Python thread of its own, while the writer goes on, and `finish` commits it. It
/// holds the writer's lock, so no other writer opens the index until it finishes, or ends and its run, if one is under
/// way, stops.
#[pyclass(frozen, name = "Job", module = "coldivf")]
struct PyJob {
    writer: Py<PyWriter>,
    state: Arc<JobState>,
    /// What spaces the job's disk work, made from begin_compact's throttle settings.
    throttle: Throttle,
}

#[pymethods]
impl PyJob {
    /// Plans and writes the next generation's data file on the calling thread, while other Python threads run. With
    /// `throttle` it waits while the index is searched, in any process, as begin_compact's throttle settings say, and
    /// its threads run at low priority. Without it, it reads and writes as fast as it can, and it neither checks the
    /// free disk space first nor records where it got. A cancel stops it with InterruptedError, and it keeps what it
    /// recorded, so the next job of the same generation goes on from there. A job runs once, so another run raises
    /// ValueError, or the error that ended the job.
    #[pyo3(signature = (throttle = true))]
    fn run(&self, py: Python<'_>, throttle: bool) -> PyResult<()> {
        let none = Throttle::none();
        let gate = if throttle { &self.throttle } else { &none };
        py.detach(|| self.state.run(gate)).map_err(error)
    }

    /// Commits the next generation that the run wrote and returns its RunReport. It first waits for the run on another
    /// thread, which may not have started yet, so a job that no thread runs keeps it waiting until a cancel or Ctrl-C.
    /// Every open index of the file in this process moves to the new generation at once, and one in another process at
    /// its next refresh or prepare. A job that a cancel, a close of its writer or a failed run ended raises that error,
    /// eg. InterruptedError, and a finished one ValueError.
    fn finish(&self, py: Python<'_>) -> PyResult<PyRunReport> {
        let done = loop {
            match py.detach(|| self.state.take(WAIT)) {
                Some(done) => break done.map_err(error)?,
                None => py.check_signals()?,
            }
        };
        let writer = self.writer.get();
        let report = py.detach(|| {
            writer
                .locked(|w| w.finish_compact(*done))
                .inspect_err(|e| self.state.fail(e))
        });
        report.map(PyRunReport::from).map_err(error)
    }

    /// Ends the job, from any thread. A run under way stops at its next read or write with InterruptedError, and it
    /// keeps what it recorded, so the next job of the same generation goes on from there. A job that hasn't run, or
    /// ran but isn't finished, ends now, and its file goes. From then on run and finish raise InterruptedError. A
    /// finished job ignores it.
    fn cancel(&self, py: Python<'_>) {
        py.detach(|| self.state.end(CANCELLED));
    }

    /// Tells the job whether the app runs in the background, where it writes as when the index is idle. Any thread may
    /// call it while the job runs.
    fn set_background(&self, background: bool) {
        self.throttle.set_background(background);
    }

    /// Whether a run of the job is under way.
    #[getter]
    fn running(&self) -> bool {
        matches!(*lock(&self.state.stage), Stage::Running { .. })
    }
}

/// What a job shares with its writer, which a close of the writer ends.
struct JobState {
    stage: Mutex<Stage>,
    /// Wakes a finish that waits for the run to end.
    ran: Condvar,
    cancel: Cancel,
}

/// Where a job is.
enum Stage {
    /// Begun, and not run yet.
    Ready(Box<Job>),
    /// Under way on some thread, and whether a cancel came meanwhile.
    Running { cancelled: bool },
    /// Run to the end, with its file to commit.
    Ran(Box<Done>),
    /// Ended without a commit, with the kind and the message of the error that ended it.
    Ended(ErrorKind, String),
    /// Committed.
    Finished,
}

impl Stage {
    /// What a call that this stage doesn't allow fails with: the error that ended the job, or the stage it is in.
    fn refusal(&self) -> Error {
        let what = match self {
            Stage::Ended(kind, message) => return Error::new(*kind, message.clone()),
            Stage::Ready(_) => "the job hasn't run",
            Stage::Running { .. } | Stage::Ran(_) => "the job ran already",
            Stage::Finished => "the job finished already",
        };
        Error::new(ErrorKind::InvalidInput, what)
    }
}

impl JobState {
    /// Runs the job through `throttle`, unless it ran already.
    fn run(&self, throttle: &Throttle) -> Result<(), Error> {
        let mut job = {
            let mut stage = lock(&self.stage);
            match std::mem::replace(&mut *stage, Stage::Running { cancelled: false }) {
                Stage::Ready(job) => job,
                other => {
                    let refused = other.refusal();
                    *stage = other;
                    return Err(refused);
                }
            }
        };
        // A run that panics ends the job too, so a finish that waits for it doesn't wait forever.
        let ran = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| job.run(throttle)))
            .unwrap_or_else(|_| Err(Error::other("the maintenance job panicked")));
        drop(job);
        let mut stage = lock(&self.stage);
        // A cancel that came as the run ended still ends the job, and its file goes.
        let ran = match (ran, &*stage) {
            (Ok(_), Stage::Running { cancelled: true }) => Err(Error::new(ErrorKind::Interrupted, CANCELLED)),
            (ran, _) => ran,
        };
        let result = match ran {
            Ok(done) => {
                *stage = Stage::Ran(Box::new(done));
                Ok(())
            }
            Err(e) => {
                *stage = Stage::Ended(e.kind(), e.to_string());
                Err(e)
            }
        };
        self.ran.notify_all();
        result
    }

    /// The file of the job's run to commit, or `None` while the run is still to start or under way after waiting up to
    /// `wait`. A finish may come before the thread that runs the job starts, eg. one of a thread pool.
    fn take(&self, wait: Duration) -> Option<Result<Box<Done>, Error>> {
        let mut stage = lock(&self.stage);

        if matches!(*stage, Stage::Ready(_) | Stage::Running { .. }) {
            stage = self
                .ran
                .wait_timeout(stage, wait)
                .unwrap_or_else(PoisonError::into_inner)
                .0;
        }

        match std::mem::replace(&mut *stage, Stage::Finished) {
            Stage::Ran(done) => Some(Ok(done)),
            waiting @ (Stage::Ready(_) | Stage::Running { .. }) => {
                *stage = waiting;
                None
            }
            other => {
                let refused = other.refusal();
                *stage = other;
                Some(Err(refused))
            }
        }
    }

    /// Ends the job with `message`. A run under way stops at its next read or write, and a job that hasn't run, or ran
    /// but didn't commit, ends now, and its file goes.
    fn end(&self, message: &str) {
        self.cancel.cancel();
        let mut stage = lock(&self.stage);

        match &mut *stage {
            Stage::Running { cancelled } => *cancelled = true,
            Stage::Ready(_) | Stage::Ran(_) => *stage = Stage::Ended(ErrorKind::Interrupted, message.to_string()),
            Stage::Ended(..) | Stage::Finished => {}
        }

        // A finish that waits for a job that hasn't run raises at once.
        self.ran.notify_all();
    }

    /// Ends the job with `e`, the error of its failed commit.
    fn fail(&self, e: &Error) {
        *lock(&self.stage) = Stage::Ended(e.kind(), e.to_string());
    }
}

/// What a maintenance run did. Times are in seconds.
#[pyclass(frozen, get_all, name = "RunReport", module = "coldivf")]
struct PyRunReport {
    /// The run it did, "fold", "minor" or "major".
    kind: &'static str,
    /// Regions that were due.
    regions_due: usize,
    /// Vectors in the regions it clustered again, before the boundary fix.
    reclustered: usize,
    /// Vectors that changed cluster.
    moved: usize,
    /// Codes measured again from their vectors' 8-bit copies.
    reencoded: usize,
    /// Codes copied.
    copied: usize,
    /// Vectors in the new file.
    count: usize,
    /// Clusters in the new file.
    lists: usize,
    /// Regions in the new file.
    regions: usize,
    /// Vector-to-center distances that its k-means and boundary fix computed.
    distances: u64,
    bytes_read: u64,
    bytes_written: u64,
    /// Processor time of the run's threads.
    cpu: f64,
    /// The time it took.
    wall: f64,
    /// The read-cost indicator of the new file, when the run measured it, which it does with `major={"indicator":
    /// ...}` set, else None.
    indicator: Option<f32>,
}

#[pymethods]
impl PyRunReport {
    fn __repr__(&self) -> String {
        let indicator = self.indicator.map_or_else(|| "None".to_string(), |i| format!("{i:?}"));
        format!(
            "RunReport(kind='{}', regions_due={}, reclustered={}, moved={}, reencoded={}, copied={}, count={}, \
             lists={}, regions={}, distances={}, bytes_read={}, bytes_written={}, cpu={:?}, wall={:?}, \
             indicator={indicator})",
            self.kind,
            self.regions_due,
            self.reclustered,
            self.moved,
            self.reencoded,
            self.copied,
            self.count,
            self.lists,
            self.regions,
            self.distances,
            self.bytes_read,
            self.bytes_written,
            self.cpu,
            self.wall,
        )
    }
}

impl From<coldivf::RunReport> for PyRunReport {
    fn from(r: coldivf::RunReport) -> PyRunReport {
        PyRunReport {
            kind: KINDS.iter().find(|(_, k)| *k == r.kind).map_or("", |&(name, _)| name),
            regions_due: r.regions_due,
            reclustered: r.reclustered,
            moved: r.moved,
            reencoded: r.reencoded,
            copied: r.copied,
            count: r.count,
            lists: r.lists,
            regions: r.regions,
            distances: r.distances,
            bytes_read: r.bytes_read,
            bytes_written: r.bytes_written,
            cpu: r.cpu.as_secs_f64(),
            wall: r.wall.as_secs_f64(),
            indicator: r.indicator,
        }
    }
}

/// What maintenance would do now.
#[pyclass(frozen, get_all, name = "Status", module = "coldivf")]
struct PyStatus {
    /// Inserts waiting in the log.
    pending: usize,
    /// Vectors of the file that a change hid.
    deleted: usize,
    /// Regions due to be clustered again.
    regions_due: usize,
    /// Whether a major run is due.
    major_due: bool,
    /// Whether a run is due, as the start settings say.
    due: bool,
    /// Whether the write budget holds a due run back, while the inserts that wait reach twice what makes one due.
    deferred: bool,
    /// Bytes that the runs of the last 24 hours wrote.
    bytes_written_24h: u64,
    /// Codes that the runs of the last 24 hours measured again from the 8-bit copies.
    reencoded_24h: u64,
    /// Processor seconds that the runs of the last 24 hours used.
    cpu_24h: f64,
}

#[pymethods]
impl PyStatus {
    fn __repr__(&self) -> String {
        format!(
            "Status(pending={}, deleted={}, regions_due={}, major_due={}, due={}, deferred={}, bytes_written_24h={}, \
             reencoded_24h={}, cpu_24h={:?})",
            self.pending,
            self.deleted,
            self.regions_due,
            py_bool(self.major_due),
            py_bool(self.due),
            py_bool(self.deferred),
            self.bytes_written_24h,
            self.reencoded_24h,
            self.cpu_24h,
        )
    }
}

impl From<coldivf::Status> for PyStatus {
    fn from(s: coldivf::Status) -> PyStatus {
        PyStatus {
            pending: s.pending,
            deleted: s.deleted,
            regions_due: s.regions_due,
            major_due: s.major_due,
            due: s.due,
            deferred: s.deferred,
            bytes_written_24h: s.bytes_written_24h,
            reencoded_24h: s.reencoded_24h,
            cpu_24h: s.cpu_24h.as_secs_f64(),
        }
    }
}

/// A bool as Python writes it.
fn py_bool(b: bool) -> &'static str {
    if b { "True" } else { "False" }
}

/// An id as the index keeps it. Searches return ids as int64, so an id is at least 0.
fn id_of(id: i64) -> PyResult<u64> {
    u64::try_from(id).map_err(|_| PyOverflowError::new_err(format!("an id must be at least 0, not {id}")))
}

/// The ids of a batch, from a sequence of ints, eg. a list or a 1-D integer array.
fn ids_of(ids: &Bound<'_, PyAny>) -> PyResult<Vec<u64>> {
    ids.extract::<Vec<i64>>()?.into_iter().map(id_of).collect()
}

/// `e` with the row of the batch that it came from, eg. "row 3: a vector has a value that isn't finite".
fn at_row(e: Error, row: usize) -> Error {
    Error::new(e.kind(), format!("row {row}: {e}"))
}

/// Maintenance settings: the defaults, with each setting that `params` names in its place. `start`, `minor`, `major`
/// and `throttle` take dicts of their own settings.
fn maintain_params(params: Option<&Bound<'_, PyDict>>) -> PyResult<MaintainParams> {
    let mut p = MaintainParams::default();
    let Some(params) = params else {
        return Ok(p);
    };

    for (name, value) in settings(params.as_any(), "maintenance")? {
        match name.as_str() {
            "start" | "minor" | "major" | "throttle" => {
                for (setting, value) in settings(&value, &name)? {
                    set(&mut p, &name, &setting, &value)?;
                }
            }
            _ => set(&mut p, "", &name, &value)?,
        }
    }

    Ok(p)
}

/// The names and values of `dict`, a dict of settings, eg. `{"min_interval": 0}`.
fn settings<'py>(dict: &Bound<'py, PyAny>, what: &str) -> PyResult<Vec<(String, Bound<'py, PyAny>)>> {
    let dict = dict
        .cast::<PyDict>()
        .map_err(|_| PyTypeError::new_err(format!("{what} takes a dict of settings")))?;
    dict.iter().map(|(name, value)| Ok((name.extract()?, value))).collect()
}

/// Sets the setting `name` of `group`, eg. "start", to `value`. The settings of MaintainParams itself have no group.
fn set(p: &mut MaintainParams, group: &str, name: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
    let v = value;

    match (group, name) {
        ("", "vectors_per_list") => p.vectors_per_list = v.extract()?,
        ("", "rounds") => p.rounds = v.extract()?,
        ("", "parents") => p.parents = v.extract()?,
        ("", "order") => {
            let orders = [("chained", LeafOrder::Chained), ("id", LeafOrder::Id)];
            p.order = pick(&v.extract::<String>()?, "order", &orders)?;
        }
        ("", "threads") => p.threads = v.extract()?,
        ("", "seed") => p.seed = v.extract()?,
        ("start", "pending_share") => p.start.pending_share = v.extract()?,
        ("start", "ram_budget") => p.start.ram_budget = v.extract()?,
        ("start", "dead_share") => p.start.dead_share = v.extract()?,
        ("start", "read_excess") => p.start.read_excess = v.extract()?,
        ("start", "excess_queries") => p.start.excess_queries = v.extract()?,
        ("start", "idle") if v.is_none() => p.start.idle = None,
        ("start", "idle") => p.start.idle = Some(seconds(v)?),
        ("start", "min_interval") => p.start.min_interval = seconds(v)?,
        ("start", "write_budget") => p.start.write_budget = v.extract()?,
        ("start", "hard_cap") => p.start.hard_cap = v.extract()?,
        ("minor", "change") => p.minor.change = v.extract()?,
        ("minor", "shrink") => p.minor.shrink = v.extract()?,
        ("minor", "hot_list") => p.minor.hot_list = v.extract()?,
        ("minor", "list_loss") => p.minor.list_loss = v.extract()?,
        ("minor", "region_lists") => p.minor.region_lists = v.extract()?,
        ("minor", "split") => p.minor.split = v.extract()?,
        ("minor", "merge") => p.minor.merge = v.extract()?,
        ("minor", "ring") => p.minor.ring = v.extract()?,
        ("minor", "pool") => p.minor.pool = v.extract()?,
        ("minor", "global_pass") => p.minor.global_pass = v.extract()?,
        ("minor", "refresh") => p.minor.refresh = v.extract()?,
        ("major", "inserts") => p.major.inserts = v.extract()?,
        ("major", "deletes") => p.major.deletes = v.extract()?,
        ("major", "indicator") => p.major.indicator = v.extract()?,
        ("major", "beta") => p.major.beta = v.extract()?,
        ("major", "sample") => p.major.sample = v.extract()?,
        ("throttle", "burst_bytes") => p.throttle.burst_bytes = v.extract()?,
        ("throttle", "idle_burst_bytes") => p.throttle.idle_burst_bytes = v.extract()?,
        ("throttle", "quiet") => p.throttle.quiet = seconds(v)?,
        ("throttle", "spacing") => p.throttle.spacing = seconds(v)?,
        ("throttle", "idle") => p.throttle.idle = seconds(v)?,
        ("throttle", "sync") => {
            let kinds = [("full", SyncKind::Full), ("barrier", SyncKind::Barrier)];
            p.throttle.sync = pick(&v.extract::<String>()?, "sync", &kinds)?;
        }
        ("throttle", "resume_bytes") => p.throttle.resume_bytes = v.extract()?,
        ("throttle", "reserve_bytes") => p.throttle.reserve_bytes = v.extract()?,
        _ => {
            let group = if group.is_empty() { "maintenance" } else { group };
            return Err(PyTypeError::new_err(format!("{group} has no setting named {name}")));
        }
    }

    Ok(())
}

/// A duration given in seconds, eg. 0.1.
fn seconds(value: &Bound<'_, PyAny>) -> PyResult<Duration> {
    let s: f64 = value.extract()?;
    Duration::try_from_secs_f64(s)
        .map_err(|_| PyValueError::new_err(format!("a duration must be 0 or more seconds, not {s}")))
}

/// The data file that holds the index at `path` now: `path` itself until the index's first maintenance run, and a file
/// in its folder `name.d` from then on.
#[pyfunction]
fn data_file(py: Python<'_>, path: PathBuf) -> PyResult<PathBuf> {
    py.detach(|| coldivf::data_file(&path)).map_err(error)
}

// The function takes another name than the module's, since PyO3 makes a Rust module of the function's name, which
// would hide the coldivf crate here.
#[pymodule]
#[pyo3(name = "coldivf")]
fn python_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyIndex>()?;
    m.add_class::<PyWriter>()?;
    m.add_class::<PyJob>()?;
    m.add_class::<PyRunReport>()?;
    m.add_class::<PyStatus>()?;
    m.add_function(wrap_pyfunction!(build, m)?)?;
    m.add_function(wrap_pyfunction!(data_file, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
