//! Maintenance runs as jobs on any thread while the writer goes on. `Writer::begin_compact` takes the snapshot,
//! `Job::run` plans the next generation's data file and writes it in bursts that a throttle gates, and
//! `Writer::finish_compact` commits it on the writer's thread. The job holds the writer's lock, so no other writer
//! opens until it ends.
//!
//! The throttle keeps the job off the disk while the index is searched. A burst of writes starts only once no search
//! of the index, in any process, ran for a while, and it ends with a sync, so the drive is idle again by the next
//! search. A search during a burst ends it at the next write. The job's reads of the old file wait the same way, and its
//! threads run at low priority.
//!
//! Once the plan fixes the next generation's centers, the writer measures each insert for both generations, and the
//! job measures the inserts made since the snapshot from their rows, so the commit routes nothing.
//!
//! Every so many bytes the job makes its writes durable and records where it got in a resume file next to its data
//! file. A job that was cancelled, or a process that died, leaves both, and the next job of the same generation goes
//! on from there, provided it plans the same file.

use std::fs::{self, File};
use std::io::{Error, ErrorKind};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use super::{Lane, Snapshot, Writes};
use crate::build::rewrite::{Gate, Mark, Out, Plan, Progress};
use crate::build::stream::{Reached, Sink};
use crate::index::{Inner, Shared};
use crate::io::{self, Activity};
use crate::quantization::rabitq::{Encoder, code_bytes};
use crate::quantization::{padded_dim, sq4, sq8};
use crate::root::{Next, Owner};
use crate::search::inmem::InMemory;
use crate::search::route::Centers;
use crate::update::{Delta, Seen, locked};
use crate::{Anchor, Limits, MaintainParams, RunKind, RunReport, SearchCtx};

/// The most bytes that one write of a burst takes, so a search that comes meanwhile waits for at most one.
const PIECE: usize = 1 << 20;

/// How often a job that waits for quiet looks at the activity word and its cancel flag again.
const POLL: Duration = Duration::from_millis(5);

/// What a resume file starts with, and its version.
const MAGIC: &[u8; 8] = b"CIVFRES1";

/// How a maintenance job spaces its disk work so that searches don't wait for the disk.
#[derive(Clone, Debug, PartialEq)]
pub struct ThrottleParams {
    /// The most bytes that a burst of writes takes before it syncs, 2 MiB, and once no search ran for `idle`, 16 MiB.
    pub burst_bytes: usize,
    pub idle_burst_bytes: usize,
    /// A burst starts only once no search ran for `quiet`, 100 ms, and `spacing` after the last burst ended, 80 ms.
    /// Each read of the old file waits for `quiet` too.
    pub quiet: Duration,
    pub spacing: Duration,
    /// Once no search ran for `idle`, 2 s, or while the app says it runs in the background, bursts of
    /// `idle_burst_bytes` run back to back.
    pub idle: Duration,
    /// How each burst ends.
    pub sync: SyncKind,
    /// How often the job records where it got, in bytes written, so a cancelled job goes on from there, 64 MiB.
    pub resume_bytes: u64,
    /// The free disk space that the job leaves beyond its new file and a tenth more, 3 GiB, about where macOS starts
    /// to warn of low space. A job that would leave less fails with `StorageFull` before it writes.
    pub reserve_bytes: u64,
}

impl Default for ThrottleParams {
    fn default() -> ThrottleParams {
        ThrottleParams {
            burst_bytes: 2 << 20,
            idle_burst_bytes: 16 << 20,
            quiet: Duration::from_millis(100),
            spacing: Duration::from_millis(80),
            idle: Duration::from_secs(2),
            sync: SyncKind::Full,
            resume_bytes: 64 << 20,
            reserve_bytes: 3 << 30,
        }
    }
}

/// How a burst of a job's writes ends.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SyncKind {
    /// The burst is on the disk when the sync returns, which on macOS is `F_FULLFSYNC`, and the drive is idle again.
    #[default]
    Full,
    /// Later writes reach the disk after the burst, which on macOS is `F_BARRIERFSYNC`, at a fraction of the cost. The
    /// drive may still be writing the burst when the next search comes.
    Barrier,
}

/// What a job's disk work waits for: its settings, or nothing, and whether the app runs in the background now.
#[derive(Debug)]
pub struct Throttle {
    params: Option<ThrottleParams>,
    background: AtomicBool,
}

impl Throttle {
    pub fn new(params: ThrottleParams) -> Throttle {
        Throttle {
            params: Some(params),
            background: AtomicBool::new(false),
        }
    }

    /// No gate: the job reads and writes as fast as it can, and syncs only its whole file, as
    /// [`Writer::maintain`](crate::Writer::maintain) does.
    pub fn none() -> Throttle {
        Throttle {
            params: None,
            background: AtomicBool::new(false),
        }
    }

    /// Tells every job that this throttle gates that the app runs in the background, so they write as when the index
    /// is idle, or that it runs in front again. Any thread may call it while the jobs run.
    pub fn set_background(&self, background: bool) {
        self.background.store(background, Ordering::Relaxed);
    }
}

impl Default for Throttle {
    fn default() -> Throttle {
        Throttle::new(ThrottleParams::default())
    }
}

/// What a writer and its job share: whether the job, or the file it wrote, is still to commit, the cancel flag, and
/// the next generation's centers once the job's plan fixes them, so the writer measures each insert for both
/// generations.
pub(crate) struct Link {
    live: AtomicBool,
    cancel: AtomicBool,
    centers: OnceLock<NextCenters>,
}

impl Link {
    fn new() -> Link {
        Link {
            live: AtomicBool::new(true),
            cancel: AtomicBool::new(false),
            centers: OnceLock::new(),
        }
    }

    /// Whether the job may still give the writer a file to commit: it hasn't failed, and neither it nor its `Done`
    /// was dropped.
    pub(crate) fn live(&self) -> bool {
        self.live.load(Ordering::Acquire)
    }

    fn end(&self) {
        self.live.store(false, Ordering::Release);
    }

    /// The next generation's centers, once the job's plan fixed them, while the job is live.
    pub(crate) fn centers(&self) -> Option<&NextCenters> {
        self.centers.get().filter(|_| self.live())
    }

    pub(crate) fn cancel(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }
}

/// The next generation's cluster centers as its data file stores them.
pub(crate) struct NextCenters {
    codes: Vec<u8>,
    factors: Vec<sq4::CenterFactors>,
    norms: Vec<f64>,
}

impl NextCenters {
    fn of(frame: &InMemory) -> NextCenters {
        NextCenters {
            codes: frame.centers.clone(),
            factors: frame.center_factors.clone(),
            norms: frame.center_norms.clone(),
        }
    }

    pub(crate) fn centers(&self) -> Centers<'_> {
        Centers {
            codes: &self.codes,
            factors: &self.factors,
            norms: &self.norms,
        }
    }
}

/// A maintenance run that runs on any thread while the writer goes on. [`Writer::begin_compact`](crate::Writer)
/// makes it, [`Job::run`] plans and writes the next generation's data file, and
/// [`Writer::finish_compact`](crate::Writer) commits it. It holds the writer's lock, so no other writer opens the
/// index until the job and what it returns are gone. Dropping its writer cancels it.
pub struct Job {
    old: Arc<Inner>,
    delta: Arc<Delta>,
    snapshot: Snapshot,
    params: MaintainParams,
    kind: RunKind,
    next: Option<(Next, File)>,
    /// The resume file, and what it held when the job began, if it was this generation's.
    resume: PathBuf,
    record: Option<Record>,
    link: Arc<Link>,
    activity: Arc<Activity>,
    writer: u64,
    /// The writer's lock until the job runs, when it passes to the `Done`, or goes with a failed run.
    lock: Option<Arc<File>>,
    /// Whether the generation that the job rewrites was warm, so the next one starts with its codes in memory, and
    /// then which of its row pages were in memory as the job began, before its own reads of the file could drop them.
    warm: bool,
    kept_rows: Vec<bool>,
    /// The most bytes of rows that a window of the rewrite gathers, 0 for the rewrite's own, which tests make small.
    pub(crate) window: usize,
}

impl Drop for Job {
    /// A job dropped before it ran gives the writer nothing to commit.
    fn drop(&mut self) {
        if self.next.is_some() {
            self.link.end();
        }
    }
}

/// A job's finished data file, which [`Writer::finish_compact`](crate::Writer) commits as the index's next
/// generation. Dropping it instead deletes the file.
pub struct Done {
    pub(crate) writer: u64,
    pub(crate) base: Arc<Inner>,
    pub(crate) next: Option<Next>,
    pub(crate) snapshot: Snapshot,
    pub(crate) report: RunReport,
    /// The lanes of the inserts made between the snapshot and the plan, measured by the job.
    pub(crate) lanes: Vec<Lane>,
    /// The bytes per query for 95% recall of the new file, when the job measured them.
    pub(crate) cost: Option<f64>,
    /// Whether the generation it rewrote was warm, and so the new one starts warm, with its codes in memory.
    pub(crate) warm: bool,
    resume: PathBuf,
    link: Arc<Link>,
    _lock: Arc<File>,
}

impl Drop for Done {
    fn drop(&mut self) {
        self.link.end();
        let _ = io::remove(&self.resume);
    }
}

/// Cancels a job from any thread. The job stops at its next write or read with `Interrupted`, and keeps what it
/// recorded for the next job of the same generation.
#[derive(Clone)]
pub struct Cancel(Arc<Link>);

impl Cancel {
    pub fn cancel(&self) {
        self.0.cancel();
    }
}

impl Job {
    /// What cancels the job from another thread.
    pub fn canceller(&self) -> Cancel {
        Cancel(Arc::clone(&self.link))
    }

    /// Whether the job goes on from a resume file that an earlier job left.
    #[cfg(test)]
    pub(crate) fn resumes(&self) -> bool {
        self.record.is_some()
    }

    /// Plans the next generation's data file and writes it, on the calling thread and a pool of the job's own, as
    /// `throttle` lets it. With [`Throttle::none`] it blocks the calling thread for as long as the run takes, and
    /// otherwise it also waits for quiet. The pool runs at utility quality of service (QoS) on macOS and a nice value
    /// of 10 on Linux, unless `throttle` is none. A cancelled job fails with `Interrupted`. It keeps its file and resume
    /// file once it recorded where it got, so the next job of the same generation goes on from there. A job runs once.
    pub fn run(&mut self, throttle: &Throttle) -> Result<Done, Error> {
        let (Some((next, file)), Some(lock)) = (self.next.take(), self.lock.take()) else {
            return Err(Error::new(ErrorKind::InvalidInput, "the job ran already"));
        };
        let pool = super::pool(self.params.threads, throttle.params.is_some())?;
        // A warm index's new codes go through the system's cache, so they are in memory once written.
        let cached = match self.warm {
            true => Some(fs::OpenOptions::new().write(true).open(&next.data)?),
            false => None,
        };
        let mut bursts = Bursts::new(&file, &next.data, throttle, &self.activity, &self.link.cancel);
        (bursts.cached, bursts.kept_rows) = (cached.as_ref(), &self.kept_rows);
        let mut tracker = Tracker {
            job: self,
            bursts: &bursts,
            record: self.record.clone(),
            start: None,
            plan: 0,
            marked: 0,
            recorded: false,
            lanes: Vec::new(),
        };
        let writes = Writes {
            out: &bursts,
            gate: &bursts,
            progress: &mut tracker,
            pool: &pool,
            window: self.window,
        };
        let result = super::run(&self.old, &self.delta, &self.snapshot, &self.params, self.kind, writes);
        let (recorded, lanes) = (tracker.recorded, std::mem::take(&mut tracker.lanes));
        // With the indicator's trigger set, the job measures the new file's read cost, which the commit records.
        let measured = match (&result, self.params.major.indicator) {
            (Ok(report), Some(_)) => {
                let runs = super::policy::runs(self.old.path());
                let beta = self.params.major.beta;
                Some(super::policy::indicator(&next.data, report.kind, &runs, beta, &bursts))
            }
            _ => None,
        };
        drop(file);
        let result = match measured {
            Some(Err(e)) => Err(e),
            Some(Ok(m)) => result.map(|r| (r, Some(m))),
            None => result.map(|r| (r, None)),
        };

        match result {
            Ok((mut report, measured)) => {
                report.indicator = measured.map(|(_, i)| i);
                Ok(Done {
                    writer: self.writer,
                    base: Arc::clone(&self.old),
                    next: Some(next),
                    snapshot: self.snapshot,
                    report,
                    lanes,
                    cost: measured.map(|(c, _)| c),
                    warm: self.warm,
                    resume: self.resume.clone(),
                    link: Arc::clone(&self.link),
                    _lock: lock,
                })
            }
            Err(e) => {
                if e.kind() == ErrorKind::Interrupted && recorded {
                    next.keep();
                } else {
                    let _ = io::remove(&self.resume);
                }
                self.link.end();
                Err(e)
            }
        }
    }
}

/// Starts a job of `kind` with `params` for the index whose writer holds `owner`, and whose current generation the cell
/// `shared` holds. When an earlier job of that generation left a resume file, the job goes on with its data file and
/// snapshot. Returns the job and its link to the writer.
pub(crate) fn begin(
    owner: &mut Owner,
    shared: &Shared,
    params: &MaintainParams,
    kind: RunKind,
) -> Result<(Job, Arc<Link>), Error> {
    super::check(params)?;
    let old = shared.load_full();
    let delta = old.changes().load_full();
    let found = resumable(owner).filter(|(_, r)| r.fits(&delta));
    // A job that goes on from a record plans the run that the record's job asked for.
    let kind = found.as_ref().and_then(|(_, r)| kind_of(r.kind)).unwrap_or(kind);
    let (next, file, record, snapshot) = match found {
        Some((data, record)) => {
            let (next, file) = owner.reopen(&data)?;
            let snapshot = record.snapshot();
            (next, file, Some(record), snapshot)
        }
        None => {
            let (next, file) = owner.next()?;
            let snapshot = Snapshot {
                seen: delta.seen(),
                log_at: delta.log_len(),
            };
            (next, file, None, snapshot)
        }
    };
    // The job's writes would only push the searches' pages out of the cache.
    io::uncached(&file);
    let activity = shared.activity();
    activity.map();
    let link = Arc::new(Link::new());
    let warm = shared.load().is_warm();
    let job = Job {
        old,
        delta,
        snapshot,
        params: params.clone(),
        kind,
        resume: resume_path(&next.data),
        next: Some((next, file)),
        record,
        link: Arc::clone(&link),
        activity,
        writer: owner.writer(),
        lock: Some(owner.lock()),
        kept_rows: match warm {
            true => shared.load().resident_rows(),
            false => Vec::new(),
        },
        warm,
        window: 0,
    };
    Ok((job, link))
}

/// The data file and the record of the resume file that an earlier job of the current generation of `owner`'s index
/// left, if there is one and it is whole. The record names the generation's data file by its header's checksum, which
/// tells it from a file that a rebuild put in its place before the index's first commit.
fn resumable(owner: &Owner) -> Option<(PathBuf, Record)> {
    let at = owner.current();
    let check = crate::root::header_of(&at.data).ok()?.check;
    let files = fs::read_dir(crate::root::folder(owner.path())).ok()?;

    files.flatten().find_map(|entry| {
        let path = entry.path();
        if path.extension().is_none_or(|e| e != "resume") {
            return None;
        }
        let record = Record::read(&path)?;
        let data = path.with_extension("data");
        let ours = record.index_id == at.index_id() && record.base == at.generation && record.base_check == check;
        (ours && data.is_file()).then_some((data, record))
    })
}

/// The run kind that a resume file records as `kind`.
fn kind_of(kind: u8) -> Option<RunKind> {
    [RunKind::Auto, RunKind::Fold, RunKind::Minor, RunKind::Major]
        .into_iter()
        .find(|&k| kind_number(k) == kind)
}

fn kind_number(kind: RunKind) -> u8 {
    match kind {
        RunKind::Auto => 0,
        RunKind::Fold => 1,
        RunKind::Minor => 2,
        RunKind::Major => 3,
    }
}

/// The resume file of the generation whose data file is `data`, eg. `g3-<r>.resume` next to `g3-<r>.data`.
pub(crate) fn resume_path(data: &Path) -> PathBuf {
    data.with_extension("resume")
}

/// The names of the data file and the resume file of a job that `owner`'s index can go on with, if there is one,
/// which a cleanup keeps.
pub(crate) fn kept(owner: &Owner) -> Vec<String> {
    let Some((data, _)) = resumable(owner) else {
        return Vec::new();
    };
    let stem = data
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    vec![format!("{stem}.data"), format!("{stem}.resume")]
}

/// What a resume file holds: the job it belongs to, and the mark that its rewrite reached and made durable.
#[derive(Clone, Debug, PartialEq)]
struct Record {
    index_id: u64,
    /// The generation that the job rewrites, by its number and its data file's header checksum.
    base: u64,
    base_check: u32,
    /// The job's snapshot: its inserts, the sequence number of its first change not taken, and the log's length.
    inserts: u64,
    next: u64,
    log_at: u64,
    /// The run that the job asked for, the fingerprint of its plan and its rewrite's window, so a job that plans
    /// another file or writes it in other windows starts over.
    kind: u8,
    plan: u32,
    window: u64,
    mark: Mark,
}

impl Record {
    fn snapshot(&self) -> Snapshot {
        Snapshot {
            seen: Seen::at(self.inserts as usize, self.next),
            log_at: self.log_at as usize,
        }
    }

    /// Whether the record's snapshot is one of `delta`, the changes of the same generation now.
    fn fits(&self, delta: &Delta) -> bool {
        self.inserts as usize <= delta.len() && self.next <= delta.next() && self.log_at as usize <= delta.log_len()
    }

    fn encode(&self) -> Vec<u8> {
        let mut out = MAGIC.to_vec();
        let words = [
            self.index_id,
            self.base,
            self.inserts,
            self.next,
            self.log_at,
            self.window,
        ];
        words.iter().for_each(|w| out.extend_from_slice(&w.to_le_bytes()));
        out.extend_from_slice(&self.base_check.to_le_bytes());
        out.extend_from_slice(&self.plan.to_le_bytes());
        out.push(self.kind);
        let m = &self.mark;
        out.extend_from_slice(&(m.windows as u64).to_le_bytes());

        for s in &m.sections {
            out.extend_from_slice(&(s.at as u64).to_le_bytes());
            out.extend_from_slice(&s.crc.to_le_bytes());
            out.extend_from_slice(&s.len.to_le_bytes());
        }

        out.extend_from_slice(&(m.page.len() as u64).to_le_bytes());
        out.extend_from_slice(&m.page);
        let counts = [m.copied as u64, m.reencoded as u64, m.bytes_read];
        counts.iter().for_each(|w| out.extend_from_slice(&w.to_le_bytes()));
        let crc = crc32fast::hash(&out);
        out.extend_from_slice(&crc.to_le_bytes());
        out
    }

    /// The record in `bytes`, or `None` for one that is cut short or damaged, eg. by a crash while it was written.
    fn decode(bytes: &[u8]) -> Option<Record> {
        let (body, crc) = bytes.split_last_chunk::<4>()?;

        if !body.starts_with(MAGIC) || crc32fast::hash(body) != u32::from_le_bytes(*crc) {
            return None;
        }

        let mut at = MAGIC.len();
        let mut take = |n: usize| -> Option<&[u8]> {
            let b = body.get(at..at + n)?;
            at += n;
            Some(b)
        };
        let mut words = [0u64; 6];

        for w in &mut words {
            *w = take(8).map(|b| u64::from_le_bytes(b.try_into().unwrap_or_default()))?;
        }

        let [index_id, base, inserts, next, log_at, window] = words;
        let base_check = take(4).map(|b| u32::from_le_bytes(b.try_into().unwrap_or_default()))?;
        let plan = take(4).map(|b| u32::from_le_bytes(b.try_into().unwrap_or_default()))?;
        let kind = *take(1)?.first()?;
        let windows = take(8).map(|b| u64::from_le_bytes(b.try_into().unwrap_or_default()))? as usize;
        let mut sections = [Reached::default(); 3];

        for s in &mut sections {
            let at = take(8).map(|b| u64::from_le_bytes(b.try_into().unwrap_or_default()))? as usize;
            let crc = take(4).map(|b| u32::from_le_bytes(b.try_into().unwrap_or_default()))?;
            let len = take(8).map(|b| u64::from_le_bytes(b.try_into().unwrap_or_default()))?;
            *s = Reached { at, crc, len };
        }

        let page_len = take(8).map(|b| u64::from_le_bytes(b.try_into().unwrap_or_default()))? as usize;
        let page = take(page_len)?.to_vec();
        let mut counts = [0u64; 3];

        for c in &mut counts {
            *c = take(8).map(|b| u64::from_le_bytes(b.try_into().unwrap_or_default()))?;
        }

        let mark = Mark {
            windows,
            sections,
            page,
            copied: counts[0] as usize,
            reencoded: counts[1] as usize,
            bytes_read: counts[2],
        };
        Some(Record {
            index_id,
            base,
            base_check,
            inserts,
            next,
            log_at,
            kind,
            plan,
            window,
            mark,
        })
    }

    fn read(path: &Path) -> Option<Record> {
        Record::decode(&fs::read(path).ok()?)
    }

    /// Writes the record over the resume file at `path` and syncs it. The first time it also syncs the folder, so the
    /// names of the resume file and the data file survive a power cut.
    fn write(&self, path: &Path, first: bool) -> Result<(), Error> {
        let bytes = self.encode();
        let file = match first {
            true => {
                let _ = io::remove(path);
                io::create_new(path)?
            }
            false => fs::OpenOptions::new().write(true).open(path)?,
        };
        file.set_len(bytes.len() as u64)?;
        io::write_all_at(&file, &bytes, 0, path)?;
        io::sync_file(&file, path)?;

        if first {
            io::sync_parent(path)?;
        }

        Ok(())
    }
}

/// A job's new data file, written in bursts that the throttle gates, and its reads of the old file, gated the same way.
struct Bursts<'a> {
    file: &'a File,
    /// A handle of the same file that writes through the system's cache, for what a warm index held in memory, and
    /// which of the old file's row pages that was.
    cached: Option<&'a File>,
    kept_rows: &'a [bool],
    path: &'a Path,
    params: Option<&'a ThrottleParams>,
    throttle: &'a Throttle,
    activity: &'a Activity,
    cancel: &'a AtomicBool,
    burst: Mutex<Burst>,
    /// Every byte written so far.
    written: AtomicU64,
}

/// The burst under way, if any.
#[derive(Default)]
struct Burst {
    open: bool,
    /// When it started, and the bytes it wrote.
    started: u64,
    bytes: usize,
    /// When the last burst ended.
    ended: u64,
}

impl<'a> Bursts<'a> {
    fn new(
        file: &'a File,
        path: &'a Path,
        throttle: &'a Throttle,
        activity: &'a Activity,
        cancel: &'a AtomicBool,
    ) -> Bursts<'a> {
        Bursts {
            file,
            cached: None,
            kept_rows: &[],
            path,
            params: throttle.params.as_ref(),
            throttle,
            activity,
            cancel,
            burst: Mutex::new(Burst::default()),
            written: AtomicU64::new(0),
        }
    }

    fn check_cancel(&self) -> Result<(), Error> {
        match self.cancel.load(Ordering::Relaxed) {
            true => Err(Error::new(ErrorKind::Interrupted, "the maintenance job was cancelled")),
            false => Ok(()),
        }
    }

    /// How long no search ran, in nanoseconds, and whether the job may work as when the index is idle.
    fn quiet_for(&self, params: &ThrottleParams) -> (u64, bool) {
        let quiet = io::now_ns().saturating_sub(self.activity.last());
        let idle = self.throttle.background.load(Ordering::Relaxed) || quiet >= params.idle.as_nanos() as u64;
        (quiet, idle)
    }

    /// Waits until a write of `len` bytes may go out: in the burst under way while no search came since it started and
    /// it has room, or else in a new burst once the last one is synced and no search ran for `quiet`, `spacing` after
    /// it ended. Without a throttle every write goes out at once.
    fn admit(&self, len: usize) -> Result<(), Error> {
        self.check_cancel()?;
        let Some(params) = self.params else {
            return Ok(());
        };
        let mut burst = locked(&self.burst);

        loop {
            self.check_cancel()?;
            let (quiet, idle) = self.quiet_for(params);
            let now = io::now_ns();

            if burst.open {
                let room = if idle {
                    params.idle_burst_bytes
                } else {
                    params.burst_bytes
                };
                let searched = self.activity.last() > burst.started;
                if !searched && burst.bytes + len <= room.max(len) {
                    return Ok(());
                }
                self.end(&mut burst, params.sync)?;
                continue;
            }

            let spacing = if idle { 0 } else { params.spacing.as_nanos() as u64 };
            if quiet >= params.quiet.as_nanos() as u64 && now.saturating_sub(burst.ended) >= spacing {
                (burst.open, burst.started, burst.bytes) = (true, now, 0);
                return Ok(());
            }

            std::thread::sleep(POLL);
        }
    }

    /// Ends the burst under way with a sync of `kind`.
    fn end(&self, burst: &mut Burst, kind: SyncKind) -> Result<(), Error> {
        match kind {
            SyncKind::Full => io::sync_file(self.file, self.path)?,
            SyncKind::Barrier => io::barrier_sync(self.file, self.path)?,
        }
        (burst.open, burst.ended) = (false, io::now_ns());
        Ok(())
    }

    /// Makes every write so far survive a power cut, eg. before a mark, and ends the burst under way.
    fn sync_all(&self) -> Result<(), Error> {
        let mut burst = locked(&self.burst);
        io::sync_file(self.file, self.path)?;
        (burst.open, burst.ended) = (false, io::now_ns());
        Ok(())
    }
}

impl Bursts<'_> {
    /// Writes all of `bytes` at `offset` through `file`, a handle of the job's file, in pieces that the throttle
    /// admits.
    fn write_through(&self, file: &File, bytes: &[u8], offset: usize) -> Result<(), Error> {
        // A write never takes more than a burst, so a burst holds no more than the throttle allows.
        let most = self.params.map_or(PIECE, |p| PIECE.min(p.burst_bytes.max(1)));

        for (k, piece) in bytes.chunks(most).enumerate() {
            self.admit(piece.len())?;
            io::write_all_at(file, piece, (offset + k * most) as u64, self.path)?;
            locked(&self.burst).bytes += piece.len();
            self.written.fetch_add(piece.len() as u64, Ordering::Relaxed);
        }

        Ok(())
    }
}

impl Sink for Bursts<'_> {
    fn write_at(&self, bytes: &[u8], offset: usize) -> Result<(), Error> {
        self.write_through(self.file, bytes, offset)
    }

    fn write_kept_at(&self, bytes: &[u8], offset: usize) -> Result<(), Error> {
        self.write_through(self.cached.unwrap_or(self.file), bytes, offset)
    }
}

impl Out for Bursts<'_> {
    fn kept_rows(&self) -> Option<&[bool]> {
        self.cached.map(|_| self.kept_rows)
    }

    fn set_len(&self, len: u64) -> Result<(), Error> {
        let has = self.file.metadata()?.len();

        if let Some(params) = self.params {
            let dir = self.path.parent().unwrap_or(Path::new("."));
            let more = len.saturating_sub(has);
            let needed = more.saturating_add(len / 10).saturating_add(params.reserve_bytes);
            if more > 0 && io::free_bytes(dir) < needed {
                return Err(Error::new(
                    ErrorKind::StorageFull,
                    "the disk lacks the free space that the maintenance job needs",
                ));
            }
        }

        match has == len {
            true => Ok(()),
            false => io::preallocate(self.file, len),
        }
    }

    fn order(&self) -> Result<(), Error> {
        let Some(params) = self.params else {
            return self.file.sync_data();
        };
        self.admit(0)?;
        let mut burst = locked(&self.burst);
        self.end(&mut burst, params.sync)
    }

    fn sync(&self) -> Result<(), Error> {
        self.admit(0)?;
        self.sync_all()
    }
}

impl Gate for Bursts<'_> {
    fn read(&self, _: usize) -> Result<(), Error> {
        self.check_cancel()?;
        let Some(params) = self.params else {
            return Ok(());
        };

        while self.quiet_for(params).0 < params.quiet.as_nanos() as u64 {
            self.check_cancel()?;
            std::thread::sleep(POLL);
        }

        Ok(())
    }
}

/// What a job's rewrite tells it: the plan, after which it shares the next generation's centers and measures the
/// inserts since the snapshot, and each mark, which it records every so many bytes.
struct Tracker<'a> {
    job: &'a Job,
    bursts: &'a Bursts<'a>,
    /// What the resume file held when the job began, until the plan shows whether it still holds, and then its mark.
    record: Option<Record>,
    start: Option<Mark>,
    plan: u32,
    /// The bytes written when the job last recorded a mark, and whether the resume file holds one of this plan.
    marked: u64,
    recorded: bool,
    lanes: Vec<Lane>,
}

impl Progress for Tracker<'_> {
    fn start(&self) -> Option<Mark> {
        self.start.clone()
    }

    fn planned(&mut self, plan: &Plan, frame: &InMemory) -> Result<(), Error> {
        self.plan = plan.fingerprint();
        let same = |r: &Record| r.plan == self.plan && r.window == self.job.window as u64;
        self.start = self.record.take().filter(same).map(|r| r.mark);
        self.recorded = self.start.is_some();
        // From here on the writer measures each insert for both generations, and the job measures the ones before.
        let link = &self.job.link;
        let _ = link.centers.set(NextCenters::of(frame));
        let centers = link.centers.get().map(NextCenters::centers);
        self.lanes = match centers {
            Some(centers) => lanes_since(&self.job.old, &self.job.snapshot, centers)?,
            None => Vec::new(),
        };
        Ok(())
    }

    fn reached(&mut self, mark: &Mark) -> Result<(), Error> {
        let (written, every) = (
            self.bursts.written.load(Ordering::Relaxed),
            self.bursts.params.map_or(u64::MAX, |p| p.resume_bytes.max(1)),
        );

        if written.saturating_sub(self.marked) < every {
            return Ok(());
        }

        let (job, at) = (self.job, self.job.old.located());
        let record = Record {
            index_id: at.index_id(),
            base: at.generation,
            base_check: job.old.layout().check,
            inserts: job.snapshot.seen.inserts() as u64,
            next: job.snapshot.seen.next(),
            log_at: job.snapshot.log_at as u64,
            kind: kind_number(job.kind),
            plan: self.plan,
            window: job.window as u64,
            mark: mark.clone(),
        };
        self.bursts.sync_all()?;
        record.write(&job.resume, !self.recorded)?;
        (self.marked, self.recorded) = (written, true);
        Ok(())
    }
}

/// The lanes in the next generation, whose centers are `centers`, of the inserts that `old` holds past `snapshot`: each
/// one's row decoded and routed to its nearest center, and with the cluster anchor its code measured again from that
/// center. A row that doesn't decode gets no lane, so the commit fails on it.
fn lanes_since(old: &Inner, snapshot: &Snapshot, centers: Centers) -> Result<Vec<Lane>, Error> {
    let (dim, mean) = (old.dim(), old.mean());
    let now = old.changes().load_full();
    let mut ctx = SearchCtx::new(Limits {
        max_k: 1,
        max_nprobe: 1,
        max_rerank: 1,
    });
    let (mut o, mut center) = (vec![0f32; dim], vec![0f32; dim]);
    let mut encoder = Encoder::new(old.rotation(), mean);
    let mut lanes = Vec::new();

    for i in snapshot.seen.inserts()..now.len() {
        sq8::decode(now.row(i), mean, &mut o);
        if !o.iter().all(|x| x.is_finite()) {
            continue;
        }
        let list = old.route_among(&o, centers, &mut ctx)?;
        let (code, factors) = match old.layout().anchor {
            Anchor::Mean => {
                let (code, factors) = now.lane(i);
                (code.to_vec(), factors)
            }
            Anchor::Cluster => {
                let (codes, f) = centers.center(list);
                sq4::decode(codes, f, &mut center);
                let mut code = vec![0u8; code_bytes(padded_dim(dim))];
                let factors = encoder.encode_from(&o, &center, &mut code);
                (code, factors)
            }
        };
        lanes.push(Lane {
            seq: now.seq(i),
            list,
            code,
            factors,
        });
    }

    Ok(lanes)
}

#[cfg(test)]
#[path = "../../tests/unit/maintain/job.rs"]
mod tests;

#[cfg(test)]
#[path = "../../tests/perf/job.rs"]
mod perf;
