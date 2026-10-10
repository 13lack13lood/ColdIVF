//! Unit tests for the root and the commit: crashes and power cuts at every step, failed syncs and writes, damaged
//! slots, indexes from before the root, and writers that compete.

use std::collections::{HashMap, HashSet};
use std::ffi::OsString;
use std::io::ErrorKind;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use super::*;
use crate::index::Inner;
use crate::io::watch::{self, Op};
use crate::maintain::Live;
use crate::maintain::tests::{DIM, build_params, churn, settings};
use crate::quantization::SplitMix64;
use crate::quantization::sq8;
use crate::quantization::testutil::{Rows, clustered, scratch_dir};
use crate::{Anchor, Builder, Index, Metric, RunKind, Throttle, ThrottleParams, Writer};

/// A folder of a test's own, by the name that the file system gives it, so it matches the paths of the index's files.
/// Dropping it removes it.
struct Folder(PathBuf);

impl Folder {
    fn new(name: &str) -> Folder {
        let dir = scratch_dir().join(format!("coldivf-{}-{name}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        Folder(fs::canonicalize(&dir).unwrap())
    }

    fn index(&self) -> PathBuf {
        self.0.join("x.coldivf")
    }
}

impl Drop for Folder {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// An index of 1500 vectors with ids 0 to 1499 in `scratch`, and its vectors.
fn built(scratch: &Folder) -> crate::quantization::testutil::Rows {
    let (rows, _) = clustered(&mut SplitMix64::new(41), 1500, DIM, 15, 0.6);
    Builder::new(build_params(Metric::L2, Anchor::Cluster))
        .build(&rows, &scratch.index())
        .unwrap();
    rows
}

/// The ids that the index at `path` holds now, sorted, as a reader that opens it sees them.
fn ids_at(path: &Path) -> Result<Vec<u64>, Error> {
    let inner = Inner::open(path)?;
    let delta = inner.changes().load();
    let (live, _) = Live::of(&inner, &delta, delta.seen())?;
    let (layout, data) = (inner.layout(), inner.in_place());
    let rows = (0..layout.count as u32).filter(|&r| live.rows[r as usize]);
    let mut ids: Vec<u64> = rows.map(|r| sq8::id(data.rows.row(r), layout.dim)).collect();
    ids.extend(live.inserts.iter().map(|&i| sq8::id(delta.row(i as usize), layout.dim)));
    ids.sort_unstable();
    Ok(ids)
}

/// The row that a search reads for each id that the index at `path` holds now, by id.
fn rows_at(path: &Path) -> HashMap<u64, Vec<u8>> {
    let inner = Inner::open(path).unwrap();
    let delta = inner.changes().load();
    let (live, _) = Live::of(&inner, &delta, delta.seen()).unwrap();
    let (layout, data) = (inner.layout(), inner.in_place());
    let rows = (0..layout.count as u32).filter(|&r| live.rows[r as usize]);
    let mut out: HashMap<u64, Vec<u8>> = rows
        .map(|r| (sq8::id(data.rows.row(r), layout.dim), data.rows.row(r).to_vec()))
        .collect();
    out.extend(live.inserts.iter().map(|&i| {
        let row = delta.row(i as usize);
        (sq8::id(row, layout.dim), row.to_vec())
    }));
    out
}

/// A throttle that waits for no search, records a mark after every window and keeps no space free.
fn eager_throttle() -> Throttle {
    Throttle::new(ThrottleParams {
        quiet: Duration::ZERO,
        spacing: Duration::ZERO,
        burst_bytes: 64 << 10,
        resume_bytes: 1,
        reserve_bytes: 0,
        ..ThrottleParams::default()
    })
}

/// The files of the index at `path` in its folder, by name.
fn names_in(dir: &Path) -> HashSet<OsString> {
    fs::read_dir(dir)
        .map(|d| d.flatten().map(|e| e.file_name()).collect())
        .unwrap_or_default()
}

/// Copies the files of the folder `from`, and of its index's folder, into `to`, as they are.
fn copy_files(from: &Path, to: &Path) {
    for dir in [from.to_path_buf(), folder(&from.join("x.coldivf"))] {
        let rel = dir.strip_prefix(from).unwrap();
        fs::create_dir_all(to.join(rel)).unwrap();
        for name in names_in(&dir) {
            let p = dir.join(&name);
            if p.is_file() {
                fs::copy(&p, to.join(rel).join(&name)).unwrap();
            }
        }
    }
}

/// What a power cut would leave of a test's folder and its index's folder: each folder's entries as of its last sync,
/// and each file's bytes as of its last sync. A file that was never synced comes back empty.
struct Durable {
    root: PathBuf,
    entries: HashMap<PathBuf, HashSet<OsString>>,
    bytes: HashMap<PathBuf, Vec<u8>>,
    /// A sync that has started, which counts once the next operation starts.
    pending: Option<(PathBuf, bool)>,
}

impl Durable {
    /// Everything in `root` and its index's folder as it is now.
    fn now(root: &Path) -> Durable {
        let mut d = Durable {
            root: root.to_path_buf(),
            entries: HashMap::new(),
            bytes: HashMap::new(),
            pending: None,
        };
        for dir in [root.to_path_buf(), folder(&root.join("x.coldivf"))] {
            d.entries.insert(dir.clone(), names_in(&dir));
            for name in names_in(&dir) {
                if let Ok(b) = fs::read(dir.join(&name)) {
                    d.bytes.insert(dir.join(name), b);
                }
            }
        }
        d
    }

    /// Counts the sync that started before `op`, then notes `op` if it is a sync.
    fn see(&mut self, op: Op, path: &Path) {
        if let Some((p, is_folder)) = self.pending.take() {
            match is_folder {
                true => {
                    self.entries.insert(p.clone(), names_in(&p));
                }
                false => {
                    if let Ok(b) = fs::read(&p) {
                        self.bytes.insert(p, b);
                    }
                }
            }
        }

        match op {
            Op::Sync => self.pending = Some((path.to_path_buf(), false)),
            Op::SyncFolder => self.pending = Some((path.to_path_buf(), true)),
            _ => {}
        }
    }

    /// Writes what a power cut now would leave into `to`.
    fn image(&self, to: &Path) {
        for (dir, names) in &self.entries {
            let rel = dir.strip_prefix(&self.root).unwrap();
            let listed = |d: &Path| {
                let (parent, name) = (d.parent().unwrap(), d.file_name().unwrap());
                self.entries.get(parent).is_some_and(|n| n.contains(name))
            };
            if *dir != self.root && !listed(dir) {
                continue;
            }
            fs::create_dir_all(to.join(rel)).unwrap();
            for name in names {
                let p = dir.join(name);
                if self.entries.contains_key(&p) {
                    continue;
                }
                let bytes = self.bytes.get(&p).cloned().unwrap_or_default();
                fs::write(to.join(rel).join(name), bytes).unwrap();
            }
        }
    }
}

/// The images that a run left, as (what made it, its folder).
type Images = Arc<Mutex<Vec<(String, PathBuf)>>>;

/// Watches `scratch` while `f` runs and takes two images of it before every file operation: the files as they are,
/// which a crash of the process leaves, and what a power cut would leave. The images go in a folder of their own.
fn imaged(scratch: &Folder, images: &Folder, f: impl FnOnce()) -> Vec<(String, PathBuf)> {
    imaged_counting(scratch, images, &Arc::default(), f)
}

/// The same, counting the operations in `ops`, so `f` can tell which images come after a step of its own. The images
/// before operation k are in the folders `k-crash` and `k-cut`.
fn imaged_counting(
    scratch: &Folder,
    images: &Folder,
    ops: &Arc<AtomicUsize>,
    f: impl FnOnce(),
) -> Vec<(String, PathBuf)> {
    let taken: Images = Arc::default();
    let durable = Arc::new(Mutex::new(Durable::now(&scratch.0)));
    let (from, to, list) = (scratch.0.clone(), images.0.clone(), Arc::clone(&taken));
    let ops = Arc::clone(ops);
    let guard = watch::set(&scratch.0, move |op, path| {
        let count = ops.fetch_add(1, Ordering::SeqCst) + 1;
        let what = format!("{op:?} {}", path.file_name().unwrap_or_default().to_string_lossy());
        // A panic here would poison the watch for every other test, so it becomes an image that fails the check.
        let taken = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut d = durable.lock().unwrap_or_else(PoisonError::into_inner);
            d.see(op, path);
            let (crash, cut) = (to.join(format!("{count}-crash")), to.join(format!("{count}-cut")));
            copy_files(&from, &crash);
            d.image(&cut);
            [
                (format!("crash before {what}"), crash),
                (format!("power cut before {what}"), cut),
            ]
        }));
        let mut list = list.lock().unwrap_or_else(PoisonError::into_inner);
        match taken {
            Ok(images) => list.extend(images),
            Err(_) => list.push((format!("no image before {what}"), PathBuf::new())),
        }
        Ok(())
    });
    f();
    drop(guard);
    taken.lock().unwrap_or_else(PoisonError::into_inner).clone()
}

/// Checks that every image opens as an index that holds exactly `want`, for a reader and for a writer, and that a
/// writer's open leaves only the current generation's files.
fn check_images(images: &[(String, PathBuf)], want: &[u64]) {
    assert!(!images.is_empty(), "the run made no file operation");

    for (what, dir) in images {
        let path = dir.join("x.coldivf");
        let ids = ids_at(&path).unwrap_or_else(|e| panic!("{what}: the index doesn't open: {e}"));
        assert!(ids == want, "{what}: the index holds other vectors");
        let writer = Writer::open(&path).unwrap_or_else(|e| panic!("{what}: a writer doesn't open: {e}"));
        let at = writer.index.inner().located().clone();
        let left: Vec<u64> = generation_files(&folder(&path))
            .unwrap()
            .into_iter()
            .filter(|(name, _)| at.root.as_ref().is_none_or(|r| *name != r.data && *name != r.log))
            .map(|(_, g)| g)
            .collect();
        assert!(left.is_empty(), "{what}: the writer left generations {left:?}");
        drop(writer);
        assert!(
            ids_at(&path).unwrap() == want,
            "{what}: the writer's open changed the vectors"
        );
        fs::remove_dir_all(dir).unwrap();
    }
}

#[test]
fn a_crash_or_a_power_cut_at_any_step_of_a_commit_loses_no_synced_change() {
    let (scratch, images) = (Folder::new("crash-steps"), Folder::new("crash-steps-images"));
    let rows = built(&scratch);
    let mut writer = Writer::open(&scratch.index()).unwrap();

    // The first commit makes the folder and deletes the index file and its log, and a later one replaces a generation.
    for round in 0..2 {
        churn(&mut writer, &rows, 50 + round);
        writer.sync().unwrap();
        let want = ids_at(&scratch.index()).unwrap();
        let taken = imaged(&scratch, &images, || {
            writer.maintain(&settings(), RunKind::Fold).unwrap();
        });
        assert_eq!(writer.index.inner().located().generation, round + 1);
        check_images(&taken, &want);
        assert!(ids_at(&scratch.index()).unwrap() == want);
    }
}

#[test]
fn a_crash_or_a_power_cut_at_any_step_of_a_job_loses_no_synced_change_and_no_insert_made_during_it() {
    let (scratch, images) = (Folder::new("crash-job"), Folder::new("crash-job-images"));
    let rows = built(&scratch);
    let mut writer = Writer::open(&scratch.index()).unwrap();
    writer.compact().unwrap();
    churn(&mut writer, &rows, 57);
    writer.sync().unwrap();
    let base = ids_at(&scratch.index()).unwrap();
    let throttle = Throttle::new(ThrottleParams {
        quiet: Duration::ZERO,
        spacing: Duration::ZERO,
        burst_bytes: 64 << 10,
        resume_bytes: 1,
        reserve_bytes: 0,
        ..ThrottleParams::default()
    });
    // Each insert during the job, with the operations done once its record was written, and once it was synced.
    let (ops, mut inserted) = (Arc::new(AtomicUsize::new(0)), Vec::new());

    let taken = imaged_counting(&scratch, &images, &ops, || {
        let mut insert = |writer: &mut Writer, i: usize| {
            let v: Vec<f32> = rows.data[i * DIM..(i + 1) * DIM].iter().map(|&x| x + 0.2).collect();
            writer.insert(5_000_000 + i as u64, &v).unwrap();
            let written = ops.load(Ordering::SeqCst);
            writer.sync().unwrap();
            inserted.push((5_000_000 + i as u64, written, ops.load(Ordering::SeqCst)));
        };
        // Inserts before the plan and after it, while the job writes in many small bursts and windows.
        let mut job = writer.begin(&settings(), RunKind::Fold).unwrap();
        job.window = 16 << 10;
        (0..3).for_each(|i| insert(&mut writer, i));
        let done = job.run(&throttle).unwrap();
        (3..6).for_each(|i| insert(&mut writer, i));
        writer.finish_compact(done).unwrap();
    });
    assert!(taken.len() > 40, "the job made only {} images", taken.len());

    // Image k holds every insert whose record was written before operation k, for a crash, or synced before it, for a
    // power cut. A writer's open keeps only the current generation and a job it can go on with.
    for (what, dir) in &taken {
        let name = dir.file_name().unwrap().to_string_lossy().into_owned();
        let (k, kind) = name.split_once('-').unwrap();
        let k: usize = k.parse().unwrap();
        let mut want = base.clone();
        let done = |&(_, written, synced): &(u64, usize, usize)| k > if kind == "crash" { written } else { synced };
        want.extend(inserted.iter().filter(|x| done(x)).map(|x| x.0));
        want.sort_unstable();
        let path = dir.join("x.coldivf");
        let ids = ids_at(&path).unwrap_or_else(|e| panic!("{what}: the index doesn't open: {e}"));
        assert!(ids == want, "{what}: the index holds other vectors");
        let mut writer = Writer::open(&path).unwrap_or_else(|e| panic!("{what}: a writer doesn't open: {e}"));
        let (at, kept) = (
            writer.index.inner().located().clone(),
            crate::maintain::job::kept(&writer.root),
        );
        let left: Vec<String> = generation_files(&folder(&path))
            .unwrap()
            .into_iter()
            .map(|(name, _)| name)
            .filter(|name| at.root.as_ref().is_none_or(|r| *name != r.data && *name != r.log) && !kept.contains(name))
            .collect();
        assert!(left.is_empty(), "{what}: the writer left {left:?}");
        assert!(
            ids_at(&path).unwrap() == want,
            "{what}: the writer's open changed the vectors"
        );

        // A job that the image lets the next one go on with writes a whole file from where it got.
        if !kept.is_empty() {
            let mut job = writer.begin_compact(&settings()).unwrap().unwrap();
            assert!(job.resumes(), "{what}: the next job started over");
            job.window = 16 << 10;
            let done = job.run(&throttle).unwrap();
            writer.finish_compact(done).unwrap();
            writer
                .index
                .verify()
                .unwrap_or_else(|e| panic!("{what}: the job that went on wrote a damaged file: {e}"));
            assert!(
                ids_at(&path).unwrap() == want,
                "{what}: the job that went on lost vectors"
            );
        }

        drop(writer);
        fs::remove_dir_all(dir).unwrap();
    }
}

#[test]
fn a_torn_slot_leaves_the_generation_before_it_current() {
    let (scratch, images) = (Folder::new("torn"), Folder::new("torn-images"));
    let rows = built(&scratch);
    let mut writer = Writer::open(&scratch.index()).unwrap();
    writer.compact().unwrap();
    churn(&mut writer, &rows, 52);
    writer.sync().unwrap();
    let want = ids_at(&scratch.index()).unwrap();
    let taken = imaged(&scratch, &images, || writer.compact().unwrap());
    let slot = folder(&scratch.index()).join(SLOTS[0]);
    let new_slot = fs::read(&slot).unwrap();
    // The image taken just before the write of the slot that commits.
    let before = taken
        .iter()
        .find(|(what, _)| what.starts_with("crash before Write root.0"))
        .unwrap();

    // A write torn inside the record, before its checksum, or right before it.
    for torn in [100, 178] {
        let path = before.1.join("x.coldivf");
        let image_slot = folder(&path).join(SLOTS[0]);
        let mut bytes = fs::read(&image_slot).unwrap();
        bytes[..torn].copy_from_slice(&new_slot[..torn]);
        fs::write(&image_slot, &bytes).unwrap();
        assert_eq!(locate(&path).unwrap().generation, 1, "torn at {torn}");
        assert!(ids_at(&path).unwrap() == want, "torn at {torn}");
    }
}

#[test]
fn a_damaged_newer_slot_is_made_again_from_the_newest_whole_generation() {
    let scratch = Folder::new("rot");
    let rows = built(&scratch);
    let mut writer = Writer::open(&scratch.index()).unwrap();
    writer.compact().unwrap();
    churn(&mut writer, &rows, 53);
    writer.compact().unwrap();
    churn(&mut writer, &rows, 54);
    writer.sync().unwrap();
    let want = ids_at(&scratch.index()).unwrap();
    drop(writer);
    // Generation 2 is current, in slot 0, and generation 1's files are gone. A run of generation 3 left its data file
    // and a log that lacks generation 2's last changes, as a crash before its commit leaves them. Then a flipped bit
    // breaks slot 0's checksum.
    let at = locate(&scratch.index()).unwrap();
    let root = at.root.clone().unwrap();
    let stem = folder(&scratch.index()).join("g3-00000000000000ab");
    fs::copy(&at.data, stem.with_extension("data")).unwrap();
    let short = log::Version::Two {
        index_id: root.index_id,
        generation: 3,
        base: root.base,
    }
    .header(&header_of(&at.data).unwrap());
    fs::write(stem.with_extension("log"), &short).unwrap();
    let slot = folder(&scratch.index()).join(SLOTS[0]);
    let mut bytes = fs::read(&slot).unwrap();
    bytes[30] ^= 1;
    fs::write(&slot, &bytes).unwrap();
    assert!(format::read_root(&bytes).is_err());

    assert_eq!(locate(&scratch.index()).unwrap().generation, 2);
    assert!(ids_at(&scratch.index()).unwrap() == want);
    let writer = Writer::open(&scratch.index()).unwrap();
    assert_eq!(format::read_root(&fs::read(&slot).unwrap()).unwrap().generation, 2);
    drop(writer);
    assert!(ids_at(&scratch.index()).unwrap() == want);
}

#[test]
fn a_failed_sync_or_write_before_the_commit_leaves_the_index_as_it_was() {
    let scratch = Folder::new("faults");
    let rows = built(&scratch);
    let mut writer = Writer::open(&scratch.index()).unwrap();
    churn(&mut writer, &rows, 55);
    writer.compact().unwrap();
    let late: Vec<f32> = rows.data[..DIM].iter().map(|&x| x + 0.3).collect();

    // Each fails the first operation of its kind on a file whose name ends so: the new data file's sync, the new log's
    // write and sync, the folder's sync, and the old log's write as a full disk fails it.
    let faults: [(Op, &str, ErrorKind); 5] = [
        (Op::Sync, ".data", ErrorKind::Other),
        (Op::Write, ".log", ErrorKind::StorageFull),
        (Op::Sync, ".log", ErrorKind::Other),
        (Op::SyncFolder, ".d", ErrorKind::Other),
        (Op::Create, ".log", ErrorKind::StorageFull),
    ];

    for (i, (op, end, kind)) in faults.into_iter().enumerate() {
        let id = 70_000 + i as u64;
        writer.insert(id, &late).unwrap();
        let (generation, want) = (
            writer.index.inner().located().generation,
            ids_at(&scratch.index()).unwrap(),
        );
        let mut failed = false;
        let guard = watch::set(&scratch.0, move |seen, path| {
            let hit = seen == op && path.to_string_lossy().ends_with(end) && !failed;
            failed |= hit;
            match hit {
                true => Err(Error::new(kind, "a test fault")),
                false => Ok(()),
            }
        });
        let err = writer.compact().unwrap_err();
        drop(guard);
        assert_eq!(err.kind(), kind, "{op:?} {end}: {err}");
        assert_eq!(locate(&scratch.index()).unwrap().generation, generation, "{op:?} {end}");
        assert!(ids_at(&scratch.index()).unwrap() == want, "{op:?} {end}");
        // The run's files are gone, and the writer goes on.
        assert_eq!(
            generation_files(&folder(&scratch.index())).unwrap().len(),
            2,
            "{op:?} {end}"
        );
        writer.delete(id).unwrap();
        writer.insert(id, &late).unwrap();
    }

    // A full disk fails an insert too, and the writer cuts off what landed and goes on.
    let guard = watch::set(&scratch.0, |op, _| match op {
        Op::Write => Err(Error::new(ErrorKind::StorageFull, "a test fault")),
        _ => Ok(()),
    });
    assert_eq!(writer.insert(80_000, &late).unwrap_err().kind(), ErrorKind::StorageFull);
    drop(guard);
    writer.insert(80_001, &late).unwrap();
    writer.compact().unwrap();
    let ids = ids_at(&scratch.index()).unwrap();
    assert!(ids.contains(&80_001) && !ids.contains(&80_000));
}

#[test]
fn a_failed_slot_write_takes_no_change_until_the_writer_opens_again() {
    for op in [Op::Write, Op::Sync] {
        let scratch = Folder::new(&format!("slot-{op:?}"));
        let rows = built(&scratch);
        let mut writer = Writer::open(&scratch.index()).unwrap();
        churn(&mut writer, &rows, 56);
        writer.compact().unwrap();
        churn(&mut writer, &rows, 57);
        writer.sync().unwrap();
        let want = ids_at(&scratch.index()).unwrap();
        let guard = watch::set(&scratch.0, move |seen, path| {
            match seen == op && path.file_name().is_some_and(|n| n == SLOTS[0]) {
                true => Err(Error::other("a test fault")),
                false => Ok(()),
            }
        });
        assert!(writer.compact().is_err(), "{op:?}");
        drop(guard);
        // Whether the slot write landed is unknown, so the writer takes nothing more, and the open that follows finds
        // the generation that is current, whichever it is, with every change.
        assert!(writer.insert(90_000, &rows.data[..DIM]).is_err(), "{op:?}");
        assert!(writer.delete(3).is_err(), "{op:?}");
        assert!(writer.compact().is_err(), "{op:?}");
        drop(writer);
        assert!(ids_at(&scratch.index()).unwrap() == want, "{op:?}");
        let mut writer = Writer::open(&scratch.index()).unwrap();
        writer.insert(90_000, &rows.data[..DIM]).unwrap();
        writer.compact().unwrap();
        assert!(ids_at(&scratch.index()).unwrap().contains(&90_000), "{op:?}");
    }
}

#[test]
fn an_index_from_before_the_root_opens_and_its_first_commit_moves_it_into_its_folder() {
    let scratch = Folder::new("legacy");
    let rows = built(&scratch);
    let path = scratch.index();
    let mut writer = Writer::open(&path).unwrap();
    churn(&mut writer, &rows, 58);
    drop(writer);
    // The index file and its log are those of a version without a root.
    assert!(!folder(&path).exists() && update::log_path(&path).exists());
    let want = ids_at(&path).unwrap();
    assert_eq!(locate(&path).unwrap(), legacy(&path));

    let mut writer = Writer::open(&path).unwrap();
    writer.compact().unwrap();
    assert!(!path.exists() && !update::log_path(&path).exists());
    let at = locate(&path).unwrap();
    assert_eq!(at.generation, 1);
    assert!(at.data.starts_with(folder(&path)) && at.log.starts_with(folder(&path)));
    assert!(ids_at(&path).unwrap() == want);
    // Changes after it go to the generation's log, and a reader that opens the index later reads them there.
    writer.insert(60_000, &rows.data[..DIM]).unwrap();
    writer.delete(5).unwrap();
    let ids = ids_at(&path).unwrap();
    assert!(ids.contains(&60_000) && !ids.contains(&5));
    assert!(Index::open(&path).unwrap().verify().is_ok());
}

#[test]
fn only_one_writer_or_build_changes_an_index_at_a_time() {
    let scratch = Folder::new("two");
    let rows = built(&scratch);
    let path = scratch.index();
    let mut writer = Writer::open(&path).unwrap();
    writer.compact().unwrap();
    assert_eq!(
        Writer::open(&path).map(|_| ()).unwrap_err().kind(),
        ErrorKind::WouldBlock
    );
    let build = Builder::new(build_params(Metric::L2, Anchor::Cluster)).build(&rows, &path);
    assert_eq!(build.unwrap_err().kind(), ErrorKind::WouldBlock);
    // The failed build left nothing behind.
    assert!(crate::quantization::testutil::temps_of(&path).is_empty());
    drop(writer);

    // Once the index has a root, a build commits its file as the next generation, with no changes, and deletes the old
    // generation's files.
    let mut writer = Writer::open(&path).unwrap();
    writer.insert(60_000, &rows.data[..DIM]).unwrap();
    let old = writer.index.inner().located().clone();
    drop(writer);
    Builder::new(build_params(Metric::L2, Anchor::Cluster))
        .build(&rows, &path)
        .unwrap();
    let at = locate(&path).unwrap();
    assert_eq!(at.generation, old.generation + 1);
    assert!(!old.data.exists() && !old.log.exists());
    assert_eq!(ids_at(&path).unwrap(), (0..1500).collect::<Vec<u64>>());
}

#[test]
fn a_writer_s_open_deletes_what_no_slot_names_and_nothing_else() {
    let scratch = Folder::new("clean");
    built(&scratch);
    let path = scratch.index();
    let mut writer = Writer::open(&path).unwrap();
    writer.compact().unwrap();
    writer.compact().unwrap();
    let at = writer.index.inner().located().clone();
    drop(writer);
    let dir = folder(&path);
    let junk = [
        "g7-00000000000000ab.data",
        "g7-00000000000000ab.log",
        "g1-0123456789abcdef.log",
        "g3-00000000000000cd.resume",
    ];
    let kept = ["notes.txt", "g7-short.data", "g7-00000000000000ab.data.bak"];

    for name in junk.iter().chain(&kept) {
        fs::write(dir.join(name), b"x").unwrap();
    }

    drop(Writer::open(&path).unwrap());
    let mut want: HashSet<OsString> = kept.iter().map(OsString::from).collect();
    want.extend([SLOTS[0], SLOTS[1], "activity", "runs"].map(OsString::from));
    want.extend([at.data, at.log].map(|p| p.file_name().unwrap().to_os_string()));
    assert_eq!(names_in(&dir), want);
}

#[test]
fn a_missing_or_foreign_log_of_the_current_generation_fails_with_invalid_data() {
    let scratch = Folder::new("nolog");
    let rows = built(&scratch);
    let path = scratch.index();
    let mut writer = Writer::open(&path).unwrap();
    churn(&mut writer, &rows, 59);
    writer.compact().unwrap();
    writer.insert(60_000, &rows.data[..DIM]).unwrap();
    let at = writer.index.inner().located().clone();
    drop(writer);
    let log = fs::read(&at.log).unwrap();

    // A log whose header names another generation, then no log at all.
    let mut foreign = log.clone();
    foreign[20] ^= 1;
    let crc = crc32fast::hash(&foreign[..log::HEADER_2_BYTES - 4]);
    foreign[log::HEADER_2_BYTES - 4..log::HEADER_2_BYTES].copy_from_slice(&crc.to_le_bytes());
    fs::write(&at.log, &foreign).unwrap();
    assert_eq!(
        Index::open(&path).map(|_| ()).unwrap_err().kind(),
        ErrorKind::InvalidData
    );
    assert_eq!(
        Writer::open(&path).map(|_| ()).unwrap_err().kind(),
        ErrorKind::InvalidData
    );
    fs::remove_file(&at.log).unwrap();
    assert_eq!(
        Index::open(&path).map(|_| ()).unwrap_err().kind(),
        ErrorKind::InvalidData
    );
    assert_eq!(
        Writer::open(&path).map(|_| ()).unwrap_err().kind(),
        ErrorKind::InvalidData
    );

    // With its log back the index opens, and a torn last record is left out, then cut off by a writer.
    let mut torn = log.clone();
    torn.extend_from_slice(&log[log::HEADER_2_BYTES..log::HEADER_2_BYTES + 30]);
    fs::write(&at.log, &torn).unwrap();
    assert!(ids_at(&path).unwrap().contains(&60_000));
    drop(Writer::open(&path).unwrap());
    assert_eq!(fs::read(&at.log).unwrap(), log);
}

#[test]
fn slots_round_trip_and_name_only_files_in_the_folder() {
    let root = Root {
        index_id: 0x1234_5678_9abc_def0,
        generation: 7,
        data: "g7-0123456789abcdef.data".into(),
        log: "g7-0123456789abcdef.log".into(),
        check: 0xdead_beef,
        base: 4321,
        writer: 99,
    };
    let bytes = format::encode_root(&root);
    assert_eq!(bytes.len(), format::ROOT_BYTES);
    assert_eq!(format::read_root(&bytes).unwrap(), root);

    // Every byte that the checksum covers is checked, and the names must be plain file names.
    for at in [0, 8, 12, 20, 28, 32, 40, 48, 49, 50, 114, 177] {
        let mut b = bytes.clone();
        b[at] ^= 1;
        assert_eq!(
            format::read_root(&b).unwrap_err().kind(),
            ErrorKind::InvalidData,
            "byte {at}"
        );
    }

    for bad in ["../x.data", "a/b.data", "..", ""] {
        let slot = format::encode_root(&Root {
            data: bad.into(),
            ..root.clone()
        });
        assert!(format::read_root(&slot).is_err(), "{bad:?}");
    }

    assert_eq!(generation_of("g12-0123456789abcdef.data"), Some(12));
    assert_eq!(generation_of("g12-0123456789abcdef.log"), Some(12));
    for name in [
        "g12-0123.data",
        "g-0123456789abcdef.data",
        "h1-0123456789abcdef.log",
        "g1-0123456789abcdeg.log",
    ] {
        assert_eq!(generation_of(name), None, "{name}");
    }
}

#[cfg(unix)]
#[test]
fn a_link_to_an_index_still_leads_to_it_after_its_first_commit() {
    let scratch = Folder::new("link");
    let rows = built(&scratch);
    let link = scratch.0.join("link.coldivf");
    std::os::unix::fs::symlink(scratch.index(), &link).unwrap();
    let mut writer = Writer::open(&link).unwrap();
    churn(&mut writer, &rows, 60);
    writer.compact().unwrap();
    // The link's target is gone, but its folder is where the index lives, and the link shares the writer's lock.
    assert!(!scratch.index().exists() && folder(&scratch.index()).is_dir());
    assert_eq!(
        Writer::open(&link).map(|_| ()).unwrap_err().kind(),
        ErrorKind::WouldBlock
    );
    let want = ids_at(&scratch.index()).unwrap();
    assert!(ids_at(&link).unwrap() == want);
    drop(writer);
    writer = Writer::open(&link).unwrap();
    writer.insert(60_000, &rows.data[..DIM]).unwrap();
    assert!(ids_at(&scratch.index()).unwrap().contains(&60_000));
}

#[test]
fn a_job_that_goes_on_after_a_power_cut_never_reuses_what_it_wrote_from_changes_that_the_cut_dropped() {
    for upsert in [false, true] {
        let name = format!("cut-resume-{upsert}");
        let (scratch, images) = (Folder::new(&name), Folder::new(&format!("{name}-images")));
        let rows = built(&scratch);
        let mut writer = Writer::open(&scratch.index()).unwrap();
        writer.compact().unwrap();
        churn(&mut writer, &rows, 61);
        writer.sync().unwrap();
        let near =
            |i: usize, by: f32| -> Vec<f32> { rows.data[i * DIM..(i + 1) * DIM].iter().map(|&x| x + by).collect() };
        let throttle = eager_throttle();
        // Changes that the app never syncs before the job's snapshot takes them: new ids, or new vectors for ids 0 to
        // 39.
        let first = |i: usize| if upsert { i as u64 } else { 5_000_000 + i as u64 };
        let taken = imaged(&scratch, &images, || {
            for i in 0..40 {
                writer.insert(first(i), &near(i, 0.2)).unwrap();
            }
            let mut job = writer.begin(&settings(), RunKind::Fold).unwrap();
            job.window = 16 << 10;
            let done = job.run(&throttle).unwrap();
            writer.finish_compact(done).unwrap();
        });
        drop(writer);
        // A power cut once the job wrote its file and recorded where it got, just before the commit makes its log.
        let (what, dir) = taken
            .iter()
            .find(|(what, _)| what.starts_with("power cut before Create") && what.ends_with(".log"))
            .unwrap();
        let path = dir.join("x.coldivf");
        let mut writer = Writer::open(&path).unwrap();
        assert!(
            !crate::maintain::job::kept(&writer.root).is_empty(),
            "{what}: no job to go on with"
        );

        // After the restart the app makes other changes of the same shape and syncs them: other ids for the same
        // vectors, or the same ids with other vectors.
        let second = |i: usize| if upsert { i as u64 } else { 6_000_000 + i as u64 };

        for i in 0..40 {
            writer
                .insert(second(i), &near(i, if upsert { 0.21 } else { 0.2 }))
                .unwrap();
        }

        writer.sync().unwrap();
        let want = rows_at(&path);
        let mut job = writer.begin_compact(&settings()).unwrap().unwrap();
        assert!(job.resumes(), "{what}: the job started over");
        job.window = 16 << 10;
        let done = job.run(&throttle).unwrap();
        writer.finish_compact(done).unwrap();
        assert!(
            rows_at(&path) == want,
            "{what}, upsert {upsert}: the commit holds other vectors"
        );
    }
}

#[test]
fn a_resume_file_from_before_the_first_commit_names_its_data_file_so_a_rebuild_starts_the_job_over() {
    let (scratch, images) = (Folder::new("legacy-resume"), Folder::new("legacy-resume-images"));
    let (rows, _) = clustered(&mut SplitMix64::new(41), 1500, DIM, 15, 0.6);
    let builder = Builder::new(build_params(Metric::L2, Anchor::Cluster));
    let clusters = builder.cluster(&rows, &scratch.index()).unwrap();
    builder.build_with_clusters(&rows, &clusters, &scratch.index()).unwrap();
    let throttle = eager_throttle();
    let mut writer = Writer::open(&scratch.index()).unwrap();
    let taken = imaged(&scratch, &images, || {
        let mut job = writer.begin(&settings(), RunKind::Fold).unwrap();
        job.window = 16 << 10;
        let done = job.run(&throttle).unwrap();
        writer.finish_compact(done).unwrap();
    });
    drop(writer);
    // The process dies once the job wrote its file, before the commit.
    let (what, dir) = taken
        .iter()
        .find(|(what, _)| what.starts_with("crash before Create") && what.ends_with(".log"))
        .unwrap();

    // Left alone, the next job goes on with the file it left.
    let alone = Folder::new("legacy-resume-alone");
    copy_files(dir, &alone.0);
    let mut writer = Writer::open(&alone.index()).unwrap();
    let mut job = writer.begin_compact(&settings()).unwrap().unwrap();
    assert!(job.resumes(), "{what}: the job started over");
    job.window = 16 << 10;
    let done = job.run(&throttle).unwrap();
    writer.finish_compact(done).unwrap();
    assert_eq!(ids_at(&alone.index()).unwrap(), (0..1500).collect::<Vec<u64>>());

    // The app builds the index again from other vectors with the same clustering, eg. after a new embedding run, so
    // the next job starts over and folds the new file's rows.
    let path = dir.join("x.coldivf");
    let mut rng = SplitMix64::new(7);
    let moved = Rows {
        dim: DIM,
        data: rows.data.iter().map(|&x| x + 0.2 * rng.normal() as f32).collect(),
    };
    builder.build_with_clusters(&moved, &clusters, &path).unwrap();
    let want = rows_at(&path);
    let mut writer = Writer::open(&path).unwrap();
    assert!(crate::maintain::job::kept(&writer.root).is_empty(), "{what}");
    let mut job = writer.begin(&settings(), RunKind::Fold).unwrap();
    assert!(
        !job.resumes(),
        "{what}: the job went on with the file before the rebuild"
    );
    job.window = 16 << 10;
    let done = job.run(&throttle).unwrap();
    writer.finish_compact(done).unwrap();
    assert!(
        rows_at(&path) == want,
        "{what}: the commit holds other vectors than the rebuild"
    );
}

#[test]
fn a_rebuild_identical_to_the_generation_before_takes_none_of_its_changes() {
    let scratch = Folder::new("rebuild-same");
    let rows = built(&scratch);
    let path = scratch.index();
    let mut writer = Writer::open(&path).unwrap();
    writer.compact().unwrap();
    drop(writer);
    let build = || {
        Builder::new(build_params(Metric::L2, Anchor::Cluster))
            .build(&rows, &path)
            .unwrap()
    };
    build();
    let before = locate(&path).unwrap();
    // A reader keeps the generation open while a writer adds a vector and goes.
    let reader = Index::open(&path).unwrap();
    let mut writer = Writer::open(&path).unwrap();
    writer.insert(60_000, &rows.data[..DIM]).unwrap();
    writer.sync().unwrap();
    drop(writer);
    build();
    let after = locate(&path).unwrap();
    let check = |at: &Located| at.root.as_ref().unwrap().check;
    assert_eq!(
        (after.generation, check(&after)),
        (before.generation + 1, check(&before))
    );

    // The rebuild holds the built vectors alone, for a new open and for the reader once it moves, and a writer opens.
    assert_eq!(ids_at(&path).unwrap(), (0..1500).collect::<Vec<u64>>());
    assert!(reader.refresh().unwrap());
    assert_eq!(reader.inner().changes().load().len(), 0);
    drop(Writer::open(&path).unwrap());
}

#[test]
fn a_damaged_length_in_the_middle_of_a_generation_s_log_fails_every_open_and_cuts_nothing_off() {
    let scratch = Folder::new("damaged-length");
    let rows = built(&scratch);
    let path = scratch.index();
    let mut writer = Writer::open(&path).unwrap();
    writer.compact().unwrap();

    for i in 0..3 {
        writer
            .insert(70_000 + i as u64, &rows.data[i * DIM..(i + 1) * DIM])
            .unwrap();
    }

    writer.sync().unwrap();
    let at = writer.index.inner().located().clone();
    drop(writer);
    let mut log = fs::read(&at.log).unwrap();
    let frame = (log.len() - log::HEADER_2_BYTES) / 3;
    // A flipped bit in the length of the second of the three synced records.
    log[log::HEADER_2_BYTES + frame + 1] ^= 1;
    fs::write(&at.log, &log).unwrap();
    assert_eq!(ids_at(&path).unwrap_err().kind(), ErrorKind::InvalidData);
    assert_eq!(
        Writer::open(&path).map(|_| ()).unwrap_err().kind(),
        ErrorKind::InvalidData
    );
    assert_eq!(fs::read(&at.log).unwrap(), log);
}

/// An index in `scratch` whose generation 2 is current with changes in its log, and whose generation 1's files are
/// still in its folder, as a power cut that undid their deletes leaves them. Returns generation 1 and the ids that the
/// index holds.
fn older_files_kept(scratch: &Folder) -> (Located, Vec<u64>) {
    let rows = built(scratch);
    let path = scratch.index();
    let mut writer = Writer::open(&path).unwrap();
    writer.compact().unwrap();
    churn(&mut writer, &rows, 63);
    writer.sync().unwrap();
    let g1 = locate(&path).unwrap();
    let kept: Vec<(PathBuf, Vec<u8>)> = [&g1.data, &g1.log]
        .iter()
        .map(|p| (p.to_path_buf(), fs::read(p).unwrap()))
        .collect();
    writer.compact().unwrap();
    churn(&mut writer, &rows, 64);
    writer.sync().unwrap();
    let want = ids_at(&path).unwrap();
    drop(writer);

    for (p, bytes) in &kept {
        fs::write(p, bytes).unwrap();
    }

    (g1, want)
}

#[test]
fn a_damaged_newer_slot_with_the_older_generation_s_files_still_there_leaves_the_newer_one_current() {
    let scratch = Folder::new("rot-older-kept");
    let (g1, want) = older_files_kept(&scratch);
    let path = scratch.index();
    // A flipped bit breaks slot 0, which names generation 2.
    let slot = folder(&path).join(SLOTS[0]);
    let mut bytes = fs::read(&slot).unwrap();
    bytes[30] ^= 1;
    fs::write(&slot, &bytes).unwrap();
    assert_eq!(locate(&path).unwrap().generation, 2);
    assert!(ids_at(&path).unwrap() == want);

    // A writer's open makes the slot again and deletes generation 1's files.
    drop(Writer::open(&path).unwrap());
    assert_eq!(format::read_root(&fs::read(&slot).unwrap()).unwrap().generation, 2);
    assert!(ids_at(&path).unwrap() == want);
    assert!(!g1.data.exists() && !g1.log.exists());
}

#[cfg(unix)]
#[test]
fn a_slot_that_can_t_be_read_fails_every_open_instead_of_going_back_a_generation() {
    use std::os::unix::fs::PermissionsExt;

    let scratch = Folder::new("unreadable-slot");
    let (_, want) = older_files_kept(&scratch);
    let path = scratch.index();
    let slot = folder(&path).join(SLOTS[0]);
    fs::set_permissions(&slot, fs::Permissions::from_mode(0o200)).unwrap();
    let located = locate(&path).map(|at| at.generation);
    let (opened, written) = (Index::open(&path).map(|_| ()), Writer::open(&path).map(|_| ()));
    fs::set_permissions(&slot, fs::Permissions::from_mode(0o644)).unwrap();
    assert_eq!(located.unwrap_err().kind(), ErrorKind::PermissionDenied);
    assert!(opened.is_err() && written.is_err());
    assert_eq!(locate(&path).unwrap().generation, 2);
    assert!(ids_at(&path).unwrap() == want);
}
