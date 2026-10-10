//! Unit tests for maintenance jobs: the throttle, cancel and resume, a job that outlives its writer, inserts during a
//! job and the lanes that the next generation gives them.

use std::sync::mpsc;
use std::time::Instant;

use super::*;
use crate::io::watch::{self, Op};
use crate::maintain::tests::{DIM, build_params, churn, settings, visible_ids};
use crate::quantization::SplitMix64;
use crate::quantization::testutil::{Rows, clustered, copy_index, scratch_dir};
use crate::{Builder, Index, Metric, SearchParams, Writer};

/// A folder of a test's own, by the name that the file system gives it, so a watch on it sees the index's files.
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

/// An index of 2000 vectors with ids 0 to 1999 in `dir`, committed once, so it has its folder and its id, then changed
/// through its writer. Returns the vectors and the writer.
fn changed(dir: &Folder, anchor: Anchor, seed: u64) -> (Rows, Writer) {
    let (rows, _) = clustered(&mut SplitMix64::new(seed), 2000, DIM, 20, 0.6);
    Builder::new(build_params(Metric::L2, anchor))
        .build(&rows, &dir.index())
        .unwrap();
    let mut writer = Writer::open(&dir.index()).unwrap();
    writer.maintain(&settings(), RunKind::Fold).unwrap();
    churn(&mut writer, &rows, seed + 1);
    writer.sync().unwrap();
    (rows, writer)
}

/// Room for the searches of the tests.
const LIMITS: Limits = Limits {
    max_k: 10,
    max_nprobe: 16,
    max_rerank: 64,
};

/// A throttle that waits for no search, records a mark after every window and keeps no space free.
fn eager_throttle() -> Throttle {
    Throttle::new(ThrottleParams {
        quiet: Duration::ZERO,
        spacing: Duration::ZERO,
        resume_bytes: 1,
        reserve_bytes: 0,
        ..ThrottleParams::default()
    })
}

/// A record with every field set.
fn record() -> Record {
    Record {
        index_id: 7,
        base: 3,
        base_check: 0xdead_beef,
        inserts: 40,
        next: 61,
        log_at: 9000,
        kind: 3,
        plan: 12345,
        window: 1 << 16,
        mark: Mark {
            windows: 2,
            sections: [
                Reached {
                    at: 100,
                    crc: 1,
                    len: 10,
                },
                Reached {
                    at: 200,
                    crc: 2,
                    len: 20,
                },
                Reached {
                    at: 300,
                    crc: 3,
                    len: 30,
                },
            ],
            page: vec![5u8; 776],
            copied: 11,
            reencoded: 12,
            bytes_read: 13,
        },
    }
}

#[test]
fn a_resume_record_round_trips_and_a_damaged_one_reads_as_none() {
    let r = record();
    let bytes = r.encode();
    assert_eq!(Record::decode(&bytes), Some(r));

    // Cut short anywhere, or with any byte changed, it is no record.
    for n in [0, 4, 8, bytes.len() / 2, bytes.len() - 1] {
        assert_eq!(Record::decode(&bytes[..n]), None, "cut to {n}");
    }

    for at in [0, 9, bytes.len() / 2, bytes.len() - 1] {
        let mut bad = bytes.clone();
        bad[at] ^= 1;
        assert_eq!(Record::decode(&bad), None, "byte {at}");
    }
}

/// The lane that a commit gives an insert whose row is `row` in the generation `inner`: its row decoded, routed to
/// its nearest cluster, and with the cluster anchor measured again from that cluster's center.
fn lane_from_row(inner: &Inner, row: &[u8]) -> (u32, Vec<u8>, crate::quantization::rabitq::Factors) {
    let (mut o, mut center) = (vec![0f32; DIM], vec![0f32; DIM]);
    sq8::decode(row, inner.mean(), &mut o);
    let mut ctx = SearchCtx::new(Limits {
        max_k: 1,
        max_nprobe: 1,
        max_rerank: 1,
    });
    let mut leaves = Vec::new();
    inner.route(&o, 1, None, &mut ctx, &mut leaves).unwrap();
    let mut code = vec![0u8; code_bytes(padded_dim(DIM))];
    let (codes, f) = inner.view(inner.tables().unwrap()).centers.center(leaves[0]);
    sq4::decode(codes, f, &mut center);
    let factors = Encoder::new(inner.rotation(), inner.mean()).encode_from(&o, &center, &mut code);
    (leaves[0], code, factors)
}

#[test]
fn inserts_during_a_job_keep_every_change_and_take_the_lanes_that_the_new_file_gives_them() {
    let dir = Folder::new("job-lanes");
    let (rows, mut writer) = changed(&dir, Anchor::Cluster, 61);
    let near = |r: usize, k: f32| -> Vec<f32> { rows.data[r * DIM..(r + 1) * DIM].iter().map(|&x| x + k).collect() };
    // A major run moves every center, so the next generation measures each insert from another one.
    let mut job = writer.begin(&settings(), RunKind::Major).unwrap();

    // Inserts after the snapshot, which the job measures from their rows, then after the plan, which the writer
    // measures from the vectors themselves, with an upsert of an earlier one and a delete of each.
    for i in 0..20 {
        writer.insert(2_000_000 + i, &near(i as usize, 0.05)).unwrap();
    }

    let mut done = job.run(&Throttle::none()).unwrap();
    assert_eq!(
        done.lanes.len(),
        20,
        "the job measured other inserts than those before its plan"
    );
    let mut after: Vec<(u64, Vec<f32>)> = (0..20)
        .map(|i| (3_000_000 + i, near(100 + i as usize, -0.05)))
        .collect();
    after.push((2_000_000, near(500, 0.0)));

    for (id, v) in &after {
        writer.insert(*id, v).unwrap();
    }

    writer.delete(3_000_001).unwrap();
    writer.delete(2_000_001).unwrap();
    // A lane that the job measured for an insert that the writer measured too loses to the writer's.
    let seq = writer
        .index
        .inner()
        .changes()
        .load()
        .seq(writer.index.inner().changes().load().len() - 21);
    done.lanes.push(Lane {
        seq,
        list: 0,
        code: vec![0u8; code_bytes(padded_dim(DIM))],
        factors: crate::quantization::rabitq::Factors::EMPTY,
    });
    let want = visible_ids(&writer);
    writer.finish_compact(done).unwrap();
    assert_eq!(visible_ids(&writer), want);

    // Each insert after the plan has the lane that an insert of its vector into the new file gets, and each one
    // before has the lane that its row gets there.
    let inner = writer.index.inner();
    let delta = inner.changes().load();
    let mut ctx = SearchCtx::new(Limits {
        max_k: 1,
        max_nprobe: 1,
        max_rerank: 1,
    });
    let place =
        |id: u64| (0..delta.len()).find(|&i| sq8::id(delta.row(i), DIM) == id && delta.insert_visible(i, delta.seen()));
    let lane = |i: usize| (delta.leaf(i), delta.lane(i).0.to_vec(), delta.lane(i).1);
    let mut differs = false;

    for (id, v) in after.iter().filter(|(id, _)| *id != 3_000_001) {
        let i = place(*id).unwrap();
        let e = inner.encode_insert(*id, v, &mut ctx, None).unwrap();
        assert_eq!(lane(i), (e.leaf, e.code, e.factors), "insert {id} after the plan");
        differs |= lane(i) != lane_from_row(&inner, delta.row(i));
    }

    for id in 2_000_002..2_000_020 {
        let i = place(id).unwrap();
        assert_eq!(
            lane(i),
            lane_from_row(&inner, delta.row(i)),
            "insert {id} before the plan"
        );
    }

    assert!(
        differs,
        "every insert after the plan got the lane of its row, so the test tells nothing"
    );
}

#[test]
fn a_job_reads_and_writes_only_once_no_search_ran_for_the_quiet_time() {
    let dir = Folder::new("job-gate");
    let (rows, mut writer) = changed(&dir, Anchor::Cluster, 62);
    let index = Index::open(&dir.index()).unwrap();
    let quiet = Duration::from_millis(60);
    let throttle = Throttle::new(ThrottleParams {
        quiet,
        spacing: Duration::from_millis(10),
        burst_bytes: 64 << 10,
        idle: Duration::from_secs(3600),
        resume_bytes: u64::MAX,
        reserve_bytes: 0,
        ..ThrottleParams::default()
    });
    let writes = Arc::new(Mutex::new(Vec::new()));
    let seen = Arc::clone(&writes);
    let _watch = watch::set(&dir.0, move |op, path| {
        let data = path.extension().is_some_and(|e| e == "data");
        if data && matches!(op, Op::Write | Op::Sync) {
            locked(&seen).push((io::now_ns(), op));
        }
        Ok(())
    });
    let mut job = writer.begin(&settings(), RunKind::Fold).unwrap();
    let sp = SearchParams {
        k: 10,
        nprobe: 4,
        rerank: 50,
        reach: None,
    };

    let link = Arc::clone(writer.job.as_ref().unwrap());
    let activity = writer.index.shared.activity();

    // A search that the word says runs until an hour from now holds the job back, however long the machine takes to
    // run anything else. Then a real one runs, and the job may go on once the quiet time after it has passed.
    let (last, planned, done) = std::thread::scope(|s| {
        activity.set(io::now_ns() + 3_600_000_000_000);
        let running = s.spawn(|| job.run(&throttle));
        std::thread::sleep(Duration::from_millis(300));
        let planned = link.centers().is_some();
        activity.set(0);
        let mut ctx = SearchCtx::new(LIMITS);
        let last = io::now_ns();
        index.search(&rows.data[..DIM], &sp, &mut ctx).unwrap();
        (last, planned, running.join().unwrap())
    });
    let done = done.unwrap();
    let writes = locked(&writes).clone();

    // The job read nothing while the searches ran, so it hadn't planned, and nothing reached the new file then, nor
    // in the quiet time after the last search began, less the few milliseconds that the word's once-a-millisecond
    // stores may be behind.
    assert!(!planned, "the job planned while the searches ran");
    assert!(!writes.is_empty(), "the job wrote nothing");
    let first = writes.iter().map(|w| w.0).min().unwrap_or(0);
    let slack = 5_000_000;
    assert!(
        first + slack >= last + quiet.as_nanos() as u64,
        "a write came {} ms after the last search began",
        first.saturating_sub(last) / 1_000_000
    );

    // Bursts of at most 64 KiB, each ended by a sync, start at least 10 ms after the one before ends.
    let len = fs::metadata(&done.next.as_ref().unwrap().data).unwrap().len() as usize;
    let syncs: Vec<u64> = writes.iter().filter(|w| w.1 == Op::Sync).map(|w| w.0).collect();
    assert!(syncs.len() >= len / (64 << 10), "{} syncs for {len} bytes", syncs.len());

    for pair in writes.windows(2).filter(|p| p[0].1 == Op::Sync && p[1].1 == Op::Write) {
        assert!(
            pair[1].0 + slack >= pair[0].0 + 10_000_000,
            "a burst started too soon after the last one"
        );
    }

    writer.finish_compact(done).unwrap();
}

#[test]
fn a_search_during_a_burst_ends_the_burst_at_the_next_write() {
    let dir = Folder::new("job-burst");
    let (_, mut writer) = changed(&dir, Anchor::Cluster, 69);
    let quiet = Duration::from_millis(40);
    let throttle = Throttle::new(ThrottleParams {
        quiet,
        spacing: Duration::ZERO,
        burst_bytes: 1 << 30,
        resume_bytes: u64::MAX,
        reserve_bytes: 0,
        ..ThrottleParams::default()
    });
    // At the second write of the new file a search runs, so the third waits for the quiet time.
    let (activity, times) = (writer.index.shared.activity(), Arc::new(Mutex::new(Vec::new())));
    let (seen, count) = (Arc::clone(&times), AtomicU64::new(0));
    let _watch = watch::set(&dir.0, move |op, path| {
        if op == Op::Write && path.extension().is_some_and(|e| e == "data") {
            locked(&seen).push(io::now_ns());
            if count.fetch_add(1, Ordering::Relaxed) == 1 {
                activity.note();
            }
        }
        Ok(())
    });
    let mut job = writer.begin(&settings(), RunKind::Fold).unwrap();
    job.window = 64 << 10;
    let done = job.run(&throttle).unwrap();
    let times = locked(&times).clone();
    assert!(times.len() > 3);
    assert!(
        times[2] + 5_000_000 >= times[1] + quiet.as_nanos() as u64,
        "the burst went on after a search"
    );
    writer.finish_compact(done).unwrap();
}

/// The data file that a job of `writer` writes through `throttle`, in windows of `window` bytes of rows, cancelled
/// after its `cancel_at`-th write when that is set. Returns the result, the bytes of the data file if it ran to the
/// end, and the writes it made.
fn run_job(
    dir: &Folder,
    writer: &mut Writer,
    window: usize,
    cancel_at: Option<usize>,
) -> (Result<Done, Error>, Vec<u8>, usize) {
    let mut job = writer.begin(&settings(), RunKind::Fold).unwrap();
    job.window = window;
    let (cancel, count) = (job.canceller(), Arc::new(AtomicU64::new(0)));
    let counted = Arc::clone(&count);
    let watch = watch::set(&dir.0, move |op, path| {
        if op == Op::Write && path.extension().is_some_and(|e| e == "data") {
            let n = counted.fetch_add(1, Ordering::Relaxed) + 1;
            if cancel_at.is_some_and(|at| n as usize == at) {
                cancel.cancel();
            }
        }
        Ok(())
    });
    let result = job.run(&eager_throttle());
    drop(watch);
    let bytes = match &result {
        Ok(done) => fs::read(&done.next.as_ref().unwrap().data).unwrap(),
        Err(_) => Vec::new(),
    };
    (result, bytes, count.load(Ordering::Relaxed) as usize)
}

#[test]
fn a_cancelled_job_goes_on_where_it_stopped_and_writes_the_same_file() {
    let (dir, other) = (Folder::new("job-resume"), Folder::new("job-resume-whole"));
    let (_, writer) = changed(&dir, Anchor::Cluster, 63);
    drop(writer);
    copy_index(&dir.index(), &other.index());
    let mut writer = Writer::open(&dir.index()).unwrap();
    let want = visible_ids(&writer);

    // The same job of a copy of the index, run to the end, writes the file to compare with.
    let mut whole = Writer::open(&other.index()).unwrap();
    let (done, file, writes) = run_job(&other, &mut whole, 64 << 10, None);
    let done = done.unwrap();
    assert!(
        writes > 8,
        "the file took {writes} writes, too few to stop in the middle"
    );
    whole.finish_compact(done).unwrap();

    // Cancelled halfway, the job keeps its file and a mark, and the next job goes on from the mark.
    let (stopped, _, _) = run_job(&dir, &mut writer, 64 << 10, Some(writes / 2));
    assert_eq!(stopped.err().map(|e| e.kind()), Some(ErrorKind::Interrupted));
    assert_eq!(kept(&writer.root).len(), 2, "the job left no file to go on with");
    let (done, again, more) = run_job(&dir, &mut writer, 64 << 10, None);
    let done = done.unwrap();
    assert!(again == file, "the job that went on wrote another file");
    assert!(more < writes, "the job that went on wrote the whole file again");
    let resume = resume_path(&done.next.as_ref().unwrap().data);
    let record = fs::read(&resume).unwrap();
    writer.finish_compact(done).unwrap();
    assert_eq!(visible_ids(&writer), want);
    assert!(kept(&writer.root).is_empty(), "the commit left the resume file");

    // A resume file that a crash left after the commit is of a generation that is gone, so the next writer deletes it,
    // and its job starts over.
    drop(writer);
    fs::write(&resume, &record).unwrap();
    let mut writer = Writer::open(&dir.index()).unwrap();
    assert!(
        !resume.exists(),
        "the writer kept a resume file of a generation that is gone"
    );
    let job = writer.begin(&settings(), RunKind::Fold).unwrap();
    assert!(!job.resumes());
}

#[test]
fn a_job_that_writes_in_other_windows_than_the_one_it_goes_on_from_starts_over() {
    let (dir, other) = (Folder::new("job-windows"), Folder::new("job-windows-whole"));
    let (_, writer) = changed(&dir, Anchor::Cluster, 70);
    drop(writer);
    copy_index(&dir.index(), &other.index());
    let mut writer = Writer::open(&dir.index()).unwrap();
    let mut whole = Writer::open(&other.index()).unwrap();
    let (done, file, writes) = run_job(&other, &mut whole, 64 << 10, None);
    whole.finish_compact(done.unwrap()).unwrap();

    // A mark counts windows, so a job of half as large windows can't go on from it, and writes the whole file.
    let (stopped, _, _) = run_job(&dir, &mut writer, 64 << 10, Some(writes / 2));
    assert_eq!(stopped.err().map(|e| e.kind()), Some(ErrorKind::Interrupted));
    let (done, again, more) = run_job(&dir, &mut writer, 32 << 10, None);
    assert!(again == file, "the job that started over wrote another file");
    assert!(more >= writes, "the job went on from a mark of other windows");
    writer.finish_compact(done.unwrap()).unwrap();
}

#[test]
fn a_job_outlives_its_writer_and_the_next_writer_goes_on_with_it() {
    let dir = Folder::new("job-outlives");
    let (_, mut writer) = changed(&dir, Anchor::Cluster, 64);
    let want = visible_ids(&writer);
    let mut job = writer.begin(&settings(), RunKind::Fold).unwrap();
    job.window = 64 << 10;
    // The job waits at its third write, which ends its first window, until the writer is gone. The watch holds every
    // other file operation of the process meanwhile, so it waits 5 seconds at most.
    let ((at, reached), (go, wait)) = (mpsc::channel::<()>(), mpsc::channel::<()>());
    let count = AtomicU64::new(0);
    let _watch = watch::set(&dir.0, move |op, path| {
        let data = op == Op::Write && path.extension().is_some_and(|e| e == "data");
        if data && count.fetch_add(1, Ordering::Relaxed) == 2 {
            let _ = at.send(());
            let _ = wait.recv_timeout(Duration::from_secs(5));
        }
        Ok(())
    });

    std::thread::scope(|s| {
        let running = s.spawn(|| job.run(&eager_throttle()));
        reached.recv().unwrap();
        drop(writer);
        // The job holds the writer's lock, so no other writer opens while it runs.
        let blocked = Writer::open(&dir.index()).err().map(|e| e.kind());
        let _ = go.send(());
        let stopped = running.join().unwrap();
        assert_eq!(blocked, Some(ErrorKind::WouldBlock));
        assert_eq!(stopped.err().map(|e| e.kind()), Some(ErrorKind::Interrupted));
    });
    drop(job);

    let mut writer = Writer::open(&dir.index()).unwrap();
    assert_eq!(visible_ids(&writer), want);
    assert_eq!(kept(&writer.root).len(), 2, "the job left no file to go on with");
    let mut job = writer.begin_compact(&settings()).unwrap().unwrap();
    assert!(job.resumes(), "the next job started over");
    job.window = 64 << 10;
    let done = job.run(&eager_throttle()).unwrap();
    writer.finish_compact(done).unwrap();
    assert_eq!(visible_ids(&writer), want);
}

#[test]
fn a_writer_runs_one_job_at_a_time_and_commits_only_its_own() {
    let (a, b) = (Folder::new("job-one-a"), Folder::new("job-one-b"));
    let (_, mut first) = changed(&a, Anchor::Cluster, 65);
    let (_, mut second) = changed(&b, Anchor::Mean, 66);
    let mut job = first.begin(&settings(), RunKind::Fold).unwrap();
    let busy = |w: &mut Writer| w.begin(&settings(), RunKind::Fold).err().map(|e| e.kind());
    assert_eq!(busy(&mut first), Some(ErrorKind::WouldBlock));
    assert_eq!(
        first.maintain(&settings(), RunKind::Fold).err().map(|e| e.kind()),
        Some(ErrorKind::WouldBlock)
    );

    // The finished job is the first writer's, so the second refuses it and its file goes.
    let done = job.run(&Throttle::none()).unwrap();
    drop(job);
    assert_eq!(busy(&mut first), Some(ErrorKind::WouldBlock));
    let (data, want) = (done.next.as_ref().unwrap().data.clone(), visible_ids(&second));
    let generation = second.index.inner().located().generation;
    assert_eq!(
        second.finish_compact(done).err().map(|e| e.kind()),
        Some(ErrorKind::InvalidInput)
    );
    assert!(!data.exists(), "the refused job's file is still there");
    assert_eq!(second.index.inner().located().generation, generation);
    assert_eq!(visible_ids(&second), want);

    // With its job gone, the first writer runs another.
    first.maintain(&settings(), RunKind::Fold).unwrap();
}

#[test]
fn a_job_that_lacks_free_disk_space_fails_before_it_writes_and_leaves_the_index_as_it_was() {
    let dir = Folder::new("job-full");
    let (_, mut writer) = changed(&dir, Anchor::Cluster, 67);
    let (want, generation) = (visible_ids(&writer), writer.index.inner().located().generation);
    let mut job = writer.begin(&settings(), RunKind::Fold).unwrap();
    let throttle = Throttle::new(ThrottleParams {
        quiet: Duration::ZERO,
        reserve_bytes: u64::MAX / 4,
        ..ThrottleParams::default()
    });
    assert_eq!(job.run(&throttle).err().map(|e| e.kind()), Some(ErrorKind::StorageFull));
    drop(job);
    assert!(kept(&writer.root).is_empty());
    assert_eq!(writer.index.inner().located().generation, generation);
    assert_eq!(visible_ids(&writer), want);
    writer.maintain(&settings(), RunKind::Fold).unwrap();
}

#[cfg(target_os = "macos")]
#[test]
fn a_job_s_threads_run_at_utility_quality_of_service_and_a_blocking_run_s_do_not() {
    let utility = |background| {
        super::super::pool(2, background)
            .unwrap()
            .broadcast(|_| io::macos::is_utility())
    };
    assert!(utility(true).iter().all(|&u| u));
    assert!(utility(false).iter().all(|&u| !u));
}

#[test]
fn a_search_notes_when_it_ran_in_the_word_that_every_process_maps() {
    let dir = Folder::new("job-activity");
    let (rows, writer) = changed(&dir, Anchor::Cluster, 68);
    let index = Index::open(&dir.index()).unwrap();
    let before = io::now_ns();
    let mut ctx = SearchCtx::new(LIMITS);
    let sp = SearchParams {
        k: 1,
        nprobe: 1,
        rerank: 1,
        reach: None,
    };
    index.search(&rows.data[..DIM], &sp, &mut ctx).unwrap();

    // Another process maps the same word, and sees the search.
    let other = Activity::open(&crate::root::folder(&dir.index()));
    let word = fs::read(crate::root::folder(&dir.index()).join("activity")).unwrap();
    let stored = u64::from_le_bytes(word[..8].try_into().unwrap());
    assert!(
        stored + 2_000_000 >= before && stored == other.last(),
        "{stored} {before}"
    );
    drop(writer);
}

#[test]
fn a_warm_index_is_warm_after_a_commit_with_its_new_codes_in_memory_and_a_cold_one_stays_cold() {
    for warm in [true, false] {
        let dir = Folder::new(&format!("job-warm-{warm}"));
        let (rows, mut writer) = changed(&dir, Anchor::Cluster, 78);
        let index = Index::open(&dir.index()).unwrap();

        // A warm index holds its codes in memory, and the row pages that searches read.
        let mut touched = std::collections::HashSet::new();

        if warm {
            index.warm_up().unwrap();
            let started = Instant::now();
            while !index.is_warm() && started.elapsed() < Duration::from_secs(20) {
                std::thread::sleep(Duration::from_millis(5));
            }
            assert!(index.is_warm(), "the warm-up didn't finish");
            let old = index.inner();
            let rows = old.in_place().rows;
            for r in (0..old.layout().count as u32).step_by(97) {
                touched.insert(sq8::id(std::hint::black_box(rows.row(r)), DIM));
            }
        }

        // A major run measures every code again, so the new file's codes differ from the old file's.
        let mut job = writer.begin(&settings(), RunKind::Major).unwrap();
        let done = job.run(&Throttle::none()).unwrap();
        drop(job);
        writer.finish_compact(done).unwrap();
        let inner = index.inner();
        assert_eq!((inner.located().generation, inner.is_warm()), (2, warm));

        // The job wrote a warm index's new codes through the system's cache, and a cold one's around it.
        #[cfg(target_os = "macos")]
        {
            let resident = io::resident(inner.in_place().blocks);
            let held = resident.iter().filter(|&&r| r).count();
            assert!(!resident.is_empty());
            match warm {
                true => assert_eq!(held, resident.len(), "pages of the new codes aren't in memory"),
                false => assert!(
                    2 * held < resident.len(),
                    "{held} of {} pages are in memory",
                    resident.len()
                ),
            }

            // The pages of the rows that were in memory are again, wherever the rows went.
            let rows = inner.in_place().rows;
            let (pages, lead) = (io::resident(rows.bytes), rows.bytes.as_ptr() as usize % io::page_size());
            let page_of = |r: u32| (lead + r as usize / rows.per_page * rows.page_size) / io::page_size();
            let missing = (0..inner.layout().count as u32)
                .filter(|&r| touched.contains(&sq8::id(rows.row(r), DIM)))
                .filter(|&r| !pages[page_of(r)])
                .count();
            assert_eq!(missing, 0, "rows that were in memory aren't");
        }

        let sp = SearchParams {
            k: 10,
            nprobe: 4,
            rerank: 50,
            reach: None,
        };
        let mut ctx = SearchCtx::new(LIMITS);
        let hits: Vec<u64> = index
            .search(&rows.data[..DIM], &sp, &mut ctx)
            .unwrap()
            .iter()
            .map(|h| h.id)
            .collect();
        index.set_warm(!warm);
        let other: Vec<u64> = index
            .search(&rows.data[..DIM], &sp, &mut ctx)
            .unwrap()
            .iter()
            .map(|h| h.id)
            .collect();
        assert_eq!(hits, other, "warm {warm}");
    }
}

/// Warms `index` and waits until it is warm.
fn warmed(index: &Index) {
    index.warm_up().unwrap();
    let started = Instant::now();

    while !index.is_warm() && started.elapsed() < Duration::from_secs(20) {
        std::thread::sleep(Duration::from_millis(5));
    }

    assert!(index.is_warm(), "the warm-up didn't finish");
}

#[test]
fn a_copy_of_the_new_generation_made_current_as_the_writer_commits_gives_way_to_the_writer_s_own() {
    let dir = Folder::new("job-own-publish");
    let (_, mut writer) = changed(&dir, Anchor::Cluster, 81);
    let index = Index::open(&dir.index()).unwrap();
    warmed(&index);
    let (path, cell) = (dir.index(), index.clone());
    // Right after the commit's slot write, a copy of the new generation, opened from the root as a refresh that began
    // before the commit opens it, becomes current.
    let guard = watch::set(&dir.0, move |op, _| {
        if op == Op::Step("retired") {
            let copy = Arc::new(Inner::open(&path).unwrap());
            assert!(cell.shared.publish(&cell.shared.load_full(), copy));
        }
        Ok(())
    });
    writer.compact().unwrap();
    drop(guard);

    // The writer's generation, warm unlike the copy, took the copy's place, and the stamp of the root's files is the
    // commit's, so no prepare looks for a newer generation.
    assert_eq!(index.inner().located().generation, 2);
    assert!(index.is_warm() && !index.shared.may_have_moved());
}

#[test]
fn a_refresh_a_prepare_or_an_open_as_the_writer_commits_opens_no_copy_of_the_new_generation() {
    let dir = Folder::new("job-commit-window");
    let (_, mut writer) = changed(&dir, Anchor::Cluster, 82);
    let index = Index::open(&dir.index()).unwrap();
    warmed(&index);
    let (path, cell) = (dir.index(), index.clone());
    let seen = Arc::new(Mutex::new(Vec::new()));
    let saw = Arc::clone(&seen);
    // Each call runs between the commit's slot write and the writer's publish, and leaves the old generation current.
    let guard = watch::set(&dir.0, move |op, _| {
        if op == Op::Step("retired") {
            let before = cell.inner().uid();
            let refreshed = cell.refresh().unwrap();
            cell.prepare(&mut SearchCtx::new(LIMITS)).unwrap();
            let opened = Index::open(&path).unwrap();
            let after = (cell.inner().uid(), opened.inner().uid());
            saw.lock().unwrap().push((refreshed, before, after));
        }
        Ok(())
    });
    writer.compact().unwrap();
    drop(guard);
    let seen = seen.lock().unwrap().clone();
    assert_eq!(seen.len(), 1, "the commit had no step that retires the old generation");
    let (refreshed, before, after) = seen[0];
    assert!(!refreshed && after == (before, before), "{seen:?}");
    assert_eq!(index.inner().located().generation, 2);
    assert!(index.is_warm());
}
