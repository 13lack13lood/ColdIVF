//! Unit tests for inserts, deletes and compaction.

use std::cell::RefCell;
use std::io::ErrorKind;
use std::path::Path;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use super::*;
use crate::build::write;
use crate::format::Part;
use crate::index::Inner;
use crate::quantization::SplitMix64;
use crate::quantization::testutil::{HOOK, Hooked, Rows, TempFile, clustered, gaussian, group, params, temp_file};
use crate::search::inmem::InMemory;
use crate::{Collections, Index, Limits, Metric, SearchCtx, SearchParams, Writer};

/// An index file of 600 clustered vectors with ids 0 to 599, and those vectors.
fn sample(name: &str, metric: Metric) -> (TempFile, Rows) {
    let (rows, mut clusters) = clustered(&mut SplitMix64::new(51), 600, 128, 12, 1.0);
    group(&mut clusters, 3);
    let index = InMemory::build(&params(metric), &rows, &clusters).unwrap();
    let path = temp_file(name);
    write(&index, &path).unwrap();
    let _ = std::fs::remove_file(log_path(&path));
    (path, rows)
}

/// Every cluster read and every vector rescored, so a search sees all of them.
fn everything() -> SearchParams {
    SearchParams {
        k: 10,
        nprobe: 12,
        rerank: 800,
        reach: None,
    }
}

fn ctx() -> SearchCtx {
    SearchCtx::new(Limits {
        max_k: 10,
        max_nprobe: 12,
        max_rerank: 800,
    })
}

/// The ids every search path of `index` finds for `q`, which must agree.
fn found(index: &Index, path: &Path, q: &[f32]) -> Vec<u64> {
    let (mut c, sp) = (ctx(), everything());
    index.set_warm(false);
    let cold: Vec<u64> = index.search(q, &sp, &mut c).unwrap().iter().map(|h| h.id).collect();
    index.set_warm(true);
    let warm: Vec<u64> = index.search(q, &sp, &mut c).unwrap().iter().map(|h| h.id).collect();
    index.set_warm(false);
    let mapped: Vec<u64> = index
        .inner
        .search_mapped(q, &sp, &mut c)
        .unwrap()
        .iter()
        .map(|h| h.id)
        .collect();
    let (set, mut out) = (Collections::open(&[path]).unwrap(), Vec::new());
    set.search(q, &sp, &mut c, &mut out).unwrap();
    let cold_set: Vec<u64> = out.iter().map(|(_, h)| h.id).collect();
    set.indexes[0].set_warm(true);
    set.search(q, &sp, &mut c, &mut out).unwrap();
    let warm_set: Vec<u64> = out.iter().map(|(_, h)| h.id).collect();
    assert!(
        cold == warm && cold == mapped && cold == cold_set && cold == warm_set,
        "the paths disagree"
    );
    cold
}

#[test]
fn an_insert_is_found_and_a_delete_hidden_by_the_next_search() {
    for metric in [Metric::L2, Metric::Cosine] {
        let (path, rows) = sample(&format!("insert-{metric:?}.coldivf"), metric);
        let before = Index::open(&path).unwrap();
        let mut writer = Writer::open(&path).unwrap();
        let v = gaussian(&mut SplitMix64::new(52), 1, 128, |_| 1.0);
        writer.insert(9000, &v).unwrap();
        assert_eq!(found(&before, &path, &v)[0], 9000);
        assert_eq!(found(&Index::open(&path).unwrap(), &path, &v)[0], 9000);
        // Deleting the vector a query lands on hides it, whether it came from the file or from an insert.
        let q = &rows.data[7 * 128..8 * 128];
        assert_eq!(found(&before, &path, q)[0], 7);
        writer.delete(7).unwrap();
        writer.delete(9000).unwrap();
        assert!(!found(&before, &path, q).contains(&7));
        assert!(!found(&before, &path, &v).contains(&9000));
    }
}

#[test]
fn an_insert_replaces_an_earlier_vector_with_its_id() {
    let (path, rows) = sample("replace.coldivf", Metric::L2);
    let index = Index::open(&path).unwrap();
    let mut writer = Writer::open(&path).unwrap();
    let old = &rows.data[5 * 128..6 * 128];
    let new = gaussian(&mut SplitMix64::new(53), 1, 128, |_| 3.0);

    writer.insert(5, &new).unwrap();
    assert!(!found(&index, &path, old).contains(&5));
    assert_eq!(found(&index, &path, &new)[0], 5);
    // Inserting it again leaves only the newest copy.
    writer.insert(5, &new).unwrap();
    assert_eq!(found(&index, &path, &new).iter().filter(|&&id| id == 5).count(), 1);
}

#[test]
fn the_log_survives_a_reopen_and_loses_only_a_torn_last_record() {
    let dim = 128;
    let (path, _) = sample("replay.coldivf", Metric::L2);
    let vs = gaussian(&mut SplitMix64::new(54), 3, 128, |_| 1.0);
    let mut writer = Writer::open(&path).unwrap();

    for (i, v) in vs.chunks_exact(dim).enumerate() {
        writer.insert(7000 + i as u64, v).unwrap();
    }

    writer.delete(7001).unwrap();
    drop(writer);

    // A crash in the middle of a write leaves part of a record at the end.
    let log = log_path(&path);
    let whole = std::fs::read(&log).unwrap();
    let mut torn = whole.clone();
    torn.extend_from_slice(&whole[HEADER_BYTES..HEADER_BYTES + 40]);
    std::fs::write(&log, &torn).unwrap();
    let index = Index::open(&path).unwrap();
    assert_eq!(found(&index, &path, &vs[..128])[0], 7000);
    assert!(!found(&index, &path, &vs[128..256]).contains(&7001));
    assert_eq!(found(&index, &path, &vs[256..])[0], 7002);
    drop(index);

    // A writer cuts the torn tail off, and its next record replays after the others.
    let mut writer = Writer::open(&path).unwrap();
    assert_eq!(std::fs::read(&log).unwrap(), whole);
    writer.delete(7000).unwrap();
    drop(writer);
    let index = Index::open(&path).unwrap();
    assert!(!found(&index, &path, &vs[..128]).contains(&7000));
}

#[test]
fn a_log_from_another_epoch_is_ignored() {
    let (path, _) = sample("epoch.coldivf", Metric::L2);
    let v = gaussian(&mut SplitMix64::new(55), 1, 128, |_| 1.0);
    Writer::open(&path).unwrap().insert(8000, &v).unwrap();

    let log = log_path(&path);
    let mut bytes = std::fs::read(&log).unwrap();
    bytes[12] ^= 1;
    std::fs::write(&log, &bytes).unwrap();
    assert!(!found(&Index::open(&path).unwrap(), &path, &v).contains(&8000));
}

#[test]
fn only_one_writer_holds_an_index() {
    let (path, _) = sample("lock.coldivf", Metric::L2);
    let first = Writer::open(&path).unwrap();

    assert_eq!(
        Writer::open(&path).map(|_| ()).unwrap_err().kind(),
        ErrorKind::WouldBlock
    );
    drop(first);
    assert!(Writer::open(&path).is_ok());
}

#[test]
fn compaction_folds_the_changes_into_the_file() {
    let dim = 128;
    let one = SearchParams {
        k: 1,
        nprobe: 1,
        rerank: 50,
        reach: None,
    };

    for metric in [Metric::L2, Metric::Cosine] {
        let (path, rows) = sample(&format!("compact-{metric:?}.coldivf"), metric);
        let mut writer = Writer::open(&path).unwrap();
        // Inserts near data rows, so each belongs in one cluster, but far enough from them that the 8-bit copies keep
        // each insert nearest to its own vector.
        let near: Vec<f32> = rows.data[100 * 128..120 * 128].iter().map(|&x| x + 0.5).collect();
        for (i, v) in near.chunks_exact(dim).enumerate() {
            writer.insert(5000 + i as u64, v).unwrap();
        }
        (0..10).for_each(|id| writer.delete(id).unwrap());
        writer.insert(20, &rows.data[50 * 128..51 * 128]).unwrap();
        // An insert that another of its id replaced, and one deleted, which the compaction leaves out.
        writer.insert(5001, &near[128..256]).unwrap();
        writer.delete(5002).unwrap();
        let queries: Vec<&[f32]> = near
            .chunks_exact(dim)
            .chain(rows.data[30 * 128..40 * 128].chunks_exact(dim))
            .collect();
        let before: Vec<u64> = {
            let index = Index::open(&path).unwrap();
            queries.iter().map(|q| found(&index, &path, q)[0]).collect()
        };
        writer.compact().unwrap();
        let index = Index::open(&path).unwrap();
        let layout = index.inner.layout();
        assert_eq!(
            (layout.count, layout.epoch, layout.nlist),
            (600 - 10 + 20 - 1, 1, 12),
            "{metric:?}"
        );
        assert_eq!(index.inner.changes().read().len(), 0);
        let ((bytes, file), data) = (index.inner.parts(), index.inner.in_place());
        let ids = &bytes[file.span(Part::Ids).range()];
        let id_of = |i: usize| sq8::id(data.rows.row(i as u32), dim).to_le_bytes();
        assert!(
            (0..file.count).all(|i| ids[8 * i..8 * i + 8] == id_of(i)),
            "the id section disagrees with the rows"
        );
        assert_eq!(std::fs::metadata(log_path(&path)).unwrap().len(), HEADER_BYTES as u64);
        let after: Vec<Vec<u64>> = queries.iter().map(|q| found(&index, &path, q)).collect();
        assert_eq!(after.iter().map(|ids| ids[0]).collect::<Vec<_>>(), before, "{metric:?}");
        assert!(
            after
                .iter()
                .all(|ids| (1..ids.len()).all(|i| !ids[..i].contains(&ids[i])))
        );
        assert!(!found(&index, &path, &near[256..384]).contains(&5002));
        assert!((0..10).all(|id| !found(&index, &path, &rows.data[id * 128..(id + 1) * 128]).contains(&(id as u64))));
        // Each vector kept its cluster, so reading one cluster finds each insert. The cosine file routes by centers of
        // the data before it was normalized, which put an insert in any cluster, so only L2 can check this.
        let checked = if metric == Metric::L2 { 20 } else { 0 };
        for (i, v) in near
            .chunks_exact(dim)
            .enumerate()
            .take(checked)
            .filter(|&(i, _)| i != 2)
        {
            assert_eq!(
                index.search(v, &one, &mut ctx()).unwrap()[0].id,
                5000 + i as u64,
                "{metric:?}"
            );
        }
        // Changes after a compaction go into the rebuilt file's log, and replay from it once nothing holds the file.
        writer.delete(5000).unwrap();
        writer.insert(6000, &near[..128]).unwrap();
        drop((writer, index));
        let ids = found(&Index::open(&path).unwrap(), &path, &near[..128]);
        assert!(ids[0] == 6000 && !ids.contains(&5000), "{metric:?}");
        drop(Writer::open(&path).unwrap());
        assert_eq!(found(&Index::open(&path).unwrap(), &path, &near[..128])[0], 6000);
    }
}

#[test]
fn a_damaged_record_ends_the_log_there() {
    let (path, _) = sample("crc.coldivf", Metric::L2);
    let vs = gaussian(&mut SplitMix64::new(65), 2, 128, |_| 1.0);
    let mut writer = Writer::open(&path).unwrap();
    writer.insert(7000, &vs[..128]).unwrap();
    writer.insert(7001, &vs[128..]).unwrap();
    drop(writer);

    // Flip the top byte of the second record's id, which keeps its length and breaks its checksum.
    let log = log_path(&path);
    let mut bytes = std::fs::read(&log).unwrap();
    let at = HEADER_BYTES;
    let second = at + 8 + u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap()) as usize;
    bytes[second + 8 + 8] ^= 1;
    std::fs::write(&log, &bytes).unwrap();
    let index = Index::open(&path).unwrap();
    assert_eq!(found(&index, &path, &vs[..128])[0], 7000);
    let ids = found(&index, &path, &vs[128..]);
    assert!(!ids.contains(&7001) && !ids.contains(&(7001 ^ 1 << 56)));
    drop(index);

    // A writer cuts the log just before it.
    let _writer = Writer::open(&path).unwrap();
    assert_eq!(std::fs::metadata(&log).unwrap().len(), second as u64);
}

#[test]
fn bad_inserts_fail_with_invalid_input() {
    let (path, _) = sample("bad-insert.coldivf", Metric::Cosine);
    let mut writer = Writer::open(&path).unwrap();
    let kind = |w: &mut Writer, v: &[f32]| w.insert(1, v).unwrap_err().kind();

    assert_eq!(kind(&mut writer, &[0.5; 127]), ErrorKind::InvalidInput);
    assert_eq!(kind(&mut writer, &[f32::NAN; 128]), ErrorKind::InvalidInput);
    assert_eq!(kind(&mut writer, &[0.0; 128]), ErrorKind::InvalidInput);
    let err = writer.insert(1, &[f32::NAN; 128]).unwrap_err();
    assert_eq!(err.to_string(), "a vector has a value that isn't finite");

    // The messages speak of the vector, not of a query.
    let message = |w: &mut Writer, v: &[f32]| w.insert(1, v).unwrap_err().to_string();
    assert_eq!(
        message(&mut writer, &[0.0; 128]),
        "a zero vector has no direction for cosine"
    );
    let (path, _) = sample("far-insert.coldivf", Metric::L2);
    let mut writer = Writer::open(&path).unwrap();
    assert_eq!(
        message(&mut writer, &[1e30; 128]),
        "a vector is too far from the mean to encode"
    );
}

#[test]
fn searches_run_while_a_writer_changes_the_index() {
    let dim = 128;
    let (path, _) = sample("stress.coldivf", Metric::L2);
    let index = Index::open(&path).unwrap();
    let vs = gaussian(&mut SplitMix64::new(57), 200, 128, |_| 1.0);
    let done = Arc::new(AtomicBool::new(false));

    std::thread::scope(|scope| {
        for t in 0..3 {
            let (index, done, vs) = (&index, Arc::clone(&done), &vs);
            scope.spawn(move || {
                let mut c = ctx();
                let mut q = 0;
                while !done.load(Ordering::Relaxed) {
                    let v = &vs[(q + t) % 200 * 128..((q + t) % 200 + 1) * 128];
                    index.search(v, &everything(), &mut c).unwrap();
                    q += 1;
                }
            });
        }
        let mut writer = Writer::open(&path).unwrap();
        for (i, v) in vs.chunks_exact(dim).enumerate() {
            writer.insert(10_000 + i as u64, v).unwrap();
            if i % 3 == 0 {
                writer.delete(i as u64).unwrap();
            }
        }
        done.store(true, Ordering::Relaxed);
    });

    for (i, v) in vs.chunks_exact(dim).enumerate().step_by(17) {
        assert_eq!(found(&index, &path, v)[0], 10_000 + i as u64);
    }
}

#[test]
fn a_search_sees_the_changes_as_they_were_when_it_scanned() {
    let (path, rows) = sample("seen.coldivf", Metric::L2);
    let index = Index::open(&path).unwrap();
    let writer = Rc::new(RefCell::new(Writer::open(&path).unwrap()));
    let q = rows.data[5 * 128..6 * 128].to_vec();
    let one = |index: &Index| {
        let mut c = ctx();
        let hits = index.search(&q, &everything(), &mut c).unwrap();
        hits.iter().filter(|h| h.id == 5).count()
    };
    // A cold search reads the clusters, then the rows. This backend replaces row 5 with an insert of the same
    // vector just before the rows, so after the search scanned the old copy and before it rescores it.
    let hooked = Index {
        inner: Arc::new(Inner::open(&path).unwrap().with_wave(|| Box::<Hooked<2>>::default())),
    };
    let (w, v) = (Rc::clone(&writer), q.clone());
    HOOK.set(Some(Box::new(move || w.borrow_mut().insert(5, &v).unwrap())));

    assert_eq!(one(&hooked), 1);
    assert_eq!(one(&index), 1);

    // The same in a search over several collections, with the next change.
    let set = Collections {
        indexes: vec![Index {
            inner: Arc::new(Inner::open(&path).unwrap().with_wave(|| Box::<Hooked<2>>::default())),
        }],
    };
    let (w, v) = (Rc::clone(&writer), q.clone());
    HOOK.set(Some(Box::new(move || w.borrow_mut().insert(5, &v).unwrap())));
    let mut out = Vec::new();
    set.search(&q, &everything(), &mut ctx(), &mut out).unwrap();
    assert_eq!(out.iter().filter(|(_, h)| h.id == 5).count(), 1);
    assert_eq!(one(&index), 1);

    // On its own, a later change hides only what came before it.
    let mut delta = Delta::new(index.inner.layout());
    let seen = delta.seen();
    delta.delete(5);
    assert!(delta.is_live(5, 0, seen) && !delta.is_live(5, 0, delta.seen()));
}

#[test]
fn an_open_reads_the_changes_that_another_process_logged() {
    let (path, rows) = sample("theirs.coldivf", Metric::L2);
    // A byte-identical copy at another path gets its own changes in this process, so its writer stands in for another
    // process that writes the first file's log.
    let twin = temp_file("theirs-twin.coldivf");
    std::fs::copy(&path, &twin).unwrap();
    let index = Index::open(&path).unwrap();
    let v = gaussian(&mut SplitMix64::new(59), 1, 128, |_| 1.0);
    let mut other = Writer::open(&twin).unwrap();
    other.insert(7777, &v).unwrap();
    other.delete(3).unwrap();
    drop(other);
    std::fs::copy(log_path(&twin), log_path(&path)).unwrap();
    let ids = |index: &Index, q: &[f32]| -> Vec<u64> {
        let mut c = ctx();
        let hits = index.search(q, &everything(), &mut c).unwrap();
        hits.iter().map(|h| h.id).collect()
    };

    // The index opened before doesn't see them until the next open of the file brings the shared changes up to date.
    assert!(!ids(&index, &v).contains(&7777));
    assert_eq!(found(&Index::open(&path).unwrap(), &path, &v)[0], 7777);
    assert_eq!(ids(&index, &v)[0], 7777);

    // A writer starts from them as well, so a compaction keeps them.
    Writer::open(&path).unwrap().compact().unwrap();
    let after = Index::open(&path).unwrap();
    assert_eq!(found(&after, &path, &v)[0], 7777);
    assert!(!found(&after, &path, &rows.data[3 * 128..4 * 128]).contains(&3));
}

#[test]
fn a_log_left_next_to_a_rebuilt_file_is_ignored() {
    let (path, _) = sample("rebuilt.coldivf", Metric::L2);
    let v = gaussian(&mut SplitMix64::new(60), 1, 128, |_| 1.0);
    let mut writer = Writer::open(&path).unwrap();
    writer.insert(9000, &v).unwrap();
    writer.delete(7).unwrap();
    drop(writer);

    // A build of other data with as many clusters, at the same path and the same epoch, so every old record would fit.
    let (rows, mut clusters) = clustered(&mut SplitMix64::new(61), 600, 128, 12, 1.0);
    group(&mut clusters, 3);
    write(&InMemory::build(&params(Metric::L2), &rows, &clusters).unwrap(), &path).unwrap();
    let index = Index::open(&path).unwrap();
    assert!(!found(&index, &path, &v).contains(&9000));
    assert_eq!(found(&index, &path, &rows.data[7 * 128..8 * 128])[0], 7);

    // A writer starts the log again for the new file, and a compaction of it works.
    let mut writer = Writer::open(&path).unwrap();
    assert_eq!(std::fs::metadata(log_path(&path)).unwrap().len(), HEADER_BYTES as u64);
    writer.compact().unwrap();
}

#[test]
fn a_record_for_a_cluster_the_file_lacks_ends_the_log() {
    let (path, _) = sample("leaf.coldivf", Metric::L2);
    let vs = gaussian(&mut SplitMix64::new(62), 2, 128, |_| 1.0);
    let mut writer = Writer::open(&path).unwrap();
    writer.insert(1000, &vs[..128]).unwrap();
    writer.insert(1001, &vs[128..]).unwrap();
    drop(writer);

    // Point the first record at cluster 999, with a checksum that matches, as a log written for another file could.
    let log = log_path(&path);
    let mut bytes = std::fs::read(&log).unwrap();
    let at = HEADER_BYTES;
    let len = u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap()) as usize;
    bytes[at + 8 + 9..at + 8 + 13].copy_from_slice(&999u32.to_le_bytes());
    let crc = crc32fast::hash(&bytes[at + 8..at + 8 + len]);
    bytes[at + 4..at + 8].copy_from_slice(&crc.to_le_bytes());
    std::fs::write(&log, &bytes).unwrap();
    let index = Index::open(&path).unwrap();
    assert!(!found(&index, &path, &vs[..128]).contains(&1000));
    assert!(!found(&index, &path, &vs[128..]).contains(&1001));

    // A writer cuts the log there, and a compaction doesn't trip over it.
    let mut writer = Writer::open(&path).unwrap();
    assert_eq!(std::fs::metadata(&log).unwrap().len(), HEADER_BYTES as u64);
    writer.compact().unwrap();
}

#[test]
fn a_failed_write_stops_the_writer_until_it_opens_again() {
    let (path, _) = sample("failed.coldivf", Metric::L2);
    let vs = gaussian(&mut SplitMix64::new(63), 2, 128, |_| 1.0);
    let mut writer = Writer::open(&path).unwrap();
    writer.insert(1000, &vs[..128]).unwrap();

    // A handle that can't write fails the next record, and can't cut the log back either, so the writer refuses
    // every later change, even through a good handle.
    let good = std::mem::replace(&mut writer.log.file, File::open(log_path(&path)).unwrap());
    assert!(writer.insert(1001, &vs[128..]).is_err());
    writer.log.file = good;
    assert!(writer.delete(1000).is_err());
    drop(writer);

    // The log holds the first insert, whole, and a new writer goes on after it.
    let mut writer = Writer::open(&path).unwrap();
    writer.insert(1001, &vs[128..]).unwrap();
    drop(writer);
    let index = Index::open(&path).unwrap();
    assert_eq!(found(&index, &path, &vs[..128])[0], 1000);
    assert_eq!(found(&index, &path, &vs[128..])[0], 1001);
}

#[cfg(unix)]
#[test]
fn every_path_to_a_file_shares_its_log_and_its_lock() {
    let (path, _) = sample("real.coldivf", Metric::L2);
    let link = temp_file("link.coldivf");
    std::os::unix::fs::symlink(&*path, &*link).unwrap();
    let writer = Writer::open(&path).unwrap();

    assert_eq!(
        Writer::open(&link).map(|_| ()).unwrap_err().kind(),
        ErrorKind::WouldBlock
    );
    drop(writer);
    let v = gaussian(&mut SplitMix64::new(64), 1, 128, |_| 1.0);
    Writer::open(&link).unwrap().insert(7777, &v).unwrap();
    assert!(!log_path(&link).exists());
    assert!(std::fs::metadata(log_path(&path)).unwrap().len() > HEADER_BYTES as u64);
}
