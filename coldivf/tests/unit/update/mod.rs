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
use crate::quantization::rabitq::{BLOCK, block_bytes, code_bytes, unpack_lane};
use crate::quantization::testutil::{
    HOOK, Hooked, Rows, TempFile, clustered, dist2, gaussian, group, params, temp_file,
};
use crate::quantization::{padded_dim, sq8};
use crate::search::inmem::InMemory;
use crate::{Anchor, BuildParams, Collections, Hit, Index, Limits, Metric, SearchCtx, SearchParams, Writer};
use log::HEADER_BYTES;

/// An index file of 600 clustered vectors with ids 0 to 599, and those vectors.
fn sample(name: &str, metric: Metric) -> (TempFile, Rows) {
    sample_with(name, &params(metric))
}

/// The same, built with `p`.
fn sample_with(name: &str, p: &BuildParams) -> (TempFile, Rows) {
    let (rows, mut clusters) = clustered(&mut SplitMix64::new(51), 600, 128, 12, 1.0);
    group(&mut clusters, 3);
    let index = InMemory::build(p, &rows, &clusters).unwrap();
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
        .inner()
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

    for (metric, anchor) in [
        (Metric::L2, Anchor::Cluster),
        (Metric::Cosine, Anchor::Cluster),
        (Metric::L2, Anchor::Mean),
    ] {
        let p = BuildParams {
            anchor,
            ..params(metric)
        };
        let (path, rows) = sample_with(&format!("compact-{metric:?}-{anchor:?}.coldivf"), &p);
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
        let inner = index.inner();
        let layout = inner.layout();
        assert_eq!(
            (layout.count, layout.epoch, layout.nlist, layout.anchor),
            (600 - 10 + 20 - 1, 1, 12, anchor),
            "{metric:?}"
        );
        assert_eq!(inner.changes().load().len(), 0);
        let ((bytes, file), data) = (inner.parts(), inner.in_place());
        let ids = &bytes[file.span(Part::Ids).range()];
        let id_of = |i: usize| sq8::id(data.rows.row(i as u32), dim).to_le_bytes();
        assert!(
            (0..file.count).all(|i| ids[8 * i..8 * i + 8] == id_of(i)),
            "the id section disagrees with the rows"
        );
        // The commit deleted the old file's log, and the new generation's log holds only its header.
        assert!(!log_path(&path).exists());
        let log = &inner.located().log;
        assert_eq!(std::fs::metadata(log).unwrap().len(), log::HEADER_2_BYTES as u64);
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
    let hooked = Index::of(Inner::open(&path).unwrap().with_wave(|| Box::<Hooked<2>>::default()));
    let (w, v) = (Rc::clone(&writer), q.clone());
    HOOK.set(Some(Box::new(move || w.borrow_mut().insert(5, &v).unwrap())));

    assert_eq!(one(&hooked), 1);
    assert_eq!(one(&index), 1);

    // The same in a search over several collections, with the next change.
    let set = Collections {
        indexes: vec![Index::of(
            Inner::open(&path).unwrap().with_wave(|| Box::<Hooked<2>>::default()),
        )],
    };
    let (w, v) = (Rc::clone(&writer), q.clone());
    HOOK.set(Some(Box::new(move || w.borrow_mut().insert(5, &v).unwrap())));
    let mut out = Vec::new();
    set.search(&q, &everything(), &mut ctx(), &mut out).unwrap();
    assert_eq!(out.iter().filter(|(_, h)| h.id == 5).count(), 1);
    assert_eq!(one(&index), 1);

    // On its own, a later change hides only what came before it, and a snapshot from before judges the row as it was.
    let inner = index.inner();
    let (data, dim) = (inner.in_place(), inner.layout().dim);
    let row = (0..600).find(|&r| sq8::id(data.rows.row(r), dim) == 6).unwrap();
    let mut a = Applier::new(inner.layout(), Version::One, &inner.located().data);
    let before = a.snapshot();
    a.apply(
        &log::encode(
            &Record::Delete {
                seq: 0,
                id: 6,
                hides: log::NONE,
            },
            Version::One,
        ),
        false,
    )
    .unwrap();
    let after = a.snapshot();
    assert!(before.row_visible(row, before.seen()) && after.row_visible(row, before.seen()));
    assert!(!after.row_visible(row, after.seen()));
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

#[test]
fn an_insert_scores_like_the_same_vector_built_into_the_file() {
    let (dim, n, d_pad) = (128, 600, padded_dim(128));
    let (rows, mut clusters) = clustered(&mut SplitMix64::new(66), n, dim, 12, 1.0);
    group(&mut clusters, 3);
    let (cb, size) = (code_bytes(d_pad), block_bytes(d_pad));
    let queries = gaussian(&mut SplitMix64::new(67), 4, dim, |_| 1.0);

    for anchor in [Anchor::Cluster, Anchor::Mean] {
        let p = BuildParams {
            anchor,
            ..params(Metric::L2)
        };
        let memory = InMemory::build(&p, &rows, &clusters).unwrap();
        let path = temp_file(&format!("same-score-{anchor:?}.coldivf"));
        write(&memory, &path).unwrap();
        let _ = std::fs::remove_file(log_path(&path));
        let index = Index::open(&path).unwrap();
        let mut writer = Writer::open(&path).unwrap();
        let mut c = ctx();
        // Rows that route to the cluster the build put them in go in again under new ids, so insert i has internal id
        // n + i. Each pair holds the row's internal id in the file and its cluster.
        let mut pairs = Vec::new();
        for r in 0..80 {
            let v = &rows.data[r * dim..(r + 1) * dim];
            let built = memory.ids.iter().position(|&id| id == r as u64).unwrap() as u32;
            let leaf = memory.lists.partition_point(|l| l.first_id + l.count as u32 <= built);
            let mut routed = Vec::new();
            index.route(v, 1, None, &mut c, &mut routed).unwrap();
            if routed[0] as usize == leaf {
                writer.insert(10_000 + r as u64, v).unwrap();
                pairs.push((built, leaf));
            }
        }
        assert!(
            pairs.len() >= 60,
            "{anchor:?}: only {} rows route to their own cluster",
            pairs.len()
        );

        // Each insert has the code and the factors of its row in the file.
        let inner = index.inner();
        let delta = inner.changes().load();
        let mut bits = vec![0u8; cb];
        for (i, &(built, leaf)) in pairs.iter().enumerate() {
            let list = memory.lists[leaf];
            let j = (built - list.first_id) as usize;
            let block = &memory.blocks[list.offset + j / BLOCK * size..list.offset + (j / BLOCK + 1) * size];
            let factors = unpack_lane(block, d_pad, j % BLOCK, &mut bits);
            let (code, f) = delta.lane(i);
            assert_eq!(code, bits, "{anchor:?}: insert {i}");
            assert_eq!(f, factors, "{anchor:?}: insert {i}");
        }
        drop(delta);

        // So every search path gives each insert the estimate of its row in the file.
        for q in queries.chunks_exact(dim).chain(rows.data[..2 * dim].chunks_exact(dim)) {
            for how in ["mapped", "cold", "warm"] {
                index.set_warm(how == "warm");
                match how {
                    "mapped" => index.inner().search_mapped(q, &everything(), &mut c).map(|_| ()),
                    _ => index.search(q, &everything(), &mut c).map(|_| ()),
                }
                .unwrap();
                let s = c.scratch.iter().find(|s| s.fits(dim)).unwrap();
                let score = |id: u32| {
                    let found = s.candidates.as_slice().iter().find(|x| x.id == id);
                    found.map(|x| x.score.to_bits())
                };
                for (i, &(built, _)) in pairs.iter().enumerate() {
                    let insert = (n + i) as u32;
                    assert!(
                        score(built).is_some(),
                        "{anchor:?}, {how}: row {built} isn't a candidate"
                    );
                    assert_eq!(
                        score(built),
                        score(insert),
                        "{anchor:?}, {how}: rows {built} and {insert}"
                    );
                }
            }
        }
    }
}

#[test]
fn an_open_while_a_run_reads_the_changes_waits_for_nothing() {
    let (path, rows) = sample("held-open.coldivf", Metric::L2);
    let mut writer = Writer::open(&path).unwrap();
    writer.insert(9000, &rows.data[..128]).unwrap();
    // A maintenance run holds the read lock of the changes for its whole length, as this guard does. An open that
    // waited for the write lock would make every later search of the file wait too.
    let inner = writer.index.inner();
    let guard = inner.changes().load();
    let (send, opened) = std::sync::mpsc::channel();

    std::thread::scope(|scope| {
        scope.spawn(|| {
            let (index, mut c) = (Index::open(&path).unwrap(), ctx());
            let hits = index.search(&rows.data[..128], &everything(), &mut c).unwrap();
            send.send(hits.iter().map(|h| h.id).collect::<Vec<u64>>()).unwrap();
        });
        let hits = opened.recv_timeout(std::time::Duration::from_secs(10));
        drop(guard);
        assert!(hits.is_ok_and(|h| h.contains(&9000)), "the open waited for the run");
    });
}

/// The external ids of the candidates that the last search with `c` kept, sorted.
fn candidate_ids(index: &Index, c: &SearchCtx) -> Vec<u64> {
    let inner = index.inner();
    let (data, layout, delta) = (inner.in_place(), inner.layout(), inner.changes().load());
    let s = c.scratch.iter().find(|s| s.fits(layout.dim)).unwrap();
    let mut ids: Vec<u64> = s
        .candidates
        .as_slice()
        .iter()
        .map(|cand| sq8::id(inner.row_of(&delta, cand.id, |id| data.rows.row(id)), layout.dim))
        .collect();
    ids.sort_unstable();
    ids
}

#[test]
fn deleted_rows_take_no_rescore_place_and_every_path_finds_what_the_folded_file_finds() {
    use crate::maintain::tests::{DIM, build_params, sample};

    let (path, rows) = sample(
        "deleted.coldivf",
        &build_params(Metric::L2, Anchor::Cluster),
        3000,
        30,
        80,
    );
    let folded = temp_file("deleted-folded.coldivf");
    crate::quantization::testutil::copy_index(&path, &folded);
    let mut rng = SplitMix64::new(81);
    let gone: Vec<u64> = (0..300).map(|_| rng.next_u64() % 3000).collect();
    let mut writer = Writer::open(&path).unwrap();
    let mut other = Writer::open(&folded).unwrap();

    for &id in &gone {
        writer.delete(id).unwrap();
        other.delete(id).unwrap();
    }

    // A fold keeps every cluster, its center and its order, and copies every code, so the two files differ only in
    // where the deleted rows went.
    other.compact().unwrap();
    let (index, fold) = (Index::open(&path).unwrap(), Index::open(&folded).unwrap());
    let limits = Limits {
        max_k: 10,
        max_nprobe: 30,
        max_rerank: 400,
    };

    for (nprobe, rerank) in [(4, 20), (8, 60), (30, 400)] {
        let sp = SearchParams {
            k: 10,
            nprobe,
            rerank,
            reach: None,
        };
        for q in rows.data.as_chunks::<DIM>().0.iter().step_by(150).map(|q| &q[..]) {
            let (mut a, mut b) = (SearchCtx::new(limits), SearchCtx::new(limits));
            for warm in [false, true] {
                index.set_warm(warm);
                fold.set_warm(warm);
                let hits = |index: &Index, c: &mut SearchCtx| -> Vec<(u64, u32)> {
                    let hits = index.search(q, &sp, c).unwrap();
                    hits.iter().map(|h| (h.id, h.score.to_bits())).collect()
                };
                assert_eq!(
                    hits(&index, &mut a),
                    hits(&fold, &mut b),
                    "nprobe {nprobe}, warm {warm}"
                );
                let kept = candidate_ids(&index, &a);
                assert!(
                    kept.iter().all(|id| !gone.contains(id)),
                    "a deleted row took a rescore place"
                );
                assert_eq!(kept, candidate_ids(&fold, &b), "nprobe {nprobe}, warm {warm}");
            }
        }
    }
}

#[test]
fn pending_inserts_add_no_read_to_a_cold_search() {
    let (path, rows) = sample("no-reads.coldivf", Metric::L2);
    let index = Index::open(&path).unwrap();
    let mut writer = Writer::open(&path).unwrap();
    let sp = SearchParams {
        k: 10,
        nprobe: 4,
        rerank: 40,
        reach: None,
    };
    let queries: Vec<&[f32]> = rows
        .data
        .as_chunks::<128>()
        .0
        .iter()
        .step_by(60)
        .map(|q| &q[..])
        .collect();
    let traffic = |c: &SearchCtx| (c.trace.cluster_bytes, c.trace.cluster_ranges);
    let before: Vec<_> = queries
        .iter()
        .map(|q| {
            let mut c = ctx();
            index.search(q, &sp, &mut c).unwrap();
            (traffic(&c), c.trace.row_bytes)
        })
        .collect();

    // Inserts near the queries, so they win candidates' places.
    for (i, q) in queries.iter().enumerate() {
        for k in 0..30 {
            let v: Vec<f32> = q.iter().map(|&x| x + 0.01 * (k as f32 + 1.0)).collect();
            writer.insert(50_000 + (i * 100 + k) as u64, &v).unwrap();
        }
    }

    for (q, (clusters, rows_before)) in queries.iter().zip(before) {
        let mut c = ctx();
        let hits = index.search(q, &sp, &mut c).unwrap();
        assert!(hits.iter().filter(|h| h.id >= 50_000).count() >= 5);
        // The same clusters, read the same way, and no more rows from the file, since the inserts' rows are in memory.
        assert_eq!(traffic(&c), clusters);
        assert!(c.trace.row_bytes <= rows_before);
    }
}

/// Sets its flag when dropped, eg. as a thread that fails a check unwinds.
struct Stops<'a>(&'a AtomicBool);

impl Drop for Stops<'_> {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Release);
    }
}

#[test]
fn searches_take_no_lock_and_see_every_id_once_while_a_writer_upserts() {
    let (path, rows) = sample("upserts.coldivf", Metric::L2);
    let index = Index::open(&path).unwrap();
    let done = AtomicBool::new(false);
    let limits = Limits {
        max_k: 600,
        max_nprobe: 12,
        max_rerank: 800,
    };
    let sp = SearchParams {
        k: 600,
        nprobe: 12,
        rerank: 800,
        reach: None,
    };
    let searches = std::sync::atomic::AtomicUsize::new(0);
    // A searcher that stops early stops the writer, so a failed check fails the test rather than leaves it waiting.
    let stopped = AtomicBool::new(false);

    std::thread::scope(|scope| {
        for t in 0..4 {
            let (index, done, searches, rows, stopped) = (&index, &done, &searches, &rows, &stopped);
            scope.spawn(move || {
                let _stops = Stops(stopped);
                let mut c = SearchCtx::new(limits);
                let q = &rows.data[t * 128..(t + 1) * 128];
                index.search(q, &sp, &mut c).unwrap();
                while !done.load(Ordering::Acquire) {
                    let locks = LOCKS.with(|n| n.get());
                    let mut ids: Vec<u64> = index.search(q, &sp, &mut c).unwrap().iter().map(|h| h.id).collect();
                    assert_eq!(LOCKS.with(|n| n.get()), locks, "a search took a lock");
                    ids.sort_unstable();
                    assert!(
                        ids == (0..600).collect::<Vec<u64>>(),
                        "a search saw an id twice or missed one"
                    );
                    searches.fetch_add(1, Ordering::Relaxed);
                }
            });
        }

        // The writer replaces ids 0 to 99 with vectors near their own, over and over, and folds now and then, which
        // moves every reader to a new generation.
        let mut writer = Writer::open(&path).unwrap();
        let mut rng = SplitMix64::new(82);
        let started = std::time::Instant::now();
        let mut changes = 0;
        let enough = || started.elapsed().as_secs() >= 3 && searches.load(Ordering::Relaxed) >= 40;
        while !enough() && !stopped.load(Ordering::Acquire) {
            let id = rng.next_u64() % 100;
            let v: Vec<f32> = rows.data[id as usize * 128..(id as usize + 1) * 128]
                .iter()
                .map(|&x| x + 0.2 * rng.normal() as f32)
                .collect();
            writer.insert(id, &v).unwrap();
            changes += 1;
            if changes % 500 == 0 {
                writer.compact().unwrap();
            }
        }
        done.store(true, Ordering::Release);
        assert!(changes > 500, "the writer folded nothing");
    });
}

#[test]
fn an_insert_into_a_cluster_with_no_vector_in_the_file_is_found() {
    // Cluster 5 has a center but no vector, far from the others, so an insert near it routes there.
    let (rows, mut clusters) = clustered(&mut SplitMix64::new(83), 600, 128, 6, 0.5);
    clusters.assignment.iter_mut().for_each(|c| *c = (*c).min(4));
    clusters.centroids[5 * 128..6 * 128].iter_mut().for_each(|x| *x += 20.0);
    group(&mut clusters, 2);
    let index = InMemory::build(&params(Metric::L2), &rows, &clusters).unwrap();
    let path = temp_file("empty-cluster.coldivf");
    write(&index, &path).unwrap();
    let opened = Index::open(&path).unwrap();
    let empty = opened.inner().tables().unwrap().lists.iter().position(|l| l.count == 0);
    assert!(empty.is_some(), "the file has no empty cluster");
    let v = clusters.centroids[5 * 128..6 * 128].to_vec();
    Writer::open(&path).unwrap().insert(9000, &v).unwrap();
    let sp = SearchParams {
        k: 1,
        nprobe: 1,
        rerank: 10,
        reach: None,
    };

    for warm in [false, true] {
        opened.set_warm(warm);
        let mut c = ctx();
        assert_eq!(opened.search(&v, &sp, &mut c).unwrap()[0].id, 9000, "warm {warm}");
        let mut leaves = Vec::new();
        opened.route(&v, 1, None, &mut c, &mut leaves).unwrap();
        let hits = opened.fetch(&leaves, &mut c).unwrap().search(&v, 1, 10).unwrap();
        assert_eq!(hits[0].id, 9000);
        // A search of clusters that a guess at the query already scanned scans them again.
        let mut fetched = opened.fetch(&leaves, &mut c).unwrap();
        fetched.prefetch_rows(&v, 10).unwrap();
        assert_eq!(
            fetched.search(&v, 1, 10).unwrap()[0].id,
            9000,
            "warm {warm}, after a guess"
        );
    }

    let mut c = ctx();
    assert_eq!(opened.inner().search_mapped(&v, &sp, &mut c).unwrap()[0].id, 9000);
    let (set, mut out) = (Collections::open(&[&*path]).unwrap(), Vec::new());
    set.search(&v, &sp, &mut c, &mut out).unwrap();
    assert_eq!(out[0].1.id, 9000);

    // With the next cluster probed too, and an insert in it, the insert near the empty cluster's center keeps the
    // estimate that it gets alone, on every path, since the search measures the empty cluster's distance itself.
    let mut leaves = Vec::new();
    opened.route(&v, 2, None, &mut c, &mut leaves).unwrap();
    let next = opened.inner().tables().unwrap().lists[leaves[1] as usize].count;
    assert!(
        leaves[0] == 5 && next > 0,
        "the query doesn't probe the empty cluster first"
    );
    let center = clusters.centroids[leaves[1] as usize * 128..(leaves[1] as usize + 1) * 128].to_vec();
    Writer::open(&path).unwrap().insert(9001, &center).unwrap();
    let estimate = |nprobe: usize, how: &str| {
        let sp = SearchParams { nprobe, ..sp };
        let mut c = ctx();
        opened.set_warm(how == "warm");
        match how {
            "mapped" => opened.inner().search_mapped(&v, &sp, &mut c).map(|_| ()),
            _ => opened.search(&v, &sp, &mut c).map(|_| ()),
        }
        .unwrap();
        let s = c.scratch.iter().find(|s| s.fits(128)).unwrap();
        let first = opened.inner().layout().count as u32;
        s.candidates
            .as_slice()
            .iter()
            .find(|x| x.id == first)
            .map(|x| x.score.to_bits())
    };

    for how in ["mapped", "cold", "warm"] {
        assert!(
            estimate(1, how).is_some() && estimate(1, how) == estimate(2, how),
            "{how}"
        );
    }
}

/// The dimension of the files of `gapped`.
const GAPPED: usize = 128;

/// An index file of 6400 vectors with ids 0 to 6399 in 320 tight clusters, more than the 256 whose pending inserts a
/// search scans, and the clusters' centers. Each cluster keeps its row's position in the file.
fn gapped(name: &str) -> (TempFile, Rows, Vec<f32>) {
    let (rows, mut clusters) = clustered(&mut SplitMix64::new(91), 6400, GAPPED, 320, 0.2);
    group(&mut clusters, 8);
    let index = InMemory::build(&params(Metric::L2), &rows, &clusters).unwrap();
    let path = temp_file(name);
    write(&index, &path).unwrap();
    let _ = std::fs::remove_file(log_path(&path));
    (path, rows, clusters.centroids)
}

/// Room to route every cluster of a file of `gapped`.
const ROOM: Limits = Limits {
    max_k: 10,
    max_nprobe: 320,
    max_rerank: 60,
};

#[test]
fn queries_aimed_at_fresh_inserts_find_them_in_clusters_past_the_ones_they_read() {
    let (path, rows, _) = gapped("aimed.coldivf");
    let folded = temp_file("aimed-folded.coldivf");
    crate::quantization::testutil::copy_index(&path, &folded);
    // Each query sits in a gap between the clusters, nearly as far from each of a few dozen centers, with 10 inserts
    // close around it, nearer than any vector of the file. Each insert joins its own nearest cluster, which the query
    // may rank past the few clusters it reads.
    let mut rng = SplitMix64::new(92);
    let queries = gaussian(&mut rng, 40, GAPPED, |_| 1.0);
    let mut inserts: Vec<f32> = Vec::new();

    for q in queries.as_chunks::<GAPPED>().0 {
        for _ in 0..10 {
            inserts.extend(q.iter().map(|&x| x + 0.5 * rng.normal() as f32));
        }
    }

    let (mut writer, mut other) = (Writer::open(&path).unwrap(), Writer::open(&folded).unwrap());

    for (i, v) in inserts.as_chunks::<GAPPED>().0.iter().enumerate() {
        writer.insert(100_000 + i as u64, v).unwrap();
        other.insert(100_000 + i as u64, v).unwrap();
    }

    other.compact().unwrap();
    let (index, fold) = (Index::open(&path).unwrap(), Index::open(&folded).unwrap());
    let truth: Vec<Vec<u64>> = queries
        .as_chunks::<GAPPED>()
        .0
        .iter()
        .map(|q| {
            let file = rows.data.as_chunks::<GAPPED>().0.iter().enumerate();
            let inserted = inserts.as_chunks::<GAPPED>().0.iter().enumerate();
            let mut all: Vec<(f64, u64)> = file
                .map(|(i, r)| (dist2(r, q), i as u64))
                .chain(inserted.map(|(i, v)| (dist2(v, q), 100_000 + i as u64)))
                .collect();
            all.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            all[..10].iter().map(|x| x.1).collect()
        })
        .collect();
    // The share of the true 10 nearest that are inserts and that `search` finds, for `nprobe`.
    let recall = |nprobe: usize, search: &mut dyn FnMut(&[f32], &SearchParams) -> Vec<u64>| {
        let sp = SearchParams {
            k: 10,
            nprobe,
            rerank: 60,
            reach: None,
        };
        let (mut found, mut inserted) = (0, 0);
        for (q, t) in queries.as_chunks::<GAPPED>().0.iter().zip(&truth) {
            let hits = search(q, &sp);
            let want: Vec<u64> = t.iter().copied().filter(|&id| id >= 100_000).collect();
            inserted += want.len();
            found += want.iter().filter(|id| hits.contains(id)).count();
        }
        assert!(inserted >= 300, "only {inserted} of the true neighbors are inserts");
        found as f64 / inserted as f64
    };
    let mut c = SearchCtx::new(ROOM);
    let mut ids = |index: &Index, q: &[f32], sp: &SearchParams| -> Vec<u64> {
        index.search(q, sp, &mut c).unwrap().iter().map(|h| h.id).collect()
    };
    let read = recall(4, &mut |q, sp| ids(&fold, q, sp));
    let all_read = recall(256, &mut |q, sp| ids(&fold, q, sp));
    let aimed = recall(4, &mut |q, sp| ids(&index, q, sp));

    // A fold leaves an insert in its cluster, so a search finds it only when it reads that cluster. Scanning the
    // pending inserts of the 256 nearest clusters finds about what reading all of them would.
    assert!(
        read <= 0.85,
        "a fold finds {read:.3} at nprobe 4, so the inserts tell nothing"
    );
    assert!(
        aimed >= 0.95 && aimed >= all_read - 0.02,
        "{aimed:.3}, against {all_read:.3} reading 256"
    );

    // Every path finds the same.
    let sp = SearchParams {
        k: 10,
        nprobe: 4,
        rerank: 60,
        reach: None,
    };
    let set = Collections::open(&[&*path]).unwrap();
    let mut c = SearchCtx::new(ROOM);

    for q in queries.as_chunks::<GAPPED>().0 {
        let mut paths: Vec<Vec<(u64, u32)>> = Vec::new();
        let pairs = |hits: &[Hit]| hits.iter().map(|h| (h.id, h.score.to_bits())).collect::<Vec<_>>();
        for warm in [false, true] {
            index.set_warm(warm);
            paths.push(pairs(index.search(q, &sp, &mut c).unwrap()));
            set.indexes[0].set_warm(warm);
            let mut out = Vec::new();
            set.search(q, &sp, &mut c, &mut out).unwrap();
            paths.push(out.iter().map(|(_, h)| (h.id, h.score.to_bits())).collect());
        }
        paths.push(pairs(index.inner().search_mapped(q, &sp, &mut c).unwrap()));
        let mut leaves = Vec::new();
        index.route(q, 4, None, &mut c, &mut leaves).unwrap();
        paths.push(pairs(index.fetch(&leaves, &mut c).unwrap().search(q, 10, 60).unwrap()));
        assert!(paths.iter().all(|p| *p == paths[0]), "the paths disagree");
    }
}

#[test]
fn a_search_scans_the_pending_inserts_of_the_256_nearest_clusters_once_each() {
    let (path, _, centers) = gapped("next-clusters.coldivf");
    let mut writer = Writer::open(&path).unwrap();

    // Two inserts at every cluster's center, which each join that cluster.
    for (c, center) in centers.as_chunks::<GAPPED>().0.iter().enumerate() {
        for k in 0..2 {
            let v: Vec<f32> = center.iter().map(|&x| x + 0.01 * (k as f32 + 1.0)).collect();
            writer.insert(50_000 + (2 * c + k) as u64, &v).unwrap();
        }
    }

    let index = Index::open(&path).unwrap();
    let held = |l: u32| index.inner().changes().load().groups(l).map(|g| g.len()).sum::<usize>();
    assert!((0..320).all(|l| held(l) == 2), "an insert joined another cluster");
    let q = gaussian(&mut SplitMix64::new(93), 1, GAPPED, |_| 1.0);
    let mut c = SearchCtx::new(ROOM);
    let mut order = Vec::new();
    index.route(&q, 320, None, &mut c, &mut order).unwrap();
    let count = |l: u32| index.inner().tables().unwrap().lists[l as usize].count;
    let read: usize = order[..8].iter().map(|&l| count(l)).sum();

    // Whatever it reads, a search scans the inserts of the 256 nearest clusters, or of more when it reads more, and
    // the vectors in the file of only the clusters it reads.
    for (nprobe, reach, clusters, vectors) in [
        (8, None, 256, Some(read)),
        (300, None, 300, None),
        (8, Some(0.0), 256, None),
    ] {
        let sp = SearchParams {
            k: 10,
            nprobe,
            rerank: 60,
            reach,
        };
        let what = format!("nprobe {nprobe}, reach {reach:?}");
        let reads = |c: &SearchCtx| vectors.is_none_or(|v| c.trace.scanned == v);
        for warm in [false, true] {
            index.set_warm(warm);
            index.search(&q, &sp, &mut c).unwrap();
            assert_eq!(c.trace.inserts_scanned, 2 * clusters, "{what}, warm {warm}");
            assert!(reads(&c), "{what}, warm {warm}: {} vectors", c.trace.scanned);
        }
        index.set_warm(false);
        index.inner().search_mapped(&q, &sp, &mut c).unwrap();
        assert_eq!(c.trace.inserts_scanned, 2 * clusters, "{what}, mapped");
        assert!(reads(&c), "{what}, mapped: {} vectors", c.trace.scanned);
        let mut leaves = Vec::new();
        index.route(&q, nprobe, reach, &mut c, &mut leaves).unwrap();
        index.fetch(&leaves, &mut c).unwrap().search(&q, 10, 60).unwrap();
        assert_eq!(c.trace.inserts_scanned, 2 * clusters, "{what}, fetched");
    }

    // A fetch of clusters of the caller's choosing, one of them among those that the route ranked next, scans the
    // inserts of each cluster once.
    let mut leaves = Vec::new();
    index.route(&q, 8, None, &mut c, &mut leaves).unwrap();
    let hits = index
        .fetch(&[order[3], order[100]], &mut c)
        .unwrap()
        .search(&q, 10, 60)
        .unwrap();
    let mut ids: Vec<u64> = hits.iter().map(|h| h.id).collect();
    ids.sort_unstable();
    ids.dedup();
    assert_eq!(ids.len(), 10);
    // The two fetched clusters and the 247 others of the 248 ranked next.
    assert_eq!(c.trace.inserts_scanned, 2 * 249);

    // A route of another index, a copy with the same inserts that reads 100 clusters, keeps the clusters that this
    // index's last route ranked next, and a fetch of the other index's route scans the inserts of the 156 that it
    // ranked next.
    let twin = temp_file("next-clusters-twin.coldivf");
    crate::quantization::testutil::copy_index(&path, &twin);
    let other = Index::open(&twin).unwrap();
    let mut other_leaves = Vec::new();
    other.route(&q, 100, None, &mut c, &mut other_leaves).unwrap();
    index
        .fetch(&[order[3], order[100]], &mut c)
        .unwrap()
        .search(&q, 10, 60)
        .unwrap();
    assert_eq!(c.trace.inserts_scanned, 2 * 249);
    other
        .fetch(&other_leaves[..2], &mut c)
        .unwrap()
        .search(&q, 10, 60)
        .unwrap();
    assert_eq!(c.trace.inserts_scanned, 2 * 158);

    // A fetch with a context that never routed this index scans the inserts of the fetched clusters alone.
    let mut fresh = SearchCtx::new(ROOM);
    index
        .fetch(&[order[3], order[100]], &mut fresh)
        .unwrap()
        .search(&q, 10, 60)
        .unwrap();
    assert_eq!(fresh.trace.inserts_scanned, 2 * 2);

    // An insert of a cluster ranked next gets the estimate that it gets once a search reads its cluster.
    let all = Limits {
        max_rerank: 3000,
        ..ROOM
    };
    let estimates = |nprobe: usize| -> Vec<u32> {
        let mut c = SearchCtx::new(all);
        let sp = SearchParams {
            k: 10,
            nprobe,
            rerank: 3000,
            reach: None,
        };
        index.inner().search_mapped(&q, &sp, &mut c).unwrap();
        let s = c.scratch.iter().find(|s| s.fits(GAPPED)).unwrap();
        let first = 6400 + 2 * order[99];
        let score = |id: u32| {
            s.candidates
                .as_slice()
                .iter()
                .find(|x| x.id == id)
                .map(|x| x.score.to_bits())
        };
        (first..first + 2)
            .map(|id| score(id).expect("an insert isn't a candidate"))
            .collect()
    };
    assert_eq!(estimates(8), estimates(120));

    // Two collections of the same file rank each cluster twice, so the 256 nearest overall are 128 of each.
    let (set, mut out) = (Collections::open(&[&*path, &*path]).unwrap(), Vec::new());
    let sp = SearchParams {
        k: 10,
        nprobe: 8,
        rerank: 60,
        reach: None,
    };

    for warm in [false, true] {
        set.indexes.iter().for_each(|i| i.set_warm(warm));
        set.search(&q, &sp, &mut c, &mut out).unwrap();
        assert_eq!(c.trace.inserts_scanned, 2 * 256, "two collections, warm {warm}");
    }
}
