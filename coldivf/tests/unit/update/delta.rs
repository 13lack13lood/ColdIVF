//! Unit tests for the snapshots of a data file's changes and the applier that makes them.

use super::*;
use crate::Metric;
use crate::quantization::testutil::{TempFile, params, written};
use crate::search::inmem::InMemory;

const DIM: usize = 128;

fn two() -> Version {
    Version::Two {
        index_id: 9,
        generation: 1,
        base: 1,
    }
}

/// A file of 600 vectors in 6 clusters, its ids by row, and an applier of its log of `version`.
fn applier(name: &str, version: Version) -> (Applier, InMemory, TempFile) {
    let (memory, inner, path) = written(name, &params(Metric::L2), 600, DIM, 6);
    let a = Applier::new(inner.layout(), version, &path);
    (a, memory, path)
}

/// An insert record of `id` into cluster `list` with a row that holds the id.
fn insert(a: &Applier, id: u64, list: u32, hides: u32) -> Vec<u8> {
    let (code, mut row) = (
        vec![id as u8; code_bytes(padded_dim(DIM))],
        vec![0u8; sq8::row_bytes(DIM)],
    );
    sq8::write_row(&vec![0.5; DIM], &vec![0.0; DIM], id, &mut row);
    let f = Factors {
        f_add: 1.0,
        f_rescale: -1.0,
        f_pop: 3.0,
        f_err: 0.5,
    };
    let r = Record::Insert {
        seq: a.next(),
        id,
        list,
        hides,
        code: &code,
        factors: f,
        row: &row,
    };
    log::encode(&r, a.version())
}

fn delete(a: &Applier, id: u64, hides: u32) -> Vec<u8> {
    log::encode(
        &Record::Delete {
            seq: a.next(),
            id,
            hides,
        },
        a.version(),
    )
}

/// The internal id of the row whose id is `id`.
fn row_of(memory: &InMemory, id: u64) -> u32 {
    memory.ids.iter().position(|&x| x == id).unwrap() as u32
}

#[test]
fn a_snapshot_keeps_what_it_saw_whatever_comes_later() {
    for version in [Version::One, two()] {
        let (mut a, memory, _path) = applier(&format!("snap-{}.coldivf", version.base()), version);
        let r7 = row_of(&memory, 7);
        for (id, list) in [(1000, 0), (1001, 3), (1002, 3)] {
            let hint = a.locate(id).unwrap();
            let record = insert(&a, id, list, hint);
            a.apply(&record, false).unwrap();
        }
        let hint = a.locate(7).unwrap();
        a.apply(&delete(&a, 7, hint), false).unwrap();
        let s = a.snapshot();
        let seen = s.seen();
        assert_eq!((s.len(), s.next()), (3, version.base() + 4));
        assert!(!s.row_visible(r7, seen) && (0..3).all(|i| s.insert_visible(i, seen)));

        // A delete of an insert and an upsert of a row, after the snapshot.
        let hint = a.locate(1001).unwrap();
        a.apply(&delete(&a, 1001, hint), false).unwrap();
        let r8 = row_of(&memory, 8);
        let hint = a.locate(8).unwrap();
        a.apply(&insert(&a, 8, 1, hint), false).unwrap();
        let later = a.snapshot();
        assert_eq!(s.len(), 3);
        assert!(s.insert_visible(1, seen) && s.row_visible(r8, seen), "{version:?}");
        assert!(!later.insert_visible(1, later.seen()) && !later.row_visible(r8, later.seen()));
        assert!(later.row_replaced(r8, later.seen()) && !later.row_replaced(r7, later.seen()));
        assert_eq!(later.groups(3).len(), 2);
        assert!(s.groups(1).is_empty());
    }
}

#[test]
fn an_upsert_hides_the_old_version_exactly_when_the_new_one_shows() {
    let (mut a, memory, _path) = applier("upsert.coldivf", two());
    let r5 = row_of(&memory, 5);

    for round in 0..3 {
        let before = a.snapshot();
        let hint = a.locate(5).unwrap();
        a.apply(&insert(&a, 5, 2, hint), false).unwrap();
        let after = a.snapshot();
        // Every version of id 5 that each snapshot shows, which is always exactly one.
        for s in [&before, &after] {
            let rows = usize::from(s.row_visible(r5, s.seen()));
            let inserts = (0..s.len()).filter(|&i| s.insert_visible(i, s.seen())).count();
            assert_eq!(rows + inserts, 1, "round {round}");
        }
        assert!(after.insert_visible(round, after.seen()) && !after.row_visible(r5, after.seen()));
    }
}

#[test]
fn a_wrong_hint_costs_a_lookup_and_never_visibility() {
    let (mut a, memory, _path) = applier("hints.coldivf", two());
    let (r3, r4) = (row_of(&memory, 3), row_of(&memory, 4));

    for id in [1000, 1001] {
        let hint = a.locate(id).unwrap();
        a.apply(&insert(&a, id, 0, hint), false).unwrap();
    }

    // Hints of another id's row, of a place past the inserts, of an insert of another id and of a hidden row, each
    // for a change of a vector that the file or the inserts hold.
    a.apply(&delete(&a, 3, r4), false).unwrap();
    a.apply(&delete(&a, 4, 600 + 9), false).unwrap();
    a.apply(&delete(&a, 1001, 600), false).unwrap();
    a.apply(&insert(&a, 1000, 1, r3), false).unwrap();
    let s = a.snapshot();
    let seen = s.seen();
    assert!(!s.row_visible(r3, seen) && !s.row_visible(r4, seen));
    assert!(!s.insert_visible(0, seen) && !s.insert_visible(1, seen) && s.insert_visible(2, seen));
    // A hint that names a hidden version is looked up again too, and finds nothing more to hide.
    a.apply(&delete(&a, 3, r3), false).unwrap();
    assert!(a.snapshot().row_visible(row_of(&memory, 2), seen));
}

#[test]
fn a_replay_packs_its_inserts_once_and_shows_what_one_by_one_shows() {
    let (mut one, memory, _path) = applier("replay.coldivf", two());
    let mut records = Vec::new();

    for i in 0..300u64 {
        let id = if i % 7 == 0 { i % 600 } else { 1000 + i };
        let hint = one.locate(id).unwrap();
        let record = match i % 11 {
            5 => delete(&one, id, hint),
            _ => insert(&one, id, (i % 6) as u32, hint),
        };
        one.apply(&record, false).unwrap();
        records.push(record);
    }

    let mut bytes: Vec<u8> = records.iter().flat_map(|r| log::frame(r)).collect();
    let mut replayed = Applier::new(
        &crate::index::Inner::open(&_path).unwrap().layout().clone(),
        two(),
        &_path,
    );
    replayed.replay(&bytes, 50).unwrap();
    let (a, b) = (one.snapshot(), replayed.snapshot());
    assert_eq!((a.len(), a.next()), (b.len(), b.next()));
    assert_eq!(b.log_len(), 50 + bytes.len());

    for r in 0..600 {
        assert_eq!(a.row_visible(r, a.seen()), b.row_visible(r, b.seen()), "row {r}");
    }

    for i in 0..a.len() {
        assert_eq!(
            a.insert_visible(i, a.seen()),
            b.insert_visible(i, b.seen()),
            "insert {i}"
        );
        assert!(a.row(i) == b.row(i) && a.lane(i) == b.lane(i));
    }

    for list in 0..6 {
        let places = |s: &Delta| {
            s.groups(list)
                .iter()
                .flat_map(|g| (0..g.len()).map(move |j| g.place(j)))
                .collect::<Vec<_>>()
        };
        assert_eq!(places(&a), places(&b), "cluster {list}");
    }

    // A damaged middle record fails a version 2 replay, which keeps the records before it.
    bytes[8 + records[0].len() + 8 + 3] ^= 1;
    let mut damaged = Applier::new(
        &crate::index::Inner::open(&_path).unwrap().layout().clone(),
        two(),
        &_path,
    );
    assert!(damaged.replay(&bytes, 0).is_err());
    let _ = memory;
}

#[test]
fn a_change_at_a_snapshot_s_next_sequence_number_hides_nothing_in_it() {
    let (mut a, memory, _path) = applier("boundary.coldivf", two());
    // A hidden row and a hidden insert, so that the snapshots hold the stamps of rows and read those of inserts, then
    // the insert that the test hides, at place 1.
    let hint = a.locate(8).unwrap();
    a.apply(&delete(&a, 8, hint), false).unwrap();

    for id in [1001, 1000] {
        let hint = a.locate(id).unwrap();
        a.apply(&insert(&a, id, 0, hint), false).unwrap();
    }

    let hint = a.locate(1001).unwrap();
    a.apply(&delete(&a, 1001, hint), false).unwrap();

    // A row of the file, then an insert, hidden by the first change after a snapshot, whose number is its next.
    let (s, r9) = (a.snapshot(), row_of(&memory, 9));
    let hint = a.locate(9).unwrap();
    a.apply(&delete(&a, 9, hint), false).unwrap();
    let t = a.snapshot();
    let hint = a.locate(1000).unwrap();
    a.apply(&delete(&a, 1000, hint), false).unwrap();
    let u = a.snapshot();
    assert!(s.row_visible(r9, s.seen()) && !t.row_visible(r9, t.seen()));
    assert!(t.insert_visible(1, t.seen()) && !u.insert_visible(1, u.seen()));
}

#[test]
fn a_snapshot_reads_the_stamps_for_an_end_past_its_own() {
    let (mut a, _memory, _path) = applier("later-end.coldivf", two());
    let hint = a.locate(1000).unwrap();
    a.apply(&insert(&a, 1000, 0, hint), false).unwrap();
    let s = a.snapshot();

    // No change that `s` holds hid an insert, but the one after it did, which a later end sees.
    let hint = a.locate(1000).unwrap();
    a.apply(&delete(&a, 1000, hint), false).unwrap();
    let t = a.snapshot();
    assert!(s.insert_visible(0, s.seen()) && !s.insert_visible(0, t.seen()));
}

#[test]
fn hiding_a_hidden_version_again_changes_nothing() {
    let (mut a, memory, _path) = applier("again.coldivf", two());
    let r2 = row_of(&memory, 2);
    let hint = a.locate(1000).unwrap();
    a.apply(&insert(&a, 1000, 0, hint), false).unwrap();

    for (id, slot) in [(1000, 600), (2, r2)] {
        a.apply(&delete(&a, id, slot), false).unwrap();
        let s = a.snapshot();
        // A second delete that names the hidden version, and a third that names nothing, leave it hidden as of `s`.
        a.apply(&delete(&a, id, slot), false).unwrap();
        a.apply(&delete(&a, id, NONE), false).unwrap();
        assert!(!s.visible(slot, s.seen()), "{id}");
    }
}

#[test]
fn a_version_2_record_from_before_hints_were_looked_up_is_looked_up_at_replay() {
    let (mut a, memory, _path) = applier("legacy-hints.coldivf", two());
    let (r7, r8, r9) = (row_of(&memory, 7), row_of(&memory, 8), row_of(&memory, 9));
    // The first version 2 logs wrote every hint as naming nothing, in records whose kind lacks the mark of a lookup.
    let legacy = |mut record: Vec<u8>| {
        record[0] &= !log::LOOKED_UP;
        record
    };
    let mut records: Vec<Vec<u8>> = Vec::new();

    // An insert of a new id, a delete of a row, an upsert of a row, and a delete of the insert, all as legacy records.
    // Then a record that looked up nothing for id 9, so it hides nothing, and an upsert of the upserted id that looked
    // it up, as a later writer appends them to the same log.
    for step in 0..6 {
        let record = match step {
            0 => legacy(insert(&a, 1000, 0, NONE)),
            1 => legacy(delete(&a, 7, NONE)),
            2 => legacy(insert(&a, 8, 1, NONE)),
            3 => legacy(delete(&a, 1000, NONE)),
            4 => delete(&a, 9, NONE),
            _ => {
                let hint = a.locate(8).unwrap();
                insert(&a, 8, 2, hint)
            }
        };
        a.apply(&record, false).unwrap();
        records.push(record);
    }

    let s = a.snapshot();
    let seen = s.seen();
    assert!(!s.row_visible(r7, seen) && !s.row_visible(r8, seen) && s.row_visible(r9, seen));
    assert!(!s.insert_visible(0, seen) && !s.insert_visible(1, seen) && s.insert_visible(2, seen));

    // A replay of the same records shows the same.
    let bytes: Vec<u8> = records.iter().flat_map(|r| log::frame(r)).collect();
    let layout = crate::index::Inner::open(&_path).unwrap().layout().clone();
    let mut replayed = Applier::new(&layout, two(), &_path);
    replayed.replay(&bytes, 0).unwrap();
    let t = replayed.snapshot();

    for r in [r7, r8, r9] {
        assert_eq!(t.row_visible(r, t.seen()), s.row_visible(r, seen), "row {r}");
    }

    for i in 0..3 {
        assert_eq!(t.insert_visible(i, t.seen()), s.insert_visible(i, seen), "insert {i}");
    }
}
