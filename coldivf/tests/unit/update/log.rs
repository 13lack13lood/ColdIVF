//! Unit tests for the log formats: their records, and which damage each version takes as a torn write.

use std::io::ErrorKind;

use super::*;
use crate::quantization::rabitq::Factors;
use crate::quantization::testutil::{params, written};
use crate::update::Applier;

const DIM: usize = 128;

fn version() -> Version {
    Version::Two {
        index_id: 5,
        generation: 3,
        base: 10,
    }
}

/// Records of a file of 128 dimensions: an insert into cluster 2 and a delete, numbered from `seq`.
fn records_from(seq: u64, version: Version) -> Vec<Vec<u8>> {
    let (code, row) = (
        vec![0xA5u8; code_bytes(padded_dim(DIM))],
        vec![7u8; sq8::row_bytes(DIM)],
    );
    let factors = Factors {
        f_add: 1.0,
        f_rescale: -2.0,
        f_pop: 60.0,
        f_err: 0.5,
    };
    let insert = Record::Insert {
        seq,
        id: 42,
        list: 2,
        hides: NONE,
        code: &code,
        factors,
        row: &row,
    };
    let delete = Record::Delete {
        seq: seq + 1,
        id: 42,
        hides: 7,
    };
    let third = Record::Delete {
        seq: seq + 2,
        id: 43,
        hides: NONE,
    };
    [insert, delete, third].iter().map(|r| encode(r, version)).collect()
}

fn framed(records: &[Vec<u8>]) -> Vec<u8> {
    records.iter().flat_map(|r| frame(r)).collect()
}

#[test]
fn records_round_trip_in_both_versions() {
    for v in [Version::One, version()] {
        let records = records_from(10, v);
        let insert = decode(&records[0], v, DIM, 4, 10).unwrap();
        let Record::Insert {
            seq,
            id,
            list,
            hides,
            code,
            factors,
            row,
        } = insert
        else {
            panic!("{v:?}: not an insert");
        };
        assert_eq!((seq, id, list, hides), (10, 42, 2, NONE), "{v:?}");
        assert!(code.iter().all(|&b| b == 0xA5) && row.iter().all(|&b| b == 7), "{v:?}");
        assert_eq!(factors.f_pop, 60.0, "{v:?}");
        // Version 1 keeps no hint for a delete.
        let hides = if v == Version::One { NONE } else { 7 };
        assert_eq!(
            decode(&records[1], v, DIM, 4, 11).unwrap(),
            Record::Delete { seq: 11, id: 42, hides },
            "{v:?}"
        );
    }
}

#[test]
fn a_version_2_record_says_its_hint_is_a_lookup_and_one_without_the_mark_still_decodes() {
    let v = version();

    for (i, record) in records_from(10, v).into_iter().enumerate() {
        let seq = 10 + i as u64;
        assert!(looked_up(&record, v), "record {i}");
        let mut legacy = record.clone();
        legacy[0] &= !LOOKED_UP;
        assert!(!looked_up(&legacy, v), "record {i}");
        assert_eq!(
            decode(&legacy, v, DIM, 4, seq).unwrap(),
            decode(&record, v, DIM, 4, seq).unwrap()
        );
    }

    // Version 1 records carry no hint, so none is a lookup.
    assert!(
        records_from(1, Version::One)
            .iter()
            .all(|r| !looked_up(r, Version::One))
    );
}

#[test]
fn a_version_2_record_must_fit_its_place_and_its_file() {
    let v = version();
    let records = records_from(10, v);

    // Another sequence number, a file with too few clusters, hints of another generation, and a record cut short.
    assert_eq!(
        decode(&records[0], v, DIM, 4, 11).unwrap_err().kind(),
        ErrorKind::InvalidData
    );
    assert_eq!(
        decode(&records[0], v, DIM, 2, 10).unwrap_err().kind(),
        ErrorKind::InvalidData
    );
    let other = Version::Two {
        index_id: 5,
        generation: 4,
        base: 10,
    };
    assert_eq!(
        decode(&records[0], other, DIM, 4, 10).unwrap_err().kind(),
        ErrorKind::InvalidData
    );
    let short = &records[1][..records[1].len() - 1];
    assert_eq!(decode(short, v, DIM, 4, 11).unwrap_err().kind(), ErrorKind::InvalidData);
    assert_eq!(decode(&[], v, DIM, 4, 11).unwrap_err().kind(), ErrorKind::InvalidData);
}

#[test]
fn only_a_damaged_last_frame_is_torn_in_a_version_2_log() {
    let v = version();
    let records = records_from(10, v);
    let whole = framed(&records);
    let ends: Vec<usize> = records
        .iter()
        .scan(0, |at, r| {
            *at += 8 + r.len();
            Some(*at)
        })
        .collect();
    assert_eq!(super::records(&whole, v, DIM).unwrap().0.len(), 3);

    // A last frame cut short, or damaged, or zeros that a power cut left after the records, are torn writes.
    let cut = &whole[..ends[2] - 5];
    assert_eq!(
        super::records(cut, v, DIM).unwrap(),
        (records[..2].iter().map(|r| &r[..]).collect(), ends[1])
    );
    let mut last = whole.clone();
    last[ends[1] + 8 + 3] ^= 1;
    assert_eq!(super::records(&last, v, DIM).unwrap().1, ends[1]);
    let mut zeros = whole.clone();
    zeros.extend([0u8; 100]);
    assert_eq!(super::records(&zeros, v, DIM).unwrap().1, ends[2]);

    // A damaged frame with a whole one after it is damage, not a torn write, in a version 2 log only.
    let mut middle = whole.clone();
    middle[ends[0] + 8 + 3] ^= 1;
    assert_eq!(
        super::records(&middle, v, DIM).unwrap_err().kind(),
        ErrorKind::InvalidData
    );
    let mut old = framed(&records_from(10, Version::One));
    let first = 8 + records_from(10, Version::One)[0].len();
    old[first + 8] ^= 1;
    assert_eq!(
        super::records(&old, Version::One, DIM).unwrap(),
        (vec![&records_from(10, Version::One)[0][..]], first)
    );
}

#[test]
fn a_damaged_length_in_a_version_2_log_is_damage_unless_no_whole_frame_comes_after_it() {
    let v = version();
    let records: Vec<Vec<u8>> = [records_from(10, v), records_from(13, v)].concat();
    let whole = framed(&records);
    let ends: Vec<usize> = records
        .iter()
        .scan(0, |at, r| {
            *at += 8 + r.len();
            Some(*at)
        })
        .collect();
    let first = |n: usize| records[..n].iter().map(|r| &r[..]).collect::<Vec<&[u8]>>();

    // A flipped bit in the length of the second frame, or of the last one, hides where the frame ends. Whole frames
    // after it make it damage, and none make it a torn last write.
    for (frame, start) in [(1, ends[0]), (5, ends[4])] {
        let mut bytes = whole.clone();
        bytes[start + 1] ^= 1;
        let read = super::records(&bytes, v, DIM);
        match frame {
            5 => assert_eq!(read.unwrap(), (first(5), ends[4])),
            _ => assert_eq!(read.unwrap_err().kind(), ErrorKind::InvalidData),
        }
    }

    // A power cut that kept later pages but lost earlier ones leaves zeros from inside a frame, whose length is whole,
    // into a later one, then whole frames. The records after the zeros were never synced, so the log ends before them.
    let mut lost = whole.clone();
    lost[ends[0] + 8..ends[3] + 4].fill(0);
    assert_eq!(super::records(&lost, v, DIM).unwrap(), (first(1), ends[0]));
    lost[ends[0]..ends[0] + 8].fill(0);
    assert_eq!(super::records(&lost, v, DIM).unwrap(), (first(1), ends[0]));
}

#[test]
fn a_whole_record_that_does_not_fit_fails_a_version_2_log_and_ends_a_version_1_log() {
    let (_, inner, _path) = written("log-fit.coldivf", &params(crate::Metric::L2), 600, DIM, 6);
    let layout = inner.layout();

    for v in [Version::One, version()] {
        let base = v.base();
        let mut records = records_from(base, v);
        // The second record's sequence number is one too far in version 2, and its id is too long in version 1, with
        // a checksum that matches either way.
        match v {
            Version::One => records[1].push(0),
            Version::Two { .. } => records[1][1] += 1,
        }
        let bytes = framed(&records);
        let mut a = Applier::new(layout, v, &inner.located().data);
        let applied = a.replay(&bytes, 0);
        let (first, delta) = (8 + records[0].len(), a.snapshot());
        match v {
            Version::One => assert!(applied.is_ok() && delta.log_len() == first && delta.len() == 1),
            Version::Two { .. } => {
                assert_eq!(applied.unwrap_err().kind(), ErrorKind::InvalidData);
                assert_eq!((delta.log_len(), delta.len()), (first, 1));
            }
        }
    }
}
