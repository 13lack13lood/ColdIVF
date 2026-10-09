//! Unit tests for the live ground truth.

use std::path::PathBuf;

use crate::data::write_fbin;
use crate::stream::{Params, Pattern, Replay, Rng, make};
use crate::testutil::counters;

use super::*;

/// A folder of its own in the system temporary folder, which tests may write to.
fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("coldivf-bench/{}-truth-{name}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// `n` rows of 128 small whole numbers, which make many exact ties, and the first quarter of them again.
fn tied(name: &str, n: usize) -> Fbin {
    let mut rng = Rng::new(17);
    let mut rows: Vec<f32> = (0..n * 128).map(|_| rng.below(5) as f32 - 2.0).collect();
    rows.extend_from_within(..n / 4 * 128);
    let path = scratch(name).join("base.fbin");
    write_fbin(&path, &rows, 128).unwrap();
    Fbin::open(&path).unwrap()
}

/// Every candidate's (distance, key) pair, from a full sort.
fn sorted(data: &Fbin, metric: Metric, query: &[f32], candidates: &[(u64, u32)]) -> Vec<Near> {
    let mut all: Vec<Near> = candidates
        .iter()
        .map(|&(key, row)| (distance(metric, query, data.row(row as usize)), key))
        .collect();
    all.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    all
}

#[test]
fn an_exact_search_matches_a_full_sort() {
    // Its threads stay busy, which the tests that read the process's counters would see.
    let _lock = counters();
    let data = tied("exact", 2000);
    // Keys out of row order, so a tie must go to the lower key, not the lower row.
    let candidates: Vec<(u64, u32)> = (0..data.len() as u32).map(|r| (u64::from(r * 7 % 2503), r)).collect();
    let queries: Vec<f32> = (0..9).flat_map(|q| data.row(q * 100).to_vec()).collect();

    for metric in [Metric::L2, Metric::Cosine] {
        for (k, threads) in [(1, 1), (10, 3), (64, 4), (3000, 2)] {
            let found = nearest(&data, metric, &queries, &candidates, k, threads);
            assert_eq!(found.len(), 9);

            for (q, f) in found.iter().enumerate() {
                let mut want = sorted(&data, metric, &queries[q * 128..(q + 1) * 128], &candidates);
                want.truncate(k);
                assert_eq!(f, &want, "{metric:?}, k = {k}, query {q}");
            }
        }
    }
}

#[test]
fn the_truth_stays_exact_through_inserts_deletes_and_upserts() {
    // Its threads stay busy, which the tests that read the process's counters would see.
    let _lock = counters();
    let data = tied("rounds", 2000);
    let p = Params {
        pattern: Pattern::Uniform,
        base: 500,
        first: false,
        rounds: 6,
        insert: 60,
        delete: 0.2,
        upsert: 0.1,
        pseudo: 0,
        hot: 0,
        topics: 1,
        hot_topics: 1,
        sample: 1,
        upsert_pool: 1000,
        seed: 2,
        threads: 2,
    };
    let stream = make(&data, Metric::L2, &p).unwrap().stream;
    let queries: Vec<f32> = (0..20).flat_map(|q| data.row(1990 - q).to_vec()).collect();

    for metric in [Metric::L2, Metric::Cosine] {
        let mut replay = Replay::new(&stream.base).unwrap();
        // 12 kept, against 20% deleted a round, leaves some queries with fewer than 10, which search again.
        let mut truth = Truth::new(&data, metric, queries.clone(), 12, &replay.live(), 3);

        for round in &stream.rounds {
            let changes = replay.apply(round).unwrap();
            truth.update(&data, &changes, &replay.live(), 3);

            for q in 0..truth.len() {
                let all = sorted(&data, metric, truth.query(q), &replay.live());
                let kept = truth.nearest(q);
                assert!(kept.len() >= K && kept.len() <= 12, "{metric:?}: {} kept", kept.len());
                assert_eq!(kept, &all[..kept.len()], "{metric:?}, query {q}");
            }
        }

        assert!(truth.redone > 0, "{metric:?}: no query was searched again");
    }
}

#[test]
fn a_query_with_every_live_vector_takes_any_insert() {
    let data = tied("whole", 400);
    let queries = data.row(0).to_vec();
    let live: Vec<(u64, u32)> = (0..20).map(|r| (r, r as u32)).collect();
    let mut truth = Truth::new(&data, Metric::L2, queries, 30, &live, 1);
    // Ten inserts that rank after every vector there: a list that holds every live vector takes them all, until it
    // holds 30.
    let far: Vec<u32> = sorted(
        &data,
        Metric::L2,
        data.row(0),
        &(20..400).map(|r| (r, r as u32)).collect::<Vec<_>>(),
    )
    .iter()
    .rev()
    .take(15)
    .map(|&(_, r)| r as u32)
    .collect();
    let changes: Vec<Change> = far
        .iter()
        .map(|&row| Change::Put {
            id: u64::from(row),
            row,
        })
        .collect();
    let after: Vec<(u64, u32)> = live
        .iter()
        .copied()
        .chain(far.iter().map(|&r| (u64::from(r), r)))
        .collect();
    truth.update(&data, &changes, &after, 1);
    assert_eq!(truth.nearest(0), &sorted(&data, Metric::L2, data.row(0), &after)[..30]);
    assert_eq!(truth.redone, 0);
}

#[test]
fn recall_counts_ties_with_the_tenth_and_each_id_once() {
    // Rows 0 to 11 are each nearer the query than the next, and row 12 is a copy of row 9, the 10th nearest.
    let mut rows = Vec::new();

    for i in 0..12 {
        rows.extend((0..128).map(|j| if j == 0 { i as f32 } else { 0.0 }));
    }

    rows.extend((0..128).map(|j| if j == 0 { 9.0 } else { 0.0 }));
    let path = scratch("recall").join("base.fbin");
    write_fbin(&path, &rows, 128).unwrap();
    let data = Fbin::open(&path).unwrap();
    let live: Vec<(u64, u32)> = (0..13).map(|r| (100 + r, r as u32)).collect();
    let truth = Truth::new(&data, Metric::L2, vec![0.0; 128], 64, &live, 1);
    assert_eq!(
        truth.nearest(0).iter().map(|n| n.1).take(10).collect::<Vec<_>>(),
        (100..110).collect::<Vec<_>>()
    );
    let vector = |id: u64| live.iter().find(|l| l.0 == id).map(|l| data.row(l.1 as usize));
    let ten: Vec<u64> = (100..110).collect();
    assert_eq!(truth.recall(0, &ten, vector), 1.0);
    // The copy of the 10th counts in its place, the 11th doesn't, a repeat counts once, and so does an unknown id.
    let copy: Vec<u64> = (100..109).chain([112]).collect();
    assert_eq!(truth.recall(0, &copy, vector), 1.0);
    let eleventh: Vec<u64> = (100..109).chain([110]).collect();
    assert_eq!(truth.recall(0, &eleventh, vector), 0.9);
    let repeat: Vec<u64> = (100..109).chain([100]).collect();
    assert_eq!(truth.recall(0, &repeat, vector), 0.9);
    assert_eq!(truth.recall(0, &[100, 5000], vector), 0.1);
    // Hits past the 10th don't count.
    let late: Vec<u64> = (101..111).chain([100]).collect();
    assert_eq!(truth.recall(0, &late, vector), 0.9);
}
