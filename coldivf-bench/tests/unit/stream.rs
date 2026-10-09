//! Unit tests for the streams of changes.

use std::collections::HashSet;
use std::path::PathBuf;

use crate::data::write_fbin;
use crate::testutil::counters;

use super::*;

/// A folder of its own in the system temporary folder, which tests may write to.
fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("coldivf-bench/{}-stream-{name}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// `blobs` tight clusters of `per` rows each, of `dim` values, far apart, in a random order. Returns the dataset and
/// each row's cluster.
fn blobs(name: &str, blobs: usize, per: usize, dim: usize) -> (Fbin, Vec<usize>) {
    let mut rng = Rng::new(99);
    let mut unit = || (rng.next_u64() >> 11) as f32 / (1u64 << 53) as f32 - 0.5;
    let centers: Vec<f32> = (0..blobs * dim).map(|_| 20.0 * unit()).collect();
    let mut order: Vec<usize> = (0..blobs * per).collect();
    Rng::new(5).shuffle(&mut order);
    let mut rows = Vec::with_capacity(blobs * per * dim);

    for &i in &order {
        let c = &centers[i / per * dim..(i / per + 1) * dim];
        rows.extend(c.iter().map(|&x| x + 0.3 * unit()));
    }

    let path = scratch(name).join("base.fbin");
    write_fbin(&path, &rows, dim).unwrap();
    (Fbin::open(&path).unwrap(), order.iter().map(|&i| i / per).collect())
}

/// Settings for a stream of `blobs(_, 8, 250, 128)`. Its topics split each cluster into ~4, so a topic lies inside one
/// cluster. Its inserts and hot queries, 270 of them, take all of that cluster's rows and a few more.
fn params(pattern: Pattern) -> Params {
    Params {
        pattern,
        base: 600,
        first: false,
        rounds: 4,
        insert: 60,
        delete: 0.05,
        upsert: 0.05,
        pseudo: 100,
        hot: 30,
        topics: 32,
        hot_topics: 1,
        sample: 1000,
        upsert_pool: 400,
        seed: 3,
        threads: 2,
    }
}

#[test]
fn a_seed_makes_the_same_stream_every_time() {
    // Its threads stay busy, which the tests that read the process's counters would see.
    let _lock = counters();
    let (data, _) = blobs("seed", 8, 250, 128);

    for pattern in [Pattern::Uniform, Pattern::Drift, Pattern::Novel] {
        let p = params(pattern);
        let made = make(&data, Metric::L2, &p).unwrap();
        assert_eq!(made, make(&data, Metric::L2, &p).unwrap(), "{pattern:?}");
        let other = make(&data, Metric::L2, &Params { seed: 4, ..p }).unwrap();
        assert_ne!(made.stream, other.stream, "{pattern:?}");
    }
}

#[test]
fn a_made_stream_draws_its_inserts_as_its_pattern_says() {
    // Its threads stay busy, which the tests that read the process's counters would see.
    let _lock = counters();
    let (data, blob) = blobs("pattern", 8, 250, 128);
    // The cluster with the most inserts, the share of the inserts in it and the base rows in it.
    let spread = |made: &Made| {
        let mut counts = [0usize; 8];
        let inserts: Vec<u32> = made.stream.rounds.iter().flat_map(|r| r.inserts.clone()).collect();
        inserts.iter().for_each(|&r| counts[blob[r as usize]] += 1);
        let top = (0..8).max_by_key(|&b| counts[b]).unwrap();
        let base = made.stream.base.iter().filter(|&&r| blob[r as usize] == top).count();
        (top, counts[top] as f64 / inserts.len() as f64, base)
    };

    for pattern in [Pattern::Uniform, Pattern::Drift, Pattern::Novel] {
        let p = params(pattern);
        let made = make(&data, Metric::L2, &p).unwrap();
        let s = &made.stream;
        assert_eq!(s.base.len(), p.base);
        assert!(s.base.is_sorted(), "{pattern:?}");
        assert_eq!(s.rounds.len(), p.rounds);
        assert!(s.rounds.iter().all(|r| r.inserts.len() == p.insert));
        assert_eq!(made.pseudo.len(), p.pseudo);
        assert_eq!(made.hot.len(), if pattern == Pattern::Uniform { 0 } else { p.hot });

        // Base rows, inserts, hot queries, pseudo-queries and new vectors are all different rows.
        let named: Vec<u32> = s
            .base
            .iter()
            .chain(s.rounds.iter().flat_map(|r| &r.inserts))
            .chain(s.rounds.iter().flat_map(|r| r.upserts.iter().map(|(_, row)| row)))
            .chain(&made.pseudo)
            .chain(&made.hot)
            .copied()
            .collect();
        assert_eq!(named.iter().collect::<HashSet<_>>().len(), named.len(), "{pattern:?}");

        let (top, share, base) = spread(&made);
        let hot = made.hot.iter().filter(|&&r| blob[r as usize] == top).count();
        match pattern {
            // Random inserts land in every cluster, and the base holds ~75 rows of each.
            Pattern::Uniform => assert!(share < 0.3 && base > 25, "{share} {base}"),
            // Topical inserts come mostly from one cluster, which the base holds too, as do the hot queries...
            Pattern::Drift => assert!(share > 0.5 && base > 25 && hot > 15, "{share} {base} {hot}"),
            // ...unless the topic is new, so the base holds none of it.
            Pattern::Novel => assert!(share > 0.8 && base == 0 && hot > 20, "{share} {base} {hot}"),
        }
    }
}

#[test]
fn a_topic_takes_the_rows_nearest_its_centers() {
    let dir = scratch("nearest");
    let dim = 128;
    // For L2, rows 3 apart along one axis, and centers at 0 and 1000, so each row's distance to the nearer center is a
    // whole number of its own.
    let along = |x: f32| {
        (0..dim)
            .map(move |j| if j == 0 { x } else { 1.0 })
            .collect::<Vec<f32>>()
    };
    let rows: Vec<f32> = (0..400).flat_map(|i| along(3.0 * i as f32)).collect();
    write_fbin(&dir.join("l2.fbin"), &rows, dim).unwrap();
    let data = Fbin::open(&dir.join("l2.fbin")).unwrap();
    let (near, far) = (along(0.0), along(1000.0));
    let candidates: Vec<u32> = (0..400).rev().collect();
    let mut want: Vec<(i64, u32)> = (0..400)
        .map(|i| {
            let x = 3 * i64::from(i);
            ((x * x).min((1000 - x) * (1000 - x)), i)
        })
        .collect();
    want.sort_unstable();
    let want: Vec<u32> = want[..50].iter().map(|&(_, i)| i).collect();
    assert_eq!(
        nearest_rows(&data, Metric::L2, &candidates, &[&near, &far], 50, 3),
        want
    );
    // Rows left out of the candidates never come back.
    let odd: Vec<u32> = (1..400).step_by(2).collect();
    assert_eq!(nearest_rows(&data, Metric::L2, &odd, &[&near], 5, 2), [1, 3, 5, 7, 9]);

    // For cosine, rows at a growing angle from the center and of lengths 1 to 3, which cosine ignores.
    let at = |t: f32, len: f32| {
        (0..dim)
            .map(move |j| match j {
                0 => len * t.cos(),
                1 => len * t.sin(),
                _ => 0.0,
            })
            .collect::<Vec<f32>>()
    };
    let rows: Vec<f32> = (0..60)
        .flat_map(|i| at(0.05 * i as f32, 1.0 + (i % 3) as f32))
        .collect();
    write_fbin(&dir.join("cos.fbin"), &rows, dim).unwrap();
    let data = Fbin::open(&dir.join("cos.fbin")).unwrap();
    let candidates: Vec<u32> = (0..60).rev().collect();
    let got = nearest_rows(&data, Metric::Cosine, &candidates, &[&at(0.0, 1.0)], 20, 2);
    assert_eq!(got, (0..20).collect::<Vec<u32>>());
}

#[test]
fn the_first_rows_form_the_base_when_asked() {
    let (data, _) = blobs("first", 8, 250, 128);
    let p = Params {
        first: true,
        ..params(Pattern::Uniform)
    };
    let made = make(&data, Metric::L2, &p).unwrap();
    assert_eq!(made.stream.base, (0..600).collect::<Vec<u32>>());
    assert!(
        made.stream
            .rounds
            .iter()
            .all(|r| r.inserts.iter().all(|&row| row >= 600))
    );
}

#[test]
fn each_round_deletes_and_upserts_its_shares_of_the_live_vectors() {
    // Its threads stay busy, which the tests that read the process's counters would see.
    let _lock = counters();
    let (data, _) = blobs("shares", 8, 250, 128);
    let p = Params {
        delete: 0.1,
        upsert: 0.2,
        upsert_pool: 1000,
        ..params(Pattern::Uniform)
    };
    let made = make(&data, Metric::L2, &p).unwrap();
    // The stream ids that are live, and the row each one holds now.
    let mut live: std::collections::HashMap<u64, u32> = made.stream.base.iter().map(|&r| (u64::from(r), r)).collect();
    let (mut moved, mut random) = (0.0, 0.0);

    for round in &made.stream.rounds {
        let n = live.len() as f64;
        assert_eq!(round.deletes.len(), (0.1 * n).round() as usize);
        assert_eq!(round.upserts.len(), (0.2 * n).round() as usize);
        let ids: HashSet<u64> = round
            .deletes
            .iter()
            .chain(round.upserts.iter().map(|(id, _)| id))
            .copied()
            .collect();
        assert_eq!(
            ids.len(),
            round.deletes.len() + round.upserts.len(),
            "an id is changed twice"
        );

        for id in &round.deletes {
            assert!(live.remove(id).is_some(), "deleted a vector that isn't live");
        }

        for (k, &(id, row)) in round.upserts.iter().enumerate() {
            let old = live.insert(id, row).expect("upserted a vector that isn't live");
            let d = |a: u32, b: u32| crate::data::distance(Metric::L2, data.row(a as usize), data.row(b as usize));
            moved += f64::from(d(old, row));
            // Another upsert's row, a random one of the pool's.
            random += f64::from(d(old, round.upserts[(k + 1) % round.upserts.len()].1));
        }

        live.extend(round.inserts.iter().map(|&r| (u64::from(r), r)));
    }

    // An upsert's new vector is a near one, far nearer than a random row of the pool.
    assert!(moved * 5.0 < random, "{moved} {random}");
}

#[test]
fn a_stream_needs_enough_rows_and_shares_from_0_to_1() {
    let (data, _) = blobs("bounds", 8, 250, 128);

    for p in [
        Params {
            base: 1900,
            ..params(Pattern::Uniform)
        },
        // One round, with held-out rows enough for its upserts, so only the shares can fail it.
        Params {
            delete: 0.6,
            upsert: 0.6,
            rounds: 1,
            upsert_pool: 1000,
            ..params(Pattern::Uniform)
        },
        Params {
            delete: -0.1,
            ..params(Pattern::Uniform)
        },
        Params {
            base: 0,
            ..params(Pattern::Uniform)
        },
    ] {
        assert_eq!(make(&data, Metric::L2, &p).unwrap_err().kind(), ErrorKind::InvalidInput);
    }
}

#[test]
fn a_stream_file_round_trips_and_a_damaged_one_fails() {
    let dir = scratch("file");
    let stream = Stream {
        base: vec![4, 1, 7],
        rounds: vec![
            Round {
                inserts: vec![9, 3],
                deletes: vec![1],
                upserts: vec![(4, 11), (9, 2)],
            },
            Round::default(),
            Round {
                inserts: vec![u32::MAX - 1],
                deletes: vec![u64::MAX, 7],
                upserts: Vec::new(),
            },
        ],
    };
    let path = dir.join("s.stream");
    write(&path, &stream).unwrap();
    assert_eq!(read(&path).unwrap(), stream);
    let bytes = std::fs::read(&path).unwrap();
    assert_eq!(&bytes[..8], b"COLDSTRM");
    // The header, then the base round's counts and rows, then the first round's counts.
    assert_eq!(u32_at(&bytes, 12), 4);
    assert_eq!([u32_at(&bytes, 16), u32_at(&bytes, 20), u32_at(&bytes, 24)], [3, 0, 0]);
    assert_eq!([u32_at(&bytes, 28), u32_at(&bytes, 32), u32_at(&bytes, 36)], [4, 1, 7]);
    assert_eq!([u32_at(&bytes, 40), u32_at(&bytes, 44), u32_at(&bytes, 48)], [2, 1, 2]);

    let mut magic = bytes.clone();
    magic[0] = b'X';
    let mut version = bytes.clone();
    version[8] = 2;
    let mut no_base = bytes[..16].to_vec();
    no_base[12..16].copy_from_slice(&0u32.to_le_bytes());
    let mut long = bytes.clone();
    long.push(0);
    // A whole file of one round, the base, which holds a delete.
    let mut base_delete = bytes[..16].to_vec();
    base_delete[12..16].copy_from_slice(&1u32.to_le_bytes());

    for n in [1u32, 1, 0, 4] {
        base_delete.extend(n.to_le_bytes());
    }

    base_delete.extend(9u64.to_le_bytes());

    for (what, damaged) in [
        ("magic", &magic[..]),
        ("version", &version[..]),
        ("no base", &no_base[..]),
        ("cut short", &bytes[..bytes.len() - 1]),
        ("header only", &bytes[..10]),
        ("long", &long[..]),
        ("base delete", &base_delete[..]),
    ] {
        std::fs::write(&path, damaged).unwrap();
        assert_eq!(read(&path).unwrap_err().kind(), ErrorKind::InvalidData, "{what}");
    }
}

#[test]
fn a_replay_numbers_the_base_then_each_new_vector() {
    let mut replay = Replay::new(&[10, 20, 30]).unwrap();
    assert_eq!(replay.live(), vec![(0, 10), (1, 20), (2, 30)]);
    let changes = replay
        .apply(&Round {
            inserts: vec![40, 50],
            deletes: vec![20, 20],
            upserts: vec![(30, 60)],
        })
        .unwrap();
    // The second delete of 20 finds it gone already.
    assert_eq!(
        changes,
        vec![
            Change::Delete { id: 1 },
            Change::Put { id: 2, row: 60 },
            Change::Put { id: 3, row: 40 },
            Change::Put { id: 4, row: 50 },
        ]
    );
    assert_eq!(replay.len(), 4);
    assert_eq!((replay.row(1), replay.row(2), replay.row(9)), (None, Some(60), None));

    // An insert of a row whose vector came first puts it back under its id, and so does an upsert of a deleted one.
    let changes = replay
        .apply(&Round {
            inserts: vec![20, 70],
            deletes: vec![40],
            upserts: vec![(40, 80)],
        })
        .unwrap();
    assert_eq!(
        changes,
        vec![
            Change::Delete { id: 3 },
            Change::Put { id: 3, row: 80 },
            Change::Put { id: 1, row: 20 },
            Change::Put { id: 5, row: 70 },
        ]
    );
    assert_eq!(
        replay.live(),
        vec![(0, 10), (1, 20), (2, 60), (3, 80), (4, 50), (5, 70)]
    );
    assert_eq!(replay.len(), 6);

    // A change to an id that was never inserted changes nothing.
    let before = replay.live();

    for round in [
        Round {
            deletes: vec![10, 99],
            ..Round::default()
        },
        Round {
            inserts: vec![90],
            upserts: vec![(98, 1)],
            ..Round::default()
        },
    ] {
        assert_eq!(replay.apply(&round).unwrap_err().kind(), ErrorKind::InvalidData);
        assert_eq!(replay.live(), before);
    }

    assert_eq!(Replay::new(&[1, 2, 1]).unwrap_err().kind(), ErrorKind::InvalidData);
}

#[test]
fn the_generator_draws_every_value_alike_and_shuffles_into_any_order() {
    let mut rng = Rng::new(1);
    let mut counts = [0usize; 7];

    for _ in 0..70_000 {
        counts[rng.below(7) as usize] += 1;
    }

    assert!(counts.iter().all(|&c| (9_400..10_600).contains(&c)), "{counts:?}");
    let mut orders = HashSet::new();

    for _ in 0..600 {
        let mut items = [0, 1, 2, 3];
        rng.shuffle(&mut items);
        orders.insert(items);
    }

    assert_eq!(orders.len(), 24);
    assert_eq!(Rng::new(8).next_u64(), Rng::new(8).next_u64());
}
