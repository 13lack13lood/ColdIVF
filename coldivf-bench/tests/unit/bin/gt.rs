//! Unit tests for the ground-truth tool.

use coldivf_bench::data::Ibin;
use coldivf_bench::stream::Round;

use super::*;

/// A folder of its own in the system temporary folder, which tests may write to.
fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("coldivf-bench/{}-gt-{name}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// The `k` nearest of the first `n` base rows to each query, from a full sort of every (distance, id) pair.
fn sorted(base: &Fbin, queries: &[&[f32]], metric: Metric, n: usize, k: usize) -> Vec<Nearest> {
    let nearest = |query: &[f32]| {
        let mut all: Nearest = (0..n)
            .map(|id| (distance(metric, query, base.row(id)), id as u32))
            .collect();
        all.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        all.truncate(k);
        all
    };
    queries.iter().map(|&q| nearest(q)).collect()
}

#[test]
fn one_pass_matches_a_full_sort_at_every_snapshot() {
    let dir = scratch("scan");
    // Small whole numbers make many exact ties, and the second half of the base repeats the first, so the order of
    // tied ids decides the truth. 20 values a row leave a tail past the 8-value chunks.
    let (rows, dim, nq, k) = (10_000, 20, 12, 100);
    let mut seed = 1u64;
    let mut next = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((seed >> 33) % 5) as f32 - 2.0
    };
    let half: Vec<f32> = (0..rows / 2 * dim).map(|_| next()).collect();
    let queries: Vec<f32> = (0..nq * dim).map(|_| next()).collect();
    write_fbin(&dir.join("base.fbin"), &[&half[..], &half[..]].concat(), dim).unwrap();
    let base = Fbin::open(&dir.join("base.fbin")).unwrap();
    let queries: Vec<&[f32]> = queries.chunks(dim).collect();
    // 3000 and 7001 aren't multiples of the tile, so the scan stops partway through a tile at each, and 7001 cuts a
    // pair of copies in two.
    let sizes = [3000, 7001, rows];

    for metric in [Metric::L2, Metric::Cosine] {
        let mut seen = Vec::new();
        scan(&base, &queries, metric, k, 3, &sizes, |size, best| {
            seen.push((size, best.to_vec()))
        });
        assert_eq!(seen.iter().map(|s| s.0).collect::<Vec<_>>(), sizes);

        for (size, best) in &seen {
            assert_eq!(
                best,
                &sorted(&base, &queries, metric, *size, k),
                "{metric:?}, n = {size}"
            );
        }
    }

    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn the_ids_and_their_distances_go_to_files_side_by_side() {
    let dir = scratch("write");
    let (path, dists) = (
        dir.join("t-euclidean-3.gt.ibin"),
        dir.join("t-euclidean-3.gt.dist.fbin"),
    );
    let best = vec![vec![(0.5, 2), (1.5, 0)], vec![(-1.0, 1), (2.0, 2)]];
    // An old distances file of another shape is replaced.
    write_fbin(&dists, &[9.0; 3], 1).unwrap();
    write(&path, &best, 2).unwrap();
    let (ids, values) = (Ibin::open(&path).unwrap(), Fbin::open(&dists).unwrap());
    assert_eq!((ids.len(), ids.row(0), ids.row(1)), (2, &[2, 0][..], &[1, 2][..]));
    assert_eq!(
        (values.len(), values.row(0), values.row(1)),
        (2, &[0.5, 1.5][..], &[-1.0, 2.0][..])
    );

    // A write whose ids fail, here since a folder holds their name, leaves no distances file, not even the old one.
    let blocked = dir.join("blocked.gt.ibin");
    std::fs::create_dir_all(&blocked).unwrap();
    write_fbin(&dist_path(&blocked), &[9.0; 4], 2).unwrap();
    assert!(write(&blocked, &best, 2).is_err());
    assert!(!dist_path(&blocked).exists());
    std::fs::remove_dir_all(&dir).unwrap();
}

/// The `k` nearest of the rows `live` marks to each query, from a full sort of every (distance, id) pair.
fn sorted_live(base: &Fbin, queries: &[&[f32]], metric: Metric, live: &[bool], k: usize) -> Vec<Nearest> {
    let nearest = |query: &[f32]| {
        let mut all: Nearest = (0..live.len())
            .filter(|&id| live[id])
            .map(|id| (distance(metric, query, base.row(id)), id as u32))
            .collect();
        all.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        all.truncate(k);
        all
    };
    queries.iter().map(|&q| nearest(q)).collect()
}

/// A stream over a base of 600 rows: three rounds that insert the next rows and delete ids live at their start.
fn small_stream() -> Stream {
    let round = |inserts: std::ops::Range<u32>, deletes: Vec<u64>| Round {
        inserts: inserts.collect(),
        deletes,
        upserts: Vec::new(),
    };
    Stream {
        base: (0..600).collect(),
        rounds: vec![
            round(600..650, vec![3, 17, 599]),
            round(650..700, vec![610, 4, 5, 6]),
            round(700..760, (100..160).collect()),
        ],
    }
}

#[test]
fn each_state_of_a_stream_gets_the_truth_of_its_own_vectors() {
    let dir = scratch("states");
    // Small whole numbers make many exact ties, and the inserts repeat base rows, so deletes and ties both decide.
    let (dim, nq, k) = (20, 9, 50);
    let mut seed = 7u64;
    let mut next = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((seed >> 33) % 5) as f32 - 2.0
    };
    let half: Vec<f32> = (0..380 * dim).map(|_| next()).collect();
    let queries: Vec<f32> = (0..nq * dim).map(|_| next()).collect();
    write_fbin(&dir.join("base.fbin"), &[&half[..], &half[..]].concat(), dim).unwrap();
    let base = Fbin::open(&dir.join("base.fbin")).unwrap();
    let queries: Vec<&[f32]> = queries.chunks(dim).collect();
    let stream = small_stream();
    let states = parse_states("s0=0,s1=1, s2=2,s3=3,s1i=1i,s3i=3i").unwrap();

    for metric in [Metric::L2, Metric::Cosine] {
        let best = truths(&base, &queries, metric, k, 3, &stream, &states).unwrap();

        for (state, lists) in states.iter().zip(&best) {
            let mut live = vec![false; 760];
            live[..600].fill(true);

            for round in &stream.rounds[..state.rounds] {
                if state.deletes {
                    round.deletes.iter().for_each(|&id| live[id as usize] = false);
                }

                round.inserts.iter().for_each(|&row| live[row as usize] = true);
            }

            assert_eq!(
                lists,
                &sorted_live(&base, &queries, metric, &live, k),
                "{metric:?}, {}",
                state.name
            );
        }
    }

    write_states(
        dir.join("w").to_str().unwrap(),
        &states[..2],
        &truths(&base, &queries, Metric::L2, k, 2, &stream, &states).unwrap(),
        k,
    )
    .unwrap();
    let ids = Ibin::open(&dir.join("w-s1.gt.ibin")).unwrap();
    assert_eq!((ids.len(), ids.width()), (nq, k));
    assert!(dir.join("w-s0.gt.dist.fbin").exists());
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn states_and_streams_that_dont_fit_fail() {
    for bad in ["", "s1", "s1=x", "s1=1,s1=2", "s-1=1", "=1"] {
        assert!(parse_states(bad).is_err(), "{bad}");
    }

    assert_eq!(
        parse_states("a=2i").unwrap(),
        vec![State {
            name: "a".to_string(),
            rounds: 2,
            deletes: false
        }]
    );
    let mut stream = small_stream();
    assert!(masks(&stream, &parse_states("s9=9").unwrap()).is_err());
    stream.rounds[1].deletes.push(3);
    assert!(masks(&stream, &parse_states("s2=2").unwrap()).is_err());
    assert!(masks(&stream, &parse_states("s2i=2i").unwrap()).is_ok());
    stream.rounds[1].deletes.pop();
    stream.rounds[2].inserts[0] = 9;
    assert!(masks(&stream, &parse_states("s1=1").unwrap()).is_err());
}

#[test]
fn a_stream_of_scattered_rows_takes_the_truth_of_those_rows() {
    let dir = scratch("drift");
    let (dim, nq, k) = (12, 6, 30);
    let mut seed = 3u64;
    let mut next = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((seed >> 33) % 7) as f32 - 3.0
    };
    let rows: Vec<f32> = (0..1000 * dim).map(|_| next()).collect();
    let queries: Vec<f32> = (0..nq * dim).map(|_| next()).collect();
    write_fbin(&dir.join("base.fbin"), &rows, dim).unwrap();
    let base = Fbin::open(&dir.join("base.fbin")).unwrap();
    let queries: Vec<&[f32]> = queries.chunks(dim).collect();
    // A drift stream inserts rows far past the base, out of order, and deletes some of them in a later round.
    let stream = Stream {
        base: (0..300).collect(),
        rounds: vec![
            Round {
                inserts: vec![950, 610, 777, 612],
                deletes: vec![5, 6],
                upserts: Vec::new(),
            },
            Round {
                inserts: vec![400, 999],
                deletes: vec![777, 7],
                upserts: Vec::new(),
            },
        ],
    };
    let states = parse_states("a=1,b=2,c=2i").unwrap();
    let (named, _) = masks(&stream, &states).unwrap();
    assert_eq!(named.len(), 306);
    assert_eq!(&named[300..], &[400, 610, 612, 777, 950, 999]);
    let best = truths(&base, &queries, Metric::L2, k, 2, &stream, &states).unwrap();
    let live = |rows: &[usize]| {
        let mut out = vec![false; 1000];
        out[..300].fill(true);
        rows.iter().for_each(|&r| out[r] = !out[r]);
        out
    };
    let want = [
        live(&[5, 6, 950, 610, 777, 612]),
        live(&[5, 6, 7, 950, 610, 612, 400, 999]),
        live(&[950, 610, 777, 612, 400, 999]),
    ];

    for ((state, lists), live) in states.iter().zip(&best).zip(&want) {
        assert_eq!(
            lists,
            &sorted_live(&base, &queries, Metric::L2, live, k),
            "{}",
            state.name
        );
    }

    std::fs::remove_dir_all(&dir).unwrap();
}
