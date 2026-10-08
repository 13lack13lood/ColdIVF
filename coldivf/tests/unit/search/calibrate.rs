//! Unit tests for calibration.

use std::sync::Arc;

use super::*;
use crate::build::write;
use crate::quantization::SplitMix64;
use crate::quantization::testutil::{TempFile, clustered, dist2, group, params, temp_file};
use crate::search::inmem::InMemory;
use crate::{Index, Metric, SearchCtx};

/// The tests' vector dimension.
const DIM: usize = 128;

/// An index file of `n` vectors in `nlist` clusters, each vector `spread` from its center, with `nq` queries near its
/// vectors and their exact 10 nearest neighbors. With a spread of 1, a query's true neighbors sit in its nearest
/// cluster, and with 2 the clusters overlap.
fn sample(name: &str, n: usize, nlist: usize, nq: usize, spread: f64) -> (Index, TempFile, Vec<f32>, Vec<u32>) {
    let mut rng = SplitMix64::new(31);
    let dim = DIM;
    let (rows, mut clusters) = clustered(&mut rng, n, dim, nlist, spread);
    group(&mut clusters, 3);
    let memory = InMemory::build(&params(Metric::L2), &rows, &clusters).unwrap();
    let path = temp_file(name);
    write(&memory, &path).unwrap();
    let index = Index {
        inner: Arc::new(Inner::open(&path).unwrap()),
    };

    let mut queries = Vec::with_capacity(nq * dim);
    for _ in 0..nq {
        let r = (rng.next_u64() % n as u64) as usize;
        queries.extend(
            rows.data[r * dim..(r + 1) * dim]
                .iter()
                .map(|&x| x + (rng.normal() * 0.5) as f32),
        );
    }
    let truth = queries
        .chunks_exact(dim)
        .flat_map(|q| {
            let mut exact: Vec<(f64, u32)> = rows
                .data
                .chunks_exact(dim)
                .enumerate()
                .map(|(i, o)| (dist2(q, o), i as u32))
                .collect();
            exact.select_nth_unstable_by(TRUTH - 1, |a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            exact[..TRUTH].iter().map(|e| e.1).collect::<Vec<_>>()
        })
        .collect();
    (index, path, queries, truth)
}

/// The share of a query's true 10 nearest neighbors that a real search with `c` finds.
fn recall_of(index: &Index, ctx: &mut SearchCtx, q: &[f32], truth: &[u32], c: &Calibration) -> f32 {
    let sp = SearchParams {
        k: TRUTH,
        nprobe: c.nprobe,
        rerank: c.rerank,
        reach: c.reach,
    };
    let hits = index.search(q, &sp, ctx).unwrap();
    hits.iter()
        .filter(|h| truth.iter().any(|&t| u64::from(t) == h.id))
        .count() as f32
        / TRUTH as f32
}

fn ctx() -> SearchCtx {
    SearchCtx::new(Limits {
        max_k: TRUTH,
        max_nprobe: 64,
        max_rerank: 256,
    })
}

/// The runs of `nq` queries on settings that cost `costs`, where `recall(q, s)` is setting s's recall on query q.
fn runs(costs: &[f64], nq: usize, recall: impl Fn(usize, usize) -> f32) -> Runs {
    let n = costs.len();
    Runs {
        settings: n,
        recall: (0..nq * n).map(|i| recall(i / n, i % n)).collect(),
        bytes: (0..nq * n).map(|i| costs[i % n]).collect(),
    }
}

#[test]
fn each_setting_s_recall_matches_a_real_search() {
    // The clusters overlap, so the clusters a setting reads change its recall.
    let (index, _path, queries, truth) = sample("calibrate-match.coldivf", 3000, 64, 20, 2.0);
    let settings: Vec<Calibration> = [1, 3, 8, 20, 64]
        .iter()
        .flat_map(|&nprobe| {
            [10, 29, 256].into_iter().flat_map(move |rerank| {
                [None, Some(0.05), Some(0.1), Some(0.3)]
                    .into_iter()
                    .map(move |reach| Calibration { nprobe, rerank, reach })
            })
        })
        .collect();
    let tables = index.inner.tables().unwrap();
    let (view, data) = (index.inner.view(tables), index.inner.in_place());
    // Rows cost 5000 bytes here, which no page size is, so the cost must come from this argument.
    let runs = Evaluator::new(&view, &data, 64, 5000)
        .run_all(&queries, &truth, &settings)
        .unwrap();
    let (mut ctx, dim) = (ctx(), DIM);
    let size = block_bytes(padded_dim(dim));
    let every = view.lists.iter().map(|l| l.count.div_ceil(BLOCK) * size).sum::<usize>() as f64;
    let all = settings
        .iter()
        .position(|c| c.nprobe == 64 && c.rerank == 29 && c.reach.is_none())
        .unwrap();

    for (q, (query, t)) in queries.chunks_exact(dim).zip(truth.as_chunks::<TRUTH>().0).enumerate() {
        for (i, c) in settings.iter().enumerate() {
            let want = recall_of(&index, &mut ctx, query, t, c);
            assert_eq!(runs.recall[q * settings.len() + i], want, "query {q}, {c:?}");
        }
        // Reading every cluster costs all their blocks, plus a row for each candidate rescored.
        assert_eq!(runs.bytes[q * settings.len() + all], every + 29.0 * 5000.0);
    }
}

#[test]
fn a_calibrated_setting_meets_its_target_on_new_queries() {
    let (index, _path, queries, truth) = sample("calibrate-target.coldivf", 3000, 64, 300, 1.0);
    let (cal, test, dim) = (200, 100, DIM);
    let (cal_q, test_q) = queries.split_at(cal * DIM);
    let (cal_t, test_t) = truth.split_at(cal * TRUTH);
    let mut ctx = ctx();
    let targets = [
        Target::MeanRecall { recall_at_10: 0.9 },
        Target::QueryShare {
            recall_at_10: 0.8,
            share: 0.9,
        },
    ];
    let tables = index.inner.tables().unwrap();
    let (view, data) = (index.inner.view(tables), index.inner.in_place());

    for target in targets {
        let c = calibrate(&index.inner, cal_q, cal_t, target).unwrap();
        let recalls: Vec<f32> = test_q
            .chunks_exact(dim)
            .zip(test_t.as_chunks::<TRUTH>().0)
            .map(|(q, t)| recall_of(&index, &mut ctx, q, t, &c))
            .collect();
        let mean = recalls.iter().sum::<f32>() / test as f32;
        let share = recalls.iter().filter(|&&r| r >= 0.8).count() as f32 / test as f32;
        match target {
            Target::MeanRecall { .. } => assert!(mean >= 0.9, "{c:?} gave a mean recall of {mean}"),
            Target::QueryShare { .. } => assert!(share >= 0.9, "{c:?} gave {share} of queries at 0.8"),
        }

        // Each query's true neighbors sit in its nearest cluster, so a reach can't read fewer clusters, and the kill
        // rule keeps the fixed setting. The pick also costs far less than reading and rescoring everything.
        assert_eq!(c.reach, None, "{c:?}");
        let most = Calibration {
            nprobe: 64,
            rerank: 256,
            reach: None,
        };
        let runs = Evaluator::new(&view, &data, 64, index.inner.read_page())
            .run_all(cal_q, cal_t, &[c, most])
            .unwrap();
        assert!(runs.cost(0, 2) * 4.0 < runs.cost(1, 2), "{c:?}");
    }
}

#[test]
fn a_lower_target_never_costs_more() {
    let (index, _path, queries, truth) = sample("calibrate-cost.coldivf", 3000, 64, 200, 1.0);
    let low = calibrate(&index.inner, &queries, &truth, Target::MeanRecall { recall_at_10: 0.7 }).unwrap();
    let high = calibrate(
        &index.inner,
        &queries,
        &truth,
        Target::MeanRecall { recall_at_10: 0.95 },
    )
    .unwrap();
    let tables = index.inner.tables().unwrap();
    let (view, data) = (index.inner.view(tables), index.inner.in_place());
    let runs = Evaluator::new(&view, &data, 64, index.inner.read_page())
        .run_all(&queries, &truth, &[low, high])
        .unwrap();

    assert!(
        runs.cost(0, 2) <= GAIN * runs.cost(1, 2),
        "{low:?} costs more than {high:?}"
    );
}

#[test]
fn the_test_stops_at_the_first_failure_and_the_cheapest_pass_wins() {
    let mean = Target::MeanRecall { recall_at_10: 0.9 };
    let fixed = |nprobe| Calibration {
        nprobe,
        rerank: 10,
        reach: None,
    };
    let pick = |r: &Runs| {
        let settings: Vec<Setting> = (0..r.settings).map(|s| (fixed(s + 1), r, s)).collect();
        select(&settings, mean, 0.1)
    };

    // Three settings are perfect on the first half and cost 300, 200 and 100. Equal losses test the costliest first,
    // and the 200 one misses every query of the second half, so the test stops before the cheapest.
    let r = runs(&[300.0, 200.0, 100.0], 200, |q, s| {
        if s == 1 && q % 2 == 1 { 0.0 } else { 1.0 }
    });
    assert_eq!(pick(&r), Some(fixed(1)));

    // A setting perfect on the first half and useless on the second fails.
    let r = runs(&[100.0], 200, |q, _| if q % 2 == 0 { 1.0 } else { 0.0 });
    assert_eq!(pick(&r), None);

    // Of two settings that pass, the cheaper wins.
    let r = runs(&[200.0, 100.0], 200, |_, _| 1.0);
    assert_eq!(pick(&r), Some(fixed(2)));
}

#[test]
fn a_reach_wins_only_when_it_costs_1_1_times_less() {
    let mean = Target::MeanRecall { recall_at_10: 0.9 };
    let fixed = Calibration {
        nprobe: 10,
        rerank: 10,
        reach: None,
    };
    let reached = Calibration {
        reach: Some(0.2),
        ..fixed
    };

    for (cost, want) in [(95.0, fixed), (90.0, reached)] {
        let r = runs(&[100.0, cost], 200, |_, _| 1.0);
        let settings = [(fixed, &r, 0), (reached, &r, 1)];
        assert_eq!(select(&settings, mean, 0.1), Some(want), "a reach costing {cost}");
    }
}

#[test]
fn the_reach_grid_caps_the_clusters_read_near_the_best_fixed_setting() {
    let base = Calibration {
        nprobe: 100,
        rerank: 39,
        reach: None,
    };
    let grid = reach_grid(&base, 6250);
    let mut caps: Vec<usize> = grid.iter().map(|c| c.nprobe).collect();
    caps.dedup();
    let mut reranks: Vec<usize> = grid.iter().map(|c| c.rerank).collect();
    reranks.sort_unstable();
    reranks.dedup();

    assert_eq!(caps, [100, 125, 150]);
    assert_eq!(reranks, [32, 35, 39, 44, 48]);
    assert_eq!(grid.len(), 3 * 60 * 5);
    assert!(grid.iter().all(|c| c.reach.is_some_and(|r| (0.01..=0.6).contains(&r))));

    // The caps stop at the index's cluster count, and the depths at the ends of the grid.
    let grid = reach_grid(&Calibration { rerank: 10, ..base }, 120);
    let mut caps: Vec<usize> = grid.iter().map(|c| c.nprobe).collect();
    caps.dedup();
    assert_eq!(caps, [100, 120]);
    assert!(grid.iter().all(|c| [10, 12, 14].contains(&c.rerank)));
}

#[test]
fn a_target_out_of_reach_fails_with_invalid_input() {
    let (index, _path, queries, truth) = sample("calibrate-strict.coldivf", 1000, 16, 40, 1.0);
    let strict = Target::QueryShare {
        recall_at_10: 1.0,
        share: 0.999,
    };

    let err = calibrate(&index.inner, &queries, &truth, strict).unwrap_err();
    assert_eq!(
        err.to_string(),
        "no setting meets the target with 90% confidence on these queries"
    );
}

#[test]
fn bad_arguments_fail_with_invalid_input() {
    let (index, _path, queries, truth) = sample("calibrate-bad.coldivf", 1000, 16, 60, 1.0);
    let mean = Target::MeanRecall { recall_at_10: 0.8 };
    let message = |q: &[f32], t: &[u32], target| calibrate(&index.inner, q, t, target).unwrap_err().to_string();
    let (count, length) = (
        "calibration needs at least 2 queries and 10 true neighbors for each",
        "the queries' length isn't a multiple of the index's dimension",
    );
    let (mean_range, share_range) = (
        "a mean target's recall must be above 0 and below 1",
        "a share target's recall must be above 0 and at most 1, and its share above 0 and below 1",
    );
    let share = |recall_at_10, share| Target::QueryShare { recall_at_10, share };

    // The same queries with good arguments pass, so each failure below comes from its argument.
    calibrate(&index.inner, &queries, &truth, mean).unwrap();
    assert_eq!(message(&queries, &truth[1..], mean), count);
    assert_eq!(message(&queries[..DIM], &truth[..TRUTH], mean), count);
    assert_eq!(message(&queries[1..], &truth, mean), length);
    for recall_at_10 in [0.0, 1.0, 1.5, f32::NAN] {
        assert_eq!(
            message(&queries, &truth, Target::MeanRecall { recall_at_10 }),
            mean_range
        );
    }
    for target in [share(0.9, 1.0), share(0.9, 0.0), share(0.0, 0.5), share(1.5, 0.5)] {
        assert_eq!(message(&queries, &truth, target), share_range, "{target:?}");
    }
}

#[test]
fn the_p_values_match_known_tails() {
    assert!((binomial_cdf(0, 10, 0.5) - 1.0 / 1024.0).abs() < 1e-12);
    assert!((binomial_cdf(2, 20, 0.1) - 0.676_926_1).abs() < 1e-6);
    assert!((binomial_cdf(10, 10, 0.3) - 1.0).abs() < 1e-12);

    // With no losses on 100 queries, Bentkus's bound is e (about 2.718) times 0.95 to the 100th, and Hoeffding's is
    // 0.95 to the 100th, which wins.
    let zeros = vec![0.0; 100];
    assert!((bounded_p(&zeros, 0.05) - 0.95f64.powi(100)).abs() < 1e-12);
    assert_eq!(bounded_p(&vec![0.06; 100], 0.05), 1.0);
    assert!(bounded_p(&vec![0.05; 100], 0.05) > 0.99);

    // 17 losses of 0.5 sum to 8.5, which rounds up to 9. Bentkus's bound, e times the chance of at most 9 of 100
    // trials at 0.2, is 0.0063433 and beats Hoeffding's 0.0066282.
    let mut losses = vec![0.5; 17];
    losses.resize(100, 0.0);
    assert!((bounded_p(&losses, 0.2) - 0.006_343_276_424_417_872).abs() < 1e-12);
    assert_eq!(binomial_p(&[1.0, 0.0, 1.0], 0.5), binomial_cdf(2, 3, 0.5));
}
