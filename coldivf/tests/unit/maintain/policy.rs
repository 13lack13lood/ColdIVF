//! Unit tests for when maintenance runs: the triggers on uniform, topical and delete-heavy streams, the interval, the
//! write budget, the hard cap and the read-cost indicator.

use super::*;
use crate::maintain::tests::{DIM, build_params, sample, settings, visible_ids};
use crate::quantization::SplitMix64;
use crate::{Anchor, BuildParams, Index, Limits, MajorParams, Metric, SearchCtx, SearchParams, Throttle, Writer};

const DAY: u64 = 86_400_000;

/// An index of 100,000 vectors of 768 dimensions with nothing to do, as the policy sees it at time `now_ms`.
fn quiet(now_ms: u64) -> Now {
    Now {
        count: 100_000,
        rows: 100_000,
        dim: 768,
        nlist: 625,
        file_bytes: 100 << 20,
        idle: Duration::ZERO,
        now_ms,
        ..Now::default()
    }
}

/// A run that ended at `ended_ms` after `wall_ms`, having written `bytes`.
fn ran(ended_ms: u64, wall_ms: u64, bytes: u64) -> Ran {
    Ran {
        ended_ms,
        wall_ms,
        bytes_written: bytes,
        count: 100_000,
        ..Ran::default()
    }
}

#[test]
fn a_run_is_due_only_once_a_trigger_holds_and_never_for_due_regions_alone() {
    let p = MaintainParams::default();
    let now = 10 * DAY;
    // 5% of 100,000 is 5,000, which 32 MB holds. In 10M vectors in 62,500 clusters 32 MB holds far fewer, since each
    // cluster's inserts take whole blocks.
    let soft = soft_limit(&p.start, 100_000, 768, 625);
    assert_eq!(soft, 5000);
    let large = soft_limit(&p.start, 10_000_000, 768, 62_500);
    assert!((5_000..10_000).contains(&large), "{large}");
    let due = |at: Now| decide(&p, &at, &[]);
    assert_eq!(
        due(quiet(now)),
        Decision {
            start: None,
            due: false,
            deferred: false
        }
    );

    // A uniform stream: due at the pending share, and only then.
    let pending = |n| Now {
        pending: n,
        ..quiet(now)
    };
    assert!(!due(pending(soft - 1)).due);
    assert_eq!(due(pending(soft)).start, Some(RunKind::Auto));

    // In 1M vectors in 6,250 clusters, the memory budget makes a run due long before the pending share.
    let wide = |n| Now {
        count: 1_000_000,
        rows: 1_000_000,
        nlist: 6250,
        pending: n,
        ..quiet(now)
    };
    let held = soft_limit(&p.start, 1_000_000, 768, 6250);
    assert!(held < 20_000 && !due(wide(held - 1)).due && due(wide(held)).due);

    // Due regions alone start nothing, even many, unless the index sits idle and idle runs are on.
    let regions = Now {
        regions_due: 40,
        idle: Duration::from_secs(3600),
        ..quiet(now)
    };
    assert!(!due(regions).due);
    let idle = MaintainParams {
        start: StartParams {
            idle: Some(Duration::from_secs(600)),
            ..StartParams::default()
        },
        ..MaintainParams::default()
    };
    assert_eq!(decide(&idle, &regions, &[]).start, Some(RunKind::Auto));
    let busy = Now {
        idle: Duration::from_secs(60),
        ..regions
    };
    assert!(!decide(&idle, &busy, &[]).due);
    assert!(
        decide(
            &idle,
            &Now {
                pending: soft / 2,
                ..quiet(now)
            }
            .with_idle(3600),
            &[]
        )
        .due
    );

    // A delete-heavy stream: due at the dead share.
    let dead = |n| Now {
        deleted: n,
        ..quiet(now)
    };
    assert!(!due(dead(4999)).due && due(dead(5000)).due);

    // A topical stream: due once the searches since the last run scanned 5% more, over 1,000 of them at least.
    let excess = |e, q| Now {
        excess: e,
        queries: q,
        ..quiet(now)
    };
    assert!(!due(excess(0.049, 5000)).due && !due(excess(0.2, 999)).due);
    assert!(due(excess(0.05, 1000)).due);

    // A major run is due as the regions say, or as the indicator does when its trigger is set.
    assert_eq!(
        due(Now {
            major_due: true,
            ..quiet(now)
        })
        .start,
        Some(RunKind::Major)
    );
}

impl Now {
    fn with_idle(self, secs: u64) -> Now {
        Now {
            idle: Duration::from_secs(secs),
            ..self
        }
    }
}

#[test]
fn the_interval_and_the_write_budget_hold_a_due_run_back_and_deferred_says_so() {
    let p = MaintainParams::default();
    let now = 10 * DAY;
    let soft = soft_limit(&p.start, 100_000, 768, 625);
    let at = Now {
        pending: soft,
        ..quiet(now)
    };
    let mb = 1 << 20;

    // An hour must pass after a run, or 10 times its length.
    let after = |runs: &[Ran]| decide(&p, &at, runs);
    assert_eq!(after(&[ran(now - 59 * 60_000, 60_000, mb)]).start, None);
    assert!(after(&[ran(now - 59 * 60_000, 60_000, mb)]).due);
    assert_eq!(after(&[ran(now - 61 * 60_000, 60_000, mb)]).start, Some(RunKind::Auto));
    assert_eq!(after(&[ran(now - 2 * 3_600_000, 15 * 60_000, mb)]).start, None);
    assert_eq!(
        after(&[ran(now - 3 * 3_600_000, 15 * 60_000, mb)]).start,
        Some(RunKind::Auto)
    );

    // The runs of the last 24 hours and this one may write twice the file's 100 MB. More holds the run back, which
    // defers maintenance once the inserts reach twice what makes a run due.
    let old = |ago_ms: u64, bytes: u64| ran(now - ago_ms, 1000, bytes);
    assert_eq!(after(&[old(5 * 3_600_000, 100 * mb)]).start, Some(RunKind::Auto));
    let over = [old(5 * 3_600_000, 60 * mb), old(4 * 3_600_000, 60 * mb)];
    assert_eq!(
        after(&over),
        Decision {
            start: None,
            due: true,
            deferred: false
        }
    );
    let piled = Now {
        pending: 2 * soft,
        ..at
    };
    assert_eq!(
        decide(&p, &piled, &over),
        Decision {
            start: None,
            due: true,
            deferred: true
        }
    );
    // A day later they no longer count.
    let later = [old(DAY + 5 * 3_600_000, 60 * mb), old(DAY + 4 * 3_600_000, 60 * mb)];
    assert_eq!(after(&later).start, Some(RunKind::Auto));
    assert_eq!(last_day(&over, now).0, 120 * mb);
    assert_eq!(last_day(&later, now).0, 0);
}

#[test]
fn the_indicator_calls_for_a_major_run_after_two_minor_runs_over_its_threshold() {
    let p = MaintainParams {
        major: MajorParams {
            indicator: Some(1.03),
            ..MajorParams::default()
        },
        ..MaintainParams::default()
    };
    let now = 10 * DAY;
    let with = |indicator: f32, major: bool| Ran {
        indicator: Some(indicator),
        major,
        ..ran(now - 2 * DAY, 1000, 0)
    };
    let start = |runs: &[Ran]| decide(&p, &quiet(now), runs).start;
    assert_eq!(
        start(&[with(1.0, true), with(1.05, false), with(1.04, false)]),
        Some(RunKind::Major)
    );
    assert_eq!(start(&[with(1.05, false), with(1.02, false)]), None);
    assert_eq!(start(&[with(1.0, true), with(1.05, false)]), None);
    assert_eq!(start(&[with(1.05, false)]), None);
    // Without the trigger the indicator starts nothing.
    let off = decide(
        &MaintainParams::default(),
        &quiet(now),
        &[with(1.5, false), with(1.5, false)],
    );
    assert_eq!(off.start, None);
}

#[test]
fn the_runs_file_keeps_the_last_runs_and_skips_a_line_cut_short() {
    let path = crate::quantization::testutil::temp_file("runs.coldivf");
    fs::create_dir_all(root::folder(&path)).unwrap();
    let mut first = ran(5, 6, 7);
    (first.cost, first.indicator, first.major) = (Some(1234.5), Some(1.25), true);
    record(&path, &first);
    assert_eq!(runs(&path), vec![first]);

    for i in 0..KEPT as u64 {
        record(&path, &ran(100 + i, 1, 1));
    }

    let all = runs(&path);
    assert_eq!(all.len(), KEPT);
    assert_eq!(all[0].ended_ms, 100);
    let file = root::folder(&path).join("runs");
    let text = fs::read_to_string(&file).unwrap();
    fs::write(&file, &text[..text.len() - 3]).unwrap();
    assert_eq!(runs(&path).len(), KEPT - 1);
}

/// Searches `index` for each of `queries`, `rounds` times over.
fn search_all(index: &Index, queries: &[f32], rounds: usize) {
    let sp = SearchParams {
        k: 10,
        nprobe: 4,
        rerank: 40,
        reach: None,
    };
    let mut ctx = SearchCtx::new(Limits {
        max_k: 10,
        max_nprobe: 4,
        max_rerank: 40,
    });

    for _ in 0..rounds {
        for q in queries.as_chunks::<DIM>().0 {
            index.search(q, &sp, &mut ctx).unwrap();
        }
    }
}

#[test]
fn topical_inserts_make_a_run_due_by_the_scanning_they_add_and_uniform_ones_by_their_count() {
    let p = build_params(Metric::L2, Anchor::Cluster);
    let params = MaintainParams {
        start: StartParams {
            excess_queries: 200,
            ..StartParams::default()
        },
        ..settings()
    };

    for topical in [true, false] {
        let (path, rows) = sample(&format!("policy-topic-{topical}.coldivf"), &p, 3000, 30, 71);
        let index = Index::open(&path).unwrap();
        let mut writer = Writer::open(&path).unwrap();
        let mut rng = SplitMix64::new(72);
        // 40 inserts, under the 150 that 5% of 3000 makes due, all near one vector, or near vectors all over.
        for i in 0..40u64 {
            let r = if topical { 7 } else { (rng.next_u64() % 3000) as usize };
            let v: Vec<f32> = rows.data[r * DIM..(r + 1) * DIM]
                .iter()
                .map(|&x| x + 0.05 * rng.normal() as f32)
                .collect();
            writer.insert(100_000 + i, &v).unwrap();
        }
        assert!(
            !writer.status(&params).unwrap().due,
            "topical {topical}: due before any search"
        );

        // Searches near that vector scan the inserts with every cluster they read, so they scan far more than 5% more.
        search_all(&index, &rows.data[7 * DIM..8 * DIM], 250);
        let status = writer.status(&params).unwrap();
        assert_eq!(status.due, topical, "topical {topical}");
        assert_eq!(writer.begin_compact(&params).unwrap().is_some(), topical);
    }
}

#[test]
fn deletes_make_a_run_due_at_the_dead_share_and_a_run_ends_what_made_it_due() {
    let (path, rows) = sample(
        "policy-deletes.coldivf",
        &build_params(Metric::L2, Anchor::Cluster),
        3000,
        30,
        73,
    );
    let mut writer = Writer::open(&path).unwrap();
    let params = settings();

    for id in 0..149 {
        writer.delete(id).unwrap();
    }

    assert!(!writer.status(&params).unwrap().due);
    writer.delete(149).unwrap();
    assert!(writer.status(&params).unwrap().due);
    let mut job = writer.begin_compact(&params).unwrap().unwrap();
    let done = job.run(&Throttle::none()).unwrap();
    drop(job);
    writer.finish_compact(done).unwrap();
    let status = writer.status(&params).unwrap();
    assert!(!status.due && status.deleted == 0, "{status:?}");
    assert!(status.bytes_written_24h > 0 && status.cpu_24h > Duration::ZERO);
    // The next run waits for the interval, whatever is due.
    for id in 150..400 {
        writer.delete(id).unwrap();
    }
    assert!(writer.status(&params).unwrap().due);
    assert!(writer.begin_compact(&params).unwrap().is_none());
    let _ = rows;
}

#[test]
fn the_memory_budget_counts_each_cluster_s_pending_inserts_in_whole_blocks() {
    // A file of 200 clusters, then inserts near its vectors, about 1, 5 and 40 per cluster on average. Each count is a
    // multiple of `REBUILD`, at which every insert has joined the sorted view.
    let p = BuildParams {
        vectors_per_list: 50,
        ..build_params(Metric::L2, Anchor::Cluster)
    };
    let (path, rows) = sample("policy-blocks.coldivf", &p, 10_000, 40, 75);
    let mut writer = Writer::open(&path).unwrap();
    let nlist = writer.index.inner().layout().nlist;
    let block = crate::quantization::rabitq::block_bytes(padded_dim(DIM)) + 4 * BLOCK;
    let mut rng = SplitMix64::new(76);
    let mut pending = 0;

    for want in [REBUILD, 4 * REBUILD, 32 * REBUILD] {
        while pending < want {
            let r = (rng.next_u64() % 10_000) as usize;
            let v: Vec<f32> = rows.data[r * DIM..(r + 1) * DIM].iter().map(|&x| x + 0.05).collect();
            writer.insert(1_000_000 + pending as u64, &v).unwrap();
            pending += 1;
        }
        let (sorted, recent) = writer.index.inner().changes().load().view_bytes();
        let blocks = nlist as f64 * blocks_per_cluster(pending as f64 / nlist as f64);
        let unpacked = pending * (code_bytes(padded_dim(DIM)) + 16);
        let ratio = sorted as f64 / (blocks * block as f64 + unpacked as f64);
        assert!(
            (0.7..=1.1).contains(&ratio),
            "{pending} inserts: the view holds {ratio:.2} of the estimate"
        );
        assert_eq!(recent, 0, "{pending} inserts");
    }

    // At 1M vectors of 768 dimensions in 6,250 clusters, 32 MB holds about 12,000 pending inserts, which leave many
    // clusters with one or two.
    let start = StartParams::default();
    let soft = soft_limit(&start, 1_000_000, 768, 6250);
    assert!((11_000..15_000).contains(&soft), "{soft}");
    let budget = start.ram_budget as f64;
    assert!(pending_bytes(soft, 6250, 768) <= budget && pending_bytes(soft + 1, 6250, 768) > budget);
}

#[test]
fn the_hard_cap_fails_inserts_only_while_the_budget_defers_maintenance() {
    let (path, rows) = sample(
        "policy-cap.coldivf",
        &build_params(Metric::L2, Anchor::Cluster),
        3000,
        30,
        74,
    );
    let mut writer = Writer::open(&path).unwrap();
    let capped = |budget: f32| MaintainParams {
        start: StartParams {
            pending_share: 0.01,
            write_budget: budget,
            hard_cap: Some(3.0),
            ..StartParams::default()
        },
        ..settings()
    };
    let row = |i: usize| -> Vec<f32> { rows.data[i % 3000 * DIM..][..DIM].iter().map(|&x| x + 0.1).collect() };

    // 30 inserts make a run due, and no budget lets one start, so 60 defer maintenance, and then inserts fail at 90.
    // Before that the cap holds nothing back.
    for i in 0..40 {
        writer.insert(200_000 + i as u64, &row(i)).unwrap();
    }

    assert!(writer.begin_compact(&capped(0.0)).unwrap().is_none());
    assert!(!writer.status(&capped(0.0)).unwrap().deferred);
    for i in 40..100 {
        writer.insert(200_000 + i as u64, &row(i)).unwrap();
    }

    assert!(writer.begin_compact(&capped(0.0)).unwrap().is_none());
    assert!(writer.status(&capped(0.0)).unwrap().deferred);
    let failed = writer.insert(300_000, &row(5)).err().map(|e| e.kind());
    assert_eq!(failed, Some(std::io::ErrorKind::WouldBlock));

    // A budget that lets the run start ends the deferral, and inserts go through again.
    let mut job = writer.begin_compact(&capped(2.0)).unwrap().unwrap();
    writer.insert(300_000, &row(5)).unwrap();
    let done = job.run(&Throttle::none()).unwrap();
    drop(job);
    writer.finish_compact(done).unwrap();
    assert!(visible_ids(&writer).contains(&300_000));
}

#[test]
fn the_indicator_is_1_after_a_major_run_and_grows_when_inserts_crowd_one_cluster() {
    let (path, rows) = sample(
        "policy-indicator.coldivf",
        &build_params(Metric::L2, Anchor::Cluster),
        3000,
        30,
        75,
    );
    let mut writer = Writer::open(&path).unwrap();
    let params = MaintainParams {
        major: MajorParams {
            indicator: Some(1.03),
            ..MajorParams::default()
        },
        ..settings()
    };
    let major = writer.maintain(&params, RunKind::Major).unwrap();
    assert_eq!(major.indicator, Some(1.0));
    let base = runs(&path).last().unwrap().cost.unwrap();
    assert!(base > 0.0);

    // 1500 inserts near one vector all join its cluster in a fold, so a query near them reads far more for its
    // neighbors, and so do the samples among them.
    let mut rng = SplitMix64::new(76);
    for i in 0..1500u64 {
        let v: Vec<f32> = rows.data[11 * DIM..12 * DIM]
            .iter()
            .map(|&x| x + 0.2 * rng.normal() as f32)
            .collect();
        writer.insert(400_000 + i, &v).unwrap();
    }
    let folded = writer.maintain(&params, RunKind::Fold).unwrap();
    assert!(folded.indicator.is_some_and(|i| i > 1.03), "{:?}", folded.indicator);
    let ran = runs(&path);
    assert_eq!(ran.len(), 2);
    assert!(ran[1].cost.unwrap() > base && !ran[1].major);
    // A major run sets the base again.
    assert_eq!(writer.maintain(&params, RunKind::Major).unwrap().indicator, Some(1.0));
}

#[test]
fn the_read_cost_excess_is_the_inserts_scanned_over_the_vectors_scanned_plus_the_dead_share() {
    let scanned = crate::index::Scanned::default();
    scanned.add([0, 0, 0]);
    assert_eq!(scanned.excess(0.0), (0.0, 0));
    // The inserts of the clusters past those that a search read count for nothing, and nor does a search of them alone.
    scanned.add([1000, 20, 300]);
    scanned.add([1000, 40, 300]);
    scanned.add([0, 0, 300]);
    let (excess, searches) = scanned.excess(0.02);
    assert!((excess - 0.05).abs() < 1e-12 && searches == 2, "{excess} {searches}");
}

#[test]
fn a_search_s_trace_counts_the_inserts_it_scanned_and_names_its_generation() {
    let (path, rows) = sample(
        "policy-trace.coldivf",
        &build_params(Metric::L2, Anchor::Cluster),
        3000,
        30,
        77,
    );
    let index = Index::open(&path).unwrap();
    let mut writer = Writer::open(&path).unwrap();
    writer.maintain(&settings(), RunKind::Fold).unwrap();

    for i in 0..30u64 {
        writer.insert(500_000 + i, &rows.data[7 * DIM..8 * DIM]).unwrap();
    }

    let sp = SearchParams {
        k: 10,
        nprobe: 2,
        rerank: 40,
        reach: None,
    };
    let mut ctx = SearchCtx::new(Limits {
        max_k: 10,
        max_nprobe: 2,
        max_rerank: 40,
    });

    for warm in [false, true] {
        index.set_warm(warm);
        index.search(&rows.data[7 * DIM..8 * DIM], &sp, &mut ctx).unwrap();
        assert_eq!(
            (ctx.trace.inserts_scanned, ctx.trace.generation),
            (30, 1),
            "warm {warm}"
        );
    }
}
