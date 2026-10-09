//! Unit tests for the warm harness.

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use coldivf_bench::data::{write_fbin, write_ibin};

use super::*;

/// A folder of its own in the system temporary folder, which tests may write to.
fn scratch(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("coldivf-bench/{}-warm-{name}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-9 * b.abs().max(1.0)
}

#[test]
fn each_pass_gives_its_latencies_and_the_line_takes_the_median_of_each() {
    let times = [[4, 1, 3, 2], [10, 20, 30, 40], [5, 6, 7, 100]];
    let mut calls = 0;
    let stats = latency(3, 4, |i| {
        let us = times[calls / 4][i];
        calls += 1;
        Duration::from_micros(us)
    });

    // Sorted, the first pass is 1, 2, 3 and 4 us, so its p50 is the 2nd, and its p90 and p99 the 4th.
    let expected = [
        [2.0, 4.0, 4.0, 2.5],
        [20.0, 40.0, 40.0, 25.0],
        [6.0, 100.0, 100.0, 29.5],
    ];
    assert_eq!(calls, 12);

    for (pass, (got, want)) in stats.iter().zip(&expected).enumerate() {
        assert!(got.iter().zip(want).all(|(&g, &w)| close(g, w)), "pass {pass}: {got:?}");
    }

    // Each of the four is the median over the passes, which here comes from a different pass for each.
    let medians = medians(&stats);
    assert!(
        medians.iter().zip([6.0, 40.0, 40.0, 25.0]).all(|(&g, w)| close(g, w)),
        "{medians:?}"
    );
}

#[test]
fn every_thread_s_searches_count_and_each_starts_at_its_own_place() {
    let (threads, count, time) = (3, 7, Duration::from_millis(200));
    let next = AtomicUsize::new(0);
    let seen = Mutex::new(Vec::new());
    let (done, seconds, mut ctxs) = throughput(
        threads,
        time,
        count,
        || next.fetch_add(1, Ordering::Relaxed),
        |thread, i| {
            seen.lock().unwrap().push((*thread, i));
            std::thread::sleep(Duration::from_millis(1));
        },
    );

    // Every search but each thread's first, which runs before the timer, counts, and the timer runs at least `time`.
    let seen = seen.into_inner().unwrap();
    assert_eq!(done as usize, seen.len() - threads);
    assert!(seconds >= time.as_secs_f64(), "{seconds} s");
    // Each thread's context comes back, so a line can say what read its searches.
    ctxs.sort();
    assert_eq!(ctxs, [0, 1, 2]);
    let mut by_thread: HashMap<usize, Vec<usize>> = HashMap::new();
    seen.iter().for_each(|&(t, i)| by_thread.entry(t).or_default().push(i));
    let mut firsts: Vec<usize> = by_thread.values().map(|v| v[0]).collect();
    firsts.sort();
    // The threads start at 0, 7 / 3 and 14 / 3 of the way through, rounded down, then each counts up from there.
    assert_eq!(firsts, [0, 2, 4]);

    for positions in by_thread.values() {
        assert!(positions.len() > 10, "{positions:?}");
        assert_eq!(positions[1], positions[0]);
        assert!(
            positions[1..].windows(2).all(|w| w[1] == (w[0] + 1) % count),
            "{positions:?}"
        );
    }
}

#[test]
fn recall_comes_at_each_level_the_k_and_the_truth_allow() {
    let (root, derived) = (scratch("recall-root"), scratch("recall-derived"));
    // One value a row. Every query is 0, so a row's distance is the square of its value. Rows 0 to 11 hold their ids,
    // and row 12 holds 9, so it ties the 10th nearest, row 9.
    let base: Vec<f32> = (0..12).map(|v| v as f32).chain([9.0]).collect();
    write_fbin(&root.join("t-euclidean.base.fbin"), &base, 1).unwrap();
    write_fbin(&root.join("t-euclidean.query.fbin"), &[0.0; 20], 1).unwrap();
    let truth: Vec<i32> = (0..20).flat_map(|_| 0..10).collect();
    write_ibin(&root.join("t-euclidean.gt.ibin"), &truth, 10).unwrap();
    let dists: Vec<f32> = truth.iter().map(|&t| (t * t) as f32).collect();
    write_fbin(&root.join("t-euclidean.gt.dist.fbin"), &dists, 10).unwrap();
    let dataset = Dataset::open_in(&root, &derived, "t-euclidean", None).unwrap();

    // Query 0 finds 5 of its true 10, and query 1 finds 7 but not its nearest. The rest find 9 and the copy of the
    // 10th, which ties it. Rows 10 and 11 are farther, and 100 to 102 aren't rows.
    let found: Vec<Vec<u64>> = (0..20)
        .map(|q| match q {
            0 => vec![0, 1, 2, 3, 4, 10, 11, 100, 101, 102],
            1 => vec![1, 0, 2, 3, 4, 5, 6, 10, 11, 100],
            _ => vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 12],
        })
        .collect();
    let queries: Vec<usize> = (0..20).collect();
    let r = Recalls::of(&dataset, &queries, &found, 10);
    assert!(
        close(r.levels[0], 19.0 / 20.0) && close(r.levels[1], 19.2 / 20.0),
        "{:?}",
        r.levels
    );
    assert!(r.levels[2].is_nan());
    // Sorted, the queries' recall@10 starts 0.5, 0.7, so those are its 5th and 10th percentiles of 20.
    assert_eq!((r.p05, r.p10), (0.5, 0.7));
    // By ids alone the copy of the 10th doesn't count.
    assert!(
        close(r.ids, 17.4 / 20.0) && close(r.share, 18.0 / 20.0),
        "{} {}",
        r.ids,
        r.share
    );

    // At k = 1 there is no recall@10, and the ground truth is too narrow for recall@100 at any k.
    let one = Recalls::of(&dataset, &queries, &found, 1);
    assert!(close(one.levels[0], 19.0 / 20.0) && one.levels[1].is_nan() && one.p05.is_nan() && one.p10.is_nan());
    let wide = Recalls::of(&dataset, &queries, &found, 100);
    assert!(close(wide.levels[1], 19.2 / 20.0) && wide.levels[2].is_nan());

    // Without the distances file the copy of the 10th still ties it, since the ties come from the base.
    std::fs::remove_file(root.join("t-euclidean.gt.dist.fbin")).unwrap();
    let bare = Dataset::open_in(&root, &derived, "t-euclidean", None).unwrap();
    let r = Recalls::of(&bare, &queries, &found, 10);
    assert!(
        bare.dists.is_none() && close(r.levels[1], 19.2 / 20.0),
        "{:?}",
        r.levels
    );
    std::fs::remove_dir_all(&root).unwrap();
    std::fs::remove_dir_all(&derived).unwrap();
}

#[cfg(target_os = "macos")]
#[test]
fn the_resident_share_counts_whole_pages_of_each_range() {
    let path = scratch("share").join("file.bin");
    std::fs::write(&path, vec![7u8; 4 << 20]).unwrap();
    let _ = std::fs::read(&path).unwrap();
    let (file, page) = (Mapped::open(&path).unwrap(), page_size());
    // Of the four pages from 2 MiB, past the first MiB that another process may read back after a new mapping, the
    // last two leave the cache.
    let start = 2 << 20;
    file.evict_range(start + 2 * page..start + 4 * page).unwrap();

    // A range a byte into its second page holds both pages, and a range past the end of the file counts the file's
    // last page only.
    let (two, four) = (start..start + page + 1, start..start + 4 * page);
    assert_eq!(share(&file, &[two]), 1.0);
    assert_eq!(share(&file, &[four]), 0.5);
    assert_eq!(
        share(&file, &[start + 3 * page..start + 4 * page, (4 << 20) - 1..usize::MAX]),
        0.5
    );
    assert!(share(&file, &[]).is_nan());
    std::fs::remove_dir_all(path.parent().unwrap()).unwrap();
}
