//! Timing checks for pending inserts, ignored by default. Run them in release mode with
//! `cargo test --release -p coldivf --all-features --lib perf -- --ignored --nocapture`.

use std::hint::black_box;
use std::time::Instant;

use crate::io::Evictor;
use crate::quantization::SplitMix64;
use crate::quantization::testutil::{clustered, group, params, temp_file};
use crate::{BuildParams, Builder, Index, LeafOrder, Limits, Metric, SearchCtx, SearchParams, Writer};

/// One index that the check times, with a context of its own and a map of its data file through which a cold search's
/// pages leave the cache.
struct Side {
    index: Index,
    ctx: SearchCtx,
    evictor: Evictor,
}

impl Side {
    fn open(path: &std::path::Path, limits: Limits) -> Side {
        let index = Index::open(path).unwrap();
        let evictor = Evictor::open(&index.inner().located().data).unwrap();
        Side {
            index,
            ctx: SearchCtx::new(limits),
            evictor,
        }
    }

    /// The time of a search of `q`, in µs. A warm search runs on the calling thread alone, so it takes the thread's
    /// processor time, which other processes' work disturbs less than the time on the clock. A cold one first drops the
    /// data file's pages from the cache, and checks that they left, then waits for reads, so it takes the time on the
    /// clock.
    fn time(&mut self, q: &[f32], sp: &SearchParams, warm: bool) -> f64 {
        if !warm {
            self.evictor.evict().unwrap();
            let left = self.evictor.resident();
            assert!(left < 0.05, "a drop left {left:.3} of the file in the cache");
        }

        let (at, cpu) = (Instant::now(), crate::io::thread_cpu());
        black_box(self.index.search(q, sp, &mut self.ctx).unwrap());

        match warm {
            true => (crate::io::thread_cpu() - cpu).as_secs_f64() * 1e6,
            false => at.elapsed().as_secs_f64() * 1e6,
        }
    }
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_unstable_by(f64::total_cmp);
    let n = v.len();
    match n % 2 {
        1 => v[n / 2],
        _ => (v[n / 2 - 1] + v[n / 2]) / 2.0,
    }
}

/// What side a's searches take more than side b's, warm or cold, in µs: the median over the queries of each query's
/// median difference over `passes` passes, with a 95% interval from resampling the queries, and b's median. The two
/// sides search the same queries in blocks of 50, in turns whose order changes from block to block, so that a busy
/// spell slows both alike. A first pass that is not counted brings a warm search's pages into memory.
fn extra_us(a: &mut Side, b: &mut Side, queries: &[f32], dim: usize, sp: &SearchParams, warm: bool) -> [f64; 4] {
    let (nq, passes) = (queries.len() / dim, if warm { 3 } else { 1 });

    for side in [&mut *a, &mut *b] {
        side.index.set_warm(warm);
        if warm {
            queries.chunks_exact(dim).for_each(|q| _ = side.time(q, sp, true));
        }
    }

    let mut times = vec![(Vec::new(), Vec::new()); nq];

    for pass in 0..passes {
        for (block, start) in (0..nq).step_by(50).enumerate() {
            for turn in 0..2 {
                let a_turn = (turn == 0) == ((pass + block) % 2 == 0);
                for (i, q) in queries.chunks_exact(dim).enumerate().skip(start).take(50) {
                    match a_turn {
                        true => times[i].0.push(a.time(q, sp, warm)),
                        false => times[i].1.push(b.time(q, sp, warm)),
                    }
                }
            }
        }
    }

    let diffs: Vec<f64> = times
        .iter()
        .map(|(x, y)| median(x.iter().zip(y).map(|(p, q)| p - q).collect()))
        .collect();
    let mut rng = SplitMix64::new(5);
    let mut resampled: Vec<f64> = (0..400)
        .map(|_| median((0..nq).map(|_| diffs[(rng.next_u64() % nq as u64) as usize]).collect()))
        .collect();
    resampled.sort_unstable_by(f64::total_cmp);
    let b_median = median(times.iter().map(|(_, y)| median(y.clone())).collect());
    [median(diffs), resampled[10], resampled[389], b_median]
}

/// What pending inserts add to a search, against the same vectors folded into the file. The file has 6250 clusters, as
/// 1M vectors have, and a search reads 160 or 64 of them and scans the pending inserts of the 256 nearest, so it sees
/// as many pending inserts as one at 1M does. A fold writes the file first, so the file with pending inserts and the
/// folded one are written alike. Pending inserts may add at most 10 µs to a warm search and 1% to a cold one.
#[test]
#[ignore]
fn pending_inserts_cost_per_search() {
    let (dim, n, nlist) = (768, 200_000, 6250);
    let mut rng = SplitMix64::new(90);
    let (rows, mut clusters) = clustered(&mut rng, n, dim, nlist, 0.3);
    group(&mut clusters, 64);
    let path = temp_file("perf-delta.coldivf");
    let p = BuildParams {
        order: LeafOrder::Chained,
        threads: 4,
        ..params(Metric::L2)
    };
    Builder::new(p).build_with_clusters(&rows, &clusters, &path).unwrap();
    let mut writer = Writer::open(&path).unwrap();
    writer.compact().unwrap();
    let centers = &clusters.centroids;
    let queries: Vec<f32> = (0..2000)
        .flat_map(|_| {
            let at = (rng.next_u64() % n as u64) as usize;
            rows.data[at * dim..(at + 1) * dim].to_vec()
        })
        .collect();
    let settings = [160, 64].map(|nprobe| SearchParams {
        k: 10,
        nprobe,
        rerank: 39,
        reach: None,
    });
    let limits = Limits {
        max_k: 10,
        max_nprobe: 160,
        max_rerank: 39,
    };
    let (mut pending, mut failed) = (0, Vec::new());

    for want in [2_000, 10_000, 32_000, 100_000] {
        while pending < want {
            let c = (rng.next_u64() % nlist as u64) as usize;
            let v: Vec<f32> = centers[c * dim..(c + 1) * dim]
                .iter()
                .map(|&x| x + 0.3 * rng.normal() as f32)
                .collect();
            writer.insert(10_000_000 + pending as u64, &v).unwrap();
            pending += 1;
        }
        let copy = temp_file(&format!("perf-delta-{want}.coldivf"));
        crate::quantization::testutil::copy_index(&path, &copy);
        Writer::open(&copy).unwrap().compact().unwrap();
        let (mut with, mut folded) = (Side::open(&path, limits), Side::open(&copy, limits));

        for sp in &settings {
            for (warm, name) in [(true, "warm"), (false, "cold")] {
                let [extra, lo, hi, base] = extra_us(&mut with, &mut folded, &queries, dim, sp, warm);
                println!(
                    "nprobe {}, pending {want} {name}: {extra:+.1} µs [{lo:+.1}, {hi:+.1}] over folded, {:+.2}% of its \
                     {base:.0} µs",
                    sp.nprobe,
                    100.0 * extra / base
                );
                let over = match warm {
                    true => extra > 10.0,
                    false => extra > 0.01 * base,
                };
                if over {
                    failed.push(format!("nprobe {} pending {want} {name}: {extra:+.1} µs", sp.nprobe));
                }
            }
        }
    }

    assert!(failed.is_empty(), "pending inserts cost too much: {failed:?}");
}
