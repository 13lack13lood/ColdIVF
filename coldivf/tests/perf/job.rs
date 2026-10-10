//! Timing checks for maintenance jobs, ignored by default. Run them in release mode on a quiet machine with
//! `cargo test --release -p coldivf --all-features --lib job::perf -- --ignored --nocapture`.

use std::time::{Duration, Instant};

use crate::io::Evictor;
use crate::quantization::SplitMix64;
use crate::quantization::testutil::{clustered, group, params, temp_file};
use crate::{BuildParams, Builder, Index, LeafOrder, Limits, MaintainParams, Metric, RunKind, SearchCtx, SearchParams};
use crate::{Throttle, Writer};

/// The 50th, 90th and 99th percentiles of `times`, in µs.
fn percentiles(mut times: Vec<f64>) -> [f64; 3] {
    times.sort_unstable_by(f64::total_cmp);
    [0.5, 0.9, 0.99].map(|p| times[((times.len() - 1) as f64 * p) as usize])
}

/// One search's time, and the time that its cluster wave held it up, its submit plus its wait, both in µs.
#[derive(Clone, Copy)]
struct Timed {
    total: f64,
    wave: f64,
}

/// The searches of a run of bursts: each burst's first one, which reads the file cold as a session's first query does,
/// and the later ones.
#[derive(Default)]
struct Waves {
    first: Vec<Timed>,
    later: Vec<Timed>,
}

impl Waves {
    /// Prints the percentiles of each kind of search and of its cluster wave, under `name`.
    fn print(&self, name: &str) {
        for (kind, times) in [("first", &self.first), ("later", &self.later)] {
            if times.is_empty() {
                continue;
            }
            let (total, wave) = (
                percentiles(times.iter().map(|t| t.total).collect()),
                percentiles(times.iter().map(|t| t.wave).collect()),
            );
            println!(
                "{name}, {} {kind} searches: p50 {:.0} µs, p90 {:.0} µs, p99 {:.0} µs, and their cluster waves \
                 p50 {:.0} µs, p90 {:.0} µs, p99 {:.0} µs",
                times.len(),
                total[0],
                total[1],
                total[2],
                wave[0],
                wave[1],
                wave[2]
            );
        }
    }

    /// The share of first searches whose cluster wave took over 2 ms.
    fn slow_firsts(&self) -> f64 {
        let slow = self.first.iter().filter(|t| t.wave > 2000.0).count();
        slow as f64 / self.first.len().max(1) as f64
    }
}

/// The cold handle of a data file that the searches read, which drops its pages from the cache, and the generation it
/// belongs to.
struct Cold {
    generation: u64,
    evictor: Evictor,
}

impl Cold {
    /// Drops the pages of the data file of `index`'s current generation from the cache, through a map of the file that
    /// it opens once for each generation.
    fn evict(cold: &mut Option<Cold>, index: &Index) {
        let at = index.inner().located().clone();

        if cold.as_ref().is_none_or(|c| c.generation != at.generation) {
            let evictor = Evictor::open(&at.data).unwrap();
            *cold = Some(Cold {
                generation: at.generation,
                evictor,
            });
        }

        if let Some(c) = cold {
            c.evictor.evict().unwrap();
        }
    }
}

/// Cold searches of `queries` in bursts of 10, each burst after a pause of `pause`, as a person typing makes them, for
/// `rounds` bursts, while `busy` says to go on. With `cold`, before each burst it drops the data file's pages from the
/// cache, so each burst's first search reads cold.
fn bursts(
    index: &Index,
    queries: &[f32],
    dim: usize,
    rounds: usize,
    pause: Duration,
    busy: &dyn Fn() -> bool,
    mut cold: Option<&mut Option<Cold>>,
) -> Waves {
    let sp = SearchParams {
        k: 10,
        nprobe: 64,
        rerank: 39,
        reach: None,
    };
    let mut ctx = SearchCtx::new(Limits {
        max_k: 10,
        max_nprobe: 64,
        max_rerank: 39,
    });
    let mut waves = Waves::default();
    let mut qs = queries.chunks_exact(dim).cycle();

    for _ in 0..rounds {
        if !busy() {
            break;
        }
        std::thread::sleep(pause);
        if let Some(cold) = cold.as_deref_mut() {
            Cold::evict(cold, index);
        }
        for (i, q) in qs.by_ref().take(10).enumerate() {
            let at = Instant::now();
            index.search(q, &sp, &mut ctx).unwrap();
            let total = at.elapsed().as_secs_f64() * 1e6;
            let wave = (ctx.trace.submit + ctx.trace.wave).as_secs_f64() * 1e6;
            let timed = Timed { total, wave };
            match i {
                0 => waves.first.push(timed),
                _ => waves.later.push(timed),
            }
        }
    }

    waves
}

/// What a job's writes do to cold searches that come in bursts between pauses, against the same searches with no
/// job, for each sync that ends a burst of the job's writes. The file has 200,000 vectors of 768 dimensions and 20,000
/// pending inserts. Each burst's first search reads cold, and with the `bench` feature the check first makes sure that
/// those reads come from the drive. Once a case has at least 1,000 first searches, it checks that at most 1% of their
/// cluster waves take over 2 ms, and that the later searches' 99th percentile stays within 0.1 ms of the one with no
/// job.
#[test]
#[ignore]
fn a_job_slows_cold_searches_little() {
    let (dim, n, nlist) = (768, 200_000, 1250);
    let mut rng = SplitMix64::new(91);
    let (rows, mut clusters) = clustered(&mut rng, n, dim, nlist, 0.3);
    group(&mut clusters, 64);
    let path = temp_file("perf-job.coldivf");
    let p = BuildParams {
        order: LeafOrder::Chained,
        threads: 4,
        ..params(Metric::L2)
    };
    Builder::new(p).build_with_clusters(&rows, &clusters, &path).unwrap();
    let mut writer = Writer::open(&path).unwrap();
    writer.maintain(&MaintainParams::default(), RunKind::Fold).unwrap();

    for i in 0..20_000u64 {
        let r = (rng.next_u64() % n as u64) as usize;
        let v: Vec<f32> = rows.data[r * dim..(r + 1) * dim]
            .iter()
            .map(|&x| x + 0.2 * rng.normal() as f32)
            .collect();
        writer.insert(10_000_000 + i, &v).unwrap();
    }

    let index = Index::open(&path).unwrap();
    index.set_warm(false);
    let queries: Vec<f32> = (0..500)
        .flat_map(|_| {
            let r = (rng.next_u64() % n as u64) as usize;
            rows.data[r * dim..(r + 1) * dim].to_vec()
        })
        .collect();
    let (pause, mut cold) = (Duration::from_millis(150), None);

    // The same searches once a pass over the queries brought their clusters into the cache, whose waves a cold
    // search's must far exceed.
    let all = queries.len() / dim / 10;
    bursts(&index, &queries, dim, all, Duration::ZERO, &|| true, None);
    let cached = bursts(&index, &queries, dim, all, Duration::ZERO, &|| true, None);
    let cached_wave = percentiles(cached.first.iter().chain(&cached.later).map(|t| t.wave).collect())[0];
    let alone = bursts(&index, &queries, dim, 200, pause, &|| true, Some(&mut cold));
    alone.print("no job");
    let first_wave = percentiles(alone.first.iter().map(|t| t.wave).collect())[0];
    let resident = cold.as_ref().map_or(1.0, |c| {
        c.evictor.evict().unwrap();
        c.evictor.resident()
    });
    println!("cached waves: p50 {cached_wave:.0} µs; after a drop {resident:.3} of the file stays in the cache");
    assert!(resident < 0.05, "a drop left {resident:.3} of the file in the cache");

    if cfg!(feature = "bench") {
        assert!(
            first_wave > 3.0 * cached_wave,
            "the first searches' waves, p50 {first_wave:.0} µs, look like cache hits, {cached_wave:.0} µs"
        );
    }

    let idle_later = percentiles(alone.later.iter().map(|t| t.total).collect())[2];

    for sync in [crate::SyncKind::Full, crate::SyncKind::Barrier] {
        let params = MaintainParams {
            throttle: crate::ThrottleParams {
                sync,
                ..crate::ThrottleParams::default()
            },
            ..MaintainParams::default()
        };
        let mut job = writer.begin(&params, RunKind::Fold).unwrap();
        let throttle = Throttle::new(params.throttle.clone());
        let done = std::sync::atomic::AtomicBool::new(false);
        let started = Instant::now();
        let (waves, finished) = std::thread::scope(|s| {
            let running = s.spawn(|| {
                let r = job.run(&throttle);
                done.store(true, std::sync::atomic::Ordering::Release);
                r
            });
            let busy = || !done.load(std::sync::atomic::Ordering::Acquire);
            let waves = bursts(&index, &queries, dim, 100_000, pause, &busy, Some(&mut cold));
            (waves, running.join().unwrap())
        });
        let took = started.elapsed();
        drop(job);
        writer.finish_compact(finished.unwrap()).unwrap();
        let name = format!("{sync:?} sync, job {:.1} s", took.as_secs_f64());
        waves.print(&name);
        let later = percentiles(waves.later.iter().map(|t| t.total).collect())[2];
        println!(
            "{name}: {:.2}% of first waves over 2 ms, later p99 {later:.0} µs against {idle_later:.0} µs with no job",
            100.0 * waves.slow_firsts()
        );

        if waves.first.len() >= 1000 {
            assert!(waves.slow_firsts() <= 0.01, "{name}");
            assert!(later <= idle_later + 100.0, "{name}");
        }
    }
}
