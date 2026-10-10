//! Unit tests for maintenance runs.

use std::collections::HashSet;

use super::*;
use crate::quantization::SplitMix64;
use crate::quantization::testutil::{Rows, TempFile, clustered, data_of, dist2, gaussian, params, temp_file};
use crate::update::log_path;
use crate::{BuildParams, Builder, Index, LeafId, LeafOrder, MajorParams, MinorParams, SearchParams, Throttle, Writer};

pub(crate) const DIM: usize = 128;

/// Settings for the small test files: 50 vectors per cluster and one thread.
pub(crate) fn settings() -> MaintainParams {
    MaintainParams {
        vectors_per_list: 50,
        threads: 1,
        ..MaintainParams::default()
    }
}

/// The same, with every region that changed at all due.
pub(crate) fn eager() -> MaintainParams {
    MaintainParams {
        minor: MinorParams {
            change: 1e-6,
            ..MinorParams::default()
        },
        ..settings()
    }
}

/// Build settings for the test files: 50 vectors per cluster in 4 groups, in the chained order.
pub(crate) fn build_params(metric: Metric, anchor: Anchor) -> BuildParams {
    BuildParams {
        vectors_per_list: 50,
        parents: 4,
        order: LeafOrder::Chained,
        anchor,
        ..params(metric)
    }
}

/// An index file of `n` vectors with ids 0 to n − 1 around `topics` centers, built with k-means as `p` says, and the
/// vectors.
pub(crate) fn sample(name: &str, p: &BuildParams, n: usize, topics: usize, seed: u64) -> (TempFile, Rows) {
    let (rows, _) = clustered(&mut SplitMix64::new(seed), n, DIM, topics, 0.6);
    let path = temp_file(name);
    Builder::new(p.clone()).build(&rows, &path).unwrap();
    let _ = std::fs::remove_file(log_path(&path));
    (path, rows)
}

/// Changes through `writer` to a file of the vectors `rows`: 300 inserts near its vectors, 150 deletes, 30 inserts
/// that replace a row, and 20 deletes and 10 replacements of the new inserts.
pub(crate) fn churn(writer: &mut Writer, rows: &Rows, seed: u64) {
    let mut rng = SplitMix64::new(seed);
    let n = (rows.data.len() / DIM) as u64;
    let near = |rng: &mut SplitMix64| {
        let r = (rng.next_u64() % n) as usize;
        let v: Vec<f32> = rows.data[r * DIM..(r + 1) * DIM]
            .iter()
            .map(|&x| x + 0.3 * rng.normal() as f32)
            .collect();
        v
    };

    for i in 0..300 {
        writer.insert(1_000_000 + i, &near(&mut rng)).unwrap();
    }

    for _ in 0..150 {
        writer.delete(rng.next_u64() % n).unwrap();
    }

    for _ in 0..30 {
        let id = rng.next_u64() % n;
        writer.insert(id, &near(&mut rng)).unwrap();
    }

    for i in 0..20 {
        writer.delete(1_000_000 + 7 * i).unwrap();
    }

    for i in 0..10 {
        writer.insert(1_000_000 + 3 * i, &near(&mut rng)).unwrap();
    }
}

/// The ids that the writer's file and its changes hold now, sorted.
pub(crate) fn visible_ids(writer: &Writer) -> Vec<u64> {
    let inner = &writer.index.inner();
    let delta = inner.changes().load();
    let (live, _) = Live::of(inner, &delta, delta.seen()).unwrap();
    let (layout, data) = (inner.layout(), inner.in_place());
    let rows = (0..layout.count as u32).filter(|&r| live.rows[r as usize]);
    let mut ids: Vec<u64> = rows.map(|r| sq8::id(data.rows.row(r), layout.dim)).collect();
    ids.extend(live.inserts.iter().map(|&i| sq8::id(delta.row(i as usize), layout.dim)));
    ids.sort_unstable();
    ids
}

/// The ids of the file at `path`, by internal id.
pub(crate) fn ids_of(path: &Path) -> Vec<u64> {
    let inner = Inner::open(path).unwrap();
    let (bytes, layout) = inner.parts();
    let ids = &bytes[layout.span(Part::Ids).range()][..8 * layout.count];
    ids.as_chunks::<8>().0.iter().map(|b| u64::from_le_bytes(*b)).collect()
}

/// The ids every cluster of `index` gives for `q`, reading every cluster and rescoring every candidate.
fn everything(index: &Index, q: &[f32]) -> Vec<u64> {
    let n = index.inner().layout().nlist;
    let sp = SearchParams {
        k: 10,
        nprobe: n,
        rerank: 5000,
        reach: None,
    };
    let mut ctx = SearchCtx::new(Limits {
        max_k: 10,
        max_nprobe: n,
        max_rerank: 5000,
    });
    index.search(q, &sp, &mut ctx).unwrap().iter().map(|h| h.id).collect()
}

#[test]
fn every_run_keeps_each_visible_vector_once_and_no_hidden_one() {
    for (kind, metric) in [
        (RunKind::Fold, Metric::L2),
        (RunKind::Minor, Metric::L2),
        (RunKind::Major, Metric::L2),
        (RunKind::Minor, Metric::Cosine),
    ] {
        let p = build_params(metric, Anchor::Cluster);
        let (path, rows) = sample(&format!("once-{kind:?}-{metric:?}.coldivf"), &p, 3000, 30, 1);
        let mut writer = Writer::open(&path).unwrap();
        churn(&mut writer, &rows, 2);
        let want = visible_ids(&writer);
        let report = writer.maintain(&eager(), kind).unwrap();
        let mut got = ids_of(&path);
        got.sort_unstable();
        assert!(got == want, "{kind:?} {metric:?}: the new file holds other ids");
        assert_eq!((report.kind, report.count), (kind, want.len()), "{kind:?}");
        assert_eq!(report.copied + report.reencoded, want.len(), "{kind:?}");
        assert_eq!(
            writer.index.inner().changes().load().len(),
            0,
            "{kind:?}: inserts were left in the log"
        );
        writer.index.verify().unwrap();
        if kind == RunKind::Minor {
            assert!(
                report.regions_due > 0 && report.reclustered > 0 && report.reencoded > 0,
                "{report:?}"
            );
        }
        // Each vector that stays finds itself, and a deleted one is gone.
        for r in [10usize, 500, 2999] {
            let found = everything(&writer.index, &rows.data[r * DIM..(r + 1) * DIM]);
            assert_eq!(
                found.contains(&(r as u64)),
                want.binary_search(&(r as u64)).is_ok(),
                "{kind:?} row {r}"
            );
        }
    }
}

#[test]
fn a_run_writes_the_same_file_for_any_number_of_threads() {
    let (path, rows) = sample(
        "threads.coldivf",
        &build_params(Metric::L2, Anchor::Cluster),
        3000,
        30,
        3,
    );
    churn(&mut Writer::open(&path).unwrap(), &rows, 4);

    for kind in [RunKind::Minor, RunKind::Major] {
        let files: Vec<Vec<u8>> = [1, 2, 4]
            .iter()
            .map(|&threads| {
                let copy = temp_file(&format!("threads-{kind:?}-{threads}.coldivf"));
                std::fs::copy(&path, &copy).unwrap();
                std::fs::copy(log_path(&path), log_path(&copy)).unwrap();
                let p = MaintainParams { threads, ..eager() };
                Writer::open(&copy).unwrap().maintain(&p, kind).unwrap();
                std::fs::read(data_of(&copy)).unwrap()
            })
            .collect();
        assert!(
            files[0] == files[1] && files[0] == files[2],
            "{kind:?}: the threads changed the file"
        );
    }
}

#[test]
fn a_minor_run_with_nothing_due_only_folds() {
    let (path, rows) = sample(
        "nothing-due.coldivf",
        &build_params(Metric::L2, Anchor::Cluster),
        3000,
        30,
        5,
    );
    let mut writer = Writer::open(&path).unwrap();
    (0..3).for_each(|i| {
        writer
            .insert(9000 + i, &rows.data[i as usize * DIM..(i as usize + 1) * DIM])
            .unwrap()
    });
    writer.delete(17).unwrap();
    let copy = temp_file("nothing-due-copy.coldivf");
    std::fs::copy(&path, &copy).unwrap();
    std::fs::copy(log_path(&path), log_path(&copy)).unwrap();

    let report = writer.maintain(&settings(), RunKind::Minor).unwrap();
    assert_eq!(
        (report.kind, report.regions_due, report.reclustered),
        (RunKind::Fold, 0, 0)
    );
    assert_eq!((report.reencoded, report.copied, report.moved), (0, 3002, 0));
    Writer::open(&copy)
        .unwrap()
        .maintain(&settings(), RunKind::Fold)
        .unwrap();
    assert!(
        std::fs::read(data_of(&path)).unwrap() == std::fs::read(data_of(&copy)).unwrap(),
        "it wrote another file than a fold"
    );

    // With no major run due either, `Auto` folds too.
    writer.insert(9100, &rows.data[..DIM]).unwrap();
    assert_eq!(writer.maintain(&settings(), RunKind::Auto).unwrap().kind, RunKind::Fold);
}

#[test]
fn an_insert_that_replaces_a_vector_counts_as_an_insert_only() {
    let (path, rows) = sample(
        "counts.coldivf",
        &build_params(Metric::L2, Anchor::Cluster),
        3000,
        30,
        13,
    );
    let mut writer = Writer::open(&path).unwrap();
    writer.maintain(&settings(), RunKind::Fold).unwrap();
    let row = |r: usize| {
        rows.data[r * DIM..(r + 1) * DIM]
            .iter()
            .map(|&x| x + 0.01)
            .collect::<Vec<f32>>()
    };
    (0..20).for_each(|id| writer.insert(id, &row(id as usize)).unwrap());
    (100..107).for_each(|id| writer.delete(id).unwrap());
    (0..5).for_each(|i| writer.insert(50_000 + i, &row(200 + i as usize)).unwrap());
    let deletes = |share: f32| MaintainParams {
        major: MajorParams {
            inserts: None,
            deletes: Some(share),
            ..MajorParams::default()
        },
        ..settings()
    };
    assert!(
        writer.status(&deletes(0.002)).unwrap().major_due,
        "7 deletes are over 0.2% of 3000"
    );
    assert!(
        !writer.status(&deletes(0.003)).unwrap().major_due,
        "7 deletes are under 0.3% of 3000"
    );
    writer.maintain(&settings(), RunKind::Fold).unwrap();
    let t = writer.index.inner().trailer().unwrap().unwrap();
    assert_eq!((t.major_inserts, t.major_deletes, t.pass_changes), (25, 7, 52));
    assert_eq!(t.region_changes.iter().sum::<u64>(), 52);
}

#[test]
fn changes_after_the_snapshot_move_to_the_new_file_s_log() {
    for anchor in [Anchor::Cluster, Anchor::Mean] {
        let p = build_params(Metric::L2, anchor);
        let (path, rows) = sample(&format!("snapshot-{anchor:?}.coldivf"), &p, 3000, 30, 7);
        let row = |r: usize| &rows.data[r * DIM..(r + 1) * DIM];
        let mut writer = Writer::open(&path).unwrap();
        writer.insert(50_000, row(1)).unwrap();
        writer.delete(2).unwrap();
        let mut job = writer.begin(&eager(), RunKind::Major).unwrap();
        let done = job.run(&Throttle::none()).unwrap();
        assert_eq!(done.report.count, 3000);

        // These land after the snapshot, while the run's new file waits.
        let late: Vec<f32> = row(3).iter().map(|&x| x + 0.3).collect();
        writer.insert(60_000, &late).unwrap();
        writer.delete(4).unwrap();
        writer.delete(50_000).unwrap();
        writer.finish_compact(done).unwrap();
        let check = |index: &Index, when: &str| {
            assert_eq!(
                everything(index, &late)[0],
                60_000,
                "{anchor:?} {when}: the late insert is missing"
            );
            for (r, id) in [(4, 4), (1, 50_000), (2, 2)] {
                assert!(
                    !everything(index, row(r)).contains(&id),
                    "{anchor:?} {when}: {id} is still there"
                );
            }
        };
        check(&writer.index, "after the commit");
        assert_eq!(writer.index.inner().layout().count, 3000);
        drop(writer);
        check(&Index::open(&path).unwrap(), "reopened");
        // A writer opens the new file's log, and the late changes replay from it.
        drop(Writer::open(&path).unwrap());
        check(&Index::open(&path).unwrap(), "after a writer opened it");
    }
}

#[test]
fn status_says_what_a_run_would_do() {
    let (path, rows) = sample(
        "status.coldivf",
        &build_params(Metric::L2, Anchor::Cluster),
        3000,
        30,
        9,
    );
    let mut writer = Writer::open(&path).unwrap();
    let s = writer.status(&settings()).unwrap();
    assert_eq!((s.pending, s.deleted, s.regions_due, s.major_due), (0, 0, 0, false));

    (0..100).for_each(|i| {
        writer
            .insert(9000 + i, &rows.data[i as usize * DIM..(i as usize + 1) * DIM])
            .unwrap()
    });
    (0..30).for_each(|id| writer.delete(id).unwrap());
    let s = writer.status(&settings()).unwrap();
    assert_eq!((s.pending, s.deleted, s.regions_due, s.major_due), (100, 30, 0, false));
    assert!(writer.status(&eager()).unwrap().regions_due > 0);
    let growth = MaintainParams {
        major: MajorParams {
            inserts: Some(0.02),
            ..MajorParams::default()
        },
        ..settings()
    };
    assert!(
        writer.status(&growth).unwrap().major_due,
        "100 inserts are over 2% of 3000"
    );

    // A run with `Auto` does what the status said.
    assert_eq!(writer.maintain(&growth, RunKind::Auto).unwrap().kind, RunKind::Major);
    let s = writer.status(&settings()).unwrap();
    assert_eq!((s.pending, s.deleted, s.regions_due, s.major_due), (0, 0, 0, false));
}

/// Vectors around topics: `topics` centers of standard deviation 1 in each value, and vectors at a spread of 0.6.
struct Topics {
    centers: Vec<f32>,
    rng: SplitMix64,
}

impl Topics {
    fn new(topics: usize, seed: u64) -> Topics {
        let mut rng = SplitMix64::new(seed);
        Topics {
            centers: gaussian(&mut rng, topics, DIM, |_| 1.0),
            rng,
        }
    }

    /// `n` vectors from topics `from..to`.
    fn draw(&mut self, n: usize, from: usize, to: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(n * DIM);
        for _ in 0..n {
            let t = from + (self.rng.next_u64() % (to - from) as u64) as usize;
            let c = &self.centers[t * DIM..(t + 1) * DIM];
            out.extend(c.iter().map(|&x| x + (0.6 * self.rng.normal()) as f32));
        }
        out
    }
}

/// Recall@10 of `index` for `queries`, against the exact neighbors among `live`, as (id, vector) pairs. Each query
/// reads its nearest clusters until they hold `budget` vectors, so files with clusters of other sizes read as much,
/// and rescores 40 candidates.
fn recall(index: &Index, queries: &[f32], live: &[(u64, Vec<f32>)], budget: usize) -> f64 {
    let lists = index.inner().tables().unwrap().lists.clone();
    let mut ctx = SearchCtx::new(Limits {
        max_k: 10,
        max_nprobe: lists.len(),
        max_rerank: 40,
    });
    let (mut found, mut leaves) = (0, Vec::new());

    for q in queries.as_chunks::<DIM>().0 {
        let mut exact: Vec<(f64, u64)> = live.iter().map(|(id, v)| (dist2(q, v), *id)).collect();
        exact.sort_by(|a, b| a.0.total_cmp(&b.0));
        let truth: HashSet<u64> = exact[..10].iter().map(|&(_, id)| id).collect();
        index.route(q, lists.len(), None, &mut ctx, &mut leaves).unwrap();
        let mut scanned = 0;
        let read = leaves.iter().take_while(|&&l| {
            let more = scanned < budget;
            scanned += lists[l as usize].count;
            more
        });
        let read: Vec<LeafId> = read.copied().collect();
        let hits = index.fetch(&read, &mut ctx).unwrap().search(q, 10, 40).unwrap();
        found += hits.iter().filter(|h| truth.contains(&h.id)).count();
    }

    found as f64 / (queries.len() / DIM * 10) as f64
}

#[test]
fn recall_after_runs_on_growth_streams_stays_near_a_fresh_build() {
    let p = build_params(Metric::L2, Anchor::Cluster);

    // Inserts from the base's topics, crowded into 2 of them, or from 8 topics that the base lacks.
    for (stream, from, to) in [("uniform", 0, 20), ("crowded", 0, 2), ("novel", 20, 28)] {
        let mut topics = Topics::new(28, 11);
        let rows = Rows {
            dim: DIM,
            data: topics.draw(2000, 0, 20),
        };
        let path = temp_file(&format!("growth-{stream}.coldivf"));
        Builder::new(p.clone()).build(&rows, &path).unwrap();
        let _ = std::fs::remove_file(log_path(&path));
        let folded = temp_file(&format!("growth-{stream}-folded.coldivf"));
        std::fs::copy(&path, &folded).unwrap();
        let mut live: Vec<(u64, Vec<f32>)> = rows
            .data
            .as_chunks::<DIM>()
            .0
            .iter()
            .enumerate()
            .map(|(i, v)| (i as u64, v.to_vec()))
            .collect();
        let (mut writer, mut fold) = (Writer::open(&path).unwrap(), Writer::open(&folded).unwrap());
        let mut rng = SplitMix64::new(12);
        let mut next = 2000u64;
        let mut runs = Vec::new();

        // Three rounds of 450 inserts and 60 deletes. The second brings the inserts past 25% of the base's 2000, so
        // it is a major run, and the third is a minor one.
        for _ in 0..3 {
            for v in topics.draw(450, from, to).as_chunks::<DIM>().0 {
                writer.insert(next, v).unwrap();
                fold.insert(next, v).unwrap();
                live.push((next, v.to_vec()));
                next += 1;
            }
            for _ in 0..60 {
                let (id, _) = live.swap_remove((rng.next_u64() % live.len() as u64) as usize);
                writer.delete(id).unwrap();
                fold.delete(id).unwrap();
            }
            runs.push(writer.maintain(&settings(), RunKind::Auto).unwrap().kind);
            fold.compact().unwrap();
        }

        assert_eq!(runs, [RunKind::Minor, RunKind::Major, RunKind::Minor], "{stream}");
        // Queries from where the vectors are now, half from the stream's topics, each reading as many vectors.
        let mut queries = topics.draw(60, 0, 20);
        queries.extend(topics.draw(60, from, to));
        let now = Rows {
            dim: DIM,
            data: live.iter().flat_map(|(_, v)| v.iter().copied()).collect(),
        };
        let by_row: Vec<(u64, Vec<f32>)> = (0..live.len() as u64)
            .zip(live.iter().map(|(_, v)| v.clone()))
            .collect();
        let budget = live.len() / 15;
        let kept = recall(&writer.index, &queries, &live, budget);
        let never = recall(&fold.index, &queries, &live, budget);
        // Fresh builds of isotropic topics differ by a few hundredths between seeds, so they count on average.
        let fresh_recall = [7u64, 8, 9]
            .iter()
            .map(|&seed| {
                let fresh = temp_file(&format!("growth-{stream}-fresh-{seed}.coldivf"));
                Builder::new(BuildParams { seed, ..p.clone() })
                    .build(&now, &fresh)
                    .unwrap();
                recall(&Index::open(&fresh).unwrap(), &queries, &by_row, budget)
            })
            .sum::<f64>()
            / 3.0;
        // A crowded topic spans several regions, and a minor run clusters only the due ones again, so the topic is
        // clustered in pieces until a major run. That costs a few hundredths here.
        let slack = if stream == "crowded" { 0.06 } else { 0.02 };
        assert!(
            kept >= fresh_recall - slack,
            "{stream}: maintained recall {kept:.3} against fresh builds' {fresh_recall:.3}"
        );
        // Inserts from topics the base lacks join clusters of other topics until a run clusters them again.
        if stream == "novel" {
            assert!(
                kept >= never + 0.05,
                "{stream}: maintained recall {kept:.3} against folds only {never:.3}"
            );
        }
    }
}

/// Vectors whose 8-bit rows decode to them exactly, in pairs v and −v, so their mean is exactly 0. Each value is a
/// multiple of 1/128, and the last is ±127/128, the largest, so the row's scale is 1/128 and every code is exact.
struct Grid {
    centers: Vec<Vec<i32>>,
    rng: SplitMix64,
}

impl Grid {
    fn new(topics: usize, seed: u64) -> Grid {
        let mut rng = SplitMix64::new(seed);
        let centers = (0..topics)
            .map(|_| (0..DIM).map(|_| (rng.normal() * 40.0) as i32).collect())
            .collect();
        Grid { centers, rng }
    }

    /// A pair of vectors around topic `t`, v and −v.
    fn pair(&mut self, t: usize) -> [Vec<f32>; 2] {
        let mut v: Vec<f32> = self.centers[t]
            .iter()
            .map(|&c| (c + (self.rng.normal() * 12.0) as i32).clamp(-126, 126) as f32 / 128.0)
            .collect();
        v[DIM - 1] = 127.0 / 128.0;
        let minus = v.iter().map(|&x| -x).collect();
        [v, minus]
    }

    /// Topic `t`'s center and its mirror image's, as f32.
    fn center(&self, t: usize, sign: f32) -> Vec<f32> {
        let mut c: Vec<f32> = self.centers[t].iter().map(|&x| sign * x as f32 / 128.0).collect();
        c[DIM - 1] = sign * 127.0 / 128.0;
        c
    }
}

#[test]
fn a_maintained_file_reads_exactly_what_a_fresh_build_of_its_clustering_reads() {
    for kind in [RunKind::Minor, RunKind::Major] {
        let mut grid = Grid::new(12, 41);
        // The base: 1200 pairs around 12 topics and their mirror images, in 24 clusters, one per topic and sign.
        let (mut data, mut assignment) = (Vec::new(), Vec::new());
        for i in 0..1200 {
            let t = i % 12;
            for (sign, v) in grid.pair(t).into_iter().enumerate() {
                data.extend(v);
                assignment.push((2 * t + sign) as u32);
            }
        }
        let centroids: Vec<f32> = (0..24)
            .flat_map(|c| grid.center(c / 2, if c % 2 == 0 { 1.0 } else { -1.0 }))
            .collect();
        let clusters = Clusters {
            dim: DIM,
            parents: centroids[..4 * DIM].to_vec(),
            parent_of: (0..24).map(|c| c % 4).collect(),
            centroids: centroids.clone(),
            assignment,
        };
        let p = build_params(Metric::L2, Anchor::Cluster);
        let path = temp_file(&format!("fresh-reads-{kind:?}.coldivf"));
        let rows = Rows { dim: DIM, data };
        crate::build::write(
            &crate::search::inmem::InMemory::build(&p, &rows, &clusters).unwrap(),
            &path,
        )
        .unwrap();
        let _ = std::fs::remove_file(log_path(&path));
        assert!(Index::open(&path).unwrap().inner().mean().iter().all(|&m| m == 0.0));

        // Pairs inserted near 3 topics, and pairs of those topics deleted, so the vectors keep their mean at 0 and the
        // regions of the other topics stay as they are.
        let mut writer = Writer::open(&path).unwrap();
        for i in 0..300u64 {
            let [a, b] = grid.pair((i % 3) as usize);
            writer.insert(10_000 + 2 * i, &a).unwrap();
            writer.insert(10_001 + 2 * i, &b).unwrap();
        }
        for pair in (0..1200u64).filter(|p| p % 12 < 3 && p % 5 == 0) {
            writer.delete(2 * pair).unwrap();
            writer.delete(2 * pair + 1).unwrap();
        }
        writer.maintain(&eager(), kind).unwrap();
        let old = Index::open(&path).unwrap();
        let inner = old.inner();
        assert!(inner.mean().iter().all(|&m| m == 0.0));

        // The fresh build gets the maintained file's clusters in its order, each with the center its codes came from,
        // and its vectors in the same order with the same ids.
        let (tables, data) = (inner.tables().unwrap(), inner.in_place());
        let ccb = sq4::code_bytes(padded_dim(DIM));
        let (mut vectors, mut ids, mut assignment, mut centers) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let mut o = vec![0f32; DIM];
        let (mut kept, mut new) = (0, 0);
        for (j, list) in tables.lists.iter().enumerate() {
            let mut sum = vec![0f64; DIM];
            for id in list.first_id..list.first_id + list.count as u32 {
                let row = data.rows.row(id);
                sq8::decode(row, inner.mean(), &mut o);
                sum.iter_mut().zip(&o).for_each(|(s, &x)| *s += f64::from(x));
                vectors.extend_from_slice(&o);
                ids.push(sq8::id(row, DIM));
                assignment.push(j as u32);
            }
            // A cluster whose codes equal an old center's kept it, and any other has its vectors' mean.
            let codes = &tables.codes[j * ccb..(j + 1) * ccb];
            let (mut z, mut tmp, mut theirs) = (vec![0f32; DIM], vec![0f32; DIM], vec![0u8; ccb]);
            let old_center = (0..24).find(|&c| {
                rotate_offset(
                    inner.rotation(),
                    &centroids[c * DIM..(c + 1) * DIM],
                    inner.mean(),
                    &mut z,
                    &mut tmp,
                );
                sq4::encode(&z, &mut theirs) == tables.factors[j] && theirs == codes
            });
            match old_center {
                Some(c) => {
                    centers.extend_from_slice(&centroids[c * DIM..(c + 1) * DIM]);
                    kept += 1;
                }
                None => {
                    centers.extend(sum.iter().map(|&s| (s / list.count as f64) as f32));
                    new += 1;
                }
            }
        }
        assert!(
            new > 0 && (kind == RunKind::Major) == (kept == 0),
            "{kind:?}: {kept} kept, {new} new"
        );
        // The groups only order a build's clusters, which `LeafOrder::Id` leaves in the given order, so any centers do.
        let parents: Vec<f32> = centers[..inner.layout().nparents * DIM].to_vec();
        let fresh_clusters = Clusters {
            dim: DIM,
            centroids: centers,
            assignment,
            parents,
            parent_of: tables.parent_of.iter().map(|&g| u32::from(g)).collect(),
        };
        let by_id = BuildParams {
            order: LeafOrder::Id,
            ..p.clone()
        };
        let src = Rows {
            dim: DIM,
            data: vectors,
        };
        let fresh = crate::search::inmem::InMemory::build_ids(&by_id, &src, &fresh_clusters, Some(&ids)).unwrap();
        let fresh_path = temp_file(&format!("fresh-reads-{kind:?}-fresh.coldivf"));
        crate::build::write(&fresh, &fresh_path).unwrap();
        let fresh_index = Index::open(&fresh_path).unwrap();

        // The same bytes in every section that a search reads, the codes, the rows, the extents and the centers.
        let fresh_inner = fresh_index.inner();
        let (a, la) = inner.parts();
        let (b, lb) = fresh_inner.parts();
        for part in [Part::Codes, Part::Rows, Part::Ids, Part::Extents] {
            assert!(
                a[la.span(part).range()] == b[lb.span(part).range()],
                "{kind:?}: the {part:?} section differs"
            );
        }
        let skip = |l: &crate::format::Layout| l.span(Part::Centers).offset + l.nparents * sq8::center_row_bytes(DIM);
        assert!(a[skip(la)..la.span(Part::Centers).end()] == b[skip(lb)..lb.span(Part::Centers).end()]);

        // So 1,000 queries read the same clusters and rows from both, in the same ranges.
        let sp = SearchParams {
            k: 10,
            nprobe: 6,
            rerank: 40,
            reach: None,
        };
        let limits = Limits {
            max_k: 10,
            max_nprobe: 6,
            max_rerank: 40,
        };
        let (mut ca, mut cb) = (SearchCtx::new(limits), SearchCtx::new(limits));
        for _ in 0..500 {
            let t = (grid.rng.next_u64() % 12) as usize;
            for q in grid.pair(t) {
                let ha: Vec<u64> = old.search(&q, &sp, &mut ca).unwrap().iter().map(|h| h.id).collect();
                let hb: Vec<u64> = fresh_index
                    .search(&q, &sp, &mut cb)
                    .unwrap()
                    .iter()
                    .map(|h| h.id)
                    .collect();
                let (ta, tb) = (ca.trace, cb.trace);
                let traffic = |t: crate::trace::Trace| (t.cluster_bytes, t.cluster_ranges, t.row_bytes, t.row_ranges);
                assert_eq!(traffic(ta), traffic(tb), "{kind:?}: a query read differently");
                assert_eq!(ha, hb, "{kind:?}");
            }
        }
    }
}

#[test]
fn an_open_index_keeps_searching_its_file_while_a_run_writes_the_new_one() {
    let (path, rows) = sample(
        "searching.coldivf",
        &build_params(Metric::L2, Anchor::Cluster),
        3000,
        30,
        15,
    );
    let mut writer = Writer::open(&path).unwrap();
    churn(&mut writer, &rows, 16);
    let index = Index::open(&path).unwrap();
    let queries: Vec<&[f32]> = rows.data.chunks(DIM).step_by(300).collect();
    let want: Vec<Vec<u64>> = queries.iter().map(|q| everything(&index, q)).collect();
    let (done, searches) = (
        std::sync::atomic::AtomicBool::new(false),
        std::sync::atomic::AtomicUsize::new(0),
    );

    std::thread::scope(|scope| {
        scope.spawn(|| {
            while !done.load(std::sync::atomic::Ordering::Acquire) {
                for (q, want) in queries.iter().zip(&want) {
                    assert_eq!(
                        &everything(&index, q),
                        want,
                        "a search during the run found other vectors"
                    );
                }
                searches.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        });
        writer.maintain(&eager(), RunKind::Major).unwrap();
        done.store(true, std::sync::atomic::Ordering::Release);
    });

    assert!(searches.into_inner() > 0, "no search ran during the run");
    // The index opened before still reads its own file, and the new file holds the same vectors.
    let reopened = Index::open(&path).unwrap();
    for (q, want) in queries.iter().zip(&want) {
        assert_eq!(&everything(&index, q), want);
        assert_eq!(&everything(&reopened, q), want);
    }
}

#[cfg(unix)]
#[test]
fn an_open_that_a_run_overtakes_opens_the_new_file() {
    let (path, rows) = sample(
        "overtaken.coldivf",
        &build_params(Metric::L2, Anchor::Cluster),
        3000,
        30,
        27,
    );
    let mut writer = Writer::open(&path).unwrap();
    let near: Vec<f32> = rows.data[..DIM].iter().map(|&x| x + 0.3).collect();
    writer.insert(50_000, &near).unwrap();
    writer.delete(1).unwrap();

    // The open starts on the old file. Then a run renames its new file over it and starts the log again for the new
    // file, all before the open reads the log, and no index holds the old file's changes any more.
    let opening = crate::index::Opening::start(&path).unwrap();
    writer.maintain(&settings(), RunKind::Fold).unwrap();
    let index = Index::of(opening.finish(&path).unwrap());
    assert_eq!(index.inner().layout().epoch, writer.index.inner().layout().epoch);
    assert_eq!(everything(&index, &near)[0], 50_000);
    assert!(!everything(&index, &rows.data[DIM..2 * DIM]).contains(&1));
}

#[test]
fn a_run_with_every_vector_deleted_fails_and_the_writer_goes_on() {
    let (path, rows) = sample(
        "emptied.coldivf",
        &build_params(Metric::L2, Anchor::Cluster),
        600,
        10,
        29,
    );
    let mut writer = Writer::open(&path).unwrap();
    writer.insert(70_000, &rows.data[..DIM]).unwrap();
    (0..600).for_each(|id| writer.delete(id).unwrap());
    writer.delete(70_000).unwrap();
    let bytes = std::fs::read(&path).unwrap();

    for kind in [RunKind::Fold, RunKind::Minor, RunKind::Major, RunKind::Auto] {
        let err = writer.maintain(&settings(), kind).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidInput, "{kind:?}");
        assert_eq!(
            err.to_string(),
            "every vector was deleted, so there is nothing to write",
            "{kind:?}"
        );
        assert!(std::fs::read(&path).unwrap() == bytes, "{kind:?} changed the file");
    }

    writer.insert(70_001, &rows.data[DIM..2 * DIM]).unwrap();
    assert_eq!(writer.maintain(&settings(), RunKind::Fold).unwrap().count, 1);
}

#[test]
fn a_run_makes_a_damaged_trailer_again() {
    let (path, rows) = sample(
        "trailer.coldivf",
        &build_params(Metric::L2, Anchor::Cluster),
        3000,
        30,
        31,
    );
    let mut writer = Writer::open(&path).unwrap();
    churn(&mut writer, &rows, 32);
    writer.compact().unwrap();
    let count = writer.index.inner().layout().count;
    drop(writer);
    // The file's last byte is part of the trailer's checksum.
    let data = data_of(&path);
    let mut bytes = std::fs::read(&data).unwrap();
    *bytes.last_mut().unwrap() ^= 1;
    std::fs::write(&data, &bytes).unwrap();
    let index = Index::open(&path).unwrap();
    assert_eq!(index.inner().trailer().unwrap_err().kind(), ErrorKind::InvalidData);
    assert_eq!(index.verify().unwrap_err().kind(), ErrorKind::InvalidData);
    assert_eq!(everything(&index, &rows.data[..DIM]).len(), 10);

    // The trailer holds no section that a run copies, so a run makes its regions again as for a fresh build.
    let mut writer = Writer::open(&path).unwrap();
    writer.status(&eager()).unwrap();
    writer.insert(9_000_000, &rows.data[..DIM]).unwrap();
    writer.maintain(&eager(), RunKind::Minor).unwrap();
    writer.index.verify().unwrap();
    let t = writer.index.inner().trailer().unwrap().unwrap();
    assert_eq!((t.major_count, t.major_inserts), (count as u64, 1));
}

#[test]
fn a_run_that_reads_a_damaged_row_fails_with_invalid_data() {
    for kind in [RunKind::Minor, RunKind::Major] {
        let (path, rows) = sample(
            &format!("scale-{kind:?}.coldivf"),
            &build_params(Metric::L2, Anchor::Cluster),
            3000,
            30,
            33,
        );
        // Row 0's scale goes from about 0.02 to 1e20, so it decodes to values that are finite but far too large. Its
        // region changes, so a minor run clusters it again.
        let at = Inner::open(&path).unwrap().layout().span(Part::Rows).offset + DIM;
        let mut bytes = std::fs::read(&path).unwrap();
        bytes[at..at + 4].copy_from_slice(&1e20f32.to_le_bytes());
        std::fs::write(&path, &bytes).unwrap();
        let mut writer = Writer::open(&path).unwrap();
        churn(&mut writer, &rows, 34);
        writer.insert(80_000, &rows.data[..DIM]).unwrap();
        let err = writer.maintain(&eager(), kind).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidData, "{kind:?}: {err}");
    }
}

#[test]
fn a_writer_keeps_one_routing_scratch_whatever_runs_do_to_the_cluster_count() {
    let (path, rows) = sample(
        "scratch.coldivf",
        &build_params(Metric::L2, Anchor::Cluster),
        3000,
        30,
        35,
    );
    let mut writer = Writer::open(&path).unwrap();
    let mut counts = HashSet::new();

    // Each insert routes through the writer's context, against the file of the run before.
    for (i, vectors_per_list) in [40, 60, 30].into_iter().enumerate() {
        writer
            .insert(9_000 + i as u64, &rows.data[i * DIM..(i + 1) * DIM])
            .unwrap();
        let p = MaintainParams {
            vectors_per_list,
            ..settings()
        };
        writer.maintain(&p, RunKind::Major).unwrap();
        counts.insert(writer.index.inner().layout().nlist);
    }

    let v = &rows.data[5 * DIM..6 * DIM];
    writer.insert(9_100, v).unwrap();
    assert_eq!((counts.len(), writer.ctx.scratch.len()), (3, 1));
    // The insert went to the cluster that a new context routes it to.
    let mut fresh = SearchCtx::new(Limits {
        max_k: 1,
        max_nprobe: 1,
        max_rerank: 1,
    });
    let mut leaves = Vec::new();
    let inner = writer.index.inner();
    inner.route(v, 1, None, &mut fresh, &mut leaves).unwrap();
    let delta = inner.changes().load();
    assert_eq!(delta.leaf(delta.len() - 1), leaves[0]);
}
