//! Unit tests for major runs.

use crate::maintain::regions::regions_for;
use crate::maintain::tests::{DIM, build_params, churn, eager, sample, settings};
use crate::quantization::testutil::{dist2, temp_file};
use crate::update::log_path;
use crate::{Anchor, Index, Limits, MaintainParams, MajorParams, Metric, RunKind, SearchCtx, SearchParams, Writer};

#[test]
fn a_major_run_measures_every_code_again_and_makes_new_regions() {
    let (path, rows) = sample(
        "major.coldivf",
        &build_params(Metric::L2, Anchor::Cluster),
        3000,
        30,
        31,
    );
    let mut writer = Writer::open(&path).unwrap();
    churn(&mut writer, &rows, 32);
    let report = writer.maintain(&eager(), RunKind::Major).unwrap();
    assert_eq!(
        (report.reencoded, report.copied, report.moved),
        (report.count, 0, report.count)
    );
    assert_eq!(report.reclustered, report.count);
    let inner = writer.index.inner();
    let t = inner.trailer().unwrap().unwrap();
    let lists = &inner.tables().unwrap().lists;
    assert_eq!(
        t.region_last.len(),
        regions_for(lists.len(), &eager()),
        "regions for {} clusters",
        lists.len()
    );
    assert_eq!(
        (t.major_count, t.major_inserts, t.major_deletes, t.pass_changes),
        (report.count as u64, 0, 0, 0)
    );
    assert!(t.region_changes.iter().all(|&c| c == 0));
    assert!(t.list_last.iter().zip(lists).all(|(&n, l)| n as usize == l.count));
    assert_eq!(t.region_last.iter().sum::<u64>(), report.count as u64);
    // About 50 vectors per cluster, as the settings say.
    assert_eq!(lists.len(), (report.count as f64 / 50.0).round() as usize);
}

#[test]
fn a_major_run_copies_every_code_measured_from_the_mean() {
    let (path, rows) = sample(
        "major-mean.coldivf",
        &build_params(Metric::L2, Anchor::Mean),
        3000,
        30,
        33,
    );
    let mut writer = Writer::open(&path).unwrap();
    churn(&mut writer, &rows, 34);
    let report = writer.maintain(&eager(), RunKind::Major).unwrap();
    assert_eq!((report.reencoded, report.copied), (0, report.count));
    assert_eq!(writer.index.inner().layout().anchor, Anchor::Mean);
}

#[test]
fn a_major_run_on_a_sample_clusters_about_as_well_as_on_every_vector() {
    let (path, rows) = sample(
        "major-sample.coldivf",
        &build_params(Metric::L2, Anchor::Cluster),
        4000,
        40,
        35,
    );
    churn(&mut Writer::open(&path).unwrap(), &rows, 36);
    let recall = |sample: usize| {
        let copy = temp_file(&format!("major-sample-{sample}.coldivf"));
        std::fs::copy(&path, &copy).unwrap();
        std::fs::copy(log_path(&path), log_path(&copy)).unwrap();
        let mut writer = Writer::open(&copy).unwrap();
        let p = MaintainParams {
            major: MajorParams {
                sample,
                ..MajorParams::default()
            },
            ..settings()
        };
        let report = writer.maintain(&p, RunKind::Major).unwrap();
        drop(writer);
        let index = Index::open(&copy).unwrap();
        let sp = SearchParams {
            k: 10,
            nprobe: 6,
            rerank: 40,
            reach: None,
        };
        let mut ctx = SearchCtx::new(Limits {
            max_k: 10,
            max_nprobe: 6,
            max_rerank: 40,
        });
        let base: Vec<&[f32]> = rows.data.as_chunks::<DIM>().0.iter().map(|v| &v[..]).collect();
        let mut found = 0;
        for q in base.iter().step_by(40) {
            let mut exact: Vec<(f64, usize)> = base.iter().enumerate().map(|(i, v)| (dist2(q, v), i)).collect();
            exact.sort_by(|a, b| a.0.total_cmp(&b.0));
            let hits = index.search(q, &sp, &mut ctx).unwrap();
            found += exact[..10]
                .iter()
                .filter(|&&(_, i)| hits.iter().any(|h| h.id == i as u64))
                .count();
        }
        (found as f64 / (base.len() / 40 * 10) as f64, report)
    };
    let ((all, a), (some, s)) = (recall(0), recall(8));
    assert_eq!(a.count, s.count);
    assert!(
        s.distances < a.distances,
        "a sample of 8 per cluster computed {} distances, all {}",
        s.distances,
        a.distances
    );
    assert!(
        (all - some).abs() <= 0.03,
        "recall {all:.3} with every vector and {some:.3} with a sample"
    );
}
