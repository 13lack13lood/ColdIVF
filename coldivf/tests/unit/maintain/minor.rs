//! Unit tests for minor runs.

use std::collections::HashMap;

use crate::maintain::decode_center;
use crate::maintain::tests::{DIM, build_params, churn, eager, sample};
use crate::quantization::rabitq::{BLOCK, block_bytes, code_bytes, unpack_lane};
use crate::quantization::testutil::{dist2, gaussian};
use crate::quantization::{SplitMix64, padded_dim, sq4, sq8};
use crate::{Anchor, Index, MaintainParams, Metric, MinorParams, RunKind, Writer};

/// A vector's cluster's center codes and factors as bytes, its code and factors as bytes, and its row.
type Stored = (Vec<u8>, Vec<u8>, Vec<u8>);

/// What the file of `index` stores for each vector, by id.
fn by_id(index: &Index) -> HashMap<u64, Stored> {
    let inner = &index.inner();
    let (tables, data) = (inner.tables().unwrap(), inner.in_place());
    let d_pad = padded_dim(DIM);
    let (cb, ccb, size) = (code_bytes(d_pad), sq4::code_bytes(d_pad), block_bytes(d_pad));
    let mut out = HashMap::new();

    for (j, list) in tables.lists.iter().enumerate() {
        let f = tables.factors[j];
        let mut center = tables.codes[j * ccb..(j + 1) * ccb].to_vec();
        [f.lo, f.step, f.norm2]
            .iter()
            .for_each(|x| center.extend(x.to_le_bytes()));
        for k in 0..list.count {
            let start = list.offset + k / BLOCK * size;
            let mut lane = vec![0u8; cb];
            let f = unpack_lane(&data.blocks[start..start + size], d_pad, k % BLOCK, &mut lane);
            [f.f_add, f.f_rescale, f.f_pop, f.f_err]
                .iter()
                .for_each(|x| lane.extend(x.to_le_bytes()));
            let row = data.rows.row(list.first_id + k as u32).to_vec();
            out.insert(sq8::id(&row, DIM), (center.clone(), lane, row));
        }
    }

    out
}

/// How many vectors of `index` sit nearer another cluster's decoded center than their own's.
fn outside(index: &Index) -> usize {
    let inner = &index.inner();
    let (tables, data) = (inner.tables().unwrap(), inner.in_place());
    let ccb = sq4::code_bytes(padded_dim(DIM));
    let centers: Vec<Vec<f32>> = (0..tables.lists.len())
        .map(|j| decode_center(inner, &tables.codes[j * ccb..(j + 1) * ccb], &tables.factors[j]))
        .collect();
    let (mut o, mut outside) = (vec![0f32; DIM], 0);

    for (j, list) in tables.lists.iter().enumerate() {
        for id in list.first_id..list.first_id + list.count as u32 {
            sq8::decode(data.rows.row(id), inner.mean(), &mut o);
            let own = dist2(&o, &centers[j]);
            outside += usize::from(centers.iter().any(|c| dist2(&o, c) < own));
        }
    }

    outside
}

/// How many vectors of `index` sit nearer another cluster's mean than their own's, from the decoded rows.
fn outside_means(index: &Index) -> usize {
    let inner = &index.inner();
    let (tables, data) = (inner.tables().unwrap(), inner.in_place());
    let mut o = vec![0f32; DIM];
    let decoded = |id: u32, o: &mut [f32]| sq8::decode(data.rows.row(id), inner.mean(), o);
    let means: Vec<Vec<f32>> = tables
        .lists
        .iter()
        .map(|l| {
            let mut sum = vec![0f64; DIM];
            for id in l.first_id..l.first_id + l.count as u32 {
                decoded(id, &mut o);
                sum.iter_mut().zip(&o).for_each(|(s, &x)| *s += f64::from(x));
            }
            sum.iter().map(|&s| (s / l.count as f64) as f32).collect()
        })
        .collect();
    let mut outside = 0;

    for (j, list) in tables.lists.iter().enumerate() {
        for id in list.first_id..list.first_id + list.count as u32 {
            decoded(id, &mut o);
            let own = dist2(&o, &means[j]);
            outside += usize::from(means.iter().any(|c| dist2(&o, c) < own));
        }
    }

    outside
}

#[test]
fn after_a_minor_run_few_vectors_sit_outside_their_nearest_cluster() {
    let (path, rows) = sample(
        "outside.coldivf",
        &build_params(Metric::L2, Anchor::Cluster),
        4000,
        40,
        21,
    );
    let mut writer = Writer::open(&path).unwrap();
    let base = outside(&writer.index);
    churn(&mut writer, &rows, 22);
    let report = writer.maintain(&eager(), RunKind::Minor).unwrap();
    assert!(report.regions_due > 1, "{report:?}");
    assert_eq!(
        report.reclustered, report.count,
        "every region changed, so every one is clustered again"
    );
    // Every cluster is new, so its center is its vectors' mean. At most 0.6% sit nearer another cluster's mean.
    let (n, exact) = (report.count as f64, outside_means(&writer.index));
    assert!(
        exact as f64 <= 0.006 * n,
        "{exact} of {n} vectors sit nearer another cluster's mean"
    );
    // The 4-bit centers move every center a little, so more sit nearer another center as it decodes, but no more than
    // in a fresh build.
    let after = outside(&writer.index);
    assert!(
        after as f64 / n <= base as f64 / 4000.0,
        "{after} of {n} against {base} of 4000 in a fresh build"
    );
}

#[test]
fn a_minor_run_keeps_the_codes_of_the_clusters_it_leaves_and_every_row() {
    let (path, rows) = sample(
        "leaves.coldivf",
        &build_params(Metric::L2, Anchor::Cluster),
        4000,
        40,
        23,
    );
    let mut writer = Writer::open(&path).unwrap();
    // Changes in a few places only, so most regions stay as they are.
    let mut rng = SplitMix64::new(24);
    for i in 0..200u64 {
        let v: Vec<f32> = rows.data[..DIM]
            .iter()
            .map(|&x| x + 0.3 * rng.normal() as f32)
            .collect();
        writer.insert(50_000 + i, &v).unwrap();
    }
    let before = by_id(&writer.index);
    let report = writer
        .maintain(&MaintainParams { threads: 2, ..eager() }, RunKind::Minor)
        .unwrap();
    assert!(report.regions_due >= 1 && report.reencoded > 0, "{report:?}");
    let after = by_id(&writer.index);
    let mut kept = 0;

    for (id, (center, lane, row)) in &after {
        let Some((old_center, old_lane, old_row)) = before.get(id) else {
            continue;
        };
        assert!(row == old_row, "vector {id}'s row changed");
        if center == old_center {
            assert!(
                lane == old_lane,
                "vector {id} kept its cluster's center but not its code"
            );
            kept += 1;
        }
    }

    assert!(
        kept > 2000 && kept + report.reencoded >= 4000,
        "{kept} kept, {report:?}"
    );
}

/// The number of regions the file at `path` has, from its trailer.
fn regions_of(index: &Index) -> usize {
    index.inner().trailer().unwrap().unwrap().region_last.len()
}

#[test]
fn a_region_splits_past_split_and_merges_under_merge() {
    let p = build_params(Metric::L2, Anchor::Cluster);
    let small = |split: f32, merge: f32| MaintainParams {
        minor: MinorParams {
            region_lists: 4,
            split,
            merge,
            ..eager().minor
        },
        ..eager()
    };

    // 1500 inserts around one vector crowd its region, which grows from ~4 clusters to ~34, past twice 4.
    let (path, rows) = sample("split.coldivf", &p, 3000, 30, 25);
    let mut writer = Writer::open(&path).unwrap();
    writer.maintain(&small(2.0, 0.4), RunKind::Fold).unwrap();
    let mut rng = SplitMix64::new(26);
    for i in 0..1500u64 {
        let v: Vec<f32> = rows.data[..DIM]
            .iter()
            .map(|&x| x + 0.6 * rng.normal() as f32)
            .collect();
        writer.insert(60_000 + i, &v).unwrap();
    }
    let copy = crate::quantization::testutil::temp_file("split-copy.coldivf");
    crate::quantization::testutil::copy_index(&path, &copy);
    writer.maintain(&small(2.0, 0.0), RunKind::Minor).unwrap();
    let mut other = Writer::open(&copy).unwrap();
    other.maintain(&small(0.0, 0.0), RunKind::Minor).unwrap();
    assert_eq!(
        regions_of(&writer.index),
        regions_of(&other.index) + 1,
        "the crowded region didn't split"
    );

    // Deleting all but 5 vectors of a region leaves it one cluster, under 0.4 times 4, so it joins another.
    let (path, _) = sample("merge.coldivf", &p, 3000, 30, 27);
    let mut writer = Writer::open(&path).unwrap();
    writer.maintain(&small(0.0, 0.0), RunKind::Fold).unwrap();
    let trailer = writer.index.inner().trailer().unwrap().unwrap();
    let tables = writer.index.inner().tables().unwrap().lists.clone();
    let lists_of = |r: u32| -> Vec<usize> { (0..tables.len()).filter(|&l| trailer.region_of[l] == r).collect() };
    let r = (0..trailer.region_last.len() as u32)
        .find(|&r| lists_of(r).len() >= 3)
        .unwrap();
    let ids: Vec<u64> = {
        let inner = writer.index.inner();
        let data = inner.in_place();
        let rows = lists_of(r)
            .into_iter()
            .flat_map(|l| tables[l].first_id..tables[l].first_id + tables[l].count as u32);
        rows.map(|id| sq8::id(data.rows.row(id), inner.layout().dim)).collect()
    };
    ids[5..].iter().for_each(|&id| writer.delete(id).unwrap());
    let copy = crate::quantization::testutil::temp_file("merge-copy.coldivf");
    crate::quantization::testutil::copy_index(&path, &copy);
    writer.maintain(&small(0.0, 0.4), RunKind::Minor).unwrap();
    let mut other = Writer::open(&copy).unwrap();
    other.maintain(&small(0.0, 0.0), RunKind::Minor).unwrap();
    assert!(
        regions_of(&writer.index) < regions_of(&other.index),
        "the emptied region didn't merge"
    );
    let t = writer.index.inner().trailer().unwrap().unwrap();
    let mut per_region = vec![0; t.region_last.len()];
    t.region_of.iter().for_each(|&r| per_region[r as usize] += 1);
    assert!(
        per_region.iter().all(|&n| n as f32 >= 0.4 * 4.0),
        "a region under the merge size remains"
    );
}

#[test]
fn refresh_moves_each_changed_cluster_to_its_mean() {
    let (path, rows) = sample(
        "refresh.coldivf",
        &build_params(Metric::L2, Anchor::Cluster),
        3000,
        30,
        51,
    );
    let mut writer = Writer::open(&path).unwrap();
    writer.maintain(&eager(), RunKind::Fold).unwrap();
    // A few inserts near one vector, far under any region's change share.
    let mut rng = SplitMix64::new(52);
    for i in 0..5u64 {
        let v: Vec<f32> = rows.data[..DIM]
            .iter()
            .map(|&x| x + 0.05 * rng.normal() as f32)
            .collect();
        writer.insert(70_000 + i, &v).unwrap();
    }
    let before = by_id(&writer.index);
    let p = MaintainParams {
        minor: MinorParams {
            refresh: true,
            ..MinorParams::default()
        },
        ..crate::maintain::tests::settings()
    };
    let report = writer.maintain(&p, RunKind::Minor).unwrap();
    assert_eq!((report.kind, report.regions_due), (RunKind::Minor, 0));
    let after = by_id(&writer.index);
    let changed: std::collections::HashSet<&Vec<u8>> = (70_000..70_005).map(|id| &after[&id].0).collect();
    let mut again = 0;

    for (id, (center, lane, row)) in &after {
        assert!(*id >= 70_000 || row == &before[id].2, "vector {id}'s row changed");
        if changed.contains(center) {
            again += 1;
        } else {
            assert!(
                (center, lane) == (&before[id].0, &before[id].1),
                "vector {id} lost its code"
            );
        }
    }

    assert!(
        again > 5 && report.reencoded == again,
        "{again} in changed clusters, {report:?}"
    );
}

#[test]
fn a_global_pass_moves_every_vector_to_its_nearest_cluster() {
    let (path, rows) = sample("pass.coldivf", &build_params(Metric::L2, Anchor::Cluster), 3000, 30, 53);
    let mut writer = Writer::open(&path).unwrap();
    crate::maintain::tests::churn(&mut writer, &rows, 54);
    let p = MaintainParams {
        minor: MinorParams {
            change: 10.0,
            global_pass: Some(0.01),
            ..MinorParams::default()
        },
        ..crate::maintain::tests::settings()
    };
    let report = writer.maintain(&p, RunKind::Minor).unwrap();
    assert_eq!((report.kind, report.regions_due), (RunKind::Minor, 0));
    assert_eq!(
        (report.reencoded, report.copied),
        (report.count, 0),
        "every cluster's center is new"
    );
    let t = writer.index.inner().trailer().unwrap().unwrap();
    assert_eq!(t.pass_changes, 0);
    let exact = outside_means(&writer.index);
    assert!(
        exact as f64 <= 0.01 * report.count as f64,
        "{exact} vectors sit nearer another cluster's mean"
    );

    // Under the pass's share, a run with nothing due only folds.
    let quiet = MaintainParams {
        minor: MinorParams {
            change: 10.0,
            global_pass: Some(0.5),
            ..MinorParams::default()
        },
        ..crate::maintain::tests::settings()
    };
    writer.insert(80_000, &rows.data[..DIM]).unwrap();
    assert_eq!(writer.maintain(&quiet, RunKind::Minor).unwrap().kind, RunKind::Fold);
}

#[test]
fn far_inserts_form_a_region_of_their_own() {
    let (path, _) = sample("pool.coldivf", &build_params(Metric::L2, Anchor::Cluster), 3000, 30, 55);
    let mut writer = Writer::open(&path).unwrap();
    writer.maintain(&eager(), RunKind::Fold).unwrap();
    let regions = writer.index.inner().trailer().unwrap().unwrap().region_last.len();
    // 120 inserts around a point far from every topic.
    let mut rng = SplitMix64::new(56);
    let far: Vec<f32> = gaussian(&mut rng, 1, DIM, |_| 1.0).iter().map(|&x| 4.0 * x).collect();
    for i in 0..120u64 {
        let v: Vec<f32> = far.iter().map(|&x| x + 0.6 * rng.normal() as f32).collect();
        writer.insert(90_000 + i, &v).unwrap();
    }
    let p = MaintainParams {
        minor: MinorParams {
            change: 10.0,
            hot_list: 0.0,
            merge: 0.0,
            pool: Some(0.99),
            ..MinorParams::default()
        },
        ..crate::maintain::tests::settings()
    };
    let report = writer.maintain(&p, RunKind::Minor).unwrap();
    assert_eq!((report.kind, report.regions_due), (RunKind::Minor, 0));
    let inner = &writer.index.inner();
    let (t, tables, data) = (
        inner.trailer().unwrap().unwrap(),
        inner.tables().unwrap(),
        inner.in_place(),
    );
    assert_eq!(t.region_last.len(), regions + 1, "the far inserts didn't make a region");

    // Each far insert sits in a cluster of far inserts only, in the new region.
    for (j, list) in tables.lists.iter().enumerate() {
        let ids: Vec<u64> = (list.first_id..list.first_id + list.count as u32)
            .map(|id| sq8::id(data.rows.row(id), DIM))
            .collect();
        let far_ids = ids.iter().filter(|&&id| id >= 90_000).count();
        assert!(
            far_ids == 0 || far_ids == ids.len(),
            "cluster {j} mixes far inserts with others"
        );
        assert!(
            far_ids == 0 || t.region_of[j] as usize == regions,
            "cluster {j} isn't in the new region"
        );
    }
}
