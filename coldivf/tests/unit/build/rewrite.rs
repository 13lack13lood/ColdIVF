//! Unit tests for rewriting an index file as a plan says.

use std::io::ErrorKind;

use super::*;
use crate::build::{threads, write as write_index};
use crate::format::Trailer;
use crate::quantization::SplitMix64;
use crate::quantization::testutil::temps_of;
use crate::quantization::testutil::{Rows, TempFile, clustered, dist2, gaussian, group, params, shuffle, temp_file};
use crate::update::log_path;
use crate::{BuildParams, Index, Limits, Metric, RunKind, SearchCtx, SearchParams, Writer};

const DIM: usize = 128;

/// An index file of `n` vectors with ids 0 to n − 1 around `nlist` centers in 4 groups, built with `p`, and those
/// vectors.
fn sample(name: &str, p: &BuildParams, n: usize, nlist: usize, seed: u64) -> (TempFile, Rows) {
    let (rows, mut clusters) = clustered(&mut SplitMix64::new(seed), n, DIM, nlist, 1.0);
    group(&mut clusters, 4);
    let index = InMemory::build(p, &rows, &clusters).unwrap();
    let path = temp_file(name);
    write_index(&index, &path).unwrap();
    let _ = std::fs::remove_file(log_path(&path));
    (path, rows)
}

/// The bytes of every section of the file at `path`, and what its header says.
fn sections(path: &Path) -> (Vec<Vec<u8>>, Layout) {
    let inner = Inner::open(path).unwrap();
    let (bytes, layout) = inner.parts();
    let parts = [
        Part::Header,
        Part::Rotation,
        Part::Centers,
        Part::Extents,
        Part::Codes,
        Part::Rows,
        Part::Ids,
    ];
    (
        parts.iter().map(|&p| bytes[layout.span(p).range()].to_vec()).collect(),
        layout.clone(),
    )
}

/// Changes through `writer`: inserts near some rows, deletes of others, an insert that replaces a row, an insert that a
/// later one of its id replaces, and an insert that is deleted again.
fn change(writer: &mut Writer, rows: &Rows, seed: u64) {
    let mut rng = SplitMix64::new(seed);
    let n = rows.data.len() / DIM;

    for i in 0..60 {
        let r = (rng.next_u64() % n as u64) as usize;
        let v: Vec<f32> = rows.data[r * DIM..(r + 1) * DIM]
            .iter()
            .map(|&x| x + 0.3 * rng.normal() as f32)
            .collect();
        writer.insert(100_000 + i, &v).unwrap();
    }

    for _ in 0..40 {
        writer.delete(rng.next_u64() % n as u64).unwrap();
    }

    writer.insert(5, &gaussian(&mut rng, 1, DIM, |_| 1.0)).unwrap();
    writer.insert(100_001, &rows.data[9 * DIM..10 * DIM]).unwrap();
    writer.delete(100_002).unwrap();
}

/// Whether each row of the old file and each insert of the snapshot is visible.
fn visible(inner: &Inner, delta: &Delta, seen: Seen) -> (Vec<bool>, Vec<bool>) {
    let rows = (0..inner.layout().count as u32).map(|r| delta.row_visible(r, seen));
    let inserts = (0..seen.inserts()).map(|i| delta.insert_visible(i, seen));
    (rows.collect(), inserts.collect())
}

/// A plan that keeps some old clusters in another order, gives the vectors of some others new centers, and moves some
/// vectors and inserts to other clusters, with new groups or the old ones. With `scatter`, every vector goes to a
/// cluster picked at random, so each window's rows lie all over the old file.
fn mixed(inner: &Inner, delta: &Delta, seen: Seen, new_groups: bool, scatter: bool, seed: u64) -> Plan {
    let mut rng = SplitMix64::new(seed);
    let (layout, tables, data) = (inner.layout(), inner.tables().unwrap(), inner.in_place());
    let nlist = tables.lists.len();
    let (rows, inserts) = visible(inner, delta, seen);
    // Old clusters 0 to 15 keep their centers in a shuffled order, and 16 to 21 get new ones at one of their rows.
    let mut kept: Vec<u32> = (0..16).collect();
    shuffle(&mut kept, &mut rng);
    let mut lists: Vec<PlanList> = kept
        .iter()
        .map(|&j| PlanList {
            center: PlanCenter::Keep(j),
            group: 0,
        })
        .collect();
    let mut decoded = vec![0f32; layout.dim];

    for j in 16..22 {
        let row = data.rows.row(tables.lists[j].first_id);
        sq8::decode(row, inner.mean(), &mut decoded);
        lists.push(PlanList {
            center: PlanCenter::New(decoded.iter().map(|&x| x + 0.01).collect()),
            group: 0,
        });
    }

    let ngroups = if new_groups { 3 } else { 2 };
    lists
        .iter_mut()
        .for_each(|l| l.group = (rng.next_u64() % ngroups as u64) as u16);
    let new_of = |j: usize| match j {
        0..16 => kept.iter().position(|&k| k as usize == j).unwrap() as u32,
        16..22 => (j - 16 + 16) as u32,
        _ => GONE,
    };
    let pick = |rng: &mut SplitMix64| (rng.next_u64() % lists.len() as u64) as u32;
    let file_rows = (0..layout.count)
        .map(|r| {
            let j = tables.lists.partition_point(|l| l.first_id as usize + l.count <= r);
            match (rows[r], new_of(j)) {
                (false, _) => GONE,
                (true, _) if scatter => pick(&mut rng),
                (true, GONE) => pick(&mut rng),
                (true, l) if r % 7 == 0 => (l + 1 + pick(&mut rng) % 21) % 22,
                (true, l) => l,
            }
        })
        .collect();
    let delta_rows = (0..seen.inserts())
        .map(|i| match (inserts[i], new_of(delta.leaf(i) as usize)) {
            (false, _) => GONE,
            (true, l) if l == GONE || i % 3 == 0 => pick(&mut rng),
            (true, l) => l,
        })
        .collect();
    let groups = match new_groups {
        true => PlanGroups::New(gaussian(&mut rng, 3, layout.dim, |_| 1.0)),
        false => PlanGroups::Keep(vec![3, 1]),
    };
    let trailer = Trailer {
        region_of: (0..22).map(|l| l % 5).collect(),
        list_last: (0..22).map(|l| 100 + l).collect(),
        region_last: vec![10, 20, 30, 40, 50],
        region_changes: vec![1, 2, 3, 4, 5],
        per_region: 4.4,
        major_count: 3000,
        major_inserts: 12,
        major_deletes: 3,
        pass_changes: 15,
    };
    assert_eq!(nlist, 24);

    Plan {
        lists,
        groups,
        file_rows,
        delta_rows,
        trailer: Some(trailer),
    }
}

/// What a rewrite must write: every section but the header, which is left empty, the vector count, and the codes
/// copied and measured again.
struct Want {
    sections: Vec<Vec<u8>>,
    count: usize,
    copied: usize,
    again: usize,
}

/// What a rewrite of `inner` and `delta` as of `seen` by `plan` must write, computed from the primitives: each
/// cluster's members in order, copied rows, copied codes from the old blocks or the inserts, codes measured again with
/// `encode_from` from the decoded rows, and the small sections from `InMemory::with_tables`.
fn oracle(inner: &Inner, delta: &Delta, seen: Seen, plan: &Plan) -> Want {
    let (layout, tables, data) = (inner.layout(), inner.tables().unwrap(), inner.in_place());
    let (dim, d_pad, count) = (layout.dim, padded_dim(layout.dim), layout.count);
    let (cb, rb, size, ccb) = (
        code_bytes(d_pad),
        sq8::row_bytes(dim),
        block_bytes(d_pad),
        sq4::code_bytes(d_pad),
    );
    let mut members = vec![Vec::new(); plan.lists.len()];
    (0..count)
        .filter(|&r| plan.file_rows[r] != GONE)
        .for_each(|r| members[plan.file_rows[r] as usize].push(r));
    let inserts = (0..seen.inserts()).filter(|&i| plan.delta_rows[i] != GONE);
    inserts.for_each(|i| members[plan.delta_rows[i] as usize].push(count + i));
    let row_of = |v: usize| {
        if v < count {
            data.rows.row(v as u32)
        } else {
            delta.row(v - count)
        }
    };

    let (mut centers, mut factors) = (Vec::new(), Vec::new());
    let (mut z, mut tmp, mut codes) = (vec![0f32; dim], vec![0f32; dim], vec![0u8; ccb]);
    for list in &plan.lists {
        match &list.center {
            PlanCenter::Keep(j) => {
                centers.extend_from_slice(&tables.codes[*j as usize * ccb..(*j as usize + 1) * ccb]);
                factors.push(tables.factors[*j as usize]);
            }
            PlanCenter::New(c) => {
                rotate_offset(inner.rotation(), c, inner.mean(), &mut z, &mut tmp);
                factors.push(sq4::encode(&z, &mut codes));
                centers.extend_from_slice(&codes);
            }
        }
    }
    let crb = sq8::center_row_bytes(dim);
    let parents = match &plan.groups {
        PlanGroups::Keep(old) => {
            let (bytes, _) = inner.parts();
            let start = layout.span(Part::Centers).offset;
            old.iter()
                .flat_map(|&g| bytes[start + g as usize * crb..start + (g as usize + 1) * crb].to_vec())
                .collect()
        }
        PlanGroups::New(g) => {
            let mut rows = vec![0u8; g.len() / dim * crb];
            g.chunks_exact(dim)
                .zip(rows.chunks_exact_mut(crb))
                .for_each(|(g, row)| sq8::write_center_row(g, inner.mean(), row));
            rows
        }
    };
    let stored = Stored {
        parents,
        centers: centers.clone(),
        center_factors: factors.clone(),
        parent_of: plan.lists.iter().map(|l| l.group).collect(),
        lists: place(members.iter().map(|m| m.len()), size),
    };
    let mut frame = InMemory::with_tables(layout, inner.mean().to_vec(), inner.rotation().clone(), stored);
    frame.epoch = layout.epoch + 1;
    let [rotation, small_centers, extents] = format::small_sections(&frame);

    let n: usize = members.iter().map(|m| m.len()).sum();
    let per_page = layout.page_size / rb;
    let mut rows = vec![0u8; format::rows_len(n, dim, layout.page_size)];
    let mut ids: Vec<u8> = Vec::new();
    let mut blocks = Vec::new();
    let (mut copied, mut again) = (0, 0);
    let mut encoder = Encoder::new(inner.rotation(), inner.mean());
    let (mut o, mut center) = (vec![0f32; dim], vec![0f32; dim]);
    let mut id = 0;

    for (l, m) in members.iter().enumerate() {
        let (mut lane_codes, mut lane_factors) = (vec![0u8; m.len() * cb], Vec::new());
        sq4::decode(&centers[l * ccb..(l + 1) * ccb], &factors[l], &mut center);
        for (k, &v) in m.iter().enumerate() {
            let row = row_of(v);
            let at = (id / per_page) * layout.page_size + (id % per_page) * rb;
            rows[at..at + rb].copy_from_slice(row);
            ids.extend(sq8::id(row, dim).to_le_bytes());
            id += 1;
            let from = match v.checked_sub(count) {
                Some(i) => delta.leaf(i) as usize,
                None => tables
                    .lists
                    .partition_point(|list| list.first_id as usize + list.count <= v),
            };
            let stays = match (&plan.lists[l].center, layout.anchor) {
                (_, Anchor::Mean) => true,
                (PlanCenter::Keep(j), _) => *j as usize == from,
                (PlanCenter::New(_), _) => false,
            };
            let bits = &mut lane_codes[k * cb..(k + 1) * cb];
            let f = match (stays, v.checked_sub(count)) {
                (true, Some(i)) => {
                    let (code, f) = delta.lane(i);
                    bits.copy_from_slice(code);
                    f
                }
                (true, None) => {
                    let list = tables.lists[from];
                    let j = v - list.first_id as usize;
                    let start = list.offset + j / BLOCK * size;
                    unpack_lane(&data.blocks[start..start + size], d_pad, j % BLOCK, bits)
                }
                (false, _) => {
                    sq8::decode(row, inner.mean(), &mut o);
                    encoder.encode_from(&o, &center, bits)
                }
            };
            if stays {
                copied += 1;
            } else {
                again += 1;
            }
            lane_factors.push(f);
        }
        let mut block = vec![0u8; size];
        for (b, chunk) in lane_factors.chunks(BLOCK).enumerate() {
            pack_block(
                &lane_codes[b * BLOCK * cb..(b * BLOCK + chunk.len()) * cb],
                chunk,
                d_pad,
                &mut block,
            );
            blocks.extend_from_slice(&block);
        }
    }

    ids.resize(format::ids_len(n), 0);
    Want {
        sections: vec![Vec::new(), rotation, small_centers, extents, blocks, rows, ids],
        count: n,
        copied,
        again,
    }
}

#[test]
fn a_rewrite_writes_exactly_the_sections_that_the_plan_gives() {
    for (metric, anchor, new_groups, scatter) in [
        (Metric::L2, Anchor::Cluster, true, false),
        (Metric::Cosine, Anchor::Cluster, false, false),
        (Metric::L2, Anchor::Mean, true, false),
        (Metric::L2, Anchor::Cluster, true, true),
        (Metric::L2, Anchor::Mean, false, true),
    ] {
        let p = BuildParams {
            anchor,
            ..params(metric)
        };
        let name = format!("layout-{metric:?}-{anchor:?}-{scatter}.coldivf");
        let (path, rows) = sample(&name, &p, 3000, 24, 71);
        let mut writer = Writer::open(&path).unwrap();
        change(&mut writer, &rows, 72);
        let inner = &writer.index.inner();
        let delta = inner.changes().load();
        let seen = delta.seen();
        let plan = mixed(inner, &delta, seen, new_groups, scatter, 73);
        let want = oracle(inner, &delta, seen, &plan);
        // Scattered rows go out in many small windows, so each window gathers rows from all over the old file.
        let window = if scatter { 64 << 10 } else { WINDOW };
        let pool = threads(3).unwrap();
        let (tmp, done) = rewrite_with(inner, &delta, seen, &plan, &path, &pool, window).unwrap();
        let (got, layout) = sections(&tmp);
        let what = format!("{metric:?} {anchor:?}, scattered {scatter}");

        let names = ["header", "rotation", "centers", "extents", "codes", "rows", "ids"];
        for part in 1..7 {
            assert!(
                got[part] == want.sections[part],
                "{what}: the {} section differs",
                names[part]
            );
        }
        let old = inner.layout();
        assert_eq!(
            (
                layout.count,
                layout.nlist,
                layout.epoch,
                layout.anchor,
                layout.metric,
                layout.seed
            ),
            (want.count, 22, old.epoch + 1, anchor, metric, old.seed),
            "{what}"
        );
        let counts = (done.count, done.nlist, done.copied, done.reencoded);
        assert_eq!(counts, (want.count, 22, want.copied, want.again), "{what}");
        assert_eq!(want.again == 0, anchor == Anchor::Mean, "{what}");
        let file = std::fs::read(&tmp).unwrap();
        let trailer = &file[format::trailer_at(layout.span(Part::Ids))..];
        assert_eq!(
            trailer,
            format::encode_trailer(plan.trailer.as_ref().unwrap()),
            "{what}"
        );
        drop(delta);
        let _ = std::fs::remove_file(&tmp);
    }
}

/// A vector's code, its factors' bits and its row.
type Lane = (Vec<u8>, [u32; 4], Vec<u8>);

/// Each vector's code, factors and row in the file at `path`, by id.
fn lanes_by_id(path: &Path) -> std::collections::HashMap<u64, Lane> {
    let inner = Inner::open(path).unwrap();
    let (layout, tables, data) = (inner.layout(), inner.tables().unwrap(), inner.in_place());
    let d_pad = padded_dim(layout.dim);
    let (cb, size) = (code_bytes(d_pad), block_bytes(d_pad));
    let mut out = std::collections::HashMap::new();

    for list in &tables.lists {
        for k in 0..list.count {
            let id = list.first_id + k as u32;
            let start = list.offset + k / BLOCK * size;
            let mut bits = vec![0u8; cb];
            let f = unpack_lane(&data.blocks[start..start + size], d_pad, k % BLOCK, &mut bits);
            let row = data.rows.row(id).to_vec();
            let factors = [f.f_add, f.f_rescale, f.f_pop, f.f_err].map(f32::to_bits);
            out.insert(sq8::id(&row, layout.dim), (bits, factors, row));
        }
    }

    out
}

#[test]
fn folds_keep_every_section_without_changes_and_every_lane_with_them() {
    let (path, rows) = sample("folds.coldivf", &params(Metric::L2), 3000, 24, 81);
    let (before, _) = sections(&path);
    let mut writer = Writer::open(&path).unwrap();

    for _ in 0..10 {
        writer.compact().unwrap();
        let (after, layout) = sections(&path);
        assert!(
            (1..7).all(|p| after[p] == before[p]),
            "a fold without changes changed a section"
        );
        assert_eq!(layout.epoch, writer.index.inner().layout().epoch);
    }

    // With changes, every vector that stays keeps its code, its factors and its row bit for bit, and so does every
    // insert, whose code was measured from the cluster it was routed to.
    let old = lanes_by_id(&path);
    change(&mut writer, &rows, 82);
    let mut inserted = std::collections::HashMap::new();
    let stays = {
        let inner = &writer.index.inner();
        let delta = inner.changes().load();
        let (rows, inserts) = visible(inner, &delta, delta.seen());
        for i in 0..delta.len() {
            let (code, f) = delta.lane(i);
            let factors = [f.f_add, f.f_rescale, f.f_pop, f.f_err].map(f32::to_bits);
            inserted.insert(
                sq8::id(delta.row(i), DIM),
                (code.to_vec(), factors, delta.row(i).to_vec()),
            );
        }
        rows.iter().chain(&inserts).filter(|&&v| v).count()
    };
    writer.compact().unwrap();
    let new = lanes_by_id(&path);
    assert_eq!(new.len(), stays, "the fold kept another number of vectors");

    for (id, lane) in &new {
        let want = inserted.get(id).or(old.get(id)).unwrap();
        assert!(lane == want, "vector {id} changed its lane or its row");
    }
}

#[test]
fn codes_measured_again_from_rows_lose_little_recall() {
    let (path, rows) = sample("again.coldivf", &params(Metric::L2), 4000, 24, 91);
    let index = Index::open(&path).unwrap();
    let inner = &index.inner();
    let (layout, tables) = (inner.layout(), inner.tables().unwrap());
    let ccb = sq4::code_bytes(padded_dim(DIM));
    // Every cluster gets its own center as it decodes as a new one, so every code is measured again from its row.
    let lists = (0..24)
        .map(|j| PlanList {
            center: PlanCenter::New(crate::maintain::decode_center(
                inner,
                &tables.codes[j * ccb..(j + 1) * ccb],
                &tables.factors[j],
            )),
            group: tables.parent_of[j],
        })
        .collect();
    let file_rows = (0..layout.count)
        .map(|r| tables.lists.partition_point(|l| l.first_id as usize + l.count <= r) as u32)
        .collect();
    let plan = Plan {
        lists,
        groups: PlanGroups::Keep((0..layout.nparents as u16).collect()),
        file_rows,
        delta_rows: Vec::new(),
        trailer: None,
    };
    let delta = inner.changes().load();
    let (tmp, done) = rewrite(inner, &delta, delta.seen(), &plan, &path, &threads(2).unwrap()).unwrap();
    assert_eq!((done.reencoded, done.copied), (4000, 0));
    // An open of the same file takes the changes' write lock.
    drop(delta);

    // The same vectors sit at the same internal ids, so their codes compare bit for bit.
    let (old, new) = (sections(&path).0, sections(&tmp).0);
    let d_pad = padded_dim(DIM);
    let (size, groups) = (block_bytes(d_pad), 4 * d_pad);
    let differ: u32 = old[4]
        .chunks_exact(size)
        .zip(new[4].chunks_exact(size))
        .map(|(a, b)| {
            a[..groups]
                .iter()
                .zip(&b[..groups])
                .map(|(x, y)| (x ^ y).count_ones())
                .sum::<u32>()
        })
        .sum();
    let share = f64::from(differ) / (4000 * DIM) as f64;
    assert!(share <= 0.01, "{:.4} of the sign bits changed", share);

    // Recall@10 against the exact neighbors of the original vectors, reading every cluster and rescoring few
    // candidates, so the codes decide which vectors reach the rescore.
    let mut rng = SplitMix64::new(92);
    let queries: Vec<f32> = (0..300)
        .flat_map(|_| {
            let r = (rng.next_u64() % 4000) as usize;
            let v: Vec<f32> = rows.data[r * DIM..(r + 1) * DIM].to_vec();
            v.into_iter()
                .map(|x| x + 0.5 * rng.normal() as f32)
                .collect::<Vec<f32>>()
        })
        .collect();
    let recall = |index: &Index| -> f64 {
        let sp = SearchParams {
            k: 10,
            nprobe: 24,
            rerank: 15,
            reach: None,
        };
        let mut ctx = SearchCtx::new(Limits {
            max_k: 10,
            max_nprobe: 24,
            max_rerank: 15,
        });
        let mut found = 0;
        for q in queries.as_chunks::<DIM>().0 {
            let mut exact: Vec<(f64, u64)> = rows
                .data
                .as_chunks::<DIM>()
                .0
                .iter()
                .enumerate()
                .map(|(i, v)| (dist2(q, v), i as u64))
                .collect();
            exact.sort_by(|a, b| a.0.total_cmp(&b.0));
            let truth: Vec<u64> = exact[..10].iter().map(|&(_, i)| i).collect();
            let hits = index.search(q, &sp, &mut ctx).unwrap();
            found += hits.iter().filter(|h| truth.contains(&h.id)).count();
        }
        found as f64 / (10 * 300) as f64
    };
    let (before, after) = (recall(&index), recall(&Index::open(&tmp).unwrap()));
    assert!((before - after).abs() <= 0.005, "recall {before:.4} became {after:.4}");
}

#[test]
fn measuring_codes_again_from_the_same_rows_gives_the_same_bytes() {
    let (path, _) = sample("same-bytes.coldivf", &params(Metric::Cosine), 2000, 24, 95);
    let index = Index::open(&path).unwrap();
    let ccb = sq4::code_bytes(padded_dim(DIM));
    let plan_for = |inner: &Inner, centers: &[Vec<f32>]| {
        let (layout, tables) = (inner.layout(), inner.tables().unwrap());
        Plan {
            lists: centers
                .iter()
                .enumerate()
                .map(|(j, c)| PlanList {
                    center: PlanCenter::New(c.clone()),
                    group: tables.parent_of[j],
                })
                .collect(),
            groups: PlanGroups::Keep((0..layout.nparents as u16).collect()),
            file_rows: (0..layout.count)
                .map(|r| tables.lists.partition_point(|l| l.first_id as usize + l.count <= r) as u32)
                .collect(),
            delta_rows: Vec::new(),
            trailer: None,
        }
    };
    let inner = index.inner();
    let tables = inner.tables().unwrap();
    // Centers a little off the old ones, so every code is measured again against a center no code was measured from.
    let centers: Vec<Vec<f32>> = (0..24)
        .map(|j| {
            let c = crate::maintain::decode_center(&inner, &tables.codes[j * ccb..(j + 1) * ccb], &tables.factors[j]);
            c.iter().map(|&x| x * 1.01).collect()
        })
        .collect();
    let pool = threads(2).unwrap();
    let delta = inner.changes().load();
    let plan = plan_for(&inner, &centers);
    let (first, _) = rewrite(&inner, &delta, delta.seen(), &plan, &path, &pool).unwrap();
    let again = Inner::open(&first).unwrap();
    let delta_again = again.changes().load();
    let plan = plan_for(&again, &centers);
    let (second, done) = rewrite(&again, &delta_again, delta_again.seen(), &plan, &first, &pool).unwrap();
    assert_eq!(done.reencoded, 2000);
    // An open of the same file takes the changes' write lock.
    drop((delta, delta_again));
    let (a, b) = (sections(&first).0, sections(&second).0);
    assert!(
        (1..7).all(|p| a[p] == b[p]),
        "measuring the codes again from the same rows changed a section"
    );
}

#[test]
fn a_damaged_input_fails_the_run_and_leaves_the_old_file() {
    for part in [Part::Rows, Part::Codes, Part::Centers] {
        let (path, rows) = sample(&format!("damaged-{part:?}.coldivf"), &params(Metric::L2), 3000, 24, 101);
        let mut bytes = std::fs::read(&path).unwrap();
        let layout = Inner::open(&path).unwrap().layout().clone();
        let span = layout.span(part);
        // In the centers section, a byte of a center's 4-bit codes, past the group rows.
        let at = match part {
            Part::Centers => span.offset + layout.nparents * sq8::center_row_bytes(DIM) + 100,
            _ => span.offset + span.len / 2,
        };
        bytes[at] ^= 0x10;
        std::fs::write(&path, &bytes).unwrap();
        let mut writer = Writer::open(&path).unwrap();
        writer.insert(9000, &rows.data[..DIM]).unwrap();
        writer.delete(7).unwrap();
        let search = |index: &Index| {
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
            let hits = index.search(&rows.data[DIM..2 * DIM], &sp, &mut ctx);
            hits.map(|h| h.iter().map(|h| (h.id, h.score.to_bits())).collect::<Vec<_>>())
                .map_err(|e| e.kind())
        };
        let before = search(&Index::open(&path).unwrap());

        // A fold copies from every section. A major run measures every code again, so it copies nothing of the old
        // codes, and a damaged codes section can't reach its new file.
        let copies = |&kind: &RunKind| (kind, part) != (RunKind::Major, Part::Codes);
        for kind in [RunKind::Fold, RunKind::Major].iter().filter(|k| copies(k)) {
            let err = writer.maintain(&crate::MaintainParams::default(), *kind).unwrap_err();
            assert_eq!(err.kind(), ErrorKind::InvalidData, "{part:?} {kind:?}: {err}");
            assert!(
                temps_of(&path).is_empty(),
                "{part:?} {kind:?}: a temporary file is still there"
            );
            assert!(
                std::fs::read(&path).unwrap() == bytes,
                "{part:?} {kind:?}: the old file changed"
            );
        }

        assert_eq!(search(&Index::open(&path).unwrap()), before, "{part:?}");

        if part == Part::Codes {
            let report = writer
                .maintain(&crate::MaintainParams::default(), RunKind::Major)
                .unwrap();
            assert_eq!(report.copied, 0);
            Index::open(&path).unwrap().verify().unwrap();
            continue;
        }
        // The writer goes on with the old file and its log.
        writer.insert(9001, &rows.data[2 * DIM..3 * DIM]).unwrap();
        drop(writer);
        assert!(std::fs::metadata(log_path(&path)).unwrap().len() > crate::update::log::HEADER_BYTES as u64);
    }
}

#[test]
fn a_plan_that_does_not_fit_fails_with_invalid_input() {
    let (path, _) = sample("bad-plan.coldivf", &params(Metric::L2), 600, 24, 103);
    let index = Index::open(&path).unwrap();
    let inner = index.inner();
    let delta = inner.changes().load();
    let pool = threads(1).unwrap();
    let good = || Plan {
        lists: (0..24)
            .map(|j| PlanList {
                center: PlanCenter::Keep(j),
                group: 0,
            })
            .collect(),
        groups: PlanGroups::Keep(vec![0]),
        file_rows: (0..600).map(|r| (r % 24) as u32).collect(),
        delta_rows: Vec::new(),
        trailer: None,
    };
    let fails = |plan: Plan| {
        let err = rewrite(&index.inner(), &delta, delta.seen(), &plan, &path, &pool)
            .map(|_| ())
            .unwrap_err();
        err.kind() == ErrorKind::InvalidInput
    };

    assert!(
        fails(Plan {
            file_rows: vec![0; 599],
            ..good()
        }),
        "a row without a cluster"
    );
    assert!(
        fails(Plan {
            file_rows: vec![24; 600],
            ..good()
        }),
        "a cluster past the plan's"
    );
    assert!(
        fails(Plan {
            groups: PlanGroups::Keep(vec![4]),
            ..good()
        }),
        "an old group the file lacks"
    );
    assert!(
        fails(Plan {
            file_rows: vec![GONE; 600],
            ..good()
        }),
        "no vector at all"
    );
    let mut far = good();
    far.lists[3].center = PlanCenter::Keep(24);
    assert!(fails(far), "an old center the file lacks");
    let mut short = good();
    short.lists[0].center = PlanCenter::New(vec![0.0; DIM - 1]);
    assert!(fails(short), "a center of another dimension");
    assert!(temps_of(&path).is_empty());
}
