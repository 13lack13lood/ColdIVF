//! Unit tests for the pending inserts by cluster.

use super::*;
use crate::quantization::SplitMix64;
use crate::quantization::padded_dim;
use crate::quantization::rabitq::unpack_lane;

const DIM: usize = 128;

/// `n` random lanes of codes of 128 dimensions, as (cluster, code, factors), in `nlist` clusters.
fn lanes(n: usize, nlist: u32, seed: u64) -> Vec<(u32, Vec<u8>, Factors)> {
    let mut rng = SplitMix64::new(seed);
    let cb = code_bytes(padded_dim(DIM));
    (0..n)
        .map(|i| {
            let code: Vec<u8> = (0..cb).map(|_| rng.next_u64() as u8).collect();
            let f = Factors {
                f_add: i as f32,
                f_rescale: -1.5,
                f_pop: (rng.next_u64() % 128) as f32,
                f_err: 0.25,
            };
            ((rng.next_u64() % u64::from(nlist)) as u32, code, f)
        })
        .collect()
}

/// Every lane that `groups` holds for cluster `list`, unpacked from the blocks, with its place, in the groups' order.
fn unpacked(groups: &Groups, list: u32) -> Vec<(u32, Vec<u8>, Factors)> {
    let d_pad = padded_dim(DIM);
    let (size, mut bits) = (block_bytes(d_pad), vec![0u8; code_bytes(d_pad)]);
    let mut out = Vec::new();

    for g in groups.of(list) {
        assert_eq!(g.blocks().len(), g.len().div_ceil(BLOCK) * size);
        for j in 0..g.len() {
            let block = &g.blocks()[j / BLOCK * size..(j / BLOCK + 1) * size];
            let f = unpack_lane(block, d_pad, j % BLOCK, &mut bits);
            out.push((g.place(j), bits.clone(), f));
        }
    }

    out
}

#[test]
fn inserts_one_at_a_time_and_all_at_once_make_the_same_groups() {
    // 700 inserts in 9 clusters, so the groups rebuild twice and keep recent ones after. From insert 256 on only
    // clusters 0 to 3 get any, so the second rebuild keeps the part-full blocks of the others as they are.
    let all: Vec<(u32, Vec<u8>, Factors)> = lanes(700, 9, 1)
        .into_iter()
        .enumerate()
        .map(|(i, (l, c, f))| (if i >= 256 { l % 4 } else { l }, c, f))
        .collect();
    let mut one = Groups::new(9, DIM);
    let mut halves = Groups::new(9, DIM);

    for (i, (l, c, f)) in all.iter().enumerate() {
        one = one.with(*l, c, *f, i as u32);
    }

    for half in [0..350, 350..700] {
        let part = half.clone().map(|i| (all[i].0, &all[i].1[..], all[i].2, i as u32));
        halves = halves.with_many(part);
    }

    let many = Groups::new(9, DIM).with_many(all.iter().enumerate().map(|(i, (l, c, f))| (*l, &c[..], *f, i as u32)));

    // Each cluster holds its inserts in place order, and every block unpacks to their codes and factors.
    for list in 0..9 {
        let want: Vec<(u32, Vec<u8>, Factors)> = (0..700)
            .filter(|&i| all[i].0 == list)
            .map(|i| (i as u32, all[i].1.clone(), all[i].2))
            .collect();
        assert!(!want.is_empty());
        assert!(unpacked(&one, list) == want, "one at a time, cluster {list}");
        assert!(unpacked(&halves, list) == want, "in two batches, cluster {list}");
        assert!(unpacked(&many, list) == want, "all at once, cluster {list}");
    }

    assert!(unpacked(&one, 9).is_empty() && unpacked(&one, 1000).is_empty());
}

#[test]
fn later_inserts_change_no_group_that_a_snapshot_holds() {
    let all = lanes(600, 5, 2);
    let mut groups = Groups::new(5, DIM);
    let mut kept = Vec::new();

    for (i, (l, c, f)) in all.iter().enumerate() {
        groups = groups.with(*l, c, *f, i as u32);
        if i % 97 == 0 {
            kept.push((
                i,
                groups.clone(),
                (0..5).map(|l| unpacked(&groups, l)).collect::<Vec<_>>(),
            ));
        }
    }

    for (i, g, lanes) in kept {
        assert!(
            (0..5).map(|l| unpacked(&g, l)).collect::<Vec<_>>() == lanes,
            "after insert {i}"
        );
        let held: usize = lanes.iter().map(Vec::len).sum();
        assert_eq!(held, i + 1);
    }
}
