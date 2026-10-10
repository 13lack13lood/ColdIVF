//! Unit tests for the scan, including the single instruction, multiple data (SIMD) scan against the portable one.

use super::*;
use crate::kernels::scan_block;
use crate::quantization::rabitq::{Factors, code_bytes, pack_block};
use crate::quantization::testutil::{clustered, gaussian, params};
use crate::quantization::{SplitMix64, padded_dim, sq4, write_f32};
use crate::search::candidates::order;
use crate::search::inmem::{InMemory, MAX_DIM};
use crate::search::{Scratch, prepare_query};
use crate::{BuildParams, Limits, Metric};

/// A query with a random lookup table for `d_pad` dimensions. Every 7th group has all its entries at 0, and the next
/// one has its largest entry at 60.
fn random_query(rng: &mut SplitMix64, d_pad: usize) -> QueryLut {
    let mut query = QueryLut::new(d_pad);
    for (g, entries) in query.lut.as_chunks_mut::<16>().0.iter_mut().enumerate() {
        let qg: [u8; 4] = match g % 7 {
            0 => [0; 4],
            1 => [15; 4],
            _ => [0; 4].map(|_: u8| (rng.next_u64() % 16) as u8),
        };
        for (j, e) in entries.iter_mut().enumerate() {
            *e = (0..4).filter(|&b| (j >> b) & 1 == 1).map(|b| qg[b]).sum();
        }
    }

    query.k = QueryConsts {
        k_s: 0.1 * rng.uniform() as f32,
        k_p: -(rng.uniform() as f32),
        k_c: rng.uniform() as f32 - 0.5,
        t: 1.5,
        t2: 2.25,
    };
    query
}

/// A block of `count` random codes with random factors. Its other lanes are unused.
fn random_block(rng: &mut SplitMix64, d_pad: usize, count: usize) -> Vec<u8> {
    let codes: Vec<u8> = (0..count * code_bytes(d_pad)).map(|_| rng.next_u64() as u8).collect();
    let factors: Vec<Factors> = (0..count)
        .map(|_| Factors {
            f_add: 50.0 * rng.uniform() as f32,
            f_rescale: -3.0 * rng.uniform() as f32,
            f_pop: (rng.next_u64() % d_pad as u64) as f32,
            f_err: rng.uniform() as f32,
        })
        .collect();

    let mut block = vec![0u8; block_bytes(d_pad)];
    pack_block(&codes, &factors, d_pad, &mut block);
    block
}

/// The bits of each candidate's score and its id.
fn bits(c: &[Candidate]) -> Vec<(u32, u32)> {
    c.iter().map(|c| (c.score.to_bits(), c.id)).collect()
}

/// One of NaN, ±∞, 3e38 and −0, values that only a damaged file or extreme data give.
fn special(rng: &mut SplitMix64) -> f32 {
    [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 3e38, -0.0][(rng.next_u64() % 5) as usize]
}

#[test]
fn simd_scan_matches_scalar_bit_for_bit() {
    #[cfg(target_arch = "aarch64")]
    assert!(
        !Tier::best().is_scalar(),
        "every 64-bit Arm CPU this runs on should have dotprod"
    );

    let mut rng = SplitMix64::new(3);
    let (mut all, mut a, mut b) = (
        [Candidate::NONE; BLOCK],
        [Candidate::NONE; BLOCK],
        [Candidate::NONE; BLOCK],
    );
    for d_pad in [128, 384, 768, 1536] {
        for i in 0..2500 {
            let count = if i % 4 == 0 {
                1 + (rng.next_u64() % BLOCK as u64) as usize
            } else {
                BLOCK
            };
            let block = random_block(&mut rng, d_pad, count);
            let query = random_query(&mut rng, d_pad);
            let first_id = (rng.next_u64() % 1000) as u32;
            // Every other block gives each lane its own t2.
            let lanes_t2: [f32; BLOCK] = std::array::from_fn(|_| 5.0 * rng.uniform() as f32);
            let t2 = (i % 2 == 1).then_some(&lanes_t2);
            // An open bound, a lane's own candidate, and a lane's score with a nearby id, so ties get tested.
            let n_all = scan_block(
                Tier::SCALAR,
                &block,
                &query,
                t2,
                first_id,
                count,
                Candidate::NONE,
                &mut all,
            );
            assert_eq!(n_all, count, "an open bound keeps every vector");
            let lane = all[(rng.next_u64() % n_all as u64) as usize];
            let bound = match i % 3 {
                0 => Candidate::NONE,
                1 => lane,
                _ => Candidate {
                    score: lane.score,
                    id: lane.id + (rng.next_u64() % 3) as u32,
                },
            };
            let na = scan_block(Tier::SCALAR, &block, &query, t2, first_id, count, bound, &mut a);
            for tier in Tier::simd() {
                let nb = scan_block(tier, &block, &query, t2, first_id, count, bound, &mut b);
                let tier = tier.name();
                assert_eq!(bits(&a[..na]), bits(&b[..nb]), "{tier}: block {i} at d_pad {d_pad}");
            }
        }
    }
}

#[test]
fn the_largest_sums_score_alike_on_every_tier() {
    // At the largest padded dimension, with every query value at 15, each set group adds 60, so a lane with every bit
    // set sums 1024·60 = 61440, near the top of 16 bits.
    let mut rng = SplitMix64::new(8);
    let d_pad = padded_dim(MAX_DIM);
    let mut query = random_query(&mut rng, d_pad);
    for entries in query.lut.as_chunks_mut::<16>().0 {
        *entries = std::array::from_fn(|j| 15 * j.count_ones() as u8);
    }
    let (mut a, mut b) = ([Candidate::NONE; BLOCK], [Candidate::NONE; BLOCK]);

    // Every lane at the top, then only lanes 0 to 15, then only lanes 16 to 31.
    for code in [0xff, 0x0f, 0xf0] {
        let mut block = random_block(&mut rng, d_pad, BLOCK);
        block[..4 * d_pad].fill(code);
        let na = scan_block(Tier::SCALAR, &block, &query, None, 0, BLOCK, Candidate::NONE, &mut a);
        assert_eq!(na, BLOCK);
        for tier in Tier::simd() {
            let nb = scan_block(tier, &block, &query, None, 0, BLOCK, Candidate::NONE, &mut b);
            assert_eq!(bits(&a[..na]), bits(&b[..nb]), "{} with codes {code:#x}", tier.name());
        }
    }
}

#[test]
fn damaged_lanes_and_unusual_bounds_rank_alike_on_every_tier() {
    // Special factors and per-lane t2 values, and bounds at ±∞ and NaN. A bound at +∞ with an id above 0 keeps every
    // finite estimate, and an estimate of +∞ or NaN only with a lower id. Every lane holds a code and factors, also
    // those past `count`, which no tier may write.
    let mut rng = SplitMix64::new(7);
    let (mut a, mut b) = ([Candidate::NONE; BLOCK], [Candidate::NONE; BLOCK]);

    for d_pad in [128, 768] {
        for i in 0..600 {
            let count = if i % 3 == 0 {
                1 + (rng.next_u64() % BLOCK as u64) as usize
            } else {
                BLOCK
            };
            let mut block = random_block(&mut rng, d_pad, BLOCK);
            // About one in 8 of f_add, f_rescale and f_pop, which est reads.
            for lane in 0..BLOCK {
                for k in 0..3 {
                    if rng.next_u64().is_multiple_of(8) {
                        let v = special(&mut rng);
                        write_f32(&mut block, 4 * d_pad + 4 * (BLOCK * k + lane), v);
                    }
                }
            }
            let query = random_query(&mut rng, d_pad);
            let first_id = (rng.next_u64() % 1000) as u32;
            let lanes_t2: [f32; BLOCK] = std::array::from_fn(|_| {
                if rng.next_u64().is_multiple_of(8) {
                    special(&mut rng)
                } else {
                    5.0 * rng.uniform() as f32
                }
            });
            let t2 = (i % 2 == 1).then_some(&lanes_t2);
            let id = first_id + (rng.next_u64() % 40) as u32;
            let score = match i % 5 {
                0 => f32::INFINITY,
                1 => f32::NAN,
                2 => f32::NEG_INFINITY,
                _ => 60.0 * rng.uniform() as f32 - 10.0,
            };
            let bound = Candidate { score, id };
            let na = scan_block(Tier::SCALAR, &block, &query, t2, first_id, count, bound, &mut a);
            for tier in Tier::simd() {
                let nb = scan_block(tier, &block, &query, t2, first_id, count, bound, &mut b);
                // A NaN's bits depend on which operand it came from, so any NaN matches any other.
                let same = |x: &Candidate, y: &Candidate| {
                    x.id == y.id && (x.score.to_bits() == y.score.to_bits() || (x.score.is_nan() && y.score.is_nan()))
                };
                assert!(
                    na == nb && a[..na].iter().zip(&b[..nb]).all(|(x, y)| same(x, y)),
                    "{}: block {i} at d_pad {d_pad} with bound {bound:?}: {:?} against {:?}",
                    tier.name(),
                    &a[..na],
                    &b[..nb]
                );
            }
        }
    }
}

#[test]
fn a_lane_with_its_own_t2_scores_as_a_scan_with_that_t2_does() {
    let mut rng = SplitMix64::new(6);
    let d_pad = 384;
    let (mut each, mut one) = ([Candidate::NONE; BLOCK], [Candidate::NONE; BLOCK]);

    for tier in Tier::supported() {
        for count in [BLOCK, 13] {
            let block = random_block(&mut rng, d_pad, count);
            let mut query = random_query(&mut rng, d_pad);
            let lanes_t2: [f32; BLOCK] = std::array::from_fn(|_| 5.0 * rng.uniform() as f32);
            let n = scan_block(
                tier,
                &block,
                &query,
                Some(&lanes_t2),
                40,
                count,
                Candidate::NONE,
                &mut each,
            );
            assert_eq!(n, count);
            for (lane, &t2) in lanes_t2.iter().enumerate().take(count) {
                query.k.t2 = t2;
                scan_block(tier, &block, &query, None, 40, count, Candidate::NONE, &mut one);
                assert_eq!(
                    each[lane].score.to_bits(),
                    one[lane].score.to_bits(),
                    "lane {lane} of {count}"
                );
            }
        }
    }
}

#[test]
fn scan_list_keeps_the_best_vectors_of_a_cluster() {
    let mut rng = SplitMix64::new(4);
    let d_pad = 256;
    let count: usize = 70;
    let first_id = 500;
    let query = random_query(&mut rng, d_pad);

    // Three blocks, the last one partial.
    let mut blocks = Vec::new();
    let mut want = Vec::new();
    for b in 0..count.div_ceil(BLOCK) {
        let lanes = (count - BLOCK * b).min(BLOCK);
        let block = random_block(&mut rng, d_pad, lanes);
        let mut out = [Candidate::NONE; BLOCK];
        let first = first_id + (BLOCK * b) as u32;
        let n = scan_block_scalar(&block, &query, None, first, lanes, Candidate::NONE, &mut out);
        want.extend_from_slice(&out[..n]);
        blocks.extend(block);
    }
    want.sort_unstable_by(order);

    for tier in Tier::supported() {
        for cap in [1, 10, 70, 100] {
            let mut candidates = Candidates::new(cap);
            let mut survivors = [Candidate::NONE; BLOCK];
            scan_list(
                tier,
                &blocks,
                count,
                first_id,
                &query,
                Hides::NONE,
                &mut candidates,
                &mut survivors,
            );
            let mut kept = candidates.as_slice().to_vec();
            kept.sort_unstable_by(order);
            assert_eq!(kept, want[..cap.min(count)], "cap {cap}");
        }
    }
}

#[test]
fn a_cluster_s_t2_is_the_exact_squared_distance_to_its_decoded_center() {
    let mut rng = SplitMix64::new(5);
    let nlist = 6;

    for dim in [200, 768] {
        let (rows, clusters) = clustered(&mut rng, 600, dim, nlist, 0.5);
        let index = InMemory::build(&params(Metric::L2), &rows, &clusters).unwrap();
        let mean_index = InMemory::build(
            &BuildParams {
                anchor: Anchor::Mean,
                ..params(Metric::L2)
            },
            &rows,
            &clusters,
        )
        .unwrap();
        // Column j of the rotation's matrix is P·e_j, so value j of ĉ − μ = Pᵀẑ is ⟨P·e_j, ẑ⟩.
        let (mut e, mut tmp) = (vec![0f32; dim], vec![0f32; dim]);
        let columns: Vec<Vec<f32>> = (0..dim)
            .map(|j| {
                let mut column = vec![0f32; dim];
                e.fill(0.0);
                e[j] = 1.0;
                index.rotation.apply(&e, &mut column, &mut tmp);
                column
            })
            .collect();
        let (ccb, size) = (sq4::code_bytes(padded_dim(dim)), block_bytes(padded_dim(dim)));
        let centers: Vec<Vec<f64>> = (0..nlist)
            .map(|leaf| {
                let (codes, f) = (&index.centers[leaf * ccb..(leaf + 1) * ccb], index.center_factors[leaf]);
                let z: Vec<f64> = (0..dim)
                    .map(|i| f64::from(f.lo) + f64::from(f.step) * f64::from(sq4::code_at(codes, i)))
                    .collect();
                let along = |j: usize| columns[j].iter().zip(&z).map(|(&p, &z)| f64::from(p) * z).sum::<f64>();
                (0..dim).map(|j| f64::from(index.mean[j]) + along(j)).collect()
            })
            .collect();
        let limits = Limits {
            max_k: 10,
            max_nprobe: nlist,
            max_rerank: 10,
        };
        let mut s = Scratch::new(dim, nlist, &limits);

        for q in gaussian(&mut rng, 5, dim, |_| 1.0).chunks_exact(dim) {
            // Each cluster's t2 is ‖q − ĉ‖² for the center ĉ that its codes decode to, on every tier, and only t and t2
            // of the query change.
            for tier in Tier::supported() {
                let view = View { tier, ..index.view() };
                prepare_query(&view, q, &mut s).unwrap();
                for leaf in 0..nlist as LeafId {
                    let list = view.lists[leaf as usize];
                    let blocks = &index.blocks[list.offset..list.offset + list.count.div_ceil(BLOCK) * size];
                    s.candidates.reset(10);
                    scan_leaf(
                        &view,
                        leaf,
                        blocks,
                        &mut s.lut,
                        &s.q32,
                        &mut s.candidates,
                        &mut s.survivors,
                    );
                    let exact: f64 = q
                        .iter()
                        .zip(&centers[leaf as usize])
                        .map(|(&x, &c)| (f64::from(x) - c).powi(2))
                        .sum();
                    let (t2, t) = (f64::from(s.lut.k.t2), f64::from(s.lut.k.t));
                    assert!(
                        (t2 - exact).abs() <= 1e-6 * exact,
                        "cluster {leaf} at dim {dim}: t2 {t2} against {exact}"
                    );
                    // The error bar's t also counts the query's 4-bit error.
                    let bar = (exact + s.lut.e2).sqrt();
                    assert!(
                        s.lut.e2 > 0.0 && (t - bar).abs() <= 1e-6 * bar,
                        "cluster {leaf} at dim {dim}: t {t}"
                    );
                    assert_eq!(
                        s.lut.k,
                        QueryConsts {
                            t: s.lut.k.t,
                            t2: s.lut.k.t2,
                            ..s.lut.mean
                        }
                    );
                }
            }

            // With the mean anchor every cluster keeps T and T², as the query prepared them.
            let view = mean_index.view();
            prepare_query(&view, q, &mut s).unwrap();
            for leaf in 0..nlist as LeafId {
                let list = view.lists[leaf as usize];
                let blocks = &mean_index.blocks[list.offset..list.offset + list.count.div_ceil(BLOCK) * size];
                scan_leaf(
                    &view,
                    leaf,
                    blocks,
                    &mut s.lut,
                    &s.q32,
                    &mut s.candidates,
                    &mut s.survivors,
                );
                assert_eq!(s.lut.k, s.lut.mean);
            }
        }
    }
}
