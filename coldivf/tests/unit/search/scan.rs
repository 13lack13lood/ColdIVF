//! Unit tests for the scan, including the SIMD scan against the portable one.

use super::*;
use crate::kernels::scan_block;
use crate::quantization::SplitMix64;
use crate::quantization::rabitq::{Factors, QueryConsts, code_bytes, pack_block};
use crate::search::candidates::order;

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

#[test]
fn simd_scan_matches_scalar_bit_for_bit() {
    let best = Tier::best();
    #[cfg(target_arch = "aarch64")]
    assert!(
        !best.is_scalar(),
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

            // An open bound, a lane's own candidate, and a lane's score with a nearby id, so ties get tested.
            let n_all = scan_block(Tier::SCALAR, &block, &query, first_id, count, Candidate::NONE, &mut all);
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

            let na = scan_block(Tier::SCALAR, &block, &query, first_id, count, bound, &mut a);
            let nb = scan_block(best, &block, &query, first_id, count, bound, &mut b);
            let bits = |c: &[Candidate]| c.iter().map(|c| (c.score.to_bits(), c.id)).collect::<Vec<_>>();
            assert_eq!(bits(&a[..na]), bits(&b[..nb]), "block {i} at d_pad {d_pad}");
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
        let n = scan_block_scalar(&block, &query, first, lanes, Candidate::NONE, &mut out);
        want.extend_from_slice(&out[..n]);
        blocks.extend(block);
    }
    want.sort_unstable_by(order);

    for tier in [Tier::SCALAR, Tier::best()] {
        for cap in [1, 10, 70, 100] {
            let mut candidates = Candidates::new(cap);
            let mut survivors = [Candidate::NONE; BLOCK];
            scan_list(tier, &blocks, count, first_id, &query, &mut candidates, &mut survivors);
            let mut kept = candidates.as_slice().to_vec();
            kept.sort_unstable_by(order);
            assert_eq!(kept, want[..cap.min(count)], "cap {cap}");
        }
    }
}
