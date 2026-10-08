//! Unit tests for routing.

use super::*;
use crate::quantization::sq4::{self, code_bytes, dots_scalar};
use crate::quantization::testutil::{dot, gaussian};
use crate::quantization::{SplitMix64, padded_dim};

#[test]
fn route_matches_a_full_sort_on_every_tier() {
    let mut rng = SplitMix64::new(2);
    for dim in [128, 200, 768] {
        let nlist = 50;
        let cb = code_bytes(padded_dim(dim));
        let mut zs = gaussian(&mut rng, nlist, dim, |_| 1.0);
        // Center 7 repeats center 3, so their scores tie and 3 must come first.
        let copy = zs[3 * dim..4 * dim].to_vec();
        zs[7 * dim..8 * dim].copy_from_slice(&copy);
        let mut codes = vec![0u8; nlist * cb];
        let factors: Vec<_> = zs
            .chunks_exact(dim)
            .zip(codes.chunks_exact_mut(cb))
            .map(|(z, c)| sq4::encode(z, c))
            .collect();
        let y = gaussian(&mut rng, 1, dim, |_| 1.0);
        let t2 = dot(&y, &y) as f32;
        let mut q8 = QueryS8::new(dim);
        q8.prepare(&y);
        let mut dots = vec![0i32; nlist];
        dots_scalar(&codes, &q8.q8, &mut dots);
        let mut want: Vec<Candidate> = factors
            .iter()
            .zip(&dots)
            .enumerate()
            .map(|(i, (f, &d))| Candidate {
                score: route_score(t2, f, &q8, d),
                id: i as u32,
            })
            .collect();
        want.sort_unstable_by(order);
        let (p3, p7) = (want.iter().position(|c| c.id == 3), want.iter().position(|c| c.id == 7));
        assert!(p3 < p7, "the tie between centers 3 and 7 must go to 3");
        let centers = Centers {
            codes: &codes,
            factors: &factors,
        };
        for tier in [Tier::SCALAR, Tier::best()] {
            for nprobe in [0, 1, 10, nlist, nlist + 5] {
                let mut routes = Vec::new();
                route(tier, &centers, &q8, t2, nprobe, &mut dots, &mut routes);
                assert_eq!(routes, want[..nprobe.min(nlist)], "dim {dim}, nprobe {nprobe}");
            }
        }
    }
}

#[test]
fn a_tie_between_two_centers_goes_to_the_lower_position() {
    let mut rng = SplitMix64::new(3);
    let (dim, nlist) = (128, 30);
    let cb = code_bytes(padded_dim(dim));

    for _ in 0..20 {
        let mut zs = gaussian(&mut rng, nlist, dim, |_| 1.0);
        let a = (rng.next_u64() % (nlist as u64 - 1)) as usize;
        let b = a + 1 + (rng.next_u64() % (nlist - a - 1) as u64) as usize;
        let copy = zs[a * dim..(a + 1) * dim].to_vec();
        zs[b * dim..(b + 1) * dim].copy_from_slice(&copy);
        let mut codes = vec![0u8; nlist * cb];
        let factors: Vec<_> = zs
            .chunks_exact(dim)
            .zip(codes.chunks_exact_mut(cb))
            .map(|(z, c)| sq4::encode(z, c))
            .collect();
        let y = gaussian(&mut rng, 1, dim, |_| 1.0);
        let t2 = dot(&y, &y) as f32;
        let mut q8 = QueryS8::new(dim);
        q8.prepare(&y);
        let centers = Centers {
            codes: &codes,
            factors: &factors,
        };
        let mut dots = vec![0i32; nlist];
        let mut full = Vec::new();
        route(Tier::SCALAR, &centers, &q8, t2, nlist, &mut dots, &mut full);
        let pa = full.iter().position(|c| c.id == a as u32).unwrap();
        assert_eq!(full[pa + 1].id, b as u32, "the copy must come right after the original");
        for tier in [Tier::SCALAR, Tier::best()] {
            let mut routes = Vec::new();
            route(tier, &centers, &q8, t2, pa + 1, &mut dots, &mut routes);
            assert_eq!(routes, full[..pa + 1], "positions {a} and {b}");
        }
    }
}

#[test]
fn reach_keeps_the_clusters_near_enough_to_the_nearest() {
    let routes = |scores: &[f32]| -> Vec<Candidate> {
        scores
            .iter()
            .enumerate()
            .map(|(i, &score)| Candidate { score, id: i as u32 })
            .collect()
    };
    // Scores are squared distances, so a reach of 0.5 keeps scores up to 2.25 times the nearest one's.
    let near = routes(&[4.0, 6.0, 9.0, 9.01, 20.0]);

    assert_eq!(within_reach(&near, None), 5);
    assert_eq!(within_reach(&near, Some(0.5)), 3);
    assert_eq!(within_reach(&near, Some(0.0)), 1);
    assert_eq!(within_reach(&near, Some(1e9)), 5);
    assert_eq!(within_reach(&[], Some(0.5)), 0);
    // A nearest score below 0 counts as 0, so only scores at most 0 stay, and the nearest always does.
    assert_eq!(within_reach(&routes(&[-1.0, 0.0, 0.5]), Some(0.5)), 2);
}
