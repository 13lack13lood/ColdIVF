//! Timing checks for the codes, ignored by default. Run them in release mode with
//! `cargo test --release -p coldivf perf -- --ignored --nocapture`.

use std::hint::black_box;
use std::time::Instant;

use crate::kernels::{Tier, center_distance2, scan_block, sq4_dots};
use crate::quantization::rabitq::{BLOCK, Factors, QueryLut, block_bytes, code_bytes, pack_block};
use crate::quantization::rotation::Rotation;
use crate::quantization::sq4::{self, QueryS8, QueryS32};
use crate::quantization::testutil::{dot, gaussian};
use crate::quantization::{SplitMix64, padded_dim};
use crate::search::candidates::Candidate;

/// One rotation at 768 dimensions.
#[test]
#[ignore]
fn rotation_cost() {
    let rotation = Rotation::new(768, 1);
    let x = gaussian(&mut SplitMix64::new(2), 1, 768, |_| 1.0);
    let (mut out, mut tmp) = (vec![0f32; 768], vec![0f32; 768]);
    let runs = 200_000;
    let start = Instant::now();
    for _ in 0..runs {
        rotation.apply(black_box(&x), &mut out, &mut tmp);
        black_box(&out);
    }
    let us = start.elapsed().as_secs_f64() * 1e6 / f64::from(runs);
    println!("one rotation at 768 dimensions: {us:.2} µs");
}

/// A scan of 6250 cluster centers at 768 dimensions, the number at 1M vectors.
#[test]
#[ignore]
fn center_scan_cost() {
    let (dim, n) = (768, 6250);
    let mut rng = SplitMix64::new(5);
    let cb = sq4::code_bytes(padded_dim(dim));
    let centers = gaussian(&mut rng, n, dim, |_| 1.0);
    let mut codes = vec![0u8; n * cb];
    for (z, c) in centers.chunks_exact(dim).zip(codes.chunks_exact_mut(cb)) {
        sq4::encode(z, c);
    }
    let mut query = QueryS8::new(dim);
    query.prepare(&gaussian(&mut rng, 1, dim, |_| 1.0));
    let mut dots = vec![0i32; n];
    for tier in Tier::supported() {
        let (name, runs) = (tier.name(), 200);
        let start = Instant::now();
        for _ in 0..runs {
            sq4_dots(tier, black_box(&codes), &query.q8, &mut dots);
            black_box(&dots);
        }
        let us = start.elapsed().as_secs_f64() * 1e6 / f64::from(runs);
        println!("{name} scan of {n} centers at {dim} dimensions: {us:.1} µs");
    }
}

/// The query's distance to 128 decoded cluster centers at 768 dimensions, which a search at nprobe 128 needs when the
/// 1-bit codes are measured from the clusters' centers.
#[test]
#[ignore]
fn center_distance_cost() {
    let (dim, n) = (768, 128);
    let mut rng = SplitMix64::new(6);
    let cb = sq4::code_bytes(padded_dim(dim));
    let centers = gaussian(&mut rng, n, dim, |_| 1.0);
    let mut codes = vec![0u8; n * cb];
    let factors: Vec<sq4::CenterFactors> = centers
        .chunks_exact(dim)
        .zip(codes.chunks_exact_mut(cb))
        .map(|(z, c)| sq4::encode(z, c))
        .collect();
    let norms = sq4::decoded_norms(&codes, &factors, dim);
    let y = gaussian(&mut rng, 1, dim, |_| 1.0);
    let mut query = QueryS32::new(dim);
    for tier in Tier::supported() {
        let (name, runs) = (tier.name(), 5000);
        let start = Instant::now();
        for _ in 0..runs {
            query.prepare(black_box(&y), dot(&y, &y));
            for ((c, f), &norm2) in codes.chunks_exact(cb).zip(&factors).zip(&norms) {
                black_box(center_distance2(tier, black_box(c), f, norm2, &query));
            }
        }
        let us = start.elapsed().as_secs_f64() * 1e6 / f64::from(runs);
        println!(
            "{name} distance to {n} decoded centers at {dim} dimensions, with the query's preparation: {us:.1} µs"
        );
    }
}

/// A scan of 625 blocks, 20000 vectors at 768 dimensions, eg. 128 clusters of ~160 vectors. With a bound of −∞ no lane
/// passes, so it times the lookup sums and the estimates alone. With an open bound every lane passes.
#[test]
#[ignore]
fn block_scan_cost() {
    let (d_pad, blocks) = (768, 625);
    let mut rng = SplitMix64::new(7);
    let codes: Vec<u8> = (0..BLOCK * code_bytes(d_pad)).map(|_| rng.next_u64() as u8).collect();
    let factors: Vec<Factors> = (0..BLOCK)
        .map(|_| Factors {
            f_add: 50.0 * rng.uniform() as f32,
            f_rescale: -3.0 * rng.uniform() as f32,
            f_pop: (rng.next_u64() % d_pad as u64) as f32,
            f_err: rng.uniform() as f32,
        })
        .collect();
    let mut block = vec![0u8; block_bytes(d_pad)];
    pack_block(&codes, &factors, d_pad, &mut block);
    let all = block.repeat(blocks);
    let mut query = QueryLut::new(d_pad);
    query.prepare(&gaussian(&mut rng, 1, d_pad, |_| 1.0), 1.0);
    let mut out = [Candidate::NONE; BLOCK];
    let none_pass = Candidate {
        score: f32::NEG_INFINITY,
        id: 0,
    };

    for tier in Tier::supported() {
        for (what, bound) in [("no lane passes", none_pass), ("every lane passes", Candidate::NONE)] {
            let runs = 200;
            let start = Instant::now();
            for _ in 0..runs {
                for (b, block) in all.chunks_exact(block.len()).enumerate() {
                    let id = (BLOCK * b) as u32;
                    black_box(scan_block(
                        tier,
                        black_box(block),
                        &query,
                        None,
                        id,
                        BLOCK,
                        bound,
                        &mut out,
                    ));
                }
            }
            let us = start.elapsed().as_secs_f64() * 1e6 / f64::from(runs);
            let ns = us * 1e3 / (BLOCK * blocks) as f64;
            println!(
                "{} scan of {blocks} blocks at {d_pad} dimensions, {what}: {us:.1} µs, {ns:.2} ns per vector",
                tier.name()
            );
        }
    }
}
