//! Timing checks for the codes, ignored by default. Run them in release mode with
//! `cargo test --release -p coldivf perf -- --ignored --nocapture`.

use std::hint::black_box;
use std::time::Instant;

use crate::kernels::{Tier, sq4_dots};
use crate::quantization::rotation::Rotation;
use crate::quantization::sq4::{self, QueryS8};
use crate::quantization::testutil::gaussian;
use crate::quantization::{SplitMix64, padded_dim};

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
    for (name, tier) in [("scalar", Tier::SCALAR), ("best", Tier::best())] {
        let runs = 200;
        let start = Instant::now();
        for _ in 0..runs {
            sq4_dots(tier, black_box(&codes), &query.q8, &mut dots);
            black_box(&dots);
        }
        let us = start.elapsed().as_secs_f64() * 1e6 / f64::from(runs);
        println!("{name} scan of {n} centers at {dim} dimensions: {us:.1} µs");
    }
}
