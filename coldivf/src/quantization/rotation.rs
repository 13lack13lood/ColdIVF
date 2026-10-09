//! A fast random rotation. Rotating each vector before encoding spreads its values evenly over all dimensions, so no
//! single dimension carries most of it. The 1-bit codes keep only the sign of each value, and this spreading is what
//! keeps them accurate.
//!
//! The rotation runs 3 rounds, and each round does three things:
//! 1. It flips the sign of the coordinates its seed picked.
//! 2. It applies a fast Walsh-Hadamard transform to each block. The blocks are the powers of two that sum to the
//!    dimension d, largest first, eg. 512 and 256 for 768. Each block is then scaled by 1/√b, where b is its size.
//! 3. It shuffles the coordinates, so the next round mixes across the blocks. Output j takes input (j·stride) mod d,
//!    where the stride is the smallest odd number of at least 7 that shares no factor with d, eg. 7 for 768.
//!
//! Every step keeps lengths and dot products, so the whole rotation does too. It costs O(d log d) per vector.

use std::io::{Error, ErrorKind};

use super::SplitMix64;

/// Rounds per rotation. With 2 rounds, some coordinates of a 768-dimension vector spread over only ~700 of the 768
/// outputs.
pub(crate) const ROUNDS: usize = 3;

/// A seeded random rotation of `dim`-dimension vectors.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Rotation {
    dim: usize,
    /// Each round's sign bits, ⌈dim/8⌉ bytes per round. Bit j % 8 of byte j / 8 is set when the round negates
    /// coordinate j.
    signs: Vec<u8>,
    /// The same bits as one mask per round and coordinate. XOR with a mask flips the sign bit of an f32.
    masks: Vec<u32>,
    /// The blocks as (start, size, 1/√size).
    blocks: Vec<(usize, usize, f32)>,
    /// The stride, reduced mod dim.
    step: usize,
}

impl Rotation {
    /// A rotation whose signs come from `seed`.
    pub(crate) fn new(dim: usize, seed: u64) -> Rotation {
        let bytes = dim.div_ceil(8);
        let mut rng = SplitMix64::new(seed);
        let mut signs = vec![0u8; ROUNDS * bytes];

        for round in signs.chunks_exact_mut(bytes) {
            for chunk in round.chunks_mut(8) {
                let len = chunk.len();
                chunk.copy_from_slice(&rng.next_u64().to_le_bytes()[..len]);
            }
            // Clears the bits past the last coordinate, so equal rotations store equal bytes.
            if !dim.is_multiple_of(8) {
                round[bytes - 1] &= (1u8 << (dim % 8)) - 1;
            }
        }

        Rotation::with_signs(dim, signs)
    }

    /// Rebuilds a rotation from the sign bits stored in an index file.
    pub(crate) fn from_signs(dim: usize, signs: &[u8]) -> Result<Rotation, Error> {
        if dim == 0 || signs.len() != ROUNDS * dim.div_ceil(8) {
            return Err(Error::new(
                ErrorKind::InvalidData,
                "the rotation's signs don't match its dimension",
            ));
        }

        Ok(Rotation::with_signs(dim, signs.to_vec()))
    }

    fn with_signs(dim: usize, signs: Vec<u8>) -> Rotation {
        assert!(dim > 0, "a rotation needs at least one dimension");
        let bytes = dim.div_ceil(8);
        let mut masks = Vec::with_capacity(ROUNDS * dim);
        for round in signs.chunks_exact(bytes) {
            masks.extend((0..dim).map(|j| u32::from((round[j / 8] >> (j % 8)) & 1) << 31));
        }
        Rotation {
            dim,
            signs,
            masks,
            blocks: blocks(dim),
            step: stride(dim) % dim,
        }
    }

    pub(crate) fn dim(&self) -> usize {
        self.dim
    }

    /// The sign bits, as stored in an index file.
    pub(crate) fn signs(&self) -> &[u8] {
        &self.signs
    }

    /// Writes the rotation of `x` into `out`. `tmp` is scratch space. All three have length `dim`.
    pub(crate) fn apply(&self, x: &[f32], out: &mut [f32], tmp: &mut [f32]) {
        out.copy_from_slice(x);
        self.apply_in_place(out, tmp);
    }

    /// Rotates `v` in place. `tmp` is scratch space of the same length.
    pub(crate) fn apply_in_place(&self, v: &mut [f32], tmp: &mut [f32]) {
        let dim = self.dim;
        assert!(v.len() == dim && tmp.len() == dim);

        let (mut input, mut output) = (v, tmp);
        flip(input, &self.masks[..dim]);

        for round in 0..ROUNDS {
            for &(start, size, scale) in &self.blocks {
                fwht_scaled(&mut input[start..start + size], scale);
            }
            // The shuffle also flips the signs for the next round.
            let next = self.masks.get((round + 1) * dim..(round + 2) * dim);
            let mut from = 0;
            for (j, out_j) in output.iter_mut().enumerate() {
                *out_j = input[from];
                if let Some(masks) = next {
                    *out_j = f32::from_bits(out_j.to_bits() ^ masks[j]);
                }
                from += self.step;
                if from >= dim {
                    from -= dim;
                }
            }
            std::mem::swap(&mut input, &mut output);
        }
        // After an odd number of rounds the result is in `tmp`.
        if ROUNDS % 2 == 1 {
            output.copy_from_slice(input);
        }
    }

    /// Undoes the rotation of `v` in place, so it gives back what `apply_in_place` rotated, up to rounding. `tmp` is
    /// scratch space of the same length. It runs the rounds backwards and undoes each step in the opposite order. Each
    /// block's scaled transform is its own inverse.
    pub(crate) fn invert_in_place(&self, v: &mut [f32], tmp: &mut [f32]) {
        let dim = self.dim;
        assert!(v.len() == dim && tmp.len() == dim);

        let (mut input, mut output) = (v, tmp);

        for round in (0..ROUNDS).rev() {
            // A round's shuffle flips the signs for the next round after it moves the values.
            if let Some(masks) = self.masks.get((round + 1) * dim..(round + 2) * dim) {
                flip(input, masks);
            }
            let mut to = 0;
            for &x in input.iter() {
                output[to] = x;
                to += self.step;
                if to >= dim {
                    to -= dim;
                }
            }
            std::mem::swap(&mut input, &mut output);
            for &(start, size, scale) in &self.blocks {
                fwht_scaled(&mut input[start..start + size], scale);
            }
        }

        flip(input, &self.masks[..dim]);

        // After an odd number of rounds the result is in `tmp`.
        if ROUNDS % 2 == 1 {
            output.copy_from_slice(input);
        }
    }
}

/// Flips the sign of each value whose mask is set.
fn flip(v: &mut [f32], masks: &[u32]) {
    for (x, &m) in v.iter_mut().zip(masks) {
        *x = f32::from_bits(x.to_bits() ^ m);
    }
}

/// The fast Walsh-Hadamard transform of a block whose length is a power of two, in place, then scaled by `scale`. It
/// does the first three stages inside each run of 8 values and folds the scale into the last stage, which gives the
/// same bits as the plain loop followed by a separate scaling pass.
fn fwht_scaled(v: &mut [f32], scale: f32) {
    let n = v.len();
    let mut h = 1;

    if n >= 8 {
        for run in v.as_chunks_mut::<8>().0 {
            let mut x = *run;
            for k in [0, 2, 4, 6] {
                (x[k], x[k + 1]) = (x[k] + x[k + 1], x[k] - x[k + 1]);
            }
            for k in [0, 1, 4, 5] {
                (x[k], x[k + 2]) = (x[k] + x[k + 2], x[k] - x[k + 2]);
            }
            for k in 0..4 {
                (x[k], x[k + 4]) = (x[k] + x[k + 4], x[k] - x[k + 4]);
            }
            *run = x;
        }

        h = 8;
    }

    if h >= n {
        v.iter_mut().for_each(|x| *x *= scale);
    }

    while h < n {
        let last = 2 * h == n;
        for pair in v.chunks_exact_mut(2 * h) {
            let (lo, hi) = pair.split_at_mut(h);
            if last {
                for (a, b) in lo.iter_mut().zip(hi) {
                    let (x, y) = (*a, *b);
                    *a = (x + y) * scale;
                    *b = (x - y) * scale;
                }
            } else {
                for (a, b) in lo.iter_mut().zip(hi) {
                    let (x, y) = (*a, *b);
                    *a = x + y;
                    *b = x - y;
                }
            }
        }
        h *= 2;
    }
}

/// The blocks of `dim`, the powers of two that sum to it, largest first, as (start, size, 1/√size).
fn blocks(dim: usize) -> Vec<(usize, usize, f32)> {
    let mut out = Vec::new();
    let (mut start, mut left) = (0, dim);

    while left > 0 {
        let size = 1 << left.ilog2();
        out.push((start, size, (1.0 / (size as f64).sqrt()) as f32));
        start += size;
        left -= size;
    }

    out
}

/// The shuffle stride for `dim`, the smallest odd number of at least 7 that shares no factor with it.
fn stride(dim: usize) -> usize {
    (7..)
        .step_by(2)
        .find(|&k| gcd(k, dim) == 1)
        .expect("a prime larger than dim always qualifies")
}

fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        (a, b) = (b, a % b);
    }

    a
}

#[cfg(test)]
#[path = "../../tests/unit/quantization/rotation.rs"]
mod tests;
