//! Quantization, i.e. storing vectors with fewer bits than 32-bit floats. This module holds the rotation, the 1-bit
//! codes that queries scan, the 8-bit copies used for rescoring and the 4-bit codes of the cluster centers.

pub(crate) mod rabitq;
pub(crate) mod rotation;
pub(crate) mod sq4;
pub(crate) mod sq8;

use rotation::Rotation;

/// Rounds `n` up to a multiple of `m`.
pub(crate) const fn round_up(n: usize, m: usize) -> usize {
    n.div_ceil(m) * m
}

/// The dimension the codes use, the vector dimension rounded up to a multiple of 64. The extra dimensions are 0.
pub(crate) const fn padded_dim(dim: usize) -> usize {
    round_up(dim, 64)
}

/// Writes P(x − mean) into `out` and returns ‖x − mean‖, where P is `rotation`. `tmp` is scratch space. All of them
/// have the rotation's dimension.
pub(crate) fn rotate_offset(rotation: &Rotation, x: &[f32], mean: &[f32], out: &mut [f32], tmp: &mut [f32]) -> f64 {
    assert!(x.len() == out.len() && mean.len() == out.len());

    let mut norm2 = 0f64;
    for ((out_i, &x_i), &mean_i) in out.iter_mut().zip(x).zip(mean) {
        *out_i = x_i - mean_i;
        norm2 += f64::from(*out_i) * f64::from(*out_i);
    }

    rotation.apply_in_place(out, tmp);
    norm2.sqrt()
}

/// Reads the little-endian f32 that starts at byte `at`.
pub(crate) fn read_f32(bytes: &[u8], at: usize) -> f32 {
    f32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
}

/// Writes `v` as a little-endian f32 starting at byte `at`.
pub(crate) fn write_f32(bytes: &mut [u8], at: usize, v: f32) {
    bytes[at..at + 4].copy_from_slice(&v.to_le_bytes());
}

/// SplitMix64, a small seeded random number generator. The same seed always gives the same sequence, so the same
/// build settings always give the same index file.
pub(crate) struct SplitMix64(u64);

impl SplitMix64 {
    pub(crate) fn new(seed: u64) -> SplitMix64 {
        SplitMix64(seed)
    }

    pub(crate) fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);

        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

#[cfg(test)]
#[path = "../../tests/unit/testutil.rs"]
pub(crate) mod testutil;

#[cfg(test)]
#[path = "../../tests/perf/quantization.rs"]
mod perf;
