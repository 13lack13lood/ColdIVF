//! The 1-bit codes that queries scan. Each vector keeps only the sign of each rotated value, 1 bit per dimension, plus
//! 4 numbers that turn a quick sum over those bits into an estimate of its distance to the query.
//!
//! Let μ be the mean, P the rotation and d the dimension. A vector o is encoded as
//!
//! ```text
//! r = o − μ    L = ‖r‖    w = P·r / L    b_i = 1 if w_i > 0, else 0    α = Σ|w_i| / √d
//! f_add = L²    f_rescale = −2L / α    f_pop = Σ b_i    f_err = 2L·ε0·√(max(1 − α², 0) / (α²(d − 1)))
//! ```
//!
//! where ε0 = 1.9 sets the width of the error bar. A query q is rotated the same way and quantized to 4 bits per
//! dimension as
//!
//! ```text
//! T = ‖q − μ‖    y = P(q − μ) / T    v_l = min y    Δ = (max y − v_l) / 15
//! q_u,i = clamp(round((y_i − v_l) / Δ), 0, 15)    S_y = Σ_i (v_l + Δ·q_u,i)
//! k_s = 2ΔT / √d    k_p = 2v_l·T / √d    k_c = −T·S_y / √d
//! ```
//!
//! For a vector with bits b, let s be the sum of q_u,i over the dimensions where b_i = 1. Then
//!
//! ```text
//! est = f_add + T² + f_rescale·(k_s·s + k_p·f_pop + k_c) ≈ ‖q − o‖²
//! ```
//!
//! and the true distance is within f_err·T of est for ~95% of pairs. Every path computes est with the same three fused
//! multiply-adds in the same order, so they all give the same bits.
//!
//! Queries scan the codes in blocks of 32 vectors, over d_pad dimensions, d rounded up to a multiple of 64. The 32
//! vector slots of a block are its lanes. A block holds the codes as 4-bit groups. Group g covers dimensions 4g to
//! 4g + 3, and bit t of a group is the bit of dimension 4g + t. Byte i of group g holds vector i's group in its low
//! half and vector 16 + i's group in its high half. After the d_pad/4 groups of 16 bytes come the factors, 32 values
//! each of f_add, f_rescale, f_pop and f_err. The query's lookup table has 16 entries per group, and entry j is the sum
//! of q_u over the bits set in j, at most 60.

use libm::fmaf;

use super::rotation::Rotation;
use super::{padded_dim, read_f32, rotate_offset, write_f32};

/// Sets the width of the error bar. With 1.9, ~95% of true distances fall within it.
pub(crate) const EPS0: f32 = 1.9;

/// Vectors per block.
pub(crate) const BLOCK: usize = 32;

/// The 4 numbers stored with each code.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Factors {
    pub(crate) f_add: f32,
    pub(crate) f_rescale: f32,
    pub(crate) f_pop: f32,
    pub(crate) f_err: f32,
}

impl Factors {
    /// The factors of an unused lane in a partial block. Its estimate is +∞, so it never ranks.
    pub(crate) const EMPTY: Factors = Factors {
        f_add: f32::INFINITY,
        f_rescale: 0.0,
        f_pop: 0.0,
        f_err: 0.0,
    };
}

/// The bytes in one vector's code, 1 bit per padded dimension.
pub(crate) const fn code_bytes(d_pad: usize) -> usize {
    d_pad / 8
}

/// The bytes in one block, the 4-bit groups and then the factors.
pub(crate) const fn block_bytes(d_pad: usize) -> usize {
    4 * d_pad + 16 * BLOCK
}

/// Encodes vectors against a mean and a rotation. It owns its scratch space, so encoding allocates nothing.
pub(crate) struct Encoder<'a> {
    rotation: &'a Rotation,
    mean: &'a [f32],
    /// P·r, before dividing by L.
    rotated: Vec<f32>,
    tmp: Vec<f32>,
}

impl<'a> Encoder<'a> {
    pub(crate) fn new(rotation: &'a Rotation, mean: &'a [f32]) -> Encoder<'a> {
        let dim = rotation.dim();
        assert!(dim >= 2, "the error bar needs at least 2 dimensions");
        assert_eq!(mean.len(), dim);

        Encoder {
            rotation,
            mean,
            rotated: vec![0.0; dim],
            tmp: vec![0.0; dim],
        }
    }

    /// Encodes `o` into `bits`, which has `code_bytes(d_pad)` bytes. Bit i % 8 of byte i / 8 is b_i, and the padding
    /// bits are 0. A vector equal to the mean gets f_add = f_rescale = 0, so its estimate is exactly T².
    pub(crate) fn encode(&mut self, o: &[f32], bits: &mut [u8]) -> Factors {
        let dim = self.rotation.dim();
        assert_eq!(bits.len(), code_bytes(padded_dim(dim)));

        let l = rotate_offset(self.rotation, o, self.mean, &mut self.rotated, &mut self.tmp);
        bits.fill(0);

        if l == 0.0 {
            return Factors {
                f_add: 0.0,
                f_rescale: 0.0,
                f_pop: 0.0,
                f_err: 0.0,
            };
        }

        let (mut abs_sum, mut pop) = (0f64, 0u32);

        for (i, &x) in self.rotated.iter().enumerate() {
            abs_sum += f64::from(x).abs();
            if x > 0.0 {
                bits[i / 8] |= 1 << (i % 8);
                pop += 1;
            }
        }

        let d = dim as f64;
        // Σ|w_i| is at least 1 for the unit vector w = P·r / L, so α is at least 1/√d. The clamp only matters when P·r
        // underflows to zero for a tiny r.
        let alpha = (abs_sum / l / d.sqrt()).max(1e-6);
        let f_err = 2.0 * l * f64::from(EPS0) * ((1.0 - alpha * alpha).max(0.0) / (alpha * alpha * (d - 1.0))).sqrt();

        Factors {
            f_add: (l * l) as f32,
            f_rescale: (-2.0 * l / alpha) as f32,
            f_pop: pop as f32,
            f_err: f_err as f32,
        }
    }
}

/// The constants of est for one query.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct QueryConsts {
    pub(crate) k_s: f32,
    pub(crate) k_p: f32,
    pub(crate) k_c: f32,
    /// T = ‖q − μ‖.
    pub(crate) t: f32,
    /// T².
    pub(crate) t2: f32,
}

/// A query prepared for scanning 1-bit codes, with its 4-bit values, its lookup table and the constants of est. It is
/// sized once, so preparing a query allocates nothing.
pub(crate) struct QueryLut {
    dim: usize,
    /// q_u,i for each padded dimension. The padding dimensions are 0.
    pub(crate) qu: Vec<u8>,
    /// 16 entries per 4-bit group, d_pad/4 groups.
    pub(crate) lut: Vec<u8>,
    pub(crate) k: QueryConsts,
}

impl QueryLut {
    pub(crate) fn new(dim: usize) -> QueryLut {
        let d_pad = padded_dim(dim);
        QueryLut {
            dim,
            qu: vec![0; d_pad],
            lut: vec![0; 4 * d_pad],
            k: QueryConsts::default(),
        }
    }

    /// Prepares the query from `y_rot` = P(q − μ) and `t` = ‖q − μ‖. A query equal to the mean (t = 0) gets zero
    /// constants, so every estimate is f_add.
    pub(crate) fn prepare(&mut self, y_rot: &[f32], t: f64) {
        assert_eq!(y_rot.len(), self.dim);

        self.qu.fill(0);
        self.lut.fill(0);
        self.k = QueryConsts::default();

        if t == 0.0 {
            return;
        }

        let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);

        for &v in y_rot {
            let y = f64::from(v) / t;
            lo = lo.min(y);
            hi = hi.max(y);
        }

        let delta = (hi - lo) / 15.0;
        let mut qu_sum = 0u64;

        if delta > 0.0 {
            for (q, &v) in self.qu.iter_mut().zip(y_rot) {
                *q = ((f64::from(v) / t - lo) / delta).round().clamp(0.0, 15.0) as u8;
                qu_sum += u64::from(*q);
            }
        }

        for (entries, qg) in self
            .lut
            .as_chunks_mut::<16>()
            .0
            .iter_mut()
            .zip(self.qu.as_chunks::<4>().0)
        {
            for (j, e) in entries.iter_mut().enumerate() {
                *e = (0..4).filter(|&b| (j >> b) & 1 == 1).map(|b| qg[b]).sum();
            }
        }

        let d = self.dim as f64;
        let s_y = d * lo + delta * qu_sum as f64;

        self.k = QueryConsts {
            k_s: (2.0 * delta * t / d.sqrt()) as f32,
            k_p: (2.0 * lo * t / d.sqrt()) as f32,
            k_c: (-t * s_y / d.sqrt()) as f32,
            t: t as f32,
            t2: (t * t) as f32,
        };
    }
}

/// The estimated squared distance to a vector, from its sum s and its factors. Every path uses these three fused
/// multiply-adds in this order, so all paths give the same bits.
#[inline]
pub(crate) fn est(s: u32, f: &Factors, k: &QueryConsts) -> f32 {
    let a = fmaf(k.k_s, s as f32, k.k_c);
    let ip = fmaf(k.k_p, f.f_pop, a);
    fmaf(f.f_rescale, ip, f.f_add + k.t2)
}

/// The low end of the error bar, est − f_err·T.
#[inline]
pub(crate) fn lower(est: f32, f: &Factors, k: &QueryConsts) -> f32 {
    fmaf(-f.f_err, k.t, est)
}

/// The sum s for one code, which adds up q_u over the dimensions whose bit is set.
pub(crate) fn code_sum(bits: &[u8], qu: &[u8]) -> u32 {
    qu.iter()
        .enumerate()
        .map(|(i, &q)| u32::from((bits[i / 8] >> (i % 8)) & 1) * u32::from(q))
        .sum()
}

/// The 4-bit group g of a code.
fn group(bits: &[u8], g: usize) -> u8 {
    (bits[g / 2] >> (4 * (g % 2))) & 0x0f
}

/// Packs up to 32 codes and their factors into one block. `codes` holds the codes back to back. Lanes past the last
/// code are unused. Their bits are 0 and their factors are `Factors::EMPTY`.
pub(crate) fn pack_block(codes: &[u8], factors: &[Factors], d_pad: usize, out: &mut [u8]) {
    let (n, cb) = (factors.len(), code_bytes(d_pad));
    assert!(n <= BLOCK && codes.len() == n * cb && out.len() == block_bytes(d_pad));

    out.fill(0);

    for (lane, code) in codes.chunks_exact(cb).enumerate() {
        let shift = if lane < 16 { 0 } else { 4 };
        for g in 0..d_pad / 4 {
            out[16 * g + lane % 16] |= group(code, g) << shift;
        }
    }

    for lane in 0..BLOCK {
        let f = factors.get(lane).copied().unwrap_or(Factors::EMPTY);
        for (k, v) in [f.f_add, f.f_rescale, f.f_pop, f.f_err].into_iter().enumerate() {
            write_f32(out, 4 * d_pad + 4 * (BLOCK * k + lane), v);
        }
    }
}

/// Reads one lane's code and factors back out of a block.
pub(crate) fn unpack_lane(block: &[u8], d_pad: usize, lane: usize, bits: &mut [u8]) -> Factors {
    assert!(lane < BLOCK && block.len() == block_bytes(d_pad) && bits.len() == code_bytes(d_pad));

    bits.fill(0);
    let shift = if lane < 16 { 0 } else { 4 };

    for g in 0..d_pad / 4 {
        bits[g / 2] |= ((block[16 * g + lane % 16] >> shift) & 0x0f) << (4 * (g % 2));
    }

    lane_factors(block, d_pad, lane)
}

/// The factors of one lane of a block.
pub(crate) fn lane_factors(block: &[u8], d_pad: usize, lane: usize) -> Factors {
    let read = |k: usize| read_f32(block, 4 * d_pad + 4 * (BLOCK * k + lane));
    Factors {
        f_add: read(0),
        f_rescale: read(1),
        f_pop: read(2),
        f_err: read(3),
    }
}

/// The sums s of all 32 lanes of a block for one query, by the scalar path.
pub(crate) fn block_sums(block: &[u8], lut: &[u8], d_pad: usize, out: &mut [u32; BLOCK]) {
    assert!(block.len() == block_bytes(d_pad) && lut.len() == 4 * d_pad);

    out.fill(0);

    for (bytes, entries) in block[..4 * d_pad]
        .as_chunks::<16>()
        .0
        .iter()
        .zip(lut.as_chunks::<16>().0)
    {
        for (i, &byte) in bytes.iter().enumerate() {
            out[i] += u32::from(entries[usize::from(byte & 0x0f)]);
            out[16 + i] += u32::from(entries[usize::from(byte >> 4)]);
        }
    }
}

#[cfg(test)]
#[path = "../../tests/unit/quantization/rabitq.rs"]
mod tests;
