//! The 4-bit codes of the cluster centers, which a query scans to pick the clusters it reads. Let μ be the mean and P
//! the rotation. A center c is stored as its rotated offset from the mean, z = P(c − μ), with one minimum lo and one
//! step per center, where
//!
//! ```text
//! lo = min z    step = (max z − lo) / 15    k_i = round((z_i − lo) / step) ∈ 0..=15
//! ```
//!
//! and a zero step gives all-zero codes. Each center also keeps ‖z‖². The query is quantized to 8 bits, with one scale
//! for both signs, and each center's route score follows from one integer dot product. With T = ‖q − μ‖,
//!
//! ```text
//! ỹ = P(q − μ)    Δ_q = max_i |ỹ_i| / 127    q8_i = round(ỹ_i / Δ_q) ∈ −127..=127    S_q = Δ_q·Σ q8_i
//! route(c) = T² + ‖z‖² − 2(lo·S_q + step·Δ_q·Σ q8_i k_i) ≈ ‖q − c‖²
//! ```
//!
//! Only Σ q8_i k_i involves both sides, and it is an exact integer, so every scan path gives the same route scores.
//!
//! A center's codes decode to ẑ_i = lo + step·k_i, the rotated offset P(ĉ − μ) of the center ĉ that the index and every
//! search use. Scanning 1-bit codes measured from ĉ needs the query's exact distance to it. Since P keeps lengths, with
//! d the dimension,
//!
//! ```text
//! ‖q − ĉ‖² = T² + ‖ẑ‖² − 2⟨ỹ, ẑ⟩    ⟨ỹ, ẑ⟩ = lo·Σ ỹ_i + step·Σ k_i ỹ_i
//! ‖ẑ‖² = d·lo² + 2·lo·step·Σ k_i + step²·Σ k_i²
//! ```
//!
//! The stored ‖z‖² belongs to the center before its codes rounded it, so an index sums each center's codes and their
//! squares for ‖ẑ‖² once, when it builds or loads them. For Σ k_i ỹ_i the query takes 32-bit fixed point,
//! ŷ_i = 2^−s·Y_i with Y_i = round(2^s·ỹ_i), where s is the largest with 2^s·max_j |ỹ_j| ≤ 127·(1 + 256 + 256² + 256³)
//! = 2139062143. So every Y_i fits in 4 signed digits in base 256, and each ŷ_i is within max_j |ỹ_j| / 2139062143 of
//! ỹ_i, just over 2^−31·max_j |ỹ_j|. Σ k_i ŷ_i is then 4 integer dot products like the route's, one per digit, so every
//! path gives the same bits.
//!
//! A center's codes take d_pad/2 bytes, where d_pad is the dimension rounded up to a multiple of 64. They come in
//! chunks of 32 bytes per 64 dimensions. In chunk m, byte b < 16 holds dimension 64m + b in its low half and
//! 64m + 16 + b in its high half. Byte 16 + b holds dimensions 64m + 32 + b and 64m + 48 + b. So each 16-byte load
//! splits into two runs of 16 consecutive dimensions, which line up with the query.

use super::padded_dim;

/// The bytes of one center's codes.
pub(crate) const fn code_bytes(d_pad: usize) -> usize {
    d_pad / 2
}

/// The minimum, step and squared length stored with each center's codes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct CenterFactors {
    pub(crate) lo: f32,
    pub(crate) step: f32,
    pub(crate) norm2: f32,
}

/// Where dimension i's code sits, as its byte and the shift of its half.
fn position(i: usize) -> (usize, u32) {
    let (chunk, r) = (i / 64, i % 64);
    (32 * chunk + 16 * (r / 32) + r % 16, if r % 32 < 16 { 0 } else { 4 })
}

/// Encodes a rotated center offset `z` into `codes`, which has `code_bytes(d_pad)` bytes. The padding dimensions get
/// code 0.
pub(crate) fn encode(z: &[f32], codes: &mut [u8]) -> CenterFactors {
    assert_eq!(codes.len(), code_bytes(padded_dim(z.len())));

    let lo = z.iter().copied().fold(f32::INFINITY, f32::min);
    let hi = z.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let step = (hi - lo) / 15.0;
    codes.fill(0);

    for (i, &v) in z.iter().enumerate() {
        let k = if step > 0.0 {
            ((v - lo) / step).round().clamp(0.0, 15.0) as u8
        } else {
            0
        };
        let (byte, shift) = position(i);
        codes[byte] |= k << shift;
    }

    let norm2 = z.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>() as f32;
    CenterFactors { lo, step, norm2 }
}

/// Dimension i's code.
pub(crate) fn code_at(codes: &[u8], i: usize) -> u8 {
    let (byte, shift) = position(i);
    (codes[byte] >> shift) & 0x0f
}

/// Writes the rotated offset that `codes` and `f` decode to, ẑ_i = lo + step·k_i, for each of the `out.len()`
/// dimensions, rounded to f32.
pub(crate) fn decode(codes: &[u8], f: &CenterFactors, out: &mut [f32]) {
    let (lo, step) = (f64::from(f.lo), f64::from(f.step));

    for (i, z) in out.iter_mut().enumerate() {
        *z = (lo + step * f64::from(code_at(codes, i))) as f32;
    }
}

/// The squared length ‖ẑ‖² of what a center's codes `codes` and factors `f` decode to, over `dim` dimensions.
pub(crate) fn decoded_norm2(codes: &[u8], f: &CenterFactors, dim: usize) -> f64 {
    let halves = |b: u8| (u32::from(b & 0x0f), u32::from(b >> 4));
    // Two passes, since the compiler turns each one into single instruction, multiple data (SIMD) code, but not a pass
    // that does both.
    let k1: u32 = codes.iter().map(|&b| halves(b).0 + halves(b).1).sum();
    let k2: u32 = codes.iter().map(|&b| halves(b).0.pow(2) + halves(b).1.pow(2)).sum();
    let (lo, step, d) = (f64::from(f.lo), f64::from(f.step), dim as f64);
    d * lo * lo + 2.0 * lo * step * f64::from(k1) + step * step * f64::from(k2)
}

/// `decoded_norm2` of each center of `dim` dimensions, whose codes `codes` holds back to back.
pub(crate) fn decoded_norms(codes: &[u8], factors: &[CenterFactors], dim: usize) -> Vec<f64> {
    let cb = code_bytes(padded_dim(dim));
    let centers = codes.chunks_exact(cb).zip(factors);
    centers.map(|(codes, f)| decoded_norm2(codes, f, dim)).collect()
}

/// A query prepared for scanning the center codes. It is sized once, so preparing a query allocates nothing.
pub(crate) struct QueryS8 {
    /// q8_i for each padded dimension. The padding dimensions are 0.
    pub(crate) q8: Vec<i8>,
    /// Δ_q.
    pub(crate) delta: f32,
    /// S_q.
    pub(crate) s_q: f32,
}

impl QueryS8 {
    pub(crate) fn new(dim: usize) -> QueryS8 {
        QueryS8 {
            q8: vec![0; padded_dim(dim)],
            delta: 0.0,
            s_q: 0.0,
        }
    }

    /// Quantizes `y_rot` = P(q − μ).
    pub(crate) fn prepare(&mut self, y_rot: &[f32]) {
        assert_eq!(self.q8.len(), padded_dim(y_rot.len()));

        let delta = y_rot.iter().fold(0f32, |m, &v| m.max(v.abs())) / 127.0;
        self.q8.fill(0);
        let mut sum = 0i64;

        if delta > 0.0 {
            for (q, &v) in self.q8.iter_mut().zip(y_rot) {
                *q = (v / delta).round().clamp(-127.0, 127.0) as i8;
                sum += i64::from(*q);
            }
        }

        self.delta = delta;
        self.s_q = (f64::from(delta) * sum as f64) as f32;
    }
}

/// The bytes of a query's fixed point. Each Y_i is Σ_t 256^t·d_t,i with every digit d_t,i in −128..=127.
pub(crate) const PLANES: usize = 4;

/// The largest |Y_i| that the digits hold, 127·(1 + 256 + 256² + 256³).
const FIXED_MAX: f64 = 2_139_062_143.0;

/// A query prepared for measuring its distance to the centers that the codes decode to, in 32-bit fixed point. It is
/// sized once, so preparing a query allocates nothing.
pub(crate) struct QueryS32 {
    dim: usize,
    /// The digits of each Y_i. Plane t holds digit t of every padded dimension, and the padding dimensions are 0.
    planes: Vec<i8>,
    /// 2^−s, the value of one step of Y.
    unit: f64,
    /// Σ ŷ_i.
    sum: f64,
    /// T².
    t2: f64,
}

impl QueryS32 {
    pub(crate) fn new(dim: usize) -> QueryS32 {
        QueryS32 {
            dim,
            planes: vec![0; PLANES * padded_dim(dim)],
            unit: 1.0,
            sum: 0.0,
            t2: 0.0,
        }
    }

    /// Prepares the query from `y_rot` = ỹ and `t2` = T².
    pub(crate) fn prepare(&mut self, y_rot: &[f32], t2: f64) {
        assert_eq!(y_rot.len(), self.dim);

        let d_pad = padded_dim(self.dim);
        let top = y_rot.iter().fold(0f64, |m, &v| m.max(f64::from(v).abs()));
        // The largest s with 2^s·top at most FIXED_MAX. A zero query can take any s.
        let mut s = if top > 0.0 {
            (FIXED_MAX / top).log2().floor() as i32
        } else {
            0
        };

        while top * 2f64.powi(s) > FIXED_MAX {
            s -= 1;
        }

        let (scale, mut sum) = (2f64.powi(s), 0i64);

        for (i, &v) in y_rot.iter().enumerate() {
            let mut y = (f64::from(v) * scale).round() as i64;
            sum += y;
            for t in 0..PLANES {
                let digit = (y + 128).rem_euclid(256) - 128;
                self.planes[t * d_pad + i] = digit as i8;
                y = (y - digit) / 256;
            }
        }

        self.unit = 2f64.powi(-s);
        self.sum = sum as f64 * self.unit;
        self.t2 = t2;
    }

    /// The planes back to back, so plane t, digit t of each Y_i, is the t-th run of d_pad values.
    pub(crate) fn planes(&self) -> &[i8] {
        &self.planes
    }

    /// ‖q − ĉ‖² for the center ĉ that a cluster's codes decode to, at least 0. `f` is the center's factors, `norm2` its
    /// `decoded_norm2` ‖ẑ‖², and `dots[t]` is Σ k_i·d_t,i, the dot product of its codes with plane t.
    pub(crate) fn distance2(&self, f: &CenterFactors, norm2: f64, dots: &[i32; PLANES]) -> f64 {
        let dot = dots.iter().rev().fold(0i64, |sum, &d| 256 * sum + i64::from(d));
        let ip = f64::from(f.lo) * self.sum + f64::from(f.step) * (dot as f64 * self.unit);
        (self.t2 + norm2 - 2.0 * ip).max(0.0)
    }
}

/// The dot product Σ q8_i k_i for each center, by the scalar path. `codes` holds `out.len()` centers back to back.
pub(crate) fn dots_scalar(codes: &[u8], q8: &[i8], out: &mut [i32]) {
    let cb = q8.len() / 2;
    assert_eq!(out.len().checked_mul(cb), Some(codes.len()));

    for (dot, center) in out.iter_mut().zip(codes.chunks_exact(cb)) {
        let mut s = 0i32;
        for (chunk, q) in center.as_chunks::<32>().0.iter().zip(q8.as_chunks::<64>().0) {
            for (half, bytes) in chunk.as_chunks::<16>().0.iter().enumerate() {
                let (q_low, q_high) = q[32 * half..32 * half + 32].split_at(16);
                for ((&byte, &a), &b) in bytes.iter().zip(q_low).zip(q_high) {
                    s += i32::from(byte & 0x0f) * i32::from(a) + i32::from(byte >> 4) * i32::from(b);
                }
            }
        }
        *dot = s;
    }
}

/// A center's route score from its dot product Σ q8_i k_i. `t2` is T².
#[inline]
pub(crate) fn route_score(t2: f32, f: &CenterFactors, q: &QueryS8, dot: i32) -> f32 {
    t2 + f.norm2 - 2.0 * (f.lo * q.s_q + f.step * q.delta * dot as f32)
}

#[cfg(test)]
#[path = "../../tests/unit/quantization/sq4.rs"]
mod tests;
