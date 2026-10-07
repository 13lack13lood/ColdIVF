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
