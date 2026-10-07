//! The 8-bit copies used for rescoring. Let μ be the mean. A vector o is stored as its offset x′ = o − μ, with each
//! value an 8-bit integer c_i and one scale per vector, where
//!
//! ```text
//! scale = max_i |x′_i| / 127    c_i = clamp(round(x′_i / scale), −127, 127)    n2 = scale²·Σ c_i²
//! ```
//!
//! and n2 is the squared length of the decoded copy scale·c. A zero scale gives all-zero codes. For a query q, with
//! q′ = q − μ, the squared distance from the query to the decoded copy is
//!
//! ```text
//! ‖q′‖² + n2 − 2·scale·Σ q′_i c_i
//! ```
//!
//! A row holds the codes, then scale and n2 as f32, then the id given at build time as u64, padded to a multiple of 8
//! bytes. That is 784 bytes at 768 dimensions. Rows never cross a page, so a page holds its size divided by the row
//! size, rounded down. A cluster group's center uses the same row without the id, 776 bytes at 768 dimensions.

use super::{read_f32, round_up, write_f32};

/// The bytes in a row with an id.
pub(crate) const fn row_bytes(dim: usize) -> usize {
    round_up(dim + 16, 8)
}

/// The bytes in a cluster group center's row, which has no id.
pub(crate) const fn center_row_bytes(dim: usize) -> usize {
    round_up(dim + 8, 8)
}

/// The rows with an id that fit in one page of `page_size` bytes.
pub(crate) const fn rows_per_page(dim: usize, page_size: usize) -> usize {
    page_size / row_bytes(dim)
}

/// Encodes o − mean into the start of `row`, as the codes followed by scale and n2. The row needs at least dim + 8
/// bytes.
pub(crate) fn encode(o: &[f32], mean: &[f32], row: &mut [u8]) {
    let dim = o.len();
    assert!(mean.len() == dim && row.len() >= dim + 8);

    let max_abs = o.iter().zip(mean).fold(0f32, |m, (&x, &mu)| m.max((x - mu).abs()));
    let scale = max_abs / 127.0;
    let mut sum_sq = 0u64;

    for ((c, &x), &mu) in row[..dim].iter_mut().zip(o).zip(mean) {
        let code = if scale > 0.0 {
            ((x - mu) / scale).round().clamp(-127.0, 127.0) as i8
        } else {
            0
        };
        *c = code as u8;
        sum_sq += u64::from(code.unsigned_abs()).pow(2);
    }

    write_f32(row, dim, scale);
    write_f32(row, dim + 4, (f64::from(scale).powi(2) * sum_sq as f64) as f32);
}

/// Writes a full row of `row_bytes(dim)` bytes, with the codes, scale, n2 and the id.
pub(crate) fn write_row(o: &[f32], mean: &[f32], id: u64, row: &mut [u8]) {
    let dim = o.len();
    assert_eq!(row.len(), row_bytes(dim));

    row.fill(0);
    encode(o, mean, row);
    row[dim + 8..dim + 16].copy_from_slice(&id.to_le_bytes());
}

/// Writes a cluster group center's row of `center_row_bytes(dim)` bytes, with the codes, scale and n2 and zero
/// padding.
pub(crate) fn write_center_row(o: &[f32], mean: &[f32], row: &mut [u8]) {
    assert_eq!(row.len(), center_row_bytes(o.len()));

    row.fill(0);
    encode(o, mean, row);
}

/// Code i of a row.
pub(crate) fn code(row: &[u8], i: usize) -> i8 {
    row[i] as i8
}

pub(crate) fn scale(row: &[u8], dim: usize) -> f32 {
    read_f32(row, dim)
}

/// The squared length of the decoded copy.
pub(crate) fn norm2(row: &[u8], dim: usize) -> f32 {
    read_f32(row, dim + 4)
}

pub(crate) fn id(row: &[u8], dim: usize) -> u64 {
    let mut b = [0u8; 8];
    b.copy_from_slice(&row[dim + 8..dim + 16]);
    u64::from_le_bytes(b)
}

/// The squared distance from a query to a row's decoded copy. `q_off` is q − μ and `q_norm2` is ‖q − μ‖². The dot
/// product keeps 16 running sums, where sum k takes dimensions k, k + 16, k + 32 and so on, then adds them in halves.
/// A SIMD version can follow the same order and match it bit for bit.
pub(crate) fn distance(row: &[u8], q_off: &[f32], q_norm2: f32) -> f32 {
    let dim = q_off.len();
    let mut acc = [0f32; 16];

    for (qs, cs) in q_off.chunks(16).zip(row[..dim].chunks(16)) {
        for ((a, &q), &c) in acc.iter_mut().zip(qs).zip(cs) {
            *a += q * f32::from(c as i8);
        }
    }

    for half in [8, 4, 2, 1] {
        let (lo, hi) = acc.split_at_mut(half);
        for (a, &b) in lo.iter_mut().zip(&hi[..half]) {
            *a += b;
        }
    }

    q_norm2 + norm2(row, dim) - 2.0 * scale(row, dim) * acc[0]
}

#[cfg(test)]
#[path = "../../tests/unit/quantization/sq8.rs"]
mod tests;
