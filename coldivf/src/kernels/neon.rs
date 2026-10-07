//! Kernels for NEON, the single instruction, multiple data (SIMD) instructions of 64-bit Arm CPUs. They also use the
//! dot-product instructions, which every Apple silicon chip has.

use std::arch::aarch64::*;

/// The dot product Σ q8_i k_i for each cluster center, where q8_i is `q8[i]` and k_i is the center's 4-bit code for
/// dimension i. It scores four centers per pass, so they share each load of the query. Each center keeps two running
/// sums, one for the low halves of its code bytes and one for the high halves, so its dot-product instructions don't
/// wait on each other.
///
/// # Safety
///
/// The CPU must support NEON and dotprod. `q8.len()` must be a multiple of 64, and `codes` must hold `out.len()`
/// centers of `q8.len() / 2` bytes each.
#[target_feature(enable = "neon,dotprod")]
pub(crate) unsafe fn sq4_dots(codes: &[u8], q8: &[i8], out: &mut [i32]) {
    let cb = q8.len() / 2;
    let n = out.len();
    let mut c = 0;

    while c + 4 <= n {
        let mut acc = [[vdupq_n_s32(0); 2]; 4];
        for chunk in 0..cb / 32 {
            // SAFETY: 64·chunk + 64 ≤ q8.len() and (c + 3)·cb + 32·chunk + 32 ≤ codes.len(), by the caller's lengths.
            unsafe {
                let q = load_query(q8.as_ptr().add(64 * chunk));
                for (j, a) in acc.iter_mut().enumerate() {
                    add_chunk(a, codes.as_ptr().add((c + j) * cb + 32 * chunk), q);
                }
            }
        }
        for (j, a) in acc.iter().enumerate() {
            out[c + j] = vaddvq_s32(vaddq_s32(a[0], a[1]));
        }
        c += 4;
    }

    while c < n {
        let mut acc = [vdupq_n_s32(0); 2];
        for chunk in 0..cb / 32 {
            // SAFETY: the same bounds as above, for one center.
            unsafe {
                add_chunk(
                    &mut acc,
                    codes.as_ptr().add(c * cb + 32 * chunk),
                    load_query(q8.as_ptr().add(64 * chunk)),
                );
            }
        }
        out[c] = vaddvq_s32(vaddq_s32(acc[0], acc[1]));
        c += 1;
    }
}

/// Loads 64 query values, the 4 runs of 16 that one 32-byte chunk of codes lines up with.
///
/// # Safety
///
/// `q` must point to 64 readable values.
#[inline]
#[target_feature(enable = "neon,dotprod")]
unsafe fn load_query(q: *const i8) -> [int8x16_t; 4] {
    // SAFETY: the caller guarantees 64 readable values.
    unsafe {
        [
            vld1q_s8(q),
            vld1q_s8(q.add(16)),
            vld1q_s8(q.add(32)),
            vld1q_s8(q.add(48)),
        ]
    }
}

/// Adds one 32-byte chunk of a center's codes, 64 dimensions, to its two running sums. The low halves go to `acc[0]`
/// and the high halves to `acc[1]`.
///
/// # Safety
///
/// `p` must point to 32 readable bytes.
#[inline]
#[target_feature(enable = "neon,dotprod")]
unsafe fn add_chunk(acc: &mut [int32x4_t; 2], p: *const u8, q: [int8x16_t; 4]) {
    // SAFETY: the caller guarantees 32 readable bytes.
    let (b0, b1) = unsafe { (vld1q_u8(p), vld1q_u8(p.add(16))) };
    let mask = vdupq_n_u8(0x0f);

    acc[0] = vdotq_s32(acc[0], vreinterpretq_s8_u8(vandq_u8(b0, mask)), q[0]);
    acc[1] = vdotq_s32(acc[1], vreinterpretq_s8_u8(vshrq_n_u8::<4>(b0)), q[1]);
    acc[0] = vdotq_s32(acc[0], vreinterpretq_s8_u8(vandq_u8(b1, mask)), q[2]);
    acc[1] = vdotq_s32(acc[1], vreinterpretq_s8_u8(vshrq_n_u8::<4>(b1)), q[3]);
}
