//! Kernels for NEON, the single instruction, multiple data (SIMD) instructions of 64-bit Arm CPUs. They also use the
//! dot-product instructions, which every Apple silicon chip has.

use std::arch::aarch64::*;

use crate::quantization::rabitq::{BLOCK, QueryLut};
use crate::search::candidates::Candidate;

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

/// Scores one block of 1-bit codes like `scan_block_scalar` does, and writes the same candidates bit for bit. Each
/// estimate uses the same three fused multiply-adds in the same order.
///
/// # Safety
///
/// The CPU must support NEON and dotprod. `query.lut` must hold 16 entries per group of 4 dimensions for a padded
/// dimension that is a multiple of 64, `block` must hold one block for that dimension, and `count` must be at most 32.
#[target_feature(enable = "neon,dotprod")]
pub(crate) unsafe fn scan_block(
    block: &[u8],
    query: &QueryLut,
    first_id: u32,
    count: usize,
    bound: Candidate,
    out: &mut [Candidate; BLOCK],
) -> usize {
    let d_pad = query.lut.len() / 4;
    let mut sums = [0u16; BLOCK];
    // SAFETY: the block holds d_pad/4 groups of 16 code bytes and the table 16 entries per group, by the caller's
    // lengths.
    unsafe { block_sums(block.as_ptr(), query.lut.as_ptr(), d_pad / 4, &mut sums) };

    let k = &query.k;
    let (k_s, k_p, k_c, t2) = (
        vdupq_n_f32(k.k_s),
        vdupq_n_f32(k.k_p),
        vdupq_n_f32(k.k_c),
        vdupq_n_f32(k.t2),
    );
    let (bound_score, bound_id) = (vdupq_n_f32(bound.score), vdupq_n_u32(bound.id));
    // SAFETY: the array holds 4 values.
    let offsets = unsafe { vld1q_u32([0u32, 1, 2, 3].as_ptr()) };
    let mut n = 0;

    for lane in (0..count).step_by(4) {
        // SAFETY: lanes `lane..lane + 4` are inside the 32 sums and the 32 values of each factor, which start at byte
        // 4·d_pad of the block.
        let (s, f_add, f_rescale, f_pop) = unsafe {
            let p = block.as_ptr().add(4 * d_pad + 4 * lane);
            (
                vcvtq_f32_u32(vmovl_u16(vld1_u16(sums.as_ptr().add(lane)))),
                vld1q_f32(p.cast()),
                vld1q_f32(p.add(4 * BLOCK).cast()),
                vld1q_f32(p.add(8 * BLOCK).cast()),
            )
        };
        let a = vfmaq_f32(k_c, k_s, s);
        let ip = vfmaq_f32(a, k_p, f_pop);
        let est = vfmaq_f32(vaddq_f32(f_add, t2), f_rescale, ip);
        let ids = vaddq_u32(vdupq_n_u32(first_id.wrapping_add(lane as u32)), offsets);
        let ties = vandq_u32(vceqq_f32(est, bound_score), vcltq_u32(ids, bound_id));
        let pass = vorrq_u32(vcltq_f32(est, bound_score), ties);
        if vmaxvq_u32(pass) == 0 {
            continue;
        }
        let (mut scores, mut passed) = ([0f32; 4], [0u32; 4]);
        // SAFETY: both arrays hold 4 values.
        unsafe {
            vst1q_f32(scores.as_mut_ptr(), est);
            vst1q_u32(passed.as_mut_ptr(), pass);
        }
        for (j, (&score, &p)) in scores.iter().zip(&passed).enumerate().take(count - lane) {
            if p != 0 {
                out[n] = Candidate {
                    score,
                    id: first_id + (lane + j) as u32,
                };
                n += 1;
            }
        }
    }

    n
}

/// The 32 lookup sums of one block. Each pass adds four groups in bytes, which can't overflow since an entry is at
/// most 60, then widens them into 16-bit totals.
///
/// # Safety
///
/// `codes` must point to `groups` groups of 16 bytes and `lut` to `groups` tables of 16 entries. `groups` must be a
/// multiple of 4.
#[inline]
#[target_feature(enable = "neon,dotprod")]
unsafe fn block_sums(codes: *const u8, lut: *const u8, groups: usize, out: &mut [u16; BLOCK]) {
    let mask = vdupq_n_u8(0x0f);
    let mut acc = [vdupq_n_u16(0); 4];

    for g in (0..groups).step_by(4) {
        // SAFETY: groups g to g + 3 are below `groups`.
        let ((l0, h0), (l1, h1), (l2, h2), (l3, h3)) = unsafe {
            (
                lookup(codes, lut, g, mask),
                lookup(codes, lut, g + 1, mask),
                lookup(codes, lut, g + 2, mask),
                lookup(codes, lut, g + 3, mask),
            )
        };
        let low = vaddq_u8(vaddq_u8(l0, l1), vaddq_u8(l2, l3));
        let high = vaddq_u8(vaddq_u8(h0, h1), vaddq_u8(h2, h3));
        acc[0] = vaddw_u8(acc[0], vget_low_u8(low));
        acc[1] = vaddw_high_u8(acc[1], low);
        acc[2] = vaddw_u8(acc[2], vget_low_u8(high));
        acc[3] = vaddw_high_u8(acc[3], high);
    }

    // SAFETY: `out` holds 32 values, 8 per running sum.
    unsafe {
        for (k, a) in acc.iter().enumerate() {
            vst1q_u16(out.as_mut_ptr().add(8 * k), *a);
        }
    }
}

/// Looks up group g's entries for all 32 lanes. Lanes 0 to 15 come from the low halves of the code bytes, and lanes 16
/// to 31 from the high halves.
///
/// # Safety
///
/// Group g must be in range for `codes` and `lut`.
#[inline]
#[target_feature(enable = "neon,dotprod")]
unsafe fn lookup(codes: *const u8, lut: *const u8, g: usize, mask: uint8x16_t) -> (uint8x16_t, uint8x16_t) {
    // SAFETY: the caller guarantees group g is in range.
    let (c, l) = unsafe { (vld1q_u8(codes.add(16 * g)), vld1q_u8(lut.add(16 * g))) };

    (vqtbl1q_u8(l, vandq_u8(c, mask)), vqtbl1q_u8(l, vshrq_n_u8::<4>(c)))
}
