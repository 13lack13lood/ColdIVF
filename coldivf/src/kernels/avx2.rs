//! Kernels for the Advanced Vector Extensions 2 (AVX2), the 256-bit single instruction, multiple data (SIMD)
//! instructions of x86 CPUs, with fused multiply-add (FMA). Each kernel gives the same bits as the scalar code, except
//! that a NaN score may carry another NaN's bits.

use std::arch::x86_64::*;

use super::{first_lanes, lower_ids};
use crate::quantization::rabitq::{BLOCK, QueryLut};
use crate::search::candidates::{Candidate, rank};

/// How many chunks of 64 dimensions a 16-bit sum adds up before it is widened to 32 bits. Each chunk adds 4 products
/// k·q to every 16-bit sum, with k at most 15 and |q| at most 128, so 4 chunks add at most 16·15·128 = 30720 in size,
/// below 2^15.
const RUN: usize = 4;

/// The dot product Σ q8_i k_i for each cluster center, where q8_i is `q8[i]` and k_i is the center's 4-bit code for
/// dimension i. It scores four centers per pass, so they share each load of the query. Each center adds `RUN` chunks
/// into 16-bit sums before it widens them to 32 bits.
///
/// # Safety
///
/// The CPU must support AVX2 and FMA. `q8.len()` must be a multiple of 64, and `codes` must hold `out.len()` centers
/// of `q8.len() / 2` bytes each.
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn sq4_dots(codes: &[u8], q8: &[i8], out: &mut [i32]) {
    let (cb, n) = (q8.len() / 2, out.len());
    let chunks = cb / 32;
    let mut c = 0;

    while c + 4 <= n {
        let mut sums = [_mm256_setzero_si256(); 4];
        for first in (0..chunks).step_by(RUN) {
            let mut run = [_mm256_setzero_si256(); 4];
            for chunk in first..chunks.min(first + RUN) {
                // SAFETY: 64·chunk + 64 ≤ q8.len() and (c + 3)·cb + 32·chunk + 32 ≤ codes.len(), by the caller's
                // lengths.
                unsafe {
                    let q = query_runs(q8.as_ptr().add(64 * chunk));
                    for (j, r) in run.iter_mut().enumerate() {
                        *r = _mm256_add_epi16(*r, chunk_products(codes.as_ptr().add((c + j) * cb + 32 * chunk), q));
                    }
                }
            }
            for (s, &r) in sums.iter_mut().zip(&run) {
                *s = _mm256_add_epi32(*s, widen(r));
            }
        }
        for (j, &s) in sums.iter().enumerate() {
            out[c + j] = total(s);
        }
        c += 4;
    }

    while c < n {
        let mut sum = _mm256_setzero_si256();
        for first in (0..chunks).step_by(RUN) {
            let mut run = _mm256_setzero_si256();
            for chunk in first..chunks.min(first + RUN) {
                // SAFETY: the same bounds as above, for one center.
                unsafe {
                    let q = query_runs(q8.as_ptr().add(64 * chunk));
                    run = _mm256_add_epi16(run, chunk_products(codes.as_ptr().add(c * cb + 32 * chunk), q));
                }
            }
            sum = _mm256_add_epi32(sum, widen(run));
        }
        out[c] = total(sum);
        c += 1;
    }
}

/// The dot product of one center's 4-bit codes with each of 4 query planes, the same as `sq4_dots` gives for that
/// center and each plane. It goes through the codes once, a chunk at a time for all 4 planes, and puts each chunk's
/// codes in the query's order once for all of them.
///
/// # Safety
///
/// The CPU must support AVX2 and FMA. `codes.len()` must be a multiple of 32, and `planes` must hold 4 planes of
/// `2 * codes.len()` values each, back to back.
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn plane_dots(codes: &[u8], planes: &[i8], out: &mut [i32; 4]) {
    let (d_pad, chunks) = (2 * codes.len(), codes.len() / 32);
    let mut sums = [_mm256_setzero_si256(); 4];

    for first in (0..chunks).step_by(RUN) {
        let mut run = [_mm256_setzero_si256(); 4];
        for chunk in first..chunks.min(first + RUN) {
            // SAFETY: 32·chunk + 32 ≤ codes.len() and t·d_pad + 64·chunk + 64 ≤ planes.len(), by the caller's lengths.
            unsafe {
                let k = ordered_codes(codes.as_ptr().add(32 * chunk));
                for (t, r) in run.iter_mut().enumerate() {
                    let q = planes.as_ptr().add(t * d_pad + 64 * chunk);
                    let (q0, q1) = (_mm256_loadu_si256(q.cast()), _mm256_loadu_si256(q.add(32).cast()));
                    let products = _mm256_add_epi16(_mm256_maddubs_epi16(k[0], q0), _mm256_maddubs_epi16(k[1], q1));
                    *r = _mm256_add_epi16(*r, products);
                }
            }
        }
        for (s, &r) in sums.iter_mut().zip(&run) {
            *s = _mm256_add_epi32(*s, widen(r));
        }
    }

    for (dot, &s) in out.iter_mut().zip(&sums) {
        *dot = total(s);
    }
}

/// Loads 64 query values, the ones that one 32-byte chunk of codes lines up with, in the order of the chunk's low
/// halves and then its high halves. The low halves hold dimensions 0 to 15 and 32 to 47, and the high halves the 16
/// after each.
///
/// # Safety
///
/// `q` must point to 64 readable values.
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn query_runs(q: *const i8) -> [__m256i; 2] {
    // SAFETY: the caller guarantees 64 readable values.
    let (q0, q1) = unsafe { (_mm256_loadu_si256(q.cast()), _mm256_loadu_si256(q.add(32).cast())) };

    [
        _mm256_permute2x128_si256::<0x20>(q0, q1),
        _mm256_permute2x128_si256::<0x31>(q0, q1),
    ]
}

/// One 32-byte chunk of a center's codes, 64 dimensions, multiplied by the query values from `query_runs`. Each 16-bit
/// value is the sum of 4 products. The codes go in as unsigned bytes and the query as signed ones, and a pair of
/// products is at most 2·15·128 = 3840 in size, so no sum saturates.
///
/// # Safety
///
/// `p` must point to 32 readable bytes.
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn chunk_products(p: *const u8, q: [__m256i; 2]) -> __m256i {
    // SAFETY: the caller guarantees 32 readable bytes.
    let b = unsafe { _mm256_loadu_si256(p.cast()) };
    let mask = _mm256_set1_epi8(0x0f);
    let (low, high) = (
        _mm256_and_si256(b, mask),
        _mm256_and_si256(_mm256_srli_epi16::<4>(b), mask),
    );

    _mm256_add_epi16(_mm256_maddubs_epi16(low, q[0]), _mm256_maddubs_epi16(high, q[1]))
}

/// One 32-byte chunk of a center's codes as bytes in the order of its dimensions, 0 to 31 and then 32 to 63, so they
/// line up with the query as it is stored.
///
/// # Safety
///
/// `p` must point to 32 readable bytes.
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn ordered_codes(p: *const u8) -> [__m256i; 2] {
    // SAFETY: the caller guarantees 32 readable bytes.
    let b = unsafe { _mm256_loadu_si256(p.cast()) };
    let mask = _mm256_set1_epi8(0x0f);
    // The low halves hold dimensions 0 to 15 and 32 to 47, and the high halves the 16 after each.
    let (low, high) = (
        _mm256_and_si256(b, mask),
        _mm256_and_si256(_mm256_srli_epi16::<4>(b), mask),
    );

    [
        _mm256_permute2x128_si256::<0x20>(low, high),
        _mm256_permute2x128_si256::<0x31>(low, high),
    ]
}

/// Adds each pair of 16-bit sums into one 32-bit sum.
#[inline]
#[target_feature(enable = "avx2,fma")]
fn widen(run: __m256i) -> __m256i {
    _mm256_madd_epi16(run, _mm256_set1_epi16(1))
}

/// The sum of the 8 32-bit values.
#[inline]
#[target_feature(enable = "avx2,fma")]
fn total(v: __m256i) -> i32 {
    let s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256::<1>(v));
    let s = _mm_add_epi32(s, _mm_unpackhi_epi64(s, s));
    let s = _mm_add_epi32(s, _mm_shuffle_epi32::<0b01>(s));
    _mm_cvtsi128_si32(s)
}

/// Scores one block of 1-bit codes like `scan_block_scalar` does, and writes the same candidates bit for bit, except
/// that a NaN score may carry another NaN's bits. Each estimate uses the same three fused multiply-adds in the same
/// order, and adds the query's t2, or lane j's `t2[j]` when `t2` is given. It ranks the estimates against `bound` as
/// `before` does, with NaN as +∞.
///
/// # Safety
///
/// The CPU must support AVX2 and FMA. `query.lut` must hold 16 entries per group of 4 dimensions for a padded dimension
/// that is a multiple of 64, `block` must hold one block for that dimension, and `count` must be at most 32.
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn scan_block(
    block: &[u8],
    query: &QueryLut,
    t2: Option<&[f32; BLOCK]>,
    first_id: u32,
    count: usize,
    bound: Candidate,
    out: &mut [Candidate; BLOCK],
) -> usize {
    let d_pad = query.lut.len() / 4;
    // SAFETY: the block holds d_pad/4 groups of 16 code bytes and the table 16 entries per group, by the caller's
    // lengths, and d_pad/4 is a multiple of 16.
    let sums = unsafe { block_sums(block.as_ptr(), query.lut.as_ptr(), d_pad / 4) };

    let k = &query.k;
    let (k_s, k_p, k_c, t2_all) = (
        _mm256_set1_ps(k.k_s),
        _mm256_set1_ps(k.k_p),
        _mm256_set1_ps(k.k_c),
        _mm256_set1_ps(k.t2),
    );
    let (bound_score, infinity) = (_mm256_set1_ps(rank(bound.score)), _mm256_set1_ps(f32::INFINITY));
    let mut n = 0;

    for (lane, &s) in (0..count).step_by(8).zip(&sums) {
        // SAFETY: lanes `lane..lane + 8` are inside the 32 values of each factor, which start at byte 4·d_pad of the
        // block.
        let (f_add, f_rescale, f_pop) = unsafe {
            let p = block.as_ptr().add(4 * d_pad + 4 * lane).cast::<f32>();
            (
                _mm256_loadu_ps(p),
                _mm256_loadu_ps(p.add(BLOCK)),
                _mm256_loadu_ps(p.add(2 * BLOCK)),
            )
        };
        let t2_lanes = match t2 {
            // SAFETY: lanes `lane..lane + 8` are inside the 32 values.
            Some(t2) => unsafe { _mm256_loadu_ps(t2.as_ptr().add(lane)) },
            None => t2_all,
        };
        let s = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(s));
        let a = _mm256_fmadd_ps(k_s, s, k_c);
        let ip = _mm256_fmadd_ps(k_p, f_pop, a);
        let est = _mm256_fmadd_ps(f_rescale, ip, _mm256_add_ps(f_add, t2_lanes));
        // The estimates as they rank, with NaN as +∞.
        let x = _mm256_blendv_ps(est, infinity, _mm256_cmp_ps::<_CMP_UNORD_Q>(est, est));
        let below = _mm256_movemask_ps(_mm256_cmp_ps::<_CMP_LT_OQ>(x, bound_score)) as u32;
        let equal = _mm256_movemask_ps(_mm256_cmp_ps::<_CMP_EQ_OQ>(x, bound_score)) as u32;
        let mut passed = (below | (equal & lower_ids(bound, first_id + lane as u32))) & first_lanes(count - lane);

        if passed == 0 {
            continue;
        }

        let mut scores = [0f32; 8];
        // SAFETY: the array holds 8 values.
        unsafe { _mm256_storeu_ps(scores.as_mut_ptr(), est) };
        while passed != 0 {
            let j = passed.trailing_zeros() as usize;
            out[n] = Candidate {
                score: scores[j],
                id: first_id + (lane + j) as u32,
            };
            n += 1;
            passed &= passed - 1;
        }
    }

    n
}

/// The 32 lookup sums of one block, as four vectors of eight 16-bit sums, for lanes 0 to 7, 8 to 15, 16 to 23 and 24
/// to 31. Each 256-bit load covers two groups, one per 128-bit half, since a byte shuffle looks up within its half.
/// Each pass adds four loads in bytes, which can't overflow since an entry is at most 60, then adds the bytes into
/// 16-bit totals.
///
/// # Safety
///
/// `codes` must point to `groups` groups of 16 bytes and `lut` to `groups` tables of 16 entries. `groups` must be a
/// multiple of 8.
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn block_sums(codes: *const u8, lut: *const u8, groups: usize) -> [__m128i; 4] {
    let mask = _mm256_set1_epi8(0x0f);
    // Lanes 0 to 15 go to acc[0] and acc[1], lanes 16 to 31 to acc[2] and acc[3]. The first of each pair adds the bytes
    // as 16-bit values, even lane + 256·odd lane, and the second adds only the odd lanes.
    let mut acc = [_mm256_setzero_si256(); 4];

    for g in (0..groups).step_by(8) {
        // SAFETY: groups g to g + 7 are below `groups`.
        let ((l0, h0), (l1, h1), (l2, h2), (l3, h3)) = unsafe {
            (
                lookup(codes, lut, g, mask),
                lookup(codes, lut, g + 2, mask),
                lookup(codes, lut, g + 4, mask),
                lookup(codes, lut, g + 6, mask),
            )
        };
        let low = _mm256_add_epi8(_mm256_add_epi8(l0, l1), _mm256_add_epi8(l2, l3));
        let high = _mm256_add_epi8(_mm256_add_epi8(h0, h1), _mm256_add_epi8(h2, h3));
        acc[0] = _mm256_add_epi16(acc[0], low);
        acc[1] = _mm256_add_epi16(acc[1], _mm256_srli_epi16::<8>(low));
        acc[2] = _mm256_add_epi16(acc[2], high);
        acc[3] = _mm256_add_epi16(acc[3], _mm256_srli_epi16::<8>(high));
    }

    let (s0, s1) = lane_totals(acc[0], acc[1]);
    let (s2, s3) = lane_totals(acc[2], acc[3]);
    [s0, s1, s2, s3]
}

/// The totals of 16 lanes from one pair of running sums, in lane order, 8 per vector. `both` holds even lane + 256·odd
/// lane in each 16-bit value and `odd` the odd lane, so the even lanes' totals are `both` − 256·`odd`. This holds even
/// when `both` wraps, since every total is below 2^16. Each 128-bit half holds the part from every other group.
#[inline]
#[target_feature(enable = "avx2,fma")]
fn lane_totals(both: __m256i, odd: __m256i) -> (__m128i, __m128i) {
    let even = _mm256_sub_epi16(both, _mm256_slli_epi16::<8>(odd));
    let even = _mm_add_epi16(_mm256_castsi256_si128(even), _mm256_extracti128_si256::<1>(even));
    let odd = _mm_add_epi16(_mm256_castsi256_si128(odd), _mm256_extracti128_si256::<1>(odd));

    (_mm_unpacklo_epi16(even, odd), _mm_unpackhi_epi16(even, odd))
}

/// Looks up groups g and g + 1 for all 32 lanes, one group per 128-bit half. Lanes 0 to 15 come from the low halves of
/// the code bytes, and lanes 16 to 31 from the high halves.
///
/// # Safety
///
/// Groups g and g + 1 must be in range for `codes` and `lut`.
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn lookup(codes: *const u8, lut: *const u8, g: usize, mask: __m256i) -> (__m256i, __m256i) {
    // SAFETY: the caller guarantees groups g and g + 1 are in range.
    let (c, l) = unsafe {
        (
            _mm256_loadu_si256(codes.add(16 * g).cast()),
            _mm256_loadu_si256(lut.add(16 * g).cast()),
        )
    };

    (
        _mm256_shuffle_epi8(l, _mm256_and_si256(c, mask)),
        _mm256_shuffle_epi8(l, _mm256_and_si256(_mm256_srli_epi16::<4>(c), mask)),
    )
}
