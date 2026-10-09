//! Kernels for AVX-512, the 512-bit single instruction, multiple data (SIMD) instructions of x86 CPUs. They use its
//! Foundation (F), Byte and Word (BW), Vector Length (VL) and Vector Neural Network Instructions (VNNI) parts, and the
//! Advanced Vector Extensions 2 (AVX2) and fused multiply-add (FMA) instructions that every such CPU has. Each kernel
//! gives the same bits as the scalar code, except that a NaN score may carry another NaN's bits.

use std::arch::x86_64::*;

use super::{first_lanes, lower_ids};
use crate::quantization::rabitq::{BLOCK, QueryLut};
use crate::search::candidates::{Candidate, rank};

/// The dot product Σ q8_i k_i for each cluster center, where q8_i is `q8[i]` and k_i is the center's 4-bit code for
/// dimension i. It scores four centers per pass, so they share each load of the query, and takes two chunks of 64
/// dimensions per step. The dot-product instruction adds 4 products of an unsigned code and a signed query value into
/// each 32-bit sum, without saturating.
///
/// # Safety
///
/// The CPU must support AVX-512 F, BW, VL and VNNI, AVX2 and FMA. `q8.len()` must be a multiple of 64, and `codes` must
/// hold `out.len()` centers of `q8.len() / 2` bytes each.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni,avx2,fma")]
pub(crate) unsafe fn sq4_dots(codes: &[u8], q8: &[i8], out: &mut [i32]) {
    let (cb, n) = (q8.len() / 2, out.len());
    let mut c = 0;

    while c + 4 <= n {
        let mut acc = [[_mm512_setzero_si512(); 2]; 4];
        for start in (0..cb).step_by(64) {
            let two = start + 64 <= cb;
            // SAFETY: 2·start + 64 ≤ q8.len() and (c + 3)·cb + start + 32 ≤ codes.len(), by the caller's lengths,
            // plus 64 more values and 32 more codes when `two` holds.
            unsafe {
                let q = query_runs(q8.as_ptr().add(2 * start), two);
                for (j, a) in acc.iter_mut().enumerate() {
                    add_codes(a, codes.as_ptr().add((c + j) * cb + start), two, q);
                }
            }
        }
        for (j, a) in acc.iter().enumerate() {
            out[c + j] = _mm512_reduce_add_epi32(_mm512_add_epi32(a[0], a[1]));
        }
        c += 4;
    }

    while c < n {
        let mut acc = [_mm512_setzero_si512(); 2];
        for start in (0..cb).step_by(64) {
            let two = start + 64 <= cb;
            // SAFETY: the same bounds as above, for one center.
            unsafe {
                let q = query_runs(q8.as_ptr().add(2 * start), two);
                add_codes(&mut acc, codes.as_ptr().add(c * cb + start), two, q);
            }
        }
        out[c] = _mm512_reduce_add_epi32(_mm512_add_epi32(acc[0], acc[1]));
        c += 1;
    }
}

/// The dot product of one center's 4-bit codes with each of 4 query planes, the same as `sq4_dots` gives for that
/// center and each plane. It goes through the codes once, two chunks at a time for all 4 planes, and puts each step's
/// codes in the query's order once for all of them.
///
/// # Safety
///
/// The CPU must support AVX-512 F, BW, VL and VNNI, AVX2 and FMA. `codes.len()` must be a multiple of 32, and `planes`
/// must hold 4 planes of `2 * codes.len()` values each, back to back.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni,avx2,fma")]
pub(crate) unsafe fn plane_dots(codes: &[u8], planes: &[i8], out: &mut [i32; 4]) {
    let d_pad = 2 * codes.len();
    let mut acc = [[_mm512_setzero_si512(); 2]; 4];

    for start in (0..codes.len()).step_by(64) {
        let two = start + 64 <= codes.len();
        // SAFETY: start + 32 ≤ codes.len() and t·d_pad + 2·start + 64 ≤ planes.len(), by the caller's lengths, plus 32
        // more codes and 64 more values when `two` holds.
        unsafe {
            let k = ordered_codes(codes.as_ptr().add(start), two);
            for (t, a) in acc.iter_mut().enumerate() {
                let q = planes.as_ptr().add(t * d_pad + 2 * start);
                a[0] = _mm512_dpbusd_epi32(a[0], k[0], _mm512_loadu_si512(q.cast()));
                if two {
                    a[1] = _mm512_dpbusd_epi32(a[1], k[1], _mm512_loadu_si512(q.add(64).cast()));
                }
            }
        }
    }

    for (dot, a) in out.iter_mut().zip(&acc) {
        *dot = _mm512_reduce_add_epi32(_mm512_add_epi32(a[0], a[1]));
    }
}

/// Loads the query values that two 32-byte chunks of codes line up with, 128 of them, or only the first chunk's 64
/// with zeros for the second when `two` is false. They come in the order of the chunks' low halves and then their high
/// halves. The low halves hold dimensions 0 to 15 and 32 to 47 of each chunk, and the high halves the 16 after each.
///
/// # Safety
///
/// `q` must point to 128 readable values, or 64 when `two` is false.
#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni,avx2,fma")]
unsafe fn query_runs(q: *const i8, two: bool) -> [__m512i; 2] {
    // SAFETY: the caller guarantees the readable values.
    let (q0, q1) = unsafe {
        let q1 = if two {
            _mm512_loadu_si512(q.add(64).cast())
        } else {
            _mm512_setzero_si512()
        };
        (_mm512_loadu_si512(q.cast()), q1)
    };

    // Quarters 0 and 2 of each, then quarters 1 and 3.
    [
        _mm512_shuffle_i64x2::<0b10_00_10_00>(q0, q1),
        _mm512_shuffle_i64x2::<0b11_01_11_01>(q0, q1),
    ]
}

/// Adds two 32-byte chunks of a center's codes, or only the first when `two` is false, times the query values from
/// `query_runs`, to its two running sums. The low halves go to `acc[0]` and the high halves to `acc[1]`.
///
/// # Safety
///
/// `p` must point to 64 readable bytes, or 32 when `two` is false.
#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni,avx2,fma")]
unsafe fn add_codes(acc: &mut [__m512i; 2], p: *const u8, two: bool, q: [__m512i; 2]) {
    // SAFETY: the caller guarantees the readable bytes. The masked load reads only the first 32.
    let b = unsafe {
        if two {
            _mm512_loadu_si512(p.cast())
        } else {
            _mm512_maskz_loadu_epi8(0xffff_ffff, p.cast())
        }
    };
    let mask = _mm512_set1_epi8(0x0f);

    acc[0] = _mm512_dpbusd_epi32(acc[0], _mm512_and_si512(b, mask), q[0]);
    acc[1] = _mm512_dpbusd_epi32(acc[1], _mm512_and_si512(_mm512_srli_epi16::<4>(b), mask), q[1]);
}

/// Two 32-byte chunks of a center's codes as bytes in the order of their dimensions, 0 to 63 and then 64 to 127, so
/// they line up with the query as it is stored. When `two` is false, only the first chunk is read, and the second
/// vector is 0.
///
/// # Safety
///
/// `p` must point to 64 readable bytes, or 32 when `two` is false.
#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni,avx2,fma")]
unsafe fn ordered_codes(p: *const u8, two: bool) -> [__m512i; 2] {
    // SAFETY: the caller guarantees the readable bytes. The masked load reads only the first 32.
    let b = unsafe {
        if two {
            _mm512_loadu_si512(p.cast())
        } else {
            _mm512_maskz_loadu_epi8(0xffff_ffff, p.cast())
        }
    };
    let mask = _mm512_set1_epi8(0x0f);
    // The low halves hold dimensions 0 to 15, 32 to 47, 64 to 79 and 96 to 111, one run of 16 per 128-bit quarter, and
    // the high halves the 16 after each. Interleaving the quarters of the two puts them in order.
    let (low, high) = (
        _mm512_and_si512(b, mask),
        _mm512_and_si512(_mm512_srli_epi16::<4>(b), mask),
    );

    [
        _mm512_permutex2var_epi64(low, _mm512_setr_epi64(0, 1, 8, 9, 2, 3, 10, 11), high),
        _mm512_permutex2var_epi64(low, _mm512_setr_epi64(4, 5, 12, 13, 6, 7, 14, 15), high),
    ]
}

/// Scores one block of 1-bit codes like `scan_block_scalar` does, and writes the same candidates bit for bit, except
/// that a NaN score may carry another NaN's bits. Each estimate uses the same three fused multiply-adds in the same
/// order, and adds the query's t2, or lane j's `t2[j]` when `t2` is given. It ranks the estimates against `bound` as
/// `before` does, with NaN as +∞, and packs the lanes that pass together in lane order.
///
/// # Safety
///
/// The CPU must support AVX-512 F, BW, VL and VNNI, AVX2 and FMA. `query.lut` must hold 16 entries per group of 4
/// dimensions for a padded dimension that is a multiple of 64, `block` must hold one block for that dimension, and
/// `count` must be at most 32.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni,avx2,fma")]
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
        _mm512_set1_ps(k.k_s),
        _mm512_set1_ps(k.k_p),
        _mm512_set1_ps(k.k_c),
        _mm512_set1_ps(k.t2),
    );
    let (bound_score, infinity) = (_mm512_set1_ps(rank(bound.score)), _mm512_set1_ps(f32::INFINITY));
    let offsets = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    let mut n = 0;

    for (lane, s) in (0..count).step_by(16).zip([(sums[0], sums[1]), (sums[2], sums[3])]) {
        // SAFETY: lanes `lane..lane + 16` are inside the 32 values of each factor, which start at byte 4·d_pad of the
        // block.
        let (f_add, f_rescale, f_pop) = unsafe {
            let p = block.as_ptr().add(4 * d_pad + 4 * lane).cast::<f32>();
            (
                _mm512_loadu_ps(p),
                _mm512_loadu_ps(p.add(BLOCK)),
                _mm512_loadu_ps(p.add(2 * BLOCK)),
            )
        };
        let t2_lanes = match t2 {
            // SAFETY: lanes `lane..lane + 16` are inside the 32 values.
            Some(t2) => unsafe { _mm512_loadu_ps(t2.as_ptr().add(lane)) },
            None => t2_all,
        };
        let s = _mm512_cvtepi32_ps(_mm512_cvtepu16_epi32(_mm256_set_m128i(s.1, s.0)));
        let a = _mm512_fmadd_ps(k_s, s, k_c);
        let ip = _mm512_fmadd_ps(k_p, f_pop, a);
        let est = _mm512_fmadd_ps(f_rescale, ip, _mm512_add_ps(f_add, t2_lanes));
        // The estimates as they rank, with NaN as +∞.
        let x = _mm512_mask_mov_ps(est, _mm512_cmp_ps_mask::<_CMP_UNORD_Q>(est, est), infinity);
        let below = _mm512_cmp_ps_mask::<_CMP_LT_OQ>(x, bound_score);
        let equal = _mm512_cmp_ps_mask::<_CMP_EQ_OQ>(x, bound_score);
        let ties = equal & lower_ids(bound, first_id + lane as u32) as __mmask16;
        let passed = (below | ties) & first_lanes(count - lane) as __mmask16;

        if passed == 0 {
            continue;
        }

        let ids = _mm512_add_epi32(_mm512_set1_epi32(first_id.wrapping_add(lane as u32) as i32), offsets);
        let (mut scores, mut kept) = ([0f32; 16], [0u32; 16]);
        // SAFETY: both arrays hold 16 values.
        unsafe {
            _mm512_storeu_ps(scores.as_mut_ptr(), _mm512_maskz_compress_ps(passed, est));
            _mm512_storeu_si512(kept.as_mut_ptr().cast(), _mm512_maskz_compress_epi32(passed, ids));
        }
        for (&score, &id) in scores.iter().zip(&kept).take(passed.count_ones() as usize) {
            out[n] = Candidate { score, id };
            n += 1;
        }
    }

    n
}

/// The 32 lookup sums of one block, as four vectors of eight 16-bit sums, for lanes 0 to 7, 8 to 15, 16 to 23 and 24
/// to 31. Each 512-bit load covers four groups, one per 128-bit quarter, since a byte shuffle looks up within its
/// quarter. Each pass adds four loads in bytes, which can't overflow since an entry is at most 60, then adds the bytes
/// into 16-bit totals.
///
/// # Safety
///
/// `codes` must point to `groups` groups of 16 bytes and `lut` to `groups` tables of 16 entries. `groups` must be a
/// multiple of 16.
#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni,avx2,fma")]
unsafe fn block_sums(codes: *const u8, lut: *const u8, groups: usize) -> [__m128i; 4] {
    let mask = _mm512_set1_epi8(0x0f);
    // Lanes 0 to 15 go to acc[0] and acc[1], lanes 16 to 31 to acc[2] and acc[3]. The first of each pair adds the bytes
    // as 16-bit values, even lane + 256·odd lane, and the second adds only the odd lanes.
    let mut acc = [_mm512_setzero_si512(); 4];

    for g in (0..groups).step_by(16) {
        // SAFETY: groups g to g + 15 are below `groups`.
        let ((l0, h0), (l1, h1), (l2, h2), (l3, h3)) = unsafe {
            (
                lookup(codes, lut, g, mask),
                lookup(codes, lut, g + 4, mask),
                lookup(codes, lut, g + 8, mask),
                lookup(codes, lut, g + 12, mask),
            )
        };
        let low = _mm512_add_epi8(_mm512_add_epi8(l0, l1), _mm512_add_epi8(l2, l3));
        let high = _mm512_add_epi8(_mm512_add_epi8(h0, h1), _mm512_add_epi8(h2, h3));
        acc[0] = _mm512_add_epi16(acc[0], low);
        acc[1] = _mm512_add_epi16(acc[1], _mm512_srli_epi16::<8>(low));
        acc[2] = _mm512_add_epi16(acc[2], high);
        acc[3] = _mm512_add_epi16(acc[3], _mm512_srli_epi16::<8>(high));
    }

    let (s0, s1) = lane_totals(acc[0], acc[1]);
    let (s2, s3) = lane_totals(acc[2], acc[3]);
    [s0, s1, s2, s3]
}

/// The totals of 16 lanes from one pair of running sums, in lane order, 8 per vector. `both` holds even lane + 256·odd
/// lane in each 16-bit value and `odd` the odd lane, so the even lanes' totals are `both` − 256·`odd`. This holds even
/// when `both` wraps, since every total is below 2^16. Each 128-bit quarter holds the part from every fourth group.
#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni,avx2,fma")]
fn lane_totals(both: __m512i, odd: __m512i) -> (__m128i, __m128i) {
    let even = quarter_sum(_mm512_sub_epi16(both, _mm512_slli_epi16::<8>(odd)));
    let odd = quarter_sum(odd);

    (_mm_unpacklo_epi16(even, odd), _mm_unpackhi_epi16(even, odd))
}

/// The sum of the four 128-bit quarters of `v`, as 16-bit values.
#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni,avx2,fma")]
fn quarter_sum(v: __m512i) -> __m128i {
    let half = _mm256_add_epi16(_mm512_castsi512_si256(v), _mm512_extracti64x4_epi64::<1>(v));
    _mm_add_epi16(_mm256_castsi256_si128(half), _mm256_extracti128_si256::<1>(half))
}

/// Looks up groups g to g + 3 for all 32 lanes, one group per 128-bit quarter. Lanes 0 to 15 come from the low halves
/// of the code bytes, and lanes 16 to 31 from the high halves.
///
/// # Safety
///
/// Groups g to g + 3 must be in range for `codes` and `lut`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni,avx2,fma")]
unsafe fn lookup(codes: *const u8, lut: *const u8, g: usize, mask: __m512i) -> (__m512i, __m512i) {
    // SAFETY: the caller guarantees groups g to g + 3 are in range.
    let (c, l) = unsafe {
        (
            _mm512_loadu_si512(codes.add(16 * g).cast()),
            _mm512_loadu_si512(lut.add(16 * g).cast()),
        )
    };

    (
        _mm512_shuffle_epi8(l, _mm512_and_si512(c, mask)),
        _mm512_shuffle_epi8(l, _mm512_and_si512(_mm512_srli_epi16::<4>(c), mask)),
    )
}
