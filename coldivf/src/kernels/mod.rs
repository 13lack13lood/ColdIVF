//! Single instruction, multiple data (SIMD) kernels, chosen at run time from what the CPU supports. Besides `io`, this
//! is the only module with `unsafe` code.

#[cfg(target_arch = "x86_64")]
mod avx2;
#[cfg(target_arch = "x86_64")]
mod avx512;
#[cfg(target_arch = "aarch64")]
mod neon;

use crate::quantization::rabitq::{BLOCK, QueryLut, block_bytes};
use crate::quantization::sq4::{self, CenterFactors, PLANES, QueryS32};
use crate::search::candidates::Candidate;
use crate::search::scan::scan_block_scalar;

/// A kernel tier, i.e. which version of the kernels runs, eg. the portable scalar code or the NEON code for Arm CPUs.
/// Only `best` and `supported` can return a SIMD tier, so a SIMD kernel never runs on a CPU without its instructions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Tier(Kind);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    /// Portable code, for any CPU.
    Scalar,
    /// NEON with its dot-product instructions, on 64-bit Arm CPUs.
    #[cfg(target_arch = "aarch64")]
    Neon,
    /// The Advanced Vector Extensions 2 (AVX2), 256 bits wide, with fused multiply-add (FMA), eg. on Intel CPUs since
    /// Haswell and AMD CPUs since Zen.
    #[cfg(target_arch = "x86_64")]
    Avx2,
    /// AVX-512, 512 bits wide, with its Foundation (F), Byte and Word (BW), Vector Length (VL) and Vector Neural
    /// Network Instructions (VNNI) parts, eg. on Intel CPUs since Ice Lake and AMD CPUs since Zen 4.
    #[cfg(target_arch = "x86_64")]
    Avx512,
}

impl Tier {
    /// The portable tier, which runs on any CPU.
    pub(crate) const SCALAR: Tier = Tier(Kind::Scalar);

    /// The fastest tier this CPU supports.
    pub(crate) fn best() -> Tier {
        Tier::supported().next().unwrap_or(Tier::SCALAR)
    }

    /// Every tier this CPU supports, fastest first. The scalar tier comes last.
    pub(crate) fn supported() -> impl Iterator<Item = Tier> {
        [
            #[cfg(target_arch = "x86_64")]
            Kind::Avx512,
            #[cfg(target_arch = "x86_64")]
            Kind::Avx2,
            #[cfg(target_arch = "aarch64")]
            Kind::Neon,
            Kind::Scalar,
        ]
        .into_iter()
        .filter(|kind| kind.detected())
        .map(Tier)
    }

    pub(crate) fn is_scalar(self) -> bool {
        self.0 == Kind::Scalar
    }
}

impl Kind {
    /// Whether this CPU has every instruction that the tier's kernels use.
    fn detected(self) -> bool {
        match self {
            Kind::Scalar => true,
            #[cfg(target_arch = "aarch64")]
            Kind::Neon => std::arch::is_aarch64_feature_detected!("dotprod"),
            #[cfg(target_arch = "x86_64")]
            Kind::Avx2 => std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma"),
            #[cfg(target_arch = "x86_64")]
            Kind::Avx512 => {
                Kind::Avx2.detected()
                    && std::arch::is_x86_feature_detected!("avx512f")
                    && std::arch::is_x86_feature_detected!("avx512bw")
                    && std::arch::is_x86_feature_detected!("avx512vl")
                    && std::arch::is_x86_feature_detected!("avx512vnni")
            }
        }
    }
}

/// The dot product Σ q8_i k_i for each cluster center, the integer part of its route score. Here q8_i is `q8[i]` and
/// k_i is the center's 4-bit code for dimension i. `codes` holds `out.len()` centers back to back, `q8.len() / 2` bytes
/// each.
pub(crate) fn sq4_dots(tier: Tier, codes: &[u8], q8: &[i8], out: &mut [i32]) {
    assert!(q8.len().is_multiple_of(64) && out.len().checked_mul(q8.len() / 2) == Some(codes.len()));

    match tier.0 {
        Kind::Scalar => sq4::dots_scalar(codes, q8, out),
        #[cfg(target_arch = "aarch64")]
        Kind::Neon => {
            // SAFETY: a Neon tier only comes from `Tier::supported`, which found dotprod on this CPU. The lengths were
            // checked above.
            unsafe { neon::sq4_dots(codes, q8, out) }
        }
        #[cfg(target_arch = "x86_64")]
        Kind::Avx2 => {
            // SAFETY: an Avx2 tier only comes from `Tier::supported`, which found AVX2 and FMA on this CPU. The
            // lengths were checked above.
            unsafe { avx2::sq4_dots(codes, q8, out) }
        }
        #[cfg(target_arch = "x86_64")]
        Kind::Avx512 => {
            // SAFETY: an Avx512 tier only comes from `Tier::supported`, which found AVX-512 F, BW, VL and VNNI, AVX2
            // and FMA on this CPU. The lengths were checked above.
            unsafe { avx512::sq4_dots(codes, q8, out) }
        }
    }
}

/// The squared distance from the query in `q` to the center that one cluster's 4-bit codes `codes` and factors `f`
/// decode to, whose squared length is `norm2`. It takes one dot product of the codes per digit of the query's fixed
/// point. They are exact integers, so every tier gives the same bits.
pub(crate) fn center_distance2(tier: Tier, codes: &[u8], f: &CenterFactors, norm2: f64, q: &QueryS32) -> f64 {
    let (planes, d_pad) = (q.planes(), 2 * codes.len());
    assert!(d_pad.is_multiple_of(64) && planes.len() == PLANES * d_pad);
    let mut dots = [0i32; PLANES];

    match tier.0 {
        Kind::Scalar => {
            for (dot, plane) in dots.iter_mut().zip(planes.chunks_exact(d_pad)) {
                sq4::dots_scalar(codes, plane, std::slice::from_mut(dot));
            }
        }
        #[cfg(target_arch = "aarch64")]
        Kind::Neon => {
            // SAFETY: a Neon tier only comes from `Tier::supported`, which found dotprod on this CPU. The lengths were
            // checked above.
            unsafe { neon::plane_dots(codes, planes, &mut dots) }
        }
        #[cfg(target_arch = "x86_64")]
        Kind::Avx2 => {
            // SAFETY: an Avx2 tier only comes from `Tier::supported`, which found AVX2 and FMA on this CPU. The
            // lengths were checked above.
            unsafe { avx2::plane_dots(codes, planes, &mut dots) }
        }
        #[cfg(target_arch = "x86_64")]
        Kind::Avx512 => {
            // SAFETY: an Avx512 tier only comes from `Tier::supported`, which found AVX-512 F, BW, VL and VNNI, AVX2
            // and FMA on this CPU. The lengths were checked above.
            unsafe { avx512::plane_dots(codes, planes, &mut dots) }
        }
    }

    q.distance2(f, norm2, &dots)
}

/// Scores one block of 1-bit codes for a query and writes the lanes that rank before `bound` into `out`, in lane order.
/// Only the first `count` lanes hold vectors, and lane j has internal id `first_id + j`. Each lane's estimate adds the
/// query's t2, or `t2[j]` when `t2` is given, eg. the squared distance to the center that lane j's code was measured
/// from. Returns how many it wrote. Every tier writes the same candidates, bit for bit, except that a NaN score may
/// carry another NaN's bits. A fused multiply-add with two NaN inputs passes on the one that its operand order picks,
/// and the tiers order their operands differently.
#[allow(clippy::too_many_arguments)]
pub(crate) fn scan_block(
    tier: Tier,
    block: &[u8],
    query: &QueryLut,
    t2: Option<&[f32; BLOCK]>,
    first_id: u32,
    count: usize,
    bound: Candidate,
    out: &mut [Candidate; BLOCK],
) -> usize {
    let d_pad = query.lut.len() / 4;
    assert!(d_pad.is_multiple_of(64) && block.len() == block_bytes(d_pad) && count <= BLOCK);

    match tier.0 {
        Kind::Scalar => scan_block_scalar(block, query, t2, first_id, count, bound, out),
        #[cfg(target_arch = "aarch64")]
        Kind::Neon => {
            // SAFETY: a Neon tier only comes from `Tier::supported`, which found dotprod on this CPU. The lengths were
            // checked above.
            unsafe { neon::scan_block(block, query, t2, first_id, count, bound, out) }
        }
        #[cfg(target_arch = "x86_64")]
        Kind::Avx2 => {
            // SAFETY: an Avx2 tier only comes from `Tier::supported`, which found AVX2 and FMA on this CPU. The
            // lengths were checked above.
            unsafe { avx2::scan_block(block, query, t2, first_id, count, bound, out) }
        }
        #[cfg(target_arch = "x86_64")]
        Kind::Avx512 => {
            // SAFETY: an Avx512 tier only comes from `Tier::supported`, which found AVX-512 F, BW, VL and VNNI, AVX2
            // and FMA on this CPU. The lengths were checked above.
            unsafe { avx512::scan_block(block, query, t2, first_id, count, bound, out) }
        }
    }
}

/// The lanes of a block whose id is below the bound's, from lane 0, as one bit per lane. Lane j has id `first_id + j`,
/// so they come first.
#[cfg(target_arch = "x86_64")]
fn lower_ids(bound: Candidate, first_id: u32) -> u32 {
    first_lanes(bound.id.saturating_sub(first_id) as usize)
}

/// The first `n` lanes of a block, as one bit per lane.
#[cfg(target_arch = "x86_64")]
fn first_lanes(n: usize) -> u32 {
    ((1u64 << n.min(BLOCK)) - 1) as u32
}

#[cfg(test)]
#[path = "../../tests/unit/kernels/mod.rs"]
mod tests;
