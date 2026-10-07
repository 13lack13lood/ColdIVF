//! Single instruction, multiple data (SIMD) kernels, chosen at run time from what the CPU supports. This is the only
//! module with `unsafe` code.

#[cfg(target_arch = "aarch64")]
mod neon;

use crate::quantization::sq4;

/// A kernel tier, i.e. which version of the kernels runs, eg. the portable scalar code or the NEON code for Arm CPUs.
/// Only `best` can return a SIMD tier, so a SIMD kernel never runs on a CPU without its instructions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Tier(Kind);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Scalar,
    #[cfg(target_arch = "aarch64")]
    Neon,
}

impl Tier {
    /// The portable tier, which runs on any CPU.
    pub(crate) const SCALAR: Tier = Tier(Kind::Scalar);

    /// The fastest tier this CPU supports.
    pub(crate) fn best() -> Tier {
        #[cfg(target_arch = "aarch64")]
        if std::arch::is_aarch64_feature_detected!("dotprod") {
            return Tier(Kind::Neon);
        }

        Tier::SCALAR
    }

    pub(crate) fn is_scalar(self) -> bool {
        self.0 == Kind::Scalar
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
            // SAFETY: a Neon tier only comes from `Tier::best`, which found dotprod on this CPU. The lengths were
            // checked above.
            unsafe { neon::sq4_dots(codes, q8, out) }
        }
    }
}
