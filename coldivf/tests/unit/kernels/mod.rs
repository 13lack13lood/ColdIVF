//! Unit tests for the kernel tiers. The tests that check each kernel's tiers against the scalar one sit with the code
//! that uses the kernel, in `search/scan.rs` for the scan and `quantization/sq4.rs` for the center codes.

use super::*;

impl Tier {
    /// Every tier this CPU supports but the scalar one, which the tests check them against.
    pub(crate) fn simd() -> impl Iterator<Item = Tier> {
        Tier::supported().filter(|tier| !tier.is_scalar())
    }

    /// The tier's name, as the test scripts and the timing checks spell it.
    pub(crate) fn name(self) -> &'static str {
        match self.0 {
            Kind::Scalar => "scalar",
            #[cfg(target_arch = "aarch64")]
            Kind::Neon => "neon",
            #[cfg(target_arch = "x86_64")]
            Kind::Avx2 => "avx2",
            #[cfg(target_arch = "x86_64")]
            Kind::Avx512 => "avx512",
        }
    }
}

#[test]
fn tiers_come_fastest_first_and_end_with_the_scalar_tier() {
    let tiers: Vec<Tier> = Tier::supported().collect();
    let fastest_first = [
        #[cfg(target_arch = "x86_64")]
        Kind::Avx512,
        #[cfg(target_arch = "x86_64")]
        Kind::Avx2,
        #[cfg(target_arch = "aarch64")]
        Kind::Neon,
        Kind::Scalar,
    ];
    let positions: Vec<usize> = tiers
        .iter()
        .map(|t| fastest_first.iter().position(|&k| k == t.0).unwrap())
        .collect();

    assert!(positions.is_sorted_by(|a, b| a < b), "{tiers:?}");
    assert_eq!(tiers.first(), Some(&Tier::best()));
    assert_eq!(tiers.last(), Some(&Tier::SCALAR));
    #[cfg(target_arch = "aarch64")]
    assert_eq!(
        tiers,
        [Tier(Kind::Neon), Tier::SCALAR],
        "every 64-bit Arm CPU this runs on should have dotprod"
    );
    // The Avx512 tier also needs every instruction that the Avx2 tier uses, so it never comes without it.
    #[cfg(target_arch = "x86_64")]
    assert!(!tiers.contains(&Tier(Kind::Avx512)) || tiers.contains(&Tier(Kind::Avx2)));
}

/// The test scripts name the fastest tier that the CPU's flags allow in `COLDIVF_EXPECT_TIER`, eg. avx512 on a CPU
/// whose flags list every AVX-512 part that tier needs. So a run on such a machine fails instead of quietly testing
/// fewer tiers. Without the variable there is nothing to check.
#[test]
fn the_best_tier_is_the_one_the_test_scripts_expect() {
    let Ok(want) = std::env::var("COLDIVF_EXPECT_TIER") else {
        return;
    };
    let tiers: Vec<&str> = Tier::supported().map(Tier::name).collect();

    assert_eq!(Tier::best().name(), want, "this CPU supports {tiers:?}");
}
