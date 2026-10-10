//! Unit tests for the wave check.

use super::*;

#[test]
fn a_trials_waves_read_pages_that_no_other_wave_of_it_reads() {
    let mut state = 7u64;
    let waves = offsets(&mut state, 64, 4, 12, 16384);
    assert_eq!(waves.len(), 12);
    assert!(waves.iter().all(|w| w.len() == 4), "{waves:?}");

    // Each wave's pages are in increasing order, on page boundaries, within the file's 64 pages.
    for wave in &waves {
        assert!(wave.windows(2).all(|p| p[0] < p[1]), "{wave:?}");
        assert!(wave.iter().all(|&at| at % 16384 == 0 && at < 64 * 16384), "{wave:?}");
    }

    let mut all: Vec<usize> = waves.iter().flatten().copied().collect();
    all.sort_unstable();
    all.dedup();
    assert_eq!(all.len(), 48);

    // The next trial draws other pages, and the same state draws the same ones.
    let mut again = 7u64;
    assert_eq!(offsets(&mut again, 64, 4, 12, 16384), waves);
    assert_ne!(offsets(&mut state, 64, 4, 12, 16384), waves);
}

#[test]
fn a_trial_takes_the_time_of_all_its_waves() {
    let (totals, laps) = spread(&[vec![3.0, 1.0], vec![2.0, 0.5]]);
    assert_eq!(totals, [2.5, 4.0]);
    assert_eq!(laps, [0.5, 1.0, 2.0, 3.0]);
}
