//! Unit tests for the candidate set.

use super::*;
use crate::quantization::SplitMix64;
use crate::quantization::testutil::shuffle;

#[test]
fn keeps_the_best_by_score_then_id() {
    let mut rng = SplitMix64::new(1);
    for cap in [1, 5, 35, 299, 300, 400] {
        // Scores repeat a lot, so many candidates tie on score and only their ids order them.
        let all: Vec<Candidate> = (0..300)
            .map(|id| Candidate {
                score: (rng.next_u64() % 40) as f32,
                id,
            })
            .collect();
        let mut offered = all.clone();
        shuffle(&mut offered, &mut rng);
        let mut set = Candidates::new(cap);
        offered.iter().for_each(|&c| set.offer(c));
        let mut kept = set.as_slice().to_vec();
        kept.sort_unstable_by(order);
        let mut want = all;
        want.sort_unstable_by(order);
        want.truncate(cap);
        assert_eq!(kept, want, "cap {cap}");
    }
}

#[test]
fn the_bound_tightens_as_the_set_fills() {
    let mut set = Candidates::new(2);
    assert_eq!(set.bound(), Candidate::NONE);

    set.offer(Candidate { score: 3.0, id: 1 });
    assert_eq!(set.bound(), Candidate::NONE);

    set.offer(Candidate { score: 1.0, id: 2 });
    assert_eq!(set.bound(), Candidate { score: 3.0, id: 1 });

    // The same score with a lower id ranks before the bound, but an equal candidate doesn't.
    set.offer(Candidate { score: 3.0, id: 0 });
    assert_eq!(set.bound(), Candidate { score: 3.0, id: 0 });
    set.offer(Candidate { score: 3.0, id: 0 });
    assert_eq!(set.as_slice().len(), 2);

    set.reset(1);
    assert_eq!(set.bound(), Candidate::NONE);
}

#[test]
fn never_keeps_an_infinite_or_nan_score() {
    let mut set = Candidates::new(4);
    for c in [
        Candidate::NONE,
        Candidate {
            score: f32::INFINITY,
            id: 9,
        },
        Candidate { score: f32::NAN, id: 1 },
    ] {
        set.offer(c);
    }

    assert!(set.as_slice().is_empty());
}

#[test]
fn a_set_with_no_room_takes_nothing() {
    let mut set = Candidates::new(0);
    set.offer(Candidate { score: -1e30, id: 0 });

    assert!(set.as_slice().is_empty());
}

#[test]
fn a_nan_score_ranks_last_and_sorting_never_panics() {
    let nan = Candidate { score: f32::NAN, id: 3 };
    assert!(before(Candidate { score: f32::MAX, id: 9 }, nan));
    // NaN ranks as +∞, so against +∞ the lower id goes first.
    let infinite = |id| Candidate {
        score: f32::INFINITY,
        id,
    };
    assert!(before(nan, infinite(9)) && !before(nan, infinite(1)));

    // The library's sorts check that the order is total once a slice has more than 20 items.
    let mut many: Vec<Candidate> = (0..40)
        .map(|id| Candidate {
            score: (id % 7) as f32,
            id,
        })
        .collect();
    many[17].score = f32::NAN;
    many.sort_unstable_by(order);
    assert_eq!(many.last().unwrap().id, 17);
}
