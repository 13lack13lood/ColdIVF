//! The bounded set of best candidates, and the one order that every ranking in a search uses.

use std::cmp::Ordering;

/// A score with an id. It is a vector's estimate or distance with its internal id, or a cluster's route score with the
/// cluster's position. A NaN score, which only a damaged file can give, ranks last.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Candidate {
    pub(crate) score: f32,
    pub(crate) id: u32,
}

impl Candidate {
    /// The bound of a set that isn't full yet. Only a finite score ranks before it.
    pub(crate) const NONE: Candidate = Candidate {
        score: f32::INFINITY,
        id: 0,
    };
}

/// Whether `a` ranks before `b`, with a lower score, or the same score and a lower id. A NaN score ranks as +∞, so the
/// order stays total. Every path uses this test, so results never depend on the order of a scan.
#[inline]
pub(crate) fn before(a: Candidate, b: Candidate) -> bool {
    let (x, y) = (rank(a.score), rank(b.score));
    x < y || (x == y && a.id < b.id)
}

/// A score as it ranks, with NaN as +∞.
#[inline]
fn rank(score: f32) -> f32 {
    if score.is_nan() { f32::INFINITY } else { score }
}

/// `before` as an ordering, for sorting.
pub(crate) fn order(a: &Candidate, b: &Candidate) -> Ordering {
    if before(*a, *b) {
        Ordering::Less
    } else if before(*b, *a) {
        Ordering::Greater
    } else {
        Ordering::Equal
    }
}

/// The best candidates seen so far, at most `cap` of them. It is a heap with the worst kept candidate on top, so one
/// comparison tells whether a new candidate gets in.
pub(crate) struct Candidates {
    cap: usize,
    heap: Vec<Candidate>,
}

impl Candidates {
    /// Room for up to `cap` candidates. It never allocates again.
    pub(crate) fn new(cap: usize) -> Candidates {
        Candidates {
            cap,
            heap: Vec::with_capacity(cap),
        }
    }

    /// Empties the set and keeps up to `cap` candidates from now on.
    pub(crate) fn reset(&mut self, cap: usize) {
        assert!(cap <= self.heap.capacity(), "the set was made for fewer candidates");

        self.cap = cap;
        self.heap.clear();
    }

    /// What a new candidate must rank before to get in. That is `Candidate::NONE` until the set is full, then the
    /// worst kept candidate.
    pub(crate) fn bound(&self) -> Candidate {
        if self.heap.len() < self.cap {
            return Candidate::NONE;
        }

        // A set with room for none takes nothing, and nothing ranks before −∞ with id 0.
        self.heap.first().copied().unwrap_or(Candidate {
            score: f32::NEG_INFINITY,
            id: 0,
        })
    }

    /// Adds `c` if it ranks before the bound. When the set is full, its worst candidate makes room.
    pub(crate) fn offer(&mut self, c: Candidate) {
        if !before(c, self.bound()) {
            return;
        }

        if self.heap.len() < self.cap {
            self.heap.push(c);
            self.sift_up(self.heap.len() - 1);
        } else {
            self.heap[0] = c;
            self.sift_down(0);
        }
    }

    /// The kept candidates, in no particular order.
    pub(crate) fn as_slice(&self) -> &[Candidate] {
        &self.heap
    }

    fn sift_up(&mut self, mut i: usize) {
        while i > 0 {
            let parent = (i - 1) / 2;
            if !before(self.heap[parent], self.heap[i]) {
                break;
            }
            self.heap.swap(parent, i);
            i = parent;
        }
    }

    fn sift_down(&mut self, mut i: usize) {
        loop {
            let mut worst = i;
            for child in [2 * i + 1, 2 * i + 2] {
                if child < self.heap.len() && before(self.heap[worst], self.heap[child]) {
                    worst = child;
                }
            }
            if worst == i {
                break;
            }
            self.heap.swap(worst, i);
            i = worst;
        }
    }
}

#[cfg(test)]
#[path = "../../tests/unit/search/candidates.rs"]
mod tests;
