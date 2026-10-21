//! The pending inserts of a data file by cluster, so a search scans only those of the clusters it probes and of the
//! clusters ranked next, up to 5/4 of `nprobe`, instead of every one. The sorted view holds them in one array, cluster
//! after cluster, in blocks of 32 for the scan, with each lane's place among the inserts, and their codes and factors
//! again unpacked, for a cluster with so few that a block would be mostly empty. The inserts since its last rebuild
//! sit in small groups by cluster, which an insert copies. Every `REBUILD` inserts they join the sorted view, which
//! copies it whole, and the snapshots that hold the old one keep it.

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::quantization::rabitq::{BLOCK, Factors, block_bytes, code_bytes, pack_block, unpack_lane};

/// The inserts after which the groups changed since the last rebuild join the sorted view.
pub(crate) const REBUILD: usize = 256;

/// The place of a lane of a block that holds no insert.
const NO_PLACE: u32 = u32::MAX;

/// Some of one cluster's pending inserts: their blocks, and each lane's place among the inserts, and the same codes
/// back to back and their factors.
#[derive(Clone, Copy)]
pub(crate) struct Group<'a> {
    blocks: &'a [u8],
    places: &'a [u32],
    codes: &'a [u8],
    factors: &'a [Factors],
}

impl<'a> Group<'a> {
    /// How many inserts it holds.
    pub(crate) fn len(&self) -> usize {
        self.places.len()
    }

    /// The blocks, `len().div_ceil(BLOCK)` of them.
    pub(crate) fn blocks(&self) -> &'a [u8] {
        self.blocks
    }

    /// The place among the inserts of lane `j`.
    pub(crate) fn place(&self, j: usize) -> u32 {
        self.places[j]
    }

    /// The lanes' codes, back to back, and their factors.
    pub(crate) fn lanes(&self) -> (&'a [u8], &'a [Factors]) {
        (self.codes, self.factors)
    }
}

/// One cluster's pending inserts, found with one lookup: its group in the sorted view, then its group of the inserts
/// since.
#[derive(Clone, Copy, Default)]
pub(crate) struct ClusterGroups<'a> {
    sorted: Option<Group<'a>>,
    recent: Option<Group<'a>>,
}

impl<'a> ClusterGroups<'a> {
    /// Whether the cluster holds no pending insert.
    pub(crate) fn is_empty(&self) -> bool {
        self.sorted.is_none() && self.recent.is_none()
    }

    /// Its groups, the sorted view's first.
    pub(crate) fn iter(&self) -> impl Iterator<Item = Group<'a>> {
        self.sorted.into_iter().chain(self.recent)
    }

    /// How many inserts they hold.
    pub(crate) fn len(&self) -> usize {
        self.iter().map(|g| g.len()).sum()
    }
}

/// Pending inserts' lanes, unpacked, with each one's place.
#[derive(Clone, Default)]
struct Lanes {
    codes: Vec<u8>,
    factors: Vec<Factors>,
    places: Vec<u32>,
}

impl Lanes {
    fn len(&self) -> usize {
        self.places.len()
    }

    fn push(&mut self, code: &[u8], factors: Factors, place: u32) {
        self.codes.extend_from_slice(code);
        self.factors.push(factors);
        self.places.push(place);
    }

    fn append(&mut self, more: &Lanes) {
        self.codes.extend_from_slice(&more.codes);
        self.factors.extend_from_slice(&more.factors);
        self.places.extend_from_slice(&more.places);
    }

    /// Packs the lanes from lane `first`, which starts a block, onto `blocks`.
    fn pack_onto(&self, first: usize, d_pad: usize, blocks: &mut Vec<u8>) {
        let (cb, size) = (code_bytes(d_pad), block_bytes(d_pad));

        for s in (first..self.len()).step_by(BLOCK) {
            let e = (s + BLOCK).min(self.len());
            let at = blocks.len();
            blocks.resize(at + size, 0);
            pack_block(
                &self.codes[s * cb..e * cb],
                &self.factors[s..e],
                d_pad,
                &mut blocks[at..],
            );
        }
    }
}

/// The inserts into one cluster since the last rebuild: their lanes, and the same packed in blocks.
struct Recent {
    lanes: Lanes,
    blocks: Vec<u8>,
}

impl Recent {
    fn group(&self) -> Group<'_> {
        Group {
            blocks: &self.blocks,
            places: &self.lanes.places,
            codes: &self.lanes.codes,
            factors: &self.lanes.factors,
        }
    }

    /// The same inserts and one more, at place `place`. It packs again only the last block.
    fn with(&self, code: &[u8], factors: Factors, place: u32, d_pad: usize) -> Recent {
        let mut lanes = self.lanes.clone();
        lanes.push(code, factors, place);
        let kept = self.lanes.len() / BLOCK;
        let mut blocks = self.blocks[..kept * block_bytes(d_pad)].to_vec();
        lanes.pack_onto(kept * BLOCK, d_pad, &mut blocks);
        Recent { lanes, blocks }
    }
}

/// Where a cluster's inserts are in the sorted view: its first block, how many there are, and the first of their
/// unpacked lanes.
#[derive(Clone, Copy, Default)]
struct Span {
    block: u32,
    count: u32,
    lane: u32,
}

/// Every cluster's inserts as of the last rebuild: the blocks, cluster after cluster, and the places of their lanes,
/// `BLOCK` per block, and the lanes' codes and factors unpacked, cluster after cluster, with no lane between clusters.
struct Sorted {
    spans: Vec<Span>,
    blocks: Vec<u8>,
    places: Vec<u32>,
    codes: Vec<u8>,
    factors: Vec<Factors>,
}

/// Groups of pending inserts by cluster, as a search finds them.
#[derive(Clone)]
pub(crate) struct Groups {
    sorted: Arc<Sorted>,
    /// The groups of the inserts since, by cluster in increasing order.
    recent: Arc<Vec<(u32, Arc<Recent>)>>,
    /// The inserts in `recent`.
    since: usize,
    d_pad: usize,
}

impl Groups {
    /// The bytes of the sorted view's blocks, places and unpacked lanes, and of the recent groups' lanes and blocks.
    #[cfg(test)]
    pub(crate) fn bytes(&self) -> (usize, usize) {
        let unpacked = self.sorted.codes.len() + 16 * self.sorted.factors.len();
        let sorted = self.sorted.blocks.len() + 4 * self.sorted.places.len() + unpacked;
        let recent = self.recent.iter().map(|(_, r)| {
            let lanes = &r.lanes;
            r.blocks.len() + lanes.codes.len() + 16 * lanes.factors.len() + 4 * lanes.places.len()
        });
        (sorted, recent.sum())
    }

    /// No pending insert, for a file of `nlist` clusters and vectors of `dim` dimensions.
    pub(crate) fn new(nlist: usize, dim: usize) -> Groups {
        Groups {
            sorted: Arc::new(Sorted {
                spans: vec![Span::default(); nlist],
                blocks: Vec::new(),
                places: Vec::new(),
                codes: Vec::new(),
                factors: Vec::new(),
            }),
            recent: Arc::new(Vec::new()),
            since: 0,
            d_pad: crate::quantization::padded_dim(dim),
        }
    }

    /// The groups of cluster `list`: its inserts in the sorted view, then those since.
    pub(crate) fn of(&self, list: u32) -> ClusterGroups<'_> {
        let sorted = self.sorted.spans.get(list as usize).filter(|s| s.count > 0).map(|s| {
            let (first, blocks) = (s.block as usize, (s.count as usize).div_ceil(BLOCK));
            let (size, cb) = (block_bytes(self.d_pad), code_bytes(self.d_pad));
            let lanes = s.lane as usize..(s.lane + s.count) as usize;
            Group {
                blocks: &self.sorted.blocks[first * size..(first + blocks) * size],
                places: &self.sorted.places[first * BLOCK..first * BLOCK + s.count as usize],
                codes: &self.sorted.codes[lanes.start * cb..lanes.end * cb],
                factors: &self.sorted.factors[lanes],
            }
        });
        let recent = match self.recent.is_empty() {
            true => None,
            false => self.recent.binary_search_by_key(&list, |(l, _)| *l).ok(),
        };
        ClusterGroups {
            sorted,
            recent: recent.map(|i| self.recent[i].1.group()),
        }
    }

    /// The same groups with the insert at place `place` in cluster `list`, with its code and factors. Every
    /// `REBUILD` inserts the recent groups join the sorted view.
    pub(crate) fn with(&self, list: u32, code: &[u8], factors: Factors, place: u32) -> Groups {
        let mut recent: Vec<(u32, Arc<Recent>)> = self.recent.as_ref().clone();
        let d_pad = self.d_pad;

        match recent.binary_search_by_key(&list, |(l, _)| *l) {
            Ok(i) => recent[i].1 = Arc::new(recent[i].1.with(code, factors, place, d_pad)),
            Err(i) => {
                let mut lanes = Lanes::default();
                lanes.push(code, factors, place);
                let mut blocks = Vec::new();
                lanes.pack_onto(0, d_pad, &mut blocks);
                recent.insert(i, (list, Arc::new(Recent { lanes, blocks })));
            }
        }

        let next = Groups {
            sorted: Arc::clone(&self.sorted),
            recent: Arc::new(recent),
            since: self.since + 1,
            d_pad,
        };

        match next.since >= REBUILD {
            true => next.rebuilt_with(BTreeMap::new()),
            false => next,
        }
    }

    /// The same groups with every insert of `lanes`, as (cluster, code, factors, place) in place order, and the recent
    /// ones joined to the sorted view, eg. for a replay of many records. Each new lane is packed once.
    pub(crate) fn with_many<'a>(&self, lanes: impl Iterator<Item = (u32, &'a [u8], Factors, u32)>) -> Groups {
        let mut more: BTreeMap<u32, Lanes> = BTreeMap::new();

        for (list, code, factors, place) in lanes {
            more.entry(list).or_default().push(code, factors, place);
        }

        self.rebuilt_with(more)
    }

    /// The same groups with the recent ones, then the lanes of `more`, after each cluster's inserts in the sorted
    /// view, which it copies. Only the last block of a cluster that gains inserts is packed again.
    fn rebuilt_with(&self, mut more: BTreeMap<u32, Lanes>) -> Groups {
        for (list, g) in self.recent.iter() {
            let mut lanes = g.lanes.clone();
            if let Some(after) = more.remove(list) {
                lanes.append(&after);
            }
            more.insert(*list, lanes);
        }

        let (old, d_pad) = (&*self.sorted, self.d_pad);
        let (size, mut bits) = (block_bytes(d_pad), vec![0u8; code_bytes(d_pad)]);
        let added: usize = more
            .iter()
            .map(|(&l, m)| {
                let count = old.spans[l as usize].count as usize;
                (count + m.len()).div_ceil(BLOCK) - count.div_ceil(BLOCK)
            })
            .sum();
        let blocks = old.blocks.len() / size + added;
        let lanes = old.factors.len() + more.values().map(Lanes::len).sum::<usize>();
        let mut new = Sorted {
            spans: Vec::with_capacity(old.spans.len()),
            blocks: Vec::with_capacity(blocks * size),
            places: Vec::with_capacity(blocks * BLOCK),
            codes: Vec::with_capacity(lanes * code_bytes(d_pad)),
            factors: Vec::with_capacity(lanes),
        };
        let (cb, mut more) = (code_bytes(d_pad), more.into_iter().peekable());

        for (list, span) in old.spans.iter().enumerate() {
            let (from, count) = (span.block as usize, span.count as usize);
            let (block, lane) = ((new.places.len() / BLOCK) as u32, new.factors.len() as u32);
            let had = span.lane as usize..span.lane as usize + count;
            new.codes.extend_from_slice(&old.codes[had.start * cb..had.end * cb]);
            new.factors.extend_from_slice(&old.factors[had]);
            // A cluster that gains no insert keeps its blocks as they are, and one that does keeps its full blocks.
            let gains = more.next_if(|(l, _)| *l as usize == list).map(|(_, m)| m);
            let kept = match gains {
                None => count.div_ceil(BLOCK),
                Some(_) => count / BLOCK,
            };
            new.blocks
                .extend_from_slice(&old.blocks[from * size..(from + kept) * size]);
            new.places
                .extend_from_slice(&old.places[from * BLOCK..(from + kept) * BLOCK]);
            let Some(gains) = gains else {
                new.spans.push(Span {
                    block,
                    count: span.count,
                    lane,
                });
                continue;
            };

            // The lanes of the last block, if it isn't full, then the new ones.
            let mut tail = Lanes::default();

            for j in kept * BLOCK..count {
                let last = &old.blocks[(from + kept) * size..(from + kept + 1) * size];
                let f = unpack_lane(last, d_pad, j % BLOCK, &mut bits);
                tail.push(&bits, f, old.places[from * BLOCK + j]);
            }

            tail.append(&gains);
            tail.pack_onto(0, d_pad, &mut new.blocks);
            new.places.extend_from_slice(&tail.places);
            new.places.resize(new.places.len().next_multiple_of(BLOCK), NO_PLACE);
            new.codes.extend_from_slice(&gains.codes);
            new.factors.extend_from_slice(&gains.factors);
            new.spans.push(Span {
                block,
                count: (count + gains.len()) as u32,
                lane,
            });
        }

        debug_assert!(more.next().is_none(), "a lane of a cluster past the file's");
        Groups {
            sorted: Arc::new(new),
            recent: Arc::new(Vec::new()),
            since: 0,
            d_pad,
        }
    }
}

#[cfg(test)]
#[path = "../../tests/unit/update/view.rs"]
mod tests;
