//! Streams of changes to replay on an index: the dataset rows of the base, then for each round the rows it inserts,
//! the vectors it deletes and the vectors it gives a new vector. A vector's id in a stream is the dataset row it was
//! first inserted with. A stream is made from a seed, so the same settings always give the same stream, or read from
//! a stream file.
//!
//! A stream file is little-endian. It starts with the magic `COLDSTRM`, a u32 version and a u32 count of rounds. Each
//! round holds three u32 counts, of its inserts, deletes and upserts, then the inserted rows as u32, the deleted ids
//! as u64, and each upsert as a u64 id and a u32 row. The first round is the base, which holds only inserts.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Error, ErrorKind, Read, Write};
use std::path::Path;

use coldivf::{Anchor, BuildParams, Builder, LeafOrder, Metric, VectorSource};

use crate::data::Fbin;
use crate::truth::nearest;

pub const MAGIC: [u8; 8] = *b"COLDSTRM";
pub const VERSION: u32 = 1;

/// The nearest held-out rows that each upsert looks at for its new vector.
const NEAREST: usize = 32;

/// One round of changes. Its deletes apply first, then its upserts, then its inserts.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Round {
    /// Dataset rows inserted as new vectors.
    pub inserts: Vec<u32>,
    /// Ids of the vectors deleted.
    pub deletes: Vec<u64>,
    /// Ids of the vectors given a new vector, each with the dataset row of the new one.
    pub upserts: Vec<(u64, u32)>,
}

/// The base and the rounds of changes after it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Stream {
    pub base: Vec<u32>,
    pub rounds: Vec<Round>,
}

/// Where a made stream's inserts come from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pattern {
    /// Random rows held out from the base.
    Uniform,
    /// Held-out rows of a few topics that the base holds too. The topics are a few of the clusters that k-means finds
    /// among the held-out rows, and the inserts are the held-out rows nearest them.
    Drift,
    /// Rows of a few topics that the base lacks. The topics are found the same way among all the rows, and the base is
    /// drawn from the rows outside them.
    Novel,
}

/// How a stream is made.
#[derive(Clone, Debug)]
pub struct Params {
    pub pattern: Pattern,
    /// Vectors in the base.
    pub base: usize,
    /// Whether the base is the dataset's first rows. Otherwise it is rows drawn at random.
    pub first: bool,
    pub rounds: usize,
    /// Rows that each round inserts.
    pub insert: usize,
    /// Shares of the live vectors that each round deletes, and gives a new vector, before its inserts. An upsert's new
    /// vector is the nearest held-out row not used yet, like an edited document.
    pub delete: f64,
    pub upsert: f64,
    /// Held-out rows set aside as queries: pseudo-queries, drawn like the base, and hot queries, drawn from the
    /// inserts' topics. Uniform streams have no hot queries.
    pub pseudo: usize,
    pub hot: usize,
    /// The clusters that k-means finds for the topics, how many of them the inserts come from, and the rows it
    /// trains on.
    pub topics: usize,
    pub hot_topics: usize,
    pub sample: usize,
    /// The held-out rows that upserts take their new vectors from.
    pub upsert_pool: usize,
    pub seed: u64,
    /// Threads for k-means and the exact searches.
    pub threads: usize,
}

/// A made stream, and the held-out rows it sets aside as queries.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Made {
    pub stream: Stream,
    pub pseudo: Vec<u32>,
    pub hot: Vec<u32>,
}

/// Makes the stream that `p` describes from the rows of `data`, which `metric` measures.
pub fn make(data: &Fbin, metric: Metric, p: &Params) -> Result<Made, Error> {
    let total = data.len();
    let inserts = p.rounds * p.insert;
    let hot = if p.pattern == Pattern::Uniform { 0 } else { p.hot };
    let need = p.base + inserts + hot + p.pseudo;

    if p.base == 0 || total > u32::MAX as usize {
        return Err(invalid("a stream needs a base, and a dataset of fewer than 2^32 rows"));
    }

    if need > total {
        return Err(invalid(&format!(
            "the stream needs {need} rows, but the dataset holds {total}"
        )));
    }

    if !(0.0..=1.0).contains(&p.delete) || !(0.0..=1.0).contains(&p.upsert) || p.delete + p.upsert > 1.0 {
        return Err(invalid(
            "the delete and upsert shares must be from 0 to 1, and add up to at most 1",
        ));
    }

    let mut rng = Rng::new(p.seed);
    let (base, ins, hot) = match p.pattern {
        Pattern::Uniform => {
            let (base, held) = split(total, p.base, p.first, &mut rng, |_| true);
            (base, held[..inserts].to_vec(), Vec::new())
        }
        Pattern::Drift => {
            let (base, held) = split(total, p.base, p.first, &mut rng, |_| true);
            let (ins, hot) = topical(data, metric, &held, inserts, hot, p, &mut rng)?;
            (base, ins, hot)
        }
        Pattern::Novel => {
            let all: Vec<u32> = (0..total as u32).collect();
            let (ins, hot) = topical(data, metric, &all, inserts, hot, p, &mut rng)?;
            let mut taken = vec![false; total];

            for &r in ins.iter().chain(&hot) {
                taken[r as usize] = true;
            }

            let (base, _) = split(total, p.base, p.first, &mut rng, |r| !taken[r as usize]);
            (base, ins, hot)
        }
    };
    let mut used = vec![false; total];

    for &r in base.iter().chain(&ins).chain(&hot) {
        used[r as usize] = true;
    }

    let mut free: Vec<u32> = (0..total as u32).filter(|&r| !used[r as usize]).collect();
    rng.shuffle(&mut free);
    let pseudo = free[..p.pseudo].to_vec();
    let pool = &free[p.pseudo..(p.pseudo + p.upsert_pool).min(free.len())];
    let rounds = rounds(data, metric, &base, &ins, pool, p, &mut rng)?;

    Ok(Made {
        stream: Stream { base, rounds },
        pseudo,
        hot,
    })
}

/// The base of `n` rows among the rows below `total` that `allowed` lets in, sorted, and the rest of those rows in a
/// random order. The base is the first `n` of them when `first` is set, and otherwise `n` drawn at random.
fn split(total: usize, n: usize, first: bool, rng: &mut Rng, allowed: impl Fn(u32) -> bool) -> (Vec<u32>, Vec<u32>) {
    let mut rows: Vec<u32> = (0..total as u32).filter(|&r| allowed(r)).collect();

    if !first {
        rng.shuffle(&mut rows);
    }

    let mut held = rows.split_off(n.min(rows.len()));
    rows.sort_unstable();

    if first {
        rng.shuffle(&mut held);
    }

    (rows, held)
}

/// The inserts and hot queries of a topical stream, drawn from `candidates`. k-means over a sample of them finds
/// `p.topics` clusters, and `p.hot_topics` of those are the topics. The rows of `candidates` nearest a topic's center,
/// as many as the inserts and hot queries need, go in a random order to the inserts first and the hot queries after.
fn topical(
    data: &Fbin,
    metric: Metric,
    candidates: &[u32],
    inserts: usize,
    hot: usize,
    p: &Params,
    rng: &mut Rng,
) -> Result<(Vec<u32>, Vec<u32>), Error> {
    let dim = data.dim();
    let n = p.sample.min(candidates.len());
    let mut sample = candidates.to_vec();
    rng.shuffle(&mut sample);
    let mut rows = Vec::with_capacity(n * dim);

    for &r in &sample[..n] {
        rows.extend_from_slice(data.row(r as usize));
    }

    let topics = p.topics.clamp(1, n.max(1));
    let params = BuildParams {
        metric,
        vectors_per_list: n.div_ceil(topics).max(1),
        parents: 1,
        order: LeafOrder::Id,
        anchor: Anchor::Cluster,
        page_size: 4096,
        threads: p.threads,
        seed: p.seed,
    };
    let src = Rows { dim, rows };
    let clusters = Builder::new(params).cluster(&src, Path::new("topics"))?;
    let mut picked: Vec<usize> = (0..clusters.centroids.len() / dim).collect();
    rng.shuffle(&mut picked);
    picked.truncate(p.hot_topics.max(1));
    picked.sort_unstable();
    let centers: Vec<&[f32]> = picked
        .iter()
        .map(|&c| &clusters.centroids[c * dim..(c + 1) * dim])
        .collect();
    let mut pool = nearest_rows(data, metric, candidates, &centers, inserts + hot, p.threads);
    rng.shuffle(&mut pool);
    let hot = pool.split_off(inserts);
    Ok((pool, hot))
}

/// The `n` rows of `candidates` nearest to any of `centers`, nearest first, with ties to the lower row.
fn nearest_rows(
    data: &Fbin,
    metric: Metric,
    candidates: &[u32],
    centers: &[&[f32]],
    n: usize,
    threads: usize,
) -> Vec<u32> {
    let mut near = topic_distances(data, metric, candidates, centers, threads);
    near.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    near.truncate(n);
    near.into_iter().map(|(_, r)| r).collect()
}

/// Each row of `rows` with its squared distance to the nearest of `centers`. For cosine the row is measured as a unit
/// vector, as k-means measures it.
fn topic_distances(data: &Fbin, metric: Metric, rows: &[u32], centers: &[&[f32]], threads: usize) -> Vec<(f32, u32)> {
    let norms: Vec<f32> = centers.iter().map(|c| dot(c, c)).collect();
    let per = rows.len().div_ceil(threads.max(1)).max(1);
    let mut out = vec![(0f32, 0u32); rows.len()];

    std::thread::scope(|scope| {
        for (part, mine) in out.chunks_mut(per).zip(rows.chunks(per)) {
            let (norms, centers) = (&norms, centers);
            scope.spawn(move || {
                for (slot, &r) in part.iter_mut().zip(mine) {
                    let x = data.row(r as usize);
                    let xx = dot(x, x);
                    let scale = match metric {
                        Metric::Cosine if xx > 0.0 => 1.0 / xx.sqrt(),
                        _ => 1.0,
                    };
                    let unit = xx * scale * scale;
                    let d = centers
                        .iter()
                        .zip(norms)
                        .map(|(c, &cc)| unit - 2.0 * scale * dot(x, c) + cc)
                        .fold(f32::INFINITY, f32::min);
                    *slot = (d, r);
                }
            });
        }
    });

    out
}

/// The dot product, in eight partial sums, so the compiler vectorizes it.
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    let (ac, ar) = a.as_chunks::<8>();
    let (bc, br) = b.as_chunks::<8>();

    for (x, y) in ac.iter().zip(bc) {
        (0..8).for_each(|i| acc[i] += x[i] * y[i]);
    }

    acc.iter().sum::<f32>() + ar.iter().zip(br).map(|(x, y)| x * y).sum::<f32>()
}

/// The rounds of a made stream. Each round first deletes its share of the live vectors and gives its share of the
/// rest a new vector, both drawn at random, then inserts the next `p.insert` rows of `inserts`. Each new vector is
/// the nearest row of `pool` to the old one that no upsert has used yet.
fn rounds(
    data: &Fbin,
    metric: Metric,
    base: &[u32],
    inserts: &[u32],
    pool: &[u32],
    p: &Params,
    rng: &mut Rng,
) -> Result<Vec<Round>, Error> {
    let mut live: Vec<u64> = base.iter().map(|&r| u64::from(r)).collect();
    // The row that each id given a new vector holds now. Every other id holds its own row.
    let mut now: HashMap<u64, u32> = HashMap::new();
    let mut used = vec![false; pool.len()];
    let mut out = Vec::with_capacity(p.rounds);

    for r in 0..p.rounds {
        let n = live.len();
        let deletes = ((p.delete * n as f64).round() as usize).min(n);
        let upserts = ((p.upsert * n as f64).round() as usize).min(n - deletes);
        let deleted: Vec<u64> = (0..deletes)
            .map(|_| live.swap_remove(rng.below(live.len() as u64) as usize))
            .collect();

        // A partial shuffle draws the upserted ids, all different, to the front.
        for i in 0..upserts {
            let j = i + rng.below((live.len() - i) as u64) as usize;
            live.swap(i, j);
        }

        let ids = &live[..upserts];
        let old: Vec<u32> = ids
            .iter()
            .map(|id| now.get(id).copied().unwrap_or(*id as u32))
            .collect();
        let rows = moved(data, metric, &old, pool, &mut used, p.threads, rng)?;
        let upserted: Vec<(u64, u32)> = ids.iter().copied().zip(rows).collect();

        for &(id, row) in &upserted {
            now.insert(id, row);
        }

        for id in &deleted {
            now.remove(id);
        }

        let added = inserts[r * p.insert..(r + 1) * p.insert].to_vec();
        live.extend(added.iter().map(|&row| u64::from(row)));
        out.push(Round {
            inserts: added,
            deletes: deleted,
            upserts: upserted,
        });
    }

    Ok(out)
}

/// The new row for each vector of `old`, rows of `data`. Taking them in a random order, each gets the nearest row of
/// `pool` that `used` doesn't mark, among its 32 nearest, or else a random unused one. Each row taken is marked.
fn moved(
    data: &Fbin,
    metric: Metric,
    old: &[u32],
    pool: &[u32],
    used: &mut [bool],
    threads: usize,
    rng: &mut Rng,
) -> Result<Vec<u32>, Error> {
    if old.is_empty() {
        return Ok(Vec::new());
    }

    let free: Vec<(u64, u32)> = (0..pool.len())
        .filter(|&i| !used[i])
        .map(|i| (i as u64, pool[i]))
        .collect();

    if free.len() < old.len() {
        return Err(invalid("too few held-out rows are left for the upserts' new vectors"));
    }

    let mut queries = Vec::with_capacity(old.len() * data.dim());

    for &r in old {
        queries.extend_from_slice(data.row(r as usize));
    }

    let near = nearest(data, metric, &queries, &free, NEAREST, threads);
    let mut order: Vec<usize> = (0..old.len()).collect();
    rng.shuffle(&mut order);
    let mut out = vec![0u32; old.len()];

    for i in order {
        let pick = match near[i].iter().find(|&&(_, at)| !used[at as usize]) {
            Some(&(_, at)) => at as usize,
            None => {
                let left: Vec<usize> = (0..pool.len()).filter(|&j| !used[j]).collect();
                left[rng.below(left.len() as u64) as usize]
            }
        };
        used[pick] = true;
        out[i] = pool[pick];
    }

    Ok(out)
}

/// Vectors held in memory one after another, as a source for k-means.
struct Rows {
    dim: usize,
    rows: Vec<f32>,
}

impl VectorSource for Rows {
    fn count(&self) -> usize {
        self.rows.len() / self.dim
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn rows(&self, start: usize, rows: usize) -> &[f32] {
        &self.rows[start * self.dim..(start + rows) * self.dim]
    }
}

/// Writes `stream` to the file `path`. It writes a temporary file next to it and renames it, so a crash never leaves
/// half a file.
pub fn write(path: &Path, stream: &Stream) -> Result<(), Error> {
    let tmp = path.with_extension("tmp");
    let mut out = BufWriter::with_capacity(1 << 20, File::create(&tmp)?);
    out.write_all(&MAGIC)?;
    out.write_all(&VERSION.to_le_bytes())?;
    out.write_all(&count(stream.rounds.len() + 1)?.to_le_bytes())?;
    let base = Round {
        inserts: stream.base.clone(),
        ..Round::default()
    };

    for round in std::iter::once(&base).chain(&stream.rounds) {
        for n in [round.inserts.len(), round.deletes.len(), round.upserts.len()] {
            out.write_all(&count(n)?.to_le_bytes())?;
        }

        for row in &round.inserts {
            out.write_all(&row.to_le_bytes())?;
        }

        for id in &round.deletes {
            out.write_all(&id.to_le_bytes())?;
        }

        for (id, row) in &round.upserts {
            out.write_all(&id.to_le_bytes())?;
            out.write_all(&row.to_le_bytes())?;
        }
    }

    out.into_inner().map_err(|e| e.into_error())?.sync_all()?;
    std::fs::rename(&tmp, path)
}

/// A count as the file's u32.
fn count(n: usize) -> Result<u32, Error> {
    u32::try_from(n).map_err(|_| invalid("a stream file holds fewer than 2^32 items of each kind"))
}

/// Reads the stream file at `path`. A file that isn't one, or whose base round holds deletes or upserts, fails with
/// `InvalidData`.
pub fn read(path: &Path) -> Result<Stream, Error> {
    let file = File::open(path)?;
    let mut left = file.metadata()?.len();
    let mut input = BufReader::with_capacity(1 << 20, file);
    let damaged = |what: &str| Error::new(ErrorKind::InvalidData, format!("{} {what}", path.display()));
    let mut head = [0u8; 16];
    input
        .read_exact(&mut head)
        .map_err(|_| damaged("is too short for a stream file"))?;
    left -= 16;

    if head[..8] != MAGIC || u32_at(&head, 8) != VERSION {
        return Err(damaged("isn't a stream file of version 1"));
    }

    let rounds = u32_at(&head, 12) as usize;

    if rounds == 0 {
        return Err(damaged("holds no base"));
    }

    // Each read is checked against what is left of the file first, so a damaged count can't ask for a huge buffer.
    let mut read = |len: usize| -> Result<Vec<u8>, Error> {
        if len as u64 > left {
            return Err(damaged("ends inside a round"));
        }

        let mut buf = vec![0u8; len];
        input.read_exact(&mut buf).map_err(|_| damaged("ends inside a round"))?;
        left -= len as u64;
        Ok(buf)
    };
    let mut all = Vec::with_capacity(rounds);

    for _ in 0..rounds {
        let counts = read(12)?;
        let (inserts, deletes, upserts) = (
            u32_at(&counts, 0) as usize,
            u32_at(&counts, 4) as usize,
            u32_at(&counts, 8) as usize,
        );
        let rows = read(4 * inserts)?;
        let ids = read(8 * deletes)?;
        let pairs = read(12 * upserts)?;
        all.push(Round {
            inserts: (0..inserts).map(|i| u32_at(&rows, 4 * i)).collect(),
            deletes: (0..deletes).map(|i| u64_at(&ids, 8 * i)).collect(),
            upserts: (0..upserts)
                .map(|i| (u64_at(&pairs, 12 * i), u32_at(&pairs, 12 * i + 8)))
                .collect(),
        });
    }

    if left > 0 {
        return Err(damaged("goes on past its last round"));
    }

    let base = all.remove(0);

    if !base.deletes.is_empty() || !base.upserts.is_empty() {
        return Err(damaged("has deletes or upserts in its base"));
    }

    Ok(Stream {
        base: base.inserts,
        rounds: all,
    })
}

fn u32_at(bytes: &[u8], at: usize) -> u32 {
    bytes[at..].first_chunk().map_or(0, |b| u32::from_le_bytes(*b))
}

fn u64_at(bytes: &[u8], at: usize) -> u64 {
    bytes[at..].first_chunk().map_or(0, |b| u64::from_le_bytes(*b))
}

/// What a change does to an index: it puts the vector of a dataset row under an id, or deletes an id's vector.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Change {
    Put { id: u64, row: u32 },
    Delete { id: u64 },
}

/// The dataset row that an id holds when its vector is deleted.
pub const GONE: u32 = u32::MAX;

/// A stream applied a round at a time. It gives each vector its id in the index and keeps the dataset row of each
/// live one. The base's vectors get the ids 0 to n − 1 in order, as a build gives them, and each inserted vector the
/// next unused id.
#[derive(Clone, Debug)]
pub struct Replay {
    /// The index id of each stream id.
    index_of: HashMap<u64, u64>,
    /// The dataset row that each index id holds, or `GONE`.
    rows: Vec<u32>,
    live: usize,
}

impl Replay {
    /// The index of a fresh build of `base`. A row named twice fails with `InvalidData`.
    pub fn new(base: &[u32]) -> Result<Replay, Error> {
        let index_of: HashMap<u64, u64> = base
            .iter()
            .enumerate()
            .map(|(i, &r)| (u64::from(r), i as u64))
            .collect();

        if index_of.len() != base.len() || base.contains(&GONE) {
            return Err(Error::new(
                ErrorKind::InvalidData,
                "the base names a row twice, or a row past the last one",
            ));
        }

        Ok(Replay {
            index_of,
            rows: base.to_vec(),
            live: base.len(),
        })
    }

    /// The changes that `round` makes to the index, in order: its deletes, then its upserts, then its inserts. Deleting
    /// a vector that is gone already does nothing. An insert of a row that some id was first inserted with puts it
    /// under that id again, and so does an upsert of a deleted id. A round that deletes or upserts an id that was never
    /// inserted fails with `InvalidData`, and changes nothing.
    pub fn apply(&mut self, round: &Round) -> Result<Vec<Change>, Error> {
        let unknown = |id: u64| {
            Error::new(
                ErrorKind::InvalidData,
                format!("the stream changes the id {id}, which it never inserted"),
            )
        };

        if let Some(&id) = round
            .deletes
            .iter()
            .chain(round.upserts.iter().map(|(id, _)| id))
            .find(|id| !self.index_of.contains_key(id))
        {
            return Err(unknown(id));
        }

        if round.inserts.contains(&GONE) || round.upserts.iter().any(|&(_, row)| row == GONE) {
            return Err(Error::new(
                ErrorKind::InvalidData,
                "the stream names a row past the last one",
            ));
        }

        let mut changes = Vec::with_capacity(round.deletes.len() + round.upserts.len() + round.inserts.len());

        for id in &round.deletes {
            let id = self.index_of[id];

            if self.rows[id as usize] != GONE {
                self.rows[id as usize] = GONE;
                self.live -= 1;
                changes.push(Change::Delete { id });
            }
        }

        for &(id, row) in &round.upserts {
            changes.push(self.put(self.index_of[&id], row));
        }

        for &row in &round.inserts {
            let id = match self.index_of.get(&u64::from(row)) {
                Some(&id) => id,
                None => {
                    let id = self.rows.len() as u64;
                    self.index_of.insert(u64::from(row), id);
                    self.rows.push(GONE);
                    id
                }
            };
            changes.push(self.put(id, row));
        }

        Ok(changes)
    }

    fn put(&mut self, id: u64, row: u32) -> Change {
        if self.rows[id as usize] == GONE {
            self.live += 1;
        }

        self.rows[id as usize] = row;
        Change::Put { id, row }
    }

    /// The dataset row that index id `id` holds, or None when its vector is gone or it was never given.
    pub fn row(&self, id: u64) -> Option<u32> {
        self.rows.get(id as usize).copied().filter(|&r| r != GONE)
    }

    /// Every live vector as its index id and dataset row, by id.
    pub fn live(&self) -> Vec<(u64, u32)> {
        let rows = self.rows.iter().enumerate();
        rows.filter(|&(_, &r)| r != GONE)
            .map(|(id, &r)| (id as u64, r))
            .collect()
    }

    /// How many vectors are live.
    pub fn len(&self) -> usize {
        self.live
    }

    pub fn is_empty(&self) -> bool {
        self.live == 0
    }
}

fn invalid(message: &str) -> Error {
    Error::new(ErrorKind::InvalidInput, message.to_string())
}

/// A seeded generator of random numbers, SplitMix64. The same seed gives the same numbers on every platform.
#[derive(Clone, Debug)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A number below `n`, each as likely as the others. `n` must be above 0.
    pub fn below(&mut self, n: u64) -> u64 {
        // Lemire's method: the high half of a 128-bit product, redrawn when the low half falls in the few values that
        // would make some results likelier.
        let floor = n.wrapping_neg() % n;

        loop {
            let m = u128::from(self.next_u64()) * u128::from(n);

            if m as u64 >= floor {
                return (m >> 64) as u64;
            }
        }
    }

    /// Puts `items` in a random order, each order as likely as the others.
    pub fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            items.swap(i, self.below(i as u64 + 1) as usize);
        }
    }
}

#[cfg(test)]
#[path = "../tests/unit/stream.rs"]
mod tests;
