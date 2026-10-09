//! Maintenance, which folds an index's changes into a new data file and clusters again the parts that drifted. A run
//! takes a snapshot of the changes, plans the new file's clusters from the old file and the snapshot, then writes the
//! new file as the plan says, as the index's next generation. It runs on the writer's thread and blocks it until the
//! commit makes the new generation current.
//!
//! There are three kinds of run:
//! - A fold keeps every cluster and its center. It drops the deleted vectors and adds the inserted ones to the clusters
//!   they were routed to, so it measures no code again.
//! - A minor run clusters again the regions that are due, each a set of nearby clusters, with k-means over their
//!   vectors. Then it moves each of their vectors to its nearest cluster of the whole file, and the vectors of nearby
//!   clusters to a new cluster when it is nearer. Clusters elsewhere keep their centers, and their codes are copied.
//! - A major run clusters every vector again with k-means over the whole file.
//!
//! The vectors exist only as 8-bit rows, so k-means runs on the decoded rows, and a vector whose cluster or center
//! changed gets its code measured again from its row. Every row is copied unchanged, and the mean and the rotation
//! never change.

pub(crate) mod major;
pub(crate) mod minor;
pub(crate) mod regions;
pub(crate) mod rows;

use std::collections::HashSet;
use std::fs::File;
use std::io::{Error, ErrorKind};
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use rayon::{ThreadPool, ThreadPoolBuilder};

use crate::build::Slice;
use crate::build::kmeans::{self, KMeans};
use crate::build::order::leaf_order;
use crate::build::rewrite::{GONE, Plan, PlanCenter, PlanGroups, PlanList, rewrite_into};
use crate::format::{Layout, Part, Tables, Trailer};
use crate::index::Inner;
use crate::io;
use crate::quantization::rabitq::{Encoder, code_bytes};
use crate::quantization::{padded_dim, rotate_offset, sq4, sq8};
use crate::root::{self, Next, Owner};
use crate::search::{LIMIT, invalid};
use crate::update::{Delta, Log, NONE, Record, Seen, Version, log};
use crate::{Anchor, Clusters, Limits, MaintainParams, Metric, RunKind, RunReport, SearchCtx, Status};
use regions::Regions;
use rows::damaged_row;

/// The rows that k-means reads at a time in a run. Its buffers hold this many rows as f32, and their distances to
/// 1,024 centers.
pub(crate) const CHUNK: usize = 4096;

/// The most bytes of a section that a run reads at once to check it whole.
const PIECE: usize = 1 << 20;

/// A run's snapshot: the changes it folds in, and where the log ended then, so the changes logged after it move to the
/// new generation's log.
pub(crate) struct Snapshot {
    pub(crate) seen: Seen,
    pub(crate) log_at: usize,
}

/// The vectors that a run keeps, as of its snapshot.
pub(crate) struct Live {
    /// Whether each row of the old file is visible, by internal id.
    pub(crate) rows: Vec<bool>,
    /// The visible inserts, by place in the changes.
    pub(crate) inserts: Vec<u32>,
    /// Each old cluster's visible rows, its visible inserts, its rows that a change hid, and how many of those a
    /// visible insert of the same id replaced.
    pub(crate) kept: Vec<u32>,
    pub(crate) added: Vec<u32>,
    pub(crate) deleted: Vec<u32>,
    pub(crate) replaced: Vec<u32>,
}

impl Live {
    /// The vectors of `old` and of `delta` visible as of `seen`. With any delete, it reads the old file's ids section
    /// for each row's id and checks it against the header. Returns the bytes it read too.
    pub(crate) fn of(old: &Inner, delta: &Delta, seen: Seen) -> Result<(Live, u64), Error> {
        let (layout, tables) = (old.layout(), old.tables()?);
        let dim = layout.dim;
        let inserts: Vec<u32> = (0..seen.inserts())
            .filter(|&i| delta.is_live(sq8::id(delta.row(i), dim), delta.seq(i), seen))
            .map(|i| i as u32)
            .collect();
        let (mut rows, mut replaced_rows) = (vec![true; layout.count], Vec::new());
        let mut read = 0;

        if !delta.hides_nothing() {
            let ids: HashSet<u64> = inserts.iter().map(|&i| sq8::id(delta.row(i as usize), dim)).collect();
            replaced_rows = vec![false; layout.count];
            read = visible_rows(old, delta, seen, &ids, &mut rows, &mut replaced_rows)?;
        }

        let nlist = tables.lists.len();
        let (mut kept, mut added, mut deleted) = (vec![0u32; nlist], vec![0u32; nlist], vec![0u32; nlist]);
        let mut replaced = vec![0u32; nlist];

        for (j, list) in tables.lists.iter().enumerate() {
            let rows_of = list.first_id as usize..list.first_id as usize + list.count;
            kept[j] = rows[rows_of.clone()].iter().filter(|&&v| v).count() as u32;
            deleted[j] = list.count as u32 - kept[j];
            replaced[j] = replaced_rows
                .get(rows_of)
                .map_or(0, |r| r.iter().filter(|&&v| v).count() as u32);
        }

        for &i in &inserts {
            added[delta.leaf(i as usize) as usize] += 1;
        }

        Ok((
            Live {
                rows,
                inserts,
                kept,
                added,
                deleted,
                replaced,
            },
            read,
        ))
    }

    /// Each old cluster's vector count as of the snapshot.
    pub(crate) fn sizes(&self) -> Vec<usize> {
        self.kept
            .iter()
            .zip(&self.added)
            .map(|(&k, &a)| (k + a) as usize)
            .collect()
    }
}

/// Marks the rows of `old` that a change in `delta` hid as of `seen`, and among them those whose id is in `inserted`,
/// the ids of the visible inserts. It reads each row's id from the ids section through the cold handle, and checks the
/// section against the header. Returns the bytes read.
fn visible_rows(
    old: &Inner,
    delta: &Delta,
    seen: Seen,
    inserted: &HashSet<u64>,
    rows: &mut [bool],
    replaced: &mut [bool],
) -> Result<u64, Error> {
    read_part(old, Part::Ids, |from, piece| {
        let first = from / 8;
        for (k, id) in piece.as_chunks::<8>().0.iter().enumerate() {
            let (id, r) = (u64::from_le_bytes(*id), first + k);
            if r < rows.len() && !delta.is_live(id, 0, seen) {
                rows[r] = false;
                replaced[r] = inserted.contains(&id);
            }
        }
    })
}

/// Reads section `part` of `old` through the cold handle, `PIECE` bytes at a time, hands each piece to `each` with its
/// offset in the section, then checks the section against the header. Returns the bytes read.
fn read_part(old: &Inner, part: Part, mut each: impl FnMut(usize, &[u8])) -> Result<u64, Error> {
    let span = old.layout().span(part);
    let (mut crc, mut buf) = (crc32fast::Hasher::new(), Vec::new());

    for at in (span.offset..span.end()).step_by(PIECE) {
        buf.resize(PIECE.min(span.end() - at), 0);
        io::read_at(old.cold_file(), &mut buf, at)?;
        crc.update(&buf);
        each(at - span.offset, &buf);
    }

    crate::format::check_section(old.layout(), part, crc.finalize())?;
    Ok(span.len as u64)
}

/// What every planner reads: the old file, the snapshot, the vectors that stay and the settings.
pub(crate) struct Ctx<'a> {
    pub(crate) old: &'a Inner,
    pub(crate) delta: &'a Delta,
    pub(crate) seen: Seen,
    pub(crate) live: &'a Live,
    pub(crate) tables: &'a Tables,
    /// The old clusters' centers as they decode, one after another in file order, in the vectors' space.
    pub(crate) centers: &'a [f32],
    pub(crate) params: &'a MaintainParams,
    pub(crate) pool: &'a ThreadPool,
    /// The seed of this run's k-means, from the settings' seed and the old file's epoch.
    pub(crate) seed: u64,
}

impl Ctx<'_> {
    pub(crate) fn dim(&self) -> usize {
        self.old.dim()
    }

    /// The center `c` as the file would store and decode it: P(c − μ) to 4 bits and back, where P is the rotation and
    /// μ the mean. A center too far from the mean to encode fails with `InvalidInput`.
    pub(crate) fn as_stored(&self, c: &[f32]) -> Result<Vec<f32>, Error> {
        let (dim, old) = (self.dim(), self.old);
        let (mut z, mut tmp) = (vec![0f32; dim], vec![0f32; dim]);
        let mut codes = vec![0u8; sq4::code_bytes(padded_dim(dim))];
        rotate_offset(old.rotation(), c, old.mean(), &mut z, &mut tmp);
        let f = sq4::encode(&z, &mut codes);

        if !(f64::from(f.norm2) <= LIMIT && f.lo.is_finite() && f.step.is_finite()) {
            return Err(invalid("a cluster center is too far from the mean to encode"));
        }

        Ok(decode_center(old, &codes, &f))
    }

    /// The vectors that stay in old cluster `j`: its visible rows by internal id, then its visible inserts.
    pub(crate) fn members_of_old(&self) -> Vec<Vec<u32>> {
        let count = self.old.layout().count as u32;
        let mut members: Vec<Vec<u32>> = self
            .tables
            .lists
            .iter()
            .map(|l| {
                (l.first_id..l.first_id + l.count as u32)
                    .filter(|&r| self.live.rows[r as usize])
                    .collect()
            })
            .collect();

        for &i in &self.live.inserts {
            members[self.delta.leaf(i as usize) as usize].push(count + i);
        }

        members
    }

    /// The vector id of each visible vector, in increasing order: file rows by internal id, then inserts.
    pub(crate) fn live_ids(&self) -> Vec<u32> {
        let count = self.old.layout().count as u32;
        let rows = (0..count).filter(|&r| self.live.rows[r as usize]);
        rows.chain(self.live.inserts.iter().map(|&i| count + i)).collect()
    }
}

/// The center that `codes` and `f` decode to, in the vectors' space: μ + P⁻¹ẑ, where ẑ is the decoded rotated offset.
pub(crate) fn decode_center(old: &Inner, codes: &[u8], f: &sq4::CenterFactors) -> Vec<f32> {
    let dim = old.dim();
    let (mut c, mut tmp) = (vec![0f32; dim], vec![0f32; dim]);
    sq4::decode(codes, f, &mut c);
    old.rotation().invert_in_place(&mut c, &mut tmp);
    c.iter_mut().zip(old.mean()).for_each(|(x, &m)| *x += m);
    c
}

/// Every old cluster's center as it decodes, one after another in file order, in the vectors' space.
fn decoded_centers(old: &Inner, tables: &Tables) -> Vec<f32> {
    let cb = sq4::code_bytes(padded_dim(old.dim()));
    let centers = tables.codes.chunks_exact(cb).zip(&tables.factors);
    centers.flat_map(|(codes, f)| decode_center(old, codes, f)).collect()
}

/// Checks the settings that a run uses.
fn check(params: &MaintainParams) -> Result<(), Error> {
    let (m, j) = (&params.minor, &params.major);
    let shares = [m.change, m.shrink, m.hot_list, m.list_loss, m.split, m.merge, j.beta];
    let optional = [j.inserts, j.deletes, j.indicator, m.pool, m.global_pass];

    if params.vectors_per_list == 0 || params.rounds == 0 || params.parents > usize::from(u16::MAX) {
        return Err(invalid(
            "maintenance needs at least one vector per cluster, one round and at most 65535 groups",
        ));
    }

    if !shares
        .iter()
        .chain(optional.iter().flatten())
        .all(|x| x.is_finite() && *x >= 0.0)
    {
        return Err(invalid("every maintenance threshold must be finite and at least 0"));
    }

    if m.shrink >= 1.0 || m.list_loss > 1.0 || m.pool.is_some_and(|q| q > 1.0) {
        return Err(invalid(
            "a region can't shrink by its whole count, and a pooling share is at most 1",
        ));
    }

    Ok(())
}

/// The pool of a run's threads, `threads` of them, or with 0 `RAYON_NUM_THREADS` when it is set and otherwise 2.
fn pool(threads: usize) -> Result<ThreadPool, Error> {
    let env = std::env::var("RAYON_NUM_THREADS")
        .ok()
        .and_then(|n| n.parse::<usize>().ok());
    let n = match threads {
        0 => env.filter(|&n| n > 0).unwrap_or(2),
        n => n,
    };
    ThreadPoolBuilder::new()
        .num_threads(n)
        .thread_name(|i| format!("coldivf-maintain-{i}"))
        .build()
        .map_err(Error::other)
}

/// The seed of a run's k-means on `old`: the settings' seed, or with 0 the file's seed plus its epoch.
fn seed_of(params: &MaintainParams, old: &Inner) -> u64 {
    match params.seed {
        0 => old.layout().seed.wrapping_add(old.layout().epoch),
        seed => seed,
    }
}

/// Plans a run of `kind` for the open index `old` and writes its new data file into `out`, the empty file at
/// `out_path`, as of a snapshot that it takes first. The changes made after the snapshot stay in the log. Returns the
/// snapshot and what the run did.
pub(crate) fn run(
    old: &Inner,
    out: &File,
    out_path: &Path,
    params: &MaintainParams,
    kind: RunKind,
) -> Result<(Snapshot, RunReport), Error> {
    check(params)?;
    let (started, cpu) = (Instant::now(), io::thread_cpu());
    let pool = pool(params.threads)?;
    let delta = old.changes().read();
    let snapshot = Snapshot {
        seen: delta.seen(),
        log_at: delta.log_len(),
    };
    let (live, mut read) = Live::of(old, &delta, snapshot.seen)?;
    let tables = old.tables()?;
    let centers = decoded_centers(old, tables);
    let seed = seed_of(params, old);
    let ctx = Ctx {
        old,
        delta: &delta,
        seen: snapshot.seen,
        live: &live,
        tables,
        centers: &centers,
        params,
        pool: &pool,
        seed,
    };
    let mut report = RunReport::default();
    let regions = match trailer_of(old)? {
        Some(t) => Regions::carried(t, &live),
        None => {
            let counts: Vec<usize> = tables.lists.iter().map(|l| l.count).collect();
            let (r, distances) = Regions::made(&centers, ctx.dim(), &counts, &live, params, seed, &pool)?;
            report.distances += distances;
            r
        }
    };
    report.kind = match kind {
        RunKind::Auto if major_due(&regions, params) => RunKind::Major,
        RunKind::Auto => RunKind::Minor,
        kind => kind,
    };
    let plan = match plan_of(&ctx, regions, &mut report) {
        // The centers of a plan are means of rows, and each row passed the range checks of a build or an insert, so
        // a center out of range means a damaged row, unless the rows section checks out.
        Err(e) if e.kind() == ErrorKind::InvalidInput => {
            read_part(old, Part::Rows, |_, _| {})?;
            return Err(e);
        }
        planned => planned?,
    };
    report.moved = moved(&ctx, &plan);
    read += report.bytes_read;
    let done = rewrite_into(old, &delta, snapshot.seen, &plan, out, out_path, &pool)?;
    report.bytes_read = read + done.bytes_read;
    report.bytes_written = done.bytes_written;
    (report.reencoded, report.copied, report.count, report.lists) =
        (done.reencoded, done.copied, done.count, done.nlist);
    report.regions = plan.trailer.as_ref().map_or(0, |t| t.region_last.len());
    let threads: Duration = pool.broadcast(|_| io::thread_cpu()).iter().sum();
    report.cpu = io::thread_cpu().saturating_sub(cpu) + threads;
    report.wall = started.elapsed();
    Ok((snapshot, report))
}

/// The plan of a run of `report.kind`, from the old file's `regions`. A minor run with nothing to do becomes a fold.
fn plan_of(ctx: &Ctx, regions: Regions, report: &mut RunReport) -> Result<Plan, Error> {
    match report.kind {
        RunKind::Minor => {
            let work = minor::Work::of(ctx, &regions, report)?;
            report.regions_due = work.due.len();
            match work.is_empty() {
                true => {
                    report.kind = RunKind::Fold;
                    Ok(fold(ctx, &regions))
                }
                false => minor::plan(ctx, regions, &work, report),
            }
        }
        RunKind::Major => major::plan(ctx, report),
        _ => Ok(fold(ctx, &regions)),
    }
}

/// The trailer of `old`, or `None` when it has none, eg. after a fresh build, or a damaged one. It holds only the
/// regions and the counts that say when runs are due, and no section that a run copies, so a run then makes the
/// regions again as for a fresh build and writes a whole trailer.
fn trailer_of(old: &Inner) -> Result<Option<Trailer>, Error> {
    match old.trailer() {
        Err(e) if e.kind() == ErrorKind::InvalidData => Ok(None),
        trailer => trailer,
    }
}

/// Whether a major run is due, i.e. whether the inserts since the last one reached `params.major.inserts` of the count
/// then, or the deletes `params.major.deletes` of it. Runs don't measure the read-cost indicator, so it never fires.
pub(crate) fn major_due(regions: &Regions, params: &MaintainParams) -> bool {
    // The shares are compared in f32, the type they are given in, as in the regions' due marks.
    let (base, j) = (regions.major_count.max(1) as f32, &params.major);
    let inserted = j.inserts.is_some_and(|f| regions.major_inserts as f32 >= f * base);
    let deleted = j.deletes.is_some_and(|f| regions.major_deletes as f32 >= f * base);
    inserted || deleted
}

/// What maintenance would do now with `params`.
pub(crate) fn status(old: &Inner, params: &MaintainParams) -> Result<Status, Error> {
    let delta = old.changes().read();
    let seen = delta.seen();
    let (live, _) = Live::of(old, &delta, seen)?;
    let tables = old.tables()?;
    let regions = match trailer_of(old)? {
        Some(t) => Regions::carried(t, &live),
        None => {
            let (centers, counts) = (
                decoded_centers(old, tables),
                tables.lists.iter().map(|l| l.count).collect::<Vec<_>>(),
            );
            let pool = pool(params.threads)?;
            Regions::made(&centers, old.dim(), &counts, &live, params, seed_of(params, old), &pool)?.0
        }
    };

    Ok(Status {
        pending: live.inserts.len(),
        deleted: live.deleted.iter().map(|&n| n as usize).sum(),
        regions_due: regions.due(&live.sizes(), &params.minor).len(),
        major_due: major_due(&regions, params),
    })
}

/// The plan of a fold: every old cluster that still holds a vector, with its center, in the old order, and the old
/// groups that still hold a cluster. Inserts join the clusters they were routed to, so every code is copied.
pub(crate) fn fold(ctx: &Ctx, regions: &Regions) -> Plan {
    let (tables, live, delta) = (ctx.tables, ctx.live, ctx.delta);
    let nlist = tables.lists.len();
    let sizes = live.sizes();
    let kept: Vec<usize> = (0..nlist).filter(|&j| sizes[j] > 0).collect();
    let mut new_of = vec![GONE; nlist];
    kept.iter().enumerate().for_each(|(l, &j)| new_of[j] = l as u32);
    let mut used = vec![false; ctx.old.layout().nparents];
    kept.iter().for_each(|&j| used[usize::from(tables.parent_of[j])] = true);
    let groups: Vec<u16> = (0..used.len() as u16).filter(|&g| used[usize::from(g)]).collect();
    let group_of = |g: u16| groups.partition_point(|&x| x < g) as u16;
    let mut file_rows = vec![GONE; ctx.old.layout().count];

    for (j, list) in tables.lists.iter().enumerate() {
        let rows = list.first_id as usize..list.first_id as usize + list.count;
        let placed = file_rows[rows.clone()].iter_mut().zip(&live.rows[rows]);
        placed.filter(|(_, live)| **live).for_each(|(r, _)| *r = new_of[j]);
    }

    let mut delta_rows = vec![GONE; ctx.seen.inserts()];
    live.inserts
        .iter()
        .for_each(|&i| delta_rows[i as usize] = new_of[delta.leaf(i as usize) as usize]);
    let lists = kept
        .iter()
        .map(|&j| PlanList {
            center: PlanCenter::Keep(j as u32),
            group: group_of(tables.parent_of[j]),
        })
        .collect();
    let new_lists: Vec<(u32, u32)> = kept.iter().map(|&j| (regions.of[j], regions.list_last[j])).collect();

    Plan {
        lists,
        groups: PlanGroups::Keep(groups),
        file_rows,
        delta_rows,
        trailer: Some(trailer(regions, &regions.last, &regions.changes, &new_lists)),
    }
}

/// The trailer of a new file whose clusters, in file order, have the regions and counts when last clustered of
/// `lists`, as (region, count) pairs. The regions are numbered as in `last` and `changes`, and the ones that hold no
/// cluster drop out.
pub(crate) fn trailer(regions: &Regions, last: &[u64], changes: &[u64], lists: &[(u32, u32)]) -> Trailer {
    let old: Vec<u32> = lists.iter().map(|&(r, _)| r).collect();
    let region_of = regions::renumber(&old);
    let nregions = region_of.iter().map(|&r| r as usize + 1).max().unwrap_or(0);
    let (mut region_last, mut region_changes) = (vec![0u64; nregions], vec![0u64; nregions]);

    for (&r, &new) in old.iter().zip(&region_of) {
        (region_last[new as usize], region_changes[new as usize]) = (last[r as usize], changes[r as usize]);
    }

    Trailer {
        region_of,
        list_last: lists.iter().map(|&(_, n)| n).collect(),
        region_last,
        region_changes,
        per_region: regions.per_region,
        major_count: regions.major_count,
        major_inserts: regions.major_inserts,
        major_deletes: regions.major_deletes,
        pass_changes: regions.pass_changes,
    }
}

/// The file order of a run's clusters.
pub(crate) struct Ordered {
    /// The cluster at each file position, as its place in the centers given.
    pub(crate) file: Vec<u32>,
    /// Each cluster's group, by place, and the group centers one after another.
    pub(crate) parent_of: Vec<u32>,
    pub(crate) parents: Vec<f32>,
    /// The distances that the groups' k-means computed.
    pub(crate) distances: u64,
}

/// The groups that order clusters whose decoded centers are `centers`, from k-means into `params.parents` groups, or
/// as many as the old file has with 0, and the file order they give.
pub(crate) fn order(ctx: &Ctx, centers: Vec<f32>) -> Result<Ordered, Error> {
    let (dim, params) = (ctx.dim(), ctx.params);
    let nlist = centers.len() / dim;
    let wanted = match params.parents {
        0 => ctx.old.layout().nparents,
        n => n,
    };
    let k = wanted.clamp(1, nlist);
    let src = Slice { dim, data: &centers };
    let groups = KMeans {
        k,
        rounds: params.rounds,
        sample: usize::MAX,
        spherical: false,
        seed: ctx.seed.wrapping_add(1),
    };
    let parents = kmeans::train(&groups, &src, Metric::L2, ctx.pool)?;
    let parent_of = kmeans::assign(&parents, &src, Metric::L2, ctx.pool)?;
    let clusters = Clusters {
        dim,
        centroids: centers,
        assignment: Vec::new(),
        parents,
        parent_of,
    };
    let file = leaf_order(params.order, &clusters, ctx.old.mean());
    Ok(Ordered {
        file,
        parent_of: clusters.parent_of,
        parents: clusters.parents,
        distances: (nlist * k * (params.rounds + 1)) as u64,
    })
}

/// How many vectors the plan moves to another cluster. Every vector of a new cluster counts, since its cluster is new.
fn moved(ctx: &Ctx, plan: &Plan) -> usize {
    let tables = ctx.tables;
    let kept_from = |l: u32| match plan.lists[l as usize].center {
        PlanCenter::Keep(j) => Some(j as usize),
        PlanCenter::New(_) => None,
    };
    let mut moved = 0;

    for (j, list) in tables.lists.iter().enumerate() {
        let first = list.first_id as usize;
        let rows = &plan.file_rows[first..first + list.count];
        moved += rows.iter().filter(|&&l| l != GONE && kept_from(l) != Some(j)).count();
    }

    let inserts = plan.delta_rows.iter().enumerate().filter(|&(_, &l)| l != GONE);
    moved
        + inserts
            .filter(|&(i, &l)| kept_from(l) != Some(ctx.delta.leaf(i) as usize))
            .count()
}

/// Makes the generation that a run wrote into `next` current, as of the run's `snapshot` of the old generation's `log`,
/// and returns the new generation and its log. It makes the new log, syncs the folder, opens the new generation,
/// moves the changes logged after the snapshot into the new log and syncs it, then commits through the root and deletes
/// the old generation's files. On an error before the commit the old generation stays current, with every change, and
/// the new files go. On one in the commit's slot write, whether it committed is unknown, so `log` refuses every change
/// until the writer opens again.
pub(crate) fn commit(
    owner: &mut Owner,
    mut next: Next,
    path: &Path,
    old: &Layout,
    snapshot: &Snapshot,
    log: &mut Log,
) -> Result<(Arc<Inner>, Log), Error> {
    let layout = root::header_of(&next.data)?;
    let base = snapshot.seen.next();
    let mut new_log = Log::create(&next.log, next.version(base), &layout, &[], owner.lock())?;
    next.seal()?;
    io::step("sealed", &next.data);
    let inner = Arc::new(Inner::open_at(path, next.located(layout.check, base))?);
    // The first searches of the new generation find its cluster tables loaded.
    inner.tables()?;
    let later = log.records_after(snapshot.log_at)?;
    carry(&inner, &later, log.version(), old, base, &mut new_log)?;
    new_log.sync()?;
    io::step("carried", &next.data);

    let old = match owner.commit(&mut next, layout.check, base) {
        Ok(old) => old,
        Err(e) => {
            if owner.unsure() {
                log.poison();
            }
            return Err(e);
        }
    };
    owner.retire(&old);
    io::step("retired", &next.data);
    Ok((inner, new_log))
}

/// Appends the changes `records`, which a log of `from` holds for the data file whose header says `old` and which start
/// at sequence number `base`, to `to`, the log of `inner`, fit for its data file. A delete stays as it is. An insert is
/// routed again to its nearest cluster of the new file. With the cluster anchor its code is measured again from its
/// row, against that cluster's center, and with the mean anchor it stays. Its row stays as it is.
fn carry(
    inner: &Inner,
    records: &[Vec<u8>],
    from: Version,
    old: &Layout,
    base: u64,
    to: &mut Log,
) -> Result<(), Error> {
    let (dim, mean) = (inner.dim(), inner.mean());
    let mut ctx = SearchCtx::new(Limits {
        max_k: 1,
        max_nprobe: 1,
        max_rerank: 1,
    });
    let (mut o, mut center, mut leaves) = (vec![0f32; dim], vec![0f32; dim], Vec::with_capacity(1));
    let mut code = vec![0u8; code_bytes(padded_dim(dim))];
    let mut encoder = Encoder::new(inner.rotation(), mean);

    for (i, bytes) in records.iter().enumerate() {
        let seq = base + i as u64;
        let carried = match log::decode(bytes, from, old.dim, old.nlist, seq)? {
            Record::Delete { id, .. } => Record::Delete { seq, id, hides: NONE },
            Record::Insert {
                id,
                code: old_code,
                factors,
                row,
                ..
            } => {
                sq8::decode(row, mean, &mut o);
                if !o.iter().all(|x| x.is_finite()) {
                    return Err(damaged_row());
                }
                inner.route(&o, 1, None, &mut ctx, &mut leaves)?;
                let leaf = leaves.first().copied().unwrap_or(0);
                let factors = match inner.layout().anchor {
                    Anchor::Mean => {
                        code.copy_from_slice(old_code);
                        factors
                    }
                    Anchor::Cluster => {
                        let (codes, f) = inner.view(inner.tables()?).centers.center(leaf);
                        sq4::decode(codes, f, &mut center);
                        encoder.encode_from(&o, &center, &mut code)
                    }
                };
                Record::Insert {
                    seq,
                    id,
                    list: leaf,
                    hides: NONE,
                    code: &code,
                    factors,
                    row,
                }
            }
        };
        to.write(inner.changes(), &log::encode(&carried, to.version()))?;
    }

    Ok(())
}

#[cfg(test)]
#[path = "../../tests/unit/maintain/mod.rs"]
pub(crate) mod tests;
