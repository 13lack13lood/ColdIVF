//! Where the read waves of a cold search wait, on macOS. For each query it plans the reads that a cold search of it
//! makes, from the index's route and the rows that a search of it rescores. Those are the first 16 KiB, the cluster
//! tables, the cluster wave and the row wave, each wave's ranges rounded out to 16 KiB pages and joined as the backend
//! joins them. Each trial empties the file's cache, then replays those reads the way the macOS backend makes them,
//! through a handle with `F_NOCACHE`. Every range of a wave gets an `F_RDADVISE` hint, then a read in order, into new
//! memory as the backend's buffers are. The trial times each call on its own, so a slow wave shows which range waited,
//! and whether its hint or its read did.
//!
//! stall --index PATH --dataset NAME [--n N] --queries 1,3,18 [--nprobe 9] [--rerank 29] [--repeats 20]
//!       [--variant idle,prewarm] [--results results/stall.jsonl] [--label TEXT]
//!
//! The trials go through the queries in turn, `--repeats` times, and each query takes the variants in turn. `idle`
//! sleeps 200 ms before the timed part and `prewarm` spins 150 ms, as the cold harness does. A search of a query with
//! k = `--rerank` returns every candidate it rescores, and each hit's row sits in page i / r of the rows, where i is
//! the hit's position in the file, which the ids section maps from its id, and r is the rows per page.
//!
//! Each line holds one trial. `open_us` is the open of the handle, and `header`, `tables`, `clusters` and `rows` list
//! each read with its `offset`, `len`, `advise_us` and `read_us`. `cluster_wave_us` and `row_wave_us` are the time from
//! a wave's first hint to the end of its last read, and `pages_resident_at_start` counts the file's cached pages right
//! before the timer.

use std::collections::HashMap;
use std::fs::File;
use std::io::{Error, ErrorKind, Read, Seek, SeekFrom};
use std::ops::Range;
use std::path::{Path, PathBuf};
#[cfg(target_os = "macos")]
use std::time::{Duration, Instant};

use coldivf::{Index, LeafId, Limits, SearchCtx, SearchParams, data_file};
use coldivf_bench::args::{Args, fail};
use coldivf_bench::coldcache::Mapped;
use coldivf_bench::data::{Dataset, data_root, derived_root};
use coldivf_bench::metrics::{Json, append, interactive, provenance, state};

/// The page size of Apple silicon, which the backend rounds its reads out to.
const PAGE: usize = 16384;

/// How much of the file an open reads first, the header and, up to ~2,800 dimensions, the rotation.
#[cfg(target_os = "macos")]
const FIRST_READ: usize = 16384;

fn main() {
    let args = Args::parse();
    let name = args.need("dataset");
    let n = args
        .str("n")
        .map(|v| v.parse().unwrap_or_else(|_| fail("--n must be a number")));
    let dataset = Dataset::open_in(&data_root(), &derived_root(), name, n)
        .unwrap_or_else(|e| fail(&format!("opening {name}: {e}")));
    let path = PathBuf::from(args.need("index"));
    let data = data_file(&path).unwrap_or_else(|e| fail(&format!("finding the data file of {}: {e}", path.display())));
    let layout = Layout::read(&data).unwrap_or_else(|e| fail(&format!("reading {}: {e}", data.display())));
    let queries = args.list::<usize>("queries", &[]);
    let (nprobe, rerank, repeats) = (
        args.get("nprobe", 9),
        args.get("rerank", 29),
        args.get("repeats", 20usize),
    );
    let variants = args.list("variant", &["idle".to_string(), "prewarm".to_string()]);
    let results = args.str("results").unwrap_or("results/stall.jsonl");

    if queries.is_empty() || queries.iter().any(|&q| q >= dataset.queries.len()) {
        fail("--queries must list positions among the dataset's queries");
    }

    if variants.iter().any(|v| v != "idle" && v != "prewarm") {
        fail("--variant must list idle or prewarm");
    }

    let extents = layout
        .section(&data, &layout.extents)
        .unwrap_or_else(|e| fail(&format!("reading the extents of {}: {e}", data.display())));
    let ids = layout
        .section(&data, &layout.ids)
        .unwrap_or_else(|e| fail(&format!("reading the ids of {}: {e}", data.display())));
    let positions = positions(&ids);
    let plans: Vec<Plan> = queries
        .iter()
        .map(|&q| {
            let leaves_and_hits = searched(&path, dataset.queries.row(q), nprobe, rerank)
                .unwrap_or_else(|e| fail(&format!("planning query {q}: {e}")));
            layout
                .plan(&extents, &positions, &leaves_and_hits.0, &leaves_and_hits.1)
                .unwrap_or_else(|e| fail(&format!("planning query {q}: {e}")))
        })
        .collect();
    let map = Mapped::open(&data).unwrap_or_else(|e| fail(&format!("mapping {}: {e}", data.display())));
    let tail = state(provenance(Json::new())).line();
    interactive();

    for repeat in 0..repeats {
        for (i, (&q, plan)) in queries.iter().zip(&plans).enumerate() {
            let variant = &variants[(repeat + i) % variants.len()];
            let line = Json::new()
                .str("label", args.str("label").unwrap_or(""))
                .str("index", &path.display().to_string())
                .str("dataset", name)
                .int("query", q as u64)
                .int("repeat", repeat as u64)
                .str("variant", variant)
                .int("nprobe", nprobe as u64)
                .int("rerank", rerank as u64)
                .int("cluster_ranges", plan.clusters.len() as u64)
                .int("row_ranges", plan.rows.len() as u64)
                .int(
                    "cluster_bytes",
                    plan.clusters.iter().map(Range::len).sum::<usize>() as u64,
                )
                .int("row_bytes", plan.rows.iter().map(Range::len).sum::<usize>() as u64);
            let line = trial(&data, &map, plan, variant, line);
            let line = format!("{},{}", line.line().trim_end_matches('}'), tail.trim_start_matches('{'));
            append(Path::new(results), &line).unwrap_or_else(|e| fail(&format!("writing {results}: {e}")));
        }

        eprintln!("stall: {} of {repeats} repeats", repeat + 1);
    }
}

/// The clusters that a search of `query` reads and the ids of its `rerank` best candidates, from a search with k =
/// `rerank` of the index at `path`, which returns every candidate it rescores.
fn searched(path: &Path, query: &[f32], nprobe: usize, rerank: usize) -> Result<(Vec<LeafId>, Vec<u64>), Error> {
    let index = Index::open(path)?;
    let mut ctx = SearchCtx::new(Limits {
        max_k: rerank,
        max_nprobe: nprobe,
        max_rerank: rerank,
    });
    let mut leaves = Vec::new();
    index.route(query, nprobe, None, &mut ctx, &mut leaves)?;
    let params = SearchParams {
        k: rerank,
        nprobe,
        rerank,
        reach: None,
    };
    let hits = index.search(query, &params, &mut ctx)?;
    Ok((leaves, hits.iter().map(|h| h.id).collect()))
}

/// Each id's position in the file, from `ids`, the bytes of the ids section, which holds the id at each position as a
/// u64.
fn positions(ids: &[u8]) -> HashMap<u64, usize> {
    let ids = ids.as_chunks::<8>().0.iter().map(|b| u64::from_le_bytes(*b));
    ids.enumerate().map(|(at, id)| (id, at)).collect()
}

/// The reads of one query's cold search, apart from the first 16 KiB.
#[derive(Debug, PartialEq)]
struct Plan {
    /// The cluster tables, read as they are, without rounding.
    tables: Range<usize>,
    clusters: Vec<Range<usize>>,
    rows: Vec<Range<usize>>,
}

/// Where the parts of an index file sit, from its header.
#[derive(Clone, Debug, PartialEq)]
struct Layout {
    /// The file's size.
    len: usize,
    dim: usize,
    /// The dimension padded to a multiple of 64, which sizes a block of codes.
    padded: usize,
    /// How many groups the clusters fall in. Their centers start the centers section.
    groups: usize,
    /// The size of a page of rows, and how many rows a page holds.
    page_size: usize,
    per_page: usize,
    centers: Range<usize>,
    extents: Range<usize>,
    codes: Range<usize>,
    rows: Range<usize>,
    ids: Range<usize>,
}

impl Layout {
    /// Reads the header of the index file at `path`. A file that isn't an index of format version 1 fails with
    /// `InvalidData`.
    fn read(path: &Path) -> Result<Layout, Error> {
        let mut file = File::open(path)?;
        let mut head = [0u8; 324];
        let damaged = || {
            Error::new(
                ErrorKind::InvalidData,
                format!("{} isn't an index file", path.display()),
            )
        };
        file.read_exact(&mut head).map_err(|_| damaged())?;
        let len = usize::try_from(file.metadata()?.len()).map_err(|_| damaged())?;
        Layout::parse(&head, len).ok_or_else(damaged)
    }

    /// The layout in `head`, the 324 bytes of the header of version 1 of a file of `len` bytes, or None for any other
    /// header. The dimension and the padded dimension are the u32s at bytes 16 and 20, the group count the u32 at
    /// byte 36, and the page size of the rows and the rows per page the u32s at bytes 48 and 64. The table of sections
    /// starts at byte 80, with 24 bytes for each, an offset and a length first. The centers, the extents, the codes,
    /// the rows and the ids are the third to the seventh.
    fn parse(head: &[u8; 324], len: usize) -> Option<Layout> {
        let u32_at = |at: usize| head[at..].first_chunk().map_or(0, |b| u32::from_le_bytes(*b)) as usize;
        let u64_at = |at: usize| head[at..].first_chunk().map_or(0, |b| u64::from_le_bytes(*b)) as usize;
        let section = |i: usize| {
            let at = 80 + 24 * i;
            u64_at(at)..u64_at(at).saturating_add(u64_at(at + 8))
        };

        if head[..8] != *b"COLDIVF\0" || u32_at(8) != 1 {
            return None;
        }

        Some(Layout {
            len,
            dim: u32_at(16),
            padded: u32_at(20),
            groups: u32_at(36),
            page_size: u32_at(48),
            per_page: u32_at(64),
            centers: section(2),
            extents: section(3),
            codes: section(4),
            rows: section(5),
            ids: section(6),
        })
    }

    /// The bytes of `range` of the file at `path`.
    fn section(&self, path: &Path, range: &Range<usize>) -> Result<Vec<u8>, Error> {
        let mut file = File::open(path)?;
        let mut bytes = vec![0u8; range.len()];
        file.seek(SeekFrom::Start(range.start as u64))?;
        file.read_exact(&mut bytes)?;
        Ok(bytes)
    }

    /// The reads of a cold search that probes the clusters `leaves` and rescores the candidates `hits`, by their ids,
    /// with `extents` the bytes of the extents section and `positions` each id's position in the file.
    fn plan(
        &self,
        extents: &[u8],
        positions: &HashMap<u64, usize>,
        leaves: &[LeafId],
        hits: &[u64],
    ) -> Result<Plan, Error> {
        let unknown = |id: &u64| Error::new(ErrorKind::InvalidData, format!("the file holds no id {id}"));
        let at: Vec<usize> = hits
            .iter()
            .map(|id| positions.get(id).copied().ok_or_else(|| unknown(id)))
            .collect::<Result<_, _>>()?;
        Ok(Plan {
            tables: self.tables(),
            clusters: coalesce(&self.cluster_spans(extents, leaves), self.len),
            rows: coalesce(&self.row_spans(&at), self.len),
        })
    }

    /// The cluster tables, from the clusters' 4-bit codes to the end of the extents. The codes start after the groups'
    /// centers, which take g·r bytes, where g is the group count and r is d + 8 rounded up to a multiple of 8, for d
    /// dimensions.
    fn tables(&self) -> Range<usize> {
        self.centers.start + self.groups * (self.dim + 8).div_ceil(8) * 8..self.extents.end
    }

    /// The blocks of the clusters `leaves` that hold vectors, in increasing order. Each cluster has 16 bytes in
    /// `extents`, where its blocks start, counted from the start of the codes, as a u64, then its vector count and the
    /// id of its first vector, as u32s. A block of 32 vectors' codes takes 4·p + 512 bytes, where p is the padded
    /// dimension.
    fn cluster_spans(&self, extents: &[u8], leaves: &[LeafId]) -> Vec<Range<usize>> {
        let (entries, block) = (extents.as_chunks::<16>().0, 4 * self.padded + 16 * 32);
        let mut leaves = leaves.to_vec();
        leaves.sort_unstable();
        leaves.dedup();
        let span = |e: &[u8; 16]| {
            let offset = u64::from_le_bytes(e[..8].try_into().unwrap_or_default()) as usize;
            let count = u32::from_le_bytes(e[8..12].try_into().unwrap_or_default()) as usize;
            let start = self.codes.start + offset;
            start..start + count.div_ceil(32) * block
        };
        let spans = leaves.iter().filter_map(|&l| entries.get(l as usize)).map(span);
        spans.filter(|s| s.start < s.end).collect()
    }

    /// The pages of rows that hold the rows at `positions`, in increasing order.
    fn row_spans(&self, positions: &[usize]) -> Vec<Range<usize>> {
        let mut pages: Vec<usize> = positions.iter().map(|&p| p / self.per_page.max(1)).collect();
        pages.sort_unstable();
        pages.dedup();
        let page = |p: usize| self.rows.start + p * self.page_size;
        pages.iter().map(|&p| page(p)..page(p + 1)).collect()
    }
}

/// The reads that cover `spans`, which must be in increasing order, as the backend makes them. Each span is rounded out
/// to whole pages of `PAGE` bytes and cut at the end of the file, of `len` bytes, and it joins the read before it when
/// they touch or overlap.
fn coalesce(spans: &[Range<usize>], len: usize) -> Vec<Range<usize>> {
    let mut out: Vec<Range<usize>> = Vec::with_capacity(spans.len());

    for s in spans {
        let (start, end) = (s.start / PAGE * PAGE, (s.end.div_ceil(PAGE) * PAGE).min(len));

        match out.last_mut() {
            Some(r) if start <= r.end => r.end = r.end.max(end),
            _ => out.push(start..end),
        }
    }

    out
}

/// One trial of `plan` in the variant `variant`, with its fields added to `line`.
#[cfg(target_os = "macos")]
fn trial(data: &Path, map: &Mapped, plan: &Plan, variant: &str, line: Json) -> Json {
    use coldivf_bench::coldcache::make_cold;

    let mut tries = 0;

    while let Err(e) = make_cold(&[map], Duration::from_millis(500)) {
        tries += 1;

        if e.kind() != ErrorKind::TimedOut || tries > 2 {
            fail(&format!("emptying the cache of {}: {e}", data.display()));
        }
    }

    if variant == "prewarm" {
        spin(Duration::from_millis(150));
    } else {
        std::thread::sleep(Duration::from_millis(200));
    }

    let cached = map.resident(0..map.len()).unwrap_or(0);
    let start = Instant::now();
    let file = File::open(data).unwrap_or_else(|e| fail(&format!("opening {}: {e}", data.display())));
    uncache(&file);
    let open_us = us(start.elapsed());
    let first = 0..FIRST_READ.min(map.len());
    let (header, _) = replay(&file, std::slice::from_ref(&first));
    let (tables, _) = replay(&file, std::slice::from_ref(&plan.tables));
    let (clusters, cluster_wave) = replay(&file, &plan.clusters);
    let (rows, row_wave) = replay(&file, &plan.rows);
    line.int("pages_resident_at_start", cached as u64)
        .num("open_us", open_us)
        .num("total_us", us(start.elapsed()))
        .num("cluster_wave_us", us(cluster_wave))
        .num("row_wave_us", us(row_wave))
        .objects("header", &header)
        .objects("tables", &tables)
        .objects("clusters", &clusters)
        .objects("rows", &rows)
}

#[cfg(not(target_os = "macos"))]
fn trial(_: &Path, _: &Mapped, _: &Plan, _: &str, _: Json) -> Json {
    fail("the stall probe is written for macOS")
}

/// Hints every range of `ranges` with `F_RDADVISE`, then reads each one in order into new memory, as the backend's
/// buffers are, and times each call. Returns each range's times and the whole wave's.
#[cfg(target_os = "macos")]
fn replay(file: &File, ranges: &[Range<usize>]) -> (Vec<Json>, Duration) {
    use std::os::fd::AsRawFd;
    use std::os::unix::fs::FileExt;

    let start = Instant::now();
    let advised: Vec<f64> = ranges
        .iter()
        .map(|r| {
            let at = Instant::now();
            let ra = libc::radvisory {
                ra_offset: r.start as libc::off_t,
                ra_count: r.len().min(i32::MAX as usize) as libc::c_int,
            };
            // SAFETY: `ra` outlives the call, and F_RDADVISE only reads it.
            unsafe { libc::fcntl(file.as_raw_fd(), libc::F_RDADVISE, &ra as *const libc::radvisory) };
            us(at.elapsed())
        })
        .collect();
    let mut arena = vec![0u8; ranges.iter().map(Range::len).sum()];
    let mut next = 0;
    let mut out = Vec::with_capacity(ranges.len());

    for (r, advise_us) in ranges.iter().zip(advised) {
        let at = Instant::now();
        file.read_exact_at(&mut arena[next..next + r.len()], r.start as u64)
            .unwrap_or_else(|e| fail(&format!("reading {} bytes at {}: {e}", r.len(), r.start)));
        next += r.len();
        out.push(
            Json::new()
                .int("offset", r.start as u64)
                .int("len", r.len() as u64)
                .num("advise_us", advise_us)
                .num("read_us", us(at.elapsed())),
        );
    }

    (out, start.elapsed())
}

/// Turns off caching for `file`, as the backend's handle has it.
#[cfg(target_os = "macos")]
fn uncache(file: &File) {
    use std::os::fd::AsRawFd;

    // SAFETY: F_NOCACHE takes an integer and touches no memory.
    if unsafe { libc::fcntl(file.as_raw_fd(), libc::F_NOCACHE, 1) } == -1 {
        fail("F_NOCACHE failed");
    }
}

/// Keeps this core busy for `time`, so it runs at full clock when the timer starts.
#[cfg(target_os = "macos")]
fn spin(time: Duration) {
    let start = Instant::now();
    let mut x = 0u64;

    while start.elapsed() < time {
        x = std::hint::black_box(x.wrapping_mul(6364136223846793005).wrapping_add(1));
    }
}

/// `time` in microseconds.
#[cfg(target_os = "macos")]
fn us(time: Duration) -> f64 {
    time.as_secs_f64() * 1e6
}

#[cfg(test)]
#[path = "../../tests/unit/bin/stall.rs"]
mod tests;
