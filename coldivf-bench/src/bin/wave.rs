//! The read wave on its own, on macOS. On a file that isn't cached, it reads `--ranges` random pages of `--page`
//! bytes the way ColdIVF's macOS backend does. It hints every range with `F_RDADVISE` first, then reads each one in
//! order through an `F_NOCACHE` handle. It maps the file once and empties the cache through that mapping before every
//! trial, since a new mapping can make another process read the file's first pages back. Right before the timer it
//! checks the trial's own pages, and it leaves out any trial that would start with one of them cached.
//!
//! wave [--file data/wave-1g.bin] [--size-mb 1024] [--trials 200] [--ranges 64] [--page 16384]
//!      [--results results/wave.jsonl]

use std::path::{Path, PathBuf};

use coldivf_bench::args::{Args, fail};
use coldivf_bench::metrics::{Json, append, interactive, percentile, provenance, state};

fn main() {
    let args = Args::parse();
    let file = PathBuf::from(args.str("file").unwrap_or("data/wave-1g.bin"));
    let size = args.get::<usize>("size-mb", 1024) << 20;
    let (trials, ranges, page) = (args.get("trials", 200), args.get("ranges", 64), args.get("page", 16384));
    let results = args.str("results").unwrap_or("results/wave.jsonl");

    if page == 0 || size < page * ranges {
        fail("the file must hold at least --ranges pages of --page bytes");
    }

    make(&file, size);
    interactive();
    let (times, dropped) = run(&file, size, trials, ranges, page);
    let mut sorted = times.clone();
    sorted.sort_by(f64::total_cmp);
    eprintln!(
        "wave: {ranges} x {page} B, {} trials ({dropped} left out for starting with cached pages): p50 {:.0} us, p99 \
         {:.0} us",
        sorted.len(),
        percentile(&sorted, 50.0),
        percentile(&sorted, 99.0)
    );
    let line = state(provenance(Json::new()))
        .str("file", &file.display().to_string())
        .int("size", size as u64)
        .int("ranges", ranges as u64)
        .int("page", page as u64)
        .int("trials", sorted.len() as u64)
        .int("dropped", dropped as u64)
        .num("p50_us", percentile(&sorted, 50.0))
        .num("p99_us", percentile(&sorted, 99.0))
        .num("mean_us", sorted.iter().sum::<f64>() / sorted.len().max(1) as f64)
        .line();
    append(Path::new(results), &line).unwrap_or_else(|e| fail(&format!("writing {results}: {e}")));
}

/// Writes `size` bytes of random data to `path`, unless a file of that size is already there.
fn make(path: &Path, size: usize) {
    if std::fs::metadata(path).is_ok_and(|m| m.len() == size as u64) {
        return;
    }

    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).unwrap_or_else(|e| fail(&format!("making {}: {e}", dir.display())));
    }

    let mut state = 1u64;
    let mut chunk = vec![0u8; 8 << 20];
    let mut out = std::fs::File::create(path).unwrap_or_else(|e| fail(&format!("creating {}: {e}", path.display())));

    for _ in 0..size.div_ceil(chunk.len()) {
        chunk
            .as_chunks_mut::<8>()
            .0
            .iter_mut()
            .for_each(|c| *c = next(&mut state).to_le_bytes());
        std::io::Write::write_all(&mut out, &chunk).unwrap_or_else(|e| fail(&format!("writing: {e}")));
    }

    out.set_len(size as u64)
        .unwrap_or_else(|e| fail(&format!("sizing: {e}")));
}

/// Times `trials` waves, and returns their times and how many trials it left out for starting with cached pages.
#[cfg(target_os = "macos")]
fn run(path: &Path, size: usize, trials: usize, ranges: usize, page: usize) -> (Vec<f64>, usize) {
    use std::os::fd::AsRawFd;
    use std::time::{Duration, Instant};

    use coldivf_bench::coldcache::{Mapped, make_cold};

    let file = std::fs::File::open(path).unwrap_or_else(|e| fail(&format!("opening {}: {e}", path.display())));
    let fd = file.as_raw_fd();

    // SAFETY: F_NOCACHE takes an integer and touches no memory.
    if unsafe { libc::fcntl(fd, libc::F_NOCACHE, 1) } == -1 {
        fail("F_NOCACHE failed");
    }

    let map = Mapped::open(path).unwrap_or_else(|e| fail(&format!("mapping {}: {e}", path.display())));
    make_cold(&[&map], Duration::from_millis(500)).unwrap_or_else(|e| fail(&format!("evicting: {e}")));
    let (mut state, mut times, mut dropped) = (7u64, Vec::with_capacity(trials), 0);
    let mut buf = vec![0u8; ranges * page];
    let pages = size / page;

    for _ in 0..trials {
        let mut offsets: Vec<usize> = Vec::with_capacity(ranges);
        while offsets.len() < ranges {
            let p = (next(&mut state) % pages as u64) as usize * page;
            if !offsets.contains(&p) {
                offsets.push(p);
            }
        }
        offsets.sort_unstable();
        make_cold(&[&map], Duration::ZERO).unwrap_or_else(|e| fail(&format!("evicting: {e}")));
        let cached: usize = offsets.iter().map(|&at| map.resident(at..at + page).unwrap_or(1)).sum();
        if cached > 0 {
            dropped += 1;
            continue;
        }
        let start = Instant::now();
        for &at in &offsets {
            let ra = libc::radvisory {
                ra_offset: at as libc::off_t,
                ra_count: page as libc::c_int,
            };
            // SAFETY: `ra` outlives the call, and F_RDADVISE only reads it.
            unsafe { libc::fcntl(fd, libc::F_RDADVISE, &ra as *const libc::radvisory) };
        }
        for (slot, &at) in buf.chunks_exact_mut(page).zip(&offsets) {
            std::os::unix::fs::FileExt::read_exact_at(&file, slot, at as u64)
                .unwrap_or_else(|e| fail(&format!("reading: {e}")));
        }
        times.push(start.elapsed().as_secs_f64() * 1e6);
    }

    (times, dropped)
}

#[cfg(not(target_os = "macos"))]
fn run(_: &Path, _: usize, _: usize, _: usize, _: usize) -> (Vec<f64>, usize) {
    fail("the wave check is written for macOS")
}

/// SplitMix64, a small fast generator of random 64-bit numbers.
fn next(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}
