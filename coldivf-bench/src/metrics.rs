//! Recall, percentiles, and the JSON-lines result files every harness appends to, with the fields that tie each line to
//! the code, the machine and the index file that made it.

use std::fs::{File, OpenOptions};
use std::io::{Error, ErrorKind, Read, Write};
use std::ops::Range;
use std::path::Path;
#[cfg(not(target_os = "linux"))]
use std::process::Command;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use coldivf::ReadPath;

/// The share of the true `k` nearest neighbors in `truth` that the first `k` of `found` holds.
pub fn recall(found: &[u64], truth: &[i32], k: usize) -> f64 {
    recall_tied(found, truth, k, |_| false)
}

/// The same, where an id among the first `k` of `found` that isn't among the true `k` still counts when `tied` says
/// so, eg. when it is as near as the true k-th neighbor. An id found twice counts once.
pub fn recall_tied(found: &[u64], truth: &[i32], k: usize, tied: impl Fn(u64) -> bool) -> f64 {
    let (truth, found) = (&truth[..k.min(truth.len())], &found[..k.min(found.len())]);
    let hits = found
        .iter()
        .enumerate()
        .filter(|&(i, id)| !found[..i].contains(id))
        .filter(|&(_, &id)| truth.iter().any(|&t| t as u64 == id) || tied(id))
        .count();
    hits as f64 / k as f64
}

/// The nearest-rank percentile `p` (0 to 100) of `sorted`, which must be sorted. It is the smallest value that at
/// least `p` percent of the values are at or below.
pub fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }

    // Multiplying first, and taking off a little for rounding, keeps eg. 99.9% of 1000 values at rank 999, not 1000.
    let rank = (p * sorted.len() as f64 / 100.0 - 1e-9).ceil() as usize;
    sorted[rank.clamp(1, sorted.len()) - 1]
}

/// One JSON object, built a field at a time and written as one line.
#[derive(Default)]
pub struct Json(Vec<String>);

impl Json {
    pub fn new() -> Json {
        Json::default()
    }

    pub fn str(mut self, key: &str, value: &str) -> Json {
        self.0.push(format!("{}:{}", quote(key), quote(value)));
        self
    }

    /// A number, or null when it isn't finite, which JSON can't hold.
    pub fn num(mut self, key: &str, value: f64) -> Json {
        self.0.push(format!("{}:{}", quote(key), number(value)));
        self
    }

    /// A list of numbers, each null when it isn't finite.
    pub fn nums(mut self, key: &str, values: &[f64]) -> Json {
        let values: Vec<String> = values.iter().map(|&v| number(v)).collect();
        self.0.push(format!("{}:[{}]", quote(key), values.join(",")));
        self
    }

    /// A list of objects.
    pub fn objects(mut self, key: &str, items: &[Json]) -> Json {
        let items: Vec<String> = items.iter().map(Json::line).collect();
        self.0.push(format!("{}:[{}]", quote(key), items.join(",")));
        self
    }

    /// An object inside this one.
    pub fn object(mut self, key: &str, value: &Json) -> Json {
        self.0.push(format!("{}:{}", quote(key), value.line()));
        self
    }

    pub fn int(mut self, key: &str, value: u64) -> Json {
        self.0.push(format!("{}:{value}", quote(key)));
        self
    }

    pub fn bool(mut self, key: &str, value: bool) -> Json {
        self.0.push(format!("{}:{value}", quote(key)));
        self
    }

    /// A null, for a value that isn't known.
    pub fn null(mut self, key: &str) -> Json {
        self.0.push(format!("{}:null", quote(key)));
        self
    }

    pub fn line(&self) -> String {
        format!("{{{}}}", self.0.join(","))
    }
}

fn number(value: f64) -> String {
    if value.is_finite() {
        format!("{value}")
    } else {
        "null".to_string()
    }
}

fn quote(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');

    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }

    out.push('"');
    out
}

/// Appends `line` to the results file `path` and flushes it, so a sleep or a kill loses at most the line in flight.
pub fn append(path: &Path, line: &str) -> Result<(), Error> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }

    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    writeln!(file, "{line}")?;
    file.flush()
}

/// The fields every result records about the code and the machine. The git commit, whether the workspace's sources
/// differed from it, and the time are from when this program was built, so a line names the code that made it.
pub fn provenance(json: Json) -> Json {
    let built = env!("COLDIVF_BUILT")
        .parse()
        .map_or(String::new(), |s| utc(UNIX_EPOCH + Duration::from_secs(s)));
    json.str("commit", env!("COLDIVF_COMMIT"))
        .bool("dirty", env!("COLDIVF_DIRTY") == "true")
        .str("built", &built)
        .str("machine", &machine())
}

/// The machine's model and chip, eg. "Mac15,7 Apple M3 Pro", or on Linux its product name and its processor's
/// model, eg. "m6i.32xlarge Intel(R) Xeon(R) Platinum 8375C CPU @ 2.90GHz".
pub fn machine() -> String {
    #[cfg(target_os = "linux")]
    {
        let read = |path: &str| std::fs::read_to_string(path).unwrap_or_default();
        linux_machine(
            &read("/sys/devices/virtual/dmi/id/product_name"),
            &read("/proc/cpuinfo"),
        )
    }

    #[cfg(not(target_os = "linux"))]
    {
        let sysctl = |name: &str| {
            let out = Command::new("sysctl").args(["-n", name]).output();
            out.map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
                .unwrap_or_default()
        };
        format!("{} {}", sysctl("hw.model"), sysctl("machdep.cpu.brand_string"))
            .trim()
            .to_string()
    }
}

/// A Linux machine's name from its product name, as the firmware gives it, and the first model name in `cpuinfo`, the
/// text of /proc/cpuinfo. Either may be empty, eg. a virtual machine may give no product name.
pub fn linux_machine(product: &str, cpuinfo: &str) -> String {
    let model = cpuinfo
        .lines()
        .filter_map(|line| line.split_once(':'))
        .find(|(key, _)| key.trim() == "model name")
        .map_or("", |(_, value)| value.trim());
    format!("{} {model}", product.trim()).trim().to_string()
}

/// Adds what read the waves of the searches that `paths` describe, one per search context: `backend`, the backend
/// every context used or "mixed", and `read_paths`, each distinct setup with its count of contexts. Under a low
/// locked-memory limit on Linux before 5.12, some contexts may get no io_uring ring and read with read hints instead,
/// and this shows how many.
pub fn read_paths(json: Json, paths: &[ReadPath]) -> Json {
    let mut kinds: Vec<(ReadPath, u64)> = Vec::new();

    for &p in paths {
        match kinds.iter_mut().find(|(k, _)| *k == p) {
            Some((_, n)) => *n += 1,
            None => kinds.push((p, 1)),
        }
    }

    let backend = match kinds.as_slice() {
        [(p, _)] => p.backend,
        [] => "",
        _ if kinds.iter().all(|(p, _)| p.backend == kinds[0].0.backend) => kinds[0].0.backend,
        _ => "mixed",
    };
    let items: Vec<Json> = kinds
        .iter()
        .map(|(p, n)| {
            Json::new()
                .str("backend", p.backend)
                .int("ring_entries", u64::from(p.ring_entries))
                .bool("direct", p.direct)
                .bool("fixed_files", p.fixed_files)
                .int("registered", p.registered as u64)
                .int("contexts", *n)
        })
        .collect();
    json.str("backend", backend).objects("read_paths", &items)
}

/// The fields every result records about the machine's state now: the time, the load averages over 1, 5 and 15
/// minutes, the power source, whether low-power mode is on, and the calling thread's quality of service (QoS) class,
/// which decides the cores it may run on.
pub fn state(json: Json) -> Json {
    let [one, five, fifteen] = load();
    let (source, low_power) = power();
    let json = json
        .str("time", &utc(SystemTime::now()))
        .num("load_1", one)
        .num("load_5", five)
        .num("load_15", fifteen)
        .str("power", &source);
    match low_power {
        Some(on) => json.bool("low_power", on),
        None => json.null("low_power"),
    }
    .str("qos", qos())
}

/// The load averages over 1, 5 and 15 minutes, or NaN where they can't be read.
fn load() -> [f64; 3] {
    #[cfg(unix)]
    {
        let mut load = [f64::NAN; 3];
        // SAFETY: getloadavg writes at most the 3 values it is asked for into `load`.
        unsafe { libc::getloadavg(load.as_mut_ptr(), 3) };
        load
    }

    #[cfg(not(unix))]
    [f64::NAN; 3]
}

/// The power source, eg. "AC Power", and whether low-power mode is on, from `pmset`. Either is empty when unknown.
fn power() -> (String, Option<bool>) {
    #[cfg(target_os = "macos")]
    {
        let pmset = |args: &[&str]| {
            let out = Command::new("pmset").args(args).output();
            out.map(|o| String::from_utf8_lossy(&o.stdout).into_owned())
                .unwrap_or_default()
        };
        // eg. "Now drawing from 'AC Power'".
        let source = pmset(&["-g", "batt"])
            .split('\'')
            .nth(1)
            .unwrap_or_default()
            .to_string();
        // eg. " lowpowermode 0", or on Macs that also have a high-power mode " powermode 1", where 1 is low power.
        let low_power = pmset(&["-g"])
            .lines()
            .find_map(|line| match line.split_whitespace().collect::<Vec<_>>()[..] {
                ["lowpowermode", value, ..] => Some(value != "0"),
                ["powermode", value, ..] => Some(value == "1"),
                _ => None,
            });
        (source, low_power)
    }

    #[cfg(not(target_os = "macos"))]
    (String::new(), None)
}

/// Moves the calling thread to the user-interactive QoS class, the one for work a user waits on, for timed work.
/// Elsewhere than macOS it does nothing.
pub fn interactive() {
    #[cfg(target_os = "macos")]
    // SAFETY: pthread_set_qos_class_self_np takes only a class and a priority, and changes only the calling thread.
    unsafe {
        libc::pthread_set_qos_class_self_np(libc::qos_class_t::QOS_CLASS_USER_INTERACTIVE, 0)
    };
}

/// The calling thread's QoS class, eg. "user-interactive", or an empty string elsewhere than macOS.
pub fn qos() -> &'static str {
    #[cfg(target_os = "macos")]
    {
        use libc::qos_class_t::*;

        let class = qos_class_self();
        let names = [
            (QOS_CLASS_USER_INTERACTIVE, "user-interactive"),
            (QOS_CLASS_USER_INITIATED, "user-initiated"),
            (QOS_CLASS_DEFAULT, "default"),
            (QOS_CLASS_UTILITY, "utility"),
            (QOS_CLASS_BACKGROUND, "background"),
            (QOS_CLASS_UNSPECIFIED, "unspecified"),
        ];
        names
            .iter()
            .find(|&&(c, _)| c as u32 == class)
            .map_or("other", |&(_, name)| name)
    }

    #[cfg(not(target_os = "macos"))]
    ""
}

// SAFETY: this matches <pthread/qos.h>, where qos_class_t is an unsigned int. qos_class_self only reads the calling
// thread's class, so any thread may call it.
#[cfg(target_os = "macos")]
unsafe extern "C" {
    safe fn qos_class_self() -> u32;
}

/// `time` in Coordinated Universal Time (UTC) to the millisecond, eg. "2026-10-08T05:12:33.123Z".
pub fn utc(time: SystemTime) -> String {
    let ms = time.duration_since(UNIX_EPOCH).map_or(0, |d| d.as_millis() as u64);
    let (days, ms) = (ms / 86_400_000, ms % 86_400_000);
    // The date comes from the count of days in years that start on 1 March, so a leap day ends its year. 719468 days
    // run from 1 March of year 0 to 1 January 1970, and every 400 years hold 146097 days.
    let days = days + 719_468;
    let (era, day_of_era) = (days / 146_097, days % 146_097);
    let year_of_era = (day_of_era - day_of_era / 1460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_from_march = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_from_march + 2) / 5 + 1;
    let month = (month_from_march + 2) % 12 + 1;
    let year = era * 400 + year_of_era + u64::from(month <= 2);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}.{:03}Z",
        ms / 3_600_000,
        ms / 60_000 % 60,
        ms / 1000 % 60,
        ms % 1000
    )
}

/// What a result line records about an index file, to tie the line to the build that wrote the file. That is its size,
/// when it was last written, and from its header the epoch, which counts its compactions, and the header's checksum.
pub struct IndexFile {
    pub bytes: u64,
    pub modified: SystemTime,
    pub epoch: u64,
    pub crc: u32,
    /// The vectors and the clusters in the file.
    pub count: u64,
    pub nlist: u32,
    /// Where the cluster centers and extents are, which the first search of a cold index reads after the header.
    pub tables: Range<usize>,
    /// Where the 1-bit codes are, which a warm-up reads and a warm search scans in place.
    pub codes: Range<usize>,
    /// Where the 8-bit rows are, which every search reads to rescore its best candidates.
    pub rows: Range<usize>,
}

impl IndexFile {
    /// Reads the data file of the index at `path`, through a plain open and read. Once maintenance has run, that is a
    /// file in the index's folder. A file that isn't an index of format version 1 fails with `InvalidData`.
    pub fn read(path: &Path) -> Result<IndexFile, Error> {
        let path = &coldivf::data_file(path)?;
        let mut file = File::open(path)?;
        let meta = file.metadata()?;
        // Version 1's header is 324 bytes. The vector count is the u64 at byte 24, and the cluster count the u32 after
        // it. The table of sections starts at byte 80, with 24 bytes for each, an offset and a length first. The
        // centers, the extents, the codes and the rows are the third to the sixth. The epoch and the checksum end it.
        let mut head = [0u8; 324];
        let damaged = || {
            Error::new(
                ErrorKind::InvalidData,
                format!("{} isn't an index file", path.display()),
            )
        };
        file.read_exact(&mut head).map_err(|_| damaged())?;
        let u32_at = |at: usize| head[at..].first_chunk().map_or(0, |b| u32::from_le_bytes(*b));
        let u64_at = |at: usize| head[at..].first_chunk().map_or(0, |b| u64::from_le_bytes(*b));

        if head[..8] != *b"COLDIVF\0" || u32_at(8) != 1 {
            return Err(damaged());
        }

        let section = |i: usize| {
            let at = 80 + 24 * i;
            u64_at(at) as usize..u64_at(at).saturating_add(u64_at(at + 8)) as usize
        };
        Ok(IndexFile {
            bytes: meta.len(),
            modified: meta.modified()?,
            epoch: u64_at(312),
            crc: u32_at(320),
            count: u64_at(24),
            nlist: u32_at(32),
            tables: section(2).start..section(3).end,
            codes: section(4),
            rows: section(5),
        })
    }

    /// Adds the fields `index_bytes`, `index_mtime`, `index_epoch` and `index_crc`.
    pub fn fields(&self, json: Json) -> Json {
        json.int("index_bytes", self.bytes)
            .str("index_mtime", &utc(self.modified))
            .int("index_epoch", self.epoch)
            .int("index_crc", u64::from(self.crc))
    }
}

#[cfg(test)]
#[path = "../tests/unit/metrics.rs"]
mod tests;
