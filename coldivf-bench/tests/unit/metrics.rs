//! Unit tests for the metrics.

use coldivf::{Anchor, BuildParams, Builder, LeafOrder, Metric, ReadPath, Writer};

use super::*;
use crate::data::{Fbin, write_fbin};

#[test]
fn recall_counts_the_true_neighbors_found() {
    assert_eq!(recall(&[1, 2, 3, 4], &[4, 3, 9, 8, 7], 4), 0.5);
    assert_eq!(recall(&[7, 8], &[8, 7, 1], 2), 1.0);
    assert_eq!(recall(&[], &[1, 2], 2), 0.0);
    // Only the first k of the truth count.
    assert_eq!(recall(&[5], &[1, 5], 1), 0.0);
}

#[test]
fn tied_recall_also_counts_what_the_tie_rule_allows_but_each_id_once() {
    let truth = [1, 2, 3, 4];
    assert_eq!(recall_tied(&[1, 7, 3], &truth, 3, |id| id == 7), 1.0);
    assert_eq!(recall_tied(&[1, 7, 3], &truth, 3, |_| false), 2.0 / 3.0);
    // An id found twice counts once, whether it is a true neighbor or a tie.
    assert_eq!(recall_tied(&[1, 1, 7, 7], &truth, 4, |id| id == 7), 0.5);
    assert_eq!(recall(&[4, 4], &[4, 3], 2), 0.5);
    // Only the first k found count, and fewer than k found leave the rest as misses.
    assert_eq!(recall_tied(&[1, 2, 7], &truth, 2, |_| true), 1.0);
    assert_eq!(recall_tied(&[7], &truth, 2, |id| id == 7), 0.5);
}

#[test]
fn json_lists_hold_numbers_and_objects() {
    let line = Json::new()
        .nums("x", &[1.5, f64::NAN, 2.0])
        .objects("o", &[Json::new().int("a", 1), Json::new()])
        .nums("none", &[])
        .object("p", &Json::new().str("b", "c").object("q", &Json::new()))
        .line();
    assert_eq!(
        line,
        r#"{"x":[1.5,null,2],"o":[{"a":1},{}],"none":[],"p":{"b":"c","q":{}}}"#
    );
}

#[test]
fn percentiles_take_the_nearest_rank() {
    let values: Vec<f64> = (1..=10).map(f64::from).collect();
    assert_eq!(percentile(&values, 50.0), 5.0);
    assert_eq!(percentile(&values, 99.0), 10.0);
    assert_eq!(percentile(&values, 0.0), 1.0);
    assert_eq!(percentile(&values, 100.0), 10.0);
    // The rank is 9.4, which rounds up to 10, not down to 9.
    assert_eq!(percentile(&values, 94.0), 10.0);
    assert_eq!(percentile(&[3.0], 90.0), 3.0);
    assert!(percentile(&[], 50.0).is_nan());
    // Ranks that are whole numbers stay whole, though eg. 99.9 / 100 * 1000 is a little over 999 in floating point.
    let thousand: Vec<f64> = (1..=1000).map(f64::from).collect();
    assert_eq!(percentile(&thousand, 99.9), 999.0);
    assert_eq!(percentile(&thousand[..100], 7.0), 7.0);
}

#[test]
fn a_linux_machine_is_named_by_its_product_and_its_cpu() {
    let cpuinfo = "processor\t: 0\nvendor_id\t: GenuineIntel\nmodel\t\t: 106\nmodel name\t: Intel(R) Xeon(R) \
                   Platinum 8375C CPU @ 2.90GHz\nflags\t\t: fpu avx2\n\nprocessor\t: 1\nmodel name\t: Other\n";
    assert_eq!(
        linux_machine("m6i.32xlarge\n", cpuinfo),
        "m6i.32xlarge Intel(R) Xeon(R) Platinum 8375C CPU @ 2.90GHz"
    );
    // A virtual machine may give no product name, and an Arm CPU no model name.
    assert_eq!(
        linux_machine("", cpuinfo),
        "Intel(R) Xeon(R) Platinum 8375C CPU @ 2.90GHz"
    );
    assert_eq!(
        linux_machine("Graviton\n", "processor\t: 0\nCPU part\t: 0xd40\n"),
        "Graviton"
    );
    assert_eq!(linux_machine("", ""), "");
}

#[test]
fn read_paths_count_the_contexts_of_each_setup() {
    let ring = ReadPath {
        backend: "io_uring",
        ring_entries: 64,
        direct: true,
        fixed_files: false,
        registered: 0,
    };
    let hints = ReadPath {
        backend: "fadvise",
        ..ReadPath::default()
    };
    let line = |paths: &[ReadPath]| read_paths(Json::new(), paths).line();
    let want = [
        r#"{"backend":"mixed","read_paths":["#,
        r#"{"backend":"io_uring","ring_entries":64,"direct":true,"fixed_files":false,"registered":0,"contexts":2},"#,
        r#"{"backend":"fadvise","ring_entries":0,"direct":false,"fixed_files":false,"registered":0,"contexts":1}]}"#,
    ];
    assert_eq!(line(&[ring, hints, ring]), want.concat());
    // Contexts of one backend set up apart, eg. with rings of two sizes, still name that backend.
    let small = ReadPath {
        ring_entries: 16,
        ..ring
    };
    assert!(line(&[ring, small]).starts_with(r#"{"backend":"io_uring","#));
    assert!(line(&[hints]).starts_with(r#"{"backend":"fadvise","#));
    assert_eq!(line(&[]), r#"{"backend":"","read_paths":[]}"#);
}

#[test]
fn json_lines_quote_and_skip_numbers_json_cannot_hold() {
    let line = Json::new()
        .str("a\"b", "c\\d\n")
        .num("x", 1.5)
        .num("nan", f64::NAN)
        .int("n", 3)
        .bool("ok", true)
        .null("none")
        .line();
    assert_eq!(
        line,
        r#"{"a\"b":"c\\d\n","x":1.5,"nan":null,"n":3,"ok":true,"none":null}"#
    );
}

#[test]
fn times_are_written_as_utc_dates() {
    for (ms, date) in [
        (0, "1970-01-01T00:00:00.000Z"),
        (951_782_400_000, "2000-02-29T00:00:00.000Z"),
        (1_791_436_353_123, "2026-10-08T05:12:33.123Z"),
        (1_798_761_599_999, "2026-12-31T23:59:59.999Z"),
        (4_107_542_400_000, "2100-03-01T00:00:00.000Z"),
        (253_402_300_799_999, "9999-12-31T23:59:59.999Z"),
    ] {
        assert_eq!(utc(UNIX_EPOCH + Duration::from_millis(ms)), date);
    }
}

#[cfg(target_os = "macos")]
#[test]
fn the_machine_state_holds_the_time_the_load_the_power_and_the_qos() {
    interactive();
    let line = state(Json::new()).line();

    assert!(line.starts_with(r#"{"time":"20"#), "{line}");
    assert!(line.contains(r#""qos":"user-interactive""#), "{line}");
    // eg. "AC Power" or "Battery Power".
    let power = line
        .split(r#""power":""#)
        .nth(1)
        .and_then(|rest| rest.split('"').next());
    assert!(power.is_some_and(|p| p.ends_with(" Power")), "{line}");

    for key in ["load_1", "load_5", "load_15", "low_power"] {
        assert!(line.contains(&format!("\"{key}\":")), "{key}: {line}");
        assert!(!line.contains(&format!("\"{key}\":null")), "{key}: {line}");
    }

    let line = provenance(Json::new()).line();
    let commit = line.split('"').nth(3).unwrap();
    assert!(
        commit.len() >= 7 && commit.chars().all(|c| c.is_ascii_hexdigit()),
        "{line}"
    );
    assert!(line.contains(r#""built":"20"#), "{line}");
}

/// The cyclic redundancy check (CRC32) of `bytes`, bit by bit, the checksum the index file's header ends with.
fn crc32(bytes: &[u8]) -> u32 {
    let step = |c: u32| if c & 1 == 1 { (c >> 1) ^ 0xEDB8_8320 } else { c >> 1 };
    !bytes
        .iter()
        .fold(!0u32, |crc, &b| (0..8).fold(crc ^ u32::from(b), |c, _| step(c)))
}

#[test]
fn an_index_file_s_header_names_its_build() {
    let dir = std::env::temp_dir().join("coldivf-bench");
    std::fs::create_dir_all(&dir).unwrap();
    let (rows, path) = (
        dir.join(format!("{}-id.fbin", std::process::id())),
        dir.join(format!("{}-id.coldivf", std::process::id())),
    );
    let mut state = 1u64;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 40) as f32 / (1u64 << 24) as f32 - 0.5
    };
    let (n, dim) = (400, 128);
    let data: Vec<f32> = (0..n * dim).map(|_| next()).collect();
    write_fbin(&rows, &data, dim).unwrap();
    let params = BuildParams {
        metric: Metric::L2,
        vectors_per_list: 100,
        parents: 2,
        order: LeafOrder::Chained,
        anchor: Anchor::Cluster,
        page_size: 4096,
        threads: 1,
        seed: 7,
    };
    Builder::new(params)
        .build(&Fbin::open(&rows).unwrap().prefix(n), &path)
        .unwrap();

    // The last cluster's extent is the last 16 bytes of the tables, and its ids end at the vector count.
    // A compaction moves the index into its folder, and the header read follows it there.
    let check = |vectors: u32, epoch: u64| {
        let data = coldivf::data_file(&path).unwrap();
        let (id, bytes) = (IndexFile::read(&path).unwrap(), std::fs::read(&data).unwrap());
        let meta = std::fs::metadata(&data).unwrap();
        assert_eq!(
            (id.bytes, id.modified, id.epoch),
            (meta.len(), meta.modified().unwrap(), epoch)
        );
        assert_eq!(id.crc, crc32(&bytes[..320]));
        // 400 vectors at 100 a cluster, and a compaction keeps every cluster.
        assert_eq!((id.count, id.nlist), (u64::from(vectors), 4));
        let tables = id.tables.clone();
        assert!(tables.start >= 16384 && tables.start.is_multiple_of(16384) && tables.end <= bytes.len());
        let last = &bytes[tables.end - 8..tables.end];
        let (count, first) = (
            u32::from_le_bytes(last[..4].try_into().unwrap()),
            u32::from_le_bytes(last[4..].try_into().unwrap()),
        );
        assert_eq!(first + count, vectors);
        id.crc
    };
    let built = check(n as u32, 0);
    let mut writer = Writer::open(&path).unwrap();
    writer.insert(9999, &data[..dim]).unwrap();
    writer.compact().unwrap();
    assert_ne!(check(n as u32 + 1, 1), built);

    assert_eq!(IndexFile::read(&rows).err().unwrap().kind(), ErrorKind::InvalidData);
    // A header of another format version, one with another magic, and one cut short aren't read either.
    let header = std::fs::read(coldivf::data_file(&path).unwrap()).unwrap()[..324].to_vec();
    let other = dir.join(format!("{}-other.coldivf", std::process::id()));
    let (mut version, mut magic) = (header.clone(), header.clone());
    (version[8], magic[0]) = (2, b'X');

    for (what, bytes) in [
        ("version 2", &version[..]),
        ("another magic", &magic[..]),
        ("cut short", &header[..300]),
    ] {
        std::fs::write(&other, bytes).unwrap();
        let kind = IndexFile::read(&other).err().map(|e| e.kind());
        assert_eq!(kind, Some(ErrorKind::InvalidData), "{what}");
    }

    std::fs::remove_file(&other).unwrap();

    for end in ["", ".delta", ".lock"] {
        let _ = std::fs::remove_file(format!("{}{end}", path.display()));
    }

    let _ = std::fs::remove_dir_all(format!("{}.d", path.display()));
    std::fs::remove_file(&rows).unwrap();
}

#[cfg(target_os = "macos")]
#[test]
fn a_warm_up_caches_an_index_file_s_codes_and_its_rows_drop_apart() {
    use std::time::Instant;

    use crate::coldcache::{Mapped, make_cold, page_size};

    let dir = std::env::temp_dir().join("coldivf-bench");
    std::fs::create_dir_all(&dir).unwrap();
    let (rows, path) = (
        dir.join(format!("{}-sections.fbin", std::process::id())),
        dir.join(format!("{}-sections.coldivf", std::process::id())),
    );
    let mut state = 3u64;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 40) as f32 / (1u64 << 24) as f32 - 0.5
    };
    let (n, dim) = (2000, 128);
    let data: Vec<f32> = (0..n * dim).map(|_| next()).collect();
    write_fbin(&rows, &data, dim).unwrap();
    let params = BuildParams {
        metric: Metric::L2,
        vectors_per_list: 100,
        parents: 4,
        order: LeafOrder::Chained,
        anchor: Anchor::Cluster,
        page_size: 4096,
        threads: 1,
        seed: 7,
    };
    Builder::new(params)
        .build(&Fbin::open(&rows).unwrap().prefix(n), &path)
        .unwrap();

    // The file starts out of the cache, so only what the warm-up reads comes back.
    let (id, file) = (IndexFile::read(&path).unwrap(), Mapped::open(&path).unwrap());
    make_cold(&[&file], Duration::from_millis(500)).unwrap();
    let index = coldivf::Index::open(&path).unwrap();
    index.warm_up().unwrap();
    let start = Instant::now();

    while !index.is_warm() && start.elapsed() < Duration::from_secs(10) {
        std::thread::sleep(Duration::from_millis(10));
    }

    // Anything that reads a file's first pages back after a new mapping, as the index's own, has done so by now.
    std::thread::sleep(Duration::from_millis(500));
    let pages = |r: &Range<usize>| r.end.div_ceil(page_size()) - r.start / page_size();
    assert!(index.is_warm() && pages(&id.codes) > 0 && pages(&id.rows) > 0);
    assert_eq!(file.resident(id.codes.clone()).unwrap(), pages(&id.codes));
    // The rows drop from the cache without the codes.
    std::fs::read(&path).unwrap();
    file.evict_range(id.rows.clone()).unwrap();
    assert_eq!(file.resident(id.rows.clone()).unwrap(), 0);
    assert_eq!(file.resident(id.codes.clone()).unwrap(), pages(&id.codes));
    drop(index);

    for end in ["", ".delta", ".lock"] {
        let _ = std::fs::remove_file(format!("{}{end}", path.display()));
    }

    std::fs::remove_file(&rows).unwrap();
}
