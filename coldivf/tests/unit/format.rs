//! Unit tests for the file format.

use std::io::ErrorKind;

use super::*;
use crate::build::write;
use crate::quantization::SplitMix64;
use crate::quantization::testutil::{clustered, group, params, temp_file};

/// The bytes of a small index file, and the index they came from.
fn built(name: &str, dim: usize) -> (Vec<u8>, InMemory) {
    let mut rng = SplitMix64::new(1);
    let (rows, mut clusters) = clustered(&mut rng, 150, dim, 6, 0.5);
    group(&mut clusters, 2);
    let index = InMemory::build(&params(Metric::Cosine), &rows, &clusters).unwrap();

    let path = temp_file(name);
    write(&index, &path).unwrap();
    let bytes = std::fs::read(&path).unwrap();
    (bytes, index)
}

fn kind<T>(r: Result<T, Error>) -> ErrorKind {
    r.map(|_| ()).unwrap_err().kind()
}

#[test]
fn the_header_reads_back_what_was_written() {
    let (bytes, index) = built("header.coldivf", 200);
    let layout = read_header(&bytes, bytes.len()).unwrap();

    assert_eq!(
        (layout.dim, layout.count, layout.nlist, layout.nparents),
        (200, 150, 6, 2)
    );
    assert_eq!(
        (layout.metric, layout.page_size, layout.seed),
        (Metric::Cosine, 4096, 7)
    );
    assert_eq!(layout.span(Part::Codes).len, index.blocks.len());
    assert_eq!(layout.span(Part::Rows).len, index.rows.len());
    for part in [Part::Centers, Part::Extents, Part::Codes, Part::Rows, Part::Ids] {
        assert!(
            layout.span(part).offset.is_multiple_of(ALIGN),
            "{part:?} isn't on a 16 KiB boundary"
        );
    }

    let (rotation, mean) = read_rotation(&bytes[layout.span(Part::Rotation).range()], &layout).unwrap();
    assert_eq!((rotation, mean), (index.rotation.clone(), index.mean.clone()));

    let tables = read_tables(bytes[tables_span(&layout).range()].to_vec(), &layout).unwrap();
    assert_eq!(tables.codes, index.centers);
    assert_eq!(
        (tables.factors, tables.parent_of, tables.lists),
        (index.center_factors, index.parent_of, index.lists)
    );
    verify(&bytes, &layout).unwrap();

    let ids: Vec<u8> = index
        .ids
        .iter()
        .flat_map(|id| id.to_le_bytes())
        .chain(vec![0; 8 * 150usize.div_ceil(64)])
        .collect();
    assert_eq!(&bytes[layout.span(Part::Ids).range()], &ids[..]);
    let centers = &bytes[layout.span(Part::Centers).range()];
    assert_eq!(&centers[..index.parents.len()], &index.parents[..]);
}

#[test]
fn a_damaged_header_or_rotation_is_rejected() {
    let (bytes, _) = built("damaged.coldivf", 128);
    let layout = read_header(&bytes, bytes.len()).unwrap();

    let mut flipped = bytes.clone();
    flipped[20] ^= 1;
    assert_eq!(kind(read_header(&flipped, flipped.len())), ErrorKind::InvalidData);

    let mut foreign = bytes.clone();
    foreign[..8].copy_from_slice(b"NOTANIDX");
    assert_eq!(kind(read_header(&foreign, foreign.len())), ErrorKind::InvalidData);

    assert_eq!(kind(read_header(&bytes, bytes.len() - 1)), ErrorKind::InvalidData);
    assert_eq!(kind(read_header(&bytes[..100], 100)), ErrorKind::InvalidData);

    let mut rotation = bytes[layout.span(Part::Rotation).range()].to_vec();
    rotation[30] ^= 1;
    assert_eq!(kind(read_rotation(&rotation, &layout)), ErrorKind::InvalidData);
}

#[test]
fn damaged_extents_are_rejected() {
    let (bytes, _) = built("extents.coldivf", 128);
    let layout = read_header(&bytes, bytes.len()).unwrap();
    let good = bytes[layout.span(Part::Extents).range()].to_vec();
    check_extents(&good, &layout).unwrap();

    // Each extent is an 8-byte offset, a 4-byte count and a 4-byte first id.
    let u64_at = |b: &[u8], at: usize| u64::from_le_bytes(b[at..at + 8].try_into().unwrap());
    let set_u64 = |b: &mut Vec<u8>, at: usize, v: u64| b[at..at + 8].copy_from_slice(&v.to_le_bytes());
    let set_u32 = |b: &mut Vec<u8>, at: usize, v: u32| b[at..at + 4].copy_from_slice(&v.to_le_bytes());

    let mut off_line = good.clone();
    set_u64(&mut off_line, 16, u64_at(&good, 16) + 4);
    let mut backwards = good.clone();
    set_u64(&mut backwards, 16, 0);
    set_u64(&mut backwards, 0, u64_at(&good, 16));
    let mut misnumbered = good.clone();
    set_u32(&mut misnumbered, 16 + 12, 1);
    let mut too_long = good.clone();
    set_u32(&mut too_long, 5 * 16 + 8, 1_000_000);
    let mut short = good.clone();
    set_u32(&mut short, 5 * 16 + 8, 0);

    for bad in [off_line, backwards, misnumbered, too_long, short] {
        assert_eq!(kind(check_extents(&bad, &layout)), ErrorKind::InvalidData);
    }

    // With room left in the codes section, a last cluster moved off its line breaks only the line rule.
    let mut roomy = layout.clone();
    roomy.spans[Part::Codes as usize].len += 4096;
    let last = 16 * (layout.nlist - 1);
    let mut moved = good.clone();
    check_extents(&moved, &roomy).unwrap();
    set_u64(&mut moved, last, u64_at(&good, last) + 4);
    assert_eq!(kind(check_extents(&moved, &roomy)), ErrorKind::InvalidData);
}

#[test]
fn verify_finds_a_flipped_byte_in_any_section() {
    let (bytes, _) = built("verify.coldivf", 128);
    let layout = read_header(&bytes, bytes.len()).unwrap();

    for part in [
        Part::Rotation,
        Part::Centers,
        Part::Extents,
        Part::Codes,
        Part::Rows,
        Part::Ids,
    ] {
        let mut flipped = bytes.clone();
        flipped[layout.span(part).offset + 3] ^= 0x10;
        assert_eq!(kind(verify(&flipped, &layout)), ErrorKind::InvalidData, "{part:?}");
    }
}

#[test]
fn the_first_read_holds_the_rotation_up_to_2806_dimensions() {
    assert!(HEADER_BYTES + rotation_len(2806) <= FIRST_READ);
    assert!(HEADER_BYTES + rotation_len(2807) > FIRST_READ);
}

#[test]
fn extents_must_stay_inside_the_codes_section() {
    let (bytes, _) = built("bounds.coldivf", 128);
    let layout = read_header(&bytes, bytes.len()).unwrap();
    let good = &bytes[layout.span(Part::Extents).range()];

    // A codes section one line shorter than its clusters' blocks breaks only this rule.
    let mut tight = layout.clone();
    tight.spans[Part::Codes as usize].len -= LINE;
    assert_eq!(kind(check_extents(good, &tight)), ErrorKind::InvalidData);
}

/// Recomputes the checksum of every section that fits in the file, then the header's own, so that a damaged field
/// meets its own check instead of a checksum.
fn reseal(bytes: &mut [u8]) {
    let Ok((mut h, _)) = Header::read_from_prefix(bytes) else {
        return;
    };
    for e in h.sections.iter_mut().skip(1) {
        let (offset, len) = (e.offset.get() as usize, e.len.get() as usize);
        if let Some(end) = offset.checked_add(len).filter(|&end| end <= bytes.len()) {
            e.crc = U32::new(crc32fast::hash(&bytes[offset..end]));
        }
    }
    h.crc = U32::new(crc32fast::hash(&h.as_bytes()[..size_of::<Header>() - 4]));

    bytes[..size_of::<Header>()].copy_from_slice(h.as_bytes());
}

/// What a damaged file must do. Nothing it does may panic.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Expect {
    /// Opening fails with `InvalidData` and this message.
    OpenFails(&'static str),
    /// It opens, but every search fails with `InvalidData` and this message, and `verify` fails with `InvalidData`.
    SearchFails(&'static str),
    /// It opens, and `verify` fails with `InvalidData`. Searches may still work.
    VerifyFails,
    /// Every error is `InvalidData`, except that a query may be too far from a mean that the damage moved.
    NoPanic,
}

/// One call's error, as its kind and message.
type Outcome = Option<(ErrorKind, String)>;

fn outcome<T>(r: Result<T, Error>) -> Outcome {
    r.err().map(|e| (e.kind(), e.to_string()))
}

/// Writes `bytes` to `path`, opens it and runs every kind of search and `verify` on it. It panics with `name` when
/// the file doesn't do what `expect` says, or when anything panics.
fn check(name: &str, bytes: &[u8], path: &std::path::Path, expect: Expect) {
    std::fs::write(path, bytes).unwrap();
    let run = std::panic::catch_unwind(|| {
        let index = match crate::index::Inner::open(path) {
            Ok(inner) => crate::Index {
                inner: std::sync::Arc::new(inner),
            },
            Err(e) => return (outcome::<()>(Err(e)), Vec::new(), None),
        };
        let mut ctx = crate::SearchCtx::new(crate::Limits {
            max_k: 10,
            max_nprobe: 64,
            max_rerank: 100,
        });
        let sp = crate::SearchParams {
            k: 10,
            nprobe: 64,
            rerank: 100,
            reach: None,
        };
        let q = vec![0.5f32; 128];
        let mut routes = Vec::new();
        let mut searches = vec![
            outcome(index.search(&q, &sp, &mut ctx)),
            outcome(index.inner.search_mapped(&q, &sp, &mut ctx)),
            outcome(index.route(&q, 64, None, &mut ctx, &mut routes)),
        ];
        match index.fetch(&[0, 1, 2, 3], &mut ctx) {
            Ok(mut fetched) => {
                searches.push(outcome(fetched.prefetch_rows(&q, 50)));
                searches.push(outcome(fetched.search(&q, 10, 100)));
            }
            Err(e) => searches.push(outcome::<()>(Err(e))),
        }
        (None, searches, outcome(index.verify()))
    });
    let Ok((open, searches, verified)) = run else {
        panic!("{name}: a damaged file made the index panic");
    };

    let data = |o: &Outcome, message: Option<&str>| match o {
        Some((ErrorKind::InvalidData, m)) => message.is_none_or(|want| m == want),
        _ => false,
    };
    let far = |o: &Outcome| match o {
        Some((ErrorKind::InvalidInput, m)) => m == "the query is too far from the index's mean",
        _ => false,
    };
    let ok = match expect {
        Expect::OpenFails(message) => data(&open, Some(message)),
        Expect::SearchFails(message) => {
            open.is_none() && searches.iter().all(|s| data(s, Some(message))) && data(&verified, None)
        }
        Expect::VerifyFails => open.is_none() && data(&verified, None),
        Expect::NoPanic => [&open, &verified]
            .into_iter()
            .chain(&searches)
            .all(|o| o.is_none() || data(o, None) || far(o)),
    };
    assert!(
        ok,
        "{name}: opening gave {open:?}, searches {searches:?} and verify {verified:?}"
    );
}

#[test]
fn every_kind_of_damage_is_caught_by_its_own_check_without_a_panic() {
    let (_, file, _base) =
        crate::quantization::testutil::written("corrupt.coldivf", &params(Metric::L2), 3000, 128, 64);
    let (base, layout) = file.parts();
    let (base, layout) = (base.to_vec(), layout.clone());
    drop(file);
    let path = temp_file("corrupt-case.coldivf");
    let header = |edit: &dyn Fn(&mut Header)| {
        let mut b = base.clone();
        let (mut h, _) = Header::read_from_prefix(&b).unwrap();
        edit(&mut h);
        b[..size_of::<Header>()].copy_from_slice(h.as_bytes());
        reseal(&mut b);
        b
    };
    let at = |part: Part| layout.span(part).offset;
    let poke = |at: usize, value: &[u8], seal: bool| {
        let mut b = base.clone();
        b[at..at + value.len()].copy_from_slice(value);
        if seal {
            reseal(&mut b);
        }
        b
    };
    let flip = |at: usize| {
        let mut b = base.clone();
        b[at] ^= 0x10;
        b
    };
    let nan = f32::NAN.to_le_bytes();
    let factors = at(Part::Centers) + layout.nparents * sq8::center_row_bytes(128) + 64 * sq4::code_bytes(128);
    let groups = factors + 12 * 64;
    let (settings, section) = (
        "the header's settings are out of range",
        "a section is missing, misplaced or the wrong size",
    );
    let (extent, cluster) = (
        "a cluster's blocks are out of order, off their line or misnumbered",
        "a cluster's factors or group are out of range",
    );
    let open = Expect::OpenFails;
    let cases: Vec<(&str, Vec<u8>, Expect)> = vec![
        (
            "magic",
            header(&|h| h.magic = *b"COLDIVF1"),
            open("the file isn't an index"),
        ),
        (
            "version",
            header(&|h| h.version = U32::new(2)),
            open("the file has an unknown version or byte order"),
        ),
        (
            "byte order",
            header(&|h| h.endian = U32::new(0x0403_0201)),
            open("the file has an unknown version or byte order"),
        ),
        (
            "header checksum",
            flip(size_of::<Header>() - 2),
            open("the header's checksum doesn't match"),
        ),
        // The other sizes in the header match 64 dimensions, so only the dimension's range is wrong.
        (
            "dimension",
            header(&|h| {
                let rb = sq8::row_bytes(64);
                h.dim = U32::new(64);
                h.d_pad = U32::new(padded_dim(64) as u32);
                h.row_bytes = U32::new(rb as u32);
                h.rows_per_page = U32::new((4096 / rb) as u32);
            }),
            open(settings),
        ),
        ("padded dimension", header(&|h| h.d_pad = U32::new(192)), open(settings)),
        (
            "page size",
            header(&|h| {
                h.page_size = U32::new(4097);
                h.rows_per_page = U32::new((4097 / sq8::row_bytes(128)) as u32);
            }),
            open(settings),
        ),
        (
            "page size 0",
            header(&|h| {
                h.page_size = U32::new(0);
                h.rows_per_page = U32::new(0);
            }),
            open(settings),
        ),
        (
            "rows per page",
            header(&|h| h.rows_per_page = U32::new(h.rows_per_page.get() + 1)),
            open(settings),
        ),
        ("block", header(&|h| h.block = U32::new(64)), open(settings)),
        ("line size", header(&|h| h.line_size = U32::new(64)), open(settings)),
        (
            "row bytes",
            header(&|h| h.row_bytes = U32::new(h.row_bytes.get() + 8)),
            open(settings),
        ),
        ("no clusters", header(&|h| h.nlist = U32::new(0)), open(settings)),
        ("no groups", header(&|h| h.nparents = U32::new(0)), open(settings)),
        (
            "metric",
            header(&|h| h.metric = 7),
            open("the header names an unknown metric"),
        ),
        (
            "vector count",
            header(&|h| h.count = U64::new(h.count.get() + 1)),
            open(section),
        ),
        ("kind of codes", header(&|h| h.codes_kind = 2), open(settings)),
        (
            "section kind",
            header(&|h| h.sections[3].kind = U32::new(9)),
            open(section),
        ),
        (
            "section off its boundary",
            header(&|h| h.sections[4].offset = U64::new(h.sections[4].offset.get() + 8)),
            open(section),
        ),
        (
            "section past the end",
            header(&|h| h.sections[6].len = U64::new(h.sections[6].len.get() + (1 << 20))),
            open(section),
        ),
        (
            "sections overlap",
            header(&|h| h.sections[3].offset = h.sections[2].offset),
            open(section),
        ),
        (
            "section length",
            header(&|h| h.sections[2].len = U64::new(h.sections[2].len.get() + 4)),
            open(section),
        ),
        (
            "rotation checksum",
            flip(at(Part::Rotation) + 9),
            open("a section's checksum doesn't match"),
        ),
        (
            "rotation rounds",
            poke(at(Part::Rotation), &2u32.to_le_bytes(), true),
            open("the rotation's settings don't match the header"),
        ),
        (
            "mean not finite",
            poke(at(Part::Rotation) + rotation_len(128) - 4, &nan, true),
            open("the mean has a value that isn't finite"),
        ),
        ("cut short", base[..base.len() - 100].to_vec(), open(section)),
        ("header only", base[..HEADER_BYTES].to_vec(), open(section)),
        ("empty", Vec::new(), open("the file is too short for its header")),
        (
            "extent misnumbered",
            flip(at(Part::Extents) + 16 + 12),
            Expect::SearchFails(extent),
        ),
        (
            "extent off its line",
            poke(at(Part::Extents) + 16, &4u64.to_le_bytes(), true),
            Expect::SearchFails(extent),
        ),
        // The end of the first cluster's blocks would overflow.
        (
            "extent far past the end",
            poke(at(Part::Extents), &(u64::MAX - 127).to_le_bytes(), true),
            Expect::SearchFails("a cluster's blocks run past the codes section"),
        ),
        (
            "cluster factor not finite",
            poke(factors, &nan, true),
            Expect::SearchFails(cluster),
        ),
        (
            "group out of range",
            poke(groups, &u16::MAX.to_le_bytes(), true),
            Expect::SearchFails(cluster),
        ),
        ("codes flipped", flip(at(Part::Codes) + 77), Expect::VerifyFails),
        (
            "row not finite",
            poke(at(Part::Rows) + 128, &nan, false),
            Expect::VerifyFails,
        ),
        ("ids flipped", flip(at(Part::Ids) + 5), Expect::VerifyFails),
        (
            "codes rewritten and resealed",
            poke(at(Part::Codes), &[0xFF; 512], true),
            Expect::NoPanic,
        ),
    ];

    assert!(cases.len() >= 20);
    for (name, bytes, expect) in &cases {
        check(name, bytes, &path, *expect);
    }

    // Resealing fixes every checksum, so a flipped byte in the codes then passes `verify`.
    let mut resealed = flip(at(Part::Codes) + 77);
    reseal(&mut resealed);
    check("codes flipped and resealed", &resealed, &path, Expect::NoPanic);
    assert!(crate::Index::open(&path).unwrap().verify().is_ok());
    check("intact", &base, &path, Expect::NoPanic);
}

#[test]
fn ten_thousand_random_damages_never_panic() {
    let mut rng = SplitMix64::new(12);
    let (rows, mut clusters) = clustered(&mut rng, 300, 128, 24, 0.5);
    group(&mut clusters, 3);
    let index = InMemory::build(&params(Metric::L2), &rows, &clusters).unwrap();
    let path = temp_file("random-base.coldivf");
    write(&index, &path).unwrap();
    let base = std::fs::read(&path).unwrap();

    // Eight threads share the cases, each with its own file and its own random damages.
    std::thread::scope(|scope| {
        for t in 0..8 {
            let base = &base;
            scope.spawn(move || {
                let mut rng = SplitMix64::new(100 + t);
                let path = temp_file(&format!("random-case-{t}.coldivf"));
                for case in 0..1250 {
                    let damaged = damage(base, &mut rng);
                    check(&format!("thread {t}, case {case}"), &damaged, &path, Expect::NoPanic);
                }
            });
        }
    });
}

/// `base` with one random kind of damage.
fn damage(base: &[u8], rng: &mut SplitMix64) -> Vec<u8> {
    let mut b = base.to_vec();
    let len = b.len();
    let at = |rng: &mut SplitMix64, n: usize| (rng.next_u64() % n as u64) as usize;

    match rng.next_u64() % 6 {
        // A few flipped bits anywhere, then the same with the checksums made to match.
        kind @ (0 | 1) => {
            for _ in 0..1 + rng.next_u64() % 4 {
                let i = at(rng, len);
                b[i] ^= 1 << (rng.next_u64() % 8);
            }
            if kind == 1 {
                reseal(&mut b);
            }
        }
        // A run of random bytes, with matching checksums.
        2 => {
            let (start, run) = (at(rng, len), 1 + at(rng, 64));
            b[start..(start + run).min(len)]
                .iter_mut()
                .for_each(|x| *x = rng.next_u64() as u8);
            reseal(&mut b);
        }
        // A random value in a random header field, with matching checksums.
        3 => {
            let start = at(rng, size_of::<Header>() - 4);
            b[start..start + 4].copy_from_slice(&(rng.next_u64() as u32).to_le_bytes());
            reseal(&mut b);
        }
        // The file cut short.
        4 => b.truncate(at(rng, len)),
        // Random bytes from the start.
        _ => {
            let cut = at(rng, len);
            b[..cut].iter_mut().for_each(|x| *x = rng.next_u64() as u8);
        }
    }

    b
}
