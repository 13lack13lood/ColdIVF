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

    let tables = read_tables(&bytes, &layout).unwrap();
    assert_eq!(&bytes[tables.codes.range()], &index.centers[..]);
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
