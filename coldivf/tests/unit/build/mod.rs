//! Unit tests for building index files.

use std::io::ErrorKind;

use super::*;
use crate::Metric;
use crate::format::{read_header, verify};
use crate::quantization::SplitMix64;
use crate::quantization::testutil::{clustered, params, temp_file, temps_of};

#[test]
fn a_built_file_checks_out() {
    let (rows, clusters) = clustered(&mut SplitMix64::new(1), 300, 128, 6, 0.5);
    let out = temp_file("built.coldivf");
    build_with_clusters(&params(Metric::L2), &rows, &clusters, &out).unwrap();

    let bytes = fs::read(&out).unwrap();
    let layout = read_header(&bytes, bytes.len()).unwrap();
    verify(&bytes, &layout).unwrap();
    assert_eq!((layout.dim, layout.count, layout.nlist), (128, 300, 6));
    assert!(temps_of(&out).is_empty(), "a temporary file is still there");
}

#[test]
fn the_same_input_gives_the_same_bytes_and_a_rebuild_replaces_the_file() {
    let (rows, clusters) = clustered(&mut SplitMix64::new(2), 200, 128, 4, 0.5);
    let (a, b) = (temp_file("same-a.coldivf"), temp_file("same-b.coldivf"));
    build_with_clusters(&params(Metric::Cosine), &rows, &clusters, &a).unwrap();
    build_with_clusters(&params(Metric::Cosine), &rows, &clusters, &b).unwrap();
    assert_eq!(fs::read(&a).unwrap(), fs::read(&b).unwrap());

    let other_seed = BuildParams {
        seed: 99,
        ..params(Metric::Cosine)
    };
    build_with_clusters(&other_seed, &rows, &clusters, &a).unwrap();
    let bytes = fs::read(&a).unwrap();
    assert_eq!(read_header(&bytes, bytes.len()).unwrap().seed, 99);
}

#[test]
fn a_rejected_build_writes_nothing() {
    let (rows, clusters) = clustered(&mut SplitMix64::new(3), 50, 128, 2, 0.5);
    let out = temp_file("rejected.coldivf");
    let odd_pages = BuildParams {
        page_size: 5000,
        ..params(Metric::L2)
    };

    let err = build_with_clusters(&odd_pages, &rows, &clusters, &out).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert!(!out.exists() && temps_of(&out).is_empty());
}

#[test]
fn a_failed_rename_leaves_no_temporary_file() {
    let (rows, clusters) = clustered(&mut SplitMix64::new(4), 50, 128, 2, 0.5);
    // A folder where the index should go makes the rename fail after the temporary file is written.
    let out = temp_file("taken.coldivf");
    fs::create_dir_all(out.join("inside")).unwrap();

    assert!(build_with_clusters(&params(Metric::L2), &rows, &clusters, &out).is_err());
    assert!(temps_of(&out).is_empty(), "a temporary file is still there");
    fs::remove_dir_all(&*out).unwrap();
}

#[test]
fn every_build_gets_its_own_temporary_file() {
    let out = temp_file("unique.coldivf");
    let ((_, a), (_, b)) = (create_temp(&out).unwrap(), create_temp(&out).unwrap());
    assert_ne!(a, b);

    fs::remove_file(a).unwrap();
    fs::remove_file(b).unwrap();
}
