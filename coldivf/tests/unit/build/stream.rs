//! Unit tests for the streaming writer.

use std::io::ErrorKind;

use super::*;
use crate::build::write;
use crate::quantization::SplitMix64;
use crate::quantization::testutil::{clustered, group, params, temp_file, temps_of};
use crate::{Anchor, LeafOrder, Metric};

#[test]
fn a_streamed_file_has_exactly_the_bytes_of_an_in_memory_build() {
    // Parts of 300 bytes hold a vector or none, so every cluster gets parts of its own, and some clusters span several
    // pages of rows. Larger parts hold several clusters.
    let cases = [
        (Metric::L2, LeafOrder::Chained, 128, 4096, 2000, 300),
        (Metric::L2, LeafOrder::Chained, 128, 4096, 2000, 64 << 10),
        (Metric::Cosine, LeafOrder::Id, 200, 16384, 1500, 20 << 10),
        (Metric::L2, LeafOrder::Chained, 768, 4096, 701, 100 << 10),
        (Metric::L2, LeafOrder::Id, 128, 4096, 33, 1 << 20),
    ];

    for ((i, (metric, order, dim, page_size, n, part_bytes)), anchor) in cases
        .into_iter()
        .enumerate()
        .flat_map(|c| [Anchor::Cluster, Anchor::Mean].map(|a| (c, a)))
    {
        let (rows, mut clusters) = clustered(&mut SplitMix64::new(70 + i as u64), n, dim, 12, 0.5);
        group(&mut clusters, 3);
        let p = BuildParams {
            order,
            page_size,
            anchor,
            ..params(metric)
        };
        let (memory, streamed) = (
            temp_file(&format!("mem-{i}-{anchor:?}.coldivf")),
            temp_file(&format!("stream-{i}-{anchor:?}.coldivf")),
        );
        write(&InMemory::build(&p, &rows, &clusters).unwrap(), &memory).unwrap();
        write_parts(&p, &rows, &clusters, &streamed, part_bytes).unwrap();

        let bytes = std::fs::read(&streamed).unwrap();
        assert!(
            std::fs::read(&memory).unwrap() == bytes,
            "case {i} with the {anchor:?} anchor"
        );
        assert_eq!(format::read_header(&bytes, bytes.len()).unwrap().anchor, anchor);
        assert!(temps_of(&streamed).is_empty());
        // The build takes the writer's lock, so it leaves the lock file, as a writer does.
        let spill = std::fs::read_dir(streamed.parent().unwrap())
            .unwrap()
            .flatten()
            .any(|e| {
                let name = e.file_name().to_string_lossy().into_owned();
                name.starts_with(&format!("{}.", streamed.file_name().unwrap().to_string_lossy()))
                    && !name.ends_with(".lock")
            });
        assert!(!spill, "case {i} left a spill or temporary file");
    }
}

#[test]
fn a_failed_streamed_build_leaves_no_files() {
    let (mut rows, mut clusters) = clustered(&mut SplitMix64::new(80), 500, 128, 6, 0.5);
    group(&mut clusters, 2);
    rows.data[300 * 128 + 5] = f32::NAN;
    let out = temp_file("failed-stream.coldivf");

    let err = write_parts(&params(Metric::L2), &rows, &clusters, &out, 4096).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert!(!out.exists() && temps_of(&out).is_empty());
    let name = out.file_name().unwrap().to_string_lossy().to_string();
    let left = std::fs::read_dir(out.parent().unwrap())
        .unwrap()
        .flatten()
        .any(|e| e.file_name().to_string_lossy().starts_with(&format!("{name}.")));
    assert!(!left, "a spill or temporary file was left");
}
