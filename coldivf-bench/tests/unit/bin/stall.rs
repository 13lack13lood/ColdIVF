//! Unit tests for the stall probe.

use super::*;

/// A layout of 768 dimensions in 2 groups, with 5 rows to a page of 4 KiB, in a file of 1 MiB.
fn layout() -> Layout {
    Layout {
        len: 1 << 20,
        dim: 768,
        padded: 768,
        groups: 2,
        page_size: 4096,
        per_page: 5,
        centers: 16384..20_000,
        extents: 20_480..20_528,
        codes: 32_768..131_072,
        rows: 131_072..1 << 20,
        ids: 0..0,
    }
}

/// One cluster's entry in the extents section.
fn extent(offset: u64, count: u32, first: u32) -> Vec<u8> {
    [
        offset.to_le_bytes().to_vec(),
        count.to_le_bytes().to_vec(),
        first.to_le_bytes().to_vec(),
    ]
    .concat()
}

#[test]
fn reads_round_out_to_pages_and_join_when_they_touch() {
    let page = PAGE;
    // The first two share a page, the third starts on the page after them, and the last is cut at the end of the file.
    let spans = [10..20, 30..page + 5, 2 * page..2 * page + 1, 5 * page + 7..6 * page + 7];
    assert_eq!(coalesce(&spans, 6 * page), [0..3 * page, 5 * page..6 * page]);
    // Reads that touch join, and reads a page apart don't.
    let touching = coalesce(&[0..1, page + 1..page + 2], 1 << 20);
    assert_eq!((touching.len(), touching.first()), (1, Some(&(0..2 * page))));
    assert_eq!(
        coalesce(&[0..1, 2 * page + 1..2 * page + 2], 1 << 20),
        [0..page, 2 * page..3 * page]
    );
    assert!(coalesce(&[], 100).is_empty());
}

#[test]
fn a_header_gives_the_sections_and_the_row_pages() {
    let mut head = [0u8; 324];
    head[..8].copy_from_slice(b"COLDIVF\0");
    head[8..12].copy_from_slice(&1u32.to_le_bytes());
    head[16..20].copy_from_slice(&760u32.to_le_bytes());
    head[20..24].copy_from_slice(&768u32.to_le_bytes());
    head[36..40].copy_from_slice(&2u32.to_le_bytes());
    head[48..52].copy_from_slice(&4096u32.to_le_bytes());
    head[64..68].copy_from_slice(&5u32.to_le_bytes());
    let sections = [
        (0u64, 4096u64),
        (4096, 3368),
        (16384, 3616),
        (20_480, 48),
        (32_768, 98_304),
        (131_072, 917_504),
        (1_048_576, 80),
    ];

    for (i, (offset, len)) in sections.iter().enumerate() {
        head[80 + 24 * i..88 + 24 * i].copy_from_slice(&offset.to_le_bytes());
        head[88 + 24 * i..96 + 24 * i].copy_from_slice(&len.to_le_bytes());
    }

    let want = Layout {
        dim: 760,
        ids: 1_048_576..1_048_656,
        ..layout()
    };
    assert_eq!(Layout::parse(&head, 1 << 20), Some(want));
    head[8] = 2;
    assert_eq!(Layout::parse(&head, 1 << 20), None);
}

#[test]
fn a_plan_reads_the_tables_the_probed_clusters_and_the_pages_of_the_rescored_rows() {
    let layout = layout();
    // 33 vectors take two blocks of 4·768 + 512 bytes, a cluster with none takes nothing, and 10 more take one block.
    let extents = [extent(0, 33, 0), extent(7168, 0, 33), extent(7168, 10, 33)].concat();
    // The file holds ids 100 to 142, at positions 0 to 42.
    let ids: Vec<u8> = (100u64..143).flat_map(|id| id.to_le_bytes()).collect();
    let positions = positions(&ids);
    assert_eq!((positions[&100], positions[&142]), (0, 42));

    // The groups' two centers of 776 bytes come before the tables.
    assert_eq!(layout.tables(), 16384 + 2 * 776..20_528);
    assert_eq!(
        layout.cluster_spans(&extents, &[2, 0, 1, 2]),
        [32_768..39_936, 39_936..43_520]
    );
    // Positions 0, 4 and 6 sit in pages 0 and 1, and position 42 in page 8.
    let page = |p: usize| 131_072 + p * 4096;
    assert_eq!(
        layout.row_spans(&[6, 42, 0, 4]),
        [page(0)..page(1), page(1)..page(2), page(8)..page(9)]
    );

    let plan = layout.plan(&extents, &positions, &[0, 2], &[106, 142, 100]).unwrap();
    let want = Plan {
        tables: 17_936..20_528,
        clusters: std::iter::once(32_768..49_152).collect(),
        rows: vec![131_072..147_456, 163_840..180_224],
    };
    assert_eq!(plan, want);
    // A hit whose id the file doesn't hold is an error.
    assert!(layout.plan(&extents, &positions, &[0], &[7]).is_err());
}
