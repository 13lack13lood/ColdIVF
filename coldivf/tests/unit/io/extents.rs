//! Unit tests for the extent cache.

use super::*;

/// `n` pages of 4096 bytes that differ from page to page, and with `seed`.
fn pages(n: usize, seed: u8) -> Vec<u8> {
    (0..n * 4096).map(|i| (i / 4096 * 31 + i % 251) as u8 ^ seed).collect()
}

#[test]
fn a_page_is_kept_the_second_time_it_is_read() {
    let mut cache = ExtentCache::new(64, 4096, 100 * 4096);
    let bytes = pages(3, 1);
    let mut out = vec![0u8; 3 * 4096];

    assert!(!cache.get(8192, &mut out));
    cache.offer(8192, &bytes);
    assert!(!cache.get(8192, &mut out), "kept after one read");
    cache.offer(8192, &bytes);
    assert!(cache.get(8192, &mut out));
    assert_eq!(out, bytes);

    // Part of what was read comes from here too, but not a range with a page more.
    let mut part = vec![0u8; 4096];
    assert!(cache.get(12_288, &mut part));
    assert_eq!(part, &bytes[4096..8192]);
    let mut more = vec![0u8; 4 * 4096];
    assert!(!cache.get(8192, &mut more));
}

#[test]
fn ranges_off_the_pages_and_the_file_s_part_page_never_come_from_here() {
    let len = 10 * 4096 + 100;
    let mut cache = ExtentCache::new(64, 4096, len);
    // The last whole page, and the 100 bytes after it.
    let bytes = pages(2, 2);
    cache.offer(9 * 4096, &bytes[..4096 + 100]);
    cache.offer(9 * 4096, &bytes[..4096 + 100]);

    let mut out = vec![0u8; 4096];
    assert!(cache.get(9 * 4096, &mut out));
    assert_eq!(out, &bytes[..4096]);
    let mut tail = vec![0u8; 4096 + 100];
    assert!(!cache.get(9 * 4096, &mut tail));
    assert!(!cache.get(10 * 4096, &mut out));
    assert!(!cache.get(9 * 4096 + 1, &mut out));

    // Bytes that start off a page offer only the whole pages after it.
    let mut cache = ExtentCache::new(64, 4096, len);
    let bytes = pages(3, 3);
    cache.offer(4096 - 10, &bytes[..2 * 4096 + 10]);
    cache.offer(4096 - 10, &bytes[..2 * 4096 + 10]);
    assert!(cache.get(4096, &mut out));
    assert_eq!(out, &bytes[10..4096 + 10]);
    assert!(cache.get(2 * 4096, &mut out));
    assert!(!cache.get(0, &mut out));
}

#[test]
fn a_page_takes_the_slot_of_the_page_that_held_it() {
    let mut cache = ExtentCache::new(4, 4096, 1000 * 4096);
    let a = 0;
    let b = (1..1000).find(|&p| cache.slot(p) == cache.slot(a)).unwrap();
    let (first, second) = (pages(1, 4), pages(1, 5));
    let mut out = vec![0u8; 4096];

    cache.offer(a * 4096, &first);
    cache.offer(a * 4096, &first);
    assert!(cache.get(a * 4096, &mut out));
    cache.offer(b * 4096, &second);
    // Noted once, it doesn't push the kept page out yet.
    assert!(cache.get(a * 4096, &mut out));
    cache.offer(b * 4096, &second);
    assert!(!cache.get(a * 4096, &mut out));
    assert!(cache.get(b * 4096, &mut out));
    assert_eq!(out, second);
}

#[test]
fn a_cache_has_the_slots_its_file_can_fill_as_a_power_of_2() {
    assert_eq!(ExtentCache::new(100, 4096, 1 << 30).keys.len(), 64);
    assert_eq!(ExtentCache::new(4096, 4096, 10 * 4096 + 5).keys.len(), 8);
    assert_eq!(ExtentCache::new(4096, 4096, 4000).keys.len(), 0);

    // With no slots it keeps nothing.
    let mut none = ExtentCache::new(0, 4096, 1 << 20);
    let bytes = pages(1, 6);
    none.offer(0, &bytes);
    none.offer(0, &bytes);
    assert!(!none.get(0, &mut vec![0u8; 4096]));
    assert!(none.bytes.is_empty());
}
