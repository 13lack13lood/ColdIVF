use super::*;

#[test]
fn the_counters_move_with_work() {
    let before = Usage::now();
    let mut x = 0u64;

    for i in 0..20_000_000u64 {
        x = x.wrapping_mul(6364136223846793005).wrapping_add(i);
    }

    std::hint::black_box(x);
    let block = vec![1u8; 32 << 20];
    std::hint::black_box(&block);
    let after = Usage::now();
    assert!(after.cpu_ns > before.cpu_ns);
    assert!(after.footprint > 0);
    assert!(after.peak_footprint >= after.footprint);
    assert!(after.peak_footprint >= 32 << 20);
}

#[test]
fn ticks_become_nanoseconds() {
    // Apple silicon counts in 125/3 ns units. Other Macs count in nanoseconds.
    let ns = ticks_to_ns(3_000);
    assert!(ns == 3_000 || ns == 125_000, "{ns}");
}
