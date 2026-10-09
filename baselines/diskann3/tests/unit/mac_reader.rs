use std::io::ErrorKind;
use std::os::fd::AsRawFd;

use super::*;
use crate::testutil::{Scratch, aligned, byte_at, sectors_file};

const SECTORS: usize = 64;

/// Reads the sectors `ids` of the scratch file with `reader` and checks every byte.
fn read_and_check<R: AlignedFileReader<Alignment = A4096>>(reader: &mut R, ids: &[u64]) {
    let mut buf = aligned(ids.len() * SECTOR);
    let mut requests: Vec<AlignedRead<'_, u8, A4096>> = buf
        .chunks_mut(SECTOR)
        .zip(ids)
        .map(|(slice, &id)| AlignedRead::new(id * SECTOR as u64, slice).unwrap())
        .collect();
    reader.read(&mut requests).unwrap();

    for (r, &id) in requests.iter().zip(ids) {
        let want: Vec<u8> = (0..SECTOR as u64).map(|i| byte_at(id * SECTOR as u64 + i)).collect();
        assert_eq!(r.aligned_buf(), &want[..], "sector {id}");
    }
}

fn batches() -> Vec<Vec<u64>> {
    vec![
        vec![5],
        vec![0, 63],
        vec![7, 3, 3, 40, 12],
        (0..16).map(|i| (i * 13 % 64) as u64).collect(),
        vec![],
    ]
}

#[test]
fn the_pool_reader_reads_what_the_file_holds() {
    let dir = Scratch::new("pool");
    let path = dir.path().join("f.bin");
    sectors_file(&path, SECTORS);
    let pool = Arc::new(ReadPool::default());
    pool.reserve(3).unwrap();
    let counters = Arc::new(Counters::default());
    let factory = PoolFactory::new(&path, Arc::clone(&pool), Arc::clone(&counters)).unwrap();
    let mut reader = factory.build().unwrap();

    for ids in batches() {
        read_and_check(&mut reader, &ids);
    }

    let tally = counters.tally();
    assert_eq!(tally.reads, 24);
    assert_eq!(tally.calls, 5);
    assert_eq!(tally.bytes, 24 * SECTOR as u64);
}

#[test]
fn a_pool_without_threads_reads_on_the_calling_thread() {
    let dir = Scratch::new("inline");
    let path = dir.path().join("f.bin");
    sectors_file(&path, SECTORS);
    let pool = Arc::new(ReadPool::default());
    let factory = PoolFactory::new(&path, pool, Arc::new(Counters::default())).unwrap();
    let mut reader = factory.build().unwrap();

    for ids in batches() {
        read_and_check(&mut reader, &ids);
    }
}

/// Which pages of the file at `path` are in the page cache, one flag per page, after dropping them all first when
/// `drop` is set. It maps the file to ask and never reads the mapping.
fn cached(path: &Path, drop: bool) -> Vec<bool> {
    let file = File::open(path).unwrap();
    file.sync_all().unwrap();
    let len = file.metadata().unwrap().len() as usize;
    // SAFETY: sysconf takes a constant and touches no memory.
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
    // SAFETY: a read-only shared mapping of the whole open file, which is unmapped below and never read.
    let addr = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            len,
            libc::PROT_READ,
            libc::MAP_SHARED,
            file.as_raw_fd(),
            0,
        )
    };
    assert_ne!(addr, libc::MAP_FAILED);

    if drop {
        // SAFETY: the span is the mapping made above, and the file has no unwritten pages after sync_all.
        assert_eq!(unsafe { libc::msync(addr, len, libc::MS_INVALIDATE) }, 0);
    }

    let mut flags = vec![0 as libc::c_char; len.div_ceil(page)];
    // SAFETY: `flags` holds one byte for each page of the mapping.
    assert_eq!(unsafe { libc::mincore(addr, len, flags.as_mut_ptr()) }, 0);
    // SAFETY: the mapping made above, which nothing uses after this.
    unsafe { libc::munmap(addr, len) };
    flags.iter().map(|&f| f as i32 & libc::MINCORE_INCORE != 0).collect()
}

#[test]
fn a_prefilling_pool_reader_leaves_its_sectors_in_the_page_cache() {
    let dir = Scratch::new("prefill");
    let path = dir.path().join("f.bin");
    sectors_file(&path, SECTORS);
    let pool = Arc::new(ReadPool::default());
    pool.reserve(2).unwrap();
    let factory = PoolFactory::new(&path, pool, Arc::new(Counters::default())).unwrap();
    let mut reader = factory.build().unwrap();
    // The switch reaches readers built before it was turned on.
    factory.prefill().store(true, Ordering::Relaxed);
    assert!(reader.prefill.load(Ordering::Relaxed));
    let ids = [5u64, 9, 40];
    assert_eq!(cached(&path, true), vec![false; cached(&path, false).len()]);
    read_and_check(&mut reader, &ids);
    // SAFETY: sysconf takes a constant and touches no memory.
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as u64;
    let start = Instant::now();

    // F_RDADVISE reads in the background, so the pages may arrive a moment after the read returns.
    while !ids
        .iter()
        .all(|&id| cached(&path, false)[(id * SECTOR as u64 / page) as usize])
    {
        assert!(
            start.elapsed().as_secs() < 5,
            "the prefilled sectors never reached the page cache"
        );
        std::thread::sleep(std::time::Duration::from_millis(10));
    }

    factory.prefill().store(false, Ordering::Relaxed);
    read_and_check(&mut reader, &ids);
}

#[test]
fn readers_built_by_one_factory_share_its_handle() {
    let dir = Scratch::new("share");
    let path = dir.path().join("f.bin");
    sectors_file(&path, SECTORS);
    let factory = PoolFactory::new(&path, Arc::new(ReadPool::default()), Arc::new(Counters::default())).unwrap();
    let (a, b) = (factory.build().unwrap(), factory.build().unwrap());
    assert!(Arc::ptr_eq(&a.file, &b.file));
    let factory = AdviseFactory::new(&path, Arc::new(Counters::default())).unwrap();
    let (a, b) = (factory.build().unwrap(), factory.build().unwrap());
    assert!(Arc::ptr_eq(&a.file, &b.file));
}

#[test]
fn a_failed_pool_read_reports_its_error_after_every_job_ends() {
    let dir = Scratch::new("eof");
    let path = dir.path().join("f.bin");
    sectors_file(&path, 4);
    let pool = Arc::new(ReadPool::default());
    pool.reserve(2).unwrap();

    for ids in [[1, 9, 2, 3], [9, 1, 2, 3]] {
        let mut buf = aligned(ids.len() * SECTOR);
        let mut requests: Vec<AlignedRead<'_, u8, A4096>> = buf
            .chunks_mut(SECTOR)
            .zip(ids)
            .map(|(slice, id)| AlignedRead::new(id * SECTOR as u64, slice).unwrap())
            .collect();
        let file = Arc::new(open_uncached(&path).unwrap());
        let e = pool.read(&file, &mut requests).unwrap_err();
        assert_eq!(e.kind(), ErrorKind::UnexpectedEof);
        // The sectors that exist were read in full before the call returned.
        assert_eq!(requests[2].aligned_buf()[1], byte_at(2 * SECTOR as u64 + 1));
        assert_eq!(requests[3].aligned_buf()[SECTOR - 1], byte_at(4 * SECTOR as u64 - 1));
    }
}

#[test]
fn many_callers_share_one_pool() {
    let dir = Scratch::new("callers");
    let path = dir.path().join("f.bin");
    sectors_file(&path, SECTORS);
    let pool = Arc::new(ReadPool::default());
    pool.reserve(4).unwrap();
    let factory = PoolFactory::new(&path, pool, Arc::new(Counters::default())).unwrap();

    std::thread::scope(|scope| {
        for t in 0..6u64 {
            let mut reader = factory.build().unwrap();
            scope.spawn(move || {
                for round in 0..50u64 {
                    let ids: Vec<u64> = (0..8).map(|i| (t * 11 + round * 5 + i * 7) % SECTORS as u64).collect();
                    read_and_check(&mut reader, &ids);
                }
            });
        }
    });
}

#[test]
fn the_advise_reader_reads_what_the_file_holds() {
    let dir = Scratch::new("advise");
    let path = dir.path().join("f.bin");
    sectors_file(&path, SECTORS);
    let counters = Arc::new(Counters::default());
    let factory = AdviseFactory::new(&path, Arc::clone(&counters)).unwrap();
    let mut reader = factory.build().unwrap();

    for ids in batches() {
        read_and_check(&mut reader, &ids);
    }

    assert_eq!(counters.tally().reads, 24);
}

#[test]
fn the_advise_reader_reports_a_read_past_the_end() {
    let dir = Scratch::new("advise-eof");
    let path = dir.path().join("f.bin");
    sectors_file(&path, 2);
    let mut reader = AdviseFactory::new(&path, Arc::new(Counters::default()))
        .unwrap()
        .build()
        .unwrap();
    let mut buf = aligned(SECTOR);
    let mut requests = [AlignedRead::new(8 * SECTOR as u64, &mut buf[..]).unwrap()];
    assert!(reader.read(&mut requests).is_err());
}

#[test]
fn the_stock_reader_copies_the_file_once_for_every_reader() {
    let dir = Scratch::new("stock");
    let path = dir.path().join("f.bin");
    sectors_file(&path, SECTORS);
    let counters = Arc::new(Counters::default());
    let factory = StockFactory::new(&path, Arc::clone(&counters)).unwrap();
    let (mut a, b) = (factory.build().unwrap(), factory.build().unwrap());
    assert!(Arc::ptr_eq(&a.reader, &b.reader));
    let mut buf = vec![0u8; 3 * SECTOR];
    let mut requests: Vec<AlignedRead<'_, u8, A1>> = buf
        .chunks_mut(SECTOR)
        .zip([4u64, 0, 63])
        .map(|(slice, id)| AlignedRead::new(id * SECTOR as u64, slice).unwrap())
        .collect();
    a.read(&mut requests).unwrap();
    assert_eq!(requests[0].aligned_buf()[10], byte_at(4 * SECTOR as u64 + 10));
    assert_eq!(requests[2].aligned_buf()[0], byte_at(63 * SECTOR as u64));
    assert_eq!(counters.tally().reads, 3);
}

#[test]
fn reads_off_the_sector_grid_are_refused() {
    let mut buf = aligned(2 * SECTOR);
    assert!(AlignedRead::<u8, A4096>::new(512, &mut buf[..SECTOR]).is_err());
    assert!(AlignedRead::<u8, A4096>::new(0, &mut buf[..100]).is_err());
    assert!(AlignedRead::<u8, A4096>::new(0, &mut buf[512..512 + SECTOR]).is_err());
    assert!(AlignedRead::<u8, A4096>::new(SECTOR as u64, &mut buf[SECTOR..]).is_ok());
}

#[test]
fn the_pool_only_grows() {
    let pool = ReadPool::default();
    assert_eq!(pool.threads(), 0);
    assert_eq!(pool.reserve(2).unwrap(), 2);
    assert_eq!(pool.reserve(1).unwrap(), 2);
    assert_eq!(pool.reserve(4).unwrap(), 4);
    assert_eq!(pool.threads(), 4);
}

#[test]
fn a_tally_subtracts_an_earlier_one() {
    let later = Tally {
        reads: 10,
        calls: 3,
        bytes: 40960,
        nanos: 900,
    };
    let earlier = Tally {
        reads: 4,
        calls: 1,
        bytes: 16384,
        nanos: 1000,
    };
    assert_eq!(
        later.since(earlier),
        Tally {
            reads: 6,
            calls: 2,
            bytes: 24576,
            nanos: 0
        }
    );
}

#[test]
fn advice_past_the_end_is_harmless() {
    let dir = Scratch::new("advice");
    let path = dir.path().join("f.bin");
    sectors_file(&path, 1);
    let file = open_uncached(&path).unwrap();
    advise(&file, 0, SECTOR).unwrap();
    advise(&file, 1 << 30, SECTOR).unwrap();
}
