//! Unit tests for the cold harness.

use coldivf_bench::data::{write_fbin, write_ibin};

use super::*;

/// A folder of its own in the system temporary folder, which tests may write to.
fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("coldivf-bench/{}-cold-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Arguments as a C1 process reads them from a line of its input.
fn words(line: &str) -> Args {
    Args::from(line.split_whitespace().map(str::to_string).collect())
}

/// The calling thread's CPU time.
#[cfg(unix)]
fn thread_cpu() -> Duration {
    let mut t = libc::timespec { tv_sec: 0, tv_nsec: 0 };
    // SAFETY: clock_gettime writes one timespec into `t`, which outlives the call.
    unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut t) };
    Duration::new(t.tv_sec as u64, t.tv_nsec as u32)
}

#[test]
fn a_block_of_rounds_balances_each_combinations_place_and_the_one_before_it() {
    for n in 1..=7 {
        let block = balanced(n).len();
        assert_eq!(block, if n % 2 == 0 { n } else { 2 * n }, "{n} combinations");
        let order = schedule(3 * block, n, 7);
        assert_eq!(order.len(), 3 * block * n);

        for (b, rounds) in order.chunks(block * n).enumerate() {
            let (mut at, mut after) = (vec![0; n * n], vec![0; n * n]);

            for (r, round) in rounds.chunks(n).enumerate() {
                // Each round runs every combination once, at the places 0 to n - 1.
                let mut combos: Vec<usize> = round.iter().map(|t| t.2).collect();
                combos.sort();
                assert_eq!(combos, (0..n).collect::<Vec<_>>(), "{round:?}");
                assert!(
                    round
                        .iter()
                        .enumerate()
                        .all(|(place, t)| t.0 == b * block + r && t.1 == place),
                    "{round:?}"
                );
                round.iter().for_each(|t| at[t.2 * n + t.1] += 1);
                round.windows(2).for_each(|w| after[w[0].2 * n + w[1].2] += 1);
            }

            // Over a block each combination runs at each place, and right after each other one, equally often.
            assert!(at.iter().all(|&c| c == block / n), "{n} combinations: {at:?}");
            let pairs = (0..n * n).filter(|i| i / n != i % n);
            assert!(
                pairs.map(|i| after[i]).all(|c| c == block / n),
                "{n} combinations: {after:?}"
            );
        }
    }

    // The seed draws the order of each block's rounds, and the same seed gives the same run.
    assert_eq!(schedule(8, 4, 3), schedule(8, 4, 3));
    assert_ne!(schedule(8, 4, 3), schedule(8, 4, 4));
    // A run that ends inside a block still runs every combination in each round.
    for round in schedule(5, 4, 3).chunks(4) {
        let mut combos: Vec<usize> = round.iter().map(|t| t.2).collect();
        combos.sort();
        assert_eq!(combos, [0, 1, 2, 3]);
    }
}

#[test]
fn each_trial_of_a_run_searches_queries_that_no_other_trial_searches() {
    let all: Vec<usize> = (0..60).flat_map(|slot| trial_queries(5, slot, 3, 1000)).collect();
    let mut once = all.clone();
    once.sort();
    once.dedup();
    assert_eq!(once.len(), all.len());

    // A trial's queries follow each other from where the trial before it ended, and wrap around the queries.
    assert_eq!(trial_queries(5, 2, 3, 1000), [11, 12, 13]);
    assert_eq!(trial_queries(0, 1, 3, 4), [3, 0, 1]);
    assert_eq!(list(&[7, 8]), "7,8");
}

#[test]
fn the_vector_arrives_the_gap_after_the_timer_started_not_after_the_open() {
    for variant in VARIANTS {
        let start = Instant::now();
        // The open and prepare take 60 ms of the 150 ms gap.
        std::thread::sleep(Duration::from_millis(60));
        let waited = wait_for_vector(start, Duration::from_millis(150), variant);
        let at = start.elapsed();
        assert!(
            at >= Duration::from_millis(150),
            "{variant}: the search started {at:?} after the text"
        );
        assert!(
            at < Duration::from_millis(185),
            "{variant}: the search started {at:?} after the text"
        );
        assert!(
            waited > Duration::from_millis(50) && waited < at,
            "{variant}: it waited {waited:?}"
        );
    }
}

#[test]
fn an_open_longer_than_the_gap_starts_the_search_at_once() {
    let start = Instant::now();
    std::thread::sleep(Duration::from_millis(5));
    assert_eq!(wait_for_vector(start, Duration::from_millis(2), "idle"), Duration::ZERO);
}

#[test]
fn an_idle_wait_spins_only_its_last_5_ms_and_a_prewarmed_one_spins_it_all() {
    assert_eq!(asleep(Duration::from_millis(30), "idle"), Duration::from_millis(25));
    assert_eq!(asleep(Duration::from_millis(3), "idle"), Duration::ZERO);
    assert_eq!(asleep(Duration::from_millis(30), "prewarm"), Duration::ZERO);

    // The thread's own time shows which part of a 100 ms wait spun, whatever the other tests run meanwhile.
    #[cfg(unix)]
    {
        let spun = |variant: &str| {
            let before = thread_cpu();
            wait_for_vector(Instant::now(), Duration::from_millis(100), variant);
            thread_cpu() - before
        };
        let (idle, prewarm) = (spun("idle"), spun("prewarm"));
        assert!(idle < Duration::from_millis(30), "idle spun {idle:?}");
        assert!(prewarm > Duration::from_millis(15), "prewarm spun {prewarm:?}");
    }
}

#[test]
fn a_c1_command_names_its_trial_and_its_queries() {
    assert_eq!(command(&words("--trial 3 --queries 7,8"), 10), Ok((3, vec![7, 8])));
    assert_eq!(command(&words("--queries 9"), 10), Ok((0, vec![9])));
    assert!(command(&words("--trial 3"), 10).is_err());
    assert!(command(&words("--trial 3 --queries 7,10"), 10).is_err());
}

#[test]
fn a_manifest_lists_files_that_follow_each_other_from_id_0() {
    let dir = Path::new("/data");
    let (files, n) = parse_manifest("/x/a.coldivf 0 50\n\n  b.coldivf 50 25\n", dir).unwrap();
    assert_eq!(n, 75);
    let want = [
        Collection {
            path: PathBuf::from("/x/a.coldivf"),
            offset: 0,
            count: 50,
        },
        Collection {
            path: PathBuf::from("/data/b.coldivf"),
            offset: 50,
            count: 25,
        },
    ];
    assert_eq!(files, want);

    // A gap, an overlap, a line that doesn't start at 0, a short line, a bad number and no files at all.
    for bad in ["a 0 50\nb 60 5", "a 0 50\nb 40 5", "a 5 10", "a 0", "a 0 x", "\n"] {
        assert!(parse_manifest(bad, dir).is_err(), "{bad:?}");
    }
}

#[test]
fn a_file_must_hold_as_many_vectors_as_the_run_says() {
    let file = |count| Collection {
        path: PathBuf::from("c.coldivf"),
        offset: 0,
        count,
    };
    // Each header says 10 vectors.
    let ids = [header(100, 0..1, 0, 1), header(100, 0..1, 0, 2)];
    assert_eq!(check_counts(&[file(10), file(10)], &ids), Ok(()));
    let e = check_counts(&[file(10), file(9)], &ids).unwrap_err();
    assert!(e.contains("holds 10 vectors, not the 9"), "{e}");
    assert!(check_counts(&[file(11)], &ids[..1]).is_err());
}

#[test]
fn a_hit_of_a_collection_gets_its_id_in_the_union() {
    let file = |offset| Collection {
        path: PathBuf::from("c.coldivf"),
        offset,
        count: 50,
    };
    let hit = |id| Hit { id, score: 0.0 };
    let files = [file(0), file(50)];
    assert_eq!(union_ids(&[(1, hit(3)), (0, hit(3)), (1, hit(0))], &files), [53, 3, 50]);
}

#[test]
fn recall_counts_a_hit_as_near_as_the_kth_neighbor() {
    // Unit vectors at 0.1 radians apart, and row 11 a copy of row 9, the 10th nearest to the query at angle 0.
    let dir = scratch("ties");
    let mut base: Vec<f32> = (0..11)
        .flat_map(|i| [(i as f32 * 0.1).cos(), (i as f32 * 0.1).sin()])
        .collect();
    base.extend_from_within(18..20);
    write_fbin(&dir.join("ties-cosine.base.fbin"), &base, 2).unwrap();
    write_fbin(&dir.join("ties-cosine.query.fbin"), &[1.0, 0.0], 2).unwrap();
    // The truth puts the copy after the row it copies.
    write_ibin(
        &dir.join("ties-cosine.gt.ibin"),
        &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 10],
        12,
    )
    .unwrap();
    let dataset = Dataset::open_in(&dir, &dir, "ties-cosine", None).unwrap();

    // The copy isn't among the true 10, but it is as near as the 10th, so it counts with ties.
    let found = [0, 1, 2, 3, 4, 5, 6, 7, 8, 11];
    let r = judge(&dataset, 0, &found, 10);
    assert!((r.ids - 0.9).abs() < 1e-9, "{}", r.ids);
    assert_eq!(r.levels[..2], [1.0, 1.0]);
    // Row 10 is farther than the 10th, so it doesn't count.
    assert!((judge(&dataset, 0, &[0, 1, 2, 3, 4, 5, 6, 7, 8, 10], 10).levels[1] - 0.9).abs() < 1e-9);
    // Recall at 100 needs a search of 100 and a truth that wide, and recall at 10 a search of 10.
    assert!(r.levels[2].is_nan());
    assert!(judge(&dataset, 0, &found[..5], 5).levels[1].is_nan());
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn the_summary_leaves_out_trials_that_started_with_too_many_cached_pages() {
    let line = |cached: u32, checked: u32, total: f64| {
        format!(
            "{{\"seq\":0,\"pages_resident_at_start\":{cached},\"pages_checked\":{checked},\"total_us\":{total},\
             \"post_embedding_us\":1,\"from_vector_us\":2}}"
        )
    };
    let mut c0 = Summary::default();
    c0.add(&line(0, 100, 10.0), 0, 0.0);
    c0.add(&line(1, 100, 20.0), 1, 0.0);
    c0.add(&line(0, 100, 30.0), 2, 0.0);
    assert_eq!((c0.totals, c0.cached), (vec![10.0, 30.0], 1));
    assert_eq!((c0.firsts, c0.others), (vec![10.0], vec![30.0]));

    // C1 allows 0.5% of its bulk pages, so 5 cached of 1000 pass, and 6 don't.
    let mut c1 = Summary::default();
    c1.add(&line(5, 1000, 30.0), 1, C1_CACHED);
    c1.add(&line(6, 1000, 40.0), 0, C1_CACHED);
    assert_eq!((c1.totals, c1.cached, c1.firsts.len()), (vec![30.0], 1, 0));
}

#[test]
fn a_trial_drops_the_files_beside_the_index_that_a_reader_may_read() {
    // The library names files by their real paths, eg. /private/var for /var on macOS.
    let dir = std::fs::canonicalize(scratch("beside")).unwrap();
    let index = dir.join("i.coldivf");
    std::fs::write(&index, b"x").unwrap();
    assert_eq!(reader_files(&index), std::slice::from_ref(&index));

    std::fs::write(dir.join("i.coldivf.delta"), b"d").unwrap();
    std::fs::write(dir.join("i.coldivf.lock"), b"").unwrap();
    std::fs::create_dir(dir.join("i.coldivf.d")).unwrap();
    std::fs::write(dir.join("i.coldivf.d").join("log.1"), b"l").unwrap();
    std::fs::write(dir.join("i.coldivf.hot.ibin"), b"h").unwrap();
    let names: Vec<String> = reader_files(&index)
        .iter()
        .map(|p| p.strip_prefix(&dir).unwrap().display().to_string())
        .collect();
    // The data file comes first, and other files beside it aren't a reader's.
    assert_eq!(
        names,
        ["i.coldivf", "i.coldivf.delta", "i.coldivf.lock", "i.coldivf.d/log.1"]
    );

    // A link to the index file leads to the files beside the file it names, as the library finds them.
    #[cfg(unix)]
    {
        let other = dir.join("elsewhere");
        std::fs::create_dir(&other).unwrap();
        std::os::unix::fs::symlink(&index, other.join("link.coldivf")).unwrap();
        assert_eq!(reader_files(&other.join("link.coldivf")), reader_files(&index));
        assert_eq!(real_path(&other.join("link.coldivf")), index);
        // A link to a file that is gone, eg. after the first maintenance run, still leads to where the file was.
        std::os::unix::fs::symlink(dir.join("gone.coldivf"), other.join("gone.coldivf")).unwrap();
        assert_eq!(real_path(&other.join("gone.coldivf")), dir.join("gone.coldivf"));
    }

    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn a_check_counts_every_cached_page_of_the_index_and_the_files_beside_it() {
    let (dir, page) = (std::fs::canonicalize(scratch("check")).unwrap(), page_size());
    let (index, log) = (dir.join("i.coldivf"), dir.join("i.coldivf.delta"));
    std::fs::write(&index, vec![1u8; 3 * page]).unwrap();
    std::fs::write(&log, vec![2u8; page + 1]).unwrap();
    // Both files are cached, since they were just written and read.
    let _ = (std::fs::read(&index).unwrap(), std::fs::read(&log).unwrap());
    let watched = Watched::map(&[Collection {
        path: index.clone(),
        offset: 0,
        count: 0,
    }]);
    assert_eq!(watched.first(), index.as_path());
    let (cached, checked) = watched.cached();
    assert_eq!(checked, 5);
    assert!(cached > 0 && cached <= checked, "{cached} of {checked}");

    // A C1 check covers a range of the data file, eg. its last two pages.
    let (cached, checked) = residency(&watched.0[0].1, &(page..3 * page)).unwrap();
    assert_eq!(checked, 2);
    assert!(cached > 0 && cached <= checked, "{cached} of {checked}");
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn a_range_counts_the_pages_it_touches_within_the_file() {
    let page = page_size();
    assert_eq!(pages(&(0..1), 10), 1);
    assert_eq!(pages(&(page - 1..page + 1), 3 * page), 2);
    assert_eq!(pages(&(page..usize::MAX), 2 * page + 1), 2);
    assert_eq!(pages(&(5..9), 5), 0);
}

/// The header fields of an index file of 10 vectors whose cluster tables are at `tables`.
fn header(bytes: u64, tables: Range<usize>, epoch: u64, crc: u32) -> IndexFile {
    IndexFile {
        bytes,
        modified: SystemTime::UNIX_EPOCH + Duration::from_secs(epoch * 1000),
        epoch,
        crc,
        count: 10,
        nlist: 2,
        tables,
        codes: 0..0,
        rows: 0..0,
    }
}

#[test]
fn a_header_gives_the_dimension_the_groups_and_the_sections() {
    let mut head = [0u8; 324];
    head[..8].copy_from_slice(b"COLDIVF\0");
    head[8..12].copy_from_slice(&1u32.to_le_bytes());
    // 760 dimensions, padded to 768.
    head[16..20].copy_from_slice(&760u32.to_le_bytes());
    head[20..24].copy_from_slice(&768u32.to_le_bytes());
    // 6,250 clusters in 128 groups, and 5 rows to a page of 4 KiB.
    head[32..36].copy_from_slice(&6250u32.to_le_bytes());
    head[36..40].copy_from_slice(&128u32.to_le_bytes());
    head[48..52].copy_from_slice(&4096u32.to_le_bytes());
    head[64..68].copy_from_slice(&5u32.to_le_bytes());
    let sections = [
        (0u64, 4096u64),
        (4096, 3368),
        (16384, 184_728),
        (212_992, 100_000),
        (327_680, 5_000_000),
        (5_332_992, 800_000),
    ];

    for (i, (offset, len)) in sections.iter().enumerate() {
        head[80 + 24 * i..88 + 24 * i].copy_from_slice(&offset.to_le_bytes());
        head[88 + 24 * i..96 + 24 * i].copy_from_slice(&len.to_le_bytes());
    }

    let want = Layout {
        len: 1 << 20,
        dim: 760,
        padded: 768,
        groups: 128,
        page_size: 4096,
        per_page: 5,
        rotation: 4096..7464,
        centers: 16384..201_112,
        extents: 212_992..312_992,
        codes: 327_680..5_327_680,
        rows: 5_332_992..6_132_992,
    };
    assert_eq!(Layout::parse(&head, 1 << 20), Some(want));
    head[8] = 2;
    assert_eq!(Layout::parse(&head, 1 << 20), None);
}

#[test]
fn the_tables_start_after_the_groups_centers_and_both_reads_take_whole_pages() {
    let page = 16384;
    // 128 groups of 776-byte centers at 768 dimensions end at byte 115,712, in the 8th page, and the extents end in the
    // 13th, so the tables take 6 pages.
    let one = Layout {
        len: 1 << 30,
        dim: 768,
        padded: 768,
        groups: 128,
        page_size: 4096,
        per_page: 5,
        rotation: 4096..7464,
        centers: 16384..201_000,
        extents: 201_728..209_000,
        codes: 212_992..300_000,
        rows: 311_296..400_000,
    };
    assert_eq!(one.reads(page), (16384, 6 * 16384));

    // At 130 dimensions a group's center takes 144 bytes, not 138, so 29 of them end past the 5th page of 4 KiB.
    let small = Layout {
        dim: 130,
        groups: 29,
        centers: 16384..30_000,
        extents: 30_000..36_000,
        ..one.clone()
    };
    assert_eq!(small.reads(4096), (16384, 9 * 4096 - 5 * 4096));

    // At 3,072 dimensions the rotation runs past the first 16 KiB, so the open reads it again. The groups' centers end
    // at byte 427,008, in the 27th page, and the extents in the 38th.
    let wide = Layout {
        dim: 3072,
        rotation: 4096..17_544,
        centers: 32_768..600_000,
        extents: 606_208..620_000,
        ..one.clone()
    };
    assert_eq!(wide.reads(page), (16384 + 2 * 16384, 12 * 16384));
    assert_eq!(wide.reads(4096).0, 16384 + 4 * 4096);
    assert_eq!(open_reads(&[one, wide], page), (4 * 16384, 18 * 16384));
}

#[test]
fn several_files_are_named_by_their_sum_their_newest_and_their_checksums() {
    let line = |ids: &[IndexFile]| identity(ids, Json::new()).line();
    let one = header(100, 0..1, 3, 7);
    assert_eq!(line(&[header(100, 0..1, 3, 7)]), one.fields(Json::new()).line());

    let two = [header(100, 0..1, 3, 7), header(50, 0..1, 5, 8)];
    let text = line(&two);
    assert_eq!(field(&text, "index_bytes"), Some(150.0));
    assert_eq!(field(&text, "index_epoch"), Some(5.0));
    assert!(
        text.contains(&format!("\"index_mtime\":\"{}\"", utc(two[1].modified))),
        "{text}"
    );
    // A change to either checksum or to their order changes the hash.
    let other = line(&[header(100, 0..1, 3, 7), header(50, 0..1, 5, 9)]);
    let swapped = line(&[header(50, 0..1, 5, 8), header(100, 0..1, 3, 7)]);
    assert_ne!(field(&text, "index_crc"), field(&other, "index_crc"));
    assert_ne!(field(&text, "index_crc"), field(&swapped, "index_crc"));
}

#[test]
fn a_run_with_resident_shares_makes_a_combination_of_each() {
    let made = combos(&words("--variant idle,prewarm --resident 0,0.5"));
    let labels: Vec<String> = made.iter().map(Combo::label).collect();
    assert_eq!(
        labels,
        [
            "wave idle 0 r0",
            "wave idle 0 r0.5",
            "wave prewarm 0 r0",
            "wave prewarm 0 r0.5"
        ]
    );

    // A trial process gets its own share, and a run without shares names none.
    let args = made[1].args(9);
    assert_eq!(args[args.len() - 2..], ["--resident", "0.5"]);
    let plain = combos(&words("--variant idle"));
    assert_eq!(plain[0].label(), "wave idle 0");
    assert!(!plain[0].args(9).contains(&"--resident".to_string()));
}

#[test]
fn a_trial_picks_its_share_of_the_clusters_from_the_seed_and_its_number() {
    let picked = pick(100, 0.25, 1, 3);
    assert_eq!(picked.len(), 25);
    assert!(picked.windows(2).all(|p| p[0] < p[1]) && picked[24] < 100, "{picked:?}");
    assert_eq!(pick(100, 0.25, 1, 3), picked);
    assert_ne!(pick(100, 0.25, 1, 4), picked);
    assert_ne!(pick(100, 0.25, 2, 3), picked);
    // The picks of a larger share hold more of the clusters, and 0 and 1 pick none and all.
    assert_eq!(pick(100, 0.5, 1, 3).len(), 50);
    assert!(pick(100, 0.0, 1, 3).is_empty());
    assert_eq!(pick(100, 1.0, 1, 3), (0..100).collect::<Vec<_>>());
    // Over many trials every cluster gets picked about as often as the share says.
    let mut counts = [0; 100];
    (0..400)
        .flat_map(|t| pick(100, 0.25, 1, t))
        .for_each(|c| counts[c] += 1);
    assert!(counts.iter().all(|&c| (50..=150).contains(&c)), "{counts:?}");
}

#[test]
fn ranges_that_touch_or_overlap_join_into_one() {
    assert_eq!(joined(vec![5..9, 0..3, 3..4, 8..12, 20..21]), [0..4, 5..12, 20..21]);
    let inside = joined(vec![0..10, 2..3]);
    assert_eq!((inside.len(), inside.first()), (1, Some(&(0..10))));
    assert!(joined(Vec::new()).is_empty());
}

#[test]
fn the_extents_give_each_clusters_codes_and_rows() {
    let layout = Layout {
        len: 1 << 30,
        dim: 768,
        padded: 768,
        groups: 1,
        page_size: 4096,
        per_page: 5,
        rotation: 4096..7464,
        centers: 16384..20_000,
        extents: 20_480..20_528,
        codes: 1000..20_000,
        rows: 100_000..200_000,
    };
    let extent = |offset: u64, count: u32, first: u32| {
        [
            offset.to_le_bytes().to_vec(),
            count.to_le_bytes().to_vec(),
            first.to_le_bytes().to_vec(),
        ]
        .concat()
    };
    // 33 vectors take two blocks of 4·768 + 512 bytes, a cluster with none takes nothing, and 10 vectors from id 33
    // take one block, with their rows on the 7th to the 9th page.
    let extents = [extent(0, 33, 0), extent(7168, 0, 33), extent(7168, 10, 33)].concat();
    let page = |p: usize| 100_000 + p * 4096;
    let want = [
        Parts {
            codes: 1000..1000 + 2 * 3584,
            rows: page(0)..page(7),
        },
        Parts {
            codes: 8168..8168,
            rows: page(0)..page(0),
        },
        Parts {
            codes: 8168..8168 + 3584,
            rows: page(6)..page(9),
        },
    ];
    assert_eq!(layout.parts_of(&extents), want);
}

#[test]
fn a_trial_caches_the_clusters_it_picks_and_counts_only_other_pages_as_cached() {
    let (dir, page) = (std::fs::canonicalize(scratch("resident")).unwrap(), page_size());
    let index = dir.join("i.coldivf");
    std::fs::write(&index, vec![1u8; 8 * page]).unwrap();
    let watched = Watched::map(&[Collection {
        path: index.clone(),
        offset: 0,
        count: 0,
    }]);
    // Cluster 0 sits in page 0 with its rows in page 5, and cluster 1 in pages 1 and 2 with its rows in page 6.
    let parts = [
        Parts {
            codes: 10..page - 10,
            rows: 5 * page..5 * page + 1,
        },
        Parts {
            codes: page + 1..2 * page + 1,
            rows: 6 * page..7 * page,
        },
    ];
    let mut file = File::open(&index).unwrap();
    no_read_ahead(&file).unwrap();
    make_cold(&watched.all(), Duration::ZERO).unwrap();
    assert_eq!(watched.cached().0, 0);

    let loaded = load(&mut file, &parts, &[1]).unwrap();
    assert_eq!(loaded.clusters, 1);
    assert_eq!(loaded.pages, [page..3 * page, 6 * page..7 * page]);
    assert_eq!(watched.cached_in(&loaded.pages), 3);
    assert_eq!(watched.cached().0, 3);

    // Pages it didn't pick stay out of the cache until something reads them.
    let other = load(&mut file, &parts, &[0]).unwrap();
    assert_eq!(other.pages, [0..page, 5 * page..6 * page]);
    assert_eq!((watched.cached().0, watched.cached_in(&loaded.pages)), (5, 3));

    // A long run of pages, read in order, caches nothing past its end, which a handle that reads ahead would.
    let long = dir.join("long.coldivf");
    std::fs::write(&long, vec![2u8; 64 * page]).unwrap();
    File::open(&long).unwrap().sync_all().unwrap();
    let watched = Watched::map(&[Collection {
        path: long.clone(),
        offset: 0,
        count: 0,
    }]);
    let mut file = File::open(&long).unwrap();
    no_read_ahead(&file).unwrap();
    make_cold(&watched.all(), Duration::ZERO).unwrap();
    let run = Parts {
        codes: 0..16 * page,
        rows: 16 * page..16 * page,
    };
    load(&mut file, &[run], &[0]).unwrap();
    assert_eq!(watched.cached().0, 16);
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn a_pool_gives_the_trials_its_queries_in_turn_and_the_warm_up_another() {
    let pool = [18, 3, 51];
    let taken: Vec<Vec<usize>> = (0..4).map(|slot| pooled(&pool, 0, slot, 2, 1000)).collect();
    assert_eq!(taken, [vec![18, 3], vec![51, 18], vec![3, 51], vec![18, 3]]);
    assert_eq!(pooled(&pool, 1, 0, 1, 1000), [3]);
    // Without a pool the trials take every query in turn, and the warm-up the one after them.
    assert_eq!(pooled(&[], 5, 2, 3, 1000), [11, 12, 13]);
    assert_eq!(spare(&[], 120, 100), 20);
    // With one the warm-up is the first position past the pool's highest that the pool doesn't hold, wrapping around.
    assert_eq!(spare(&pool, 0, 1000), 52);
    assert_eq!(spare(&[8, 9, 0], 0, 10), 1);
}
