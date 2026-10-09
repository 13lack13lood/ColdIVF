//! Unit tests for the maintenance benchmark.

use coldivf_bench::data::write_fbin;

use super::*;

/// A folder of its own in the system temporary folder, which tests may write to. It starts empty.
fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("coldivf-bench/{}-maintain-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Writes the dataset `synth-euclidean` to `dir`: 8 clusters of 400 rows of 128 values, and 40 queries near rows. Each
/// cluster's rows spread over 8 directions and barely off them, as embeddings do, so a search can find their
/// nearest neighbors.
fn dataset(dir: &Path) {
    let (dim, mut rng) = (128, Rng::new(31));
    let mut unit = move || (rng.next_u64() >> 11) as f32 / (1u64 << 53) as f32 - 0.5;
    let centers: Vec<f32> = (0..8 * dim).map(|_| 4.0 * unit()).collect();
    let basis: Vec<f32> = (0..8 * dim).map(|_| unit()).collect();
    let mut rows = Vec::with_capacity(3200 * dim);

    for i in 0..3200 {
        let z: Vec<f32> = (0..8).map(|_| 4.0 * unit()).collect();

        for j in 0..dim {
            let spread: f32 = (0..8).map(|k| z[k] * basis[k * dim + j]).sum();
            rows.push(centers[i % 8 * dim + j] + spread + 0.02 * unit());
        }
    }

    let queries: Vec<f32> = (0..40 * dim)
        .map(|i| rows[i / dim * 79 % 3200 * dim + i % dim] + 0.05 * unit())
        .collect();
    write_fbin(&dir.join("synth-euclidean.base.fbin"), &rows, dim).unwrap();
    write_fbin(&dir.join("synth-euclidean.query.fbin"), &queries, dim).unwrap();
}

/// The settings of a three-round replay in `dir`, with `extra` arguments after them.
fn config(dir: &Path, extra: &[&str]) -> Config {
    let (data, index, results) = (
        dir.display().to_string(),
        dir.join("x.coldivf").display().to_string(),
        dir.join("results.jsonl").display().to_string(),
    );
    let mut args = vec![
        "--dataset",
        "synth-euclidean",
        "--data",
        &data,
        "--index",
        &index,
        "--label",
        "t",
        "--results",
        &results,
        "--base",
        "1000",
        "--rounds",
        "3",
        "--insert",
        "150",
        "--delete",
        "0.05",
        "--upsert",
        "0.05",
        "--pseudo",
        "300",
        "--queries",
        "40",
        "--upsert-pool",
        "400",
        "--threads",
        "2",
        "--build-threads",
        "2",
        "--vectors-per-list",
        "50",
        "--parents",
        "4",
        "--settings",
        "threads=2,vectors_per_list=50",
    ];
    args.extend(extra);
    Config::from(&Args::from(args.into_iter().map(str::to_string).collect()))
}

/// The lines of the results file in `dir`.
fn lines(dir: &Path) -> Vec<String> {
    let text = std::fs::read_to_string(dir.join("results.jsonl")).unwrap();
    text.lines().map(str::to_string).collect()
}

/// `line` without what varies from one run to the next: its usage, which comes last, and the index's path.
fn stable(line: &str, dir: &Path) -> String {
    let cut = line.find(",\"usage\":").unwrap();
    line[..cut].replace(&dir.display().to_string(), "DIR") + "}"
}

#[test]
fn a_replay_writes_a_line_a_round_and_a_killed_one_resumes_to_the_same_lines() {
    let whole = scratch("whole");
    dataset(&whole);
    replay(&config(&whole, &[]), None).unwrap();
    let want = lines(&whole);
    assert_eq!(want.len(), 4, "round 0 and three rounds");

    for (r, line) in want.iter().enumerate() {
        assert_eq!(field(line, "round"), Some(&r.to_string()[..]));
        assert_eq!(field(line, "rounds"), Some("3"));
        // The run and the fresh build come from round 1 on. Each holds its measures.
        assert_eq!(line.contains("\"run\":null"), r == 0, "{line}");
        assert_eq!(line.contains("\"fresh\":null"), r == 0, "{line}");
        assert!(line.contains("\"maintained\":{\"lists\":"), "{line}");

        for set in ["tune", "test", "pseudo"] {
            assert!(line.contains(&format!("\"{set}\":{{\"queries\":")), "{set}: {line}");
        }

        assert!(!line.contains("\"at95\":null"), "{line}");
        assert!(
            line.contains("\"maintained\":{\"lists\":") && !line.contains("\"calibration\":null"),
            "{line}"
        );
    }

    // The live count follows the stream: 50 deleted, then 150 inserted, a round, and so on.
    let live: Vec<&str> = want.iter().map(|l| field(l, "live").unwrap()).collect();
    assert_eq!(live, ["1000", "1100", "1195", "1285"]);
    let counts: Vec<[&str; 3]> = want
        .iter()
        .map(|l| ["inserted", "deleted", "upserted"].map(|k| field(l, k).unwrap()))
        .collect();
    assert_eq!(
        counts,
        [
            ["1000", "0", "0"],
            ["150", "50", "50"],
            ["150", "55", "55"],
            ["150", "60", "60"]
        ]
    );

    // Killed after round 2 applied its changes, then after round 3's run, it resumes to the same lines.
    let killed = scratch("killed");
    dataset(&killed);
    replay(&config(&killed, &[]), Some(Stop::Applied(2))).unwrap();
    assert_eq!(lines(&killed).len(), 2);
    replay(&config(&killed, &[]), Some(Stop::Ran(3))).unwrap();
    assert_eq!(lines(&killed).len(), 3);
    replay(&config(&killed, &[]), None).unwrap();
    let got = lines(&killed);
    assert_eq!(got.len(), 4);

    for (g, w) in got.iter().zip(&want) {
        assert_eq!(stable(g, &killed), stable(w, &whole));
    }

    // A rerun of a finished replay writes nothing, and one with other settings won't resume.
    replay(&config(&killed, &[]), None).unwrap();
    assert_eq!(lines(&killed).len(), 4);
    let other = replay(&config(&killed, &["--settings", "minor.change=0.3"]), None);
    assert_eq!(other.unwrap_err().kind(), ErrorKind::InvalidInput);
    let other = replay(&config(&killed, &["--seed", "2"]), None);
    assert_eq!(other.unwrap_err().kind(), ErrorKind::InvalidInput);
}

#[test]
fn a_replay_runs_the_kind_it_is_given_and_skips_fresh_builds_it_isnt() {
    let dir = scratch("kinds");
    dataset(&dir);
    replay(
        &config(
            &dir,
            &["--kind", "major", "--fresh", "2", "--rounds", "2", "--calibrate", "off"],
        ),
        None,
    )
    .unwrap();
    let got = lines(&dir);
    assert_eq!(got.len(), 3);
    assert!(got[1].contains("\"run\":{\"kind\":\"major\""), "{}", got[1]);
    assert!(got[1].contains("\"fresh\":null"), "{}", got[1]);
    assert!(!got[2].contains("\"fresh\":null"), "{}", got[2]);
    assert!(got[2].contains("\"calibration\":null"), "{}", got[2]);
    // A major run measures every code again.
    let run = &got[2][got[2].find("\"run\":").unwrap()..];
    assert_eq!(field(run, "reencoded"), field(run, "count"));
}

#[test]
fn an_evaluation_averages_what_each_search_reads() {
    let dir = scratch("evaluate");
    dataset(&dir);
    let data = Fbin::open(&dir.join("synth-euclidean.base.fbin")).unwrap();
    let params = BuildParams {
        metric: coldivf::Metric::L2,
        vectors_per_list: 50,
        parents: 4,
        order: LeafOrder::Chained,
        anchor: Anchor::Cluster,
        page_size: 4096,
        threads: 2,
        seed: 7,
    };
    let path = dir.join("e.coldivf");
    Builder::new(params.clone()).build(&data.prefix(1000), &path).unwrap();
    let replay = Replay::new(&(0..1000).collect::<Vec<u32>>()).unwrap();
    let queries: Vec<f32> = (0..30).flat_map(|q| data.row(2000 + q).to_vec()).collect();
    let truth = Truth::new(&data, coldivf::Metric::L2, queries, 64, &replay.live(), 2);
    let judge = Judge {
        data: &data,
        replay: &replay,
    };
    let limits = Limits {
        max_k: K,
        max_nprobe: MAX_NPROBE,
        max_rerank: MAX_RERANK,
    };
    let mut ctx = SearchCtx::new(limits);
    let set: Vec<usize> = (0..30).step_by(3).collect();
    let s = SearchParams {
        k: K,
        nprobe: 3,
        rerank: 20,
        reach: None,
    };
    let p = Subject::new(&path, None)
        .unwrap()
        .evaluate(&s, &set, &truth, &judge, &mut ctx)
        .unwrap();
    // The same searches by hand.
    let (index, mut leaves, mut sums) = (Index::open(&path).unwrap(), Vec::new(), [0f64; 6]);

    for &q in &set {
        let query = truth.query(q);
        index.route(query, 3, None, &mut ctx, &mut leaves).unwrap();
        let hits = index.fetch(&leaves, &mut ctx).unwrap().search(query, K, 20).unwrap();
        let ids: Vec<u64> = hits.iter().map(|h| h.id).collect();
        let t = ctx.trace();
        let recall = truth.recall(q, &ids, |id| judge.vector(id));
        let read = [t.scanned, t.cluster_bytes, t.cluster_ranges, t.row_bytes, t.row_ranges].map(|x| x as f64);

        for (sum, x) in sums.iter_mut().zip(std::iter::once(recall).chain(read)) {
            *sum += x / set.len() as f64;
        }
    }

    let got = [p.recall, p.scanned, p.t1_bytes, p.t1_ranges, p.t2_bytes, p.t2_ranges];

    for (g, w) in got.iter().zip(sums) {
        assert!((g - w).abs() <= 1e-9 * w.abs().max(1.0), "{got:?} against {sums:?}");
    }

    assert!(
        p.nprobe == 3.0 && p.t1_bytes > 0.0 && p.t2_bytes > 0.0 && p.t1_bytes != p.t2_bytes,
        "{p:?}"
    );

    // A fresh build of the base in reverse holds index id 999 − i at its id i, and its hits count as those ids.
    let rows: Vec<f32> = (0..1000).rev().flat_map(|r| data.row(r).to_vec()).collect();
    write_fbin(&dir.join("reversed.fbin"), &rows, 128).unwrap();
    let reversed = Fbin::open(&dir.join("reversed.fbin")).unwrap();
    let fresh = dir.join("f.coldivf");
    Builder::new(params).build(&reversed.prefix(1000), &fresh).unwrap();
    let all = SearchParams { nprobe: 20, ..s };
    let subject = Subject::new(&fresh, Some((0..1000).rev().collect())).unwrap();
    let p = subject.evaluate(&all, &set, &truth, &judge, &mut ctx).unwrap();
    assert!(p.recall > 0.8, "{p:?}");
}

#[test]
fn every_setting_has_a_path_that_sets_only_it() {
    // A value off the default for each setting, in the order `settings` lists them.
    let changed = [
        ("vectors_per_list", "100"),
        ("rounds", "7"),
        ("parents", "33"),
        ("order", "id"),
        ("threads", "3"),
        ("seed", "99"),
        ("minor.change", "0.25"),
        ("minor.shrink", "0.3"),
        ("minor.hot_list", "5"),
        ("minor.list_loss", "0.6"),
        ("minor.region_lists", "12"),
        ("minor.split", "2.5"),
        ("minor.merge", "0.3"),
        ("minor.ring", "8"),
        ("minor.pool", "0.99"),
        ("minor.global_pass", "0.2"),
        ("minor.refresh", "true"),
        ("major.inserts", "none"),
        ("major.deletes", "0.4"),
        ("major.indicator", "1.03"),
        ("major.beta", "0.62"),
        ("major.sample", "64"),
    ];
    let defaults = settings(&MaintainParams::default());
    assert_eq!(defaults.len(), changed.len());

    for (i, &(path, text)) in changed.iter().enumerate() {
        let mut p = MaintainParams::default();
        set(&mut p, path, text).unwrap();
        let now = settings(&p);

        for (j, ((name, value), (_, default))) in now.iter().zip(&defaults).enumerate() {
            match i == j {
                true => assert_eq!((*name, value.text()), (path, text.to_string())),
                false => assert_eq!(value, default, "setting {path} changed {name}"),
            }
        }
    }

    // Every setting's text sets it back.
    let mut p = MaintainParams::default();
    apply(&mut p, &changed.map(|(a, b)| (a.to_string(), b.to_string()))).unwrap();
    let mut q = MaintainParams::default();

    for (path, value) in settings(&p) {
        set(&mut q, path, &value.text()).unwrap();
    }

    assert_eq!(format!("{q:?}"), format!("{p:?}"));
    assert_eq!(settings_text(&q), settings_text(&p));
    assert!(settings_text(&p).starts_with("vectors_per_list=100,rounds=7,"));
}

#[test]
fn a_bad_setting_fails() {
    let mut p = MaintainParams::default();

    for (path, text) in [
        ("minor.change", "x"),
        ("minor.chnge", "0.1"),
        ("order", "random"),
        ("minor.refresh", "1"),
        ("major.sample", "-1"),
        ("minor.split", "inf"),
        ("major.inserts", "NaN"),
    ] {
        assert!(set(&mut p, path, text).is_err(), "{path} = {text}");
    }

    assert_eq!(format!("{p:?}"), format!("{:?}", MaintainParams::default()));
    assert!(pairs("minor.change=0.2,major.sample").is_err());
    assert_eq!(
        pairs("minor.change = 0.2, major.sample=64,").unwrap(),
        [("minor.change", "0.2"), ("major.sample", "64")].map(|(a, b)| (a.to_string(), b.to_string()))
    );
}

#[test]
fn a_toml_file_sets_keys_by_their_section() {
    let text = "# Settings\nrounds = 15\n\n[minor]\nchange = 0.25 # a comment\npool = \"none\"\nsplit=3\n\n[ major ]\n\
                inserts = 0.5\nindicator = \"1.05\"  \n";
    let pairs = toml(text).unwrap();
    let want = [
        ("rounds", "15"),
        ("minor.change", "0.25"),
        ("minor.pool", "none"),
        ("minor.split", "3"),
        ("major.inserts", "0.5"),
        ("major.indicator", "1.05"),
    ];
    assert_eq!(pairs, want.map(|(a, b)| (a.to_string(), b.to_string())));
    let mut p = MaintainParams::default();
    apply(&mut p, &pairs).unwrap();
    assert_eq!(
        (p.rounds, p.minor.change, p.minor.pool, p.minor.split),
        (15, 0.25, None, 3.0)
    );
    assert_eq!((p.major.inserts, p.major.indicator), (Some(0.5), Some(1.05)));
    assert_eq!(uncomment("name = \"a#b\" # c"), "name = \"a#b\" ");
    assert!(toml("[minor]\nchange 0.25\n").is_err());
}

/// A point of the grid at position `i` whose searches find `recall`.
fn point(i: usize, recall: f64) -> Point {
    let x = (i + 1) as f64;
    Point {
        nprobe: 10.0 * x,
        recall,
        scanned: 100.0 * x,
        t1_bytes: 1000.0 * x,
        t1_ranges: x,
        t2_bytes: 2000.0 * x,
        t2_ranges: 3.0 * x,
    }
}

#[test]
fn the_crossing_lies_between_the_points_either_side_of_the_target() {
    let recalls = [0.5, 0.8, 0.9, 0.94, 0.96, 0.98, 0.99];

    for hint in 0..recalls.len() + 2 {
        let mut calls = Vec::new();
        let c = crossing(recalls.len(), hint, |i| {
            calls.push(i);
            Ok(point(i, recalls[i]))
        })
        .unwrap();
        // Halfway from 0.94 at position 3 to 0.96 at 4.
        let at = c.at.unwrap();
        assert!((at.recall - 0.95).abs() < 1e-12);
        assert!(
            (at.nprobe - 45.0).abs() < 1e-9 && (at.scanned - 450.0).abs() < 1e-9,
            "{at:?}"
        );
        assert!(
            (at.t1_bytes - 4500.0).abs() < 1e-9 && (at.t2_ranges - 13.5).abs() < 1e-9,
            "{at:?}"
        );
        // It evaluates each point from the hint to the target once.
        let hint = hint.min(recalls.len() - 1);
        let want: Vec<usize> = match hint <= 3 {
            true => (hint..=4).collect(),
            false => (3..=hint).rev().collect(),
        };
        assert_eq!(calls, want, "hint {hint}");
        let nprobes: Vec<f64> = c.points.iter().map(|p| p.nprobe).collect();
        let mut sorted = want.clone();
        sorted.sort_unstable();
        assert_eq!(
            nprobes,
            sorted.iter().map(|&i| 10.0 * (i + 1) as f64).collect::<Vec<_>>()
        );
    }

    // A grid that never reaches the target has no crossing.
    let c = crossing(3, 1, |i| Ok(point(i, 0.5))).unwrap();
    assert!(c.at.is_none());
    assert_eq!(c.points.len(), 2);

    // A grid that starts above it crosses from a point of zeros.
    let c = crossing(3, 2, |i| Ok(point(i, 0.97 + 0.01 * i as f64))).unwrap();
    let at = c.at.unwrap();
    assert!((at.nprobe - 10.0 * 0.95 / 0.97).abs() < 1e-9, "{at:?}");
    assert_eq!(c.points.len(), 3);
}

#[test]
fn a_grid_runs_up_to_the_cluster_count() {
    assert_eq!(grid(1), [1]);
    assert_eq!(grid(5), [1, 2, 3, 4, 5]);
    assert_eq!(grid(6), [1, 2, 3, 4, 5, 6]);
    assert_eq!(grid(100).last(), Some(&100));
    assert_eq!(grid(100)[grid(100).len() - 2], 96);
    assert_eq!(grid(9000).last(), Some(&4096));
    assert!(grid(9000).windows(2).all(|w| w[0] < w[1]));
}

#[test]
fn a_run_round_trips_through_its_saved_text() {
    let ran = Ran {
        report: RunReport {
            kind: RunKind::Minor,
            regions_due: 3,
            reclustered: 1000,
            moved: 120,
            reencoded: 900,
            copied: 5000,
            count: 5900,
            lists: 40,
            regions: 6,
            distances: 123_456_789_012,
            bytes_read: 11,
            bytes_written: 12,
            cpu: Duration::from_nanos(13),
            wall: Duration::from_nanos(14),
        },
        cost: Cost {
            wall_ns: 15,
            cpu_ns: 16,
            disk_read: 17,
            disk_written: 18,
            footprint: 19,
            peak: 20,
        },
    };
    let (round, back) = Ran::parse(&ran.text(7)).unwrap();
    assert_eq!(round, 7);
    assert_eq!(format!("{:?}", back.report), format!("{:?}", ran.report));
    assert_eq!(back.cost, ran.cost);

    let dir = scratch("ran");
    let index = dir.join("x.coldivf");
    ran.save(&index, 7).unwrap();
    assert!(Ran::load(&index, 7).is_some());
    assert!(Ran::load(&index, 8).is_none());
    assert!(Ran::parse("kind=minor round=1").is_none());
    assert!(Ran::parse(&ran.text(1).replace("kind=minor", "kind=other")).is_none());
}

#[test]
fn a_fixed_setting_round_trips_through_its_text() {
    for s in [
        SearchParams {
            k: K,
            nprobe: 64,
            rerank: 39,
            reach: None,
        },
        SearchParams {
            k: K,
            nprobe: 7,
            rerank: 2048,
            reach: Some(0.25),
        },
    ] {
        let back = parse_fixed(&fixed_text(&s)).unwrap();
        assert_eq!((back.nprobe, back.rerank, back.reach), (s.nprobe, s.rerank, s.reach));
    }

    assert!(parse_fixed("64:39").is_none());
    assert!(parse_fixed("64:x:none").is_none());
}

#[test]
fn a_field_reads_numbers_and_strings_of_a_line() {
    let line = r#"{"label":"a","round":12,"rounds":3,"fixed_setting":"8:39:none","x":{"round":4}}"#;
    assert_eq!(field(line, "round"), Some("12"));
    assert_eq!(field(line, "rounds"), Some("3"));
    assert_eq!(field(line, "fixed_setting"), Some("8:39:none"));
    assert_eq!(field(line, "label"), Some("a"));
    assert_eq!(field(line, "missing"), None);
}

#[test]
fn a_results_file_cut_short_loses_its_last_line() {
    let dir = scratch("earlier");
    let path = dir.join("r.jsonl");
    std::fs::write(&path, "{\"round\":0}\n{\"round\":1}\n{\"round\":2,\"x\":").unwrap();
    let earlier = Earlier::read(&path).unwrap();
    assert_eq!(earlier.lines.len(), 2);
    assert_eq!(
        std::fs::read_to_string(&path).unwrap(),
        "{\"round\":0}\n{\"round\":1}\n"
    );
    std::fs::write(&path, "{\"round\":0}\n{\"round\":2}\n").unwrap();
    assert_eq!(Earlier::read(&path).unwrap_err().kind(), ErrorKind::InvalidData);
    assert!(Earlier::read(&dir.join("none.jsonl")).unwrap().lines.is_empty());

    // A whole last line without its newline goes too. The cut leaves every byte before it as it was, so it never
    // writes them again.
    std::fs::write(&path, "{\"round\":0}\r\n{\"round\":1}\r\n{\"round\":2}").unwrap();
    assert_eq!(Earlier::read(&path).unwrap().lines.len(), 2);
    assert_eq!(
        std::fs::read_to_string(&path).unwrap(),
        "{\"round\":0}\r\n{\"round\":1}\r\n"
    );
}

#[test]
fn a_round_whose_run_lost_its_report_says_so() {
    let dir = scratch("lost");
    dataset(&dir);
    let cfg = config(&dir, &["--rounds", "2", "--fresh", "none", "--calibrate", "off"]);
    // Killed after round 1's run renamed its new file in but before it saved its report, as removing the report
    // stands for.
    replay(&cfg, Some(Stop::Ran(1))).unwrap();
    std::fs::remove_file(side(&dir.join("x.coldivf"), "report")).unwrap();
    replay(&cfg, None).unwrap();
    let got = lines(&dir);
    assert_eq!(got.len(), 3);
    assert!(got[1].contains("\"run\":{\"lost\":true}"), "{}", got[1]);
    assert!(got[2].contains("\"run\":{\"kind\":"), "{}", got[2]);
}

#[test]
fn a_round_s_summary_compares_the_bytes_of_codes_with_the_fresh_build_s() {
    let measured = |t1_bytes: f64, t2_bytes: f64| Measured {
        lists: 1,
        count: 1,
        bytes: 1,
        calibration: None,
        sets: vec![SetMeasure {
            name: "tune",
            queries: 1,
            crossing: Crossing {
                at: Some(Point {
                    t1_bytes,
                    t2_bytes,
                    ..Point::default()
                }),
                points: Vec::new(),
            },
            fixed: Point::default(),
            calibrated: None,
        }],
    };
    // The maintained file reads 10% more codes but fewer rows, whose bytes depend on their order in the file too.
    let (maintained, fresh) = (measured(1100.0, 100.0), measured(1000.0, 300.0));
    let file = IndexFile {
        bytes: 1,
        modified: std::time::UNIX_EPOCH,
        epoch: 1,
        crc: 0,
        count: 1,
        nlist: 1,
        tables: 0..0,
        codes: 0..0,
        rows: 0..0,
    };
    let line = Line {
        round: 1,
        rounds: 1,
        base_epoch: 0,
        fixed: SearchParams {
            k: K,
            nprobe: 1,
            rerank: K,
            reach: None,
        },
        changes: [0; 4],
        ran: None,
        file: &file,
        maintained: &maintained,
        fresh: Some(&fresh),
        usage: Json::new(),
    };
    let text = summary(&line);
    assert!(text.contains("tune 1 KB (1.100x fresh)"), "{text}");
    assert!(!Point::default().json().line().contains("\"bytes\""));
}

#[test]
fn temporary_files_of_killed_builds_are_removed_and_others_kept() {
    let dir = scratch("temps");
    let index = dir.join("x.coldivf");
    let names = [
        ("x.coldivf.123.0.tmp", false),
        ("x.coldivf.fresh.9.12.spill", false),
        ("x.coldivf", true),
        ("x.coldivf.delta", true),
        ("x.coldivf.stream", true),
        ("x.coldivf.1.tmp", true),
        ("x.coldivf.a.0.tmp", true),
        ("y.coldivf.1.2.tmp", true),
    ];

    for (name, _) in names {
        std::fs::write(dir.join(name), b"x").unwrap();
    }

    remove_temps(&index).unwrap();

    for (name, kept) in names {
        assert_eq!(dir.join(name).exists(), kept, "{name}");
    }
}
