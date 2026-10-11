"""charts.py on a small tree of result files: its picks, the numbers it records, and its drawing where matplotlib is
installed."""

import contextlib
import importlib.util
import io
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import charts
import schema

HAS_MATPLOTLIB = importlib.util.find_spec("matplotlib") is not None
DATASET, N = "tiny-768-cosine", 1000
MACHINE = {"model": "Mac15,7", "brand": "Apple M3 Pro", "memsize": 1, "perflevels": [], "os": {}, "ssd": {}}
RQ = "rq1-default"


def driver_line(kind: str, system: str, batch: str, ts: str, index: str = "new", **fields) -> dict:
    """A line as the driver writes it, of a batch at time `ts` that searched the index named `index`."""
    line = schema.blank(kind)
    line.update(run_id="r", run_key="k", ts=ts, batch_id=batch, phase="smoke", system=system, dataset=DATASET, n=N,
                machine=MACHINE, load_before=[3.0, 2.0, 1.0], system_version=f"{system} 1.0",
                provenance={"head": "h1", "unstaged": False, "wheel_sha256": {system: "w1"},
                            "env": {"COLDIVF_BIN": "/frozen/bin"}},
                index_identity={"path": "/i", "inode": 1, "mtime_ns": 1, "bytes": 10, "checksum": index},
                gate={"mode": "warn", "passed": False, "solo": False, "cpu_idle_pct": 50.0, "problems": []})
    line.update(fields)
    return line


def warm(system, batch, ts, setting, p50, recall, **fields):
    defaults = {"mode": "whole", "threads": 1, "queries": 500, "passes": 2, "p50_us": p50, "recall10": recall,
                "search_params": setting, "footprint": 30_000_000, "resident_file_bytes": 900_000,
                "extra": {"footprint_since_open": 5_000_000}}
    return driver_line("warm", system, batch, ts, **(defaults | fields))


def cold(system, batch, ts, setting, total, **fields):
    defaults = {"regime": "c0", "trial": 0, "threads": 1, "cpu_state": "prewarm", "total_us": total, "recall10": 0.9,
                "search_params": setting, "footprint_delta": 4_000_000}
    return driver_line("cold", system, batch, ts, **(defaults | fields))


def own(variant, seq=0, cached=0, mode="wave", nprobe=96, rerank=35, total=6000.0):
    return {"seq": seq, "pages_resident_at_start": cached, "mode": mode, "variant": variant, "nprobe": nprobe,
            "rerank": rerank, "total_us": total, "recall10": 1.0, "footprint_delta": 7_000_000, "dataset": "dev",
            "n": N, "load_1": 2.0, "power": "AC Power", "low_power": False, "commit": "abc", "dirty": False}


def own_warm(nprobe, rerank, recall, p50=None, threads=1, qps=None):
    return {"threads": threads, "p50_us": p50, "qps": qps, "nprobe": nprobe, "rerank": rerank, "recall10": recall,
            "queries": 1000, "passes": 3, "split": "all", "dataset": "dev", "n": N, "load_1": 2.0, "power": "AC Power",
            "low_power": False, "time": "2026-10-10T08:00:00Z", "commit": "abc", "dirty": False}


def write(path: Path, lines: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(line) + "\n" for line in lines) + "not json\n")


class ChartsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self.tmp.name)
        self.suite = self.repo / "suite.toml"
        self.suite.write_text(f'phase = "smoke"\ndataset = "{DATASET}"\nn = {N}\n\n[[system]]\nname = "coldivf"\n\n'
                              '[[system]]\nname = "lancedb"\n')
        old, new = ("b1", "2026-10-09T10:00:00Z"), ("b2", "2026-10-10T10:00:00Z")
        # The smoke charts show ColdIVF's python variant, so the binaries' newest lines are left out.
        binaries = ("b9", "2026-10-11T10:00:00Z")
        python = [
            driver_line("build", "coldivf", *old, index="old", seconds=1.0, index_bytes=2000, peak_footprint=50),
            warm("coldivf", *old, {"nprobe": 8}, 90.0, 0.7, index_identity={"checksum": "old"}),
            cold("coldivf", *old, {"nprobe": 16}, 1000.0, index_identity={"checksum": "old"}),
            driver_line("build", "coldivf", *new, seconds=1.0, index_bytes=1000, peak_footprint=40),
            driver_line("build", "coldivf", *new, index="dummy", seconds=1.0, index_bytes=5, n=50),
            warm("coldivf", *new, {"nprobe": 16}, 250.0, 0.95),
            warm("coldivf", *new, {"nprobe": 8}, 150.0, 0.85),
            *(cold("coldivf", *new, {"nprobe": 16}, t) for t in (7000.0, 5000.0, 6000.0)),
            cold("coldivf", *new, {"nprobe": 16}, 100.0, status="rejected"),
            cold("coldivf", *new, {"nprobe": 16}, 200.0, regime="c1"),
        ]
        write(self.repo / "results/bench/smoke/coldivf.jsonl", [
            *(dict(line, variant="python", binding="python") for line in python),
            warm("coldivf", *binaries, {"nprobe": 16}, 50.0, 0.95), cold("coldivf", *binaries, {"nprobe": 16}, 900.0)])
        write(self.repo / "results/bench/smoke/lancedb.jsonl", [
            driver_line("build", "lancedb", "b3", "2026-10-09T12:00:00Z", seconds=1.0, index_bytes=3000,
                        peak_footprint=90, exact_f32_stored=True),
            warm("lancedb", "b3", "2026-10-09T12:00:00Z", {"nprobes": 8}, 4000.0, 0.9, overhead_us=100.0),
            cold("lancedb", "b3", "2026-10-09T12:00:00Z", {"nprobes": 8}, 30000.0, threads=10),
        ])
        quiet = self.repo / "results/quiet2"
        night, grid = ("q", "2026-10-10T07:00:00Z"), ("w", "2026-10-10T03:00:00Z")

        def rq(nprobes, totals, p50, recall):
            setting = {"nprobes": nprobes, "refine_factor": 4}
            return [*(cold("lancedb", *night, setting, t, config_id=RQ, split="test") for t in totals),
                    warm("lancedb", *grid, dict(reversed(setting.items())), p50, recall, config_id=RQ, queries=1000,
                         split="all")]

        write(quiet / "driver/m1a/lancedb.jsonl", [
            # The main cell. 20 is cheaper warm but has fewer trials, 12 is faster cold but doesn't reach 0.95, and
            # 24 reaches it with as many trials but costs more warm.
            *rq(18, (22000.0, 21000.0, 23000.0), 3000.0, 0.951),
            *rq(20, (20000.0, 20500.0), 2900.0, 0.956),
            *rq(12, (15000.0, 15500.0, 16000.0), 2000.0, 0.93),
            *rq(24, (24000.0, 25000.0, 26000.0), 3500.0, 0.97),
        ])
        write(quiet / "driver/m1a/faiss-ivf.jsonl", [
            *(cold("faiss-ivf", *night, {"nprobe": 8}, t, config_id="vpl", split="test") for t in (9e4, 8e4)),
            warm("faiss-ivf", *grid, {"nprobe": 8}, 900.0, 0.9, config_id="vpl", queries=1000, split="all"),
        ])
        write(quiet / "driver/m1a/coldivf.jsonl", [
            cold("coldivf", *night, {"nprobe": 96, "rerank": 35}, t, config_id="faiss-chained", split="test")
            for t in (6000.0, 6500.0)])
        write(quiet / "a3-m1a.jsonl", [own("idle", total=7000.0), own("prewarm"), own("prewarm", total=6400.0),
                                       own("prewarm", seq=1, total=1.0), own("prewarm", cached=5, total=2.0),
                                       own("prewarm", nprobe=32, rerank=39, total=5000.0)])
        write(quiet / "a5-mapped.jsonl", [own("prewarm", mode="mapped", nprobe=128, rerank=29, total=25000.0),
                                          own("prewarm", mode="wave", nprobe=128, rerank=29, total=7000.0)])
        write(quiet / "a7-aisaq.jsonl", [
            {"rounds": r, "ranges": 4, "page": 16384, "trials": 200, "p50_us": 160.0 * r, "p99_us": 200.0 * r,
             "round_p50_us": 160.0, "size": 1 << 30, "load_1": 5.0, "power": "AC Power", "low_power": False,
             "commit": "abc", "dirty": False} for r in (1, 12)])
        write(quiet / "b3-warm.jsonl", [own_warm(32, 39, 0.90, 182.0), own_warm(96, 39, 0.952, 246.0),
                                        own_warm(96, 35, 0.9513, 231.0), own_warm(128, 29, 0.9566, 260.0),
                                        own_warm(96, 35, 0.9513, qps=4000.0), own_warm(96, 35, 0.9513, threads=6,
                                                                                       qps=18000.0)])
        (quiet / "summary.md").write_text("```\n23:58:38 a3-m1a: CPU idle 85.16 91.21\n00:12:38 a5-mapped: CPU idle "
                                          "88.82 85.37\n01:16:25 b3-warm: CPU idle 89.76 87.62\n```\n")
        self.quiet = quiet

    def tearDown(self):
        self.tmp.cleanup()

    def data(self) -> dict:
        return charts.chart_data(self.suite, self.quiet, self.repo)

    def test_percentile_is_nearest_rank(self):
        values = [float(v) for v in range(1, 11)]
        self.assertEqual(charts.percentile(values, 50), 5.0)
        self.assertEqual(charts.percentile(values, 95), 10.0)
        self.assertEqual(charts.percentile(values, 99), 10.0)
        self.assertEqual(charts.percentile(values[:1], 99), 1.0)
        self.assertIsNone(charts.percentile([], 50))

    def test_a_smoke_pick_takes_the_newest_batch_and_the_build_it_searched(self):
        lines = charts.read_lines(self.repo / "results/bench/smoke/coldivf.jsonl")
        self.assertEqual([w["p50_us"] for w in charts.smoke_pick(lines, "coldivf", DATASET, N)["warm"]], [50.0])
        pick = charts.smoke_pick(lines, "coldivf", DATASET, N, "python")
        self.assertEqual([w["p50_us"] for w in pick["warm"]], [150.0, 250.0])
        self.assertEqual(sorted(c["total_us"] for c in pick["cold"]), [5000.0, 6000.0, 7000.0])
        self.assertEqual(pick["build"]["index_bytes"], 1000)
        summary = charts.cold_summary(pick["cold"])
        self.assertEqual((summary["trials"], summary["p50_us"], summary["p95_us"], summary["min_us"]),
                         (3, 6000.0, 7000.0, 5000.0))

    def test_the_matched_cell_reaches_the_target_with_the_most_trials_and_is_the_cheapest_warm(self):
        matched = self.data()["charts"][1]
        self.assertEqual(matched["id"], "cold-head-to-head")
        lance = [r for r in matched["rows"] if r["system"] == "lancedb"]
        self.assertEqual([r["setting"] for r in lance], [{"nprobes": 18, "refine_factor": 4}])
        self.assertEqual((lance[0]["recall10_warm"], lance[0]["recall10_warm_split"]), (0.951, "all"))
        self.assertEqual(lance[0]["recall_source"], "results/quiet2/driver/m1a/lancedb.jsonl")
        gone = {(g["system"], g.get("config_id"), g.get("job")): g["best_recall10_warm"] for g in matched["left_out"]}
        self.assertEqual(gone, {("faiss-ivf", "vpl", None): 0.9, ("coldivf", None, "a3-m1a"): 0.9})
        self.assertNotIn("faiss-ivf", {r["system"] for r in matched["rows"]})
        others = [(c["system"], c["setting"].get("nprobes") or c["setting"].get("nprobe"), c["trials"])
                  for c in matched["other_cells"]]
        self.assertEqual(others, [("faiss-ivf", 8, 2), ("lancedb", 12, 3), ("lancedb", 20, 2), ("lancedb", 24, 3)])
        self.assertEqual(next(r for r in matched["rows"] if r["system"] == "lancedb")["slowest_us"],
                         [21000.0, 22000.0, 23000.0])

        for r in matched["rows"]:
            if not r["modeled"]:
                self.assertGreaterEqual(r["recall10_warm"], charts.TARGET, r["label"])

    def test_a_matched_cell_is_picked_on_the_tune_split_and_shows_the_split_its_trials_searched(self):
        lines = [cold("x", "q", "t", {"k": 1}, 10.0, split="test"), cold("x", "q", "t", {"k": 2}, 5.0, split="test")]
        warm_lines = [warm("x", "w", "t", {"k": 1}, 100.0, 0.96, split="tune"),
                      warm("x", "w", "t", {"k": 1}, 100.0, 0.948, split="test"),
                      warm("x", "w", "t", {"k": 2}, 90.0, 0.94, split="tune")]

        def warm_of(line, splits):
            return charts.warm_at(warm_lines, "x", None, line["search_params"], splits)

        chosen, left_out = charts.matched_cells(lines, warm_of)
        self.assertEqual(left_out, [])
        ((cell, picked, shown),) = chosen
        self.assertEqual(cell[0]["search_params"], {"k": 1})
        self.assertEqual((picked["split"], picked["recall10"]), ("tune", 0.96))
        self.assertEqual((shown["split"], shown["recall10"]), ("test", 0.948))

    def test_own_cells_count_only_first_queries_that_started_cold(self):
        cells = charts.own_cells(charts.read_lines(self.quiet / "a3-m1a.jsonl"))
        trials = {(c["cpu_state"], c["setting"]["nprobe"]): c["trials"] for c in cells}
        self.assertEqual(trials, {("idle", 96): 1, ("prewarm", 96): 2, ("prewarm", 32): 1})
        self.assertEqual(charts.runner_idle((self.quiet / "summary.md").read_text())["a3-m1a"], [85.16, 91.21])

    def test_each_ratio_is_against_the_coldivf_row_measured_the_same_way(self):
        data = self.data()
        scenarios, head = data["charts"][:2]
        self.assertEqual(len({r["system"] for r in head["rows"]}), len(head["rows"]))
        self.assertTrue(all(r["harness"] == "driver" for r in head["rows"]))
        self.assertTrue(all(r["system"] == "coldivf" for r in scenarios["rows"]))
        rows = {r["label"]: r for r in scenarios["rows"] + head["rows"]}
        driver = rows["ColdIVF, through the driver"]
        self.assertEqual((driver["p50_us"], driver["recall10_warm"], driver["p50_ratio"]), (6000.0, 0.9513, 1.0))
        self.assertEqual(rows["LanceDB IVF_RQ"]["against"]["label"], "ColdIVF, through the driver")
        self.assertAlmostEqual(rows["LanceDB IVF_RQ"]["p50_ratio"], 22000 / 6000, places=5)
        self.assertAlmostEqual(rows["ColdIVF, own harness, idle core"]["p50_ratio"], 7000 / 6000, places=5)
        lazy = rows["ColdIVF's file, lazily mapped, prewarmed core (control)"]
        self.assertEqual((lazy["against"]["label"], lazy["against"]["p50_us"]),
                         ("ColdIVF, own harness, read waves, prewarmed core", 7000.0))
        self.assertAlmostEqual(lazy["p50_ratio"], 25000 / 7000, places=5)
        table = charts.headline_table(data, "cold-head-to-head")
        self.assertIn("| LanceDB IVF_RQ | nprobes 18, refine_factor 4 | 0.951 | 3 | 22.0 ms | 23.0 ms | 23.0 ms | "
                      "3.67x |", table)
        self.assertNotIn("lazily mapped", table)
        self.assertIn("| 25.0 ms | 25.0 ms | 25.0 ms | 3.57x |", charts.headline_table(data, "cold-coldivf"))
        self.assertEqual([m["label"] for m in head["models"]], ["AiSAQ-style graph, 12 rounds (modeled)"])
        self.assertAlmostEqual(head["models"][0]["p50_ratio"], 1920 / 6000, places=5)

    def test_every_number_records_its_source_setting_trials_machine_state_and_code(self):
        scenarios, matched, warm_chart, cold_chart, memory = self.data()["charts"]
        self.assertEqual([r["group"] for r in matched["models"]], ["model"])
        rows = {r["label"]: r for r in scenarios["rows"] + matched["rows"]}
        driver = rows["ColdIVF, through the driver"]
        self.assertEqual(driver["source"], "results/quiet2/driver/m1a/coldivf.jsonl")
        self.assertEqual(driver["code"], {"system_version": "coldivf 1.0", "wheel_sha256": {"coldivf": "w1"},
                                          "bin": "/frozen/bin", "harness_head": "h1", "harness_unstaged": False})
        self.assertNotIn("bin", rows["LanceDB IVF_RQ"]["code"])
        self.assertEqual(rows["ColdIVF, own harness, prewarmed core"]["code"], {"commit": "abc", "dirty": False})
        self.assertEqual(driver["recall_code"], {"commit": "abc", "dirty": False})

        for row in scenarios["rows"] + matched["rows"] + matched["models"] + cold_chart["points"] + memory["rows"]:
            self.assertTrue(row["source"].startswith("results/"), row)
            self.assertIn("machine_state", row)
            self.assertTrue(row["code"], row)

        for row in scenarios["rows"] + matched["rows"] + cold_chart["points"]:
            self.assertGreater(row["trials"], 0)
            self.assertIsInstance(row["setting"], dict)

        self.assertEqual([c["dataset"] for c in warm_chart["curves"]], [DATASET, DATASET])
        self.assertTrue(all(c["code"] for c in warm_chart["curves"]))
        point = next(p for p in cold_chart["points"] if p["system"] == "coldivf")
        self.assertEqual((point["recall10_warm"], point["trials"]), (0.95, 3))
        self.assertEqual(next(p for p in cold_chart["points"] if p["system"] == "lancedb")["label"],
                         "LanceDB IVF_RQ (10 threads)")
        row = next(r for r in memory["rows"] if r["system"] == "coldivf")
        self.assertEqual((row["disk"]["bytes_per_vector"], row["warm"]["process_bytes"]), (1.0, 5_000_000))

    def test_coldivf_warm_holds_the_quiet_runs_p50_and_qps_at_the_driver_setting(self):
        found = self.data()["charts"][1]["coldivf_warm"]
        self.assertEqual((found["setting"], found["p50_us"], found["recall10"]), ({"nprobe": 96, "rerank": 35}, 231.0,
                                                                                 0.9513))
        self.assertEqual([(q["threads"], q["qps"]) for q in found["qps"]], [(1, 4000.0), (6, 18000.0)])
        self.assertEqual((found["source"], found["machine_state"]["runner_cpu_idle_pct"]),
                         ("results/quiet2/b3-warm.jsonl", [89.76, 87.62]))

    def test_the_warm_chart_compares_each_point_with_coldivf_at_the_same_recall(self):
        curves = {c["system"]: c for c in self.data()["charts"][2]["curves"]}
        self.assertNotIn("against_coldivf", curves["coldivf"])
        ((p,),) = [curves["lancedb"]["against_coldivf"]["points"]]
        base = math.sqrt(150 * 250)
        self.assertAlmostEqual(p["coldivf_p50_us"], base, places=2)
        self.assertAlmostEqual(p["ratio"], 4000 / base, places=4)
        self.assertAlmostEqual(p["ratio_less_call"], 3900 / base, places=4)
        self.assertIsNone(charts.interpolate(curves["coldivf"]["points"], 0.96))
        slow = {"points": [{"setting": {}, "recall10": 0.9, "p50_us": 80.0, "overhead_us": 100.0},
                           {"setting": {}, "recall10": 0.92, "p50_us": 300.0, "overhead_us": 100.0}]}
        found = charts.against_coldivf(slow, curves["coldivf"])
        self.assertTrue(found["call_longer_than_a_search"])
        self.assertEqual([q["ratio_less_call"] for q in found["points"]], [None, None])

    def test_the_leaderboard_ranks_each_target_and_leaves_off_grids_that_reach_none(self):
        boards, left_out = charts.leaderboard(self.data()["charts"][2]["curves"])
        self.assertEqual([b["recall10"] for b in boards], list(charts.WARM_TARGETS))
        low, mid, high = boards
        self.assertEqual([(r["system"], round(r["p50_us"], 6), r["between"]) for r in low["rows"]],
                         [("coldivf", 150.0, True), ("lancedb", 4000.0, False)])
        self.assertEqual((low["rows"][1]["recall10"], low["rows"][1]["p50_less_call_us"]), (0.9, 3900.0))
        self.assertAlmostEqual(mid["rows"][0]["p50_us"], math.sqrt(150 * 250), places=4)
        self.assertEqual([r["system"] for r in high["rows"]], ["coldivf"])
        self.assertEqual(high["not_reached"], [{"system": "lancedb", "label": "LanceDB IVF_RQ", "best_recall10": 0.9}])
        self.assertEqual(left_out, [])
        capped = {"system": "x", "label": "X i8", "points": [{"setting": {}, "recall10": 0.84, "p50_us": 10.0}]}
        boards, left_out = charts.leaderboard([capped])
        self.assertEqual(left_out, [{"system": "x", "label": "X i8", "best_recall10": 0.84}])
        self.assertTrue(all(not b["rows"] and not b["not_reached"] for b in boards))

    def test_a_drawing_that_fails_writes_nothing(self):
        out = self.repo / "docs/charts"
        out.mkdir(parents=True)
        (out / "chart-data.json").write_text("old\n")
        args = ["--suite", str(self.suite), "--quiet", str(self.quiet), "--out", str(out), "--repo", str(self.repo)]

        with mock.patch.object(charts, "_mpl", side_effect=ImportError("no matplotlib")), \
                contextlib.redirect_stderr(io.StringIO()) as said:
            self.assertEqual(charts.main(args), 1)

        self.assertIn("wrote nothing", said.getvalue())
        self.assertEqual([p.name for p in out.iterdir()], ["chart-data.json"])
        self.assertEqual((out / "chart-data.json").read_text(), "old\n")

        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(charts.main(args + ["--data-only"]), 0)

        self.assertEqual([p.name for p in out.iterdir()], ["chart-data.json"])
        self.assertIsNone(json.loads((out / "chart-data.json").read_text())["drawn_with"])

    def test_the_drawing_settings_leave_out_the_hinting_factor_matplotlib_removes(self):
        self.assertNotIn("text.hinting_factor", charts._rc(charts.THEMES["light"]))

    @unittest.skipUnless(HAS_MATPLOTLIB, "matplotlib isn't installed")
    def test_main_draws_every_chart_in_both_themes(self):
        out = self.repo / "docs/charts"
        args = ["--suite", str(self.suite), "--quiet", str(self.quiet), "--out", str(out), "--repo", str(self.repo)]

        with contextlib.redirect_stdout(io.StringIO()) as printed, contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(charts.main(args), 0)

        self.assertIn("| LanceDB IVF_RQ | nprobes 18, refine_factor 4 | 0.951 | 3 | 22.0 ms |", printed.getvalue())
        names = sorted(p.name for p in out.iterdir())
        self.assertEqual(len(names), 11)
        self.assertIn("chart-data.json", names)
        self.assertIn("cold-head-to-head.svg", names)
        self.assertIn("cold-coldivf-dark.svg", names)
        self.assertIn("warm-leaderboard-dark.svg", names)
        drawn_with = json.loads((out / "chart-data.json").read_text())["drawn_with"]
        self.assertEqual(set(drawn_with), {"matplotlib"})
        light, dark = (out / "memory.svg").read_text(), (out / "memory-dark.svg").read_text()
        self.assertNotEqual(light, dark)
        self.assertNotIn("<dc:date>", light)


class HeadToHeadChartsTest(unittest.TestCase):
    """The 10M charts on the small phase of the head-to-head report's tests."""

    def setUp(self):
        from tests import test_headtohead

        self.phase = test_headtohead.CollectTest("test_markdown_and_json")
        self.phase.setUp()
        self.root = self.phase.root
        self.charts, self.machine = charts.h2h_charts(self.root / "suite.toml", self.root)
        self.by_id = {c["id"]: c for c in self.charts}

    def tearDown(self):
        self.phase.tearDown()

    def test_each_chart_is_there_and_final(self):
        self.assertEqual(list(self.by_id), ["cold-10m-95", "cold-10m-90", "warm-10m", "qps-10m", "memory-10m"])
        self.assertFalse(any(c["preliminary"] for c in self.charts))

    def test_a_cold_chart_ranks_each_systems_pick_against_coldivf(self):
        rows = self.by_id["cold-10m-95"]["rows"]
        self.assertEqual([(r["label"], r["group"]) for r in rows],
                         [("ColdIVF", "coldivf"), ("faiss IVF as shipped, 640 per list", "system")])
        ours, theirs = rows
        self.assertEqual((ours["p50_ratio"], ours["trials"], ours["queries"]), (1.0, 4, 4))
        self.assertAlmostEqual(theirs["p50_ratio"], 34_000.0 / 10_004.0)
        self.assertEqual(theirs["paired_ratio"]["queries"], 3)
        self.assertEqual((theirs["setting_text"], charts.fmt_setting(theirs)), ("nprobe 48, k_factor 4",) * 2)
        self.assertEqual(theirs["source"], "results/t/faiss-ivf.jsonl")
        self.assertIn("| faiss IVF as shipped, 640 per list | nprobe 48, k_factor 4 | 0.950 | 4 (4) | 34.0 ms |",
                      charts.h2h_table({"charts": self.charts}, "cold-10m-95"))

    def test_warm_panels_rank_by_p50_and_by_queries_per_second(self):
        p50 = self.by_id["warm-10m"]["boards"][1]
        self.assertEqual([(r["label"], r["p50_us"]) for r in p50["rows"]],
                         [("ColdIVF", 410.0), ("faiss IVF", 2100.0)])
        qps = self.by_id["qps-10m"]["boards"][1]
        self.assertEqual([(r["label"], r["qps"]) for r in qps["rows"]], [("ColdIVF", 5000.0)])
        self.assertEqual({r["label"] for r in self.by_id["warm-10m"]["boards"][2]["not_reached"]},
                         {"ColdIVF", "faiss IVF", "turbovec"})

    def test_memory_counts_the_process_and_the_index_pages(self):
        rows = {r["label"]: r for r in self.by_id["memory-10m"]["rows"]}
        self.assertEqual(set(rows), {"ColdIVF", "faiss IVF", "turbovec"})
        self.assertEqual(rows["ColdIVF"]["cold"]["total"], 48_000_000)
        self.assertEqual(rows["ColdIVF"]["warm"]["resident_file_bytes"], 9_000_000_000)
        self.assertEqual(rows["faiss IVF"]["cold"]["total"], 63_000_000)
        self.assertEqual(rows["turbovec"]["target_recall10"], None)

    def test_coldivf_shows_once_through_its_module_and_its_binaries_stay_in_the_report(self):
        from tests import test_headtohead as h

        binaries = [h.warm("coldivf", "", "tl", h.OURS, 0.955, 380.0),
                    h.warm("coldivf", "", "tl", h.OURS, 0.952, 390.0, split="test")]

        with open(self.root / "results" / "t" / "coldivf.jsonl", "a") as f:
            f.write("".join(json.dumps(x) + "\n" for x in binaries))

        made, _ = charts.h2h_charts(self.root / "suite.toml", self.root)
        rows = [r["label"] for r in next(c for c in made if c["id"] == "cold-10m-95")["rows"]]
        self.assertEqual(rows, ["ColdIVF", "faiss IVF as shipped, 640 per list"])
        board = next(c for c in made if c["id"] == "warm-10m")["boards"]
        self.assertNotIn("ColdIVF binaries", [r["label"] for b in board for r in b["rows"] + b["not_reached"]])
        report = h.headtohead.collect(h.driver.load_suite(self.root / "suite.toml"),
                                      h.headtohead.Results(self.root / "results" / "t"))
        self.assertIn("ColdIVF, binaries", [r["label"] for r in report["targets"][0.95]["rows"]])

    def test_one_row_per_system_keeps_the_fastest_and_coldivfs_module(self):
        rows = [{"system": "coldivf", "variant": "", "p50_us": 9.0},
                {"system": "faiss-ivf", "variant": "a", "p50_us": 30.0},
                {"system": "coldivf", "variant": "python", "p50_us": 10.0},
                {"system": "faiss-ivf", "variant": "b", "p50_us": 20.0},
                {"system": "lancedb", "variant": "", "p50_us": 25.0}]
        kept = charts.one_per_system(rows, lambda r: r["p50_us"])
        self.assertEqual([(r["system"], r["variant"]) for r in kept],
                         [("coldivf", "python"), ("faiss-ivf", "b"), ("lancedb", "")])
        # Without a row of the module, ColdIVF's binaries keep theirs.
        self.assertEqual(charts.one_per_system(rows[:2], lambda r: r["p50_us"]), rows[:2])

    def test_chart_data_puts_the_10m_charts_first(self):
        smoke = self.root / "smoke.toml"
        smoke.write_text(f'phase = "smoke"\ndataset = "{DATASET}"\nn = {N}\n')
        data = charts.chart_data(smoke, self.root / "no-quiet-run", self.root, self.root / "suite.toml")
        self.assertEqual([c["id"] for c in data["charts"][:5]],
                         ["cold-10m-95", "cold-10m-90", "warm-10m", "qps-10m", "memory-10m"])
        self.assertEqual(data["h2h_suite"], "suite.toml")


if __name__ == "__main__":
    unittest.main()
