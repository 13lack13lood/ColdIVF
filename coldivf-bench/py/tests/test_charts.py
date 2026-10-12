"""charts.py on a small tree of result files: its picks, the numbers it records, and its drawing where matplotlib is
installed."""

import contextlib
import html
import importlib.util
import io
import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import charts
import schema

HAS_MATPLOTLIB = importlib.util.find_spec("matplotlib") is not None
DATASET, N = "tiny-768-cosine", 1000
MACHINE = {"model": "Mac15,7", "brand": "Apple M3 Pro", "memsize": 1, "perflevels": [], "os": {}, "ssd": {}}


def driver_line(kind: str, system: str, batch: str, ts: str, **fields) -> dict:
    """A line as the driver writes it, of a batch at time `ts`."""
    line = schema.blank(kind)
    line.update(run_id="r", run_key="k", ts=ts, batch_id=batch, phase="m1a", system=system, dataset=DATASET, n=N,
                machine=MACHINE, load_before=[3.0, 2.0, 1.0], system_version=f"{system} 1.0",
                provenance={"head": "h1", "unstaged": False, "wheel_sha256": {system: "w1"},
                            "env": {"COLDIVF_BIN": "/frozen/bin"}},
                gate={"mode": "warn", "passed": False, "solo": False, "cpu_idle_pct": 50.0, "problems": []})
    line.update(fields)
    return line


def warm(system, batch, ts, setting, p50, recall, **fields):
    defaults = {"mode": "whole", "threads": 1, "queries": 500, "passes": 2, "p50_us": p50, "recall10": recall,
                "search_params": setting}
    return driver_line("warm", system, batch, ts, **(defaults | fields))


def cold(system, batch, ts, setting, total, **fields):
    defaults = {"regime": "c0", "trial": 0, "threads": 1, "cpu_state": "prewarm", "total_us": total, "recall10": 0.9,
                "search_params": setting}
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


def svg_texts(path: Path) -> list[str]:
    """The strings a chart's SVG draws."""
    return [html.unescape(s) for s in re.findall(r"<text[^>]*>([^<]*)</text>", path.read_text())]


def assert_value_columns(test: unittest.TestCase, path: Path) -> None:
    """A dot-and-line chart heads its two columns of times once, with "p50" and "p99", and writes neither in a row."""
    texts = svg_texts(path)
    test.assertEqual((texts.count("p50"), texts.count("p99")), (1, 1), path.name)
    test.assertFalse([s for s in texts if s.startswith(("p50 ", "p99 "))], path.name)


class ChartsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self.tmp.name)
        quiet = self.repo / "results/quiet2"
        write(quiet / "driver/m1a/coldivf.jsonl", [
            cold("coldivf", "q", "2026-10-10T07:00:00Z", {"nprobe": 96, "rerank": 35}, t, config_id="faiss-chained",
                 split="test") for t in (6000.0, 6500.0)])
        write(quiet / "a3-m1a.jsonl", [own("idle", total=7000.0), own("prewarm"), own("prewarm", total=6400.0),
                                       own("prewarm", seq=1, total=1.0), own("prewarm", cached=5, total=2.0),
                                       own("prewarm", nprobe=32, rerank=39, total=5000.0)])
        write(quiet / "a5-mapped.jsonl", [own("prewarm", mode="mapped", nprobe=128, rerank=29, total=25000.0),
                                          own("prewarm", mode="wave", nprobe=128, rerank=29, total=7000.0)])
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
        return charts.chart_data(self.quiet, self.repo)

    def test_tidy_cuts_float_noise_and_keeps_counts_to_the_unit(self):
        self.assertEqual(charts.tidy({"a": [0.9005999999999982, 11543.452, 3]}), {"a": [0.9006, 11543.45, 3]})
        self.assertEqual(f"{charts.tidy(11543.452):,.0f}", "11,543")

    def test_percentile_is_nearest_rank(self):
        values = [float(v) for v in range(1, 11)]
        self.assertEqual(charts.percentile(values, 50), 5.0)
        self.assertEqual(charts.percentile(values, 95), 10.0)
        self.assertEqual(charts.percentile(values, 99), 10.0)
        self.assertEqual(charts.percentile(values[:1], 99), 1.0)
        self.assertIsNone(charts.percentile([], 50))

    def test_the_matched_cell_reaches_the_target_with_the_most_trials_and_is_the_cheapest_warm(self):
        lines, warm_lines = [], []

        # 18 is the main cell. 20 is cheaper warm but has fewer trials, 12 is faster cold but doesn't reach 0.95, and
        # 24 reaches it with as many trials but costs more warm.
        for nprobes, totals, p50, recall in ((18, (22000.0, 21000.0, 23000.0), 3000.0, 0.951),
                                             (20, (20000.0, 20500.0), 2900.0, 0.956),
                                             (12, (15000.0, 15500.0, 16000.0), 2000.0, 0.93),
                                             (24, (24000.0, 25000.0, 26000.0), 3500.0, 0.97)):
            setting = {"nprobes": nprobes, "refine_factor": 4}
            lines += [cold("x", "q", "t", setting, t, split="test") for t in totals]
            warm_lines.append(warm("x", "w", "t", dict(reversed(setting.items())), p50, recall, split="all"))

        def warm_of(line, splits):
            return charts.warm_at(warm_lines, "x", None, line["search_params"], splits)

        ((cell, picked, shown),) = charts.matched_cells(lines, warm_of)
        self.assertEqual(cell[0]["search_params"], {"nprobes": 18, "refine_factor": 4})
        self.assertEqual(sorted(line["total_us"] for line in cell), [21000.0, 22000.0, 23000.0])
        self.assertEqual((picked["recall10"], shown["recall10"]), (0.951, 0.951))

    def test_a_matched_cell_is_picked_on_the_tune_split_and_shows_the_split_its_trials_searched(self):
        lines = [cold("x", "q", "t", {"k": 1}, 10.0, split="test"), cold("x", "q", "t", {"k": 2}, 5.0, split="test")]
        warm_lines = [warm("x", "w", "t", {"k": 1}, 100.0, 0.96, split="tune"),
                      warm("x", "w", "t", {"k": 1}, 100.0, 0.948, split="test"),
                      warm("x", "w", "t", {"k": 2}, 90.0, 0.94, split="tune")]

        def warm_of(line, splits):
            return charts.warm_at(warm_lines, "x", None, line["search_params"], splits)

        ((cell, picked, shown),) = charts.matched_cells(lines, warm_of)
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
        (chart,) = data["charts"]
        self.assertEqual(chart["id"], "cold-coldivf")
        self.assertTrue(all(r["system"] == "coldivf" for r in chart["rows"]))
        rows = {r["label"]: r for r in chart["rows"]}
        driver = rows["ColdIVF, through the driver"]
        self.assertEqual((driver["p50_us"], driver["recall10_warm"], driver["p50_ratio"]), (6000.0, 0.9513, 1.0))
        idle = rows["ColdIVF, own harness, idle core"]
        self.assertEqual(idle["against"]["label"], "ColdIVF, through the driver")
        self.assertAlmostEqual(idle["p50_ratio"], 7000 / 6000, places=5)
        lazy = rows["ColdIVF's file, lazily mapped, prewarmed core (control)"]
        self.assertEqual((lazy["against"]["label"], lazy["against"]["p50_us"]),
                         ("ColdIVF, own harness, read waves, prewarmed core", 7000.0))
        self.assertAlmostEqual(lazy["p50_ratio"], 25000 / 7000, places=5)
        # The setting whose warm recall is 0.90 has no row.
        self.assertNotIn({"nprobe": 32, "rerank": 39}, [r["setting"] for r in chart["rows"]])

        for r in chart["rows"]:
            self.assertGreaterEqual(r["recall10_warm"], charts.TARGET, r["label"])

        table = charts.headline_table(data, "cold-coldivf")
        self.assertIn("| ColdIVF, through the driver | nprobe 96, rerank 35 | 0.951 | 2 | 6.00 ms | 6.50 ms | 6.50 ms "
                      "| 1.00x |", table)
        self.assertIn("| 25.0 ms | 25.0 ms | 25.0 ms | 3.57x |", table)

    def test_every_number_records_its_source_setting_trials_machine_state_and_code(self):
        (chart,) = self.data()["charts"]
        rows = {r["label"]: r for r in chart["rows"]}
        driver = rows["ColdIVF, through the driver"]
        self.assertEqual(driver["source"], "results/quiet2/driver/m1a/coldivf.jsonl")
        self.assertEqual(driver["code"], {"system_version": "coldivf 1.0", "wheel_sha256": {"coldivf": "w1"},
                                          "bin": "/frozen/bin", "harness_head": "h1", "harness_unstaged": False})
        self.assertEqual((driver["recall_source"], driver["recall_code"]),
                         ("results/quiet2/b3-warm.jsonl", {"commit": "abc", "dirty": False}))
        self.assertEqual(driver["slowest_us"], [6000.0, 6500.0])
        own_row = rows["ColdIVF, own harness, prewarmed core"]
        self.assertEqual(own_row["code"], {"commit": "abc", "dirty": False})
        self.assertEqual(own_row["machine_state"]["runner_cpu_idle_pct"], [85.16, 91.21])

        for row in chart["rows"]:
            self.assertTrue(row["source"].startswith("results/"), row)
            self.assertIn("machine_state", row)
            self.assertTrue(row["code"], row)
            self.assertGreater(row["trials"], 0)
            self.assertIsInstance(row["setting"], dict)

    def test_a_drawing_that_fails_writes_nothing(self):
        out = self.repo / "docs/charts"
        out.mkdir(parents=True)
        (out / "chart-data.json").write_text("old\n")
        args = ["--quiet", str(self.quiet), "--out", str(out), "--repo", str(self.repo)]

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
        self.assertNotIn("text.hinting_factor", charts._rc(charts.THEME))

    @unittest.skipUnless(HAS_MATPLOTLIB, "matplotlib isn't installed")
    def test_main_draws_the_chart_of_coldivf_in_each_setup(self):
        out = self.repo / "docs/charts"
        args = ["--quiet", str(self.quiet), "--out", str(out), "--repo", str(self.repo)]

        with contextlib.redirect_stdout(io.StringIO()) as printed, contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(charts.main(args), 0)

        self.assertIn("| ColdIVF, through the driver | nprobe 96, rerank 35 | 0.951 | 2 | 6.00 ms |",
                      printed.getvalue())
        self.assertEqual(sorted(p.name for p in out.iterdir()), ["chart-data.json", "cold-coldivf-dark.svg"])
        drawn_with = json.loads((out / "chart-data.json").read_text())["drawn_with"]
        self.assertEqual(set(drawn_with), {"matplotlib"})
        self.assertNotIn("<dc:date>", (out / "cold-coldivf-dark.svg").read_text())
        assert_value_columns(self, out / "cold-coldivf-dark.svg")
        texts = svg_texts(out / "cold-coldivf-dark.svg")
        self.assertIn("nprobe 96, rerank 35 · recall 0.951 · 2 trials", texts)
        self.assertIn("6.50 ms", texts)


# A same-batch check of ColdIVF's module, its binaries and faiss with batched reads, on the head-to-head's indexes.
CHECK = """
phase = "check"
dataset = "tiny-16-cosine"
n = 1000
index_root = "{root}/indexes"
results_root = "{root}/results"

[[system]]
name = "coldivf"
adapter = "coldivf"

  [[system.build]]
  config_id = "tl"
  index = "{root}/indexes/coldivf/tl-n1000-s1"

  [[system.cold]]
  config_id = "tl"
  params = {{ nprobe = 64, rerank = 39 }}
  split = "test"
  trials = 2

[[system]]
name = "coldivf"
adapter = "coldivf_py"
variant = "python"

  [[system.build]]
  config_id = "tl"
  index = "{root}/indexes/coldivf/tl-n1000-s1"

  [[system.cold]]
  config_id = "tl"
  params = {{ nprobe = 64, rerank = 39 }}
  split = "test"
  trials = 3

[[system]]
name = "faiss-ivf"
adapter = "faiss_ivf"
variant = "batched"

  [[system.build]]
  config_id = "vpl"
  index = "{root}/indexes/faiss/vpl"

  [[system.cold]]
  config_id = "vpl"
  params = {{ nprobe = 48, k_factor = 4, qb = 4 }}
  split = "test"
  trials = 3
"""


# A second faiss setup, with batched reads on faiss's own partition, for a test where faiss's two setups rank one way
# cold and the other way warm.
BATCHED = """
[[system]]
name = "faiss-ivf"
adapter = "faiss_ivf"
variant = "batched"

  [[system.build]]
  config_id = "vpl2560-rq1"
  config = { bits = 1 }

  [[system.cold]]
  config_id = "vpl2560-rq1"
  params = { nprobe = 24, k_factor = 4, qb = 4 }
  split = "test"
  trials = 4
  group = "faiss"
"""


class HeadToHeadChartsTest(unittest.TestCase):
    """The 10M charts on the small phase of the head-to-head report's tests."""

    def setUp(self):
        from tests import test_headtohead

        self.phase = test_headtohead.CollectTest("test_markdown_and_json")
        self.phase.setUp()
        self.root = self.phase.root

        # ColdIVF's queries per second on 6 threads at its 0.90 setting too, so each target ColdIVF reaches has a QPS
        # chart.
        six = test_headtohead.warm("coldivf", "python", "tl", test_headtohead.BELOW, None, None, split="test",
                                   threads=6, qps=6000.0, batch_api=False, processes=1)

        with open(self.root / "results" / "t" / "coldivf.jsonl", "a") as f:
            f.write(json.dumps(six) + "\n")

        # Each warm line at 1 thread gets the p99 and the counts the driver records, a p99 of 1.5 times its p50 over
        # 500 queries in 3 passes.
        for path in (self.root / "results" / "t").glob("*.jsonl"):
            lines = [json.loads(x) for x in path.read_text().splitlines()]
            spread = [x | {"p99_us": 1.5 * x["p50_us"], "queries": 500, "passes": 3}
                      if x["kind"] == "warm" and x["threads"] == 1 else x for x in lines]
            path.write_text("".join(json.dumps(x) + "\n" for x in spread))

        self.charts, self.machine = charts.h2h_charts(self.root / "suite.toml", self.root)
        self.by_id = {c["id"]: c for c in self.charts}

    def tearDown(self):
        self.phase.tearDown()

    def test_each_chart_is_there_and_final(self):
        self.assertEqual(list(self.by_id), ["cold-10m-95", "cold-10m-90", "warm-10m-95", "warm-10m-90", "qps-10m-95",
                                            "qps-10m-90", "memory-10m", "build-10m"])
        self.assertFalse(any(c["preliminary"] for c in self.charts))
        # The README shows memory and build as two images of the same rows.
        memory, build = self.by_id["memory-10m"], self.by_id["build-10m"]
        self.assertEqual((memory["panels"], build["panels"]), ("memory", "build"))
        self.assertIs(memory["rows"], build["rows"])

    def test_a_cold_chart_ranks_each_systems_pick_against_coldivf(self):
        rows = self.by_id["cold-10m-95"]["rows"]
        self.assertEqual([(r["label"], r["group"]) for r in rows],
                         [("ColdIVF", "coldivf"), ("faiss IVF as shipped, 640 per list", "system")])
        ours, theirs = rows
        self.assertEqual((ours["p50_ratio"], ours["trials"], ours["queries"]), (1.0, 4, 4))
        self.assertAlmostEqual(theirs["p50_ratio"], 34_000.0 / 10_004.0)
        self.assertEqual(theirs["paired_ratio"]["queries"], 3)
        # Its setting names every search parameter, the qb its whole grid shares too, as the warm and QPS charts do.
        self.assertEqual((theirs["setting_text"], charts.fmt_setting(theirs)), ("nprobe 48, k_factor 4, qb 4",) * 2)
        self.assertEqual(theirs["source"], "results/t/faiss-ivf.jsonl")
        self.assertIn("| faiss IVF as shipped, 640 per list | nprobe 48, k_factor 4, qb 4 | 0.950 | 4 (4) | 34.0 ms |",
                      charts.h2h_table({"charts": self.charts}, "cold-10m-95"))

    def test_a_warm_chart_shows_each_systems_pick_as_the_cold_charts_do(self):
        from tests import test_headtohead as h

        chart = self.by_id["warm-10m-95"]
        self.assertEqual((chart["kind"], chart["target_recall10"], chart["title"]),
                         ("h2h-warm", 0.95, "Warm query at 0.001M vectors, recall@10 0.95"))
        self.assertEqual([(r["label"], r["group"], r["p50_us"], r["p99_us"]) for r in chart["rows"]],
                         [("ColdIVF", "coldivf", 410.0, 615.0),
                          ("faiss IVF as shipped, 640 per list", "system", 2100.0, 3150.0)])
        ours, theirs = chart["rows"]
        self.assertEqual((ours["queries"], ours["passes"], ours["threads"], ours["recall10_warm_split"]),
                         (500, 3, 1, "test"))
        self.assertAlmostEqual(theirs["p50_ratio"], 2100.0 / 410.0)
        self.assertEqual((theirs["source"], theirs["short"]), ("results/t/faiss-ivf.jsonl", "faiss IVF"))
        self.assertTrue({"machine_state", "code"} <= set(theirs))
        self.assertEqual((ours["invalid"], theirs["invalid"]), ([], []))
        # Its setting names every search parameter, as the cold chart's row does.
        self.assertEqual(charts._row_detail(theirs), "nprobe 48, k_factor 4, qb 4 · recall 0.950 · 500 queries")
        self.assertEqual(self.by_id["cold-10m-95"]["rows"][1]["setting_text"], theirs["setting_text"])
        self.assertIn("Each dot is the p50 of the 500 test queries at 1 thread", chart["subtitle"])
        self.assertIn("Both are the median of 3 passes over the queries.", chart["subtitle"])
        # A system that reaches no target has no row, as in the cold charts.
        self.assertEqual(chart["not_reached"], [{"system": "turbovec", "label": "turbovec"}])
        self.assertEqual([(r["setting"], r["p50_us"], r["p99_us"]) for r in self.by_id["warm-10m-90"]["rows"]],
                         [(h.BELOW, 310.0, 465.0), (h.FAISS, 2100.0, 3150.0)])

    def test_a_warm_chart_ranks_its_rows_by_warm_p50(self):
        # faiss is slower than ColdIVF cold but faster warm, so the warm chart's order differs from the cold one's.
        path = self.root / "results" / "t" / "faiss-ivf.jsonl"
        lines = [json.loads(x) for x in path.read_text().splitlines()]
        fast = [x | {"p50_us": 300.0, "p99_us": 450.0}
                if x["kind"] == "warm" and x["split"] == "test" and x["threads"] == 1 else x for x in lines]
        path.write_text("".join(json.dumps(x) + "\n" for x in fast))
        made = {c["id"]: c for c in charts.h2h_charts(self.root / "suite.toml", self.root)[0]}
        self.assertEqual([r["system"] for r in made["cold-10m-95"]["rows"]], ["coldivf", "faiss-ivf"])
        self.assertEqual([r["system"] for r in made["warm-10m-95"]["rows"]], ["faiss-ivf", "coldivf"])

    def test_a_warm_chart_keeps_each_systems_setup_of_lowest_warm_p50(self):
        from tests import test_headtohead as h

        # faiss with batched reads is faster than faiss as shipped cold and slower warm, so the cold charts show it and
        # the warm charts show faiss as shipped.
        with open(self.root / "suite.toml", "a") as f:
            f.write(BATCHED)

        batched = {"nprobe": 24, "k_factor": 4, "qb": 4}
        lines = [h.warm("faiss-ivf", "batched", "vpl2560-rq1", batched, 0.953, 2600.0),
                 h.warm("faiss-ivf", "batched", "vpl2560-rq1", batched, 0.952, 2700.0, split="test", p99_us=4000.0,
                        queries=500, passes=3)]
        lines += [h.cold("faiss-ivf", "batched", "vpl2560-rq1", batched, q, 20_000.0) for q in (1, 3, 5, 11)]

        with open(self.root / "results" / "t" / "faiss-ivf.jsonl", "a") as f:
            f.write("".join(json.dumps(x) + "\n" for x in lines))

        made = {c["id"]: c for c in charts.h2h_charts(self.root / "suite.toml", self.root)[0]}

        for t in ("95", "90"):
            self.assertEqual([(r["system"], r["variant"]) for r in made[f"cold-10m-{t}"]["rows"]],
                             [("coldivf", "python"), ("faiss-ivf", "batched")])
            self.assertEqual([(r["system"], r["variant"]) for r in made[f"warm-10m-{t}"]["rows"]],
                             [("coldivf", "python"), ("faiss-ivf", "shipped")])

    def test_every_setting_names_each_search_parameter(self):
        self.assertEqual(charts.every_setting({"nprobe": 48, "k_factor": 4, "qb": 4, "quantizer_ef": 128}),
                         "nprobe 48, k_factor 4, qb 4, quantizer_ef 128")
        self.assertEqual(charts.every_setting({"nprobes": 16, "refine_factor": 8}), "nprobes 16, refine 8")
        self.assertEqual(charts.every_setting({}), "none")

    def test_a_qps_chart_ranks_each_systems_pick_by_queries_per_second_on_6_threads(self):
        chart = self.by_id["qps-10m-95"]
        self.assertEqual((chart["kind"], chart["target_recall10"], chart["title"], chart["legend"]),
                         ("h2h-qps", 0.95, "Queries per second at 0.001M vectors, recall@10 0.95",
                          ["ColdIVF", "Other systems"]))
        # The highest stands first, with its numbers on 6 and on 12 threads. ColdIVF has no line on 12 threads.
        self.assertEqual([(r["label"], r["group"], r["qps"], r["qps_12"]) for r in chart["rows"]],
                         [("ColdIVF", "coldivf", 5000.0, None),
                          ("faiss IVF as shipped, 640 per list", "system", 2000.0, 2500.0)])
        ours, theirs = chart["rows"]
        self.assertEqual((theirs["short"], theirs["threads"], theirs["recall10_warm"], theirs["recall10_warm_split"]),
                         ("faiss IVF", 6, 0.95, "test"))
        self.assertEqual([(r["method"], r["method_12"]) for r in chart["rows"]],
                         [("threads", None), ("batch call", "processes")])
        self.assertEqual((ours["invalid"], theirs["invalid"]), ([], []))
        self.assertEqual(theirs["source"], "results/t/faiss-ivf.jsonl")
        self.assertTrue({"machine_state", "code"} <= set(theirs))
        # Under each name, every search parameter, the recall and how its 6-thread queries ran side by side. The footer
        # says that faiss ran its 12 threads another way.
        self.assertEqual([charts._row_detail(r) for r in chart["rows"]],
                         ["nprobe 64, rerank 39 · recall 0.952 · threads",
                          "nprobe 48, k_factor 4, qb 4 · recall 0.950 · batch call"])
        self.assertEqual([charts.twelve_threads(r) for r in chart["rows"]], ["-", "2,500"])
        self.assertIn("so a dot has no line to a p99.", chart["subtitle"])
        # A system that reaches no target has no row, as in the cold and warm charts.
        self.assertEqual(chart["not_reached"], [{"system": "turbovec", "label": "turbovec"}])
        # The chart at 0.90 shows each system's pick there.
        self.assertEqual([(r["label"], r["qps"], r["setting"]) for r in self.by_id["qps-10m-90"]["rows"]],
                         [("ColdIVF", 6000.0, {"nprobe": 32, "rerank": 39}),
                          ("faiss IVF as shipped, 640 per list", 2000.0, {"nprobe": 48, "k_factor": 4, "qb": 4})])
        # No chart for recall@10 0.99, which ColdIVF doesn't reach, since a chart without it compares nothing.
        qps = [c for c in self.charts if c["kind"] == "h2h-qps"]
        self.assertEqual([c["id"] for c in qps], ["qps-10m-95", "qps-10m-90"])
        self.assertTrue(all(any(r["system"] == "coldivf" for r in c["rows"]) for c in qps))

    def test_a_qps_row_lists_an_invalid_12_thread_line(self):
        # faiss's 12-thread line is its only one, and swap-ins or compressions during its runs made it invalid.
        path = self.root / "results" / "t" / "faiss-ivf.jsonl"
        lines = [json.loads(x) for x in path.read_text().splitlines()]
        flagged = [x | {"status": "invalid"} if x["kind"] == "warm" and x["threads"] == 12 else x for x in lines]
        path.write_text("".join(json.dumps(x) + "\n" for x in flagged))
        made = {c["id"]: c for c in charts.h2h_charts(self.root / "suite.toml", self.root)[0]}
        theirs = made["qps-10m-95"]["rows"][1]
        self.assertEqual((theirs["qps_12"], theirs["invalid"]), (2500.0, [12]))
        self.assertEqual(charts.twelve_threads(theirs), "2,500 (invalid)")
        self.assertEqual(charts.twelve_threads({"qps_12": 2500.0, "invalid": [6]}), "2,500")

    def test_each_qps_number_keeps_the_machine_state_code_and_mode_of_its_own_line(self):
        # faiss's 6-thread line ran in another batch and harness than its 12-thread line and its other lines, and kept
        # only its PQ codes in memory, while its line at 1 thread read the whole index.
        path = self.root / "results" / "t" / "faiss-ivf.jsonl"
        lines = [json.loads(x) for x in path.read_text().splitlines()]
        own = {6: {"batch_id": "b6", "provenance": {"head": "h6"}, "mode": "pq"},
               12: {"batch_id": "b12", "provenance": {"head": "h12"}}}
        path.write_text("".join(json.dumps(x | {"batch_id": "b1"} | own.get(x.get("threads"), {})) + "\n"
                                for x in lines))
        made = {c["id"]: c for c in charts.h2h_charts(self.root / "suite.toml", self.root)[0]}
        ours, theirs = made["qps-10m-95"]["rows"]
        self.assertEqual((theirs["machine_state"]["batches"], theirs["machine_state_12"]["batches"]), (["b6"], ["b12"]))
        self.assertEqual((theirs["code"], theirs["code_12"]), ({"harness_head": "h6"}, {"harness_head": "h12"}))
        # ColdIVF has no 12-thread line.
        self.assertEqual((ours["machine_state_12"], ours["code_12"]), (None, None))
        self.assertEqual((ours["mode"], theirs["mode"]), ("whole", "pq"))
        self.assertIn("p99. faiss IVF as shipped, 640 per list keeps only its PQ codes in memory",
                      made["qps-10m-95"]["subtitle"])
        self.assertNotIn("PQ codes", made["warm-10m-95"]["subtitle"])

    def test_a_target_gets_a_qps_chart_only_with_coldivfs_number_on_6_threads(self):
        def drop(gone) -> dict:
            for path in (self.root / "results" / "t").glob("*.jsonl"):
                lines = [json.loads(x) for x in path.read_text().splitlines()]
                path.write_text("".join(json.dumps(x) + "\n" for x in lines if not gone(x)))

            return charts.chart_data(self.root / "no-quiet-run", self.root, {"10m": self.root / "suite.toml"}, {})

        # Without ColdIVF's 6-thread line at 0.90, faiss's number there gets no chart, since it compares nothing with
        # ColdIVF.
        below = {"nprobe": 32, "rerank": 39}
        data = drop(lambda x: x["system"] == "coldivf" and x.get("threads") == 6 and x.get("search_params") == below)
        self.assertEqual([c["id"] for c in data["charts"] if c.get("kind") == "h2h-qps"], ["qps-10m-95"])
        # Without any 6-thread line, eg. before a suite's runs on several threads, the other charts and the tables are
        # still made.
        data = drop(lambda x: x.get("threads") == 6)
        self.assertEqual([c["id"] for c in data["charts"]],
                         ["cold-10m-95", "cold-10m-90", "warm-10m-95", "warm-10m-90", "memory-10m", "build-10m",
                          "cold-coldivf"])
        self.assertIn("| **ColdIVF** | **0.310 ms** | - | **0.410 ms** | - | - |", charts.h2h_warm_table(data, "10m"))

    def test_memory_counts_the_process_and_the_index_pages(self):
        rows = {r["label"]: r for r in self.by_id["memory-10m"]["rows"]}
        self.assertEqual(set(rows), {"ColdIVF", "faiss IVF", "turbovec"})
        self.assertEqual(rows["ColdIVF"]["cold"]["total"], 48_000_000)
        self.assertEqual(rows["ColdIVF"]["warm"]["resident_file_bytes"], 9_000_000_000)
        self.assertEqual(rows["faiss IVF"]["cold"]["total"], 63_000_000)
        self.assertEqual(rows["turbovec"]["target_recall10"], None)

    def test_coldivf_shows_once_through_its_module_and_its_binaries_stay_in_the_report(self):
        from tests import test_headtohead as h

        # The binaries serve more queries per second than the module, and the QPS chart still shows the module.
        binaries = [h.warm("coldivf", "", "tl", h.OURS, 0.955, 380.0),
                    h.warm("coldivf", "", "tl", h.OURS, 0.952, 390.0, split="test"),
                    h.warm("coldivf", "", "tl", h.OURS, None, None, split="test", threads=6, qps=9000.0,
                           batch_api=False, processes=1)]

        with open(self.root / "results" / "t" / "coldivf.jsonl", "a") as f:
            f.write("".join(json.dumps(x) + "\n" for x in binaries))

        made, _ = charts.h2h_charts(self.root / "suite.toml", self.root)
        rows = [r["label"] for r in next(c for c in made if c["id"] == "cold-10m-95")["rows"]]
        self.assertEqual(rows, ["ColdIVF", "faiss IVF as shipped, 640 per list"])
        warm = [r["label"] for c in made if c["kind"] == "h2h-warm" for r in c["rows"] + c["not_reached"]]
        self.assertEqual(warm, ["ColdIVF", "faiss IVF as shipped, 640 per list", "turbovec"] * 2)
        qps = next(c for c in made if c["id"] == "qps-10m-95")["rows"]
        self.assertEqual([(r["system"], r["variant"], r["qps"]) for r in qps],
                         [("coldivf", "python", 5000.0), ("faiss-ivf", "shipped", 2000.0)])
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

    def test_chart_data_puts_the_head_to_head_charts_first_and_takes_their_machine(self):
        path = self.root / "results" / "t" / "turbovec.jsonl"
        path.write_text("".join(json.dumps(json.loads(x) | {"machine": MACHINE}) + "\n"
                                for x in path.read_text().splitlines()))
        data = charts.chart_data(self.root / "no-quiet-run", self.root, {"10m": self.root / "suite.toml"}, {})
        self.assertEqual([c["id"] for c in data["charts"]],
                         ["cold-10m-95", "cold-10m-90", "warm-10m-95", "warm-10m-90", "qps-10m-95", "qps-10m-90",
                          "memory-10m", "build-10m", "cold-coldivf"])
        self.assertEqual(data["h2h_suites"], {"10m": "suite.toml"})
        self.assertEqual((data["machine"]["model"], data["machine"]["chip"]), ("Mac15,7", "Apple M3 Pro"))

    def test_each_scale_names_its_charts_and_says_which_rows_it_takes(self):
        with mock.patch.dict(charts.DATASET_ROWS, {"tiny-16-cosine": 5000}):
            made, _ = charts.h2h_charts(self.root / "suite.toml", self.root, "1m")

        self.assertEqual([(c["id"], c["kind"]) for c in made],
                         [("cold-1m-95", "h2h-cold"), ("cold-1m-90", "h2h-cold"), ("warm-1m-95", "h2h-warm"),
                          ("warm-1m-90", "h2h-warm"), ("qps-1m-95", "h2h-qps"), ("qps-1m-90", "h2h-qps"),
                          ("memory-1m", "h2h-memory"), ("build-1m", "h2h-memory")])
        self.assertTrue(all(c["subtitle"].startswith("The first 0.001M rows of tiny-16-cosine") for c in made))
        self.assertTrue(self.by_id["cold-10m-95"]["subtitle"].startswith("0.001M vectors of tiny-16-cosine"))
        # Every kind has a drawing.
        self.assertTrue(all(c["kind"] in charts.DRAW for c in made))

    def test_the_qps_chart_names_how_each_system_ran_its_queries_side_by_side(self):
        footer = self.by_id["qps-10m-95"]["footer"]
        self.assertIn("faiss IVF searches through a call for a batch of queries", footer)
        self.assertIn("ColdIVF searches from 6 threads in one process.", footer)
        self.assertIn("ColdIVF's module has no batch call.", footer)
        # faiss ran its 12 threads as processes, and ColdIVF has no 12-thread line.
        self.assertTrue(footer.endswith(" On 12 threads, faiss IVF searches from 12 processes."), footer)
        self.assertEqual({r["method"] for c in self.charts if c["kind"] == "h2h-qps" for r in c["rows"]},
                         {"threads", "batch call"})
        self.assertEqual(charts.method_note([{"short": "a", "method": "processes", "method_12": "processes"},
                                             {"short": "b", "method": "processes"},
                                             {"short": "c", "method": "threads", "method_12": "threads"}]),
                         "a and b search from 6 processes, and c from 6 threads in one process.")

    def test_the_memory_chart_says_how_each_build_ran_and_which_rows_reach_no_target(self):
        chart = self.by_id["memory-10m"]
        self.assertIn("ColdIVF's build ran with macOS malloc's cache of freed large blocks off, and the other builds "
                      "with it on", chart["footer"])
        self.assertIn(", and turbovec at its setting of most recall, since it reaches neither.", chart["subtitle"])
        rows = {r["label"]: r for r in chart["rows"]}
        self.assertEqual((rows["ColdIVF"]["build"]["malloc_cache"], rows["faiss IVF"]["build"]["malloc_cache"]),
                         ("off", "on"))
        self.assertEqual(rows["ColdIVF"]["build"]["cluster_seconds"], 540.0)
        both_off = [{"label": x, "build": {"peak_footprint": 1, "malloc_cache": "off"}} for x in ("a", "b")]
        self.assertEqual(charts.malloc_note(both_off), "Every build ran with macOS malloc's cache of freed large "
                                                       "blocks off, so no build peak holds freed memory.")

    def test_each_partial_warm_mode_says_what_stays_on_disk(self):
        self.assertEqual(charts.warm_reads_note({"index": ["LanceDB IVF_RQ"], "pq": ["DiskANN"]}),
                         "LanceDB IVF_RQ caches only its index files, so its rows are read from its table. DiskANN "
                         "keeps only its PQ codes in memory and reads every sector from the disk, since its index "
                         "doesn't fit in memory.")
        self.assertEqual(charts.warm_reads_note({"pq": ["a", "b"]}),
                         "a and b keep only their PQ codes in memory and read every sector from the disk, since their "
                         "index doesn't fit in memory.")

        def row(label: str, mode: str, stored: int, index: int) -> dict:
            return {"full_label": label, "warm": {"mode": mode}, "build": {"stored_vector_bytes": stored},
                    "disk": {"index_bytes": index}}

        rows = [row("ColdIVF", "whole", 9_000_000_000, 9_430_000_000),
                row("LanceDB IVF_RQ", "index", 30_720_000_000, 31_900_000_000),
                row("DiskANN", "pq", 31_680_000_000, 41_930_000_000)]
        self.assertEqual(charts.warm_left_note(rows),
                         "LanceDB's warm runs cache only its index files and leave its 30.7 GB table on disk. "
                         "DiskANN's warm runs keep only its PQ codes in memory and read the sectors of its 41.9 GB "
                         "index from the disk.")
        self.assertEqual(charts.warm_left_note(rows[:1]), "")

    def test_rows_of_fewer_queries_get_a_note_on_their_subsets(self):
        self.assertNotIn("subset", self.by_id["cold-10m-95"]["footer"])
        from tests import test_headtohead as h

        more = [h.cold("coldivf", "python", "tl", h.OURS, 13, 10_100.0, footprint_delta=24_000_000,
                       disk_bytes=24_000_000, extra={"resident_loaded_bytes_after": 0})]

        with open(self.root / "results" / "t" / "coldivf.jsonl", "a") as f:
            f.write("".join(json.dumps(x) + "\n" for x in more))

        made, _ = charts.h2h_charts(self.root / "suite.toml", self.root)
        cold = next(c for c in made if c["id"] == "cold-10m-95")
        self.assertEqual([r["queries"] for r in cold["rows"]], [5, 4])
        self.assertIn("Rows with fewer queries time a subset of them", cold["footer"])

    def test_the_cold_chart_at_095_keeps_the_control_cells(self):
        controls = self.by_id["cold-10m-95"]["controls"]
        self.assertEqual([(c["config_id"], c["fresh_clone"], c["trials"]) for c in controls], [("tl-clone", True, 2)])
        self.assertNotIn("controls", self.by_id["cold-10m-90"])

    def test_readme_tables_of_warm_queries_and_memory(self):
        data = {"charts": self.charts}
        warm = charts.h2h_warm_table(data, "10m")
        self.assertEqual(warm.splitlines()[0], "| System | Warm p50 at 0.90 | QPS, 6 Threads, at 0.90 | Warm p50 at "
                                               "0.95 | QPS, 6 Threads, at 0.95 | QPS Method |")
        self.assertIn("| **ColdIVF** | **0.310 ms** | **6,000** | **0.410 ms** | **5,000** | threads |", warm)
        self.assertIn("| faiss IVF as shipped, 640 per list | 2.10 ms | 2,000 | 2.10 ms | 2,000 | batch call |", warm)
        memory = charts.h2h_memory_table(data, "10m")
        lines = memory.splitlines()
        self.assertTrue(lines[2].startswith("| **ColdIVF** | **48 MB** | **30 MB + 9 GB** | **10.0 min** | **800 MB**"))
        self.assertIn("| turbovec 4-bit, at its setting of most recall |", memory)
        self.assertEqual({x.count("|") for x in lines}, {7})

    def test_the_warm_table_takes_every_target_of_the_warm_charts_and_the_qps_charts(self):
        # Were ColdIVF to reach recall@10 0.99, it would get a QPS chart, while the warm charts keep their targets.
        qps = json.loads(json.dumps(self.by_id["qps-10m-95"]))
        qps |= {"id": "qps-10m-99", "target_recall10": 0.99, "rows": [qps["rows"][0] | {"qps": 1200.0}]}
        warm = charts.h2h_warm_table({"charts": self.charts + [qps]}, "10m")
        self.assertTrue(warm.splitlines()[0].endswith("| Warm p50 at 0.99 | QPS, 6 Threads, at 0.99 | QPS Method |"))
        self.assertIn("| **ColdIVF** | **0.310 ms** | **6,000** | **0.410 ms** | **5,000** | - | **1,200** | threads |",
                      warm)

    def test_a_same_batch_check_compares_each_cell_with_coldivfs_module(self):
        from tests import test_headtohead as h

        check = self.root / "check.toml"
        check.write_text(CHECK.format(root=self.root))
        folder = self.root / "results" / "check"
        folder.mkdir(parents=True)
        lines = {"coldivf": [h.cold("coldivf", "python", "tl", h.OURS, q, 10_000.0) for q in (1, 3, 5)]
                 + [h.cold("coldivf", "", "tl", h.OURS, q, 9_000.0) for q in (1, 3)],
                 "faiss-ivf": [h.cold("faiss-ivf", "batched", "vpl", h.FAISS, q, t)
                               for q, t in ((1, 11_000.0), (3, 12_000.0), (5, 9_000.0), (5, 99_000.0))]}
        lines["faiss-ivf"][-1]["status"] = "invalid"

        for name, rows in lines.items():
            (folder / f"{name}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))

        got = charts.same_batch(check, self.root)
        cells = {c["label"]: c for c in got["cells"]}
        self.assertEqual(set(cells), {"ColdIVF", "ColdIVF, binaries", "faiss IVF, batched reads"})
        theirs = cells["faiss IVF, batched reads"]
        self.assertEqual((theirs["trials"], theirs["p50_us"]), (3, 11_000.0))
        self.assertAlmostEqual(theirs["against"]["c0_p50"], 1.1)
        self.assertEqual((theirs["against"]["c0_paired"]["queries"], theirs["against"]["c0_paired"]["faster"]), (3, 1))
        self.assertEqual(cells["ColdIVF, binaries"]["against"]["c0_paired"]["median"], 0.9)
        data = charts.chart_data(self.root / "no-quiet-run", self.root, {"10m": self.root / "suite.toml"},
                                 {"10m": check})
        self.assertEqual(charts.chart_of(data, "cold-10m-95")["same_batch"]["folder"], "results/check")
        self.assertNotIn("same_batch", charts.chart_of(data, "cold-10m-90"))

    @unittest.skipUnless(HAS_MATPLOTLIB, "matplotlib isn't installed")
    def test_a_draw_writes_only_dark_charts(self):
        out = self.root / "charts"
        args = ["--quiet", str(self.root / "no-quiet-run"), "--h2h", f"10m={self.root / 'suite.toml'}", "--out",
                str(out), "--repo", str(self.root)]

        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(charts.main(args), 0)

        # One file per chart that has lines. With no quiet run, the chart of ColdIVF in each setup has none.
        self.assertEqual(sorted(p.name for p in out.iterdir()),
                         sorted(["chart-data.json"] + [f"{c['id']}-dark.svg" for c in self.charts]))

        for name in ("cold-10m-95", "cold-10m-90", "warm-10m-95", "warm-10m-90"):
            assert_value_columns(self, out / f"{name}-dark.svg")
            self.assertEqual(svg_texts(out / f"{name}-dark.svg").count("Line to the p99"), 1, name)

        warm = svg_texts(out / "warm-10m-95-dark.svg")
        self.assertIn("Warm query at 1 thread (ms), log scale", warm)
        self.assertIn("nprobe 48, k_factor 4, qb 4 · recall 0.950 · 500 queries", warm)
        self.assertEqual(warm[warm.index("0.410 ms") + 1], "0.615 ms")
        self.assertEqual(warm[warm.index("2.10 ms") + 1], "3.15 ms")

        # A QPS chart heads its two columns once, with the thread counts, and its legend has no line to a p99. Like the
        # cold and warm charts, it leaves out the systems that don't reach its target.
        for name in ("qps-10m-95", "qps-10m-90"):
            texts = svg_texts(out / f"{name}-dark.svg")
            self.assertEqual((texts.count("6 threads"), texts.count("12 threads")), (1, 1), name)
            self.assertFalse({"p50", "p99", "Line to the p99", "turbovec"} & set(texts), name)
            self.assertFalse([s for s in texts if s.startswith("Not reached")], name)

        qps = svg_texts(out / "qps-10m-95-dark.svg")
        self.assertIn("Queries per second at 0.001M vectors, recall@10 0.95", qps)
        self.assertIn("Queries per second on 6 threads, log scale", qps)
        self.assertIn("nprobe 64, rerank 39 · recall 0.952 · threads", qps)
        self.assertIn("nprobe 48, k_factor 4, qb 4 · recall 0.950 · batch call", qps)
        # The highest stands first. ColdIVF's 12-thread number is a dash, since it has no 12-thread line.
        self.assertLess(qps.index("ColdIVF"), qps.index("faiss IVF as shipped, 640 per list"))
        self.assertEqual(qps[qps.index("5,000") + 1], "-")
        self.assertEqual(qps[qps.index("2,000") + 1], "2,500")

    @unittest.skipUnless(HAS_MATPLOTLIB, "matplotlib isn't installed")
    def test_a_qps_chart_draws_only_its_dots(self):
        from matplotlib.axes import Axes

        plot, marks = Axes.plot, []

        def record(ax, *args, **kwargs):
            marks.append(kwargs.get("marker"))
            return plot(ax, *args, **kwargs)

        for name in ("qps-10m-95", "warm-10m-95"):
            chart = self.by_id[name]

            with mock.patch.object(Axes, "plot", record), charts._mpl().matplotlib.rc_context(charts._rc(charts.THEME)):
                charts.DRAW[chart["kind"]]({}, chart, charts.THEME, self.root / f"{name}.svg")

        # One dot per row, where the warm chart also draws each row's line and tick at its p99.
        self.assertEqual(marks, ["o", "o"] + [None, "|", "o"] * 2)
        # The chart leaves its last dot the room it leaves its first, so a top just under a power of ten takes the next,
        # and a tick of a million is written in full.
        chart = json.loads(json.dumps(self.by_id["qps-10m-95"]))
        ticks = []

        for qps in (5000.0, 9000.0, 90000.0):
            chart["rows"][0]["qps"] = qps

            with charts._mpl().matplotlib.rc_context(charts._rc(charts.THEME)):
                charts.DRAW[chart["kind"]]({}, chart, charts.THEME, self.root / "qps.svg")

            ticks.append([s for s in svg_texts(self.root / "qps.svg")
                          if s in ("1,000", "10,000", "100,000", "1,000,000", "1e+06")])

        self.assertEqual(ticks, [["1,000", "10,000"], ["1,000", "10,000", "100,000"],
                                 ["1,000", "10,000", "100,000", "1,000,000"]])

    @unittest.skipUnless(HAS_MATPLOTLIB, "matplotlib isn't installed")
    def test_a_qps_chart_keeps_its_rows_text_inside_the_margins_with_the_widest_setting(self):
        # The widest setting of the 1M and 10M charts, on a row whose 12 threads ran another way, which only the footer
        # names, and a 12-thread number as wide as any.
        chart = json.loads(json.dumps(self.by_id["qps-10m-95"]))
        chart["rows"][1] |= {"setting_text": "nprobe 128, k_factor 8, qb 4, quantizer_ef 128", "recall10_warm": 0.953,
                             "qps_12": 16020.07, "invalid": [12]}
        save, pages, rc = charts.Page.save, [], charts._rc(charts.THEME)

        def keep(page, path):
            pages.append(page)
            save(page, path)

        with mock.patch.object(charts.Page, "save", keep), charts._mpl().matplotlib.rc_context(rc):
            charts.DRAW[chart["kind"]]({}, chart, charts.THEME, self.root / "qps.svg")
            (ax,) = pages[0].fig.axes
            renderer = pages[0].fig.canvas.get_renderer()
            boxes = {t.get_text(): t.get_window_extent(renderer) for t in ax.texts}

        self.assertGreater(boxes["nprobe 128, k_factor 8, qb 4, quantizer_ef 128 · recall 0.953 · batch call"].x0,
                           charts.MARGIN)
        self.assertLess(boxes["16,020 (invalid)"].x1, charts.WIDTH - charts.MARGIN)
        self.assertGreater(min(b.x0 for b in boxes.values()), charts.MARGIN)

    @unittest.skipUnless(HAS_MATPLOTLIB, "matplotlib isn't installed")
    def test_an_invalid_12_thread_number_is_marked_and_a_missing_one_is_a_dash(self):
        chart = json.loads(json.dumps(self.by_id["qps-10m-95"]))
        chart["rows"][1]["invalid"] = [12]
        path = self.root / "qps.svg"

        with charts._mpl().matplotlib.rc_context(charts._rc(charts.THEME)):
            charts.DRAW[chart["kind"]]({}, chart, charts.THEME, path)

        texts = svg_texts(path)
        self.assertEqual([texts[texts.index(v) + 1] for v in ("5,000", "2,000")], ["-", "2,500 (invalid)"])

    @unittest.skipUnless(HAS_MATPLOTLIB, "matplotlib isn't installed")
    def test_a_row_without_a_p99_draws_only_its_dot(self):
        chart = json.loads(json.dumps(self.by_id["warm-10m-95"]))
        chart["rows"][1]["p99_us"] = None
        path = self.root / "warm.svg"

        with charts._mpl().matplotlib.rc_context(charts._rc(charts.THEME)):
            charts.DRAW[chart["kind"]]({}, chart, charts.THEME, path)

        texts = svg_texts(path)
        self.assertEqual(texts[texts.index("2.10 ms") + 1], "-")

    @unittest.skipUnless(HAS_MATPLOTLIB, "matplotlib isn't installed")
    def test_a_p99_under_its_dot_draws_no_tick_and_keeps_its_column(self):
        chart = json.loads(json.dumps(self.by_id["warm-10m-95"]))
        path = self.root / "warm.svg"
        marks = []

        # At 3.15 ms faiss's tick stands clear of its dot at 2.10 ms, and at 2.11 ms the dot would hide it.
        for p99 in (3150.0, 2110.0):
            chart["rows"][1]["p99_us"] = p99

            with charts._mpl().matplotlib.rc_context(charts._rc(charts.THEME)):
                charts.DRAW[chart["kind"]]({}, chart, charts.THEME, path)

            marks.append(path.read_text().count("<use "))

        self.assertEqual(marks[0] - marks[1], 1)
        texts = svg_texts(path)
        self.assertEqual(texts[texts.index("2.10 ms") + 1], "2.11 ms")


if __name__ == "__main__":
    unittest.main()
