import json
import math
import tempfile
import unittest
from pathlib import Path

import driver
import headtohead

SUITE = """
phase = "t"
dataset = "tiny-16-cosine"
n = 1000
index_root = "{root}/indexes"
results_root = "{root}/results"

[[system]]
name = "coldivf"
adapter = "coldivf"

  [[system.build]]
  config_id = "tl"
  config = {{ vectors_per_list = 20 }}

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
  trials = 4

  [[system.cold]]
  config_id = "tl"
  params = {{ nprobe = 32, rerank = 39 }}
  split = "test"
  trials = 4

  [[system.build]]
  config_id = "tl-clone"
  index = "{root}/indexes/coldivf/tl-n1000-s1"

  [[system.cold]]
  config_id = "tl-clone"
  params = {{ nprobe = 64, rerank = 39 }}
  split = "test"
  trials = 2
  fresh_clone = true

[[system]]
name = "faiss-ivf"
adapter = "faiss_ivf"
variant = "shipped"

  [[system.build]]
  config_id = "cp640-rq1"
  config = {{ bits = 1 }}

  [[system.cold]]
  config_id = "cp640-rq1"
  params = {{ nprobe = 48, k_factor = 4, qb = 4 }}
  split = "test"
  trials = 4
  group = "faiss"

[[system]]
name = "turbovec"
adapter = "turbovec"

  [[system.build]]
  config_id = "b4"
  config = {{ bits = 4 }}

  [[system.cold]]
  config_id = "b4"
  params = {{}}
  split = "test"
  trials = 2
  group = "turbovec"
"""

OURS = {"nprobe": 64, "rerank": 39}
BELOW = {"nprobe": 32, "rerank": 39}
FAISS = {"nprobe": 48, "k_factor": 4, "qb": 4}


def line(system: str, variant: str, config: str, kind: str, **more) -> dict:
    out = {"kind": kind, "status": "ok", "system": system, "variant": variant, "config_id": config, "n": 1000}
    out.update(more)
    return out


def warm(system, variant, config, params, recall, p50, split="tune", threads=1, **more) -> dict:
    return line(system, variant, config, "warm", split=split, threads=threads, search_params=params, recall10=recall,
                p50_us=p50, bytes_scanned=None, mode="whole", **more)


def cold(system, variant, config, params, query, total, split="test", **more) -> dict:
    return line(system, variant, config, "cold", split=split, regime="c0", search_params=params, query_id=query,
                total_us=total, **more)


class MemoryTest(unittest.TestCase):
    def test_a_faiss_trial_counts_the_parts_cached_after_its_first_query_less_the_loaded_ones(self):
        x = {"footprint_delta": 3_000_000, "disk_bytes": 99_000_000,
             "extra": {"resident_after_first": {"rotation": 2_000_000, "coarse": 48_000_000, "lists": 11_000_000},
                       "resident_loaded_bytes_after": 2_000_000}}
        self.assertEqual(headtohead.first_query_pages(x), 59_000_000)
        self.assertEqual(headtohead.cold_memory(x), {"process": 3_000_000, "pages": 59_000_000, "total": 62_000_000})

    def test_a_reader_without_that_counts_the_bytes_it_read_less_the_loaded_ones(self):
        mapped = {"footprint_delta": 164_000_000, "disk_bytes": 2_400_000_000,
                  "extra": {"resident_loaded_bytes_after": 0}}
        self.assertEqual(headtohead.first_query_pages(mapped), 2_400_000_000)
        loaded = {"footprint_delta": 5_100_000_000, "disk_bytes": 5_000_000_000,
                  "extra": {"resident_loaded_bytes_after": 5_000_000_100}}
        self.assertEqual(headtohead.cold_memory(loaded)["total"], 5_100_000_000)

    def test_a_trial_without_disk_bytes_has_no_pages(self):
        self.assertEqual(headtohead.cold_memory({"footprint_delta": 10}), {"process": 10, "pages": None,
                                                                           "total": None})


class StatisticsTest(unittest.TestCase):
    def test_exact_target_is_linear_in_log_latency_between_the_points_around_it(self):
        points = [{"setting": {"a": 1}, "recall10": 0.94, "p50_us": 1000.0},
                  {"setting": {"a": 2}, "recall10": 0.96, "p50_us": 4000.0},
                  {"setting": {"a": 3}, "recall10": 0.99, "p50_us": 9000.0},
                  {"setting": {"a": 4}, "recall10": 0.90, "p50_us": 500.0}]
        got = headtohead.at_exactly(points, 0.95)
        self.assertAlmostEqual(got["p50_us"], 2000.0)
        self.assertEqual([b["setting"] for b in got["between"]], [{"a": 1}, {"a": 2}])
        self.assertAlmostEqual(headtohead.at_exactly(points, 0.96)["p50_us"], 4000.0)
        self.assertIsNone(headtohead.at_exactly(points, 0.995))
        self.assertIsNone(headtohead.at_exactly(points, 0.85))
        self.assertAlmostEqual(headtohead.at_exactly(points[1:3], 0.96)["p50_us"], 4000.0)

    def test_paired_ratios_take_each_querys_median_over_the_queries_both_timed(self):
        theirs = [{"query_id": 1, "total_us": 30.0}, {"query_id": 1, "total_us": 50.0},
                  {"query_id": 3, "total_us": 90.0}, {"query_id": 7, "total_us": 5.0}]
        ours = [{"query_id": 1, "total_us": 10.0}, {"query_id": 3, "total_us": 30.0}, {"query_id": 5, "total_us": 1.0}]
        got = headtohead.paired(theirs, ours, resamples=200)
        self.assertEqual((got["queries"], got["median"], got["min"], got["max"]), (2, 3.5, 3.0, 4.0))
        self.assertTrue(3.0 <= got["low"] <= got["median"] <= got["high"] <= 4.0)
        self.assertEqual(got, headtohead.paired(theirs, ours, resamples=200))
        self.assertIsNone(headtohead.paired(theirs, [{"query_id": 9, "total_us": 1.0}]))

    def test_names_and_settings(self):
        self.assertEqual(headtohead.label("faiss-ivf", "ondisk", "vpl2560-rq1"),
                         "faiss IVF, on-disk lists, 2,560 per list")
        self.assertEqual(headtohead.label("rabitq-ivf", "", "cp640-b9"), "RaBitQ-Library IVF 9-bit")
        self.assertEqual(headtohead.label("turbovec", "", "b4-cal"), "turbovec 4-bit, calibrated")
        self.assertEqual(headtohead.setting_text({"nprobe": 48, "k_factor": 4, "qb": 4, "prefetch_threads": 32}),
                         "nprobe 48, k_factor 4")
        self.assertEqual(headtohead.setting_text({"nprobes": 48, "refine_factor": 8}), "nprobes 48, refine 8")
        self.assertEqual(headtohead.setting_text({}), "none")
        self.assertEqual((headtohead.fmt_ms(332.0), headtohead.fmt_ms(10_350.0), headtohead.fmt_ms(4_424_850.0)),
                         ("0.332", "10.3", "4,425"))


class CollectTest(unittest.TestCase):
    """A phase with ColdIVF's Python module, one faiss setup and turbovec, which reaches no target."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "suite.toml").write_text(SUITE.format(root=self.root))
        self.suite = driver.load_suite(self.root / "suite.toml")
        folder = self.root / "results" / "t"
        folder.mkdir(parents=True)
        lines = {"coldivf": [], "faiss-ivf": [], "turbovec": []}
        c = lines["coldivf"]
        c += [warm("coldivf", "python", "tl", BELOW, 0.93, 300.0), warm("coldivf", "python", "tl", OURS, 0.955, 400.0)]
        c += [warm("coldivf", "python", "tl", BELOW, 0.944, 310.0, split="test"),
              warm("coldivf", "python", "tl", OURS, 0.952, 410.0, split="test", footprint=30_000_000,
                   resident_file_bytes=9_000_000_000, memory_total=9_030_000_000)]
        c += [warm("coldivf", "python", "tl", OURS, None, None, split="test", threads=6, qps=5000.0)]
        memory = {"footprint_delta": 24_000_000, "disk_bytes": 24_000_000, "extra": {"resident_loaded_bytes_after": 0}}
        c += [cold("coldivf", "python", "tl", OURS, q, 10_000.0 + q, **memory) for q in (1, 3, 5, 7)]
        c += [cold("coldivf", "python", "tl", BELOW, q, 8_000.0, **memory) for q in (1, 3, 5, 7)]
        c += [cold("coldivf", "python", "tl", OURS, q, 99_000.0, status="invalid") for q in (9,)]
        c += [cold("coldivf", "python", "tl-clone", OURS, q, 11_000.0, extra={"clone": "x"}) for q in (1, 3)]
        c.append(line("coldivf", "", "tl", "build", seconds=600.0, peak_footprint=800_000_000,
                      index_bytes=9_430_000_000))
        f = lines["faiss-ivf"]
        f += [warm("faiss-ivf", "shipped", "cp640-rq1", FAISS, 0.951, 2000.0),
              warm("faiss-ivf", "shipped", "cp640-rq1", FAISS, 0.95, 2100.0, split="test")]
        f += [cold("faiss-ivf", "shipped", "cp640-rq1", FAISS, q, 30_000.0 + 1000 * q, footprint_delta=3_000_000,
                   extra={"resident_after_first": {"coarse": 48_000_000, "lists": 12_000_000},
                          "resident_loaded_bytes_after": 0}) for q in (1, 3, 5, 11)]
        f.append(line("faiss-ivf", "shipped", "cp640-rq1", "build", seconds=150.0, peak_footprint=9_400_000_000,
                      index_bytes=8_850_000_000))
        t = lines["turbovec"]
        t += [warm("turbovec", "", "b4", {}, 0.89, 27_000.0), warm("turbovec", "", "b4", {}, 0.894, 26_800.0,
                                                                     split="test")]
        t += [cold("turbovec", "", "b4", {}, q, 997_000.0, footprint_delta=4_880_000_000, disk_bytes=3_880_000_000,
                   extra={"resident_loaded_bytes_after": 3_880_000_000}) for q in (1, 3)]

        for name, rows in lines.items():
            (folder / f"{name}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))

        self.data = headtohead.collect(self.suite, headtohead.Results(folder))

    def tearDown(self):
        self.tmp.cleanup()

    def test_each_target_takes_each_builds_pick_and_lists_what_reaches_none(self):
        t90, t95, t99 = (self.data["targets"][t] for t in (0.90, 0.95, 0.99))
        self.assertEqual([r["setting"] for r in t90["rows"]], [BELOW, FAISS])
        self.assertEqual([r["label"] for r in t95["rows"]], ["ColdIVF", "faiss IVF as shipped, 640 per list"])
        self.assertEqual(t95["reference"], "ColdIVF")
        self.assertEqual({m["label"] for m in t99["not_reached"]}, {"ColdIVF", "faiss IVF as shipped, 640 per list",
                                                                    "turbovec 4-bit"})
        self.assertEqual([r["label"] for r in self.data["hand"]], ["turbovec 4-bit"])

    def test_a_row_counts_ok_trials_only_and_leaves_the_clones_apart(self):
        ours = self.data["targets"][0.95]["rows"][0]
        self.assertEqual((ours["trials"], ours["queries"], ours["status"]), (4, 4, {"ok": 4, "invalid": 1}))
        self.assertEqual((ours["c0_p50_us"], ours["recall10"], ours["qps"][6]), (10_004.0, 0.952, 5000.0))
        self.assertEqual((ours["cold_process"], ours["cold_pages"], ours["cold_total"]),
                         (24_000_000, 24_000_000, 48_000_000))
        self.assertEqual((ours["warm_total"], ours["warm_mode"]), (9_030_000_000, "whole"))
        # The Python variant searches the binaries' build, so its build numbers are that build's.
        self.assertTrue(ours["shared_file"])
        self.assertEqual(ours["index_bytes"], 9_430_000_000)
        clone = [c for c in self.data["controls"] if c["config_id"] == "tl-clone"]
        self.assertEqual([(c["fresh_clone"], c["trials"], c["c0_p50_us"]) for c in clone], [(True, 2, 11_000.0)])

    def test_comparisons_pair_queries_and_match_recall(self):
        ours, theirs = self.data["targets"][0.95]["rows"]
        against = theirs["against"]
        # faiss timed queries 1, 3, 5 and 11, ColdIVF 1, 3, 5 and 7, so 3 pair up.
        self.assertEqual(against["c0_paired"]["queries"], 3)
        self.assertAlmostEqual(against["c0_paired"]["median"], 33_000.0 / 10_003.0)
        self.assertAlmostEqual(against["c0_p50"], 34_000.0 / 10_004.0)
        self.assertEqual((theirs["cold_pages"], theirs["cold_total"]), (60_000_000, 63_000_000))
        # ColdIVF at exactly 0.95 lies between its cells at 0.944 and 0.952, and faiss's only cell is at 0.95.
        exact = ours["c0_exact"]["p50_us"]
        self.assertAlmostEqual(exact, math.exp(0.25 * math.log(8_000.0) + 0.75 * math.log(10_004.0)))
        self.assertAlmostEqual(theirs["c0_exact"]["p50_us"], 34_000.0)
        self.assertAlmostEqual(against["c0_exact"], 34_000.0 / exact)
        self.assertAlmostEqual(against["warm_p50"], 2100.0 / 410.0)

    def test_markdown_and_json(self):
        text = headtohead.markdown(self.data)
        self.assertIn("### Recall@10 0.95", text)
        self.assertIn("| ColdIVF | nprobe 64, rerank 39 | 0.9520 | 0.410 | 5,000 |", text)
        self.assertIn("24 MB + 24 MB = 48 MB", text)
        self.assertIn("| turbovec 4-bit | not reached, best tune recall@10 0.8900 at none |", text)
        out = json.loads(json.dumps(headtohead.public(self.data)))
        self.assertNotIn("_trials", out["targets"]["0.95"]["rows"][0])


if __name__ == "__main__":
    unittest.main()
