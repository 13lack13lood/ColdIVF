import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import driver
import pick
import schema

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

  [[system.cold]]
  config_id = "tl"
  params = {{ nprobe = 1, rerank = 1 }}
  trials = 1
  dummy_n = 200

# The rival's own comment, which stays right above its section.
[[system]]
name = "rival"
adapter = "flat"
variant = "fast"

  [[system.build]]
  config_id = "r"

  [[system.cold]]
  config_id = "r"
  params = {{ ef = 10 }}
  trials = 1
  group = "rival"
  threads = 4
"""

# nprobe, rerank, recall@10 and warm p50 of each setting of the coldivf build. nprobe 96 with rerank 39 reaches 0.95
# at the lowest p50, and no setting reaches 0.99.
GRID = [(32, 39, 0.880, 300.0), (64, 39, 0.905, 400.0), (64, 64, 0.909, 450.0), (96, 29, 0.921, 470.0),
        (96, 39, 0.951, 500.0), (128, 39, 0.952, 520.0), (128, 64, 0.960, 600.0), (192, 64, 0.975, 800.0),
        (256, 100, 0.986, 1100.0)]


def put(path: Path, line: dict) -> None:
    """Appends a result line as the driver writes it, without the fields pick.py doesn't read."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "a") as f:
        f.write(json.dumps(line) + "\n")


def warm_line(system: str, config: str, params: dict, recall: float, p50: float, **more) -> dict:
    line = {"kind": "warm", "status": "ok", "system": system, "variant": "", "config_id": config, "n": 1000,
            "split": "tune", "threads": 1, "search_params": params, "recall10": recall, "p50_us": p50,
            "bytes_scanned": None}
    line.update(more)
    return line


def cold_line(params: dict, total_us: float, **more) -> dict:
    line = {"kind": "cold", "status": "ok", "system": "coldivf", "variant": "", "config_id": "tl", "n": 1000,
            "split": "tune", "regime": "c0", "search_params": params, "total_us": total_us}
    line.update(more)
    return line


class PickTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.suite = self.root / "s.toml"
        self.suite.write_text(SUITE.format(root=self.root))
        self.results = self.root / "results" / "t" / "coldivf.jsonl"

        for nprobe, rerank, recall, p50 in GRID:
            put(self.results, warm_line("coldivf", "tl", {"nprobe": nprobe, "rerank": rerank}, recall, p50))

        # Lines that must not count: another build, 6 threads, the test split, a failed line.
        put(self.results, warm_line("coldivf", "other", {"nprobe": 8, "rerank": 39}, 0.99, 10.0))
        put(self.results, warm_line("coldivf", "tl", {"nprobe": 8, "rerank": 39}, 0.99, 10.0, threads=6))
        put(self.results, warm_line("coldivf", "tl", {"nprobe": 8, "rerank": 39}, 0.99, 10.0, split="test"))
        put(self.results, warm_line("coldivf", "tl", {"nprobe": 8, "rerank": 39}, 0.99, 10.0,
                                              status="failed"))

    def tearDown(self):
        self.tmp.cleanup()

    def run_pick(self, *args: str) -> str:
        out = io.StringIO()

        with redirect_stdout(out):
            self.assertEqual(pick.main([args[0], str(self.suite), *args[1:]]), 0)

        return out.getvalue()

    def cells(self, system: str = "coldivf", variant: str = "") -> list[driver.Cell]:
        return pick.find_system(driver.load_suite(self.suite), system, variant).cold

    def test_screen_takes_the_cheapest_settings_that_reach_each_target(self):
        grid = pick.points(schema.read(self.results), *self.system_build())
        self.assertEqual(len(grid), len(GRID))
        picks = {p.target: [x.params for x, _ in p.cells] for p in pick.screen(grid, [0.90, 0.95, 0.99], 3)}
        self.assertEqual(picks[0.90], [{"nprobe": 64, "rerank": 39}, {"nprobe": 64, "rerank": 64},
                                       {"nprobe": 96, "rerank": 29}])
        self.assertEqual(picks[0.95], [{"nprobe": 96, "rerank": 39}, {"nprobe": 128, "rerank": 39},
                                       {"nprobe": 128, "rerank": 64}])
        self.assertEqual(picks[0.99], [])

    def test_a_setting_with_only_invalid_lines_ranks_by_its_latest_and_says_so(self):
        # nprobe 80 has only invalid lines, and the latest makes it the cheapest at 0.95. nprobe 96 with rerank 39 has
        # an invalid line after its ok one, which must not replace it.
        params = {"nprobe": 80, "rerank": 39}
        put(self.results, warm_line("coldivf", "tl", params, 0.951, 700.0, status="invalid"))
        put(self.results, warm_line("coldivf", "tl", params, 0.953, 450.0, status="invalid"))
        put(self.results, warm_line("coldivf", "tl", {"nprobe": 96, "rerank": 39}, 0.999, 1.0, status="invalid"))
        grid = pick.points(schema.read(self.results), *self.system_build())
        by = {p.key: p for p in grid}
        self.assertTrue(by[pick.canonical(params)].invalid)
        self.assertEqual((by[pick.canonical(params)].recall10, by[pick.canonical(params)].p50_us), (0.953, 450.0))
        kept = by[pick.canonical({"nprobe": 96, "rerank": 39})]
        self.assertEqual((kept.invalid, kept.recall10), (False, 0.951))
        self.assertEqual(pick.cheapest(grid, 0.95, 1)[0].params, params)
        text = self.run_pick("screen", "--system", "coldivf", "--config", "tl", "--targets", "0.95")
        self.assertIn("warm from a line the machine's swapping made invalid", text)

    def test_a_tie_on_p50_goes_to_fewer_bytes(self):
        grid = [pick.Point({"a": 1}, 0.96, 500.0, 900.0), pick.Point({"a": 2}, 0.96, 500.0, 100.0)]
        self.assertEqual(pick.cheapest(grid, 0.95, 1)[0].params, {"a": 2})

    def test_screen_prints_cells_on_the_tune_split_with_the_first_cells_dummy(self):
        text = self.run_pick("screen", "--system", "coldivf", "--config", "tl")
        self.assertIn("0.99 not reached. The best is", text)
        self.assertIn('{"nprobe": 256, "rerank": 100}', text)
        self.assertEqual(text.count("[[system.cold]]"), 6)
        self.assertIn('split = "tune"', text)
        self.assertIn("trials = 30", text)
        self.assertIn("dummy_n = 200", text)
        self.assertNotIn("group =", text)

    def test_write_puts_the_cells_in_the_systems_own_section_once(self):
        self.run_pick("screen", "--system", "coldivf", "--config", "tl", "--targets", "0.95", "--write")
        params = [c.params for c in self.cells() if c.split == "tune"]
        self.assertEqual(params, [{"nprobe": 96, "rerank": 39}, {"nprobe": 128, "rerank": 39},
                                  {"nprobe": 128, "rerank": 64}])
        self.assertEqual([c.params for c in self.cells("rival", "fast")], [{"ef": 10}])
        self.assertTrue(all(c.dummy.n == 200 and c.trials == 30 for c in self.cells() if c.split == "tune"))
        text = self.suite.read_text()
        comment = "# The rival's own comment, which stays right above its section.\n"
        self.assertIn(comment + '[[system]]\nname = "rival"', text)
        self.assertLess(text.index("rerank = 64 }"), text.index("# The rival's own comment"))
        before = self.suite.read_text()
        self.run_pick("screen", "--system", "coldivf", "--config", "tl", "--targets", "0.95", "--write")
        self.assertEqual(self.suite.read_text().count("[[system.cold]]"), before.count("[[system.cold]]"))

    def test_write_into_the_last_section_keeps_its_group_and_threads(self):
        put(self.root / "results" / "t" / "rival.jsonl",
                      warm_line("rival", "r", {"ef": 64}, 0.96, 200.0, variant="fast"))
        self.run_pick("screen", "--system", "rival", "--variant", "fast", "--config", "r", "--targets", "0.95",
                      "--write")
        new = [c for c in self.cells("rival", "fast") if c.split == "tune"]
        self.assertEqual([(c.params, c.group, c.threads) for c in new], [({"ef": 64}, "rival", 4)])
        self.assertEqual(len([c for c in self.cells() if c.split == "tune"]), 0)

    def test_verdict_takes_the_lowest_cold_p50_and_the_point_below(self):
        # 30 trials each, apart enough that no two medians' intervals overlap. 128 with rerank 39 has the lowest
        # median, though its 5 slow outliers give it the highest mean.
        for params, times in (({"nprobe": 96, "rerank": 39}, [9000.0 + i for i in range(30)]),
                              ({"nprobe": 128, "rerank": 39}, [8000.0 + i for i in range(25)] + [30000.0] * 5),
                              ({"nprobe": 128, "rerank": 64}, [8500.0 + i for i in range(30)])):
            for t in times:
                put(self.results, cold_line(params, t))

        # A rejected trial and a test-split trial don't count.
        put(self.results, cold_line({"nprobe": 96, "rerank": 39}, 10.0, status="rejected"))
        put(self.results, cold_line({"nprobe": 96, "rerank": 39}, 10.0, split="test"))
        grid = pick.points(schema.read(self.results), *self.system_build())
        (only,) = pick.verdict(grid, [0.95], 3, 20, True)
        self.assertEqual([p.params for p, _ in only.cells], [{"nprobe": 128, "rerank": 39},
                                                            {"nprobe": 96, "rerank": 29}])
        text = self.run_pick("verdict", "--system", "coldivf", "--config", "tl", "--targets", "0.95")
        self.assertIn('split = "test"', text)
        self.assertIn("trials = 100", text)
        self.assertIn("the fastest cold of 3", text)
        self.assertIn("C0 p50 8.01 ms in 30 tune trials", text)

    def test_a_cheaper_setting_as_fast_cold_within_the_noise_wins(self):
        # 128 with rerank 39 has the lower median by 10 us, but its median's 95% interval, the 10th to 21st of 30,
        # overlaps that of the cheaper 96 with rerank 39, so 96 wins. 128 with rerank 64 is far slower.
        for params, times in (({"nprobe": 96, "rerank": 39}, [8000.0 + i for i in range(30)]),
                              ({"nprobe": 128, "rerank": 39}, [7990.0 + i for i in range(30)]),
                              ({"nprobe": 128, "rerank": 64}, [9500.0 + i for i in range(30)])):
            for t in times:
                put(self.results, cold_line(params, t))

        self.assertEqual(pick.median_interval([float(i) for i in range(1, 31)]), (10.0, 21.0))
        self.assertEqual(pick.median_interval([float(i) for i in range(1, 101)]), (40.0, 61.0))
        grid = pick.points(schema.read(self.results), *self.system_build())
        (only,) = pick.verdict(grid, [0.95], 3, 20, False)
        self.assertEqual([p.params for p, _ in only.cells], [{"nprobe": 96, "rerank": 39}])
        self.assertIn('the fastest\'s, {"nprobe": 128, "rerank": 39} at 8.00 ms', only.cells[0][1])

    def test_verdict_with_warm_adds_test_split_warm_entries_once(self):
        for params, times in (({"nprobe": 96, "rerank": 39}, [9000.0] * 3),
                              ({"nprobe": 128, "rerank": 39}, [8000.0] * 3),
                              ({"nprobe": 128, "rerank": 64}, [8500.0] * 3)):
            for x in times:
                put(self.results, cold_line(params, x))

        args = ("verdict", "--system", "coldivf", "--config", "tl", "--targets", "0.95", "--min-trials", "3", "--warm",
                "--write")
        self.run_pick(*args)
        suite = driver.load_suite(self.suite)
        warm = [w for w in pick.find_system(suite, "coldivf", "").warm if w.split == "test"]
        self.assertEqual([w.threads for w in warm], [1, 6, 12])
        self.assertTrue(all(w.settings == [{"nprobe": 128, "rerank": 39}, {"nprobe": 96, "rerank": 29}] for w in warm))
        self.assertEqual((warm[0].passes, warm[1].qps_seconds, warm[2].qps_repeats), (3, 10.0, 3))
        before = self.suite.read_text()
        self.run_pick(*args)
        self.assertEqual(self.suite.read_text(), before)

    def test_a_dummy_path_takes_the_place_of_the_first_cells_dummy(self):
        text = self.run_pick("screen", "--system", "coldivf", "--config", "tl", "--targets", "0.95", "--dummy",
                             "data/bench.noindex/x-dummy")
        self.assertIn('dummy = "data/bench.noindex/x-dummy"', text)
        self.assertNotIn("dummy_n", text)

    def test_verdict_needs_the_screen_first(self):
        grid = pick.points(schema.read(self.results), *self.system_build())

        with self.assertRaises(SystemExit):
            pick.verdict(grid, [0.95], 3, 20, True)

    def system_build(self) -> tuple[driver.System, driver.Build]:
        system = pick.find_system(driver.load_suite(self.suite), "coldivf", "")
        return system, pick.find_build(system, "tl")


if __name__ == "__main__":
    unittest.main()
