import unittest

import writes_report as wr

HIGH = {"nprobe": 128, "rerank": 64}
LOW = {"nprobe": 64, "rerank": 39}


def line(kind: str, config: str = "tl320", status: str = "ok", **fields) -> dict:
    return {"kind": kind, "system": "coldivf", "variant": "python", "config_id": config, "status": status, **fields}


def warm(state: str, params: dict, recall: float, p50: float, config: str = "tl320", **fields) -> dict:
    return line("warm", config, state=state, split="test", threads=1, search_params=params, recall10=recall,
                p50_us=p50, **fields)


def cold(state: str, params: dict, total: float, config: str = "tl320") -> dict:
    return line("cold", config, state=state, split="test", regime="c0", search_params=params, total_us=total,
                open_us=total / 2, search_us=total / 2, recall10=1.0, disk_bytes=1000, footprint_delta=10)


class WritesReportTest(unittest.TestCase):
    def lines(self) -> list[dict]:
        found = "{'inserted': 1.0, 'base': 0.99, 'probes': {'inserted': 10, 'base': 10}, 'missed': []}"
        return [
            warm("s0", HIGH, 0.961, 350.0), warm("s0", LOW, 0.921, 210.0),
            warm("s10", HIGH, 0.958, 420.0, dead_returned=0.0, recall10_inserted=0.95,
                 extra={"findability": found}),
            warm("fresh", HIGH, 0.959, 365.0, config="tl320-fresh"),
            *[cold("s0", HIGH, t) for t in (8000.0, 9000.0, 10000.0)],
            *[cold("s10", HIGH, t) for t in (180000.0, 190000.0)],
            cold("fresh", HIGH, 7000.0, config="tl320-fresh"),
            line("cold", state="s10", split="test", regime="c0", search_params=HIGH, total_us=1.0, status="failed"),
            line("write", mode="latency", op="insert", state="s0", p50_us=64.0, p99_us=80.0, durability="process"),
            line("write", mode="latency", op="insert", state="s0", p50_us=99.0, p99_us=99.0, durability="process",
                 status="failed"),
            line("write", mode="stream", op="insert", state_from="s0", state_to="s1", rows=10000, rows_per_s=15000.0),
            line("write", mode="stream", op="insert", state_from="s5", state_to="s10p", rows=50000, rows_per_s=25000.0),
            line("write", mode="stream", op="delete", state_from="s5", state_to="s10p", rows=50000, rows_per_s=50000.0),
            line("maintain", action="fold", state_from="s10", state_to="s10-fold", seconds=1.5, threads=10),
            line("maintain", action="auto", state_from="s10p", state_to="s10p", seconds=3.0,
                 extra={"report": {"kind": "minor"}}),
            warm("s10-fold", HIGH, 0.961, 354.0), cold("s10-fold", HIGH, 13000.0),
        ]

    def test_a_pick_is_the_setting_of_least_s0_recall_that_reaches_the_target(self):
        lines = wr.Lines(self.lines())
        b = ("coldivf", "python", "tl320")
        self.assertEqual(lines.builds(), [b])
        self.assertEqual(lines.picks(b), {0.95: HIGH, 0.90: LOW})
        # The fresh build's lines join the build it was named after.
        self.assertEqual(lines.warm(b, "fresh", HIGH)["p50_us"], 365.0)

    def test_the_tables_take_ok_lines_and_each_state_s_trials(self):
        lines = wr.Lines(self.lines())
        row = wr.search_rows(lines, 0.95)[0]
        self.assertEqual(row["cold"]["s0"]["p50_us"], 9000.0)
        self.assertEqual(row["cold"]["s10"]["trials"], 2)
        self.assertEqual(row["cold"]["fresh"]["p50_us"], 7000.0)
        self.assertEqual(list(row["cold"]), ["s0", "s10", "s10-fold", "fresh"])
        self.assertEqual(wr.write_rows(lines)[0]["latency"]["s0"]["insert"]["p50_us"], 64.0)

        m = wr.maintenance_rows(lines)
        self.assertEqual([r["action"] for r in m], ["fold"])
        self.assertEqual(m[0]["cells"]["s10-fold"]["cold"]["p50_us"], 13000.0)

        # 100,000 rows in 2 + 1 s of writes and 3 s of runs.
        p = wr.policy_rows(lines)[0]
        self.assertEqual((p["runs"], p["kinds"], p["rows_per_s"], p["rows_per_s_with_runs"]),
                         (1, ["minor"], 100000 / 3, 100000 / 6))

        c = wr.check_rows(lines)[0]
        self.assertEqual(c["dead_returned_max"], 0.0)
        self.assertEqual(c["findability"]["s10"], {"inserted": 1.0, "base": 0.99, "recall10_inserted": 0.95})

    def test_a_system_with_two_builds_picks_the_cheaper_build_for_each_target(self):
        lines = wr.Lines([warm("s0", {"nprobe": 48}, 0.960, 28000.0, config="b8"),
                          warm("s0", {"nprobe": 48}, 0.917, 13000.0, config="b4")])
        picks = lines.system_picks(("coldivf", "python"))
        self.assertEqual((picks[0.95][0][2], picks[0.90][0][2]), ("b8", "b4"))
        self.assertEqual([r["build"][2] for r in wr.search_rows(lines, 0.90)], ["b4"])

    def test_the_drift_ladder_takes_the_place_of_the_uniform_one(self):
        lines = wr.Lines([warm("s0", HIGH, 0.961, 350.0), *[cold(s, HIGH, 9000.0) for s in ("s0", "d1", "d10")],
                          cold("fresh", HIGH, 7000.0, config="tl320-fresh"), cold("d10-fold", HIGH, 12000.0),
                          line("maintain", action="fold", state_from="d10", state_to="d10-fold", seconds=1.0),
                          line("write", mode="stream", op="insert", state_from="s0", state_to="d1", rows_per_s=9.0)])
        rows = wr.search_rows(lines, 0.95)
        self.assertEqual(wr.columns(rows), ["s0", "d1", "d10", "fresh"])
        self.assertIn("D10 Against S0", "\n".join(wr.search_tables(rows, "0.95")))
        self.assertEqual(wr.maintenance_rows(lines)[0]["cells"]["d10-fold"]["cold"]["p50_us"], 12000.0)
        self.assertEqual(wr.write_rows(lines)[0]["stream"]["insert"]["rows_per_s"], 9.0)

    def test_the_markdown_has_a_row_for_each_build(self):
        text = wr.markdown(wr.tables(wr.Lines(self.lines())))
        self.assertIn("| ColdIVF | process | 64 µs | 80 µs |", text)
        self.assertIn("| ColdIVF | nprobe 128, rerank 64 | 9.00 (0.961) |", text)
        self.assertIn("| ColdIVF | fold | 10 | 1.5 s |", text)


if __name__ == "__main__":
    unittest.main()
