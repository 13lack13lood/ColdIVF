import subprocess
import sys
import threading
import time
import unittest

import rusage


def busy(seconds: float) -> None:
    end = time.perf_counter() + seconds

    while time.perf_counter() < end:
        pass


@unittest.skipUnless(rusage.MACOS, "the counters are read on macOS")
class UsageTest(unittest.TestCase):
    def test_counters_move_with_work(self):
        before = rusage.Usage.settled()
        busy(0.1)
        used = rusage.Usage.settled() - before
        self.assertGreater(used.cpu_ms, 80)
        self.assertLess(used.cpu_ms, 400)
        self.assertGreater(used.cycles, 0)
        self.assertGreater(used.energy_nj, 0)
        self.assertGreater(used.ghz, 0.5)
        self.assertTrue(0 <= used.p_share <= 1.01)

    def test_qos_times_add_up_to_cpu_time(self):
        before = rusage.Usage.settled()
        busy(0.05)
        used = rusage.Usage.settled() - before
        self.assertAlmostEqual(sum(used.qos_ms.values()), used.cpu_ms, delta=0.1 * used.cpu_ms + 2)

    def test_background_work_lands_in_its_class(self):
        # The thread spins until it has used 50 ms of CPU, however long that takes. A background thread gets the CPU
        # last, so on a loaded machine 50 ms of wall time can hold much less CPU time than the bound below.
        def work():
            rusage._libc.pthread_set_qos_class_self_np(0x09, 0)
            end = time.thread_time() + 0.05

            while time.thread_time() < end:
                pass

        before = rusage.Usage.settled()
        t = threading.Thread(target=work)
        t.start()
        t.join()
        used = rusage.Usage.settled() - before
        self.assertGreater(used.qos_ms["background"], 30)

    def test_footprint_and_peak(self):
        now = rusage.Usage.now()
        self.assertGreater(now.footprint, 1 << 20)
        self.assertGreaterEqual(now.lifetime_max_footprint, now.footprint)
        block = bytearray(64 << 20)
        block[:: 1 << 14] = b"x" * len(block[:: 1 << 14])
        grown = rusage.Usage.now()
        self.assertGreater(grown.footprint - now.footprint, 32 << 20)
        del block

    def test_another_process(self):
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(5)"])

        try:
            time.sleep(0.3)
            other = rusage.Usage.now(child.pid)
            self.assertGreater(other.footprint, 0)
        finally:
            child.kill()
            child.wait()

    def test_missing_process_reads_zero(self):
        self.assertEqual(rusage.Usage.now(2 ** 22 + 12345).footprint, 0)

    def test_a_child_at_exit(self):
        code = "b = bytearray(96 << 20)\nb[:: 1 << 14] = b'x' * len(b[:: 1 << 14])"
        child = subprocess.Popen([sys.executable, "-c", code])
        used = rusage.Usage.at_exit(child.pid)
        # The child is gone but not yet reaped, so its record holds its peak and its CPU time.
        self.assertGreater(used.lifetime_max_footprint, 96 << 20)
        self.assertGreater(used.cpu_ns, 0)
        self.assertEqual(child.wait(), 0)

    def test_delta_add(self):
        a = rusage.Delta(cpu_ns=1, energy_nj=2, qos_ns={"legacy": 1})
        b = rusage.Delta(cpu_ns=3, energy_nj=4, qos_ns={"legacy": 2})
        c = a + b
        self.assertEqual((c.cpu_ns, c.energy_nj, c.qos_ns["legacy"]), (4, 6, 3))
        self.assertIsNone(rusage.Delta().p_share)
        self.assertIsNone(rusage.Delta().ghz)

    def test_qos(self):
        self.assertTrue(rusage.set_interactive())
        self.assertEqual(rusage.qos_name(), "user-interactive")

    def test_vm_counts_match_vm_stat(self):
        counts = rusage.vm_counts()
        out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
        stat = {line.split(":")[0]: int(line.split(":")[1].strip(" .")) for line in out.splitlines()[1:]
                if ":" in line and line.split(":")[1].strip(" .").isdigit()}
        self.assertAlmostEqual(counts["swapouts"], stat["Swapouts"], delta=10000)
        self.assertAlmostEqual(counts["compressions"], stat["Compressions"], delta=200000)
        self.assertLessEqual(counts["swapins"], rusage.vm_counts()["swapins"])
        self.assertEqual(rusage.vm_delta({"swapins": 5}, {"swapins": 7})["swapins"], 2)


if __name__ == "__main__":
    unittest.main()
