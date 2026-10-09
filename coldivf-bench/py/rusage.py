"""Process and machine counters on macOS.

`Usage.now()` reads proc_pid_rusage's version 6 record through ctypes. It holds CPU time, the part of it on the
performance cores (P-cores), cycles, energy, page-ins, disk bytes, the physical footprint and its lifetime peak, and the
CPU time spent at each quality of service (QoS) class. macOS adds to the energy count only at a context switch, so
`Usage.settled()` sleeps 1 ms first, which forces one. `Usage.at_exit(pid)` reads a child's record once it has exited
and before it is reaped. `vm_counts()` reads the machine's swap-in and compression counts. Elsewhere than macOS every
counter reads 0.
"""

from __future__ import annotations

import ctypes
import errno
import os
import sys
import time
from dataclasses import dataclass, field, fields

MACOS = sys.platform == "darwin"

RUSAGE_INFO_V6 = 6

# The 64-bit counters of struct rusage_info_v6 after its 16-byte uuid, in the order of <sys/resource.h>.
V6_FIELDS = (
    "user_time", "system_time", "pkg_idle_wkups", "interrupt_wkups", "pageins", "wired_size", "resident_size",
    "phys_footprint", "proc_start_abstime", "proc_exit_abstime", "child_user_time", "child_system_time",
    "child_pkg_idle_wkups", "child_interrupt_wkups", "child_pageins", "child_elapsed_abstime", "diskio_bytesread",
    "diskio_byteswritten", "cpu_time_qos_default", "cpu_time_qos_maintenance", "cpu_time_qos_background",
    "cpu_time_qos_utility", "cpu_time_qos_legacy", "cpu_time_qos_user_initiated", "cpu_time_qos_user_interactive",
    "billed_system_time", "serviced_system_time", "logical_writes", "lifetime_max_phys_footprint", "instructions",
    "cycles", "billed_energy", "serviced_energy", "interval_max_phys_footprint", "runnable_time", "flags",
    "user_ptime", "system_ptime", "pinstructions", "pcycles", "energy_nj", "penergy_nj", "secure_time_in_system",
    "secure_ptime_in_system", "neural_footprint", "lifetime_max_neural_footprint", "interval_max_neural_footprint",
    "conclave_footprint", "page_wait_time_mach", "page_cache_hits",
)

QOS_CLASSES = ("default", "maintenance", "background", "utility", "legacy", "user_initiated", "user_interactive")

QOS_NAMES = {
    0x21: "user-interactive", 0x19: "user-initiated", 0x15: "default", 0x11: "utility", 0x09: "background",
    0x00: "unspecified",
}
QOS_CLASS_USER_INTERACTIVE = 0x21


class _RusageInfoV6(ctypes.Structure):
    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [(name, ctypes.c_uint64) for name in V6_FIELDS] + [
        ("reserved", ctypes.c_uint64 * 6)
    ]


assert ctypes.sizeof(_RusageInfoV6) == 16 + 56 * 8


class _Timebase(ctypes.Structure):
    _fields_ = [("numer", ctypes.c_uint32), ("denom", ctypes.c_uint32)]


_libc = ctypes.CDLL(None, use_errno=True) if MACOS else None
_libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True) if MACOS else None

if MACOS:
    _libproc.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    _libproc.proc_pid_rusage.restype = ctypes.c_int
    _libc.mach_timebase_info.argtypes = [ctypes.POINTER(_Timebase)]
    _libc.pthread_set_qos_class_self_np.argtypes = [ctypes.c_uint, ctypes.c_int]
    _libc.pthread_set_qos_class_self_np.restype = ctypes.c_int
    _libc.qos_class_self.restype = ctypes.c_uint
    _libc.mach_host_self.restype = ctypes.c_uint
    _libc.host_statistics64.argtypes = [ctypes.c_uint, ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint)]
    _libc.host_statistics64.restype = ctypes.c_int
    _libc.waitid.argtypes = [ctypes.c_int, ctypes.c_uint, ctypes.c_void_p, ctypes.c_int]
    _libc.waitid.restype = ctypes.c_int

# Room for waitid's siginfo_t, which is 104 bytes on macOS.
_SIGINFO_BYTES = 256


def _timebase() -> tuple[int, int]:
    if not MACOS:
        return 1, 1

    tb = _Timebase()
    _libc.mach_timebase_info(ctypes.byref(tb))
    return (tb.numer, tb.denom) if tb.denom else (1, 1)


_NUMER, _DENOM = _timebase()


def ticks_to_ns(ticks: int) -> int:
    """Mach time units to nanoseconds. Apple silicon's unit is 125/3 ns, Intel's 1 ns."""
    return ticks * _NUMER // _DENOM


def rusage_v6(pid: int | None = None) -> dict[str, int] | None:
    """The raw version 6 record of process `pid`, this one by default, or None if it can't be read. A child that has
    exited but isn't reaped yet still has its record."""
    if not MACOS:
        return None

    info = _RusageInfoV6()

    if _libproc.proc_pid_rusage(os.getpid() if pid is None else pid, RUSAGE_INFO_V6, ctypes.byref(info)) != 0:
        return None

    return {name: int(getattr(info, name)) for name in V6_FIELDS}


@dataclass
class Usage:
    """A process's counters at one moment. Times are in nanoseconds and sizes in bytes."""

    user_ns: int = 0
    system_ns: int = 0
    p_user_ns: int = 0
    p_system_ns: int = 0
    cycles: int = 0
    p_cycles: int = 0
    instructions: int = 0
    energy_nj: int = 0
    p_energy_nj: int = 0
    pageins: int = 0
    bytes_read: int = 0
    bytes_written: int = 0
    footprint: int = 0
    lifetime_max_footprint: int = 0
    runnable_ns: int = 0
    page_wait_ns: int = 0
    page_cache_hits: int = 0
    qos_ns: dict[str, int] = field(default_factory=lambda: dict.fromkeys(QOS_CLASSES, 0))

    @classmethod
    def now(cls, pid: int | None = None) -> Usage:
        """The counters of process `pid`, this one by default. They are all 0 when they can't be read."""
        raw = rusage_v6(pid)

        if raw is None:
            return cls()

        ns = ticks_to_ns
        return cls(
            user_ns=ns(raw["user_time"]),
            system_ns=ns(raw["system_time"]),
            p_user_ns=ns(raw["user_ptime"]),
            p_system_ns=ns(raw["system_ptime"]),
            cycles=raw["cycles"],
            p_cycles=raw["pcycles"],
            instructions=raw["instructions"],
            energy_nj=raw["energy_nj"],
            p_energy_nj=raw["penergy_nj"],
            pageins=raw["pageins"],
            bytes_read=raw["diskio_bytesread"],
            bytes_written=raw["diskio_byteswritten"],
            footprint=raw["phys_footprint"],
            lifetime_max_footprint=raw["lifetime_max_phys_footprint"],
            runnable_ns=ns(raw["runnable_time"]),
            page_wait_ns=ns(raw["page_wait_time_mach"]),
            page_cache_hits=raw["page_cache_hits"],
            qos_ns={c: ns(raw[f"cpu_time_qos_{c}"]) for c in QOS_CLASSES},
        )

    @classmethod
    def settled(cls, pid: int | None = None) -> Usage:
        """The counters after a 1 ms sleep. The sleep is a context switch, which brings the energy count up to date."""
        time.sleep(0.001)
        return cls.now(pid)

    @classmethod
    def at_exit(cls, pid: int) -> Usage:
        """Waits for the child `pid` to exit without reaping it, then returns its counters as it left them. An exited
        child keeps its record until it is reaped, so this reads its CPU time and lifetime peak footprint, and the
        caller reaps it afterwards, eg. with Popen.wait. Elsewhere than macOS it only waits."""
        if not MACOS:
            os.waitpid(pid, 0)
            return cls()

        info = (ctypes.c_uint8 * _SIGINFO_BYTES)()

        while _libc.waitid(os.P_PID, pid, info, os.WEXITED | os.WNOWAIT) != 0:
            if ctypes.get_errno() != errno.EINTR:
                raise OSError(ctypes.get_errno(), f"waiting for process {pid}: {os.strerror(ctypes.get_errno())}")

        return cls.now(pid)

    @property
    def cpu_ns(self) -> int:
        return self.user_ns + self.system_ns

    def __sub__(self, before: Usage) -> Delta:
        sub = lambda a, b: max(0, a - b)
        return Delta(
            cpu_ns=sub(self.cpu_ns, before.cpu_ns),
            p_cpu_ns=sub(self.p_user_ns + self.p_system_ns, before.p_user_ns + before.p_system_ns),
            cycles=sub(self.cycles, before.cycles),
            p_cycles=sub(self.p_cycles, before.p_cycles),
            instructions=sub(self.instructions, before.instructions),
            energy_nj=sub(self.energy_nj, before.energy_nj),
            pageins=sub(self.pageins, before.pageins),
            bytes_read=sub(self.bytes_read, before.bytes_read),
            bytes_written=sub(self.bytes_written, before.bytes_written),
            footprint=self.footprint - before.footprint,
            runnable_ns=sub(self.runnable_ns, before.runnable_ns),
            page_wait_ns=sub(self.page_wait_ns, before.page_wait_ns),
            page_cache_hits=sub(self.page_cache_hits, before.page_cache_hits),
            qos_ns={c: sub(self.qos_ns.get(c, 0), before.qos_ns.get(c, 0)) for c in QOS_CLASSES},
        )


@dataclass
class Delta:
    """What a process used between two readings. The footprint stays a signed difference."""

    cpu_ns: int = 0
    p_cpu_ns: int = 0
    cycles: int = 0
    p_cycles: int = 0
    instructions: int = 0
    energy_nj: int = 0
    pageins: int = 0
    bytes_read: int = 0
    bytes_written: int = 0
    footprint: int = 0
    runnable_ns: int = 0
    page_wait_ns: int = 0
    page_cache_hits: int = 0
    qos_ns: dict[str, int] = field(default_factory=lambda: dict.fromkeys(QOS_CLASSES, 0))

    def __add__(self, other: Delta) -> Delta:
        out = Delta()

        for f in fields(Delta):
            if f.name == "qos_ns":
                out.qos_ns = {c: self.qos_ns.get(c, 0) + other.qos_ns.get(c, 0) for c in QOS_CLASSES}
            else:
                setattr(out, f.name, getattr(self, f.name) + getattr(other, f.name))

        return out

    @property
    def cpu_ms(self) -> float:
        return self.cpu_ns / 1e6

    @property
    def qos_ms(self) -> dict[str, float]:
        return {c: v / 1e6 for c, v in self.qos_ns.items()}

    @property
    def p_share(self) -> float | None:
        """The share of the CPU time on P-cores, or None without CPU time."""
        return self.p_cpu_ns / self.cpu_ns if self.cpu_ns else None

    @property
    def ghz(self) -> float | None:
        """Cycles per nanosecond of CPU time, i.e. the clock the work ran at."""
        return self.cycles / self.cpu_ns if self.cpu_ns else None


def set_interactive() -> bool:
    """Moves the calling thread to the user-interactive QoS class, the class of work a user waits on."""
    if not MACOS:
        return False

    return _libc.pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0) == 0


def qos_name() -> str:
    """The calling thread's QoS class, eg. "user-interactive", or "" elsewhere than macOS."""
    if not MACOS:
        return ""

    return QOS_NAMES.get(int(_libc.qos_class_self()), "other")


# host_statistics64's flavor for struct vm_statistics64, and the byte offsets of its counters used here.
_HOST_VM_INFO64 = 4
_VM_OFFSETS = {
    "pageins": 32, "pageouts": 40, "decompressions": 96, "compressions": 104, "swapins": 112, "swapouts": 120,
}
_host = _libc.mach_host_self() if MACOS else 0


def vm_counts() -> dict[str, int]:
    """The machine's lifetime counts of page-ins, page-outs, compressions, decompressions, swap-ins and swap-outs, as
    vm_stat reports them. They are all 0 when they can't be read."""
    if not MACOS:
        return dict.fromkeys(_VM_OFFSETS, 0)

    buf = (ctypes.c_uint8 * 1024)()
    count = ctypes.c_uint(len(buf) // 4)

    if _libc.host_statistics64(_host, _HOST_VM_INFO64, buf, ctypes.byref(count)) != 0 or count.value * 4 < 128:
        return dict.fromkeys(_VM_OFFSETS, 0)

    raw = bytes(buf)
    return {name: int.from_bytes(raw[at : at + 8], "little") for name, at in _VM_OFFSETS.items()}


def vm_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    return {k: max(0, after.get(k, 0) - before.get(k, 0)) for k in _VM_OFFSETS}
