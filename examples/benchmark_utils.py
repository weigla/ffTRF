"""Shared helpers for reproducible ffTRF benchmark reports."""

from __future__ import annotations

import math
import os
import platform
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from statistics import median
from typing import Any

THREAD_ENVIRONMENT_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def package_version(distribution: str) -> str:
    """Return an installed distribution version or ``"unknown"``."""

    try:
        return version(distribution)
    except PackageNotFoundError:
        return "unknown"


def isolated_worker_environment(*, threads: int) -> dict[str, str]:
    """Return an environment with deterministic native thread limits."""

    if threads < 1:
        raise ValueError("threads must be at least 1.")
    environment = os.environ.copy()
    for name in THREAD_ENVIRONMENT_VARIABLES:
        environment[name] = str(threads)
    environment["PYTHONHASHSEED"] = "0"
    return environment


def current_process_peak_memory_mib() -> float:
    """Return current-process peak RSS in MiB when supported."""

    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        get_current_process = ctypes.windll.kernel32.GetCurrentProcess
        get_current_process.restype = wintypes.HANDLE
        get_process_memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
        get_process_memory_info.argtypes = (
            wintypes.HANDLE,
            ctypes.POINTER(ProcessMemoryCounters),
            wintypes.DWORD,
        )
        get_process_memory_info.restype = wintypes.BOOL
        success = get_process_memory_info(
            get_current_process(),
            ctypes.byref(counters),
            counters.cb,
        )
        if not success:
            return float("nan")
        return float(counters.PeakWorkingSetSize) / (1024.0**2)

    try:
        import resource
    except ModuleNotFoundError:
        return float("nan")

    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return float(peak_rss) / (1024.0**2)
    return float(peak_rss) / 1024.0


def additional_peak_memory_mib(*, before: float, after: float) -> float:
    """Return peak RSS growth above the pre-fit process peak."""

    if not math.isfinite(before) or not math.isfinite(after):
        return float("nan")
    return max(after - before, 0.0)


def summarize(values: list[float]) -> dict[str, float]:
    """Return median, minimum, and maximum for non-empty measurements."""

    if not values:
        raise ValueError("Cannot summarize an empty measurement sequence.")
    return {
        "median": float(median(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def format_median_range(
    summary: dict[str, float],
    *,
    precision: int,
) -> str:
    """Format a median followed by the observed repeated-run range."""

    median_value = summary["median"]
    minimum = summary["min"]
    maximum = summary["max"]
    if not all(math.isfinite(value) for value in (median_value, minimum, maximum)):
        return "n/a"
    return f"{median_value:.{precision}f} [{minimum:.{precision}f}, {maximum:.{precision}f}]"


def cpu_name() -> str:
    """Return a concise CPU identifier without shell parsing."""

    commands: tuple[tuple[str, ...], ...]
    if sys.platform == "darwin":
        commands = (("sysctl", "-n", "machdep.cpu.brand_string"),)
    elif sys.platform.startswith("linux"):
        commands = (("lscpu",),)
    else:
        commands = ()

    for command in commands:
        try:
            output = subprocess.check_output(
                command,
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            continue
        if command[0] == "lscpu":
            for line in output.splitlines():
                if line.lower().startswith("model name:"):
                    return line.split(":", maxsplit=1)[1].strip()
        elif output:
            return output

    processor = platform.processor().strip()
    return processor or platform.machine()


def git_source_state(repo_root: Path) -> str:
    """Return the current Git revision plus clean/dirty state."""

    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{revision} ({'dirty' if status.strip() else 'clean'})"


def environment_metadata(*, repo_root: Path, threads: int) -> dict[str, Any]:
    """Return common benchmark provenance as JSON-safe values."""

    return {
        "cpu": cpu_name(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "fftrf": package_version("fftrf"),
        "mtrf": package_version("mtrf"),
        "numpy": package_version("numpy"),
        "scipy": package_version("scipy"),
        "source": git_source_state(repo_root),
        "native_threads_per_worker": threads,
        "thread_environment_variables": list(THREAD_ENVIRONMENT_VARIABLES),
    }


def replace_marked_section(
    path: Path,
    *,
    start_marker: str,
    end_marker: str,
    content: str,
) -> None:
    """Replace one generated Markdown section delimited by exact markers."""

    text = path.read_text(encoding="utf-8")
    if text.count(start_marker) != 1 or text.count(end_marker) != 1:
        raise ValueError(
            f"{path} must contain exactly one {start_marker!r} and one {end_marker!r}."
        )
    prefix, remainder = text.split(start_marker, maxsplit=1)
    _, suffix = remainder.split(end_marker, maxsplit=1)
    replacement = f"{start_marker}\n{content.rstrip()}\n{end_marker}"
    path.write_text(prefix + replacement + suffix, encoding="utf-8")
