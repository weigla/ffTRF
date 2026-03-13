#!/usr/bin/env python3
"""Benchmark ``FrequencyTRF`` against a standard time-domain ``mTRFpy`` fit.

The benchmark is intentionally simple and reproducible:

- simulate continuous stimulus/response pairs from a known kernel
- fit ``fft_trf.FrequencyTRF`` with a fixed ridge value
- fit ``mTRFpy`` with the same lag window and regularization
- report median training time across repeated runs

The resulting Markdown report is suitable for inclusion in project
documentation or manuscripts.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from statistics import median
from subprocess import DEVNULL, CalledProcessError, check_output
from time import perf_counter
import platform
import sys

import numpy as np

from fft_trf import FrequencyTRF

EXAMPLES_DIR = Path(__file__).resolve().parent
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from comparison_utils import default_kernel, simulate_trials


@dataclass(slots=True)
class BenchmarkScenario:
    """Configuration for one runtime benchmark."""

    name: str
    fs: float
    n_trials: int
    n_samples: int
    tmin: float
    tmax: float
    regularization: float
    noise_scale: float = 0.05
    seed: int = 0
    segment_length: int | None = None
    overlap: float = 0.0
    window: str | None = None

    @property
    def n_lags(self) -> int:
        return int(round((self.tmax - self.tmin) * self.fs))

    @property
    def lag_matrix_mebibytes(self) -> float:
        n_elements = self.n_trials * self.n_samples * self.n_lags
        return n_elements * np.dtype(np.float64).itemsize / (1024.0**2)


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Benchmark FrequencyTRF against mTRFpy and emit a Markdown summary."
        )
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of timed repetitions per scenario and method.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Number of untimed warmup runs per scenario and method.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/runtime_benchmark.md"),
        help="Markdown file where the benchmark report should be written.",
    )
    return parser


def benchmark_call(fn, *, repeats: int, warmup: int) -> list[float]:
    """Time a callable repeatedly and return per-run durations in seconds."""

    durations = []
    for index in range(repeats + warmup):
        start = perf_counter()
        fn()
        duration = perf_counter() - start
        if index >= warmup:
            durations.append(duration)
    return durations


def fit_frequency_trf(
    stimulus: list[np.ndarray],
    response: list[np.ndarray],
    scenario: BenchmarkScenario,
) -> FrequencyTRF:
    """Fit ``FrequencyTRF`` for one scenario."""

    model = FrequencyTRF(direction=1)
    model.train(
        stimulus=stimulus,
        response=response,
        fs=scenario.fs,
        tmin=scenario.tmin,
        tmax=scenario.tmax,
        regularization=scenario.regularization,
        segment_length=scenario.segment_length,
        overlap=scenario.overlap,
        window=scenario.window,
    )
    return model


def fit_mtrf(
    stimulus: list[np.ndarray],
    response: list[np.ndarray],
    scenario: BenchmarkScenario,
):
    """Fit ``mTRFpy`` for one scenario."""

    from mtrf.model import TRF

    model = TRF(direction=1)
    model.train(
        stimulus=stimulus,
        response=response,
        fs=scenario.fs,
        tmin=scenario.tmin,
        tmax=scenario.tmax - (1.0 / scenario.fs),
        regularization=scenario.regularization,
    )
    return model


def safe_corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    """Return a robust Pearson correlation for two 1D vectors."""

    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.shape != b.shape:
        raise ValueError("Vectors must have the same shape.")
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def cpu_name() -> str:
    """Return a concise CPU identifier when available."""

    for command in (
        ["sysctl", "-n", "machdep.cpu.brand_string"],
        ["bash", "-lc", "lscpu | awk -F: '/Model name/ {print $2}' | xargs"],
    ):
        try:
            output = check_output(command, stderr=DEVNULL, text=True).strip()
        except (CalledProcessError, FileNotFoundError):
            continue
        if output:
            return output
    processor = platform.processor().strip()
    if processor:
        return processor
    return platform.machine()


def default_scenarios() -> list[BenchmarkScenario]:
    """Return the benchmark scenarios used in the README."""

    return [
        BenchmarkScenario(
            name="Moderate length, 1 kHz",
            fs=1_000.0,
            n_trials=8,
            n_samples=4_096,
            tmin=0.0,
            tmax=0.040,
            regularization=1e-3,
            seed=11,
        ),
        BenchmarkScenario(
            name="Long recording, 1 kHz",
            fs=1_000.0,
            n_trials=4,
            n_samples=60_000,
            tmin=0.0,
            tmax=0.040,
            regularization=1e-3,
            seed=17,
        ),
        BenchmarkScenario(
            name="High rate, 10 kHz",
            fs=10_000.0,
            n_trials=2,
            n_samples=30_000,
            tmin=0.0,
            tmax=0.030,
            regularization=1e-3,
            seed=23,
        ),
        BenchmarkScenario(
            name="Long high rate, 10 kHz",
            fs=10_000.0,
            n_trials=2,
            n_samples=60_000,
            tmin=0.0,
            tmax=0.030,
            regularization=1e-3,
            seed=29,
        ),
    ]


def run_scenario(
    scenario: BenchmarkScenario,
    *,
    repeats: int,
    warmup: int,
) -> dict[str, float | str | int]:
    """Run one scenario and return a summary row."""

    kernel = default_kernel(fs=scenario.fs, tmin=scenario.tmin, tmax=scenario.tmax)
    stimulus, response = simulate_trials(
        fs=scenario.fs,
        n_trials=scenario.n_trials,
        n_samples=scenario.n_samples,
        tmin=scenario.tmin,
        kernel=kernel,
        noise_scale=scenario.noise_scale,
        seed=scenario.seed,
    )

    fft_durations = benchmark_call(
        lambda: fit_frequency_trf(stimulus, response, scenario),
        repeats=repeats,
        warmup=warmup,
    )
    mtrf_durations = benchmark_call(
        lambda: fit_mtrf(stimulus, response, scenario),
        repeats=repeats,
        warmup=warmup,
    )

    fft_model = fit_frequency_trf(stimulus, response, scenario)
    mtrf_model = fit_mtrf(stimulus, response, scenario)
    mtrf_kernel = np.asarray(mtrf_model.weights)[0, :, 0] / scenario.fs

    return {
        "name": scenario.name,
        "fs_hz": int(scenario.fs),
        "n_trials": scenario.n_trials,
        "n_samples": scenario.n_samples,
        "n_lags": scenario.n_lags,
        "lag_matrix_mib": scenario.lag_matrix_mebibytes,
        "fft_seconds": median(fft_durations),
        "mtrf_seconds": median(mtrf_durations),
        "speedup": median(mtrf_durations) / median(fft_durations),
        "kernel_corr": safe_corrcoef(fft_model.weights[0, :, 0], mtrf_kernel),
    }


def format_report(
    rows: list[dict[str, float | str | int]],
    *,
    repeats: int,
    warmup: int,
) -> str:
    """Render the benchmark results as Markdown."""

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    python_version = platform.python_version()
    numpy_version = metadata.version("numpy")
    scipy_version = metadata.version("scipy")
    mtrf_version = metadata.version("mtrf")

    lines = [
        "# Runtime benchmark",
        "",
        f"Generated: {timestamp}",
        "",
        "Environment:",
        f"- CPU: {cpu_name()}",
        f"- Platform: {platform.platform()}",
        f"- Python: {python_version}",
        f"- NumPy: {numpy_version}",
        f"- SciPy: {scipy_version}",
        f"- mTRFpy: {mtrf_version}",
        f"- Timed repetitions: {repeats}",
        f"- Warmup runs: {warmup}",
        "",
        "All scenarios use a forward single-feature / single-output regression with",
        "the same fixed ridge value for both methods. `FrequencyTRF` is run in the",
        "closest mTRF-like setting: `segment_length=None` and `window=None`.",
        "",
        "| Scenario | fs (Hz) | Trials | Samples/trial | Lags | Lag matrix size (MiB) | FrequencyTRF median fit (s) | mTRFpy median fit (s) | Speedup | Kernel corr. |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for row in rows:
        lines.append(
            "| "
            f"{row['name']} | "
            f"{row['fs_hz']} | "
            f"{row['n_trials']} | "
            f"{row['n_samples']} | "
            f"{row['n_lags']} | "
            f"{row['lag_matrix_mib']:.1f} | "
            f"{row['fft_seconds']:.4f} | "
            f"{row['mtrf_seconds']:.4f} | "
            f"{row['speedup']:.2f}x | "
            f"{row['kernel_corr']:.4f} |"
        )

    lines.extend(
        [
            "",
            "Interpretation:",
            "- The approximate lag-matrix size is shown because it dominates the memory footprint of a standard time-domain fit.",
            "- Kernel correlations close to 1 indicate that the two methods recover nearly the same filter under these settings.",
            "- The runtime gap grows as the number of samples and lag coefficients grows.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    """Run the benchmark and write the Markdown report."""

    try:
        metadata.version("mtrf")
    except metadata.PackageNotFoundError as exc:
        raise SystemExit(
            "mTRFpy is required for this benchmark. Install it with "
            '`pip install mtrf` or use `pixi run -e compare`.'
        ) from exc

    parser = build_parser()
    args = parser.parse_args()

    rows = [
        run_scenario(scenario, repeats=args.repeats, warmup=args.warmup)
        for scenario in default_scenarios()
    ]
    report = format_report(rows, repeats=args.repeats, warmup=args.warmup)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    sys.stdout.write(report)
    print(f"\nSaved report to {args.output}")


if __name__ == "__main__":
    main()
