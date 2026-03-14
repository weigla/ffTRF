#!/usr/bin/env python3
"""Benchmark ``FrequencyTRF`` against a standard time-domain ``mTRFpy`` fit.

The benchmark is intentionally simple and reproducible:

- simulate continuous stimulus/response pairs from a known kernel
- fit ``fft_trf.FrequencyTRF`` with a fixed ridge value
- fit ``mTRFpy`` with the same lag window and regularization
- report median training time and per-fit peak memory across repeated runs

The resulting Markdown report is suitable for inclusion in project
documentation or manuscripts.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata
import json
from pathlib import Path
from statistics import median
from subprocess import DEVNULL, CalledProcessError, check_output
from time import perf_counter
import platform
import sys
from typing import Sequence

import numpy as np

from fft_trf import FrequencyTRF

EXAMPLES_DIR = Path(__file__).resolve().parent
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from comparison_utils import default_kernel

_MTRF_TRF = None


@dataclass(slots=True)
class BenchmarkScenario:
    """Configuration for one runtime benchmark."""

    name: str
    fs: float
    n_trials: int
    n_samples: int
    tmin: float
    tmax: float
    regularization: float | Sequence[float]
    n_features: int = 1
    n_outputs: int = 1
    noise_scale: float = 0.05
    seed: int = 0
    segment_length: int | None = None
    overlap: float = 0.0
    window: str | None = None
    k: int = -1

    @property
    def n_lags(self) -> int:
        return int(round((self.tmax - self.tmin) * self.fs))

    @property
    def lag_matrix_mebibytes(self) -> float:
        n_elements = self.n_trials * self.n_samples * self.n_lags * self.n_features
        return n_elements * np.dtype(np.float64).itemsize / (1024.0**2)

    @property
    def regularization_values(self) -> tuple[float, ...]:
        if np.isscalar(self.regularization):
            return (float(self.regularization),)
        return tuple(float(value) for value in self.regularization)

    @property
    def fit_mode(self) -> str:
        if len(self.regularization_values) == 1:
            return "fixed"
        folds = "loo" if self.k == -1 else str(self.k)
        return f"cv-{len(self.regularization_values)} (k={folds})"

    @property
    def fft_setting(self) -> str:
        if self.segment_length is None:
            return "whole-trial"
        window_label = self.window if self.window is not None else "rect"
        return f"seg={self.segment_length}, ov={self.overlap:.1f}, {window_label}"

    @property
    def shape_label(self) -> str:
        return f"{self.n_features}->{self.n_outputs}"


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
    parser.add_argument(
        "--worker-scenario-index",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--worker-method",
        choices=("fft", "mtrf"),
        default=None,
        help=argparse.SUPPRESS,
    )
    return parser


def current_process_peak_memory_mib() -> float:
    """Return peak resident memory for the current process in MiB when possible."""

    try:
        import resource
    except ModuleNotFoundError:
        return float("nan")

    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return float(peak_rss) / (1024.0**2)
    return float(peak_rss) / 1024.0


def benchmark_worker(
    scenario_index: int,
    *,
    method: str,
    repeats: int,
    warmup: int,
) -> tuple[list[float], list[float]]:
    """Run isolated worker processes and return durations and peak memory."""

    script_path = Path(__file__).resolve()
    durations = []
    peak_memories = []
    for run_index in range(repeats + warmup):
        output = check_output(
            [
                sys.executable,
                str(script_path),
                "--worker-scenario-index",
                str(scenario_index),
                "--worker-method",
                method,
            ],
            text=True,
        )
        payload = json.loads(output.strip().splitlines()[-1])
        if run_index >= warmup:
            durations.append(float(payload["duration_seconds"]))
            peak_memories.append(float(payload["peak_memory_mib"]))
    return durations, peak_memories


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
        regularization=list(scenario.regularization_values)
        if len(scenario.regularization_values) > 1
        else scenario.regularization_values[0],
        segment_length=scenario.segment_length,
        overlap=scenario.overlap,
        window=scenario.window,
        k=scenario.k,
    )
    return model


def fit_mtrf(
    stimulus: list[np.ndarray],
    response: list[np.ndarray],
    scenario: BenchmarkScenario,
):
    """Fit ``mTRFpy`` for one scenario."""

    TRF = get_mtrf_class()

    model = TRF(direction=1)
    model.train(
        stimulus=stimulus,
        response=response,
        fs=scenario.fs,
        tmin=scenario.tmin,
        tmax=scenario.tmax - (1.0 / scenario.fs),
        regularization=list(scenario.regularization_values)
        if len(scenario.regularization_values) > 1
        else scenario.regularization_values[0],
        k=scenario.k,
        verbose=False,
    )
    return model


def get_mtrf_class():
    """Import and cache the ``mTRFpy`` estimator class."""

    global _MTRF_TRF
    if _MTRF_TRF is None:
        from mtrf.model import TRF

        _MTRF_TRF = TRF
    return _MTRF_TRF


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
            name="Moderate length",
            fs=1_000.0,
            n_trials=8,
            n_samples=4_096,
            tmin=0.0,
            tmax=0.040,
            regularization=1e-3,
            seed=11,
        ),
        BenchmarkScenario(
            name="Long recording",
            fs=1_000.0,
            n_trials=4,
            n_samples=60_000,
            tmin=0.0,
            tmax=0.040,
            regularization=1e-3,
            seed=17,
        ),
        BenchmarkScenario(
            name="High rate",
            fs=10_000.0,
            n_trials=2,
            n_samples=30_000,
            tmin=0.0,
            tmax=0.030,
            regularization=1e-3,
            seed=23,
        ),
        BenchmarkScenario(
            name="Long high rate",
            fs=10_000.0,
            n_trials=2,
            n_samples=60_000,
            tmin=0.0,
            tmax=0.030,
            regularization=1e-3,
            seed=29,
        ),
        BenchmarkScenario(
            name="Multifeature / multichannel",
            fs=1_000.0,
            n_trials=6,
            n_samples=4_096,
            tmin=0.0,
            tmax=0.040,
            regularization=1e-3,
            n_features=3,
            n_outputs=2,
            seed=31,
        ),
        BenchmarkScenario(
            name="Longer lag window",
            fs=10_000.0,
            n_trials=2,
            n_samples=30_000,
            tmin=0.0,
            tmax=0.060,
            regularization=1e-3,
            seed=37,
        ),
        BenchmarkScenario(
            name="Cross-validated ridge",
            fs=10_000.0,
            n_trials=4,
            n_samples=30_000,
            tmin=0.0,
            tmax=0.030,
            regularization=np.logspace(-6, 1, 8),
            seed=41,
            k=4,
        ),
        BenchmarkScenario(
            name="Segmented Hann estimate",
            fs=10_000.0,
            n_trials=2,
            n_samples=60_000,
            tmin=0.0,
            tmax=0.030,
            regularization=1e-3,
            seed=43,
            segment_length=4_096,
            overlap=0.5,
            window="hann",
        ),
    ]


def build_kernel_bank(scenario: BenchmarkScenario) -> np.ndarray:
    """Create a deterministic kernel bank for one benchmark scenario."""

    base_kernel = default_kernel(fs=scenario.fs, tmin=scenario.tmin, tmax=scenario.tmax)
    kernels = np.zeros(
        (scenario.n_features, base_kernel.shape[0], scenario.n_outputs),
        dtype=float,
    )

    for input_index in range(scenario.n_features):
        for output_index in range(scenario.n_outputs):
            shift = int(round((0.0015 * scenario.fs) * (input_index + output_index)))
            shifted = np.roll(base_kernel, shift)
            if shift > 0:
                shifted[:shift] = 0.0
            scale = 1.0 / (1.0 + 0.30 * input_index + 0.20 * output_index)
            sign = -1.0 if (input_index + output_index) % 2 else 1.0
            kernels[input_index, :, output_index] = sign * scale * shifted
    return kernels


def shifted_convolution(
    signal_in: np.ndarray,
    kernel: np.ndarray,
    *,
    lag_start: int,
    out_length: int,
) -> np.ndarray:
    """Convolve one predictor with one kernel while respecting lag origin."""

    full = np.convolve(signal_in, kernel, mode="full")
    offset = -lag_start

    prediction = np.zeros(out_length, dtype=float)
    src_start = max(offset, 0)
    dst_start = max(-offset, 0)
    length = min(full.shape[0] - src_start, out_length - dst_start)
    if length > 0:
        prediction[dst_start : dst_start + length] = full[src_start : src_start + length]
    return prediction


def simulate_multivariate_trials(
    scenario: BenchmarkScenario,
    kernel_bank: np.ndarray,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Simulate multifeature / multichannel trials for one scenario."""

    rng = np.random.default_rng(scenario.seed)
    lag_start = int(round(scenario.tmin * scenario.fs))

    stimulus = []
    response = []
    for _ in range(scenario.n_trials):
        x = rng.standard_normal((scenario.n_samples, scenario.n_features))
        y = np.zeros((scenario.n_samples, scenario.n_outputs), dtype=float)
        for input_index in range(scenario.n_features):
            for output_index in range(scenario.n_outputs):
                y[:, output_index] += shifted_convolution(
                    x[:, input_index],
                    kernel_bank[input_index, :, output_index],
                    lag_start=lag_start,
                    out_length=scenario.n_samples,
                )
        y += scenario.noise_scale * rng.standard_normal(y.shape)
        stimulus.append(x)
        response.append(y)
    return stimulus, response


def run_worker_once(
    scenario_index: int,
    *,
    method: str,
) -> dict[str, float]:
    """Run one fit in the current process and report timing and peak memory."""

    scenario = default_scenarios()[scenario_index]
    kernel_bank = build_kernel_bank(scenario)
    stimulus, response = simulate_multivariate_trials(scenario, kernel_bank)

    if method == "mtrf":
        get_mtrf_class()

    start = perf_counter()
    if method == "fft":
        fit_frequency_trf(stimulus, response, scenario)
    else:
        fit_mtrf(stimulus, response, scenario)
    duration = perf_counter() - start
    return {
        "duration_seconds": duration,
        "peak_memory_mib": current_process_peak_memory_mib(),
    }


def run_scenario(
    scenario: BenchmarkScenario,
    *,
    scenario_index: int,
    repeats: int,
    warmup: int,
) -> dict[str, float | str | int]:
    """Run one scenario and return a summary row."""

    kernel_bank = build_kernel_bank(scenario)
    stimulus, response = simulate_multivariate_trials(scenario, kernel_bank)

    fft_durations, fft_peak_memories = benchmark_worker(
        scenario_index,
        method="fft",
        repeats=repeats,
        warmup=warmup,
    )
    mtrf_durations, mtrf_peak_memories = benchmark_worker(
        scenario_index,
        method="mtrf",
        repeats=repeats,
        warmup=warmup,
    )

    fft_model = fit_frequency_trf(stimulus, response, scenario)
    mtrf_model = fit_mtrf(stimulus, response, scenario)
    mtrf_kernel = np.asarray(mtrf_model.weights, dtype=float) / scenario.fs

    return {
        "name": scenario.name,
        "shape": scenario.shape_label,
        "fit_mode": scenario.fit_mode,
        "fft_setting": scenario.fft_setting,
        "fs_hz": int(scenario.fs),
        "n_trials": scenario.n_trials,
        "n_samples": scenario.n_samples,
        "n_lags": scenario.n_lags,
        "lag_matrix_mib": scenario.lag_matrix_mebibytes,
        "fft_seconds": median(fft_durations),
        "fft_peak_mib": median(fft_peak_memories),
        "mtrf_seconds": median(mtrf_durations),
        "mtrf_peak_mib": median(mtrf_peak_memories),
        "speedup": median(mtrf_durations) / median(fft_durations),
        "kernel_corr": safe_corrcoef(fft_model.weights, mtrf_kernel),
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
        "- Peak memory: median per-fit peak RSS measured in isolated worker processes",
        "",
        "All scenarios use forward regression on the same simulated data for both",
        "methods. Fixed-ridge scenarios use the same lambda in both toolboxes, and",
        "the cross-validated scenario uses the same candidate grid in both. Kernel",
        "correlation is computed over the flattened full kernel bank.",
        "",
        "| Scenario | Shape | Fit mode | FFT setting | fs (Hz) | Trials | Samples/trial | Lags | Lag matrix size (MiB) | FrequencyTRF median fit (s) | FrequencyTRF peak RSS (MiB) | mTRFpy median fit (s) | mTRFpy peak RSS (MiB) | Speedup | Kernel corr. |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for row in rows:
        lines.append(
            "| "
            f"{row['name']} | "
            f"{row['shape']} | "
            f"{row['fit_mode']} | "
            f"{row['fft_setting']} | "
            f"{row['fs_hz']} | "
            f"{row['n_trials']} | "
            f"{row['n_samples']} | "
            f"{row['n_lags']} | "
            f"{row['lag_matrix_mib']:.1f} | "
            f"{row['fft_seconds']:.4f} | "
            f"{row['fft_peak_mib']:.1f} | "
            f"{row['mtrf_seconds']:.4f} | "
            f"{row['mtrf_peak_mib']:.1f} | "
            f"{row['speedup']:.2f}x | "
            f"{row['kernel_corr']:.4f} |"
        )

    lines.extend(
        [
            "",
            "Interpretation:",
            "- The approximate lag-matrix size is shown because it dominates the memory footprint of a standard time-domain fit and grows with both lag count and feature count.",
            "- Kernel correlations close to 1 indicate that the two methods recover nearly the same flattened kernel bank under the matched settings used here.",
            "- Cached spectra matter most in the cross-validated scenario because `FrequencyTRF` can reuse FFT work across lambda candidates, even if that does not automatically make it faster than `mTRFpy` on every machine.",
            "- The segmented Hann scenario is intentionally not the closest mTRF-like setting; it shows the cost of a more typical spectral-estimation workflow.",
            "- Peak RSS is measured per fit in a fresh worker process, so the reported memory is not inflated by earlier benchmark runs.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    """Run the benchmark and write the Markdown report."""

    parser = build_parser()
    args = parser.parse_args()

    if args.worker_method is not None:
        if args.worker_scenario_index is None:
            raise SystemExit("--worker-scenario-index is required with --worker-method.")
        if args.worker_method == "mtrf":
            try:
                metadata.version("mtrf")
            except metadata.PackageNotFoundError as exc:
                raise SystemExit(
                    "mTRFpy is required for the mTRF benchmark worker."
                ) from exc
        payload = run_worker_once(
            args.worker_scenario_index,
            method=args.worker_method,
        )
        sys.stdout.write(json.dumps(payload) + "\n")
        return

    try:
        metadata.version("mtrf")
    except metadata.PackageNotFoundError as exc:
        raise SystemExit(
            "mTRFpy is required for this benchmark. Install it with "
            '`pip install mtrf` or use `pixi run -e compare`.'
        ) from exc

    rows = [
        run_scenario(
            scenario,
            scenario_index=index,
            repeats=args.repeats,
            warmup=args.warmup,
        )
        for index, scenario in enumerate(default_scenarios())
    ]
    report = format_report(rows, repeats=args.repeats, warmup=args.warmup)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    sys.stdout.write(report)
    print(f"\nSaved report to {args.output}")


if __name__ == "__main__":
    main()
