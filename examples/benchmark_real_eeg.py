#!/usr/bin/env python3
"""Benchmark practical ffTRF settings against mTRF on real speech EEG."""

from __future__ import annotations

import argparse
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from statistics import median

from example_mtrf_sample_eeg import benchmark_worker_payloads

DEFAULT_OUTPUT = Path("artifacts/real_eeg_benchmark.md")
SEGMENT_DURATION_SECONDS = 2.0
OVERLAP = 0.5
WINDOW = "hann"
BACKWARD_STOP_SECONDS = 0.35
BACKWARD_REGULARIZATION_MIN = 1e-8
BACKWARD_REGULARIZATION_MAX = 1e6
BACKWARD_REGULARIZATION_COUNT = 15
BACKWARD_K_FOLDS = 3
BACKWARD_ENVELOPE_COMPRESSION = 0.4


@dataclass(slots=True, frozen=True)
class BenchmarkRow:
    """Aggregated measurements for one toolbox and model direction."""

    direction: str
    toolbox: str
    configuration: str
    regularization: float
    mean_correlation: float
    median_correlation: float
    duration_seconds: float
    peak_memory_mib: float


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Reproduce the practical 2 s Hann ffTRF versus mTRF real-EEG "
            "benchmark and write a Markdown report."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Markdown report path (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Number of measured isolated-process runs per table row.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="Number of additional unreported isolated warmup runs per row.",
    )
    parser.add_argument(
        "--direction",
        choices=("both", "forward", "backward"),
        default="both",
        help="Limit the report to one direction; the default reproduces all rows.",
    )
    return parser


def _package_version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "unknown"


def _selected_methods(direction: str) -> tuple[str, ...]:
    directions = ("forward", "backward") if direction == "both" else (direction,)
    return tuple(
        f"{toolbox}-{model_direction}"
        for model_direction in directions
        for toolbox in ("fftrf", "mtrf")
    )


def _aggregate(method: str, payloads: list[dict[str, float]]) -> BenchmarkRow:
    toolbox, direction = method.split("-", maxsplit=1)
    configuration = (
        "2 s / 50% overlap / Hann"
        if toolbox == "fftrf"
        else "finite-lag baseline"
    )
    return BenchmarkRow(
        direction=direction.capitalize(),
        toolbox="ffTRF" if toolbox == "fftrf" else "mTRF",
        configuration=configuration,
        regularization=median(item["regularization"] for item in payloads),
        mean_correlation=median(item["mean_correlation"] for item in payloads),
        median_correlation=median(item["median_correlation"] for item in payloads),
        duration_seconds=median(item["duration_seconds"] for item in payloads),
        peak_memory_mib=median(item["peak_memory_mib"] for item in payloads),
    )


def run_benchmark(*, repeats: int, warmup: int, direction: str) -> list[BenchmarkRow]:
    """Run each selected fit in isolated worker processes."""

    rows = []
    for method in _selected_methods(direction):
        toolbox, model_direction = method.split("-", maxsplit=1)
        print(f"Running {toolbox} {model_direction} benchmark...")
        payloads = benchmark_worker_payloads(
            method=method,
            repeats=repeats,
            warmup=warmup,
            forward_segment_duration=SEGMENT_DURATION_SECONDS,
            forward_overlap=OVERLAP,
            forward_window=WINDOW,
            backward_stop_seconds=BACKWARD_STOP_SECONDS,
            backward_regularization_min=BACKWARD_REGULARIZATION_MIN,
            backward_regularization_max=BACKWARD_REGULARIZATION_MAX,
            backward_regularization_count=BACKWARD_REGULARIZATION_COUNT,
            backward_k_folds=BACKWARD_K_FOLDS,
            backward_segment_duration=SEGMENT_DURATION_SECONDS,
            backward_overlap=OVERLAP,
            backward_window=WINDOW,
            backward_envelope_compression=BACKWARD_ENVELOPE_COMPRESSION,
        )
        rows.append(_aggregate(method, payloads))
    return rows


def render_markdown(
    rows: list[BenchmarkRow],
    *,
    repeats: int,
    warmup: int,
) -> str:
    """Render benchmark measurements and protocol details as Markdown."""

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Real EEG benchmark",
        "",
        "Practical 2 s Hann-windowed ffTRF fits compared with the finite-lag",
        "mTRF baseline on the public mTRF speech-EEG sample.",
        "",
        "## Environment",
        "",
        f"- Generated: {generated_at}",
        f"- Platform: {platform.platform()}",
        f"- Machine: {platform.machine()}",
        f"- Python: {platform.python_version()}",
        f"- ffTRF: {_package_version('fftrf')}",
        f"- mTRF: {_package_version('mtrf')}",
        f"- NumPy: {_package_version('numpy')}",
        f"- SciPy: {_package_version('scipy')}",
        "",
        "## Protocol",
        "",
        "- Command: `pixi run -e compare python examples/benchmark_real_eeg.py`",
        "- Data: 10 twelve-second segments at 128 Hz; 7 train/CV and 3 held out.",
        "- Forward model: 16 speech bands to 128 EEG channels, 5-fold CV.",
        "- Backward model: 128 EEG channels to a compressed broadband envelope, 3-fold CV.",
        "- Selection: seeded folds (`seed=7`) and negative MSE over the same lambda grid per direction.",
        "- Accuracy: held-out Pearson correlation, summarized across EEG channels (forward) or segments (backward).",
        f"- Timing and memory: medians of {repeats} isolated process run(s) per row; {warmup} warmup run(s).",
        "- Peak RSS is sampled immediately after CV fitting; prediction is excluded from runtime and RSS.",
        "",
        "## Results",
        "",
        "| Direction | Model | Configuration | Lambda | Mean held-out r | Median held-out r | CV fit (s) | Peak RSS (MiB) |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row.direction} | {row.toolbox} | {row.configuration} | "
            f"{row.regularization:.6g} | {row.mean_correlation:.4f} | "
            f"{row.median_correlation:.4f} | {row.duration_seconds:.4f} | "
            f"{row.peak_memory_mib:.1f} |"
        )
    lines.extend(
        [
            "",
            "The ffTRF and mTRF rows use identical data splits, lag samples, CV seeds,",
            "selection metrics, and direction-specific lambda grids. The 2 s Hann setting",
            "changes ffTRF's spectral estimator, so this is a practical performance",
            "comparison rather than a strict solver-equivalence test.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    """Run the benchmark and write its Markdown report."""

    parser = build_parser()
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")

    rows = run_benchmark(
        repeats=int(args.repeats),
        warmup=int(args.warmup),
        direction=str(args.direction),
    )
    report = render_markdown(
        rows,
        repeats=int(args.repeats),
        warmup=int(args.warmup),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
