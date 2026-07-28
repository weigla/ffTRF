#!/usr/bin/env python3
"""Benchmark matched and practical ffTRF settings on real speech EEG."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from benchmark_utils import (
    environment_metadata,
    format_median_range,
    replace_marked_section,
    summarize,
)
from compare_real_eeg_with_mtrf import benchmark_worker_payloads
from mtrf_sample_data import SAMPLE_DATA_COMMIT, SAMPLE_DATA_SHA256

DEFAULT_OUTPUT = Path("artifacts/real_eeg_benchmark.md")
DEFAULT_JSON_OUTPUT = Path("artifacts/real_eeg_benchmark.json")
README_START_MARKER = "<!-- REAL_EEG_BENCHMARK_SUMMARY_START -->"
README_END_MARKER = "<!-- REAL_EEG_BENCHMARK_SUMMARY_END -->"
REPO_ROOT = Path(__file__).resolve().parents[1]

BACKWARD_STOP_SECONDS = 0.35
BACKWARD_REGULARIZATION_MIN = 1e-8
BACKWARD_REGULARIZATION_MAX = 1e6
BACKWARD_REGULARIZATION_COUNT = 15
BACKWARD_K_FOLDS = 3
BACKWARD_ENVELOPE_COMPRESSION = 0.4

PROFILE_SETTINGS = {
    "matched": {
        "label": "Matched whole-trial",
        "segment_duration": 0.0,
        "overlap": 0.0,
        "window": "none",
    },
    "practical": {
        "label": "Practical 2 s Hann",
        "segment_duration": 2.0,
        "overlap": 0.5,
        "window": "hann",
    },
}


@dataclass(slots=True, frozen=True)
class BenchmarkRow:
    """Repeated measurements for one toolbox, profile, and direction."""

    profile: str
    direction: str
    toolbox: str
    configuration: str
    regularization: float
    mean_correlation: float
    median_correlation: float
    duration_seconds: dict[str, float]
    peak_memory_mib: dict[str, float]
    additional_peak_memory_mib: dict[str, float]
    samples: tuple[dict[str, float], ...]


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Reproduce matched and practical ffTRF versus mTRF real-EEG "
            "benchmarks and write Markdown plus raw JSON reports."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Markdown report path (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=DEFAULT_JSON_OUTPUT,
        help=f"Raw JSON report path (default: {DEFAULT_JSON_OUTPUT}).",
    )
    parser.add_argument(
        "--readme-summary",
        type=Path,
        default=None,
        help="README whose generated real-EEG summary block should be updated.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of measured isolated-process runs per unique fit.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="Number of additional unreported isolated warmup runs per unique fit.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="Native BLAS/OpenMP threads per isolated worker.",
    )
    parser.add_argument(
        "--direction",
        choices=("both", "forward", "backward"),
        default="both",
        help="Limit the report to one direction.",
    )
    parser.add_argument(
        "--profile",
        choices=("both", "matched", "practical"),
        default="both",
        help="Limit the report to matched or practical ffTRF settings.",
    )
    return parser


def _directions(selection: str) -> tuple[str, ...]:
    return ("forward", "backward") if selection == "both" else (selection,)


def _profiles(selection: str) -> tuple[str, ...]:
    return ("matched", "practical") if selection == "both" else (selection,)


def _worker_payloads(
    *,
    method: str,
    profile: str,
    repeats: int,
    warmup: int,
    threads: int,
) -> list[dict[str, float]]:
    settings = PROFILE_SETTINGS[profile]
    return benchmark_worker_payloads(
        method=method,
        repeats=repeats,
        warmup=warmup,
        forward_segment_duration=float(settings["segment_duration"]),
        forward_overlap=float(settings["overlap"]),
        forward_window=str(settings["window"]),
        backward_stop_seconds=BACKWARD_STOP_SECONDS,
        backward_regularization_min=BACKWARD_REGULARIZATION_MIN,
        backward_regularization_max=BACKWARD_REGULARIZATION_MAX,
        backward_regularization_count=BACKWARD_REGULARIZATION_COUNT,
        backward_k_folds=BACKWARD_K_FOLDS,
        backward_segment_duration=float(settings["segment_duration"]),
        backward_overlap=float(settings["overlap"]),
        backward_window=str(settings["window"]),
        backward_envelope_compression=BACKWARD_ENVELOPE_COMPRESSION,
        threads=threads,
    )


def _aggregate(
    *,
    profile: str,
    method: str,
    payloads: list[dict[str, float]],
) -> BenchmarkRow:
    toolbox, direction = method.split("-", maxsplit=1)
    configuration = (
        str(PROFILE_SETTINGS[profile]["label"]) if toolbox == "fftrf" else "Finite-lag baseline"
    )
    return BenchmarkRow(
        profile=profile,
        direction=direction,
        toolbox="ffTRF" if toolbox == "fftrf" else "mTRF",
        configuration=configuration,
        regularization=float(payloads[0]["regularization"]),
        mean_correlation=float(payloads[0]["mean_correlation"]),
        median_correlation=float(payloads[0]["median_correlation"]),
        duration_seconds=summarize([item["duration_seconds"] for item in payloads]),
        peak_memory_mib=summarize([item["peak_memory_mib"] for item in payloads]),
        additional_peak_memory_mib=summarize(
            [item["additional_peak_memory_mib"] for item in payloads]
        ),
        samples=tuple(payloads),
    )


def run_benchmark(
    *,
    repeats: int,
    warmup: int,
    threads: int,
    direction: str,
    profile: str,
) -> list[BenchmarkRow]:
    """Run each unique fit in isolated worker processes."""

    profiles = _profiles(profile)
    rows = []
    for model_direction in _directions(direction):
        print(f"Running mtrf {model_direction} baseline...")
        mtrf_payloads = _worker_payloads(
            method=f"mtrf-{model_direction}",
            profile=profiles[0],
            repeats=repeats,
            warmup=warmup,
            threads=threads,
        )
        for profile_name in profiles:
            print(f"Running fftrf {model_direction} ({profile_name})...")
            fftrf_payloads = _worker_payloads(
                method=f"fftrf-{model_direction}",
                profile=profile_name,
                repeats=repeats,
                warmup=warmup,
                threads=threads,
            )
            rows.extend(
                [
                    _aggregate(
                        profile=profile_name,
                        method=f"fftrf-{model_direction}",
                        payloads=fftrf_payloads,
                    ),
                    _aggregate(
                        profile=profile_name,
                        method=f"mtrf-{model_direction}",
                        payloads=mtrf_payloads,
                    ),
                ]
            )
    return rows


def _comparison_pairs(
    rows: list[BenchmarkRow],
) -> list[tuple[BenchmarkRow, BenchmarkRow]]:
    pairs = []
    for profile in _profiles_present(rows):
        for direction in _directions_present(rows):
            fftrf_row = next(
                row
                for row in rows
                if row.profile == profile and row.direction == direction and row.toolbox == "ffTRF"
            )
            mtrf_row = next(
                row
                for row in rows
                if row.profile == profile and row.direction == direction and row.toolbox == "mTRF"
            )
            pairs.append((fftrf_row, mtrf_row))
    return pairs


def _profiles_present(rows: list[BenchmarkRow]) -> tuple[str, ...]:
    return tuple(
        profile for profile in PROFILE_SETTINGS if any(row.profile == profile for row in rows)
    )


def _directions_present(rows: list[BenchmarkRow]) -> tuple[str, ...]:
    return tuple(
        direction
        for direction in ("forward", "backward")
        if any(row.direction == direction for row in rows)
    )


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator


def _format_ratio(value: float) -> str:
    return f"{value:.2f}×"


def render_markdown(
    rows: list[BenchmarkRow],
    *,
    repeats: int,
    warmup: int,
    threads: int,
    metadata: dict[str, object],
) -> str:
    """Render a complete, self-contained benchmark report."""

    lines = [
        "# Real EEG benchmark",
        "",
        "Matched solver and practical estimator comparisons on the pinned public",
        "mTRF speech-EEG sample.",
        "",
        "## Environment and provenance",
        "",
        f"- Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"- Source: {metadata['source']}",
        f"- CPU: {metadata['cpu']}",
        f"- Platform: {metadata['platform']}",
        f"- Machine: {metadata['machine']}",
        f"- Python: {metadata['python']}",
        f"- ffTRF: {metadata['fftrf']}",
        f"- mTRF: {metadata['mtrf']}",
        f"- NumPy: {metadata['numpy']}",
        f"- SciPy: {metadata['scipy']}",
        f"- Native threads per worker: {threads}",
        f"- Dataset commit: `{SAMPLE_DATA_COMMIT}`",
        f"- Dataset SHA-256: `{SAMPLE_DATA_SHA256}`",
        "",
        "## Protocol",
        "",
        "- Command: `pixi run -e compare real-eeg-benchmark`",
        "- Data: 10 twelve-second segments at 128 Hz; 7 train/CV and 3 held out.",
        "- Forward model: 16 speech bands to 128 EEG channels, 5-fold CV.",
        "- Backward model: 128 EEG channels to a compressed broadband envelope, 3-fold CV.",
        "- Selection: seeded folds (`seed=7`) and negative MSE over the same direction-specific lambda grid.",
        "- Accuracy: held-out Pearson correlation across channels (forward) or segments (backward). It is a prediction check, not ground-truth kernel accuracy.",
        f"- Timing: median and [minimum, maximum] of {repeats} isolated fit-only run(s); {warmup} unreported warmup run(s).",
        "- Total peak RSS includes the interpreter, imports, and loaded data.",
        "- Additional peak RSS is growth above the process peak immediately before fitting.",
        "- Prediction, plotting, data download, and data loading are excluded from fit time.",
        "",
    ]

    for profile in _profiles_present(rows):
        if profile == "matched":
            lines.extend(
                [
                    "## Matched whole-trial comparison",
                    "",
                    "ffTRF uses whole-trial rectangular spectra with no segmentation,",
                    "windowing, multitaper smoothing, or detrending. This is the closest",
                    "comparison to the finite-lag mTRF fit.",
                    "",
                ]
            )
        else:
            lines.extend(
                [
                    "## Practical 2 s Hann comparison",
                    "",
                    "ffTRF uses 2-second Hann windows with 50% overlap. This changes the",
                    "spectral estimator, so the result is a practical workflow comparison,",
                    "not a strict solver-equivalence claim. The mTRF baseline is unchanged.",
                    "",
                ]
            )

        lines.extend(
            [
                "| Direction | Model | Configuration | Lambda | Mean held-out r | Median held-out r | Fit seconds median [range] | Total peak RSS MiB median [range] | Additional peak RSS MiB median [range] |",
                "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in rows:
            if row.profile != profile:
                continue
            lines.append(
                f"| {row.direction.capitalize()} | {row.toolbox} | "
                f"{row.configuration} | {row.regularization:.6g} | "
                f"{row.mean_correlation:.4f} | {row.median_correlation:.4f} | "
                f"{format_median_range(row.duration_seconds, precision=4)} | "
                f"{format_median_range(row.peak_memory_mib, precision=1)} | "
                f"{format_median_range(row.additional_peak_memory_mib, precision=1)} |"
            )

        lines.extend(
            [
                "",
                "| Direction | Runtime ratio (mTRF / ffTRF) | Total peak RSS ratio (mTRF / ffTRF) |",
                "| --- | ---: | ---: |",
            ]
        )
        for fftrf_row, mtrf_row in _comparison_pairs(rows):
            if fftrf_row.profile != profile:
                continue
            lines.append(
                f"| {fftrf_row.direction.capitalize()} | "
                f"{_format_ratio(_ratio(mtrf_row.duration_seconds['median'], fftrf_row.duration_seconds['median']))} | "
                f"{_format_ratio(_ratio(mtrf_row.peak_memory_mib['median'], fftrf_row.peak_memory_mib['median']))} |"
            )
        lines.extend(
            [
                "",
                "Ratios above 1 favor ffTRF; ratios below 1 favor mTRF.",
                "",
            ]
        )

    return "\n".join(lines)


def render_readme_summary(rows: list[BenchmarkRow]) -> str:
    """Render the compact generated real-EEG README table."""

    lines = [
        "| Comparison | Direction | Runtime ratio (mTRF / ffTRF) | Peak RSS ratio (mTRF / ffTRF) | Held-out r (ffTRF / mTRF) |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for fftrf_row, mtrf_row in _comparison_pairs(rows):
        label = str(PROFILE_SETTINGS[fftrf_row.profile]["label"])
        lines.append(
            f"| {label} | {fftrf_row.direction.capitalize()} | "
            f"{_format_ratio(_ratio(mtrf_row.duration_seconds['median'], fftrf_row.duration_seconds['median']))} | "
            f"{_format_ratio(_ratio(mtrf_row.peak_memory_mib['median'], fftrf_row.peak_memory_mib['median']))} | "
            f"{fftrf_row.mean_correlation:.4f} / {mtrf_row.mean_correlation:.4f} |"
        )
    lines.extend(
        [
            "",
            "Ratios above 1 favor ffTRF. Matched rows compare the closest available",
            "solver settings. Practical rows use 2-second Hann-windowed spectra in",
            "ffTRF and therefore compare workflows rather than identical estimators.",
        ]
    )
    return "\n".join(lines)


def _write_json_report(
    path: Path,
    *,
    rows: list[BenchmarkRow],
    repeats: int,
    warmup: int,
    threads: int,
    metadata: dict[str, object],
) -> None:
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "environment": metadata,
        "protocol": {
            "command": "pixi run -e compare real-eeg-benchmark",
            "repeats": repeats,
            "warmup": warmup,
            "native_threads_per_worker": threads,
            "dataset_commit": SAMPLE_DATA_COMMIT,
            "dataset_sha256": SAMPLE_DATA_SHA256,
        },
        "rows": [asdict(row) for row in rows],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    """Run the benchmark and write synchronized reports."""

    parser = build_parser()
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.threads < 1:
        parser.error("--threads must be at least 1")

    metadata = environment_metadata(repo_root=REPO_ROOT, threads=int(args.threads))
    rows = run_benchmark(
        repeats=int(args.repeats),
        warmup=int(args.warmup),
        threads=int(args.threads),
        direction=str(args.direction),
        profile=str(args.profile),
    )
    report = render_markdown(
        rows,
        repeats=int(args.repeats),
        warmup=int(args.warmup),
        threads=int(args.threads),
        metadata=metadata,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    _write_json_report(
        args.json_output,
        rows=rows,
        repeats=int(args.repeats),
        warmup=int(args.warmup),
        threads=int(args.threads),
        metadata=metadata,
    )
    if args.readme_summary is not None:
        replace_marked_section(
            args.readme_summary,
            start_marker=README_START_MARKER,
            end_marker=README_END_MARKER,
            content=render_readme_summary(rows),
        )

    print(f"Wrote {args.output}")
    print(f"Wrote {args.json_output}")
    if args.readme_summary is not None:
        print(f"Updated {args.readme_summary}")


if __name__ == "__main__":
    main()
