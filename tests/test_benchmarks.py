from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import pytest

EXAMPLES_DIR = Path(__file__).parents[1] / "examples"
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

benchmark_real_eeg = importlib.import_module("benchmark_real_eeg")
benchmark_runtime = importlib.import_module("benchmark_runtime")
benchmark_utils = importlib.import_module("benchmark_utils")


def _summary(median: float, minimum: float | None = None, maximum: float | None = None):
    return {
        "median": median,
        "min": median if minimum is None else minimum,
        "max": median if maximum is None else maximum,
    }


def _real_row(
    *,
    profile: str,
    toolbox: str,
    duration: float,
    peak: float,
) -> Any:
    return benchmark_real_eeg.BenchmarkRow(
        profile=profile,
        direction="forward",
        toolbox=toolbox,
        configuration=(
            str(benchmark_real_eeg.PROFILE_SETTINGS[profile]["label"])
            if toolbox == "ffTRF"
            else "Finite-lag baseline"
        ),
        regularization=1e3,
        mean_correlation=0.2 if toolbox == "ffTRF" else 0.19,
        median_correlation=0.18,
        duration_seconds=_summary(duration, duration * 0.9, duration * 1.1),
        peak_memory_mib=_summary(peak, peak - 5.0, peak + 5.0),
        additional_peak_memory_mib=_summary(peak / 2.0),
        samples=(),
    )


def _runtime_row(name: str, *, speedup: float = 2.0) -> dict[str, Any]:
    return {
        "name": name,
        "shape": "1->1",
        "direction": "forward",
        "fit_mode": "fixed",
        "fft_setting": "whole-trial",
        "fs_hz": 1_000,
        "n_trials": 2,
        "n_samples": 1_024,
        "n_lags": 40,
        "lag_matrix_mib": 1.0,
        "fft_seconds": _summary(1.0, 0.9, 1.1),
        "mtrf_seconds": _summary(speedup, speedup * 0.9, speedup * 1.1),
        "fft_peak_mib": _summary(100.0, 99.0, 101.0),
        "mtrf_peak_mib": _summary(200.0, 199.0, 201.0),
        "fft_additional_peak_mib": _summary(20.0),
        "mtrf_additional_peak_mib": _summary(80.0),
        "speedup": speedup,
        "peak_memory_ratio": 2.0,
        "fft_prediction_score": 0.99,
        "mtrf_prediction_score": 0.99,
        "kernel_corr": 1.0,
        "fft_samples": [],
        "mtrf_samples": [],
    }


def _metadata() -> dict[str, object]:
    return {
        "source": "abcdef123456 (clean)",
        "cpu": "Test CPU",
        "platform": "Test OS",
        "machine": "test-machine",
        "python": "3.13",
        "fftrf": "0.1.0",
        "mtrf": "2.1.2",
        "numpy": "2.4",
        "scipy": "1.17",
    }


def test_benchmark_helpers_summarize_and_replace_marked_section(tmp_path: Path) -> None:
    summary = benchmark_utils.summarize([3.0, 1.0, 2.0])
    assert summary == {"median": 2.0, "min": 1.0, "max": 3.0}
    assert benchmark_utils.format_median_range(summary, precision=1) == "2.0 [1.0, 3.0]"

    report = tmp_path / "README.md"
    report.write_text("before\n<!-- START -->\nold\n<!-- END -->\nafter\n", encoding="utf-8")
    benchmark_utils.replace_marked_section(
        report,
        start_marker="<!-- START -->",
        end_marker="<!-- END -->",
        content="new",
    )
    assert report.read_text(encoding="utf-8") == (
        "before\n<!-- START -->\nnew\n<!-- END -->\nafter\n"
    )


def test_worker_environment_controls_native_threads() -> None:
    environment = benchmark_utils.isolated_worker_environment(threads=2)
    for name in benchmark_utils.THREAD_ENVIRONMENT_VARIABLES:
        assert environment[name] == "2"
    assert environment["PYTHONHASHSEED"] == "0"

    with pytest.raises(ValueError, match="threads must be at least 1"):
        benchmark_utils.isolated_worker_environment(threads=0)


def test_real_benchmark_report_separates_matched_and_practical_profiles() -> None:
    rows = [
        _real_row(profile="matched", toolbox="ffTRF", duration=2.0, peak=200.0),
        _real_row(profile="matched", toolbox="mTRF", duration=4.0, peak=300.0),
        _real_row(profile="practical", toolbox="ffTRF", duration=1.0, peak=100.0),
        _real_row(profile="practical", toolbox="mTRF", duration=4.0, peak=300.0),
    ]

    report = benchmark_real_eeg.render_markdown(
        rows,
        repeats=3,
        warmup=0,
        threads=1,
        metadata=_metadata(),
    )
    assert "Matched whole-trial comparison" in report
    assert "Practical 2 s Hann comparison" in report
    assert "2.00×" in report
    assert "4.00×" in report
    assert "prediction check, not ground-truth kernel accuracy" in report
    assert benchmark_real_eeg.SAMPLE_DATA_SHA256 in report
    assert "pixi run -e compare real-eeg-benchmark" in report

    summary = benchmark_real_eeg.render_readme_summary(rows)
    assert "| Matched whole-trial | Forward | 2.00× | 1.50× |" in summary
    assert "| Practical 2 s Hann | Forward | 4.00× | 3.00× |" in summary


def test_synthetic_report_exposes_ranges_memory_and_honest_crossover() -> None:
    row = _runtime_row("Moderate length", speedup=0.5)
    report = benchmark_runtime.format_report(
        [row],
        repeats=3,
        warmup=1,
        threads=1,
        environment=_metadata(),
    )
    assert "1.0000 [0.9000, 1.1000]" in report
    assert "Additional peak RSS" in report
    assert "Ratios are `mTRF / ffTRF`" in report
    assert "0.50×" in report
    assert "pixi run -e compare benchmark-demo" in report

    summary_rows = [
        _runtime_row("Moderate length", speedup=0.5),
        _runtime_row("Longer lag window"),
        _runtime_row("Cross-validated ridge"),
        _runtime_row("102-channel backward decoder"),
    ]
    summary = benchmark_runtime.render_readme_summary(summary_rows)
    assert "ffTRF is not universally" in summary
    assert "| Moderate length | 1->1, fixed | 0.50× |" in summary


def test_release_benchmark_defaults_use_repeated_single_thread_runs() -> None:
    runtime_args = benchmark_runtime.build_parser().parse_args([])
    real_args = benchmark_real_eeg.build_parser().parse_args([])

    assert runtime_args.repeats == 3
    assert runtime_args.warmup == 1
    assert runtime_args.threads == 1
    assert real_args.repeats == 3
    assert real_args.warmup == 0
    assert real_args.threads == 1
    assert real_args.profile == "both"
