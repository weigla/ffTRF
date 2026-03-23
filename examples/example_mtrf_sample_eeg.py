#!/usr/bin/env python3
"""Example: compare ffTRF and mTRF on the public speech EEG sample."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from statistics import median
from subprocess import check_output
import sys
from time import perf_counter

import numpy as np

from fftrf import TRF, pearsonr

from mtrf_sample_data import exact_lag_window_seconds, load_sample_data
from simulated_data import finalize_figure, require_matplotlib

OUTPUT_PATH = Path("artifacts/examples/mtrf_sample_eeg_comparison.png")
SELECTION_METRIC_NAME = "neg_mse"
WORKER_METHODS = (
    "fftrf-forward",
    "mtrf-forward",
    "fftrf-backward",
    "mtrf-backward",
)


@dataclass(slots=True, frozen=True)
class ComparisonSetup:
    """Shared configuration for the real EEG toolbox comparison."""

    stimulus: list[np.ndarray]
    response: list[np.ndarray]
    backward_stimulus: list[np.ndarray]
    train_stimulus: list[np.ndarray]
    train_response: list[np.ndarray]
    backward_train_stimulus: list[np.ndarray]
    test_stimulus: list[np.ndarray]
    test_response: list[np.ndarray]
    backward_test_stimulus: list[np.ndarray]
    fs: int
    regularization_grid: np.ndarray
    k_folds: int
    cv_seed: int
    tmin: float
    n_lags: int
    tmax: float


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Compare ffTRF and mTRF on the official speech EEG sample with "
            "matched forward and backward CV benchmarks."
        )
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Number of isolated timing and memory runs per toolbox and model direction.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="Number of untimed isolated warmup runs per toolbox and model direction.",
    )
    parser.add_argument(
        "--skip-backward",
        action="store_true",
        help="Skip the additional backward CV benchmark.",
    )
    parser.add_argument(
        "--backward-stop-seconds",
        type=float,
        default=0.35,
        help="Lag-window stop in seconds for the backward decoder.",
    )
    parser.add_argument(
        "--backward-regularization-min",
        type=float,
        default=1e-8,
        help="Smallest lambda in the backward CV grid.",
    )
    parser.add_argument(
        "--backward-regularization-max",
        type=float,
        default=1e6,
        help="Largest lambda in the backward CV grid.",
    )
    parser.add_argument(
        "--backward-regularization-count",
        type=int,
        default=15,
        help="Number of lambda candidates in the backward CV grid.",
    )
    parser.add_argument(
        "--backward-k-folds",
        type=int,
        default=3,
        help="Number of CV folds used for the backward benchmark.",
    )
    parser.add_argument(
        "--backward-segment-duration",
        type=float,
        default=2.0,
        help="Segment duration in seconds for the backward ffTRF benchmark.",
    )
    parser.add_argument(
        "--backward-overlap",
        type=float,
        default=0.5,
        help="Segment overlap fraction for the backward ffTRF benchmark.",
    )
    parser.add_argument(
        "--backward-window",
        type=str,
        default="hann",
        help="Window name for the backward ffTRF benchmark. Use 'none' to disable.",
    )
    parser.add_argument(
        "--backward-envelope-compression",
        type=float,
        default=0.4,
        help=(
            "Compression exponent applied to the mean raw stimulus bands when "
            "building the 1D backward envelope target."
        ),
    )
    parser.add_argument(
        "--worker-method",
        choices=WORKER_METHODS,
        default=None,
        help=argparse.SUPPRESS,
    )
    return parser


def current_process_peak_memory_mib() -> float:
    """Return the current-process peak RSS in MiB when available."""

    try:
        import resource
    except ModuleNotFoundError:
        return float("nan")

    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return float(peak_rss) / (1024.0**2)
    return float(peak_rss) / 1024.0


def load_comparison_setup(
    *,
    backward_envelope_compression: float = 0.4,
) -> ComparisonSetup:
    """Load the public speech EEG example and derive the train/test split."""

    stimulus, response, fs = load_sample_data(n_segments=10, normalize=True)
    raw_stimulus, _, raw_fs = load_sample_data(n_segments=10, normalize=False)
    if raw_fs != fs:
        raise ValueError("Normalized and raw sample-data loaders disagreed on the sampling rate.")

    backward_stimulus = [
        _backward_envelope_target(
            trial,
            compression_exponent=backward_envelope_compression,
        )
        for trial in raw_stimulus
    ]
    train_stimulus = stimulus[:-3]
    train_response = response[:-3]
    backward_train_stimulus = backward_stimulus[:-3]
    test_stimulus = stimulus[-3:]
    test_response = response[-3:]
    backward_test_stimulus = backward_stimulus[-3:]

    regularization_grid = np.logspace(-4, 4, 17)
    k_folds = 5
    cv_seed = 7
    tmin = 0.0
    n_lags, tmax = exact_lag_window_seconds(fs=fs, nominal_stop_seconds=0.4)

    return ComparisonSetup(
        stimulus=stimulus,
        response=response,
        backward_stimulus=backward_stimulus,
        train_stimulus=train_stimulus,
        train_response=train_response,
        backward_train_stimulus=backward_train_stimulus,
        test_stimulus=test_stimulus,
        test_response=test_response,
        backward_test_stimulus=backward_test_stimulus,
        fs=fs,
        regularization_grid=regularization_grid,
        k_folds=k_folds,
        cv_seed=cv_seed,
        tmin=tmin,
        n_lags=n_lags,
        tmax=tmax,
    )


def fit_fftrf(
    stimulus: list[np.ndarray],
    response: list[np.ndarray],
    *,
    fs: int,
    tmin: float,
    tmax: float,
    regularization: float | list[float] | np.ndarray,
    direction: int,
    metric: str = SELECTION_METRIC_NAME,
    segment_duration: float | None = None,
    overlap: float | None = None,
    window: str | None = None,
    k: int | None = None,
    seed: int | None = None,
) -> tuple[TRF, np.ndarray | float | None]:
    """Fit a matched ffTRF model."""

    model = TRF(direction=direction, metric=metric)
    train_kwargs = {
        "stimulus": stimulus,
        "response": response,
        "fs": fs,
        "tmin": tmin,
        "tmax": tmax,
        "regularization": regularization,
    }
    if segment_duration is not None:
        train_kwargs["segment_duration"] = segment_duration
    else:
        train_kwargs["segment_length"] = None
    if overlap is not None:
        train_kwargs["overlap"] = overlap
    train_kwargs["window"] = window
    if k is not None:
        train_kwargs["k"] = k
    if seed is not None:
        train_kwargs["seed"] = seed
    cv_scores = model.train(**train_kwargs)
    return model, cv_scores


def fit_mtrf(
    stimulus: list[np.ndarray],
    response: list[np.ndarray],
    *,
    fs: int,
    tmin: float,
    tmax: float,
    regularization: float | list[float] | np.ndarray,
    direction: int,
    metric: str = SELECTION_METRIC_NAME,
    k: int | None = None,
    seed: int | None = None,
):
    """Fit a matched reference mTRF model."""

    try:
        from mtrf.model import TRF
        from mtrf.stats import neg_mse
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "mTRF is required for this example. Use the compare extras or "
            "the Pixi compare environment."
        ) from exc

    if metric == "neg_mse":
        metric_callable = neg_mse
    elif metric == "pearsonr":
        metric_callable = None
    else:
        raise ValueError(f"Unsupported comparison metric for mTRF: {metric!r}")

    model = TRF(direction=direction, **({} if metric_callable is None else {"metric": metric_callable}))
    train_kwargs = {
        "stimulus": stimulus,
        "response": response,
        "fs": fs,
        "tmin": tmin,
        "tmax": tmax - (1.0 / fs),
        "regularization": regularization,
        "verbose": False,
    }
    if k is not None:
        train_kwargs["k"] = k
    if seed is not None:
        train_kwargs["seed"] = seed
    cv_scores = model.train(**train_kwargs)
    return model, cv_scores


def benchmark_worker(
    *,
    method: str,
    repeats: int,
    warmup: int,
    backward_stop_seconds: float,
    backward_regularization_min: float,
    backward_regularization_max: float,
    backward_regularization_count: int,
    backward_k_folds: int,
    backward_segment_duration: float,
    backward_overlap: float,
    backward_window: str,
    backward_envelope_compression: float,
) -> tuple[list[float], list[float]]:
    """Run isolated worker fits and return durations and peak RSS values."""

    script_path = Path(__file__).resolve()
    durations = []
    peak_memories = []
    for run_index in range(repeats + warmup):
        command = [
            sys.executable,
            str(script_path),
            "--worker-method",
            method,
            "--skip-backward",
            "--backward-stop-seconds",
            str(backward_stop_seconds),
            "--backward-regularization-min",
            str(backward_regularization_min),
            "--backward-regularization-max",
            str(backward_regularization_max),
            "--backward-regularization-count",
            str(backward_regularization_count),
            "--backward-k-folds",
            str(backward_k_folds),
            "--backward-segment-duration",
            str(backward_segment_duration),
            "--backward-overlap",
            str(backward_overlap),
            "--backward-window",
            str(backward_window),
            "--backward-envelope-compression",
            str(backward_envelope_compression),
        ]
        output = check_output(command, text=True)
        payload = json.loads(output.strip().splitlines()[-1])
        if run_index >= warmup:
            durations.append(float(payload["duration_seconds"]))
            peak_memories.append(float(payload["peak_memory_mib"]))
    return durations, peak_memories


def run_worker_once(
    args: argparse.Namespace,
    *,
    method: str,
) -> dict[str, float]:
    """Run one isolated CV fit for timing and peak RSS reporting."""

    setup = load_comparison_setup(
        backward_envelope_compression=float(args.backward_envelope_compression),
    )
    toolbox_name, direction = method.split("-", maxsplit=1)
    backward_tmax = exact_lag_window_seconds(
        fs=setup.fs,
        nominal_stop_seconds=float(args.backward_stop_seconds),
    )[1]
    backward_regularization_grid = np.logspace(
        np.log10(float(args.backward_regularization_min)),
        np.log10(float(args.backward_regularization_max)),
        int(args.backward_regularization_count),
    )
    backward_window = None if str(args.backward_window).strip().lower() == "none" else str(args.backward_window)

    start = perf_counter()
    if toolbox_name == "fftrf":
        fit_fftrf(
            setup.train_stimulus if direction == "forward" else setup.backward_train_stimulus,
            setup.train_response,
            fs=setup.fs,
            tmin=0.0,
            tmax=setup.tmax if direction == "forward" else backward_tmax,
            regularization=(
                setup.regularization_grid
                if direction == "forward"
                else backward_regularization_grid
            ),
            k=setup.k_folds if direction == "forward" else int(args.backward_k_folds),
            seed=setup.cv_seed,
            direction=1 if direction == "forward" else -1,
            segment_duration=None if direction == "forward" else float(args.backward_segment_duration),
            overlap=None if direction == "forward" else float(args.backward_overlap),
            window=None if direction == "forward" else backward_window,
        )
    else:
        fit_mtrf(
            setup.train_stimulus if direction == "forward" else setup.backward_train_stimulus,
            setup.train_response,
            fs=setup.fs,
            tmin=0.0,
            tmax=setup.tmax if direction == "forward" else backward_tmax,
            regularization=(
                setup.regularization_grid
                if direction == "forward"
                else backward_regularization_grid
            ),
            k=setup.k_folds if direction == "forward" else int(args.backward_k_folds),
            seed=setup.cv_seed,
            direction=1 if direction == "forward" else -1,
        )
    duration = perf_counter() - start
    return {
        "duration_seconds": duration,
        "peak_memory_mib": current_process_peak_memory_mib(),
    }


def global_field_power(x: np.ndarray) -> np.ndarray:
    """Return the across-channel standard deviation at each time sample."""

    return np.std(np.asarray(x, dtype=float), axis=1)


def plot_prediction_trace(
    ax,
    *,
    times_seconds: np.ndarray,
    observed: np.ndarray,
    fftrf_prediction: np.ndarray,
    mtrf_prediction: np.ndarray,
    title: str,
    ylabel: str = "Amplitude (z)",
) -> None:
    """Plot observed and predicted held-out data for one 1D trace."""

    ax.plot(times_seconds, observed, color="#222222", linewidth=1.5, label="Observed")
    ax.plot(
        times_seconds,
        fftrf_prediction,
        color="#0B6E4F",
        linewidth=1.4,
        label="ffTRF",
    )
    ax.plot(
        times_seconds,
        mtrf_prediction,
        color="#3366CC",
        linewidth=1.3,
        label="mTRF",
    )
    ax.axhline(0.0, color="#999999", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.2, linewidth=0.6)


def plot_score_distribution(
    ax,
    *,
    fftrf_scores: np.ndarray,
    mtrf_scores: np.ndarray,
    title: str,
    xlabel: str,
) -> None:
    """Plot sorted held-out scores for both toolboxes."""

    score_order = np.argsort(0.5 * (fftrf_scores + mtrf_scores))[::-1]
    rank = np.arange(1, len(score_order) + 1)
    fftrf_sorted = fftrf_scores[score_order]
    mtrf_sorted = mtrf_scores[score_order]

    ax.plot(rank, fftrf_sorted, color="#0B6E4F", linewidth=1.8, label="ffTRF")
    ax.plot(rank, mtrf_sorted, color="#3366CC", linewidth=1.6, label="mTRF")
    ax.fill_between(
        rank,
        fftrf_sorted,
        mtrf_sorted,
        color="#C84C09",
        alpha=0.12,
        linewidth=0.0,
    )
    ax.axhline(0.0, color="#999999", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Pearson r")
    ax.grid(alpha=0.2, linewidth=0.6)


def plot_summary_bars(
    ax,
    *,
    forward_fftrf_scores: np.ndarray,
    forward_mtrf_scores: np.ndarray,
    backward_fftrf_scores: np.ndarray,
    backward_mtrf_scores: np.ndarray,
) -> None:
    """Plot a compact held-out summary across forward and backward models."""

    metrics = (
        "Forward mean r",
        "Forward median r",
        "Backward mean r",
        "Backward median r",
    )
    fftrf_values = (
        float(np.mean(forward_fftrf_scores)),
        float(np.median(forward_fftrf_scores)),
        float(np.mean(backward_fftrf_scores)),
        float(np.median(backward_fftrf_scores)),
    )
    mtrf_values = (
        float(np.mean(forward_mtrf_scores)),
        float(np.median(forward_mtrf_scores)),
        float(np.mean(backward_mtrf_scores)),
        float(np.median(backward_mtrf_scores)),
    )

    x = np.arange(len(metrics), dtype=float)
    width = 0.35
    ax.bar(x - 0.5 * width, fftrf_values, width=width, color="#0B6E4F", label="ffTRF")
    ax.bar(x + 0.5 * width, mtrf_values, width=width, color="#3366CC", label="mTRF")
    ax.axhline(0.0, color="#999999", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=15, ha="right")
    ax.set_ylabel("Pearson r")
    ax.set_title("Held-out summary")
    ax.grid(alpha=0.2, linewidth=0.6, axis="y")


def trial_correlations(
    *,
    observed_trials: list[np.ndarray],
    predicted_trials,
) -> np.ndarray:
    """Return one Pearson correlation per observed/predicted trial pair."""

    predicted_list = _coerce_trial_list(predicted_trials)
    if len(observed_trials) != len(predicted_list):
        raise ValueError("Observed and predicted trial lists must have the same length.")

    scores = []
    for observed_trial, predicted_trial in zip(observed_trials, predicted_list, strict=True):
        observed = _ensure_2d_column_array(observed_trial)
        predicted = _ensure_2d_column_array(predicted_trial)
        trial_length = min(observed.shape[0], predicted.shape[0])
        scores.append(float(pearsonr(observed[:trial_length], predicted[:trial_length])[0]))
    return np.asarray(scores, dtype=float)


def feature_correlations(
    *,
    observed_trials: list[np.ndarray],
    predicted_trials,
) -> np.ndarray:
    """Return one average Pearson correlation per feature across held-out trials."""

    predicted_list = _coerce_trial_list(predicted_trials)
    if len(observed_trials) != len(predicted_list):
        raise ValueError("Observed and predicted trial lists must have the same length.")
    return np.mean(
        np.vstack(
            [
                pearsonr(np.asarray(observed, dtype=float), np.asarray(predicted, dtype=float))
                for observed, predicted in zip(observed_trials, predicted_list, strict=True)
            ]
        ),
        axis=0,
    )


def main() -> None:
    """Fit both toolboxes on the public speech EEG example and compare them."""

    parser = build_parser()
    args = parser.parse_args()

    if args.worker_method is not None:
        payload = run_worker_once(args, method=args.worker_method)
        sys.stdout.write(json.dumps(payload) + "\n")
        return

    setup = load_comparison_setup(
        backward_envelope_compression=float(args.backward_envelope_compression),
    )

    fftrf_durations, fftrf_peak_memories = benchmark_worker(
        method="fftrf-forward",
        repeats=args.repeats,
        warmup=args.warmup,
        backward_stop_seconds=float(args.backward_stop_seconds),
        backward_regularization_min=float(args.backward_regularization_min),
        backward_regularization_max=float(args.backward_regularization_max),
        backward_regularization_count=int(args.backward_regularization_count),
        backward_k_folds=int(args.backward_k_folds),
        backward_segment_duration=float(args.backward_segment_duration),
        backward_overlap=float(args.backward_overlap),
        backward_window=str(args.backward_window),
        backward_envelope_compression=float(args.backward_envelope_compression),
    )
    mtrf_durations, mtrf_peak_memories = benchmark_worker(
        method="mtrf-forward",
        repeats=args.repeats,
        warmup=args.warmup,
        backward_stop_seconds=float(args.backward_stop_seconds),
        backward_regularization_min=float(args.backward_regularization_min),
        backward_regularization_max=float(args.backward_regularization_max),
        backward_regularization_count=int(args.backward_regularization_count),
        backward_k_folds=int(args.backward_k_folds),
        backward_segment_duration=float(args.backward_segment_duration),
        backward_overlap=float(args.backward_overlap),
        backward_window=str(args.backward_window),
        backward_envelope_compression=float(args.backward_envelope_compression),
    )
    backward_fftrf_durations = []
    backward_fftrf_peak_memories = []
    backward_mtrf_durations = []
    backward_mtrf_peak_memories = []
    backward_regularization_grid = None
    backward_k_folds = None

    fftrf_model, fftrf_cv_scores = fit_fftrf(
        setup.train_stimulus,
        setup.train_response,
        fs=setup.fs,
        tmin=setup.tmin,
        tmax=setup.tmax,
        regularization=setup.regularization_grid,
        k=setup.k_folds,
        seed=setup.cv_seed,
        direction=1,
    )
    mtrf_model, mtrf_cv_scores = fit_mtrf(
        setup.train_stimulus,
        setup.train_response,
        fs=setup.fs,
        tmin=setup.tmin,
        tmax=setup.tmax,
        regularization=setup.regularization_grid,
        k=setup.k_folds,
        seed=setup.cv_seed,
        direction=1,
    )

    fftrf_prediction, _ = fftrf_model.predict(
        stimulus=setup.test_stimulus,
        response=setup.test_response,
        average=False,
    )
    mtrf_prediction, _ = mtrf_model.predict(
        stimulus=setup.test_stimulus,
        response=setup.test_response,
        average=False,
    )

    fftrf_scores = feature_correlations(
        observed_trials=setup.test_response,
        predicted_trials=fftrf_prediction,
    )
    mtrf_scores = feature_correlations(
        observed_trials=setup.test_response,
        predicted_trials=mtrf_prediction,
    )
    channel_index = int(np.argmax(0.5 * (fftrf_scores + mtrf_scores)))

    observed_segment = np.asarray(setup.test_response[0], dtype=float)
    fftrf_segment = np.asarray(fftrf_prediction[0], dtype=float)
    mtrf_segment = np.asarray(mtrf_prediction[0], dtype=float)
    times_seconds = np.arange(observed_segment.shape[0], dtype=float) / float(setup.fs)

    observed_channel = observed_segment[:, channel_index]
    fftrf_channel = fftrf_segment[:, channel_index]
    mtrf_channel = mtrf_segment[:, channel_index]
    observed_gfp = global_field_power(observed_segment)
    fftrf_gfp = global_field_power(fftrf_segment)
    mtrf_gfp = global_field_power(mtrf_segment)

    backward_scores = None
    backward_mtrf_scores = None
    backward_fftrf_cv_scores = None
    backward_mtrf_cv_scores = None
    backward_trial_index = None
    backward_times_seconds = None
    backward_observed = None
    fftrf_backward_trial = None
    mtrf_backward_trial = None
    backward_fftrf_model = None
    backward_mtrf_model = None
    backward_tmax = None
    backward_n_lags = None
    backward_window = None if str(args.backward_window).strip().lower() == "none" else str(args.backward_window)

    if not args.skip_backward:
        backward_n_lags, backward_tmax = exact_lag_window_seconds(
            fs=setup.fs,
            nominal_stop_seconds=float(args.backward_stop_seconds),
        )
        backward_regularization_grid = np.logspace(
            np.log10(float(args.backward_regularization_min)),
            np.log10(float(args.backward_regularization_max)),
            int(args.backward_regularization_count),
        )
        backward_k_folds = int(args.backward_k_folds)
        backward_fftrf_durations, backward_fftrf_peak_memories = benchmark_worker(
            method="fftrf-backward",
            repeats=args.repeats,
            warmup=args.warmup,
            backward_stop_seconds=float(args.backward_stop_seconds),
            backward_regularization_min=float(args.backward_regularization_min),
            backward_regularization_max=float(args.backward_regularization_max),
            backward_regularization_count=int(args.backward_regularization_count),
            backward_k_folds=int(args.backward_k_folds),
            backward_segment_duration=float(args.backward_segment_duration),
            backward_overlap=float(args.backward_overlap),
            backward_window=str(args.backward_window),
            backward_envelope_compression=float(args.backward_envelope_compression),
        )
        backward_mtrf_durations, backward_mtrf_peak_memories = benchmark_worker(
            method="mtrf-backward",
            repeats=args.repeats,
            warmup=args.warmup,
            backward_stop_seconds=float(args.backward_stop_seconds),
            backward_regularization_min=float(args.backward_regularization_min),
            backward_regularization_max=float(args.backward_regularization_max),
            backward_regularization_count=int(args.backward_regularization_count),
            backward_k_folds=int(args.backward_k_folds),
            backward_segment_duration=float(args.backward_segment_duration),
            backward_overlap=float(args.backward_overlap),
            backward_window=str(args.backward_window),
            backward_envelope_compression=float(args.backward_envelope_compression),
        )
        backward_fftrf_model, backward_fftrf_cv_scores = fit_fftrf(
            setup.backward_train_stimulus,
            setup.train_response,
            fs=setup.fs,
            tmin=0.0,
            tmax=backward_tmax,
            regularization=backward_regularization_grid,
            direction=-1,
            k=backward_k_folds,
            seed=setup.cv_seed,
            segment_duration=float(args.backward_segment_duration),
            overlap=float(args.backward_overlap),
            window=backward_window,
        )
        backward_mtrf_model, backward_mtrf_cv_scores = fit_mtrf(
            setup.backward_train_stimulus,
            setup.train_response,
            fs=setup.fs,
            tmin=0.0,
            tmax=backward_tmax,
            regularization=backward_regularization_grid,
            direction=-1,
            k=backward_k_folds,
            seed=setup.cv_seed,
        )
        fftrf_backward_prediction, _ = backward_fftrf_model.predict(
            stimulus=setup.backward_test_stimulus,
            response=setup.test_response,
            average=False,
        )
        mtrf_backward_prediction, _ = backward_mtrf_model.predict(
            stimulus=setup.backward_test_stimulus,
            response=setup.test_response,
            average=False,
        )
        backward_scores = trial_correlations(
            observed_trials=setup.backward_test_stimulus,
            predicted_trials=fftrf_backward_prediction,
        )
        backward_mtrf_scores = trial_correlations(
            observed_trials=setup.backward_test_stimulus,
            predicted_trials=mtrf_backward_prediction,
        )
        backward_trial_index = int(np.argmax(0.5 * (backward_scores + backward_mtrf_scores)))
        backward_observed = _ensure_2d_column_array(setup.backward_test_stimulus[backward_trial_index])
        fftrf_backward_trial = _ensure_2d_column_array(
            _coerce_trial_list(fftrf_backward_prediction)[backward_trial_index]
        )
        mtrf_backward_trial = _ensure_2d_column_array(
            _coerce_trial_list(mtrf_backward_prediction)[backward_trial_index]
        )
        backward_trial_length = min(
            backward_observed.shape[0],
            fftrf_backward_trial.shape[0],
            mtrf_backward_trial.shape[0],
        )
        backward_times_seconds = np.arange(backward_trial_length, dtype=float) / float(setup.fs)

    print("Example: public mTRF speech EEG sample")
    print("  dataset: 16-band speech spectrogram to 128-channel EEG")
    print(
        f"  segments: {len(setup.stimulus)} total, "
        f"{len(setup.train_stimulus)} train, {len(setup.test_stimulus)} test"
    )
    print(f"  sampling rate: {setup.fs} Hz")
    print(
        f"  lag window: {setup.n_lags} samples "
        f"({fftrf_model.times[0]:.6f} to {fftrf_model.times[-1]:.6f} s)"
    )
    print(f"  exact tmax passed to ffTRF: {setup.tmax:.6f} s")
    print(f"  CV lambda grid: {np.array2string(setup.regularization_grid, precision=3)}")
    print(f"  CV folds: {setup.k_folds}")
    print(f"  CV seed: {setup.cv_seed}")
    print("  regularization selection metric: neg_mse")
    print(
        f"  isolated benchmark runs per toolbox: {args.repeats} "
        f"(warmup: {args.warmup})"
    )
    print("Forward comparison")
    print(f"  ffTRF selected lambda: {fftrf_model.regularization}")
    print(f"  mTRF selected lambda: {mtrf_model.regularization}")
    print(f"  ffTRF CV scores (neg MSE): {np.array2string(np.asarray(fftrf_cv_scores), precision=4)}")
    print(f"  mTRF CV scores (neg MSE): {np.array2string(np.asarray(mtrf_cv_scores), precision=4)}")
    print(f"  reference channel for prediction trace: {channel_index + 1}")
    print(f"  ffTRF mean held-out channel correlation: {float(fftrf_scores.mean()):.4f}")
    print(f"  mTRF mean held-out channel correlation: {float(mtrf_scores.mean()):.4f}")
    print(f"  ffTRF median held-out channel correlation: {float(np.median(fftrf_scores)):.4f}")
    print(f"  mTRF median held-out channel correlation: {float(np.median(mtrf_scores)):.4f}")
    print(f"  ffTRF isolated benchmark CV fit time: {median(fftrf_durations):.4f} s")
    print(f"  mTRF isolated benchmark CV fit time: {median(mtrf_durations):.4f} s")
    print(f"  ffTRF peak RSS: {median(fftrf_peak_memories):.1f} MiB")
    print(f"  mTRF peak RSS: {median(mtrf_peak_memories):.1f} MiB")

    if not args.skip_backward:
        print("Backward comparison")
        print(
            "  target: broadband envelope proxy built from the mean raw "
            "16-band stimulus and compressed with exponent "
            f"{float(args.backward_envelope_compression):.3f}"
        )
        print(
            f"  lag window: {backward_n_lags} samples "
            f"(0.000000 to {float(backward_tmax):.6f} s)"
        )
        print(
            f"  CV lambda grid: {np.array2string(np.asarray(backward_regularization_grid), precision=3)}"
        )
        print(f"  CV folds: {backward_k_folds}")
        print(
            "  ffTRF backward spectral settings: "
            f"segment_duration={float(args.backward_segment_duration):.3f} s, "
            f"overlap={float(args.backward_overlap):.2f}, "
            f"window={backward_window if backward_window is not None else 'none'}"
        )
        print(f"  ffTRF selected lambda: {backward_fftrf_model.regularization}")
        print(f"  mTRF selected lambda: {backward_mtrf_model.regularization}")
        print(
            f"  ffTRF CV scores (neg MSE): {np.array2string(np.asarray(backward_fftrf_cv_scores), precision=4)}"
        )
        print(
            f"  mTRF CV scores (neg MSE): {np.array2string(np.asarray(backward_mtrf_cv_scores), precision=4)}"
        )
        print(f"  reference held-out segment for envelope trace: {backward_trial_index + 1}")
        print(f"  ffTRF mean held-out segment correlation: {float(np.mean(backward_scores)):.4f}")
        print(f"  mTRF mean held-out segment correlation: {float(np.mean(backward_mtrf_scores)):.4f}")
        print(f"  ffTRF median held-out segment correlation: {float(np.median(backward_scores)):.4f}")
        print(
            f"  mTRF median held-out segment correlation: {float(np.median(backward_mtrf_scores)):.4f}"
        )
        print(
            f"  ffTRF isolated benchmark CV fit time: {median(backward_fftrf_durations):.4f} s"
        )
        print(
            f"  mTRF isolated benchmark CV fit time: {median(backward_mtrf_durations):.4f} s"
        )
        print(f"  ffTRF peak RSS: {median(backward_fftrf_peak_memories):.1f} MiB")
        print(f"  mTRF peak RSS: {median(backward_mtrf_peak_memories):.1f} MiB")

    print(f"  saved figure: {OUTPUT_PATH}")

    plt = require_matplotlib()
    if args.skip_backward:
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
        axes = np.asarray(axes, dtype=object)
        plot_prediction_trace(
            axes[0],
            times_seconds=times_seconds,
            observed=observed_channel,
            fftrf_prediction=fftrf_channel,
            mtrf_prediction=mtrf_channel,
            title=f"Held-out EEG channel {channel_index + 1}",
        )
        plot_prediction_trace(
            axes[1],
            times_seconds=times_seconds,
            observed=observed_gfp,
            fftrf_prediction=fftrf_gfp,
            mtrf_prediction=mtrf_gfp,
            title="Held-out global field power",
            ylabel="GFP (z)",
        )
        plot_score_distribution(
            axes[2],
            fftrf_scores=fftrf_scores,
            mtrf_scores=mtrf_scores,
            title="Held-out channel correlations",
            xlabel="Channel rank",
        )
        axes[0].legend(loc="upper right", frameon=False)
        axes[2].text(
            0.02,
            0.98,
            (
                f"mean / median r\n"
                f"ffTRF: {float(fftrf_scores.mean()):.3f} / {float(np.median(fftrf_scores)):.3f}\n"
                f"mTRF: {float(mtrf_scores.mean()):.3f} / {float(np.median(mtrf_scores)):.3f}\n"
                f"lambda_ffTRF: {float(fftrf_model.regularization):.0f}\n"
                f"lambda_mTRF: {float(mtrf_model.regularization):.0f}"
            ),
            transform=axes[2].transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.9},
        )
        fig.suptitle(
            "Official mTRF speech EEG sample: held-out prediction traces and channel-wise accuracy"
        )
    else:
        fig, axes = plt.subplots(2, 3, figsize=(16, 9.0), constrained_layout=True)
        axes = np.asarray(axes, dtype=object)
        plot_prediction_trace(
            axes[0, 0],
            times_seconds=times_seconds,
            observed=observed_channel,
            fftrf_prediction=fftrf_channel,
            mtrf_prediction=mtrf_channel,
            title=f"Forward: held-out EEG channel {channel_index + 1}",
        )
        plot_prediction_trace(
            axes[0, 1],
            times_seconds=times_seconds,
            observed=observed_gfp,
            fftrf_prediction=fftrf_gfp,
            mtrf_prediction=mtrf_gfp,
            title="Forward: held-out global field power",
            ylabel="GFP (z)",
        )
        plot_score_distribution(
            axes[0, 2],
            fftrf_scores=fftrf_scores,
            mtrf_scores=mtrf_scores,
            title="Forward: held-out channel correlations",
            xlabel="Channel rank",
        )
        axes[0, 2].text(
            0.02,
            0.98,
            (
                f"mean / median r\n"
                f"ffTRF: {float(fftrf_scores.mean()):.3f} / {float(np.median(fftrf_scores)):.3f}\n"
                f"mTRF: {float(mtrf_scores.mean()):.3f} / {float(np.median(mtrf_scores)):.3f}\n"
                f"lambda_ffTRF: {float(fftrf_model.regularization):.0f}\n"
                f"lambda_mTRF: {float(mtrf_model.regularization):.0f}"
            ),
            transform=axes[0, 2].transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.9},
        )
        plot_prediction_trace(
            axes[1, 0],
            times_seconds=backward_times_seconds,
            observed=backward_observed[: len(backward_times_seconds), 0],
            fftrf_prediction=fftrf_backward_trial[: len(backward_times_seconds), 0],
            mtrf_prediction=mtrf_backward_trial[: len(backward_times_seconds), 0],
            title=f"Backward: held-out envelope segment {backward_trial_index + 1}",
            ylabel="Envelope (z)",
        )
        plot_score_distribution(
            axes[1, 1],
            fftrf_scores=backward_scores,
            mtrf_scores=backward_mtrf_scores,
            title="Backward: held-out segment correlations",
            xlabel="Test-segment rank",
        )
        axes[1, 1].text(
            0.02,
            0.98,
            (
                f"mean / median r\n"
                f"ffTRF: {float(np.mean(backward_scores)):.3f} / {float(np.median(backward_scores)):.3f}\n"
                f"mTRF: {float(np.mean(backward_mtrf_scores)):.3f} / {float(np.median(backward_mtrf_scores)):.3f}\n"
                f"lambda_ffTRF: {float(backward_fftrf_model.regularization):.4g}\n"
                f"lambda_mTRF: {float(backward_mtrf_model.regularization):.4g}"
            ),
            transform=axes[1, 1].transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.9},
        )
        plot_summary_bars(
            axes[1, 2],
            forward_fftrf_scores=fftrf_scores,
            forward_mtrf_scores=mtrf_scores,
            backward_fftrf_scores=backward_scores,
            backward_mtrf_scores=backward_mtrf_scores,
        )
        axes[0, 0].legend(loc="upper right", frameon=False)
        axes[1, 0].legend(loc="upper right", frameon=False)
        axes[1, 2].legend(loc="upper right", frameon=False)
        fig.suptitle(
            "Official mTRF speech EEG sample: forward benchmark plus backward envelope reconstruction"
        )

    finalize_figure(fig, output_path=OUTPUT_PATH, show=False)


def _coerce_trial_list(prediction) -> list[np.ndarray]:
    if isinstance(prediction, list):
        return [np.asarray(trial, dtype=float) for trial in prediction]
    return [np.asarray(prediction, dtype=float)]


def _ensure_2d_column_array(x: np.ndarray) -> np.ndarray:
    array = np.asarray(x, dtype=float)
    if array.ndim == 1:
        return array[:, np.newaxis]
    return array


def _zscore_columns(x: np.ndarray) -> np.ndarray:
    """Z-score a 2D array column-wise with safe zero-variance handling."""

    x = np.asarray(x, dtype=float)
    centered = x - x.mean(axis=0, keepdims=True)
    scale = np.clip(centered.std(axis=0, keepdims=True), np.finfo(float).eps, None)
    return centered / scale


def _backward_envelope_target(
    trial: np.ndarray,
    *,
    compression_exponent: float = 0.4,
) -> np.ndarray:
    """Build a compressed broadband envelope proxy from raw stimulus bands."""

    mean_band_signal = np.mean(np.asarray(trial, dtype=float), axis=1, keepdims=True)
    compressed = np.power(
        np.clip(mean_band_signal, 0.0, None),
        float(compression_exponent),
    )
    return _zscore_columns(compressed)

if __name__ == "__main__":
    main()
