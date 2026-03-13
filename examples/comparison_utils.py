"""Optional comparison helpers for validating fft_trf against reference fits.

This module intentionally lives outside the installable `fft_trf` package so the
core toolbox stays focused on model fitting and preprocessing. The functions
here are for examples, sanity checks, and development-time comparisons only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from fft_trf import FrequencyTRF


@dataclass(slots=True)
class KernelComparisonResult:
    """Container holding the results of a simulated kernel comparison."""

    fs: float
    tmin: float
    tmax: float
    regularization: float
    times: np.ndarray
    true_kernel: np.ndarray
    fft_trf_kernel: np.ndarray
    time_domain_kernel: np.ndarray
    mtrf_kernel: np.ndarray | None
    metrics: dict[str, float]
    model: FrequencyTRF


def default_kernel(
    *,
    fs: float,
    tmin: float,
    tmax: float,
) -> np.ndarray:
    """Create a simple sparse simulation kernel."""

    lag_start = int(round(tmin * fs))
    lag_stop = int(round(tmax * fs))
    lag_indices = np.arange(lag_start, lag_stop, dtype=int)
    kernel = np.zeros(len(lag_indices), dtype=float)

    peaks = {
        int(round(0.003 * fs)): 1.0,
        int(round(0.009 * fs)): -0.4,
        int(round(0.018 * fs)): 0.2,
    }
    index_lookup = {lag: idx for idx, lag in enumerate(lag_indices)}
    for lag, amplitude in peaks.items():
        if lag in index_lookup:
            kernel[index_lookup[lag]] = amplitude
    return kernel


def simulate_trials(
    *,
    fs: float,
    n_trials: int,
    n_samples: int,
    tmin: float,
    kernel: np.ndarray,
    noise_scale: float = 0.05,
    seed: int = 0,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Simulate continuous stimulus/response pairs from a known kernel."""

    rng = np.random.default_rng(seed)
    lag_start = int(round(tmin * fs))
    stimulus = []
    response = []
    for _ in range(n_trials):
        x = rng.standard_normal((n_samples, 1))
        y = _predict_from_kernel(
            x[:, 0],
            kernel,
            lag_start=lag_start,
            out_length=n_samples,
        )
        y += noise_scale * rng.standard_normal(n_samples)
        stimulus.append(x)
        response.append(y[:, np.newaxis])
    return stimulus, response


def time_domain_ridge_kernel(
    stimulus: list[np.ndarray],
    response: list[np.ndarray],
    *,
    fs: float,
    tmin: float,
    tmax: float,
    regularization: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a direct time-domain lagged ridge regression reference."""

    lags = np.arange(int(round(tmin * fs)), int(round(tmax * fs)), dtype=int)
    n_features = stimulus[0].shape[1]
    n_outputs = response[0].shape[1]

    x_blocks = []
    y_blocks = []
    for x_trial, y_trial in zip(stimulus, response):
        x_blocks.append(_lagged_design_matrix(x_trial, lags))
        y_blocks.append(y_trial)

    x_all = np.vstack(x_blocks)
    y_all = np.vstack(y_blocks)
    ridge = regularization * np.eye(x_all.shape[1])
    weights = np.linalg.solve(x_all.T @ x_all + ridge, x_all.T @ y_all)
    weights = weights.reshape(n_features, len(lags), n_outputs)
    return weights, lags / fs


def compare_simulated_kernels(
    *,
    fs: float = 1_000,
    n_trials: int = 8,
    n_samples: int = 4_096,
    tmin: float = 0.0,
    tmax: float = 0.040,
    regularization: float = 1e-3,
    noise_scale: float = 0.05,
    seed: int = 0,
    segment_length: int | None = None,
    overlap: float = 0.0,
    window: None | str = None,
    include_mtrf: bool = True,
) -> KernelComparisonResult:
    """Run a full synthetic comparison between true, FFT, and mTRF kernels."""

    true_kernel = default_kernel(fs=fs, tmin=tmin, tmax=tmax)
    stimulus, response = simulate_trials(
        fs=fs,
        n_trials=n_trials,
        n_samples=n_samples,
        tmin=tmin,
        kernel=true_kernel,
        noise_scale=noise_scale,
        seed=seed,
    )

    model = FrequencyTRF(direction=1)
    model.train(
        stimulus=stimulus,
        response=response,
        fs=fs,
        tmin=tmin,
        tmax=tmax,
        regularization=regularization,
        segment_length=segment_length,
        overlap=overlap,
        window=window,
    )
    fft_kernel = model.weights[0, :, 0]

    time_weights, times = time_domain_ridge_kernel(
        stimulus,
        response,
        fs=fs,
        tmin=tmin,
        tmax=tmax,
        regularization=regularization,
    )
    time_kernel = time_weights[0, :, 0]

    mtrf_kernel = None
    if include_mtrf:
        mtrf_kernel = fit_mtrf_kernel(
            stimulus,
            response,
            fs=fs,
            tmin=tmin,
            tmax=tmax,
            regularization=regularization,
        )

    metrics = {
        "fft_vs_true": _safe_corrcoef(fft_kernel, true_kernel),
        "time_vs_true": _safe_corrcoef(time_kernel, true_kernel),
        "fft_vs_time": _safe_corrcoef(fft_kernel, time_kernel),
    }
    if mtrf_kernel is not None:
        metrics["mtrf_vs_true"] = _safe_corrcoef(mtrf_kernel, true_kernel)
        metrics["fft_vs_mtrf"] = _safe_corrcoef(fft_kernel, mtrf_kernel)
        metrics["time_vs_mtrf"] = _safe_corrcoef(time_kernel, mtrf_kernel)

    return KernelComparisonResult(
        fs=fs,
        tmin=tmin,
        tmax=tmax,
        regularization=regularization,
        times=times,
        true_kernel=true_kernel,
        fft_trf_kernel=fft_kernel,
        time_domain_kernel=time_kernel,
        mtrf_kernel=mtrf_kernel,
        metrics=metrics,
        model=model,
    )


def fit_mtrf_kernel(
    stimulus: list[np.ndarray],
    response: list[np.ndarray],
    *,
    fs: float,
    tmin: float,
    tmax: float,
    regularization: float,
) -> np.ndarray | None:
    """Fit ``mTRFpy`` if available and return its single-feature kernel."""

    try:
        from mtrf.model import TRF
    except ModuleNotFoundError:
        return None

    tmax_mtrf = tmax - (1.0 / fs)
    model = TRF(direction=1)
    model.train(
        stimulus=stimulus,
        response=response,
        fs=fs,
        tmin=tmin,
        tmax=tmax_mtrf,
        regularization=regularization,
    )
    return np.asarray(model.weights)[0, :, 0] / fs


def plot_kernel_comparison(
    result: KernelComparisonResult,
    *,
    output_path: str | Path | None = None,
    show: bool = True,
) -> Any:
    """Plot the kernel comparison and optionally save it to disk."""

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required for plotting. Install the compare extras "
            "or use the Pixi compare environment."
        ) from exc

    times_ms = result.times * 1e3
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(times_ms, result.true_kernel, label="True kernel", linewidth=2.5, color="#111111")
    ax.plot(times_ms, result.fft_trf_kernel, label="fft_trf", linewidth=2.0, color="#0B6E4F")
    ax.plot(
        times_ms,
        result.time_domain_kernel,
        label="Time-domain ridge",
        linewidth=1.8,
        color="#C84C09",
    )
    if result.mtrf_kernel is not None:
        ax.plot(times_ms, result.mtrf_kernel, label="mTRFpy", linewidth=1.6, color="#3366CC")

    ax.axhline(0.0, color="#999999", linewidth=0.8)
    ax.set_xlabel("Lag (ms)")
    ax.set_ylabel("Kernel weight")
    ax.set_title("Kernel Comparison on Simulated Data")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.2, linewidth=0.6)

    metric_text = "\n".join(
        f"{key}: {value:.3f}"
        for key, value in result.metrics.items()
    )
    ax.text(
        0.01,
        0.99,
        metric_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.9},
    )

    fig.tight_layout()
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def _safe_corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("Correlation inputs must have identical shape.")
    if np.allclose(a, 0.0) or np.allclose(b, 0.0):
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _lagged_design_matrix(signal: np.ndarray, lags: np.ndarray) -> np.ndarray:
    signal = np.asarray(signal, dtype=float)
    if signal.ndim == 1:
        signal = signal[:, np.newaxis]

    n_samples, n_features = signal.shape
    design = np.zeros((n_samples, n_features * len(lags)), dtype=float)

    for feature_index in range(n_features):
        x = signal[:, feature_index]
        for lag_index, lag in enumerate(lags):
            column = feature_index * len(lags) + lag_index
            if lag >= 0:
                design[lag:, column] = x[: n_samples - lag]
            else:
                design[: n_samples + lag, column] = x[-lag:]
    return design


def _predict_from_kernel(
    signal: np.ndarray,
    kernel: np.ndarray,
    *,
    lag_start: int,
    out_length: int,
) -> np.ndarray:
    full = np.convolve(signal, kernel, mode="full")
    offset = -lag_start
    prediction = np.zeros(out_length, dtype=float)
    src_start = max(offset, 0)
    dst_start = max(-offset, 0)
    length = min(full.shape[0] - src_start, out_length - dst_start)
    if length > 0:
        prediction[dst_start : dst_start + length] = full[src_start : src_start + length]
    return prediction
