#!/usr/bin/env python3
"""Generate the real-data figures used in the documentation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

from fftrf import TRF

EXAMPLES_DIR = Path(__file__).resolve().parent
ROOT = EXAMPLES_DIR.parent
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from example_mtrf_sample_eeg import (  # noqa: E402
    feature_correlations,
    fit_fftrf,
    fit_mtrf,
    load_comparison_setup,
    trial_correlations,
    _coerce_trial_list,
    _ensure_2d_column_array,
)
from mtrf_sample_data import exact_lag_window_seconds  # noqa: E402
from simulated_data import (  # noqa: E402
    finalize_figure,
    require_matplotlib,
)


DOC_IMAGE_DIR = ROOT / "docs" / "images" / "examples"
ARTIFACT_IMAGE_DIR = ROOT / "artifacts" / "examples" / "documentation"


@dataclass(slots=True)
class ForwardFits:
    """Forward real-EEG fits shared across documentation figures."""

    setup: object
    mtrf_matched: object
    fftrf_practical: TRF
    fftrf_scores: np.ndarray
    mtrf_scores: np.ndarray


def main() -> None:
    """Generate all documentation figures."""

    plt = require_matplotlib()
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 180,
            "axes.spines.top": True,
            "axes.spines.right": True,
        }
    )

    DOC_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    setup = load_comparison_setup()
    forward = fit_forward_models(setup)

    save_figure(plot_forward_kernel_comparison(forward), "real_eeg_forward_kernels.png")
    save_figure(plot_kernel_agreement(forward), "real_eeg_kernel_agreement.png")
    save_figure(plot_real_eeg_frequency_resolved(forward), "real_eeg_frequency_resolved.png")
    save_figure(plot_backward_model(setup), "real_eeg_backward_model.png")


def fit_forward_models(setup) -> ForwardFits:
    """Fit the practical ffTRF forward model and the mTRF reference."""

    print("Fitting matched forward mTRF model...")
    mtrf_matched, _ = fit_mtrf(
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

    print("Fitting practical 2 s Hann ffTRF forward model...")
    fftrf_practical, _ = fit_fftrf(
        setup.train_stimulus,
        setup.train_response,
        fs=setup.fs,
        tmin=setup.tmin,
        tmax=setup.tmax,
        regularization=setup.regularization_grid,
        k=setup.k_folds,
        seed=setup.cv_seed,
        direction=1,
        segment_duration=2.0,
        overlap=0.5,
        window="hann",
    )

    fftrf_prediction, _ = fftrf_practical.predict(
        stimulus=setup.test_stimulus,
        response=setup.test_response,
        average=False,
    )
    mtrf_prediction, _ = mtrf_matched.predict(
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
    return ForwardFits(
        setup=setup,
        mtrf_matched=mtrf_matched,
        fftrf_practical=fftrf_practical,
        fftrf_scores=fftrf_scores,
        mtrf_scores=mtrf_scores,
    )


def plot_forward_kernel_comparison(forward: ForwardFits):
    """Plot practical ffTRF forward kernels against the mTRF reference."""

    plt = require_matplotlib()
    ff_weights = np.asarray(forward.fftrf_practical.weights, dtype=float)
    mtrf_weights = np.asarray(forward.mtrf_matched.weights, dtype=float)
    times_ms = np.asarray(forward.fftrf_practical.times, dtype=float) * 1e3

    ff_rms = np.sqrt(np.mean(ff_weights**2, axis=0))
    mtrf_rms = np.sqrt(np.mean(mtrf_weights**2, axis=0))
    ff_rms_normalized = ff_rms / _robust_abs_limit(ff_rms)
    mtrf_rms_normalized = mtrf_rms / _robust_abs_limit(mtrf_rms)
    diff_normalized = ff_rms_normalized - mtrf_rms_normalized
    eeg_channels = np.arange(1, ff_rms.shape[1] + 1)

    reference_channel = int(np.argmax(0.5 * (forward.fftrf_scores + forward.mtrf_scores)))
    reference_input = int(np.argmax(np.linalg.norm(ff_weights[:, :, reference_channel], axis=1)))

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.0), constrained_layout=True)
    axes = np.asarray(axes, dtype=object)

    _image(
        axes[0, 0],
        ff_rms_normalized.T,
        x=times_ms,
        y=eeg_channels,
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        title="ffTRF 2 s Hann forward kernel RMS, normalized",
        xlabel="Lag (ms)",
        ylabel="EEG channel",
        colorbar_label="Normalized RMS",
    )
    _image(
        axes[0, 1],
        mtrf_rms_normalized.T,
        x=times_ms,
        y=eeg_channels,
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        title="mTRF forward kernel RMS, normalized",
        xlabel="Lag (ms)",
        ylabel="EEG channel",
        colorbar_label="Normalized RMS",
    )
    diff_limit = _robust_abs_limit(diff_normalized)
    _image(
        axes[1, 0],
        diff_normalized.T,
        x=times_ms,
        y=eeg_channels,
        cmap="RdBu_r",
        vmin=-diff_limit,
        vmax=diff_limit,
        title="Normalized RMS difference",
        xlabel="Lag (ms)",
        ylabel="EEG channel",
        colorbar_label="ffTRF 2 s Hann - mTRF",
    )

    ax = axes[1, 1]
    ff_curve = _normalize_curve(ff_weights[reference_input, :, reference_channel])
    mtrf_curve = _normalize_curve(mtrf_weights[reference_input, :, reference_channel])
    ax.plot(
        times_ms,
        ff_curve,
        color="#0B6E4F",
        linewidth=2.0,
        label="ffTRF 2 s Hann",
    )
    ax.plot(
        times_ms,
        mtrf_curve,
        color="#3366CC",
        linewidth=1.8,
        label="mTRF",
    )
    ax.axhline(0.0, color="#888888", linewidth=0.8)
    ax.set_title(f"Shape-normalized kernel: band {reference_input + 1}, channel {reference_channel + 1}")
    ax.set_xlabel("Lag (ms)")
    ax.set_ylabel("Normalized weight")
    ax.grid(alpha=0.2, linewidth=0.6)
    ax.legend(frameon=False)
    fig.suptitle("Real speech EEG forward model: 2 s Hann ffTRF and mTRF kernels")
    return fig


def plot_kernel_agreement(forward: ForwardFits):
    """Plot channel-wise agreement between practical ffTRF and mTRF kernels."""

    plt = require_matplotlib()
    ff_weights = np.asarray(forward.fftrf_practical.weights, dtype=float)
    mtrf_weights = np.asarray(forward.mtrf_matched.weights, dtype=float)
    times_ms = np.asarray(forward.fftrf_practical.times, dtype=float) * 1e3

    correlations = _kernel_channel_correlations(ff_weights, mtrf_weights)
    representative_channel = 79
    representative_input = 10
    if representative_channel >= ff_weights.shape[-1]:
        raise IndexError("The requested example channel 80 is not available.")
    if representative_input >= ff_weights.shape[0]:
        raise IndexError("The requested example stimulus band 11 is not available.")

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)

    channel_index = np.arange(1, correlations.shape[0] + 1)
    axes[0].plot(channel_index, correlations, color="#C84C09", linewidth=1.5)
    axes[0].scatter(
        [representative_channel + 1],
        [correlations[representative_channel]],
        color="#0B6E4F",
        s=42,
        zorder=3,
    )
    axes[0].annotate(
        "channel 80",
        xy=(representative_channel + 1, correlations[representative_channel]),
        xytext=(8, -16),
        textcoords="offset points",
        color="#0B6E4F",
        fontsize=9,
    )
    axes[0].axhline(np.nanmedian(correlations), color="#111111", linestyle="--", linewidth=1.0)
    axes[0].set_title("Kernel correlation by EEG channel")
    axes[0].set_xlabel("EEG channel")
    axes[0].set_ylabel("Kernel correlation")
    axes[0].set_ylim(0.85, 1.0)

    axes[1].plot(
        times_ms,
        _normalize_curve(ff_weights[representative_input, :, representative_channel]),
        color="#0B6E4F",
        linewidth=2.0,
        label="ffTRF 2 s Hann",
    )
    axes[1].plot(
        times_ms,
        _normalize_curve(mtrf_weights[representative_input, :, representative_channel]),
        color="#3366CC",
        linewidth=1.8,
        label="mTRF",
    )
    axes[1].axhline(0.0, color="#888888", linewidth=0.8)
    axes[1].set_title("Example kernel: channel 80, stimulus band 11")
    axes[1].set_xlabel("Lag (ms)")
    axes[1].set_ylabel("Normalized weight")
    axes[1].legend(frameon=False)

    for ax in axes:
        ax.grid(alpha=0.2, linewidth=0.6)
    fig.suptitle("Real speech EEG forward model: 2 s Hann ffTRF/mTRF kernel agreement")
    return fig


def plot_real_eeg_frequency_resolved(forward: ForwardFits):
    """Plot frequency-resolved weights from the practical real EEG ffTRF fit."""

    plt = require_matplotlib()
    model = forward.fftrf_practical
    resolved = model.frequency_resolved_weights(
        n_bands=24,
        fmax=40.0,
        value_mode="real",
    )
    power = model.time_frequency_power(
        n_bands=24,
        fmax=40.0,
    )
    weights = np.asarray(model.weights, dtype=float)
    input_index, output_index = _select_frequency_resolved_pair(resolved)

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(11, 9),
        gridspec_kw={"height_ratios": [1.0, 1.35, 1.35]},
        constrained_layout=True,
    )

    axes[0].plot(
        model.times * 1e3,
        weights[input_index, :, output_index],
        color="#0B6E4F",
        linewidth=2.0,
    )
    axes[0].axhline(0.0, color="#888888", linewidth=0.8)
    axes[0].set_title(
        f"Real EEG ffTRF kernel, stimulus band {input_index + 1}, channel {output_index + 1}"
    )
    axes[0].set_xlabel("Lag (ms)")
    axes[0].set_ylabel("Weight")
    axes[0].grid(alpha=0.2, linewidth=0.6)

    resolved_map = resolved.at(input_index=input_index, output_index=output_index)
    resolved_limit = _robust_abs_limit(resolved_map, percentile=99.0)
    _image(
        axes[1],
        resolved_map,
        x=resolved.times * 1e3,
        y=resolved.band_centers,
        cmap="RdBu_r",
        vmin=-resolved_limit,
        vmax=resolved_limit,
        title="Frequency-resolved signed weights",
        xlabel="Lag (ms)",
        ylabel="Frequency (Hz)",
        colorbar_label="Weight",
    )
    power_map = power.at(input_index=input_index, output_index=output_index)
    _image(
        axes[2],
        power_map,
        x=power.times * 1e3,
        y=power.band_centers,
        cmap="magma",
        vmin=0.0,
        vmax=np.nanpercentile(power_map, 99.0),
        title="Time-frequency power of the fitted kernel",
        xlabel="Lag (ms)",
        ylabel="Frequency (Hz)",
        colorbar_label="Power",
    )
    fig.suptitle("Real speech EEG example: frequency-resolved ffTRF kernel")
    return fig


def plot_backward_model(setup):
    """Plot one held-out backward reconstruction for ffTRF and mTRF."""

    plt = require_matplotlib()
    _, backward_tmax = exact_lag_window_seconds(fs=setup.fs, nominal_stop_seconds=0.35)
    regularization_grid = np.logspace(-8, 6, 15)

    print("Fitting practical 2 s Hann backward ffTRF model...")
    fftrf_model, _ = fit_fftrf(
        setup.backward_train_stimulus,
        setup.train_response,
        fs=setup.fs,
        tmin=0.0,
        tmax=backward_tmax,
        regularization=regularization_grid,
        direction=-1,
        k=3,
        seed=setup.cv_seed,
        segment_duration=2.0,
        overlap=0.5,
        window="hann",
    )
    print("Fitting backward mTRF reference model...")
    mtrf_model, _ = fit_mtrf(
        setup.backward_train_stimulus,
        setup.train_response,
        fs=setup.fs,
        tmin=0.0,
        tmax=backward_tmax,
        regularization=regularization_grid,
        direction=-1,
        k=3,
        seed=setup.cv_seed,
    )
    fftrf_prediction, _ = fftrf_model.predict(
        stimulus=setup.backward_test_stimulus,
        response=setup.test_response,
        average=False,
    )
    mtrf_prediction, _ = mtrf_model.predict(
        stimulus=setup.backward_test_stimulus,
        response=setup.test_response,
        average=False,
    )
    fftrf_scores = trial_correlations(
        observed_trials=setup.backward_test_stimulus,
        predicted_trials=fftrf_prediction,
    )
    mtrf_scores = trial_correlations(
        observed_trials=setup.backward_test_stimulus,
        predicted_trials=mtrf_prediction,
    )
    best_trial = int(np.nanargmax(0.5 * (fftrf_scores + mtrf_scores)))
    observed = _ensure_2d_column_array(setup.backward_test_stimulus[best_trial])
    fftrf_decoded = _ensure_2d_column_array(_coerce_trial_list(fftrf_prediction)[best_trial])
    mtrf_decoded = _ensure_2d_column_array(_coerce_trial_list(mtrf_prediction)[best_trial])
    length = min(observed.shape[0], fftrf_decoded.shape[0], mtrf_decoded.shape[0])
    time = np.arange(length, dtype=float) / float(setup.fs)
    snippet = time <= 6.0
    observed_display = _zscore_vector(observed[:length, 0])
    fftrf_display = _zscore_vector(fftrf_decoded[:length, 0])
    mtrf_display = _zscore_vector(mtrf_decoded[:length, 0])

    fig, ax_trace = plt.subplots(figsize=(12, 4.8), constrained_layout=True)
    ax_trace.plot(time[snippet], observed_display[snippet], color="#111111", linewidth=1.4, label="Observed envelope")
    ax_trace.plot(time[snippet], fftrf_display[snippet], color="#0B6E4F", linewidth=1.35, label="ffTRF 2 s Hann")
    ax_trace.plot(time[snippet], mtrf_display[snippet], color="#3366CC", linewidth=1.25, label="mTRF")
    ax_trace.axhline(0.0, color="#888888", linewidth=0.8)
    ax_trace.set_title(
        f"Held-out backward reconstruction, segment {best_trial + 1} "
        f"(r: ffTRF={fftrf_scores[best_trial]:.3f}, mTRF={mtrf_scores[best_trial]:.3f})"
    )
    ax_trace.set_xlabel("Time (s)")
    ax_trace.set_ylabel("Envelope (z, display-normalized)")
    ax_trace.legend(frameon=False)
    ax_trace.grid(alpha=0.2, linewidth=0.6)
    fig.suptitle("Real speech EEG backward model: response -> compressed envelope")
    return fig


def save_figure(fig, filename: str) -> None:
    """Save a figure to docs and artifact directories."""

    for directory in (DOC_IMAGE_DIR, ARTIFACT_IMAGE_DIR):
        output = directory / filename
        finalize_figure(fig, output_path=output, show=False)
        print(f"saved {output}")
    require_matplotlib().close(fig)


def _select_frequency_resolved_pair(resolved) -> tuple[int, int]:
    band_mask = (resolved.band_centers >= 4.0) & (resolved.band_centers <= 20.0)
    if not np.any(band_mask):
        band_mask = np.ones_like(resolved.band_centers, dtype=bool)
    energy = np.linalg.norm(resolved.weights[:, band_mask, :, :], axis=(1, 2))
    return tuple(int(index) for index in np.unravel_index(np.nanargmax(energy), energy.shape))


def _normalize_curve(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    scale = np.nanmax(np.abs(values))
    if not np.isfinite(scale) or scale <= 0.0:
        return np.zeros_like(values)
    return values / scale


def _zscore_vector(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    centered = values - np.nanmean(values)
    scale = np.nanstd(centered)
    if not np.isfinite(scale) or scale <= 0.0:
        return np.zeros_like(values)
    return centered / scale


def _kernel_channel_correlations(ff_weights: np.ndarray, mtrf_weights: np.ndarray) -> np.ndarray:
    correlations = np.empty(ff_weights.shape[-1], dtype=float)
    for channel_index in range(ff_weights.shape[-1]):
        x = ff_weights[:, :, channel_index].ravel()
        y = mtrf_weights[:, :, channel_index].ravel()
        if np.std(x) == 0.0 or np.std(y) == 0.0:
            correlations[channel_index] = np.nan
        else:
            correlations[channel_index] = float(np.corrcoef(x, y)[0, 1])
    return correlations


def _robust_abs_limit(values: np.ndarray, *, percentile: float = 98.0) -> float:
    values = np.asarray(values, dtype=float)
    finite = np.abs(values[np.isfinite(values)])
    if finite.size == 0:
        return 1.0
    limit = float(np.nanpercentile(finite, percentile))
    return max(limit, np.finfo(float).eps)


def _image(
    ax,
    values: np.ndarray,
    *,
    x: np.ndarray,
    y: np.ndarray,
    cmap: str,
    vmin: float | None,
    vmax: float | None,
    title: str,
    xlabel: str,
    ylabel: str,
    colorbar_label: str,
) -> None:
    values = np.asarray(values, dtype=float)
    x_edges = _axis_edges(np.asarray(x, dtype=float))
    y_edges = _axis_edges(np.asarray(y, dtype=float))
    mesh = ax.pcolormesh(
        x_edges,
        y_edges,
        values,
        shading="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.15, linewidth=0.5)
    ax.figure.colorbar(mesh, ax=ax, label=colorbar_label)


def _axis_edges(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 1:
        return np.asarray([values[0] - 0.5, values[0] + 0.5], dtype=float)
    deltas = np.diff(values)
    first = values[0] - deltas[0] / 2.0
    last = values[-1] + deltas[-1] / 2.0
    return np.concatenate([[first], values[:-1] + deltas / 2.0, [last]])


if __name__ == "__main__":
    main()
