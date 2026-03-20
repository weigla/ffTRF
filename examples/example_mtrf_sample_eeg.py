#!/usr/bin/env python3
"""Example: compare ffTRF to mTRFpy on the public speech EEG sample."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np

from fftrf import TRF

from mtrf_sample_data import exact_lag_window_seconds, load_sample_data
from simulated_data import finalize_figure, require_matplotlib

OUTPUT_PATH = Path("artifacts/examples/mtrf_sample_eeg_comparison.png")


def fit_fftrf(
    stimulus: list[np.ndarray],
    response: list[np.ndarray],
    *,
    fs: int,
    tmin: float,
    tmax: float,
    regularization: float | list[float] | np.ndarray,
    k: int,
    seed: int,
) -> tuple[TRF, np.ndarray | float | None]:
    """Fit the ffTRF forward model with matched lag settings."""

    model = TRF(direction=1, metric="pearsonr")
    cv_scores = model.train(
        stimulus=stimulus,
        response=response,
        fs=fs,
        tmin=tmin,
        tmax=tmax,
        regularization=regularization,
        segment_length=None,
        window=None,
        k=k,
        seed=seed,
    )
    return model, cv_scores


def fit_mtrf(
    stimulus: list[np.ndarray],
    response: list[np.ndarray],
    *,
    fs: int,
    tmin: float,
    tmax: float,
    regularization: float | list[float] | np.ndarray,
    k: int,
    seed: int,
):
    """Fit the reference mTRFpy forward model with matched lag settings."""

    try:
        from mtrf.model import TRF
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "mTRFpy is required for this example. Use the compare extras or "
            "the Pixi compare environment."
        ) from exc

    model = TRF(direction=1)
    cv_scores = model.train(
        stimulus=stimulus,
        response=response,
        fs=fs,
        tmin=tmin,
        tmax=tmax - (1.0 / fs),
        regularization=regularization,
        k=k,
        seed=seed,
        verbose=False,
    )
    return model, cv_scores


def timed_fit(fit_callable):
    """Return the fit result together with one wall-clock training time."""

    start = perf_counter()
    result = fit_callable()
    return result, perf_counter() - start


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
        label="mTRFpy",
    )
    ax.axhline(0.0, color="#999999", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.2, linewidth=0.6)


def plot_channel_score_distribution(
    ax,
    *,
    fftrf_scores: np.ndarray,
    mtrf_scores: np.ndarray,
) -> None:
    """Plot sorted held-out channel correlations for both toolboxes."""

    channel_order = np.argsort(0.5 * (fftrf_scores + mtrf_scores))[::-1]
    rank = np.arange(1, len(channel_order) + 1)
    fftrf_sorted = fftrf_scores[channel_order]
    mtrf_sorted = mtrf_scores[channel_order]

    ax.plot(rank, fftrf_sorted, color="#0B6E4F", linewidth=1.8, label="ffTRF")
    ax.plot(rank, mtrf_sorted, color="#3366CC", linewidth=1.6, label="mTRFpy")
    ax.fill_between(
        rank,
        fftrf_sorted,
        mtrf_sorted,
        color="#C84C09",
        alpha=0.12,
        linewidth=0.0,
    )
    ax.axhline(0.0, color="#999999", linewidth=0.8)
    ax.set_title("Held-out channel correlations")
    ax.set_xlabel("Channel rank")
    ax.set_ylabel("Pearson r")
    ax.grid(alpha=0.2, linewidth=0.6)


def main() -> None:
    """Fit both toolboxes on the public speech EEG example and compare them."""

    stimulus, response, fs = load_sample_data(n_segments=10, normalize=True)
    train_stimulus = stimulus[:-3]
    train_response = response[:-3]
    test_stimulus = stimulus[-3:]
    test_response = response[-3:]

    regularization_grid = np.logspace(-4, 4, 17)
    k_folds = 5
    cv_seed = 7
    tmin = 0.0
    n_lags, tmax = exact_lag_window_seconds(fs=fs, nominal_stop_seconds=0.4)

    (fftrf_model, fftrf_cv_scores), fftrf_seconds = timed_fit(
        lambda: fit_fftrf(
            train_stimulus,
            train_response,
            fs=fs,
            tmin=tmin,
            tmax=tmax,
            regularization=regularization_grid,
            k=k_folds,
            seed=cv_seed,
        )
    )
    (mtrf_model, mtrf_cv_scores), mtrf_seconds = timed_fit(
        lambda: fit_mtrf(
            train_stimulus,
            train_response,
            fs=fs,
            tmin=tmin,
            tmax=tmax,
            regularization=regularization_grid,
            k=k_folds,
            seed=cv_seed,
        )
    )

    fftrf_prediction, fftrf_scores = fftrf_model.predict(
        stimulus=test_stimulus,
        response=test_response,
        average=False,
    )
    mtrf_prediction, mtrf_scores = mtrf_model.predict(
        stimulus=test_stimulus,
        response=test_response,
        average=False,
    )

    fftrf_scores = np.asarray(fftrf_scores, dtype=float)
    mtrf_scores = np.asarray(mtrf_scores, dtype=float)
    channel_index = int(np.argmax(0.5 * (fftrf_scores + mtrf_scores)))

    observed_segment = np.asarray(test_response[0], dtype=float)
    fftrf_segment = np.asarray(fftrf_prediction[0], dtype=float)
    mtrf_segment = np.asarray(mtrf_prediction[0], dtype=float)
    times_seconds = np.arange(observed_segment.shape[0], dtype=float) / float(fs)

    observed_channel = observed_segment[:, channel_index]
    fftrf_channel = fftrf_segment[:, channel_index]
    mtrf_channel = mtrf_segment[:, channel_index]
    observed_gfp = global_field_power(observed_segment)
    fftrf_gfp = global_field_power(fftrf_segment)
    mtrf_gfp = global_field_power(mtrf_segment)

    print("Example: public mTRF speech EEG sample")
    print("  dataset: 16-band speech spectrogram to 128-channel EEG")
    print(f"  segments: {len(stimulus)} total, {len(train_stimulus)} train, {len(test_stimulus)} test")
    print(f"  sampling rate: {fs} Hz")
    print(f"  lag window: {n_lags} samples ({fftrf_model.times[0]:.6f} to {fftrf_model.times[-1]:.6f} s)")
    print(f"  exact tmax passed to ffTRF: {tmax:.6f} s")
    print(f"  CV lambda grid: {np.array2string(regularization_grid, precision=3)}")
    print(f"  CV folds: {k_folds}")
    print(f"  CV seed: {cv_seed}")
    print(f"  ffTRF selected lambda: {fftrf_model.regularization}")
    print(f"  mTRFpy selected lambda: {mtrf_model.regularization}")
    print(f"  ffTRF CV scores: {np.array2string(np.asarray(fftrf_cv_scores), precision=4)}")
    print(f"  mTRFpy CV scores: {np.array2string(np.asarray(mtrf_cv_scores), precision=4)}")
    print(f"  reference channel for prediction trace: {channel_index + 1}")
    print(f"  ffTRF mean held-out channel correlation: {float(fftrf_scores.mean()):.4f}")
    print(f"  mTRFpy mean held-out channel correlation: {float(mtrf_scores.mean()):.4f}")
    print(f"  ffTRF median held-out channel correlation: {float(np.median(fftrf_scores)):.4f}")
    print(f"  mTRFpy median held-out channel correlation: {float(np.median(mtrf_scores)):.4f}")
    print(f"  ffTRF CV fit time: {fftrf_seconds:.4f} s")
    print(f"  mTRFpy CV fit time: {mtrf_seconds:.4f} s")
    print(f"  saved figure: {OUTPUT_PATH}")

    plt = require_matplotlib()
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
    plot_channel_score_distribution(
        axes[2],
        fftrf_scores=fftrf_scores,
        mtrf_scores=mtrf_scores,
    )
    axes[0].legend(loc="upper right", frameon=False)
    axes[2].text(
        0.02,
        0.98,
        (
            f"mean / median r\n"
            f"ffTRF: {float(fftrf_scores.mean()):.3f} / {float(np.median(fftrf_scores)):.3f}\n"
            f"mTRFpy: {float(mtrf_scores.mean()):.3f} / {float(np.median(mtrf_scores)):.3f}\n"
            f"λ_ffTRF: {float(fftrf_model.regularization):.0f}\n"
            f"λ_mTRF: {float(mtrf_model.regularization):.0f}"
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
    finalize_figure(fig, output_path=OUTPUT_PATH, show=False)


if __name__ == "__main__":
    main()
