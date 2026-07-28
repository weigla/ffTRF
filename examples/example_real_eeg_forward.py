#!/usr/bin/env python3
"""Focused real-data example: speech features predict held-out EEG."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from mtrf_sample_data import exact_lag_window_seconds, load_sample_data
from simulated_data import finalize_figure, require_matplotlib

from fftrf import TRF, pearsonr

OUTPUT_PATH = Path("artifacts/examples/real_eeg_forward.png")


def main() -> None:
    """Fit a forward model and summarize all held-out EEG channels."""

    # This public sample follows the upstream tutorial convention: each
    # 12-second segment is standardized independently before the model split.
    stimulus, response, fs = load_sample_data(n_segments=10, normalize=True)
    train_stimulus, test_stimulus = stimulus[:7], stimulus[7:]
    train_response, test_response = response[:7], response[7:]
    _, tmax = exact_lag_window_seconds(fs=fs, nominal_stop_seconds=0.4)
    ridge_grid = np.logspace(-4, 4, 17)

    model = TRF(direction=1, metric="neg_mse")
    cv_scores = model.train(
        stimulus=train_stimulus,
        response=train_response,
        fs=fs,
        tmin=0.0,
        tmax=tmax,
        regularization=ridge_grid,
        k=5,
        seed=7,
        segment_duration=2.0,
        overlap=0.5,
        window="hann",
        detrend=None,
    )

    predictions = model.predict(stimulus=test_stimulus)
    heldout_r = np.mean(
        np.vstack(
            [
                pearsonr(observed, predicted)
                for observed, predicted in zip(
                    test_response,
                    predictions,
                    strict=True,
                )
            ]
        ),
        axis=0,
    )

    print("Real speech EEG: forward encoding")
    print("  split: 7 training segments, 3 segments held out from fitting and CV")
    print(f"  data shape: 16 stimulus bands -> {response[0].shape[1]} EEG channels")
    print(f"  selected lambda: {float(model.regularization):.6g}")
    print(f"  mean held-out channel r: {float(heldout_r.mean()):.4f}")
    print(f"  median held-out channel r: {float(np.median(heldout_r)):.4f}")
    print(f"  CV score shape: {np.asarray(cv_scores).shape}")
    print(f"  saved figure: {OUTPUT_PATH}")

    plt = require_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    model.plot(
        input_index=0,
        output_index=0,
        ax=axes[0],
        title="Band 1 to EEG channel 1",
    )
    axes[1].plot(
        np.arange(1, heldout_r.size + 1),
        np.sort(heldout_r)[::-1],
        color="#0B6E4F",
        marker=".",
        linewidth=1.2,
    )
    axes[1].axhline(0.0, color="#777777", linewidth=0.8)
    axes[1].set(
        title="Held-out correlation across all channels",
        xlabel="Channel rank",
        ylabel="Pearson r",
    )
    axes[1].grid(alpha=0.2, linewidth=0.6)
    finalize_figure(fig, output_path=OUTPUT_PATH, show=False)


if __name__ == "__main__":
    main()
