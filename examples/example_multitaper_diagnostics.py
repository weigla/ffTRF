#!/usr/bin/env python3
"""Example: multi-taper fitting with transfer-function and coherence diagnostics."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from fftrf import FrequencyTRF, pearsonr

from simulated_data import (
    build_multi_trial_single_channel_dataset,
    finalize_figure,
    require_matplotlib,
)

OUTPUT_PATH = Path("artifacts/examples/multitaper_diagnostics.png")


def main() -> None:
    """Fit a multi-taper model and visualize spectral diagnostics."""

    dataset = build_multi_trial_single_channel_dataset()
    train_stimulus = dataset.stimulus[:-1]
    train_response = dataset.response[:-1]
    test_stimulus = dataset.stimulus[-1]
    test_response = dataset.response[-1]

    model = FrequencyTRF(direction=1, metric="r2")
    cv_scores = model.train(
        stimulus=train_stimulus,
        response=train_response,
        fs=dataset.fs,
        tmin=dataset.tmin,
        tmax=dataset.tmax,
        regularization=np.logspace(-5, -1, 5),
        segment_length=1_024,
        overlap=0.5,
        spectral_method="multitaper",
        time_bandwidth=3.5,
        n_tapers=4,
        k=4,
    )
    prediction, held_out_r2 = model.predict(
        stimulus=test_stimulus,
        response=test_response,
    )
    held_out_corr = float(pearsonr(test_response, prediction).mean())
    diagnostics = model.diagnostics(
        stimulus=test_stimulus,
        response=test_response,
    )

    print("Example: multi-taper diagnostics")
    print(f"  description: {dataset.description}")
    print(f"  metric: {model.metric_name}")
    print(f"  selected lambda: {model.regularization}")
    print(f"  held-out R^2: {float(held_out_r2):.4f}")
    print(f"  held-out correlation: {held_out_corr:.4f}")
    print(f"  spectral method: {model.spectral_method}")
    print(f"  time_bandwidth: {model.time_bandwidth}")
    print(f"  n_tapers: {model.n_tapers}")
    print(f"  mean low-frequency coherence: {float(np.mean(diagnostics.coherence[:40, 0])):.4f}")
    print(f"  CV scores: {np.array2string(np.asarray(cv_scores), precision=4)}")
    print(f"  saved figure: {OUTPUT_PATH}")

    plt = require_matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))

    axes[0, 0].plot(
        dataset.times * 1e3,
        dataset.true_weights[0, :, 0],
        color="#111111",
        linewidth=1.6,
        linestyle="--",
        label="True kernel",
    )
    axes[0, 0].plot(
        model.times * 1e3,
        model.weights[0, :, 0],
        color="#3366CC",
        linewidth=1.8,
        label="Recovered kernel",
    )
    axes[0, 0].set_title("Recovered kernel")
    axes[0, 0].set_xlabel("Lag (ms)")
    axes[0, 0].set_ylabel("Weight")
    axes[0, 0].legend(loc="upper right", frameon=False)
    axes[0, 0].grid(alpha=0.2, linewidth=0.6)

    model.plot_coherence(
        diagnostics=diagnostics,
        ax=axes[0, 1],
        color="#0B6E4F",
        title="Observed vs predicted coherence",
    )

    model.plot_transfer_function(
        ax=np.asarray([axes[1, 0], axes[1, 1]], dtype=object),
        color="#C84C09",
        phase_color="#6A994E",
        phase_unit="deg",
        title="Learned transfer function",
    )

    fig.tight_layout()
    finalize_figure(fig, output_path=OUTPUT_PATH, show=False)


if __name__ == "__main__":
    main()
