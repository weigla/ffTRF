#!/usr/bin/env python3
"""Example: optional multi-taper estimation with spectral diagnostics."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from fftrf import TRF, pearsonr

from simulated_data import (
    build_multi_trial_single_channel_dataset,
    finalize_figure,
    require_matplotlib,
)

OUTPUT_PATH = Path("artifacts/examples/multitaper_estimator.png")


def main() -> None:
    """Fit a multi-taper model and visualize transfer/cross-spectral outputs."""

    dataset = build_multi_trial_single_channel_dataset()
    train_stimulus = dataset.stimulus[:-1]
    train_response = dataset.response[:-1]
    test_stimulus = dataset.stimulus[-1]
    test_response = dataset.response[-1]

    model = TRF(direction=1, metric="r2")
    cv_scores = model.train_multitaper(
        stimulus=train_stimulus,
        response=train_response,
        fs=dataset.fs,
        tmin=dataset.tmin,
        tmax=dataset.tmax,
        regularization=np.logspace(-5, -1, 5),
        segment_duration=1.024,
        overlap=0.5,
        time_bandwidth=3.5,
        n_tapers=4,
        k="loo",
        show_progress=True,
    )
    prediction, held_out_r2 = model.predict(
        stimulus=test_stimulus,
        response=test_response,
    )
    held_out_corr = float(pearsonr(test_response, prediction).mean())
    diagnostics = model.cross_spectral_diagnostics(
        stimulus=test_stimulus,
        response=test_response,
    )

    print("Example: optional multi-taper estimator")
    print(f"  description: {dataset.description}")
    print(f"  metric: {model.metric_name}")
    print(f"  selected lambda: {model.regularization}")
    print(f"  held-out R^2: {float(held_out_r2):.4f}")
    print(f"  held-out correlation: {held_out_corr:.4f}")
    print(f"  spectral method: {model.spectral_method}")
    print(f"  segment_duration: {model.segment_duration}")
    print(f"  time_bandwidth: {model.time_bandwidth}")
    print(f"  n_tapers: {model.n_tapers}")
    print(f"  mean low-frequency coherence: {float(np.mean(diagnostics.coherence[:40, 0])):.4f}")
    print(f"  CV scores: {np.array2string(np.asarray(cv_scores), precision=4)}")
    print(f"  saved figure: {OUTPUT_PATH}")

    plt = require_matplotlib()
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))

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
        ax=axes[1, 0],
        color="#0B6E4F",
        title="Observed vs predicted coherence",
    )

    model.plot_cross_spectrum(
        diagnostics=diagnostics,
        kind="magnitude",
        ax=axes[2, 0],
        color="#7B2CBF",
        title="Predicted vs observed cross-spectrum magnitude",
    )

    model.plot_transfer_function(
        kind="all",
        ax=np.asarray([axes[0, 1], axes[1, 1], axes[2, 1]], dtype=object),
        color="#C84C09",
        phase_color="#6A994E",
        group_delay_color="#1D6996",
        phase_unit="deg",
        group_delay_unit="ms",
        title="Learned transfer function",
    )

    axes[0, 0].set_title("Recovered kernel")
    fig.tight_layout()
    finalize_figure(fig, output_path=OUTPUT_PATH, show=False)


if __name__ == "__main__":
    main()
