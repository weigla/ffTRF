#!/usr/bin/env python3
"""Example: pointwise percentile bootstrap intervals for one kernel."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from simulated_data import (
    build_multi_trial_single_channel_dataset,
    finalize_figure,
)

from fftrf import TRF

OUTPUT_PATH = Path("artifacts/examples/bootstrap_confidence_interval.png")


def main() -> None:
    """Fit a model with a pointwise bootstrap interval and visualize it."""

    dataset = build_multi_trial_single_channel_dataset()
    train_stimulus = dataset.stimulus[:-1]
    train_response = dataset.response[:-1]
    test_stimulus = dataset.stimulus[-1]
    test_response = dataset.response[-1]

    model = TRF(direction=1)
    model.train(
        stimulus=train_stimulus,
        response=train_response,
        fs=dataset.fs,
        tmin=dataset.tmin,
        tmax=dataset.tmax,
        regularization=1e-3,
        segment_length=1_024,
        overlap=0.5,
        window="hann",
        bootstrap_samples=100,
        bootstrap_level=0.95,
        bootstrap_seed=0,
    )

    interval, _ = model.bootstrap_interval_at()
    _, heldout_score = model.predict(
        stimulus=test_stimulus,
        response=test_response,
    )
    mean_interval_width = float(np.mean(interval[1] - interval[0]))

    print("Example: pointwise percentile bootstrap interval")
    print(f"  description: {dataset.description}")
    print(f"  held-out prediction correlation: {float(heldout_score):.4f}")
    print(f"  bootstrap level: {model.bootstrap_level}")
    print(f"  bootstrap samples: {model.bootstrap_samples}")
    print(f"  interval shape: {interval.shape}")
    print(f"  mean interval width: {mean_interval_width:.4f}")
    print("  interpretation: pointwise across lags; not a simultaneous confidence band")
    print(f"  saved figure: {OUTPUT_PATH}")

    fig, ax = model.plot(
        input_index=0,
        output_index=0,
        show_bootstrap_interval=True,
        color="#3366CC",
        interval_color="#9BB7FF",
        title="Recovered kernel with pointwise bootstrap interval",
        label="Recovered kernel",
    )
    ax.plot(
        dataset.times * 1e3,
        dataset.true_weights[0, :, 0],
        color="#111111",
        linewidth=1.2,
        linestyle="--",
        label="True kernel",
    )
    ax.legend(frameon=False)
    fig.tight_layout()
    finalize_figure(fig, output_path=OUTPUT_PATH, show=False)


if __name__ == "__main__":
    main()
