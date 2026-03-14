#!/usr/bin/env python3
"""Example: bootstrap confidence intervals for a single recovered kernel."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from fft_trf import FrequencyTRF

from simulated_data import (
    build_multi_trial_single_channel_dataset,
    finalize_figure,
    require_matplotlib,
)

OUTPUT_PATH = Path("artifacts/examples/bootstrap_confidence_interval.png")


def main() -> None:
    """Fit a model with bootstrap confidence intervals and visualize them."""

    dataset = build_multi_trial_single_channel_dataset()

    model = FrequencyTRF(direction=1)
    model.train(
        stimulus=dataset.stimulus,
        response=dataset.response,
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
    prediction, score = model.predict(stimulus=dataset.stimulus, response=dataset.response)
    mean_prediction_score = float(score)
    mean_interval_width = float(np.mean(interval[1] - interval[0]))

    print("Example: bootstrap confidence interval")
    print(f"  description: {dataset.description}")
    print(f"  prediction correlation: {mean_prediction_score:.4f}")
    print(f"  bootstrap level: {model.bootstrap_level}")
    print(f"  bootstrap samples: {model.bootstrap_samples}")
    print(f"  interval shape: {interval.shape}")
    print(f"  mean interval width: {mean_interval_width:.4f}")
    print(f"  saved figure: {OUTPUT_PATH}")

    fig, ax = model.plot(
        input_index=0,
        output_index=0,
        show_bootstrap_interval=True,
        color="#3366CC",
        interval_color="#9BB7FF",
        title="Recovered kernel with bootstrap confidence interval",
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
