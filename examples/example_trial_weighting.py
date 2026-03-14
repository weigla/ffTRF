#!/usr/bin/env python3
"""Example: compare unweighted and trial-weighted fitting."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from fftrf import FrequencyTRF, inverse_variance_weights

from simulated_data import (
    build_multi_trial_single_channel_dataset,
    finalize_figure,
    require_matplotlib,
)

OUTPUT_PATH = Path("artifacts/examples/trial_weighting.png")


def main() -> None:
    """Fit the same dataset with and without trial weighting."""

    dataset = build_multi_trial_single_channel_dataset()
    train_stimulus = dataset.stimulus[:-1]
    train_response = [trial.copy() for trial in dataset.response[:-1]]
    test_stimulus = dataset.stimulus[-1]
    test_response = dataset.response[-1]

    rng = np.random.default_rng(0)
    train_response[0] = 3.0 * train_response[0] + 0.80 * rng.standard_normal(train_response[0].shape)

    trial_weights = inverse_variance_weights(train_response)

    unweighted = FrequencyTRF(direction=1)
    unweighted.train(
        stimulus=train_stimulus,
        response=train_response,
        fs=dataset.fs,
        tmin=dataset.tmin,
        tmax=dataset.tmax,
        regularization=1e-3,
        segment_length=1_024,
        overlap=0.5,
        window="hann",
    )
    _, unweighted_score = unweighted.predict(
        stimulus=test_stimulus,
        response=test_response,
    )

    weighted = FrequencyTRF(direction=1)
    weighted.train(
        stimulus=train_stimulus,
        response=train_response,
        fs=dataset.fs,
        tmin=dataset.tmin,
        tmax=dataset.tmax,
        regularization=1e-3,
        segment_length=1_024,
        overlap=0.5,
        window="hann",
        trial_weights=trial_weights,
    )
    prediction, weighted_score = weighted.predict(
        stimulus=test_stimulus,
        response=test_response,
    )

    print("Example: trial weighting")
    print(f"  description: {dataset.description}")
    print("  note: training trial 1 was deliberately rescaled and made noisier")
    print(f"  inverse-variance weights: {np.array2string(trial_weights, precision=3)}")
    print(f"  stored weighting mode: {np.array2string(weighted.trial_weights, precision=3)}")
    print(f"  unweighted held-out correlation: {float(unweighted_score):.4f}")
    print(f"  weighted held-out correlation: {float(weighted_score):.4f}")
    print(f"  saved figure: {OUTPUT_PATH}")

    plt = require_matplotlib()
    fig, axes = plt.subplots(3, 1, figsize=(10, 9))

    axes[0].bar(np.arange(1, len(trial_weights) + 1), trial_weights, color="#0B6E4F")
    axes[0].set_title("Inverse-variance trial weights")
    axes[0].set_xlabel("Training trial")
    axes[0].set_ylabel("Weight")

    axes[1].plot(
        dataset.times * 1e3,
        dataset.true_weights[0, :, 0],
        color="#111111",
        linewidth=1.8,
        linestyle="--",
        label="True kernel",
    )
    axes[1].plot(
        unweighted.times * 1e3,
        unweighted.weights[0, :, 0],
        color="#7A3E9D",
        linewidth=1.3,
        label="Unweighted",
    )
    axes[1].plot(
        weighted.times * 1e3,
        weighted.weights[0, :, 0],
        color="#3366CC",
        linewidth=1.3,
        label="Weighted",
    )
    axes[1].set_title("Recovered kernel with and without trial weighting")
    axes[1].set_xlabel("Lag (ms)")
    axes[1].set_ylabel("Weight")
    axes[1].legend(loc="upper right")

    time = np.arange(test_stimulus.shape[0]) / dataset.fs
    snippet = time <= 2.0
    axes[2].plot(
        time[snippet],
        test_response[snippet, 0],
        color="#111111",
        linewidth=1.2,
        label="Observed",
    )
    axes[2].plot(
        time[snippet],
        prediction[snippet, 0],
        color="#C84C09",
        linewidth=1.0,
        label="Weighted prediction",
    )
    axes[2].set_title("Held-out response")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Response")
    axes[2].legend(loc="upper right")

    for ax in axes:
        ax.grid(alpha=0.2, linewidth=0.6)

    fig.tight_layout()
    finalize_figure(fig, output_path=OUTPUT_PATH, show=False)


if __name__ == "__main__":
    main()
