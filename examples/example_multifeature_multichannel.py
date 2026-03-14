#!/usr/bin/env python3
"""Example: multifeature stimulus and multichannel response modeling."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from fft_trf import FrequencyTRF

from simulated_data import (
    build_multifeature_multichannel_dataset,
    finalize_figure,
    require_matplotlib,
)

OUTPUT_PATH = Path("artifacts/examples/multifeature_multichannel.png")

def main() -> None:
    """Fit a multivariate forward model and visualize all kernels."""

    dataset = build_multifeature_multichannel_dataset()
    train_stimulus = dataset.stimulus[:-1]
    train_response = dataset.response[:-1]
    test_stimulus = dataset.stimulus[-1]
    test_response = dataset.response[-1]

    model = FrequencyTRF(direction=1)
    model.train(
        stimulus=train_stimulus,
        response=train_response,
        fs=dataset.fs,
        tmin=dataset.tmin,
        tmax=dataset.tmax,
        regularization=5e-3,
        segment_length=1_024,
        overlap=0.5,
        window="hann",
    )
    prediction, scores = model.predict(
        stimulus=test_stimulus,
        response=test_response,
        average=False,
    )
    scores = np.asarray(scores, dtype=float)

    print("Example: multiple features, multiple outputs")
    print(f"  description: {dataset.description}")
    print(f"  held-out channel scores: {np.array2string(scores, precision=4)}")
    print(f"  regularization: {model.regularization}")
    print(f"  segment_length: {model.segment_length}")
    print(f"  n_fft: {model.n_fft}")
    print(f"  weights shape: {model.weights.shape}")
    print(f"  saved figure: {OUTPUT_PATH}")

    plt = require_matplotlib()
    fig, axes = plt.subplots(
        3,
        2,
        figsize=(11, 9),
        gridspec_kw={"height_ratios": [1.0, 1.0, 0.8]},
    )
    feature_names = ["Envelope", "Onset"]
    channel_names = ["Channel 1", "Channel 2"]

    for input_index in range(2):
        for output_index in range(2):
            ax = axes[input_index, output_index]
            ax.plot(
                dataset.times * 1e3,
                dataset.true_weights[input_index, :, output_index],
                label="True",
                color="#111111",
                linewidth=2.0,
            )
            ax.plot(
                model.times * 1e3,
                model.weights[input_index, :, output_index],
                label="Recovered",
                color="#3366CC",
                linewidth=1.6,
            )
            ax.set_title(f"{feature_names[input_index]} -> {channel_names[output_index]}")
            ax.set_ylabel("Weight")
            ax.grid(alpha=0.2, linewidth=0.6)

    for ax in axes[1, :]:
        ax.set_xlabel("Lag (ms)")
    axes[0, 0].legend(loc="upper right")

    time = np.arange(test_stimulus.shape[0]) / dataset.fs
    snippet = time <= 1.5
    axes[2, 0].set_visible(True)
    axes[2, 0].plot(time[snippet], test_response[snippet, 0], label="Observed", color="#111111", linewidth=1.1)
    axes[2, 0].plot(time[snippet], prediction[snippet, 0], label="Predicted", color="#C84C09", linewidth=1.0)
    axes[2, 0].set_title("Held-out prediction, channel 1")
    axes[2, 0].set_xlabel("Time (s)")
    axes[2, 0].set_ylabel("Response")
    axes[2, 0].grid(alpha=0.2, linewidth=0.6)
    axes[2, 0].legend(loc="upper right")

    axes[2, 1].set_visible(True)
    axes[2, 1].plot(time[snippet], test_response[snippet, 1], label="Observed", color="#111111", linewidth=1.1)
    axes[2, 1].plot(time[snippet], prediction[snippet, 1], label="Predicted", color="#C84C09", linewidth=1.0)
    axes[2, 1].set_title("Held-out prediction, channel 2")
    axes[2, 1].set_xlabel("Time (s)")
    axes[2, 1].set_ylabel("Response")
    axes[2, 1].grid(alpha=0.2, linewidth=0.6)
    axes[2, 1].legend(loc="upper right")

    fig.tight_layout()
    finalize_figure(fig, output_path=OUTPUT_PATH, show=False)


if __name__ == "__main__":
    main()
