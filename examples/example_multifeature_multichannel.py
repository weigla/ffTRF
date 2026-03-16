#!/usr/bin/env python3
"""Example: multifeature stimulus and multichannel response modeling."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from fftrf import FrequencyTRF

from simulated_data import (
    build_multifeature_multichannel_dataset,
    finalize_figure,
    require_matplotlib,
)

KERNEL_OUTPUT_PATH = Path("artifacts/examples/multifeature_multichannel_kernels.png")
PREDICTION_OUTPUT_PATH = Path("artifacts/examples/multifeature_multichannel_predictions.png")

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
    print(f"  kernel figure: {KERNEL_OUTPUT_PATH}")
    print(f"  prediction figure: {PREDICTION_OUTPUT_PATH}")

    plt = require_matplotlib()
    kernel_fig, _ = model.plot_grid(
        input_labels=["Envelope", "Onset"],
        output_labels=["Channel 1", "Channel 2"],
        title="Recovered multifeature / multichannel kernels",
        sharey=False,
    )
    for ax_index, ax in enumerate(kernel_fig.axes):
        input_index = ax_index // 2
        output_index = ax_index % 2
        ax.plot(
            dataset.times * 1e3,
            dataset.true_weights[input_index, :, output_index],
            color="#111111",
            linewidth=1.2,
            linestyle="--",
            label="True kernel" if ax_index == 0 else None,
        )
    if kernel_fig.axes:
        kernel_fig.axes[0].legend(loc="upper right", frameon=False)
    finalize_figure(kernel_fig, output_path=KERNEL_OUTPUT_PATH, show=False)

    time = np.arange(test_stimulus.shape[0]) / dataset.fs
    snippet = time <= 2.5
    prediction_fig, prediction_axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for channel_index, ax in enumerate(prediction_axes):
        ax.plot(time[snippet], test_response[snippet, channel_index], label="Observed", color="#111111", linewidth=1.2)
        ax.plot(time[snippet], prediction[snippet, channel_index], label="Predicted", color="#C84C09", linewidth=1.0)
        ax.set_title(f"Held-out response, channel {channel_index + 1}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Response")
        ax.grid(alpha=0.2, linewidth=0.6)
        ax.legend(loc="upper right")
    prediction_fig.tight_layout()
    finalize_figure(prediction_fig, output_path=PREDICTION_OUTPUT_PATH, show=False)


if __name__ == "__main__":
    main()
