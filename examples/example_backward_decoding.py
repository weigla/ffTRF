#!/usr/bin/env python3
"""Example: backward decoding from multichannel responses to one stimulus."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from fftrf import FrequencyTRF

from simulated_data import (
    build_backward_decoding_dataset,
    finalize_figure,
    require_matplotlib,
)

OUTPUT_PATH = Path("artifacts/examples/backward_decoding.png")

def main() -> None:
    """Fit a backward model and visualize the decoded stimulus."""

    dataset = build_backward_decoding_dataset()
    train_stimulus = dataset.stimulus[:-1]
    train_response = dataset.response[:-1]
    test_stimulus = dataset.stimulus[-1]
    test_response = dataset.response[-1]

    model = FrequencyTRF(direction=-1)
    model.train(
        stimulus=train_stimulus,
        response=train_response,
        fs=dataset.fs,
        tmin=-0.180,
        tmax=0.040,
        regularization=2e-2,
        segment_length=1_024,
        overlap=0.5,
        window="hann",
    )
    prediction, score = model.predict(
        response=test_response,
        stimulus=test_stimulus,
    )

    print("Example: backward decoding")
    print(f"  description: {dataset.description}")
    print(f"  decoding correlation: {float(score):.4f}")
    print(f"  regularization: {model.regularization}")
    print(f"  segment_length: {model.segment_length}")
    print(f"  n_fft: {model.n_fft}")
    print(f"  weights shape: {model.weights.shape}")
    print(f"  saved figure: {OUTPUT_PATH}")

    plt = require_matplotlib()
    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    time = np.arange(test_stimulus.shape[0]) / dataset.fs
    snippet = time <= 2.0

    axes[0].plot(time[snippet], test_stimulus[snippet, 0], label="True envelope", color="#111111", linewidth=1.2)
    axes[0].plot(time[snippet], prediction[snippet, 0], label="Decoded envelope", color="#C84C09", linewidth=1.0)
    axes[0].set_title("Backward Reconstruction On A Held-out Trial")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.2, linewidth=0.6)

    for channel_index in range(model.weights.shape[0]):
        axes[1].plot(
            model.times * 1e3,
            model.weights[channel_index, :, 0],
            linewidth=1.2,
            label=f"Channel {channel_index + 1}",
        )
    axes[1].set_title("Learned Decoder Weights")
    axes[1].set_xlabel("Lag (ms)")
    axes[1].set_ylabel("Weight")
    axes[1].legend(loc="upper right", ncols=2)
    axes[1].grid(alpha=0.2, linewidth=0.6)

    fig.tight_layout()
    finalize_figure(fig, output_path=OUTPUT_PATH, show=False)


if __name__ == "__main__":
    main()
