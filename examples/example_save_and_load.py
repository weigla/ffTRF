#!/usr/bin/env python3
"""Example: save, load, and reuse a fitted `TRF` model."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from fftrf import TRF

from simulated_data import (
    build_single_trial_single_channel_dataset,
    finalize_figure,
    require_matplotlib,
)

MODEL_PATH = Path("artifacts/examples/frequency_trf_model.pkl")
OUTPUT_PATH = Path("artifacts/examples/save_and_load.png")


def main() -> None:
    """Train a model, serialize it, reload it, and compare predictions."""

    dataset = build_single_trial_single_channel_dataset()
    stimulus = dataset.stimulus[0]
    response = dataset.response[0]

    model = TRF(direction=1)
    model.train(
        stimulus=stimulus,
        response=response,
        fs=dataset.fs,
        tmin=dataset.tmin,
        tmax=dataset.tmax,
        regularization=1e-2,
        segment_length=2_048,
        overlap=0.5,
        window="hann",
    )
    prediction, score = model.predict(stimulus=stimulus, response=response)

    model.save(MODEL_PATH)
    restored = TRF(direction=1)
    restored.load(MODEL_PATH)
    restored_prediction, restored_score = restored.predict(
        stimulus=stimulus,
        response=response,
    )
    short_weights, short_times = restored.to_impulse_response(tmin=0.0, tmax=0.120)

    print("Example: save and load")
    print(f"  description: {dataset.description}")
    print(f"  original score: {float(score):.4f}")
    print(f"  restored score: {float(restored_score):.4f}")
    print(f"  weights identical: {np.allclose(model.weights, restored.weights)}")
    print(f"  prediction identical: {np.allclose(prediction, restored_prediction)}")
    print(f"  shortened impulse response shape: {short_weights.shape}")
    print(f"  saved model: {MODEL_PATH}")
    print(f"  saved figure: {OUTPUT_PATH}")

    plt = require_matplotlib()
    fig, axes = plt.subplots(2, 1, figsize=(10, 7))

    axes[0].plot(
        dataset.times * 1e3,
        dataset.true_weights[0, :, 0],
        color="#111111",
        linewidth=1.8,
        linestyle="--",
        label="True kernel",
    )
    axes[0].plot(
        restored.times * 1e3,
        restored.weights[0, :, 0],
        color="#3366CC",
        linewidth=1.4,
        label="Restored model kernel",
    )
    axes[0].plot(
        short_times * 1e3,
        short_weights[0, :, 0],
        color="#C84C09",
        linewidth=1.2,
        label="Shorter exported window",
    )
    axes[0].set_title("Saved and restored kernel")
    axes[0].set_xlabel("Lag (ms)")
    axes[0].set_ylabel("Weight")
    axes[0].legend(loc="upper right")

    time = np.arange(stimulus.shape[0]) / dataset.fs
    snippet = time <= 2.0
    axes[1].plot(
        time[snippet],
        response[snippet, 0],
        color="#111111",
        linewidth=1.2,
        label="Observed",
    )
    axes[1].plot(
        time[snippet],
        restored_prediction[snippet, 0],
        color="#C84C09",
        linewidth=1.0,
        label="Prediction from restored model",
    )
    axes[1].set_title("Prediction from the restored estimator")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Response")
    axes[1].legend(loc="upper right")

    for ax in axes:
        ax.grid(alpha=0.2, linewidth=0.6)

    fig.tight_layout()
    finalize_figure(fig, output_path=OUTPUT_PATH, show=False)


if __name__ == "__main__":
    main()
