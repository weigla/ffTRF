#!/usr/bin/env python3
"""Example: multiple trials with cross-validated regularization selection."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from fftrf import FrequencyTRF, inverse_variance_weights, r2_score

from simulated_data import (
    build_multi_trial_single_channel_dataset,
    finalize_figure,
    require_matplotlib,
)

OUTPUT_PATH = Path("artifacts/examples/multi_trial_single_channel.png")

def main() -> None:
    """Fit the model on multiple trials and visualize CV behavior."""

    dataset = build_multi_trial_single_channel_dataset()
    train_stimulus = dataset.stimulus[:-1]
    train_response = dataset.response[:-1]
    test_stimulus = dataset.stimulus[-1]
    test_response = dataset.response[-1]

    regularization_grid = np.logspace(-4, 1, 7)
    model = FrequencyTRF(direction=1)
    cv_scores = model.train(
        stimulus=train_stimulus,
        response=train_response,
        fs=dataset.fs,
        tmin=dataset.tmin,
        tmax=dataset.tmax,
        regularization=regularization_grid,
        segment_length=1_024,
        overlap=0.5,
        window="hann",
        k=len(train_stimulus),
        trial_weights=inverse_variance_weights(train_response),
    )
    prediction, held_out_score = model.predict(stimulus=test_stimulus, response=test_response)
    held_out_r2 = float(r2_score(test_response, prediction).mean())
    kernel_corr = np.corrcoef(dataset.true_weights[0, :, 0], model.weights[0, :, 0])[0, 1]

    print("Example: multiple trials, single feature, single output")
    print(f"  description: {dataset.description}")
    print(f"  selected lambda: {float(model.regularization):.6f}")
    print(f"  held-out correlation: {float(held_out_score):.4f}")
    print(f"  held-out R^2: {held_out_r2:.4f}")
    print(f"  kernel correlation: {float(kernel_corr):.4f}")
    print(f"  segment_length: {model.segment_length}")
    print(f"  n_fft: {model.n_fft}")
    print(f"  weights shape: {model.weights.shape}")
    print(f"  saved figure: {OUTPUT_PATH}")

    plt = require_matplotlib()
    fig, axes = plt.subplots(3, 1, figsize=(10, 9))

    axes[0].semilogx(regularization_grid, np.asarray(cv_scores), marker="o", color="#0B6E4F")
    axes[0].axvline(float(model.regularization), color="#C84C09", linewidth=1.2, linestyle="--")
    axes[0].set_title("Cross-validation Over Ridge Values")
    axes[0].set_xlabel("Regularization")
    axes[0].set_ylabel("Mean score")

    axes[1].plot(dataset.times * 1e3, dataset.true_weights[0, :, 0], label="True", color="#111111", linewidth=2.0)
    axes[1].plot(model.times * 1e3, model.weights[0, :, 0], label="Recovered", color="#3366CC", linewidth=1.6)
    axes[1].set_title("Recovered Kernel")
    axes[1].set_xlabel("Lag (ms)")
    axes[1].set_ylabel("Weight")
    axes[1].legend(loc="upper right")

    time = np.arange(test_stimulus.shape[0]) / dataset.fs
    snippet = time <= 2.0
    axes[2].plot(time[snippet], test_response[snippet, 0], label="Observed", color="#111111", linewidth=1.2)
    axes[2].plot(time[snippet], prediction[snippet, 0], label="Predicted", color="#C84C09", linewidth=1.0)
    axes[2].set_title("Held-out Prediction")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Response")
    axes[2].legend(loc="upper right")

    for ax in axes:
        ax.grid(alpha=0.2, linewidth=0.6)

    fig.tight_layout()
    finalize_figure(fig, output_path=OUTPUT_PATH, show=False)


if __name__ == "__main__":
    main()
