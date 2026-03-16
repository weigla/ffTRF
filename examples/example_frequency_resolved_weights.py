#!/usr/bin/env python3
"""Example: frequency-resolved weights for a time-locked alpha-burst response."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from fftrf import FrequencyTRF

from simulated_data import (
    build_frequency_resolved_dataset,
    finalize_figure,
    require_matplotlib,
)

OUTPUT_PATH = Path("artifacts/examples/frequency_resolved_weights.png")


def main() -> None:
    """Fit a model and visualize a time-locked alpha burst in the kernel."""

    dataset = build_frequency_resolved_dataset()
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
        regularization=1e-2,
        segment_length=4096,
        overlap=0.5,
        window=None,
    )
    prediction, score = model.predict(stimulus=test_stimulus, response=test_response)
    resolved = model.frequency_resolved_weights(
        n_bands=20,
        fmax=40.0,
        value_mode="real",
    )
    time_frequency_power = model.time_frequency_power(
        n_bands=20,
        fmax=40.0,
    )

    print("Example: frequency-resolved weights")
    print(f"  description: {dataset.description}")
    print(f"  held-out correlation: {float(score):.4f}")
    print(
        "  kernel correlation: "
        f"{float(np.corrcoef(dataset.true_weights[0, :, 0], model.weights[0, :, 0])[0, 1]):.4f}"
    )
    print(f"  selected regularization: {model.regularization}")
    print(f"  transfer function shape: {model.transfer_function.shape}")
    print(f"  resolved weights shape: {resolved.weights.shape}")
    print(f"  time-frequency power shape: {time_frequency_power.power.shape}")
    print(
        "  first five band centers (Hz): "
        + np.array2string(resolved.band_centers[:5], precision=1, separator=", ")
    )
    print(f"  saved figure: {OUTPUT_PATH}")

    plt = require_matplotlib()
    fig, axes = plt.subplots(
        4,
        1,
        figsize=(10, 11),
        gridspec_kw={"height_ratios": [1.1, 1.1, 1.5, 1.5]},
    )

    time = np.arange(test_stimulus.shape[0]) / dataset.fs
    snippet = time <= 2.0
    axes[0].plot(time[snippet], test_response[snippet, 0], label="Observed", color="#111111", linewidth=1.2)
    axes[0].plot(time[snippet], prediction[snippet, 0], label="Predicted", color="#C84C09", linewidth=1.0)
    axes[0].set_title("Held-Out Simulated Brain Response With Alpha Burst")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Response")
    axes[0].legend(loc="upper right")

    axes[1].plot(dataset.times * 1e3, dataset.true_weights[0, :, 0], label="True", color="#111111", linewidth=2.0)
    axes[1].plot(model.times * 1e3, model.weights[0, :, 0], label="Recovered", color="#3366CC", linewidth=1.6)
    axes[1].set_title("Recovered Time-Domain Kernel With Alpha Burst")
    axes[1].set_xlabel("Lag (ms)")
    axes[1].set_ylabel("Weight")
    axes[1].legend(loc="upper right")

    model.plot_frequency_resolved_weights(
        resolved=resolved,
        ax=axes[2],
        title="Frequency-Resolved Weights (Signed, Alpha Range Emphasis)",
        time_unit="ms",
    )
    model.plot_time_frequency_power(
        power=time_frequency_power,
        ax=axes[3],
        title="Time-Frequency Power (Hilbert, Alpha Range Emphasis)",
        time_unit="ms",
    )

    for axis in axes[:2]:
        axis.grid(alpha=0.2, linewidth=0.6)

    fig.tight_layout()
    finalize_figure(fig, output_path=OUTPUT_PATH, show=False)


if __name__ == "__main__":
    main()
