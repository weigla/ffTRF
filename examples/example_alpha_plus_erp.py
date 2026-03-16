#!/usr/bin/env python3
"""Example: mixed ERP and alpha-burst response in a frequency-resolved view."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from fftrf import FrequencyTRF

from simulated_data import (
    build_alpha_plus_erp_dataset,
    finalize_figure,
    require_matplotlib,
)

OUTPUT_PATH = Path("artifacts/examples/alpha_plus_erp.png")


def main() -> None:
    """Fit a mixed ERP-plus-alpha response and visualize both components."""

    dataset = build_alpha_plus_erp_dataset()
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
        segment_duration=4.096,
        overlap=0.5,
        window=None,
    )
    prediction, score = model.predict(stimulus=test_stimulus, response=test_response)
    resolved = model.frequency_resolved_weights(
        n_bands=24,
        fmax=35.0,
        value_mode="real",
    )
    kernel_corr = float(np.corrcoef(dataset.true_weights[0, :, 0], model.weights[0, :, 0])[0, 1])

    print("Example: mixed alpha-plus-ERP response")
    print(f"  description: {dataset.description}")
    print(f"  held-out correlation: {float(score):.4f}")
    print(f"  kernel correlation: {kernel_corr:.4f}")
    print(f"  selected regularization: {model.regularization}")
    print(f"  segment_duration: {model.segment_duration}")
    print(f"  resolved weights shape: {resolved.weights.shape}")
    print(
        "  alpha-relevant band centers (Hz): "
        + np.array2string(resolved.band_centers[4:10], precision=1, separator=", ")
    )
    print(f"  saved figure: {OUTPUT_PATH}")

    plt = require_matplotlib()
    fig, axes = plt.subplots(
        4,
        1,
        figsize=(10, 11),
        gridspec_kw={"height_ratios": [0.9, 1.1, 1.2, 1.8]},
    )

    time = np.arange(test_stimulus.shape[0]) / dataset.fs
    snippet = time <= 2.3
    axes[0].plot(time[snippet], test_stimulus[snippet, 0], color="#0B6E4F", linewidth=1.2)
    axes[0].set_title("Event-Locked Stimulus Driver")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")

    axes[1].plot(time[snippet], test_response[snippet, 0], label="Observed", color="#111111", linewidth=1.2)
    axes[1].plot(time[snippet], prediction[snippet, 0], label="Predicted", color="#C84C09", linewidth=1.0)
    axes[1].set_title("Held-Out Simulated Response")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Response")
    axes[1].legend(loc="upper right")

    axes[2].plot(dataset.times * 1e3, dataset.true_weights[0, :, 0], label="True", color="#111111", linewidth=2.0)
    axes[2].plot(model.times * 1e3, model.weights[0, :, 0], label="Recovered", color="#3366CC", linewidth=1.6)
    axes[2].axvspan(30, 190, color="#D9EAF7", alpha=0.35, linewidth=0.0)
    axes[2].axvspan(190, 340, color="#FCE8D5", alpha=0.35, linewidth=0.0)
    axes[2].text(95, axes[2].get_ylim()[1] * 0.86, "ERP", ha="center", va="center", fontsize=10)
    axes[2].text(265, axes[2].get_ylim()[1] * 0.86, "Alpha burst", ha="center", va="center", fontsize=10)
    axes[2].set_title("Recovered Kernel With ERP And Alpha Components")
    axes[2].set_xlabel("Lag (ms)")
    axes[2].set_ylabel("Weight")
    axes[2].legend(loc="upper right")

    model.plot_frequency_resolved_weights(
        resolved=resolved,
        ax=axes[3],
        title="Frequency-Resolved Weights (ERP + Alpha)",
        time_unit="ms",
    )

    for axis in axes[:3]:
        axis.grid(alpha=0.2, linewidth=0.6)

    fig.tight_layout()
    finalize_figure(fig, output_path=OUTPUT_PATH, show=False)


if __name__ == "__main__":
    main()
