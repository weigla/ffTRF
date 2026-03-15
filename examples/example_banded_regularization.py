#!/usr/bin/env python3
"""Example: optional banded regularization for multifeature predictors."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from fftrf import FrequencyTRF

from simulated_data import (
    build_banded_regularization_dataset,
    finalize_figure,
    require_matplotlib,
)

OUTPUT_PATH = Path("artifacts/examples/banded_regularization.png")


def main() -> None:
    """Fit a banded-ridge model and visualize the selected coefficients."""

    dataset = build_banded_regularization_dataset()
    train_stimulus = dataset.stimulus[:-1]
    train_response = dataset.response[:-1]
    test_stimulus = dataset.stimulus[-1]
    test_response = dataset.response[-1]

    regularization_grid = np.logspace(-4, 0.5, 6)
    model = FrequencyTRF(direction=1)
    cv_scores = model.train(
        stimulus=train_stimulus,
        response=train_response,
        fs=dataset.fs,
        tmin=dataset.tmin,
        tmax=dataset.tmax,
        regularization=regularization_grid,
        bands=[1, 1],
        segment_length=1_024,
        overlap=0.5,
        window="hann",
        k=4,
    )
    prediction, held_out_score = model.predict(stimulus=test_stimulus, response=test_response)
    score_grid = np.asarray(cv_scores, dtype=float).reshape(len(regularization_grid), len(regularization_grid))

    print("Example: banded regularization")
    print(f"  description: {dataset.description}")
    print(f"  selected band coefficients: {model.regularization}")
    print(f"  expanded feature penalties: {model.feature_regularization}")
    print(f"  candidate count: {len(model.regularization_candidates)}")
    print(f"  held-out correlation: {float(held_out_score):.4f}")
    print(f"  saved figure: {OUTPUT_PATH}")

    plt = require_matplotlib()
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(11, 7),
        gridspec_kw={"width_ratios": [1.1, 1.0], "height_ratios": [1.0, 1.0]},
    )

    heatmap_ax = axes[0, 0]
    image = heatmap_ax.imshow(
        score_grid,
        origin="lower",
        aspect="auto",
        cmap="viridis",
    )
    best_coefficients = tuple(float(value) for value in model.regularization)
    feature_0_index = int(np.argmin(np.abs(regularization_grid - best_coefficients[0])))
    feature_1_index = int(np.argmin(np.abs(regularization_grid - best_coefficients[1])))
    heatmap_ax.scatter(feature_1_index, feature_0_index, color="#F4D35E", s=60, edgecolor="#111111")
    heatmap_ax.set_title("Cross-validation over banded coefficients")
    heatmap_ax.set_xlabel("Feature 2 regularization")
    heatmap_ax.set_ylabel("Feature 1 regularization")
    heatmap_ax.set_xticks(np.arange(len(regularization_grid)))
    heatmap_ax.set_yticks(np.arange(len(regularization_grid)))
    heatmap_ax.set_xticklabels([f"{value:.1e}" for value in regularization_grid], rotation=45, ha="right")
    heatmap_ax.set_yticklabels([f"{value:.1e}" for value in regularization_grid])
    fig.colorbar(image, ax=heatmap_ax, fraction=0.046, pad=0.04, label="Mean CV score")

    time_ms = dataset.times * 1e3
    kernel_colors = ["#3366CC", "#C84C09"]
    kernel_titles = ["Envelope kernel", "Onset kernel"]
    for feature_index, kernel_ax in enumerate([axes[0, 1], axes[1, 1]]):
        kernel_ax.plot(
            time_ms,
            dataset.true_weights[feature_index, :, 0],
            color="#111111",
            linewidth=1.5,
            linestyle="--",
            label="True kernel",
        )
        kernel_ax.plot(
            model.times * 1e3,
            model.weights[feature_index, :, 0],
            color=kernel_colors[feature_index],
            linewidth=1.8,
            label="Recovered kernel",
        )
        kernel_ax.set_title(kernel_titles[feature_index])
        kernel_ax.set_xlabel("Lag (ms)")
        kernel_ax.set_ylabel("Weight")
        kernel_ax.grid(alpha=0.2, linewidth=0.6)
        kernel_ax.legend(loc="upper right", frameon=False)

    prediction_ax = axes[1, 0]
    time = np.arange(test_stimulus.shape[0]) / dataset.fs
    snippet = time <= 2.0
    prediction_ax.plot(
        time[snippet],
        test_response[snippet, 0],
        color="#111111",
        linewidth=1.2,
        label="Observed",
    )
    prediction_ax.plot(
        time[snippet],
        prediction[snippet, 0],
        color="#0B6E4F",
        linewidth=1.0,
        label="Predicted",
    )
    prediction_ax.set_title("Held-out prediction")
    prediction_ax.set_xlabel("Time (s)")
    prediction_ax.set_ylabel("Response")
    prediction_ax.grid(alpha=0.2, linewidth=0.6)
    prediction_ax.legend(loc="upper right", frameon=False)

    fig.tight_layout()
    finalize_figure(fig, output_path=OUTPUT_PATH, show=False)


if __name__ == "__main__":
    main()
