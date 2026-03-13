#!/usr/bin/env python3
"""Example: single-trial single-channel forward modeling with `FrequencyTRF`."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from fft_trf import FrequencyTRF

from simulated_data import (
    build_single_trial_single_channel_dataset,
    finalize_figure,
    require_matplotlib,
)


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/examples/single_trial_single_channel.png"),
        help="Where to save the figure.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save the figure without opening a window.",
    )
    return parser


def run_example(*, output_path: str | Path | None, show: bool) -> None:
    """Fit the model, print summary metrics, and generate the figure."""

    dataset = build_single_trial_single_channel_dataset()
    stimulus = dataset.stimulus[0]
    response = dataset.response[0]

    model = FrequencyTRF(direction=1)
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
    kernel_corr = np.corrcoef(dataset.true_weights[0, :, 0], model.weights[0, :, 0])[0, 1]

    print("Example: single trial, single feature, single output")
    print(f"  description: {dataset.description}")
    print(f"  prediction correlation: {float(score):.4f}")
    print(f"  kernel correlation: {float(kernel_corr):.4f}")

    plt = require_matplotlib()
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    time = np.arange(stimulus.shape[0]) / dataset.fs
    snippet = time <= 2.0

    axes[0].plot(time[snippet], stimulus[snippet, 0], color="#0B6E4F", linewidth=1.2)
    axes[0].set_title("Simulated Envelope")
    axes[0].set_ylabel("Amplitude")

    axes[1].plot(time[snippet], response[snippet, 0], label="Observed", color="#111111", linewidth=1.2)
    axes[1].plot(time[snippet], prediction[snippet, 0], label="Predicted", color="#C84C09", linewidth=1.0)
    axes[1].set_title("Simulated Brain Response")
    axes[1].set_ylabel("Response")
    axes[1].legend(loc="upper right")

    axes[2].plot(dataset.times * 1e3, dataset.true_weights[0, :, 0], label="True", color="#111111", linewidth=2.0)
    axes[2].plot(model.times * 1e3, model.weights[0, :, 0], label="Recovered", color="#3366CC", linewidth=1.6)
    axes[2].set_title("Recovered Kernel")
    axes[2].set_xlabel("Lag (ms)")
    axes[2].set_ylabel("Weight")
    axes[2].legend(loc="upper right")

    for ax in axes:
        ax.grid(alpha=0.2, linewidth=0.6)

    fig.tight_layout()
    finalize_figure(fig, output_path=output_path, show=show)


def main() -> None:
    """Run the example script."""

    args = build_parser().parse_args()
    run_example(output_path=args.output, show=not args.no_show)


if __name__ == "__main__":
    main()
