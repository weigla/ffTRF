#!/usr/bin/env python3
"""Example: test one held-out prediction against circular-shift surrogates."""

from __future__ import annotations

from pathlib import Path

from simulated_data import (
    build_multi_trial_single_channel_dataset,
    finalize_figure,
    require_matplotlib,
)

from fftrf import TRF

OUTPUT_PATH = Path("artifacts/examples/permutation_significance.png")


def main() -> None:
    """Fit on training trials and test an untouched trial against a null."""

    dataset = build_multi_trial_single_channel_dataset()
    train_stimulus = dataset.stimulus[:-1]
    train_response = dataset.response[:-1]
    test_stimulus = dataset.stimulus[-1]
    test_response = dataset.response[-1]

    model = TRF(direction=1)
    model.train(
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
    result = model.permutation_test(
        stimulus=test_stimulus,
        response=test_response,
        n_permutations=199,
        surrogate="circular_shift",
        min_shift=0.5,
        tail="greater",
        seed=0,
    )

    print("Example: fixed-kernel permutation significance test")
    print(f"  observed held-out r: {float(result.observed_score):.4f}")
    print(f"  greater-tail p-value: {float(result.p_value):.4f}")
    print(f"  permutations: {result.n_permutations}")
    print(f"  minimum attainable p: {1.0 / (result.n_permutations + 1):.4f}")
    print("  inference: one prespecified output; no multiple-output correction needed")
    print(f"  saved figure: {OUTPUT_PATH}")

    plt = require_matplotlib()
    fig, ax = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
    ax.hist(result.null_scores, bins=24, color="#9BB7FF", edgecolor="white")
    ax.axvline(
        float(result.observed_score),
        color="#C84C09",
        linewidth=2.0,
        label=f"Observed r = {float(result.observed_score):.3f}",
    )
    ax.set(
        title="Circular-shift null on an untouched test trial",
        xlabel="Pearson r",
        ylabel="Surrogate count",
    )
    ax.legend(frameon=False)
    finalize_figure(fig, output_path=OUTPUT_PATH, show=False)


if __name__ == "__main__":
    main()
