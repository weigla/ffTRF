#!/usr/bin/env python3
"""Focused real-data example: EEG reconstructs a held-out speech envelope."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from mtrf_sample_data import exact_lag_window_seconds, load_sample_data
from simulated_data import finalize_figure, require_matplotlib

from fftrf import TRF, pearsonr

OUTPUT_PATH = Path("artifacts/examples/real_eeg_backward.png")


def compressed_envelope(stimulus_trial: np.ndarray, exponent: float = 0.4) -> np.ndarray:
    """Create the broadband envelope target used by the public mTRF example."""

    broadband = np.asarray(stimulus_trial, dtype=float).mean(axis=1, keepdims=True)
    compressed = np.clip(broadband, 0.0, None) ** exponent
    scale = np.clip(compressed.std(axis=0, keepdims=True), np.finfo(float).eps, None)
    return (compressed - compressed.mean(axis=0, keepdims=True)) / scale


def main() -> None:
    """Fit a backward decoder and inspect one prespecified test segment."""

    # The response loader reproduces the upstream tutorial convention of
    # standardizing each 12-second segment independently.
    raw_stimulus, _, fs = load_sample_data(n_segments=10, normalize=False)
    _, response, normalized_fs = load_sample_data(n_segments=10, normalize=True)
    if normalized_fs != fs:
        raise ValueError("Sample-data loaders returned different sampling rates.")

    envelope = [compressed_envelope(trial) for trial in raw_stimulus]
    train_envelope, test_envelope = envelope[:7], envelope[7:]
    train_response, test_response = response[:7], response[7:]
    _, tmax = exact_lag_window_seconds(fs=fs, nominal_stop_seconds=0.35)

    model = TRF(direction=-1, metric="neg_mse")
    model.train(
        stimulus=train_envelope,
        response=train_response,
        fs=fs,
        tmin=0.0,
        tmax=tmax,
        regularization=np.logspace(-8, 6, 15),
        k=3,
        seed=7,
        segment_duration=2.0,
        overlap=0.5,
        window="hann",
        detrend=None,
    )

    predictions = model.predict(response=test_response)
    heldout_r = np.asarray(
        [
            float(pearsonr(observed, predicted)[0])
            for observed, predicted in zip(test_envelope, predictions, strict=True)
        ]
    )

    print("Real speech EEG: backward decoding")
    print("  split: 7 training segments, 3 segments held out from fitting and CV")
    print(f"  data shape: {response[0].shape[1]} EEG channels -> 1 envelope")
    print(f"  selected lambda: {float(model.regularization):.6g}")
    print(f"  requested decoder interval: [0, {tmax:.6f}) s")
    print(f"  stored physical lags: {model.times[0]:.6f} to {model.times[-1]:.6f} s")
    print(f"  held-out segment r: {np.array2string(heldout_r, precision=4)}")
    print(f"  mean held-out r: {float(heldout_r.mean()):.4f}")
    print(f"  saved figure: {OUTPUT_PATH}")

    display_seconds = 6.0
    n_display = min(
        int(round(display_seconds * fs)),
        test_envelope[0].shape[0],
        predictions[0].shape[0],
    )
    time = np.arange(n_display) / fs
    plt = require_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    axes[0].plot(
        time,
        test_envelope[0][:n_display, 0],
        color="#111111",
        linewidth=1.2,
        label="Observed",
    )
    axes[0].plot(
        time,
        predictions[0][:n_display, 0],
        color="#C84C09",
        linewidth=1.0,
        label="Decoded",
    )
    axes[0].set(
        title=f"Prespecified test segment 1 (r = {heldout_r[0]:.3f})",
        xlabel="Time (s)",
        ylabel="Standardized envelope",
    )
    axes[0].legend(frameon=False)
    model.plot(
        input_index=0,
        output_index=0,
        ax=axes[1],
        title="Decoder weights for EEG channel 1",
    )
    finalize_figure(fig, output_path=OUTPUT_PATH, show=False)


if __name__ == "__main__":
    main()
