"""Shared simulation helpers for the example scripts.

The functions in this module generate small, interpretable datasets that are
useful for demonstrating the main `FrequencyTRF` API patterns without depending
on any external files.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.signal import butter, filtfilt, fftconvolve


@dataclass(slots=True)
class SimulatedTRFDataset:
    """Container holding simulated stimulus-response pairs and ground truth."""

    stimulus: list[np.ndarray]
    response: list[np.ndarray]
    true_weights: np.ndarray
    times: np.ndarray
    fs: float
    tmin: float
    tmax: float
    description: str


def require_matplotlib():
    """Import `matplotlib.pyplot` with a helpful error if missing."""

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required for the example plots. "
            "Install the compare extras or use the Pixi compare environment."
        ) from exc
    return plt


def finalize_figure(fig, *, output_path: str | Path | None, show: bool) -> None:
    """Save and optionally show a matplotlib figure."""

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
    if show:
        require_matplotlib().show()


def lag_times(fs: float, tmin: float, tmax: float) -> np.ndarray:
    """Return the lag vector used by the simulated kernels."""

    lag_start = int(round(tmin * fs))
    lag_stop = int(round(tmax * fs))
    return np.arange(lag_start, lag_stop, dtype=int) / fs


def gaussian_kernel(
    *,
    fs: float,
    tmin: float,
    tmax: float,
    bumps: Sequence[tuple[float, float, float]],
) -> tuple[np.ndarray, np.ndarray]:
    """Create a smooth kernel from Gaussian bumps.

    Each tuple in `bumps` is `(latency_seconds, amplitude, width_seconds)`.
    """

    times = lag_times(fs, tmin, tmax)
    kernel = np.zeros(times.shape[0], dtype=float)
    for latency, amplitude, width in bumps:
        kernel += amplitude * np.exp(-0.5 * ((times - latency) / width) ** 2)
    return kernel, times


def make_envelope(
    *,
    n_samples: int,
    fs: float,
    rng: np.random.Generator,
    lowpass_hz: float = 8.0,
) -> np.ndarray:
    """Generate a smooth positive envelope-like regressor."""

    cutoff = min(lowpass_hz / (0.5 * fs), 0.99)
    b, a = butter(3, cutoff, btype="low")
    signal = filtfilt(b, a, rng.standard_normal(n_samples))
    signal = signal - signal.min()
    signal = signal / np.clip(signal.std(), np.finfo(float).eps, None)
    return signal


def make_onset_feature(envelope: np.ndarray) -> np.ndarray:
    """Return a positive onset-like feature derived from an envelope."""

    onset = np.diff(envelope, prepend=envelope[:1])
    onset = np.maximum(onset, 0.0)
    onset = onset / np.clip(onset.std(), np.finfo(float).eps, None)
    return onset


def shifted_convolution(
    signal_in: np.ndarray,
    kernel: np.ndarray,
    *,
    lag_start: int,
    out_length: int,
) -> np.ndarray:
    """Convolve one predictor with a kernel while honoring the lag origin."""

    full = fftconvolve(signal_in, kernel, mode="full")
    offset = -lag_start

    prediction = np.zeros(out_length, dtype=float)
    src_start = max(offset, 0)
    dst_start = max(-offset, 0)
    length = min(full.shape[0] - src_start, out_length - dst_start)
    if length > 0:
        prediction[dst_start : dst_start + length] = full[src_start : src_start + length]
    return prediction


def simulate_response(
    stimulus: np.ndarray,
    true_weights: np.ndarray,
    *,
    fs: float,
    tmin: float,
    noise_scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a multichannel response from a stimulus and kernel bank."""

    n_samples, n_inputs = stimulus.shape
    n_outputs = true_weights.shape[-1]
    lag_start = int(round(tmin * fs))
    response = np.zeros((n_samples, n_outputs), dtype=float)
    for input_index in range(n_inputs):
        for output_index in range(n_outputs):
            response[:, output_index] += shifted_convolution(
                stimulus[:, input_index],
                true_weights[input_index, :, output_index],
                lag_start=lag_start,
                out_length=n_samples,
            )
    response += noise_scale * rng.standard_normal(response.shape)
    return response


def build_single_trial_single_channel_dataset(
    *,
    fs: float = 1_000.0,
    n_samples: int = 8_000,
    tmin: float = 0.0,
    tmax: float = 0.250,
    noise_scale: float = 0.08,
    seed: int = 1,
) -> SimulatedTRFDataset:
    """Create one stimulus trial and one simulated response channel."""

    rng = np.random.default_rng(seed)
    envelope = make_envelope(n_samples=n_samples, fs=fs, rng=rng)
    kernel, times = gaussian_kernel(
        fs=fs,
        tmin=tmin,
        tmax=tmax,
        bumps=[
            (0.040, 1.10, 0.012),
            (0.105, -0.55, 0.020),
            (0.180, 0.25, 0.025),
        ],
    )
    true_weights = kernel[np.newaxis, :, np.newaxis]
    stimulus = envelope[:, np.newaxis]
    response = simulate_response(
        stimulus,
        true_weights,
        fs=fs,
        tmin=tmin,
        noise_scale=noise_scale,
        rng=rng,
    )
    return SimulatedTRFDataset(
        stimulus=[stimulus],
        response=[response],
        true_weights=true_weights,
        times=times,
        fs=fs,
        tmin=tmin,
        tmax=tmax,
        description="Single-trial single-feature forward model.",
    )


def build_multi_trial_single_channel_dataset(
    *,
    fs: float = 1_000.0,
    n_trials: int = 6,
    n_samples: int = 6_000,
    tmin: float = 0.0,
    tmax: float = 0.250,
    seed: int = 2,
) -> SimulatedTRFDataset:
    """Create multiple trials for cross-validated single-channel fitting."""

    rng = np.random.default_rng(seed)
    kernel, times = gaussian_kernel(
        fs=fs,
        tmin=tmin,
        tmax=tmax,
        bumps=[
            (0.030, 0.95, 0.010),
            (0.085, -0.45, 0.017),
            (0.140, 0.20, 0.020),
        ],
    )
    true_weights = kernel[np.newaxis, :, np.newaxis]

    stimulus = []
    response = []
    for _ in range(n_trials):
        envelope = make_envelope(n_samples=n_samples, fs=fs, rng=rng)
        trial_stimulus = envelope[:, np.newaxis]
        trial_noise = 0.05 + 0.05 * rng.random()
        trial_response = simulate_response(
            trial_stimulus,
            true_weights,
            fs=fs,
            tmin=tmin,
            noise_scale=trial_noise,
            rng=rng,
        )
        stimulus.append(trial_stimulus)
        response.append(trial_response)

    return SimulatedTRFDataset(
        stimulus=stimulus,
        response=response,
        true_weights=true_weights,
        times=times,
        fs=fs,
        tmin=tmin,
        tmax=tmax,
        description="Multiple noisy trials for cross-validated forward fitting.",
    )


def build_multifeature_multichannel_dataset(
    *,
    fs: float = 1_000.0,
    n_trials: int = 5,
    n_samples: int = 7_000,
    tmin: float = 0.0,
    tmax: float = 0.220,
    noise_scale: float = 0.06,
    seed: int = 3,
) -> SimulatedTRFDataset:
    """Create a dataset with two stimulus features and two response channels."""

    rng = np.random.default_rng(seed)
    env_kernel_0, times = gaussian_kernel(
        fs=fs,
        tmin=tmin,
        tmax=tmax,
        bumps=[(0.050, 0.80, 0.014), (0.120, -0.35, 0.022)],
    )
    env_kernel_1, _ = gaussian_kernel(
        fs=fs,
        tmin=tmin,
        tmax=tmax,
        bumps=[(0.080, -0.55, 0.018), (0.155, 0.30, 0.026)],
    )
    onset_kernel_0, _ = gaussian_kernel(
        fs=fs,
        tmin=tmin,
        tmax=tmax,
        bumps=[(0.018, 0.55, 0.008)],
    )
    onset_kernel_1, _ = gaussian_kernel(
        fs=fs,
        tmin=tmin,
        tmax=tmax,
        bumps=[(0.030, 0.45, 0.010), (0.095, -0.18, 0.012)],
    )

    true_weights = np.stack(
        [
            np.column_stack([env_kernel_0, env_kernel_1]),
            np.column_stack([onset_kernel_0, onset_kernel_1]),
        ],
        axis=0,
    )

    stimulus = []
    response = []
    for _ in range(n_trials):
        envelope = make_envelope(n_samples=n_samples, fs=fs, rng=rng)
        onset = make_onset_feature(envelope)
        trial_stimulus = np.column_stack([envelope, onset])
        trial_response = simulate_response(
            trial_stimulus,
            true_weights,
            fs=fs,
            tmin=tmin,
            noise_scale=noise_scale,
            rng=rng,
        )
        stimulus.append(trial_stimulus)
        response.append(trial_response)

    return SimulatedTRFDataset(
        stimulus=stimulus,
        response=response,
        true_weights=true_weights,
        times=times,
        fs=fs,
        tmin=tmin,
        tmax=tmax,
        description="Two-feature, two-channel forward model.",
    )


def build_banded_regularization_dataset(
    *,
    fs: float = 1_000.0,
    n_trials: int = 6,
    n_samples: int = 6_000,
    tmin: float = 0.0,
    tmax: float = 0.220,
    noise_scale: float = 0.07,
    seed: int = 5,
) -> SimulatedTRFDataset:
    """Create a two-feature dataset suited to banded regularization demos."""

    rng = np.random.default_rng(seed)
    envelope_kernel, times = gaussian_kernel(
        fs=fs,
        tmin=tmin,
        tmax=tmax,
        bumps=[(0.045, 0.85, 0.014), (0.115, -0.32, 0.022)],
    )
    onset_kernel, _ = gaussian_kernel(
        fs=fs,
        tmin=tmin,
        tmax=tmax,
        bumps=[(0.014, 0.42, 0.006), (0.030, 0.18, 0.008)],
    )
    true_weights = np.stack([envelope_kernel, onset_kernel], axis=0)[:, :, np.newaxis]

    stimulus = []
    response = []
    for _ in range(n_trials):
        envelope = make_envelope(n_samples=n_samples, fs=fs, rng=rng)
        onset = make_onset_feature(envelope)
        onset += 0.35 * rng.standard_normal(n_samples)
        onset /= np.clip(onset.std(), np.finfo(float).eps, None)
        trial_stimulus = np.column_stack([envelope, onset])
        trial_response = simulate_response(
            trial_stimulus,
            true_weights,
            fs=fs,
            tmin=tmin,
            noise_scale=noise_scale,
            rng=rng,
        )
        stimulus.append(trial_stimulus)
        response.append(trial_response)

    return SimulatedTRFDataset(
        stimulus=stimulus,
        response=response,
        true_weights=true_weights,
        times=times,
        fs=fs,
        tmin=tmin,
        tmax=tmax,
        description="Two-feature forward model for banded regularization.",
    )


def build_backward_decoding_dataset(
    *,
    fs: float = 500.0,
    n_trials: int = 6,
    n_samples: int = 5_000,
    tmin: float = 0.0,
    tmax: float = 0.180,
    noise_scale: float = 0.07,
    seed: int = 4,
) -> SimulatedTRFDataset:
    """Create a dataset for backward decoding from multichannel responses."""

    rng = np.random.default_rng(seed)
    channel_kernels = []
    times = None
    bump_sets = [
        [(0.018, 0.55, 0.010), (0.090, -0.20, 0.018)],
        [(0.028, 0.45, 0.012), (0.110, -0.24, 0.016)],
        [(0.040, 0.38, 0.014), (0.125, -0.18, 0.020)],
        [(0.060, 0.32, 0.018), (0.145, -0.14, 0.024)],
    ]
    for bumps in bump_sets:
        kernel, times = gaussian_kernel(fs=fs, tmin=tmin, tmax=tmax, bumps=bumps)
        channel_kernels.append(kernel)
    true_weights = np.stack(channel_kernels, axis=-1)[np.newaxis, :, :]

    stimulus = []
    response = []
    for _ in range(n_trials):
        envelope = make_envelope(n_samples=n_samples, fs=fs, rng=rng)
        trial_stimulus = envelope[:, np.newaxis]
        trial_response = simulate_response(
            trial_stimulus,
            true_weights,
            fs=fs,
            tmin=tmin,
            noise_scale=noise_scale,
            rng=rng,
        )
        stimulus.append(trial_stimulus)
        response.append(trial_response)

    return SimulatedTRFDataset(
        stimulus=stimulus,
        response=response,
        true_weights=true_weights,
        times=times if times is not None else lag_times(fs, tmin, tmax),
        fs=fs,
        tmin=tmin,
        tmax=tmax,
        description="Backward decoding from four simulated brain channels.",
    )
