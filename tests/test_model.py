from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np

from fft_trf import FrequencyTRF, half_wave_rectify, inverse_variance_weights, resample_signal


def _simulate_trials(
    *,
    rng: np.random.Generator,
    n_trials: int,
    n_samples: int,
    kernel: np.ndarray,
    noise_scale: float,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    stimulus = []
    response = []
    for _ in range(n_trials):
        x = rng.standard_normal((n_samples, 1))
        y = np.convolve(x[:, 0], kernel, mode="full")[:n_samples]
        y = y + noise_scale * rng.standard_normal(n_samples)
        stimulus.append(x)
        response.append(y[:, np.newaxis])
    return stimulus, response


def test_frequency_trf_recovers_impulse_response() -> None:
    rng = np.random.default_rng(2)
    fs = 1_000
    kernel = np.zeros(40)
    kernel[4] = 1.0
    kernel[11] = -0.45
    kernel[19] = 0.2

    stimulus, response = _simulate_trials(
        rng=rng,
        n_trials=8,
        n_samples=4_096,
        kernel=kernel,
        noise_scale=0.02,
    )

    model = FrequencyTRF(direction=1)
    model.train(
        stimulus=stimulus,
        response=response,
        fs=fs,
        tmin=0.0,
        tmax=0.040,
        regularization=1e-3,
        segment_length=512,
        overlap=0.5,
    )

    recovered = model.weights[0, :, 0]
    correlation = np.corrcoef(kernel, recovered)[0, 1]
    assert correlation > 0.95


def test_train_selects_regularization_and_predicts_held_out_data() -> None:
    rng = np.random.default_rng(7)
    fs = 1_000
    kernel = np.zeros(30)
    kernel[3] = 0.9
    kernel[8] = -0.35
    kernel[15] = 0.12

    stimulus, response = _simulate_trials(
        rng=rng,
        n_trials=10,
        n_samples=3_072,
        kernel=kernel,
        noise_scale=0.05,
    )

    train_x, test_x = stimulus[:8], stimulus[8:]
    train_y, test_y = response[:8], response[8:]

    model = FrequencyTRF(direction=1)
    scores = model.train(
        stimulus=train_x,
        response=train_y,
        fs=fs,
        tmin=0.0,
        tmax=0.030,
        regularization=np.logspace(-6, 0, 7),
        segment_length=512,
        overlap=0.5,
        k=4,
        trial_weights="inverse_variance",
    )

    assert scores.shape == (7,)
    _, held_out_score = model.predict(stimulus=test_x, response=test_y)
    assert held_out_score > 0.8


def test_frequency_trf_matches_time_domain_ridge_lambda_scale() -> None:
    rng = np.random.default_rng(21)
    fs = 1_000
    n_samples = 2_048
    kernel = np.zeros(40)
    kernel[3] = 1.0
    kernel[9] = -0.4
    kernel[18] = 0.2

    x = rng.standard_normal((n_samples, 1))
    y = np.convolve(x[:, 0], kernel, mode="full")[:n_samples]
    y += 0.1 * rng.standard_normal(n_samples)

    lags = np.arange(40)
    design = np.zeros((n_samples, len(lags)))
    for index, lag in enumerate(lags):
        design[lag:, index] = x[: n_samples - lag, 0]

    regularization = 1_000.0
    reference = np.linalg.solve(
        design.T @ design + regularization * np.eye(len(lags)),
        design.T @ y,
    )

    model = FrequencyTRF(direction=1)
    model.train(
        stimulus=x,
        response=y[:, np.newaxis],
        fs=fs,
        tmin=0.0,
        tmax=0.040,
        regularization=regularization,
        window=None,
    )

    recovered = model.weights[0, :, 0]
    assert np.corrcoef(reference, recovered)[0, 1] > 0.99
    assert 0.7 < (np.linalg.norm(recovered) / np.linalg.norm(reference)) < 1.3


def test_multichannel_prediction_and_helpers() -> None:
    rng = np.random.default_rng(11)
    fs = 500
    n_samples = 2_048

    k00 = np.zeros(25)
    k00[2] = 0.8
    k00[7] = -0.2
    k10 = np.zeros(25)
    k10[4] = 0.35
    k01 = np.zeros(25)
    k01[1] = -0.1
    k11 = np.zeros(25)
    k11[3] = 0.6

    stimulus = []
    response = []
    for _ in range(6):
        x = rng.standard_normal((n_samples, 2))
        y0 = np.convolve(x[:, 0], k00, mode="full")[:n_samples]
        y0 += np.convolve(x[:, 1], k10, mode="full")[:n_samples]
        y1 = np.convolve(x[:, 0], k01, mode="full")[:n_samples]
        y1 += np.convolve(x[:, 1], k11, mode="full")[:n_samples]
        y = np.column_stack([y0, y1])
        y += 0.03 * rng.standard_normal(y.shape)
        stimulus.append(x)
        response.append(y)

    model = FrequencyTRF(direction=1)
    model.train(
        stimulus=stimulus[:-1],
        response=response[:-1],
        fs=fs,
        tmin=0.0,
        tmax=0.050,
        regularization=1e-3,
        segment_length=256,
        overlap=0.5,
    )

    prediction, scores = model.predict(
        stimulus=stimulus[-1],
        response=response[-1],
        average=False,
    )
    assert prediction.shape == response[-1].shape
    assert scores.shape == (2,)
    assert np.all(scores > 0.75)

    pos, neg = half_wave_rectify(np.array([-2.0, -1.0, 0.5, 1.5]))
    assert np.allclose(pos, [0.0, 0.0, 0.5, 1.5])
    assert np.allclose(neg, [2.0, 1.0, 0.0, 0.0])

    weights = inverse_variance_weights(response)
    assert np.isclose(weights.sum(), 1.0)

    resampled = resample_signal(np.arange(100.0), orig_fs=100.0, target_fs=50.0)
    assert abs(resampled.shape[0] - 50) <= 1


def test_optional_comparison_helper_runs_without_mtrf_dependency() -> None:
    comparison_path = Path(__file__).resolve().parent.parent / "examples" / "comparison_utils.py"
    spec = importlib.util.spec_from_file_location("comparison_utils", comparison_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    result = module.compare_simulated_kernels(
        fs=1_000,
        n_trials=6,
        n_samples=2_048,
        regularization=1e-3,
        include_mtrf=False,
    )

    assert result.true_kernel.shape == result.fft_trf_kernel.shape
    assert result.true_kernel.shape == result.time_domain_kernel.shape
    assert result.mtrf_kernel is None
    assert result.metrics["fft_vs_true"] > 0.95
    assert result.metrics["fft_vs_time"] > 0.95
