from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Sequence

import fftrf.model as model_module
import numpy as np
import pytest

from fftrf import (
    FrequencyResolvedWeights,
    FrequencyTRF,
    available_metrics,
    explained_variance_score,
    half_wave_rectify,
    inverse_variance_weights,
    r2_score,
    resample_signal,
)


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


def _simulate_multifeature_trials(
    *,
    rng: np.random.Generator,
    n_trials: int,
    n_samples: int,
    kernels: Sequence[np.ndarray],
    noise_scale: float,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    stimulus = []
    response = []
    for _ in range(n_trials):
        x = rng.standard_normal((n_samples, len(kernels)))
        y = np.zeros(n_samples, dtype=float)
        for feature_index, kernel in enumerate(kernels):
            y += np.convolve(x[:, feature_index], kernel, mode="full")[:n_samples]
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


def test_cross_validation_builds_spectral_cache_once(monkeypatch: pytest.MonkeyPatch) -> None:
    rng = np.random.default_rng(8)
    fs = 1_000
    kernel = np.zeros(30)
    kernel[3] = 0.9
    kernel[8] = -0.35
    kernel[15] = 0.12

    stimulus, response = _simulate_trials(
        rng=rng,
        n_trials=8,
        n_samples=2_048,
        kernel=kernel,
        noise_scale=0.05,
    )

    call_count = 0
    original = model_module._build_spectral_cache

    def counting_cache(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(model_module, "_build_spectral_cache", counting_cache)

    model = FrequencyTRF(direction=1)
    scores = model.train(
        stimulus=stimulus,
        response=response,
        fs=fs,
        tmin=0.0,
        tmax=0.030,
        regularization=np.logspace(-5, 0, 6),
        segment_length=512,
        overlap=0.5,
        k=4,
    )

    assert scores.shape == (6,)
    assert call_count == 1


def test_cross_validation_aggregates_fold_spectra_once_per_fold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(29)
    fs = 1_000
    kernel = np.zeros(30)
    kernel[3] = 0.9
    kernel[8] = -0.35
    kernel[15] = 0.12

    stimulus, response = _simulate_trials(
        rng=rng,
        n_trials=8,
        n_samples=2_048,
        kernel=kernel,
        noise_scale=0.05,
    )

    call_count = 0
    original = model_module._aggregate_cached_spectra

    def counting_aggregate(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(model_module, "_aggregate_cached_spectra", counting_aggregate)

    model = FrequencyTRF(direction=1)
    model.train(
        stimulus=stimulus,
        response=response,
        fs=fs,
        tmin=0.0,
        tmax=0.030,
        regularization=np.logspace(-5, 0, 6),
        segment_length=512,
        overlap=0.5,
        k=4,
    )

    assert call_count == 5


def test_segment_duration_alias_matches_segment_length() -> None:
    rng = np.random.default_rng(30)
    fs = 1_000
    kernel = np.zeros(25)
    kernel[4] = 0.95
    kernel[9] = -0.3

    stimulus, response = _simulate_trials(
        rng=rng,
        n_trials=5,
        n_samples=2_048,
        kernel=kernel,
        noise_scale=0.04,
    )

    by_samples = FrequencyTRF(direction=1)
    by_samples.train(
        stimulus=stimulus,
        response=response,
        fs=fs,
        tmin=0.0,
        tmax=0.025,
        regularization=1e-3,
        segment_length=512,
        overlap=0.5,
        window="hann",
    )

    by_seconds = FrequencyTRF(direction=1)
    by_seconds.train(
        stimulus=stimulus,
        response=response,
        fs=fs,
        tmin=0.0,
        tmax=0.025,
        regularization=1e-3,
        segment_duration=0.512,
        overlap=0.5,
        window="hann",
    )

    assert by_seconds.segment_length == 512
    assert by_seconds.segment_duration == pytest.approx(0.512)
    assert np.allclose(by_samples.weights, by_seconds.weights)


def test_k_accepts_loo_alias() -> None:
    rng = np.random.default_rng(31)
    fs = 1_000
    kernel = np.zeros(25)
    kernel[3] = 0.9
    kernel[7] = -0.25

    stimulus, response = _simulate_trials(
        rng=rng,
        n_trials=6,
        n_samples=2_048,
        kernel=kernel,
        noise_scale=0.04,
    )

    model = FrequencyTRF(direction=1)
    scores = model.train(
        stimulus=stimulus,
        response=response,
        fs=fs,
        tmin=0.0,
        tmax=0.025,
        regularization=np.logspace(-5, -1, 5),
        segment_duration=0.512,
        overlap=0.5,
        k="loo",
    )

    assert scores.shape == (5,)


def test_single_lambda_warns_about_unused_cv_arguments() -> None:
    rng = np.random.default_rng(32)
    fs = 1_000
    kernel = np.zeros(20)
    kernel[2] = 0.8
    kernel[6] = -0.2

    stimulus, response = _simulate_trials(
        rng=rng,
        n_trials=4,
        n_samples=2_048,
        kernel=kernel,
        noise_scale=0.04,
    )

    model = FrequencyTRF(direction=1)
    with pytest.warns(UserWarning, match="ignored because cross-validation requires more than one regularization candidate"):
        model.train(
            stimulus=stimulus,
            response=response,
            fs=fs,
            tmin=0.0,
            tmax=0.020,
            regularization=1e-3,
            segment_duration=0.512,
            k=4,
            show_progress=True,
        )


def test_cv_progress_indicator_emits_output(capsys: pytest.CaptureFixture[str]) -> None:
    rng = np.random.default_rng(33)
    fs = 1_000
    kernel = np.zeros(25)
    kernel[3] = 0.85
    kernel[8] = -0.25

    stimulus, response = _simulate_trials(
        rng=rng,
        n_trials=5,
        n_samples=2_048,
        kernel=kernel,
        noise_scale=0.04,
    )

    model = FrequencyTRF(direction=1)
    model.train(
        stimulus=stimulus,
        response=response,
        fs=fs,
        tmin=0.0,
        tmax=0.025,
        regularization=np.logspace(-5, -1, 5),
        segment_duration=0.512,
        k=3,
        show_progress=True,
    )

    captured = capsys.readouterr()
    assert "Cross-validating" in captured.err
    assert "15/15" in captured.err


def test_builtin_metric_helpers_and_registry() -> None:
    y_true = np.array([[0.0], [1.0], [2.0], [3.0]])
    y_pred = np.array([[0.0], [0.8], [2.2], [3.1]])

    assert "r2" in available_metrics()
    assert "explained_variance" in available_metrics()
    assert float(r2_score(y_true, y_pred)[0]) > 0.95
    assert float(explained_variance_score(y_true, y_pred)[0]) > 0.95


def test_banded_regularization_matches_scalar_ridge_for_equal_penalties() -> None:
    rng = np.random.default_rng(18)
    fs = 1_000
    kernels = [
        np.array([0.0, 0.8, -0.25, 0.1]),
        np.array([0.0, 0.2, 0.0, -0.05]),
    ]
    stimulus, response = _simulate_multifeature_trials(
        rng=rng,
        n_trials=6,
        n_samples=2_048,
        kernels=kernels,
        noise_scale=0.03,
    )

    scalar_model = FrequencyTRF(direction=1)
    scalar_model.train(
        stimulus=stimulus,
        response=response,
        fs=fs,
        tmin=0.0,
        tmax=0.004,
        regularization=1e-3,
        window=None,
    )

    banded_model = FrequencyTRF(direction=1)
    banded_model.train(
        stimulus=stimulus,
        response=response,
        fs=fs,
        tmin=0.0,
        tmax=0.004,
        regularization=[(1e-3, 1e-3)],
        bands=[1, 1],
        window=None,
    )

    assert banded_model.regularization == (1e-3, 1e-3)
    assert np.allclose(banded_model.feature_regularization, [1e-3, 1e-3])
    assert np.allclose(scalar_model.weights, banded_model.weights, rtol=1e-7, atol=1e-9)


def test_banded_regularization_cross_validation_expands_cartesian_grid() -> None:
    rng = np.random.default_rng(23)
    fs = 1_000
    kernels = [
        np.array([0.0, 0.9, -0.3, 0.15]),
        np.array([0.0, 0.05, 0.02, 0.0]),
    ]
    stimulus, response = _simulate_multifeature_trials(
        rng=rng,
        n_trials=6,
        n_samples=2_048,
        kernels=kernels,
        noise_scale=0.04,
    )

    model = FrequencyTRF(direction=1)
    scores = model.train(
        stimulus=stimulus,
        response=response,
        fs=fs,
        tmin=0.0,
        tmax=0.004,
        regularization=[1e-4, 1e-1],
        bands=[1, 1],
        k=3,
        segment_length=512,
        overlap=0.5,
    )

    assert scores.shape == (4,)
    assert model.regularization_candidates == [
        (1e-4, 1e-4),
        (1e-4, 1e-1),
        (1e-1, 1e-4),
        (1e-1, 1e-1),
    ]
    assert isinstance(model.regularization, tuple)
    assert len(model.regularization) == 2
    assert model.feature_regularization.shape == (2,)


def test_frequency_trf_supports_named_metric_and_multitaper() -> None:
    rng = np.random.default_rng(24)
    fs = 1_000
    kernel = np.zeros(35)
    kernel[3] = 0.9
    kernel[9] = -0.35
    kernel[16] = 0.15

    stimulus, response = _simulate_trials(
        rng=rng,
        n_trials=8,
        n_samples=2_048,
        kernel=kernel,
        noise_scale=0.05,
    )

    model = FrequencyTRF(direction=1, metric="r2")
    model.train_multitaper(
        stimulus=stimulus[:-1],
        response=response[:-1],
        fs=fs,
        tmin=0.0,
        tmax=0.035,
        regularization=np.logspace(-5, -1, 5),
        segment_length=512,
        overlap=0.5,
        time_bandwidth=3.5,
        n_tapers=4,
        k=4,
    )

    _, score = model.predict(stimulus=stimulus[-1], response=response[-1])
    assert model.metric_name == "r2"
    assert model.spectral_method == "multitaper"
    assert model.n_tapers == 4
    assert model.time_bandwidth == pytest.approx(3.5)
    assert score > 0.75


def test_frequency_trf_diagnostics_returns_coherence() -> None:
    rng = np.random.default_rng(25)
    fs = 1_000
    kernel = np.zeros(30)
    kernel[4] = 1.0
    kernel[10] = -0.42
    kernel[17] = 0.18

    stimulus, response = _simulate_trials(
        rng=rng,
        n_trials=6,
        n_samples=2_048,
        kernel=kernel,
        noise_scale=0.04,
    )

    model = FrequencyTRF(direction=1)
    model.train_multitaper(
        stimulus=stimulus[:-1],
        response=response[:-1],
        fs=fs,
        tmin=0.0,
        tmax=0.030,
        regularization=1e-3,
        segment_length=512,
        overlap=0.5,
        time_bandwidth=3.5,
        n_tapers=4,
    )

    diagnostics = model.cross_spectral_diagnostics(
        stimulus=stimulus[-1],
        response=response[-1],
    )
    assert diagnostics.transfer_function.shape == model.transfer_function.shape
    assert diagnostics.predicted_spectrum.shape == (model.frequencies.shape[0], 1)
    assert diagnostics.observed_spectrum.shape == (model.frequencies.shape[0], 1)
    assert diagnostics.cross_spectrum.shape == (model.frequencies.shape[0], 1)
    assert diagnostics.coherence.shape == (model.frequencies.shape[0], 1)
    assert np.all((diagnostics.coherence >= 0.0) & (diagnostics.coherence <= 1.0))
    assert float(np.mean(diagnostics.coherence[:30, 0])) > 0.7
    components = model.transfer_function_components_at()
    assert components.magnitude.shape == model.frequencies.shape
    assert components.phase.shape == model.frequencies.shape
    assert components.group_delay.shape == model.frequencies.shape


def test_frequency_resolved_weights_reconstruct_kernel() -> None:
    rng = np.random.default_rng(27)
    fs = 1_000
    times = np.arange(0, 0.080, 1.0 / fs)
    kernel = (
        np.exp(-0.5 * ((times - 0.020) / 0.006) ** 2) * np.cos(2.0 * np.pi * 18.0 * times)
        + 0.6 * np.exp(-0.5 * ((times - 0.050) / 0.004) ** 2) * np.cos(2.0 * np.pi * 55.0 * times)
    )

    stimulus, response = _simulate_trials(
        rng=rng,
        n_trials=6,
        n_samples=4_096,
        kernel=kernel,
        noise_scale=0.03,
    )

    model = FrequencyTRF(direction=1)
    model.train(
        stimulus=stimulus,
        response=response,
        fs=fs,
        tmin=0.0,
        tmax=0.080,
        regularization=1e-3,
        segment_length=1024,
        overlap=0.5,
        window="hann",
    )

    resolved = model.frequency_resolved_weights(n_bands=12)
    assert isinstance(resolved, FrequencyResolvedWeights)
    assert resolved.weights.shape == (1, 12, model.weights.shape[1], 1)
    assert resolved.at().shape == (12, model.weights.shape[1])
    assert np.allclose(resolved.weights.sum(axis=1), model.weights, atol=1e-7, rtol=1e-7)

    magnitude = model.frequency_resolved_weights(n_bands=10, value_mode="magnitude")
    assert np.all(magnitude.weights >= 0.0)
    assert magnitude.value_mode == "magnitude"


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


def test_frequency_trf_stores_bootstrap_interval() -> None:
    rng = np.random.default_rng(12)
    fs = 1_000
    kernel = np.zeros(40)
    kernel[4] = 1.0
    kernel[11] = -0.45
    kernel[19] = 0.2

    stimulus, response = _simulate_trials(
        rng=rng,
        n_trials=6,
        n_samples=3_072,
        kernel=kernel,
        noise_scale=0.03,
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
        bootstrap_samples=24,
        bootstrap_level=0.9,
        bootstrap_seed=0,
    )

    interval, times = model.bootstrap_interval_at()
    assert interval.shape == (2, *model.weights.shape)
    assert times.shape == model.times.shape
    assert model.bootstrap_level == 0.9
    assert model.bootstrap_samples == 24
    assert np.all(interval[0] <= interval[1])
    assert np.mean((model.weights >= interval[0]) & (model.weights <= interval[1])) > 0.7


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

    assert result.true_kernel.shape == result.fftrf_kernel.shape
    assert result.true_kernel.shape == result.time_domain_kernel.shape
    assert result.mtrf_kernel is None
    assert result.metrics["fft_vs_true"] > 0.95
    assert result.metrics["fft_vs_time"] > 0.95


def test_frequency_trf_plot_if_matplotlib_available() -> None:
    plt = pytest.importorskip("matplotlib.pyplot")

    rng = np.random.default_rng(19)
    fs = 1_000
    kernel = np.zeros(20)
    kernel[2] = 0.9
    kernel[5] = -0.25

    stimulus, response = _simulate_trials(
        rng=rng,
        n_trials=4,
        n_samples=1_024,
        kernel=kernel,
        noise_scale=0.03,
    )

    model = FrequencyTRF(direction=1)
    model.train(
        stimulus=stimulus,
        response=response,
        fs=fs,
        tmin=0.0,
        tmax=0.020,
        regularization=1e-3,
    )

    fig, ax = model.plot(label="ffTRF")
    assert ax.get_xlabel() == "Lag (ms)"
    assert ax.get_ylabel() == "Weight"
    plt.close(fig)


def test_frequency_trf_plot_grid_if_matplotlib_available() -> None:
    plt = pytest.importorskip("matplotlib.pyplot")

    rng = np.random.default_rng(20)
    fs = 500
    n_samples = 2_048

    stimulus = []
    response = []
    for _ in range(5):
        x = rng.standard_normal((n_samples, 2))
        y = np.column_stack(
            [
                np.convolve(x[:, 0], np.array([0.0, 0.8, -0.2]), mode="full")[:n_samples]
                + np.convolve(x[:, 1], np.array([0.0, 0.0, 0.3]), mode="full")[:n_samples],
                np.convolve(x[:, 0], np.array([0.0, -0.1, 0.4]), mode="full")[:n_samples]
                + np.convolve(x[:, 1], np.array([0.0, 0.6, 0.0]), mode="full")[:n_samples],
            ]
        )
        y += 0.04 * rng.standard_normal(y.shape)
        stimulus.append(x)
        response.append(y)

    model = FrequencyTRF(direction=1)
    model.train(
        stimulus=stimulus,
        response=response,
        fs=fs,
        tmin=0.0,
        tmax=0.030,
        regularization=1e-3,
        segment_length=256,
        overlap=0.5,
        bootstrap_samples=12,
        bootstrap_seed=1,
    )

    fig, axes = model.plot_grid(
        input_labels=["Feature 1", "Feature 2"],
        output_labels=["Channel 1", "Channel 2"],
        show_bootstrap_interval=True,
    )
    assert axes.shape == (2, 2)
    assert axes[1, 0].get_xlabel() == "Lag (ms)"
    assert "Feature 1" in axes[0, 0].get_title()
    plt.close(fig)


def test_frequency_trf_transfer_and_coherence_plots_if_matplotlib_available() -> None:
    plt = pytest.importorskip("matplotlib.pyplot")

    rng = np.random.default_rng(26)
    fs = 1_000
    kernel = np.zeros(24)
    kernel[3] = 0.85
    kernel[7] = -0.22

    stimulus, response = _simulate_trials(
        rng=rng,
        n_trials=5,
        n_samples=2_048,
        kernel=kernel,
        noise_scale=0.04,
    )

    model = FrequencyTRF(direction=1)
    model.train_multitaper(
        stimulus=stimulus[:-1],
        response=response[:-1],
        fs=fs,
        tmin=0.0,
        tmax=0.024,
        regularization=1e-3,
        segment_length=512,
        overlap=0.5,
        time_bandwidth=3.5,
        n_tapers=4,
    )

    transfer_fig, transfer_axes = model.plot_transfer_function(kind="all", phase_unit="deg")
    assert transfer_axes.shape == (3,)
    assert transfer_axes[2].get_xlabel() == "Frequency (Hz)"
    plt.close(transfer_fig)

    diagnostics = model.cross_spectral_diagnostics(stimulus=stimulus[-1], response=response[-1])
    coherence_fig, coherence_ax = model.plot_coherence(diagnostics=diagnostics)
    assert coherence_ax.get_ylabel() == "Coherence"
    plt.close(coherence_fig)

    cross_fig, cross_axes = model.plot_cross_spectrum(
        diagnostics=diagnostics,
        phase_unit="deg",
    )
    assert cross_axes.shape == (2,)
    assert cross_axes[1].get_xlabel() == "Frequency (Hz)"
    plt.close(cross_fig)


def test_frequency_resolved_weight_plot_if_matplotlib_available() -> None:
    plt = pytest.importorskip("matplotlib.pyplot")

    rng = np.random.default_rng(28)
    fs = 1_000
    times = np.arange(0, 0.060, 1.0 / fs)
    kernel = (
        np.exp(-0.5 * ((times - 0.018) / 0.004) ** 2) * np.cos(2.0 * np.pi * 22.0 * times)
        + 0.5 * np.exp(-0.5 * ((times - 0.042) / 0.003) ** 2) * np.cos(2.0 * np.pi * 70.0 * times)
    )

    stimulus, response = _simulate_trials(
        rng=rng,
        n_trials=5,
        n_samples=3_072,
        kernel=kernel,
        noise_scale=0.03,
    )

    model = FrequencyTRF(direction=1)
    model.train(
        stimulus=stimulus,
        response=response,
        fs=fs,
        tmin=0.0,
        tmax=0.060,
        regularization=1e-3,
        segment_length=1024,
        overlap=0.5,
        window="hann",
    )

    resolved = model.frequency_resolved_weights(n_bands=14)
    fig, ax = model.plot_frequency_resolved_weights(resolved=resolved)
    assert ax.get_xlabel() == "Lag (ms)"
    assert ax.get_ylabel() == "Frequency (Hz)"
    plt.close(fig)


def test_frequency_trf_plot_rejects_invalid_indices_if_matplotlib_available() -> None:
    pytest.importorskip("matplotlib.pyplot")

    rng = np.random.default_rng(22)
    fs = 1_000
    kernel = np.zeros(20)
    kernel[2] = 0.9
    kernel[5] = -0.25

    stimulus, response = _simulate_trials(
        rng=rng,
        n_trials=3,
        n_samples=2_048,
        kernel=kernel,
        noise_scale=0.04,
    )

    model = FrequencyTRF(direction=1)
    model.train(
        stimulus=stimulus,
        response=response,
        fs=fs,
        tmin=0.0,
        tmax=0.020,
        regularization=1e-3,
    )

    with pytest.raises(IndexError):
        model.plot(input_index=1)
    with pytest.raises(IndexError):
        model.plot(output_index=1)


def test_legacy_fft_trf_import_aliases_new_package() -> None:
    import fft_trf
    import fft_trf.model as legacy_model_module

    assert fft_trf.FrequencyTRF is FrequencyTRF
    assert legacy_model_module.FrequencyTRF is model_module.FrequencyTRF
    assert hasattr(legacy_model_module, "_build_spectral_cache")
