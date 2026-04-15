from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Sequence

import fftrf
import fftrf.estimator as estimator_module
import fftrf.prediction as prediction_module
import fftrf.spectral as spectral_module
import numpy as np
import pytest
from scipy.signal import fftconvolve

from fftrf import (
    FrequencyResolvedWeights,
    TRF,
    TimeFrequencyPower,
    available_metrics,
    explained_variance_score,
    half_wave_rectify,
    inverse_variance_weights,
    neg_mse,
    pearsonr,
    r2_score,
    resample_signal,
    suggest_segment_settings,
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


def _direct_transfer_function(
    cxx: np.ndarray,
    cxy: np.ndarray,
    *,
    feature_regularization: np.ndarray,
) -> np.ndarray:
    feature_regularization = np.asarray(feature_regularization, dtype=float)
    if np.allclose(feature_regularization, feature_regularization[0]):
        ridge_matrix = float(feature_regularization[0]) * np.eye(cxx.shape[1], dtype=np.complex128)
    else:
        ridge_matrix = np.diag(feature_regularization.astype(np.complex128))

    transfer_function = np.zeros_like(cxy)
    for frequency_index in range(cxx.shape[0]):
        transfer_function[frequency_index] = np.linalg.solve(
            cxx[frequency_index] + ridge_matrix,
            cxy[frequency_index],
        )
    return transfer_function
def _shifted_convolution_reference(
    signal_in: np.ndarray,
    kernel: np.ndarray,
    *,
    lag_start: int,
    out_length: int,
) -> np.ndarray:
    full = fftconvolve(signal_in, kernel, mode="full")
    offset = -lag_start

    prediction = np.zeros(out_length, dtype=float)
    src_start = max(offset, 0)
    dst_start = max(-offset, 0)
    length = min(full.shape[0] - src_start, out_length - dst_start)
    if length > 0:
        prediction[dst_start : dst_start + length] = full[src_start : src_start + length]
    return prediction


def _predict_trials_from_weights_reference(
    predictor_trials: Sequence[np.ndarray],
    *,
    weights: np.ndarray,
    lag_start: int,
) -> list[np.ndarray]:
    weights = np.asarray(weights, dtype=float)
    n_inputs, _, n_outputs = weights.shape
    kernel = np.transpose(weights, (1, 0, 2))

    predictions: list[np.ndarray] = []
    for predictor_trial in predictor_trials:
        prediction = np.zeros((predictor_trial.shape[0], n_outputs), dtype=float)
        for input_index in range(n_inputs):
            for output_index in range(n_outputs):
                prediction[:, output_index] += _shifted_convolution_reference(
                    predictor_trial[:, input_index],
                    kernel[:, input_index, output_index],
                    lag_start=lag_start,
                    out_length=predictor_trial.shape[0],
                )
        predictions.append(prediction)
    return predictions


def _score_regularization_grid_for_fold_reference(
    *,
    cxx: np.ndarray,
    cxy: np.ndarray,
    val_predictors: Sequence[np.ndarray],
    val_targets: Sequence[np.ndarray],
    feature_regularization_values: Sequence[np.ndarray],
    fs: float,
    n_fft: int,
    tmin: float,
    tmax: float,
    metric,
) -> np.ndarray:
    scores = np.zeros((len(feature_regularization_values), val_targets[0].shape[1]), dtype=float)
    lag_start = int(round(float(tmin) * float(fs)))
    scalar_grid = spectral_module._scalar_regularization_grid(feature_regularization_values)
    scalar_decomposition = (
        spectral_module._prepare_scalar_ridge_decomposition(cxx, cxy)
        if scalar_grid is not None
        else None
    )

    for reg_index, feature_regularization in enumerate(feature_regularization_values):
        transfer_function = spectral_module._solve_transfer_function(
            cxx,
            cxy,
            feature_regularization=feature_regularization,
            scalar_decomposition=scalar_decomposition,
            scalar_regularization=None if scalar_grid is None else float(scalar_grid[reg_index]),
        )
        weights, _ = prediction_module._extract_impulse_response(
            transfer_function,
            fs=float(fs),
            n_fft=n_fft,
            tmin=float(tmin),
            tmax=float(tmax),
        )
        predictions = _predict_trials_from_weights_reference(
            val_predictors,
            weights=weights,
            lag_start=lag_start,
        )
        scores[reg_index, :] = prediction_module._score_prediction_trials(metric, val_targets, predictions)
    return scores


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

    model = TRF(direction=1)
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

    model = TRF(direction=1)
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
    aggregate_flags: list[bool] = []
    original = estimator_module._build_spectral_cache

    def counting_cache(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        aggregate_flags.append(bool(kwargs.get("aggregate_only", False)))
        return original(*args, **kwargs)

    monkeypatch.setattr(estimator_module, "_build_spectral_cache", counting_cache)

    model = TRF(direction=1)
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
    assert aggregate_flags == [False]


def test_fixed_lambda_fit_uses_aggregated_spectra_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(18)
    fs = 1_000
    kernel = np.zeros(30)
    kernel[4] = 0.85
    kernel[10] = -0.30
    kernel[17] = 0.10

    stimulus, response = _simulate_trials(
        rng=rng,
        n_trials=6,
        n_samples=2_048,
        kernel=kernel,
        noise_scale=0.05,
    )

    aggregate_flags: list[bool] = []
    original = estimator_module._build_spectral_cache

    def counting_cache(*args, **kwargs):
        aggregate_flags.append(bool(kwargs.get("aggregate_only", False)))
        return original(*args, **kwargs)

    monkeypatch.setattr(estimator_module, "_build_spectral_cache", counting_cache)

    model = TRF(direction=1)
    model.train(
        stimulus=stimulus,
        response=response,
        fs=fs,
        tmin=0.0,
        tmax=0.030,
        regularization=1e-3,
        segment_length=512,
        overlap=0.5,
    )

    assert aggregate_flags == [True]


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
    original = estimator_module._aggregate_cached_spectra

    def counting_aggregate(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(estimator_module, "_aggregate_cached_spectra", counting_aggregate)

    model = TRF(direction=1)
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


def test_solve_transfer_function_matches_direct_solver() -> None:
    rng = np.random.default_rng(34)
    n_frequencies = 16
    n_inputs = 3
    n_outputs = 2

    a = rng.standard_normal((n_frequencies, n_inputs, n_inputs))
    b = rng.standard_normal((n_frequencies, n_inputs, n_inputs))
    cxx = np.matmul(a + 1j * b, np.swapaxes(np.conjugate(a + 1j * b), 1, 2))
    cxy = rng.standard_normal((n_frequencies, n_inputs, n_outputs)) + 1j * rng.standard_normal(
        (n_frequencies, n_inputs, n_outputs)
    )

    scalar_regularization = np.full(n_inputs, 1e-2)
    direct_scalar = _direct_transfer_function(
        cxx,
        cxy,
        feature_regularization=scalar_regularization,
    )
    optimized_scalar = spectral_module._solve_transfer_function(
        cxx,
        cxy,
        feature_regularization=scalar_regularization,
    )
    assert np.allclose(optimized_scalar, direct_scalar, rtol=1e-10, atol=1e-12)

    feature_regularization = np.array([1e-3, 3e-3, 1e-2])
    direct_featurewise = _direct_transfer_function(
        cxx,
        cxy,
        feature_regularization=feature_regularization,
    )
    optimized_featurewise = spectral_module._solve_transfer_function(
        cxx,
        cxy,
        feature_regularization=feature_regularization,
    )
    assert np.allclose(optimized_featurewise, direct_featurewise, rtol=1e-10, atol=1e-12)


def test_cross_validation_n_jobs_matches_serial_results() -> None:
    rng = np.random.default_rng(35)
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

    serial = TRF(direction=1)
    serial_scores = serial.train(
        stimulus=stimulus,
        response=response,
        fs=fs,
        tmin=0.0,
        tmax=0.030,
        regularization=np.logspace(-5, 0, 6),
        segment_length=512,
        overlap=0.5,
        k=4,
        seed=3,
        n_jobs=1,
    )

    parallel = TRF(direction=1)
    parallel_scores = parallel.train(
        stimulus=stimulus,
        response=response,
        fs=fs,
        tmin=0.0,
        tmax=0.030,
        regularization=np.logspace(-5, 0, 6),
        segment_length=512,
        overlap=0.5,
        k=4,
        seed=3,
        n_jobs=2,
    )

    assert np.allclose(parallel_scores, serial_scores)
    assert parallel.regularization == serial.regularization
    assert np.allclose(parallel.weights, serial.weights, rtol=1e-10, atol=1e-12)
    assert np.allclose(parallel.transfer_function, serial.transfer_function, rtol=1e-10, atol=1e-12)


def test_predict_trials_from_weights_matches_reference_convolution() -> None:
    rng = np.random.default_rng(135)
    predictor_trials = [
        rng.standard_normal((257, 3)),
        rng.standard_normal((191, 3)),
    ]
    weights = rng.standard_normal((3, 17, 2))

    optimized = prediction_module._predict_trials_from_weights(
        predictor_trials,
        weights=weights,
        lag_start=-4,
    )
    reference = _predict_trials_from_weights_reference(
        predictor_trials,
        weights=weights,
        lag_start=-4,
    )

    assert len(optimized) == len(reference)
    for optimized_trial, reference_trial in zip(optimized, reference, strict=True):
        assert np.allclose(optimized_trial, reference_trial, rtol=1e-10, atol=1e-12)


def test_cross_validation_fold_scorer_matches_reference() -> None:
    rng = np.random.default_rng(136)
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

    spectral_cache = estimator_module._build_spectral_cache(
        stimulus,
        response,
        segment_length=512,
        overlap=0.5,
        n_fft=None,
        spectral_method="standard",
        time_bandwidth=3.5,
        n_tapers=None,
        window=None,
        detrend="constant",
    )
    val_idx = np.array([1, 5])
    train_weights = np.ones(len(stimulus), dtype=float)
    train_weights[val_idx] = 0.0
    cxx, cxy = estimator_module._aggregate_cached_spectra(
        spectral_cache,
        raw_trial_weights=train_weights,
    )

    feature_regularization_values = [
        np.full(stimulus[0].shape[1], value, dtype=float)
        for value in np.logspace(-5, 0, 6)
    ]
    val_predictors = [stimulus[i] for i in val_idx]
    val_targets = [response[i] for i in val_idx]

    optimized = prediction_module._score_regularization_grid_for_fold(
        cxx=cxx,
        cxy=cxy,
        val_predictors=val_predictors,
        val_targets=val_targets,
        feature_regularization_values=feature_regularization_values,
        fs=float(fs),
        n_fft=spectral_cache.n_fft,
        tmin=0.0,
        tmax=0.030,
        metric=pearsonr,
    )
    reference = _score_regularization_grid_for_fold_reference(
        cxx=cxx,
        cxy=cxy,
        val_predictors=val_predictors,
        val_targets=val_targets,
        feature_regularization_values=feature_regularization_values,
        fs=float(fs),
        n_fft=spectral_cache.n_fft,
        tmin=0.0,
        tmax=0.030,
        metric=pearsonr,
    )

    assert np.allclose(optimized, reference, rtol=1e-10, atol=1e-12)


def test_banded_cross_validation_fold_scorer_matches_reference() -> None:
    rng = np.random.default_rng(137)
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

    spectral_cache = estimator_module._build_spectral_cache(
        stimulus,
        response,
        segment_length=512,
        overlap=0.5,
        n_fft=None,
        spectral_method="standard",
        time_bandwidth=3.5,
        n_tapers=None,
        window=None,
        detrend="constant",
    )
    val_idx = np.array([0, 4])
    train_weights = np.ones(len(stimulus), dtype=float)
    train_weights[val_idx] = 0.0
    cxx, cxy = estimator_module._aggregate_cached_spectra(
        spectral_cache,
        raw_trial_weights=train_weights,
    )

    feature_regularization_values = [
        np.array(candidate, dtype=float)
        for candidate in [
            (1e-4, 1e-4),
            (1e-4, 1e-1),
            (1e-1, 1e-4),
            (1e-1, 1e-1),
        ]
    ]
    val_predictors = [stimulus[i] for i in val_idx]
    val_targets = [response[i] for i in val_idx]

    optimized = prediction_module._score_regularization_grid_for_fold(
        cxx=cxx,
        cxy=cxy,
        val_predictors=val_predictors,
        val_targets=val_targets,
        feature_regularization_values=feature_regularization_values,
        fs=float(fs),
        n_fft=spectral_cache.n_fft,
        tmin=0.0,
        tmax=0.004,
        metric=pearsonr,
    )
    reference = _score_regularization_grid_for_fold_reference(
        cxx=cxx,
        cxy=cxy,
        val_predictors=val_predictors,
        val_targets=val_targets,
        feature_regularization_values=feature_regularization_values,
        fs=float(fs),
        n_fft=spectral_cache.n_fft,
        tmin=0.0,
        tmax=0.004,
        metric=pearsonr,
    )

    assert np.allclose(optimized, reference, rtol=1e-10, atol=1e-12)


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

    by_samples = TRF(direction=1)
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

    by_seconds = TRF(direction=1)
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

    model = TRF(direction=1)
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

    model = TRF(direction=1)
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

    model = TRF(direction=1)
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
        n_jobs=2,
    )

    captured = capsys.readouterr()
    assert "Cross-validating" in captured.err
    assert "1/15" in captured.err
    assert "15/15" in captured.err


def test_builtin_metric_helpers_and_registry() -> None:
    y_true = np.array([[0.0], [1.0], [2.0], [3.0]])
    y_pred = np.array([[0.0], [0.8], [2.2], [3.1]])

    assert "r2" in available_metrics()
    assert "explained_variance" in available_metrics()
    assert "neg_mse" in available_metrics()
    assert float(r2_score(y_true, y_pred)[0]) > 0.95
    assert float(explained_variance_score(y_true, y_pred)[0]) > 0.95
    assert float(neg_mse(y_true, y_pred)[0]) == pytest.approx(-0.0225)


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

    scalar_model = TRF(direction=1)
    scalar_model.train(
        stimulus=stimulus,
        response=response,
        fs=fs,
        tmin=0.0,
        tmax=0.004,
        regularization=1e-3,
        window=None,
    )

    banded_model = TRF(direction=1)
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

    model = TRF(direction=1)
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

    model = TRF(direction=1, metric="r2")
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


def test_trf_supports_negative_mse_metric() -> None:
    rng = np.random.default_rng(124)
    fs = 1_000
    kernel = np.zeros(30)
    kernel[2] = 0.7
    kernel[8] = -0.2

    stimulus, response = _simulate_trials(
        rng=rng,
        n_trials=6,
        n_samples=2_048,
        kernel=kernel,
        noise_scale=0.03,
    )

    model = TRF(direction=1, metric="neg_mse")
    cv_scores = model.train(
        stimulus=stimulus[:-1],
        response=response[:-1],
        fs=fs,
        tmin=0.0,
        tmax=0.030,
        regularization=np.logspace(-5, -1, 5),
        k=3,
        segment_duration=0.512,
    )

    _, score = model.predict(stimulus=stimulus[-1], response=response[-1])
    assert cv_scores.shape == (5,)
    assert model.metric_name == "neg_mse"
    assert score < 0.0


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

    model = TRF(direction=1)
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

    model = TRF(direction=1)
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

    resolved_subset = model.frequency_resolved_weights(n_bands=12, fmax=80.0)
    time_frequency_power = model.time_frequency_power(n_bands=12, fmax=80.0)
    assert isinstance(time_frequency_power, TimeFrequencyPower)
    assert time_frequency_power.power.shape == resolved_subset.weights.shape
    assert np.all(time_frequency_power.power >= 0.0)
    assert np.allclose(time_frequency_power.band_centers, resolved_subset.band_centers)
    assert np.allclose(time_frequency_power.times, resolved_subset.times)


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

    model = TRF(direction=1)
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

    model = TRF(direction=1)
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


def test_suggest_segment_settings_prefers_overlapping_hann_segments() -> None:
    suggestion = suggest_segment_settings(
        fs=128.0,
        tmin=0.0,
        tmax=0.350,
        trial_duration=60.0,
    )

    assert suggestion == {
        "segment_length": 256,
        "segment_duration": 2.0,
        "overlap": 0.5,
        "window": "hann",
    }


def test_suggest_segment_settings_prefers_full_trial_for_short_trials() -> None:
    suggestion = suggest_segment_settings(
        fs=128.0,
        tmin=0.0,
        tmax=0.350,
        trial_duration=1.0,
    )

    assert suggestion == {
        "segment_length": None,
        "segment_duration": None,
        "overlap": 0.0,
        "window": None,
    }


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

    model = TRF(direction=1)
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


def test_bootstrap_n_jobs_matches_serial_results() -> None:
    rng = np.random.default_rng(36)
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

    serial = TRF(direction=1)
    serial.train(
        stimulus=stimulus,
        response=response,
        fs=fs,
        tmin=0.0,
        tmax=0.040,
        regularization=1e-3,
        segment_length=512,
        overlap=0.5,
        bootstrap_samples=12,
        bootstrap_level=0.9,
        bootstrap_seed=2,
        n_jobs=1,
    )

    parallel = TRF(direction=1)
    parallel.train(
        stimulus=stimulus,
        response=response,
        fs=fs,
        tmin=0.0,
        tmax=0.040,
        regularization=1e-3,
        segment_length=512,
        overlap=0.5,
        bootstrap_samples=12,
        bootstrap_level=0.9,
        bootstrap_seed=2,
        n_jobs=2,
    )

    assert np.allclose(parallel.weights, serial.weights, rtol=1e-10, atol=1e-12)
    assert np.allclose(parallel.bootstrap_interval, serial.bootstrap_interval, rtol=1e-10, atol=1e-12)


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

    model = TRF(direction=1)
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

    model = TRF(direction=1)
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

    model = TRF(direction=1)
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

    model = TRF(direction=1)
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

    power = model.time_frequency_power(n_bands=14)
    power_fig, power_ax = model.plot_time_frequency_power(power=power)
    assert power_ax.get_xlabel() == "Lag (ms)"
    assert power_ax.get_ylabel() == "Frequency (Hz)"
    plt.close(power_fig)


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

    model = TRF(direction=1)
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


def test_top_level_api_exports_expected_symbols() -> None:
    assert fftrf.TRF is TRF
    assert fftrf.FrequencyResolvedWeights is FrequencyResolvedWeights
    assert fftrf.TimeFrequencyPower is TimeFrequencyPower
    assert callable(fftrf.available_metrics)
