from __future__ import annotations

import numpy as np
import pytest

from fft_trf import (
    BayesianFrequencyTRF,
    fit_bayesian_frequency_trf,
    predict_bayesian_frequency_trf,
)


def _lagged_design_matrix(x: np.ndarray, lags: np.ndarray) -> np.ndarray:
    design = np.zeros((x.shape[0], len(lags)), dtype=float)
    for index, lag in enumerate(lags):
        if lag >= 0:
            design[lag:, index] = x[: x.shape[0] - lag]
        else:
            step = -lag
            design[: x.shape[0] - step, index] = x[step:]
    return design


def _simulate(
    *,
    rng: np.random.Generator,
    n_trials: int,
    n_samples: int,
    kernel: np.ndarray,
    lag_start: int,
    noise_scale: float,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    stimulus = []
    response = []
    for _ in range(n_trials):
        x = rng.standard_normal(n_samples)
        full = np.convolve(x, kernel, mode="full")
        y = np.zeros(n_samples)
        offset = -lag_start
        src_start = max(offset, 0)
        dst_start = max(-offset, 0)
        length = min(full.shape[0] - src_start, n_samples - dst_start)
        if length > 0:
            y[dst_start : dst_start + length] = full[src_start : src_start + length]
        y += noise_scale * rng.standard_normal(n_samples)
        stimulus.append(x)
        response.append(y)
    return stimulus, response


def test_bayesian_frequency_trf_recovers_kernel() -> None:
    rng = np.random.default_rng(31)
    fs = 1_000
    tmin = 0.0
    tmax = 0.040
    lag_start = int(round(tmin * fs))
    kernel = np.zeros(int(round((tmax - tmin) * fs)))
    kernel[3] = 1.0
    kernel[9] = -0.4
    kernel[18] = 0.2

    stimulus, response = _simulate(
        rng=rng,
        n_trials=8,
        n_samples=4_096,
        kernel=kernel,
        lag_start=lag_start,
        noise_scale=0.05,
    )

    model = BayesianFrequencyTRF(direction=1)
    trained = model.train(
        stimulus,
        response,
        fs=fs,
        tmin=tmin,
        tmax=tmax,
        prior="smooth",
    )
    result = model.result()

    assert trained is None
    assert model.weights.shape == (1, kernel.size, 1)
    assert model.transfer_function.ndim == 3
    assert np.corrcoef(result.weights[0, :, 0], kernel)[0, 1] > 0.95
    assert np.corrcoef(model.weights[0, :, 0], kernel)[0, 1] > 0.95
    assert result.posterior_cov.shape == (1, kernel.size, kernel.size)
    assert result.posterior_std.shape == (1, kernel.size, 1)
    assert result.credible_interval.shape == (2, 1, kernel.size, 1)
    assert np.all(result.alpha > 0)
    assert np.all(result.beta > 0)
    assert result.fit_mode == "evidence"
    assert result.direction == 1
    assert result.fs == fs
    assert result.segment_length == max(trial.shape[0] for trial in stimulus)
    assert result.regularization > 0.0
    prediction, score = model.predict(stimulus=stimulus, response=response)
    assert isinstance(prediction, list)
    assert score > 0.8


def test_bayesian_frequency_trf_matches_time_domain_ridge_for_fixed_lambda() -> None:
    rng = np.random.default_rng(32)
    fs = 1_000
    n_samples = 2_048
    tmin = 0.0
    tmax = 0.040
    regularization = 1_000.0
    kernel = np.zeros(int(round((tmax - tmin) * fs)))
    kernel[3] = 1.0
    kernel[9] = -0.4
    kernel[18] = 0.2

    stimulus = rng.standard_normal(n_samples)
    response = np.convolve(stimulus, kernel, mode="full")[:n_samples]
    response += 0.1 * rng.standard_normal(n_samples)

    lags = np.arange(int(round(tmin * fs)), int(round(tmax * fs)), dtype=int)
    design = _lagged_design_matrix(stimulus, lags)
    reference = np.linalg.solve(
        design.T @ design + regularization * np.eye(len(lags)),
        design.T @ response,
    )

    model = BayesianFrequencyTRF(direction=1)
    trained = model.train(
        stimulus=stimulus,
        response=response[:, np.newaxis],
        fs=fs,
        tmin=tmin,
        tmax=tmax,
        regularization=regularization,
        prior="ridge",
        window=None,
    )

    assert trained is None
    assert model.fit_mode == "fixed_regularization"
    assert np.isclose(model.regularization, regularization)
    recovered = model.weights[0, :, 0]
    assert np.corrcoef(reference, recovered)[0, 1] > 0.99
    assert 0.7 < (np.linalg.norm(recovered) / np.linalg.norm(reference)) < 1.3

    wrapped = fit_bayesian_frequency_trf(
        stimulus,
        response[:, np.newaxis],
        fs=fs,
        tmin=tmin,
        tmax=tmax,
        regularization=regularization,
        prior="ridge",
        window=None,
    )
    assert wrapped.fit_mode == "fixed_regularization"
    assert np.isclose(wrapped.regularization, regularization)


def test_bayesian_frequency_trf_cross_validates_regularization() -> None:
    rng = np.random.default_rng(34)
    fs = 1_000
    kernel = np.zeros(30)
    kernel[3] = 0.9
    kernel[8] = -0.35
    kernel[15] = 0.12

    stimulus, response = _simulate(
        rng=rng,
        n_trials=10,
        n_samples=3_072,
        kernel=kernel,
        lag_start=0,
        noise_scale=0.05,
    )

    model = BayesianFrequencyTRF(direction=1)
    scores = model.train(
        stimulus=stimulus[:8],
        response=response[:8],
        fs=fs,
        tmin=0.0,
        tmax=0.030,
        regularization=np.logspace(-4, 2, 7),
        prior="ridge",
        k=4,
        trial_weights="inverse_variance",
    )

    assert scores.shape == (7,)
    assert model.fit_mode == "fixed_regularization"
    assert isinstance(model.regularization, float)
    _, held_out_score = model.predict(stimulus=stimulus[8:], response=response[8:])
    assert held_out_score > 0.8


def test_bayesian_frequency_trf_decay_ridge_suppresses_late_lags() -> None:
    rng = np.random.default_rng(35)
    fs = 1_000
    tmin = 0.0
    tmax = 0.080
    kernel = np.zeros(int(round((tmax - tmin) * fs)))
    kernel[3] = 1.0
    kernel[9] = -0.35
    kernel[16] = 0.12

    stimulus, response = _simulate(
        rng=rng,
        n_trials=8,
        n_samples=4_096,
        kernel=kernel,
        lag_start=0,
        noise_scale=0.08,
    )

    ridge = BayesianFrequencyTRF(direction=1)
    ridge.train(
        stimulus=stimulus,
        response=response,
        fs=fs,
        tmin=tmin,
        tmax=tmax,
        prior="ridge",
    )

    decay = BayesianFrequencyTRF(direction=1)
    decay.train(
        stimulus=stimulus,
        response=response,
        fs=fs,
        tmin=tmin,
        tmax=tmax,
        prior="decay_ridge",
        decay_tau=0.010,
    )

    late_mask = decay.times >= 0.040
    ridge_late = np.linalg.norm(ridge.weights[0, late_mask, 0])
    decay_late = np.linalg.norm(decay.weights[0, late_mask, 0])

    assert decay.decay_tau == 0.010
    assert decay_late < ridge_late
    assert np.corrcoef(decay.weights[0, :, 0], kernel)[0, 1] > 0.9


def test_bayesian_frequency_trf_ard_downweights_irrelevant_features() -> None:
    rng = np.random.default_rng(36)
    fs = 1_000
    tmin = 0.0
    tmax = 0.030
    n_samples = 3_072
    kernel = np.zeros(int(round((tmax - tmin) * fs)))
    kernel[2] = 0.8
    kernel[7] = -0.25
    kernel[12] = 0.1

    stimulus = []
    response = []
    for _ in range(8):
        x = rng.standard_normal((n_samples, 2))
        y = np.convolve(x[:, 0], kernel, mode="full")[:n_samples]
        y += 0.05 * rng.standard_normal(n_samples)
        stimulus.append(x)
        response.append(y[:, np.newaxis])

    model = BayesianFrequencyTRF(direction=1)
    model.train(
        stimulus=stimulus,
        response=response,
        fs=fs,
        tmin=tmin,
        tmax=tmax,
        prior="ard",
    )

    feature_norms = np.linalg.norm(model.weights[:, :, 0], axis=1)
    feature_regularization = np.asarray(model.regularization, dtype=float)
    feature_alpha = np.asarray(model.alpha, dtype=float)

    assert feature_norms[0] > 5.0 * feature_norms[1]
    assert feature_regularization[1] > feature_regularization[0]
    assert feature_alpha.shape == (2,)
    _, score = model.predict(stimulus=stimulus, response=response)
    assert score > 0.8


def test_bayesian_frequency_trf_ard_requires_evidence_mode() -> None:
    rng = np.random.default_rng(37)
    stimulus = rng.standard_normal((1_024, 2))
    response = rng.standard_normal((1_024, 1))

    model = BayesianFrequencyTRF(direction=1)
    with pytest.raises(ValueError, match="evidence-only"):
        model.train(
            stimulus=stimulus,
            response=response,
            fs=1_000,
            tmin=0.0,
            tmax=0.020,
            regularization=1e-3,
            prior="ard",
        )


def test_bayesian_frequency_trf_backward_api() -> None:
    rng = np.random.default_rng(33)
    fs = 500
    tmin = 0.0
    tmax = 0.020
    lag_start = int(round(tmin * fs))
    kernel = np.zeros(int(round((tmax - tmin) * fs)))
    kernel[2] = 0.7
    kernel[6] = -0.15

    response, stimulus = _simulate(
        rng=rng,
        n_trials=5,
        n_samples=1_536,
        kernel=kernel,
        lag_start=lag_start,
        noise_scale=0.05,
    )

    model = BayesianFrequencyTRF(direction=-1)
    model.train(
        response=response,
        stimulus=stimulus,
        fs=fs,
        tmin=tmin,
        tmax=tmax,
        prior="smooth",
    )

    prediction, score = model.predict(response=response, stimulus=stimulus)
    assert isinstance(prediction, list)
    assert score > 0.75

    result = model.result()
    wrapped_prediction, wrapped_score = predict_bayesian_frequency_trf(
        stimulus,
        result,
        response=response,
        average=True,
    )
    assert isinstance(wrapped_prediction, list)
    assert wrapped_score > 0.75


def test_bayesian_frequency_trf_plot_methods_if_matplotlib_available() -> None:
    plt = pytest.importorskip("matplotlib.pyplot")

    rng = np.random.default_rng(38)
    fs = 1_000
    kernel = np.zeros(20)
    kernel[2] = 0.8
    kernel[6] = -0.2

    stimulus, response = _simulate(
        rng=rng,
        n_trials=4,
        n_samples=1_024,
        kernel=kernel,
        lag_start=0,
        noise_scale=0.03,
    )

    model = BayesianFrequencyTRF(direction=1)
    model.train(
        stimulus=stimulus,
        response=response,
        fs=fs,
        tmin=0.0,
        tmax=0.020,
        prior="smooth",
    )

    fig_model, ax_model = model.plot(label="posterior")
    fig_result, ax_result = model.result().plot(label="result")

    assert ax_model.get_xlabel() == "Lag (ms)"
    assert ax_result.get_ylabel() == "Weight"

    plt.close(fig_model)
    plt.close(fig_result)


def test_bayesian_frequency_trf_supports_custom_credible_levels() -> None:
    rng = np.random.default_rng(39)
    fs = 1_000
    tmin = 0.0
    tmax = 0.020
    kernel = np.zeros(int(round((tmax - tmin) * fs)))
    kernel[2] = 0.8
    kernel[6] = -0.2

    stimulus, response = _simulate(
        rng=rng,
        n_trials=6,
        n_samples=2_048,
        kernel=kernel,
        lag_start=0,
        noise_scale=0.04,
    )

    model = BayesianFrequencyTRF(direction=1)
    model.train(
        stimulus=stimulus,
        response=response,
        fs=fs,
        tmin=tmin,
        tmax=tmax,
        prior="smooth",
        credible_level=0.8,
    )

    assert np.isclose(model.credible_level, 0.8)
    assert np.isclose(model.result().credible_level, 0.8)

    default_width = model.credible_interval[1] - model.credible_interval[0]
    interval_95, times_95 = model.credible_interval_at(0.95)
    interval_80_slice, times_80_slice = model.credible_interval_at(0.8, tmin=0.0, tmax=0.010)

    assert np.allclose(model.result().credible_interval, model.credible_interval)
    assert np.mean(interval_95[1] - interval_95[0]) > np.mean(default_width)
    assert interval_80_slice.shape[2] == times_80_slice.size
    assert times_80_slice[0] >= 0.0
    assert times_80_slice[-1] < 0.010


def test_bayesian_frequency_trf_rejects_invalid_credible_levels() -> None:
    rng = np.random.default_rng(40)
    stimulus = rng.standard_normal(1_024)
    response = rng.standard_normal((1_024, 1))

    model = BayesianFrequencyTRF(direction=1)
    with pytest.raises(ValueError, match="credible_level"):
        model.train(
            stimulus=stimulus,
            response=response,
            fs=1_000,
            tmin=0.0,
            tmax=0.020,
            prior="ridge",
            credible_level=1.0,
        )
