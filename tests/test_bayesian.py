from __future__ import annotations

import numpy as np

from fft_trf.experimental import (
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
