from __future__ import annotations

from pathlib import Path

import fftrf
import fftrf.experimental as experimental_api
import numpy as np
import pytest

from fftrf import BayesianTRF, BayesianTRFResult, PermutationTestResult, TRF


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


def _fit_forward_model(
    *,
    seed: int = 200,
    n_trials: int = 6,
    n_samples: int = 2_048,
    regularization: float = 1e-3,
    bootstrap_samples: int = 0,
) -> tuple[list[np.ndarray], list[np.ndarray], TRF]:
    rng = np.random.default_rng(seed)
    fs = 1_000
    kernel = np.zeros(30)
    kernel[3] = 0.9
    kernel[8] = -0.35
    kernel[15] = 0.12

    stimulus, response = _simulate_trials(
        rng=rng,
        n_trials=n_trials,
        n_samples=n_samples,
        kernel=kernel,
        noise_scale=0.04,
    )

    model = TRF(direction=1)
    model.train(
        stimulus=stimulus,
        response=response,
        fs=fs,
        tmin=0.0,
        tmax=0.030,
        regularization=regularization,
        segment_length=512,
        overlap=0.5,
        window="hann",
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=0,
    )
    return stimulus, response, model


def _fit_cross_validated_forward_model() -> tuple[list[np.ndarray], list[np.ndarray], TRF]:
    rng = np.random.default_rng(240)
    fs = 1_000
    kernel = np.zeros(24)
    kernel[2] = 0.9
    kernel[7] = -0.25
    kernel[13] = 0.10

    stimulus, response = _simulate_trials(
        rng=rng,
        n_trials=7,
        n_samples=1_024,
        kernel=kernel,
        noise_scale=0.05,
    )

    model = TRF(direction=1)
    model.train(
        stimulus=stimulus[:-2],
        response=response[:-2],
        fs=fs,
        tmin=0.0,
        tmax=0.024,
        regularization=np.logspace(-5, -1, 5),
        segment_length=256,
        overlap=0.5,
        window="hann",
        k=3,
        seed=2,
    )
    return stimulus, response, model


def test_score_requires_observed_targets() -> None:
    stimulus, _, model = _fit_forward_model()

    with pytest.raises(ValueError, match="response is required for score"):
        model.score(stimulus=stimulus[0])


def test_transfer_function_at_returns_copy_and_validates_indices() -> None:
    _, _, model = _fit_forward_model()

    frequencies, transfer = model.transfer_function_at()
    assert np.allclose(frequencies, model.frequencies)
    assert np.allclose(transfer, model.transfer_function[:, 0, 0])

    original_value = model.transfer_function[0, 0, 0]
    transfer[0] = original_value + (1.0 + 0.0j)
    assert model.transfer_function[0, 0, 0] == original_value

    with pytest.raises(IndexError, match="input_index out of bounds"):
        model.transfer_function_at(input_index=1)
    with pytest.raises(IndexError, match="output_index out of bounds"):
        model.transfer_function_at(output_index=1)


def test_diagnostics_alias_matches_cross_spectral_diagnostics() -> None:
    stimulus, response, model = _fit_forward_model()

    diagnostics = model.cross_spectral_diagnostics(
        stimulus=stimulus[-1],
        response=response[-1],
    )
    alias = model.diagnostics(
        stimulus=stimulus[-1],
        response=response[-1],
    )

    assert np.allclose(alias.frequencies, diagnostics.frequencies)
    assert np.allclose(alias.transfer_function, diagnostics.transfer_function)
    assert np.allclose(alias.predicted_spectrum, diagnostics.predicted_spectrum)
    assert np.allclose(alias.observed_spectrum, diagnostics.observed_spectrum)
    assert np.allclose(alias.cross_spectrum, diagnostics.cross_spectrum)
    assert np.allclose(alias.coherence, diagnostics.coherence)


def test_bootstrap_confidence_interval_post_fit_stores_interval() -> None:
    stimulus, response, model = _fit_forward_model()

    interval, times = model.bootstrap_confidence_interval(
        stimulus=stimulus,
        response=response,
        n_bootstraps=12,
        level=0.9,
        seed=3,
        n_jobs=2,
    )
    stored_interval, stored_times = model.bootstrap_interval_at()

    assert interval.shape == (2, *model.weights.shape)
    assert np.all(interval[0] <= interval[1])
    assert np.allclose(stored_interval, interval, rtol=1e-10, atol=1e-12)
    assert np.allclose(stored_times, times, rtol=1e-10, atol=1e-12)
    assert model.bootstrap_level == pytest.approx(0.9)
    assert model.bootstrap_samples == 12


def test_save_load_and_copy_roundtrip(tmp_path: Path) -> None:
    stimulus, response, model = _fit_forward_model(bootstrap_samples=10)
    path = tmp_path / "model.pkl"

    prediction, score = model.predict(
        stimulus=stimulus[-1],
        response=response[-1],
    )
    model.save(path)

    restored = TRF(direction=1)
    restored.load(path)
    restored_prediction, restored_score = restored.predict(
        stimulus=stimulus[-1],
        response=response[-1],
    )

    assert np.allclose(restored.weights, model.weights, rtol=1e-10, atol=1e-12)
    assert np.allclose(
        restored.transfer_function,
        model.transfer_function,
        rtol=1e-10,
        atol=1e-12,
    )
    assert np.allclose(
        restored.bootstrap_interval,
        model.bootstrap_interval,
        rtol=1e-10,
        atol=1e-12,
    )
    assert restored.regularization == model.regularization
    assert restored.segment_duration == pytest.approx(model.segment_duration)
    assert restored.metric_name == model.metric_name
    assert np.allclose(restored_prediction, prediction, rtol=1e-10, atol=1e-12)
    assert restored_score == pytest.approx(score)

    copied = model.copy()
    assert np.allclose(copied.weights, model.weights, rtol=1e-10, atol=1e-12)
    copied.weights[0, 0, 0] += 1.0
    copied.bootstrap_interval[0, 0, 0, 0] += 1.0
    assert not np.allclose(copied.weights, model.weights)
    assert not np.allclose(copied.bootstrap_interval, model.bootstrap_interval)


def test_permutation_test_circular_shift_returns_significant_score() -> None:
    stimulus, response, model = _fit_forward_model()

    result = model.permutation_test(
        stimulus=stimulus[0],
        response=response[0],
        n_permutations=63,
        average=False,
        surrogate="circular_shift",
        min_shift=0.2,
        seed=4,
        n_jobs=2,
    )

    assert isinstance(result, PermutationTestResult)
    assert result.surrogate == "circular_shift"
    assert result.tail == "greater"
    assert result.null_scores.shape == (63, 1)
    assert result.observed_score.shape == (1,)
    assert result.p_value.shape == (1,)
    assert float(result.p_value[0]) < 0.05
    assert float(result.observed_score[0]) > float(np.mean(result.null_scores[:, 0]))


def test_permutation_test_trial_shuffle_returns_significant_score() -> None:
    stimulus, response, model = _fit_forward_model(n_trials=7)

    result = model.permutation_test(
        stimulus=stimulus,
        response=response,
        n_permutations=63,
        surrogate="trial_shuffle",
        seed=5,
        n_jobs=2,
    )

    assert isinstance(result, PermutationTestResult)
    assert result.surrogate == "trial_shuffle"
    assert result.null_scores.shape == (63,)
    assert result.p_value < 0.05
    assert result.observed_score > float(np.mean(result.null_scores))


def test_permutation_test_validates_trial_shuffle_for_variable_length_trials() -> None:
    stimulus, response, model = _fit_forward_model()
    stimulus_variable = stimulus[:-1]
    response_variable = response[:-1]
    stimulus_variable[0] = stimulus_variable[0][:-16]
    response_variable[0] = response_variable[0][:-16]

    with pytest.raises(ValueError, match="trial_shuffle requires all evaluation trials to have the same sample count"):
        model.permutation_test(
            stimulus=stimulus_variable,
            response=response_variable,
            surrogate="trial_shuffle",
            n_permutations=8,
            seed=1,
        )


def test_permutation_test_validates_circular_shift_bounds() -> None:
    stimulus, response, model = _fit_forward_model(n_samples=256)

    with pytest.raises(ValueError, match="min_shift is too large"):
        model.permutation_test(
            stimulus=stimulus[0],
            response=response[0],
            surrogate="circular_shift",
            min_shift=0.2,
            n_permutations=8,
            seed=1,
        )


def test_refit_permutation_test_circular_shift_returns_significant_score() -> None:
    stimulus, response, model = _fit_forward_model(n_trials=8, n_samples=1_024)

    result = model.refit_permutation_test(
        train_stimulus=stimulus[:-2],
        train_response=response[:-2],
        test_stimulus=stimulus[-2:],
        test_response=response[-2:],
        n_permutations=31,
        surrogate="circular_shift",
        min_shift=0.1,
        seed=8,
        n_jobs=2,
    )

    assert isinstance(result, PermutationTestResult)
    assert result.surrogate == "circular_shift"
    assert result.p_value < 0.05
    assert result.observed_score > float(np.mean(result.null_scores))


def test_refit_permutation_test_supports_stored_cv_configuration() -> None:
    stimulus, response, model = _fit_cross_validated_forward_model()

    result = model.refit_permutation_test(
        train_stimulus=stimulus[:-2],
        train_response=response[:-2],
        test_stimulus=stimulus[-2:],
        test_response=response[-2:],
        n_permutations=7,
        surrogate="circular_shift",
        min_shift=0.1,
        seed=9,
    )

    assert isinstance(result, PermutationTestResult)
    assert result.null_scores.shape == (7,)
    assert np.isfinite(result.observed_score)


def test_refit_permutation_test_requires_fit_configuration() -> None:
    stimulus, response, _ = _fit_forward_model()
    fresh = TRF(direction=1)

    with pytest.raises(ValueError, match="stored fit configuration or fit_kwargs"):
        fresh.refit_permutation_test(
            train_stimulus=stimulus[:-1],
            train_response=response[:-1],
            test_stimulus=stimulus[-1],
            test_response=response[-1],
            n_permutations=4,
            seed=0,
        )


def test_top_level_api_exports_permutation_result() -> None:
    assert fftrf.PermutationTestResult is PermutationTestResult


def test_top_level_api_exports_bayesian_objects() -> None:
    assert fftrf.BayesianTRF is BayesianTRF
    assert fftrf.BayesianTRFResult is BayesianTRFResult
    assert experimental_api.BayesianTRF is BayesianTRF
