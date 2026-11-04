"""Regression coverage for numerical methods and estimator workflows."""

from __future__ import annotations

import pickle
from itertools import permutations

import numpy as np
import pytest

from fftrf import TRF, explained_variance_score, inverse_variance_weights, pearsonr, r2_score
from fftrf.prediction import (
    _build_permutation_plan,
    _compute_permutation_test_scores,
    _permutation_p_value_and_z_score,
)
from fftrf.spectral import _aggregate_cached_spectra, _build_spectral_cache, _matched_spectra


def _trials(seed: int, count: int = 4, inputs: int = 1) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.normal(size=(64, inputs)) for _ in range(count)]


@pytest.mark.parametrize("scale", [1e-150, 1e-13, 1e-9, 1.0, 1e100])
def test_dimensionless_metrics_preserve_signal_units(scale: float) -> None:
    truth = np.array([-1.0, 0.0, 1.0]) * scale
    np.testing.assert_allclose(pearsonr(truth, truth), [1.0], atol=1e-14)
    np.testing.assert_allclose(r2_score(truth, -truth), [-3.0], atol=1e-14)
    np.testing.assert_allclose(explained_variance_score(truth, -truth), [-3.0], atol=1e-14)
    assert pearsonr(np.ones(3) * scale, truth)[0] == 0.0
    assert r2_score(np.ones(3) * scale, np.ones(3) * scale)[0] == 1.0
    assert r2_score(np.ones(3) * scale, np.zeros(3))[0] == 0.0


@pytest.mark.parametrize("scale", [1e-100, 1e-9, 1.0, 1e100])
def test_inverse_variance_weights_preserve_units(scale: float) -> None:
    data = [np.array([-1.0, 1.0]) * scale, np.array([-2.0, 2.0]) * scale]
    np.testing.assert_allclose(inverse_variance_weights(data), [0.8, 0.2], atol=1e-14)


def test_inverse_variance_constant_and_invalid_trials() -> None:
    np.testing.assert_array_equal(inverse_variance_weights([np.ones(4), np.arange(4.0)]), [1, 0])
    np.testing.assert_allclose(inverse_variance_weights([np.zeros(4), np.ones(4)]), [0.5, 0.5])
    for data in ([], [np.zeros(0)], [np.array([0.0, np.nan])], [np.zeros((2, 2, 2))]):
        with pytest.raises(ValueError):
            inverse_variance_weights(data)


@pytest.mark.parametrize("direction", [1, -1])
def test_small_banded_penalties_match_independent_normal_equations(direction: int) -> None:
    predictors = [trial * 1e-6 for trial in _trials(803, inputs=2)]
    targets = [trial.sum(axis=1, keepdims=True) for trial in predictors]
    stimulus, response = (predictors, targets) if direction == 1 else (targets, predictors)
    model = TRF(direction=direction)
    model.train(stimulus, response, fs=100, tmin=0, tmax=0.03,
                regularization=[[1e-12, 1e-9]], bands=[1, 1], detrend=None)
    x_fft = np.stack([np.fft.rfft(trial, axis=0) for trial in predictors])
    y_fft = np.stack([np.fft.rfft(trial, axis=0) for trial in targets])
    cxx = np.einsum("tfi,tfj->fij", x_fft.conj(), x_fft) / len(predictors)
    cxy = np.einsum("tfi,tfj->fij", x_fft.conj(), y_fft) / len(predictors)
    reference = np.linalg.solve(cxx + np.diag([1e-12, 1e-9]), cxy)
    np.testing.assert_allclose(model.transfer_function, reference, rtol=1e-11, atol=1e-12)


@pytest.mark.parametrize("direction", [1, -1])
@pytest.mark.parametrize("during_fit", [True, False])
def test_fractional_bootstrap_lags_and_plotting(direction: int, during_fit: bool) -> None:
    import matplotlib.pyplot as plt

    data = _trials(805)
    model = TRF(direction=direction)
    model.train(data, data, fs=100, tmin=0.004, tmax=0.026, regularization=1.0,
                bootstrap_samples=4 if during_fit else 0, bootstrap_seed=0)
    if not during_fit:
        model.bootstrap_confidence_interval(data, data, n_bootstraps=4, seed=0)
    interval, times = model.bootstrap_interval_at()
    assert interval.shape == (2, *model.weights.shape)
    np.testing.assert_array_equal(times, model.times)
    tmin = 0.004 if direction == 1 else -0.016
    tmax = 0.026 if direction == 1 else 0.006
    selected, selected_times = model.bootstrap_interval_at(tmin=tmin, tmax=tmax)
    weights, weight_times = model.to_impulse_response(tmin=tmin, tmax=tmax)
    assert selected.shape == (2, *weights.shape)
    np.testing.assert_allclose(selected_times, weight_times, atol=1e-15)
    fig, _ = model.plot(tmin=tmin, tmax=tmax, show_bootstrap_interval=True)
    plt.close(fig)


@pytest.mark.parametrize("direction", [1, -1])
def test_bootstrap_rejects_incompatible_fitted_dimensions(direction: int) -> None:
    data = _trials(806)
    model = TRF(direction=direction)
    model.train(data, data, fs=100, tmin=0, tmax=0.03, regularization=1.0,
                bootstrap_samples=3, bootstrap_seed=0)
    before = model.bootstrap_interval.copy()
    extra_outputs = [np.repeat(trial, 3, axis=1) for trial in data]
    stimulus, response = (data, extra_outputs) if direction == 1 else (extra_outputs, data)
    with pytest.raises(ValueError, match="target count"):
        model.bootstrap_confidence_interval(stimulus, response, n_bootstraps=3)
    np.testing.assert_array_equal(model.bootstrap_interval, before)
    stimulus, response = (extra_outputs, data) if direction == 1 else (data, extra_outputs)
    with pytest.raises(ValueError, match="predictor count"):
        model.bootstrap_confidence_interval(stimulus, response, n_bootstraps=3)


def test_zero_weight_trials_are_excluded_from_bootstrap_population() -> None:
    data = _trials(807, count=3)
    intervals = []
    for n_jobs in (1, 2):
        model = TRF()
        model.train(data, data, fs=100, tmin=0, tmax=0.03, regularization=1.0,
                    trial_weights=[1, 2, 0])
        intervals.append(model.bootstrap_confidence_interval(data, data, n_bootstraps=20,
                                                            seed=0, n_jobs=n_jobs)[0])
    reference = TRF()
    reference.train(data[:2], data[:2], fs=100, tmin=0, tmax=0.03,
                    regularization=1.0, trial_weights=[1, 2])
    expected, _ = reference.bootstrap_confidence_interval(data[:2], data[:2],
                                                         n_bootstraps=20, seed=0)
    for interval in intervals:
        np.testing.assert_allclose(interval, expected, rtol=1e-12, atol=1e-14)
    with pytest.raises(ValueError, match="two trials with positive weight"):
        reference.bootstrap_confidence_interval(data[:2], data[:2], n_bootstraps=4,
                                                trial_weights=[1, 0])


@pytest.mark.parametrize("average", [True, False, [0]])
def test_uneven_cv_folds_average_trials_equally(average) -> None:
    predictors = [np.zeros((32, 1)) for _ in range(5)]
    targets = [np.full((32, 1), value) for value in [0, 0, 0, 0, np.sqrt(10.0)]]
    for n_jobs in (1, 2):
        result = TRF(metric="neg_mse").train(predictors, targets, fs=100, tmin=0, tmax=0.02,
                                             regularization=[1.0, 10.0], k=2,
                                             average=average, n_jobs=n_jobs)
        np.testing.assert_allclose(result, -2.0, atol=1e-14)


@pytest.mark.parametrize("direction", [1, -1])
def test_failed_retraining_preserves_complete_state(direction: int) -> None:
    data = _trials(809)
    model = TRF(direction=direction)
    model.train(data, data, fs=100, tmin=0, tmax=0.03, regularization=1.0,
                bootstrap_samples=3, bootstrap_seed=0)
    before = pickle.dumps(model)
    failures = [
        {"tmax": 3.0},
        {"window": "not-a-window"},
        {"k": 1},
        {"trial_weights": [1, 0, 0, 0], "bootstrap_samples": 4},
    ]
    for overrides in failures:
        kwargs = dict(fs=100, tmin=0, tmax=0.03, regularization=5.0)
        kwargs.update(overrides)
        with pytest.raises(ValueError):
            model.train(data, data, **kwargs)
        assert pickle.dumps(model) == before


@pytest.mark.parametrize("single", [True, False])
@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_nonfinite_training_and_scoring_are_rejected(single: bool, invalid: float) -> None:
    data = _trials(810)
    model = TRF()
    model.train(data, data, fs=100, tmin=0, tmax=0.03, regularization=1.0)
    bad = data[0].copy()
    bad[2] = invalid
    provided = bad if single else [bad, *data[1:]]
    with pytest.raises(ValueError, match="finite"):
        model.train(provided, provided, fs=100, tmin=0, tmax=0.03, regularization=1.0)
    with pytest.raises(ValueError, match="finite"):
        model.predict(provided)
    with pytest.raises(ValueError, match="finite"):
        pearsonr(bad, data[0])


@pytest.mark.parametrize("field", ["fs", "tmin", "tmax"])
@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_nonfinite_fit_settings_are_rejected(field: str, invalid: float) -> None:
    data = _trials(810)
    kwargs = dict(fs=100, tmin=0, tmax=0.03, regularization=1.0)
    kwargs[field] = invalid
    with pytest.raises(ValueError, match="finite"):
        TRF().train(data, data, **kwargs)


@pytest.mark.parametrize("data", [np.array([]), np.zeros((3, 0)), [np.array([])]])
def test_empty_trial_axes_are_rejected(data) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        TRF().train(data, data, fs=100, tmin=0, tmax=0.03, regularization=1.0)


@pytest.mark.parametrize("direction", [1, -1])
def test_frequency_views_share_lag_representability_checks(direction: int) -> None:
    data = _trials(811)
    model = TRF(direction=direction)
    model.train(data, data, fs=100, tmin=0, tmax=0.03, regularization=1.0)
    for method in (model.to_impulse_response, model.frequency_resolved_weights,
                   model.time_frequency_power):
        with pytest.raises(ValueError, match="longer than n_fft"):
            method(tmin=0, tmax=2.0)


@pytest.mark.parametrize("bandwidth", [1e-300, 0.001, 1.0])
def test_narrow_frequency_filters_still_partition_the_transfer(bandwidth: float) -> None:
    data = _trials(812)
    model = TRF()
    model.train(data, data, fs=100, tmin=0, tmax=0.03, regularization=1.0)
    resolved = model.frequency_resolved_weights(n_bands=4, bandwidth=bandwidth)
    np.testing.assert_allclose(resolved.filters.sum(axis=1), 1.0, atol=1e-14)
    np.testing.assert_allclose(resolved.weights.sum(axis=1), model.weights, atol=1e-14)


def test_nested_tuple_configuration_is_copied_independently() -> None:
    data = _trials(815, inputs=2)
    targets = [trial[:, :1] for trial in data]
    grid = ([1.0, 10.0], np.array([10.0, 1.0]))
    model = TRF()
    model.train(data, targets, fs=100, tmin=0, tmax=0.03,
                regularization=grid, bands=[1, 1])
    copied = model.copy()
    grid[0][0] = 100.0
    copied._fit_config["regularization"][1][0] = 999.0
    assert model._fit_config["regularization"][0][0] == 1.0
    assert model._fit_config["regularization"][1][0] == 10.0


@pytest.mark.parametrize("direction", [1, -1])
def test_coherence_preserves_evaluation_units(direction: int) -> None:
    data = _trials(802)
    model = TRF(direction=direction)
    model.train(data, data, fs=100, tmin=0, tmax=0.01, regularization=1e-10)
    for scale in [1.0, 1e-6, 1e-13, 1e-150]:
        scaled = [trial * scale for trial in data]
        diagnostic = model.diagnostics(stimulus=scaled, response=scaled)
        np.testing.assert_allclose(diagnostic.coherence[1:], 1.0, atol=1e-12)


@pytest.mark.parametrize("method,window", [("standard", None), ("standard", "hann"),
                                         ("multitaper", None)])
def test_matched_diagnostics_agree_with_full_matrix_reference(method: str, window) -> None:
    predicted = _trials(814, count=3, inputs=4)
    targets = _trials(8140, count=3, inputs=4)
    weights = np.array([1.0, 2.0, 0.5])
    settings = dict(segment_length=24, n_fft=32, overlap=0.5, spectral_method=method,
                    time_bandwidth=2.5, n_tapers=3, window=window, detrend="constant")
    pred, obs, cross, count = _matched_spectra(predicted, targets,
                                             raw_trial_weights=weights, **settings)
    pred_cache = _build_spectral_cache(predicted, targets, **settings)
    obs_cache = _build_spectral_cache(targets, targets, **settings)
    cxx, cxy = _aggregate_cached_spectra(pred_cache, raw_trial_weights=weights)
    cyy, _ = _aggregate_cached_spectra(obs_cache, raw_trial_weights=weights)
    for actual, expected in [(pred, cxx), (obs, cyy), (cross, cxy)]:
        np.testing.assert_allclose(actual, np.diagonal(expected, axis1=1, axis2=2),
                                   rtol=1e-12, atol=1e-12)
    assert count > 1


def test_matched_spectra_use_linear_output_storage_and_one_pair_of_ffts(monkeypatch) -> None:
    shapes = []
    fft_calls = []
    original_zeros = np.zeros
    original_rfft = np.fft.rfft

    def zeros(shape, *args, **kwargs):
        shapes.append(shape)
        return original_zeros(shape, *args, **kwargs)

    def rfft(*args, **kwargs):
        fft_calls.append(1)
        return original_rfft(*args, **kwargs)

    sizes = []
    for outputs in [4, 8]:
        data = _trials(814, count=3, inputs=outputs)
        shapes.clear()
        fft_calls.clear()
        with monkeypatch.context() as context:
            context.setattr(np, "zeros", zeros)
            context.setattr(np.fft, "rfft", rfft)
            result = _matched_spectra(data, data, raw_trial_weights=np.ones(3),
                                      segment_length=64, n_fft=64, overlap=0,
                                      spectral_method="standard", time_bandwidth=3.5,
                                      n_tapers=None, window=None, detrend="constant")
        assert all(len(shape) == 2 for shape in shapes)
        assert len(fft_calls) == 6
        sizes.append(sum(array.nbytes for array in result[:3]))
    assert sizes[1] == 2 * sizes[0]


def test_single_realization_coherence_is_unavailable_but_spectra_remain_valid() -> None:
    data = _trials(816)
    model = TRF()
    model.train(data, data, fs=100, tmin=0, tmax=0.02, regularization=1.0)
    with pytest.warns(RuntimeWarning, match="two positive-weight spectral realizations"):
        diagnostic = model.diagnostics(stimulus=data[0], response=data[1])
    assert np.isnan(diagnostic.coherence).all()
    assert np.isfinite(diagnostic.predicted_spectrum).all()
    assert np.isfinite(diagnostic.observed_spectrum).all()
    with pytest.warns(RuntimeWarning, match="two positive-weight"):
        diagnostic = model.diagnostics(stimulus=data[:2], response=data[:2], trial_weights=[1, 0])
    assert np.isnan(diagnostic.coherence).all()
    diagnostic = model.diagnostics(stimulus=data[:2], response=data[:2])
    assert np.isfinite(diagnostic.coherence).all()


def test_exact_permutation_null_matches_enumeration_and_is_calibrated() -> None:
    train = _trials(804)
    predictors = _trials(8040)
    targets = _trials(8041)
    model = TRF()
    model.train(train, train, fs=100, tmin=0, tmax=0.01, regularization=1.0)
    orders = list(permutations(range(4)))
    reference = np.asarray([model.score(predictors, [targets[i] for i in order]) for order in orders])
    p_values = []
    for order in orders:
        result = model.permutation_test(predictors, [targets[i] for i in order],
                                        surrogate="trial_shuffle", n_permutations=99, seed=0)
        assert result.exact and result.n_permutations == 24
        np.testing.assert_allclose(np.sort(result.null_scores), np.sort(reference), atol=1e-14)
        expected_p = np.mean(reference >= result.observed_score - 1e-14)
        assert result.p_value == pytest.approx(expected_p)
        p_values.append(result.p_value)
    assert np.mean(np.asarray(p_values) <= 0.05) <= 0.05


@pytest.mark.parametrize("surrogate", ["trial_shuffle", "circular_shift"])
def test_random_permutations_include_identity_and_reproduce_across_workers(surrogate: str) -> None:
    data = _trials(8042, count=5)
    plan = _build_permutation_plan(target_trials=data, fs=100, surrogate=surrogate,
                                    min_shift=None, n_permutations=16, seed=0)
    assert not plan.exact
    model = TRF()
    model.train(data, data, fs=100, tmin=0, tmax=0.01, regularization=1.0)
    results = [model.permutation_test(data, data, surrogate=surrogate,
                                      n_permutations=16, seed=0, n_jobs=n_jobs)
               for n_jobs in (1, 2)]
    assert not results[0].exact
    np.testing.assert_array_equal(results[0].null_scores, results[1].null_scores)
    # A Monte Carlo sample from a larger finite space must allow the identity.
    observed_identity = False
    for seed in range(50):
        small_data = data[:3] if surrogate == "trial_shuffle" else [np.arange(4.0)[:, None]]
        small_plan = _build_permutation_plan(target_trials=small_data, fs=1, surrogate=surrogate,
                                             min_shift=None, n_permutations=2, seed=seed)
        identity = np.arange(3) if surrogate == "trial_shuffle" else np.zeros(1)
        observed_identity |= any(np.array_equal(spec, identity) for spec in small_plan.specs)
    assert observed_identity


def test_short_circular_shift_space_has_honest_resolution() -> None:
    data = [np.array([[-1.0], [1.0]])]
    observed, null, p_value, _, exact = _compute_permutation_test_scores(
        prediction_trials=data, target_trials=data, metric=pearsonr, average=True,
        fs=1, n_permutations=999, surrogate="circular_shift", min_shift=None,
        tail="greater", seed=0,
    )
    assert observed == pytest.approx(1.0) and exact and p_value == 0.5
    np.testing.assert_allclose(null, [1.0, -1.0], atol=1e-14)


def test_permutation_ties_are_scale_aware_and_invalid_statistics_rejected() -> None:
    for scale in [1.0, 1e-20]:
        p_value, _ = _permutation_p_value_and_z_score(
            observed_score=scale, null_scores=np.array([scale * (1 - 1e-15), 0.0]),
            tail="greater", exact=True,
        )
        assert p_value == 0.5
    with pytest.raises(ValueError, match="finite"):
        _permutation_p_value_and_z_score(observed_score=np.nan, null_scores=np.ones(3), tail="greater")


@pytest.mark.parametrize("result", [np.array([np.nan]), np.ones(2), 1.0])
def test_custom_metric_contract_is_validated(result) -> None:
    data = _trials(8101)
    model = TRF(metric=lambda true, prediction: result)
    model.train(data, data, fs=100, tmin=0, tmax=0.01, regularization=1.0)
    with pytest.raises(ValueError, match="one finite score per output"):
        model.score(data, data)


def test_inverse_variance_training_shortcut_matches_prespecified_quality_weights() -> None:
    signal = _trials(813, count=1)[0]
    predictors = [signal.copy(), signal.copy()]
    for scale in [1.0, 1e-9]:
        targets = [signal * scale, 2 * signal * scale]
        fits = []
        for weights in ["inverse_variance", [0.8, 0.2]]:
            model = TRF()
            model.train(predictors, targets, fs=100, tmin=0, tmax=0.03,
                        regularization=1.0, trial_weights=weights)
            fits.append(model.transfer_function)
        np.testing.assert_allclose(*fits, rtol=1e-13, atol=1e-24)


def test_stored_single_candidate_loo_configuration_repeats_validation() -> None:
    data = _trials(817)
    model = TRF()
    expected = model.train(data, data, fs=100, tmin=0, tmax=0.03,
                           regularization=1.0, k="loo")
    result = TRF().train(data, data, **model._fit_config)
    np.testing.assert_allclose(result, expected, atol=1e-14)
