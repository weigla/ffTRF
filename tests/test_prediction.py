from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from fftrf.prediction import (
    _aggregate_null_scores,
    _build_permutation_specs,
    _circular_shift_bounds,
    _compute_bootstrap_interval_from_cache,
    _compute_permutation_test_scores,
    _extract_impulse_response,
    _permutation_p_value_and_z_score,
    _predict_prepared_trials_from_weights,
    _predict_trials_from_weights,
    _prepare_prediction_trials,
    _resolve_permutation_surrogate,
    _resolve_permutation_tail,
    _sample_non_identity_permutation,
    _score_prediction_trials,
    _slice_interval,
    _surrogate_target_trials,
    _validate_confidence_level,
)


def _neg_mse(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return -np.mean((y_true - y_pred) ** 2, axis=0)


def test_impulse_response_extraction_supports_wrapped_negative_lags() -> None:
    full_kernel = np.arange(8.0)[:, np.newaxis, np.newaxis]
    transfer = np.fft.rfft(full_kernel, axis=0)

    weights, times = _extract_impulse_response(
        transfer,
        fs=2.0,
        n_fft=8,
        tmin=-1.0,
        tmax=1.0,
    )

    assert np.array_equal(times, [-1.0, -0.5, 0.0, 0.5])
    assert np.allclose(weights[0, :, 0], [6.0, 7.0, 0.0, 1.0])

    with pytest.raises(ValueError, match="tmax"):
        _extract_impulse_response(transfer, fs=2.0, n_fft=8, tmin=1.0, tmax=1.0)
    with pytest.raises(ValueError, match="longer than n_fft"):
        _extract_impulse_response(transfer, fs=2.0, n_fft=8, tmin=-3.0, tmax=2.0)


def test_interval_slicing_handles_open_bounds_and_returns_copies() -> None:
    interval = np.arange(2 * 1 * 4 * 1, dtype=float).reshape(2, 1, 4, 1)
    times = np.array([-0.1, 0.0, 0.1, 0.2])

    full, full_times = _slice_interval(interval, times, tmin=None, tmax=None)
    left_open, left_times = _slice_interval(interval, times, tmin=None, tmax=0.1)
    right_open, right_times = _slice_interval(interval, times, tmin=0.1, tmax=None)

    assert np.array_equal(full, interval)
    full[0, 0, 0, 0] = -1.0
    assert interval[0, 0, 0, 0] == 0.0
    assert np.array_equal(full_times, times)
    assert np.array_equal(left_times, [-0.1, 0.0])
    assert np.allclose(right_times, [0.1, 0.2])
    assert left_open.shape[2] == right_open.shape[2] == 2

    with pytest.raises(ValueError, match="does not overlap"):
        _slice_interval(interval, times, tmin=1.0, tmax=2.0)


@pytest.mark.parametrize("level", [0.0, 1.0, -0.5])
def test_confidence_level_must_be_strictly_between_zero_and_one(level: float) -> None:
    with pytest.raises(ValueError, match="confidence"):
        _validate_confidence_level(level, name="confidence")


def test_permutation_option_resolution_accepts_aliases_and_rejects_unknown_values() -> None:
    assert _resolve_permutation_surrogate(" TRIAL_SHUFFLE ") == "trial_shuffle"
    assert _resolve_permutation_tail("two_sided") == "two-sided"
    assert _resolve_permutation_tail("less") == "less"

    with pytest.raises(ValueError, match="surrogate"):
        _resolve_permutation_surrogate("reverse")
    with pytest.raises(ValueError, match="tail"):
        _resolve_permutation_tail("upper")


def test_nonidentity_permutation_has_deterministic_fallback() -> None:
    class IdentityGenerator:
        @staticmethod
        def permutation(n_trials: int) -> np.ndarray:
            return np.arange(n_trials)

    order = _sample_non_identity_permutation(IdentityGenerator(), n_trials=3)  # type: ignore[arg-type]
    assert np.array_equal(order, [2, 0, 1])


def test_circular_shift_bounds_validate_trial_length_and_minimum_shift() -> None:
    assert _circular_shift_bounds(n_samples=10, fs=10.0, min_shift=None) == (1, 9)
    assert _circular_shift_bounds(n_samples=10, fs=10.0, min_shift=0.2) == (2, 8)

    with pytest.raises(ValueError, match="at least two"):
        _circular_shift_bounds(n_samples=1, fs=10.0, min_shift=None)
    with pytest.raises(ValueError, match="finite and non-negative"):
        _circular_shift_bounds(n_samples=10, fs=10.0, min_shift=-0.1)
    with pytest.raises(ValueError, match="too large"):
        _circular_shift_bounds(n_samples=10, fs=10.0, min_shift=0.6)


def test_permutation_specs_and_surrogate_trials_preserve_expected_structure() -> None:
    equal_trials = [
        np.arange(6.0)[:, np.newaxis],
        (10.0 + np.arange(6.0))[:, np.newaxis],
    ]
    shuffle_specs = _build_permutation_specs(
        target_trials=equal_trials,
        surrogate="trial_shuffle",
        fs=10.0,
        min_shift=None,
        n_permutations=3,
        seed=1,
    )
    shuffled = _surrogate_target_trials(
        equal_trials,
        surrogate="trial_shuffle",
        spec=shuffle_specs[0],
    )
    assert len(shuffle_specs) == 3
    assert all(not np.array_equal(spec, [0, 1]) for spec in shuffle_specs)
    assert np.array_equal(shuffled[0], equal_trials[int(shuffle_specs[0][0])])

    shift_specs = _build_permutation_specs(
        target_trials=equal_trials,
        surrogate="circular_shift",
        fs=10.0,
        min_shift=0.1,
        n_permutations=2,
        seed=2,
    )
    shifted = _surrogate_target_trials(
        equal_trials,
        surrogate="circular_shift",
        spec=shift_specs[0],
    )
    assert all(spec.shape == (2,) for spec in shift_specs)
    assert np.array_equal(shifted[0], np.roll(equal_trials[0], shift_specs[0][0], axis=0))

    with pytest.raises(ValueError, match="at least two evaluation trials"):
        _build_permutation_specs(
            target_trials=equal_trials[:1],
            surrogate="trial_shuffle",
            fs=10.0,
            min_shift=None,
            n_permutations=1,
            seed=0,
        )
    with pytest.raises(ValueError, match="same sample count"):
        _build_permutation_specs(
            target_trials=[equal_trials[0], equal_trials[1][:-1]],
            surrogate="trial_shuffle",
            fs=10.0,
            min_shift=None,
            n_permutations=1,
            seed=0,
        )


def test_null_score_aggregation_and_tail_statistics_cover_scalar_and_vector_outputs() -> None:
    scores = np.array([[0.0, 1.0], [2.0, 3.0]])
    assert np.array_equal(_aggregate_null_scores(scores, average=False), scores)
    assert np.allclose(_aggregate_null_scores(scores, average=True), [0.5, 2.5])

    constant_null = np.array(
        [
            [1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
        ]
    )
    p_value, z_score = _permutation_p_value_and_z_score(
        observed_score=np.array([2.0, -2.0, 0.0]),
        null_scores=constant_null,
        tail="greater",
    )
    assert np.array_equal(p_value, [0.25, 1.0, 1.0])
    assert np.array_equal(z_score, [np.inf, -np.inf, 0.0])

    scalar_p, scalar_z = _permutation_p_value_and_z_score(
        observed_score=2.0,
        null_scores=np.array([0.0, 1.0, 2.0]),
        tail="less",
    )
    assert isinstance(scalar_p, float)
    assert isinstance(scalar_z, float)

    two_sided_p, _ = _permutation_p_value_and_z_score(
        observed_score=np.array([3.0]),
        null_scores=np.array([[0.0], [1.0], [2.0]]),
        tail="two-sided",
    )
    assert two_sided_p.shape == (1,)


def test_permutation_score_computation_runs_serial_path_and_validates_count() -> None:
    predictions = [
        np.arange(8.0)[:, np.newaxis],
        np.arange(8.0, 16.0)[:, np.newaxis],
    ]
    targets = [trial.copy() for trial in predictions]

    observed, null_scores, p_value, z_score = _compute_permutation_test_scores(
        prediction_trials=predictions,
        target_trials=targets,
        metric=_neg_mse,
        average=True,
        fs=10.0,
        n_permutations=4,
        surrogate="circular_shift",
        min_shift=0.1,
        tail="greater",
        seed=3,
        n_jobs=1,
    )

    assert observed == pytest.approx(0.0)
    assert null_scores.shape == (4,)
    assert 0.0 < p_value <= 1.0
    assert np.isfinite(z_score)

    with pytest.raises(ValueError, match="n_permutations"):
        _compute_permutation_test_scores(
            prediction_trials=predictions,
            target_trials=targets,
            metric=_neg_mse,
            average=True,
            fs=10.0,
            n_permutations=0,
            surrogate="circular_shift",
            min_shift=None,
            tail="greater",
            seed=0,
        )


def test_bootstrap_helper_rejects_invalid_requests_before_computation() -> None:
    common = {
        "fs": 10.0,
        "tmin": 0.0,
        "tmax": 0.2,
        "feature_regularization": np.ones(1),
        "raw_trial_weights": np.ones(2),
        "seed": 0,
    }
    two_trial_cache = SimpleNamespace(trial_cxx=np.zeros((2, 1, 1, 1)))
    one_trial_cache = SimpleNamespace(trial_cxx=np.zeros((1, 1, 1, 1)))

    with pytest.raises(ValueError, match="n_bootstraps"):
        _compute_bootstrap_interval_from_cache(
            two_trial_cache,  # type: ignore[arg-type]
            n_bootstraps=0,
            level=0.9,
            **common,
        )
    with pytest.raises(ValueError, match="level"):
        _compute_bootstrap_interval_from_cache(
            two_trial_cache,  # type: ignore[arg-type]
            n_bootstraps=2,
            level=1.0,
            **common,
        )
    with pytest.raises(ValueError, match="at least two trials"):
        _compute_bootstrap_interval_from_cache(
            one_trial_cache,  # type: ignore[arg-type]
            n_bootstraps=2,
            level=0.9,
            **common,
        )


def test_fft_prediction_handles_positive_and_negative_lag_offsets() -> None:
    predictor = np.array([[1.0], [2.0], [3.0]])
    weights = np.array([[[1.0], [10.0]]])

    negative_lag = _predict_trials_from_weights(
        [predictor],
        weights=weights,
        lag_start=-1,
    )[0]
    positive_lag = _predict_trials_from_weights(
        [predictor],
        weights=weights,
        lag_start=1,
    )[0]

    assert np.allclose(negative_lag[:, 0], [12.0, 23.0, 30.0])
    assert np.allclose(positive_lag[:, 0], [0.0, 1.0, 12.0])

    with pytest.raises(ValueError, match="weights must have shape"):
        _predict_trials_from_weights([predictor], weights=np.ones((2, 2)), lag_start=0)

    prepared = _prepare_prediction_trials([predictor], n_lags=2)
    with pytest.raises(ValueError, match="weights must have shape"):
        _predict_prepared_trials_from_weights(prepared, weights=np.ones((2, 2)), lag_start=0)
    with pytest.raises(ValueError, match="number of input"):
        _predict_prepared_trials_from_weights(
            prepared,
            weights=np.ones((2, 2, 1)),
            lag_start=0,
        )


def test_prediction_scoring_requires_matching_trial_counts() -> None:
    trial = np.zeros((4, 1))
    assert np.array_equal(
        _score_prediction_trials(_neg_mse, [trial], [trial]),
        np.array([0.0]),
    )
    with pytest.raises(ValueError, match="same length"):
        _score_prediction_trials(_neg_mse, [trial], [trial, trial])
