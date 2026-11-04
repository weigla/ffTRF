
"""Prediction, scoring, and bootstrap helpers for ffTRF."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import permutations, product

import numpy as np
from scipy.fft import next_fast_len

from .spectral import (
    _aggregate_cached_spectra,
    _prepare_scalar_ridge_decomposition,
    _scalar_regularization_grid,
    _solve_transfer_function,
    _SpectralCache,
)
from .utils import _aggregate_metric, _resolve_lag_samples, _resolve_n_jobs


@dataclass(slots=True)
class _PreparedPredictionTrial:
    """Cached FFT representation of one predictor trial for repeated predictions."""

    predictor_fft: np.ndarray
    output_length: int
    convolution_length: int
    fft_length: int


def _extract_impulse_response(
    transfer_function: np.ndarray,
    *,
    fs: float,
    n_fft: int,
    tmin: float,
    tmax: float,
) -> tuple[np.ndarray, np.ndarray]:
    lag_start, lag_stop = _resolve_lag_samples(fs=fs, tmin=tmin, tmax=tmax, n_fft=n_fft)

    full_kernel = np.fft.irfft(transfer_function, n=n_fft, axis=0).real
    lag_indices = np.arange(lag_start, lag_stop, dtype=int)
    kernel = full_kernel[np.mod(lag_indices, n_fft), :, :]
    times = lag_indices / fs
    return np.transpose(kernel, (1, 0, 2)), times


def _slice_interval(
    interval: np.ndarray,
    times: np.ndarray,
    *,
    tmin: float | None,
    tmax: float | None,
    fs: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if tmin is None and tmax is None:
        return interval.copy(), times.copy()
    if fs is None:
        fs = 1.0 / (times[1] - times[0]) if times.size > 1 else 1.0
    tmin = times[0] if tmin is None else tmin
    tmax = times[-1] + 1.0 / fs if tmax is None else tmax
    start, stop = _resolve_lag_samples(fs=fs, tmin=tmin, tmax=tmax)
    samples = np.rint(times * fs).astype(int)
    mask = (samples >= start) & (samples < stop)
    if not np.any(mask):
        raise ValueError("Requested lag window does not overlap the stored interval.")
    return interval[:, :, mask, :].copy(), times[mask].copy()


def _validate_confidence_level(level: float, *, name: str) -> None:
    if not 0.0 < float(level) < 1.0:
        raise ValueError(f"{name} must lie strictly between 0 and 1.")


def _resolve_permutation_surrogate(surrogate: str) -> str:
    resolved = str(surrogate).strip().lower()
    if resolved not in {"circular_shift", "trial_shuffle"}:
        raise ValueError(
            "surrogate must be 'circular_shift' or 'trial_shuffle'."
        )
    return resolved


def _resolve_permutation_tail(tail: str) -> str:
    resolved = str(tail).strip().lower()
    if resolved in {"two-sided", "two_sided"}:
        return "two-sided"
    if resolved not in {"greater", "less"}:
        raise ValueError("tail must be 'greater', 'less', or 'two-sided'.")
    return resolved


@dataclass(slots=True)
class _PermutationPlan:
    specs: list[np.ndarray]
    exact: bool


def _circular_shift_bounds(
    *,
    n_samples: int,
    fs: float,
    min_shift: float | None,
) -> tuple[int, int]:
    if n_samples < 2:
        raise ValueError("circular_shift requires trials with at least two samples.")

    if min_shift is None:
        min_shift_samples = 0
    else:
        min_shift_value = float(min_shift)
        if not np.isfinite(min_shift_value) or min_shift_value < 0.0:
            raise ValueError("min_shift must be finite and non-negative when provided.")
        min_shift_samples = int(round(min_shift_value * float(fs)))

    lower = max(1, min_shift_samples)
    upper = n_samples - lower
    if upper < lower:
        raise ValueError(
            "min_shift is too large for at least one trial under circular_shift."
        )
    return lower, upper


def _build_permutation_plan(
    *,
    target_trials: Sequence[np.ndarray],
    surrogate: str,
    fs: float,
    min_shift: float | None,
    n_permutations: int,
    seed: int | None,
) -> _PermutationPlan:
    if n_permutations <= 0:
        raise ValueError("n_permutations must be positive.")
    rng = np.random.default_rng(seed)
    n_trials = len(target_trials)
    if n_trials == 0:
        raise ValueError("Permutation testing requires at least one trial.")
    # Bound exact work even when a caller requests a very large Monte Carlo run.
    exact_limit = min(n_permutations, 10_000)

    if surrogate == "trial_shuffle":
        if n_trials < 2:
            raise ValueError("trial_shuffle requires at least two evaluation trials.")
        lengths = {trial.shape[0] for trial in target_trials}
        if len(lengths) != 1:
            raise ValueError(
                "trial_shuffle requires all evaluation trials to have the same sample count."
            )
        space_size = 1
        for factor in range(2, n_trials + 1):
            space_size *= factor
            if space_size > exact_limit:
                break
        if space_size <= exact_limit:
            return _PermutationPlan([
                np.asarray(order, dtype=int) for order in permutations(range(n_trials))
            ], exact=True)
        return _PermutationPlan([
            rng.permutation(n_trials) for _ in range(n_permutations)
        ], exact=False)

    bounds = [
        _circular_shift_bounds(n_samples=trial.shape[0], fs=fs, min_shift=min_shift)
        for trial in target_trials
    ]
    space_size = 1
    for lower, upper in bounds:
        space_size *= upper - lower + 2  # Nonzero shifts plus the identity.
        if space_size > exact_limit:
            break
    if space_size <= exact_limit:
        choices = [[0, *range(lower, upper + 1)] for lower, upper in bounds]
        return _PermutationPlan([
            np.asarray(shifts, dtype=int) for shifts in product(*choices)
        ], exact=True)
    specs: list[np.ndarray] = []
    for _ in range(n_permutations):
        shifts = np.zeros(n_trials, dtype=int)
        for trial_index, (lower, upper) in enumerate(bounds):
            index = int(rng.integers(0, upper - lower + 2))
            shifts[trial_index] = 0 if index == 0 else lower + index - 1
        specs.append(shifts)
    return _PermutationPlan(specs, exact=False)


def _build_permutation_specs(
    *, target_trials: Sequence[np.ndarray], surrogate: str, fs: float,
    min_shift: float | None, n_permutations: int, seed: int | None,
) -> list[np.ndarray]:
    """Compatibility helper returning specifications from a permutation plan."""

    return _build_permutation_plan(
        target_trials=target_trials, surrogate=surrogate, fs=fs,
        min_shift=min_shift, n_permutations=n_permutations, seed=seed,
    ).specs


def _surrogate_target_trials(
    target_trials: Sequence[np.ndarray],
    *,
    surrogate: str,
    spec: np.ndarray,
) -> list[np.ndarray]:
    if surrogate == "trial_shuffle":
        return [target_trials[int(index)] for index in spec.tolist()]
    return [
        np.roll(target_trial, int(shift), axis=0)
        for target_trial, shift in zip(target_trials, spec, strict=True)
    ]


def _aggregate_null_scores(
    scores: np.ndarray,
    *,
    average: bool | Sequence[int],
) -> np.ndarray:
    if average is False:
        return scores
    return np.asarray(
        [_aggregate_metric(score, average) for score in scores],
        dtype=float,
    )


def _permutation_p_value_and_z_score(
    *,
    observed_score: np.ndarray | float,
    null_scores: np.ndarray,
    tail: str,
    exact: bool = False,
) -> tuple[np.ndarray | float, np.ndarray | float]:
    observed = np.asarray(observed_score, dtype=float)
    null_scores = np.asarray(null_scores, dtype=float)
    if not np.all(np.isfinite(observed)) or not np.all(np.isfinite(null_scores)):
        raise ValueError("Permutation statistics must be finite.")

    scalar_output = observed.ndim == 0
    if scalar_output:
        observed = observed[np.newaxis]
        null_scores = null_scores[:, np.newaxis]

    tolerance = 100 * np.finfo(float).eps * np.maximum(
        np.abs(observed), np.max(np.abs(null_scores), axis=0),
    )
    if tail == "greater":
        exceed = null_scores >= observed[np.newaxis, :] - tolerance
    elif tail == "less":
        exceed = null_scores <= observed[np.newaxis, :] + tolerance
    else:
        centered_sample = null_scores if exact else np.vstack([null_scores, observed])
        center = centered_sample.mean(axis=0, keepdims=True)
        exceed = np.abs(null_scores - center) >= np.abs(observed[np.newaxis, :] - center) - tolerance

    correction = 0.0 if exact else 1.0
    p_value = (correction + exceed.sum(axis=0)) / (null_scores.shape[0] + correction)
    null_mean = null_scores.mean(axis=0)
    null_std = null_scores.std(axis=0, ddof=1 if null_scores.shape[0] > 1 else 0)
    z_score = np.zeros_like(null_mean, dtype=float)

    finite = null_std > 0.0
    z_score[finite] = (observed[finite] - null_mean[finite]) / null_std[finite]
    positive = (~finite) & (observed > null_mean + tolerance)
    negative = (~finite) & (observed < null_mean - tolerance)
    z_score[positive] = np.inf
    z_score[negative] = -np.inf

    if scalar_output:
        return float(p_value[0]), float(z_score[0])
    return p_value, z_score


def _compute_permutation_test_scores(
    *,
    prediction_trials: Sequence[np.ndarray],
    target_trials: Sequence[np.ndarray],
    metric: Callable[[np.ndarray, np.ndarray], np.ndarray],
    average: bool | Sequence[int],
    fs: float,
    n_permutations: int,
    surrogate: str,
    min_shift: float | None,
    tail: str,
    seed: int | None,
    n_jobs: int = 1,
) -> tuple[np.ndarray | float, np.ndarray, np.ndarray | float, np.ndarray | float, bool]:
    if n_permutations <= 0:
        raise ValueError("n_permutations must be positive.")

    resolved_surrogate = _resolve_permutation_surrogate(surrogate)
    resolved_tail = _resolve_permutation_tail(tail)
    observed_per_output = _score_prediction_trials(
        metric,
        target_trials,
        prediction_trials,
    )
    plan = _build_permutation_plan(
        target_trials=target_trials,
        surrogate=resolved_surrogate,
        fs=fs,
        min_shift=min_shift,
        n_permutations=n_permutations,
        seed=seed,
    )
    permutation_specs = plan.specs
    n_permutations = len(permutation_specs)
    null_per_output = np.zeros(
        (n_permutations, observed_per_output.shape[0]),
        dtype=float,
    )

    def _score_surrogate(spec: np.ndarray) -> np.ndarray:
        return _score_prediction_trials(
            metric,
            _surrogate_target_trials(
                target_trials,
                surrogate=resolved_surrogate,
                spec=spec,
            ),
            prediction_trials,
        )

    resolved_n_jobs = min(_resolve_n_jobs(n_jobs), n_permutations)
    if resolved_n_jobs == 1:
        for permutation_index, spec in enumerate(permutation_specs):
            null_per_output[permutation_index] = _score_surrogate(spec)
    else:
        with ThreadPoolExecutor(max_workers=resolved_n_jobs) as executor:
            futures = {
                executor.submit(_score_surrogate, spec): permutation_index
                for permutation_index, spec in enumerate(permutation_specs)
            }
            for future in as_completed(futures):
                permutation_index = futures[future]
                null_per_output[permutation_index] = future.result()

    observed_score = _aggregate_metric(observed_per_output, average)
    null_scores = _aggregate_null_scores(null_per_output, average=average)
    p_value, z_score = _permutation_p_value_and_z_score(
        observed_score=observed_score,
        null_scores=null_scores,
        tail=resolved_tail,
        exact=plan.exact,
    )
    return observed_score, null_scores, p_value, z_score, plan.exact


def _compute_bootstrap_interval_from_cache(
    spectral_cache: _SpectralCache,
    *,
    fs: float,
    tmin: float,
    tmax: float,
    feature_regularization: np.ndarray,
    raw_trial_weights: np.ndarray,
    n_bootstraps: int,
    level: float,
    seed: int | None,
    n_jobs: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    if n_bootstraps <= 0:
        raise ValueError("n_bootstraps must be positive.")
    _validate_confidence_level(level, name="level")
    if spectral_cache.trial_cxx.shape[0] < 2:
        raise ValueError("Bootstrap confidence intervals require at least two trials.")

    rng = np.random.default_rng(seed)
    n_trials = spectral_cache.trial_cxx.shape[0]
    active_trials = np.flatnonzero(raw_trial_weights > 0.0)
    if active_trials.size < 2:
        raise ValueError("Bootstrap confidence intervals require at least two trials with positive weight.")
    sampled_indices = rng.choice(active_trials, size=(n_bootstraps, active_trials.size))
    lag_start, lag_stop = _resolve_lag_samples(
        fs=fs, tmin=tmin, tmax=tmax, n_fft=spectral_cache.n_fft,
    )
    kernels = np.zeros(
        (
            n_bootstraps,
            spectral_cache.trial_cxy.shape[2],
            lag_stop - lag_start,
            spectral_cache.trial_cxy.shape[3],
        ),
        dtype=float,
    )

    def _bootstrap_kernel(sampled_trial_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        sampled_weights = np.bincount(sampled_trial_indices, minlength=n_trials).astype(float)
        sampled_weights *= raw_trial_weights
        cxx, cxy = _aggregate_cached_spectra(
            spectral_cache,
            raw_trial_weights=sampled_weights,
        )
        transfer_function = _solve_transfer_function(
            cxx,
            cxy,
            feature_regularization=feature_regularization,
        )
        return _extract_impulse_response(
            transfer_function,
            fs=fs,
            n_fft=spectral_cache.n_fft,
            tmin=tmin,
            tmax=tmax,
        )

    resolved_n_jobs = min(_resolve_n_jobs(n_jobs), n_bootstraps)
    if resolved_n_jobs == 1:
        for bootstrap_index, sampled_trial_indices in enumerate(sampled_indices):
            kernels[bootstrap_index], times = _bootstrap_kernel(sampled_trial_indices)
    else:
        with ThreadPoolExecutor(max_workers=resolved_n_jobs) as executor:
            futures = {
                executor.submit(_bootstrap_kernel, sampled_trial_indices): bootstrap_index
                for bootstrap_index, sampled_trial_indices in enumerate(sampled_indices)
            }
            for future in as_completed(futures):
                bootstrap_index = futures[future]
                kernels[bootstrap_index], times = future.result()

    alpha = (1.0 - level) / 2.0
    interval = np.quantile(kernels, [alpha, 1.0 - alpha], axis=0)
    return interval, times


def _prepare_prediction_trials(
    predictor_trials: Sequence[np.ndarray],
    *,
    n_lags: int,
) -> list[_PreparedPredictionTrial]:
    """Cache predictor FFTs so repeated predictions only transform kernels."""

    prepared_trials: list[_PreparedPredictionTrial] = []
    for predictor_trial in predictor_trials:
        output_length = int(predictor_trial.shape[0])
        convolution_length = output_length + int(n_lags) - 1
        fft_length = next_fast_len(convolution_length)
        prepared_trials.append(
            _PreparedPredictionTrial(
                predictor_fft=np.fft.rfft(predictor_trial, n=fft_length, axis=0),
                output_length=output_length,
                convolution_length=convolution_length,
                fft_length=fft_length,
            )
        )
    return prepared_trials


def _predict_prepared_trials_from_weights(
    prepared_trials: Sequence[_PreparedPredictionTrial],
    *,
    weights: np.ndarray,
    lag_start: int,
) -> list[np.ndarray]:
    """Generate predictions from cached predictor FFTs and time-domain weights."""

    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 3:
        raise ValueError("weights must have shape (n_inputs, n_lags, n_outputs).")

    _, _, n_outputs = weights.shape
    kernel = np.transpose(weights, (1, 0, 2))
    kernel_fft_cache: dict[int, np.ndarray] = {}
    predictions: list[np.ndarray] = []
    offset = -lag_start

    for prepared_trial in prepared_trials:
        if prepared_trial.predictor_fft.shape[1] != weights.shape[0]:
            raise ValueError(
                "weights and predictor trials must agree on the number of input channels/features."
            )
        kernel_fft = kernel_fft_cache.get(prepared_trial.fft_length)
        if kernel_fft is None:
            kernel_fft = np.fft.rfft(kernel, n=prepared_trial.fft_length, axis=0)
            kernel_fft_cache[prepared_trial.fft_length] = kernel_fft

        full_prediction = np.fft.irfft(
            np.einsum(
                "fi,fio->fo",
                prepared_trial.predictor_fft,
                kernel_fft,
                optimize=True,
            ),
            n=prepared_trial.fft_length,
            axis=0,
        )[: prepared_trial.convolution_length]

        prediction = np.zeros((prepared_trial.output_length, n_outputs), dtype=float)
        src_start = max(offset, 0)
        dst_start = max(-offset, 0)
        length = min(
            full_prediction.shape[0] - src_start,
            prepared_trial.output_length - dst_start,
        )
        if length > 0:
            prediction[dst_start : dst_start + length] = full_prediction[src_start : src_start + length]
        predictions.append(prediction)

    return predictions


def _predict_trials_from_weights(
    predictor_trials: Sequence[np.ndarray],
    *,
    weights: np.ndarray,
    lag_start: int,
) -> list[np.ndarray]:
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 3:
        raise ValueError("weights must have shape (n_inputs, n_lags, n_outputs).")

    prepared_trials = _prepare_prediction_trials(
        predictor_trials,
        n_lags=weights.shape[1],
    )
    return _predict_prepared_trials_from_weights(
        prepared_trials,
        weights=weights,
        lag_start=lag_start,
    )


def _score_prediction_trials(
    metric: Callable[[np.ndarray, np.ndarray], np.ndarray],
    target_trials: Sequence[np.ndarray],
    prediction_trials: Sequence[np.ndarray],
) -> np.ndarray:
    if len(target_trials) != len(prediction_trials):
        raise ValueError("target_trials and prediction_trials must have the same length.")
    scores = []
    for target, prediction in zip(target_trials, prediction_trials, strict=True):
        score = np.asarray(metric(target, prediction), dtype=float)
        if score.shape != (target.shape[1],) or not np.all(np.isfinite(score)):
            raise ValueError("metric must return one finite score per output column.")
        scores.append(score)
    return np.mean(np.vstack(scores), axis=0)


def _score_regularization_grid_for_fold(
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
    metric: Callable[[np.ndarray, np.ndarray], np.ndarray],
    progress_callback: Callable[[int], None] | None = None,
) -> np.ndarray:
    scores = np.zeros(
        (len(feature_regularization_values), val_targets[0].shape[1]),
        dtype=float,
    )
    lag_start, lag_stop = _resolve_lag_samples(fs=fs, tmin=tmin, tmax=tmax, n_fft=n_fft)
    prepared_predictors = _prepare_prediction_trials(
        val_predictors,
        n_lags=lag_stop - lag_start,
    )
    scalar_grid = _scalar_regularization_grid(feature_regularization_values)
    scalar_decomposition = (
        _prepare_scalar_ridge_decomposition(cxx, cxy)
        if scalar_grid is not None
        else None
    )

    for reg_index, feature_regularization in enumerate(feature_regularization_values):
        transfer_function = _solve_transfer_function(
            cxx,
            cxy,
            feature_regularization=feature_regularization,
            scalar_decomposition=scalar_decomposition,
            scalar_regularization=None if scalar_grid is None else float(scalar_grid[reg_index]),
        )
        weights, _ = _extract_impulse_response(
            transfer_function,
            fs=float(fs),
            n_fft=n_fft,
            tmin=float(tmin),
            tmax=float(tmax),
        )
        predictions = _predict_prepared_trials_from_weights(
            prepared_predictors,
            weights=weights,
            lag_start=lag_start,
        )
        scores[reg_index, :] = _score_prediction_trials(
            metric,
            val_targets,
            predictions,
        )
        if progress_callback is not None:
            progress_callback(1)
    return scores
