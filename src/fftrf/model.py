"""Main frequency-domain TRF estimator and core scoring utilities.

The central object exported by this module is :class:`FrequencyTRF`. It mirrors
the basic workflow of mTRF-style toolboxes while estimating the mapping in the
frequency domain:

1. compute cross-spectra between predictor and target signals
2. solve a ridge-regularized linear system at each frequency bin
3. convert the learned transfer function back into a time-domain kernel for
   interpretation and prediction

This is often attractive when continuous signals are long, sampled at high
rates, or represented by many lagged samples.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import pickle
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
from scipy.fft import next_fast_len
from scipy.signal import detrend as scipy_detrend
from scipy.signal import fftconvolve, get_window
from scipy.signal.windows import dpss

_USE_STORED_TRIAL_WEIGHTS = object()
RegularizationSpec = float | tuple[float, ...]
MetricSpec = str | Callable[[np.ndarray, np.ndarray], np.ndarray]
SpectralMethod = str


def pearsonr(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Compute column-wise Pearson correlation.

    Parameters
    ----------
    y_true:
        Observed samples arranged as ``(n_samples, n_outputs)``.
    y_pred:
        Predicted samples with the same shape as ``y_true``.

    Returns
    -------
    numpy.ndarray
        One correlation coefficient per output channel / feature.

    Notes
    -----
    This is the default scoring metric used by :class:`FrequencyTRF`. It is
    intentionally lightweight and does not return p-values.
    """

    y_true = _ensure_2d(y_true, "y_true")
    y_pred = _ensure_2d(y_pred, "y_pred")
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")

    y_true = y_true - y_true.mean(axis=0, keepdims=True)
    y_pred = y_pred - y_pred.mean(axis=0, keepdims=True)

    numerator = np.sum(y_true * y_pred, axis=0)
    denominator = np.sqrt(
        np.sum(y_true**2, axis=0) * np.sum(y_pred**2, axis=0)
    )
    out = np.zeros(y_true.shape[1], dtype=float)
    valid = denominator > np.finfo(float).eps
    out[valid] = numerator[valid] / denominator[valid]
    return out


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Compute column-wise coefficient of determination.

    Parameters
    ----------
    y_true:
        Observed samples arranged as ``(n_samples, n_outputs)``.
    y_pred:
        Predicted samples with the same shape as ``y_true``.

    Returns
    -------
    numpy.ndarray
        One :math:`R^2` value per output column.

    Notes
    -----
    Scores can become negative when predictions are worse than a constant mean
    predictor. This makes the metric suitable for model comparison and
    cross-validation because larger values remain better.
    """

    y_true = _ensure_2d(y_true, "y_true")
    y_pred = _ensure_2d(y_pred, "y_pred")
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")

    residual = np.sum((y_true - y_pred) ** 2, axis=0)
    total = np.sum((y_true - y_true.mean(axis=0, keepdims=True)) ** 2, axis=0)
    eps = np.finfo(float).eps

    out = np.zeros(y_true.shape[1], dtype=float)
    valid = total > eps
    out[valid] = 1.0 - (residual[valid] / total[valid])
    perfect = (~valid) & (residual <= eps)
    out[perfect] = 1.0
    return out


def explained_variance_score(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Compute column-wise explained variance.

    Parameters
    ----------
    y_true:
        Observed samples arranged as ``(n_samples, n_outputs)``.
    y_pred:
        Predicted samples with the same shape as ``y_true``.

    Returns
    -------
    numpy.ndarray
        One explained-variance score per output column.

    Notes
    -----
    Explained variance focuses on residual variance rather than absolute error
    magnitude. Like :func:`r2_score`, larger values are better.
    """

    y_true = _ensure_2d(y_true, "y_true")
    y_pred = _ensure_2d(y_pred, "y_pred")
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")

    residual = y_true - y_pred
    variance_true = np.var(y_true, axis=0)
    variance_residual = np.var(residual, axis=0)
    eps = np.finfo(float).eps

    out = np.zeros(y_true.shape[1], dtype=float)
    valid = variance_true > eps
    out[valid] = 1.0 - (variance_residual[valid] / variance_true[valid])
    perfect = (~valid) & (variance_residual <= eps)
    out[perfect] = 1.0
    return out


_METRIC_REGISTRY: dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    "pearsonr": pearsonr,
    "r2": r2_score,
    "r2_score": r2_score,
    "explained_variance": explained_variance_score,
    "explained_variance_score": explained_variance_score,
}


def available_metrics() -> tuple[str, ...]:
    """Return the names of built-in scoring metrics."""

    return tuple(sorted(_METRIC_REGISTRY))


def _resolve_phase_unit(phase_unit: str) -> str:
    resolved = str(phase_unit).strip().lower()
    if resolved not in {"rad", "deg"}:
        raise ValueError("phase_unit must be either 'rad' or 'deg'.")
    return resolved


def _phase_values(
    transfer_function: np.ndarray,
    *,
    phase_unit: str,
) -> tuple[np.ndarray, str]:
    resolved_unit = _resolve_phase_unit(phase_unit)
    phase = np.unwrap(np.angle(np.asarray(transfer_function, dtype=np.complex128)))
    if resolved_unit == "deg":
        return np.rad2deg(phase), resolved_unit
    return phase, resolved_unit


def _group_delay_values(
    frequencies: np.ndarray,
    transfer_function: np.ndarray,
) -> np.ndarray:
    frequencies = np.asarray(frequencies, dtype=float)
    transfer_function = np.asarray(transfer_function, dtype=np.complex128)
    if frequencies.ndim != 1 or transfer_function.ndim != 1:
        raise ValueError("frequencies and transfer_function must both be 1D arrays.")
    if frequencies.shape[0] != transfer_function.shape[0]:
        raise ValueError("frequencies and transfer_function must have matching lengths.")
    if frequencies.size <= 1:
        return np.zeros_like(frequencies, dtype=float)

    phase = np.unwrap(np.angle(transfer_function))
    omega = 2.0 * np.pi * frequencies
    return -np.gradient(phase, omega, edge_order=1)


def _ensure_2d(array: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(array, dtype=float)
    if array.ndim == 1:
        return array[:, np.newaxis]
    if array.ndim != 2:
        raise ValueError(f"{name} must be 1D or 2D, got shape {array.shape!r}.")
    return array


def _coerce_trials(
    data: np.ndarray | Sequence[np.ndarray],
    name: str,
) -> tuple[list[np.ndarray], bool]:
    if isinstance(data, np.ndarray):
        return [_ensure_2d(data, name)], True
    if not isinstance(data, Sequence) or len(data) == 0:
        raise ValueError(f"{name} must be a non-empty array or sequence of arrays.")
    trials = [_ensure_2d(trial, name) for trial in data]
    return trials, False


def _check_trial_lengths(a_trials: Sequence[np.ndarray], b_trials: Sequence[np.ndarray]) -> None:
    if len(a_trials) != len(b_trials):
        raise ValueError("Stimulus and response must contain the same number of trials.")
    for index, (a_trial, b_trial) in enumerate(zip(a_trials, b_trials)):
        if a_trial.shape[0] != b_trial.shape[0]:
            raise ValueError(
                f"Trial {index} has mismatched lengths: "
                f"{a_trial.shape[0]} vs {b_trial.shape[0]} samples."
            )


def _validate_average_arg(average: bool | Sequence[int]) -> None:
    if average is False or average is True:
        return
    if not isinstance(average, Sequence) or len(average) == 0:
        raise ValueError("average must be True, False, or a non-empty sequence of indices.")


def _aggregate_metric(
    metric: np.ndarray,
    average: bool | Sequence[int],
) -> np.ndarray | float:
    _validate_average_arg(average)
    metric = np.asarray(metric, dtype=float)
    if average is False:
        return metric
    if average is True:
        return float(metric.mean())
    return float(metric[np.asarray(list(average), dtype=int)].mean())


def _normalize_trial_weights(
    y_trials: Sequence[np.ndarray],
    trial_weights: None | str | Sequence[float],
) -> np.ndarray:
    return _normalize_weight_vector(_resolve_raw_trial_weights(y_trials, trial_weights))


def _resolve_metric(
    metric: MetricSpec,
) -> tuple[Callable[[np.ndarray, np.ndarray], np.ndarray], str | None]:
    if isinstance(metric, str):
        key = metric.strip().lower()
        if key not in _METRIC_REGISTRY:
            available = ", ".join(sorted(_METRIC_REGISTRY))
            raise ValueError(f"Unknown metric {metric!r}. Available built-ins are: {available}.")
        return _METRIC_REGISTRY[key], key

    if not callable(metric):
        raise ValueError("metric must be callable or a known metric name.")
    return metric, getattr(metric, "__name__", None)


def _copy_value(value):
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, list):
        return [_copy_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _copy_value(item) for key, item in value.items()}
    return value


def _coerce_nonnegative_float(value: float, *, name: str) -> float:
    coerced = float(value)
    if not np.isfinite(coerced) or coerced < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")
    return coerced


def _is_scalar_like(value: object) -> bool:
    try:
        return np.asarray(value).ndim == 0
    except Exception:
        return False


def _validate_bands(
    bands: None | Sequence[int],
    *,
    n_inputs: int,
) -> tuple[int, ...] | None:
    if bands is None:
        return None
    try:
        raw_bands = list(bands)
    except TypeError as exc:
        raise ValueError("bands must be a non-empty sequence of positive integers.") from exc
    if len(raw_bands) == 0:
        raise ValueError("bands must be a non-empty sequence of positive integers.")

    resolved_bands = []
    for band_index, band in enumerate(raw_bands):
        band_float = float(band)
        if (
            not np.isfinite(band_float)
            or band_float <= 0.0
            or not band_float.is_integer()
        ):
            raise ValueError(
                f"bands[{band_index}] must be a positive integer, got {band!r}."
            )
        resolved_bands.append(int(band_float))

    if sum(resolved_bands) != n_inputs:
        raise ValueError(
            "bands must sum to the number of predictor features. "
            f"Got sum(bands)={sum(resolved_bands)} and n_inputs={n_inputs}."
        )
    return tuple(resolved_bands)


def _expand_feature_regularization(
    coefficients: Sequence[float],
    *,
    n_inputs: int,
    bands: tuple[int, ...] | None,
) -> np.ndarray:
    if bands is None:
        if len(coefficients) != 1:
            raise ValueError("Scalar ridge regularization expects exactly one coefficient.")
        return np.full(
            n_inputs,
            _coerce_nonnegative_float(coefficients[0], name="regularization"),
            dtype=float,
        )

    if len(coefficients) != len(bands):
        raise ValueError("One regularization coefficient is required for each entry in bands.")

    return np.concatenate(
        [
            np.full(
                band_size,
                _coerce_nonnegative_float(coefficient, name="regularization"),
                dtype=float,
            )
            for coefficient, band_size in zip(coefficients, bands)
        ]
    )


def _resolve_regularization_candidates(
    regularization: float | Sequence[float] | Sequence[Sequence[float]],
    *,
    n_inputs: int,
    bands: tuple[int, ...] | None,
) -> tuple[list[np.ndarray], list[RegularizationSpec]]:
    if _is_scalar_like(regularization):
        value = _coerce_nonnegative_float(float(regularization), name="regularization")
        spec: RegularizationSpec = value if bands is None else tuple([value] * len(bands))
        return [
            _expand_feature_regularization(
                [value] if bands is None else [value] * len(bands),
                n_inputs=n_inputs,
                bands=bands,
            )
        ], [spec]

    raw_items = list(regularization)
    if len(raw_items) == 0:
        raise ValueError("regularization must be a scalar or a non-empty sequence.")

    if bands is None:
        if any(not _is_scalar_like(item) for item in raw_items):
            raise ValueError(
                "Without bands, regularization must be a scalar or a 1D sequence of scalars."
            )
        values = [
            _coerce_nonnegative_float(float(item), name="regularization")
            for item in raw_items
        ]
        return [
            _expand_feature_regularization([value], n_inputs=n_inputs, bands=None)
            for value in values
        ], values

    n_bands = len(bands)
    if all(_is_scalar_like(item) for item in raw_items):
        pool = [
            _coerce_nonnegative_float(float(item), name="regularization")
            for item in raw_items
        ]
        if len(pool) == 1:
            coefficients = tuple([pool[0]] * n_bands)
            return [
                _expand_feature_regularization(
                    coefficients,
                    n_inputs=n_inputs,
                    bands=bands,
                )
            ], [coefficients]

        coefficient_sets = [tuple(combo) for combo in product(pool, repeat=n_bands)]
        return [
            _expand_feature_regularization(
                coefficients,
                n_inputs=n_inputs,
                bands=bands,
            )
            for coefficients in coefficient_sets
        ], coefficient_sets

    if any(_is_scalar_like(item) for item in raw_items):
        raise ValueError(
            "When bands is provided, explicit banded regularization candidates "
            "must either be all scalars or all sequences of per-band coefficients."
        )

    specs: list[RegularizationSpec] = []
    penalties: list[np.ndarray] = []
    for candidate_index, candidate in enumerate(raw_items):
        coefficients = np.asarray(candidate, dtype=float)
        if coefficients.ndim != 1 or coefficients.shape[0] != n_bands:
            raise ValueError(
                "Each explicit banded regularization candidate must be a 1D sequence "
                f"with {n_bands} values, got shape {coefficients.shape!r} for "
                f"candidate {candidate_index}."
            )
        coefficient_tuple = tuple(
            _coerce_nonnegative_float(value, name="regularization")
            for value in coefficients.tolist()
        )
        specs.append(coefficient_tuple)
        penalties.append(
            _expand_feature_regularization(
                coefficient_tuple,
                n_inputs=n_inputs,
                bands=bands,
            )
        )
    return penalties, specs


@dataclass(slots=True)
class _SpectralCache:
    """Per-trial spectral sufficient statistics reused across repeated fits."""

    trial_cxx: np.ndarray
    trial_cxy: np.ndarray
    segment_length: int
    n_fft: int
    overlap: float
    spectral_method: SpectralMethod
    time_bandwidth: float | None
    n_tapers: int | None
    window: None | str | tuple[str, float] | np.ndarray
    detrend: None | str


@dataclass(slots=True)
class FrequencyTRFDiagnostics:
    """Observed-vs-predicted spectral diagnostics for a fitted model.

    Attributes
    ----------
    frequencies:
        Frequency vector in Hz.
    transfer_function:
        Complex transfer function copied from the fitted model. Shape is
        ``(n_frequencies, n_inputs, n_outputs)``.
    predicted_spectrum:
        Diagonal auto-spectrum of the model predictions for each output. Shape
        is ``(n_frequencies, n_outputs)``.
    observed_spectrum:
        Diagonal auto-spectrum of the observed targets. Shape is
        ``(n_frequencies, n_outputs)``.
    cross_spectrum:
        Matched predicted-vs-observed cross-spectrum for each output. Shape is
        ``(n_frequencies, n_outputs)``.
    coherence:
        Magnitude-squared coherence between each predicted output and the
        corresponding observed target. Shape is ``(n_frequencies, n_outputs)``.
    """

    frequencies: np.ndarray
    transfer_function: np.ndarray
    predicted_spectrum: np.ndarray
    observed_spectrum: np.ndarray
    cross_spectrum: np.ndarray
    coherence: np.ndarray


CrossSpectralDiagnostics = FrequencyTRFDiagnostics


@dataclass(slots=True)
class TransferFunctionComponents:
    """Derived one-pair transfer-function quantities.

    Attributes
    ----------
    frequencies:
        Frequency vector in Hz.
    transfer_function:
        Complex transfer-function values for one input/output pair.
    magnitude:
        Absolute value of the transfer function.
    phase:
        Unwrapped transfer-function phase, reported in the requested units.
    phase_unit:
        Unit used for :attr:`phase`, either ``"rad"`` or ``"deg"``.
    group_delay:
        Group delay derived from the unwrapped phase, expressed in seconds.
    """

    frequencies: np.ndarray
    transfer_function: np.ndarray
    magnitude: np.ndarray
    phase: np.ndarray
    phase_unit: str
    group_delay: np.ndarray


def _resolve_raw_trial_weights(
    y_trials: Sequence[np.ndarray],
    trial_weights: None | str | Sequence[float],
) -> np.ndarray:
    if trial_weights is None:
        weights = np.ones(len(y_trials), dtype=float)
    elif isinstance(trial_weights, str):
        if trial_weights != "inverse_variance":
            raise ValueError("trial_weights must be None, 'inverse_variance', or a weight vector.")
        variances = [
            np.var(y_trial, axis=0).mean()
            for y_trial in y_trials
        ]
        weights = 1.0 / np.clip(variances, np.finfo(float).eps, None)
    else:
        weights = np.asarray(trial_weights, dtype=float)
        if weights.shape != (len(y_trials),):
            raise ValueError("Explicit trial weights must match the number of trials.")

    if np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("Trial weights must be finite and non-negative.")
    return weights


def _normalize_weight_vector(weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Trial weights must sum to a positive finite value.")
    return weights / total


def _validate_spectral_method(spectral_method: str) -> str:
    resolved = str(spectral_method).strip().lower()
    if resolved not in {"standard", "multitaper"}:
        raise ValueError("spectral_method must be either 'standard' or 'multitaper'.")
    return resolved


def _resolve_multitaper_parameters(
    *,
    time_bandwidth: float,
    n_tapers: int | None,
) -> tuple[float, int]:
    resolved_time_bandwidth = float(time_bandwidth)
    if not np.isfinite(resolved_time_bandwidth) or resolved_time_bandwidth <= 0.5:
        raise ValueError("time_bandwidth must be finite and greater than 0.5 for multitaper.")

    max_tapers = max(1, int(np.floor(2.0 * resolved_time_bandwidth - 1.0)))
    if n_tapers is None:
        resolved_n_tapers = max_tapers
    else:
        resolved_n_tapers = int(n_tapers)
        if resolved_n_tapers <= 0:
            raise ValueError("n_tapers must be positive.")
        if resolved_n_tapers > max_tapers:
            raise ValueError(
                "n_tapers is too large for the requested time_bandwidth. "
                f"Use at most {max_tapers} tapers for time_bandwidth={resolved_time_bandwidth}."
            )
    return resolved_time_bandwidth, resolved_n_tapers


def _prepare_segment(
    segment: np.ndarray,
    *,
    target_length: int,
    window_cache: dict[int, np.ndarray],
    window: None | str | tuple[str, float] | np.ndarray,
    detrend: None | str,
) -> np.ndarray:
    if detrend is not None:
        segment = scipy_detrend(segment, axis=0, type=detrend)
    if window is not None:
        if isinstance(window, np.ndarray):
            if window.shape != (segment.shape[0],):
                raise ValueError("Explicit window array must match the segment length.")
            window_values = window
        else:
            window_values = window_cache.get(segment.shape[0])
            if window_values is None:
                window_values = get_window(window, segment.shape[0], fftbins=True)
                window_cache[segment.shape[0]] = window_values
        segment = segment * window_values[:, np.newaxis]
    if segment.shape[0] == target_length:
        return segment

    padded = np.zeros((target_length, segment.shape[1]), dtype=float)
    padded[: segment.shape[0], :] = segment
    return padded


def _iter_segments(
    x_trial: np.ndarray,
    y_trial: np.ndarray,
    segment_length: int,
    overlap: float,
) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    n_samples = x_trial.shape[0]
    if segment_length >= n_samples:
        yield x_trial, y_trial
        return

    step = max(1, int(round(segment_length * (1.0 - overlap))))
    starts = list(range(0, n_samples - segment_length + 1, step))
    if not starts:
        yield x_trial, y_trial
        return

    last_stop = -1
    for start in starts:
        stop = start + segment_length
        yield x_trial[start:stop], y_trial[start:stop]
        last_stop = stop

    if last_stop < n_samples:
        start = n_samples - segment_length
        yield x_trial[start:n_samples], y_trial[start:n_samples]


def _resolve_spectral_settings(
    x_trials: Sequence[np.ndarray],
    *,
    segment_length: int | None,
    n_fft: int | None,
) -> tuple[int, int]:
    resolved_segment_length = (
        max(trial.shape[0] for trial in x_trials)
        if segment_length is None
        else int(segment_length)
    )
    resolved_n_fft = (
        next_fast_len(resolved_segment_length)
        if n_fft is None
        else int(n_fft)
    )
    if resolved_n_fft < resolved_segment_length:
        raise ValueError("n_fft must be at least as large as segment_length.")
    return resolved_segment_length, resolved_n_fft


def _build_spectral_cache(
    x_trials: Sequence[np.ndarray],
    y_trials: Sequence[np.ndarray],
    *,
    segment_length: int | None,
    overlap: float,
    n_fft: int | None,
    spectral_method: SpectralMethod,
    time_bandwidth: float,
    n_tapers: int | None,
    window: None | str | tuple[str, float] | np.ndarray,
    detrend: None | str,
) -> _SpectralCache:
    spectral_method = _validate_spectral_method(spectral_method)
    resolved_segment_length, resolved_n_fft = _resolve_spectral_settings(
        x_trials,
        segment_length=segment_length,
        n_fft=n_fft,
    )
    n_trials = len(x_trials)
    n_inputs = x_trials[0].shape[1]
    n_outputs = y_trials[0].shape[1]
    n_frequencies = resolved_n_fft // 2 + 1

    trial_cxx = np.zeros((n_trials, n_frequencies, n_inputs, n_inputs), dtype=np.complex128)
    trial_cxy = np.zeros((n_trials, n_frequencies, n_inputs, n_outputs), dtype=np.complex128)

    window_cache: dict[int, np.ndarray] = {}
    taper_cache: dict[tuple[int, float, int], np.ndarray] = {}
    if spectral_method == "multitaper":
        resolved_time_bandwidth, resolved_n_tapers = _resolve_multitaper_parameters(
            time_bandwidth=float(time_bandwidth),
            n_tapers=n_tapers,
        )
    else:
        resolved_time_bandwidth = None
        resolved_n_tapers = None

    for trial_index, (x_trial, y_trial) in enumerate(zip(x_trials, y_trials)):
        segments = list(
            _iter_segments(
                x_trial,
                y_trial,
                segment_length=resolved_segment_length,
                overlap=overlap,
            )
        )
        segment_weight = 1.0 / len(segments)
        for x_segment, y_segment in segments:
            if spectral_method == "multitaper":
                taper_key = (
                    x_segment.shape[0],
                    float(resolved_time_bandwidth),
                    int(resolved_n_tapers),
                )
                tapers = taper_cache.get(taper_key)
                if tapers is None:
                    tapers = np.asarray(
                        dpss(
                            x_segment.shape[0],
                            NW=float(resolved_time_bandwidth),
                            Kmax=int(resolved_n_tapers),
                            sym=False,
                        ),
                        dtype=float,
                    )
                    tapers /= np.sqrt(np.mean(tapers**2, axis=1, keepdims=True))
                    taper_cache[taper_key] = tapers
                taper_weight = segment_weight / tapers.shape[0]
                for taper in tapers:
                    x_prepared = _prepare_segment(
                        x_segment,
                        target_length=resolved_segment_length,
                        window_cache=window_cache,
                        window=taper,
                        detrend=detrend,
                    )
                    y_prepared = _prepare_segment(
                        y_segment,
                        target_length=resolved_segment_length,
                        window_cache=window_cache,
                        window=taper,
                        detrend=detrend,
                    )

                    x_fft = np.fft.rfft(x_prepared, n=resolved_n_fft, axis=0)
                    y_fft = np.fft.rfft(y_prepared, n=resolved_n_fft, axis=0)

                    trial_cxx[trial_index] += taper_weight * np.einsum(
                        "fi,fj->fij", np.conjugate(x_fft), x_fft, optimize=True
                    )
                    trial_cxy[trial_index] += taper_weight * np.einsum(
                        "fi,fj->fij", np.conjugate(x_fft), y_fft, optimize=True
                    )
            else:
                x_prepared = _prepare_segment(
                    x_segment,
                    target_length=resolved_segment_length,
                    window_cache=window_cache,
                    window=window,
                    detrend=detrend,
                )
                y_prepared = _prepare_segment(
                    y_segment,
                    target_length=resolved_segment_length,
                    window_cache=window_cache,
                    window=window,
                    detrend=detrend,
                )

                x_fft = np.fft.rfft(x_prepared, n=resolved_n_fft, axis=0)
                y_fft = np.fft.rfft(y_prepared, n=resolved_n_fft, axis=0)

                trial_cxx[trial_index] += segment_weight * np.einsum(
                    "fi,fj->fij", np.conjugate(x_fft), x_fft, optimize=True
                )
                trial_cxy[trial_index] += segment_weight * np.einsum(
                    "fi,fj->fij", np.conjugate(x_fft), y_fft, optimize=True
                )

    return _SpectralCache(
        trial_cxx=trial_cxx,
        trial_cxy=trial_cxy,
        segment_length=resolved_segment_length,
        n_fft=resolved_n_fft,
        overlap=overlap,
        spectral_method=spectral_method,
        time_bandwidth=resolved_time_bandwidth,
        n_tapers=resolved_n_tapers,
        window=_copy_value(window),
        detrend=detrend,
    )


def _aggregate_cached_spectra(
    spectral_cache: _SpectralCache,
    *,
    trial_indices: np.ndarray | None = None,
    raw_trial_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if trial_indices is None:
        trial_indices = np.arange(spectral_cache.trial_cxx.shape[0], dtype=int)
    else:
        trial_indices = np.asarray(trial_indices, dtype=int)
    if raw_trial_weights is None:
        normalized_weights = np.full(trial_indices.shape[0], 1.0 / trial_indices.shape[0], dtype=float)
    else:
        normalized_weights = _normalize_weight_vector(raw_trial_weights[trial_indices])

    cxx = np.tensordot(
        normalized_weights,
        spectral_cache.trial_cxx[trial_indices],
        axes=(0, 0),
    )
    cxy = np.tensordot(
        normalized_weights,
        spectral_cache.trial_cxy[trial_indices],
        axes=(0, 0),
    )
    return cxx, cxy


def _solve_transfer_function(
    cxx: np.ndarray,
    cxy: np.ndarray,
    *,
    feature_regularization: np.ndarray,
) -> np.ndarray:
    feature_regularization = np.asarray(feature_regularization, dtype=float)
    if feature_regularization.shape != (cxx.shape[1],):
        raise ValueError(
            "feature_regularization must provide one penalty per predictor feature."
        )

    if np.allclose(feature_regularization, feature_regularization[0]):
        ridge_matrix = (
            _coerce_nonnegative_float(
                float(feature_regularization[0]),
                name="feature_regularization",
            )
            * np.eye(cxx.shape[1], dtype=np.complex128)
        )
    else:
        ridge_matrix = np.diag(feature_regularization.astype(np.complex128))

    transfer_function = np.zeros_like(cxy)
    for frequency_index in range(cxx.shape[0]):
        system = cxx[frequency_index] + ridge_matrix
        transfer_function[frequency_index] = np.linalg.solve(system, cxy[frequency_index])
    return transfer_function


def _extract_impulse_response(
    transfer_function: np.ndarray,
    *,
    fs: float,
    n_fft: int,
    tmin: float,
    tmax: float,
) -> tuple[np.ndarray, np.ndarray]:
    lag_start = int(round(float(tmin) * fs))
    lag_stop = int(round(float(tmax) * fs))
    if lag_stop <= lag_start:
        raise ValueError("tmax must be greater than tmin.")
    if lag_stop - lag_start > n_fft:
        raise ValueError("Requested lag window is longer than n_fft.")

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
) -> tuple[np.ndarray, np.ndarray]:
    if tmin is None and tmax is None:
        return interval.copy(), times.copy()
    tmin = times[0] if tmin is None else tmin
    if tmax is None:
        step = (times[1] - times[0]) if times.size > 1 else 1.0
        tmax = times[-1] + step
    mask = (times >= tmin) & (times < tmax)
    if not np.any(mask):
        raise ValueError("Requested lag window does not overlap the stored interval.")
    return interval[:, :, mask, :].copy(), times[mask].copy()


def _validate_confidence_level(level: float, *, name: str) -> None:
    if not 0.0 < float(level) < 1.0:
        raise ValueError(f"{name} must lie strictly between 0 and 1.")


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
) -> tuple[np.ndarray, np.ndarray]:
    if n_bootstraps <= 0:
        raise ValueError("n_bootstraps must be positive.")
    _validate_confidence_level(level, name="level")
    if spectral_cache.trial_cxx.shape[0] < 2:
        raise ValueError("Bootstrap confidence intervals require at least two trials.")

    rng = np.random.default_rng(seed)
    n_trials = spectral_cache.trial_cxx.shape[0]
    kernels = np.zeros(
        (
            n_bootstraps,
            spectral_cache.trial_cxy.shape[2],
            int(round((tmax - tmin) * fs)),
            spectral_cache.trial_cxy.shape[3],
        ),
        dtype=float,
    )

    for bootstrap_index in range(n_bootstraps):
        sampled_indices = rng.integers(0, n_trials, size=n_trials)
        sampled_weights = np.bincount(sampled_indices, minlength=n_trials).astype(float)
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
        kernels[bootstrap_index], times = _extract_impulse_response(
            transfer_function,
            fs=fs,
            n_fft=spectral_cache.n_fft,
            tmin=tmin,
            tmax=tmax,
        )

    alpha = (1.0 - level) / 2.0
    interval = np.quantile(kernels, [alpha, 1.0 - alpha], axis=0)
    return interval, times


def _shifted_convolution(
    signal_in: np.ndarray,
    kernel: np.ndarray,
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


class FrequencyTRF:
    """
    Estimate stimulus-response mappings in the frequency domain.

    ``FrequencyTRF`` is the main estimator of this toolbox. Its public API is
    intentionally close to ``mTRFpy``:

    - call :meth:`train` to fit the model
    - call :meth:`predict` to generate predicted responses or stimuli
    - call :meth:`score` to evaluate predictions
    - call :meth:`plot` to visualize the fitted kernel
    - call :meth:`plot_grid` to visualize all input/output kernels at once
    - call :meth:`plot_transfer_function` to inspect magnitude, phase, or group delay
    - call :meth:`cross_spectral_diagnostics`, :meth:`plot_coherence`, and
      :meth:`plot_cross_spectrum` for spectral prediction diagnostics
    - inspect :attr:`weights` and :attr:`times` as the time-domain kernel

    Unlike a classic time-domain TRF, the fit is performed through
    ridge-regularized spectral deconvolution. This is often attractive for
    high-rate continuous data where explicitly building large lag matrices is
    cumbersome. When multiple regularization values are supplied, the estimator
    caches per-trial spectral statistics so cross-validation can reuse them
    instead of recomputing FFTs for every fold and candidate value.

    Parameters
    ----------
    direction:
        Modeling direction. Use ``1`` for a forward model
        (stimulus -> neural response) and ``-1`` for a backward model
        (neural response -> stimulus).
    metric:
        Callable or built-in metric name used to score predictions. It must
        accept ``(y_true, y_pred)`` and return one score per output column.
        Built-ins currently include ``"pearsonr"``, ``"r2"``, and
        ``"explained_variance"``.

    Attributes
    ----------
    transfer_function:
        Complex-valued frequency-domain mapping with shape
        ``(n_frequencies, n_inputs, n_outputs)``.
    frequencies:
        Frequency vector in Hz corresponding to ``transfer_function``.
    weights:
        Time-domain kernel extracted from ``transfer_function`` over the fitted
        lag window. Shape is ``(n_inputs, n_lags, n_outputs)``.
    times:
        Lag values in seconds corresponding to the second axis of
        :attr:`weights`.
    regularization:
        Selected ridge parameter. In ordinary ridge mode this is a scalar. When
        ``bands`` are used, it stores one coefficient per band.
    bands:
        Optional contiguous feature-group definition used for banded
        regularization.
    feature_regularization:
        Expanded per-feature penalty vector actually used by the spectral
        solver. This is especially useful when banded regularization is active.
    regularization_candidates:
        Candidate ridge values or banded coefficient tuples evaluated during
        training. This lets cross-validation scores be mapped back to the
        tested values.
    fs:
        Sampling rate used during fitting.
    bootstrap_interval:
        Optional trial-bootstrap confidence interval with shape
        ``(2, n_inputs, n_lags, n_outputs)``.
    bootstrap_level:
        Confidence level used for :attr:`bootstrap_interval`.
    spectral_method:
        Spectral estimator used during fitting. ``"standard"`` denotes the
        default windowed FFT estimator and ``"multitaper"`` activates DPSS
        multi-taper averaging.
    time_bandwidth, n_tapers:
        Multi-taper settings stored for fitted models that use
        ``spectral_method="multitaper"``.

    Examples
    --------
    >>> import numpy as np
    >>> from fftrf import FrequencyTRF
    >>> x = np.random.randn(2000, 1)
    >>> y = np.random.randn(2000, 1)
    >>> model = FrequencyTRF(direction=1)
    >>> model.train(x, y, fs=1000, tmin=0.0, tmax=0.03, regularization=1e-3)
    >>> prediction = model.predict(stimulus=x)
    """

    def __init__(
        self,
        direction: int = 1,
        metric: MetricSpec = pearsonr,
    ) -> None:
        if direction not in (1, -1):
            raise ValueError("direction must be 1 (forward) or -1 (backward).")

        self.direction = direction
        self.metric, self.metric_name = _resolve_metric(metric)

        self.transfer_function: np.ndarray | None = None
        self.frequencies: np.ndarray | None = None
        self.weights: np.ndarray | None = None
        self.times: np.ndarray | None = None

        self.fs: float | None = None
        self.regularization: RegularizationSpec | None = None
        self.bands: tuple[int, ...] | None = None
        self.feature_regularization: np.ndarray | None = None
        self.regularization_candidates: list[RegularizationSpec] | None = None
        self.segment_length: int | None = None
        self.n_fft: int | None = None
        self.overlap: float | None = None
        self.spectral_method: SpectralMethod = "standard"
        self.time_bandwidth: float | None = None
        self.n_tapers: int | None = None
        self.window: None | str | tuple[str, float] | np.ndarray = None
        self.detrend: None | str = None
        self.tmin: float | None = None
        self.tmax: float | None = None
        self.trial_weights: None | str | np.ndarray = None
        self.bootstrap_interval: np.ndarray | None = None
        self.bootstrap_level: float | None = None
        self.bootstrap_samples: int | None = None

    def train(
        self,
        stimulus: np.ndarray | Sequence[np.ndarray],
        response: np.ndarray | Sequence[np.ndarray],
        fs: float,
        tmin: float,
        tmax: float,
        regularization: float | Sequence[float] | Sequence[Sequence[float]],
        *,
        bands: None | Sequence[int] = None,
        segment_length: int | None = None,
        overlap: float = 0.0,
        n_fft: int | None = None,
        spectral_method: SpectralMethod = "standard",
        time_bandwidth: float = 3.5,
        n_tapers: int | None = None,
        window: None | str | tuple[str, float] | np.ndarray = None,
        detrend: None | str = "constant",
        k: int = -1,
        average: bool | Sequence[int] = True,
        seed: int | None = None,
        trial_weights: None | str | Sequence[float] = None,
        bootstrap_samples: int = 0,
        bootstrap_level: float = 0.95,
        bootstrap_seed: int | None = None,
    ) -> np.ndarray | float | None:
        """
        Fit the frequency-domain TRF.

        Parameters
        ----------
        stimulus:
            One trial as a 1D/2D array or multiple trials as a list of arrays.
            Each trial must have shape ``(n_samples, n_features)``. A 1D vector
            is treated as a single-feature input.
        response:
            Neural data with one trial as a 1D/2D array or multiple trials as a
            list of arrays. Each trial must have shape
            ``(n_samples, n_outputs)``.
        fs:
            Sampling rate in Hz shared by stimulus and response.
        tmin, tmax:
            Time window, in seconds, that should be extracted from the learned
            transfer function as a time-domain kernel.
        regularization:
            Regularization specification. The default behavior matches a
            standard ridge TRF fit:

            - scalar: fit directly with one ridge value
            - 1D sequence of scalars: cross-validate over those candidates

            When ``bands`` is provided, each feature group gets its own ridge
            coefficient. In that mode, a 1D scalar sequence follows the
            ``mTRFpy`` banded-ridge convention: the Cartesian product across
            bands is evaluated during cross-validation. You can also pass an
            explicit sequence of per-band coefficient tuples.
        bands:
            Optional contiguous feature-group sizes for banded ridge
            regularization. For example, if the predictor contains one envelope
            feature followed by a 16-band spectrogram, use ``bands=[1, 16]``.
            Leaving this as ``None`` keeps the estimator in ordinary scalar
            ridge mode.
        segment_length:
            Segment size used to estimate cross-spectra. If ``None``, each trial
            is treated as a single segment.
        overlap:
            Fractional overlap between neighboring segments. Must lie in
            ``[0, 1)``.
        n_fft:
            FFT size used for spectral estimation. If omitted, a fast FFT length
            is chosen automatically from ``segment_length``.
        spectral_method:
            Spectral estimator used to compute the sufficient statistics.
            ``"standard"`` keeps the current windowed FFT behavior.
            ``"multitaper"`` averages DPSS-tapered spectra and is often more
            stable for noisy continuous data.
        time_bandwidth:
            Time-bandwidth product used when ``spectral_method="multitaper"``.
            Larger values produce broader spectral smoothing and allow more
            tapers.
        n_tapers:
            Number of DPSS tapers used for ``spectral_method="multitaper"``.
            If omitted, the default ``2 * time_bandwidth - 1`` rule is used.
        window:
            Window applied to each segment before FFT. By default no window is
            applied, which keeps the behavior closer to a standard lagged ridge
            fit. When using short overlapping segments, ``window="hann"`` is
            often a good choice. In multi-taper mode this must be ``None``
            because the DPSS tapers already define the segment weighting.
        detrend:
            Optional detrending passed to :func:`scipy.signal.detrend`.
        k:
            Number of cross-validation folds when multiple regularization values
            are supplied. ``-1`` means leave-one-out over trials.
        average:
            How cross-validation scores should be reduced across output
            channels/features. ``True`` returns a single score per regularization
            value, ``False`` returns one score per output, and a sequence of
            indices averages only over the selected outputs.
        seed:
            Optional random seed for shuffling trial order before creating folds.
        trial_weights:
            Optional trial weights. Use ``"inverse_variance"`` for
            inverse-variance weighting or pass an explicit vector with one
            weight per training trial.
        bootstrap_samples:
            Number of trial-bootstrap resamples used to estimate a confidence
            interval for the fitted kernel. ``0`` disables the bootstrap.
        bootstrap_level:
            Confidence level used for the stored bootstrap interval.
        bootstrap_seed:
            Optional random seed used for bootstrap resampling.

        Returns
        -------
        None or numpy.ndarray
            ``None`` when a single regularization value is fitted directly.
            Otherwise returns cross-validation scores for each candidate
            regularization setting in the order stored by
            :attr:`regularization_candidates`.

        Notes
        -----
        The fitted model is always stored on the instance, even when
        cross-validation is used. In that case the final fit uses the selected
        regularization value and all provided trials. When multiple
        regularization values are supplied, the per-trial spectra are cached so
        the FFT work is performed only once. Banded regularization is entirely
        opt-in through ``bands``; leaving it unset preserves the default
        "mTRF in Fourier space" workflow.
        """

        stimulus_trials, _ = _coerce_trials(stimulus, "stimulus")
        response_trials, _ = _coerce_trials(response, "response")
        _check_trial_lengths(stimulus_trials, response_trials)

        x_trials, y_trials = self._get_xy(stimulus_trials, response_trials)
        self._validate_dimensions(x_trials, y_trials)
        self._validate_fit_arguments(
            fs,
            tmin,
            tmax,
            segment_length,
            overlap,
            n_fft,
            spectral_method,
            time_bandwidth,
            n_tapers,
            window,
            detrend,
        )
        if int(bootstrap_samples) < 0:
            raise ValueError("bootstrap_samples must be non-negative.")
        _validate_confidence_level(bootstrap_level, name="bootstrap_level")
        spectral_method = _validate_spectral_method(spectral_method)
        resolved_bands = _validate_bands(bands, n_inputs=x_trials[0].shape[1])
        feature_regularization_values, regularization_specs = _resolve_regularization_candidates(
            regularization,
            n_inputs=x_trials[0].shape[1],
            bands=resolved_bands,
        )
        raw_trial_weights = _resolve_raw_trial_weights(y_trials, trial_weights)
        self.trial_weights = _copy_value(trial_weights)
        self.bands = resolved_bands
        self.regularization_candidates = [_copy_value(spec) for spec in regularization_specs]
        self.bootstrap_interval = None
        self.bootstrap_level = None
        self.bootstrap_samples = None

        needs_cache = len(feature_regularization_values) > 1 or int(bootstrap_samples) > 0
        spectral_cache = None
        if needs_cache:
            spectral_cache = _build_spectral_cache(
                x_trials,
                y_trials,
                segment_length=segment_length,
                overlap=overlap,
                n_fft=n_fft,
                spectral_method=spectral_method,
                time_bandwidth=time_bandwidth,
                n_tapers=n_tapers,
                window=window,
                detrend=detrend,
            )

        if len(feature_regularization_values) == 1:
            if spectral_cache is None:
                self._fit(
                    x_trials,
                    y_trials,
                    fs=fs,
                    tmin=tmin,
                    tmax=tmax,
                    regularization=regularization_specs[0],
                    feature_regularization=feature_regularization_values[0],
                    bands=resolved_bands,
                    segment_length=segment_length,
                    overlap=overlap,
                    n_fft=n_fft,
                    spectral_method=spectral_method,
                    time_bandwidth=time_bandwidth,
                    n_tapers=n_tapers,
                    window=window,
                    detrend=detrend,
                    trial_weights=trial_weights,
                )
            else:
                self._fit_from_cache(
                    spectral_cache,
                    fs=fs,
                    tmin=tmin,
                    tmax=tmax,
                    regularization=regularization_specs[0],
                    feature_regularization=feature_regularization_values[0],
                    bands=resolved_bands,
                    raw_trial_weights=raw_trial_weights,
                )
                if bootstrap_samples > 0:
                    self.bootstrap_interval, _ = _compute_bootstrap_interval_from_cache(
                        spectral_cache,
                        fs=fs,
                        tmin=tmin,
                        tmax=tmax,
                        feature_regularization=feature_regularization_values[0],
                        raw_trial_weights=raw_trial_weights,
                        n_bootstraps=int(bootstrap_samples),
                        level=float(bootstrap_level),
                        seed=bootstrap_seed,
                    )
                    self.bootstrap_level = float(bootstrap_level)
                    self.bootstrap_samples = int(bootstrap_samples)
            return None

        cv_scores = self._cross_validate(
            x_trials,
            y_trials,
            fs=fs,
            tmin=tmin,
            tmax=tmax,
            feature_regularization_values=feature_regularization_values,
            segment_length=segment_length,
            overlap=overlap,
            n_fft=n_fft,
            spectral_method=spectral_method,
            time_bandwidth=time_bandwidth,
            n_tapers=n_tapers,
            window=window,
            detrend=detrend,
            k=k,
            seed=seed,
            average=average,
            raw_trial_weights=raw_trial_weights,
            spectral_cache=spectral_cache,
        )

        if cv_scores.ndim == 1:
            best_index = int(np.argmax(cv_scores))
        else:
            best_index = int(np.argmax(cv_scores.mean(axis=1)))

        assert spectral_cache is not None
        self._fit_from_cache(
            spectral_cache,
            fs=fs,
            tmin=tmin,
            tmax=tmax,
            regularization=regularization_specs[best_index],
            feature_regularization=feature_regularization_values[best_index],
            bands=resolved_bands,
            raw_trial_weights=raw_trial_weights,
        )
        if bootstrap_samples > 0:
            self.bootstrap_interval, _ = _compute_bootstrap_interval_from_cache(
                spectral_cache,
                fs=fs,
                tmin=tmin,
                tmax=tmax,
                feature_regularization=feature_regularization_values[best_index],
                raw_trial_weights=raw_trial_weights,
                n_bootstraps=int(bootstrap_samples),
                level=float(bootstrap_level),
                seed=bootstrap_seed,
            )
            self.bootstrap_level = float(bootstrap_level)
            self.bootstrap_samples = int(bootstrap_samples)
        return cv_scores

    def train_multitaper(
        self,
        stimulus: np.ndarray | Sequence[np.ndarray],
        response: np.ndarray | Sequence[np.ndarray],
        fs: float,
        tmin: float,
        tmax: float,
        regularization: float | Sequence[float] | Sequence[Sequence[float]],
        *,
        bands: None | Sequence[int] = None,
        segment_length: int | None = None,
        overlap: float = 0.0,
        n_fft: int | None = None,
        time_bandwidth: float = 3.5,
        n_tapers: int | None = None,
        detrend: None | str = "constant",
        k: int = -1,
        average: bool | Sequence[int] = True,
        seed: int | None = None,
        trial_weights: None | str | Sequence[float] = None,
        bootstrap_samples: int = 0,
        bootstrap_level: float = 0.95,
        bootstrap_seed: int | None = None,
    ) -> np.ndarray | float | None:
        """Fit the model with DPSS multi-taper spectral estimation.

        This is a convenience wrapper around :meth:`train` for users who want
        a named multi-taper estimation path without manually setting
        ``spectral_method="multitaper"``.
        """

        return self.train(
            stimulus=stimulus,
            response=response,
            fs=fs,
            tmin=tmin,
            tmax=tmax,
            regularization=regularization,
            bands=bands,
            segment_length=segment_length,
            overlap=overlap,
            n_fft=n_fft,
            spectral_method="multitaper",
            time_bandwidth=time_bandwidth,
            n_tapers=n_tapers,
            window=None,
            detrend=detrend,
            k=k,
            average=average,
            seed=seed,
            trial_weights=trial_weights,
            bootstrap_samples=bootstrap_samples,
            bootstrap_level=bootstrap_level,
            bootstrap_seed=bootstrap_seed,
        )

    def _fit(
        self,
        x_trials: Sequence[np.ndarray],
        y_trials: Sequence[np.ndarray],
        *,
        fs: float,
        tmin: float,
        tmax: float,
        regularization: RegularizationSpec,
        feature_regularization: np.ndarray,
        bands: tuple[int, ...] | None,
        segment_length: int | None,
        overlap: float,
        n_fft: int | None,
        spectral_method: SpectralMethod,
        time_bandwidth: float,
        n_tapers: int | None,
        window: None | str | tuple[str, float] | np.ndarray,
        detrend: None | str,
        trial_weights: None | str | Sequence[float],
    ) -> None:
        spectral_cache = _build_spectral_cache(
            x_trials,
            y_trials,
            segment_length=segment_length,
            overlap=overlap,
            n_fft=n_fft,
            spectral_method=spectral_method,
            time_bandwidth=time_bandwidth,
            n_tapers=n_tapers,
            window=window,
            detrend=detrend,
        )
        self._fit_from_cache(
            spectral_cache,
            fs=fs,
            tmin=tmin,
            tmax=tmax,
            regularization=regularization,
            feature_regularization=feature_regularization,
            bands=bands,
            raw_trial_weights=_resolve_raw_trial_weights(y_trials, trial_weights),
        )
        self.window = _copy_value(window)
        self.detrend = detrend

    def _fit_from_cache(
        self,
        spectral_cache: _SpectralCache,
        *,
        fs: float,
        tmin: float,
        tmax: float,
        regularization: RegularizationSpec,
        feature_regularization: np.ndarray,
        bands: tuple[int, ...] | None,
        raw_trial_weights: np.ndarray | None,
        trial_indices: np.ndarray | None = None,
    ) -> None:
        cxx, cxy = _aggregate_cached_spectra(
            spectral_cache,
            trial_indices=trial_indices,
            raw_trial_weights=raw_trial_weights,
        )
        transfer_function = _solve_transfer_function(
            cxx,
            cxy,
            feature_regularization=feature_regularization,
        )

        self.transfer_function = transfer_function
        self.frequencies = np.fft.rfftfreq(spectral_cache.n_fft, d=1.0 / float(fs))
        self.fs = float(fs)
        self.regularization = _copy_value(regularization)
        self.bands = _copy_value(bands)
        self.feature_regularization = np.asarray(feature_regularization, dtype=float).copy()
        self.segment_length = spectral_cache.segment_length
        self.n_fft = spectral_cache.n_fft
        self.overlap = spectral_cache.overlap
        self.spectral_method = spectral_cache.spectral_method
        self.time_bandwidth = spectral_cache.time_bandwidth
        self.n_tapers = spectral_cache.n_tapers
        self.window = _copy_value(spectral_cache.window)
        self.detrend = spectral_cache.detrend
        self.tmin = float(tmin)
        self.tmax = float(tmax)
        self.weights, self.times = _extract_impulse_response(
            self.transfer_function,
            fs=float(fs),
            n_fft=spectral_cache.n_fft,
            tmin=float(tmin),
            tmax=float(tmax),
        )

    def to_impulse_response(
        self,
        tmin: float | None = None,
        tmax: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract a time-domain kernel from the fitted transfer function.

        Parameters
        ----------
        tmin, tmax:
            Optional lag window in seconds. If omitted, the window used during
            :meth:`train` is reused.

        Returns
        -------
        weights, times:
            ``weights`` has shape ``(n_inputs, n_lags, n_outputs)`` and ``times``
            contains the corresponding lag values in seconds.

        Notes
        -----
        This method is useful when you want to inspect a different lag window
        without refitting the spectral model.
        """

        if self.transfer_function is None or self.fs is None or self.n_fft is None:
            raise ValueError("Model must be trained before extracting an impulse response.")

        tmin = self.tmin if tmin is None else tmin
        tmax = self.tmax if tmax is None else tmax
        if tmin is None or tmax is None:
            raise ValueError("tmin and tmax must be defined.")

        return _extract_impulse_response(
            self.transfer_function,
            fs=float(self.fs),
            n_fft=int(self.n_fft),
            tmin=float(tmin),
            tmax=float(tmax),
        )

    def plot(
        self,
        *,
        input_index: int = 0,
        output_index: int = 0,
        tmin: float | None = None,
        tmax: float | None = None,
        ax=None,
        time_unit: str = "ms",
        color: str | None = None,
        linewidth: float = 2.0,
        show_bootstrap_interval: bool = False,
        interval_color: str | None = None,
        interval_alpha: float = 0.2,
        title: str | None = None,
        label: str | None = None,
    ):
        """Plot one fitted time-domain kernel.

        Parameters
        ----------
        input_index, output_index:
            Select which input/output pair should be plotted.
        tmin, tmax:
            Optional lag window to visualize. If omitted, the fitted window is
            used.
        ax:
            Existing matplotlib axes. When omitted, a new figure is created.
        time_unit:
            Either ``"ms"`` or ``"s"`` for the x-axis.
        color, linewidth, title, label:
            Standard matplotlib styling arguments for the kernel line.
        show_bootstrap_interval:
            If ``True``, plot the stored bootstrap confidence interval.
        interval_color, interval_alpha:
            Styling for the bootstrap interval shading.

        Returns
        -------
        fig, ax:
            The matplotlib figure and axes containing the plot.
        """

        if self.weights is None or self.times is None:
            raise ValueError("Model must be trained before plotting.")

        from .plotting import plot_kernel

        weights, times = self.to_impulse_response(tmin=tmin, tmax=tmax)
        bootstrap_interval = None
        if show_bootstrap_interval:
            bootstrap_interval, _ = self.bootstrap_interval_at(tmin=tmin, tmax=tmax)
        return plot_kernel(
            weights=weights,
            times=times,
            credible_interval=bootstrap_interval,
            input_index=input_index,
            output_index=output_index,
            ax=ax,
            time_unit=time_unit,
            color=color,
            interval_color=interval_color,
            linewidth=linewidth,
            interval_alpha=interval_alpha,
            title=title,
            label=label,
        )

    def plot_grid(
        self,
        *,
        tmin: float | None = None,
        tmax: float | None = None,
        ax=None,
        time_unit: str = "ms",
        color: str | None = None,
        linewidth: float = 1.8,
        show_bootstrap_interval: bool = False,
        interval_color: str | None = None,
        interval_alpha: float = 0.2,
        input_labels: Sequence[str] | None = None,
        output_labels: Sequence[str] | None = None,
        title: str | None = None,
        sharey: bool = False,
    ):
        """Plot every input/output kernel in a grid.

        This is convenient for multifeature or multichannel models where
        calling :meth:`plot` repeatedly would be cumbersome.
        """

        if self.weights is None or self.times is None:
            raise ValueError("Model must be trained before plotting.")

        from .plotting import plot_kernel_grid

        weights, times = self.to_impulse_response(tmin=tmin, tmax=tmax)
        bootstrap_interval = None
        if show_bootstrap_interval:
            bootstrap_interval, _ = self.bootstrap_interval_at(tmin=tmin, tmax=tmax)
        return plot_kernel_grid(
            weights=weights,
            times=times,
            credible_interval=bootstrap_interval,
            ax=ax,
            time_unit=time_unit,
            color=color,
            interval_color=interval_color,
            linewidth=linewidth,
            interval_alpha=interval_alpha,
            title=title,
            input_labels=input_labels,
            output_labels=output_labels,
            sharey=sharey,
        )

    def transfer_function_at(
        self,
        *,
        input_index: int = 0,
        output_index: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return one complex-valued transfer function slice.

        Parameters
        ----------
        input_index, output_index:
            Select the predictor-target pair to inspect.

        Returns
        -------
        frequencies, transfer_function:
            Frequency vector in Hz and the matching complex transfer-function
            values for the selected input/output pair.
        """

        if self.transfer_function is None or self.frequencies is None:
            raise ValueError("Model must be trained before extracting a transfer function.")
        if not 0 <= int(input_index) < self.transfer_function.shape[1]:
            raise IndexError(f"input_index out of bounds: {input_index}")
        if not 0 <= int(output_index) < self.transfer_function.shape[2]:
            raise IndexError(f"output_index out of bounds: {output_index}")
        return (
            self.frequencies.copy(),
            self.transfer_function[:, int(input_index), int(output_index)].copy(),
        )

    def transfer_function_components_at(
        self,
        *,
        input_index: int = 0,
        output_index: int = 0,
        phase_unit: str = "rad",
    ) -> TransferFunctionComponents:
        """Return magnitude, phase, and group delay for one transfer function."""

        frequencies, transfer = self.transfer_function_at(
            input_index=input_index,
            output_index=output_index,
        )
        phase, resolved_phase_unit = _phase_values(
            transfer,
            phase_unit=phase_unit,
        )
        return TransferFunctionComponents(
            frequencies=frequencies,
            transfer_function=transfer,
            magnitude=np.abs(transfer),
            phase=phase,
            phase_unit=resolved_phase_unit,
            group_delay=_group_delay_values(frequencies, transfer),
        )

    def plot_transfer_function(
        self,
        *,
        input_index: int = 0,
        output_index: int = 0,
        kind: str = "both",
        ax=None,
        color: str | None = None,
        phase_color: str | None = None,
        group_delay_color: str | None = None,
        linewidth: float = 2.0,
        phase_unit: str = "rad",
        group_delay_unit: str = "ms",
        title: str | None = None,
    ):
        """Plot magnitude, phase, and/or group delay of one transfer function."""

        from .plotting import plot_transfer_function

        components = self.transfer_function_components_at(
            input_index=input_index,
            output_index=output_index,
            phase_unit=phase_unit,
        )
        return plot_transfer_function(
            frequencies=components.frequencies,
            transfer_function=components.transfer_function,
            kind=kind,
            ax=ax,
            color=color,
            phase_color=phase_color,
            group_delay_color=group_delay_color,
            linewidth=linewidth,
            phase_unit=components.phase_unit,
            group_delay_unit=group_delay_unit,
            title=title,
        )

    def cross_spectral_diagnostics(
        self,
        *,
        stimulus: np.ndarray | Sequence[np.ndarray] | None = None,
        response: np.ndarray | Sequence[np.ndarray] | None = None,
        tmin: float | None = None,
        tmax: float | None = None,
        trial_weights: None | str | Sequence[float] | object = _USE_STORED_TRIAL_WEIGHTS,
    ) -> FrequencyTRFDiagnostics:
        """Compute observed-vs-predicted cross-spectral diagnostics.

        This method reuses the fitted kernel to generate predictions for the
        provided data, then compares predicted and observed targets in the
        frequency domain. The returned diagnostics include:

        - the learned complex transfer function
        - predicted and observed output spectra
        - matched predicted-vs-observed cross-spectra
        - magnitude-squared coherence between prediction and target
        """

        if self.transfer_function is None or self.frequencies is None:
            raise ValueError("Model must be trained before computing diagnostics.")
        if self.segment_length is None or self.overlap is None or self.n_fft is None:
            raise ValueError("Stored spectral settings are unavailable on this model.")

        predictor_input = stimulus if self.direction == 1 else response
        target_input = response if self.direction == 1 else stimulus
        predictor_name = "stimulus" if self.direction == 1 else "response"
        target_name = "response" if self.direction == 1 else "stimulus"
        if predictor_input is None or target_input is None:
            raise ValueError(
                f"{predictor_name} and {target_name} are both required for diagnostics."
            )

        predictor_trials, _ = _coerce_trials(predictor_input, predictor_name)
        target_trials, _ = _coerce_trials(target_input, target_name)
        _check_trial_lengths(predictor_trials, target_trials)

        predictions = self.predict(
            stimulus=stimulus,
            response=response,
            tmin=tmin,
            tmax=tmax,
        )
        if isinstance(predictions, tuple):
            prediction_trials_raw = predictions[0]
        else:
            prediction_trials_raw = predictions
        prediction_trials, _ = _coerce_trials(
            prediction_trials_raw,
            "prediction",
        )

        resolved_trial_weights = (
            self.trial_weights if trial_weights is _USE_STORED_TRIAL_WEIGHTS else trial_weights
        )
        raw_trial_weights = _resolve_raw_trial_weights(target_trials, resolved_trial_weights)
        prediction_cache = _build_spectral_cache(
            prediction_trials,
            target_trials,
            segment_length=self.segment_length,
            overlap=float(self.overlap),
            n_fft=self.n_fft,
            spectral_method=self.spectral_method,
            time_bandwidth=float(self.time_bandwidth or 3.5),
            n_tapers=self.n_tapers,
            window=self.window,
            detrend=self.detrend,
        )
        observed_cache = _build_spectral_cache(
            target_trials,
            target_trials,
            segment_length=self.segment_length,
            overlap=float(self.overlap),
            n_fft=self.n_fft,
            spectral_method=self.spectral_method,
            time_bandwidth=float(self.time_bandwidth or 3.5),
            n_tapers=self.n_tapers,
            window=self.window,
            detrend=self.detrend,
        )
        predicted_auto, predicted_vs_observed = _aggregate_cached_spectra(
            prediction_cache,
            raw_trial_weights=raw_trial_weights,
        )
        observed_auto, _ = _aggregate_cached_spectra(
            observed_cache,
            raw_trial_weights=raw_trial_weights,
        )

        predicted_spectrum = np.real(np.diagonal(predicted_auto, axis1=1, axis2=2))
        observed_spectrum = np.real(np.diagonal(observed_auto, axis1=1, axis2=2))
        cross_spectrum = np.diagonal(predicted_vs_observed, axis1=1, axis2=2)
        denominator = np.clip(
            predicted_spectrum * observed_spectrum,
            np.finfo(float).eps,
            None,
        )
        coherence = np.clip((np.abs(cross_spectrum) ** 2) / denominator, 0.0, 1.0)

        return FrequencyTRFDiagnostics(
            frequencies=self.frequencies.copy(),
            transfer_function=self.transfer_function.copy(),
            predicted_spectrum=predicted_spectrum,
            observed_spectrum=observed_spectrum,
            cross_spectrum=cross_spectrum,
            coherence=coherence,
        )

    def diagnostics(
        self,
        *,
        stimulus: np.ndarray | Sequence[np.ndarray] | None = None,
        response: np.ndarray | Sequence[np.ndarray] | None = None,
        tmin: float | None = None,
        tmax: float | None = None,
        trial_weights: None | str | Sequence[float] | object = _USE_STORED_TRIAL_WEIGHTS,
    ) -> FrequencyTRFDiagnostics:
        """Compatibility alias for :meth:`cross_spectral_diagnostics`."""

        return self.cross_spectral_diagnostics(
            stimulus=stimulus,
            response=response,
            tmin=tmin,
            tmax=tmax,
            trial_weights=trial_weights,
        )

    def plot_coherence(
        self,
        *,
        stimulus: np.ndarray | Sequence[np.ndarray] | None = None,
        response: np.ndarray | Sequence[np.ndarray] | None = None,
        diagnostics: FrequencyTRFDiagnostics | None = None,
        output_index: int = 0,
        ax=None,
        color: str | None = None,
        linewidth: float = 2.0,
        title: str | None = None,
    ):
        """Plot magnitude-squared coherence between predictions and targets."""

        from .plotting import plot_coherence

        if diagnostics is None:
            diagnostics = self.cross_spectral_diagnostics(
                stimulus=stimulus,
                response=response,
            )
        return plot_coherence(
            frequencies=diagnostics.frequencies,
            coherence=diagnostics.coherence,
            output_index=output_index,
            ax=ax,
            color=color,
            linewidth=linewidth,
            title=title,
        )

    def plot_cross_spectrum(
        self,
        *,
        stimulus: np.ndarray | Sequence[np.ndarray] | None = None,
        response: np.ndarray | Sequence[np.ndarray] | None = None,
        diagnostics: FrequencyTRFDiagnostics | None = None,
        output_index: int = 0,
        kind: str = "both",
        ax=None,
        color: str | None = None,
        phase_color: str | None = None,
        linewidth: float = 2.0,
        phase_unit: str = "rad",
        title: str | None = None,
    ):
        """Plot the predicted-vs-observed cross spectrum for one output."""

        from .plotting import plot_cross_spectrum

        if diagnostics is None:
            diagnostics = self.cross_spectral_diagnostics(
                stimulus=stimulus,
                response=response,
            )
        return plot_cross_spectrum(
            frequencies=diagnostics.frequencies,
            cross_spectrum=diagnostics.cross_spectrum,
            output_index=output_index,
            kind=kind,
            ax=ax,
            color=color,
            phase_color=phase_color,
            linewidth=linewidth,
            phase_unit=phase_unit,
            title=title,
        )

    def bootstrap_interval_at(
        self,
        *,
        tmin: float | None = None,
        tmax: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the stored bootstrap interval over the requested lag window."""

        if self.bootstrap_interval is None or self.times is None:
            raise ValueError("No bootstrap interval is stored on this model.")
        return _slice_interval(
            self.bootstrap_interval,
            self.times,
            tmin=tmin,
            tmax=tmax,
        )

    def bootstrap_confidence_interval(
        self,
        stimulus: np.ndarray | Sequence[np.ndarray],
        response: np.ndarray | Sequence[np.ndarray],
        *,
        n_bootstraps: int = 200,
        level: float = 0.95,
        seed: int | None = None,
        trial_weights: None | str | Sequence[float] | object = _USE_STORED_TRIAL_WEIGHTS,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Estimate and store a trial-bootstrap confidence interval.

        The estimator must already be fitted. By default the method reuses the
        same fit settings and the same trial-weighting strategy as the model.
        Bootstrap resampling is performed over trials, so at least two trials
        are required.
        """

        if self.regularization is None or self.fs is None or self.tmin is None or self.tmax is None:
            raise ValueError("Model must be trained before bootstrapping.")
        if self.feature_regularization is None:
            raise ValueError("Stored feature regularization is unavailable on this model.")

        stimulus_trials, _ = _coerce_trials(stimulus, "stimulus")
        response_trials, _ = _coerce_trials(response, "response")
        _check_trial_lengths(stimulus_trials, response_trials)
        x_trials, y_trials = self._get_xy(stimulus_trials, response_trials)
        self._validate_dimensions(x_trials, y_trials)

        resolved_trial_weights = self.trial_weights if trial_weights is _USE_STORED_TRIAL_WEIGHTS else trial_weights
        spectral_cache = _build_spectral_cache(
            x_trials,
            y_trials,
            segment_length=self.segment_length,
            overlap=float(self.overlap),
            n_fft=self.n_fft,
            spectral_method=self.spectral_method,
            time_bandwidth=float(self.time_bandwidth or 3.5),
            n_tapers=self.n_tapers,
            window=self.window,
            detrend=self.detrend,
        )
        interval, times = _compute_bootstrap_interval_from_cache(
            spectral_cache,
            fs=float(self.fs),
            tmin=float(self.tmin),
            tmax=float(self.tmax),
            feature_regularization=self.feature_regularization,
            raw_trial_weights=_resolve_raw_trial_weights(y_trials, resolved_trial_weights),
            n_bootstraps=int(n_bootstraps),
            level=float(level),
            seed=seed,
        )
        self.bootstrap_interval = interval
        self.bootstrap_level = float(level)
        self.bootstrap_samples = int(n_bootstraps)
        return interval, times

    def predict(
        self,
        stimulus: np.ndarray | Sequence[np.ndarray] | None = None,
        response: np.ndarray | Sequence[np.ndarray] | None = None,
        *,
        average: bool | Sequence[int] = True,
        tmin: float | None = None,
        tmax: float | None = None,
    ) -> list[np.ndarray] | np.ndarray | tuple[list[np.ndarray] | np.ndarray, np.ndarray | float]:
        """Generate predictions from a fitted model.

        Parameters
        ----------
        stimulus, response:
            Inputs follow the same single-trial / multi-trial conventions as
            :meth:`train`. For forward models, ``stimulus`` is required. For
            backward models, ``response`` is required. If the corresponding
            observed target is also provided, the method additionally returns a
            prediction score.
        average:
            Reduction strategy for the returned score. ``True`` averages over
            outputs, ``False`` returns one score per output, and a sequence of
            indices averages only over selected outputs.
        tmin, tmax:
            Optional lag window used during prediction. If omitted, the fitted
            lag window is used.

        Returns
        -------
        prediction or (prediction, metric):
            Predicted trials are returned in the same single-trial vs list form
            as the predictor input. When observed targets are supplied, the
            method also returns the metric defined on the estimator.

        Notes
        -----
        Prediction is performed by convolving the predictor with the extracted
        time-domain kernel over the requested lag window.
        """

        if self.weights is None or self.fs is None:
            raise ValueError("Model must be trained before prediction.")

        if self.direction == 1 and stimulus is None:
            raise ValueError("stimulus is required for a forward model.")
        if self.direction == -1 and response is None:
            raise ValueError("response is required for a backward model.")

        predictor_input = stimulus if self.direction == 1 else response
        target_input = response if self.direction == 1 else stimulus
        predictor_name = "stimulus" if self.direction == 1 else "response"
        target_name = "response" if self.direction == 1 else "stimulus"

        predictor_trials, predictor_is_single = _coerce_trials(predictor_input, predictor_name)
        target_trials = None
        if target_input is not None:
            target_trials, _ = _coerce_trials(target_input, target_name)
            _check_trial_lengths(predictor_trials, target_trials)

        n_inputs = self.weights.shape[0]
        n_outputs = self.weights.shape[-1]
        for predictor_trial in predictor_trials:
            if predictor_trial.shape[1] != n_inputs:
                raise ValueError(
                    f"Expected {n_inputs} predictor channels/features, got {predictor_trial.shape[1]}."
                )
        if target_trials is not None:
            for target_trial in target_trials:
                if target_trial.shape[1] != n_outputs:
                    raise ValueError(
                        f"Expected {n_outputs} target channels/features, got {target_trial.shape[1]}."
                    )

        weights, times = self.to_impulse_response(tmin=tmin, tmax=tmax)
        lag_start = int(round(times[0] * self.fs))
        kernel = np.transpose(weights, (1, 0, 2))

        predictions: list[np.ndarray] = []
        metrics = []
        for index, predictor_trial in enumerate(predictor_trials):
            prediction = np.zeros((predictor_trial.shape[0], n_outputs), dtype=float)
            for input_index in range(n_inputs):
                for output_index in range(n_outputs):
                    prediction[:, output_index] += _shifted_convolution(
                        predictor_trial[:, input_index],
                        kernel[:, input_index, output_index],
                        lag_start=lag_start,
                        out_length=predictor_trial.shape[0],
                    )
            predictions.append(prediction)
            if target_trials is not None:
                metrics.append(self.metric(target_trials[index], prediction))

        returned_predictions: list[np.ndarray] | np.ndarray
        if predictor_is_single:
            returned_predictions = predictions[0]
        else:
            returned_predictions = predictions

        if target_trials is None:
            return returned_predictions

        metric = np.mean(np.vstack(metrics), axis=0)
        return returned_predictions, _aggregate_metric(metric, average)

    def score(
        self,
        stimulus: np.ndarray | Sequence[np.ndarray] | None = None,
        response: np.ndarray | Sequence[np.ndarray] | None = None,
        *,
        average: bool | Sequence[int] = True,
        tmin: float | None = None,
        tmax: float | None = None,
    ) -> np.ndarray | float:
        """Score predictions without returning the predicted signals.

        This is a convenience wrapper around :meth:`predict` for workflows where
        only the metric is needed.
        """

        _, metric = self.predict(
            stimulus=stimulus,
            response=response,
            average=average,
            tmin=tmin,
            tmax=tmax,
        )
        return metric

    def _cross_validate(
        self,
        x_trials: Sequence[np.ndarray],
        y_trials: Sequence[np.ndarray],
        *,
        fs: float,
        tmin: float,
        tmax: float,
        feature_regularization_values: Sequence[np.ndarray],
        segment_length: int | None,
        overlap: float,
        n_fft: int | None,
        spectral_method: SpectralMethod,
        time_bandwidth: float,
        n_tapers: int | None,
        window: None | str | tuple[str, float] | np.ndarray,
        detrend: None | str,
        k: int,
        seed: int | None,
        average: bool | Sequence[int],
        raw_trial_weights: np.ndarray,
        spectral_cache: _SpectralCache | None,
    ) -> np.ndarray:
        n_trials = len(x_trials)
        if n_trials < 2:
            raise ValueError("Cross-validation needs at least two trials.")

        n_folds = n_trials if k == -1 else int(k)
        if n_folds < 2:
            raise ValueError("k must be -1 or at least 2.")
        n_folds = min(n_folds, n_trials)

        indices = np.arange(n_trials)
        if seed is not None:
            rng = np.random.default_rng(seed)
            rng.shuffle(indices)
        folds = [fold for fold in np.array_split(indices, n_folds) if len(fold) > 0]

        per_reg_scores = np.zeros(
            (len(feature_regularization_values), len(folds), y_trials[0].shape[1]),
            dtype=float,
        )
        if spectral_cache is None:
            spectral_cache = _build_spectral_cache(
                x_trials,
                y_trials,
                segment_length=segment_length,
                overlap=overlap,
                n_fft=n_fft,
                spectral_method=spectral_method,
                time_bandwidth=time_bandwidth,
                n_tapers=n_tapers,
                window=window,
                detrend=detrend,
            )

        candidate_specs = self.regularization_candidates
        if candidate_specs is None or len(candidate_specs) != len(feature_regularization_values):
            raise RuntimeError("Regularization candidates are inconsistent with the CV search grid.")

        for reg_index, feature_regularization in enumerate(feature_regularization_values):
            for fold_index, val_idx in enumerate(folds):
                train_idx = np.setdiff1d(indices, val_idx, assume_unique=True)
                candidate = FrequencyTRF(direction=self.direction, metric=self.metric)
                candidate._fit_from_cache(
                    spectral_cache,
                    fs=fs,
                    tmin=tmin,
                    tmax=tmax,
                    regularization=candidate_specs[reg_index],
                    feature_regularization=feature_regularization,
                    bands=self.bands,
                    raw_trial_weights=raw_trial_weights,
                    trial_indices=train_idx,
                )
                per_reg_scores[reg_index, fold_index, :] = candidate.score(
                    stimulus=[x_trials[i] for i in val_idx] if self.direction == 1 else [y_trials[i] for i in val_idx],
                    response=[y_trials[i] for i in val_idx] if self.direction == 1 else [x_trials[i] for i in val_idx],
                    average=False,
                    tmin=tmin,
                    tmax=tmax,
                )

        fold_mean = per_reg_scores.mean(axis=1)
        if average is False:
            return fold_mean
        if average is True:
            return fold_mean.mean(axis=1)
        return fold_mean[:, np.asarray(list(average), dtype=int)].mean(axis=1)

    def save(self, path: str | Path) -> None:
        """Serialize the fitted estimator to disk using :mod:`pickle`."""
        path = Path(path)
        if not path.parent.exists():
            raise FileNotFoundError(f"Directory does not exist: {path.parent}")
        with path.open("wb") as handle:
            pickle.dump(self, handle, pickle.HIGHEST_PROTOCOL)

    def load(self, path: str | Path) -> None:
        """Load estimator state from a pickle file into this instance."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File does not exist: {path}")
        with path.open("rb") as handle:
            loaded = pickle.load(handle)
        self.__dict__ = loaded.__dict__
        if not hasattr(self, "metric_name"):
            self.metric_name = getattr(self.metric, "__name__", None)
        if not hasattr(self, "bands"):
            self.bands = None
        if not hasattr(self, "feature_regularization") or self.feature_regularization is None:
            if self.weights is not None and isinstance(self.regularization, (float, np.floating)):
                self.feature_regularization = np.full(
                    self.weights.shape[0],
                    float(self.regularization),
                    dtype=float,
                )
            else:
                self.feature_regularization = None
        if not hasattr(self, "regularization_candidates"):
            self.regularization_candidates = (
                None if self.regularization is None else [_copy_value(self.regularization)]
            )
        if not hasattr(self, "spectral_method"):
            self.spectral_method = "standard"
        if not hasattr(self, "time_bandwidth"):
            self.time_bandwidth = None
        if not hasattr(self, "n_tapers"):
            self.n_tapers = None

    def copy(self) -> "FrequencyTRF":
        """Return a copy of the estimator and all learned arrays."""

        copied = FrequencyTRF(direction=self.direction, metric=self.metric)
        for key, value in self.__dict__.items():
            setattr(copied, key, _copy_value(value))
        return copied

    def _get_xy(
        self,
        stimulus_trials: Sequence[np.ndarray],
        response_trials: Sequence[np.ndarray],
    ) -> tuple[Sequence[np.ndarray], Sequence[np.ndarray]]:
        if self.direction == 1:
            return stimulus_trials, response_trials
        return response_trials, stimulus_trials

    @staticmethod
    def _validate_dimensions(
        x_trials: Sequence[np.ndarray],
        y_trials: Sequence[np.ndarray],
    ) -> None:
        n_inputs = x_trials[0].shape[1]
        n_outputs = y_trials[0].shape[1]
        for trial in x_trials:
            if trial.shape[1] != n_inputs:
                raise ValueError("All predictor trials must have the same feature count.")
        for trial in y_trials:
            if trial.shape[1] != n_outputs:
                raise ValueError("All response trials must have the same channel count.")

    @staticmethod
    def _validate_fit_arguments(
        fs: float,
        tmin: float,
        tmax: float,
        segment_length: int | None,
        overlap: float,
        n_fft: int | None,
        spectral_method: SpectralMethod,
        time_bandwidth: float,
        n_tapers: int | None,
        window: None | str | tuple[str, float] | np.ndarray,
        detrend: None | str,
    ) -> None:
        if fs <= 0:
            raise ValueError("fs must be positive.")
        if tmax <= tmin:
            raise ValueError("tmax must be greater than tmin.")
        if segment_length is not None and int(segment_length) <= 0:
            raise ValueError("segment_length must be positive.")
        if not 0.0 <= overlap < 1.0:
            raise ValueError("overlap must lie in [0, 1).")
        if n_fft is not None and int(n_fft) <= 0:
            raise ValueError("n_fft must be positive.")
        resolved_method = _validate_spectral_method(spectral_method)
        if resolved_method == "multitaper":
            _resolve_multitaper_parameters(
                time_bandwidth=float(time_bandwidth),
                n_tapers=n_tapers,
            )
            if window is not None:
                raise ValueError("window must be None when spectral_method='multitaper'.")
        if detrend not in (None, "constant", "linear"):
            raise ValueError("detrend must be None, 'constant', or 'linear'.")
