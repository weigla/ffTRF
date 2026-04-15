
"""Validation, coercion, and small utility helpers for ffTRF."""

from __future__ import annotations

from itertools import product
import os
import sys
from threading import Lock
from typing import Sequence
import warnings

import numpy as np

RegularizationSpec = float | tuple[float, ...]
_NICE_SEGMENT_DURATIONS = np.asarray([0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0], dtype=float)


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


def _resolve_segment_length(
    *,
    fs: float,
    segment_length: int | None,
    segment_duration: float | None,
) -> int | None:
    if segment_length is not None and segment_duration is not None:
        raise ValueError("Provide either segment_length or segment_duration, not both.")
    if segment_duration is None:
        return None if segment_length is None else int(segment_length)

    duration = float(segment_duration)
    if not np.isfinite(duration) or duration <= 0.0:
        raise ValueError("segment_duration must be finite and positive.")
    return max(1, int(round(duration * float(fs))))


def suggest_segment_settings(
    *,
    fs: float,
    tmin: float,
    tmax: float,
    trial_duration: float | None = None,
) -> dict[str, int | float | str | None]:
    """Suggest practical segment settings for the standard spectral estimator.

    This helper is meant as a lightweight starting point for choosing
    ``segment_length`` / ``segment_duration`` and ``window`` when you use the
    default ``spectral_method="standard"`` workflow. The heuristic is simple:

    - keep the segment clearly longer than the lag window
    - prefer full-trial spectra when trials are already short
    - otherwise default to overlapping Hann-windowed segments

    Parameters
    ----------
    fs:
        Sampling rate in Hz.
    tmin, tmax:
        Lag window, in seconds, that you plan to extract from the fitted TRF.
    trial_duration:
        Optional approximate trial duration in seconds. When supplied, the
        helper can decide whether a full-trial spectrum is more sensible than
        splitting each trial into shorter overlapping segments.

    Returns
    -------
    dict
        Dictionary with the keys ``segment_length``, ``segment_duration``,
        ``overlap``, and ``window``. When a full-trial estimate is suggested,
        both segment fields are ``None``, ``overlap`` is ``0.0``, and
        ``window`` is ``None``.

    Notes
    -----
    The returned values are intended as rule-of-thumb defaults, not as an
    optimizer. If you later switch to ``train_multitaper(...)`` or
    ``spectral_method="multitaper"``, you can usually keep the suggested
    ``segment_duration`` and ``overlap`` but should set ``window=None``.
    """

    sampling_rate = float(fs)
    if not np.isfinite(sampling_rate) or sampling_rate <= 0.0:
        raise ValueError("fs must be finite and positive.")

    lag_start = float(tmin)
    lag_stop = float(tmax)
    if not np.isfinite(lag_start) or not np.isfinite(lag_stop) or lag_stop <= lag_start:
        raise ValueError("tmax must be finite and greater than tmin.")

    lag_span = lag_stop - lag_start

    resolved_trial_duration = None
    if trial_duration is not None:
        resolved_trial_duration = float(trial_duration)
        if not np.isfinite(resolved_trial_duration) or resolved_trial_duration <= 0.0:
            raise ValueError("trial_duration must be finite and positive when provided.")

    # Very short trials are usually better treated as one segment instead of
    # forcing a tiny Hann-windowed FFT estimate.
    if resolved_trial_duration is not None and resolved_trial_duration <= max(1.0, 3.0 * lag_span):
        return {
            "segment_length": None,
            "segment_duration": None,
            "overlap": 0.0,
            "window": None,
        }

    target_duration = max(1.0, 6.0 * lag_span)
    if resolved_trial_duration is not None:
        target_duration = min(target_duration, 0.5 * resolved_trial_duration)

    candidate_durations = _NICE_SEGMENT_DURATIONS
    if resolved_trial_duration is not None:
        candidate_durations = candidate_durations[candidate_durations <= 0.5 * resolved_trial_duration + 1e-12]

    if candidate_durations.size == 0:
        return {
            "segment_length": None,
            "segment_duration": None,
            "overlap": 0.0,
            "window": None,
        }

    segment_duration_value = float(candidate_durations[np.argmin(np.abs(candidate_durations - target_duration))])
    segment_length_value = max(1, int(round(segment_duration_value * sampling_rate)))
    return {
        "segment_length": segment_length_value,
        "segment_duration": segment_duration_value,
        "overlap": 0.5,
        "window": "hann",
    }


def _resolve_k_folds(k: int | str) -> int:
    if isinstance(k, str):
        resolved = k.strip().lower()
        if resolved in {"loo", "leave-one-out", "leave_one_out"}:
            return -1
        raise ValueError("k must be an integer or one of {'loo', 'leave-one-out'}.")
    return int(k)


def _resolve_n_jobs(n_jobs: int | None) -> int:
    if n_jobs is None:
        return 1
    resolved = int(n_jobs)
    if resolved == -1:
        return max(1, int(os.cpu_count() or 1))
    if resolved < 1:
        raise ValueError("n_jobs must be a positive integer, None, or -1.")
    return resolved


def _warn_if_cv_arguments_are_unused(
    *,
    n_candidates: int,
    k: int | str,
    average: bool | Sequence[int],
    seed: int | None,
    show_progress: bool,
) -> None:
    if n_candidates != 1:
        return

    unused: list[str] = []
    if isinstance(k, str) or int(k) != -1:
        unused.append("k")
    if average is not True:
        unused.append("average")
    if seed is not None:
        unused.append("seed")
    if bool(show_progress):
        unused.append("show_progress")

    if unused:
        formatted = ", ".join(unused)
        warnings.warn(
            f"{formatted} {'is' if len(unused) == 1 else 'are'} ignored because "
            "cross-validation requires more than one regularization candidate.",
            UserWarning,
            stacklevel=2,
        )


class _SimpleProgressBar:
    """Minimal stderr progress indicator used for optional CV feedback."""

    def __init__(self, *, total: int, label: str) -> None:
        self.total = max(1, int(total))
        self.label = str(label)
        self.current = 0
        self.stream = sys.stderr
        self.use_carriage = bool(getattr(self.stream, "isatty", lambda: False)())
        self._lock = Lock()
        self._emit()

    def update(self, step: int = 1) -> None:
        with self._lock:
            self.current = min(self.total, self.current + int(step))
            self._emit()

    def close(self) -> None:
        with self._lock:
            if self.use_carriage:
                self.stream.write("\n")
                self.stream.flush()

    def _emit(self) -> None:
        fraction = self.current / self.total
        width = 20
        filled = min(width, int(round(fraction * width)))
        bar = f"[{'#' * filled}{'-' * (width - filled)}] {self.current}/{self.total}"
        if self.use_carriage:
            self.stream.write(f"\r{self.label} {bar}")
        else:
            self.stream.write(f"{self.label} {bar}\n")
        self.stream.flush()


def _resolve_frequency_scale(scale: str) -> str:
    resolved = str(scale).strip().lower()
    if resolved not in {"linear", "log"}:
        raise ValueError("scale must be either 'linear' or 'log'.")
    return resolved


def _resolve_frequency_weight_value_mode(value_mode: str) -> str:
    resolved = str(value_mode).strip().lower()
    if resolved not in {"real", "magnitude", "power"}:
        raise ValueError("value_mode must be 'real', 'magnitude', or 'power'.")
    return resolved


def _smallest_positive_frequency(frequencies: np.ndarray) -> float | None:
    positive = np.asarray(frequencies, dtype=float)
    positive = positive[positive > 0.0]
    if positive.size == 0:
        return None
    return float(positive[0])


def _build_frequency_filterbank(
    frequencies: np.ndarray,
    *,
    n_bands: int,
    fmin: float | None,
    fmax: float | None,
    scale: str,
    bandwidth: float | None,
) -> tuple[np.ndarray, np.ndarray, str, float]:
    frequencies = np.asarray(frequencies, dtype=float)
    if frequencies.ndim != 1 or frequencies.size == 0:
        raise ValueError("frequencies must be a non-empty 1D array.")

    resolved_scale = _resolve_frequency_scale(scale)
    n_bands = int(n_bands)
    if n_bands <= 0:
        raise ValueError("n_bands must be a positive integer.")

    nyquist = float(frequencies[-1])
    if fmin is None:
        if resolved_scale == "linear":
            fmin = 0.0
        else:
            fmin = _smallest_positive_frequency(frequencies)
            if fmin is None:
                raise ValueError(
                    "Log-spaced frequency bands require at least one positive frequency bin."
                )
    if fmax is None:
        fmax = nyquist

    fmin = _coerce_nonnegative_float(float(fmin), name="fmin")
    fmax = _coerce_nonnegative_float(float(fmax), name="fmax")
    if resolved_scale == "log" and fmin <= 0.0:
        raise ValueError("fmin must be positive when scale='log'.")
    if fmax <= fmin:
        raise ValueError("fmax must be greater than fmin.")
    if fmax > nyquist + np.finfo(float).eps:
        raise ValueError("fmax cannot exceed the fitted Nyquist frequency.")

    if n_bands == 1:
        band_centers = np.array([(fmin + fmax) / 2.0], dtype=float)
    elif resolved_scale == "linear":
        band_centers = np.linspace(fmin, fmax, n_bands, dtype=float)
    else:
        band_centers = np.geomspace(fmin, fmax, n_bands, dtype=float)

    if bandwidth is None:
        if n_bands == 1:
            positive_step = _smallest_positive_frequency(np.diff(frequencies))
            inferred = max(fmax - fmin, positive_step or max(fmax, 1.0))
        else:
            inferred = float(np.median(np.diff(band_centers)))
        bandwidth = inferred
    bandwidth = float(bandwidth)
    if not np.isfinite(bandwidth) or bandwidth <= 0.0:
        raise ValueError("bandwidth must be finite and positive.")

    active = (frequencies >= fmin) & (frequencies <= fmax)
    filters = np.zeros((frequencies.shape[0], n_bands), dtype=float)
    if n_bands == 1:
        filters[active, 0] = 1.0
        return band_centers, filters, resolved_scale, bandwidth

    scaled_distance = (frequencies[:, np.newaxis] - band_centers[np.newaxis, :]) / bandwidth
    filters = np.exp(-0.5 * scaled_distance**2)
    filters[~active, :] = 0.0

    row_sums = filters.sum(axis=1, keepdims=True)
    valid_rows = active & (row_sums[:, 0] > np.finfo(float).eps)
    filters[valid_rows, :] /= row_sums[valid_rows, :]
    return band_centers, filters, resolved_scale, bandwidth


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
