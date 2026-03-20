
"""Spectral estimation and solver helpers for ffTRF."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from scipy.fft import next_fast_len
from scipy.linalg import LinAlgError as scipy_linalg_LinAlgError
from scipy.linalg import cho_factor, cho_solve
from scipy.signal import detrend as scipy_detrend
from scipy.signal import get_window
from scipy.signal.windows import dpss

from .utils import _coerce_nonnegative_float, _copy_value, _normalize_weight_vector

SpectralMethod = str


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
class _ScalarRidgeDecomposition:
    """Cached eigen-decomposition reused across scalar-ridge solves."""

    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    projected_rhs: np.ndarray


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
    aggregate_only: bool = False,
    raw_trial_weights: np.ndarray | None = None,
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
    stored_trials = 1 if aggregate_only else n_trials
    normalized_trial_weights = None
    if aggregate_only:
        if raw_trial_weights is None:
            normalized_trial_weights = np.full(n_trials, 1.0 / n_trials, dtype=float)
        else:
            if np.asarray(raw_trial_weights, dtype=float).shape != (n_trials,):
                raise ValueError("raw_trial_weights must match the number of trials.")
            normalized_trial_weights = _normalize_weight_vector(raw_trial_weights)

    trial_cxx = np.zeros((stored_trials, n_frequencies, n_inputs, n_inputs), dtype=np.complex128)
    trial_cxy = np.zeros((stored_trials, n_frequencies, n_inputs, n_outputs), dtype=np.complex128)

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
        stored_trial_index = 0 if aggregate_only else trial_index
        trial_weight = 1.0 if normalized_trial_weights is None else float(normalized_trial_weights[trial_index])
        segments = list(
            _iter_segments(
                x_trial,
                y_trial,
                segment_length=resolved_segment_length,
                overlap=overlap,
            )
        )
        segment_weight = trial_weight / len(segments)
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

                    trial_cxx[stored_trial_index] += taper_weight * np.einsum(
                        "fi,fj->fij", np.conjugate(x_fft), x_fft, optimize=True
                    )
                    trial_cxy[stored_trial_index] += taper_weight * np.einsum(
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

                trial_cxx[stored_trial_index] += segment_weight * np.einsum(
                    "fi,fj->fij", np.conjugate(x_fft), x_fft, optimize=True
                )
                trial_cxy[stored_trial_index] += segment_weight * np.einsum(
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


def _scalar_regularization_value(feature_regularization: np.ndarray) -> float | None:
    penalties = np.asarray(feature_regularization, dtype=float)
    if penalties.ndim != 1 or penalties.size == 0:
        raise ValueError("feature_regularization must be a non-empty 1D array.")
    if np.allclose(penalties, penalties[0]):
        return _coerce_nonnegative_float(
            float(penalties[0]),
            name="feature_regularization",
        )
    return None


def _scalar_regularization_grid(
    feature_regularization_values: Sequence[np.ndarray],
) -> np.ndarray | None:
    scalar_values = []
    for feature_regularization in feature_regularization_values:
        scalar_value = _scalar_regularization_value(feature_regularization)
        if scalar_value is None or scalar_value <= 0.0:
            return None
        scalar_values.append(scalar_value)
    return np.asarray(scalar_values, dtype=float)


def _prepare_scalar_ridge_decomposition(
    cxx: np.ndarray,
    cxy: np.ndarray,
) -> _ScalarRidgeDecomposition:
    cxx = np.asarray(cxx, dtype=np.complex128)
    cxy = np.asarray(cxy, dtype=np.complex128)
    hermitian_cxx = 0.5 * (cxx + np.swapaxes(np.conjugate(cxx), 1, 2))
    eigenvalues, eigenvectors = np.linalg.eigh(hermitian_cxx)
    eigenvalues = np.maximum(eigenvalues.real, 0.0)
    projected_rhs = np.matmul(
        np.swapaxes(np.conjugate(eigenvectors), 1, 2),
        cxy,
    )
    return _ScalarRidgeDecomposition(
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        projected_rhs=projected_rhs,
    )


def _solve_scalar_ridge_from_decomposition(
    decomposition: _ScalarRidgeDecomposition,
    *,
    regularization: float,
) -> np.ndarray:
    ridge = _coerce_nonnegative_float(regularization, name="feature_regularization")
    denominator = decomposition.eigenvalues + ridge
    scaled_rhs = decomposition.projected_rhs / denominator[..., np.newaxis]
    return np.matmul(decomposition.eigenvectors, scaled_rhs)


def _solve_transfer_function(
    cxx: np.ndarray,
    cxy: np.ndarray,
    *,
    feature_regularization: np.ndarray,
    scalar_decomposition: _ScalarRidgeDecomposition | None = None,
    scalar_regularization: float | None = None,
) -> np.ndarray:
    feature_regularization = np.asarray(feature_regularization, dtype=float)
    if feature_regularization.shape != (cxx.shape[1],):
        raise ValueError(
            "feature_regularization must provide one penalty per predictor feature."
        )

    scalar_value = (
        _scalar_regularization_value(feature_regularization)
        if scalar_regularization is None
        else _coerce_nonnegative_float(
            float(scalar_regularization),
            name="feature_regularization",
        )
    )
    if scalar_decomposition is not None:
        if scalar_value is None:
            raise ValueError(
                "scalar_decomposition can only be used with scalar ridge regularization."
            )
        return _solve_scalar_ridge_from_decomposition(
            scalar_decomposition,
            regularization=scalar_value,
        )

    ridge_diagonal = (
        np.full(cxx.shape[1], scalar_value, dtype=float)
        if scalar_value is not None
        else feature_regularization.astype(float, copy=True)
    )

    transfer_function = np.zeros_like(cxy)
    for frequency_index in range(cxx.shape[0]):
        system = np.array(cxx[frequency_index], dtype=np.complex128, copy=True)
        system = 0.5 * (system + np.conjugate(system.T))
        system.flat[:: system.shape[0] + 1] += ridge_diagonal
        try:
            factor = cho_factor(system, lower=False, check_finite=False)
            transfer_function[frequency_index] = cho_solve(
                factor,
                np.asarray(cxy[frequency_index], dtype=np.complex128),
                check_finite=False,
            )
        except (np.linalg.LinAlgError, scipy_linalg_LinAlgError):
            transfer_function[frequency_index] = np.linalg.solve(
                system,
                cxy[frequency_index],
            )
    return transfer_function
