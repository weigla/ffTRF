from __future__ import annotations

from fractions import Fraction
from typing import Sequence

import numpy as np
from scipy.signal import resample_poly


def half_wave_rectify(signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return positive and negative half-wave rectified versions of ``signal``."""

    signal = np.asarray(signal, dtype=float)
    return np.maximum(signal, 0.0), np.maximum(-signal, 0.0)


def resample_signal(
    signal: np.ndarray,
    orig_fs: float,
    target_fs: float,
    axis: int = 0,
    max_denominator: int = 10_000,
) -> np.ndarray:
    """Resample a signal with ``scipy.signal.resample_poly``."""

    ratio = Fraction(float(target_fs) / float(orig_fs)).limit_denominator(
        max_denominator
    )
    return resample_poly(
        np.asarray(signal, dtype=float),
        up=ratio.numerator,
        down=ratio.denominator,
        axis=axis,
    )


def inverse_variance_weights(trials: Sequence[np.ndarray]) -> np.ndarray:
    """
    Compute normalized inverse-variance weights for a list of trials.

    Variance is averaged across channels/features, which is usually appropriate
    for ABR-like recordings where noisier trials should contribute less.
    """

    if len(trials) == 0:
        raise ValueError("Need at least one trial to compute weights.")

    variances = []
    for trial in trials:
        trial = np.asarray(trial, dtype=float)
        if trial.ndim == 1:
            trial = trial[:, np.newaxis]
        variances.append(np.var(trial, axis=0).mean())

    variances = np.asarray(variances, dtype=float)
    variances = np.clip(variances, np.finfo(float).eps, None)
    weights = 1.0 / variances
    return weights / weights.sum()
