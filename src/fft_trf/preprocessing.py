"""Small preprocessing helpers commonly needed before fitting frequency TRFs.

The functions in this module are intentionally minimal and dependency-light.
They cover a few operations that come up frequently in ABR-style pipelines:

- create positive/negative half-wave regressors from an audio waveform
- resample stimulus-derived regressors to the neural recording rate
- compute inverse-variance trial weights for noisy recordings
"""

from __future__ import annotations

from fractions import Fraction
from typing import Sequence

import numpy as np
from scipy.signal import resample_poly


def half_wave_rectify(signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a waveform into positive and negative half-wave components.

    Parameters
    ----------
    signal:
        One-dimensional audio or regressor signal.

    Returns
    -------
    positive, negative:
        Arrays with the same shape as ``signal``. ``positive`` contains the
        positive half-wave and ``negative`` contains the magnitude of the
        negative half-wave.

    Notes
    -----
    This mirrors a common ABR regressor construction strategy where positive and
    negative polarities are modeled separately and later averaged.
    """

    signal = np.asarray(signal, dtype=float)
    return np.maximum(signal, 0.0), np.maximum(-signal, 0.0)


def resample_signal(
    signal: np.ndarray,
    orig_fs: float,
    target_fs: float,
    axis: int = 0,
    max_denominator: int = 10_000,
) -> np.ndarray:
    """Resample a signal using polyphase filtering.

    Parameters
    ----------
    signal:
        Input signal or array.
    orig_fs:
        Original sampling rate in Hz.
    target_fs:
        Target sampling rate in Hz.
    axis:
        Axis along which resampling should be performed.
    max_denominator:
        Controls the rational approximation used to derive up/down factors.

    Returns
    -------
    numpy.ndarray
        Resampled signal.

    Notes
    -----
    This is a practical helper for bringing audio-derived regressors to the same
    sampling rate as EEG/MEG data before fitting a model.
    """

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

    Parameters
    ----------
    trials:
        Sequence of trial arrays. Each trial may be 1D or 2D. For 2D arrays, the
        variance is averaged across columns to obtain one variance estimate per
        trial.

    Returns
    -------
    numpy.ndarray
        One normalized weight per trial. The weights sum to 1.

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
