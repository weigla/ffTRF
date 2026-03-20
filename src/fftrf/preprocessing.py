"""Small preprocessing helpers commonly needed before fitting frequency TRFs.

The functions in this module are intentionally minimal and dependency-light.
They cover a few operations that are broadly useful when preparing continuous
predictor and response signals for TRF analysis:

- create positive/negative half-wave regressors from a waveform
- resample derived regressors to a target sampling rate
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
        One-dimensional waveform, envelope, or other continuous regressor.
        Values above zero are routed into the positive output and values below
        zero are routed into the negative output. The input is converted to a
        floating-point NumPy array before the split is performed.

    Returns
    -------
    positive, negative:
        Arrays with the same shape as ``signal``. ``positive`` contains the
        positive half-wave and ``negative`` contains the magnitude of the
        negative half-wave. Both outputs are non-negative. At every sample,
        exactly one of them is non-zero unless the original sample value is
        zero.

    Notes
    -----
    Modeling positive and negative polarities separately can be useful when the
    sign of a waveform carries different predictive structure. A common use
    case is constructing separate regressors for compression-like or
    rectification-like hypotheses in auditory models without losing the timing
    of the original waveform.
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
        Input signal or array. The function accepts 1D or multi-dimensional
        arrays and only resamples along ``axis``.
    orig_fs:
        Original sampling rate in Hz. This must describe the spacing of the
        samples already present in ``signal``.
    target_fs:
        Target sampling rate in Hz. The returned array is sampled on this new
        grid.
    axis:
        Axis along which resampling should be performed. By default the helper
        assumes the first axis is time, which matches the rest of ``ffTRF``.
    max_denominator:
        Controls the rational approximation used to derive the integer up/down
        factors for :func:`scipy.signal.resample_poly`. Larger values allow a
        closer approximation when the ratio between ``orig_fs`` and
        ``target_fs`` is not a simple rational number, at the cost of a more
        complex resampling filter.

    Returns
    -------
    numpy.ndarray
        Resampled signal with the same number of non-time dimensions as the
        input.

    Notes
    -----
    This is a practical helper for bringing derived regressors to the same
    sampling rate as the target signal before fitting a model. Polyphase
    resampling is usually preferable to naive interpolation because it combines
    rate conversion with proper anti-alias filtering.
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
        variance is averaged across columns to obtain one scalar variance
        estimate per trial. Trials with larger variance therefore receive
        smaller weights.

    Returns
    -------
    numpy.ndarray
        One normalized weight per trial. The weights sum to 1 and can be passed
        directly to ``TRF.train(..., trial_weights=weights)``.

    Notes
    -----
    This helper is intentionally simple: it treats variance as a proxy for
    trial noisiness and does not attempt to estimate structured noise
    covariance. It is a good default when some trials are visibly noisier than
    others and you want those trials to contribute less to the aggregate
    cross-spectra during fitting or bootstrap diagnostics.

    Variance is averaged across channels/features so noisier trials contribute
    less to the final fit.
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
