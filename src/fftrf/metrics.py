
"""Metric utilities for ffTRF estimators."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .utils import _ensure_2d

MetricSpec = str | Callable[[np.ndarray, np.ndarray], np.ndarray]


def _scaled_metric_pair(
    y_true: np.ndarray, y_pred: np.ndarray, *, independent: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Rescale each pair of columns before computing dimensionless metrics."""

    y_true = _ensure_2d(y_true, "y_true")
    y_pred = _ensure_2d(y_pred, "y_pred")
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    true_scale = np.max(np.abs(y_true), axis=0)
    pred_scale = np.max(np.abs(y_pred), axis=0)
    if not independent:
        true_scale = pred_scale = np.maximum(true_scale, pred_scale)
    true_scale[true_scale == 0.0] = 1.0
    pred_scale[pred_scale == 0.0] = 1.0
    return y_true / true_scale, y_pred / pred_scale


def _center_columns(values: np.ndarray) -> np.ndarray:
    """Center scaled columns while keeping exactly constant columns zero."""

    shifted = values - values[:1]
    return shifted - shifted.mean(axis=0, keepdims=True)


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
    This is the default scoring metric used by :class:`TRF`. It is
    intentionally lightweight and does not return p-values.
    Correlation is invariant to positive amplitude rescaling. Exactly constant
    columns return zero. Empty or non-finite inputs are rejected.
    """

    y_true, y_pred = _scaled_metric_pair(y_true, y_pred, independent=True)
    y_true = _center_columns(y_true)
    y_pred = _center_columns(y_pred)

    numerator = np.sum(y_true * y_pred, axis=0)
    denominator = np.linalg.norm(y_true, axis=0) * np.linalg.norm(y_pred, axis=0)
    out = np.zeros(y_true.shape[1], dtype=float)
    valid = denominator > 0.0
    out[valid] = numerator[valid] / denominator[valid]
    return np.clip(out, -1.0, 1.0)


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

    Unlike :func:`explained_variance_score`, :math:`R^2` penalizes a constant
    offset between prediction and observation. The two scores are equal only
    when the prediction residual has zero mean.
    The score is invariant to a common nonzero rescaling of both arrays.
    For exactly constant targets, identical predictions return one and other
    predictions return zero. Empty or non-finite inputs are rejected.
    """

    y_true, y_pred = _scaled_metric_pair(y_true, y_pred)

    residual = np.sum((y_true - y_pred) ** 2, axis=0)
    total = np.sum(_center_columns(y_true)**2, axis=0)
    out = np.zeros(y_true.shape[1], dtype=float)
    valid = total > 0.0
    out[valid] = 1.0 - (residual[valid] / total[valid])
    perfect = (~valid) & (residual == 0.0)
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
    magnitude. Like :func:`r2_score`, larger values are better. A prediction
    that has the correct fluctuations but a constant offset can therefore have
    perfect explained variance while its :math:`R^2` is below one.
    The score is invariant to a common nonzero rescaling of both arrays.
    For exactly constant targets, constant residuals return one and varying
    residuals return zero. Empty or non-finite inputs are rejected.
    """

    y_true, y_pred = _scaled_metric_pair(y_true, y_pred)

    residual = y_true - y_pred
    variance_true = np.mean(_center_columns(y_true)**2, axis=0)
    variance_residual = np.mean(_center_columns(residual)**2, axis=0)
    out = np.zeros(y_true.shape[1], dtype=float)
    valid = variance_true > 0.0
    out[valid] = 1.0 - (variance_residual[valid] / variance_true[valid])
    perfect = (~valid) & (variance_residual == 0.0)
    out[perfect] = 1.0
    return out


def neg_mse(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Compute column-wise negative mean squared error.

    Parameters
    ----------
    y_true:
        Observed samples arranged as ``(n_samples, n_outputs)``.
    y_pred:
        Predicted samples with the same shape as ``y_true``.

    Returns
    -------
    numpy.ndarray
        One negative-MSE score per output column.

    Notes
    -----
    The sign convention matches :func:`mtrf.stats.neg_mse`: larger values are
    better because the underlying MSE is multiplied by ``-1``. Despite the
    name, this function is therefore directly suitable for cross-validation
    model selection in :class:`TRF`.
    """

    y_true = _ensure_2d(y_true, "y_true")
    y_pred = _ensure_2d(y_pred, "y_pred")
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")

    return -np.mean((y_true - y_pred) ** 2, axis=0)


_METRIC_REGISTRY: dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    "pearsonr": pearsonr,
    "r2": r2_score,
    "r2_score": r2_score,
    "explained_variance": explained_variance_score,
    "explained_variance_score": explained_variance_score,
    "neg_mse": neg_mse,
}


def available_metrics() -> tuple[str, ...]:
    """Return the names of built-in scoring metrics.

    Returns
    -------
    tuple of str
        Sorted metric names that can be passed to ``TRF(metric=...)`` or used
        when reconstructing a saved configuration.

    Notes
    -----
    Some metrics intentionally expose both a short alias and a more explicit
    function-style name, for example ``"r2"`` and ``"r2_score"``. They resolve
    to the same scoring function.
    """

    return tuple(sorted(_METRIC_REGISTRY))


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
