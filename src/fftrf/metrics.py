
"""Metric utilities for ffTRF estimators."""

from __future__ import annotations

from typing import Callable

import numpy as np

from .utils import _ensure_2d

MetricSpec = str | Callable[[np.ndarray, np.ndarray], np.ndarray]


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
