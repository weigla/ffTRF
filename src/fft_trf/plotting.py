"""Shared plotting helpers for TRF estimators.

Plotting is kept optional so the core toolbox does not require matplotlib at
install time. Estimator methods call into this module lazily.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def plot_kernel(
    *,
    weights: np.ndarray,
    times: np.ndarray,
    input_index: int = 0,
    output_index: int = 0,
    credible_interval: np.ndarray | None = None,
    ax: Any = None,
    time_unit: str = "ms",
    color: str | None = None,
    interval_color: str | None = None,
    linewidth: float = 2.0,
    interval_alpha: float = 0.2,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str = "Weight",
    label: str | None = None,
) -> tuple[Any, Any]:
    """Plot one input/output kernel, optionally with a credible interval."""

    plt = _require_matplotlib()
    weights = np.asarray(weights, dtype=float)
    times = np.asarray(times, dtype=float)
    if weights.ndim != 3:
        raise ValueError("weights must have shape (n_inputs, n_lags, n_outputs).")
    if times.ndim != 1 or times.shape[0] != weights.shape[1]:
        raise ValueError("times must be 1D and match the lag axis of weights.")

    if not 0 <= input_index < weights.shape[0]:
        raise IndexError(f"input_index out of bounds: {input_index}")
    if not 0 <= output_index < weights.shape[2]:
        raise IndexError(f"output_index out of bounds: {output_index}")

    fig, ax = _coerce_axes(plt, ax)
    time_values, time_label = _time_axis(times, time_unit=time_unit)
    kernel = weights[input_index, :, output_index]

    if credible_interval is not None:
        credible_interval = np.asarray(credible_interval, dtype=float)
        if credible_interval.shape != (2, *weights.shape):
            raise ValueError(
                "credible_interval must have shape (2, n_inputs, n_lags, n_outputs)."
            )
        lower = credible_interval[0, input_index, :, output_index]
        upper = credible_interval[1, input_index, :, output_index]
        fill_color = interval_color or color or "#4C72B0"
        ax.fill_between(
            time_values,
            lower,
            upper,
            color=fill_color,
            alpha=interval_alpha,
            linewidth=0.0,
            label=None if label is None else f"{label} interval",
        )

    ax.plot(
        time_values,
        kernel,
        color=color,
        linewidth=linewidth,
        label=label,
    )
    ax.axhline(0.0, color="#666666", linewidth=1.0, linestyle="--")
    ax.set_xlabel(xlabel or time_label)
    ax.set_ylabel(ylabel)
    if title is None:
        ax.set_title(f"Kernel (input {input_index}, output {output_index})")
    else:
        ax.set_title(title)
    if label is not None:
        ax.legend(frameon=False)
    return fig, ax


def _coerce_axes(plt: Any, ax: Any) -> tuple[Any, Any]:
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
        return fig, ax
    return ax.figure, ax


def _time_axis(times: np.ndarray, *, time_unit: str) -> tuple[np.ndarray, str]:
    if time_unit == "s":
        return times, "Lag (s)"
    if time_unit == "ms":
        return times * 1e3, "Lag (ms)"
    raise ValueError("time_unit must be either 's' or 'ms'.")


def _require_matplotlib() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required for plotting. Install matplotlib directly "
            "or use the compare environment / compare extras."
        ) from exc
    return plt
