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
    weights, times, credible_interval = _validate_plot_inputs(
        weights,
        times,
        credible_interval=credible_interval,
    )
    _validate_kernel_indices(
        weights,
        input_index=input_index,
        output_index=output_index,
    )

    fig, ax = _coerce_axes(plt, ax)
    _plot_kernel_on_axes(
        ax,
        weights=weights,
        times=times,
        input_index=input_index,
        output_index=output_index,
        credible_interval=credible_interval,
        time_unit=time_unit,
        color=color,
        interval_color=interval_color,
        linewidth=linewidth,
        interval_alpha=interval_alpha,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        label=label,
    )
    return fig, ax


def plot_kernel_grid(
    *,
    weights: np.ndarray,
    times: np.ndarray,
    credible_interval: np.ndarray | None = None,
    ax: Any = None,
    time_unit: str = "ms",
    color: str | None = None,
    interval_color: str | None = None,
    linewidth: float = 1.8,
    interval_alpha: float = 0.2,
    title: str | None = None,
    input_labels: list[str] | tuple[str, ...] | None = None,
    output_labels: list[str] | tuple[str, ...] | None = None,
    sharey: bool = False,
) -> tuple[Any, np.ndarray]:
    """Plot all input/output kernels in a grid."""

    plt = _require_matplotlib()
    weights, times, credible_interval = _validate_plot_inputs(
        weights,
        times,
        credible_interval=credible_interval,
    )

    n_inputs, _, n_outputs = weights.shape
    if input_labels is not None and len(input_labels) != n_inputs:
        raise ValueError("input_labels must match the number of input features.")
    if output_labels is not None and len(output_labels) != n_outputs:
        raise ValueError("output_labels must match the number of outputs.")

    if ax is None:
        fig, axes = plt.subplots(
            n_inputs,
            n_outputs,
            figsize=(4.2 * n_outputs, 2.8 * n_inputs),
            squeeze=False,
            sharex=True,
            sharey=sharey,
        )
    else:
        axes = np.asarray(ax, dtype=object)
        if axes.shape != (n_inputs, n_outputs):
            raise ValueError(
                f"ax must have shape ({n_inputs}, {n_outputs}) for a full kernel grid."
            )
        fig = axes.flat[0].figure

    for input_index in range(n_inputs):
        for output_index in range(n_outputs):
            kernel_title = None
            if input_labels is not None or output_labels is not None:
                left = input_labels[input_index] if input_labels is not None else f"Input {input_index + 1}"
                right = output_labels[output_index] if output_labels is not None else f"Output {output_index + 1}"
                kernel_title = f"{left} -> {right}"
            elif title is None:
                kernel_title = f"Kernel ({input_index}, {output_index})"

            xlabel = "Lag (ms)" if time_unit == "ms" else "Lag (s)"
            if input_index < n_inputs - 1:
                xlabel = ""
            ylabel = "Weight" if output_index == 0 else ""

            _plot_kernel_on_axes(
                axes[input_index, output_index],
                weights=weights,
                times=times,
                input_index=input_index,
                output_index=output_index,
                credible_interval=credible_interval,
                time_unit=time_unit,
                color=color,
                interval_color=interval_color,
                linewidth=linewidth,
                interval_alpha=interval_alpha,
                title=kernel_title,
                xlabel=xlabel,
                ylabel=ylabel,
                label=None,
            )

    if title is not None:
        fig.suptitle(title)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    else:
        fig.tight_layout()
    return fig, axes


def _plot_kernel_on_axes(
    ax: Any,
    *,
    weights: np.ndarray,
    times: np.ndarray,
    input_index: int,
    output_index: int,
    credible_interval: np.ndarray | None,
    time_unit: str,
    color: str | None,
    interval_color: str | None,
    linewidth: float,
    interval_alpha: float,
    title: str | None,
    xlabel: str | None,
    ylabel: str,
    label: str | None,
) -> None:
    time_values, time_label = _time_axis(times, time_unit=time_unit)
    kernel = weights[input_index, :, output_index]

    if credible_interval is not None:
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


def _validate_plot_inputs(
    weights: np.ndarray,
    times: np.ndarray,
    *,
    credible_interval: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    weights = np.asarray(weights, dtype=float)
    times = np.asarray(times, dtype=float)
    if weights.ndim != 3:
        raise ValueError("weights must have shape (n_inputs, n_lags, n_outputs).")
    if times.ndim != 1 or times.shape[0] != weights.shape[1]:
        raise ValueError("times must be 1D and match the lag axis of weights.")
    if credible_interval is not None:
        credible_interval = np.asarray(credible_interval, dtype=float)
        if credible_interval.shape != (2, *weights.shape):
            raise ValueError(
                "credible_interval must have shape (2, n_inputs, n_lags, n_outputs)."
            )
    return weights, times, credible_interval


def _validate_kernel_indices(
    weights: np.ndarray,
    *,
    input_index: int,
    output_index: int,
) -> None:
    if not 0 <= int(input_index) < weights.shape[0]:
        raise IndexError(f"input_index out of bounds: {input_index}")
    if not 0 <= int(output_index) < weights.shape[2]:
        raise IndexError(f"output_index out of bounds: {output_index}")


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
