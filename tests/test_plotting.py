from __future__ import annotations

import matplotlib.pyplot as pyplot
import numpy as np
import pytest

import fftrf.plotting as plotting


@pytest.fixture()
def plt():
    yield pyplot
    pyplot.close("all")


def _kernel_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    weights = np.arange(2 * 4 * 2, dtype=float).reshape(2, 4, 2) / 10.0
    times = np.array([-0.01, 0.0, 0.01, 0.02])
    interval = np.stack([weights - 0.05, weights + 0.05])
    return weights, times, interval


def test_plot_kernel_uses_supplied_axes_interval_and_labels(plt) -> None:
    weights, times, interval = _kernel_inputs()
    fig, ax = plt.subplots()

    returned_fig, returned_ax = plotting.plot_kernel(
        weights=weights,
        times=times,
        input_index=1,
        output_index=1,
        credible_interval=interval,
        ax=ax,
        time_unit="s",
        color="black",
        interval_color="orange",
        title="Custom kernel",
        xlabel="Custom lag",
        label="decoder",
    )

    assert returned_fig is fig
    assert returned_ax is ax
    assert ax.get_xlabel() == "Custom lag"
    assert ax.get_ylabel() == "Weight"
    assert ax.get_title() == "Custom kernel"
    assert ax.get_legend() is not None
    assert len(ax.collections) == 1
    assert len(ax.lines) == 2


def test_plot_kernel_validates_shapes_and_time_units(plt) -> None:
    weights, times, interval = _kernel_inputs()

    with pytest.raises(ValueError, match="weights must have shape"):
        plotting.plot_kernel(weights=weights[0], times=times)

    with pytest.raises(ValueError, match="times must be 1D"):
        plotting.plot_kernel(weights=weights, times=times[:-1])

    with pytest.raises(ValueError, match="credible_interval must have shape"):
        plotting.plot_kernel(weights=weights, times=times, credible_interval=interval[:, :1])

    with pytest.raises(ValueError, match="time_unit"):
        plotting.plot_kernel(weights=weights, times=times, time_unit="samples")


def test_plot_kernel_grid_uses_supplied_axes_and_validates_layout(plt) -> None:
    weights, times, interval = _kernel_inputs()
    fig, axes = plt.subplots(2, 2)

    returned_fig, returned_axes = plotting.plot_kernel_grid(
        weights=weights,
        times=times,
        credible_interval=interval,
        ax=axes,
        time_unit="s",
        input_labels=["Envelope", "Onsets"],
        title="All kernels",
        sharey=True,
    )

    assert returned_fig is fig
    assert returned_axes.shape == (2, 2)
    assert fig._suptitle is not None
    assert fig._suptitle.get_text() == "All kernels"
    assert axes[0, 0].get_title() == "Envelope -> Output 1"
    assert axes[1, 0].get_xlabel() == "Lag (s)"
    assert axes[0, 1].get_ylabel() == ""

    with pytest.raises(ValueError, match="input_labels"):
        plotting.plot_kernel_grid(weights=weights, times=times, input_labels=["only one"])

    with pytest.raises(ValueError, match="output_labels"):
        plotting.plot_kernel_grid(weights=weights, times=times, output_labels=["only one"])

    wrong_fig, wrong_axes = plt.subplots(3, 1)
    with pytest.raises(ValueError, match="ax must have shape"):
        plotting.plot_kernel_grid(weights=weights, times=times, ax=wrong_axes)
    plt.close(wrong_fig)


def test_frequency_resolved_weight_plot_options_and_validation(plt) -> None:
    weights = np.array([[[[1.0], [-0.5], [0.25]]]])
    times = np.array([0.0, 0.01, 0.02])
    band_centers = np.array([10.0])

    fig, ax = plt.subplots()
    returned_fig, returned_ax = plotting.plot_frequency_resolved_weights(
        weights=weights,
        times=times,
        band_centers=band_centers,
        value_mode="power",
        bandwidth=4.0,
        ax=ax,
        time_unit="s",
        colorbar=False,
        frequency_axis_scale="log",
        title="Band-limited weights",
    )

    assert returned_fig is fig
    assert returned_ax is ax
    assert ax.get_xlabel() == "Lag (s)"
    assert ax.get_ylabel() == "Frequency (Hz)"
    assert ax.get_yscale() == "log"
    assert len(fig.axes) == 1

    colorbar_fig, colorbar_ax = plotting.plot_frequency_resolved_weights(
        weights=weights,
        times=times,
        band_centers=band_centers,
        value_mode="magnitude",
        colorbar_label="Custom magnitude",
    )
    assert colorbar_ax.get_title().startswith("Frequency-resolved weights")
    assert colorbar_fig.axes[1].get_ylabel() == "Custom magnitude"

    with pytest.raises(ValueError, match="value_mode"):
        plotting.plot_frequency_resolved_weights(
            weights=weights,
            times=times,
            band_centers=band_centers,
            value_mode="complex",
        )

    with pytest.raises(ValueError, match="frequency_axis_scale"):
        plotting.plot_frequency_resolved_weights(
            weights=weights,
            times=times,
            band_centers=band_centers,
            frequency_axis_scale="mel",
        )

    with pytest.raises(ValueError, match="strictly positive"):
        plotting.plot_frequency_resolved_weights(
            weights=weights,
            times=times,
            band_centers=np.array([0.0]),
            frequency_axis_scale="log",
        )

    with pytest.raises(IndexError, match="input_index"):
        plotting.plot_frequency_resolved_weights(
            weights=weights,
            times=times,
            band_centers=band_centers,
            input_index=1,
        )


def test_transfer_function_plot_variants_and_validation(plt) -> None:
    frequencies = np.array([0.0, 10.0, 20.0, 30.0])
    transfer = np.exp(-1j * 2.0 * np.pi * frequencies * 0.003)

    phase_fig, phase_ax = plotting.plot_transfer_function(
        frequencies=frequencies,
        transfer_function=transfer,
        kind="phase",
        phase_unit="deg",
    )
    assert phase_ax.get_ylabel() == "Phase (deg)"
    assert phase_ax.get_xlabel() == "Frequency (Hz)"
    plt.close(phase_fig)

    delay_fig, delay_ax = plotting.plot_transfer_function(
        frequencies=frequencies,
        transfer_function=transfer,
        kind="group_delay",
        group_delay_unit="s",
    )
    assert delay_ax.get_ylabel() == "Group delay (s)"
    plt.close(delay_fig)

    axes_fig, axes = plt.subplots(3, 1)
    returned_fig, returned_axes = plotting.plot_transfer_function(
        frequencies=frequencies,
        transfer_function=transfer,
        kind="all",
        ax=axes,
        title="All transfer views",
    )
    assert returned_fig is axes_fig
    assert returned_axes.shape == (3,)
    assert axes[0].get_title() == "All transfer views"
    assert axes[2].get_xlabel() == "Frequency (Hz)"

    wrong_fig, wrong_axes = plt.subplots(2, 1)
    with pytest.raises(ValueError, match="ax must contain 3 axes"):
        plotting.plot_transfer_function(
            frequencies=frequencies,
            transfer_function=transfer,
            kind="all",
            ax=wrong_axes,
        )
    plt.close(wrong_fig)

    with pytest.raises(ValueError, match="matching lengths"):
        plotting.plot_transfer_function(
            frequencies=frequencies[:-1],
            transfer_function=transfer,
        )

    with pytest.raises(ValueError, match="phase_unit"):
        plotting.plot_transfer_function(
            frequencies=frequencies,
            transfer_function=transfer,
            kind="phase",
            phase_unit="turns",
        )

    with pytest.raises(ValueError, match="group_delay_unit"):
        plotting.plot_transfer_function(
            frequencies=frequencies,
            transfer_function=transfer,
            kind="group_delay",
            group_delay_unit="samples",
        )


def test_cross_spectrum_and_coherence_plot_variants_and_validation(plt) -> None:
    frequencies = np.array([0.0, 5.0, 10.0])
    base = np.exp(1j * frequencies / 10.0)
    cross_spectrum = np.column_stack([base, 2.0 * base])
    coherence = np.column_stack([
        np.linspace(0.1, 0.8, frequencies.size),
        np.linspace(0.2, 0.9, frequencies.size),
    ])

    cross_fig, cross_ax = plotting.plot_cross_spectrum(
        frequencies=frequencies,
        cross_spectrum=cross_spectrum,
        output_index=1,
        kind="magnitude",
    )
    assert cross_ax.get_title() == "Cross-spectrum magnitude"
    assert cross_ax.get_ylabel() == "Magnitude"
    plt.close(cross_fig)

    axes_fig, axes = plt.subplots(2, 1)
    returned_fig, returned_axes = plotting.plot_cross_spectrum(
        frequencies=frequencies,
        cross_spectrum=cross_spectrum,
        kind="both",
        ax=axes,
        phase_unit="deg",
    )
    assert returned_fig is axes_fig
    assert returned_axes.shape == (2,)
    assert axes[1].get_ylabel() == "Phase (deg)"
    assert axes[1].get_xlabel() == "Frequency (Hz)"

    coherence_fig, coherence_ax = plt.subplots()
    returned_fig, returned_ax = plotting.plot_coherence(
        frequencies=frequencies,
        coherence=coherence,
        output_index=1,
        ax=coherence_ax,
        title="Custom coherence",
    )
    assert returned_fig is coherence_fig
    assert returned_ax is coherence_ax
    assert coherence_ax.get_title() == "Custom coherence"
    assert coherence_ax.get_ylim()[1] == pytest.approx(1.05)

    wrong_fig, wrong_axes = plt.subplots(3, 1)
    with pytest.raises(ValueError, match="two axes"):
        plotting.plot_cross_spectrum(
            frequencies=frequencies,
            cross_spectrum=cross_spectrum,
            kind="both",
            ax=wrong_axes,
        )
    plt.close(wrong_fig)

    with pytest.raises(ValueError, match="kind"):
        plotting.plot_cross_spectrum(
            frequencies=frequencies,
            cross_spectrum=cross_spectrum,
            kind="power",
        )

    with pytest.raises(IndexError, match="output_index"):
        plotting.plot_coherence(
            frequencies=frequencies,
            coherence=coherence,
            output_index=2,
        )

    with pytest.raises(ValueError, match="coherence must have shape"):
        plotting.plot_coherence(
            frequencies=frequencies,
            coherence=coherence[:, 0],
        )
