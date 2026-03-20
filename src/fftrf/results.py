
"""Public result containers returned by ffTRF."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class TRFDiagnostics:
    """Observed-vs-predicted spectral diagnostics for a fitted model.

    Attributes
    ----------
    frequencies:
        Frequency vector in Hz.
    transfer_function:
        Complex transfer function copied from the fitted model. Shape is
        ``(n_frequencies, n_inputs, n_outputs)``.
    predicted_spectrum:
        Diagonal auto-spectrum of the model predictions for each output. Shape
        is ``(n_frequencies, n_outputs)``.
    observed_spectrum:
        Diagonal auto-spectrum of the observed targets. Shape is
        ``(n_frequencies, n_outputs)``.
    cross_spectrum:
        Matched predicted-vs-observed cross-spectrum for each output. Shape is
        ``(n_frequencies, n_outputs)``.
    coherence:
        Magnitude-squared coherence between each predicted output and the
        corresponding observed target. Shape is ``(n_frequencies, n_outputs)``.
    """

    frequencies: np.ndarray
    transfer_function: np.ndarray
    predicted_spectrum: np.ndarray
    observed_spectrum: np.ndarray
    cross_spectrum: np.ndarray
    coherence: np.ndarray


CrossSpectralDiagnostics = TRFDiagnostics


@dataclass(slots=True)
class TransferFunctionComponents:
    """Derived one-pair transfer-function quantities.

    Attributes
    ----------
    frequencies:
        Frequency vector in Hz.
    transfer_function:
        Complex transfer-function values for one input/output pair.
    magnitude:
        Absolute value of the transfer function.
    phase:
        Unwrapped transfer-function phase, reported in the requested units.
    phase_unit:
        Unit used for :attr:`phase`, either ``"rad"`` or ``"deg"``.
    group_delay:
        Group delay derived from the unwrapped phase, expressed in seconds.
    """

    frequencies: np.ndarray
    transfer_function: np.ndarray
    magnitude: np.ndarray
    phase: np.ndarray
    phase_unit: str
    group_delay: np.ndarray


@dataclass(slots=True)
class FrequencyResolvedWeights:
    """Frequency-resolved time-domain view of a fitted transfer function.

    Attributes
    ----------
    frequencies:
        Original frequency vector of the fitted transfer function in Hz.
    band_centers:
        Center frequency of each analysis band in Hz.
    filters:
        Frequency-domain filter bank used to partition the transfer function.
        Shape is ``(n_frequencies, n_bands)``.
    times:
        Lag vector in seconds corresponding to the third axis of
        :attr:`weights`.
    weights:
        Frequency-resolved kernel tensor with shape
        ``(n_inputs, n_bands, n_lags, n_outputs)``.
    scale:
        Spacing used for :attr:`band_centers`, either ``"linear"`` or
        ``"log"``.
    value_mode:
        Representation stored in :attr:`weights`. ``"real"`` preserves the
        signed band-limited kernels, ``"magnitude"`` stores their absolute
        value, and ``"power"`` stores squared magnitude.
    bandwidth:
        Gaussian filter width in Hz used for the analysis bands.
    """

    frequencies: np.ndarray
    band_centers: np.ndarray
    filters: np.ndarray
    times: np.ndarray
    weights: np.ndarray
    scale: str
    value_mode: str
    bandwidth: float

    def at(
        self,
        *,
        input_index: int = 0,
        output_index: int = 0,
    ) -> np.ndarray:
        """Return one frequency-by-lag map from the resolved kernel bank."""

        if not 0 <= int(input_index) < self.weights.shape[0]:
            raise IndexError(f"input_index out of bounds: {input_index}")
        if not 0 <= int(output_index) < self.weights.shape[3]:
            raise IndexError(f"output_index out of bounds: {output_index}")
        return self.weights[int(input_index), :, :, int(output_index)].copy()


@dataclass(slots=True)
class TimeFrequencyPower:
    """Spectrogram-like time-frequency power derived from a fitted kernel.

    Attributes
    ----------
    frequencies:
        Original frequency vector of the fitted transfer function in Hz.
    band_centers:
        Center frequency of each analysis band in Hz.
    filters:
        Frequency-domain filter bank used to partition the transfer function.
        Shape is ``(n_frequencies, n_bands)``.
    times:
        Lag vector in seconds corresponding to the third axis of :attr:`power`.
    power:
        Time-frequency power tensor with shape
        ``(n_inputs, n_bands, n_lags, n_outputs)``.
    scale:
        Spacing used for :attr:`band_centers`, either ``"linear"`` or
        ``"log"``.
    method:
        Power-estimation method. Currently ``"hilbert"`` computes the squared
        magnitude of the analytic signal for each band-limited kernel.
    bandwidth:
        Gaussian filter width in Hz used for the analysis bands.
    """

    frequencies: np.ndarray
    band_centers: np.ndarray
    filters: np.ndarray
    times: np.ndarray
    power: np.ndarray
    scale: str
    method: str
    bandwidth: float

    def at(
        self,
        *,
        input_index: int = 0,
        output_index: int = 0,
    ) -> np.ndarray:
        """Return one frequency-by-lag power map."""

        if not 0 <= int(input_index) < self.power.shape[0]:
            raise IndexError(f"input_index out of bounds: {input_index}")
        if not 0 <= int(output_index) < self.power.shape[3]:
            raise IndexError(f"output_index out of bounds: {output_index}")
        return self.power[int(input_index), :, :, int(output_index)].copy()
