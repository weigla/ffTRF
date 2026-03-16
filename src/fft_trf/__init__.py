"""Compatibility alias for the renamed :mod:`fftrf` package."""

from fftrf import CrossSpectralDiagnostics, FrequencyResolvedWeights, FrequencyTRF, FrequencyTRFDiagnostics, TimeFrequencyPower, TransferFunctionComponents, pearsonr
from fftrf.preprocessing import half_wave_rectify, inverse_variance_weights, resample_signal

__all__ = [
    "CrossSpectralDiagnostics",
    "FrequencyResolvedWeights",
    "FrequencyTRF",
    "FrequencyTRFDiagnostics",
    "TimeFrequencyPower",
    "TransferFunctionComponents",
    "half_wave_rectify",
    "inverse_variance_weights",
    "pearsonr",
    "resample_signal",
]
