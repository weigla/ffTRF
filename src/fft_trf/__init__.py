"""Compatibility alias for the renamed :mod:`fftrf` package."""

from fftrf import FrequencyTRF, pearsonr
from fftrf.preprocessing import half_wave_rectify, inverse_variance_weights, resample_signal

__all__ = ["FrequencyTRF", "half_wave_rectify", "inverse_variance_weights", "pearsonr", "resample_signal"]
