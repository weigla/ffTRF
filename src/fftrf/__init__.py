from importlib.metadata import PackageNotFoundError, version as _metadata_version

from .model import (
    CrossSpectralDiagnostics,
    FrequencyResolvedWeights,
    PermutationTestResult,
    TRF,
    TRFDiagnostics,
    TimeFrequencyPower,
    TransferFunctionComponents,
    available_metrics,
    explained_variance_score,
    neg_mse,
    pearsonr,
    r2_score,
)
from .preprocessing import half_wave_rectify, inverse_variance_weights, resample_signal
from .utils import suggest_segment_settings

try:
    __version__ = _metadata_version("fftrf")
except PackageNotFoundError:  # pragma: no cover - used only from an uninstalled source tree
    __version__ = "0.1.0"

__all__ = [
    "CrossSpectralDiagnostics",
    "FrequencyResolvedWeights",
    "PermutationTestResult",
    "TRF",
    "TRFDiagnostics",
    "TimeFrequencyPower",
    "TransferFunctionComponents",
    "__version__",
    "available_metrics",
    "explained_variance_score",
    "half_wave_rectify",
    "inverse_variance_weights",
    "neg_mse",
    "pearsonr",
    "r2_score",
    "resample_signal",
    "suggest_segment_settings",
]
