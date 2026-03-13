"""Compatibility aliases for legacy experimental imports."""

from ..bayesian import (
    BayesianFrequencyTRF,
    BayesianTRFResult,
    fit_bayesian_frequency_trf,
    predict_bayesian_frequency_trf,
)

__all__ = [
    "BayesianFrequencyTRF",
    "BayesianTRFResult",
    "fit_bayesian_frequency_trf",
    "predict_bayesian_frequency_trf",
]
