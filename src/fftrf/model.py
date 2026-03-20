
"""Convenience import surface for ffTRF internals and public classes."""

from __future__ import annotations

from .estimator import (
    TRF,
    RegularizationSpec,
    SpectralMethod,
    _USE_STORED_TRIAL_WEIGHTS,
)
from .metrics import MetricSpec, _resolve_metric, available_metrics, explained_variance_score, pearsonr, r2_score
from .prediction import (
    _compute_bootstrap_interval_from_cache,
    _extract_impulse_response,
    _predict_trials_from_weights,
    _score_prediction_trials,
    _score_regularization_grid_for_fold,
    _slice_interval,
    _validate_confidence_level,
)
from .results import (
    CrossSpectralDiagnostics,
    FrequencyResolvedWeights,
    TRFDiagnostics,
    TimeFrequencyPower,
    TransferFunctionComponents,
)
from .spectral import (
    _ScalarRidgeDecomposition,
    _SpectralCache,
    _aggregate_cached_spectra,
    _build_spectral_cache,
    _prepare_scalar_ridge_decomposition,
    _scalar_regularization_grid,
    _scalar_regularization_value,
    _solve_scalar_ridge_from_decomposition,
    _solve_transfer_function,
    _validate_spectral_method,
)
from .utils import (
    _SimpleProgressBar,
    _aggregate_metric,
    _build_frequency_filterbank,
    _check_trial_lengths,
    _coerce_nonnegative_float,
    _coerce_trials,
    _copy_value,
    _ensure_2d,
    _expand_feature_regularization,
    _group_delay_values,
    _is_scalar_like,
    _normalize_trial_weights,
    _normalize_weight_vector,
    _phase_values,
    _resolve_frequency_scale,
    _resolve_frequency_weight_value_mode,
    _resolve_k_folds,
    _resolve_n_jobs,
    _resolve_raw_trial_weights,
    _resolve_regularization_candidates,
    _resolve_segment_length,
    _resolve_phase_unit,
    _smallest_positive_frequency,
    _validate_average_arg,
    _validate_bands,
    _warn_if_cv_arguments_are_unused,
)

TRF.__module__ = __name__
TRFDiagnostics.__module__ = __name__
CrossSpectralDiagnostics.__module__ = __name__
FrequencyResolvedWeights.__module__ = __name__
TimeFrequencyPower.__module__ = __name__
TransferFunctionComponents.__module__ = __name__
