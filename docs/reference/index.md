# Reference

This section contains the detailed API documentation for the public `ffTRF`
surface.

## Start Here

- Read [TRF Estimator](trf.md) for the main class and its shared parameter
  semantics.
- Read [Result Containers](results.md) for the objects returned by the
  frequency-resolved, transfer-function, and diagnostic helpers.
- Read [Metrics](metrics.md) for built-in scoring functions and custom-metric
  requirements.
- Read [Segment Settings Helper](settings.md) for the public rule-of-thumb
  helper that suggests `segment_duration`, `overlap`, and `window`.
- Read [Preprocessing Helpers](preprocessing.md) for small signal-preparation
  utilities shipped with the package.

## Public API Summary

- `TRF`: train, predict, score, inspect, plot, diagnose, save, and load models
- `TRFDiagnostics` / `CrossSpectralDiagnostics`: observed-vs-predicted spectral
  diagnostics
- `FrequencyResolvedWeights`: frequency-by-lag decomposition of the kernel
- `TimeFrequencyPower`: spectrogram-like power view of the kernel
- `TransferFunctionComponents`: magnitude, phase, and group delay for one
  transfer-function slice
- `pearsonr`, `r2_score`, `explained_variance_score`, `neg_mse`: built-in metrics
- `suggest_segment_settings`: helper for choosing practical segment and window
  defaults for the standard estimator
- `half_wave_rectify`, `resample_signal`, `inverse_variance_weights`: small
  preprocessing utilities
