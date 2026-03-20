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
- `pearsonr`, `r2_score`, `explained_variance_score`: built-in metrics
- `half_wave_rectify`, `resample_signal`, `inverse_variance_weights`: small
  preprocessing utilities
