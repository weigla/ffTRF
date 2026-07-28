# Preprocessing Helpers

`ffTRF` intentionally keeps preprocessing lightweight. The helpers below are
small utilities for common signal-preparation tasks that often come up before
TRF fitting.

## Typical Use Cases

- split a waveform into positive and negative half-wave regressors
- resample derived regressors to match the target sampling rate
- compute heuristic inverse-variance trial weights in controlled cases where
  total variance is a defensible proxy for noise

These helpers do not try to replace a full preprocessing pipeline. They are
meant to cover the small but common operations that are convenient to keep next
to the estimator API. In particular, inverse-variance weighting is not a
general data-quality estimator: genuine neural effects can change trial
variance too. Compare it with an unweighted fit and prefer prespecified
artifact or noise measures when available.

::: fftrf.half_wave_rectify

::: fftrf.resample_signal

::: fftrf.inverse_variance_weights
