# Preprocessing Helpers

`ffTRF` intentionally keeps preprocessing lightweight. The helpers below are
small utilities for common signal-preparation tasks that often come up before
TRF fitting.

## Typical Use Cases

- split a waveform into positive and negative half-wave regressors
- resample derived regressors to match the target sampling rate
- compute simple inverse-variance trial weights for noisy multi-trial data

These helpers do not try to replace a full preprocessing pipeline. They are
meant to cover the small but common operations that are convenient to keep next
to the estimator API.

::: fftrf.half_wave_rectify

::: fftrf.resample_signal

::: fftrf.inverse_variance_weights
