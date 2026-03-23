# Segment Settings Helper

`ffTRF.suggest_segment_settings(...)` is a small rule-of-thumb helper for
choosing practical defaults for the standard spectral estimator.

Use it when you want a first guess for:

- `segment_duration`
- `segment_length`
- `overlap`
- `window`

It does not replace model selection or domain knowledge. It just gives a
reasonable starting point from the sampling rate, lag window, and optional
trial duration.

::: fftrf.suggest_segment_settings
