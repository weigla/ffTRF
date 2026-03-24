# TRF Estimator

`fftrf.TRF` is the main public API of the toolbox. This page explains the
shared semantics of its parameters and attributes before showing the generated
function-by-function reference.

## Constructor

Create a forward model with:

```python
model = TRF(direction=1)
```

Create a backward model with:

```python
model = TRF(direction=-1)
```

The `metric` argument controls how predictions are scored whenever you call
`predict(...)` with observed targets or `score(...)`. It is also the criterion
used to pick the best regularization value during cross-validation. It does not
change the actual fitting objective, which remains ridge-regularized
frequency-domain TRF estimation.

The same metric also defines the observed and surrogate scores returned by
`permutation_test(...)` and `refit_permutation_test(...)`.

## Common Parameter Meanings

These parameters appear repeatedly across the API:

- `stimulus`, `response`: one trial as a NumPy array or multiple trials as a
  list of arrays
- `fs`: sampling rate in Hz used to interpret lags and frequency bins
- `tmin`, `tmax`: lag window in seconds for extracting the time-domain kernel
- `regularization`: ridge value or candidate grid; can also describe banded
  regularization
- `bands`: contiguous feature-group sizes for grouped ridge penalties
- `segment_length`: segment size in samples for spectral estimation
- `segment_duration`: segment size in seconds; a friendlier alternative to
  `segment_length`
- `overlap`: fractional overlap between neighboring segments
- `n_fft`: FFT size used when constructing sufficient statistics
- `spectral_method`: `"standard"` or `"multitaper"`
- `time_bandwidth`, `n_tapers`: DPSS settings used in multi-taper mode
- `window`: optional window applied before the FFT in standard mode
- `detrend`: optional per-segment detrending
- `k`: number of cross-validation folds or `"loo"` for leave-one-out over
  trials
- `average`: how output-channel scores are reduced
- `trial_weights`: optional weighting over trials during aggregation
- `input_index`, `output_index`: which predictor/target pair to inspect or plot

If you want a first guess for the segment-related arguments, see
[`suggest_segment_settings`](settings.md) and the
[Choosing Segment Settings](../guides/choosing-segment-settings.md) guide.

## What Training Stores

After a successful fit, the estimator keeps enough state to support prediction,
plotting, and diagnostics without re-running the fit:

- `transfer_function`: complex frequency-domain solution
- `frequencies`: frequency axis in Hz
- `weights`: lag-domain kernel
- `times`: lag axis in seconds
- `regularization`: chosen scalar ridge or banded tuple
- `regularization_candidates`: evaluated grid, if applicable
- `segment_length`, `segment_duration`, `n_fft`, `overlap`: spectral settings
- `spectral_method`, `time_bandwidth`, `n_tapers`, `window`, `detrend`:
  estimation settings
- `bootstrap_interval`, `bootstrap_level`, `bootstrap_samples`: uncertainty
  information when bootstrap intervals are computed

## Reading the Generated API

The generated reference below is the authoritative source for signatures,
arguments, return values, and stored attributes. Use the guides for conceptual
advice and the reference for exact behavior.

::: fftrf.TRF
