# Multitaper Estimation

`ffTRF` includes an optional DPSS multi-taper estimator for more stable
spectral estimates in noisy continuous-data settings.

## Convenience API

```python
scores = model.train_multitaper(
    stimulus=stimulus,
    response=response,
    fs=fs,
    tmin=0.0,
    tmax=0.120,
    regularization=np.logspace(-5, -1, 5),
    segment_duration=1.024,
    time_bandwidth=3.5,
    k="loo",
)
```

## Equivalent Low-Level Form

```python
scores = model.train(
    stimulus=stimulus,
    response=response,
    fs=fs,
    tmin=0.0,
    tmax=0.120,
    regularization=np.logspace(-5, -1, 5),
    spectral_method="multitaper",
    time_bandwidth=3.5,
    n_tapers=None,
    window=None,
)
```

## Notes

- `window` must stay `None` in multi-taper mode because the DPSS tapers
  already define the segment weighting.
- Larger `time_bandwidth` values allow more tapers and stronger spectral
  smoothing.
- Multi-taper fitting works with the same downstream analysis tools as the
  standard estimator: `predict`, `score`, transfer-function plots, and
  cross-spectral diagnostics.
