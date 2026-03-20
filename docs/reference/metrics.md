# Metrics

`ffTRF` scores predictions column-wise. In other words, each metric function
expects observed and predicted arrays with shape `(n_samples, n_outputs)` and
returns one score per output column before any optional averaging.

## Built-In Metrics

- `pearsonr`: default correlation-based score
- `r2_score`: coefficient of determination
- `explained_variance_score`: variance-based goodness of fit
- `available_metrics()`: list built-in metric names accepted by `TRF(metric=...)`

## Custom Metrics

You can also pass your own callable to `TRF(metric=...)`. A custom metric must:

- accept `(y_true, y_pred)`
- return one score per output column
- use "larger is better" semantics if you want cross-validation to pick the
  best value sensibly

::: fftrf.available_metrics

::: fftrf.pearsonr

::: fftrf.r2_score

::: fftrf.explained_variance_score
