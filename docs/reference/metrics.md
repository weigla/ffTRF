# Metrics

`ffTRF` scores predictions column-wise. In other words, each metric function
expects observed and predicted arrays with shape `(n_samples, n_outputs)` and
returns one score per output column before any optional averaging.

These metrics are used for:

- `predict(..., response=...)` and `score(...)`
- choosing the best regularization value during cross-validation

They are not alternative fitting objectives. The TRF itself is always fitted
with the same ridge-regularized spectral solver.

## Built-In Metrics

- `pearsonr`: default correlation-based score
- `r2_score`: coefficient of determination
- `explained_variance_score`: variance-based goodness of fit
- `neg_mse`: mTRF-compatible negative MSE where larger values are better
- `available_metrics()`: list built-in metric names accepted by `TRF(metric=...)`

## Custom Metrics

You can also pass your own callable to `TRF(metric=...)`. A custom metric must:

- accept `(y_true, y_pred)`
- return one score per output column
- use "larger is better" semantics if you want cross-validation to pick the
  best value sensibly

For compatibility with `mTRF`, `ffTRF.neg_mse` follows the same "negative MSE"
convention: larger values are still better during cross-validation, even
though the underlying quantity is the mean squared error.

::: fftrf.available_metrics

::: fftrf.pearsonr

::: fftrf.r2_score

::: fftrf.explained_variance_score

::: fftrf.neg_mse
