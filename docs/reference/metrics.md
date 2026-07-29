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
- `r2_score`: fraction of squared error removed relative to predicting the
  observed mean
- `explained_variance_score`: fraction of target variance not left in the
  residual
- `neg_mse`: mTRF-compatible negative MSE where larger values are better
- `available_metrics()`: list built-in metric names accepted by `TRF(metric=...)`

## R² and Explained Variance Are Not Interchangeable

The two scores are equal when the residual
`y_true - y_pred` has zero mean. They differ when a model has a constant
prediction bias:

```python
import numpy as np

from fftrf import explained_variance_score, r2_score

y_true = np.arange(5.0)
y_pred = y_true + 1.0

print(explained_variance_score(y_true, y_pred))  # [1.]
print(r2_score(y_true, y_pred))                  # [0.5]
```

Explained variance is perfect here because the residual is constant and
therefore has zero variance. R² is lower because the predictions are displaced
from the observations. Use R² when absolute calibration matters. Explained
variance is useful when the scientific question concerns whether the model
captures fluctuations after disregarding a constant offset.

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
