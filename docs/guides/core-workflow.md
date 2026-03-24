# Core Workflow

This page describes the intended lifecycle of a typical `ffTRF` analysis from
raw arrays to a saved model.

## 1. Organize Your Data

`ffTRF` expects matched stimulus/response pairs.

- For single-trial data, pass a single NumPy array.
- For multi-trial data, pass a list of arrays.
- Within one fit, each stimulus trial and its matching response trial must have
  the same number of samples.
- Across trials, sample counts may differ.

The predictor side should be shaped `(n_samples, n_features)` and the target
side `(n_samples, n_outputs)`.

## 2. Choose Modeling Direction

Construct the estimator with:

- `TRF(direction=1)` for forward encoding: stimulus predicts response
- `TRF(direction=-1)` for backward decoding: response predicts stimulus

This choice affects how `train(...)`, `predict(...)`, `score(...)`, and the
diagnostic methods interpret `stimulus` and `response`.

## 3. Fit the Model

The main entry point is `train(...)`.

Use a direct fit when you already know the ridge value:

```python
model.train(
    stimulus=stimulus,
    response=response,
    fs=fs,
    tmin=0.0,
    tmax=0.150,
    regularization=1e-3,
)
```

Use cross-validation when you want the model to select a ridge value:

```python
scores = model.train(
    stimulus=stimulus,
    response=response,
    fs=fs,
    tmin=0.0,
    tmax=0.150,
    regularization=np.logspace(-6, 0, 7),
    k="loo",
    segment_duration=1.024,
    overlap=0.5,
)
```

Important fit-time choices:

- `tmin` / `tmax`: lag window of the recovered kernel
- `regularization`: scalar ridge or CV grid
- `segment_length` / `segment_duration`: spectral segment size
- `overlap`: segment reuse
- `bands`: grouped feature regularization
- `trial_weights`: how much each trial contributes

## 4. Inspect the Fitted Kernel

After fitting, the instance stores:

- `model.transfer_function`: complex frequency-domain solution
- `model.frequencies`: corresponding frequency axis
- `model.weights`: lag-domain kernel
- `model.times`: lag axis in seconds

Use:

- `plot(...)` for a single input/output pair
- `plot_grid(...)` for a full grid
- `to_impulse_response(...)` to extract a different lag window without refitting

## 5. Predict and Score

Use:

- `predict(...)` when you want the predicted signals
- `score(...)` when you only want the metric
- `permutation_test(...)` when you want a surrogate-null significance check
  for a held-out score
- `refit_permutation_test(...)` when you want the stronger retrain-and-score
  null

If you pass observed targets to `predict(...)`, it returns both predictions and
the score.

## 6. Inspect Frequency-Domain Behavior

The model can also be explored spectrally:

- `transfer_function_at(...)`
- `transfer_function_components_at(...)`
- `plot_transfer_function(...)`
- `cross_spectral_diagnostics(...)`
- `plot_coherence(...)`
- `plot_cross_spectrum(...)`

These methods are especially useful when you care about gain, phase,
coherence, or whether the model reproduces the observed output spectrum.

## 7. Quantify Uncertainty and Significance

If you have multiple trials, you can estimate a trial-bootstrap confidence
interval:

- during fitting with `bootstrap_samples=...`
- after fitting with `bootstrap_confidence_interval(...)`

The stored interval can then be shown in kernel plots or extracted with
`bootstrap_interval_at(...)`.

If you want to know whether a held-out prediction score beats a surrogate null
distribution, use `permutation_test(...)` on the evaluation data.

If you want the null to include retraining and regularization selection, use
`refit_permutation_test(...)` with explicit train/test splits.

## 8. Save or Copy the Model

- `save(...)` serializes the estimator with pickle
- `load(...)` restores a saved estimator state
- `copy(...)` creates an in-memory deep copy

## A Good First Workflow

For many users, a good default sequence is:

1. start with `TRF(direction=1)`
2. fit with a small CV grid over ridge values
3. inspect one kernel with `plot(...)`
4. compute held-out performance with `score(...)`
5. inspect coherence with `plot_coherence(...)`
6. save the trained model
