# Regularization and Cross-Validation

`TRF.train(...)` supports direct fixed-ridge fitting and cross-validated
regularization search. Cross-validation chooses a setting from the supplied
training trials; it does not replace evaluation on independent held-out data.

## Before Choosing a Ridge Grid

Ridge values are tied to the scale of the predictors and targets. Multiplying a
predictor by 100 changes the useful range of regularization values and the units
of its recovered kernel.

For multifeature models, put comparable predictors on intentional scales before
fitting. A common approach is to estimate a mean and scale from the training
data, apply that transformation to the training trials, and reuse the same
values for validation and held-out data. Do not independently learn a
normalization from every validation or test set unless per-trial normalization
is part of the scientific definition of the signal.

There is therefore no universal default ridge grid. Start broadly on a
logarithmic scale, inspect the CV curve, and extend the grid if the best value is
at an endpoint.

## Direct Fixed-Ridge Fit

Pass one scalar to fit directly:

```python
model.train(
    stimulus=train_stimulus,
    response=train_response,
    fs=fs,
    tmin=0.0,
    tmax=0.120,
    regularization=1e-3,
)
```

Use this when the value was fixed in advance, inherited from a previous
independent analysis, or is part of a frozen reproducible pipeline. In this mode
`train(...)` returns `None` because no validation run is requested.

To evaluate one fixed value with cross-validation before the final refit, set
`k` explicitly:

```python
scores = model.train(
    stimulus=train_stimulus,
    response=train_response,
    fs=fs,
    tmin=0.0,
    tmax=0.120,
    regularization=1e-3,
    k=5,
)
```

This returns a one-entry score array and then refits the final model on all
supplied training trials using the same ridge value.

## Cross-Validated Search

Pass a 1D grid to evaluate multiple candidates:

```python
grid = np.logspace(-6, 2, 9)
scores = model.train(
    stimulus=train_stimulus,
    response=train_response,
    fs=fs,
    tmin=0.0,
    tmax=0.120,
    regularization=grid,
    k=5,
)

best_index = int(np.argmax(scores))
print(grid[best_index], model.regularization)
```

In this mode:

- one validation score is computed per candidate and output reduction
- the best candidate is selected automatically
- the final model is refit on all supplied training trials
- the grid is stored in `model.regularization_candidates`
- the selected value is stored in `model.regularization`

`ffTRF` caches per-trial spectra and validation predictor FFTs across candidates
and folds. This changes the runtime, not the scores or selected solution.

## What Counts as a Fold

Cross-validation folds contain whole arrays from the supplied trial list:

- `k="loo"` or `k=-1` leaves out one trial at a time
- `k=4`, `k=5`, and similar values divide trials into that many folds
- `seed` controls the optional trial-order shuffle before folds are created

The trial list must represent scientifically defensible exchangeable or
independent units. Repeated segments from the same subject, recording, story,
or stimulus can remain highly dependent.

For one continuous recording, create blocks that respect temporal dependence
and experimental boundaries. Randomly interleaving adjacent chunks between
training and validation can inflate scores because slow neural activity,
stimulus autocorrelation, and preprocessing state are shared across folds.
Where appropriate, leave out an entire run, stimulus, session, or subject.

Preprocessing must follow the same separation. Any data-driven scaling,
feature selection, artifact threshold, or other fitted transformation should be
estimated inside the training portion of each fold when it could otherwise
leak information.

## CV Scores Are Not Final Performance

The maximum CV score is optimistic because it was used to choose the model.
Report predictive performance on data that did not participate in:

- fitting the kernel
- selecting regularization
- choosing segment or multitaper settings
- selecting predictors, channels, or lag windows

A simple workflow uses training trials for CV and a separate test set for the
final score. If all observations must contribute to both selection and
evaluation, use nested cross-validation: the inner loop selects regularization
and the outer loop estimates generalization.

## Choosing the Selection Metric

The estimator's `metric` is used during CV:

```python
model = TRF(direction=1, metric="neg_mse")
```

Choose it to match the scientific objective:

- `pearsonr` measures temporal tracking but is insensitive to multiplicative
  prediction scale and constant offsets
- `neg_mse` rewards calibrated predictions and follows the larger-is-better
  convention
- `r2` compares residual error with a constant-mean baseline and may be
  negative on difficult held-out data
- `explained_variance` focuses on residual variance and does not penalize a
  constant prediction offset

R² and explained variance coincide only when the prediction residual has zero
mean. If absolute response calibration matters, prefer R². If the analysis is
specifically about capturing fluctuations while ignoring a constant offset,
explained variance can express that question more directly. See
[Metrics](../reference/metrics.md#r2-and-explained-variance-are-not-interchangeable)
for a concrete example.

It is reasonable to select with one prespecified metric and report additional
held-out metrics, but do not choose whichever metric looks best after seeing
the test data.

## The Meaning of `average`

`average` controls reduction across output channels:

- `average=True` returns one score per candidate, averaged across all outputs
- `average=[0, 3, 5]` selects using only the listed outputs
- `average=False` returns one score per candidate and output

`average=False` is diagnostic: the final model still uses one global
regularization candidate, selected by the mean score across outputs. It does
not select a separate ridge value for each response channel. To base selection
on a prespecified channel subset, pass those indices as `average=[...]`.

Scores are first calculated per validation trial and then averaged equally
across trials. A long trial does not automatically receive more weight than a
short trial.

## How Trials and Segments Are Weighted

With `trial_weights=None`, each training trial contributes equally to the
aggregate spectral statistics. When a trial contains several FFT segments,
those segments divide that trial's contribution equally. Consequently,
unequal-length trials do not contribute in direct proportion to their sample
counts.

This behavior makes the supplied trial the unit of analysis. Use explicit trial
weights only when a different contribution is scientifically justified; see
[Trial Weighting and Bootstrap](trial-weighting-and-bootstrap.md).

## Banded Regularization

Grouped predictors can receive separate ridge coefficients:

```python
model.train(
    stimulus=train_stimulus,
    response=train_response,
    fs=fs,
    tmin=0.0,
    tmax=0.120,
    regularization=np.logspace(-5, 1, 7),
    bands=[1, 16],
    k=5,
)
```

Here the first feature is one group and the next 16 features form another.
Feature scaling remains important: banded ridge can accommodate different
penalty strengths, but it does not make arbitrary feature units comparable.
The number of band combinations also grows quickly, so use a held-out test set
and avoid treating the best inner-CV score as final evidence.

## Segment Settings Are Hyperparameters Too

Segment length, overlap, windowing, and multitaper settings change the spectral
estimator:

- longer segments provide finer frequency resolution
- shorter segments provide more local spectral estimates but less frequency
  resolution
- overlapping segments are not independent observations
- windowing can reduce leakage at the cost of changing spectral weighting

If these settings are chosen by looking at CV or test performance, they are
part of model selection and must be included in the validation design. See
[Choosing Segment Settings](choosing-segment-settings.md) for their signal
processing interpretation.

## Practical Checklist

1. Define the independent trial, run, session, stimulus, or subject unit.
2. Reserve the final test data before choosing settings.
3. Fit preprocessing only on training data where applicable.
4. Choose a metric that matches the analysis goal.
5. Search a broad log-spaced ridge grid.
6. Check that the optimum is not pinned to a grid boundary.
7. Refit on all training data with the selected value.
8. Report the untouched held-out score and the complete selection procedure.
