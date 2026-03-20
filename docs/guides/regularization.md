# Regularization and Cross-Validation

`TRF.train(...)` supports both direct fixed-ridge fitting and
cross-validated regularization search.

## Scalar Ridge

Pass one scalar to fit directly:

```python
model.train(
    stimulus=stimulus,
    response=response,
    fs=fs,
    tmin=0.0,
    tmax=0.120,
    regularization=1e-3,
)
```

## Cross-Validated Search

Pass a 1D grid to evaluate multiple candidates:

```python
scores = model.train(
    stimulus=stimulus,
    response=response,
    fs=fs,
    tmin=0.0,
    tmax=0.120,
    regularization=np.logspace(-6, 0, 7),
    k=4,
    segment_duration=1.024,
    overlap=0.5,
)
```

Useful options:

- `k="loo"` enables leave-one-out cross-validation over trials.
- `average=True` reduces scores across outputs.
- `average=False` keeps one score per output channel.
- `segment_duration` is the user-friendly alternative to `segment_length`.

## Banded Regularization

If your predictor contains grouped features, provide `bands` so each group can
receive its own ridge coefficient:

```python
model.train(
    stimulus=stimulus,
    response=response,
    fs=fs,
    tmin=0.0,
    tmax=0.120,
    regularization=np.logspace(-5, 0, 5),
    bands=[1, 16],
    k=4,
)
```

## Practical Advice

- Use direct fitting when you already know a sensible ridge value.
- Use `k="loo"` when you have only a small number of trials.
- Use `k=4` or `k=5` when trial count is larger and runtime matters more.
- Use longer segments when you care about lag resolution and narrower spectral
  smoothing.
