# Trial Weighting and Bootstrap

This page covers two related ideas:

- not all trials need to contribute equally during fitting
- trial structure can be reused to estimate uncertainty with bootstrap
  resampling

## Trial Weighting

If some trials are much noisier than others, either pass explicit weights or
use the provided inverse-variance helper.

```python
from fftrf import inverse_variance_weights

weights = inverse_variance_weights(response)

model.train(
    stimulus=stimulus,
    response=response,
    fs=fs,
    tmin=0.0,
    tmax=0.250,
    regularization=np.logspace(-4, 1, 6),
    trial_weights=weights,
)
```

You can also use `trial_weights="inverse_variance"` directly in `train(...)`.

## What Trial Weights Actually Do

Weights affect how per-trial spectra are aggregated.

That means they change:

- the effective contribution of each trial during fitting
- the training part of each cross-validation fold
- bootstrap resampling when intervals are estimated
- optional diagnostics if those diagnostics reuse the stored weighting strategy

Weights do not rescale the original arrays sample by sample.

## When Weighting Helps

Trial weighting is most useful when:

- some trials clearly have much larger noise variance
- artifact rejection is too aggressive a solution
- you want to keep all trials but reduce the influence of the worst ones

## Bootstrap Confidence Intervals

Store a bootstrap interval during training:

```python
model.train(
    stimulus=stimulus,
    response=response,
    fs=fs,
    tmin=0.0,
    tmax=0.250,
    regularization=1e-3,
    bootstrap_samples=200,
    bootstrap_level=0.95,
)
```

Or compute it afterwards on a fitted model:

```python
interval, times = model.bootstrap_confidence_interval(
    stimulus=stimulus,
    response=response,
    n_bootstraps=200,
    level=0.95,
)
```

Then access it with:

```python
interval, times = model.bootstrap_interval_at()
```

## How to Read the Interval

- the first axis of `interval` contains lower and upper bounds
- the remaining axes match the stored kernel shape:
  `(n_inputs, n_lags, n_outputs)`
- the interval reflects variability across trials, not across individual
  samples

## Important Limitation

Bootstrap resampling is trial-based, so it requires at least two trials.

If you only have one continuous recording, you can still fit the model, but a
trial-bootstrap interval is not meaningful in the same way.

## Practical Advice

- Use weighting when trial quality varies a lot.
- Use bootstrap intervals when you want uncertainty estimates on the recovered
  kernel.
- Keep your natural trial boundaries intact if you plan to use weighting or
  bootstrap later.
