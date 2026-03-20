# Trial Weighting and Bootstrap

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

Bootstrap resampling is trial-based, so it requires at least two trials.
