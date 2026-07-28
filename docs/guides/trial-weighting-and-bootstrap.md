# Trial Weighting and Bootstrap

Trial weighting controls how trials contribute to the fitted spectral
statistics. Trial bootstrap resampling uses those same trial boundaries to
describe uncertainty in the recovered kernel. They are related computationally
but answer different scientific questions.

## Default: Equal Trial Contribution

Without explicit weights, every trial contributes equally. If trials contain
different numbers of FFT segments, the segments within each trial divide that
trial's contribution equally. A longer trial therefore does not automatically
dominate a shorter one.

This is usually the clearest starting point when the supplied trials are the
units intended to contribute equally to the analysis.

## Explicit Quality Weights

Pass one non-negative value per training trial when you have a prespecified
quality measure:

```python
quality_weights = np.asarray([...], dtype=float)

model.train(
    stimulus=stimulus,
    response=response,
    fs=fs,
    tmin=0.0,
    tmax=0.250,
    regularization=1e-3,
    trial_weights=quality_weights,
)
```

Weights change the aggregation of per-trial auto- and cross-spectra. They also
affect the training part of CV folds and weighted bootstrap fits. They do not
rescale samples within the original arrays.

Prefer weights derived from an analysis-independent quality measure, such as a
prespecified artifact score, sensor-noise estimate, or noise-only time window.
Document how the weights were computed and inspect whether conclusions depend
on the weighting strategy.

## Inverse-Variance Weighting Is a Heuristic

`inverse_variance_weights(response)` and
`trial_weights="inverse_variance"` use total target variance as a proxy for
trial noise:

```python
from fftrf import inverse_variance_weights

weights = inverse_variance_weights(response)
```

This can be useful in a controlled situation where greater variance is known to
come from added noise. In neuroscience data, however, variance can also reflect
genuine evoked responses, state changes, gain differences, or experimental
effects. The heuristic can then downweight biologically meaningful trials.

Do not use it automatically, and do not compute quality weights from a held-out
test set. Compare weighted and unweighted fits, report the rule, and prefer
artifact rejection when a trial is not scientifically usable.

## Trial Bootstrap Intervals

Store an interval while training:

```python
model.train(
    stimulus=stimulus,
    response=response,
    fs=fs,
    tmin=0.0,
    tmax=0.250,
    regularization=1e-3,
    bootstrap_samples=1000,
    bootstrap_level=0.95,
    bootstrap_seed=0,
)
```

Or compute one after fitting:

```python
interval, times = model.bootstrap_confidence_interval(
    stimulus=stimulus,
    response=response,
    n_bootstraps=1000,
    level=0.95,
    seed=0,
)
```

The implementation samples trials with replacement, refits the transfer
function from the resampled cached spectra, and takes percentile quantiles of
the resulting kernels. It requires at least two trials.

`interval` has shape `(2, n_inputs, n_lags, n_outputs)`: the first axis contains
the lower and upper bounds and the remaining axes match `model.weights`.

## What the Interval Does and Does Not Mean

The stored bounds are **pointwise percentile intervals**. Each lag, input, and
output is summarized separately.

They are not:

- a simultaneous confidence band over the complete kernel
- corrected for testing many lags, features, or channels
- evidence that a kernel component is significantly different from zero
- a subject- or population-level interval unless trials are genuinely the
  exchangeable population units relevant to that claim

The bootstrap holds the fitted spectral configuration and selected
regularization fixed. It therefore does not include uncertainty from ridge
selection, segment-setting selection, predictor selection, or other upstream
analysis choices.

Natural trials must also be exchangeable for the intended inference. Repeated
epochs nested within one subject do not by themselves support a population
claim across subjects. For a single continuous recording, arbitrary sample
resampling is invalid; define defensible blocks or use a time-series-specific
resampling design outside the built-in trial bootstrap.

## Recommended Reporting

Report:

- the resampling unit and number of independent units
- the number of bootstrap resamples and random seed
- whether intervals are pointwise or simultaneous
- the percentile level
- the fixed regularization and spectral settings
- any trial-weighting rule
- whether inference is within-recording, within-subject, or across subjects

Use [Significance Testing](significance-testing.md) when the question is whether
held-out prediction exceeds a surrogate null. Bootstrap intervals instead
describe variability of the fitted kernel across the supplied resampling units.
