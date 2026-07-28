# Significance Testing

`ffTRF` provides two permutation tests for held-out prediction scores:

- `TRF.permutation_test(...)` evaluates one fitted kernel against surrogate
  alignments of the held-out target
- `TRF.refit_permutation_test(...)` retrains the complete model on each
  surrogate-aligned training set

These tests ask whether prediction exceeds a specified null alignment.
Bootstrap intervals ask a different question: how the fitted kernel varies
across resampled trials.

## Choose the Null Before Running the Test

Use `permutation_test(...)` when the inferential target is the held-out score of
an already fitted kernel. It conditions on all training and model-selection
decisions.

Use `refit_permutation_test(...)` when the null should include retraining and
regularization selection:

```python
result = model.refit_permutation_test(
    train_stimulus=train_stimulus,
    train_response=train_response,
    test_stimulus=test_stimulus,
    test_response=test_response,
    n_permutations=999,
    surrogate="circular_shift",
    min_shift=0.5,
    seed=0,
    n_jobs=4,
)
```

This is computationally slower but tests the fitted pipeline rather than only
the final fixed kernel. It reuses the stored training configuration unless
`fit_kwargs` overrides it.

## Fast Fixed-Kernel Test

```python
result = model.permutation_test(
    stimulus=test_stimulus,
    response=test_response,
    n_permutations=999,
    surrogate="circular_shift",
    min_shift=0.5,
    average=True,
    seed=0,
    n_jobs=4,
)
```

`PermutationTestResult` stores:

- `observed_score`: aligned held-out score
- `null_scores`: surrogate score distribution
- `p_value`: permutation p-value
- `z_score`: standardized distance from the null mean

The default `tail="greater"` matches all built-in metrics because larger values
are better.

## Circular-Shift Surrogates

`surrogate="circular_shift"` rolls each target trial by a random non-zero
offset. This preserves its marginal values and within-trial autocorrelation,
but it does not make every shift scientifically valid.

Consider:

- `min_shift` should be long enough to destroy plausible stimulus-response
  alignment, including slow autocorrelation and the complete lag window
- circular wrapping creates an artificial join between the end and beginning
  of a trial
- periodic or strongly structured stimuli can remain partially aligned at
  particular shifts
- filtering can lengthen temporal dependence beyond the nominal TRF window
- short trials may provide too few distinct admissible shifts

Inspect the signal autocorrelation and the distribution of allowed shifts.
State `min_shift` in the methods section rather than treating it as a generic
default.

## Trial-Shuffle Surrogates

`surrogate="trial_shuffle"` permutes complete target trials. It requires at
least two equal-length trials.

This null is appropriate only when trial identities are exchangeable under the
null. Do not freely shuffle across subjects, sessions, conditions, stimulus
items, or acquisition blocks when those labels carry structure. For a
restricted design, construct a design-aware permutation procedure outside the
built-in unrestricted shuffle.

For `refit_permutation_test(...)`, shuffling applies to training targets. The
held-out aligned test set remains the common evaluation set for the observed
and surrogate models.

## P-Value Resolution

For a greater-tail test, `ffTRF` uses
`p = (1 + count(null_score >= observed_score)) / (n_permutations + 1)`.

The smallest attainable p-value is therefore
`1 / (n_permutations + 1)`. For example, 99 permutations cannot yield
`p < 0.01`; use at least 999 permutations when that resolution matters.

Two-sided tests compare absolute deviations from the null mean. Choose the
tail in advance and interpret it together with the metric.

## Multiple Outputs and Analysis Choices

`average=False` returns one observed score and one p-value per output. These
p-values are not corrected for multiple channels. Apply an appropriate
correction or use a prespecified aggregate statistic through `average=True` or
`average=[...]`.

The same multiplicity issue applies when testing many:

- lag windows
- feature sets
- frequency bands
- preprocessing pipelines
- surrogate definitions

Selecting the best result after inspecting them invalidates the nominal
p-value unless the selection is included in the null procedure.

## Held-Out Data Still Matters

Use evaluation data that did not determine regularization, segment settings,
features, channels, lag windows, or preprocessing choices. A permutation test
does not repair circular model evaluation. The null distribution can be
computed perfectly while the observed score remains biased by prior test-set
selection.

## Reporting Template

A concise methods statement should identify:

- fixed-kernel or full-refit test
- prediction metric and output aggregation
- held-out unit and sample size
- surrogate type and exchangeability rationale
- `min_shift` and admissible shifts for circular surrogates
- number of permutations, tail, and random seed
- observed score, p-value, and minimum attainable p-value
- multiplicity correction when more than one output or hypothesis was tested

Example:

> Held-out Pearson correlation was tested with a greater-tail, fixed-kernel
> circular-shift permutation test (999 surrogates; minimum shift 0.5 s; seed
> 0). Scores were averaged across the prespecified EEG channels, giving a
> minimum attainable p-value of 0.001.

Use pointwise bootstrap intervals alongside permutation testing only when both
kernel variability and held-out predictive significance are relevant, and
describe the two inferential targets separately.
