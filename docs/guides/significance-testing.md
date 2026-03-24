# Significance Testing

`ffTRF` supports two layers of permutation-based significance testing for
held-out prediction scores:

- `TRF.permutation_test(...)`: fast score-level null using one fitted model
- `TRF.refit_permutation_test(...)`: slower refit null that retrains the full
  model on surrogate-aligned training data

This answers a different question than bootstrap confidence intervals:

- bootstrap asks how stable the recovered kernel is across trials
- permutation testing asks whether the held-out prediction score is larger than
  expected under a surrogate null alignment

## Which Method Should I Use?

Use `permutation_test(...)` when:

- you want the fastest practical null model
- you are comfortable conditioning on the already fitted model
- you mainly want to know whether the held-out score beats a surrogate score
  distribution for that fixed kernel

Use `refit_permutation_test(...)` when:

- you want a stronger null that includes retraining
- you used cross-validation and want that model-selection step inside the null
- runtime is acceptable for many repeated refits

## Quick Example

```python
result = model.permutation_test(
    stimulus=test_stimulus,
    response=test_response,
    n_permutations=1000,
    surrogate="circular_shift",
    min_shift=0.5,
    average=False,
    seed=0,
    n_jobs=4,
)
```

The returned `PermutationTestResult` stores:

- `observed_score`: the aligned held-out score
- `null_scores`: the surrogate null distribution
- `p_value`: permutation p-value
- `z_score`: standardized score relative to the null mean and variance

## Stronger Refit Null

```python
result = model.refit_permutation_test(
    train_stimulus=train_stimulus,
    train_response=train_response,
    test_stimulus=test_stimulus,
    test_response=test_response,
    n_permutations=100,
    surrogate="circular_shift",
    min_shift=0.5,
    seed=0,
    n_jobs=4,
)
```

This method trains one model on the original training alignment and one fresh
model for each surrogate alignment, then scores all of them on the same
held-out test set.

By default it reuses the most recent training configuration stored on the
estimator, but disables bootstrap estimation and progress output during the
surrogate refits for speed.

## Supported Surrogates

### `circular_shift`

This rolls each evaluation target trial by a random non-zero offset.

Use it when:

- you have one long continuous evaluation trial
- trial lengths vary
- you want to preserve within-trial autocorrelation and amplitude distribution

`min_shift` is given in seconds. Increase it when you want to avoid
near-aligned shifts for slowly varying signals.

### `trial_shuffle`

This permutes whole evaluation target trials.

Use it when:

- you have at least two evaluation trials
- all evaluation trials have the same sample count
- trial identity is the natural exchangeable unit

This null is often easier to explain than circular shifts, but it is only valid
when trial boundaries are meaningful and exchangeable.

For `refit_permutation_test(...)`, `trial_shuffle` applies to the training
target trials, so those training trials must also have equal sample counts.

## How to Read the Result

- lower p-values mean the observed score is more extreme than the surrogate
  null
- the default `tail="greater"` is the natural choice for the built-in metrics,
  because larger values are better in `ffTRF`
- `average=False` keeps one p-value per output channel
- aggregated scores use the same `average` rules as `score(...)`

## Practical Advice

- Start with `surrogate="circular_shift"` for continuous data.
- Use `trial_shuffle` when you have clean repeated trials of equal length.
- Keep held-out evaluation data separate from the training data when you want a
  genuine generalization test.
- Start with `permutation_test(...)` and move to
  `refit_permutation_test(...)` only when you need the stronger training-pipeline
  null.
- Use bootstrap intervals and permutation tests together when you care about
  both kernel stability and predictive significance.
