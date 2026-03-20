# Inputs and Shapes

Most confusion in TRF workflows comes from array conventions. This page makes
those rules explicit.

## Single-Trial Inputs

Single-trial inputs can be passed as:

- a 1D array with shape `(n_samples,)`
- a 2D array with shape `(n_samples, n_features)` or `(n_samples, n_outputs)`

In a 1D input, `ffTRF` treats the array as a single feature or single output
channel.

Examples:

- stimulus envelope: `(n_samples,)`
- spectrogram-like predictor: `(n_samples, n_features)`
- EEG response: `(n_samples, n_channels)`

## Multi-Trial Inputs

Multi-trial inputs must be passed as a list of arrays:

```python
stimulus = [trial_1, trial_2, trial_3]
response = [resp_1, resp_2, resp_3]
```

Rules:

- the stimulus list and response list must have the same length
- each stimulus trial must match the corresponding response trial in sample
  count
- all predictor trials must have the same number of features
- all target trials must have the same number of outputs
- trial lengths may differ across trials

## Predictor Side vs Target Side

The meaning of `stimulus` and `response` depends on `direction`:

- `direction=1`: predictor is stimulus, target is response
- `direction=-1`: predictor is response, target is stimulus

This applies consistently across:

- `train(...)`
- `train_multitaper(...)`
- `predict(...)`
- `score(...)`
- `cross_spectral_diagnostics(...)`

## Shape Summary

| Role | Expected shape |
| --- | --- |
| Predictor trial | `(n_samples, n_features)` |
| Target trial | `(n_samples, n_outputs)` |
| Stored kernel | `(n_inputs, n_lags, n_outputs)` |
| Frequency-resolved weights | `(n_inputs, n_bands, n_lags, n_outputs)` |
| Bootstrap interval | `(2, n_inputs, n_lags, n_outputs)` |

## Trial Weights

When you pass `trial_weights`, the weight vector must contain one value per
trial in the training data.

Weights affect:

- how much each trial contributes to aggregated training spectra
- cross-validation training folds
- bootstrap estimation when enabled
- cross-spectral diagnostics if you ask those methods to reuse stored weights

Weights do not change the raw arrays themselves; they only change aggregation.

## Common Mistakes

- Passing a Python list of numbers instead of a NumPy array for one trial
- Mixing feature counts across trials
- Mixing output counts across trials
- Using `stimulus` as the predictor in a backward model
- Passing one trial to a workflow that expects bootstrap resampling across
  several trials

## Practical Advice

- Use 2D arrays whenever possible, even for single-feature or single-channel
  data, because it makes shapes visually obvious.
- Use lists of trials rather than concatenating everything into one long trial
  if you want leave-one-trial-out cross-validation or bootstrap intervals.
- If you need trial weighting, keep the natural trial boundaries intact.
