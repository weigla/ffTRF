# Examples

This directory contains optional demo, comparison, and benchmarking code that
is not part of the core `ffTRF` library API.

The main installable toolbox lives in:

- `src/fftrf/`

The files here are intended for:

- runnable usage examples for the main `FrequencyTRF` API patterns
- sanity checks against time-domain references
- side-by-side comparisons with `mTRFpy`
- runtime benchmarking
- exploratory plotting for development and validation

## Example coverage

The example scripts are organized around the main modeling combinations that are
practical with `FrequencyTRF`:

- `example_single_trial_single_channel.py`
  Single trial, single stimulus feature, single response channel, forward model.
- `example_multi_trial_single_channel.py`
  Multiple trials, single feature, single channel, cross-validated
  regularization.
- `example_multifeature_multichannel.py`
  Multiple stimulus features, multiple response channels, forward model.
- `example_banded_regularization.py`
  Optional banded ridge search over grouped multifeature predictors.
- `example_multitaper_estimator.py`
  Optional multi-taper estimation with `R^2`, transfer-function plots, cross spectra, and coherence.
- `example_frequency_resolved_weights.py`
  Spectrogram-like frequency-resolved view of one recovered kernel.
- `example_alpha_plus_erp.py`
  Event-related response that mixes an ERP-like component with a later alpha burst.
- `example_backward_decoding.py`
  Backward model: multichannel responses used to reconstruct a single stimulus.
- `example_bootstrap_confidence_interval.py`
  Forward model with a stored bootstrap confidence interval.
- `example_trial_weighting.py`
  Compare an unweighted fit to a fit that uses inverse-variance trial weights.
- `example_save_and_load.py`
  Save a fitted model, load it again, and export a different lag window.

Each example is intentionally a plain Python script showing the API calls,
learned attributes, and one corresponding figure. Running a script prints the
relevant `FrequencyTRF` attributes and saves a figure under `artifacts/examples/`.
They are meant to be read alongside the main README: each script follows the
same pattern of `train(...)`, attribute inspection, `predict(...)`, and
plotting, but focuses on one concrete use case. The optional features are
covered with dedicated examples rather than changing the baseline examples into
advanced-only workflows. The examples now also demonstrate friendlier options
such as `segment_duration=...` in seconds and `k="loo"` for leave-one-out
cross-validation.

Example commands:

```bash
python examples/example_single_trial_single_channel.py
python examples/example_multi_trial_single_channel.py
python examples/example_multifeature_multichannel.py
python examples/example_banded_regularization.py
python examples/example_multitaper_estimator.py
python examples/example_frequency_resolved_weights.py
python examples/example_alpha_plus_erp.py
python examples/example_backward_decoding.py
python examples/example_bootstrap_confidence_interval.py
python examples/example_trial_weighting.py
python examples/example_save_and_load.py
```

Run the comparison demo with Pixi:

```bash
pixi run -e compare compare-demo
pixi run -e compare benchmark-demo
```

Or with pip:

```bash
pip install -e ".[compare]" mtrf
python examples/compare_with_mtrf.py --output artifacts/kernel_comparison.png --no-show
python examples/benchmark_runtime.py --output artifacts/runtime_benchmark.md
```
