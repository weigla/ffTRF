# Examples

This directory contains optional demo, comparison, and benchmarking code that
is not part of the core `fft_trf` library API.

The main installable toolbox lives in:

- `src/fft_trf/`

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
- `example_backward_decoding.py`
  Backward model: multichannel responses used to reconstruct a single stimulus.
- `run_all_examples.py`
  Runs all four examples and saves the resulting figures.

Individual example commands:

```bash
python examples/example_single_trial_single_channel.py --output artifacts/examples/single_trial_single_channel.png --no-show
python examples/example_multi_trial_single_channel.py --output artifacts/examples/multi_trial_single_channel.png --no-show
python examples/example_multifeature_multichannel.py --output artifacts/examples/multifeature_multichannel.png --no-show
python examples/example_backward_decoding.py --output artifacts/examples/backward_decoding.png --no-show
python examples/run_all_examples.py --output-dir artifacts/examples --no-show
```

Run the comparison demo with Pixi:

```bash
pixi run -e compare compare-demo
pixi run -e compare benchmark-demo
pixi run -e compare examples-demo
```

Or with pip:

```bash
pip install -e ".[compare]" mtrf
python examples/compare_with_mtrf.py --output artifacts/kernel_comparison.png --no-show
python examples/benchmark_runtime.py --output artifacts/runtime_benchmark.md
python examples/run_all_examples.py --output-dir artifacts/examples --no-show
```
