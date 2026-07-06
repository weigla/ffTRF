# Examples

The repository ships runnable scripts under `examples/`. They are meant to be
small, focused walkthroughs of the main API patterns. The descriptions below
explain what each script is meant to teach so you can jump straight to the
example that matches your specific question or workflow.

## Core Examples

| Script | Use case |
| --- | --- |
| `example_single_trial_single_channel.py` | Smallest forward-model example: one predictor, one output, one trial |
| `example_multi_trial_single_channel.py` | Multi-trial fit with cross-validation and explicit trial weighting |
| `example_multifeature_multichannel.py` | Multiple predictors and multiple outputs, including grid plotting |
| `example_banded_regularization.py` | Grouped predictor regularization for feature blocks with different ridge penalties |
| `example_multitaper_estimator.py` | DPSS multi-taper estimation through `train_multitaper(...)` |
| `example_frequency_resolved_weights.py` | Frequency-resolved lag-domain maps and spectrogram-like kernel views |
| `example_backward_decoding.py` | Backward decoding with `direction=-1` |
| `example_bootstrap_confidence_interval.py` | Stored bootstrap intervals and uncertainty-aware kernel plots |
| `example_trial_weighting.py` | Inverse-variance trial weighting and weighted vs unweighted fits |
| `example_save_and_load.py` | Serialization, deserialization, and impulse-response export |

## Real EEG Comparison

| Script | Use case |
| --- | --- |
| `example_mtrf_sample_eeg.py` | Public speech-EEG comparison against `mTRF`, with `neg_mse` lambda selection and held-out Pearson reporting for a forward benchmark plus a backward compressed-envelope benchmark; optional 2 s Hann settings are available for practical ffTRF forward and backward runs |
| `benchmark_real_eeg.py` | Reproduces the practical real-EEG ffTRF/mTRF benchmark and writes runtime, peak RSS, and held-out accuracy as Markdown |
| `generate_documentation_figures.py` | Regenerates the real EEG documentation figures used throughout the docs |

## Which Example Should I Start With?

- Start with `example_single_trial_single_channel.py` if you want the shortest
  possible end-to-end script.
- Start with `example_multi_trial_single_channel.py` if your real data come in
  multiple trials and you expect to use cross-validation.
- Start with `example_multifeature_multichannel.py` if your predictors are
  multi-dimensional or your response has several channels.
- Start with `example_backward_decoding.py` if your use case is decoding rather
  than forward encoding.
- Start with `example_multitaper_estimator.py` if you already know you want the
  DPSS workflow.

## Running Examples

Core examples:

```bash
python examples/example_single_trial_single_channel.py
python examples/example_multi_trial_single_channel.py
python examples/example_multitaper_estimator.py
```

Optional comparison environment:

```bash
pixi run -e compare python examples/example_mtrf_sample_eeg.py
pixi run -e compare python examples/benchmark_real_eeg.py
pixi run -e compare python examples/generate_documentation_figures.py
```

## Rendered Notebooks

If you want a more tutorial-style presentation than the plain scripts, the docs
site also renders lightweight notebooks:

- [Getting Started Notebook](../notebooks/getting-started/)
- [Frequency-Resolved Notebook](../notebooks/frequency-resolved/)

These notebooks mirror the same public API as the scripts while interleaving
code, explanation, and representative plots.

## What to Look For

When reading the examples, pay attention to:

- how single arrays differ from lists of trials
- how `direction` changes which side is treated as predictor vs target
- when `train(...)` returns `None` versus cross-validation scores, including
  fixed-ridge validation with an explicit `k`
- how the same fitted model can be inspected with lag-domain plots,
  frequency-domain diagnostics, and bootstrap intervals

## Gallery

### Real EEG Forward Kernels

![Real EEG forward kernel comparison](images/examples/real_eeg_forward_kernels.png)

### Real EEG Frequency-Resolved Kernel

![Real EEG frequency-resolved weights](images/examples/real_eeg_frequency_resolved.png)

### Real EEG Backward Model

![Real EEG backward model](images/examples/real_eeg_backward_model.png)

### Kernel Agreement Summary

![Real EEG kernel agreement summary](images/examples/real_eeg_kernel_agreement.png)
