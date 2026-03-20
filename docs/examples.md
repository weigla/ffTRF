# Examples

The repository ships runnable scripts under `examples/`. They are meant to be
small, focused walkthroughs of the main API patterns.

## Core Examples

| Script | Use case |
| --- | --- |
| `example_single_trial_single_channel.py` | Single-trial forward model |
| `example_multi_trial_single_channel.py` | Multi-trial fit with cross-validation |
| `example_multifeature_multichannel.py` | Multiple predictors and outputs |
| `example_banded_regularization.py` | Grouped predictor regularization |
| `example_multitaper_estimator.py` | DPSS multi-taper estimation |
| `example_frequency_resolved_weights.py` | Frequency-resolved lag-domain maps |
| `example_backward_decoding.py` | Backward decoding |
| `example_bootstrap_confidence_interval.py` | Stored bootstrap intervals |
| `example_trial_weighting.py` | Inverse-variance trial weighting |
| `example_save_and_load.py` | Serialization and impulse-response export |

## Comparison and Benchmarking

| Script | Use case |
| --- | --- |
| `compare_with_mtrf.py` | Synthetic kernel comparison against time-domain references |
| `example_mtrf_sample_eeg.py` | Public speech-EEG comparison |
| `benchmark_runtime.py` | Runtime benchmark against `mTRFpy` |

## Running Examples

Core examples:

```bash
python examples/example_single_trial_single_channel.py
python examples/example_multi_trial_single_channel.py
python examples/example_multitaper_estimator.py
```

Optional comparison environment:

```bash
pixi run -e compare compare-demo
pixi run -e compare benchmark-demo
pixi run -e compare python examples/example_mtrf_sample_eeg.py
```

## Gallery

![Single-trial example](images/examples/single_trial_single_channel.png)

![Multifeature / multichannel example](images/examples/multifeature_multichannel_kernels.png)

![Multi-taper example](images/examples/multitaper_estimator.png)
