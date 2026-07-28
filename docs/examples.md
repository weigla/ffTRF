# Examples

The repository ships plain Python scripts under `examples/`. Start with the
smallest script matching your data and move to inference or advanced spectral
features only when the scientific question requires them.

## Start Here

| Script | Use case |
| --- | --- |
| `example_single_trial_single_channel.py` | Smallest forward fit with one predictor, one output, and one trial; its score is an in-sample reconstruction diagnostic |
| `example_multi_trial_single_channel.py` | Multi-trial CV plus a genuinely held-out prediction |
| `example_multifeature_multichannel.py` | Multiple predictors and EEG-like output channels |
| `example_backward_decoding.py` | Backward decoding with `direction=-1` |
| `example_save_and_load.py` | Save, restore, and export a fitted model |

For a typical neuroscience analysis, start with
`example_multi_trial_single_channel.py`. It shows the separation between
training/CV trials and the untouched evaluation trial without adding optional
weighting or inference choices.

## Inference and Trial Quality

| Script | Use case |
| --- | --- |
| `example_bootstrap_confidence_interval.py` | Pointwise percentile intervals from trial resampling plus an independent held-out score |
| `example_permutation_significance.py` | Fixed-kernel circular-shift test on an untouched test trial, including attainable p-value resolution |
| `example_trial_weighting.py` | Controlled demonstration of inverse-variance weighting when excess variance is known to be noise |

Inverse-variance weighting is a heuristic, not a default preprocessing step.
The example deliberately injects independent noise so the source of excess
variance is known. See [Trial Weighting and Bootstrap](guides/trial-weighting-and-bootstrap.md)
before applying it to neural data.

## Advanced Estimation

| Script | Use case |
| --- | --- |
| `example_banded_regularization.py` | Separate ridge penalties for grouped predictor blocks |
| `example_multitaper_estimator.py` | DPSS multitaper fitting and spectral diagnostics |
| `example_frequency_resolved_weights.py` | Frequency-resolved lag-domain maps of a fitted kernel |

## Focused Real EEG Workflows

These scripts use the public mTRF speech-EEG sample. The first run downloads the
dataset; neither focused example requires the `mtrf` Python package.

| Script | Use case |
| --- | --- |
| `example_real_eeg_forward.py` | Short `16 speech bands -> 128 EEG channels` workflow with 7 training/CV and 3 held-out segments |
| `example_real_eeg_backward.py` | Short `128 EEG channels -> 1 envelope` decoder with the same held-out split and explicit physical lag reporting |

Both scripts report aggregate held-out results. The forward figure shows a
prespecified input/channel pair and all channel scores; the backward figure
shows the first held-out segment rather than selecting the best test segment.
To reproduce the public dataset's tutorial convention, the loader standardizes
each 12-second segment independently. Treat that as a documented choice for
this example, not a universal preprocessing template.

### Sample-Data Provenance and Integrity

The loader fetches
[`tests/data/speech_data.npy` from mTRFpy commit `9b89449c…`](https://github.com/powerfulbean/mTRFpy/blob/9b89449caaed3a4b7c80ea238a52c34a723cb8de/tests/data/speech_data.npy)
and accepts only SHA-256
`5726060e254caac865c5ca7cf56a8218937f4c05b7784fb08d11658748daee36`.
It verifies cached files on every use and writes new downloads to a temporary
file before atomically installing a digest-matching copy.

This check is especially important because the upstream `.npy` stores a Python
object and therefore uses pickle internally. ffTRF verifies the exact bytes
from the same open file and restricts decoding to the few NumPy constructors
required by this artifact. Do not bypass the helper for files from another
source. If a cache integrity check fails, remove only
`artifacts/mtrf_data/speech_data.npy` and rerun the example.

## Advanced Validation and Benchmarks

The following are development and validation programs, not first tutorials:

| Script | Use case |
| --- | --- |
| `compare_real_eeg_with_mtrf.py` | Comprehensive forward/backward ffTRF versus mTRF comparison, isolated timing, memory measurement, and plotting |
| `benchmark_real_eeg.py` | Reproduce matched and practical real-EEG benchmarks with repeated fit time, total and additional peak RSS, held-out prediction checks, and raw JSON |
| `generate_documentation_figures.py` | Regenerate the real EEG documentation gallery |
| `compare_with_mtrf.py` | Small synthetic solver comparison |
| `benchmark_runtime.py` | Synthetic crossover scenarios with repeated runtime, memory ratios, held-out prediction, and kernel-agreement checks |

These scripts expose more options and infrastructure because their purpose is
reproducibility and implementation validation.

## Running Examples

Core and inference examples:

```bash
python examples/example_multi_trial_single_channel.py
python examples/example_backward_decoding.py
python examples/example_bootstrap_confidence_interval.py
python examples/example_permutation_significance.py
```

Focused real EEG examples after a standard `pip install fftrf`:

```bash
python examples/example_real_eeg_forward.py
python examples/example_real_eeg_backward.py
```

Advanced comparison programs:

```bash
pixi run -e compare python examples/compare_real_eeg_with_mtrf.py
pixi run -e compare benchmark-demo
pixi run -e compare real-eeg-benchmark
pixi run -e compare python examples/generate_documentation_figures.py
```

The benchmark tasks limit each isolated worker to one native BLAS/OpenMP
thread, preserve every measured run in JSON, and update generated summary
blocks in the main README. Runtime ratios are hardware-dependent; regenerate
both reports from a clean revision for release-facing claims.

## Rendered Notebooks

- [Getting Started Notebook](notebooks/getting-started.ipynb)
- [Frequency-Resolved Notebook](notebooks/frequency-resolved.ipynb)

The getting-started notebook intentionally uses equal trial contribution and
the baseline spectral settings. The frequency-resolved notebook is an advanced
interpretation workflow.

## Reading Results Carefully

When working through the scripts, distinguish:

- in-sample reconstruction from held-out generalization
- CV model-selection scores from the final untouched test score
- pointwise bootstrap variability from simultaneous inference
- one prespecified channel from a channel selected after inspecting results
- user examples from benchmark and validation infrastructure

## Real EEG Validation Gallery

These figures come from the advanced comparison and documentation generator,
not from the two focused quickstarts.

### Forward Kernels

![Real EEG forward kernel comparison](images/examples/real_eeg_forward_kernels.png)

### Frequency-Resolved Kernel

![Real EEG frequency-resolved weights](images/examples/real_eeg_frequency_resolved.png)

### Backward Model

![Real EEG backward model](images/examples/real_eeg_backward_model.png)

### Kernel Agreement

![Real EEG kernel agreement summary](images/examples/real_eeg_kernel_agreement.png)
