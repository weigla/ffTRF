# Examples

Use the rendered notebooks when learning a workflow. They split the analysis
into short executable steps and explain which scientific question each option
addresses. The plain Python scripts under `examples/` remain useful for
copying into pipelines and for CI smoke tests.

Related scripts are grouped into one notebook when the scientific decisions
belong together. This keeps the notebook navigation extensive without turning
it into a second API reference.

## Choose a Notebook

| Question | Notebook |
| --- | --- |
| How do I fit and evaluate my first forward model? | [Getting Started](notebooks/getting-started.ipynb) |
| How do predictor features map onto several EEG channels? | [Multiple Features and Channels](notebooks/multifeature-multichannel.ipynb) |
| How do I reconstruct a stimulus from multichannel responses? | [Backward Decoding](notebooks/backward-decoding.ipynb) |
| How should I select scalar or feature-banded ridge penalties? | [Regularization](notebooks/regularization.ipynb) |
| When is multitaper spectral estimation useful? | [Multitaper Estimation](notebooks/multitaper.ipynb) |
| Which frequencies contribute to different parts of a kernel? | [Frequency-Resolved Analysis](notebooks/frequency-resolved.ipynb) |
| How do I inspect gain, phase, coherence, and cross-spectra? | [Spectral Diagnostics](notebooks/diagnostics.ipynb) |
| How do I quantify kernel uncertainty or test held-out tracking? | [Uncertainty and Significance](notebooks/uncertainty-and-significance.ipynb) |
| When are trial weights scientifically defensible? | [Trial Weighting](notebooks/trial-weighting.ipynb) |
| How do I save, restore, and export a fitted kernel? | [Persistence](notebooks/persistence.ipynb) |

For a typical neuroscience analysis, begin with Getting Started. Continue only
to the notebook connected to a prespecified analysis question.

## Runnable Script Counterparts

| Goal | Scripts |
| --- | --- |
| Basic forward workflows | `example_single_trial_single_channel.py`, `example_multi_trial_single_channel.py` |
| Multivariate and backward models | `example_multifeature_multichannel.py`, `example_backward_decoding.py` |
| Regularization and spectral estimation | `example_banded_regularization.py`, `example_multitaper_estimator.py` |
| Interpretation and diagnostics | `example_frequency_resolved_weights.py` |
| Inference and data quality | `example_bootstrap_confidence_interval.py`, `example_permutation_significance.py`, `example_trial_weighting.py` |
| Persistence | `example_save_and_load.py` |

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
and checks SHA-256.
It verifies cached files on every use and writes new downloads to a temporary
file before atomically installing a digest-matching copy.

If a cache integrity check fails, remove only
`artifacts/mtrf_data/speech_data.npy` and rerun the example.

## Comparing ffTRF With mTRF

Toolbox agreement, real-EEG validation figures, and benchmark interpretation
have a dedicated page: [Comparison with mTRF](comparison-with-mtrf.md).
That page distinguishes matched solver comparisons from ffTRF workflows that
use a different spectral estimator.

## Validation and Benchmark Programs

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

## Reading Results Carefully

When working through the scripts, distinguish:

- in-sample reconstruction from held-out generalization
- CV model-selection scores from the final untouched test score
- pointwise bootstrap variability from simultaneous inference
- one prespecified channel from a channel selected after inspecting results
- user examples from benchmark and validation infrastructure
