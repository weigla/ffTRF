# Examples

The scripts in this directory are organized by user goal. The installable
toolbox itself lives in `src/fftrf/`.

## Start here

- `example_single_trial_single_channel.py`: smallest forward fit; its prediction
  score is an in-sample reconstruction diagnostic.
- `example_multi_trial_single_channel.py`: multi-trial CV and an untouched test
  trial.
- `example_multifeature_multichannel.py`: multiple predictors and outputs.
- `example_backward_decoding.py`: response-to-stimulus decoding.
- `example_save_and_load.py`: persistence and lag-window export.

For most workflows, begin with the multi-trial example. It keeps
equal trial contribution and clearly separates training/CV data from final
evaluation.

## Inference and trial quality

- `example_bootstrap_confidence_interval.py`: pointwise percentile bootstrap
  intervals and an independent held-out score.
- `example_permutation_significance.py`: fixed-kernel circular-shift test on an
  untouched trial.
- `example_trial_weighting.py`: inverse-variance weighting in a controlled
  simulation where added noise is known. Total variance is not a general neural
  data-quality measure.

## Advanced estimation

- `example_banded_regularization.py`: grouped predictor penalties.
- `example_multitaper_estimator.py`: DPSS multitaper estimation.
- `example_frequency_resolved_weights.py`: lag-frequency views of a fitted
  kernel.

## Focused real EEG workflows

- `example_real_eeg_forward.py`: public speech bands to multichannel EEG with
  seven training/CV and three held-out segments.
- `example_real_eeg_backward.py`: multichannel EEG to a compressed broadband
  envelope on the same held-out split.

The dataset is downloaded on first use.
Cached files are rechecked before decoding, and incomplete or mismatched
downloads are never installed. The upstream `.npy` uses a pickle-backed object
representation, so the helper also restricts decoding to the few NumPy
constructors this file requires. Do not bypass the check or substitute an
untrusted file.

A standard ffTRF installation includes the plotting dependency, and these
scripts do not import the optional `mtrf` package.

```bash
python examples/example_real_eeg_forward.py
python examples/example_real_eeg_backward.py
```

## Advanced validation and benchmarks

- `compare_real_eeg_with_mtrf.py`: comprehensive forward/backward toolbox
  comparison, plotting, timing, and memory measurement.
- `benchmark_real_eeg.py`: repeated isolated-process matched and practical
  real-EEG benchmark, with Markdown and raw JSON output.
- `generate_documentation_figures.py`: real-data gallery generator.
- `compare_with_mtrf.py`: small synthetic reference comparison.
- `benchmark_runtime.py`: synthetic crossover scenarios with correctness,
  repeated runtime, and total plus additional peak-memory checks.
- `benchmark_utils.py`: shared provenance, thread-control, and reporting helpers.
- `comparison_utils.py`: shared synthetic comparison helpers.

These files are validation infrastructure rather than introductory tutorials:

```bash
pixi run -e compare python examples/compare_real_eeg_with_mtrf.py
pixi run -e compare benchmark-demo
pixi run -e compare real-eeg-benchmark
pixi run -e compare python examples/generate_documentation_figures.py
```

The two benchmark tasks also synchronize their generated summary blocks in the
main README. Run them from a clean revision before publishing release claims.

Each user example prints the relevant fitted state and saves a figure under
`artifacts/examples/`. Read the inference examples alongside the corresponding
documentation guides so the resampling unit, null model, and multiplicity
assumptions remain explicit.
