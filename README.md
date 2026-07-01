# ffTRF
[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-orange.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Repo
size](https://img.shields.io/github/repo-size/weigla/ffTRF)](https://github.com/weigla/ffTRF)
[![Coverage Status](https://coveralls.io/repos/github/weigla/ffTRF/badge.svg?branch=main)](https://coveralls.io/github/weigla/ffTRF?branch=main)

`ffTRF` is a Python toolbox for fitting temporal response functions (TRFs) and
related linear deconvolution models in the frequency domain. It is designed for
users that work with continuous stimulus-response data, for example speech
features and MEG/EEG, and want a workflow that feels familiar if they have used e.g.
`mTRFpy` or other lag-matrix TRF tools.

The main public API is centered on `fftrf.TRF`. It supports forward encoding
models, backward decoding models, cross-validated ridge regularization,
multi-trial data, optional segmented or multi-taper spectral estimation,
prediction and scoring, bootstrap intervals, permutation tests, diagnostics,
and plotting helpers.

Full documentation: [weigla.github.io/ffTRF](https://weigla.github.io/ffTRF/)

## Why ffTRF?

Traditional time-domain TRF estimators build an explicit lagged design matrix:
each predictor is copied once per requested lag. That representation is direct
and interpretable, but it can become large when recordings are long, sampling
rates are high, lag windows are wide, regularization is cross-validated, or the
predictor side has many channels.

`ffTRF` fits the same kind of linear stimulus-response model from spectral
sufficient statistics instead. In regimes where lag matrices become expensive,
this can reduce memory use and, in many cases, runtime. The trade-off is that
the spectral estimator matters: whole-trial spectra are the closest matched
comparison to a standard finite-lag mTRF fit, while segmented/windowed spectra
are often better practical settings for noisy continuous data.

## Installation

For a released package:

```bash
pip install fftrf
```

For a local editable checkout:

```bash
pip install -e .
```

Optional extras:

```bash
pip install -e ".[compare]"  # mTRF comparisons and plotting
pip install -e ".[test]"     # test suite
pip install -e ".[docs]"     # documentation build
```

For reproducible development and benchmark runs, the repository uses Pixi:

```bash
pixi install
pixi run import-check
pixi run -e test test
```

## Quick Start

`ffTRF` uses time as the first axis. A single trial is a NumPy array with shape
`(n_samples, n_features)` or `(n_samples, n_outputs)`. Multiple trials are
passed as lists of arrays.

```python
import numpy as np
from fftrf import TRF

# Example shapes:
# stimulus_train: list of (n_samples, n_features) arrays
# response_train: list of (n_samples, n_channels) arrays
# stimulus_test:  list of held-out stimulus trials
# response_test:  list of held-out response trials

model = TRF(direction=1, metric="pearsonr")
cv_scores = model.train(
    stimulus=stimulus_train,
    response=response_train,
    fs=128,
    tmin=0.0,
    tmax=0.4,
    regularization=np.logspace(-4, 4, 17),
    k=5,
    seed=7,
)

predicted_response, heldout_r = model.predict(
    stimulus=stimulus_test,
    response=response_test,
    average=False,
)

fig, ax = model.plot(input_index=0, output_index=0)
```

For a backward decoder, use `TRF(direction=-1)`. As in `mTRF`, backward fitting
reverses the requested lag samples: a user-facing request such as
`tmin=0.0, tmax=0.4` stores physical decoder lags from -0.4 ending at zero in
`model.times`.

## Under the Hood

Instead of constructing an explicit lag matrix, `ffTRF`:

1. estimates predictor auto-spectra and predictor-target cross-spectra,
2. solves a ridge-regularized transfer function at each frequency, and
3. converts the transfer function into a lag-domain impulse response over the
   requested `tmin, tmax)` interval.

By default, each trial is treated as one FFT segment. That is the closest
setting to a standard mTRF-style finite-lag comparison:

```python
model.train(..., segment_length=None, window=None)
```

For noisy continuous data (real world data), it is often useful to estimate spectra from shorter
overlapping segments:

```python
model.train(
    ...,
    segment_duration=2.0,
    overlap=0.5,
    window="hann",
)
```

With multiple regularization candidates, `ffTRF` caches per-trial spectra so
cross-validation can reuse the FFT work across folds and lambda values. Direct
single-lambda fits use a lower-memory aggregate spectral path.

## Core Conventions

- Time is always axis 0.
- A single trial can be 1D or 2D.
- Multiple trials are represented as a list of arrays.
- `TRF(direction=1)` fits stimulus -> response.
- `TRF(direction=-1)` fits response -> stimulus.
- Stored lag-domain weights have shape `(n_inputs, n_lags, n_outputs)`.
- The lag interval is sample based and half-open: `[tmin, tmax)`.

## Real EEG Benchmark

The primary practical benchmark uses the public speech-EEG sample distributed
with the mTRF ecosystem. Reproduce the practical 2 s Hann ffTRF and mTRF rows
and write a Markdown report containing runtime, peak RSS, and held-out accuracy:

```bash
pixi run -e compare python examples/benchmark_real_eeg.py
```

The report is written to `artifacts/real_eeg_benchmark.md`. The first run
downloads the public sample data into `artifacts/mtrf_data/`.
Runtime and peak RSS depend on the machine and current system load; selected
lambdas and held-out accuracy use seeded folds and should remain stable apart
from small platform-dependent numerical differences.

The dataset contains 10 twelve-second segments sampled at 128 Hz. The benchmark
uses seven segments for training and cross-validation and three held-out
segments for evaluation. Both toolboxes use the same lag samples, lambda grids,
seeded folds, `neg_mse` selection metric, and held-out Pearson-correlation
evaluation.

Representative isolated-process results on Apple M3 with Python 3.13,
NumPy 2.4, SciPy 1.17, and mTRF 2.1.2:

### Matched mTRF-Like Configuration

This setting uses whole-trial rectangular spectra in `ffTRF`: no segmentation,
no windowing, no multitaper smoothing, and no detrending. It is the closest
parameter match to the finite-lag mTRF fit.

| Direction | Shape | ffTRF lambda | mTRF lambda | ffTRF mean r | mTRF mean r | ffTRF fit (s) | mTRF fit (s) | ffTRF RSS (MiB) | mTRF RSS (MiB) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Forward encoding | `16 -> 128` | 10000 | 3162.28 | 0.0296 | 0.0200 | 8.7038 | 3.9760 | 834.3 | 445.4 |
| Backward decoding | `128 -> 1` | 1000000 | 1000 | 0.0469 | 0.1109 | 15.0707 | 211.3287 | 3222.3 | 3910.7 |

Interpretation:

- The forward model is small enough on the predictor-lag side that mTRF remains
  faster in the strict matched comparison.
- The backward model is much harder for a time-domain lag matrix because the
  predictor side has 128 EEG channels; here `ffTRF` is much faster and uses less
  memory.
- The matched whole-trial backward `ffTRF` fit is not the most accurate
  practical setting for this small noisy sample, because cross spectra are estimated from the whole trial. Therefore `ffTRF`tends to overregularize and shows worse prediction accuracy. The more practical appraoch is shown below.

### Practical 2 s Hann Settings

For this EEG example, 2-second Hann-windowed segments with 50% overlap are more
useful practical `ffTRF` settings. They change the spectral estimator, so these
rows are not strict solver-equivalence claims, but they limit the number of
frequency bins created internally and additionally help with this kind of
continuous noisy data.

| Model | Configuration | Lambda | Mean held-out r | Median held-out r | CV fit (s) | Peak RSS (MiB) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Forward ffTRF | whole trial / rectangular | 10000 | 0.0296 | 0.0345 | 8.7038 | 834.3 |
| Forward ffTRF | 2 s / 50% overlap / Hann | 10000 | 0.0367 | 0.0386 | 2.6262 | 311.2 |
| Forward mTRF | finite-lag baseline | 3162.28 | 0.0200 | 0.0172 | 3.9760 | 445.4 |
| Backward ffTRF | whole trial / rectangular | 1000000 | 0.0469 | 0.0370 | 15.0707 | 3222.3 |
| Backward ffTRF | 2 s / 50% overlap / Hann | 1000 | 0.1954 | 0.1762 | 4.0444 | 813.9 |
| Backward mTRF | finite-lag baseline | 1000 | 0.1109 | 0.1046 | 211.3287 | 3910.7 |

- Using these settings not only helps with computation time and memory-usage, but also improves the accuracy of the resulting kernel.

For an interactive comparison with prediction plots, run:

```bash
pixi run -e compare python examples/example_mtrf_sample_eeg.py \
  --forward-segment-duration 2.0 \
  --forward-overlap 0.5 \
  --forward-window hann \
  --backward-segment-duration 2.0 \
  --backward-overlap 0.5 \
  --backward-window hann
```

## Docs

Useful entry points:

- [Getting Started](https://weigla.github.io/ffTRF/getting-started/)
- [Examples](https://weigla.github.io/ffTRF/examples/)
- [API Reference](https://weigla.github.io/ffTRF/reference/)
- [Choosing Segment Settings](https://weigla.github.io/ffTRF/guides/choosing-segment-settings/)
- [Regularization and CV](https://weigla.github.io/ffTRF/guides/regularization/)


## License

`ffTRF` is distributed under the BSD 3-Clause License.
