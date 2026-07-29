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

The whole Toolbox and its API is designed to work similar to mTRFpy:

Bialas et al., (2023). mTRFpy: A Python package for temporal response function analysis. Journal of Open Source Software, 8(89), 5657, https://doi.org/10.21105/joss.05657

The workflow behind it is loosely based on previous work from the Maddox-Lab:

Tong Shan, Ross K. Maddox; Comparing methods for deriving the auditory brainstem response to continuous speech in human listeners. Imaging Neuroscience 2025; 3 IMAG.a.19. doi: https://doi.org/10.1162/IMAG.a.19

Ross K. Maddox, Adrian K. C. Lee; Auditory Brainstem Responses to Continuous Natural Speech in Human Listeners. eNeuro 31 January 2018, 5 (1) ENEURO.0441-17.2018; DOI: https://doi.org/10.1523/ENEURO.0441-17.2018



## Why ffTRF?

Traditional time-domain TRF estimators build an explicit lagged design matrix:
each predictor is copied once per requested lag. That representation is direct
and interpretable, but it can become large when recordings are long, sampling
rates are high, lag windows are wide, or the predictor side has many channels.

`ffTRF` estimates the same class of linear stimulus–response models as conventional time-domain mTRF approaches, but does so from frequency-domain sufficient statistics. Instead of forming an explicit lagged design matrix, the model is fitted from auto- and cross-spectral estimates of the stimulus and response.

The main point here is that different spectral estimators emphasize different goals. If you are looking for the closest frequency-domain analogue of a standard finite-lag mTRF fit, whole-trial spectra are the way to go. For "real world data" (and especially in backward-models), segmented or windowed spectra are often preferable: by averaging spectral statistics across shorter segments, they can improve robustness to noise and nonstationarity, and in practice may produce kernels with higher predictive accuracy than conventional time-domain TRF estimates.

## Installation

For a released package:

```bash
pip install fftrf
```

or if you use [Pixi](https://pixi.prefix.dev/latest/installation/#install-from-source):

```bash
pixi add --pypi fftrf
```

alternatively you can point directly to this repo in your `pixi.toml`file:
```toml
[pypi-dependencies]
ffTRF = { git = "https://github.com/weigla/ffTRF.git"}
```
Then run `pixi install`. If you want to pin a specific revision, add
`rev = "<commit>"` to the dependency entry.

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

## When ffTRF Saves Computation

ffTRF avoids explicitly constructing the time-lagged predictor matrix. This can
substantially reduce peak memory and cross-validation time when the
predictor-lag dimension or regularization grid is large. It is not a universal
speed win: a small fixed-ridge model can still be faster in a time-domain
solver.

The benchmark ratios below are `mTRF / ffTRF`. Values above 1 favor ffTRF;
values below 1 favor mTRF. Fit time excludes imports, data generation/loading,
prediction, and plotting. Each method runs in a fresh process with one native
BLAS/OpenMP thread.

### Synthetic Crossover Scenarios

<!-- RUNTIME_BENCHMARK_SUMMARY_START -->
| Workload | Shape and fit | Runtime ratio (mTRF / ffTRF) | Peak RSS ratio (mTRF / ffTRF) | Correctness check |
| --- | --- | ---: | ---: | --- |
| Moderate length | 1->1, fixed | 0.36× | 1.31× | held-out r 0.9990 / 0.9990 |
| Longer lag window | 1->1, fixed | 3.01× | 1.00× | held-out r 0.9989 / 0.9989 |
| Cross-validated ridge | 1->1, cv-8 (k=4) | 9.19× | 1.00× | held-out r 0.9989 / 0.9990 |
| 102-channel backward decoder | 102->1, fixed | 21.60× | 1.80× | held-out r 0.9711 / 0.8695 |

Ratios above 1 favor ffTRF; ratios below 1 favor mTRF. The small
fixed-ridge row is included deliberately: ffTRF is not universally
faster. Savings emerge as lag count, CV work, or predictor dimension
makes explicit lag-matrix construction expensive.
<!-- RUNTIME_BENCHMARK_SUMMARY_END -->

The full
[synthetic benchmark report](https://github.com/weigla/ffTRF/blob/main/artifacts/runtime_benchmark.md)
includes ten workloads, repeated-run ranges, total and additional peak RSS,
held-out prediction, kernel agreement, raw dimensions, and complete
environment metadata.

### Real Speech-EEG Case Study

The real-data benchmark uses seven training/CV and three held-out segments from
the pinned public mTRF speech-EEG sample. It reports both the closest
whole-trial solver comparison and a practical ffTRF workflow using 2-second
Hann windows with 50% overlap.

<!-- REAL_EEG_BENCHMARK_SUMMARY_START -->
| Comparison | Direction | Runtime ratio (mTRF / ffTRF) | Peak RSS ratio (mTRF / ffTRF) | Held-out r (ffTRF / mTRF) |
| --- | --- | ---: | ---: | ---: |
| Matched whole-trial | Forward | 1.03× | 0.42× | 0.0296 / 0.0200 |
| Matched whole-trial | Backward | 63.75× | 1.17× | 0.0469 / 0.1109 |
| Practical 2 s Hann | Forward | 2.16× | 1.23× | 0.0367 / 0.0200 |
| Practical 2 s Hann | Backward | 207.69× | 5.81× | 0.1954 / 0.1109 |

Ratios above 1 favor ffTRF. Matched rows compare the closest available
solver settings. Practical rows use 2-second Hann-windowed spectra in
ffTRF and therefore compare workflows rather than identical estimators.
<!-- REAL_EEG_BENCHMARK_SUMMARY_END -->

Held-out correlation is included as a prediction check, not as evidence of
ground-truth kernel accuracy. The practical rows change ffTRF's spectral
estimator and must not be interpreted as strict solver-equivalence results.
See the full
[real EEG benchmark report](https://github.com/weigla/ffTRF/blob/main/artifacts/real_eeg_benchmark.md)
for the protocol and repeated measurements.

Reproduce both reports and synchronize these generated README tables:

```bash
pixi run -e compare benchmark-demo
pixi run -e compare real-eeg-benchmark
```

The Markdown reports are accompanied by raw JSON measurements under
`artifacts/`. Runtime depends on hardware and system load; the reports record
the source revision, package versions, platform, CPU, thread limit, and—for the
real-data benchmark—the pinned dataset commit and SHA-256.

## License

`ffTRF` is distributed under the BSD 3-Clause License.

## AI usage disclosure
This project uses AI-assisted development tools, including OpenAI Codex and OpenAI language models from version 4 onward through GPT-5.5. These tools were used to assist with generating code, composing tests, improving documentation, and reviewing code.

All AI-generated or AI-assisted contributions were reviewed, validated, and - if necessary - edited by the project author before inclusion. The author remains responsible for the correctness, design decisions, and maintenance of the codebase.
