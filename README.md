# ffTRF

`ffTRF` is a Python toolbox for fitting temporal response functions in the
frequency domain. It is designed for continuous stimulus-response modeling with
a small public API centered on `fftrf.TRF`.

The full documentation is hosted at
[weigla.github.io/ffTRF](https://weigla.github.io/ffTRF/), with dedicated pages
for:

- [Getting Started](https://weigla.github.io/ffTRF/getting-started/)
- [Examples](https://weigla.github.io/ffTRF/examples/)
- [API Reference](https://weigla.github.io/ffTRF/reference/)
- [Development](https://weigla.github.io/ffTRF/development/)

## Real EEG Benchmark

The primary practical benchmark uses the official speech-EEG sample dataset
from the mTRF ecosystem:

```bash
pixi run -e compare python examples/example_mtrf_sample_eeg.py
```

It contains 10 twelve-second segments sampled at 128 Hz. Seven segments are
used for training and cross-validation, and the final three are held out. Both
toolboxes use the same lag samples, lambda grids, seeded folds, `neg_mse`
selection metric, and held-out Pearson-correlation evaluation.

- Forward encoding predicts 128 EEG channels from the 16-band speech
  spectrogram using lags from 0 to approximately 400 ms.
- Backward decoding reconstructs a one-dimensional compressed speech-envelope
  proxy from 128 EEG channels. The requested 0 to 350 ms window becomes the
  mTRF-compatible physical decoder window from -343.75 to 0 ms.
- The matched ffTRF baseline uses whole-trial rectangular estimation without
  segmentation, windowing, multitaper smoothing, or detrending.

Results from a representative isolated run on Apple M3 with Python 3.13,
NumPy 2.4, SciPy 1.17, and mTRF 2.1.2:

### Matched Configuration

| Direction | ffTRF lambda | mTRF lambda | ffTRF mean held-out r | mTRF mean held-out r | ffTRF median held-out r | mTRF median held-out r |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Forward (`16 -> 128`) | 10000 | 3162.28 | 0.0296 | 0.0200 | 0.0345 | 0.0172 |
| Backward (`128 -> 1`) | 1000000 | 1000 | 0.0469 | 0.1109 | 0.0370 | 0.1046 |

| Direction | ffTRF CV fit (s) | mTRF CV fit (s) | Speedup | ffTRF peak RSS (MiB) | mTRF peak RSS (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Forward (`16 -> 128`) | 8.7038 | 3.9760 | 0.46x | 834.3 | 445.4 |
| Backward (`128 -> 1`) | 15.0707 | 211.3287 | 14.02x | 3222.3 | 3910.7 |

The forward models have similarly low held-out correlations on this small,
noisy sample, with ffTRF slightly higher on this split while mTRF is faster and
uses less memory. In the matched backward comparison, ffTRF is much faster and
uses less peak memory, but it does **not** currently reproduce mTRF's held-out
accuracy.

### Why the Forward Model Also Benefits from 2-Second Hann Segments

The whole-trial forward row above is the closest mTRF comparison, but it is not
the most efficient practical ffTRF setting. Each 12-second trial contributes an
FFT with about `769` positive-frequency bins. A 2-second segment uses `256`
samples and about `129` bins, so cross-validation solves and validation
predictions repeat substantially less frequency-domain work. The overlapping
Hann segments also provide more averaged spectral observations on this noisy
sample.

On the same train/test split, the practical forward setting improves the
resource profile while preserving the same lambda search and held-out
evaluation:

| Forward configuration | Selected lambda | Mean held-out r | Median held-out r | CV fit (s) | Peak RSS (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| ffTRF, whole trial / rectangular | 10000 | 0.0296 | 0.0345 | 8.7038 | 834.3 |
| ffTRF, 2 s / 50% overlap / Hann | 10000 | 0.0367 | 0.0386 | 2.6262 | 311.2 |
| mTRF, finite-lag baseline | 3162.28 | 0.0200 | 0.0172 | 3.9760 | 445.4 |

As with the segmented backward setting, this is not a strict solver-equivalence
claim because the spectral estimator has changed. Reproduce the focused
forward run with:

```bash
pixi run -e compare python examples/example_mtrf_sample_eeg.py \
  --skip-backward \
  --forward-segment-duration 2.0 \
  --forward-overlap 0.5 \
  --forward-window hann
```

### Why the Backward Model Uses 2-Second Hann Segments

The whole-trial setting above is the closest parameter match, but it is a poor
spectral estimator for this particular backward problem. ffTRF solves one
`128 x 128` EEG covariance system at every frequency. With whole-trial
estimation, each of the seven training trials contributes only one observation
per frequency, so each covariance has rank at most seven before ridge
regularization. Cross-validation folds contain even fewer training trials.
This makes the high-dimensional decoder strongly underdetermined.

mTRF does not have the same per-frequency sample limitation. It directly fits
the requested 45-lag finite impulse response using the time-domain lag matrix.
ffTRF instead estimates an unrestricted frequency response and extracts the
requested lag interval afterward. That difference is small in the forward
model, but important for this `128 -> 1` backward model.

For the practical ffTRF backward fit, the trials are therefore split into
overlapping 2-second segments. This supplies many more spectral observations
per frequency, while the Hann window reduces spectral leakage at the segment
boundaries. On the same train/test split:

| Backward configuration | Selected lambda | Mean held-out r | Median held-out r | CV fit (s) | Peak RSS (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| ffTRF, whole trial / rectangular | 1000000 | 0.0469 | 0.0370 | 15.0707 | 3222.3 |
| ffTRF, 2 s / 50% overlap / Hann | 1000 | 0.1954 | 0.1762 | 4.0444 | 813.9 |
| mTRF, finite-lag baseline | 1000 | 0.1109 | 0.1046 | 211.3287 | 3910.7 |

The 2-second Hann configuration is consequently the recommended practical
setting for this ffTRF backward example: it is both more accurate and less
resource-intensive than the whole-trial frequency estimate. It is still a
different spectral estimator from mTRF, so this result demonstrates practical
performance rather than strict solver equivalence. Reproduce it with:

```bash
pixi run -e compare python examples/example_mtrf_sample_eeg.py \
  --backward-segment-duration 2.0 \
  --backward-overlap 0.5 \
  --backward-window hann
```

## Controlled mTRF Validation

One of the main reasons `ffTRF` exists is to avoid explicit lag-matrix
construction in the regimes where that becomes expensive: high sample rates,
long lag windows, cross-validated ridge grids, segmented spectral estimation,
and high-dimensional forward or backward models.

The benchmark in [`examples/benchmark_runtime.py`](examples/benchmark_runtime.py)
compares `ffTRF` and `mTRF` on identical simulated data with the same lag
samples, ridge values, trial splits, and held-out evaluation data. The primary
rows below use whole-trial rectangular estimation in ffTRF: no short segments,
no Hann window, and no multitaper smoothing. They isolate solver agreement
under known ground truth.

Each fit is measured in an isolated process. The table reports median runtime
over 3 repetitions after 1 warmup, peak RSS, held-out Pearson correlation, and
the correlation between the recovered kernel banks. Regenerate the complete
report with `pixi run -e compare benchmark-demo`.

Representative results on the same system:

| Scenario | ffTRF fit (s) | mTRF fit (s) | ffTRF peak RSS (MiB) | mTRF peak RSS (MiB) | ffTRF held-out r | mTRF held-out r | Kernel r |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Long high rate (`fs=10 kHz`, `60k` samples/trial, `300` lags) | 0.2926 | 0.2661 | 103.0 | 540.8 | 0.9990 | 0.9990 | 1.0000 |
| Longer lag window (`600` lags) | 0.1456 | 0.3466 | 100.5 | 543.4 | 0.9989 | 0.9989 | 1.0000 |
| Cross-validated ridge (`8` lambdas, `k=4`) | 0.1655 | 1.1971 | 105.7 | 367.9 | 0.9989 | 0.9990 | 1.0000 |
| EEG-scale forward model (`16 -> 102`) | 0.0553 | 0.0830 | 162.9 | 231.0 | 0.9450 | 0.9293 | 0.9884 |

These controlled rows show that ffTRF can recover essentially the same known
kernels and held-out predictions as mTRF. Small fixed-ridge problems can still
favor mTRF on runtime, but ffTRF avoids constructing the full lag matrix, so
its memory use grows more gently and it becomes faster as lag count,
cross-validation work, or dimensionality increases.

### Optional Spectral Estimation

Short overlapping segments, windows, and multitaper estimation are useful
ffTRF features, but they change the spectral estimator and are not a strict
mTRF comparison:

| Optional ffTRF setting | ffTRF fit (s) | mTRF baseline fit (s) | ffTRF held-out r | mTRF held-out r |
| --- | ---: | ---: | ---: | ---: |
| `4096`-sample segments, `50%` overlap, Hann window | 0.0239 | 0.2906 | 0.9989 | 0.9990 |

This row demonstrates the segmented workflow and its computational cost; it is
not used as evidence that the two solvers are configured identically.

## Installation

Pixi is the primary supported development workflow:

```bash
pixi install
pixi run import-check
pixi run -e test test
```

For a lightweight editable install:

```bash
pip install -e .
```

Optional extras:

```bash
pip install -e ".[test]"
pip install -e ".[compare]"
pip install -e ".[docs]"
```

For docs builds, prefer the locked Pixi environment:

```bash
pixi run -e docs docs-build
```

For an existing Pixi project, you can link `ffTRF` directly from GitHub via
Pixi's `pypi-dependencies`:

```toml
[pypi-dependencies]
fftrf = { git = "https://github.com/weigla/ffTRF" }
```

Then run:

```bash
pixi install
```

If you want to pin a specific revision, add `rev = "<commit>"` to that table
entry.

## Quick Example

```python
import numpy as np

from fftrf import TRF, inverse_variance_weights

def simulate_trial(
    rng: np.random.Generator,
    *,
    n_samples: int,
    kernel: np.ndarray,
    noise_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    stimulus = rng.standard_normal((n_samples, 1))
    response = np.convolve(stimulus[:, 0], kernel, mode="full")[:n_samples]
    response += noise_scale * rng.standard_normal(n_samples)
    return stimulus, response[:, np.newaxis]


rng = np.random.default_rng(0)
fs = 512
kernel = np.zeros(60)
kernel[6] = 1.0
kernel[18] = -0.4
kernel[32] = 0.2

trials = [simulate_trial(rng, n_samples=4_096, kernel=kernel, noise_scale=0.05) for _ in range(6)]
stimulus = [trial_stimulus for trial_stimulus, _ in trials]
response = [trial_response for _, trial_response in trials]

model = TRF(direction=1)
cv_scores = model.train(
    stimulus=stimulus[:-1],
    response=response[:-1],
    fs=fs,
    tmin=0.0,
    tmax=kernel.shape[0] / fs,
    regularization=np.logspace(-6, 0, 7),
    segment_duration=1.0,
    overlap=0.5,
    window="hann",
    k="loo",
    trial_weights=inverse_variance_weights(response[:-1]),
)

prediction = model.predict(stimulus=stimulus[-1])
score = model.score(stimulus=stimulus[-1], response=response[-1])
fig, ax = model.plot(input_index=0, output_index=0)
```

This example uses a known simulated kernel and keeps the last trial held out,
so `score` is a real generalization check rather than a training-set-only
sanity check.

## Examples

Runnable demos live in [`examples/`](examples/README.md). Useful entry points:

```bash
python examples/example_single_trial_single_channel.py
python examples/example_multi_trial_single_channel.py
python examples/example_multitaper_estimator.py
python examples/example_frequency_resolved_weights.py
```

Optional comparison tools:

```bash
pixi run -e compare compare-demo
pixi run -e compare benchmark-demo
```
