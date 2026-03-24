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

## Performance

One of the main reasons `ffTRF` exists is to avoid explicit lag-matrix
construction in the regimes where that becomes expensive: high sample rates,
long lag windows, cross-validated ridge grids, segmented spectral estimation,
and high-dimensional forward or backward models.

The benchmark in [`examples/benchmark_runtime.py`](examples/benchmark_runtime.py)
compares `ffTRF` against `mTRF` on identical simulated data, measures
median fit time over 3 timed repetitions after 1 warmup run, records per-fit
peak RSS in isolated worker processes, and reports held-out Pearson
correlation on a separate simulation split. The latest full report generated
with `pixi run -e compare benchmark-demo` is stored in
[`artifacts/runtime_benchmark.md`](artifacts/runtime_benchmark.md).
Cross-validated `TRF` fits now also batch validation-time prediction by caching
predictor FFTs within each fold, so the CV-heavy rows better reflect the
current implementation rather than the older per-kernel convolution path.

Representative results on Apple M3, Python 3.13, NumPy 2.4, SciPy 1.17, and
mTRF 2.1:

| Scenario | ffTRF fit (s) | mTRF fit (s) | Speedup | ffTRF peak RSS (MiB) | mTRF peak RSS (MiB) | ffTRF held-out r | mTRF held-out r |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Long high rate (`fs=10 kHz`, `60k` samples/trial, `300` lags) | 0.2880 | 0.2420 | 0.84x | 103.5 | 539.5 | 0.9990 | 0.9990 |
| Longer lag window (`600` lags) | 0.1438 | 0.3356 | 2.33x | 99.9 | 542.5 | 0.9989 | 0.9989 |
| Cross-validated ridge (`8` lambdas, `k=4`) | 0.1631 | 1.1293 | 6.92x | 108.3 | 367.7 | 0.9989 | 0.9990 |
| Segmented Hann estimate (`4096`-sample segments, `50%` overlap) | 0.0242 | 0.2890 | 11.96x | 99.7 | 539.8 | 0.9989 | 0.9990 |
| EEG-scale forward model (`16 -> 102`) | 0.0547 | 0.0736 | 1.35x | 161.9 | 224.4 | 0.9450 | 0.9293 |
| 102-channel backward decoder (`102 -> 1`) | 0.3082 | 3.0998 | 10.06x | 355.4 | 1148.2 | 0.9813 | 0.8695 |

The benchmark outcome is not "ffTRF is always faster." In the small fixed-ridge
1-to-1 cases, mTRF can be comparable or faster. The main pattern is that ffTRF
pulls ahead once lag count, channel count, CV grid size, or segmented spectral
workflows get heavy, and the memory advantage becomes much clearer in those
same regimes. In the current benchmark runs, the most pronounced gains are the
cross-validated ridge case (`6.92x`), the segmented Hann workflow (`11.96x`),
and the 102-channel backward decoder (`10.06x`), all while preserving very
similar or better held-out accuracy. The improved CV row is especially
relevant for current `ffTRF`: validation predictors are now transformed once
per fold and reused across lambda candidates, which lowers CV scoring cost
without changing the selected model or the reported scores.

## Real EEG Comparison

The repository also includes a comparison on the official speech-EEG sample
dataset used by the mTRF ecosystem:

```bash
pixi run -e compare python examples/example_mtrf_sample_eeg.py
```

On the current run with 7 training segments, 3 held-out test segments, a
5-fold CV grid over 17 lambdas, and a 0 to 400 ms forward lag window,
regularization is selected with the mTRF-compatible `neg_mse` metric while
the table below still reports held-out Pearson `r`:

| Dataset | ffTRF selected lambda | mTRF selected lambda | ffTRF mean held-out r | mTRF mean held-out r | ffTRF median held-out r | mTRF median held-out r | ffTRF CV fit (s) | ffTRF peak RSS (MiB) | mTRF CV fit (s) | mTRF peak RSS (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Official speech EEG sample (`16 -> 128`, `fs=128 Hz`) | 10000 | 3162.28 | 0.0296 | 0.0200 | 0.0345 | 0.0172 | 14.2327 | 735.6 | 7.3751 | 423.0 |

That real-data example is useful as a sanity check rather than a pure runtime
benchmark. In this setting, ffTRF produced better held-out channel
correlations, while mTRF completed the forward CV fit faster and with lower
peak RSS on the current run. The reported fit times and peak RSS values come
from isolated worker processes so they are not inflated by plotting or previous
fits in the same Python process.

The same script also includes an additional backward comparison on the same
train/test split: EEG is used to reconstruct a broadband speech-envelope proxy
built from the mean raw 16-band stimulus, compressed with exponent `0.4`, and
then z-scored per segment. As above, lambda selection uses `neg_mse` while
held-out Pearson `r` is reported separately. To keep the real-data example
practical to run, that backward part uses a lighter default setup (`15`
lambdas from `1e-8` to `1e6`, `k=3`, `0 to 350 ms`) and configures the ffTRF
fit with `segment_duration=2.0`, `overlap=0.5`, and `window="hann"`:

| Dataset | ffTRF selected lambda | mTRF selected lambda | ffTRF mean held-out r | mTRF mean held-out r | ffTRF median held-out r | mTRF median held-out r | ffTRF CV fit (s) | ffTRF peak RSS (MiB) | mTRF CV fit (s) | mTRF peak RSS (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Backward envelope reconstruction (`128 -> 1`, `fs=128 Hz`, `0 to 350 ms`) | 100000 | 1000 | 0.0536 | 0.1109 | 0.0850 | 0.1046 | 6.3356 | 766.3 | 452.0112 | 4083.4 |

The backward comparison is substantially heavier than the forward one on this
sample, so the reduced grid/fold defaults are intentional.

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
