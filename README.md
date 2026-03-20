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

Representative results on Apple M3, Python 3.13, NumPy 2.4, SciPy 1.17, and
mTRF 2.1:

| Scenario | ffTRF fit (s) | mTRF fit (s) | Speedup | ffTRF peak RSS (MiB) | mTRF peak RSS (MiB) | ffTRF held-out r | mTRF held-out r |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Long high rate (`fs=10 kHz`, `60k` samples/trial, `300` lags) | 0.2827 | 0.3018 | 1.07x | 104.7 | 540.4 | 0.9990 | 0.9990 |
| Longer lag window (`600` lags) | 0.1429 | 0.3478 | 2.43x | 100.2 | 542.1 | 0.9989 | 0.9989 |
| Cross-validated ridge (`8` lambdas, `k=4`) | 0.1660 | 1.1965 | 7.21x | 105.8 | 367.6 | 0.9989 | 0.9990 |
| Segmented Hann estimate (`4096`-sample segments, `50%` overlap) | 0.0240 | 0.2828 | 11.76x | 99.6 | 539.4 | 0.9989 | 0.9990 |
| EEG-scale forward model (`16 -> 102`) | 0.0547 | 0.0832 | 1.52x | 163.3 | 220.6 | 0.9450 | 0.9293 |
| 102-channel backward decoder (`102 -> 1`) | 0.3029 | 3.0364 | 10.02x | 355.6 | 1086.8 | 0.9813 | 0.8695 |

The benchmark outcome is not "ffTRF is always faster." In the small fixed-ridge
1-to-1 cases, mTRF can be comparable or faster. The main pattern is that ffTRF
pulls ahead once lag count, channel count, CV grid size, or segmented spectral
workflows get heavy, and the memory advantage becomes much clearer in those
same regimes. In the current benchmark runs, the most pronounced gains are the
cross-validated ridge case (`7.21x`), the segmented Hann workflow (`11.76x`),
and the 102-channel backward decoder (`10.02x`), all while preserving very
similar or better held-out accuracy.

## Real EEG Comparison

The repository also includes a comparison on the official speech-EEG sample
dataset used by the mTRF ecosystem:

```bash
pixi run -e compare python examples/example_mtrf_sample_eeg.py
```

On the current run with 7 training segments, 3 held-out test segments, a
5-fold CV grid over 17 lambdas, and a 0 to 400 ms forward lag window:

| Dataset | ffTRF selected lambda | mTRF selected lambda | ffTRF mean held-out r | mTRF mean held-out r | ffTRF median held-out r | mTRF median held-out r | ffTRF CV fit (s) | mTRF CV fit (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Official speech EEG sample (`16 -> 128`, `fs=128 Hz`) | 0.0001 | 10000 | 0.0235 | 0.0185 | 0.0253 | 0.0147 | 15.1023 | 2.7373 |

That real-data example is useful as a sanity check rather than a pure runtime
benchmark. In this setting, ffTRF produced better held-out channel
correlations, while mTRF completed the CV fit faster. This is consistent with
the synthetic benchmark story above: ffTRF's biggest performance wins show up
most clearly once the lag-matrix burden becomes more extreme.

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

rng = np.random.default_rng(0)
fs = 1_000

stimulus = [rng.standard_normal((8_000, 3)) for _ in range(4)]
response = [rng.standard_normal((8_000, 2)) for _ in range(4)]

model = TRF(direction=1)
cv_scores = model.train(
    stimulus=stimulus,
    response=response,
    fs=fs,
    tmin=-0.050,
    tmax=0.250,
    regularization=np.logspace(-6, 1, 8),
    segment_duration=2.048,
    overlap=0.5,
    window="hann",
    k="loo",
    trial_weights=inverse_variance_weights(response),
)

prediction, score = model.predict(stimulus=stimulus, response=response)
fig, ax = model.plot(input_index=0, output_index=0)
```

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
