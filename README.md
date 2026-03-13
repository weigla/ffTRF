# fft-trf

`fft-trf` is a small Python toolbox for fitting temporal response functions
(TRFs) in the frequency domain. It is designed as a general continuous
stimulus-response modeling library rather than a toolbox tied to one modality or
one experimental paradigm.

The main branch intentionally keeps the public API focused on a single core
estimator:

- `FrequencyTRF` for forward or backward TRF fitting
- scalar or cross-validated ridge regularization
- multi-trial input via Python lists of arrays
- time-domain kernel export for interpretation
- lightweight preprocessing helpers for resampling, half-wave rectification, and
  inverse-variance trial weighting

The public workflow is intentionally close to `mTRFpy`: call `train(...)`,
inspect `weights` and `times`, then call `predict(...)`, `score(...)`, or
`plot(...)`.

## Installation

```bash
pip install -e .
```

For optional comparison plots and the runtime benchmark against `mTRFpy`:

```bash
pip install -e ".[compare]" mtrf
```

### Pixi

Pixi can use the repository directly:

```bash
pixi install
pixi run import-check
pixi run -e test test
pixi run -e compare compare-demo
pixi run -e compare benchmark-demo
```

That provides:

- a default environment with Python, NumPy, SciPy, and this package installed
  in editable mode
- a `test` environment with `pytest`
- a `compare` environment with `matplotlib` and `mTRFpy`

## Quick Example

```python
import numpy as np

from fft_trf import FrequencyTRF, inverse_variance_weights

rng = np.random.default_rng(0)
fs = 1_000

stimulus = [rng.standard_normal((8_000, 3)) for _ in range(4)]
response = [rng.standard_normal((8_000, 2)) for _ in range(4)]

model = FrequencyTRF(direction=1)
cv_scores = model.train(
    stimulus=stimulus,
    response=response,
    fs=fs,
    tmin=-0.050,
    tmax=0.250,
    regularization=np.logspace(-6, 1, 8),
    segment_length=2_048,
    overlap=0.5,
    window="hann",
    k=4,
    trial_weights=inverse_variance_weights(response),
)

prediction, score = model.predict(stimulus=stimulus, response=response)
fig, ax = model.plot(input_index=0, output_index=0)

print("selected lambda:", model.regularization)
print("cross-validation scores:", cv_scores)
print("prediction score:", score)
```

## Runnable Examples

The repository also includes simulated end-to-end examples under
`examples/`. They cover the main `FrequencyTRF` usage patterns:

- single trial, single feature, single output
- multiple trials with cross-validated regularization
- multiple stimulus features and multiple response channels
- backward decoding from multichannel responses to one stimulus

Run all of them with:

```bash
pixi run -e compare examples-demo
```

or:

```bash
python examples/run_all_examples.py --output-dir artifacts/examples --no-show
```

## What A Frequency-Domain TRF Solves

In a standard linear TRF model, the target signal is approximated as a lagged
convolution of the predictor:

```text
y(t) ≈ Στ h(τ) x(t - τ)
```

where `h(τ)` is the temporal response function or impulse response.

In a classic time-domain mTRF implementation, one explicitly builds a lagged
design matrix and solves a ridge regression problem:

```text
w = (X^T X + λI)^-1 X^T y
```

`FrequencyTRF` takes the same modeling goal but solves it in the spectral
domain. Instead of forming the full lag matrix, it estimates cross-spectra and
auto-spectra and solves, independently at each frequency bin:

```text
H(f) = (Sxx(f) + λI)^-1 Sxy(f)
```

The learned transfer function `H(f)` is then converted back to a time-domain
kernel with an inverse FFT, and the requested lag window is returned through
`weights` and `times`.

In practice the estimator does this:

1. Split each trial into one or more segments.
2. Optionally detrend and window each segment.
3. Estimate `Sxx(f)` and `Sxy(f)` across segments and trials.
4. Solve the ridge-regularized spectral system.
5. Convert the solution to a time-domain impulse response for interpretation and
   prediction.

## Relationship To Standard mTRF Fits

`FrequencyTRF` is deliberately similar to a standard time-domain mTRF API, but
it is not exactly the same estimator in every setting.

Shared ideas:

- same forward (`direction=1`) and backward (`direction=-1`) modeling concept
- same ridge-regularization concept
- same notion of a time-domain kernel exposed as `weights`
- same prediction and scoring workflow

Important differences:

- the fit is based on spectral statistics rather than an explicit lag matrix
- `segment_length`, `overlap`, and `window` control how the spectra are
  estimated
- `tmin` and `tmax` define the impulse-response window extracted after fitting
  the frequency response

If you want the closest comparison to a standard lag-matrix ridge fit, start
with:

```python
segment_length=None
window=None
```

If you want a smoothed spectral-estimation workflow for long continuous signals,
shorter overlapping segments and a window such as `"hann"` are often useful.

## When This Formulation Is Useful

The frequency-domain formulation becomes attractive when one or more of these is
true:

- recordings are long and continuous
- the sampling rate is high
- the lag window contains many samples
- explicit lag matrices would be large or inconvenient to materialize
- you want to work naturally with segment-wise spectral estimates

For short recordings and small lag windows, a standard time-domain TRF can still
be faster. The benchmark below shows that crossover clearly.

## Practical Meaning Of The Main Parameters

- `direction`: `1` fits a forward model from predictor to target; `-1` fits a
  backward model.
- `tmin`, `tmax`: lag window reported in the time-domain kernel.
- `regularization`: ridge value, or a sequence of candidate values for
  cross-validation.
- `segment_length`: segment size used for spectral estimation. `None` means one
  segment per trial.
- `overlap`: fractional overlap between adjacent segments.
- `window`: optional segment window. `None` is the most direct mTRF-like
  comparison setting.
- `trial_weights`: optional per-trial weighting. The helper
  `inverse_variance_weights(...)` provides a simple weighting scheme when trial
  quality varies.

## Runtime Benchmark Against mTRFpy

The repository includes a reproducible runtime benchmark in
`examples/benchmark_runtime.py`. It compares `FrequencyTRF` to a standard
time-domain `mTRFpy` fit using the same lag window and fixed ridge value.

Run it with:

```bash
pixi run -e compare benchmark-demo
```

Benchmark environment used for the table below:

- CPU: Apple M1 Pro
- Platform: macOS 15.5 arm64
- Python: 3.13.12
- NumPy: 2.4.2
- SciPy: 1.17.1
- mTRFpy: 2.1.2
- 3 timed repetitions after 1 warmup run

All scenarios use a forward single-feature / single-output model with
`segment_length=None` and `window=None`, which is the closest mTRF-like
configuration.

| Scenario | fs (Hz) | Trials | Samples/trial | Lags | Approx. lag matrix (MiB) | FrequencyTRF median fit (s) | mTRFpy median fit (s) | mTRFpy / FrequencyTRF | Kernel corr. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Moderate length, 1 kHz | 1000 | 8 | 4096 | 40 | 10.0 | 0.0111 | 0.0086 | 0.77x | 1.0000 |
| Long recording, 1 kHz | 1000 | 4 | 60000 | 40 | 73.2 | 0.1490 | 0.0412 | 0.28x | 1.0000 |
| High rate, 10 kHz | 10000 | 2 | 30000 | 300 | 137.3 | 0.0859 | 0.1274 | 1.48x | 1.0000 |
| Long high rate, 10 kHz | 10000 | 2 | 60000 | 300 | 274.7 | 0.1463 | 0.2758 | 1.89x | 1.0000 |

Interpretation:

- At modest lag counts, the standard time-domain fit is still faster on this
  machine.
- As the lag count grows with sampling rate, the frequency-domain formulation
  becomes more competitive and eventually faster.
- In the table above, values below `1.0x` mean `mTRFpy` was faster; values
  above `1.0x` mean `FrequencyTRF` was faster.
- Kernel correlations of `1.0000` in these synthetic tests indicate that both
  approaches recover essentially the same filter under the matched settings used
  here.
- The approximate lag-matrix size is included because it is the dominant memory
  object for a standard time-domain fit and scales linearly with both recording
  length and lag count.

For a JOSS-style evaluation, the next natural step would be to extend this
benchmark grid across:

- multiple numbers of input features and output channels
- longer lag windows
- cross-validated regularization
- segmented spectral estimation settings
- peak memory usage in addition to wall-clock time

## API Summary

### `FrequencyTRF`

Core estimator. Important attributes after fitting:

- `transfer_function`: complex spectral mapping with shape
  `(n_frequencies, n_inputs, n_outputs)`
- `frequencies`: frequency vector in Hz
- `weights`: time-domain kernel with shape `(n_inputs, n_lags, n_outputs)`
- `times`: lag vector in seconds
- `regularization`: selected ridge value after fitting

Important methods:

- `train(...)`
- `predict(...)`
- `score(...)`
- `plot(...)`
- `to_impulse_response(...)`
- `save(...)` / `load(...)`

### Preprocessing Helpers

- `half_wave_rectify(x)`: split a waveform into positive and negative
  half-waves
- `resample_signal(x, orig_fs, target_fs)`: polyphase resampling helper
- `inverse_variance_weights(trials)`: normalized inverse-variance trial weights

## Optional Comparison Tools

The installable toolbox lives under `src/fft_trf/`. Optional validation and
benchmarking utilities live under `examples/` so the main package stays focused
on core fitting functionality.

Available example entry points:

- `examples/compare_with_mtrf.py`: side-by-side kernel comparison against
  time-domain ridge and `mTRFpy`
- `examples/benchmark_runtime.py`: reproducible runtime benchmark against
  `mTRFpy`

## Scope

The main branch is intentionally a focused frequency-domain TRF toolbox. It
does not impose a modality-specific workflow, a fixed preprocessing pipeline, or
domain-specific feature extraction. Those pieces are better kept in user or
project-level code built on top of the core estimator.
