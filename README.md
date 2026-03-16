# ffTRF

`ffTRF` is a small Python toolbox for fitting temporal response functions
(TRFs) in the frequency domain. It is designed as a general continuous
stimulus-response modeling library rather than a toolbox tied to one modality or
one experimental paradigm.

The main branch intentionally keeps the public API focused on a single core
estimator:

- `FrequencyTRF` for forward or backward TRF fitting
- scalar or cross-validated ridge regularization
- optional banded ridge regularization for grouped predictor features
- built-in scoring metrics such as Pearson correlation, `R^2`, and explained variance
- multi-trial input via Python lists of arrays
- time-domain kernel export for interpretation
- cached spectral statistics for faster regularization search
- optional DPSS multi-taper estimation through `train_multitaper(...)`
- optional trial-bootstrap confidence intervals
- single-kernel and full-kernel-grid plotting
- frequency-resolved spectrotemporal kernel maps
- spectrogram-like time-frequency power derived from band-limited kernels
- direct transfer-function magnitude/phase/group-delay tools and observed-vs-predicted cross-spectral diagnostics
- lightweight preprocessing helpers for resampling, half-wave rectification, and
  inverse-variance trial weighting

The public workflow is intentionally close to `mTRFpy`: call `train(...)`,
inspect `weights` and `times`, then call `predict(...)`, `score(...)`, or
`plot(...)`. Optional features such as banded regularization, trial weighting,
and bootstrap intervals are only activated when their dedicated arguments are
provided, so the default usage remains a straightforward "mTRF in Fourier
space" workflow.

The primary Python import is now `fftrf`. The older `fft_trf` import path is
still available as a compatibility alias.

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

from fftrf import FrequencyTRF, inverse_variance_weights

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
    segment_duration=2.048,
    overlap=0.5,
    window="hann",
    k="loo",
    show_progress=True,
    trial_weights=inverse_variance_weights(response),
)

prediction, score = model.predict(stimulus=stimulus, response=response)
fig, ax = model.plot(input_index=0, output_index=0)

print("selected lambda:", model.regularization)
print("cross-validation scores:", cv_scores)
print("prediction score:", score)
print("segment duration (s):", model.segment_duration)
```

### Quick Frequency-Resolved Example

```python
from fftrf import FrequencyTRF

model = FrequencyTRF(direction=1)
model.train(
    stimulus=train_stimulus,
    response=train_response,
    fs=fs,
    tmin=0.0,
    tmax=0.120,
    regularization=1e-2,
    segment_length=4096,
    overlap=0.5,
    window=None,
)

resolved = model.frequency_resolved_weights(
    n_bands=18,
    fmax=160.0,
    value_mode="real",
)
fig, ax = model.plot_frequency_resolved_weights(resolved=resolved)
```

This produces a lag-by-frequency view of one recovered kernel while keeping the
ordinary time-domain kernel available in `model.weights`. The shipped demo uses
an event-related response with a time-locked alpha burst so the resolved map
shows a clear low-frequency oscillatory component at a specific latency.

## Runnable Examples

The repository also includes simulated end-to-end examples under
`examples/`. They cover the main `FrequencyTRF` usage patterns:

- single trial, single feature, single output
- multiple trials with cross-validated regularization
- multiple stimulus features and multiple response channels
- optional banded regularization for grouped multifeature predictors
- optional multi-taper estimation with transfer and cross-spectral diagnostics
- frequency-resolved weights for a spectrogram-like kernel view
- backward decoding from multichannel responses to one stimulus
- bootstrap confidence intervals for one recovered kernel
- trial weighting with `inverse_variance_weights(...)`
- model serialization with `save(...)`, `load(...)`, and `to_impulse_response(...)`

Each example is a simple Python script that shows how the API is called, which
attributes are available after fitting, and what a typical visualization looks
like. For example:

```bash
python examples/example_multi_trial_single_channel.py
python examples/example_banded_regularization.py
python examples/example_multitaper_estimator.py
python examples/example_frequency_resolved_weights.py
python examples/example_bootstrap_confidence_interval.py
python examples/example_trial_weighting.py
python examples/example_save_and_load.py
```

## Example Gallery

Each script in `examples/` is meant to double as a small API walkthrough. The
figures below are generated by the shipped example scripts.

### Single-Trial Forward Model

`examples/example_single_trial_single_channel.py`

![Single-trial forward model](docs/images/examples/single_trial_single_channel.png)

### Multi-Trial Cross-Validation

`examples/example_multi_trial_single_channel.py`

![Multi-trial forward model with cross-validation](docs/images/examples/multi_trial_single_channel.png)

### Multifeature / Multichannel Kernel Grid

`examples/example_multifeature_multichannel.py`

![Multifeature and multichannel kernel grid](docs/images/examples/multifeature_multichannel_kernels.png)

### Optional Banded Regularization

`examples/example_banded_regularization.py`

![Banded regularization example](docs/images/examples/banded_regularization.png)

### Optional Multi-Taper Estimator

`examples/example_multitaper_estimator.py`

![Multi-taper estimator example](docs/images/examples/multitaper_estimator.png)

### Frequency-Resolved Weights (Time-Locked Alpha Burst)

`examples/example_frequency_resolved_weights.py`

![Frequency-resolved weights example](docs/images/examples/frequency_resolved_weights.png)

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
- `metric`: scoring function used by `predict(...)` and `score(...)`. Built-in
  names include `"pearsonr"`, `"r2"`, and `"explained_variance"`.
- `tmin`, `tmax`: lag window reported in the time-domain kernel.
- `regularization`: ridge value, or a sequence of candidate values for
  cross-validation. If `bands` is provided, scalar grids are expanded into a
  banded search over grouped coefficients.
- `bands`: optional contiguous predictor-group sizes for banded ridge. Leaving
  this as `None` keeps ordinary scalar ridge behavior.
- `segment_length`: segment size used for spectral estimation in samples.
- `segment_duration`: user-friendly alternative to `segment_length` in seconds.
  `None` means one segment per trial.
- `overlap`: fractional overlap between adjacent segments.
- `spectral_method`: `"standard"` for the default segment-wise FFT estimator or
  `"multitaper"` for DPSS multi-taper averaging.
- `time_bandwidth`, `n_tapers`: optional multi-taper settings.
- `window`: optional segment window. `None` is the most direct mTRF-like
  comparison setting. It must remain `None` in multi-taper mode.
- `trial_weights`: optional per-trial weighting. The helper
  `inverse_variance_weights(...)` provides a simple weighting scheme when trial
  quality varies.
- `k`: cross-validation fold count. Use `-1` or `"loo"` for leave-one-out
  over trials.
- `show_progress`: optional stderr progress indicator for multi-candidate CV
  runs.

## Optional Advanced Features

The default `FrequencyTRF` fit is still the simplest Fourier-domain analogue of
a standard ridge-regularized mTRF fit:

```python
model.train(
    stimulus=stimulus,
    response=response,
    fs=fs,
    tmin=0.0,
    tmax=0.250,
    regularization=1e-3,
    segment_length=None,
    window=None,
)
```

Everything below is opt-in.

### Built-In Metrics

`FrequencyTRF` accepts either a custom metric callable or one of the built-in
metric names:

```python
from fftrf import FrequencyTRF, available_metrics, explained_variance_score

print(available_metrics())

model = FrequencyTRF(direction=1, metric="r2")
model.train(...)
prediction, held_out_r2 = model.predict(stimulus=test_stimulus, response=test_response)

# You can also evaluate additional metrics explicitly.
held_out_ev = explained_variance_score(test_response, prediction)
```

### Banded Regularization

If different predictor groups should receive different shrinkage, provide
`bands`. The values define contiguous feature groups in the order they appear in
the predictor matrix. For example, one envelope feature followed by a
16-channel spectrogram would use `bands=[1, 16]`.

```python
regularization_grid = np.logspace(-4, 1, 6)

model = FrequencyTRF(direction=1)
cv_scores = model.train(
    stimulus=stimulus,
    response=response,
    fs=fs,
    tmin=0.0,
    tmax=0.250,
    regularization=regularization_grid,
    bands=[1, 16],
    k="loo",
    show_progress=True,
)

print(model.regularization)           # selected per-band coefficients
print(model.feature_regularization)   # expanded per-feature penalty vector
print(model.regularization_candidates[:3])
```

With `bands` enabled, a 1D scalar grid follows the same convention as
`mTRFpy`: the Cartesian product across feature groups is cross-validated. If
you want to test only specific combinations, pass explicit tuples such as
`[(1e-4, 1e-2), (1e-3, 1e-1)]`.


### Optional Multi-Taper Estimator

The default `spectral_method="standard"` uses one FFT per segment, optionally
after applying a segment window such as `"hann"`. If you want smoother and
usually more stable spectral estimates, use the dedicated multi-taper training
path:

```python
model.train_multitaper(
    stimulus=stimulus,
    response=response,
    fs=fs,
    tmin=0.0,
    tmax=0.250,
    regularization=np.logspace(-5, -1, 5),
    segment_duration=1.024,
    overlap=0.5,
    time_bandwidth=3.5,
    n_tapers=4,
    show_progress=True,
)
```

Internally this uses the same estimator class, but with DPSS tapers replacing
the ordinary segment window. The lower-level `spectral_method="multitaper"`
option is still available on `train(...)` when you want one unified entry
point.

### Transfer-Function And Cross-Spectral Diagnostics

The fitted model always stores the complex transfer function. You can inspect
it directly, derive magnitude/phase/group delay, and compare predictions and
observations in the frequency domain:

```python
diagnostics = model.cross_spectral_diagnostics(
    stimulus=test_stimulus,
    response=test_response,
)

frequencies, transfer = model.transfer_function_at(input_index=0, output_index=0)
components = model.transfer_function_components_at(input_index=0, output_index=0)

fig_tf, axes_tf = model.plot_transfer_function(kind="all")
fig_coh, ax_coh = model.plot_coherence(diagnostics=diagnostics)
fig_cross, axes_cross = model.plot_cross_spectrum(diagnostics=diagnostics, kind="both")
```

`diagnostics.coherence` contains one magnitude-squared coherence curve per
output channel, while `diagnostics.predicted_spectrum` and
`diagnostics.observed_spectrum` expose the matching output spectra and
`diagnostics.cross_spectrum` exposes the predicted-vs-observed cross spectrum.

What these quantities mean in practice:

- `weights` / `times`: the familiar time-domain TRF. This is usually the first
  thing to inspect if you want a direct analogue of an mTRF kernel.
- `transfer_function`: the complex frequency-domain mapping learned by the
  spectral solver. It is the most direct representation of what `ffTRF`
  estimates internally.
- `magnitude`: frequency-dependent gain. Peaks indicate frequencies where the
  predictor has the strongest linear coupling to the target.
- `phase`: frequency-dependent timing relationship between predictor and target.
  Smooth phase slopes often correspond to consistent delays.
- `group_delay`: phase slope converted into delay as a function of frequency.
  This is useful when you want a latency-like interpretation in the frequency
  domain rather than a single lag-domain kernel.
- `coherence`: how strongly predicted and observed signals agree at each
  frequency, bounded between `0` and `1`. This is often the most intuitive
  frequency-specific performance diagnostic.
- `cross_spectrum`: the complex predicted-vs-observed spectral relationship.
  Its magnitude shows shared spectral energy, and its phase shows systematic
  lead/lag structure between prediction and observation.
- `predicted_spectrum` / `observed_spectrum`: the output power spectra used to
  contextualize coherence and cross-spectrum values.

### Frequency-Resolved Weights

If you want a time-frequency view of one recovered kernel, you can partition
the fitted transfer function into smooth frequency bands and transform each
band back into the lag domain:

```python
resolved = model.frequency_resolved_weights(
    n_bands=18,
    value_mode="real",
)

print(resolved.weights.shape)      # (n_inputs, n_bands, n_lags, n_outputs)
print(resolved.band_centers[:5])   # center frequency of each analysis band

fig, ax = model.plot_frequency_resolved_weights(
    resolved=resolved,
    input_index=0,
    output_index=0,
)

power = model.time_frequency_power(
    n_bands=18,
    method="hilbert",
)
fig_power, ax_power = model.plot_time_frequency_power(
    power=power,
    input_index=0,
    output_index=0,
)
```

This is useful when a plain time-domain kernel hides oscillatory structure.
The default `value_mode="real"` keeps signed band-limited kernels, so summing
the resolved maps across the band axis reconstructs the ordinary kernel when
the full fitted frequency range is used. Use `value_mode="magnitude"` or
`"power"` if you prefer a simple pointwise squared band-limited kernel. If you
want a spectrogram-like representation instead, use
`time_frequency_power(...)`, which computes Hilbert-envelope power for each
band-limited kernel. If you want the resolved maps to stay visually close to a
known simulated kernel, prefer `window=None` and sufficiently long segments;
short tapered segments smooth the recovered time-domain kernel more
aggressively. These representations are especially useful when one ordinary
TRF hides multiple oscillatory components at different latencies or frequency
ranges. The repository includes a focused time-locked alpha-burst demo under
`examples/`, showing the signed map together with the spectrogram-like
time-frequency power map.

### Trial Bootstrap Confidence Intervals

Set `bootstrap_samples > 0` during `train(...)`, or call
`bootstrap_confidence_interval(...)` afterwards, to store a trial-bootstrap
interval for the recovered kernel:

```python
model.train(
    stimulus=stimulus,
    response=response,
    fs=fs,
    tmin=0.0,
    tmax=0.250,
    regularization=1e-3,
    bootstrap_samples=200,
    bootstrap_level=0.95,
)

interval, times = model.bootstrap_interval_at()
fig, ax = model.plot(show_bootstrap_interval=True)
```

### Trial Weighting

If some trials are much noisier than others, you can either pass explicit trial
weights or use the provided inverse-variance helper:

```python
from fftrf import inverse_variance_weights

weights = inverse_variance_weights(response)
model.train(
    stimulus=stimulus,
    response=response,
    fs=fs,
    tmin=0.0,
    tmax=0.250,
    regularization=np.logspace(-4, 1, 6),
    trial_weights=weights,
)
```

## Runtime Benchmark Against mTRFpy

The repository includes a reproducible runtime benchmark in
`examples/benchmark_runtime.py`. It compares `FrequencyTRF` to a standard
time-domain `mTRFpy` fit across a wider grid of synthetic scenarios.

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
- median per-fit peak RSS measured in isolated worker processes

The current grid covers:

- fixed-ridge whole-trial fits
- multifeature / multichannel fits
- longer lag windows
- cross-validated ridge selection
- segmented spectral estimation with a Hann window

All scenarios use forward regression on the same simulated data for both
methods. Fixed-ridge scenarios use the same lambda in both toolboxes, and the
cross-validated scenario uses the same candidate grid in both. Kernel
correlation is computed over the flattened full kernel bank.

| Scenario | Shape | Fit mode | FFT setting | fs (Hz) | Trials | Samples/trial | Lags | Approx. lag matrix (MiB) | FrequencyTRF median fit (s) | FrequencyTRF peak RSS (MiB) | mTRFpy median fit (s) | mTRFpy peak RSS (MiB) | mTRFpy / FrequencyTRF | Kernel corr. |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Moderate length | 1->1 | fixed | whole-trial | 1000 | 8 | 4096 | 40 | 10.0 | 0.0097 | 97.6 | 0.0094 | 136.1 | 0.96x | 1.0000 |
| Long recording | 1->1 | fixed | whole-trial | 1000 | 4 | 60000 | 40 | 73.2 | 0.1290 | 109.8 | 0.0440 | 181.9 | 0.34x | 1.0000 |
| High rate | 1->1 | fixed | whole-trial | 10000 | 2 | 30000 | 300 | 137.3 | 0.0638 | 98.6 | 0.1395 | 337.9 | 2.19x | 1.0000 |
| Long high rate | 1->1 | fixed | whole-trial | 10000 | 2 | 60000 | 300 | 274.7 | 0.1277 | 103.0 | 0.2763 | 540.0 | 2.16x | 1.0000 |
| Multifeature / multichannel | 3->2 | fixed | whole-trial | 1000 | 6 | 4096 | 40 | 22.5 | 0.0127 | 106.5 | 0.0255 | 149.0 | 2.01x | 1.0000 |
| Longer lag window | 1->1 | fixed | whole-trial | 10000 | 2 | 30000 | 600 | 274.7 | 0.0627 | 100.8 | 0.3775 | 542.4 | 6.02x | 1.0000 |
| Cross-validated ridge | 1->1 | cv-8 (k=4) | whole-trial | 10000 | 4 | 30000 | 300 | 274.7 | 1.9566 | 110.5 | 1.3946 | 367.8 | 0.71x | 1.0000 |
| Segmented Hann estimate | 1->1 | fixed | seg=4096, ov=0.5, hann | 10000 | 2 | 60000 | 300 | 274.7 | 0.0139 | 99.2 | 0.2680 | 539.0 | 19.34x | 1.0000 |

Interpretation:

- At modest lag counts, the standard time-domain fit is still faster on this
  machine.
- As lag count, sampling rate, or feature count grows, the frequency-domain
  formulation becomes more competitive and often faster.
- In the table above, values below `1.0x` mean `mTRFpy` was faster; values
  above `1.0x` mean `FrequencyTRF` was faster.
- Kernel correlations of `1.0000` in these synthetic tests indicate that both
  approaches recover essentially the same flattened kernel bank under the
  matched settings used here.
- The approximate lag-matrix size is included because it is the dominant memory
  object for a standard time-domain fit and scales linearly with both recording
  length and feature-weight count.
- Peak RSS is consistently lower for `FrequencyTRF` in this grid, with the gap
  widening substantially in the high-rate and longer-lag scenarios.
- The cross-validated scenario is where cached spectra matter most for
  `FrequencyTRF`, because the FFT work and fold-wise spectral aggregation are
  reused across lambda candidates, even though `mTRFpy` is still faster in
  that specific benchmark on this machine.
- The segmented Hann scenario is intentionally not the closest mTRF-like
  setting; it illustrates a more typical spectral-estimation workflow for
  frequency-domain fitting.

Further extensions that would still be useful are negative-lag settings and
trial-weighted fits in the benchmark grid.

## API Summary

### `FrequencyTRF`

Core estimator. Important attributes after fitting:

- `transfer_function`: complex spectral mapping with shape
  `(n_frequencies, n_inputs, n_outputs)`
- `frequencies`: frequency vector in Hz
- `weights`: time-domain kernel with shape `(n_inputs, n_lags, n_outputs)`
- `times`: lag vector in seconds
- `regularization`: selected ridge value after fitting
- `segment_length` / `segment_duration`: fitted segment size in samples / seconds

Important methods:

- `train(...)`
- `predict(...)`
- `score(...)`
- `plot(...)`
- `plot_grid(...)`
- `frequency_resolved_weights(...)`
- `plot_frequency_resolved_weights(...)`
- `time_frequency_power(...)`
- `plot_time_frequency_power(...)`
- `bootstrap_confidence_interval(...)`
- `bootstrap_interval_at(...)`
- `to_impulse_response(...)`
- `save(...)` / `load(...)`

### Preprocessing Helpers

- `half_wave_rectify(x)`: split a waveform into positive and negative
  half-waves
- `resample_signal(x, orig_fs, target_fs)`: polyphase resampling helper
- `inverse_variance_weights(trials)`: normalized inverse-variance trial weights

## Optional Comparison Tools

The installable toolbox lives under `src/fftrf/`. Optional validation and
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
