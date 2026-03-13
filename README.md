# fft-trf

`fft-trf` is a small Python toolbox for estimating stimulus-response models in the
frequency domain. It is aimed at the use case where a classic time-lagged mTRF
design matrix becomes expensive or awkward, for example ABR-style analyses on
10 kHz EEG/MEG data.

The package is intentionally close in spirit to `mTRFpy`:

- a `FrequencyTRF` estimator with `train`, `predict`, and `score`
- multi-trial input using lists of arrays
- scalar or cross-validated ridge regularization
- weights exposed as a time-domain impulse response for interpretation

The estimation itself follows the Maddox Lab style of spectral deconvolution:

`H(f) = (Sxx(f) + lambda I)^-1 Sxy(f)`

where `H(f)` is the learned transfer function. The reported `weights` are the
inverse FFT of that transfer function over the requested lag window.

## Why this is useful for ABR / brainstem work

For high-rate recordings, especially when you only care about a short response
window such as 0 to 30 ms, building large time-lagged regression matrices can be
wasteful. A frequency-domain formulation keeps the computation in terms of
cross-spectra and then converts the result back to an impulse response that you
can inspect like a TRF kernel.

This is not mathematically identical to the standard time-domain mTRF fit with a
hard lag constraint. Here, `tmin` and `tmax` define the impulse-response window
that is extracted and used for prediction after fitting the frequency response.
That matches the deconvolution flavor used in the Maddox Lab ABR pipeline more
closely than the classic lag-matrix formulation.

## Installation

```bash
pip install -e .
```

For the optional examples/plotting demo:

```bash
pip install -e ".[compare]" mtrf
```

### Pixi

Pixi can use this `pyproject.toml` directly as its manifest:

```bash
pixi install
pixi run import-check
pixi run -e test test
pixi run -e compare compare-demo
```

That gives you:

- a default Pixi environment with Python, NumPy, SciPy, and this package as an
  editable local dependency
- a `test` environment that also includes `pytest`
- a `compare` environment with `matplotlib` and `mTRFpy` for side-by-side kernel plots

## Quick example

```python
import numpy as np

from fft_trf import FrequencyTRF, half_wave_rectify, inverse_variance_weights

fs = 10_000

# Two trials so cross-validation over lambda is meaningful.
audio_trials = [np.random.randn(fs * 12) for _ in range(2)]
meg_trials = [np.random.randn(fs * 12, 1) for _ in range(2)]

stimulus = []
for audio_trial in audio_trials:
    stim_pos, stim_neg = half_wave_rectify(audio_trial)
    stimulus.append(np.column_stack([stim_pos, stim_neg]))

response = meg_trials

model = FrequencyTRF(direction=1)
metric = model.train(
    stimulus=stimulus,
    response=response,
    fs=fs,
    tmin=-0.005,
    tmax=0.030,
    regularization=np.logspace(-6, 1, 8),
    segment_length=4096,
    overlap=0.5,
    window="hann",
    k=-1,
    trial_weights=inverse_variance_weights(response),
)

prediction, r = model.predict(stimulus=stimulus, response=response)
print("selected lambda:", model.regularization)
print("validation scores:", metric)
print("held-out correlation:", r)

# mTRF-like public attributes
times_ms = model.times * 1e3
abr_kernel = model.weights[0, :, 0]
```

## API summary

### `FrequencyTRF`

Core estimator. Important attributes after fitting:

- `transfer_function`: complex spectral mapping with shape
  `(n_frequencies, n_inputs, n_outputs)`
- `frequencies`: frequency vector in Hz
- `weights`: mTRF-like time-domain kernel with shape
  `(n_inputs, n_lags, n_outputs)`
- `times`: lag vector in seconds

Important methods:

- `train(...)`
- `predict(...)`
- `score(...)`
- `to_impulse_response(...)`
- `save(...)` / `load(...)`

### Preprocessing helpers

- `half_wave_rectify(x)`: returns positive and negative half-wave components
- `resample_signal(x, orig_fs, target_fs)`: polyphase resampling helper
- `inverse_variance_weights(trials)`: convenient trial weights for ABR averaging

## Practical notes for your MEG workflow

- Start with regressors similar to the Maddox Lab pipeline: rectified audio,
  IHC output, or ANM output.
- Resample the regressor to the neural sampling rate before fitting.
- For ABR-like kernels, use a short output window such as `tmin=-0.005`,
  `tmax=0.030`.
- If you have many long trials, use `segment_length` around 2048 to 8192 samples
  and `overlap=0.5` to stabilize the cross-spectral estimates; in that case
  `window="hann"` is usually a good default.
- If trial noise differs strongly, pass `trial_weights="inverse_variance"` or
  an explicit weight vector.
- If you want the closest comparison to a standard time-domain mTRF simulation,
  start with `segment_length=None` and `window=None`.

## Optional Comparison Example

The core installable toolbox lives under `src/fft_trf/`. Optional validation
and comparison code lives under `examples/` so it stays separate from the public
library API.

You can generate a side-by-side figure comparing the true kernel, `fft_trf`, a
direct time-domain lagged ridge solution, and `mTRFpy` with:

```bash
python examples/compare_with_mtrf.py --output artifacts/kernel_comparison.png --no-show
```

For a comparison that is as close as possible to a standard mTRF simulation,
start with:

```bash
python examples/compare_with_mtrf.py --segment-length none --window none --no-show
```

If you instead want a more ABR-like spectral-estimation setup, use overlapping
segments and a Hann window:

```bash
python examples/compare_with_mtrf.py --segment-length 512 --overlap 0.5 --window hann --no-show
```

## Packaging notes

- `pip install -e .` works through the standard Python packaging metadata.
- Pixi uses the same [pyproject.toml](/Users/alexanderweigl/Documents/fft_trf/pyproject.toml),
  so there is no second manifest to keep in sync.
- In Pixi, NumPy and SciPy are installed from Conda-forge while this project
  itself is installed as a local editable PyPI dependency.
