# Getting Started

This page walks through the shortest path from install to a fitted model, while
also explaining what the key arguments do and what is stored on the estimator
after fitting.

## Install

Pixi is the most complete setup for contributors:

```bash
pixi install
pixi run import-check
```

If you only want the package:

```bash
pip install -e .
```

If you want plotting, tests, comparison scripts, or docs tooling in a plain
`pip` workflow, install the matching extras:

```bash
pip install -e ".[compare]"
pip install -e ".[test]"
pip install -e ".[docs]"
```

For an existing Pixi project, you can link `ffTRF` directly from GitHub via
Pixi's `pypi-dependencies`:

```toml
[pypi-dependencies]
fftrf = { git = "https://github.com/weigla/ffTRF" }
```

Then run `pixi install`. If you want to pin a specific revision, add
`rev = "<commit>"` to the dependency entry.

## First Model

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

The kind of lag-domain kernel this produces is shown below for the bundled
multi-trial example workflow:

![Multi-trial kernel example](images/examples/multi_trial_single_channel.png)

## What This Example Is Doing

- `stimulus` and `response` are lists of arrays, so the fit runs in
  multi-trial mode.
- `direction=1` means a forward model: stimulus features predict response
  channels.
- `tmin=-0.050` and `tmax=0.250` request a lag window from -50 ms to +250 ms.
- `regularization=np.logspace(-6, 1, 8)` asks the model to cross-validate over
  eight ridge values instead of fitting one fixed value.
- `segment_duration=2.048` sets the segment size used to estimate the
  cross-spectra.
- `overlap=0.5` reuses half of each segment in the next one, which can help
  stabilize estimates when segments are short.
- `window="hann"` applies a Hann window before the FFT.
- `k="loo"` means leave-one-out cross-validation over trials.
- `trial_weights=inverse_variance_weights(response)` downweights noisier trials
  in the aggregate training spectra.

## What `train(...)` Returns

- If `regularization` is a single scalar, `train(...)` returns `None` and fits
  directly.
- If `regularization` contains multiple candidates, `train(...)` returns the
  cross-validation scores for those candidates.
- Regardless of return value, the fitted state is stored on the estimator.

After training, the most important attributes are:

- `model.transfer_function`: complex frequency-domain solution
- `model.frequencies`: frequency axis in Hz
- `model.weights`: lag-domain kernel
- `model.times`: lag axis in seconds
- `model.regularization`: chosen ridge value or banded tuple
- `model.regularization_candidates`: all evaluated regularization candidates

## What `predict(...)` Returns

- If you pass only the predictor side, `predict(...)` returns predictions.
- If you also pass the observed target side, it returns
  `(predictions, score)`.
- The default score is Pearson correlation, but you can choose a different
  metric when constructing the model.

## Input Conventions

- Single-trial input can be a 1D or 2D NumPy array.
- Multi-trial input should be a list of arrays.
- Each trial must be shaped `(n_samples, n_features)` for the predictor side
  and `(n_samples, n_outputs)` for the target side.
- A 1D vector is treated as a single feature or output channel.
- All stimulus/response pairs within one fit must have matching sample counts
  per trial.
- In a backward model (`direction=-1`), the predictor side is the response and
  the target side is the stimulus, but the same shape rules apply.

## A Minimal Direct Fit

If you already know the ridge value you want, use a direct fit:

```python
model = TRF(direction=1)
model.train(
    stimulus=stimulus,
    response=response,
    fs=fs,
    tmin=0.0,
    tmax=0.120,
    regularization=1e-3,
)
```

This avoids cross-validation and fits immediately with one regularization
setting.

## Common Next Actions After Training

- `model.plot(...)`: show one kernel
- `model.plot_grid(...)`: show all kernels in a feature-by-output grid
- `model.to_impulse_response(...)`: extract a different lag window without
  refitting
- `model.score(...)`: compute only the metric
- `model.transfer_function_components_at(...)`: inspect magnitude, phase, and
  group delay
- `model.cross_spectral_diagnostics(...)`: compare predicted and observed
  spectra
- `model.bootstrap_confidence_interval(...)`: estimate uncertainty over trials
- `model.save(...)`: persist the fitted model

## Next Steps

- Walk through the intended lifecycle in [Core Workflow](guides/core-workflow.md)
- Learn the array rules in [Inputs and Shapes](guides/inputs-and-shapes.md)
- Learn how regularization search works in [Regularization and CV](guides/regularization.md)
- See the multitaper workflow in [Multitaper Estimation](guides/multitaper.md)
- Explore time-frequency views in [Frequency-Resolved Analysis](guides/frequency-resolved.md)
- Explore spectral inspection tools in
  [Diagnostics and Transfer Functions](guides/diagnostics-and-transfer-functions.md)
- Browse runnable scripts in [Examples](examples.md)
- Open the rendered tutorial in
  [Getting Started Notebook](notebooks/getting-started.ipynb)
