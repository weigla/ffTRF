# Getting Started

This page walks through the shortest path from install to a fitted model.

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

## Input Conventions

- Single-trial input can be a 1D or 2D NumPy array.
- Multi-trial input should be a list of arrays.
- Each trial must be shaped `(n_samples, n_features)` for the predictor side
  and `(n_samples, n_outputs)` for the target side.
- A 1D vector is treated as a single feature or output channel.

## Next Steps

- Learn how regularization search works in [Regularization and CV](guides/regularization.md)
- See the multitaper workflow in [Multitaper Estimation](guides/multitaper.md)
- Explore time-frequency views in [Frequency-Resolved Analysis](guides/frequency-resolved.md)
- Browse runnable scripts in [Examples](examples.md)
