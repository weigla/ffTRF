# ffTRF

`ffTRF` is a Python toolbox for fitting temporal response functions in the
frequency domain. It is designed for continuous stimulus-response modeling with
a small public API centered on `fftrf.TRF`.

The full documentation now lives in [`docs/`](docs/index.md), with dedicated
pages for:

- [Getting Started](docs/getting-started.md)
- [Examples](docs/examples.md)
- [API Reference](docs/reference/index.md)
- [Development](docs/development.md)

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
