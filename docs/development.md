# Development

## Pixi Environments

The repository is organized around a small set of Pixi environments:

- `default`: editable package install plus core runtime dependencies
- `test`: adds `pytest`
- `compare`: adds `matplotlib` and `mtrf`
- `docs`: adds MkDocs and API-reference tooling

## Common Commands

```bash
pixi install
pixi run import-check
pixi run -e test test
pixi run -e docs docs-build
pixi run -e docs docs-serve
pixi run -e compare compare-demo
pixi run -e compare benchmark-demo
```

## Editable Install Without Pixi

```bash
pip install -e .
pip install -e ".[test]"
pip install -e ".[docs]"
```

Use the Pixi `docs` environment for the most reproducible site builds, because
CI builds the documentation from the lockfile-backed toolchain:

```bash
pixi run -e docs docs-build
```

## Package Layout

- `src/fftrf/estimator.py`: `TRF`
- `src/fftrf/metrics.py`: scoring functions and metric resolution
- `src/fftrf/results.py`: result dataclasses
- `src/fftrf/spectral.py`: spectral cache and solver helpers
- `src/fftrf/prediction.py`: prediction, CV scoring, and bootstrap helpers
- `src/fftrf/utils.py`: validation and small shared utilities

`src/fftrf/model.py` remains as a thin import surface inside the package, while
the main implementation lives in the smaller submodules above.

## Performance Notes

Cross-validation performance relies on two different kinds of reuse:

- `src/fftrf/spectral.py` caches per-trial spectral sufficient statistics so
  folds and ridge candidates do not repeat FFT-based training statistics.
- `src/fftrf/prediction.py` now caches validation-side predictor FFTs within
  each fold, so repeated candidate scoring reuses the same transformed
  predictors instead of recomputing one convolution per input/output kernel.

That second optimization is especially relevant for larger CV grids and banded
regularization, because it lowers scoring cost without changing any public API,
fit result, or score value.
