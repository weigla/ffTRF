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

## Package Layout

- `src/fftrf/estimator.py`: `TRF`
- `src/fftrf/metrics.py`: scoring functions and metric resolution
- `src/fftrf/results.py`: result dataclasses
- `src/fftrf/spectral.py`: spectral cache and solver helpers
- `src/fftrf/prediction.py`: prediction, CV scoring, and bootstrap helpers
- `src/fftrf/utils.py`: validation and small shared utilities

`src/fftrf/model.py` remains as a thin import surface inside the package, while
the main implementation lives in the smaller submodules above.
