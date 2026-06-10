# AGENTS.md

This file contains repository-wide guidance for coding agents working on
`ffTRF`. It applies to the entire repository.

## Project Overview

`ffTRF` is a Python toolbox for estimating temporal response functions (TRFs)
and related deconvolution models in the frequency domain. The main public API is
`fftrf.TRF`.

Instead of constructing an explicit time-lagged design matrix, the toolbox:

1. estimates predictor auto-spectra and predictor-target cross-spectra,
2. solves a ridge-regularized transfer function at each frequency, and
3. converts that transfer function into a lag-domain impulse response.

The package supports forward encoding (`stimulus -> response`) and backward
decoding (`response -> stimulus`), scalar and feature-banded regularization,
cross-validation, standard and multitaper spectral estimation, prediction and
scoring, bootstrap intervals, permutation tests, diagnostics, and optional
plotting.

## Core Data Conventions

- Time is always the first array axis.
- A single trial may be a 1D array or a 2D array with shape
  `(n_samples, n_features)`.
- Multiple trials are represented by a sequence of arrays. Matching stimulus
  and response trials must have the same sample count, although different
  trials may have different lengths.
- `TRF(direction=1)` uses stimulus as predictor and response as target.
- `TRF(direction=-1)` uses response as predictor and stimulus as target.
- Stored lag-domain weights have shape
  `(n_inputs, n_lags, n_outputs)`.
- The lag interval is sample based and follows `[tmin, tmax)`.

Preserve these conventions across fitting, prediction, scoring, diagnostics,
serialization, and plotting. Shape or direction changes are public behavioral
changes and require explicit tests and documentation.

## Repository Map

### Package

- `src/fftrf/__init__.py`: top-level public exports.
- `src/fftrf/model.py`: compatibility and convenience import surface. It is
  intentionally thin; implementations belong in the focused modules below.
- `src/fftrf/estimator.py`: the `TRF` class and orchestration of training,
  diagnostics, uncertainty estimation, prediction, persistence, and plotting.
- `src/fftrf/spectral.py`: spectral estimation, cached sufficient statistics,
  regularized frequency-domain solvers, and multitaper helpers.
- `src/fftrf/prediction.py`: transfer-function-to-kernel conversion,
  convolution-based prediction, CV scoring, bootstrap helpers, and permutation
  helpers.
- `src/fftrf/metrics.py`: built-in metrics and metric resolution.
- `src/fftrf/results.py`: public result dataclasses.
- `src/fftrf/preprocessing.py`: public signal preprocessing and trial-weight
  helpers.
- `src/fftrf/plotting.py`: optional Matplotlib plotting functions. Matplotlib
  must remain an optional dependency and should be imported lazily.
- `src/fftrf/utils.py`: validation, input coercion, regularization expansion,
  segment settings, and small shared helpers.

### Tests, Documentation, and Examples

- `tests/test_model.py`: numerical behavior, solver/reference equivalence,
  fitting and CV behavior, shapes, metrics, preprocessing, performance-related
  cache behavior, multitaper support, and optional plotting.
- `tests/test_public_api.py`: user-facing workflows and contracts such as
  diagnostics aliases, bootstrap updates, persistence, copying, permutation
  tests, and top-level exports.
- `docs/`: MkDocs documentation. `docs/development.md` records the supported
  development commands; guides describe public behavior and conventions.
- `examples/`: runnable workflows and optional comparisons with `mTRF`.
- `artifacts/`: checked-in benchmark output. Update benchmark claims only after
  rerunning the relevant benchmark in a documented environment.
- `pyproject.toml`: package metadata, dependencies, pytest configuration, Pixi
  environments, and task definitions.

## Implementation Guidelines

- Follow the existing `src/` layout and keep code in the module that owns the
  behavior. Do not move implementation back into `model.py`.
- Prefer NumPy/SciPy operations and the existing validation and spectral
  helpers over parallel implementations of the same logic.
- Use type hints and NumPy-style public docstrings, matching surrounding code.
- Validate invalid values and incompatible shapes early with clear
  `ValueError` or `IndexError` messages.
- Keep public results copy-safe where the existing API returns copies rather
  than mutable internal state.
- Preserve reproducibility controls such as explicit seeds and deterministic
  serial/parallel equivalence.
- Avoid adding runtime dependencies unless the feature cannot reasonably be
  implemented with NumPy, SciPy, or the standard library. Keep plotting and
  comparison dependencies optional.
- Treat spectral and CV cache reuse as part of the performance design.
  Changes to `spectral.py`, `prediction.py`, or CV loops should not silently
  restore repeated FFTs, repeated decompositions, or per-candidate work that
  can be shared.
- Numerical optimizations must be checked against a direct or existing
  reference implementation, not only against expected shapes.

## Public API Changes

When adding or changing public behavior:

1. implement it in the appropriate focused module,
2. expose public symbols through `src/fftrf/model.py` and
   `src/fftrf/__init__.py` when appropriate,
3. add or update tests in `tests/test_public_api.py` and/or
   `tests/test_model.py`,
4. update docstrings and the relevant page under `docs/`, and
5. add or update an example when the workflow is not obvious from the API
   reference.

Maintain backward compatibility unless the task explicitly requires a breaking
change. Do not change result shapes, direction semantics, regularization
selection, lag indexing, or serialization state accidentally.

## Testing Requirements

Every change must be verified. Add a regression test for every bug fix and
tests for both successful and invalid-input paths of new behavior.

Tests should:

- use seeded `numpy.random.Generator` instances,
- use small synthetic datasets to keep the suite fast,
- compare numerical paths with explicit tolerances,
- use held-out data when asserting predictive quality,
- cover forward and backward direction when direction-dependent logic changes,
- cover single-trial and multi-trial forms when input coercion changes, and
- verify serial and parallel results agree when changing `n_jobs` code.

Run the narrowest relevant tests while developing, then run the full suite
before finishing:

```bash
pixi run -e test python -m pytest -q tests/test_model.py -k "<relevant test>"
pixi run -e test test
```

Additional verification by change type:

- Public imports or packaging:

  ```bash
  pixi run import-check
  ```

- Public API or documentation:

  ```bash
  pixi run -e docs docs-build
  ```

- Example changes: run the changed example directly in the appropriate Pixi
  environment.
- Performance changes: run the relevant focused tests and, when benchmark
  claims or artifacts are affected:

  ```bash
  pixi run -e compare benchmark-demo
  ```

The CI baseline is `python -m pytest -q`, smoke tests for core examples, and a
strict documentation build. Report which commands were run and any checks that
could not be run.

## Development Workflow

Pixi is the primary supported environment manager:

```bash
pixi install
pixi run import-check
pixi run -e test test
pixi run -e docs docs-build
```

An editable pip installation is also supported through the extras in
`pyproject.toml`, but prefer the locked Pixi environments for reproducible
repository work.

Before editing, inspect the relevant implementation, tests, and documentation.
Keep changes scoped to the requested behavior, preserve unrelated work in the
working tree, and do not commit generated directories such as `site/`,
`dist/`, caches, or bytecode.
