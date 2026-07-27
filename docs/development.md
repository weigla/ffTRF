# Development

## Pixi Environments

The repository is organized around a small set of Pixi environments:

- `default`: editable package install plus core runtime dependencies
- `test`: adds `matplotlib`, `pytest`, and coverage tooling
- `lint`: adds Ruff
- `package`: adds build and metadata-checking tools for local distribution checks
- `compare`: adds `matplotlib` and `mtrf`
- `docs`: adds MkDocs and API-reference tooling

## Common Commands

```bash
pixi install
pixi run import-check
pixi run -e lint lint
pixi run -e test test
pixi run -e package hatch version
pixi run -e package package-build
pixi run -e package package-check
pixi run -e docs docs-build
pixi run -e docs docs-serve
pixi run -e compare compare-demo
pixi run -e compare real-eeg-benchmark
```

For API behavior changes, the usual check is:

1. `pixi run -e test test`
2. `pixi run -e docs docs-build`

That combination catches both regression failures and documentation drift in
the generated reference pages.

## Editable Install Without Pixi

```bash
pip install -e .
pip install -e ".[test]"
pip install -e ".[lint]"
pip install -e ".[docs]"
```

Use the Pixi `docs` environment for the most reproducible site builds, because
CI builds the documentation from the lockfile-backed toolchain:

```bash
pixi run -e docs docs-build
```

## Release Workflow

TestPyPI publishing is manual through the `publish-testpypi` GitHub Actions
workflow. Production PyPI publishing is triggered when a GitHub Release is
published from a tag named `v<version>`, for example `v0.1.0`.

The package version is stored in `src/fftrf/_version.py` and managed with Hatch:

```bash
pixi run -e package hatch version
pixi run -e package hatch version fix
pixi run -e package hatch version minor
pixi run -e package hatch version major
```

Hatch updates the version file but does not create a Git tag. Update
`CITATION.CFF` to the same release version, then create and push the matching
`v<version>` tag before publishing the GitHub Release. The production workflow
checks that the release tag matches `hatch version` before building or
uploading artifacts. Both TestPyPI and PyPI publishing use Trusted Publishing.

## Package Layout

- `src/fftrf/estimator.py`: `TRF`
- `src/fftrf/metrics.py`: scoring functions and metric resolution
- `src/fftrf/results.py`: result dataclasses
- `src/fftrf/spectral.py`: spectral cache and solver helpers
- `src/fftrf/prediction.py`: prediction, CV scoring, and bootstrap helpers
- `src/fftrf/utils.py`: validation and small shared utilities

`src/fftrf/model.py` remains as a thin import surface inside the package, while
the main implementation lives in the smaller submodules above.
