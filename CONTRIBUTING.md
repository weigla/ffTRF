# How to contribute to ffTRF

Contributions to `ffTRF` are always welcome!

## General rules

The repository uses the following tools:

- [Pixi](https://pixi.sh/) for dependency and development environment
  management.
- [Ruff](https://docs.astral.sh/ruff/) for linting.
- [Hatch](https://hatch.pypa.io/latest/) for package version management.
- [Hatchling](https://hatch.pypa.io/latest/) as the package build backend.
- [pytest](https://docs.pytest.org/) for tests.
- [MkDocs](https://www.mkdocs.org/) for the documentation site.

## Setting up the development environment

Install [Pixi](https://pixi.sh/latest/#installation), then run this from the
repository root:

```bash
pixi install
```

Pixi creates the project environments and installs `ffTRF` as an editable
package. You can either prefix commands with `pixi run ...` or start a shell:

```bash
pixi shell
```

The main environments are:

- `default`: runtime dependencies, including Matplotlib, and the editable
  package install.
- `test`: pytest and coverage tooling used by the test suite.
- `lint`: Ruff.
- `package`: build and package metadata checking tools.
- `docs`: MkDocs and API-reference tooling.
- `compare`: optional `mtrf` comparison and benchmark dependencies.

## How to do common tasks

### Add dependencies

Runtime package dependencies are declared in the `[project] -> dependencies`
section of `pyproject.toml`. The Pixi development environment dependencies are
declared under `[tool.pixi.dependencies]` and the relevant
`[tool.pixi.feature.*]` sections.

When adding a runtime dependency, update `pyproject.toml` so both the packaged
metadata and the Pixi environment stay consistent. Prefer packages available
from `conda-forge` for Pixi dependencies when possible. After changing
dependencies, run:

```bash
pixi install
```

Commit the updated `pyproject.toml` and `pixi.lock` when dependency changes are
intentional.

### Linting

Run Ruff before opening a pull request:

```bash
pixi run -e lint lint
```

The lint task currently runs `ruff check .`.

### Testing

Run the full test suite with:

```bash
pixi run -e test test
```

For focused development, run a targeted pytest command, for example:

```bash
pixi run -e test python -m pytest -q tests/test_model.py -k "<relevant test>"
```

Add tests for bug fixes and new behavior. Use small seeded synthetic datasets
where possible, compare numerical paths with explicit tolerances, and cover
invalid inputs for new validation logic.

Coverage is collected in CI and reported to Coveralls. To reproduce the local
coverage command:

```bash
pixi run -e test coverage
```

### Documentation

Build the documentation with:

```bash
pixi run -e docs docs-build
```

To serve the docs locally:

```bash
pixi run -e docs docs-serve
```

Public API changes should include documentation updates and should pass the
strict docs build.

### Examples and benchmarks

Examples are smoke-tested in CI. If you change an example, run it directly in
the relevant environment. For comparison examples and benchmarks, use the
`compare` environment:

```bash
pixi run -e compare compare-demo
pixi run -e compare benchmark-demo
pixi run -e compare real-eeg-benchmark
```

Only update benchmark claims or checked-in benchmark artifacts after rerunning
the relevant benchmark in a documented environment. The benchmark tasks write
raw JSON alongside Markdown and synchronize their generated README tables.
Run them from a clean source revision for release-facing results.

### Building and checking the package

`ffTRF` uses Hatchling as the build backend. Local build and metadata checks are
wrapped as Pixi tasks:

```bash
pixi run -e package package-build
pixi run -e package package-check
```

The build command creates distributions in `dist/`; do not commit generated
distribution files.

### Versioning and releases

The package version is stored in `src/fftrf/_version.py` and managed with
[Hatch](https://hatch.pypa.io/latest/version/). View the current version or
increment it from the Pixi package environment:

```bash
pixi run -e package hatch version
pixi run -e package hatch version fix
pixi run -e package hatch version minor
pixi run -e package hatch version major
```

Hatch updates the package version but does not create Git tags. Production PyPI
publishing is triggered by publishing a GitHub Release from a tag named
`v<version>`, for example `v0.1.0`. The release workflow verifies that the tag
matches the Hatch-managed version before building and uploading the package.
When preparing a release, update `CITATION.CFF` to the same version, create the
matching tag, and push it:

```bash
git tag v<version>
git push origin v<version>
```

## Pull request checklist

Before opening a pull request, run the narrowest relevant checks while
developing and the broader checks before finishing:

```bash
pixi run import-check
pixi run -e lint lint
pixi run -e test test
pixi run -e docs docs-build
pixi run -e package package-build
pixi run -e package package-check
```

Not every change needs every command. For example, a small documentation-only
change usually needs the docs build, while a packaging change should include
the package checks. In the pull request description, mention which commands you
ran and any checks you could not run.
