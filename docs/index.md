# ffTRF

`ffTRF` is a Python toolbox for fitting temporal response functions in the
frequency domain. It is designed for continuous stimulus-response modeling
without tying the workflow to one modality, one preprocessing stack, or one
experimental paradigm.

The main estimator is `fftrf.TRF`. It supports:

- forward and backward TRF fitting
- scalar ridge and cross-validated ridge selection
- optional banded regularization for grouped predictors
- optional DPSS multi-taper spectral estimation
- multi-trial input as Python lists of arrays
- time-domain kernel export for interpretation
- bootstrap confidence intervals
- transfer-function and cross-spectral diagnostics
- frequency-resolved lag-domain views of recovered kernels

## Start Here

- New to the package: go to [Getting Started](getting-started.md)
- Looking for runnable demos: go to [Examples](examples.md)
- Looking for API docs: go to [Reference](reference/index.md)
- Working on the repo itself: go to [Development](development.md)

## Installation

For local development, Pixi is the primary supported workflow:

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

## Example Gallery

### Single-Trial Forward Model

![Single-trial forward model](images/examples/single_trial_single_channel.png)

### Multi-Trial Cross-Validation

![Multi-trial forward model](images/examples/multi_trial_single_channel.png)

### Frequency-Resolved Weights

![Frequency-resolved weights](images/examples/frequency_resolved_weights.png)
