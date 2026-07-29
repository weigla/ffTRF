# Diagnostics and Transfer Functions

`ffTRF` exposes both lag-domain and frequency-domain views of a fitted model.
This page focuses on choosing the spectral tool that answers your question.
The [Diagnostics Notebook](../notebooks/diagnostics.ipynb) fits one model and
shows the output of each function.

## Raw Transfer Function

Use `transfer_function_at(...)` when you want the complex-valued frequency
response for one input/output pair:

```python
frequencies, transfer = model.transfer_function_at(
    input_index=0,
    output_index=0,
)
```

The returned complex values encode both amplitude and phase.

Use this numerical interface when you need to export values, define a custom
summary, or combine the transfer function with another analysis. For routine
inspection, the plotting helpers below are shorter.

## Derived Transfer-Function Components

Use `transfer_function_components_at(...)` when you want the common derived
quantities in one container:

- magnitude
- unwrapped phase
- group delay

This is convenient when you want values for custom plotting or downstream
analysis.

```python
components = model.transfer_function_components_at(
    input_index=0,
    output_index=0,
)

components.magnitude
components.phase
components.group_delay
```

## Transfer-Function Plotting

Use `plot_transfer_function(...)` for quick inspection:

- `kind="magnitude"`: show only magnitude
- `kind="phase"`: show only phase
- `kind="group_delay"`: show only group delay
- `kind="both"`: show magnitude and phase
- `kind="all"`: show magnitude, phase, and group delay

Group delay can be especially informative when you want to know whether the
fitted mapping behaves like a delayed filter across frequencies rather than a
single lag-domain peak.

Plot magnitude, phase, and group delay separately while learning the API. They
have different units and answer different questions:

```python
fig, ax = model.plot_transfer_function(kind="magnitude")
fig, ax = model.plot_transfer_function(kind="phase", phase_unit="deg")
fig, ax = model.plot_transfer_function(
    kind="group_delay",
    group_delay_unit="ms",
)
```

The combined `kind="all"` layout is useful once you already know which panel
you need.

## Cross-Spectral Diagnostics

Use `cross_spectral_diagnostics(...)` when you want to compare the model's
predictions against observed targets in the frequency domain.

The returned container includes:

- predicted output spectra
- observed output spectra
- predicted-vs-observed cross-spectra
- magnitude-squared coherence

This is useful when a lag-domain kernel looks plausible but you still want to
know whether the model captures the spectral structure of the target signal.

Compute these diagnostics on held-out data whenever the goal is to assess
generalization:

```python
diagnostics = model.cross_spectral_diagnostics(
    stimulus=heldout_stimulus,
    response=heldout_response,
)
```

## Coherence

`plot_coherence(...)` shows the magnitude-squared coherence between predicted
and observed targets for one output channel.

Interpretation:

- values near 1 indicate strong frequency-specific agreement
- values near 0 indicate poor agreement at those frequencies

Coherence is bounded, so it is often easier to compare across channels than raw
spectral magnitudes.

```python
fig, ax = model.plot_coherence(
    diagnostics=diagnostics,
    output_index=0,
)
```

High coherence describes frequency-specific linear agreement. It does not by
itself show that a model is unbiased or that its prediction has the correct
amplitude.

## Cross Spectrum

`plot_cross_spectrum(...)` shows the predicted-vs-observed cross spectrum for
one output channel.

- magnitude shows how strongly the prediction and observation covary by
  frequency
- phase shows whether they align or lag relative to each other in the spectral
  domain

```python
fig, ax = model.plot_cross_spectrum(
    diagnostics=diagnostics,
    output_index=0,
    kind="magnitude",
)
```

The magnitude and phase views are demonstrated separately in the
[Diagnostics Notebook](../notebooks/diagnostics.ipynb#predicted-observed-cross-spectrum).

## When to Use Which Tool

- Use `plot(...)` when you mainly care about lag-domain kernel shape.
- Use `plot_transfer_function(...)` when you care about gain and phase.
- Use `plot_coherence(...)` when you care about prediction quality by
  frequency.
- Use `plot_cross_spectrum(...)` when you want a fuller spectral relationship
  between predictions and observed targets.
