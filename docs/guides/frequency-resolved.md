# Frequency-Resolved Analysis

The fitted transfer function can be partitioned into smooth frequency bands and
transformed back to the lag domain. This gives a lag-by-frequency view of the
recovered kernel instead of a single kernel collapsed across the whole fitted
spectrum.

This guide explains what each representation means. The
[Frequency-Resolved Notebook](../notebooks/frequency-resolved.ipynb) contains a
complete simulation, fit, and separate plots for each representation.

## Start With the Ordinary Kernel

Inspect `model.weights` first. Frequency resolution is most useful when the
ordinary kernel contains structure that could plausibly differ across
frequency, such as a transient followed by an oscillatory response. It should
not be used to turn an uninterpretable kernel into an automatically meaningful
time-frequency result.

```python
fig, ax = model.plot(input_index=0, output_index=0)
```

## Signed Frequency-Resolved Weights

```python
resolved = model.frequency_resolved_weights(
    n_bands=20,
    fmax=30.0,
    value_mode="real",
)

fig, ax = model.plot_frequency_resolved_weights(
    resolved=resolved,
    input_index=0,
    output_index=0,
)
```

The signed map retains polarity. Positive and negative weights can therefore
cancel when bands are combined, just as they do in the ordinary lag-domain
kernel. Use this view when the sign and timing of a response are part of the
scientific question.

See the notebook section
[Signed frequency-resolved weights](../notebooks/frequency-resolved.ipynb#signed-frequency-resolved-weights)
for the corresponding code and plot.

## What the Parameters Mean

- `n_bands`: number of analysis bands
- `fmin`, `fmax`: frequency range to resolve; `fmax` must stay at or below the
  fitted Nyquist frequency (`fs / 2`)
- `scale`: `"linear"` or `"log"` spacing of band centers
- `bandwidth`: width of the Gaussian analysis bands
- `value_mode`: how the band-limited kernels are represented

## Choosing `value_mode`

- `value_mode="real"` keeps signed band-limited kernels
- `value_mode="magnitude"` takes their absolute value
- `value_mode="power"` squares the magnitude

Use `real` when you care about polarity and cancellation across lags. Use
`magnitude` or `power` when the question concerns the strength of structure
rather than its sign.

## Time-Frequency Power

```python
power = model.time_frequency_power(
    n_bands=18,
    method="hilbert",
)

fig, ax = model.plot_time_frequency_power(
    power=power,
    input_index=0,
    output_index=0,
)
```

This view starts from the signed band-limited kernels and turns each band into
a smoother positive power estimate using the analytic-signal magnitude. It is
appropriate for questions such as “when is kernel energy concentrated around
10 Hz?”, but it does not estimate induced power in the original EEG.

See
[Time-frequency power](../notebooks/frequency-resolved.ipynb#time-frequency-power)
for a separate example and plot.

## When to Use Which View

- Use `frequency_resolved_weights(..., value_mode="real")` when you want signed
  structure in the recovered kernel.
- Use `frequency_resolved_weights(..., value_mode="magnitude")` when you want a
  simpler non-negative map without the extra Hilbert step.
- Use `time_frequency_power(...)` when you want a spectrogram-like summary of
  kernel energy.

## Interpretation Tips

- These plots describe the fitted kernel, not the raw stimulus or response.
- Summing across the band axis in the default signed view approximates the
  ordinary lag-domain kernel.
- Log-spaced bands are often more interpretable when you care about a broad
  range of frequencies.
- Band centers are analysis choices, not statistically independent frequency
  bins.
- Validate prominent features on held-out data or across independent
  participants before interpreting them neuroscientifically.

## Diagnostics Around the Transfer Function

The estimator also exposes direct frequency-domain views:

- `transfer_function_at(...)`
- `transfer_function_components_at(...)`
- `plot_transfer_function(...)`
- `cross_spectral_diagnostics(...)`
- `plot_coherence(...)`
- `plot_cross_spectrum(...)`

For worked examples of these separate diagnostics, continue with the
[Diagnostics Notebook](../notebooks/diagnostics.ipynb).
