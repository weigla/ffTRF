# Frequency-Resolved Analysis

The fitted transfer function can be partitioned into smooth frequency bands and
transformed back to the lag domain.

## Frequency-Resolved Weights

```python
resolved = model.frequency_resolved_weights(
    n_bands=18,
    fmax=160.0,
    value_mode="real",
)

fig, ax = model.plot_frequency_resolved_weights(
    resolved=resolved,
    input_index=0,
    output_index=0,
)
```

This produces a frequency-by-lag map for one input/output pair while leaving
the ordinary kernel available in `model.weights`.

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

## When to Use Which View

- Use `value_mode="real"` when you want signed, band-limited kernel structure.
- Use `value_mode="magnitude"` or `"power"` when you want a simpler positive map.
- Use `time_frequency_power(...)` when you want a spectrogram-like view of the
  recovered kernel energy.

## Diagnostics Around the Transfer Function

The estimator also exposes frequency-domain views directly:

- `transfer_function_at(...)`
- `transfer_function_components_at(...)`
- `plot_transfer_function(...)`
- `cross_spectral_diagnostics(...)`
- `plot_coherence(...)`
- `plot_cross_spectrum(...)`
