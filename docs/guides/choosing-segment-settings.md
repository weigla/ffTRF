# Choosing Segment Settings

`ffTRF` can either estimate spectra from full trials or split each trial into
shorter segments before the FFT. The right choice depends mostly on how long
your lag window is and how much data each trial contains.

## Short Rule of Thumb

- If trials are already short, start with full-trial spectra:
  `segment_length=None`, `window=None`.
- If trials are long and continuous, start with overlapping Hann-windowed
  segments.
- A good first target is a segment that is about `6x` the lag-window span.
- For many speech EEG fits with lag windows around `200-400 ms`, a very good
  starting point is `segment_duration=2.0`, `overlap=0.5`, `window="hann"`.

## Why This Matters

Segment settings change the tradeoff between:

- frequency resolution
- spectral stability
- the number of effectively independent observations
- leakage at segment boundaries

In practice:

- longer segments preserve finer frequency resolution
- shorter segments create more averages and can stabilize noisy data
- overlap helps when segments are short
- a Hann window usually makes short standard-FFT segments behave better

## Start Here

Use these defaults for the standard estimator:

- Full-trial fit:
  use when trials are short or already well-bounded
  `segment_length=None`, `overlap=0.0`, `window=None`
- Segmented fit:
  use when trials are long continuous recordings
  `segment_duration=2.0`, `overlap=0.5`, `window="hann"`

For multi-taper estimation, keep the same rough segment duration and overlap,
but let the DPSS tapers provide the weighting:

- `spectral_method="multitaper"`
- `window=None`

## Public Helper

`ffTRF` includes a small helper for this first guess:

```python
from fftrf import suggest_segment_settings

settings = suggest_segment_settings(
    fs=128,
    tmin=0.0,
    tmax=0.35,
    trial_duration=60.0,
)

print(settings)
# {'segment_length': 256, 'segment_duration': 2.0, 'overlap': 0.5, 'window': 'hann'}
```

You can pass the returned values straight into `train(...)`:

```python
model.train(
    stimulus=stimulus,
    response=response,
    fs=128,
    tmin=0.0,
    tmax=0.35,
    regularization=np.logspace(-6, 1, 8),
    segment_duration=settings["segment_duration"],
    overlap=settings["overlap"],
    window=settings["window"],
    k=4,
)
```

If the helper suggests a full-trial estimate, it returns:

- `segment_length=None`
- `segment_duration=None`
- `overlap=0.0`
- `window=None`

## Interpreting the Suggestion

- A full-trial suggestion means the trials are short enough that splitting them
  further is usually not worth it.
- A segmented Hann suggestion means your lag window is short relative to the
  trial duration, so short overlapping FFT segments are likely to be a more
  stable default.

## When to Deviate

- Increase segment duration if you care more about frequency resolution.
- Decrease segment duration if estimates are noisy and trials are long enough
  to support more segments.
- Try a slightly longer lag window before changing segment settings if backward
  decoding feels too weak.
- Switch to `train_multitaper(...)` when short standard segments still feel
  unstable.

## Practical Defaults by Lag Window

- `0-150 ms`:
  start around `1.0 s` segments with `window="hann"`
- `0-250 ms`:
  start around `2.0 s` segments with `window="hann"`
- `0-400 ms`:
  start around `2.0-3.0 s` segments with `window="hann"`

These are starting points, not guarantees. Once you know the rough regime that
works for your data, tune the lag window and regularization together with the
segment settings instead of changing one knob in isolation.
