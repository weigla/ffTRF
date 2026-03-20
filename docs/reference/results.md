# Result Containers

Several `TRF` methods return lightweight dataclass-based containers instead of
bare tuples. These objects make it easier to keep related arrays together.

## Why These Containers Exist

They serve three purposes:

- they document the shape and meaning of returned arrays
- they provide small convenience methods such as `at(...)`
- they keep related metadata, such as band centers or phase units, attached to
  the data they describe

`CrossSpectralDiagnostics` is a public alias of `TRFDiagnostics`, so users can
choose whichever name reads more naturally in their code.

## Containers

::: fftrf.TRFDiagnostics

::: fftrf.FrequencyResolvedWeights

::: fftrf.TimeFrequencyPower

::: fftrf.TransferFunctionComponents
