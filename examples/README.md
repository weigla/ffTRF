# Examples

This directory contains optional demo and comparison code that is not part of
the core `fft_trf` library API.

The main installable toolbox lives in:

- `src/fft_trf/`

The files here are intended for:

- sanity checks against time-domain references
- side-by-side comparisons with `mTRFpy`
- exploratory plotting for development and validation

Run the comparison demo with Pixi:

```bash
pixi run -e compare compare-demo
```

Or with pip:

```bash
pip install -e ".[compare]" mtrf
python examples/compare_with_mtrf.py --output artifacts/kernel_comparison.png --no-show
```
