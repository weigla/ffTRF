# Real EEG benchmark

Matched solver and practical estimator comparisons on the pinned public
mTRF speech-EEG sample.

## Environment and provenance

- Generated: 2026-07-28 09:44 UTC
- Source: 1df09985750d (dirty)
- CPU: Apple M3
- Platform: macOS-26.4-arm64-arm-64bit-Mach-O
- Machine: arm64
- Python: 3.13.12
- ffTRF: 0.1.0
- mTRF: 2.1.2
- NumPy: 2.4.2
- SciPy: 1.17.1
- Native threads per worker: 1
- Dataset commit: `9b89449caaed3a4b7c80ea238a52c34a723cb8de`
- Dataset SHA-256: `5726060e254caac865c5ca7cf56a8218937f4c05b7784fb08d11658748daee36`

## Protocol

- Command: `pixi run -e compare real-eeg-benchmark`
- Data: 10 twelve-second segments at 128 Hz; 7 train/CV and 3 held out.
- Forward model: 16 speech bands to 128 EEG channels, 5-fold CV.
- Backward model: 128 EEG channels to a compressed broadband envelope, 3-fold CV.
- Selection: seeded folds (`seed=7`) and negative MSE over the same direction-specific lambda grid.
- Accuracy: held-out Pearson correlation across channels (forward) or segments (backward). It is a prediction check, not ground-truth kernel accuracy.
- Timing: median and [minimum, maximum] of 3 isolated fit-only run(s); 0 unreported warmup run(s).
- Total peak RSS includes the interpreter, imports, and loaded data.
- Additional peak RSS is growth above the process peak immediately before fitting.
- Prediction, plotting, data download, and data loading are excluded from fit time.

## Matched whole-trial comparison

ffTRF uses whole-trial rectangular spectra with no segmentation,
windowing, multitaper smoothing, or detrending. This is the closest
comparison to the finite-lag mTRF fit.

| Direction | Model | Configuration | Lambda | Mean held-out r | Median held-out r | Fit seconds median [range] | Total peak RSS MiB median [range] | Additional peak RSS MiB median [range] |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Forward | ffTRF | Matched whole-trial | 10000 | 0.0296 | 0.0345 | 5.0341 [5.0047, 5.0771] | 737.1 [692.0, 818.1] | 586.4 [546.9, 671.9] |
| Forward | mTRF | Finite-lag baseline | 3162.28 | 0.0200 | 0.0172 | 5.1227 [5.0373, 5.2225] | 433.4 [431.6, 435.0] | 261.6 [260.2, 262.2] |
| Backward | ffTRF | Matched whole-trial | 1e+06 | 0.0469 | 0.0370 | 7.3326 [7.3229, 7.7221] | 3352.6 [3240.0, 3400.4] | 3202.0 [3093.0, 3254.5] |
| Backward | mTRF | Finite-lag baseline | 1000 | 0.1109 | 0.1046 | 585.0948 [517.6117, 619.4002] | 3852.9 [3792.2, 3986.8] | 3675.6 [3620.6, 3808.0] |

| Direction | Runtime ratio (mTRF / ffTRF) | Total peak RSS ratio (mTRF / ffTRF) |
| --- | ---: | ---: |
| Forward | 1.02× | 0.59× |
| Backward | 79.79× | 1.15× |

Ratios above 1 favor ffTRF; ratios below 1 favor mTRF.

## Practical 2 s Hann comparison

ffTRF uses 2-second Hann windows with 50% overlap. This changes the
spectral estimator, so the result is a practical workflow comparison,
not a strict solver-equivalence claim. The mTRF baseline is unchanged.

| Direction | Model | Configuration | Lambda | Mean held-out r | Median held-out r | Fit seconds median [range] | Total peak RSS MiB median [range] | Additional peak RSS MiB median [range] |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Forward | ffTRF | Practical 2 s Hann | 10000 | 0.0367 | 0.0386 | 2.1748 [2.1545, 2.1880] | 327.5 [324.2, 344.1] | 176.2 [173.8, 182.6] |
| Forward | mTRF | Finite-lag baseline | 3162.28 | 0.0200 | 0.0172 | 5.1227 [5.0373, 5.2225] | 433.4 [431.6, 435.0] | 261.6 [260.2, 262.2] |
| Backward | ffTRF | Practical 2 s Hann | 1000 | 0.1954 | 0.1762 | 1.6923 [1.6586, 1.7151] | 810.4 [808.2, 812.6] | 664.1 [663.0, 665.9] |
| Backward | mTRF | Finite-lag baseline | 1000 | 0.1109 | 0.1046 | 585.0948 [517.6117, 619.4002] | 3852.9 [3792.2, 3986.8] | 3675.6 [3620.6, 3808.0] |

| Direction | Runtime ratio (mTRF / ffTRF) | Total peak RSS ratio (mTRF / ffTRF) |
| --- | ---: | ---: |
| Forward | 2.36× | 1.32× |
| Backward | 345.74× | 4.75× |

Ratios above 1 favor ffTRF; ratios below 1 favor mTRF.
