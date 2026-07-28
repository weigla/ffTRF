# Real EEG benchmark

Matched solver and practical estimator comparisons on the pinned public
mTRF speech-EEG sample.

## Environment and provenance

- Generated: 2026-07-28 11:33 UTC
- Source: e714c712670c (dirty)
- CPU: AMD EPYC 9334 32-Core Processor
- Platform: Linux-5.14.0-687.26.1.el9_8.x86_64-x86_64-with-glibc2.34
- Machine: x86_64
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
| Forward | ffTRF | Matched whole-trial | 10000 | 0.0296 | 0.0345 | 5.9376 [5.9366, 5.9420] | 619.3 [618.0, 620.7] | 489.5 [489.5, 489.5] |
| Forward | mTRF | Finite-lag baseline | 3162.28 | 0.0200 | 0.0172 | 6.0387 [6.0210, 6.0615] | 260.2 [259.4, 260.3] | 113.7 [113.3, 114.1] |
| Backward | ffTRF | Matched whole-trial | 1e+06 | 0.0469 | 0.0370 | 7.6348 [7.6043, 7.6469] | 3401.8 [3400.2, 3401.8] | 3270.6 [3270.6, 3270.7] |
| Backward | mTRF | Finite-lag baseline | 1000 | 0.1109 | 0.1046 | 474.7536 [473.3494, 476.9489] | 3963.6 [3963.6, 3963.6] | 3817.2 [3817.1, 3817.4] |

| Direction | Runtime ratio (mTRF / ffTRF) | Total peak RSS ratio (mTRF / ffTRF) |
| --- | ---: | ---: |
| Forward | 1.02× | 0.42× |
| Backward | 62.18× | 1.17× |

Ratios above 1 favor ffTRF; ratios below 1 favor mTRF.

## Practical 2 s Hann comparison

ffTRF uses 2-second Hann windows with 50% overlap. This changes the
spectral estimator, so the result is a practical workflow comparison,
not a strict solver-equivalence claim. The mTRF baseline is unchanged.

| Direction | Model | Configuration | Lambda | Mean held-out r | Median held-out r | Fit seconds median [range] | Total peak RSS MiB median [range] | Additional peak RSS MiB median [range] |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Forward | ffTRF | Practical 2 s Hann | 10000 | 0.0367 | 0.0386 | 2.8469 [2.8346, 2.8527] | 211.5 [210.9, 211.6] | 80.8 [80.2, 81.1] |
| Forward | mTRF | Finite-lag baseline | 3162.28 | 0.0200 | 0.0172 | 6.0387 [6.0210, 6.0615] | 260.2 [259.4, 260.3] | 113.7 [113.3, 114.1] |
| Backward | ffTRF | Practical 2 s Hann | 1000 | 0.1954 | 0.1762 | 2.4517 [2.4325, 2.5982] | 681.9 [680.2, 682.3] | 550.9 [549.7, 551.1] |
| Backward | mTRF | Finite-lag baseline | 1000 | 0.1109 | 0.1046 | 474.7536 [473.3494, 476.9489] | 3963.6 [3963.6, 3963.6] | 3817.2 [3817.1, 3817.4] |

| Direction | Runtime ratio (mTRF / ffTRF) | Total peak RSS ratio (mTRF / ffTRF) |
| --- | ---: | ---: |
| Forward | 2.12× | 1.23× |
| Backward | 193.64× | 5.81× |

Ratios above 1 favor ffTRF; ratios below 1 favor mTRF.
