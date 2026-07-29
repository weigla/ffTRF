# Real EEG benchmark

Matched solver and practical estimator comparisons on the pinned public
mTRF speech-EEG sample.

## Environment and provenance

- Generated: 2026-07-29 11:58 UTC
- Source: 293209731bf4 (clean)
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
| Forward | ffTRF | Matched whole-trial | 10000 | 0.0296 | 0.0345 | 5.6706 [5.6673, 5.6977] | 620.7 [620.4, 621.0] | 489.5 [489.5, 490.5] |
| Forward | mTRF | Finite-lag baseline | 3162.28 | 0.0200 | 0.0172 | 5.8575 [5.8331, 5.8725] | 260.2 [257.8, 260.2] | 113.6 [113.2, 113.6] |
| Backward | ffTRF | Matched whole-trial | 1e+06 | 0.0469 | 0.0370 | 7.1850 [7.1820, 7.1963] | 3401.8 [3400.2, 3401.8] | 3271.2 [3269.9, 3271.4] |
| Backward | mTRF | Finite-lag baseline | 1000 | 0.1109 | 0.1046 | 458.0585 [457.3234, 458.3719] | 3963.7 [3963.5, 3963.7] | 3817.3 [3816.8, 3818.2] |

| Direction | Runtime ratio (mTRF / ffTRF) | Total peak RSS ratio (mTRF / ffTRF) |
| --- | ---: | ---: |
| Forward | 1.03× | 0.42× |
| Backward | 63.75× | 1.17× |

Ratios above 1 favor ffTRF; ratios below 1 favor mTRF.

## Practical 2 s Hann comparison

ffTRF uses 2-second Hann windows with 50% overlap. This changes the
spectral estimator, so the result is a practical workflow comparison,
not a strict solver-equivalence claim. The mTRF baseline is unchanged.

| Direction | Model | Configuration | Lambda | Mean held-out r | Median held-out r | Fit seconds median [range] | Total peak RSS MiB median [range] | Additional peak RSS MiB median [range] |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Forward | ffTRF | Practical 2 s Hann | 10000 | 0.0367 | 0.0386 | 2.7150 [2.7142, 2.7603] | 211.6 [210.0, 211.6] | 80.2 [79.5, 80.6] |
| Forward | mTRF | Finite-lag baseline | 3162.28 | 0.0200 | 0.0172 | 5.8575 [5.8331, 5.8725] | 260.2 [257.8, 260.2] | 113.6 [113.2, 113.6] |
| Backward | ffTRF | Practical 2 s Hann | 1000 | 0.1954 | 0.1762 | 2.2055 [2.1735, 2.2158] | 681.8 [681.6, 681.9] | 551.0 [550.7, 551.4] |
| Backward | mTRF | Finite-lag baseline | 1000 | 0.1109 | 0.1046 | 458.0585 [457.3234, 458.3719] | 3963.7 [3963.5, 3963.7] | 3817.3 [3816.8, 3818.2] |

| Direction | Runtime ratio (mTRF / ffTRF) | Total peak RSS ratio (mTRF / ffTRF) |
| --- | ---: | ---: |
| Forward | 2.16× | 1.23× |
| Backward | 207.69× | 5.81× |

Ratios above 1 favor ffTRF; ratios below 1 favor mTRF.
