# Runtime benchmark

Synthetic scaling and crossover scenarios for ffTRF and mTRF.

## Environment and provenance

- Generated: 2026-07-28 09:13 UTC
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

## Protocol

- Command: `pixi run -e compare benchmark-demo`
Each row uses the same simulated data for both methods. Forward rows fit
predictor-to-response TRFs, while backward rows fit response-to-predictor
decoders. Fixed-ridge scenarios use the same lambda in both toolboxes,
and the cross-validated scenario uses the same candidate grid, seed,
and Python-random fold shuffle in both.
Fit time is the median and [minimum, maximum] of 3 isolated
fit-only run(s), following 1 unreported warmup run(s).
Held-out prediction scores are mean Pearson correlations over outputs.
Kernel correlation is computed over the flattened full kernel bank.
Total peak RSS includes the interpreter, imports, and simulated data.
Additional peak RSS is growth above the process peak immediately before fitting.

Ratios are `mTRF / ffTRF`; values above 1 favor ffTRF, while values
below 1 favor mTRF.

## Results

| Scenario | Direction | Shape | Fit mode | FFT setting | fs (Hz) | Trials | Samples/trial | Lags | Implied lag matrix (MiB) | ffTRF fit s median [range] | mTRF fit s median [range] | Runtime ratio | ffTRF total peak MiB median [range] | mTRF total peak MiB median [range] | Peak RSS ratio | ffTRF additional peak MiB median [range] | mTRF additional peak MiB median [range] | Held-out r ffTRF / mTRF | Kernel corr. |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Moderate length | forward | 1->1 | fixed | whole-trial | 1000 | 8 | 4096 | 40 | 10.0 | 0.0205 [0.0205, 0.0207] | 0.0107 [0.0105, 0.0119] | 0.52× | 97.8 [97.6, 98.1] | 126.9 [126.1, 129.3] | 1.30× | 0.0 [0.0, 0.0] | 2.6 [2.5, 3.9] | 0.9990 / 0.9990 | 1.0000 |
| Long recording | forward | 1->1 | fixed | whole-trial | 1000 | 4 | 60000 | 40 | 73.2 | 0.2928 [0.2868, 0.3664] | 0.0569 [0.0537, 0.0600] | 0.19× | 104.8 [103.6, 104.8] | 182.6 [178.1, 183.4] | 1.74× | 3.6 [2.1, 3.8] | 54.6 [52.9, 54.6] | 0.9990 / 0.9990 | 1.0000 |
| High rate | forward | 1->1 | fixed | whole-trial | 10000 | 2 | 30000 | 300 | 137.3 | 0.1436 [0.1426, 0.1759] | 0.2135 [0.2051, 0.2440] | 1.49× | 99.8 [99.2, 100.4] | 334.4 [334.2, 334.8] | 3.35× | 1.2 [0.1, 1.5] | 208.5 [206.9, 209.7] | 0.9989 / 0.9989 | 1.0000 |
| Long high rate | forward | 1->1 | fixed | whole-trial | 10000 | 2 | 60000 | 300 | 274.7 | 0.2957 [0.2942, 0.3029] | 0.3769 [0.3556, 0.3954] | 1.27× | 103.0 [102.4, 104.0] | 540.2 [539.2, 540.5] | 5.25× | 2.7 [2.0, 2.9] | 414.2 [413.8, 414.3] | 0.9990 / 0.9990 | 1.0000 |
| Multifeature / multichannel | forward | 3->2 | fixed | whole-trial | 1000 | 6 | 4096 | 40 | 22.5 | 0.0232 [0.0229, 0.0239] | 0.0253 [0.0247, 0.0264] | 1.09× | 99.0 [98.6, 99.6] | 136.6 [134.6, 136.7] | 1.38× | 0.8 [0.7, 1.0] | 10.6 [10.6, 10.7] | 0.9994 / 0.9994 | 1.0000 |
| Longer lag window | forward | 1->1 | fixed | whole-trial | 10000 | 2 | 30000 | 600 | 274.7 | 0.1474 [0.1434, 0.1492] | 0.6428 [0.6008, 0.6831] | 4.36× | 99.8 [98.5, 100.1] | 541.5 [475.7, 543.4] | 5.43× | 0.8 [0.4, 1.5] | 417.2 [350.5, 417.5] | 0.9989 / 0.9989 | 1.0000 |
| Cross-validated ridge | forward | 1->1 | cv-8 (k=4) | whole-trial | 10000 | 4 | 30000 | 300 | 274.7 | 0.1663 [0.1638, 0.1718] | 1.4144 [1.3940, 1.4394] | 8.51× | 104.8 [102.6, 108.2] | 346.5 [340.6, 350.3] | 3.31× | 4.8 [4.8, 8.2] | 221.6 [216.0, 224.0] | 0.9989 / 0.9990 | 1.0000 |
| Segmented Hann estimate | forward | 1->1 | fixed | seg=4096, ov=0.5, hann | 10000 | 2 | 60000 | 300 | 274.7 | 0.0245 [0.0236, 0.0245] | 0.4198 [0.4174, 0.4307] | 17.16× | 98.8 [98.5, 100.3] | 540.0 [539.5, 541.0] | 5.46× | 0.0 [0.0, 0.0] | 413.8 [413.8, 414.2] | 0.9989 / 0.9990 | 1.0000 |
| EEG-scale forward channels | forward | 16->102 | fixed | whole-trial | 128 | 6 | 1024 | 52 | 39.0 | 0.0413 [0.0393, 0.0416] | 0.1393 [0.1379, 0.1424] | 3.37× | 160.7 [160.3, 168.7] | 164.6 [164.3, 165.0] | 1.02× | 58.5 [57.5, 63.8] | 34.7 [34.6, 34.7] | 0.9450 / 0.9293 | 0.9884 |
| 102-channel backward decoder | backward | 102->1 | fixed | whole-trial | 128 | 6 | 1024 | 52 | 248.6 | 0.1609 [0.1596, 0.2152] | 5.7212 [5.7120, 5.8533] | 35.55× | 347.9 [347.6, 352.0] | 1054.9 [852.5, 1127.5] | 3.03× | 246.8 [246.6, 249.3] | 926.8 [723.0, 998.7] | 0.9711 / 0.8695 | 0.9240 |

Interpretation:
- The approximate lag-matrix size is shown because it dominates the memory footprint of a standard time-domain fit and grows with both lag count and predictor count.
- `ffTRF held-out r` and `mTRF held-out r` are the main accuracy columns: they measure mean Pearson correlation on a separate held-out simulation split generated from the same ground-truth kernel.
- Kernel correlations close to 1 indicate that the two methods recover nearly the same flattened kernel bank. This is most interpretable for forward models; backward decoders can differ more in weight space while still making very similar predictions.
- Direct fixed-lambda `TRF` fits now use an aggregated lower-memory spectral path automatically, so the fixed-ridge rows reflect the lighter-weight solver rather than the heavier CV cache path.
- The key outcome is not a universal speed win. Short, simple fixed-ridge 1-to-1 fits can be similar to or slower than mTRF, while larger lag counts, CV grids, segmented spectra, and high-dimensional decoders favor ffTRF much more strongly.
- Cached spectra matter most in the cross-validated scenario because `TRF` can reuse FFT work across lambda candidates, even if that does not automatically make it faster than `mTRF` on every machine.
- The segmented Hann scenario is intentionally not the closest mTRF-like setting; it shows the cost of a more typical spectral-estimation workflow.
- The EEG-scale forward and 102-channel backward rows show how the trade-off changes once the output side becomes sensor-rich or the backward decoder has many predictor channels.
- Memory is part of the story: the main practical advantage of ffTRF often shows up as a combination of competitive runtime and much lower peak RSS once the implied lag matrix would become large.
- Peak RSS is measured per fit in a fresh worker process. Total and additional peaks are both reported so process overhead is not mistaken for fit allocation.
