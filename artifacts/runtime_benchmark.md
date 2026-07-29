# Runtime benchmark

Synthetic scaling and crossover scenarios for ffTRF and mTRF.

## Environment and provenance

- Generated: 2026-07-29 12:04 UTC
- Source: 293209731bf4 (dirty)
- CPU: AMD EPYC 9334 32-Core Processor
- Platform: Linux-5.14.0-687.26.1.el9_8.x86_64-x86_64-with-glibc2.34
- Machine: x86_64
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
| Moderate length | forward | 1->1 | fixed | whole-trial | 1000 | 8 | 4096 | 40 | 10.0 | 0.0262 [0.0261, 0.0267] | 0.0095 [0.0095, 0.0096] | 0.36× | 84.3 [84.1, 84.3] | 110.1 [108.5, 110.2] | 1.31× | 2.0 [1.8, 2.0] | 4.0 [3.3, 4.6] | 0.9990 / 0.9990 | 1.0000 |
| Long recording | forward | 1->1 | fixed | whole-trial | 1000 | 4 | 60000 | 40 | 73.2 | 0.3681 [0.3659, 0.3685] | 0.0617 [0.0614, 0.0637] | 0.17× | 114.5 [114.5, 114.5] | 166.6 [166.1, 166.7] | 1.46× | 0.0 [0.0, 0.0] | 52.2 [51.6, 52.2] | 0.9990 / 0.9990 | 1.0000 |
| High rate | forward | 1->1 | fixed | whole-trial | 10000 | 2 | 30000 | 300 | 137.3 | 0.1831 [0.1797, 0.1844] | 0.1971 [0.1969, 0.1993] | 1.08× | 179.1 [179.1, 179.1] | 316.0 [315.9, 316.8] | 1.76× | 0.0 [0.0, 0.0] | 136.9 [136.8, 137.7] | 0.9989 / 0.9989 | 1.0000 |
| Long high rate | forward | 1->1 | fixed | whole-trial | 10000 | 2 | 60000 | 300 | 274.7 | 0.3623 [0.3598, 0.3639] | 0.3877 [0.3871, 0.3878] | 1.07× | 329.8 [329.8, 329.8] | 522.3 [521.8, 522.4] | 1.58× | 0.0 [0.0, 0.0] | 192.5 [192.0, 192.6] | 0.9990 / 0.9990 | 1.0000 |
| Multifeature / multichannel | forward | 3->2 | fixed | whole-trial | 1000 | 6 | 4096 | 40 | 22.5 | 0.0296 [0.0296, 0.0298] | 0.0255 [0.0247, 0.0258] | 0.86× | 535.9 [535.9, 535.9] | 535.9 [535.9, 535.9] | 1.00× | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.9994 / 0.9994 | 1.0000 |
| Longer lag window | forward | 1->1 | fixed | whole-trial | 10000 | 2 | 30000 | 600 | 274.7 | 0.1861 [0.1830, 0.1867] | 0.5611 [0.5603, 0.5620] | 3.01× | 535.9 [535.9, 535.9] | 535.9 [535.9, 535.9] | 1.00× | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.9989 / 0.9989 | 1.0000 |
| Cross-validated ridge | forward | 1->1 | cv-8 (k=4) | whole-trial | 10000 | 4 | 30000 | 300 | 274.7 | 0.2169 [0.2167, 0.2203] | 1.9923 [1.9881, 2.0328] | 9.19× | 548.6 [548.6, 548.6] | 548.6 [548.6, 548.6] | 1.00× | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.9989 / 0.9990 | 1.0000 |
| Segmented Hann estimate | forward | 1->1 | fixed | seg=4096, ov=0.5, hann | 10000 | 2 | 60000 | 300 | 274.7 | 0.0317 [0.0313, 0.0318] | 0.3879 [0.3875, 0.3882] | 12.24× | 548.6 [548.6, 548.6] | 548.6 [548.6, 548.6] | 1.00× | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.9989 / 0.9990 | 1.0000 |
| EEG-scale forward channels | forward | 16->102 | fixed | whole-trial | 128 | 6 | 1024 | 52 | 39.0 | 0.0559 [0.0556, 0.0566] | 0.1387 [0.1382, 0.1397] | 2.48× | 549.8 [549.8, 549.8] | 549.8 [549.8, 549.8] | 1.00× | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.9450 / 0.9293 | 0.9884 |
| 102-channel backward decoder | backward | 102->1 | fixed | whole-trial | 128 | 6 | 1024 | 52 | 248.6 | 0.2457 [0.2434, 0.2471] | 5.3056 [5.2852, 5.3126] | 21.60× | 549.8 [549.8, 549.8] | 987.5 [986.7, 988.9] | 1.80× | 0.0 [0.0, 0.0] | 437.7 [437.0, 439.1] | 0.9711 / 0.8695 | 0.9240 |

Interpretation:
- The approximate lag-matrix size is shown because it dominates the memory footprint of a standard time-domain fit and grows with both lag count and predictor count.
- `ffTRF held-out r` and `mTRF held-out r` are the main accuracy columns: they measure mean Pearson correlation on a separate held-out simulation split generated from the same ground-truth kernel.
- Kernel correlations close to 1 indicate that the two methods recover nearly the same flattened kernel bank. This is most interpretable for forward models; backward decoders can differ more in weight space while still making very similar predictions.
- The key outcome is not a universal speed win. Short, simple fixed-ridge 1-to-1 fits can be similar to or slower than mTRF, while larger lag counts, CV grids, segmented spectra, and high-dimensional decoders favor ffTRF much more strongly.
- Cached spectra matter most in the cross-validated scenario because `TRF` can reuse FFT work across lambda candidates, even if that does not automatically make it faster than `mTRF` on every machine.
- The segmented Hann scenario is intentionally not the closest mTRF-like setting; it shows the cost of a more typical spectral-estimation workflow.
- The EEG-scale forward and 102-channel backward rows show how the trade-off changes once the output side becomes sensor-rich or the backward decoder has many predictor channels.
- Memory is part of the story: the main practical advantage of ffTRF often shows up as a combination of competitive runtime and much lower peak RSS once the implied lag matrix would become large.
- Peak RSS is measured per fit in a fresh worker process. Total and additional peaks are both reported so process overhead is not mistaken for fit allocation.
