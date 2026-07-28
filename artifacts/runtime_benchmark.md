# Runtime benchmark

Synthetic scaling and crossover scenarios for ffTRF and mTRF.

## Environment and provenance

- Generated: 2026-07-28 11:06 UTC
- Source: e714c712670c (clean)
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
| Moderate length | forward | 1->1 | fixed | whole-trial | 1000 | 8 | 4096 | 40 | 10.0 | 0.0273 [0.0269, 0.0276] | 0.0099 [0.0094, 0.0100] | 0.36× | 84.2 [84.0, 84.2] | 111.2 [110.1, 111.7] | 1.32× | 3.0 [3.0, 3.1] | 5.2 [4.2, 6.3] | 0.9990 / 0.9990 | 1.0000 |
| Long recording | forward | 1->1 | fixed | whole-trial | 1000 | 4 | 60000 | 40 | 73.2 | 0.3762 [0.3757, 0.3809] | 0.0632 [0.0620, 0.0646] | 0.17× | 112.6 [112.6, 112.6] | 166.7 [166.6, 166.8] | 1.48× | 0.0 [0.0, 0.0] | 54.0 [54.0, 54.1] | 0.9990 / 0.9990 | 1.0000 |
| High rate | forward | 1->1 | fixed | whole-trial | 10000 | 2 | 30000 | 300 | 137.3 | 0.1903 [0.1903, 0.1948] | 0.2042 [0.2038, 0.2047] | 1.07× | 178.3 [178.3, 178.3] | 316.4 [313.2, 316.6] | 1.77× | 0.0 [0.0, 0.0] | 138.0 [134.9, 138.3] | 0.9989 / 0.9989 | 1.0000 |
| Long high rate | forward | 1->1 | fixed | whole-trial | 10000 | 2 | 60000 | 300 | 274.7 | 0.3747 [0.3671, 0.3807] | 0.4021 [0.4006, 0.4027] | 1.07× | 330.0 [330.0, 330.0] | 522.5 [520.5, 523.4] | 1.58× | 0.0 [0.0, 0.0] | 192.5 [190.5, 193.4] | 0.9990 / 0.9990 | 1.0000 |
| Multifeature / multichannel | forward | 3->2 | fixed | whole-trial | 1000 | 6 | 4096 | 40 | 22.5 | 0.0301 [0.0300, 0.0312] | 0.0259 [0.0245, 0.0268] | 0.86× | 536.3 [536.3, 536.3] | 536.3 [536.3, 536.3] | 1.00× | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.9994 / 0.9994 | 1.0000 |
| Longer lag window | forward | 1->1 | fixed | whole-trial | 10000 | 2 | 30000 | 600 | 274.7 | 0.1915 [0.1876, 0.1928] | 0.5799 [0.5783, 0.5804] | 3.03× | 536.3 [536.3, 536.3] | 536.3 [536.3, 536.3] | 1.00× | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.9989 / 0.9989 | 1.0000 |
| Cross-validated ridge | forward | 1->1 | cv-8 (k=4) | whole-trial | 10000 | 4 | 30000 | 300 | 274.7 | 0.2248 [0.2246, 0.2313] | 2.0396 [2.0373, 2.0543] | 9.07× | 548.5 [548.5, 548.5] | 548.5 [548.5, 548.5] | 1.00× | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.9989 / 0.9990 | 1.0000 |
| Segmented Hann estimate | forward | 1->1 | fixed | seg=4096, ov=0.5, hann | 10000 | 2 | 60000 | 300 | 274.7 | 0.0321 [0.0318, 0.0330] | 0.4010 [0.4009, 0.4020] | 12.49× | 548.5 [548.5, 548.5] | 548.5 [548.5, 548.5] | 1.00× | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.9989 / 0.9990 | 1.0000 |
| EEG-scale forward channels | forward | 16->102 | fixed | whole-trial | 128 | 6 | 1024 | 52 | 39.0 | 0.0585 [0.0581, 0.0588] | 0.1435 [0.1433, 0.1488] | 2.45× | 549.0 [549.0, 549.0] | 549.0 [549.0, 549.0] | 1.00× | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.9450 / 0.9293 | 0.9884 |
| 102-channel backward decoder | backward | 102->1 | fixed | whole-trial | 128 | 6 | 1024 | 52 | 248.6 | 0.2546 [0.2535, 0.2560] | 5.4834 [5.4803, 5.5029] | 21.54× | 549.0 [549.0, 549.0] | 987.7 [987.7, 988.5] | 1.80× | 0.0 [0.0, 0.0] | 438.8 [438.7, 439.5] | 0.9711 / 0.8695 | 0.9240 |

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
