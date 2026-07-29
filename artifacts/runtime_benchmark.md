# Runtime benchmark

Synthetic scaling and crossover scenarios for ffTRF and mTRF.

## Environment and provenance

- Generated: 2026-07-29 12:15 UTC
- Source: 5ea0ebd2875f (clean)
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
| Moderate length | forward | 1->1 | fixed | whole-trial | 1000 | 8 | 4096 | 40 | 10.0 | 0.0269 [0.0263, 0.0272] | 0.0096 [0.0095, 0.0096] | 0.36× | 84.2 [83.8, 84.3] | 110.1 [109.1, 110.2] | 1.31× | 1.9 [1.6, 2.0] | 3.6 [2.4, 3.8] | 0.9990 / 0.9990 | 1.0000 |
| Long recording | forward | 1->1 | fixed | whole-trial | 1000 | 4 | 60000 | 40 | 73.2 | 0.3671 [0.3610, 0.3708] | 0.0657 [0.0652, 0.0668] | 0.18× | 115.2 [115.2, 115.2] | 166.6 [165.8, 166.6] | 1.45× | 0.0 [0.0, 0.0] | 51.4 [50.6, 51.4] | 0.9990 / 0.9990 | 1.0000 |
| High rate | forward | 1->1 | fixed | whole-trial | 10000 | 2 | 30000 | 300 | 137.3 | 0.1863 [0.1835, 0.1871] | 0.2030 [0.2024, 0.2061] | 1.09× | 179.2 [179.2, 179.2] | 315.6 [315.6, 316.6] | 1.76× | 0.0 [0.0, 0.0] | 136.4 [136.4, 137.3] | 0.9989 / 0.9989 | 1.0000 |
| Long high rate | forward | 1->1 | fixed | whole-trial | 10000 | 2 | 60000 | 300 | 274.7 | 0.3650 [0.3539, 0.3665] | 0.4019 [0.4017, 0.4031] | 1.10× | 330.4 [330.4, 330.4] | 523.1 [523.0, 523.8] | 1.58× | 0.0 [0.0, 0.0] | 192.7 [192.6, 193.4] | 0.9990 / 0.9990 | 1.0000 |
| Multifeature / multichannel | forward | 3->2 | fixed | whole-trial | 1000 | 6 | 4096 | 40 | 22.5 | 0.0297 [0.0295, 0.0297] | 0.0255 [0.0255, 0.0263] | 0.86× | 536.5 [536.5, 536.5] | 536.5 [536.5, 536.5] | 1.00× | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.9994 / 0.9994 | 1.0000 |
| Longer lag window | forward | 1->1 | fixed | whole-trial | 10000 | 2 | 30000 | 600 | 274.7 | 0.1828 [0.1825, 0.1854] | 0.5720 [0.5716, 0.5745] | 3.13× | 536.5 [536.5, 536.5] | 536.5 [536.5, 536.5] | 1.00× | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.9989 / 0.9989 | 1.0000 |
| Cross-validated ridge | forward | 1->1 | cv-8 (k=4) | whole-trial | 10000 | 4 | 30000 | 300 | 274.7 | 0.2181 [0.2171, 0.2181] | 2.1056 [2.0736, 2.1089] | 9.65× | 548.4 [548.4, 548.4] | 548.4 [548.4, 548.4] | 1.00× | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.9989 / 0.9990 | 1.0000 |
| Segmented Hann estimate | forward | 1->1 | fixed | seg=4096, ov=0.5, hann | 10000 | 2 | 60000 | 300 | 274.7 | 0.0317 [0.0313, 0.0318] | 0.4016 [0.3986, 0.4021] | 12.66× | 548.4 [548.4, 548.4] | 548.4 [548.4, 548.4] | 1.00× | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.9989 / 0.9990 | 1.0000 |
| EEG-scale forward channels | forward | 16->102 | fixed | whole-trial | 128 | 6 | 1024 | 52 | 39.0 | 0.0589 [0.0566, 0.0589] | 0.1404 [0.1400, 0.1412] | 2.38× | 548.7 [548.7, 548.7] | 548.7 [548.7, 548.7] | 1.00× | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.9450 / 0.9293 | 0.9884 |
| 102-channel backward decoder | backward | 102->1 | fixed | whole-trial | 128 | 6 | 1024 | 52 | 248.6 | 0.2626 [0.2593, 0.2639] | 5.3657 [5.3564, 5.3768] | 20.44× | 548.7 [548.7, 548.7] | 988.3 [987.5, 988.7] | 1.80× | 0.0 [0.0, 0.0] | 439.6 [438.8, 440.0] | 0.9711 / 0.8695 | 0.9240 |

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
