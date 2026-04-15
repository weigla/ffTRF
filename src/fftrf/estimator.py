
"""Main frequency-domain TRF estimator."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from scipy.signal import hilbert

from .metrics import MetricSpec, pearsonr, _resolve_metric
from .prediction import (
    _aggregate_null_scores,
    _build_permutation_specs,
    _compute_bootstrap_interval_from_cache,
    _compute_permutation_test_scores,
    _extract_impulse_response,
    _permutation_p_value_and_z_score,
    _predict_trials_from_weights,
    _resolve_permutation_surrogate,
    _resolve_permutation_tail,
    _score_prediction_trials,
    _score_regularization_grid_for_fold,
    _slice_interval,
    _surrogate_target_trials,
    _validate_confidence_level,
)
from .results import (
    PermutationTestResult,
    FrequencyResolvedWeights,
    TRFDiagnostics,
    TimeFrequencyPower,
    TransferFunctionComponents,
)
from .spectral import (
    SpectralMethod,
    _SpectralCache,
    _aggregate_cached_spectra,
    _build_spectral_cache,
    _prepare_scalar_ridge_decomposition,
    _resolve_multitaper_parameters,
    _scalar_regularization_grid,
    _solve_transfer_function,
    _validate_spectral_method,
)
from .utils import (
    _SimpleProgressBar,
    _aggregate_metric,
    _build_frequency_filterbank,
    _check_trial_lengths,
    _coerce_trials,
    _copy_value,
    _group_delay_values,
    _normalize_trial_weights,
    _phase_values,
    _resolve_frequency_weight_value_mode,
    _resolve_k_folds,
    _resolve_n_jobs,
    _resolve_raw_trial_weights,
    _resolve_regularization_candidates,
    _resolve_segment_length,
    _validate_bands,
    _warn_if_cv_arguments_are_unused,
)

_USE_STORED_TRIAL_WEIGHTS = object()
RegularizationSpec = float | tuple[float, ...]


class TRF:
    """
    Estimate stimulus-response mappings in the frequency domain.

    ``TRF`` is the main estimator of this toolbox. Its public API is
    intentionally close to ``mTRF``:

    - call :meth:`train` to fit the model
    - call :meth:`predict` to generate predicted responses or stimuli
    - call :meth:`score` to evaluate predictions
    - call :meth:`permutation_test` to assess held-out prediction scores
      against surrogate nulls
    - call :meth:`plot` to visualize the fitted kernel
    - call :meth:`plot_grid` to visualize all input/output kernels at once
    - call :meth:`frequency_resolved_weights` or
      :meth:`plot_frequency_resolved_weights` for a spectrogram-like kernel view
    - call :meth:`time_frequency_power` or :meth:`plot_time_frequency_power`
      for a smoothed spectrogram-like power view of the kernel
    - call :meth:`plot_transfer_function` to inspect magnitude, phase, or group delay
    - call :meth:`cross_spectral_diagnostics`, :meth:`plot_coherence`, and
      :meth:`plot_cross_spectrum` for spectral prediction diagnostics
    - inspect :attr:`weights` and :attr:`times` as the time-domain kernel

    Unlike a classic time-domain TRF, the fit is performed through
    ridge-regularized spectral deconvolution. This is often attractive for
    high-rate continuous data where explicitly building large lag matrices is
    cumbersome. When multiple regularization values are supplied, the estimator
    caches per-trial spectral statistics so cross-validation can reuse them
    instead of recomputing FFTs for every fold and candidate value. In direct
    single-lambda fits, the estimator automatically uses an aggregated
    lower-memory spectral path because no per-trial cache is needed.

    Parameters
    ----------
    direction:
        Modeling direction. Use ``1`` for a forward model
        (stimulus -> neural response) and ``-1`` for a backward model
        (neural response -> stimulus).
    metric:
        Callable or built-in metric name used to score predictions. It must
        accept ``(y_true, y_pred)`` and return one score per output column.
        Built-ins currently include ``"pearsonr"``, ``"r2"``,
        ``"explained_variance"``, and ``"neg_mse"``.
        This metric does not change the underlying
        TRF solver: fitting is still ridge-regularized spectral deconvolution.
        The metric is only used when scoring predictions, selecting among
        regularization candidates during cross-validation, and returning scores
        from :meth:`predict` or :meth:`score`.

    Attributes
    ----------
    transfer_function:
        Complex-valued frequency-domain mapping with shape
        ``(n_frequencies, n_inputs, n_outputs)``.
    frequencies:
        Frequency vector in Hz corresponding to ``transfer_function``.
    weights:
        Time-domain kernel extracted from ``transfer_function`` over the fitted
        lag window. Shape is ``(n_inputs, n_lags, n_outputs)``.
    times:
        Lag values in seconds corresponding to the second axis of
        :attr:`weights`.
    regularization:
        Selected ridge parameter. In ordinary ridge mode this is a scalar. When
        ``bands`` are used, it stores one coefficient per band.
    bands:
        Optional contiguous feature-group definition used for banded
        regularization.
    feature_regularization:
        Expanded per-feature penalty vector actually used by the spectral
        solver. This is especially useful when banded regularization is active.
    regularization_candidates:
        Candidate ridge values or banded coefficient tuples evaluated during
        training. This lets cross-validation scores be mapped back to the
        tested values.
    fs:
        Sampling rate used during fitting.
    segment_duration:
        Segment length expressed in seconds. This mirrors :attr:`segment_length`
        in a more user-friendly unit.
    bootstrap_interval:
        Optional trial-bootstrap confidence interval with shape
        ``(2, n_inputs, n_lags, n_outputs)``.
    bootstrap_level:
        Confidence level used for :attr:`bootstrap_interval`.
    spectral_method:
        Spectral estimator used during fitting. ``"standard"`` denotes the
        default windowed FFT estimator and ``"multitaper"`` activates DPSS
        multi-taper averaging.
    time_bandwidth, n_tapers:
        Multi-taper settings stored for fitted models that use
        ``spectral_method="multitaper"``.

    Examples
    --------
    >>> import numpy as np
    >>> from fftrf import TRF
    >>> x = np.random.randn(2000, 1)
    >>> y = np.random.randn(2000, 1)
    >>> model = TRF(direction=1)
    >>> model.train(x, y, fs=1000, tmin=0.0, tmax=0.03, regularization=1e-3)
    >>> prediction = model.predict(stimulus=x)
    """

    def __init__(
        self,
        direction: int = 1,
        metric: MetricSpec = pearsonr,
    ) -> None:
        if direction not in (1, -1):
            raise ValueError("direction must be 1 (forward) or -1 (backward).")

        self.direction = direction
        self.metric, self.metric_name = _resolve_metric(metric)

        self.transfer_function: np.ndarray | None = None
        self.frequencies: np.ndarray | None = None
        self.weights: np.ndarray | None = None
        self.times: np.ndarray | None = None

        self.fs: float | None = None
        self.regularization: RegularizationSpec | None = None
        self.bands: tuple[int, ...] | None = None
        self.feature_regularization: np.ndarray | None = None
        self.regularization_candidates: list[RegularizationSpec] | None = None
        self.segment_length: int | None = None
        self.segment_duration: float | None = None
        self.n_fft: int | None = None
        self.overlap: float | None = None
        self.spectral_method: SpectralMethod = "standard"
        self.time_bandwidth: float | None = None
        self.n_tapers: int | None = None
        self.window: None | str | tuple[str, float] | np.ndarray = None
        self.detrend: None | str = None
        self.tmin: float | None = None
        self.tmax: float | None = None
        self.trial_weights: None | str | np.ndarray = None
        self.bootstrap_interval: np.ndarray | None = None
        self.bootstrap_level: float | None = None
        self.bootstrap_samples: int | None = None
        self._fit_config: dict[str, object] | None = None

    def train(
        self,
        stimulus: np.ndarray | Sequence[np.ndarray],
        response: np.ndarray | Sequence[np.ndarray],
        fs: float,
        tmin: float,
        tmax: float,
        regularization: float | Sequence[float] | Sequence[Sequence[float]],
        *,
        bands: None | Sequence[int] = None,
        segment_length: int | None = None,
        segment_duration: float | None = None,
        overlap: float = 0.0,
        n_fft: int | None = None,
        spectral_method: SpectralMethod = "standard",
        time_bandwidth: float = 3.5,
        n_tapers: int | None = None,
        window: None | str | tuple[str, float] | np.ndarray = None,
        detrend: None | str = "constant",
        k: int | str = -1,
        average: bool | Sequence[int] = True,
        seed: int | None = None,
        show_progress: bool = False,
        n_jobs: int | None = 1,
        trial_weights: None | str | Sequence[float] = None,
        bootstrap_samples: int = 0,
        bootstrap_level: float = 0.95,
        bootstrap_seed: int | None = None,
    ) -> np.ndarray | float | None:
        """
        Fit the frequency-domain TRF.

        Parameters
        ----------
        stimulus:
            One trial as a 1D/2D array or multiple trials as a list of arrays.
            Each trial must have shape ``(n_samples, n_features)``. A 1D vector
            is treated as a single-feature input.
        response:
            Neural data with one trial as a 1D/2D array or multiple trials as a
            list of arrays. Each trial must have shape
            ``(n_samples, n_outputs)``.
        fs:
            Sampling rate in Hz shared by stimulus and response.
        tmin, tmax:
            Time window, in seconds, that should be extracted from the learned
            transfer function as a time-domain kernel.
        regularization:
            Regularization specification. The default behavior matches a
            standard ridge TRF fit:

            - scalar: fit directly with one ridge value
            - 1D sequence of scalars: cross-validate over those candidates

            When ``bands`` is provided, each feature group gets its own ridge
            coefficient. In that mode, a 1D scalar sequence follows the
            ``mTRF`` banded-ridge convention: the Cartesian product across
            bands is evaluated during cross-validation. You can also pass an
            explicit sequence of per-band coefficient tuples.
        bands:
            Optional contiguous feature-group sizes for banded ridge
            regularization. For example, if the predictor contains one envelope
            feature followed by a 16-band spectrogram, use ``bands=[1, 16]``.
            Leaving this as ``None`` keeps the estimator in ordinary scalar
            ridge mode.
        segment_length:
            Segment size used to estimate cross-spectra. If ``None``, each trial
            is treated as a single segment.
        segment_duration:
            Segment duration in seconds. This is a user-friendly alternative to
            ``segment_length`` for workflows that prefer time-based settings.
            Provide either ``segment_length`` or ``segment_duration``, not both.
        overlap:
            Fractional overlap between neighboring segments. Must lie in
            ``[0, 1)``.
        n_fft:
            FFT size used for spectral estimation. If omitted, a fast FFT length
            is chosen automatically from ``segment_length``.
        spectral_method:
            Spectral estimator used to compute the sufficient statistics.
            ``"standard"`` keeps the current windowed FFT behavior.
            ``"multitaper"`` averages DPSS-tapered spectra and is often more
            stable for noisy continuous data.
        time_bandwidth:
            Time-bandwidth product used when ``spectral_method="multitaper"``.
            Larger values produce broader spectral smoothing and allow more
            tapers.
        n_tapers:
            Number of DPSS tapers used for ``spectral_method="multitaper"``.
            If omitted, the default ``2 * time_bandwidth - 1`` rule is used.
        window:
            Window applied to each segment before FFT. By default no window is
            applied, which keeps the behavior closer to a standard lagged ridge
            fit. When using short overlapping segments, ``window="hann"`` is
            often a good choice. In multi-taper mode this must be ``None``
            because the DPSS tapers already define the segment weighting.
        detrend:
            Optional detrending passed to :func:`scipy.signal.detrend`.
        k:
            Number of cross-validation folds when multiple regularization values
            are supplied. ``-1`` or ``"loo"`` means leave-one-out over trials.
        average:
            How cross-validation scores should be reduced across output
            channels/features. ``True`` returns a single score per regularization
            value, ``False`` returns one score per output, and a sequence of
            indices averages only over the selected outputs.
        seed:
            Optional random seed for shuffling trial order before creating folds.
        show_progress:
            If ``True`` and cross-validation is active, print a small progress
            indicator to standard error while fold/candidate evaluations run.
        n_jobs:
            Number of worker threads used for cross-validation folds and
            bootstrap resamples. ``1`` keeps the serial behavior. ``-1`` uses
            all available CPU cores.
        trial_weights:
            Optional trial weights. Use ``"inverse_variance"`` for
            inverse-variance weighting or pass an explicit vector with one
            weight per training trial.
        bootstrap_samples:
            Number of trial-bootstrap resamples used to estimate a confidence
            interval for the fitted kernel. ``0`` disables the bootstrap.
        bootstrap_level:
            Confidence level used for the stored bootstrap interval.
        bootstrap_seed:
            Optional random seed used for bootstrap resampling.

        Returns
        -------
        None or numpy.ndarray
            ``None`` when a single regularization value is fitted directly.
            Otherwise returns cross-validation scores for each candidate
            regularization setting in the order stored by
            :attr:`regularization_candidates`.

        Notes
        -----
        The fitted model is always stored on the instance, even when
        cross-validation is used. In that case the final fit uses the selected
        regularization value and all provided trials. When multiple
        regularization values are supplied, the per-trial spectra are cached so
        the FFT work is performed only once. Direct single-lambda fits use a
        lower-memory aggregated-spectra path automatically because no trialwise
        cache is needed. Banded regularization is entirely opt-in through
        ``bands``; leaving it unset preserves the default "mTRF in Fourier
        space" workflow.
        """

        stimulus_trials, _ = _coerce_trials(stimulus, "stimulus")
        response_trials, _ = _coerce_trials(response, "response")
        _check_trial_lengths(stimulus_trials, response_trials)

        segment_length = _resolve_segment_length(
            fs=fs,
            segment_length=segment_length,
            segment_duration=segment_duration,
        )
        k = _resolve_k_folds(k)

        x_trials, y_trials = self._get_xy(stimulus_trials, response_trials)
        self._validate_dimensions(x_trials, y_trials)
        self._validate_fit_arguments(
            fs,
            tmin,
            tmax,
            segment_length,
            overlap,
            n_fft,
            spectral_method,
            time_bandwidth,
            n_tapers,
            window,
            detrend,
        )
        if int(bootstrap_samples) < 0:
            raise ValueError("bootstrap_samples must be non-negative.")
        _validate_confidence_level(bootstrap_level, name="bootstrap_level")
        n_jobs = _resolve_n_jobs(n_jobs)
        spectral_method = _validate_spectral_method(spectral_method)
        resolved_bands = _validate_bands(bands, n_inputs=x_trials[0].shape[1])
        feature_regularization_values, regularization_specs = _resolve_regularization_candidates(
            regularization,
            n_inputs=x_trials[0].shape[1],
            bands=resolved_bands,
        )
        _warn_if_cv_arguments_are_unused(
            n_candidates=len(feature_regularization_values),
            k=k,
            average=average,
            seed=seed,
            show_progress=show_progress,
        )
        raw_trial_weights = _resolve_raw_trial_weights(y_trials, trial_weights)
        self.trial_weights = _copy_value(trial_weights)
        self.bands = resolved_bands
        self.regularization_candidates = [_copy_value(spec) for spec in regularization_specs]
        self.bootstrap_interval = None
        self.bootstrap_level = None
        self.bootstrap_samples = None
        self._fit_config = {
            "fs": float(fs),
            "tmin": float(tmin),
            "tmax": float(tmax),
            "regularization": _copy_value(regularization),
            "bands": None if resolved_bands is None else tuple(resolved_bands),
            "segment_length": None if segment_length is None else int(segment_length),
            "segment_duration": None,
            "overlap": float(overlap),
            "n_fft": None if n_fft is None else int(n_fft),
            "spectral_method": spectral_method,
            "time_bandwidth": float(time_bandwidth),
            "n_tapers": None if n_tapers is None else int(n_tapers),
            "window": _copy_value(window),
            "detrend": detrend,
            "k": int(k),
            "average": _copy_value(average),
            "seed": seed,
            "show_progress": bool(show_progress),
            "n_jobs": n_jobs,
            "trial_weights": _copy_value(trial_weights),
            "bootstrap_samples": int(bootstrap_samples),
            "bootstrap_level": float(bootstrap_level),
            "bootstrap_seed": bootstrap_seed,
        }

        needs_cache = len(feature_regularization_values) > 1 or int(bootstrap_samples) > 0
        spectral_cache = None
        if needs_cache:
            spectral_cache = _build_spectral_cache(
                x_trials,
                y_trials,
                segment_length=segment_length,
                overlap=overlap,
                n_fft=n_fft,
                spectral_method=spectral_method,
                time_bandwidth=time_bandwidth,
                n_tapers=n_tapers,
                window=window,
                detrend=detrend,
            )

        if len(feature_regularization_values) == 1:
            if spectral_cache is None:
                self._fit(
                    x_trials,
                    y_trials,
                    fs=fs,
                    tmin=tmin,
                    tmax=tmax,
                    regularization=regularization_specs[0],
                    feature_regularization=feature_regularization_values[0],
                    bands=resolved_bands,
                    segment_length=segment_length,
                    overlap=overlap,
                    n_fft=n_fft,
                    spectral_method=spectral_method,
                    time_bandwidth=time_bandwidth,
                    n_tapers=n_tapers,
                    window=window,
                    detrend=detrend,
                    trial_weights=trial_weights,
                )
            else:
                self._fit_from_cache(
                    spectral_cache,
                    fs=fs,
                    tmin=tmin,
                    tmax=tmax,
                    regularization=regularization_specs[0],
                    feature_regularization=feature_regularization_values[0],
                    bands=resolved_bands,
                    raw_trial_weights=raw_trial_weights,
                )
                if bootstrap_samples > 0:
                    self.bootstrap_interval, _ = _compute_bootstrap_interval_from_cache(
                        spectral_cache,
                        fs=fs,
                        tmin=tmin,
                        tmax=tmax,
                        feature_regularization=feature_regularization_values[0],
                        raw_trial_weights=raw_trial_weights,
                        n_bootstraps=int(bootstrap_samples),
                        level=float(bootstrap_level),
                        seed=bootstrap_seed,
                        n_jobs=n_jobs,
                    )
                    self.bootstrap_level = float(bootstrap_level)
                    self.bootstrap_samples = int(bootstrap_samples)
            return None

        cv_scores = self._cross_validate(
            x_trials,
            y_trials,
            fs=fs,
            tmin=tmin,
            tmax=tmax,
            feature_regularization_values=feature_regularization_values,
            segment_length=segment_length,
            overlap=overlap,
            n_fft=n_fft,
            spectral_method=spectral_method,
            time_bandwidth=time_bandwidth,
            n_tapers=n_tapers,
            window=window,
            detrend=detrend,
            k=k,
            seed=seed,
            average=average,
            show_progress=show_progress,
            n_jobs=n_jobs,
            raw_trial_weights=raw_trial_weights,
            spectral_cache=spectral_cache,
        )

        if cv_scores.ndim == 1:
            best_index = int(np.argmax(cv_scores))
        else:
            best_index = int(np.argmax(cv_scores.mean(axis=1)))

        assert spectral_cache is not None
        self._fit_from_cache(
            spectral_cache,
            fs=fs,
            tmin=tmin,
            tmax=tmax,
            regularization=regularization_specs[best_index],
            feature_regularization=feature_regularization_values[best_index],
            bands=resolved_bands,
            raw_trial_weights=raw_trial_weights,
        )
        if bootstrap_samples > 0:
            self.bootstrap_interval, _ = _compute_bootstrap_interval_from_cache(
                spectral_cache,
                fs=fs,
                tmin=tmin,
                tmax=tmax,
                feature_regularization=feature_regularization_values[best_index],
                raw_trial_weights=raw_trial_weights,
                n_bootstraps=int(bootstrap_samples),
                level=float(bootstrap_level),
                seed=bootstrap_seed,
                n_jobs=n_jobs,
            )
            self.bootstrap_level = float(bootstrap_level)
            self.bootstrap_samples = int(bootstrap_samples)
        return cv_scores

    def train_multitaper(
        self,
        stimulus: np.ndarray | Sequence[np.ndarray],
        response: np.ndarray | Sequence[np.ndarray],
        fs: float,
        tmin: float,
        tmax: float,
        regularization: float | Sequence[float] | Sequence[Sequence[float]],
        *,
        bands: None | Sequence[int] = None,
        segment_length: int | None = None,
        segment_duration: float | None = None,
        overlap: float = 0.0,
        n_fft: int | None = None,
        time_bandwidth: float = 3.5,
        n_tapers: int | None = None,
        detrend: None | str = "constant",
        k: int | str = -1,
        average: bool | Sequence[int] = True,
        seed: int | None = None,
        show_progress: bool = False,
        n_jobs: int | None = 1,
        trial_weights: None | str | Sequence[float] = None,
        bootstrap_samples: int = 0,
        bootstrap_level: float = 0.95,
        bootstrap_seed: int | None = None,
    ) -> np.ndarray | float | None:
        """Fit the model with DPSS multi-taper spectral estimation.

        This is a convenience wrapper around :meth:`train` for users who want
        a named multi-taper estimation path without manually setting
        ``spectral_method="multitaper"`` and ``window=None`` themselves.

        Parameters
        ----------
        stimulus, response, fs, tmin, tmax, regularization:
            Identical to :meth:`train`.
        bands:
            Optional grouped-feature definition for banded ridge, exactly as in
            :meth:`train`.
        segment_length, segment_duration, overlap, n_fft:
            Segmentation and FFT settings for the cross-spectral estimates.
            These behave the same as in :meth:`train`.
        time_bandwidth:
            Time-bandwidth product of the DPSS tapers. Larger values create
            broader spectral smoothing and permit more orthogonal tapers.
        n_tapers:
            Number of DPSS tapers to use. If omitted, the conventional
            ``2 * time_bandwidth - 1`` rule is used.
        detrend:
            Optional per-segment detrending passed to the underlying spectral
            estimator.
        k, average, seed, show_progress, n_jobs:
            Cross-validation controls with the same semantics as in
            :meth:`train`.
        trial_weights:
            Optional trial weights used when aggregating training spectra.
        bootstrap_samples, bootstrap_level, bootstrap_seed:
            Optional trial-bootstrap interval settings with the same semantics
            as in :meth:`train`.

        Returns
        -------
        None or numpy.ndarray
            Same return contract as :meth:`train`: ``None`` for a direct fit or
            cross-validation scores when multiple regularization candidates are
            evaluated.

        Notes
        -----
        Multi-taper fitting is often useful when trial data are noisy or when
        short segments make ordinary single-window spectra unstable. Because
        the DPSS tapers already define the segment weighting, this wrapper
        always disables the ordinary ``window`` parameter.
        """

        return self.train(
            stimulus=stimulus,
            response=response,
            fs=fs,
            tmin=tmin,
            tmax=tmax,
            regularization=regularization,
            bands=bands,
            segment_length=segment_length,
            segment_duration=segment_duration,
            overlap=overlap,
            n_fft=n_fft,
            spectral_method="multitaper",
            time_bandwidth=time_bandwidth,
            n_tapers=n_tapers,
            window=None,
            detrend=detrend,
            k=k,
            average=average,
            seed=seed,
            show_progress=show_progress,
            n_jobs=n_jobs,
            trial_weights=trial_weights,
            bootstrap_samples=bootstrap_samples,
            bootstrap_level=bootstrap_level,
            bootstrap_seed=bootstrap_seed,
        )

    def _fit(
        self,
        x_trials: Sequence[np.ndarray],
        y_trials: Sequence[np.ndarray],
        *,
        fs: float,
        tmin: float,
        tmax: float,
        regularization: RegularizationSpec,
        feature_regularization: np.ndarray,
        bands: tuple[int, ...] | None,
        segment_length: int | None,
        overlap: float,
        n_fft: int | None,
        spectral_method: SpectralMethod,
        time_bandwidth: float,
        n_tapers: int | None,
        window: None | str | tuple[str, float] | np.ndarray,
        detrend: None | str,
        trial_weights: None | str | Sequence[float],
    ) -> None:
        raw_trial_weights = _resolve_raw_trial_weights(y_trials, trial_weights)
        spectral_cache = _build_spectral_cache(
            x_trials,
            y_trials,
            segment_length=segment_length,
            overlap=overlap,
            n_fft=n_fft,
            spectral_method=spectral_method,
            time_bandwidth=time_bandwidth,
            n_tapers=n_tapers,
            window=window,
            detrend=detrend,
            aggregate_only=True,
            raw_trial_weights=raw_trial_weights,
        )
        self._fit_from_cache(
            spectral_cache,
            fs=fs,
            tmin=tmin,
            tmax=tmax,
            regularization=regularization,
            feature_regularization=feature_regularization,
            bands=bands,
            raw_trial_weights=None,
        )
        self.window = _copy_value(window)
        self.detrend = detrend

    def _fit_from_cache(
        self,
        spectral_cache: _SpectralCache,
        *,
        fs: float,
        tmin: float,
        tmax: float,
        regularization: RegularizationSpec,
        feature_regularization: np.ndarray,
        bands: tuple[int, ...] | None,
        raw_trial_weights: np.ndarray | None,
        trial_indices: np.ndarray | None = None,
    ) -> None:
        cxx, cxy = _aggregate_cached_spectra(
            spectral_cache,
            trial_indices=trial_indices,
            raw_trial_weights=raw_trial_weights,
        )
        transfer_function = _solve_transfer_function(
            cxx,
            cxy,
            feature_regularization=feature_regularization,
        )

        self.transfer_function = transfer_function
        self.frequencies = np.fft.rfftfreq(spectral_cache.n_fft, d=1.0 / float(fs))
        self.fs = float(fs)
        self.regularization = _copy_value(regularization)
        self.bands = _copy_value(bands)
        self.feature_regularization = np.asarray(feature_regularization, dtype=float).copy()
        self.segment_length = spectral_cache.segment_length
        self.segment_duration = spectral_cache.segment_length / float(fs)
        self.n_fft = spectral_cache.n_fft
        self.overlap = spectral_cache.overlap
        self.spectral_method = spectral_cache.spectral_method
        self.time_bandwidth = spectral_cache.time_bandwidth
        self.n_tapers = spectral_cache.n_tapers
        self.window = _copy_value(spectral_cache.window)
        self.detrend = spectral_cache.detrend
        self.tmin = float(tmin)
        self.tmax = float(tmax)
        self.weights, self.times = _extract_impulse_response(
            self.transfer_function,
            fs=float(fs),
            n_fft=spectral_cache.n_fft,
            tmin=float(tmin),
            tmax=float(tmax),
        )

    def to_impulse_response(
        self,
        tmin: float | None = None,
        tmax: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract a time-domain kernel from the fitted transfer function.

        Parameters
        ----------
        tmin, tmax:
            Optional lag window in seconds. If omitted, the window used during
            :meth:`train` is reused. These values only control which portion of
            the already fitted transfer function is transformed back to the lag
            domain; they do not trigger a refit.

        Returns
        -------
        weights, times:
            ``weights`` has shape ``(n_inputs, n_lags, n_outputs)`` and ``times``
            contains the corresponding lag values in seconds. The time axis is
            sampled at the model's stored sampling rate.

        Notes
        -----
        This method is useful when you want to inspect a different lag window
        without refitting the spectral model. It is the same operation used
        internally to populate :attr:`weights` and :attr:`times` after
        training, and it powers downstream helpers such as :meth:`plot`,
        :meth:`predict`, and :meth:`bootstrap_interval_at`.
        """

        if self.transfer_function is None or self.fs is None or self.n_fft is None:
            raise ValueError("Model must be trained before extracting an impulse response.")

        tmin = self.tmin if tmin is None else tmin
        tmax = self.tmax if tmax is None else tmax
        if tmin is None or tmax is None:
            raise ValueError("tmin and tmax must be defined.")

        return _extract_impulse_response(
            self.transfer_function,
            fs=float(self.fs),
            n_fft=int(self.n_fft),
            tmin=float(tmin),
            tmax=float(tmax),
        )

    def plot(
        self,
        *,
        input_index: int = 0,
        output_index: int = 0,
        tmin: float | None = None,
        tmax: float | None = None,
        ax=None,
        time_unit: str = "ms",
        color: str | None = None,
        linewidth: float = 2.0,
        show_bootstrap_interval: bool = False,
        interval_color: str | None = None,
        interval_alpha: float = 0.2,
        title: str | None = None,
        label: str | None = None,
    ):
        """Plot one fitted time-domain kernel.

        Parameters
        ----------
        input_index, output_index:
            Select which input/output pair should be plotted. ``input_index``
            refers to the predictor feature dimension and ``output_index`` to
            the predicted target channel.
        tmin, tmax:
            Optional lag window to visualize. If omitted, the fitted window is
            used. This is applied by re-extracting the impulse response from
            the stored transfer function.
        ax:
            Existing matplotlib axes. When omitted, a new figure and axes are
            created.
        time_unit:
            Either ``"ms"`` or ``"s"`` for the x-axis labels and plotted lag
            values.
        color, linewidth, title, label:
            Standard matplotlib styling arguments for the kernel line.
        show_bootstrap_interval:
            If ``True``, shade the stored bootstrap confidence interval behind
            the kernel. This requires that the model already contains a stored
            interval from training or :meth:`bootstrap_confidence_interval`.
        interval_color, interval_alpha:
            Styling for the bootstrap interval shading.

        Returns
        -------
        fig, ax:
            The matplotlib figure and axes containing the plot.

        Notes
        -----
        This helper is a thin wrapper around :mod:`matplotlib`. It imports the
        plotting backend lazily so that the core toolbox stays usable without
        installing plotting dependencies.
        """

        if self.weights is None or self.times is None:
            raise ValueError("Model must be trained before plotting.")

        from .plotting import plot_kernel

        weights, times = self.to_impulse_response(tmin=tmin, tmax=tmax)
        bootstrap_interval = None
        if show_bootstrap_interval:
            bootstrap_interval, _ = self.bootstrap_interval_at(tmin=tmin, tmax=tmax)
        return plot_kernel(
            weights=weights,
            times=times,
            credible_interval=bootstrap_interval,
            input_index=input_index,
            output_index=output_index,
            ax=ax,
            time_unit=time_unit,
            color=color,
            interval_color=interval_color,
            linewidth=linewidth,
            interval_alpha=interval_alpha,
            title=title,
            label=label,
        )

    def plot_grid(
        self,
        *,
        tmin: float | None = None,
        tmax: float | None = None,
        ax=None,
        time_unit: str = "ms",
        color: str | None = None,
        linewidth: float = 1.8,
        show_bootstrap_interval: bool = False,
        interval_color: str | None = None,
        interval_alpha: float = 0.2,
        input_labels: Sequence[str] | None = None,
        output_labels: Sequence[str] | None = None,
        title: str | None = None,
        sharey: bool = False,
    ):
        """Plot every input/output kernel in a grid.

        Parameters
        ----------
        tmin, tmax:
            Optional lag window to visualize. If omitted, the fitted lag window
            is used for every panel.
        ax:
            Optional array of pre-created matplotlib axes with shape
            ``(n_inputs, n_outputs)``. When omitted, a matching grid is created
            automatically.
        time_unit:
            Either ``"ms"`` or ``"s"`` for the shared lag axis.
        color, linewidth:
            Styling applied to every kernel trace in the grid.
        show_bootstrap_interval:
            If ``True``, overlay the stored bootstrap interval in every panel.
        interval_color, interval_alpha:
            Styling for the interval shading.
        input_labels, output_labels:
            Optional human-readable labels used to title the individual panels.
            When omitted, generic ``Input N`` / ``Output N`` labels are used.
        title:
            Optional figure-level title.
        sharey:
            If ``True``, all axes share the same y limits. This is useful for
            visual comparison across channels but can hide small kernels when
            one panel contains much larger values than the others.

        Returns
        -------
        fig, axes:
            The matplotlib figure and the full axes grid.

        Notes
        -----
        This is convenient for multifeature or multichannel models where
        calling :meth:`plot` repeatedly would be cumbersome.
        """

        if self.weights is None or self.times is None:
            raise ValueError("Model must be trained before plotting.")

        from .plotting import plot_kernel_grid

        weights, times = self.to_impulse_response(tmin=tmin, tmax=tmax)
        bootstrap_interval = None
        if show_bootstrap_interval:
            bootstrap_interval, _ = self.bootstrap_interval_at(tmin=tmin, tmax=tmax)
        return plot_kernel_grid(
            weights=weights,
            times=times,
            credible_interval=bootstrap_interval,
            ax=ax,
            time_unit=time_unit,
            color=color,
            interval_color=interval_color,
            linewidth=linewidth,
            interval_alpha=interval_alpha,
            title=title,
            input_labels=input_labels,
            output_labels=output_labels,
            sharey=sharey,
        )

    def frequency_resolved_weights(
        self,
        *,
        n_bands: int = 24,
        fmin: float | None = None,
        fmax: float | None = None,
        tmin: float | None = None,
        tmax: float | None = None,
        scale: str = "linear",
        bandwidth: float | None = None,
        value_mode: str = "real",
    ) -> FrequencyResolvedWeights:
        """Return a spectrotemporal decomposition of the fitted kernel.

        The learned transfer function is partitioned into smooth frequency
        bands, and each band is transformed back into the lag domain. This
        yields a frequency-by-lag representation that can be plotted like a
        spectrogram. In the default ``value_mode="real"`` setting, summing the
        returned weights across the band axis reconstructs the ordinary
        time-domain kernel, provided the full fitted frequency range is used.

        Parameters
        ----------
        n_bands:
            Number of analysis bands used for the decomposition.
        fmin, fmax:
            Frequency range in Hz to analyze. The default covers the full
            fitted range from DC to Nyquist.
        tmin, tmax:
            Optional lag window to extract. If omitted, the fitted lag window is
            reused.
        scale:
            Spacing of the band centers. Use ``"linear"`` for evenly spaced
            bands or ``"log"`` for logarithmic spacing.
        bandwidth:
            Gaussian band width in Hz. When omitted, it is inferred from the
            spacing between neighboring band centers.
        value_mode:
            ``"real"`` returns the signed band-limited kernels,
            ``"magnitude"`` returns their absolute value, and ``"power"``
            returns squared magnitude.

        Returns
        -------
        FrequencyResolvedWeights
            Container holding the filter bank, lag axis, and resolved kernel
            tensor with shape ``(n_inputs, n_bands, n_lags, n_outputs)``.
        """

        if self.transfer_function is None or self.frequencies is None:
            raise ValueError("Model must be trained before resolving weights by frequency.")
        if self.n_fft is None or self.fs is None:
            raise ValueError("Stored FFT settings are unavailable on this model.")

        tmin = self.tmin if tmin is None else float(tmin)
        tmax = self.tmax if tmax is None else float(tmax)
        if tmin is None or tmax is None:
            raise ValueError("tmin and tmax must be defined.")

        band_centers, filters, resolved_scale, resolved_bandwidth = _build_frequency_filterbank(
            self.frequencies,
            n_bands=n_bands,
            fmin=fmin,
            fmax=fmax,
            scale=scale,
            bandwidth=bandwidth,
        )
        resolved_value_mode = _resolve_frequency_weight_value_mode(value_mode)

        band_transfer = (
            self.transfer_function[:, :, :, np.newaxis]
            * filters[:, np.newaxis, np.newaxis, :]
        )
        full_kernel = np.fft.irfft(
            band_transfer,
            n=int(self.n_fft),
            axis=0,
        ).real

        lag_start = int(round(float(tmin) * float(self.fs)))
        lag_stop = int(round(float(tmax) * float(self.fs)))
        if lag_stop <= lag_start:
            raise ValueError("tmax must be greater than tmin.")
        lag_indices = np.arange(lag_start, lag_stop, dtype=int)
        kernel = full_kernel[np.mod(lag_indices, int(self.n_fft)), :, :, :]
        kernel = np.transpose(kernel, (1, 3, 0, 2))

        if resolved_value_mode == "magnitude":
            kernel = np.abs(kernel)
        elif resolved_value_mode == "power":
            kernel = kernel**2

        return FrequencyResolvedWeights(
            frequencies=self.frequencies.copy(),
            band_centers=band_centers,
            filters=filters,
            times=(lag_indices / float(self.fs)),
            weights=kernel,
            scale=resolved_scale,
            value_mode=resolved_value_mode,
            bandwidth=resolved_bandwidth,
        )

    def time_frequency_power(
        self,
        *,
        n_bands: int = 24,
        fmin: float | None = None,
        fmax: float | None = None,
        tmin: float | None = None,
        tmax: float | None = None,
        scale: str = "linear",
        bandwidth: float | None = None,
        method: str = "hilbert",
    ) -> TimeFrequencyPower:
        """Estimate spectrogram-like power from the fitted kernel.

        This method starts from the signed band-limited kernels returned by
        :meth:`frequency_resolved_weights` and converts each frequency band into
        a smoother power representation. With the default ``method="hilbert"``,
        power is the squared magnitude of the analytic signal of each
        band-limited kernel. The result is closer to what users expect from a
        spectrogram than simply squaring the oscillatory kernel itself.

        Parameters
        ----------
        n_bands:
            Number of analysis bands used to partition the fitted transfer
            function.
        fmin, fmax:
            Frequency range in Hz to analyze. By default the full fitted range
            is used.
        tmin, tmax:
            Optional lag window in seconds to extract from the reconstructed
            band-limited kernels.
        scale:
            Placement of band centers. ``"linear"`` uses evenly spaced bands
            and ``"log"`` uses logarithmic spacing.
        bandwidth:
            Width of the Gaussian analysis filters in Hz. If omitted, it is
            inferred from neighboring band centers.
        method:
            Power estimation method. Currently only ``"hilbert"`` is
            implemented, which converts each band-limited kernel to an analytic
            signal and then squares its magnitude.

        Returns
        -------
        TimeFrequencyPower
            Container holding band centers, lag axis, and the power tensor with
            shape ``(n_inputs, n_bands, n_lags, n_outputs)``.

        Notes
        -----
        This representation is best interpreted as a descriptive view of the
        fitted kernel, not as a spectrogram of the original stimulus or
        response recordings.
        """

        resolved_method = str(method).strip().lower()
        if resolved_method != "hilbert":
            raise ValueError("method must currently be 'hilbert'.")

        resolved = self.frequency_resolved_weights(
            n_bands=n_bands,
            fmin=fmin,
            fmax=fmax,
            tmin=tmin,
            tmax=tmax,
            scale=scale,
            bandwidth=bandwidth,
            value_mode="real",
        )
        analytic = hilbert(resolved.weights, axis=2)
        power = np.abs(analytic) ** 2

        return TimeFrequencyPower(
            frequencies=resolved.frequencies.copy(),
            band_centers=resolved.band_centers.copy(),
            filters=resolved.filters.copy(),
            times=resolved.times.copy(),
            power=power,
            scale=resolved.scale,
            method=resolved_method,
            bandwidth=resolved.bandwidth,
        )

    def plot_frequency_resolved_weights(
        self,
        *,
        resolved: FrequencyResolvedWeights | None = None,
        input_index: int = 0,
        output_index: int = 0,
        n_bands: int = 24,
        fmin: float | None = None,
        fmax: float | None = None,
        tmin: float | None = None,
        tmax: float | None = None,
        scale: str = "linear",
        bandwidth: float | None = None,
        value_mode: str = "real",
        ax=None,
        time_unit: str = "ms",
        cmap: str | None = None,
        colorbar: bool = True,
        title: str | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
        frequency_axis_scale: str | None = None,
    ):
        """Plot one frequency-resolved kernel map as a heatmap.

        Parameters
        ----------
        resolved:
            Precomputed output of :meth:`frequency_resolved_weights`. Supplying
            this is useful when you want to plot several channels from the same
            decomposition without recomputing it. If omitted, the decomposition
            is computed on demand from the remaining keyword arguments.
        input_index, output_index:
            Select which input/output pair to display.
        n_bands, fmin, fmax, tmin, tmax, scale, bandwidth, value_mode:
            Parameters used when ``resolved`` is not supplied. They map exactly
            to :meth:`frequency_resolved_weights`.
        ax:
            Existing matplotlib axes. When omitted, a new figure is created.
        time_unit:
            Either ``"ms"`` or ``"s"`` for the lag axis.
        cmap:
            Optional matplotlib colormap. When omitted, a diverging map is used
            for signed weights and a sequential map is used for non-negative
            views.
        colorbar:
            If ``True``, add a colorbar describing the displayed values.
        title:
            Optional plot title.
        vmin, vmax:
            Optional explicit color limits.
        frequency_axis_scale:
            Optional display scaling for the frequency axis. If omitted, the
            scale stored in ``resolved`` is used.

        Returns
        -------
        fig, ax:
            The matplotlib figure and axes containing the heatmap.
        """

        from .plotting import plot_frequency_resolved_weights

        if resolved is None:
            resolved = self.frequency_resolved_weights(
                n_bands=n_bands,
                fmin=fmin,
                fmax=fmax,
                tmin=tmin,
                tmax=tmax,
                scale=scale,
                bandwidth=bandwidth,
                value_mode=value_mode,
            )

        return plot_frequency_resolved_weights(
            weights=resolved.weights,
            times=resolved.times,
            band_centers=resolved.band_centers,
            input_index=input_index,
            output_index=output_index,
            value_mode=resolved.value_mode,
            bandwidth=resolved.bandwidth,
            ax=ax,
            time_unit=time_unit,
            cmap=cmap,
            colorbar=colorbar,
            title=title,
            vmin=vmin,
            vmax=vmax,
            frequency_axis_scale=resolved.scale if frequency_axis_scale is None else frequency_axis_scale,
        )

    def plot_time_frequency_power(
        self,
        *,
        power: TimeFrequencyPower | None = None,
        input_index: int = 0,
        output_index: int = 0,
        n_bands: int = 24,
        fmin: float | None = None,
        fmax: float | None = None,
        tmin: float | None = None,
        tmax: float | None = None,
        scale: str = "linear",
        bandwidth: float | None = None,
        method: str = "hilbert",
        ax=None,
        time_unit: str = "ms",
        cmap: str | None = None,
        colorbar: bool = True,
        title: str | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
        frequency_axis_scale: str | None = None,
    ):
        """Plot a spectrogram-like time-frequency power map.

        Parameters
        ----------
        power:
            Precomputed output of :meth:`time_frequency_power`. Supplying this
            avoids recomputing the decomposition when plotting multiple panels.
            If omitted, the power map is computed on demand from the remaining
            keyword arguments.
        input_index, output_index:
            Select which input/output pair to display.
        n_bands, fmin, fmax, tmin, tmax, scale, bandwidth, method:
            Parameters used when ``power`` is not supplied. They map exactly to
            :meth:`time_frequency_power`.
        ax:
            Existing matplotlib axes. When omitted, a new figure is created.
        time_unit:
            Either ``"ms"`` or ``"s"`` for the lag axis.
        cmap:
            Optional matplotlib colormap for the heatmap.
        colorbar:
            If ``True``, add a colorbar labeled as power.
        title:
            Optional plot title.
        vmin, vmax:
            Optional explicit color limits.
        frequency_axis_scale:
            Optional display scaling for the frequency axis. If omitted, the
            scale stored in ``power`` is used.

        Returns
        -------
        fig, ax:
            The matplotlib figure and axes containing the heatmap.
        """

        from .plotting import plot_frequency_resolved_weights

        if power is None:
            power = self.time_frequency_power(
                n_bands=n_bands,
                fmin=fmin,
                fmax=fmax,
                tmin=tmin,
                tmax=tmax,
                scale=scale,
                bandwidth=bandwidth,
                method=method,
            )

        return plot_frequency_resolved_weights(
            weights=power.power,
            times=power.times,
            band_centers=power.band_centers,
            input_index=input_index,
            output_index=output_index,
            value_mode="power",
            bandwidth=power.bandwidth,
            ax=ax,
            time_unit=time_unit,
            cmap=cmap,
            colorbar=colorbar,
            colorbar_label="Power",
            title=title,
            vmin=vmin,
            vmax=vmax,
            frequency_axis_scale=power.scale if frequency_axis_scale is None else frequency_axis_scale,
        )

    def transfer_function_at(
        self,
        *,
        input_index: int = 0,
        output_index: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return one complex-valued transfer function slice.

        Parameters
        ----------
        input_index, output_index:
            Select the predictor-target pair to inspect.

        Returns
        -------
        frequencies, transfer_function:
            Frequency vector in Hz and the matching complex transfer-function
            values for the selected input/output pair.

        Notes
        -----
        The returned complex values encode both gain and phase. If you want
        ready-to-plot derived quantities such as magnitude, phase, or group
        delay, use :meth:`transfer_function_components_at` instead.
        """

        if self.transfer_function is None or self.frequencies is None:
            raise ValueError("Model must be trained before extracting a transfer function.")
        if not 0 <= int(input_index) < self.transfer_function.shape[1]:
            raise IndexError(f"input_index out of bounds: {input_index}")
        if not 0 <= int(output_index) < self.transfer_function.shape[2]:
            raise IndexError(f"output_index out of bounds: {output_index}")
        return (
            self.frequencies.copy(),
            self.transfer_function[:, int(input_index), int(output_index)].copy(),
        )

    def transfer_function_components_at(
        self,
        *,
        input_index: int = 0,
        output_index: int = 0,
        phase_unit: str = "rad",
    ) -> TransferFunctionComponents:
        """Return magnitude, phase, and group delay for one transfer function.

        Parameters
        ----------
        input_index, output_index:
            Select the predictor-target pair to inspect.
        phase_unit:
            Unit used for the returned phase values. Must be ``"rad"`` or
            ``"deg"``.

        Returns
        -------
        TransferFunctionComponents
            Container with the raw complex transfer function plus magnitude,
            unwrapped phase, and group delay for the selected pair.
        """

        frequencies, transfer = self.transfer_function_at(
            input_index=input_index,
            output_index=output_index,
        )
        phase, resolved_phase_unit = _phase_values(
            transfer,
            phase_unit=phase_unit,
        )
        return TransferFunctionComponents(
            frequencies=frequencies,
            transfer_function=transfer,
            magnitude=np.abs(transfer),
            phase=phase,
            phase_unit=resolved_phase_unit,
            group_delay=_group_delay_values(frequencies, transfer),
        )

    def plot_transfer_function(
        self,
        *,
        input_index: int = 0,
        output_index: int = 0,
        kind: str = "both",
        ax=None,
        color: str | None = None,
        phase_color: str | None = None,
        group_delay_color: str | None = None,
        linewidth: float = 2.0,
        phase_unit: str = "rad",
        group_delay_unit: str = "ms",
        title: str | None = None,
    ):
        """Plot magnitude, phase, and/or group delay of one transfer function.

        Parameters
        ----------
        input_index, output_index:
            Select the predictor-target pair to display.
        kind:
            Which quantity to plot. Use ``"magnitude"``, ``"phase"``,
            ``"group_delay"``, ``"both"`` (magnitude plus phase), or ``"all"``
            (magnitude, phase, and group delay).
        ax:
            Existing matplotlib axes. For ``kind="both"`` provide two stacked
            axes, and for ``kind="all"`` provide three. When omitted, a new
            figure is created.
        color, phase_color, group_delay_color:
            Optional line colors for the displayed curves.
        linewidth:
            Line width passed to matplotlib.
        phase_unit:
            Unit for plotted phase values, either ``"rad"`` or ``"deg"``.
        group_delay_unit:
            Unit for plotted group delay values, either ``"s"`` or ``"ms"``.
        title:
            Optional plot title.

        Returns
        -------
        fig, ax:
            The matplotlib figure and axes containing the requested view.
        """

        from .plotting import plot_transfer_function

        components = self.transfer_function_components_at(
            input_index=input_index,
            output_index=output_index,
            phase_unit=phase_unit,
        )
        return plot_transfer_function(
            frequencies=components.frequencies,
            transfer_function=components.transfer_function,
            kind=kind,
            ax=ax,
            color=color,
            phase_color=phase_color,
            group_delay_color=group_delay_color,
            linewidth=linewidth,
            phase_unit=components.phase_unit,
            group_delay_unit=group_delay_unit,
            title=title,
        )

    def cross_spectral_diagnostics(
        self,
        *,
        stimulus: np.ndarray | Sequence[np.ndarray] | None = None,
        response: np.ndarray | Sequence[np.ndarray] | None = None,
        tmin: float | None = None,
        tmax: float | None = None,
        trial_weights: None | str | Sequence[float] | object = _USE_STORED_TRIAL_WEIGHTS,
    ) -> TRFDiagnostics:
        """Compute observed-vs-predicted cross-spectral diagnostics.

        This method reuses the fitted kernel to generate predictions for the
        provided data, then compares predicted and observed targets in the
        frequency domain. The returned diagnostics include:

        - the learned complex transfer function
        - predicted and observed output spectra
        - matched predicted-vs-observed cross-spectra
        - magnitude-squared coherence between prediction and target

        Parameters
        ----------
        stimulus, response:
            Data to evaluate. The method follows the same directional
            conventions as :meth:`predict`: forward models require
            ``stimulus`` and compare predictions against ``response``, while
            backward models require ``response`` and compare predictions
            against ``stimulus``.
        tmin, tmax:
            Optional lag window used when generating predictions for the
            diagnostics. If omitted, the fitted lag window is reused.
        trial_weights:
            Trial weights used when aggregating the diagnostic spectra. The
            default sentinel value means "reuse the weighting strategy stored on
            the model". You can also pass ``None``, ``"inverse_variance"``, or
            an explicit vector of weights.

        Returns
        -------
        TRFDiagnostics
            Container holding transfer-function slices, spectra, cross-spectra,
            and coherence.

        Notes
        -----
        These diagnostics answer a different question than the lag-domain
        kernel plots: they show whether the model captures the spectral content
        of the observed targets, output by output, under the same spectral
        estimation settings used during fitting.
        """

        if self.transfer_function is None or self.frequencies is None:
            raise ValueError("Model must be trained before computing diagnostics.")
        if self.segment_length is None or self.overlap is None or self.n_fft is None:
            raise ValueError("Stored spectral settings are unavailable on this model.")

        predictor_input = stimulus if self.direction == 1 else response
        target_input = response if self.direction == 1 else stimulus
        predictor_name = "stimulus" if self.direction == 1 else "response"
        target_name = "response" if self.direction == 1 else "stimulus"
        if predictor_input is None or target_input is None:
            raise ValueError(
                f"{predictor_name} and {target_name} are both required for diagnostics."
            )

        predictor_trials, _ = _coerce_trials(predictor_input, predictor_name)
        target_trials, _ = _coerce_trials(target_input, target_name)
        _check_trial_lengths(predictor_trials, target_trials)

        predictions = self.predict(
            stimulus=stimulus,
            response=response,
            tmin=tmin,
            tmax=tmax,
        )
        if isinstance(predictions, tuple):
            prediction_trials_raw = predictions[0]
        else:
            prediction_trials_raw = predictions
        prediction_trials, _ = _coerce_trials(
            prediction_trials_raw,
            "prediction",
        )

        resolved_trial_weights = (
            self.trial_weights if trial_weights is _USE_STORED_TRIAL_WEIGHTS else trial_weights
        )
        raw_trial_weights = _resolve_raw_trial_weights(target_trials, resolved_trial_weights)
        prediction_cache = _build_spectral_cache(
            prediction_trials,
            target_trials,
            segment_length=self.segment_length,
            overlap=float(self.overlap),
            n_fft=self.n_fft,
            spectral_method=self.spectral_method,
            time_bandwidth=float(self.time_bandwidth or 3.5),
            n_tapers=self.n_tapers,
            window=self.window,
            detrend=self.detrend,
        )
        observed_cache = _build_spectral_cache(
            target_trials,
            target_trials,
            segment_length=self.segment_length,
            overlap=float(self.overlap),
            n_fft=self.n_fft,
            spectral_method=self.spectral_method,
            time_bandwidth=float(self.time_bandwidth or 3.5),
            n_tapers=self.n_tapers,
            window=self.window,
            detrend=self.detrend,
        )
        predicted_auto, predicted_vs_observed = _aggregate_cached_spectra(
            prediction_cache,
            raw_trial_weights=raw_trial_weights,
        )
        observed_auto, _ = _aggregate_cached_spectra(
            observed_cache,
            raw_trial_weights=raw_trial_weights,
        )

        predicted_spectrum = np.real(np.diagonal(predicted_auto, axis1=1, axis2=2))
        observed_spectrum = np.real(np.diagonal(observed_auto, axis1=1, axis2=2))
        cross_spectrum = np.diagonal(predicted_vs_observed, axis1=1, axis2=2)
        denominator = np.clip(
            predicted_spectrum * observed_spectrum,
            np.finfo(float).eps,
            None,
        )
        coherence = np.clip((np.abs(cross_spectrum) ** 2) / denominator, 0.0, 1.0)

        return TRFDiagnostics(
            frequencies=self.frequencies.copy(),
            transfer_function=self.transfer_function.copy(),
            predicted_spectrum=predicted_spectrum,
            observed_spectrum=observed_spectrum,
            cross_spectrum=cross_spectrum,
            coherence=coherence,
        )

    def diagnostics(
        self,
        *,
        stimulus: np.ndarray | Sequence[np.ndarray] | None = None,
        response: np.ndarray | Sequence[np.ndarray] | None = None,
        tmin: float | None = None,
        tmax: float | None = None,
        trial_weights: None | str | Sequence[float] | object = _USE_STORED_TRIAL_WEIGHTS,
    ) -> TRFDiagnostics:
        """Compatibility alias for :meth:`cross_spectral_diagnostics`."""

        return self.cross_spectral_diagnostics(
            stimulus=stimulus,
            response=response,
            tmin=tmin,
            tmax=tmax,
            trial_weights=trial_weights,
        )

    def plot_coherence(
        self,
        *,
        stimulus: np.ndarray | Sequence[np.ndarray] | None = None,
        response: np.ndarray | Sequence[np.ndarray] | None = None,
        diagnostics: TRFDiagnostics | None = None,
        output_index: int = 0,
        ax=None,
        color: str | None = None,
        linewidth: float = 2.0,
        title: str | None = None,
    ):
        """Plot magnitude-squared coherence between predictions and targets.

        Parameters
        ----------
        stimulus, response:
            Evaluation data used when ``diagnostics`` is not supplied.
        diagnostics:
            Precomputed diagnostics from :meth:`cross_spectral_diagnostics`.
            Passing this is useful when plotting several outputs from the same
            evaluation result.
        output_index:
            Target output channel to display.
        ax:
            Existing matplotlib axes. When omitted, a new figure is created.
        color, linewidth, title:
            Standard matplotlib styling options for the coherence curve.

        Returns
        -------
        fig, ax:
            The matplotlib figure and axes containing the coherence plot.
        """

        from .plotting import plot_coherence

        if diagnostics is None:
            diagnostics = self.cross_spectral_diagnostics(
                stimulus=stimulus,
                response=response,
            )
        return plot_coherence(
            frequencies=diagnostics.frequencies,
            coherence=diagnostics.coherence,
            output_index=output_index,
            ax=ax,
            color=color,
            linewidth=linewidth,
            title=title,
        )

    def plot_cross_spectrum(
        self,
        *,
        stimulus: np.ndarray | Sequence[np.ndarray] | None = None,
        response: np.ndarray | Sequence[np.ndarray] | None = None,
        diagnostics: TRFDiagnostics | None = None,
        output_index: int = 0,
        kind: str = "both",
        ax=None,
        color: str | None = None,
        phase_color: str | None = None,
        linewidth: float = 2.0,
        phase_unit: str = "rad",
        title: str | None = None,
    ):
        """Plot the predicted-vs-observed cross spectrum for one output.

        Parameters
        ----------
        stimulus, response:
            Evaluation data used when ``diagnostics`` is not supplied.
        diagnostics:
            Precomputed diagnostics from :meth:`cross_spectral_diagnostics`.
        output_index:
            Target output channel to display.
        kind:
            Which quantity to plot. Use ``"magnitude"``, ``"phase"``, or
            ``"both"``.
        ax:
            Existing matplotlib axes. For ``kind="both"`` provide two stacked
            axes. When omitted, a new figure is created.
        color, phase_color, linewidth, phase_unit, title:
            Standard plotting options passed through to the underlying
            matplotlib helper.

        Returns
        -------
        fig, ax:
            The matplotlib figure and axes containing the requested view.
        """

        from .plotting import plot_cross_spectrum

        if diagnostics is None:
            diagnostics = self.cross_spectral_diagnostics(
                stimulus=stimulus,
                response=response,
            )
        return plot_cross_spectrum(
            frequencies=diagnostics.frequencies,
            cross_spectrum=diagnostics.cross_spectrum,
            output_index=output_index,
            kind=kind,
            ax=ax,
            color=color,
            phase_color=phase_color,
            linewidth=linewidth,
            phase_unit=phase_unit,
            title=title,
        )

    def bootstrap_interval_at(
        self,
        *,
        tmin: float | None = None,
        tmax: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the stored bootstrap interval over the requested lag window.

        Parameters
        ----------
        tmin, tmax:
            Optional lag window in seconds to extract from the stored interval.
            If omitted, the full stored interval is returned.

        Returns
        -------
        interval, times:
            ``interval`` has shape ``(2, n_inputs, n_lags, n_outputs)`` where
            the first axis contains lower and upper bounds. ``times`` contains
            the matching lag values in seconds.
        """

        if self.bootstrap_interval is None or self.times is None:
            raise ValueError("No bootstrap interval is stored on this model.")
        return _slice_interval(
            self.bootstrap_interval,
            self.times,
            tmin=tmin,
            tmax=tmax,
        )

    def bootstrap_confidence_interval(
        self,
        stimulus: np.ndarray | Sequence[np.ndarray],
        response: np.ndarray | Sequence[np.ndarray],
        *,
        n_bootstraps: int = 200,
        level: float = 0.95,
        seed: int | None = None,
        n_jobs: int | None = 1,
        trial_weights: None | str | Sequence[float] | object = _USE_STORED_TRIAL_WEIGHTS,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Estimate and store a trial-bootstrap confidence interval.

        The estimator must already be fitted. By default the method reuses the
        same fit settings and the same trial-weighting strategy as the model.
        Bootstrap resampling is performed over trials, so at least two trials
        are required. ``n_jobs`` controls optional parallel execution across
        bootstrap resamples.

        Parameters
        ----------
        stimulus, response:
            Trial data used for the bootstrap resampling. They follow the same
            single-trial vs multi-trial conventions as :meth:`train`.
        n_bootstraps:
            Number of bootstrap resamples. Larger values yield a more stable
            interval estimate but increase runtime.
        level:
            Confidence level between 0 and 1. For example, ``0.95`` stores a
            95% interval.
        seed:
            Optional random seed for reproducible bootstrap resampling.
        n_jobs:
            Number of worker threads used to evaluate bootstrap resamples.
            ``1`` runs serially and ``-1`` uses all available cores.
        trial_weights:
            Trial-weighting strategy used while aggregating spectra inside each
            bootstrap replicate. The default reuses the model's stored
            weighting behavior.

        Returns
        -------
        interval, times:
            Stored confidence interval with shape
            ``(2, n_inputs, n_lags, n_outputs)`` and the corresponding lag axis
            in seconds.
        """

        if self.regularization is None or self.fs is None or self.tmin is None or self.tmax is None:
            raise ValueError("Model must be trained before bootstrapping.")
        if self.feature_regularization is None:
            raise ValueError("Stored feature regularization is unavailable on this model.")

        stimulus_trials, _ = _coerce_trials(stimulus, "stimulus")
        response_trials, _ = _coerce_trials(response, "response")
        _check_trial_lengths(stimulus_trials, response_trials)
        x_trials, y_trials = self._get_xy(stimulus_trials, response_trials)
        self._validate_dimensions(x_trials, y_trials)

        resolved_trial_weights = self.trial_weights if trial_weights is _USE_STORED_TRIAL_WEIGHTS else trial_weights
        spectral_cache = _build_spectral_cache(
            x_trials,
            y_trials,
            segment_length=self.segment_length,
            overlap=float(self.overlap),
            n_fft=self.n_fft,
            spectral_method=self.spectral_method,
            time_bandwidth=float(self.time_bandwidth or 3.5),
            n_tapers=self.n_tapers,
            window=self.window,
            detrend=self.detrend,
        )
        interval, times = _compute_bootstrap_interval_from_cache(
            spectral_cache,
            fs=float(self.fs),
            tmin=float(self.tmin),
            tmax=float(self.tmax),
            feature_regularization=self.feature_regularization,
            raw_trial_weights=_resolve_raw_trial_weights(y_trials, resolved_trial_weights),
            n_bootstraps=int(n_bootstraps),
            level=float(level),
            seed=seed,
            n_jobs=n_jobs,
        )
        self.bootstrap_interval = interval
        self.bootstrap_level = float(level)
        self.bootstrap_samples = int(n_bootstraps)
        return interval, times

    def permutation_test(
        self,
        stimulus: np.ndarray | Sequence[np.ndarray] | None = None,
        response: np.ndarray | Sequence[np.ndarray] | None = None,
        *,
        n_permutations: int = 1000,
        average: bool | Sequence[int] = True,
        tmin: float | None = None,
        tmax: float | None = None,
        surrogate: str = "circular_shift",
        min_shift: float | None = None,
        tail: str = "greater",
        seed: int | None = None,
        n_jobs: int | None = 1,
    ) -> PermutationTestResult:
        """Estimate score significance against a surrogate null distribution.

        This method evaluates a fitted model on aligned data, then compares the
        observed prediction score against surrogate scores obtained from the
        same predictions and a permuted target side. It answers the question
        "is this held-out score larger than would be expected under a null
        alignment?" rather than the stronger and slower "would the entire
        training pipeline beat chance if retrained on permuted data?".

        Parameters
        ----------
        stimulus, response:
            Evaluation data using the same directional conventions as
            :meth:`score`. Forward models require both ``stimulus`` and
            ``response``. Backward models require both ``response`` and
            ``stimulus``.
        n_permutations:
            Number of surrogate scores used to form the null distribution.
        average:
            Score reduction applied to the observed and surrogate scores. This
            follows the same rules as :meth:`score`.
        tmin, tmax:
            Optional lag window used during prediction before scoring.
        surrogate:
            Strategy used to break the alignment between predictions and
            observed targets. ``"circular_shift"`` rolls each evaluation trial
            by a random non-zero offset and works for single-trial or
            multi-trial data. ``"trial_shuffle"`` permutes whole target trials
            and therefore requires at least two equal-length evaluation trials.
        min_shift:
            Minimum circular shift, in seconds, used when
            ``surrogate="circular_shift"``. ``None`` allows any non-zero shift.
        tail:
            Tail convention for the p-value calculation: ``"greater"``,
            ``"less"``, or ``"two-sided"``.
        seed:
            Optional random seed for reproducible surrogate generation.
        n_jobs:
            Number of worker threads used to score the surrogate targets.
            ``1`` runs serially and ``-1`` uses all available cores.

        Returns
        -------
        PermutationTestResult
            Container with the observed score, surrogate null scores, p-value,
            and z-score.

        Notes
        -----
        All built-in metrics in ``ffTRF`` use the "larger is better" convention,
        so ``tail="greater"`` is the natural default. The returned null scores
        follow the same scalar-vs-array contract as :meth:`score`: aggregated
        scores are stored as a 1D array of length ``n_permutations``, while
        ``average=False`` keeps one surrogate score per output.
        """

        if self.weights is None or self.fs is None:
            raise ValueError("Model must be trained before permutation testing.")

        predictor_input = stimulus if self.direction == 1 else response
        target_input = response if self.direction == 1 else stimulus
        predictor_name = "stimulus" if self.direction == 1 else "response"
        target_name = "response" if self.direction == 1 else "stimulus"
        if predictor_input is None or target_input is None:
            raise ValueError(
                f"{predictor_name} and {target_name} are both required for permutation testing."
            )

        predictor_trials, _ = _coerce_trials(predictor_input, predictor_name)
        target_trials, _ = _coerce_trials(target_input, target_name)
        _check_trial_lengths(predictor_trials, target_trials)
        resolved_surrogate = _resolve_permutation_surrogate(surrogate)
        resolved_tail = _resolve_permutation_tail(tail)

        predictions, _ = self.predict(
            stimulus=stimulus,
            response=response,
            average=False,
            tmin=tmin,
            tmax=tmax,
        )
        prediction_trials, _ = _coerce_trials(predictions, "prediction")
        observed_score, null_scores, p_value, z_score = _compute_permutation_test_scores(
            prediction_trials=prediction_trials,
            target_trials=target_trials,
            metric=self.metric,
            average=average,
            fs=float(self.fs),
            n_permutations=int(n_permutations),
            surrogate=resolved_surrogate,
            min_shift=min_shift,
            tail=resolved_tail,
            seed=seed,
            n_jobs=1 if n_jobs is None else n_jobs,
        )
        return PermutationTestResult(
            observed_score=observed_score,
            null_scores=null_scores,
            p_value=p_value,
            z_score=z_score,
            tail=resolved_tail,
            surrogate=resolved_surrogate,
            n_permutations=int(n_permutations),
        )

    def refit_permutation_test(
        self,
        *,
        train_stimulus: np.ndarray | Sequence[np.ndarray],
        train_response: np.ndarray | Sequence[np.ndarray],
        test_stimulus: np.ndarray | Sequence[np.ndarray],
        test_response: np.ndarray | Sequence[np.ndarray],
        n_permutations: int = 100,
        average: bool | Sequence[int] = True,
        surrogate: str = "circular_shift",
        min_shift: float | None = None,
        tail: str = "greater",
        seed: int | None = None,
        n_jobs: int | None = 1,
        fit_n_jobs: int | None = 1,
        fit_kwargs: dict[str, object] | None = None,
    ) -> PermutationTestResult:
        """Estimate a stronger null by retraining on surrogate-aligned data.

        This method fits one model on the original training alignment and then
        fits one fresh surrogate model per permutation after breaking the
        predictor-target alignment on the training set. All models are scored
        on the same held-out aligned evaluation data.

        Compared with :meth:`permutation_test`, this is slower but answers a
        stronger question: whether the full training pipeline, including
        regularization selection, outperforms chance under a surrogate null.

        Parameters
        ----------
        train_stimulus, train_response:
            Training data used to fit the observed and surrogate models.
        test_stimulus, test_response:
            Held-out aligned evaluation data used to score every fitted model.
        n_permutations:
            Number of surrogate refits used to form the null distribution.
        average:
            Score reduction applied to the observed and surrogate scores. This
            follows the same rules as :meth:`score`.
        surrogate:
            Strategy used to break the training alignment. ``"circular_shift"``
            rolls each training target trial by a random non-zero offset.
            ``"trial_shuffle"`` permutes whole training target trials and
            therefore requires at least two equal-length training trials.
        min_shift:
            Minimum circular shift, in seconds, used when
            ``surrogate="circular_shift"``.
        tail:
            Tail convention for the p-value calculation: ``"greater"``,
            ``"less"``, or ``"two-sided"``.
        seed:
            Optional random seed for reproducible surrogate generation.
        n_jobs:
            Number of worker threads used across surrogate refits.
        fit_n_jobs:
            Number of worker threads used inside each individual refit.
            The default ``1`` avoids nested oversubscription.
        fit_kwargs:
            Optional overrides for the stored fit configuration. When omitted,
            the most recent training configuration of this estimator is reused,
            except that bootstrap estimation is disabled and progress output is
            suppressed during the surrogate refits.

        Returns
        -------
        PermutationTestResult
            Container with the observed held-out score, surrogate null scores,
            p-value, and z-score.
        """

        train_stimulus_trials, _ = _coerce_trials(train_stimulus, "train_stimulus")
        train_response_trials, _ = _coerce_trials(train_response, "train_response")
        test_stimulus_trials, _ = _coerce_trials(test_stimulus, "test_stimulus")
        test_response_trials, _ = _coerce_trials(test_response, "test_response")
        _check_trial_lengths(train_stimulus_trials, train_response_trials)
        _check_trial_lengths(test_stimulus_trials, test_response_trials)

        train_x_trials, train_y_trials = self._get_xy(
            train_stimulus_trials,
            train_response_trials,
        )
        self._validate_dimensions(train_x_trials, train_y_trials)
        test_x_trials, test_y_trials = self._get_xy(
            test_stimulus_trials,
            test_response_trials,
        )
        self._validate_dimensions(test_x_trials, test_y_trials)

        train_config = self._resolve_refit_train_config(
            fit_kwargs=fit_kwargs,
            fit_n_jobs=fit_n_jobs,
        )
        resolved_surrogate = _resolve_permutation_surrogate(surrogate)
        resolved_tail = _resolve_permutation_tail(tail)
        permutation_specs = _build_permutation_specs(
            target_trials=train_y_trials,
            surrogate=resolved_surrogate,
            fs=float(train_config["fs"]),
            min_shift=min_shift,
            n_permutations=int(n_permutations),
            seed=seed,
        )

        def _fit_and_score(
            local_train_stimulus: Sequence[np.ndarray],
            local_train_response: Sequence[np.ndarray],
            *,
            local_train_config: dict[str, object],
        ) -> np.ndarray:
            refit = TRF(direction=self.direction, metric=self.metric)
            refit.train(
                stimulus=local_train_stimulus,
                response=local_train_response,
                **local_train_config,
            )
            return np.asarray(
                refit.score(
                    stimulus=test_stimulus,
                    response=test_response,
                    average=False,
                ),
                dtype=float,
            )

        observed_per_output = _fit_and_score(
            train_stimulus_trials,
            train_response_trials,
            local_train_config=self._copy_refit_train_config(train_config),
        )
        null_per_output = np.zeros(
            (int(n_permutations), observed_per_output.shape[0]),
            dtype=float,
        )

        def _score_permutation(spec: np.ndarray) -> np.ndarray:
            surrogate_targets = _surrogate_target_trials(
                train_y_trials,
                surrogate=resolved_surrogate,
                spec=spec,
            )
            local_train_config = self._copy_refit_train_config(train_config)
            local_train_config["trial_weights"] = self._surrogate_trial_weights(
                local_train_config.get("trial_weights"),
                surrogate=resolved_surrogate,
                spec=spec,
            )
            if self.direction == 1:
                local_train_stimulus = train_stimulus_trials
                local_train_response = surrogate_targets
            else:
                local_train_stimulus = surrogate_targets
                local_train_response = train_response_trials
            return _fit_and_score(
                local_train_stimulus,
                local_train_response,
                local_train_config=local_train_config,
            )

        resolved_n_jobs = min(_resolve_n_jobs(n_jobs), int(n_permutations))
        if resolved_n_jobs == 1:
            for permutation_index, spec in enumerate(permutation_specs):
                null_per_output[permutation_index] = _score_permutation(spec)
        else:
            with ThreadPoolExecutor(max_workers=resolved_n_jobs) as executor:
                futures = {
                    executor.submit(_score_permutation, spec): permutation_index
                    for permutation_index, spec in enumerate(permutation_specs)
                }
                for future in as_completed(futures):
                    permutation_index = futures[future]
                    null_per_output[permutation_index] = future.result()

        observed_score = _aggregate_metric(observed_per_output, average)
        null_scores = _aggregate_null_scores(null_per_output, average=average)
        p_value, z_score = _permutation_p_value_and_z_score(
            observed_score=observed_score,
            null_scores=null_scores,
            tail=resolved_tail,
        )
        return PermutationTestResult(
            observed_score=observed_score,
            null_scores=null_scores,
            p_value=p_value,
            z_score=z_score,
            tail=resolved_tail,
            surrogate=resolved_surrogate,
            n_permutations=int(n_permutations),
        )

    def predict(
        self,
        stimulus: np.ndarray | Sequence[np.ndarray] | None = None,
        response: np.ndarray | Sequence[np.ndarray] | None = None,
        *,
        average: bool | Sequence[int] = True,
        tmin: float | None = None,
        tmax: float | None = None,
    ) -> list[np.ndarray] | np.ndarray | tuple[list[np.ndarray] | np.ndarray, np.ndarray | float]:
        """Generate predictions from a fitted model.

        Parameters
        ----------
        stimulus, response:
            Inputs follow the same single-trial / multi-trial conventions as
            :meth:`train`. For forward models, ``stimulus`` is required. For
            backward models, ``response`` is required. If the corresponding
            observed target is also provided, the method additionally returns a
            prediction score.
        average:
            Reduction strategy for the returned score. ``True`` averages over
            outputs, ``False`` returns one score per output, and a sequence of
            indices averages only over selected outputs.
        tmin, tmax:
            Optional lag window used during prediction. If omitted, the fitted
            lag window is used.

        Returns
        -------
        prediction or (prediction, metric):
            Predicted trials are returned in the same single-trial vs list form
            as the predictor input. When observed targets are supplied, the
            method also returns the metric defined on the estimator. The score
            shape is controlled by ``average``.

        Notes
        -----
        Prediction is performed by convolving the predictor with the extracted
        time-domain kernel over the requested lag window. This means you can
        evaluate alternative lag windows on a fitted model without repeating
        spectral training, as long as the requested window stays within what is
        representable by the stored transfer function.
        """

        if self.weights is None or self.fs is None:
            raise ValueError("Model must be trained before prediction.")

        if self.direction == 1 and stimulus is None:
            raise ValueError("stimulus is required for a forward model.")
        if self.direction == -1 and response is None:
            raise ValueError("response is required for a backward model.")

        predictor_input = stimulus if self.direction == 1 else response
        target_input = response if self.direction == 1 else stimulus
        predictor_name = "stimulus" if self.direction == 1 else "response"
        target_name = "response" if self.direction == 1 else "stimulus"

        predictor_trials, predictor_is_single = _coerce_trials(predictor_input, predictor_name)
        target_trials = None
        if target_input is not None:
            target_trials, _ = _coerce_trials(target_input, target_name)
            _check_trial_lengths(predictor_trials, target_trials)

        n_inputs = self.weights.shape[0]
        n_outputs = self.weights.shape[-1]
        for predictor_trial in predictor_trials:
            if predictor_trial.shape[1] != n_inputs:
                raise ValueError(
                    f"Expected {n_inputs} predictor channels/features, got {predictor_trial.shape[1]}."
                )
        if target_trials is not None:
            for target_trial in target_trials:
                if target_trial.shape[1] != n_outputs:
                    raise ValueError(
                        f"Expected {n_outputs} target channels/features, got {target_trial.shape[1]}."
                    )

        weights, times = self.to_impulse_response(tmin=tmin, tmax=tmax)
        lag_start = int(round(times[0] * self.fs))
        predictions = _predict_trials_from_weights(
            predictor_trials,
            weights=weights,
            lag_start=lag_start,
        )

        returned_predictions: list[np.ndarray] | np.ndarray
        if predictor_is_single:
            returned_predictions = predictions[0]
        else:
            returned_predictions = predictions

        if target_trials is None:
            return returned_predictions

        metric = _score_prediction_trials(
            self.metric,
            target_trials,
            predictions,
        )
        return returned_predictions, _aggregate_metric(metric, average)

    def score(
        self,
        stimulus: np.ndarray | Sequence[np.ndarray] | None = None,
        response: np.ndarray | Sequence[np.ndarray] | None = None,
        *,
        average: bool | Sequence[int] = True,
        tmin: float | None = None,
        tmax: float | None = None,
    ) -> np.ndarray | float:
        """Score predictions without returning the predicted signals.

        This is a convenience wrapper around :meth:`predict` for workflows where
        only the metric is needed. Unlike :meth:`predict`, this method always
        requires the observed target side because it returns only the metric,
        not the predicted signals themselves.

        Parameters
        ----------
        stimulus, response, average, tmin, tmax:
            Identical to :meth:`predict`.

        Returns
        -------
        numpy.ndarray or float
            Prediction score computed with the estimator's configured metric.
        """

        target_input = response if self.direction == 1 else stimulus
        target_name = "response" if self.direction == 1 else "stimulus"
        if target_input is None:
            raise ValueError(
                f"{target_name} is required for score(). Pass the observed target side."
            )

        _, metric = self.predict(
            stimulus=stimulus,
            response=response,
            average=average,
            tmin=tmin,
            tmax=tmax,
        )
        return metric

    def _cross_validate(
        self,
        x_trials: Sequence[np.ndarray],
        y_trials: Sequence[np.ndarray],
        *,
        fs: float,
        tmin: float,
        tmax: float,
        feature_regularization_values: Sequence[np.ndarray],
        segment_length: int | None,
        overlap: float,
        n_fft: int | None,
        spectral_method: SpectralMethod,
        time_bandwidth: float,
        n_tapers: int | None,
        window: None | str | tuple[str, float] | np.ndarray,
        detrend: None | str,
        k: int,
        seed: int | None,
        average: bool | Sequence[int],
        show_progress: bool,
        n_jobs: int,
        raw_trial_weights: np.ndarray,
        spectral_cache: _SpectralCache | None,
    ) -> np.ndarray:
        n_trials = len(x_trials)
        if n_trials < 2:
            raise ValueError("Cross-validation needs at least two trials.")

        n_folds = n_trials if int(k) == -1 else int(k)
        if n_folds < 2:
            raise ValueError("k must be -1 or at least 2.")
        n_folds = min(n_folds, n_trials)

        indices = np.arange(n_trials)
        if seed is not None:
            rng = np.random.default_rng(seed)
            rng.shuffle(indices)
        folds = [fold for fold in np.array_split(indices, n_folds) if len(fold) > 0]

        per_reg_scores = np.zeros(
            (len(feature_regularization_values), len(folds), y_trials[0].shape[1]),
            dtype=float,
        )
        if spectral_cache is None:
            spectral_cache = _build_spectral_cache(
                x_trials,
                y_trials,
                segment_length=segment_length,
                overlap=overlap,
                n_fft=n_fft,
                spectral_method=spectral_method,
                time_bandwidth=time_bandwidth,
                n_tapers=n_tapers,
                window=window,
                detrend=detrend,
            )

        if (
            self.regularization_candidates is None
            or len(self.regularization_candidates) != len(feature_regularization_values)
        ):
            raise RuntimeError("Regularization candidates are inconsistent with the CV search grid.")

        fold_predictors: list[list[np.ndarray]] = []
        fold_targets: list[list[np.ndarray]] = []
        fold_spectra: list[tuple[np.ndarray, np.ndarray]] = []
        for val_idx in folds:
            val_predictors = [x_trials[i] for i in val_idx]
            val_targets = [y_trials[i] for i in val_idx]
            fold_predictors.append(val_predictors)
            fold_targets.append(val_targets)

            train_weights = raw_trial_weights.copy()
            train_weights[val_idx] = 0.0
            fold_spectra.append(
                _aggregate_cached_spectra(
                    spectral_cache,
                    raw_trial_weights=train_weights,
                )
            )

        progress_bar = (
            _SimpleProgressBar(
                total=len(feature_regularization_values) * len(folds),
                label="Cross-validating",
            )
            if show_progress
            else None
        )
        try:
            fold_inputs = list(zip(fold_spectra, fold_predictors, fold_targets, strict=True))
            resolved_n_jobs = min(_resolve_n_jobs(n_jobs), len(fold_inputs))
            if resolved_n_jobs == 1:
                for fold_index, ((cxx, cxy), val_predictors, val_targets) in enumerate(fold_inputs):
                    per_reg_scores[:, fold_index, :] = _score_regularization_grid_for_fold(
                        cxx=cxx,
                        cxy=cxy,
                        val_predictors=val_predictors,
                        val_targets=val_targets,
                        feature_regularization_values=feature_regularization_values,
                        fs=float(fs),
                        n_fft=spectral_cache.n_fft,
                        tmin=float(tmin),
                        tmax=float(tmax),
                        metric=self.metric,
                        progress_callback=None if progress_bar is None else progress_bar.update,
                    )
            else:
                with ThreadPoolExecutor(max_workers=resolved_n_jobs) as executor:
                    futures = {
                        executor.submit(
                            _score_regularization_grid_for_fold,
                            cxx=cxx,
                            cxy=cxy,
                            val_predictors=val_predictors,
                            val_targets=val_targets,
                            feature_regularization_values=feature_regularization_values,
                            fs=float(fs),
                            n_fft=spectral_cache.n_fft,
                            tmin=float(tmin),
                            tmax=float(tmax),
                            metric=self.metric,
                            progress_callback=None if progress_bar is None else progress_bar.update,
                        ): fold_index
                        for fold_index, ((cxx, cxy), val_predictors, val_targets) in enumerate(fold_inputs)
                    }
                    for future in as_completed(futures):
                        fold_index = futures[future]
                        per_reg_scores[:, fold_index, :] = future.result()
        finally:
            if progress_bar is not None:
                progress_bar.close()

        fold_mean = per_reg_scores.mean(axis=1)
        if average is False:
            return fold_mean
        if average is True:
            return fold_mean.mean(axis=1)
        return fold_mean[:, np.asarray(list(average), dtype=int)].mean(axis=1)

    def save(self, path: str | Path) -> None:
        """Serialize the estimator to disk using :mod:`pickle`.

        Parameters
        ----------
        path:
            Destination file. Parent directories must already exist.

        Notes
        -----
        The entire estimator instance is serialized, including fitted weights,
        spectral settings, chosen regularization, bootstrap intervals, and the
        configured scoring metric. Pickle files should only be loaded from
        trusted sources.
        """
        path = Path(path)
        if not path.parent.exists():
            raise FileNotFoundError(f"Directory does not exist: {path.parent}")
        with path.open("wb") as handle:
            pickle.dump(self, handle, pickle.HIGHEST_PROTOCOL)

    def load(self, path: str | Path) -> None:
        """Load estimator state from a pickle file into this instance.

        Parameters
        ----------
        path:
            Pickle file previously created by :meth:`save`.

        Notes
        -----
        The current instance is updated in place. After loading, compatibility
        checks fill in any newer attributes that may not have existed in older
        saved files.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File does not exist: {path}")
        with path.open("rb") as handle:
            loaded = pickle.load(handle)
        self.__dict__ = loaded.__dict__
        if not hasattr(self, "metric_name"):
            self.metric_name = getattr(self.metric, "__name__", None)
        if not hasattr(self, "bands"):
            self.bands = None
        if not hasattr(self, "feature_regularization") or self.feature_regularization is None:
            if self.weights is not None and isinstance(self.regularization, (float, np.floating)):
                self.feature_regularization = np.full(
                    self.weights.shape[0],
                    float(self.regularization),
                    dtype=float,
                )
            else:
                self.feature_regularization = None
        if not hasattr(self, "regularization_candidates"):
            self.regularization_candidates = (
                None if self.regularization is None else [_copy_value(self.regularization)]
            )
        if not hasattr(self, "segment_duration"):
            if self.segment_length is not None and self.fs is not None:
                self.segment_duration = float(self.segment_length) / float(self.fs)
            else:
                self.segment_duration = None
        if not hasattr(self, "spectral_method"):
            self.spectral_method = "standard"
        if not hasattr(self, "time_bandwidth"):
            self.time_bandwidth = None
        if not hasattr(self, "n_tapers"):
            self.n_tapers = None
        if not hasattr(self, "_fit_config"):
            self._fit_config = None

    def copy(self) -> "TRF":
        """Return a copy of the estimator and all learned arrays.

        Returns
        -------
        TRF
            Independent copy of the estimator state. NumPy arrays and other
            stored values are copied so that subsequent mutations on the copy
            do not affect the original instance.
        """

        copied = TRF(direction=self.direction, metric=self.metric)
        for key, value in self.__dict__.items():
            setattr(copied, key, _copy_value(value))
        return copied

    def _resolve_refit_train_config(
        self,
        *,
        fit_kwargs: dict[str, object] | None,
        fit_n_jobs: int | None,
    ) -> dict[str, object]:
        base_config = {} if self._fit_config is None else self._copy_refit_train_config(self._fit_config)
        if fit_kwargs is not None:
            if {"stimulus", "response"} & set(fit_kwargs):
                raise ValueError("fit_kwargs must not include stimulus or response.")
            for key, value in fit_kwargs.items():
                base_config[key] = _copy_value(value)

        required = {"fs", "tmin", "tmax", "regularization"}
        missing = sorted(required.difference(base_config))
        if missing:
            missing_formatted = ", ".join(missing)
            raise ValueError(
                "refit_permutation_test requires a stored fit configuration or fit_kwargs "
                f"containing: {missing_formatted}."
            )

        base_config["show_progress"] = False
        base_config["bootstrap_samples"] = 0
        base_config["bootstrap_seed"] = None
        if fit_n_jobs is not None:
            base_config["n_jobs"] = fit_n_jobs
        elif "n_jobs" not in base_config:
            base_config["n_jobs"] = 1
        return base_config

    @staticmethod
    def _copy_refit_train_config(config: dict[str, object]) -> dict[str, object]:
        return {key: _copy_value(value) for key, value in config.items()}

    @staticmethod
    def _surrogate_trial_weights(
        trial_weights: object,
        *,
        surrogate: str,
        spec: np.ndarray,
    ) -> object:
        copied = _copy_value(trial_weights)
        if surrogate != "trial_shuffle" or copied is None or isinstance(copied, str):
            return copied

        weights = np.asarray(copied, dtype=float)
        if weights.shape != spec.shape:
            raise ValueError(
                "Explicit trial_weights must match the number of training trials "
                "when using surrogate='trial_shuffle'."
            )
        return weights[spec].copy()

    def _get_xy(
        self,
        stimulus_trials: Sequence[np.ndarray],
        response_trials: Sequence[np.ndarray],
    ) -> tuple[Sequence[np.ndarray], Sequence[np.ndarray]]:
        if self.direction == 1:
            return stimulus_trials, response_trials
        return response_trials, stimulus_trials

    @staticmethod
    def _validate_dimensions(
        x_trials: Sequence[np.ndarray],
        y_trials: Sequence[np.ndarray],
    ) -> None:
        n_inputs = x_trials[0].shape[1]
        n_outputs = y_trials[0].shape[1]
        for trial in x_trials:
            if trial.shape[1] != n_inputs:
                raise ValueError("All predictor trials must have the same feature count.")
        for trial in y_trials:
            if trial.shape[1] != n_outputs:
                raise ValueError("All response trials must have the same channel count.")

    @staticmethod
    def _validate_fit_arguments(
        fs: float,
        tmin: float,
        tmax: float,
        segment_length: int | None,
        overlap: float,
        n_fft: int | None,
        spectral_method: SpectralMethod,
        time_bandwidth: float,
        n_tapers: int | None,
        window: None | str | tuple[str, float] | np.ndarray,
        detrend: None | str,
    ) -> None:
        if fs <= 0:
            raise ValueError("fs must be positive.")
        if tmax <= tmin:
            raise ValueError("tmax must be greater than tmin.")
        if segment_length is not None and int(segment_length) <= 0:
            raise ValueError("segment_length must be positive.")
        if not 0.0 <= overlap < 1.0:
            raise ValueError("overlap must lie in [0, 1).")
        if n_fft is not None and int(n_fft) <= 0:
            raise ValueError("n_fft must be positive.")
        resolved_method = _validate_spectral_method(spectral_method)
        if resolved_method == "multitaper":
            _resolve_multitaper_parameters(
                time_bandwidth=float(time_bandwidth),
                n_tapers=n_tapers,
            )
            if window is not None:
                raise ValueError("window must be None when spectral_method='multitaper'.")
        if detrend not in (None, "constant", "linear"):
            raise ValueError("detrend must be None, 'constant', or 'linear'.")
