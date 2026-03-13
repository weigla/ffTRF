"""Main frequency-domain TRF estimator and core scoring utilities.

The central object exported by this module is :class:`FrequencyTRF`. It mirrors
the basic workflow of mTRF-style toolboxes while estimating the mapping in the
frequency domain:

1. compute cross-spectra between stimulus and response
2. solve a ridge-regularized linear system at each frequency bin
3. convert the learned transfer function back into a time-domain kernel for
   interpretation and prediction

This makes the implementation convenient for high-sample-rate analyses such as
auditory brainstem response (ABR) work on EEG or MEG recordings.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
from scipy.fft import next_fast_len
from scipy.signal import detrend as scipy_detrend
from scipy.signal import fftconvolve, get_window


def pearsonr(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Compute column-wise Pearson correlation.

    Parameters
    ----------
    y_true:
        Observed samples arranged as ``(n_samples, n_outputs)``.
    y_pred:
        Predicted samples with the same shape as ``y_true``.

    Returns
    -------
    numpy.ndarray
        One correlation coefficient per output channel / feature.

    Notes
    -----
    This is the default scoring metric used by :class:`FrequencyTRF`. It is
    intentionally lightweight and does not return p-values.
    """

    y_true = _ensure_2d(y_true, "y_true")
    y_pred = _ensure_2d(y_pred, "y_pred")
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")

    y_true = y_true - y_true.mean(axis=0, keepdims=True)
    y_pred = y_pred - y_pred.mean(axis=0, keepdims=True)

    numerator = np.sum(y_true * y_pred, axis=0)
    denominator = np.sqrt(
        np.sum(y_true**2, axis=0) * np.sum(y_pred**2, axis=0)
    )
    out = np.zeros(y_true.shape[1], dtype=float)
    valid = denominator > np.finfo(float).eps
    out[valid] = numerator[valid] / denominator[valid]
    return out


def _ensure_2d(array: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(array, dtype=float)
    if array.ndim == 1:
        return array[:, np.newaxis]
    if array.ndim != 2:
        raise ValueError(f"{name} must be 1D or 2D, got shape {array.shape!r}.")
    return array


def _coerce_trials(
    data: np.ndarray | Sequence[np.ndarray],
    name: str,
) -> tuple[list[np.ndarray], bool]:
    if isinstance(data, np.ndarray):
        return [_ensure_2d(data, name)], True
    if not isinstance(data, Sequence) or len(data) == 0:
        raise ValueError(f"{name} must be a non-empty array or sequence of arrays.")
    trials = [_ensure_2d(trial, name) for trial in data]
    return trials, False


def _check_trial_lengths(a_trials: Sequence[np.ndarray], b_trials: Sequence[np.ndarray]) -> None:
    if len(a_trials) != len(b_trials):
        raise ValueError("Stimulus and response must contain the same number of trials.")
    for index, (a_trial, b_trial) in enumerate(zip(a_trials, b_trials)):
        if a_trial.shape[0] != b_trial.shape[0]:
            raise ValueError(
                f"Trial {index} has mismatched lengths: "
                f"{a_trial.shape[0]} vs {b_trial.shape[0]} samples."
            )


def _validate_average_arg(average: bool | Sequence[int]) -> None:
    if average is False or average is True:
        return
    if not isinstance(average, Sequence) or len(average) == 0:
        raise ValueError("average must be True, False, or a non-empty sequence of indices.")


def _aggregate_metric(
    metric: np.ndarray,
    average: bool | Sequence[int],
) -> np.ndarray | float:
    _validate_average_arg(average)
    metric = np.asarray(metric, dtype=float)
    if average is False:
        return metric
    if average is True:
        return float(metric.mean())
    return float(metric[np.asarray(list(average), dtype=int)].mean())


def _normalize_trial_weights(
    y_trials: Sequence[np.ndarray],
    trial_weights: None | str | Sequence[float],
) -> np.ndarray:
    if trial_weights is None:
        weights = np.ones(len(y_trials), dtype=float)
    elif isinstance(trial_weights, str):
        if trial_weights != "inverse_variance":
            raise ValueError("trial_weights must be None, 'inverse_variance', or a weight vector.")
        variances = [
            np.var(y_trial, axis=0).mean()
            for y_trial in y_trials
        ]
        weights = 1.0 / np.clip(variances, np.finfo(float).eps, None)
    else:
        weights = np.asarray(trial_weights, dtype=float)
        if weights.shape != (len(y_trials),):
            raise ValueError("Explicit trial weights must match the number of trials.")

    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Trial weights must sum to a positive finite value.")
    return weights / total


def _copy_value(value):
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, list):
        return [_copy_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _copy_value(item) for key, item in value.items()}
    return value


def _prepare_segment(
    segment: np.ndarray,
    *,
    target_length: int,
    window_cache: dict[int, np.ndarray],
    window: None | str | tuple[str, float] | np.ndarray,
    detrend: None | str,
) -> np.ndarray:
    if detrend is not None:
        segment = scipy_detrend(segment, axis=0, type=detrend)
    if window is not None:
        if isinstance(window, np.ndarray):
            if window.shape != (segment.shape[0],):
                raise ValueError("Explicit window array must match the segment length.")
            window_values = window
        else:
            window_values = window_cache.get(segment.shape[0])
            if window_values is None:
                window_values = get_window(window, segment.shape[0], fftbins=True)
                window_cache[segment.shape[0]] = window_values
        segment = segment * window_values[:, np.newaxis]
    if segment.shape[0] == target_length:
        return segment

    padded = np.zeros((target_length, segment.shape[1]), dtype=float)
    padded[: segment.shape[0], :] = segment
    return padded


def _iter_segments(
    x_trial: np.ndarray,
    y_trial: np.ndarray,
    segment_length: int,
    overlap: float,
) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    n_samples = x_trial.shape[0]
    if segment_length >= n_samples:
        yield x_trial, y_trial
        return

    step = max(1, int(round(segment_length * (1.0 - overlap))))
    starts = list(range(0, n_samples - segment_length + 1, step))
    if not starts:
        yield x_trial, y_trial
        return

    last_stop = -1
    for start in starts:
        stop = start + segment_length
        yield x_trial[start:stop], y_trial[start:stop]
        last_stop = stop

    if last_stop < n_samples:
        start = n_samples - segment_length
        yield x_trial[start:n_samples], y_trial[start:n_samples]


def _shifted_convolution(
    signal_in: np.ndarray,
    kernel: np.ndarray,
    lag_start: int,
    out_length: int,
) -> np.ndarray:
    full = fftconvolve(signal_in, kernel, mode="full")
    offset = -lag_start

    prediction = np.zeros(out_length, dtype=float)
    src_start = max(offset, 0)
    dst_start = max(-offset, 0)
    length = min(full.shape[0] - src_start, out_length - dst_start)
    if length > 0:
        prediction[dst_start : dst_start + length] = full[src_start : src_start + length]
    return prediction


class FrequencyTRF:
    """
    Estimate stimulus-response mappings in the frequency domain.

    ``FrequencyTRF`` is the main estimator of this toolbox. Its public API is
    intentionally close to ``mTRFpy``:

    - call :meth:`train` to fit the model
    - call :meth:`predict` to generate predicted responses or stimuli
    - call :meth:`score` to evaluate predictions
    - inspect :attr:`weights` and :attr:`times` as the time-domain kernel

    Unlike a classic time-domain TRF, the fit is performed through
    ridge-regularized spectral deconvolution. This is often attractive for
    high-rate continuous data where explicitly building large lag matrices is
    cumbersome.

    Parameters
    ----------
    direction:
        Modeling direction. Use ``1`` for a forward model
        (stimulus -> neural response) and ``-1`` for a backward model
        (neural response -> stimulus).
    metric:
        Callable used to score predictions. It must accept ``(y_true, y_pred)``
        and return one score per output column. By default a column-wise Pearson
        correlation is used.

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
        Selected ridge parameter.
    fs:
        Sampling rate used during fitting.

    Examples
    --------
    >>> import numpy as np
    >>> from fft_trf import FrequencyTRF
    >>> x = np.random.randn(2000, 1)
    >>> y = np.random.randn(2000, 1)
    >>> model = FrequencyTRF(direction=1)
    >>> model.train(x, y, fs=1000, tmin=0.0, tmax=0.03, regularization=1e-3)
    >>> prediction = model.predict(stimulus=x)
    """

    def __init__(
        self,
        direction: int = 1,
        metric: Callable[[np.ndarray, np.ndarray], np.ndarray] = pearsonr,
    ) -> None:
        if direction not in (1, -1):
            raise ValueError("direction must be 1 (forward) or -1 (backward).")
        if not callable(metric):
            raise ValueError("metric must be callable.")

        self.direction = direction
        self.metric = metric

        self.transfer_function: np.ndarray | None = None
        self.frequencies: np.ndarray | None = None
        self.weights: np.ndarray | None = None
        self.times: np.ndarray | None = None

        self.fs: float | None = None
        self.regularization: float | None = None
        self.segment_length: int | None = None
        self.n_fft: int | None = None
        self.overlap: float | None = None
        self.window: None | str | tuple[str, float] | np.ndarray = None
        self.detrend: None | str = None
        self.tmin: float | None = None
        self.tmax: float | None = None

    def train(
        self,
        stimulus: np.ndarray | Sequence[np.ndarray],
        response: np.ndarray | Sequence[np.ndarray],
        fs: float,
        tmin: float,
        tmax: float,
        regularization: float | Sequence[float],
        *,
        segment_length: int | None = None,
        overlap: float = 0.0,
        n_fft: int | None = None,
        window: None | str | tuple[str, float] | np.ndarray = None,
        detrend: None | str = "constant",
        k: int = -1,
        average: bool | Sequence[int] = True,
        seed: int | None = None,
        trial_weights: None | str | Sequence[float] = None,
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
            transfer function as a time-domain kernel. For ABR-style work this
            is often something like ``tmin=-0.005`` and ``tmax=0.030``.
        regularization:
            Either a single ridge value or a sequence of candidate values. The
            value is applied directly as ``lambda * I`` in the spectral linear
            system. When a sequence is provided, cross-validation is used to
            select the best value before fitting the final model on all training
            trials.
        segment_length:
            Segment size used to estimate cross-spectra. If ``None``, each trial
            is treated as a single segment.
        overlap:
            Fractional overlap between neighboring segments. Must lie in
            ``[0, 1)``.
        n_fft:
            FFT size used for spectral estimation. If omitted, a fast FFT length
            is chosen automatically from ``segment_length``.
        window:
            Window applied to each segment before FFT. By default no window is
            applied, which keeps the behavior closer to a standard lagged ridge
            fit. When using short overlapping segments, ``window="hann"`` is
            often a good choice.
        detrend:
            Optional detrending passed to :func:`scipy.signal.detrend`.
        k:
            Number of cross-validation folds when multiple regularization values
            are supplied. ``-1`` means leave-one-out over trials.
        average:
            How cross-validation scores should be reduced across output
            channels/features. ``True`` returns a single score per regularization
            value, ``False`` returns one score per output, and a sequence of
            indices averages only over the selected outputs.
        seed:
            Optional random seed for shuffling trial order before creating folds.
        trial_weights:
            Optional trial weights. Use ``"inverse_variance"`` for ABR-style
            inverse-variance weighting or pass an explicit vector with one weight
            per training trial.

        Returns
        -------
        None or numpy.ndarray
            ``None`` when a single regularization value is fitted directly.
            Otherwise returns cross-validation scores for each candidate
            regularization value.

        Notes
        -----
        The fitted model is always stored on the instance, even when
        cross-validation is used. In that case the final fit uses the selected
        regularization value and all provided trials.
        """

        stimulus_trials, _ = _coerce_trials(stimulus, "stimulus")
        response_trials, _ = _coerce_trials(response, "response")
        _check_trial_lengths(stimulus_trials, response_trials)

        x_trials, y_trials = self._get_xy(stimulus_trials, response_trials)
        self._validate_dimensions(x_trials, y_trials)
        self._validate_fit_arguments(fs, tmin, tmax, segment_length, overlap, n_fft, detrend)

        regularization_values = np.atleast_1d(np.asarray(regularization, dtype=float))
        if regularization_values.size == 1:
            self._fit(
                x_trials,
                y_trials,
                fs=fs,
                tmin=tmin,
                tmax=tmax,
                regularization=float(regularization_values[0]),
                segment_length=segment_length,
                overlap=overlap,
                n_fft=n_fft,
                window=window,
                detrend=detrend,
                trial_weights=trial_weights,
            )
            return None

        cv_scores = self._cross_validate(
            x_trials,
            y_trials,
            fs=fs,
            tmin=tmin,
            tmax=tmax,
            regularization_values=regularization_values,
            segment_length=segment_length,
            overlap=overlap,
            n_fft=n_fft,
            window=window,
            detrend=detrend,
            k=k,
            seed=seed,
            average=average,
            trial_weights=trial_weights,
        )

        if cv_scores.ndim == 1:
            best_index = int(np.argmax(cv_scores))
        else:
            best_index = int(np.argmax(cv_scores.mean(axis=1)))

        self._fit(
            x_trials,
            y_trials,
            fs=fs,
            tmin=tmin,
            tmax=tmax,
            regularization=float(regularization_values[best_index]),
            segment_length=segment_length,
            overlap=overlap,
            n_fft=n_fft,
            window=window,
            detrend=detrend,
            trial_weights=trial_weights,
        )
        return cv_scores

    def _fit(
        self,
        x_trials: Sequence[np.ndarray],
        y_trials: Sequence[np.ndarray],
        *,
        fs: float,
        tmin: float,
        tmax: float,
        regularization: float,
        segment_length: int | None,
        overlap: float,
        n_fft: int | None,
        window: None | str | tuple[str, float] | np.ndarray,
        detrend: None | str,
        trial_weights: None | str | Sequence[float],
    ) -> None:
        resolved_segment_length = (
            max(trial.shape[0] for trial in x_trials)
            if segment_length is None
            else int(segment_length)
        )
        resolved_n_fft = (
            next_fast_len(resolved_segment_length)
            if n_fft is None
            else int(n_fft)
        )
        if resolved_n_fft < resolved_segment_length:
            raise ValueError("n_fft must be at least as large as segment_length.")

        trial_weight_values = _normalize_trial_weights(y_trials, trial_weights)
        n_inputs = x_trials[0].shape[1]
        n_outputs = y_trials[0].shape[1]
        n_frequencies = resolved_n_fft // 2 + 1

        cxx = np.zeros((n_frequencies, n_inputs, n_inputs), dtype=np.complex128)
        cxy = np.zeros((n_frequencies, n_inputs, n_outputs), dtype=np.complex128)

        window_cache: dict[int, np.ndarray] = {}
        for trial_index, (x_trial, y_trial) in enumerate(zip(x_trials, y_trials)):
            segments = list(
                _iter_segments(
                    x_trial,
                    y_trial,
                    segment_length=resolved_segment_length,
                    overlap=overlap,
                )
            )
            segment_weight = trial_weight_values[trial_index] / len(segments)
            for x_segment, y_segment in segments:
                x_prepared = _prepare_segment(
                    x_segment,
                    target_length=resolved_segment_length,
                    window_cache=window_cache,
                    window=window,
                    detrend=detrend,
                )
                y_prepared = _prepare_segment(
                    y_segment,
                    target_length=resolved_segment_length,
                    window_cache=window_cache,
                    window=window,
                    detrend=detrend,
                )

                x_fft = np.fft.rfft(x_prepared, n=resolved_n_fft, axis=0)
                y_fft = np.fft.rfft(y_prepared, n=resolved_n_fft, axis=0)

                cxx += segment_weight * np.einsum(
                    "fi,fj->fij", np.conjugate(x_fft), x_fft, optimize=True
                )
                cxy += segment_weight * np.einsum(
                    "fi,fj->fij", np.conjugate(x_fft), y_fft, optimize=True
                )

        eye = np.eye(n_inputs, dtype=np.complex128)
        transfer_function = np.zeros_like(cxy)
        for frequency_index in range(n_frequencies):
            system = cxx[frequency_index] + regularization * eye
            transfer_function[frequency_index] = np.linalg.solve(
                system,
                cxy[frequency_index],
            )

        self.transfer_function = transfer_function
        self.frequencies = np.fft.rfftfreq(resolved_n_fft, d=1.0 / float(fs))
        self.fs = float(fs)
        self.regularization = float(regularization)
        self.segment_length = resolved_segment_length
        self.n_fft = resolved_n_fft
        self.overlap = overlap
        self.window = window
        self.detrend = detrend
        self.tmin = float(tmin)
        self.tmax = float(tmax)
        self.weights, self.times = self.to_impulse_response(tmin=tmin, tmax=tmax)

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
            :meth:`train` is reused.

        Returns
        -------
        weights, times:
            ``weights`` has shape ``(n_inputs, n_lags, n_outputs)`` and ``times``
            contains the corresponding lag values in seconds.

        Notes
        -----
        This method is useful when you want to inspect a different lag window
        without refitting the spectral model.
        """

        if self.transfer_function is None or self.fs is None or self.n_fft is None:
            raise ValueError("Model must be trained before extracting an impulse response.")

        tmin = self.tmin if tmin is None else tmin
        tmax = self.tmax if tmax is None else tmax
        if tmin is None or tmax is None:
            raise ValueError("tmin and tmax must be defined.")

        lag_start = int(round(float(tmin) * self.fs))
        lag_stop = int(round(float(tmax) * self.fs))
        if lag_stop <= lag_start:
            raise ValueError("tmax must be greater than tmin.")
        if lag_stop - lag_start > self.n_fft:
            raise ValueError("Requested lag window is longer than n_fft.")

        full_kernel = np.fft.irfft(self.transfer_function, n=self.n_fft, axis=0).real
        lag_indices = np.arange(lag_start, lag_stop, dtype=int)
        kernel = full_kernel[np.mod(lag_indices, self.n_fft), :, :]
        times = lag_indices / self.fs
        return np.transpose(kernel, (1, 0, 2)), times

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
            method also returns the metric defined on the estimator.

        Notes
        -----
        Prediction is performed by convolving the predictor with the extracted
        time-domain kernel over the requested lag window.
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
        kernel = np.transpose(weights, (1, 0, 2))

        predictions: list[np.ndarray] = []
        metrics = []
        for index, predictor_trial in enumerate(predictor_trials):
            prediction = np.zeros((predictor_trial.shape[0], n_outputs), dtype=float)
            for input_index in range(n_inputs):
                for output_index in range(n_outputs):
                    prediction[:, output_index] += _shifted_convolution(
                        predictor_trial[:, input_index],
                        kernel[:, input_index, output_index],
                        lag_start=lag_start,
                        out_length=predictor_trial.shape[0],
                    )
            predictions.append(prediction)
            if target_trials is not None:
                metrics.append(self.metric(target_trials[index], prediction))

        returned_predictions: list[np.ndarray] | np.ndarray
        if predictor_is_single:
            returned_predictions = predictions[0]
        else:
            returned_predictions = predictions

        if target_trials is None:
            return returned_predictions

        metric = np.mean(np.vstack(metrics), axis=0)
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
        only the metric is needed.
        """

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
        regularization_values: np.ndarray,
        segment_length: int | None,
        overlap: float,
        n_fft: int | None,
        window: None | str | tuple[str, float] | np.ndarray,
        detrend: None | str,
        k: int,
        seed: int | None,
        average: bool | Sequence[int],
        trial_weights: None | str | Sequence[float],
    ) -> np.ndarray:
        n_trials = len(x_trials)
        if n_trials < 2:
            raise ValueError("Cross-validation needs at least two trials.")

        n_folds = n_trials if k == -1 else int(k)
        if n_folds < 2:
            raise ValueError("k must be -1 or at least 2.")
        n_folds = min(n_folds, n_trials)

        indices = np.arange(n_trials)
        if seed is not None:
            rng = np.random.default_rng(seed)
            rng.shuffle(indices)
        folds = [fold for fold in np.array_split(indices, n_folds) if len(fold) > 0]

        per_reg_scores = np.zeros(
            (len(regularization_values), len(folds), y_trials[0].shape[1]),
            dtype=float,
        )
        explicit_weights = None
        if trial_weights is not None and not isinstance(trial_weights, str):
            explicit_weights = np.asarray(trial_weights, dtype=float)

        for reg_index, reg_value in enumerate(regularization_values):
            for fold_index, val_idx in enumerate(folds):
                train_idx = np.setdiff1d(indices, val_idx, assume_unique=True)
                candidate = FrequencyTRF(direction=self.direction, metric=self.metric)

                fold_trial_weights = trial_weights
                if explicit_weights is not None:
                    fold_trial_weights = explicit_weights[train_idx]

                candidate._fit(
                    [x_trials[i] for i in train_idx],
                    [y_trials[i] for i in train_idx],
                    fs=fs,
                    tmin=tmin,
                    tmax=tmax,
                    regularization=float(reg_value),
                    segment_length=segment_length,
                    overlap=overlap,
                    n_fft=n_fft,
                    window=window,
                    detrend=detrend,
                    trial_weights=fold_trial_weights,
                )
                per_reg_scores[reg_index, fold_index, :] = candidate.score(
                    stimulus=[x_trials[i] for i in val_idx] if self.direction == 1 else [y_trials[i] for i in val_idx],
                    response=[y_trials[i] for i in val_idx] if self.direction == 1 else [x_trials[i] for i in val_idx],
                    average=False,
                    tmin=tmin,
                    tmax=tmax,
                )

        fold_mean = per_reg_scores.mean(axis=1)
        if average is False:
            return fold_mean
        if average is True:
            return fold_mean.mean(axis=1)
        return fold_mean[:, np.asarray(list(average), dtype=int)].mean(axis=1)

    def save(self, path: str | Path) -> None:
        """Serialize the fitted estimator to disk using :mod:`pickle`."""
        path = Path(path)
        if not path.parent.exists():
            raise FileNotFoundError(f"Directory does not exist: {path.parent}")
        with path.open("wb") as handle:
            pickle.dump(self, handle, pickle.HIGHEST_PROTOCOL)

    def load(self, path: str | Path) -> None:
        """Load estimator state from a pickle file into this instance."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File does not exist: {path}")
        with path.open("rb") as handle:
            loaded = pickle.load(handle)
        self.__dict__ = loaded.__dict__

    def copy(self) -> "FrequencyTRF":
        """Return a copy of the estimator and all learned arrays."""

        copied = FrequencyTRF(direction=self.direction, metric=self.metric)
        for key, value in self.__dict__.items():
            setattr(copied, key, _copy_value(value))
        return copied

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
        if detrend not in (None, "constant", "linear"):
            raise ValueError("detrend must be None, 'constant', or 'linear'.")
