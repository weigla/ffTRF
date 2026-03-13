"""Experimental Bayesian frequency-domain TRF fitting.

This module provides a Bayesian estimator with an API intentionally close to
``FrequencyTRF`` while keeping Bayesian-specific outputs such as posterior
covariance and credible intervals.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from scipy.fft import next_fast_len
from scipy.linalg import cho_factor, cho_solve

from ..model import (
    _aggregate_metric,
    _check_trial_lengths,
    _coerce_trials,
    _copy_value,
    _iter_segments,
    _normalize_trial_weights,
    _prepare_segment,
    _shifted_convolution,
    pearsonr,
)


@dataclass(slots=True)
class BayesianTRFResult:
    """Posterior summary for the experimental Bayesian TRF estimator.

    Attributes
    ----------
    weights:
        Posterior mean kernel weights with shape
        ``(n_inputs, n_lags, n_outputs)``.
    times:
        Lag axis in seconds corresponding to ``weights``.
    transfer_function:
        Complex-valued transfer function with shape
        ``(n_frequencies, n_inputs, n_outputs)``.
    frequencies:
        Frequency vector in Hz corresponding to ``transfer_function``.
    posterior_cov:
        Posterior covariance matrices with shape
        ``(n_outputs, n_parameters, n_parameters)``. Can be ``None`` if
        covariance storage was disabled.
    posterior_std:
        Marginal posterior standard deviation with the same shape as
        ``weights``.
    credible_interval:
        Approximate 95% marginal credible interval with shape
        ``(2, n_inputs, n_lags, n_outputs)``.
    alpha:
        Final prior precision for each output channel.
    beta:
        Final noise precision for each output channel.
    regularization:
        Equivalent ridge parameter per output on the same scale as
        :class:`fft_trf.FrequencyTRF`. Internally the spectral solver uses a
        corresponding FFT-scaled precision.
    prior:
        Prior type used during fitting.
    fit_mode:
        ``"evidence"`` when the regularization was learned from the data and
        ``"fixed_regularization"`` when it was supplied explicitly.
    converged:
        Boolean convergence flag per output channel.
    n_iter:
        Number of evidence iterations used per output channel.
    history:
        Hyperparameter update history per output channel.
    meta:
        Additional metadata about the constructed regression system.
    fs, tmin, tmax, segment_length, n_fft, overlap, window, detrend:
        Core fit metadata mirrored from :class:`fft_trf.FrequencyTRF`.
    direction:
        Modeling direction used during fitting.
    """

    weights: np.ndarray
    times: np.ndarray
    transfer_function: np.ndarray
    frequencies: np.ndarray
    posterior_cov: np.ndarray | None
    posterior_std: np.ndarray
    credible_interval: np.ndarray
    alpha: np.ndarray
    beta: np.ndarray
    regularization: float | np.ndarray
    prior: str
    fit_mode: str
    converged: np.ndarray
    n_iter: np.ndarray
    history: list[dict[str, list[float]]]
    meta: dict[str, Any]
    fs: float
    tmin: float
    tmax: float
    segment_length: int
    n_fft: int
    overlap: float
    window: None | str | tuple[str, float] | np.ndarray
    detrend: None | str
    direction: int


class BayesianFrequencyTRF:
    """Bayesian frequency-domain TRF estimator with a ``FrequencyTRF``-like API.

    This estimator mirrors the overall workflow of :class:`fft_trf.FrequencyTRF`
    as closely as practical:

    - call :meth:`train` to fit the model
    - inspect :attr:`weights`, :attr:`times`, :attr:`transfer_function`
    - call :meth:`predict` and :meth:`score`

    Unlike the standard estimator, this class fits a Gaussian posterior over the
    weights. Hyperparameters are chosen by empirical Bayes (evidence
    maximization), and the class retains posterior uncertainty information in
    addition to the posterior mean kernel.
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

        self.weights: np.ndarray | None = None
        self.times: np.ndarray | None = None
        self.transfer_function: np.ndarray | None = None
        self.frequencies: np.ndarray | None = None

        self.posterior_cov: np.ndarray | None = None
        self.posterior_std: np.ndarray | None = None
        self.credible_interval: np.ndarray | None = None

        self.alpha: np.ndarray | None = None
        self.beta: np.ndarray | None = None
        self.regularization: float | np.ndarray | None = None
        self.prior: str | None = None
        self.fit_mode: str | None = None
        self.converged: np.ndarray | None = None
        self.n_iter: np.ndarray | None = None
        self.history: list[dict[str, list[float]]] | None = None
        self.meta: dict[str, Any] | None = None

        self.fs: float | None = None
        self.tmin: float | None = None
        self.tmax: float | None = None
        self.segment_length: int | None = None
        self.n_fft: int | None = None
        self.overlap: float | None = None
        self.window: None | str | tuple[str, float] | np.ndarray = None
        self.detrend: None | str = None
        self.smoothness_order: int | None = None
        self.energy_thresh: float | None = None
        self.eps: float | None = None
        self.jitter: float | None = None

    def train(
        self,
        stimulus: np.ndarray | Sequence[np.ndarray],
        response: np.ndarray | Sequence[np.ndarray],
        fs: float,
        tmin: float,
        tmax: float,
        regularization: float | Sequence[float] | None = None,
        *,
        prior: str = "ridge",
        smoothness_order: int = 2,
        alpha_init: float = 1.0,
        beta_init: float = 1.0,
        max_iter: int = 500,
        tol: float = 1e-6,
        energy_thresh: float = 1e-8,
        eps: float = 0.0,
        jitter: float = 1e-6,
        segment_length: int | None = None,
        overlap: float = 0.0,
        n_fft: int | None = None,
        window: None | str | tuple[str, float] | np.ndarray = None,
        detrend: None | str = "constant",
        k: int = -1,
        average: bool | Sequence[int] = True,
        seed: int | None = None,
        trial_weights: None | str | Sequence[float] = None,
        store_covariance: bool = True,
    ) -> np.ndarray | float | None:
        """Fit the Bayesian TRF and store the posterior summary on the instance.

        The public signature mirrors :meth:`fft_trf.FrequencyTRF.train` as
        closely as practical:

        - ``regularization=None`` enables empirical-Bayes hyperparameter
          learning.
        - a scalar ``regularization`` fits a fixed-lambda Bayesian model.
        - a sequence of candidate values triggers cross-validation and returns
          per-candidate scores, just like :class:`fft_trf.FrequencyTRF`.
        """

        stimulus_trials, _ = _coerce_trials(stimulus, "stimulus")
        response_trials, _ = _coerce_trials(response, "response")
        _check_trial_lengths(stimulus_trials, response_trials)

        x_trials, y_trials = self._get_xy(stimulus_trials, response_trials)
        _validate_dimensions(x_trials, y_trials)
        _validate_fit_arguments(fs, tmin, tmax, segment_length, overlap, n_fft, detrend)

        if regularization is None:
            self._fit(
                x_trials,
                y_trials,
                fs=fs,
                tmin=tmin,
                tmax=tmax,
                regularization=None,
                prior=prior,
                smoothness_order=smoothness_order,
                alpha_init=alpha_init,
                beta_init=beta_init,
                max_iter=max_iter,
                tol=tol,
                energy_thresh=energy_thresh,
                eps=eps,
                jitter=jitter,
                segment_length=segment_length,
                overlap=overlap,
                n_fft=n_fft,
                window=window,
                detrend=detrend,
                trial_weights=trial_weights,
                store_covariance=store_covariance,
            )
            return None

        regularization_values = np.atleast_1d(np.asarray(regularization, dtype=float))
        if np.any(~np.isfinite(regularization_values)) or np.any(regularization_values < 0.0):
            raise ValueError("regularization values must be finite and non-negative.")

        if regularization_values.size == 1:
            self._fit(
                x_trials,
                y_trials,
                fs=fs,
                tmin=tmin,
                tmax=tmax,
                regularization=float(regularization_values[0]),
                prior=prior,
                smoothness_order=smoothness_order,
                alpha_init=alpha_init,
                beta_init=beta_init,
                max_iter=max_iter,
                tol=tol,
                energy_thresh=energy_thresh,
                eps=eps,
                jitter=jitter,
                segment_length=segment_length,
                overlap=overlap,
                n_fft=n_fft,
                window=window,
                detrend=detrend,
                trial_weights=trial_weights,
                store_covariance=store_covariance,
            )
            return None

        cv_scores = self._cross_validate(
            x_trials,
            y_trials,
            fs=fs,
            tmin=tmin,
            tmax=tmax,
            regularization_values=regularization_values,
            prior=prior,
            smoothness_order=smoothness_order,
            alpha_init=alpha_init,
            beta_init=beta_init,
            max_iter=max_iter,
            tol=tol,
            energy_thresh=energy_thresh,
            eps=eps,
            jitter=jitter,
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

        if np.ndim(cv_scores) == 1:
            best_index = int(np.argmax(cv_scores))
        else:
            best_index = int(np.argmax(np.asarray(cv_scores).mean(axis=1)))

        self._fit(
            x_trials,
            y_trials,
            fs=fs,
            tmin=tmin,
            tmax=tmax,
            regularization=float(regularization_values[best_index]),
            prior=prior,
            smoothness_order=smoothness_order,
            alpha_init=alpha_init,
            beta_init=beta_init,
            max_iter=max_iter,
            tol=tol,
            energy_thresh=energy_thresh,
            eps=eps,
            jitter=jitter,
            segment_length=segment_length,
            overlap=overlap,
            n_fft=n_fft,
            window=window,
            detrend=detrend,
            trial_weights=trial_weights,
            store_covariance=store_covariance,
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
        regularization: float | None,
        prior: str,
        smoothness_order: int,
        alpha_init: float,
        beta_init: float,
        max_iter: int,
        tol: float,
        energy_thresh: float,
        eps: float,
        jitter: float,
        segment_length: int | None,
        overlap: float,
        n_fft: int | None,
        window: None | str | tuple[str, float] | np.ndarray,
        detrend: None | str,
        trial_weights: None | str | Sequence[float],
        store_covariance: bool,
    ) -> None:
        result = _fit_bayesian_frequency_trf(
            x_trials,
            y_trials,
            fs=fs,
            tmin=tmin,
            tmax=tmax,
            direction=self.direction,
            regularization=regularization,
            prior=prior,
            smoothness_order=smoothness_order,
            alpha_init=alpha_init,
            beta_init=beta_init,
            max_iter=max_iter,
            tol=tol,
            energy_thresh=energy_thresh,
            eps=eps,
            jitter=jitter,
            segment_length=segment_length,
            overlap=overlap,
            n_fft=n_fft,
            window=window,
            detrend=detrend,
            trial_weights=trial_weights,
            store_covariance=store_covariance,
        )

        self.weights = result.weights.copy()
        self.times = result.times.copy()
        self.transfer_function = result.transfer_function.copy()
        self.frequencies = result.frequencies.copy()
        self.posterior_cov = None if result.posterior_cov is None else result.posterior_cov.copy()
        self.posterior_std = result.posterior_std.copy()
        self.credible_interval = result.credible_interval.copy()
        self.alpha = result.alpha.copy()
        self.beta = result.beta.copy()
        self.regularization = _copy_value(result.regularization)
        self.prior = result.prior
        self.fit_mode = result.fit_mode
        self.converged = result.converged.copy()
        self.n_iter = result.n_iter.copy()
        self.history = _copy_history(result.history)
        self.meta = _copy_meta(result.meta)

        self.fs = result.fs
        self.tmin = result.tmin
        self.tmax = result.tmax
        self.segment_length = result.segment_length
        self.n_fft = result.n_fft
        self.overlap = result.overlap
        self.window = _copy_value(result.window)
        self.detrend = result.detrend
        self.smoothness_order = smoothness_order
        self.energy_thresh = energy_thresh
        self.eps = eps
        self.jitter = jitter

    def _cross_validate(
        self,
        x_trials: Sequence[np.ndarray],
        y_trials: Sequence[np.ndarray],
        *,
        fs: float,
        tmin: float,
        tmax: float,
        regularization_values: np.ndarray,
        prior: str,
        smoothness_order: int,
        alpha_init: float,
        beta_init: float,
        max_iter: int,
        tol: float,
        energy_thresh: float,
        eps: float,
        jitter: float,
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
                candidate = BayesianFrequencyTRF(direction=self.direction, metric=self.metric)

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
                    prior=prior,
                    smoothness_order=smoothness_order,
                    alpha_init=alpha_init,
                    beta_init=beta_init,
                    max_iter=max_iter,
                    tol=tol,
                    energy_thresh=energy_thresh,
                    eps=eps,
                    jitter=jitter,
                    segment_length=segment_length,
                    overlap=overlap,
                    n_fft=n_fft,
                    window=window,
                    detrend=detrend,
                    trial_weights=fold_trial_weights,
                    store_covariance=False,
                )
                per_reg_scores[reg_index, fold_index, :] = candidate.score(
                    stimulus=(
                        [x_trials[i] for i in val_idx]
                        if self.direction == 1
                        else [y_trials[i] for i in val_idx]
                    ),
                    response=(
                        [y_trials[i] for i in val_idx]
                        if self.direction == 1
                        else [x_trials[i] for i in val_idx]
                    ),
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

    def result(self) -> BayesianTRFResult:
        """Return the current posterior summary as a dataclass."""

        _require_trained(self)
        return BayesianTRFResult(
            weights=self.weights.copy(),
            times=self.times.copy(),
            transfer_function=self.transfer_function.copy(),
            frequencies=self.frequencies.copy(),
            posterior_cov=None if self.posterior_cov is None else self.posterior_cov.copy(),
            posterior_std=self.posterior_std.copy(),
            credible_interval=self.credible_interval.copy(),
            alpha=self.alpha.copy(),
            beta=self.beta.copy(),
            regularization=_copy_value(self.regularization),
            prior=str(self.prior),
            fit_mode=str(self.fit_mode),
            converged=self.converged.copy(),
            n_iter=self.n_iter.copy(),
            history=_copy_history(self.history),
            meta=_copy_meta(self.meta),
            fs=float(self.fs),
            tmin=float(self.tmin),
            tmax=float(self.tmax),
            segment_length=int(self.segment_length),
            n_fft=int(self.n_fft),
            overlap=float(self.overlap),
            window=_copy_value(self.window),
            detrend=self.detrend,
            direction=int(self.direction),
        )

    def to_impulse_response(
        self,
        tmin: float | None = None,
        tmax: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the time-domain kernel over the requested lag window."""

        _require_trained(self)
        if tmin is None and tmax is None:
            return self.weights.copy(), self.times.copy()

        tmin = self.tmin if tmin is None else tmin
        tmax = self.tmax if tmax is None else tmax
        mask = (self.times >= tmin) & (self.times < tmax)
        if not np.any(mask):
            raise ValueError("Requested lag window does not overlap the fitted kernel.")
        return self.weights[:, mask, :].copy(), self.times[mask].copy()

    def predict(
        self,
        stimulus: np.ndarray | Sequence[np.ndarray] | None = None,
        response: np.ndarray | Sequence[np.ndarray] | None = None,
        *,
        average: bool | Sequence[int] = True,
        tmin: float | None = None,
        tmax: float | None = None,
    ) -> np.ndarray | list[np.ndarray] | tuple[np.ndarray | list[np.ndarray], np.ndarray | float]:
        """Generate predictions and optionally score them against observations."""

        _require_trained(self)
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

        weights, times = self.to_impulse_response(tmin=tmin, tmax=tmax)
        lag_start = int(round(times[0] * self.fs))
        kernel = np.transpose(weights, (1, 0, 2))

        predictions = []
        metrics = []
        n_inputs = weights.shape[0]
        n_outputs = weights.shape[-1]
        for index, predictor_trial in enumerate(predictor_trials):
            if predictor_trial.shape[1] != n_inputs:
                raise ValueError(
                    f"Expected {n_inputs} predictor channels/features, got {predictor_trial.shape[1]}."
                )
            if target_trials is not None and target_trials[index].shape[1] != n_outputs:
                raise ValueError(
                    f"Expected {n_outputs} target channels/features, got {target_trials[index].shape[1]}."
                )
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

        returned_predictions: np.ndarray | list[np.ndarray]
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
        """Return the prediction metric without returning the predictions."""

        _, metric = self.predict(
            stimulus=stimulus,
            response=response,
            average=average,
            tmin=tmin,
            tmax=tmax,
        )
        return metric

    def save(self, path: str | Path) -> None:
        """Serialize the estimator to disk with :mod:`pickle`."""

        path = Path(path)
        if not path.parent.exists():
            raise FileNotFoundError(f"Directory does not exist: {path.parent}")
        with path.open("wb") as handle:
            pickle.dump(self, handle, pickle.HIGHEST_PROTOCOL)

    def load(self, path: str | Path) -> None:
        """Load estimator state from disk into this instance."""

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File does not exist: {path}")
        with path.open("rb") as handle:
            loaded = pickle.load(handle)
        self.__dict__ = loaded.__dict__

    def copy(self) -> "BayesianFrequencyTRF":
        """Return a copy of the estimator and its learned attributes."""

        copied = BayesianFrequencyTRF(direction=self.direction, metric=self.metric)
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


def fit_bayesian_frequency_trf(
    stimulus: np.ndarray | Sequence[np.ndarray],
    response: np.ndarray | Sequence[np.ndarray],
    *,
    fs: float,
    tmin: float,
    tmax: float,
    direction: int = 1,
    regularization: float | None = None,
    prior: str = "ridge",
    smoothness_order: int = 2,
    alpha_init: float = 1.0,
    beta_init: float = 1.0,
    max_iter: int = 500,
    tol: float = 1e-6,
    energy_thresh: float = 1e-8,
    eps: float = 0.0,
    jitter: float = 1e-6,
    segment_length: int | None = None,
    overlap: float = 0.0,
    n_fft: int | None = None,
    window: None | str | tuple[str, float] | np.ndarray = None,
    detrend: None | str = "constant",
    trial_weights: None | str | Sequence[float] = None,
    store_covariance: bool = True,
) -> BayesianTRFResult:
    """Compatibility wrapper that returns the Bayesian posterior summary."""

    model = BayesianFrequencyTRF(direction=direction)
    model.train(
        stimulus=stimulus,
        response=response,
        fs=fs,
        tmin=tmin,
        tmax=tmax,
        regularization=regularization,
        prior=prior,
        smoothness_order=smoothness_order,
        alpha_init=alpha_init,
        beta_init=beta_init,
        max_iter=max_iter,
        tol=tol,
        energy_thresh=energy_thresh,
        eps=eps,
        jitter=jitter,
        segment_length=segment_length,
        overlap=overlap,
        n_fft=n_fft,
        window=window,
        detrend=detrend,
        trial_weights=trial_weights,
        store_covariance=store_covariance,
    )
    return model.result()


def predict_bayesian_frequency_trf(
    stimulus: np.ndarray | Sequence[np.ndarray] | None,
    result: BayesianTRFResult | BayesianFrequencyTRF,
    *,
    response: np.ndarray | Sequence[np.ndarray] | None = None,
    average: bool | Sequence[int] = True,
    tmin: float | None = None,
    tmax: float | None = None,
) -> np.ndarray | list[np.ndarray] | tuple[np.ndarray | list[np.ndarray], np.ndarray | float]:
    """Compatibility prediction wrapper for a result object or estimator."""

    if isinstance(result, BayesianFrequencyTRF):
        return result.predict(
            stimulus=stimulus,
            response=response,
            average=average,
            tmin=tmin,
            tmax=tmax,
        )

    if result.direction == 1:
        if stimulus is None:
            raise ValueError("stimulus must be provided for a forward result object.")
        predictor_input = stimulus
        target_input = response
        predictor_name = "stimulus"
        target_name = "response"
    else:
        if response is None:
            raise ValueError("response must be provided for a backward result object.")
        predictor_input = response
        target_input = stimulus
        predictor_name = "response"
        target_name = "stimulus"

    predictor_trials, is_single = _coerce_trials(predictor_input, predictor_name)
    target_trials = None
    if target_input is not None:
        target_trials, _ = _coerce_trials(target_input, target_name)
        _check_trial_lengths(predictor_trials, target_trials)

    weights, times = _slice_result_weights(result, tmin=tmin, tmax=tmax)
    lag_start = int(round(times[0] * result.meta["fs"]))
    kernel = np.transpose(weights, (1, 0, 2))

    predictions = []
    for predictor_trial in predictor_trials:
        prediction = np.zeros((predictor_trial.shape[0], weights.shape[-1]), dtype=float)
        for input_index in range(weights.shape[0]):
            for output_index in range(weights.shape[-1]):
                prediction[:, output_index] += _shifted_convolution(
                    predictor_trial[:, input_index],
                    kernel[:, input_index, output_index],
                    lag_start=lag_start,
                    out_length=predictor_trial.shape[0],
                )
        predictions.append(prediction)
    if target_trials is None:
        return predictions[0] if is_single else predictions

    metric = np.mean(
        np.vstack([pearsonr(true, pred) for true, pred in zip(target_trials, predictions)]),
        axis=0,
    )
    returned = predictions[0] if is_single else predictions
    return returned, _aggregate_metric(metric, average)


def _fit_bayesian_frequency_trf(
    x_trials: Sequence[np.ndarray],
    y_trials: Sequence[np.ndarray],
    *,
    fs: float,
    tmin: float,
    tmax: float,
    direction: int,
    regularization: float | None,
    prior: str,
    smoothness_order: int,
    alpha_init: float,
    beta_init: float,
    max_iter: int,
    tol: float,
    energy_thresh: float,
    eps: float,
    jitter: float,
    segment_length: int | None,
    overlap: float,
    n_fft: int | None,
    window: None | str | tuple[str, float] | np.ndarray,
    detrend: None | str,
    trial_weights: None | str | Sequence[float],
    store_covariance: bool,
) -> BayesianTRFResult:
    n_taps = int(round((tmax - tmin) * fs))
    resolved_segment_length = (
        max(trial.shape[0] for trial in x_trials)
        if segment_length is None
        else int(segment_length)
    )
    minimum_n_fft = resolved_segment_length + n_taps - 1
    resolved_n_fft = (
        next_fast_len(minimum_n_fft)
        if n_fft is None
        else int(n_fft)
    )
    if resolved_n_fft < minimum_n_fft:
        raise ValueError("n_fft must be at least segment_length + n_lags - 1.")

    design, targets, times, meta = _build_regression_system(
        x_trials,
        y_trials,
        fs=fs,
        tmin=tmin,
        tmax=tmax,
        segment_length=resolved_segment_length,
        overlap=overlap,
        n_fft=resolved_n_fft,
        window=window,
        detrend=detrend,
        energy_thresh=energy_thresh,
        eps=eps,
        trial_weights=trial_weights,
    )

    n_inputs = x_trials[0].shape[1]
    n_outputs = y_trials[0].shape[1]
    n_taps = len(times)
    n_parameters = n_inputs * n_taps
    regularization_scale = float(resolved_n_fft)

    temporal_precision = _make_temporal_prior_precision(
        n_taps=n_taps,
        prior=prior,
        smoothness_order=smoothness_order,
        jitter=jitter,
    )
    prior_precision = np.kron(np.eye(n_inputs, dtype=float), temporal_precision)

    gram = design.T @ design
    weights_flat = np.zeros((n_parameters, n_outputs), dtype=float)
    posterior_std_flat = np.zeros((n_parameters, n_outputs), dtype=float)
    posterior_cov = None if not store_covariance else np.zeros(
        (n_outputs, n_parameters, n_parameters),
        dtype=float,
    )
    alpha = np.zeros(n_outputs, dtype=float)
    beta = np.zeros(n_outputs, dtype=float)
    regularization_values = np.zeros(n_outputs, dtype=float)
    converged = np.zeros(n_outputs, dtype=bool)
    n_iter = np.zeros(n_outputs, dtype=int)
    history = []

    for output_index in range(n_outputs):
        cross = design.T @ targets[:, output_index]
        if regularization is None:
            result = _evidence_updates(
                design=design,
                gram=gram,
                cross=cross,
                target=targets[:, output_index],
                prior_precision=prior_precision,
                alpha_init=alpha_init,
                beta_init=beta_init,
                max_iter=max_iter,
                tol=tol,
                regularization_scale=regularization_scale,
            )
        else:
            result = _fixed_regularization_updates(
                design=design,
                gram=gram,
                cross=cross,
                target=targets[:, output_index],
                prior_precision=prior_precision,
                regularization=float(regularization),
                regularization_scale=regularization_scale,
            )
        weights_flat[:, output_index] = result["weights"]
        posterior_std_flat[:, output_index] = np.sqrt(
            np.clip(np.diag(result["posterior_cov"]), 0.0, None)
        )
        if posterior_cov is not None:
            posterior_cov[output_index] = result["posterior_cov"]
        alpha[output_index] = result["alpha"]
        beta[output_index] = result["beta"]
        regularization_values[output_index] = result["regularization"]
        converged[output_index] = result["converged"]
        n_iter[output_index] = result["n_iter"]
        history.append(result["history"])

    weights = weights_flat.reshape(n_inputs, n_taps, n_outputs)
    posterior_std = posterior_std_flat.reshape(n_inputs, n_taps, n_outputs)
    credible_interval = np.stack(
        [weights - 1.96 * posterior_std, weights + 1.96 * posterior_std],
        axis=0,
    )
    transfer_function = np.fft.rfft(
        np.transpose(weights, (1, 0, 2)),
        n=resolved_n_fft,
        axis=0,
    )
    frequencies = np.fft.rfftfreq(resolved_n_fft, d=1.0 / fs)

    result = BayesianTRFResult(
        weights=weights,
        times=times,
        transfer_function=transfer_function,
        frequencies=frequencies,
        posterior_cov=posterior_cov,
        posterior_std=posterior_std,
        credible_interval=credible_interval,
        alpha=alpha,
        beta=beta,
        regularization=_collapse_output_values(regularization_values),
        prior=prior,
        fit_mode="evidence" if regularization is None else "fixed_regularization",
        converged=converged,
        n_iter=n_iter,
        history=history,
        meta=meta,
        fs=float(fs),
        tmin=float(tmin),
        tmax=float(tmax),
        segment_length=int(resolved_segment_length),
        n_fft=int(resolved_n_fft),
        overlap=float(overlap),
        window=_copy_value(window),
        detrend=detrend,
        direction=direction,
    )
    return result


def _build_regression_system(
    x_trials: Sequence[np.ndarray],
    y_trials: Sequence[np.ndarray],
    *,
    fs: float,
    tmin: float,
    tmax: float,
    segment_length: int,
    overlap: float,
    n_fft: int,
    window: None | str | tuple[str, float] | np.ndarray,
    detrend: None | str,
    energy_thresh: float,
    eps: float,
    trial_weights: None | str | Sequence[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    n_inputs = x_trials[0].shape[1]
    n_outputs = y_trials[0].shape[1]
    n_taps = int(round((tmax - tmin) * fs))
    lag_start = int(round(tmin * fs))

    cos_part, sin_part = _rfft_design(n_fft, n_taps)
    design_blocks = []
    target_blocks = [[] for _ in range(n_outputs)]
    trial_meta = []
    window_cache: dict[int, np.ndarray] = {}
    weight_values = _normalize_trial_weights(y_trials, trial_weights)

    for trial_index, (x_trial, y_trial) in enumerate(zip(x_trials, y_trials)):
        segments = list(
            _iter_segments(
                x_trial,
                y_trial,
                segment_length=segment_length,
                overlap=overlap,
            )
        )
        segment_scale = np.sqrt(weight_values[trial_index] / len(segments))
        segment_meta = []
        for x_segment, y_segment in segments:
            x_prepared = _prepare_segment(
                x_segment,
                target_length=segment_length,
                window_cache=window_cache,
                window=window,
                detrend=detrend,
            )
            y_prepared = _prepare_segment(
                y_segment,
                target_length=segment_length,
                window_cache=window_cache,
                window=window,
                detrend=detrend,
            )
            x_shifted = _shift_multichannel(x_prepared, lag_start)
            x_fft = np.fft.rfft(x_shifted, n=n_fft, axis=0)
            y_fft = np.fft.rfft(y_prepared, n=n_fft, axis=0)

            total_power = np.sum((np.conjugate(x_fft) * x_fft).real, axis=1)
            mask = total_power > energy_thresh
            if not np.any(mask):
                continue

            selected_bins = np.flatnonzero(mask)
            row_weights = _stacked_rfft_row_weights(selected_bins, n_fft)
            design_block = _complex_design_matrix(
                x_fft[mask],
                cos_part[mask],
                sin_part[mask],
            )
            design_blocks.append(segment_scale * row_weights[:, np.newaxis] * design_block)
            for output_index in range(n_outputs):
                target = np.concatenate([
                    y_fft[mask, output_index].real,
                    y_fft[mask, output_index].imag,
                ])
                target_blocks[output_index].append(segment_scale * row_weights * target)

            segment_meta.append(
                {
                    "n_bins_used": int(mask.sum()),
                    "segment_length": int(segment_length),
                    "energy_floor": float(total_power[mask].min() + eps),
                }
            )
        trial_meta.append({"n_segments": len(segment_meta), "segments": segment_meta})

    if not design_blocks:
        raise ValueError("All segments were masked; lower energy_thresh.")

    design = np.vstack(design_blocks)
    targets = np.column_stack([np.concatenate(blocks) for blocks in target_blocks])
    times = tmin + np.arange(n_taps) / fs
    meta = {
        "design_shape": design.shape,
        "n_trials": len(x_trials),
        "n_inputs": n_inputs,
        "n_outputs": n_outputs,
        "n_taps": n_taps,
        "lag_start": lag_start,
        "fs": fs,
        "energy_thresh": energy_thresh,
        "eps": eps,
        "regularization_scale": n_fft,
        "trial_meta": trial_meta,
    }
    return design, targets, times, meta


def _complex_design_matrix(
    x_fft: np.ndarray,
    cos_part: np.ndarray,
    sin_part: np.ndarray,
) -> np.ndarray:
    n_bins, n_inputs = x_fft.shape
    n_taps = cos_part.shape[1]
    real_block = np.zeros((n_bins, n_inputs * n_taps), dtype=float)
    imag_block = np.zeros((n_bins, n_inputs * n_taps), dtype=float)

    for input_index in range(n_inputs):
        start = input_index * n_taps
        stop = start + n_taps
        real = x_fft[:, input_index].real[:, None]
        imag = x_fft[:, input_index].imag[:, None]
        real_block[:, start:stop] = real * cos_part - imag * sin_part
        imag_block[:, start:stop] = real * sin_part + imag * cos_part
    return np.vstack([real_block, imag_block])


def _shift_multichannel(data: np.ndarray, shift: int) -> np.ndarray:
    shifted = np.zeros_like(data)
    if shift > 0:
        if shift < data.shape[0]:
            shifted[shift:, :] = data[: data.shape[0] - shift, :]
    elif shift < 0:
        step = -shift
        if step < data.shape[0]:
            shifted[: data.shape[0] - step, :] = data[step:, :]
    else:
        shifted[:] = data
    return shifted


def _make_temporal_prior_precision(
    *,
    n_taps: int,
    prior: str,
    smoothness_order: int,
    jitter: float,
) -> np.ndarray:
    if prior == "ridge":
        return np.eye(n_taps, dtype=float)
    if prior != "smooth":
        raise ValueError("prior must be either 'ridge' or 'smooth'.")
    if smoothness_order < 1:
        raise ValueError("smoothness_order must be at least 1.")

    differences = np.eye(n_taps, dtype=float)
    for _ in range(smoothness_order):
        differences = np.diff(differences, axis=0)
    precision = differences.T @ differences
    precision += jitter * np.eye(n_taps, dtype=float)
    return precision


def _rfft_design(n_fft: int, n_taps: int) -> tuple[np.ndarray, np.ndarray]:
    k = np.arange(n_fft // 2 + 1)[:, None]
    n = np.arange(n_taps)[None, :]
    theta = 2 * np.pi * k * n / n_fft
    return np.cos(theta), -np.sin(theta)


def _stacked_rfft_row_weights(indices: np.ndarray, n_fft: int) -> np.ndarray:
    weights = np.ones(indices.shape[0], dtype=float)
    interior = indices != 0
    if n_fft % 2 == 0:
        interior &= indices != (n_fft // 2)
    weights[interior] = np.sqrt(2.0)
    return np.concatenate([weights, weights])


def _evidence_updates(
    *,
    design: np.ndarray,
    gram: np.ndarray,
    cross: np.ndarray,
    target: np.ndarray,
    prior_precision: np.ndarray,
    alpha_init: float,
    beta_init: float,
    max_iter: int,
    tol: float,
    regularization_scale: float,
) -> dict[str, Any]:
    n_obs = target.shape[0]
    n_parameters = gram.shape[0]
    alpha = float(alpha_init)
    beta = float(beta_init)
    history = {
        "alpha": [alpha],
        "beta": [beta],
        "regularization": [
            (alpha / max(beta, np.finfo(float).eps)) / regularization_scale
        ],
    }
    identity = np.eye(n_parameters, dtype=float)
    floor = np.finfo(float).eps

    posterior_cov = identity.copy()
    weights = np.zeros(n_parameters, dtype=float)
    converged = False
    for iteration in range(1, max_iter + 1):
        posterior_precision = alpha * prior_precision + beta * gram
        chol = cho_factor(posterior_precision, lower=True, check_finite=False)
        posterior_cov = cho_solve(chol, identity, check_finite=False)
        weights = beta * cho_solve(chol, cross, check_finite=False)

        gamma = n_parameters - alpha * np.trace(posterior_cov @ prior_precision)
        weight_energy = max(float(weights.T @ prior_precision @ weights), floor)
        residual = target - design @ weights
        residual_energy = max(float(residual.T @ residual), floor)

        alpha_new = max(gamma / weight_energy, floor)
        beta_new = max((n_obs - gamma) / residual_energy, floor)

        history["alpha"].append(alpha_new)
        history["beta"].append(beta_new)
        history["regularization"].append(
            (alpha_new / max(beta_new, floor)) / regularization_scale
        )

        alpha_change = abs(alpha_new - alpha) / max(alpha, floor)
        beta_change = abs(beta_new - beta) / max(beta, floor)
        alpha, beta = alpha_new, beta_new
        if max(alpha_change, beta_change) < tol:
            converged = True
            break

    return {
        "weights": weights,
        "posterior_cov": posterior_cov,
        "alpha": alpha,
        "beta": beta,
        "regularization": (alpha / max(beta, floor)) / regularization_scale,
        "converged": converged,
        "n_iter": iteration,
        "history": history,
    }


def _fixed_regularization_updates(
    *,
    design: np.ndarray,
    gram: np.ndarray,
    cross: np.ndarray,
    target: np.ndarray,
    prior_precision: np.ndarray,
    regularization: float,
    regularization_scale: float,
) -> dict[str, Any]:
    n_obs = target.shape[0]
    n_parameters = gram.shape[0]
    floor = np.finfo(float).eps
    identity = np.eye(n_parameters, dtype=float)

    spectral_regularization = regularization * regularization_scale
    posterior_system = gram + spectral_regularization * prior_precision
    chol = cho_factor(posterior_system, lower=True, check_finite=False)
    system_inv = cho_solve(chol, identity, check_finite=False)
    weights = cho_solve(chol, cross, check_finite=False)

    gamma = n_parameters - spectral_regularization * np.trace(system_inv @ prior_precision)
    gamma = float(np.clip(gamma, 0.0, n_obs - floor))
    residual = target - design @ weights
    residual_energy = max(float(residual.T @ residual), floor)
    beta = max((n_obs - gamma) / residual_energy, floor)
    alpha = spectral_regularization * beta

    return {
        "weights": weights,
        "posterior_cov": system_inv / beta,
        "alpha": alpha,
        "beta": beta,
        "regularization": regularization,
        "converged": True,
        "n_iter": 1,
        "history": {
            "alpha": [alpha],
            "beta": [beta],
            "regularization": [regularization],
        },
    }


def _require_trained(model: BayesianFrequencyTRF) -> None:
    if model.weights is None or model.times is None or model.meta is None:
        raise ValueError("Model must be trained before using this method.")


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


def _slice_result_weights(
    result: BayesianTRFResult,
    *,
    tmin: float | None,
    tmax: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    if tmin is None and tmax is None:
        return result.weights.copy(), result.times.copy()
    tmin = result.times[0] if tmin is None else tmin
    if tmax is None:
        step = (result.times[1] - result.times[0]) if result.times.size > 1 else (1.0 / result.meta["fs"])
        tmax = result.times[-1] + step
    mask = (result.times >= tmin) & (result.times < tmax)
    if not np.any(mask):
        raise ValueError("Requested lag window does not overlap the stored result.")
    return result.weights[:, mask, :].copy(), result.times[mask].copy()


def _collapse_output_values(values: np.ndarray) -> float | np.ndarray:
    if values.shape == (1,):
        return float(values[0])
    return values.copy()


def _copy_meta(meta: dict[str, Any] | None) -> dict[str, Any] | None:
    if meta is None:
        return None
    copied: dict[str, Any] = {}
    for key, value in meta.items():
        if isinstance(value, np.ndarray):
            copied[key] = value.copy()
        elif isinstance(value, dict):
            copied[key] = _copy_meta(value)
        elif isinstance(value, list):
            copied[key] = [
                _copy_meta(item) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            copied[key] = value
    return copied


def _copy_history(history: list[dict[str, list[float]]] | None) -> list[dict[str, list[float]]] | None:
    if history is None:
        return None
    return [{key: values[:] for key, values in item.items()} for item in history]
