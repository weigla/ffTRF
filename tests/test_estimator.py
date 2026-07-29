from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

from fftrf import TRF
from fftrf.estimator import _resolve_directional_lag_window


@pytest.fixture(scope="module")
def fitted_model_and_data() -> tuple[TRF, list[np.ndarray], list[np.ndarray]]:
    rng = np.random.default_rng(501)
    fs = 100.0
    kernel = np.array([0.0, 0.8, -0.25, 0.1])
    stimulus: list[np.ndarray] = []
    response: list[np.ndarray] = []
    for _ in range(3):
        x = rng.standard_normal((128, 1))
        y = np.convolve(x[:, 0], kernel, mode="full")[: x.shape[0]]
        y += 0.02 * rng.standard_normal(x.shape[0])
        stimulus.append(x)
        response.append(y[:, np.newaxis])

    model = TRF(direction=1)
    model.train(
        stimulus=stimulus,
        response=response,
        fs=fs,
        tmin=0.0,
        tmax=0.04,
        regularization=1e-3,
        segment_length=64,
        overlap=0.5,
        window="hann",
    )
    return model, stimulus, response


def test_direction_validation_and_backward_lag_window() -> None:
    assert _resolve_directional_lag_window(
        direction=1,
        fs=100.0,
        tmin=0.0,
        tmax=0.04,
    ) == (0.0, 0.04)
    assert _resolve_directional_lag_window(
        direction=-1,
        fs=100.0,
        tmin=0.0,
        tmax=0.04,
    ) == (-0.03, 0.01)

    with pytest.raises(ValueError, match="direction"):
        _resolve_directional_lag_window(
            direction=0,
            fs=100.0,
            tmin=0.0,
            tmax=0.04,
        )
    with pytest.raises(ValueError, match="direction"):
        TRF(direction=0)


def test_untrained_estimator_methods_report_required_state() -> None:
    model = TRF()
    trial = np.zeros((8, 1))

    calls = [
        (lambda: model.to_impulse_response(), "trained"),
        (lambda: model.frequency_resolved_weights(), "trained"),
        (lambda: model.transfer_function_at(), "trained"),
        (
            lambda: model.cross_spectral_diagnostics(stimulus=trial, response=trial),
            "trained",
        ),
        (lambda: model.bootstrap_interval_at(), "No bootstrap"),
        (
            lambda: model.bootstrap_confidence_interval(trial, trial),
            "trained",
        ),
        (
            lambda: model.permutation_test(
                stimulus=trial,
                response=trial,
                n_permutations=2,
            ),
            "trained",
        ),
        (lambda: model.predict(stimulus=trial), "trained"),
    ]
    for call, match in calls:
        with pytest.raises(ValueError, match=match):
            call()


def test_frequency_domain_views_cover_value_modes_and_validation(
    fitted_model_and_data: tuple[TRF, list[np.ndarray], list[np.ndarray]],
) -> None:
    model, _, _ = fitted_model_and_data

    magnitude = model.frequency_resolved_weights(
        n_bands=3,
        value_mode="magnitude",
    )
    power = model.frequency_resolved_weights(
        n_bands=3,
        value_mode="power",
    )
    components = model.transfer_function_components_at(phase_unit="deg")
    sliced_weights, sliced_times = model.to_impulse_response(tmin=0.01, tmax=0.03)

    assert np.all(magnitude.weights >= 0.0)
    assert np.all(power.weights >= 0.0)
    assert np.allclose(power.weights, magnitude.weights**2)
    assert components.phase_unit == "deg"
    assert sliced_weights.shape[1] == sliced_times.size == 2

    with pytest.raises(ValueError, match="method"):
        model.time_frequency_power(method="wavelet")
    with pytest.raises(ValueError, match="tmax"):
        model.frequency_resolved_weights(tmin=0.03, tmax=0.03)
    with pytest.raises(ValueError, match="phase_unit"):
        model.transfer_function_components_at(phase_unit="turns")


def test_prediction_and_diagnostics_validate_directional_inputs_and_shapes(
    fitted_model_and_data: tuple[TRF, list[np.ndarray], list[np.ndarray]],
) -> None:
    model, stimulus, response = fitted_model_and_data

    with pytest.raises(ValueError, match="stimulus is required"):
        model.predict()
    with pytest.raises(ValueError, match="both required"):
        model.cross_spectral_diagnostics(stimulus=stimulus[0])
    with pytest.raises(ValueError, match="Expected 1 predictor"):
        model.predict(stimulus=np.zeros((32, 2)))
    with pytest.raises(ValueError, match="Expected 1 target"):
        model.predict(
            stimulus=np.zeros((32, 1)),
            response=np.zeros((32, 2)),
        )

    backward = model.copy()
    backward.direction = -1
    with pytest.raises(ValueError, match="response is required"):
        backward.predict(stimulus=stimulus[0])
    with pytest.raises(ValueError, match="stimulus is required for score"):
        backward.score(response=response[0])


def test_plot_diagnostics_can_be_computed_on_demand(
    fitted_model_and_data: tuple[TRF, list[np.ndarray], list[np.ndarray]],
) -> None:
    import matplotlib.pyplot as plt

    model, stimulus, response = fitted_model_and_data

    coherence_figure, _ = model.plot_coherence(
        stimulus=stimulus[-1],
        response=response[-1],
    )
    spectrum_figure, _ = model.plot_cross_spectrum(
        stimulus=stimulus[-1],
        response=response[-1],
        kind="magnitude",
    )

    plt.close(coherence_figure)
    plt.close(spectrum_figure)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"fs": 0.0}, "fs"),
        ({"tmin": 0.1, "tmax": 0.1}, "tmax"),
        ({"segment_length": 0}, "segment_length"),
        ({"overlap": 1.0}, "overlap"),
        ({"n_fft": 0}, "n_fft"),
        (
            {"spectral_method": "multitaper", "window": "hann"},
            "window must be None",
        ),
        ({"detrend": "quadratic"}, "detrend"),
    ],
)
def test_fit_argument_validation_rejects_invalid_settings(
    overrides: dict[str, object],
    match: str,
) -> None:
    settings = {
        "fs": 100.0,
        "tmin": 0.0,
        "tmax": 0.04,
        "segment_length": 32,
        "overlap": 0.0,
        "n_fft": 64,
        "spectral_method": "standard",
        "time_bandwidth": 3.5,
        "n_tapers": None,
        "window": None,
        "detrend": "constant",
    }
    settings.update(overrides)

    trial = np.zeros((32, 1))
    train_settings = {
        "segment_length": settings["segment_length"],
        "overlap": settings["overlap"],
        "n_fft": settings["n_fft"],
        "spectral_method": settings["spectral_method"],
        "time_bandwidth": settings["time_bandwidth"],
        "n_tapers": settings["n_tapers"],
        "window": settings["window"],
        "detrend": settings["detrend"],
    }
    with pytest.raises(ValueError, match=match):
        TRF().train(
            stimulus=trial,
            response=trial,
            fs=settings["fs"],
            tmin=settings["tmin"],
            tmax=settings["tmax"],
            regularization=1e-3,
            **train_settings,
        )


def test_dimension_validation_rejects_inconsistent_trial_features() -> None:
    with pytest.raises(ValueError, match="predictor trials"):
        TRF._validate_dimensions(
            [np.zeros((4, 1)), np.zeros((4, 2))],
            [np.zeros((4, 1)), np.zeros((4, 1))],
        )
    with pytest.raises(ValueError, match="response trials"):
        TRF._validate_dimensions(
            [np.zeros((4, 1)), np.zeros((4, 1))],
            [np.zeros((4, 1)), np.zeros((4, 2))],
        )


def test_training_rejects_negative_bootstrap_count() -> None:
    trial = np.zeros((16, 1))
    with pytest.raises(ValueError, match="bootstrap_samples"):
        TRF().train(
            stimulus=trial,
            response=trial,
            fs=100.0,
            tmin=0.0,
            tmax=0.02,
            regularization=1e-3,
            bootstrap_samples=-1,
        )


def test_cross_validation_requires_valid_independent_folds() -> None:
    trial = np.zeros((8, 1))
    with pytest.raises(ValueError, match="at least two trials"):
        TRF().train(
            stimulus=[trial],
            response=[trial],
            fs=100.0,
            tmin=0.0,
            tmax=0.02,
            regularization=[0.1, 1.0],
            k=2,
        )
    with pytest.raises(ValueError, match="at least 2"):
        TRF().train(
            stimulus=[trial, trial],
            response=[trial, trial],
            fs=100.0,
            tmin=0.0,
            tmax=0.02,
            regularization=[0.1, 1.0],
            k=1,
        )


@pytest.mark.parametrize("average", [False, [0]])
def test_small_cross_validated_fit_supports_output_reductions(average) -> None:
    rng = np.random.default_rng(502)
    stimulus = [rng.standard_normal((64, 1)) for _ in range(3)]
    response = [
        np.column_stack(
            [
                np.roll(trial[:, 0], 1),
                0.5 * np.roll(trial[:, 0], 2),
            ]
        )
        for trial in stimulus
    ]

    model = TRF()
    scores = model.train(
        stimulus=stimulus,
        response=response,
        fs=100.0,
        tmin=0.0,
        tmax=0.03,
        regularization=[1e-3, 1e-1],
        segment_length=32,
        k=3,
        average=average,
    )

    if average is False:
        assert scores.shape == (2, 2)
    else:
        assert scores.shape == (2,)
    assert model.regularization in {1e-3, 1e-1}


def test_save_load_errors_copy_and_legacy_state_migration(
    tmp_path: Path,
    fitted_model_and_data: tuple[TRF, list[np.ndarray], list[np.ndarray]],
) -> None:
    model, _, _ = fitted_model_and_data

    with pytest.raises(FileNotFoundError, match="Directory does not exist"):
        model.save(tmp_path / "missing" / "model.pkl")
    with pytest.raises(FileNotFoundError, match="File does not exist"):
        TRF().load(tmp_path / "absent.pkl")

    copied = model.copy()
    copied.weights[0, 0, 0] += 1.0
    assert not np.array_equal(copied.weights, model.weights)

    legacy = TRF(metric="r2")
    legacy.weights = np.ones((2, 3, 1))
    legacy.regularization = 0.25
    legacy.segment_length = 20
    legacy.fs = 100.0
    for attribute in [
        "metric_name",
        "bands",
        "feature_regularization",
        "regularization_candidates",
        "segment_duration",
        "spectral_method",
        "time_bandwidth",
        "n_tapers",
        "_fit_config",
    ]:
        delattr(legacy, attribute)

    path = tmp_path / "legacy.pkl"
    with path.open("wb") as handle:
        pickle.dump(legacy, handle, pickle.HIGHEST_PROTOCOL)

    restored = TRF()
    restored.load(path)

    assert restored.metric_name == "r2_score"
    assert restored.bands is None
    assert np.array_equal(restored.feature_regularization, [0.25, 0.25])
    assert restored.regularization_candidates == [0.25]
    assert restored.segment_duration == pytest.approx(0.2)
    assert restored.spectral_method == "standard"
    assert restored.time_bandwidth is None
    assert restored.n_tapers is None
    assert restored._fit_config is None


def test_refit_configuration_and_surrogate_weight_helpers_copy_inputs() -> None:
    model = TRF()

    with pytest.raises(ValueError, match="must not include stimulus"):
        model._resolve_refit_train_config(
            fit_kwargs={
                "stimulus": np.zeros((2, 1)),
                "fs": 100.0,
                "tmin": 0.0,
                "tmax": 0.02,
                "regularization": 0.1,
            },
            fit_n_jobs=1,
        )

    config = model._resolve_refit_train_config(
        fit_kwargs={
            "fs": 100.0,
            "tmin": 0.0,
            "tmax": 0.02,
            "regularization": np.array([0.1, 1.0]),
        },
        fit_n_jobs=None,
    )
    assert config["n_jobs"] == 1
    assert config["show_progress"] is False
    assert config["bootstrap_samples"] == 0
    assert config["bootstrap_seed"] is None

    copied = model._surrogate_trial_weights(
        np.array([0.2, 0.8]),
        surrogate="circular_shift",
        spec=np.array([1, 0]),
    )
    shuffled = model._surrogate_trial_weights(
        np.array([0.2, 0.8]),
        surrogate="trial_shuffle",
        spec=np.array([1, 0]),
    )
    assert np.array_equal(copied, [0.2, 0.8])
    assert np.array_equal(shuffled, [0.8, 0.2])

    with pytest.raises(ValueError, match="match the number"):
        model._surrogate_trial_weights(
            np.array([1.0]),
            surrogate="trial_shuffle",
            spec=np.array([1, 0]),
        )
