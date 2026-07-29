from __future__ import annotations

import numpy as np
import pytest

from fftrf.utils import (
    _aggregate_metric,
    _build_frequency_filterbank,
    _check_trial_lengths,
    _coerce_trials,
    _ensure_2d,
    _expand_feature_regularization,
    _group_delay_values,
    _normalize_trial_weights,
    _normalize_weight_vector,
    _phase_values,
    _resolve_frequency_scale,
    _resolve_frequency_weight_value_mode,
    _resolve_k_folds,
    _resolve_n_jobs,
    _resolve_raw_trial_weights,
    _resolve_regularization_candidates,
    _resolve_segment_length,
    _single_candidate_cv_requested,
    _validate_average_arg,
    _validate_bands,
    _warn_if_cv_arguments_are_unused,
    suggest_segment_settings,
)


def test_phase_and_group_delay_helpers_validate_and_convert() -> None:
    frequencies = np.linspace(0.0, 40.0, 9)
    delay = 0.025
    transfer = np.exp(-2j * np.pi * frequencies * delay)

    phase_rad, unit = _phase_values(transfer, phase_unit=" RAD ")
    phase_deg, degree_unit = _phase_values(transfer, phase_unit="deg")
    group_delay = _group_delay_values(frequencies, transfer)

    assert unit == "rad"
    assert degree_unit == "deg"
    assert np.allclose(phase_deg, np.rad2deg(phase_rad))
    assert np.allclose(group_delay, delay)
    assert np.array_equal(
        _group_delay_values(np.array([0.0]), np.array([1.0 + 0.0j])),
        np.array([0.0]),
    )

    with pytest.raises(ValueError, match="phase_unit"):
        _phase_values(transfer, phase_unit="turns")
    with pytest.raises(ValueError, match="1D arrays"):
        _group_delay_values(frequencies[:, np.newaxis], transfer)
    with pytest.raises(ValueError, match="matching lengths"):
        _group_delay_values(frequencies[:-1], transfer)


def test_array_and_trial_coercion_contracts() -> None:
    vector = np.arange(4.0)
    matrix = _ensure_2d(vector, "signal")
    trials, is_single = _coerce_trials(vector, "signal")
    multiple, multiple_is_single = _coerce_trials([vector, vector + 1.0], "signal")

    assert matrix.shape == (4, 1)
    assert trials[0].shape == (4, 1)
    assert is_single is True
    assert len(multiple) == 2
    assert multiple_is_single is False

    with pytest.raises(ValueError, match="1D or 2D"):
        _ensure_2d(np.zeros((2, 2, 2)), "signal")
    with pytest.raises(ValueError, match="non-empty"):
        _coerce_trials([], "signal")
    with pytest.raises(ValueError, match="same number of trials"):
        _check_trial_lengths([matrix], [matrix, matrix])
    with pytest.raises(ValueError, match="mismatched lengths"):
        _check_trial_lengths([matrix], [np.zeros((3, 1))])


def test_metric_aggregation_validates_reduction_and_selects_outputs() -> None:
    values = np.array([0.2, 0.6, 1.0])

    assert np.array_equal(_aggregate_metric(values, False), values)
    assert _aggregate_metric(values, True) == pytest.approx(0.6)
    assert _aggregate_metric(values, [0, 2]) == pytest.approx(0.6)

    for invalid in (None, [], 1):
        with pytest.raises(ValueError, match="average"):
            _validate_average_arg(invalid)  # type: ignore[arg-type]


def test_segment_length_resolution_and_suggestions_cover_boundaries() -> None:
    assert _resolve_segment_length(fs=100.0, segment_length=None, segment_duration=None) is None
    assert _resolve_segment_length(fs=100.0, segment_length=25, segment_duration=None) == 25
    assert _resolve_segment_length(fs=100.0, segment_length=None, segment_duration=0.255) == 26

    with pytest.raises(ValueError, match="either segment_length"):
        _resolve_segment_length(fs=100.0, segment_length=10, segment_duration=0.1)
    with pytest.raises(ValueError, match="finite and positive"):
        _resolve_segment_length(fs=100.0, segment_length=None, segment_duration=0.0)

    for kwargs, match in [
        ({"fs": 0.0, "tmin": 0.0, "tmax": 0.2}, "fs"),
        ({"fs": 100.0, "tmin": 0.2, "tmax": 0.2}, "tmax"),
        (
            {"fs": 100.0, "tmin": 0.0, "tmax": 0.2, "trial_duration": np.inf},
            "trial_duration",
        ),
    ]:
        with pytest.raises(ValueError, match=match):
            suggest_segment_settings(**kwargs)

    capped = suggest_segment_settings(
        fs=100.0,
        tmin=0.0,
        tmax=0.6,
        trial_duration=4.0,
    )
    assert capped["segment_duration"] == 2.0
    assert capped["segment_length"] == 200


def test_cv_argument_resolution_and_warnings(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _resolve_k_folds(" leave_one_out ") == -1
    assert _resolve_k_folds(4) == 4
    assert _resolve_n_jobs(None) == 1
    assert _resolve_n_jobs(3) == 3
    assert _single_candidate_cv_requested("loo") is True
    assert _single_candidate_cv_requested(-1) is False

    monkeypatch.setattr("fftrf.utils.os.cpu_count", lambda: 0)
    assert _resolve_n_jobs(-1) == 1

    with pytest.raises(ValueError, match="k must be"):
        _resolve_k_folds("five")
    with pytest.raises(ValueError, match="n_jobs"):
        _resolve_n_jobs(0)

    _warn_if_cv_arguments_are_unused(
        n_candidates=2,
        k=-1,
        average=False,
        seed=1,
        show_progress=True,
    )
    with pytest.warns(UserWarning, match="average, seed, show_progress are ignored"):
        _warn_if_cv_arguments_are_unused(
            n_candidates=1,
            k=-1,
            average=False,
            seed=1,
            show_progress=True,
        )


@pytest.mark.parametrize(
    ("function", "value", "match"),
    [
        (_resolve_frequency_scale, "octave", "scale"),
        (_resolve_frequency_weight_value_mode, "phase", "value_mode"),
    ],
)
def test_frequency_option_resolvers_reject_unknown_values(
    function, value: str, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        function(value)


def test_frequency_filterbank_covers_linear_log_and_single_band_modes() -> None:
    frequencies = np.linspace(0.0, 50.0, 11)

    centers, filters, scale, bandwidth = _build_frequency_filterbank(
        frequencies,
        n_bands=3,
        fmin=None,
        fmax=None,
        scale="linear",
        bandwidth=None,
    )
    assert scale == "linear"
    assert np.allclose(centers, [0.0, 25.0, 50.0])
    assert np.allclose(filters.sum(axis=1), 1.0)
    assert bandwidth > 0.0

    log_centers, log_filters, log_scale, _ = _build_frequency_filterbank(
        frequencies,
        n_bands=3,
        fmin=None,
        fmax=50.0,
        scale="log",
        bandwidth=8.0,
    )
    assert log_scale == "log"
    assert log_centers[0] == pytest.approx(5.0)
    assert np.all(log_filters[0] == 0.0)

    one_center, one_filter, _, one_bandwidth = _build_frequency_filterbank(
        frequencies,
        n_bands=1,
        fmin=10.0,
        fmax=30.0,
        scale="linear",
        bandwidth=None,
    )
    assert np.allclose(one_center, [20.0])
    assert np.array_equal(one_filter[:, 0], (frequencies >= 10.0) & (frequencies <= 30.0))
    assert one_bandwidth == pytest.approx(20.0)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"frequencies": np.zeros((2, 2)), "n_bands": 2}, "1D array"),
        ({"frequencies": np.array([0.0, 1.0]), "n_bands": 0}, "n_bands"),
        (
            {"frequencies": np.array([0.0]), "n_bands": 2, "scale": "log"},
            "positive frequency",
        ),
        (
            {
                "frequencies": np.array([0.0, 1.0]),
                "n_bands": 2,
                "fmin": 0.0,
                "scale": "log",
            },
            "fmin must be positive",
        ),
        (
            {
                "frequencies": np.array([0.0, 1.0]),
                "n_bands": 2,
                "fmin": 1.0,
                "fmax": 1.0,
            },
            "fmax must be greater",
        ),
        (
            {"frequencies": np.array([0.0, 1.0]), "n_bands": 2, "fmax": 2.0},
            "Nyquist",
        ),
        (
            {
                "frequencies": np.array([0.0, 1.0]),
                "n_bands": 2,
                "bandwidth": 0.0,
            },
            "bandwidth",
        ),
    ],
)
def test_frequency_filterbank_rejects_invalid_settings(
    kwargs: dict[str, object],
    match: str,
) -> None:
    defaults = {
        "fmin": None,
        "fmax": None,
        "scale": "linear",
        "bandwidth": None,
    }
    defaults.update(kwargs)
    with pytest.raises(ValueError, match=match):
        _build_frequency_filterbank(**defaults)


def test_band_validation_and_regularization_expansion() -> None:
    assert _validate_bands(None, n_inputs=3) is None
    assert _validate_bands([1, 2], n_inputs=3) == (1, 2)
    assert np.array_equal(
        _expand_feature_regularization([0.1, 2.0], n_inputs=3, bands=(1, 2)),
        [0.1, 2.0, 2.0],
    )

    for bands, match in [
        (1, "sequence"),
        ([], "non-empty"),
        ([1, 0], "positive integer"),
        ([1, 1.5], "positive integer"),
        ([1, 1], "must sum"),
    ]:
        with pytest.raises(ValueError, match=match):
            _validate_bands(bands, n_inputs=3)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="exactly one"):
        _expand_feature_regularization([0.1, 0.2], n_inputs=2, bands=None)
    with pytest.raises(ValueError, match="each entry"):
        _expand_feature_regularization([0.1], n_inputs=3, bands=(1, 2))


def test_regularization_candidates_cover_scalar_grid_and_explicit_bands() -> None:
    penalties, specs = _resolve_regularization_candidates(
        0.5,
        n_inputs=3,
        bands=(1, 2),
    )
    assert specs == [(0.5, 0.5)]
    assert np.array_equal(penalties[0], [0.5, 0.5, 0.5])

    penalties, specs = _resolve_regularization_candidates(
        [0.1, 1.0],
        n_inputs=3,
        bands=(1, 2),
    )
    assert specs == [(0.1, 0.1), (0.1, 1.0), (1.0, 0.1), (1.0, 1.0)]
    assert np.array_equal(penalties[-1], [1.0, 1.0, 1.0])

    penalties, specs = _resolve_regularization_candidates(
        [(0.1, 1.0), (1.0, 0.1)],
        n_inputs=3,
        bands=(1, 2),
    )
    assert specs == [(0.1, 1.0), (1.0, 0.1)]
    assert np.array_equal(penalties[0], [0.1, 1.0, 1.0])

    for regularization, bands, match in [
        ([], None, "non-empty"),
        ([-0.1], None, "finite and non-negative"),
        ([[0.1, 1.0]], None, "Without bands"),
        ([0.1, [1.0, 2.0]], (1, 2), "all scalars or all sequences"),
        ([[0.1]], (1, 2), "with 2 values"),
    ]:
        with pytest.raises(ValueError, match=match):
            _resolve_regularization_candidates(
                regularization,
                n_inputs=3,
                bands=bands,
            )


def test_trial_weight_resolution_and_normalization() -> None:
    trials = [np.array([[0.0], [1.0]]), np.array([[0.0], [2.0]])]

    assert np.array_equal(_resolve_raw_trial_weights(trials, None), [1.0, 1.0])
    inverse = _normalize_trial_weights(trials, "inverse_variance")
    assert inverse.sum() == pytest.approx(1.0)
    assert inverse[0] > inverse[1]
    assert np.allclose(_normalize_weight_vector(np.array([1.0, 3.0])), [0.25, 0.75])

    with pytest.raises(ValueError, match="trial_weights"):
        _resolve_raw_trial_weights(trials, "uniform")
    with pytest.raises(ValueError, match="number of trials"):
        _resolve_raw_trial_weights(trials, [1.0])
    with pytest.raises(ValueError, match="finite and non-negative"):
        _resolve_raw_trial_weights(trials, [1.0, -1.0])
    with pytest.raises(ValueError, match="positive finite"):
        _normalize_weight_vector(np.zeros(2))
