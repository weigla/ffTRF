from __future__ import annotations

import numpy as np
import pytest

from fftrf.results import FrequencyResolvedWeights, TimeFrequencyPower


def _frequency_resolved_result() -> FrequencyResolvedWeights:
    return FrequencyResolvedWeights(
        frequencies=np.linspace(0.0, 10.0, 6),
        band_centers=np.array([2.0, 6.0]),
        filters=np.ones((6, 2)),
        times=np.arange(3.0),
        weights=np.arange(2 * 2 * 3 * 2, dtype=float).reshape(2, 2, 3, 2),
        scale="linear",
        value_mode="real",
        bandwidth=2.0,
    )


def _time_frequency_result() -> TimeFrequencyPower:
    return TimeFrequencyPower(
        frequencies=np.linspace(0.0, 10.0, 6),
        band_centers=np.array([2.0, 6.0]),
        filters=np.ones((6, 2)),
        times=np.arange(3.0),
        power=np.arange(2 * 2 * 3 * 2, dtype=float).reshape(2, 2, 3, 2),
        scale="linear",
        method="hilbert",
        bandwidth=2.0,
    )


@pytest.mark.parametrize(
    ("factory", "attribute"),
    [
        (_frequency_resolved_result, "weights"),
        (_time_frequency_result, "power"),
    ],
)
def test_frequency_by_lag_accessors_select_and_copy(factory, attribute: str) -> None:
    result = factory()
    values = result.at(input_index=1, output_index=1)
    stored = getattr(result, attribute)

    assert np.array_equal(values, stored[1, :, :, 1])
    values[0, 0] = -1.0
    assert stored[1, 0, 0, 1] != -1.0

    for kwargs, match in [
        ({"input_index": -1}, "input_index"),
        ({"input_index": 2}, "input_index"),
        ({"output_index": -1}, "output_index"),
        ({"output_index": 2}, "output_index"),
    ]:
        with pytest.raises(IndexError, match=match):
            result.at(**kwargs)
