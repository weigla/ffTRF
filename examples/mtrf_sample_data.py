"""Helpers for loading the public mTRF speech EEG sample dataset.

The functions here intentionally live under ``examples/`` because they are only
needed for optional comparison scripts. The main ``ffTRF`` package remains free
of external-data download logic.
"""

from __future__ import annotations

from pathlib import Path
from urllib.error import URLError
from urllib.request import urlretrieve

import numpy as np

SAMPLE_DATA_URL = (
    "https://github.com/powerfulbean/mTRFpy/raw/master/tests/data/speech_data.npy"
)


def ensure_sample_data(
    cache_dir: str | Path = "artifacts/mtrf_data",
) -> Path:
    """Return the local path to the speech EEG sample, downloading if needed."""

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    sample_path = cache_dir / "speech_data.npy"
    if sample_path.exists():
        return sample_path

    try:
        urlretrieve(SAMPLE_DATA_URL, sample_path)
    except OSError as exc:
        raise RuntimeError(
            "Unable to download the optional mTRF sample dataset. "
            f"Tried: {SAMPLE_DATA_URL}"
        ) from exc
    except URLError as exc:
        raise RuntimeError(
            "Network access is required the first time the optional mTRF sample "
            "dataset is used."
        ) from exc
    return sample_path


def load_sample_data(
    *,
    cache_dir: str | Path = "artifacts/mtrf_data",
    n_segments: int = 10,
    normalize: bool = True,
) -> tuple[list[np.ndarray], list[np.ndarray], int]:
    """Load the public speech EEG sample used in the mTRF tutorials.

    The returned arrays match the structure of ``mtrf.model.load_sample_data``:
    lists of ``(samples, features)`` and ``(samples, channels)`` arrays,
    optionally z-scored segment-wise.
    """

    data = np.load(ensure_sample_data(cache_dir), allow_pickle=True).item()
    stimulus = np.array_split(np.asarray(data["stimulus"], dtype=float), n_segments)
    response = np.array_split(np.asarray(data["response"], dtype=float), n_segments)
    fs = int(np.asarray(data["samplerate"]).squeeze())

    if normalize:
        for index, (x_trial, y_trial) in enumerate(zip(stimulus, response)):
            stimulus[index] = _zscore_columns(x_trial)
            response[index] = _zscore_columns(y_trial)

    return stimulus, response, fs


def exact_lag_window_seconds(
    *,
    fs: float,
    nominal_stop_seconds: float = 0.4,
) -> tuple[int, float]:
    """Return an integer-lag window close to the requested stop time."""

    n_lags = int(np.ceil(nominal_stop_seconds * fs))
    return n_lags, n_lags / float(fs)


def _zscore_columns(x: np.ndarray) -> np.ndarray:
    """Z-score a 2D array column-wise with safe zero-variance handling."""

    x = np.asarray(x, dtype=float)
    centered = x - x.mean(axis=0, keepdims=True)
    scale = np.clip(centered.std(axis=0, keepdims=True), np.finfo(float).eps, None)
    return centered / scale
