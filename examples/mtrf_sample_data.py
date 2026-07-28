"""Helpers for loading the public mTRF speech EEG sample dataset.

The functions here intentionally live under ``examples/`` because they are only
needed for optional comparison scripts. The main ``ffTRF`` package remains free
of external-data download logic.
"""

from __future__ import annotations

import hashlib
import os
import pickle
import shutil
import tempfile
from pathlib import Path
from typing import BinaryIO
from urllib.error import URLError
from urllib.request import urlopen

import numpy as np
from numpy.lib import format as npy_format

SAMPLE_DATA_COMMIT = "9b89449caaed3a4b7c80ea238a52c34a723cb8de"
SAMPLE_DATA_URL = (
    "https://raw.githubusercontent.com/powerfulbean/mTRFpy/"
    f"{SAMPLE_DATA_COMMIT}/tests/data/speech_data.npy"
)
SAMPLE_DATA_SHA256 = "5726060e254caac865c5ca7cf56a8218937f4c05b7784fb08d11658748daee36"
_DOWNLOAD_TIMEOUT_SECONDS = 60
_NUMPY_RECONSTRUCT = np.empty(0).__reduce__()[0]
_ALLOWED_PICKLE_GLOBALS = {
    ("numpy.core.multiarray", "_reconstruct"): _NUMPY_RECONSTRUCT,
    ("numpy._core.multiarray", "_reconstruct"): _NUMPY_RECONSTRUCT,
    ("numpy", "ndarray"): np.ndarray,
    ("numpy", "dtype"): np.dtype,
}


class _RestrictedNumpyUnpickler(pickle.Unpickler):
    """Decode only the NumPy constructors required by the pinned artifact."""

    def find_class(self, module: str, name: str) -> object:
        try:
            return _ALLOWED_PICKLE_GLOBALS[(module, name)]
        except KeyError as exc:
            raise pickle.UnpicklingError(
                f"Forbidden pickle global in sample dataset: {module}.{name}"
            ) from exc


def ensure_sample_data(
    cache_dir: str | Path = "artifacts/mtrf_data",
) -> Path:
    """Return a verified local path to the pinned speech EEG sample.

    Cached data are checked on every call. New data are downloaded to a
    temporary file and moved into place only after their SHA-256 digest matches
    the expected artifact.
    """

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    sample_path = cache_dir / "speech_data.npy"
    if sample_path.exists():
        _require_expected_sha256(sample_path)
        return sample_path

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=cache_dir,
            prefix=f".{sample_path.name}.",
            suffix=".part",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            _download_file(SAMPLE_DATA_URL, temporary_file)

        _require_expected_sha256(temporary_path)
        os.replace(temporary_path, sample_path)
    except (OSError, URLError) as exc:
        raise RuntimeError(
            "Unable to securely download the optional mTRF sample dataset. "
            f"Tried: {SAMPLE_DATA_URL}"
        ) from exc
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

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

    The upstream file uses NumPy's pickle-backed object representation. It is
    loaded only after :func:`ensure_sample_data` verifies the pinned SHA-256
    digest.
    """

    if (
        isinstance(n_segments, bool)
        or not isinstance(n_segments, (int, np.integer))
        or n_segments <= 0
    ):
        raise ValueError("n_segments must be a positive integer.")

    sample_path = ensure_sample_data(cache_dir)
    try:
        with sample_path.open("rb") as sample_file:
            _require_expected_sha256(sample_file)
            sample_file.seek(0)
            data = _decode_sample_file(sample_file)
    except (EOFError, OSError, ValueError, pickle.PickleError) as exc:
        raise RuntimeError("The verified mTRF sample dataset could not be decoded.") from exc

    stimulus_data, response_data, fs = _validate_sample_data(data)
    if n_segments > stimulus_data.shape[0]:
        raise ValueError("n_segments cannot exceed the number of samples in the dataset.")

    stimulus = np.array_split(stimulus_data, n_segments)
    response = np.array_split(response_data, n_segments)

    if normalize:
        for index, (x_trial, y_trial) in enumerate(zip(stimulus, response, strict=True)):
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


def _download_file(url: str, destination: BinaryIO) -> None:
    """Stream ``url`` into an already-open binary destination."""

    with urlopen(url, timeout=_DOWNLOAD_TIMEOUT_SECONDS) as response:
        shutil.copyfileobj(response, destination)


def _decode_sample_file(sample_file: BinaryIO) -> object:
    """Decode the pinned object-array NPY with a restricted unpickler."""

    version = npy_format.read_magic(sample_file)
    if version == (1, 0):
        shape, _, dtype = npy_format.read_array_header_1_0(sample_file)
    elif version == (2, 0):
        shape, _, dtype = npy_format.read_array_header_2_0(sample_file)
    else:
        raise ValueError(f"Unsupported NPY format version: {version}.")

    if shape != () or dtype != np.dtype(object):
        raise ValueError("Sample dataset must be a scalar NumPy object array.")

    array = _RestrictedNumpyUnpickler(sample_file).load()
    if not isinstance(array, np.ndarray) or array.shape != shape or array.dtype != dtype:
        raise ValueError("Sample dataset payload does not match its NPY header.")
    return array.item()


def _sha256(source: Path | BinaryIO) -> str:
    """Return the lowercase SHA-256 digest for a path or open binary file."""

    digest = hashlib.sha256()
    if isinstance(source, Path):
        file = source.open("rb")
        should_close = True
    else:
        file = source
        should_close = False

    try:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    finally:
        if should_close:
            file.close()
    return digest.hexdigest()


def _require_expected_sha256(source: Path | BinaryIO) -> None:
    """Raise when ``source`` is not the pinned upstream artifact."""

    actual_sha256 = _sha256(source)
    if actual_sha256 != SAMPLE_DATA_SHA256:
        source_description = str(source) if isinstance(source, Path) else "open sample file"
        raise RuntimeError(
            f"Sample-data integrity check failed for {source_description}. "
            f"Expected SHA-256 {SAMPLE_DATA_SHA256}, got {actual_sha256}. "
            "Remove this file and retry to fetch the pinned copy."
        )


def _validate_sample_data(
    data: object,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Validate and coerce the decoded upstream sample-data schema."""

    if not isinstance(data, dict):
        raise ValueError("Sample dataset must contain a dictionary.")

    required_keys = {"stimulus", "response", "samplerate"}
    missing_keys = required_keys.difference(data)
    if missing_keys:
        missing = ", ".join(sorted(missing_keys))
        raise ValueError(f"Sample dataset is missing required fields: {missing}.")

    try:
        stimulus = np.asarray(data["stimulus"], dtype=float)
        response = np.asarray(data["response"], dtype=float)
        samplerate = np.asarray(data["samplerate"], dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Sample dataset fields must contain numeric values.") from exc

    for name, array in (("stimulus", stimulus), ("response", response)):
        if array.ndim != 2 or 0 in array.shape:
            raise ValueError(f"Sample dataset field {name!r} must be a non-empty 2D array.")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"Sample dataset field {name!r} contains non-finite values.")

    if stimulus.shape[0] != response.shape[0]:
        raise ValueError(
            "Sample dataset stimulus and response must have the same number of samples."
        )
    if samplerate.size != 1:
        raise ValueError("Sample dataset samplerate must contain exactly one value.")

    samplerate_value = float(samplerate.reshape(-1)[0])
    if not np.isfinite(samplerate_value) or samplerate_value <= 0:
        raise ValueError("Sample dataset samplerate must be finite and positive.")
    if not samplerate_value.is_integer():
        raise ValueError("Sample dataset samplerate must be an integer.")

    return stimulus, response, int(samplerate_value)
