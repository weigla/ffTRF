from __future__ import annotations

import hashlib
import importlib.util
import io
from pathlib import Path
from types import ModuleType
from urllib.error import URLError

import numpy as np
import pytest

MODULE_PATH = Path(__file__).parents[1] / "examples" / "mtrf_sample_data.py"


@pytest.fixture
def sample_data_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("mtrf_sample_data_for_tests", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sample_payload(*, include_response: bool = True) -> bytes:
    data: dict[str, object] = {
        "stimulus": np.arange(24, dtype=float).reshape(8, 3),
        "samplerate": np.asarray([[128]]),
    }
    if include_response:
        data["response"] = np.arange(16, dtype=float).reshape(8, 2)

    buffer = io.BytesIO()
    np.save(buffer, data, allow_pickle=True)
    return buffer.getvalue()


def _set_expected_payload(
    module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    payload: bytes,
) -> None:
    monkeypatch.setattr(module, "SAMPLE_DATA_SHA256", hashlib.sha256(payload).hexdigest())


def test_load_sample_data_verifies_cached_file_before_decoding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sample_data_module: ModuleType,
) -> None:
    payload = _sample_payload()
    _set_expected_payload(sample_data_module, monkeypatch, payload)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "speech_data.npy").write_bytes(payload)

    def unexpected_download(url: str, destination: object) -> None:
        raise AssertionError(f"Unexpected download from {url} to {destination}")

    monkeypatch.setattr(sample_data_module, "_download_file", unexpected_download)
    stimulus, response, fs = sample_data_module.load_sample_data(
        cache_dir=cache_dir,
        n_segments=2,
        normalize=False,
    )

    assert fs == 128
    assert [trial.shape for trial in stimulus] == [(4, 3), (4, 3)]
    assert [trial.shape for trial in response] == [(4, 2), (4, 2)]


def test_corrupted_cache_is_rejected_before_decoding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sample_data_module: ModuleType,
) -> None:
    payload = _sample_payload()
    _set_expected_payload(sample_data_module, monkeypatch, payload)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "speech_data.npy").write_bytes(b"not the trusted sample")

    def unexpected_decode(sample_file: object) -> None:
        raise AssertionError(f"Decoder must not see unverified sample data: {sample_file}")

    monkeypatch.setattr(sample_data_module, "_decode_sample_file", unexpected_decode)
    with pytest.raises(RuntimeError, match="integrity check failed"):
        sample_data_module.load_sample_data(cache_dir=cache_dir)


def test_download_is_promoted_only_after_hash_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sample_data_module: ModuleType,
) -> None:
    payload = _sample_payload()
    _set_expected_payload(sample_data_module, monkeypatch, payload)
    cache_dir = tmp_path / "cache"

    def download(url: str, destination: object) -> None:
        assert url == sample_data_module.SAMPLE_DATA_URL
        destination.write(payload)

    monkeypatch.setattr(sample_data_module, "_download_file", download)
    sample_path = sample_data_module.ensure_sample_data(cache_dir)

    assert sample_path.read_bytes() == payload
    assert not list(cache_dir.glob("*.part"))
    assert not list(cache_dir.glob(".*.part"))


def test_interrupted_download_leaves_no_cached_or_partial_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sample_data_module: ModuleType,
) -> None:
    cache_dir = tmp_path / "cache"

    def interrupted_download(url: str, destination: object) -> None:
        destination.write(b"partial")
        raise URLError(f"interrupted download from {url}")

    monkeypatch.setattr(sample_data_module, "_download_file", interrupted_download)
    with pytest.raises(RuntimeError, match="securely download"):
        sample_data_module.ensure_sample_data(cache_dir)

    assert not (cache_dir / "speech_data.npy").exists()
    assert not list(cache_dir.iterdir())


def test_download_with_wrong_hash_is_never_cached(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sample_data_module: ModuleType,
) -> None:
    trusted_payload = _sample_payload()
    _set_expected_payload(sample_data_module, monkeypatch, trusted_payload)
    cache_dir = tmp_path / "cache"

    def tampered_download(url: str, destination: object) -> None:
        destination.write(b"tampered")

    monkeypatch.setattr(sample_data_module, "_download_file", tampered_download)
    with pytest.raises(RuntimeError, match="integrity check failed"):
        sample_data_module.ensure_sample_data(cache_dir)

    assert not (cache_dir / "speech_data.npy").exists()
    assert not list(cache_dir.iterdir())


def test_verified_data_must_match_expected_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sample_data_module: ModuleType,
) -> None:
    payload = _sample_payload(include_response=False)
    _set_expected_payload(sample_data_module, monkeypatch, payload)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "speech_data.npy").write_bytes(payload)

    with pytest.raises(ValueError, match="missing required fields: response"):
        sample_data_module.load_sample_data(cache_dir=cache_dir)


def test_restricted_decoder_rejects_unexpected_pickle_globals(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sample_data_module: ModuleType,
) -> None:
    buffer = io.BytesIO()
    np.save(buffer, {"unexpected": Path("not-allowed")}, allow_pickle=True)
    payload = buffer.getvalue()
    _set_expected_payload(sample_data_module, monkeypatch, payload)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "speech_data.npy").write_bytes(payload)

    with pytest.raises(RuntimeError, match="could not be decoded") as exc_info:
        sample_data_module.load_sample_data(cache_dir=cache_dir)

    assert "Forbidden pickle global" in str(exc_info.value.__cause__)
