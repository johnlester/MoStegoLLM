"""Tests for the cross-compatibility test-vector layer."""

from __future__ import annotations

from pathlib import Path

from mostegollm.compat import (
    TestVector,
    current_env,
    lib_version,
    model_commit_hash,
    read_vectors,
    write_vectors,
)


def _sample_vector() -> TestVector:
    return TestVector(
        schema="mostegollm-testvector/1",
        library_version="0.3.0",
        model="HuggingFaceTB/SmolLM-135M",
        model_revision="abc123",
        prompt="According to experts,",
        settings={"top_k": 256, "temperature": 1.0, "sentence_boundary": False, "password": None},
        payload_sha256="00" * 32,
        payload_hex="68656c6c6f",
        cover_text="Some cover text with unicode café 🙂.",
        generated_token_ids=[1, 2, 3, 4],
        source_env={"torch": "2.5.1", "device": "cpu", "os": "Linux", "arch": "x86_64"},
    )


def test_testvector_json_round_trip():
    v = _sample_vector()
    restored = TestVector.from_json(v.to_json())
    assert restored == v


def test_write_read_vectors_round_trip(tmp_path: Path):
    vectors = [_sample_vector(), _sample_vector()]
    path = tmp_path / "vectors.jsonl"
    write_vectors(vectors, path)
    loaded = read_vectors(path)
    assert loaded == vectors
    assert len(path.read_text(encoding="utf-8").splitlines()) == 2


def test_current_env_has_required_keys():
    env = current_env("cpu")
    for key in ("torch", "transformers", "tokenizers", "device", "os", "arch"):
        assert key in env and isinstance(env[key], str)
    assert env["device"] == "cpu"


def test_lib_version_is_string():
    v = lib_version()
    assert isinstance(v, str) and v


def test_model_commit_hash_handles_missing(codec):
    model, _tok, _dev = codec._ensure_model()
    result = model_commit_hash(model)
    assert result is None or isinstance(result, str)
