"""Tests for the cross-compatibility test-vector layer."""

from __future__ import annotations

from pathlib import Path

import dataclasses

import pytest

from mostegollm.compat import (
    FailureClass,
    TestVector,
    current_env,
    lib_version,
    make_vector,
    model_commit_hash,
    read_vectors,
    verify_vector,
    write_vectors,
)


def _sample_vector() -> TestVector:
    return TestVector(
        schema="mostegollm-testvector/1",
        library_version="0.3.0",
        model="HuggingFaceTB/SmolLM-135M",
        model_revision="abc123",
        prompt="According to experts,",
        settings={"top_k": 256, "temperature": 1.0, "sentence_boundary": False, "encrypted": False},
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


@pytest.mark.parametrize(
    "payload", [b"hello", b"\x42", b"\x00\x01\x02\x03\xfe\xff", b"The quick brown fox."]
)
def test_make_then_verify_round_trips(codec, payload):
    model, tok, dev = codec._ensure_model()
    vector = make_vector(
        payload,
        model=model,
        tokenizer=tok,
        device=dev,
        prompt="According to experts,",
        model_name=codec._model_name,
    )
    assert tok.encode(vector.cover_text, add_special_tokens=False) == vector.generated_token_ids
    result = verify_vector(vector, model=model, tokenizer=tok, device=dev)
    assert result.ok, result.detail
    assert result.failure_class is None


def test_verify_detects_retokenization_drift(codec):
    model, tok, dev = codec._ensure_model()
    good = make_vector(
        b"hello",
        model=model,
        tokenizer=tok,
        device=dev,
        prompt="According to experts,",
        model_name=codec._model_name,
    )
    bad = dataclasses.replace(good, generated_token_ids=good.generated_token_ids + [999999])
    result = verify_vector(bad, model=model, tokenizer=tok, device=dev)
    assert not result.ok
    assert result.failure_class == FailureClass.RETOK_DRIFT


def test_verify_detects_payload_mismatch(codec):
    model, tok, dev = codec._ensure_model()
    good = make_vector(
        b"hello",
        model=model,
        tokenizer=tok,
        device=dev,
        prompt="According to experts,",
        model_name=codec._model_name,
    )
    bad = dataclasses.replace(good, payload_sha256="de" * 32)
    result = verify_vector(bad, model=model, tokenizer=tok, device=dev)
    assert not result.ok
    assert result.failure_class == FailureClass.LOGIT_DIVERGENCE


def test_encrypted_vector_round_trips(codec):
    model, tok, dev = codec._ensure_model()
    vector = make_vector(
        b"secret payload",
        model=model,
        tokenizer=tok,
        device=dev,
        prompt="According to experts,",
        model_name=codec._model_name,
        password="pw-123",
    )
    # SECURITY: the password is NOT persisted; only a non-secret marker is.
    assert vector.settings["encrypted"] is True
    assert "password" not in vector.settings
    import hashlib as _h

    assert vector.payload_sha256 == _h.sha256(b"secret payload").hexdigest()
    result = verify_vector(vector, model=model, tokenizer=tok, device=dev, password="pw-123")
    assert result.ok, result.detail


def test_verify_encrypted_without_password_raises(codec):
    model, tok, dev = codec._ensure_model()
    vector = make_vector(
        b"secret payload",
        model=model,
        tokenizer=tok,
        device=dev,
        prompt="According to experts,",
        model_name=codec._model_name,
        password="pw-123",
    )
    with pytest.raises(ValueError, match="encrypted"):
        verify_vector(vector, model=model, tokenizer=tok, device=dev)


def test_verify_rejects_unknown_schema(codec):
    model, tok, dev = codec._ensure_model()
    good = make_vector(
        b"hello",
        model=model,
        tokenizer=tok,
        device=dev,
        prompt="According to experts,",
        model_name=codec._model_name,
    )
    bad = dataclasses.replace(good, schema="mostegollm-testvector/999")
    with pytest.raises(ValueError, match="schema"):
        verify_vector(bad, model=model, tokenizer=tok, device=dev)


def test_dump_step_logits_matches_sequence(codec):
    from mostegollm.compat import dump_step_logits

    model, tok, dev = codec._ensure_model()
    vector = make_vector(
        b"hello world",
        model=model,
        tokenizer=tok,
        device=dev,
        prompt="According to experts,",
        model_name=codec._model_name,
    )
    steps = dump_step_logits(
        "According to experts,", vector.generated_token_ids, model=model, tokenizer=tok, device=dev
    )
    # One record per replayed token.
    assert len(steps) == len(vector.generated_token_ids)
    for i, step in enumerate(steps):
        ids, logits = step["top_ids"], step["top_logits"]
        assert len(ids) == len(logits) and ids  # paired and non-empty
        # top-k is logit-descending.
        assert logits == sorted(logits, reverse=True)
        # The token actually generated at this step survives the filter and is present.
        assert vector.generated_token_ids[i] in ids


def test_canonical_payloads_are_deterministic():
    import importlib.util
    from pathlib import Path

    spec_path = Path(__file__).parent.parent / "compat" / "payloads.py"
    spec = importlib.util.spec_from_file_location("compat_payloads", spec_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    mod2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod2)
    assert mod.CANONICAL_PAYLOADS == mod2.CANONICAL_PAYLOADS
    assert all(isinstance(v, bytes) for v in mod.CANONICAL_PAYLOADS.values())
    assert len(mod.CANONICAL_PROMPTS) >= 2
