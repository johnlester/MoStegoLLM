# tests/test_cross_platform.py
"""Cross-platform reproducibility: encode here, decode under a different
floating-point regime. These are the empirical proof of the rank-interval
design (they fail on the old magnitude-based coder)."""

from __future__ import annotations

import copy

import pytest
import torch

from mostegollm.decoder import decode
from mostegollm.encoder import encode

PROMPT = "According to experts,"
PAYLOADS = [
    b"cross-platform compatibility test",
    b"hello",
    b"\x00\x01\x02\x03\xfe\xff",
    b"The quick brown fox jumps over the lazy dog.",
    b"a",
]


@pytest.mark.parametrize("payload", PAYLOADS)
def test_float32_to_float64_roundtrip(codec, payload):
    """Encoding with float32 matmul must decode with a float64 model copy."""
    model, tok, dev = codec._ensure_model()
    cover, ids, _ = encode(payload, model=model, tokenizer=tok, device=dev, prompt=PROMPT)
    model64 = copy.deepcopy(model).double()
    recovered = decode(
        cover, model=model64, tokenizer=tok, device=dev, prompt=PROMPT, token_ids=ids
    )
    assert recovered == payload


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize("payload", PAYLOADS)
def test_cpu_encode_gpu_decode_roundtrip(codec, payload):
    """Encoding on CPU must decode on GPU (the caveat scenario)."""
    model, tok, _ = codec._ensure_model()  # codec fixture is CPU
    cover, ids, _ = encode(
        payload, model=model, tokenizer=tok, device=torch.device("cpu"), prompt=PROMPT
    )
    gpu = torch.device("cuda")
    model_gpu = copy.deepcopy(model).to(gpu)
    recovered = decode(
        cover, model=model_gpu, tokenizer=tok, device=gpu, prompt=PROMPT, token_ids=ids
    )
    assert recovered == payload
