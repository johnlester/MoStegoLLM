"""The non-float32 dtype warning and the config-mismatch error hint."""

from __future__ import annotations

import types

import pytest
import torch

from mostegollm import StegoDecodeError
from mostegollm.encoder import warn_if_non_canonical_dtype
from mostegollm.utils import CONFIG_MISMATCH_HINT, MAGIC, unpack_header


def test_warns_on_fp16():
    model = types.SimpleNamespace(dtype=torch.float16)
    with pytest.warns(UserWarning, match="float32"):
        warn_if_non_canonical_dtype(model)


def test_warns_on_bf16():
    model = types.SimpleNamespace(dtype=torch.bfloat16)
    with pytest.warns(UserWarning, match="portable"):
        warn_if_non_canonical_dtype(model)


def test_silent_on_fp32(recwarn):
    warn_if_non_canonical_dtype(types.SimpleNamespace(dtype=torch.float32))
    assert len(recwarn) == 0


def test_silent_when_dtype_unknown(recwarn):
    warn_if_non_canonical_dtype(types.SimpleNamespace())  # no .dtype attribute
    assert len(recwarn) == 0


def test_bad_magic_error_includes_config_hint():
    # Wrong magic, correct 10-byte length -> StegoDecodeError carrying the hint.
    bad_header = b"\x00\x00" + b"\x00" * 8
    assert bad_header[:2] != MAGIC
    with pytest.raises(StegoDecodeError) as exc:
        unpack_header(bad_header)
    assert CONFIG_MISMATCH_HINT in str(exc.value)
