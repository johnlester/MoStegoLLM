"""Cross-compatibility test vectors: encode here, decode (and verify) elsewhere.

A TestVector is the only artifact that crosses an environment boundary. It carries
the cover text plus everything a *different* PyTorch/device/dtype needs to decode
it and check the result. See
docs/superpowers/specs/2026-06-03-cloud-cross-compatibility-testing-design.md
"""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import asdict, dataclass
from enum import Enum
from os import PathLike

import torch

from .crypto import decrypt as _decrypt
from .crypto import encrypt as _encrypt
from .decoder import decode as _decode
from .encoder import (
    TOP_K,
    _filter_tokens,
    _get_topk_logits,
    encode as _encode,
    get_non_roundtrip_tokens,
)
from .utils import StegoCryptoError, StegoDecodeError

SCHEMA = "mostegollm-testvector/1"

_DEFAULT_SETTINGS = {"top_k": 256, "temperature": 1.0, "sentence_boundary": False}


@dataclass(frozen=True)
class TestVector:
    """A portable record of one encode result plus everything needed to decode it.

    Attributes:
        schema: Schema identifier for forward-compatibility (``SCHEMA``).
        library_version: Version of ``mostegollm`` that produced the vector.
        model: Hugging Face model id used for encoding.
        model_revision: Resolved model commit hash, or ``None`` if unknown.
        prompt: Prompt prepended before generation.
        settings: Coding settings (``top_k``, ``temperature``, ``sentence_boundary``,
            ``encrypted``). The password is never stored.
        payload_sha256: Hex SHA-256 of the *plaintext* payload (pre-encryption).
        payload_hex: Hex encoding of the plaintext payload.
        cover_text: Generated cover text carrying the payload.
        generated_token_ids: Token ids the encoder produced for ``cover_text``.
        source_env: Environment metadata of the producing machine.
    """

    # Tell pytest this is not a test class (name starts with "Test").
    __test__ = False

    schema: str
    library_version: str
    model: str
    model_revision: str | None
    prompt: str
    settings: dict
    payload_sha256: str
    payload_hex: str
    cover_text: str
    generated_token_ids: list[int]
    source_env: dict

    def to_json(self) -> str:
        """Serialize this vector to a single-line JSON string (UTF-8, non-escaped)."""
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, line: str) -> "TestVector":
        """Reconstruct a :class:`TestVector` from a JSON string produced by ``to_json``."""
        return cls(**json.loads(line))


def write_vectors(vectors: list[TestVector], path: str | PathLike) -> None:
    """Write *vectors* to *path* as JSONL (one vector per line)."""
    with open(path, "w", encoding="utf-8") as f:
        for v in vectors:
            f.write(v.to_json() + "\n")


def read_vectors(path: str | PathLike) -> list[TestVector]:
    """Read a JSONL file written by :func:`write_vectors` into a list of vectors."""
    out: list[TestVector] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(TestVector.from_json(line))
    return out


def current_env(device: object) -> dict[str, str]:
    """Capture environment metadata relevant to cross-platform decode reproducibility.

    Args:
        device: The torch device (or its string form) in use.

    Returns:
        A dict of library versions plus device/OS/architecture, all as strings.
    """
    import tokenizers
    import transformers

    return {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "tokenizers": tokenizers.__version__,
        "device": str(device),
        "os": platform.system(),
        "arch": platform.machine(),
    }


def lib_version() -> str:
    """Return the installed ``mostegollm`` version, or ``"unknown"`` if undeterminable."""
    try:
        from importlib.metadata import version

        return version("mostegollm")
    except Exception:
        return "unknown"


def model_commit_hash(model: object) -> str | None:
    """Return the model's resolved HF commit hash, or ``None`` if not recorded."""
    return getattr(getattr(model, "config", None), "_commit_hash", None)


class FailureClass(str, Enum):
    """Classification of why a :class:`TestVector` failed to verify elsewhere."""

    RETOK_DRIFT = "retokenization_drift"
    LOGIT_DIVERGENCE = "logit_divergence"
    LOAD_ERROR = "load_error"


@dataclass(frozen=True)
class VerifyResult:
    """Outcome of verifying a :class:`TestVector` against the local environment.

    Attributes:
        ok: ``True`` iff the payload was recovered and its hash matched.
        failure_class: The category of failure, or ``None`` on success.
        detail: Human-readable explanation of the outcome.
        decoder_env: Environment metadata captured on the verifying machine.
    """

    ok: bool
    failure_class: FailureClass | None
    detail: str
    decoder_env: dict


def make_vector(
    payload: bytes,
    *,
    model: object,
    tokenizer: object,
    device: object,
    prompt: str,
    model_name: str,
    model_revision: str | None = None,
    settings: dict | None = None,
    password: str | None = None,
) -> TestVector:
    """Encode *payload* and package the result as a portable :class:`TestVector`.

    Args:
        payload: The plaintext bytes to hide.
        model: Loaded language model.
        tokenizer: Matching tokenizer.
        device: Torch device the model lives on.
        prompt: Prompt to prepend before generation.
        model_name: Hugging Face model id to record in the vector.
        model_revision: Explicit model commit hash; falls back to
            :func:`model_commit_hash` when ``None``.
        settings: Optional overrides merged over ``_DEFAULT_SETTINGS``.
        password: Optional password; when set, the payload is AES-256-GCM encrypted
            before encoding (but ``payload_sha256``/``payload_hex`` record the plaintext).

    Returns:
        A fully populated :class:`TestVector`.
    """
    s = {**_DEFAULT_SETTINGS, **(settings or {})}
    to_encode = _encrypt(payload, password) if password else payload
    cover, ids, _bits = _encode(
        to_encode,
        model=model,
        tokenizer=tokenizer,
        device=device,
        prompt=prompt,
        top_k=s["top_k"],
        temperature=s["temperature"],
        sentence_boundary=s["sentence_boundary"],
    )
    return TestVector(
        schema=SCHEMA,
        library_version=lib_version(),
        model=model_name,
        model_revision=model_revision if model_revision is not None else model_commit_hash(model),
        prompt=prompt,
        # SECURITY: never persist the password in a portable/committed artifact.
        # Record only a non-secret marker; the verifier supplies it out-of-band.
        settings={**s, "encrypted": bool(password)},
        payload_sha256=hashlib.sha256(payload).hexdigest(),
        payload_hex=payload.hex(),
        cover_text=cover,
        generated_token_ids=list(ids),
        source_env=current_env(device),
    )


def verify_vector(
    vector: TestVector,
    *,
    model: object,
    tokenizer: object,
    device: object,
    password: str | None = None,
) -> VerifyResult:
    """Re-decode *vector* locally and classify any divergence from the producer.

    The check has two stages. First, the cover text is re-tokenized; a mismatch
    against ``generated_token_ids`` is a deterministic tokenizer disagreement
    (:attr:`FailureClass.RETOK_DRIFT`). Otherwise the payload is decoded (and
    decrypted if the vector is encrypted); any decode/crypto failure or a payload
    hash mismatch is attributed to logit divergence between environments.

    For encrypted vectors (``settings["encrypted"]`` is True) the caller must
    supply *password* out-of-band — it is deliberately not stored in the vector.

    Args:
        vector: The vector to verify.
        model: Loaded language model on the verifying machine.
        tokenizer: Matching tokenizer.
        device: Torch device the model lives on.
        password: Decryption password for encrypted vectors, supplied out-of-band.

    Returns:
        A :class:`VerifyResult` describing success or the failure category.

    Raises:
        ValueError: If the vector's schema is unrecognized, or if it is
            encrypted but no *password* is supplied.
    """
    env = current_env(device)
    if vector.schema != SCHEMA:
        raise ValueError(f"unsupported test-vector schema {vector.schema!r} (expected {SCHEMA!r})")
    retok = tokenizer.encode(vector.cover_text, add_special_tokens=False)
    if retok != vector.generated_token_ids:
        return VerifyResult(
            False,
            FailureClass.RETOK_DRIFT,
            f"re-tokenization mismatch: {len(retok)} vs {len(vector.generated_token_ids)} tokens",
            env,
        )
    encrypted = vector.settings.get("encrypted", False)
    if encrypted and password is None:
        raise ValueError("vector is encrypted; supply password= to verify_vector")
    try:
        decoded = _decode(
            vector.cover_text,
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompt=vector.prompt,
            # Reuse the ids we just validated equal generated_token_ids, so decode
            # uses the encoder's canonical token stream instead of tokenizing a
            # third time (decoder.decode exposes token_ids for exactly this).
            token_ids=vector.generated_token_ids,
            top_k=vector.settings.get("top_k", _DEFAULT_SETTINGS["top_k"]),
            temperature=vector.settings.get("temperature", _DEFAULT_SETTINGS["temperature"]),
        )
        if encrypted:
            decoded = _decrypt(decoded, password)
    except (StegoDecodeError, StegoCryptoError) as exc:
        return VerifyResult(
            False,
            FailureClass.LOGIT_DIVERGENCE,
            f"decode failed though re-tokenization matched: {exc}",
            env,
        )
    if hashlib.sha256(decoded).hexdigest() != vector.payload_sha256:
        return VerifyResult(
            False,
            FailureClass.LOGIT_DIVERGENCE,
            "decoded payload hash mismatch (silent desync)",
            env,
        )
    return VerifyResult(True, None, "ok", env)


def dump_step_logits(
    prompt: str,
    token_ids: list[int],
    *,
    model: object,
    tokenizer: object,
    device: object,
    top_k: int = TOP_K,
    temperature: float = 1.0,
) -> list[dict]:
    """Replay *token_ids* and record the filtered top-k distribution at each step.

    This is the white-box input for cross-environment comparison: two machines
    that produce the same per-step filtered top-k *ordering* build identical
    rank intervals and therefore decode identically. Mirrors the decoder's
    per-step loop exactly (same prompt handling, top-k, and BPE filter), but
    records ``(top_ids, top_logits)`` instead of decoding bits.

    Args:
        prompt: The seed prompt (must match how the sequence was produced).
        token_ids: The token sequence to replay (e.g. a vector's
            ``generated_token_ids``).
        model: Loaded language model.
        tokenizer: Matching tokenizer.
        device: Torch device.
        top_k: Top-k width (must match encoding).
        temperature: Logit scaling (must match encoding).

    Returns:
        One dict per step with ``top_ids`` (list[int], logit-descending) and
        ``top_logits`` (list[float]), already BPE-filtered.
    """
    prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    if prompt_ids.numel() == 0:
        bos = tokenizer.bos_token_id or 0
        prompt_ids = torch.tensor([[bos]], device=device)
    next_input = prompt_ids.clone()
    past_kv = None

    non_rt_tokens = get_non_roundtrip_tokens(tokenizer)
    merge_cache: dict[tuple[int, int], bool] = {}
    prev_token_id: int | None = None

    steps: list[dict] = []
    for token_id in token_ids:
        tok_ids, logits, past_kv = _get_topk_logits(
            model, next_input, device, top_k=top_k, temperature=temperature, past_key_values=past_kv
        )
        tok_ids, logits = _filter_tokens(
            tokenizer, prev_token_id, tok_ids, logits, non_rt_tokens, merge_cache
        )
        steps.append({"top_ids": list(tok_ids), "top_logits": [float(x) for x in logits]})
        next_input = torch.tensor([[token_id]], device=device)
        prev_token_id = token_id
    return steps
