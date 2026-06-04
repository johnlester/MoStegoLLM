"""Cross-compatibility test vectors: encode here, decode (and verify) elsewhere.

A TestVector is the only artifact that crosses an environment boundary. It carries
the cover text plus everything a *different* PyTorch/device/dtype needs to decode
it and check the result. See
docs/superpowers/specs/2026-06-03-cloud-cross-compatibility-testing-design.md
"""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass
from os import PathLike

import torch

SCHEMA = "mostegollm-testvector/1"


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
            ``password``).
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
