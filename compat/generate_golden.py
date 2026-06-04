"""Generate the committed reference golden corpus.

Run from the repo root:  python -m compat.generate_golden
The reference environment is CPU + the default model (SmolLM-135M), matching the
session test fixture so tests/test_golden_vectors.py decodes it deterministically.
"""

from __future__ import annotations

from pathlib import Path

from mostegollm import StegoCodec
from mostegollm.compat import make_vector, model_commit_hash, write_vectors

from compat.payloads import CANONICAL_PAYLOADS, CANONICAL_PROMPTS

OUT = Path(__file__).parent / "golden_vectors.jsonl"


def main() -> None:
    codec = StegoCodec(device="cpu")
    model, tok, dev = codec._ensure_model()
    revision = model_commit_hash(model)
    vectors = [
        make_vector(
            payload,
            model=model,
            tokenizer=tok,
            device=dev,
            prompt=prompt,
            model_name=codec._model_name,
            model_revision=revision,
        )
        for prompt in CANONICAL_PROMPTS
        for payload in CANONICAL_PAYLOADS.values()
    ]
    write_vectors(vectors, OUT)
    print(f"wrote {len(vectors)} vectors to {OUT}")


if __name__ == "__main__":
    main()
