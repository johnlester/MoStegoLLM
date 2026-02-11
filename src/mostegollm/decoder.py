"""Decoding logic: reconstruct bits from cover text by replaying LLM distributions.

The stego decoder is conceptually an arithmetic *encoder* — it observes which
token was chosen at each step, looks up that token's probability interval, and
emits the corresponding bits.
"""

from __future__ import annotations

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .encoder import HALF, PRECISION, QUARTER, TOP_K, WHOLE, _get_token_distribution
from .utils import (
    HEADER_BITS,
    HEADER_SIZE,
    StegoDecodeError,
    bits_to_bytes,
    unpack_header,
)


def decode(
    cover_text: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    device: torch.device,
    prompt: str,
    top_k: int = TOP_K,
    temperature: float = 1.0,
    token_ids: list[int] | None = None,
) -> bytes:
    """Decode cover text back to the original binary payload.

    Reconstructs the same probability distributions that the encoder used and
    reads off the bits that each token choice encodes.

    Args:
        cover_text: The steganographic cover text (without the prompt).
        model: The language model (must be the same as used for encoding).
        tokenizer: The tokenizer.
        device: Torch device.
        prompt: The seed prompt (must match what was used for encoding).
        top_k: Number of top tokens (must match encoding).
        temperature: Softmax temperature (must match encoding).
        token_ids: Optional pre-computed token IDs. If ``None``, the cover
            text is re-tokenized.  Passing token IDs directly avoids any
            risk of tokenizer round-trip mismatches.

    Returns:
        The original binary payload (without the header).

    Raises:
        StegoDecodeError: If decoding fails (bad header, token mismatch, etc.).
    """
    # Resolve token IDs for the cover text
    if token_ids is None:
        cover_token_ids = tokenizer.encode(cover_text, add_special_tokens=False)
        if not cover_token_ids:
            raise StegoDecodeError("Cover text produced no tokens when re-tokenized")
    else:
        cover_token_ids = list(token_ids)

    if not cover_token_ids:
        raise StegoDecodeError("No tokens to decode")

    # Tokenize the prompt (same as encoder)
    prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    input_ids = prompt_ids.clone()

    # --- Arithmetic coding state (encoder/compressor side) ---
    low = 0
    high = WHOLE
    pending = 0
    extracted_bits: list[int] = []

    # Process each token in the cover text
    for step, token_id in enumerate(cover_token_ids):
        # Get the same distribution the encoder saw at this position
        tok_ids, cum_probs = _get_token_distribution(
            model, input_ids, device, top_k=top_k, temperature=temperature
        )

        # Find the index of this token in the distribution
        try:
            j = tok_ids.index(token_id)
        except ValueError:
            raise StegoDecodeError(
                f"Token ID {token_id} ('{tokenizer.decode([token_id])}') at step {step} "
                f"was not found in the top-{top_k} distribution. "
                "This usually means the cover text was corrupted, the wrong model "
                "was used, or the prompt does not match."
            )

        # Narrow the interval (must match the encoder exactly)
        range_size = high - low
        sym_low = low + (range_size * cum_probs[j]) // WHOLE
        sym_high = low + (range_size * cum_probs[j + 1]) // WHOLE
        low = sym_low
        high = sym_high

        # Renormalize and emit bits (arithmetic encoder side)
        while True:
            if high <= HALF:
                # Emit 0, then `pending` 1-bits
                extracted_bits.append(0)
                extracted_bits.extend([1] * pending)
                pending = 0
            elif low >= HALF:
                # Emit 1, then `pending` 0-bits
                extracted_bits.append(1)
                extracted_bits.extend([0] * pending)
                pending = 0
                low -= HALF
                high -= HALF
            elif low >= QUARTER and high <= 3 * QUARTER:
                pending += 1
                low -= QUARTER
                high -= QUARTER
            else:
                break

            low <<= 1
            high <<= 1

        # Advance the context (same as encoder)
        new_token = torch.tensor([[token_id]], device=device)
        input_ids = torch.cat([input_ids, new_token], dim=1)

    # Flush remaining state: emit one more disambiguating bit plus pending
    pending += 1
    if low < QUARTER:
        extracted_bits.append(0)
        extracted_bits.extend([1] * pending)
    else:
        extracted_bits.append(1)
        extracted_bits.extend([0] * pending)

    # We now have enough bits to recover the header + payload.
    if len(extracted_bits) < HEADER_BITS:
        raise StegoDecodeError(
            f"Extracted only {len(extracted_bits)} bits, need at least "
            f"{HEADER_BITS} for the header. The cover text may be too short."
        )

    # Parse header
    header_bytes = bits_to_bytes(extracted_bits[:HEADER_BITS])
    payload_length = unpack_header(header_bytes)

    # Extract payload
    payload_bits_needed = payload_length * 8
    total_bits_needed = HEADER_BITS + payload_bits_needed

    if len(extracted_bits) < total_bits_needed:
        raise StegoDecodeError(
            f"Extracted {len(extracted_bits)} bits but need {total_bits_needed} "
            f"(header says payload is {payload_length} bytes). "
            "The cover text may be truncated."
        )

    payload_bits = extracted_bits[HEADER_BITS:total_bits_needed]
    payload = bits_to_bytes(payload_bits)

    # Sanity check: the payload should be exactly payload_length bytes
    assert len(payload) == payload_length, (
        f"Internal error: expected {payload_length} bytes, got {len(payload)}"
    )

    return payload
