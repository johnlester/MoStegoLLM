"""Decoding logic: reconstruct bits from cover text by replaying LLM distributions.

The stego decoder is conceptually an arithmetic *encoder* — it observes which
token was chosen at each step, looks up that token's probability interval, and
emits the corresponding bits.
"""

from __future__ import annotations

import zlib

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .coding import HALF, QUARTER, WHOLE, step_coding
from .encoder import (
    TOP_K,
    _filter_tokens,
    _get_topk_logits,
    get_non_roundtrip_tokens,
)
from .utils import (
    HEADER_BITS,
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

    # Tokenize the prompt (same as encoder — ensure at least the BOS token)
    prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    if prompt_ids.numel() == 0:
        bos = tokenizer.bos_token_id or 0
        prompt_ids = torch.tensor([[bos]], device=device)
    next_input = prompt_ids.clone()
    past_kv = None

    # Pre-compute BPE filter state (must match the encoder exactly).
    non_rt_tokens = get_non_roundtrip_tokens(tokenizer)
    merge_cache: dict[tuple[int, int], bool] = {}
    prev_token_id: int | None = None

    # --- Arithmetic coding state (encoder/compressor side) ---
    low = 0
    high = WHOLE
    pending = 0
    extracted_bits: list[int] = []

    # Process each token in the cover text
    for step_idx, token_id in enumerate(cover_token_ids):
        # Same reproducible coding the encoder used.
        tok_ids, logits, past_kv = _get_topk_logits(
            model,
            next_input,
            device,
            top_k=top_k,
            temperature=temperature,
            past_key_values=past_kv,
        )
        tok_ids, logits = _filter_tokens(
            tokenizer, prev_token_id, tok_ids, logits, non_rt_tokens, merge_cache
        )
        step = step_coding(tok_ids, logits)

        try:
            ilo, ihi = step.token_to_run[token_id]
        except KeyError:
            raise StegoDecodeError(
                f"Token ID {token_id} ('{tokenizer.decode([token_id])}') at step {step_idx} "
                f"was not found in the reproducible top-{top_k} distribution. "
                "This usually means the cover text was corrupted, the wrong model "
                "was used, or the prompt does not match."
            )

        range_size = high - low
        sym_low = low + (range_size * ilo) // WHOLE
        sym_high = low + (range_size * ihi) // WHOLE
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
        next_input = torch.tensor([[token_id]], device=device)
        prev_token_id = token_id

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
    payload_length, expected_crc = unpack_header(header_bytes)

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
    if len(payload) != payload_length:
        raise StegoDecodeError(
            f"Internal error: expected {payload_length} bytes, got {len(payload)}"
        )

    # Verify CRC-32 integrity
    actual_crc = zlib.crc32(payload) & 0xFFFFFFFF
    if actual_crc != expected_crc:
        raise StegoDecodeError(
            "Payload integrity check failed -- data may be corrupted, "
            "the wrong model was used, or the cover text was modified."
        )

    return payload
