"""Encoding logic: consume bits from secret data to select tokens via arithmetic coding.

The stego encoder is conceptually an arithmetic *decoder* — it reads bits from
the secret message and uses them to navigate the probability intervals to choose
which token to emit at each step.
"""

from __future__ import annotations

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .utils import (
    StegoEncodeError,
    bytes_to_bits,
    pack_header,
)

# Arithmetic coding precision (number of bits for the interval)
PRECISION = 32
WHOLE = 1 << PRECISION  # 2^32
HALF = WHOLE >> 1  # 2^31
QUARTER = WHOLE >> 2  # 2^30

# Maximum tokens to generate as a safety limit
MAX_TOKENS = 8192

# Top-k tokens to consider from the distribution
TOP_K = 256


def _get_token_distribution(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    device: torch.device,
    top_k: int = TOP_K,
    temperature: float = 1.0,
) -> tuple[list[int], list[int]]:
    """Get the top-k token distribution from the model.

    Returns token IDs and their corresponding integer-scaled cumulative
    probabilities suitable for arithmetic coding.

    Args:
        model: The language model.
        input_ids: Current token sequence (batch_size=1).
        device: Torch device.
        top_k: Number of top tokens to consider.
        temperature: Temperature for softmax (1.0 = no change).

    Returns:
        A tuple of (token_ids, cum_probs) where:
        - token_ids: list of top-k token IDs sorted by probability (descending)
        - cum_probs: list of cumulative probability boundaries scaled to [0, WHOLE),
          with length len(token_ids) + 1 (starts at 0, ends at WHOLE).
    """
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[0, -1, :]  # Last position logits

    # Apply temperature
    if temperature != 1.0:
        logits = logits / temperature

    # Use float64 for precision in probability computation
    logits_f64 = logits.to(dtype=torch.float64)
    probs = torch.softmax(logits_f64, dim=-1)

    # Get top-k
    actual_k = min(top_k, probs.shape[0])
    top_probs, top_indices = torch.topk(probs, actual_k)

    # Renormalize top-k probabilities to sum to 1
    top_probs = top_probs / top_probs.sum()

    # Convert to integer-scaled cumulative distribution for arithmetic coding
    token_ids = top_indices.cpu().tolist()
    prob_values = top_probs.cpu().tolist()

    # Build cumulative distribution scaled to [0, WHOLE)
    cum_probs = [0]
    running = 0
    for p in prob_values:
        running += int(p * WHOLE)
        cum_probs.append(running)

    # Fix rounding: ensure the last entry equals WHOLE exactly
    cum_probs[-1] = WHOLE

    # Ensure no zero-width intervals (each token must have at least width 1)
    for i in range(1, len(cum_probs)):
        if cum_probs[i] <= cum_probs[i - 1]:
            cum_probs[i] = cum_probs[i - 1] + 1

    # If adjustments pushed past WHOLE, truncate the token list
    if cum_probs[-1] > WHOLE:
        for i in range(1, len(cum_probs)):
            if cum_probs[i] >= WHOLE:
                cum_probs[i] = WHOLE
                token_ids = token_ids[:i]
                cum_probs = cum_probs[: i + 1]
                break

    return token_ids, cum_probs


def encode(
    data: bytes,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    device: torch.device,
    prompt: str,
    top_k: int = TOP_K,
    temperature: float = 1.0,
) -> tuple[str, list[int], int]:
    """Encode binary data into cover text using arithmetic coding over LLM distributions.

    Prepends a header (magic + payload length) to the data before encoding.

    The encoder acts as an arithmetic *decoder*: it maintains a ``value`` register
    filled from the secret bit stream, and at each step picks the token whose
    probability interval contains that value.

    Args:
        data: The binary payload to encode.
        model: The language model.
        tokenizer: The tokenizer.
        device: Torch device.
        prompt: The seed prompt (must match during decoding).
        top_k: Number of top tokens to consider per step.
        temperature: Softmax temperature.

    Returns:
        A tuple of (cover_text, generated_token_ids, total_bits_encoded) where:
        - cover_text: the generated text (excluding the prompt)
        - generated_token_ids: list of token IDs generated
        - total_bits_encoded: number of payload+header bits encoded

    Raises:
        StegoEncodeError: If encoding fails.
    """
    # Prepend header to payload
    header = pack_header(len(data))
    full_payload = header + data
    bits = bytes_to_bits(full_payload)
    total_bits = len(bits)

    if total_bits == 0:
        raise StegoEncodeError("No data to encode")

    # Helper to read the next bit (pads with 0 beyond the payload)
    bit_pos = 0

    def next_bit() -> int:
        nonlocal bit_pos
        b = bits[bit_pos] if bit_pos < total_bits else 0
        bit_pos += 1
        return b

    # Tokenize the prompt
    prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    input_ids = prompt_ids.clone()

    # --- Arithmetic coding state (decoder side) ---
    low = 0
    high = WHOLE

    # Fill the value register with the first PRECISION bits
    value = 0
    for _ in range(PRECISION):
        value = (value << 1) | next_bit()

    generated_token_ids: list[int] = []
    tokens_generated = 0

    # Track how many bits the decoder would emit during renormalization.
    # The decoder emits bits when MSBs match (1 + pending), and increments
    # pending on straddle conditions.  The encoder must keep generating tokens
    # until the decoder has emitted at least `total_bits` bits — otherwise
    # the decoder's flush will produce canonical interval-identifying bits
    # that may not match the original secret bits.
    bits_emitted = 0
    decoder_pending = 0

    # Generate tokens until the decoder would have emitted all secret bits
    # during its renormalization (not counting the flush).
    while bits_emitted < total_bits:
        if tokens_generated >= MAX_TOKENS:
            raise StegoEncodeError(
                f"Exceeded maximum token limit ({MAX_TOKENS}) before encoding all data. "
                f"Emitted {bits_emitted}/{total_bits} bits."
            )

        # Get distribution at current position
        token_ids, cum_probs = _get_token_distribution(
            model, input_ids, device, top_k=top_k, temperature=temperature
        )
        n_tokens = len(token_ids)
        range_size = high - low

        if range_size <= 0:
            raise StegoEncodeError(
                "Arithmetic coding interval collapsed (range_size <= 0). "
                "This indicates a numerical precision issue."
            )

        # Find which token interval contains `value`.
        # Interval for token j: [low + range_size*cum_probs[j]//WHOLE,
        #                         low + range_size*cum_probs[j+1]//WHOLE)
        chosen_idx = n_tokens - 1
        for j in range(n_tokens):
            sym_high = low + (range_size * cum_probs[j + 1]) // WHOLE
            if value < sym_high:
                chosen_idx = j
                break

        # Narrow the interval
        sym_low = low + (range_size * cum_probs[chosen_idx]) // WHOLE
        sym_high = low + (range_size * cum_probs[chosen_idx + 1]) // WHOLE
        low = sym_low
        high = sym_high

        # Renormalize: shift out matching MSBs and read new bits into value.
        # Mirror the decoder's bit-emission logic to track committed bits.
        while True:
            if high <= HALF:
                # Both low and high in lower half — decoder emits 0 + pending 1s
                bits_emitted += 1 + decoder_pending
                decoder_pending = 0
            elif low >= HALF:
                # Both in upper half — decoder emits 1 + pending 0s
                bits_emitted += 1 + decoder_pending
                decoder_pending = 0
                value -= HALF
                low -= HALF
                high -= HALF
            elif low >= QUARTER and high <= 3 * QUARTER:
                # Straddle the middle — decoder increments pending
                decoder_pending += 1
                value -= QUARTER
                low -= QUARTER
                high -= QUARTER
            else:
                break

            # Shift interval and read next bit into value
            low <<= 1
            high <<= 1
            value = (value << 1) | next_bit()

        # Append chosen token
        chosen_token_id = token_ids[chosen_idx]
        generated_token_ids.append(chosen_token_id)
        new_token = torch.tensor([[chosen_token_id]], device=device)
        input_ids = torch.cat([input_ids, new_token], dim=1)
        tokens_generated += 1

    # Decode generated tokens to text
    cover_text = tokenizer.decode(generated_token_ids, skip_special_tokens=False)

    return cover_text, generated_token_ids, total_bits
