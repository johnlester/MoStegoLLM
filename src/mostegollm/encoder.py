"""Encoding logic: consume bits from secret data to select tokens via arithmetic coding.

The stego encoder is conceptually an arithmetic *decoder* — it reads bits from
the secret message and uses them to navigate the probability intervals to choose
which token to emit at each step.
"""

from __future__ import annotations

import warnings
import zlib

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from transformers.cache_utils import DynamicCache

from .coding import HALF, K, PRECISION, QUARTER, WHOLE, step_coding
from .utils import (
    StegoEncodeError,
    bytes_to_bits,
    pack_header,
)


def warn_if_non_canonical_dtype(model: PreTrainedModel) -> None:
    """Warn if *model* is not float32, the only dtype portable across platforms.

    Encode and decode must reproduce identical probability distributions. fp16/bf16
    quantization (~1e-3) is the same order as the coder's GUARD merge threshold and
    flips token ordering across devices, so a non-float32 model only round-trips
    when BOTH sides use the *identical* dtype — and never across CPU/GPU. Empirically
    confirmed by the cross-compatibility matrix (cpu-fp32 <-> t4-fp32 = 100%;
    fp16 <-> fp32 = 0%). float32 is the supported, portable default. Warns rather
    than raises, since a deliberately-matched non-float32 setup can still work.
    """
    dtype = getattr(model, "dtype", None)
    if dtype is not None and dtype != torch.float32:
        warnings.warn(
            f"Model dtype is {dtype}, not float32. Reproducible decoding requires the "
            "decoder to use the identical dtype, and only float32 is portable across "
            "devices/PyTorch versions. See the reproducibility contract in the README.",
            stacklevel=3,
        )

# Cache of non-round-tripping token sets, keyed by tokenizer identity.
_NON_RT_CACHE: dict[int, frozenset[int]] = {}

# Maximum tokens to generate as a safety limit
MAX_TOKENS = 8192

# Maximum extra tokens past data-recoverable to search for a sentence boundary
MAX_EXTRA_TOKENS = 100

# Top-k tokens to consider from the distribution (kept for backward-compatible imports)
TOP_K = K


def get_non_roundtrip_tokens(tokenizer: PreTrainedTokenizerBase) -> frozenset[int]:
    """Return the set of token IDs that don't survive a decode→encode round-trip.

    These are typically byte-fallback tokens that decode to the Unicode
    replacement character and re-encode to a different token ID.  Cached
    per tokenizer instance.
    """
    key = id(tokenizer)
    if key not in _NON_RT_CACHE:
        bad: set[int] = set()
        for tid in range(tokenizer.vocab_size):
            text = tokenizer.decode([tid])
            re_encoded = tokenizer.encode(text, add_special_tokens=False)
            if re_encoded != [tid]:
                bad.add(tid)
        _NON_RT_CACHE[key] = frozenset(bad)
    return _NON_RT_CACHE[key]


def _filter_tokens(
    tokenizer: PreTrainedTokenizerBase,
    prev_token_id: int | None,
    token_ids: list[int],
    logits: list[float],
    non_rt_tokens: frozenset[int],
    merge_cache: dict[tuple[int, int], bool],
) -> tuple[list[int], list[float]]:
    """Drop tokens that break a decode->encode round-trip, keeping logits paired.

    Filters (1) tokens that don't round-trip individually and (2) tokens that
    would BPE-merge with *prev_token_id*. These checks are pure string ops, so
    they are identical across platforms. Falls back to the unfiltered lists if
    everything would be removed.
    """
    keep_ids: list[int] = []
    keep_logits: list[float] = []
    for tid, lg in zip(token_ids, logits):
        if tid in non_rt_tokens:
            continue
        if prev_token_id is not None:
            pair = (prev_token_id, tid)
            if pair not in merge_cache:
                text = tokenizer.decode([prev_token_id, tid])
                re_enc = tokenizer.encode(text, add_special_tokens=False)
                merge_cache[pair] = re_enc != [prev_token_id, tid]
            if merge_cache[pair]:
                continue
        keep_ids.append(tid)
        keep_logits.append(lg)
    if not keep_ids:
        return token_ids, logits
    return keep_ids, keep_logits


def _is_sentence_ending(tokenizer: PreTrainedTokenizerBase, token_id: int) -> bool:
    """Check if a token ends a sentence (its decoded text ends with . ! or ?)."""
    text = tokenizer.decode([token_id]).rstrip()
    return len(text) > 0 and text[-1] in ".!?"


def _get_topk_logits(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    device: torch.device,
    top_k: int = K,
    temperature: float = 1.0,
    past_key_values: DynamicCache | None = None,
) -> tuple[list[int], list[float], DynamicCache]:
    """Return the top-k token ids and their (temperature-scaled) float64 logits.

    No softmax: the reproducible coder uses logit *order* and *gaps*, not
    probability magnitudes.
    """
    with torch.no_grad():
        outputs = model(input_ids, past_key_values=past_key_values, use_cache=True)
        logits = outputs.logits[0, -1, :]

    if temperature != 1.0:
        logits = logits / temperature

    logits_f64 = logits.to(dtype=torch.float64)
    actual_k = min(top_k, logits_f64.shape[0])
    top = torch.topk(logits_f64, actual_k)
    token_ids = top.indices.cpu().tolist()
    logit_vals = top.values.cpu().tolist()
    return token_ids, logit_vals, outputs.past_key_values


def encode(
    data: bytes,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    device: torch.device,
    prompt: str,
    top_k: int = TOP_K,
    temperature: float = 1.0,
    sentence_boundary: bool = False,
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
        temperature: Logit scaling factor (divides logits before top-k selection; 1.0 = no change).
        sentence_boundary: If ``True``, continue generating tokens past the
            data-recoverable point until the cover text ends at a sentence
            boundary (``.``, ``!``, or ``?``).

    Returns:
        A tuple of (cover_text, generated_token_ids, total_bits_encoded) where:
        - cover_text: the generated text (excluding the prompt)
        - generated_token_ids: list of token IDs generated
        - total_bits_encoded: number of payload+header bits encoded

    Raises:
        StegoEncodeError: If encoding fails.
    """
    warn_if_non_canonical_dtype(model)

    # Prepend header to payload
    payload_crc = zlib.crc32(data) & 0xFFFFFFFF
    header = pack_header(len(data), crc32=payload_crc)
    full_payload = header + data
    bits = bytes_to_bits(full_payload)
    total_bits = len(bits)

    if total_bits == 0:
        raise StegoEncodeError("No data to encode")

    # Helper to read the next bit.  Beyond the payload we feed a deterministic
    # pseudo-random stream rather than zeros.  A zero-valued arithmetic register
    # makes token selection hug the bottom of every probability interval
    # (degenerate, repetitive output with no sentence boundaries); a random
    # value instead behaves like temperature-1 sampling and keeps the prose
    # natural.  The decoder never reads these padding bits — the header length
    # bounds the payload — so any deterministic stream is round-trip safe.
    bit_pos = 0
    pad_state = 0x5354BEEF  # xorshift32 seed (deterministic encoding)

    def next_bit() -> int:
        nonlocal bit_pos, pad_state
        if bit_pos < total_bits:
            b = bits[bit_pos]
        else:
            x = pad_state
            x ^= (x << 13) & 0xFFFFFFFF
            x ^= x >> 17
            x ^= (x << 5) & 0xFFFFFFFF
            pad_state = x & 0xFFFFFFFF
            b = pad_state & 1
        bit_pos += 1
        return b

    # Tokenize the prompt (ensure at least the BOS token for empty prompts)
    prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    if prompt_ids.numel() == 0:
        bos = tokenizer.bos_token_id or 0
        prompt_ids = torch.tensor([[bos]], device=device)
    next_input = prompt_ids.clone()
    past_kv: DynamicCache | None = None

    # Pre-compute tokens that never round-trip and a cache for merge checks.
    non_rt_tokens = get_non_roundtrip_tokens(tokenizer)
    merge_cache: dict[tuple[int, int], bool] = {}
    prev_token_id: int | None = None

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
    ever_recoverable = False
    extra_tokens = 0

    while True:
        if tokens_generated >= MAX_TOKENS:
            raise StegoEncodeError(
                f"Exceeded maximum token limit ({MAX_TOKENS}) before encoding all data. "
                f"Emitted {bits_emitted}/{total_bits} bits."
            )

        # Reproducible rank-interval coding: top-k logits -> BPE filter ->
        # sorted/run-merged fixed integer intervals.
        token_ids, logits, past_kv = _get_topk_logits(
            model,
            next_input,
            device,
            top_k=top_k,
            temperature=temperature,
            past_key_values=past_kv,
        )
        token_ids, logits = _filter_tokens(
            tokenizer, prev_token_id, token_ids, logits, non_rt_tokens, merge_cache
        )
        step = step_coding(token_ids, logits)

        range_size = high - low
        if range_size <= 0:
            raise StegoEncodeError(
                "Arithmetic coding interval collapsed (range_size <= 0). "
                "This indicates a numerical precision issue."
            )

        # Navigate `value` into the interval that contains it. Pre-seed with the
        # last interval as a safety net: its hi is WHOLE, so the loop's invariant
        # (value < high) guarantees a match before the list is exhausted.
        ilo, ihi, chosen_token_id = step.intervals[-1]
        sym_low = low + (range_size * ilo) // WHOLE
        sym_high = low + (range_size * ihi) // WHOLE
        for cand_lo, cand_hi, rep in step.intervals:
            cand_high = low + (range_size * cand_hi) // WHOLE
            if value < cand_high:
                chosen_token_id = rep
                sym_low = low + (range_size * cand_lo) // WHOLE
                sym_high = cand_high
                break
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
        generated_token_ids.append(chosen_token_id)
        next_input = torch.tensor([[chosen_token_id]], device=device)
        tokens_generated += 1
        prev_token_id = chosen_token_id

        # Re-evaluate whether data is recoverable RIGHT NOW (not latched).
        # The flush check is valid only at the instant it's evaluated — if we
        # continue generating, the arithmetic state changes and the flush may
        # no longer produce the right bits.  So we must re-check each iteration.
        data_recoverable = bits_emitted >= total_bits
        if not data_recoverable:
            remaining = total_bits - bits_emitted
            flush_pending = decoder_pending + 1
            if 1 + flush_pending >= remaining:
                if low < QUARTER:
                    flush_bits = [0] + [1] * flush_pending
                else:
                    flush_bits = [1] + [0] * flush_pending
                if flush_bits[:remaining] == bits[bits_emitted:total_bits]:
                    data_recoverable = True

        if not data_recoverable:
            continue

        # Data is currently recoverable — decide whether to stop.
        if not sentence_boundary:
            break

        if _is_sentence_ending(tokenizer, chosen_token_id):
            break

        # Track tokens since we first became recoverable for the safety limit.
        if not ever_recoverable:
            ever_recoverable = True
        extra_tokens += 1
        if extra_tokens >= MAX_EXTRA_TOKENS:
            break

    # Decode generated tokens to text
    cover_text = tokenizer.decode(generated_token_ids, skip_special_tokens=False)

    return cover_text, generated_token_ids, total_bits
