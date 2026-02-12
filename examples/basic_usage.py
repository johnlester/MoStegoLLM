#!/usr/bin/env python3
"""Basic usage example for MoStegoLLM.

Demonstrates encoding a secret message into natural-looking English text
and decoding it back.
"""

from mostegollm import StegoCodec


def main() -> None:
    # Initialize the codec (model downloads on first run)
    print("Initializing StegoCodec...")
    codec = StegoCodec(
        device="auto",
        prompt="The following is a passage from a book:",
    )

    # --- Encode bytes → cover text ---
    secret = b"Hello, World!"
    print(f"\nOriginal message: {secret}")

    cover_text = codec.encode(secret)
    print(f"\nCover text:\n{cover_text}")

    # --- Decode cover text → bytes ---
    recovered = codec.decode(cover_text)
    print(f"\nRecovered: {recovered}")
    assert recovered == secret, "Round-trip failed!"
    print("Round-trip successful!")

    # --- String convenience ---
    print("\n--- String convenience ---")
    cover = codec.encode_str("steganography is fun")
    print(f"Cover: {cover[:80]}...")
    print(f"Decoded: {codec.decode_str(cover)}")

    # --- Diagnostics ---
    print("\n--- Encoding stats ---")
    stats = codec.encode_with_stats(b"diagnostic payload")
    print(f"Payload size:   {stats.payload_size_bytes} bytes")
    print(f"Total tokens:   {stats.total_tokens}")
    print(f"Bits per token: {stats.bits_per_token:.2f}")
    print(f"Cover text len: {len(stats.cover_text)} chars")


if __name__ == "__main__":
    main()
