#!/usr/bin/env python3
"""Basic usage example for MoStegoLLM.

Demonstrates encoding secret data into natural-looking English text
and decoding it back, including encryption, chunking, and diagnostics.
"""

from mostegollm import StegoCodec


def main() -> None:
    # --- Initialize ---
    # Model downloads on first run (~270 MB for SmolLM-135M).
    # Use device="cuda" for GPU acceleration.
    print("Initializing StegoCodec...")
    codec = StegoCodec(device="auto")

    # --- 1. Basic encode/decode (bytes) ---
    print("\n--- Basic encode/decode ---")
    secret = b"Hello, World!"
    cover_text = codec.encode(secret)
    print(f"Original:   {secret}")
    print(f"Cover text: {cover_text[:80]}...")
    recovered = codec.decode(cover_text)
    print(f"Recovered:  {recovered}")
    assert recovered == secret

    # --- 2. String convenience ---
    print("\n--- String convenience ---")
    cover = codec.encode_str("steganography is fun")
    decoded = codec.decode_str(cover)
    print(f"Cover:   {cover[:80]}...")
    print(f"Decoded: {decoded}")

    # --- 3. Encryption ---
    print("\n--- Encryption (AES-256-GCM) ---")
    secure_codec = StegoCodec(device="auto", password="my-secret-key")
    cover = secure_codec.encode(b"classified information")
    print(f"Encrypted cover: {cover[:80]}...")
    recovered = secure_codec.decode(cover)
    print(f"Recovered: {recovered}")

    # --- 4. Chunked encoding for large data ---
    print("\n--- Chunked encoding ---")
    large_data = b"A" * 200
    covers = codec.encode(large_data, chunk_size=50)
    print(f"Payload: {len(large_data)} bytes -> {len(covers)} chunks")
    recovered = codec.decode(covers)
    assert recovered == large_data
    print("Chunked round-trip successful!")

    # --- 5. Encoding stats ---
    print("\n--- Encoding stats ---")
    stats = codec.encode_with_stats(b"diagnostic payload")
    print(f"Payload size:   {stats.payload_size_bytes} bytes")
    print(f"Total tokens:   {stats.total_tokens}")
    print(f"Bits per token: {stats.bits_per_token:.2f}")
    print(f"Cover text len: {len(stats.cover_text)} chars")


if __name__ == "__main__":
    main()
