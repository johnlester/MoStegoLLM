#!/usr/bin/env bash
# MoStegoLLM CLI Walkthrough
# Demonstrates encoding and decoding from the command line.

set -euo pipefail

echo "=== MoStegoLLM CLI Walkthrough ==="
echo

# 1. Encode a text message
echo "--- Encoding a message ---"
mostegollm encode "attack at dawn" -o /tmp/cover.txt
echo "Cover text written to /tmp/cover.txt"
head -c 200 /tmp/cover.txt
echo "..."
echo

# 2. Decode it
echo "--- Decoding ---"
mostegollm decode -f /tmp/cover.txt --text
echo

# 3. Encode with encryption
echo "--- Encrypted encode ---"
mostegollm encode "classified information" --password mysecret -o /tmp/encrypted_cover.txt
echo "Encrypted cover text written to /tmp/encrypted_cover.txt"
echo

# 4. Decode with encryption
echo "--- Encrypted decode ---"
mostegollm decode -f /tmp/encrypted_cover.txt --text --password mysecret
echo

# 5. Encoding stats
echo "--- Encoding with stats ---"
mostegollm encode "stats demo" --stats 2>&1
echo

# 6. List models
echo "--- Available models ---"
mostegollm models
echo

echo "=== Done ==="
