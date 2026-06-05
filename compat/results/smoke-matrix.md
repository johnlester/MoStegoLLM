# Cross-compatibility matrix (Modal smoke)

## Decode matrix (rows = encoder, cols = decoder)

| enc \ dec | cpu-fp32 | t4-fp32 | t4-fp16 |
|---|---|---|---|
| **cpu-fp32** | ✓ 24/24 | ✓ 24/24 | ✗ 4/24 logit_divergence |
| **t4-fp32** | ✓ 24/24 | ✓ 24/24 | ✗ 4/24 logit_divergence |
| **t4-fp16** | ✗ 4/24 logit_divergence | ✗ 4/24 logit_divergence | ✓ 24/24 |

**Overall: 136/216 (encoder × decoder × payload) pairs recovered.**

## Distribution agreement vs `cpu-fp32` (top-k ordering per step)

| cell | steps matching reference |
|---|---|
| cpu-fp32 | 28/28 (100.0%) |
| t4-fp32 | 28/28 (100.0%) |
| t4-fp16 | 0/28 (0.0%) |
