# Changelog

## 0.3.0

### Changed (breaking)
- Encoding now uses reproducible **rank-interval coding** instead of
  probability-magnitude arithmetic coding. Cover text produced by 0.2.0 and
  earlier **cannot be decoded** by 0.3.0, and vice versa.

### Added
- Cross-platform reproducibility: cover text encoded on one PyTorch
  version / device (CPU or GPU) now decodes on another. Decoding no longer
  requires an identical floating-point environment. Any rare residual
  divergence is detected by the existing CRC-32 (decode raises; never returns
  wrong bytes).
- TF32 disabled and deterministic algorithms enabled to further reduce
  cross-platform logit divergence.
