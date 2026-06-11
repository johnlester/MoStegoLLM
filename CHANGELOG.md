# Changelog

## 0.4.0

### Added
- **Cover-story prompt modes.** Choose how the cover text reads:
  - **Auto topic (Mode A, default):** pass `topic=` (or `--topic`) to select a
    themed opener — `cooking`, `travel`, `science`, `personal`, `work`,
    `sports`, or `general`. Zero coordination: the decoder recovers the opener
    from the text. Run `mostegollm topics` to list them.
  - **Custom prompt (Mode C):** pass `prompt=` to supply your own opener, shared
    out of band with the recipient.
- Multi-sentence themed openers for stronger cover-story anchoring.
- `seeds.list_topics()` and a `topics` CLI subcommand.

### Changed (breaking)
- Explicit `prompt=` now **prepends** the prompt to the cover text (it was hidden
  context before). Decode strips it. Round-trip is preserved; output strings
  differ from 0.3.x. `topic=` and `prompt=` are mutually exclusive.

### Compatibility
- Cover text produced by 0.3.x still decodes (the legacy phrase set is preserved
  as the `general` topic and the codebook stays globally prefix-free).

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
