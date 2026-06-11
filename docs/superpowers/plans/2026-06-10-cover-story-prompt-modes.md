# Cover-Story Prompt Modes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a sender choose the cover story of stego text via a topic-organized public codebook (zero coordination) or a full custom prompt (shared out of band), with the opener always prepended and recovered byte-exactly on decode.

**Architecture:** Two prompt modes resolved in `StegoCodec`. **Mode A (auto topic):** an opener is chosen from `seeds.TOPICS` and recovered on decode by `match_seed` against the public codebook. **Mode C (custom prompt):** the opener is the configured `prompt=`, recovered because the decoder already knows it. In both modes the visible cover text is `opener + generated_text`; decode strips the opener (by codebook match in A, by known-length prefix in C) before replaying the arithmetic coder. The encoder/decoder coding layer is untouched.

**Tech Stack:** Python 3.10+, pytest, ruff, HuggingFace transformers/torch (model loaded once via the session `codec` fixture).

**Spec:** `docs/superpowers/specs/2026-06-10-cover-story-prompt-modes-design.md`

**Two notes that change the spec's conservative assumptions (both verified against the code):**
1. **Golden vectors are NOT affected and need NO regeneration.** `compat/generate_golden.py` builds vectors with the low-level explicit-prompt `_encode`/`make_vector` API (`CANONICAL_PROMPTS`), never `StegoCodec`'s Mode-A seed selection. Regrouping the codebook cannot change them. Task 6 only *verifies* the golden test still passes.
2. **`seeds.SEED_PHRASES` must stay the exact 256-entry legacy tuple** — `tests/test_edge_cases.py:105` asserts `len(SEED_PHRASES) == 256`. So `SEED_PHRASES` becomes the `"general"` topic verbatim; new topics are added alongside as `TOPICS`, and a new `ALL_PHRASES` is the union used for selection/matching.

---

## File Structure

- **Modify `src/mostegollm/seeds.py`** — introduce `TOPICS` (general = legacy 256 + 6 new themed topics), `ALL_PHRASES`, topic-aware `select_seed`, `list_topics`; move prefix-free validation behind a function run over `ALL_PHRASES`. Keep `SEED_PHRASES` and `match_seed` signatures.
- **Modify `src/mostegollm/codec.py`** — `__init__` gains `topic`; mode resolution + conflict `ValueError`; single-shot and chunked encode prepend the opener; decode strips it.
- **Modify `src/mostegollm/cli.py`** — `--topic` on `encode`; `topics` subcommand; catch `ValueError` from codec construction.
- **Modify `tests/conftest.py`** — add a `codec_topic` session fixture sharing the loaded model.
- **Create `tests/test_topics.py`** — codebook structure, selection, matching, prefix-free union, `list_topics`.
- **Create `tests/test_prompt_modes.py`** — Mode A topic round-trip, Mode C prepend round-trip, conflict error, prefix-mismatch fail-closed.
- **Modify `tests/test_cli.py`** — `topics` subcommand and `encode --topic`.
- **Modify `README.md`, `CHANGELOG.md`, `pyproject.toml`** — docs, changelog, version bump to 0.4.0.

---

## Task 1: Topic-organized codebook (`seeds.py`)

**Files:**
- Modify: `src/mostegollm/seeds.py`
- Test: `tests/test_topics.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/test_topics.py`:

```python
"""Tests for the topic-organized seed codebook."""

from __future__ import annotations

import pytest

from mostegollm.seeds import (
    ALL_PHRASES,
    SEED_PHRASES,
    TOPICS,
    list_topics,
    match_seed,
    select_seed,
)


def test_general_topic_is_legacy_256() -> None:
    """The 'general' topic preserves the exact legacy phrase set."""
    assert TOPICS["general"] == SEED_PHRASES
    assert len(SEED_PHRASES) == 256


def test_all_phrases_is_union_of_topics() -> None:
    expected = tuple(p for ps in TOPICS.values() for p in ps)
    assert ALL_PHRASES == expected
    assert len(ALL_PHRASES) > 256  # new topics added


def test_list_topics_includes_new_topics() -> None:
    topics = list_topics()
    for name in ("general", "cooking", "travel", "science", "personal", "work", "sports"):
        assert name in topics


def test_all_phrases_globally_prefix_free() -> None:
    """No phrase across all topics may be a prefix of another (unambiguous match)."""
    phrases = ALL_PHRASES
    for i, a in enumerate(phrases):
        for b in phrases[i + 1 :]:
            assert not a.startswith(b), f"{a!r} starts with {b!r}"
            assert not b.startswith(a), f"{b!r} starts with {a!r}"


def test_select_seed_from_topic_stays_in_topic() -> None:
    for _ in range(20):
        # vary input so the hash lands on different phrases
        data = bytes([_]) * 4
        phrase = select_seed(data, topic="cooking")
        assert phrase in TOPICS["cooking"]


def test_select_seed_default_uses_full_union() -> None:
    phrase = select_seed(b"anything")
    assert phrase in ALL_PHRASES


def test_select_seed_is_deterministic() -> None:
    assert select_seed(b"hello", topic="travel") == select_seed(b"hello", topic="travel")


def test_select_seed_unknown_topic_raises() -> None:
    with pytest.raises(ValueError, match="Unknown topic"):
        select_seed(b"x", topic="nonsense")


def test_match_seed_recovers_new_topic_opener() -> None:
    opener = TOPICS["science"][0]
    phrase, remainder = match_seed(opener + " and then the rest of the prose.")
    assert phrase == opener
    assert remainder == " and then the rest of the prose."
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_topics.py -v`
Expected: FAIL with `ImportError: cannot import name 'ALL_PHRASES'` / `'TOPICS'` / `'list_topics'`.

- [ ] **Step 3: Restructure `seeds.py`**

In `src/mostegollm/seeds.py`, keep the existing `SEED_PHRASES` tuple exactly as-is (the 256 legacy phrases). **Replace** the trailing block — from `# Pre-sort by length descending...` through the end of `match_seed` — with the following. Note `select_seed` gains a `topic` parameter and `match_seed` is unchanged except it now sorts/searches `ALL_PHRASES`.

```python
# New themed topics. Openers are multi-sentence lead-ins that anchor a cover
# story harder than a single clause. They must stay globally prefix-free with
# every phrase in SEED_PHRASES and with each other (enforced below).
_COOKING: tuple[str, ...] = (
    "Last weekend I finally tried making fresh pasta from scratch. The kitchen was a mess, but it was worth every minute.",
    "There's a small trick to caramelizing onions that nobody tells you about. Patience, and a splash of water when they start to stick.",
    "My grandmother kept her best recipes in a battered notebook. I've been slowly cooking my way through every page.",
    "If you've never roasted a whole head of garlic, you're missing out. It turns sweet and spreadable, like butter.",
    "The secret to a good weeknight curry is toasting the spices first. Everything else is just simmering and waiting.",
)

_TRAVEL: tuple[str, ...] = (
    "We landed in a town whose name I still can't pronounce. By the second morning it already felt like somewhere I'd lived for years.",
    "The train wound along the coast for hours without a single tunnel. I didn't open my book once.",
    "Nobody warns you that the best meals abroad are found down the narrowest alleys. We followed the smell of bread and got lucky.",
    "I packed for two weeks and ended up wearing the same three shirts. Travel has a way of simplifying things.",
    "The mountain pass was closed when we arrived, so we changed plans entirely. That detour turned into the best part of the trip.",
)

_SCIENCE: tuple[str, ...] = (
    "Researchers have spent decades arguing about how migratory birds find their way. The latest answer involves quantum effects in the eye.",
    "Every cell in your body runs on a molecular battery older than complex life itself. It's a design nature settled on billions of years ago.",
    "The more we map the deep ocean, the stranger it gets down there. Whole ecosystems thrive with no sunlight at all.",
    "A single teaspoon of soil holds more living organisms than there are people on Earth. Most of them we've never named.",
    "Light from the most distant galaxies left before the Sun existed. When we look up, we're really looking backward in time.",
)

_PERSONAL: tuple[str, ...] = (
    "I've been keeping a journal for about a year now. Reading the early entries back is like meeting a slightly different person.",
    "It took me a long time to learn how to say no without guilt. I'm still not great at it, but I'm getting there.",
    "My father rarely gave advice, but the little he did has stuck with me. Most of it I only understood much later.",
    "Some mornings I wake up certain of everything, and by noon I've changed my mind twice. I've stopped fighting it.",
    "I used to think confidence was something you either had or didn't. Now I suspect it's just practice wearing a disguise.",
)

_WORK: tuple[str, ...] = (
    "The project looked impossible on the whiteboard that first Monday. Three months later it shipped, mostly in one piece.",
    "Half of every meeting I sit through could have been a single short message. The other half, oddly, are the ones that matter most.",
    "I learned more from the deadline we missed than from any we hit. Failure leaves better notes.",
    "Our team finally agreed to write things down instead of remembering them. Productivity quietly doubled.",
    "The hardest part of any new role isn't the work itself. It's figuring out who actually makes the decisions.",
)

_SPORTS: tuple[str, ...] = (
    "The match was already lost by halftime, or so everyone thought. What happened in the second half is still talked about.",
    "I started running to clear my head, not to compete with anyone. Somewhere around the third month that changed.",
    "There's a particular silence in a stadium right before a penalty kick. Forty thousand people holding one breath.",
    "My coach used to say defense is just patience with a plan. It took me years on the court to understand him.",
    "The rookie nobody had heard of stole the whole season. By spring his jersey was sold out everywhere.",
)

# Topic registry. "general" is the legacy 256-phrase set, preserved verbatim so
# cover text from earlier versions still matches and decodes.
TOPICS: dict[str, tuple[str, ...]] = {
    "general": SEED_PHRASES,
    "cooking": _COOKING,
    "travel": _TRAVEL,
    "science": _SCIENCE,
    "personal": _PERSONAL,
    "work": _WORK,
    "sports": _SPORTS,
}

# Flat union used for default selection and for decode-side matching.
ALL_PHRASES: tuple[str, ...] = tuple(p for ps in TOPICS.values() for p in ps)

# Pre-sort by length descending for unambiguous longest-prefix matching.
_SORTED_PHRASES = sorted(ALL_PHRASES, key=len, reverse=True)


def _validate_prefix_free(phrases: tuple[str, ...]) -> None:
    """Raise ValueError if any phrase is a prefix of another (ambiguous matching)."""
    for i, a in enumerate(phrases):
        for b in phrases[i + 1 :]:
            if a.startswith(b) or b.startswith(a):
                raise ValueError(f"Seed phrase prefix collision: {a!r} / {b!r}")


# Enforce the invariant over the whole union at import time.
_validate_prefix_free(ALL_PHRASES)


def list_topics() -> tuple[str, ...]:
    """Return the available topic names (for Mode A cover-story selection)."""
    return tuple(TOPICS)


def select_seed(data: bytes, topic: str | None = None) -> str:
    """Pick a seed phrase deterministically from the SHA-256 hash of *data*.

    Args:
        data: The payload bytes to hash.
        topic: If given, choose from that topic's phrases; otherwise choose from
            the full union of all topics. Unknown topic raises ``ValueError``.
    """
    if topic is None:
        phrases = ALL_PHRASES
    else:
        phrases = TOPICS.get(topic)
        if phrases is None:
            raise ValueError(
                f"Unknown topic {topic!r}. Valid topics: {', '.join(TOPICS)}"
            )
    h = hashlib.sha256(data).digest()
    idx = int.from_bytes(h[:2], "big") % len(phrases)
    return phrases[idx]


def match_seed(cover_text: str) -> tuple[str, str]:
    """Find the seed phrase that prefixes *cover_text*.

    Returns:
        A tuple of ``(seed_phrase, remainder)`` where *remainder* is the cover
        text with the seed phrase stripped.

    Raises:
        ValueError: If no known seed phrase matches the start of *cover_text*.
    """
    for phrase in _SORTED_PHRASES:
        if cover_text.startswith(phrase):
            return phrase, cover_text[len(phrase) :]
    raise ValueError(
        "Cover text does not start with any known seed phrase. "
        "It may have been encoded with a different version or a custom prompt."
    )
```

Also **delete** the old inline validation block that ran over `SEED_PHRASES` (the
`for _i, _a in enumerate(SEED_PHRASES): ... del _i, _a, _b` lines) — it is
replaced by `_validate_prefix_free(ALL_PHRASES)` above.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_topics.py tests/test_edge_cases.py -v`
Expected: PASS (new topic tests pass; legacy `len(SEED_PHRASES) == 256` and the existing prefix-free test still pass).

- [ ] **Step 5: Lint**

Run: `ruff check src/mostegollm/seeds.py && ruff format src/mostegollm/seeds.py`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add src/mostegollm/seeds.py tests/test_topics.py
git commit -m "feat(seeds): topic-organized codebook with multi-sentence openers"
```

---

## Task 2: Codec prompt modes — single-shot (`codec.py`)

**Files:**
- Modify: `src/mostegollm/codec.py` (`__init__`, `encode`, `decode`, `encode_with_stats`)
- Test: `tests/test_prompt_modes.py` (create), `tests/conftest.py` (add fixture)

- [ ] **Step 1: Add the `codec_topic` fixture**

In `tests/conftest.py`, append:

```python
@pytest.fixture(scope="session")
def codec_topic(codec: StegoCodec) -> StegoCodec:
    """Mode-A codec pinned to the 'cooking' topic, sharing the loaded model."""
    c = StegoCodec(device="cpu", topic="cooking")
    model, tokenizer, device = codec._ensure_model()
    c._model = model
    c._tokenizer = tokenizer
    c._device = device
    return c
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_prompt_modes.py`:

```python
"""Tests for cover-story prompt modes (auto-topic and custom prompt)."""

from __future__ import annotations

import pytest

from mostegollm import StegoCodec, StegoDecodeError
from mostegollm.seeds import TOPICS


def _share_model(codec: StegoCodec, **kwargs) -> StegoCodec:
    """Build a StegoCodec reusing an already-loaded model (avoids reloading)."""
    c = StegoCodec(device="cpu", **kwargs)
    model, tokenizer, device = codec._ensure_model()
    c._model, c._tokenizer, c._device = model, tokenizer, device
    return c


class TestModeA:
    """Auto-topic mode: opener from the codebook, recovered on decode."""

    def test_topic_roundtrip(self, codec_topic: StegoCodec) -> None:
        data = b"meet me at noon"
        cover = codec_topic.encode(data)
        assert isinstance(cover, str)
        assert codec_topic.decode(cover) == data

    @pytest.mark.parametrize("topic", list(TOPICS))
    def test_every_topic_roundtrips(self, codec: StegoCodec, topic: str) -> None:
        c = _share_model(codec, topic=topic)
        data = b"payload"
        cover = c.encode(data)
        assert isinstance(cover, str)
        assert cover.startswith(tuple(TOPICS[topic]))
        assert c.decode(cover) == data

    def test_topic_opener_is_prepended(self, codec_topic: StegoCodec) -> None:
        cover = codec_topic.encode(b"x")
        assert isinstance(cover, str)
        assert cover.startswith(tuple(TOPICS["cooking"]))

    def test_default_mode_still_roundtrips(self, codec: StegoCodec) -> None:
        data = b"no topic given"
        cover = codec.encode(data)
        assert codec.decode(cover) == data


class TestModeC:
    """Custom-prompt mode: prompt is prepended and stripped by known length."""

    def test_custom_prompt_roundtrip(self, codec_alt_prompt: StegoCodec) -> None:
        data = b"hidden payload"
        cover = codec_alt_prompt.encode(data)
        assert isinstance(cover, str)
        assert cover.startswith("Once upon a time in a faraway land,")
        assert codec_alt_prompt.decode(cover) == data

    def test_decode_rejects_mismatched_prefix(self, codec_alt_prompt: StegoCodec) -> None:
        with pytest.raises(StegoDecodeError):
            codec_alt_prompt.decode("Some text that does not start with the prompt.")


class TestModeConflict:
    """topic and prompt are mutually exclusive."""

    def test_topic_plus_prompt_raises(self) -> None:
        with pytest.raises(ValueError, match="topic"):
            StegoCodec(device="cpu", topic="cooking", prompt="A custom prompt,")

    def test_unknown_topic_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown topic"):
            StegoCodec(device="cpu", topic="not-a-topic")
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_prompt_modes.py::TestModeConflict -v`
Expected: FAIL — `StegoCodec` does not accept `topic=` yet (`TypeError`).

- [ ] **Step 4: Update `__init__`**

In `src/mostegollm/codec.py`, add the import and `topic` parameter. Change the import line:

```python
from .seeds import list_topics, match_seed, select_seed
```

Add `topic: str | None = None` to the signature (place it right after `prompt`).
All callers in this repo (CLI, fixtures, tests) use keyword arguments, so
inserting it mid-signature is safe:

```python
    def __init__(
        self,
        model_name: str = PRIMARY_MODEL,
        device: str = "auto",
        prompt: str = DEFAULT_PROMPT,
        topic: str | None = None,
        top_k: int = TOP_K,
        temperature: float = 1.0,
        sentence_boundary: bool = False,
        token: str | None = None,
        password: str | None = None,
    ) -> None:
```

In the body, after the existing `self._prompt = prompt` assignment, add validation and storage:

```python
        if topic is not None:
            if prompt:
                raise ValueError(
                    "topic and prompt are mutually exclusive: a custom prompt is "
                    "its own opener, so a topic would be ignored."
                )
            if topic not in list_topics():
                raise ValueError(
                    f"Unknown topic {topic!r}. Valid topics: {', '.join(list_topics())}"
                )
        self._topic = topic
```

- [ ] **Step 5: Update `encode` (single-shot)**

Replace the body of `encode` *after* the `if chunk_size is not None:` block (the `seed = ...` through `return seed + cover_text` lines) with:

```python
        opener = self._prompt if self._prompt else select_seed(data, self._topic)

        if self._password is not None:
            data = _encrypt(data, self._password)
        model, tokenizer, device = self._ensure_model()
        cover_text, _token_ids, _total_bits = _encode(
            data,
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompt=opener,
            top_k=self._top_k,
            temperature=self._temperature,
            sentence_boundary=self._sentence_boundary,
        )
        return opener + cover_text
```

- [ ] **Step 6: Update `decode` (single-shot)**

Replace the seed/prompt resolution block (the `if not self._prompt: ... else: prompt = self._prompt` block) with:

```python
        # Recover the opener and strip it: by codebook match (Mode A) or by the
        # known prompt prefix (Mode C). Both are byte-exact string splits.
        if not self._prompt:
            try:
                opener, cover_text = match_seed(cover_text)
            except ValueError as exc:
                raise StegoDecodeError(str(exc)) from exc
        else:
            opener = self._prompt
            if not cover_text.startswith(opener):
                raise StegoDecodeError(
                    "Cover text does not start with the configured prompt. "
                    "The wrong prompt was supplied, or the text was not produced "
                    "by this prompt."
                )
            cover_text = cover_text[len(opener) :]
        prompt = opener
```

(The subsequent `_decode(cover_text, ..., prompt=prompt, ...)` call is unchanged.)

- [ ] **Step 7: Update `encode_with_stats`**

Replace its `seed = select_seed(data) if not self._prompt else ""` / `prompt = self._prompt or seed` lines and the `cover_text = seed + cover_text` line so it uses the same opener logic:

```python
        opener = self._prompt if self._prompt else select_seed(data, self._topic)
        if self._password is not None:
            data = _encrypt(data, self._password)
        model, tokenizer, device = self._ensure_model()

        cover_text, token_ids, total_bits = _encode(
            data,
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompt=opener,
            top_k=self._top_k,
            temperature=self._temperature,
            sentence_boundary=self._sentence_boundary,
        )
        cover_text = opener + cover_text
```

(Remove the now-unused `original_size`-adjacent `seed`/`prompt` lines; keep `original_size = len(data)` at the top of the method.)

- [ ] **Step 8: Run tests to verify they pass**

Run: `pytest tests/test_prompt_modes.py tests/test_codec.py -v`
Expected: PASS. In particular `test_wrong_prompt_fails` still passes (the Mode-A cover does not start with the alt prompt, so the new prefix check raises `StegoDecodeError`).

- [ ] **Step 9: Lint**

Run: `ruff check src/mostegollm/codec.py && ruff format src/mostegollm/codec.py`
Expected: no errors.

- [ ] **Step 10: Commit**

```bash
git add src/mostegollm/codec.py tests/conftest.py tests/test_prompt_modes.py
git commit -m "feat(codec): topic mode and prepended custom prompt"
```

---

## Task 3: Codec prompt modes — chunked (`codec.py`)

**Files:**
- Modify: `src/mostegollm/codec.py` (`_encode_chunked`, `_decode_chunked`)
- Test: `tests/test_prompt_modes.py` (extend)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_prompt_modes.py`:

```python
class TestChunkedModes:
    def test_topic_chunked_roundtrip(self, codec_topic: StegoCodec) -> None:
        data = b"A" * 60 + b"B" * 60
        covers = codec_topic.encode(data, chunk_size=60)
        assert isinstance(covers, list) and len(covers) == 2
        assert covers[0].startswith(tuple(TOPICS["cooking"]))
        assert codec_topic.decode(covers) == data

    def test_custom_prompt_chunked_roundtrip(self, codec_alt_prompt: StegoCodec) -> None:
        data = b"X" * 60 + b"Y" * 60
        covers = codec_alt_prompt.encode(data, chunk_size=60)
        assert covers[0].startswith("Once upon a time in a faraway land,")
        assert codec_alt_prompt.decode(covers) == data
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_prompt_modes.py::TestChunkedModes -v`
Expected: FAIL — chunk 0 of a custom-prompt encode is not prepended yet, so `covers[0].startswith("Once upon a time...")` is `False` (and/or decode mismatches).

- [ ] **Step 3: Update `_encode_chunked`**

Replace the per-chunk opener/prompt block inside the `for idx, chunk in enumerate(chunks):` loop (the `if idx == 0: ... else: ...` block and the `cover_texts.append(seed + cover_text)` line) with:

```python
            if idx == 0:
                opener = self._prompt if self._prompt else select_seed(data, self._topic)
                prompt = opener
                prefix = opener
            else:
                prompt = cover_texts[idx - 1][-context_size:]
                prefix = ""

            cover_text, _ids, _bits = _encode(
                chunk,
                model=model,
                tokenizer=tokenizer,
                device=device,
                prompt=prompt,
                top_k=self._top_k,
                temperature=self._temperature,
                sentence_boundary=self._sentence_boundary,
            )
            cover_texts.append(prefix + cover_text)
```

- [ ] **Step 4: Update `_decode_chunked`**

Replace the chunk-0 prompt-resolution block inside `for idx, cover_text in enumerate(cover_texts):` (the `if idx == 0: ... else: prev = ...; prompt = prev[-context_size:]` block) with:

```python
            if idx == 0:
                if not self._prompt:
                    try:
                        opener, cover_text = match_seed(cover_text)
                    except ValueError as exc:
                        raise StegoDecodeError(str(exc)) from exc
                else:
                    opener = self._prompt
                    if not cover_text.startswith(opener):
                        raise StegoDecodeError(
                            "Cover text does not start with the configured prompt."
                        )
                    cover_text = cover_text[len(opener) :]
                prompt = opener
            else:
                prompt = cover_texts[idx - 1][-context_size:]
```

(The `_decode(cover_text, ..., prompt=prompt, ...)` call below is unchanged. Note `cover_texts[idx - 1]` is the original received chunk, including the opener on chunk 0 — matching the encoder's `prev` reference.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_prompt_modes.py tests/test_chunked.py -v`
Expected: PASS (new chunked-mode tests pass; existing chunked round-trips still pass).

- [ ] **Step 6: Lint and commit**

```bash
ruff check src/mostegollm/codec.py && ruff format src/mostegollm/codec.py
git add src/mostegollm/codec.py tests/test_prompt_modes.py
git commit -m "feat(codec): prepend/recover opener for chunked encoding"
```

---

## Task 4: CLI — `--topic` and `topics` subcommand (`cli.py`)

**Files:**
- Modify: `src/mostegollm/cli.py`
- Test: `tests/test_cli.py` (extend)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_cli.py`:

```python
class TestCLITopics:
    def test_topics_subcommand_lists_topics(self) -> None:
        result = _run(["topics"])
        assert result.returncode == 0
        assert "cooking" in result.stdout
        assert "science" in result.stdout

    def test_encode_with_topic_roundtrips(self) -> None:
        enc = _run(["encode", "--topic", "travel", "hi there"])
        assert enc.returncode == 0
        cover = enc.stdout
        dec = _run(["decode", "--text"], input_text=cover)
        assert dec.returncode == 0
        assert dec.stdout.strip() == "hi there"

    def test_encode_unknown_topic_errors(self) -> None:
        result = _run(["encode", "--topic", "nonsense", "hi"])
        assert result.returncode == 1
        assert "topic" in result.stderr.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli.py::TestCLITopics -v`
Expected: FAIL — `topics` is not a valid subcommand and `--topic` is unrecognized.

- [ ] **Step 3: Add the `--topic` arg and `topics` subcommand**

In `src/mostegollm/cli.py`, in `_build_parser`, add a `--topic` option to the `encode` subparser (after the `--chunk-size` line):

```python
    enc.add_argument("--topic", default=None, help="cover-story topic for the opener (see 'topics')")
```

Add the `topics` subcommand registration next to the `models` one:

```python
    sub.add_parser("topics", help="list cover-story topics")
```

- [ ] **Step 4: Add the `_cmd_topics` printer**

Add this function near `_cmd_models` in `src/mostegollm/cli.py`:

```python
def _cmd_topics() -> None:
    """Print available cover-story topics with an example opener."""
    from .seeds import TOPICS

    name_w = max(len(name) for name in TOPICS)
    print(f"  {'Topic':<{name_w}}  Example opener")
    print("  " + "-" * (name_w + 16))
    for name, phrases in TOPICS.items():
        example = phrases[0]
        if len(example) > 60:
            example = example[:57] + "..."
        print(f"  {name:<{name_w}}  {example}")
```

- [ ] **Step 5: Wire `topics` and `--topic` into `main`**

In `main`, add the `topics` branch next to the `models` branch:

```python
    if args.command == "topics":
        _cmd_topics()
        return
```

Resolve the topic and pass it into the codec. After the `password = getattr(args, "password", None)` line, add:

```python
    topic = getattr(args, "topic", None)
```

Wrap the `StegoCodec(...)` construction so a bad `topic`/`prompt` combination is reported cleanly. Replace the existing `codec = StegoCodec(...)` call with:

```python
    try:
        codec = StegoCodec(
            model_name=model_name,
            device=device,
            prompt=prompt,
            topic=topic,
            top_k=top_k,
            sentence_boundary=sentence_boundary,
            password=password,
        )
    except ValueError as exc:
        print(f"mostegollm: {exc}", file=sys.stderr)
        sys.exit(1)
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_cli.py -v`
Expected: PASS (new topic tests pass; existing CLI tests unaffected).

- [ ] **Step 7: Lint and commit**

```bash
ruff check src/mostegollm/cli.py && ruff format src/mostegollm/cli.py
git add src/mostegollm/cli.py tests/test_cli.py
git commit -m "feat(cli): --topic flag and topics subcommand"
```

---

## Task 5: Docs, changelog, version bump

**Files:**
- Modify: `README.md`, `CHANGELOG.md`, `pyproject.toml`

- [ ] **Step 1: Bump the version**

In `pyproject.toml`, change `version = "0.3.0"` to:

```toml
version = "0.4.0"
```

- [ ] **Step 2: Add a CHANGELOG entry**

At the top of `CHANGELOG.md` (above the `## 0.3.0` heading), insert:

```markdown
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
- `StegoCodec.list_topics()` and a `topics` CLI subcommand.

### Changed (breaking)
- Explicit `prompt=` now **prepends** the prompt to the cover text (it was hidden
  context before). Decode strips it. Round-trip is preserved; output strings
  differ from 0.3.x. `topic=` and `prompt=` are mutually exclusive.

### Compatibility
- Cover text produced by 0.3.x still decodes (the legacy phrase set is preserved
  as the `general` topic and the codebook stays globally prefix-free).
```

- [ ] **Step 3: Update the README**

In `README.md`, replace the **Security Model** subsection's opening note about seed phrases with a "Cover stories" subsection. Insert this block immediately after the `## How It Works` section:

```markdown
## Cover Stories

By default the cover text reads as a short personal note. You can steer what it
*looks like it's about* so a casual reader sees plausible, on-topic prose.

```python
from mostegollm import StegoCodec

# Mode A — auto topic (zero coordination; the decoder recovers the opener):
codec = StegoCodec(topic="cooking")
cover = codec.encode_str("meet me at noon")
# -> "Last weekend I finally tried making fresh pasta from scratch. ..."
assert StegoCodec(topic="cooking").decode_str(cover) == "meet me at noon"

# Mode C — your own opener (share the prompt with the recipient out of band):
codec = StegoCodec(prompt="Here's the quarterly summary you asked for. ")
cover = codec.encode_str("meet me at noon")
assert StegoCodec(prompt="Here's the quarterly summary you asked for. ").decode_str(cover) == "meet me at noon"
```

Topics: `general`, `cooking`, `travel`, `science`, `personal`, `work`, `sports`
(`mostegollm topics` or `StegoCodec.list_topics()`).

**Cover-story quality is bounded by the model.** The opener sets the reader's
expectation, but the rest of the prose is generated by the model and drifts
off-topic — more so with the tiny default `SmolLM-135M`. For a more convincing
cover story, use a stronger model (`HuggingFaceTB/SmolLM-360M` or
`Qwen/Qwen2.5-0.5B`). The cover story is strongest when the claimed context only
needs a plausible opening plus loosely on-topic text, not a rigorous document.

> **Threat model.** Cover stories defend against a *casual* reader who has no
> reason to suspect hidden data. They are not a defense against an adversary who
> runs the model to test for steganography — for confidentiality against a
> capable adversary, use `password=` (AES-256-GCM).
```

Update the `Configuration` table: add a `topic` row after the `prompt` row:

```markdown
| `topic`            | `None`                        | Cover-story topic for the auto opener (Mode A); mutually exclusive with `prompt` |
```

- [ ] **Step 4: Sanity-check the docs build/format**

Run: `python -c "import mostegollm; print(mostegollm.StegoCodec.list_models()[0].name)"`
Expected: prints `HuggingFaceTB/SmolLM-135M` (import still works; no syntax errors introduced).

- [ ] **Step 5: Commit**

```bash
git add README.md CHANGELOG.md pyproject.toml
git commit -m "docs: cover-story modes, threat model, version 0.4.0"
```

---

## Task 6: Full verification

**Files:** none (verification only)

- [ ] **Step 1: Run the whole test suite**

Run (model-dependent tests load SmolLM-135M on CPU; this can exceed the 120s foreground limit, so allow time / run detached):

`pytest tests/ -q`
Expected: all tests pass, including `tests/test_golden_vectors.py` (golden corpus still decodes — it uses the low-level explicit-prompt API and is unaffected by codebook regrouping) and `tests/test_edge_cases.py` (`len(SEED_PHRASES) == 256`).

- [ ] **Step 2: Lint the whole tree**

Run: `ruff check . && ruff format --check .`
Expected: no errors.

- [ ] **Step 3: Manual smoke test of the new modes**

Run:

```bash
mostegollm topics
mostegollm encode --topic science "secret" | mostegollm decode --text
```

Expected: `topics` prints the table; the encode→decode pipe prints `secret`.

- [ ] **Step 4: Final commit (if any formatting changed)**

```bash
git add -A
git commit -m "chore: final formatting for cover-story modes" || echo "nothing to commit"
```
```
