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


def test_select_seed_default_uses_general() -> None:
    phrase = select_seed(b"anything")
    assert phrase in TOPICS["general"]


def test_select_seed_is_deterministic() -> None:
    # Pinned golden value: changing the hash slicing or topic order breaks this.
    assert (
        select_seed(b"hello", topic="travel")
        == "The train wound along the coast for hours without a single tunnel. I didn't open my book once."
    )


def test_select_seed_unknown_topic_raises() -> None:
    with pytest.raises(ValueError, match="Unknown topic"):
        select_seed(b"x", topic="nonsense")


def test_match_seed_recovers_new_topic_opener() -> None:
    opener = TOPICS["science"][0]
    phrase, remainder = match_seed(opener + " and then the rest of the prose.")
    assert phrase == opener
    assert remainder == " and then the rest of the prose."
