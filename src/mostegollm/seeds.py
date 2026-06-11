"""Seed phrases for varying the opening of steganographic cover text.

A fixed list of short phrases is used so that different input messages produce
cover text with different openings.  The phrase is selected deterministically
via a SHA-256 hash of the plaintext data and is prepended to the cover text so
the decoder can recover it without knowing the original message.
"""

from __future__ import annotations

import hashlib

# 256 short seed phrases — each is a plausible English sentence opener.
# IMPORTANT: no phrase may be a prefix of another (ensures unambiguous matching).
SEED_PHRASES: tuple[str, ...] = (
    # --- General / expository ---
    "According to experts,",
    "After years of research,",
    "Although it may seem surprising,",
    "I actually think",
    "You remember when",
    "As we look ahead,",
    "Every time I see",
    "I was just thinking",
    "So here's the thing,",
    "My friend once told",
    "By all accounts,",
    "Contrary to popular belief,",
    "I keep hearing about",
    "Despite initial setbacks,",
    "During the past decade,",
    "Even in ancient times,",
    "Honestly, I never",
    "It's funny how",
    "Few people realize that",
    "You know what's weird,",
    "I spent a lot of",
    "The other day I",
    "Given the circumstances,",
    "I used to believe",
    "Someone recently asked",
    "However, recent findings",
    "I'm not sure why",
    "Back when I was",
    "We all know that",
    "In hindsight, it seems",
    "In many ways, the story",
    "In recent decades,",
    "Whenever I think about",
    "In the early morning hours,",
    "I've always wondered",
    "In the years that followed,",
    "Most people don't",
    "Interestingly enough,",
    "If you've ever tried",
    "I finally figured out",
    "It is often said that",
    "It is worth noting that",
    "It may come as a surprise,",
    "My favorite part about",
    "I read somewhere that",
    "Just as importantly,",
    "Growing up, I always",
    "So apparently,",
    "I never really understood",
    "Many people are unaware that",
    "You'd be surprised how",
    "More importantly,",
    "A while back, I",
    "Not everyone is aware that",
    "Not long ago, a team",
    "Nowadays, it is common",
    "I was talking to",
    "On closer inspection,",
    "On the other hand,",
    "Last year, I decided",
    "One could argue that",
    "One of the most fascinating",
    "One of the most overlooked",
    "I know it sounds",
    "Here's something I",
    "When I first heard",
    "Rather than focusing on",
    "Recent studies have shown",
    "I don't usually talk",
    "It took me years",
    "I just found out",
    "Nobody ever tells you",
    "I've been meaning to",
    "What really gets me",
    "I can't help but",
    "Surprisingly, the answer",
    "The answer, it turns out,",
    "If I'm being honest,",
    "So I was reading",
    "The weird thing is",
    "The discovery was made",
    "I remember the first",
    "The evidence is clear:",
    "The fact remains that",
    "The first thing to understand",
    "People always ask me",
    "I had no idea",
    "The idea behind this",
    "You wouldn't believe how",
    "The importance of this",
    "The interesting thing about",
    "The key to understanding",
    "At some point, I",
    "I think the reason",
    "It's hard to explain,",
    "My whole life, I",
    "I only recently learned",
    "The thing about this",
    "The problem, of course,",
    "The question of how",
    "The reason for this",
    "The relationship between",
    "The results were unexpected:",
    "I've noticed that",
    "The story begins in",
    "When you really think",
    "The truth of the matter",
    "I didn't expect to",
    "It still surprises me",
    "I keep coming back",
    "There is no denying that",
    "The way I see",
    "I'm starting to think",
    "Have you ever noticed",
    "To put it in context,",
    "To this day, many",
    "Today, we take for granted",
    "For what it's worth,",
    "Under normal circumstances,",
    "Unlike what many assume,",
    "Until recently, scientists",
    "What began as a small",
    "What is less well known",
    "What makes this particularly",
    "When the first reports",
    "While it is true that",
    "While much attention has",
    "With each passing year,",
    "I wish someone had",
    "Lately, I've been",
    "I got curious about",
    # --- Nature / science ---
    "Across the vast plains,",
    "Along the rocky coastline,",
    "At the edge of the forest,",
    "Beneath the surface of",
    "If you ask me,",
    "I want to talk",
    "One thing I know is",
    "Deep beneath the ocean,",
    "I tried explaining this",
    "You might not know",
    "I heard that in",
    "It always seemed like",
    "High in the mountain passes,",
    "In the depths of winter,",
    "In the world of marine",
    "Life in the desert requires",
    "I could be wrong,",
    "Mountains have always inspired",
    "Near the banks of the river,",
    "When we were kids,",
    "On the African savanna,",
    "I sometimes wonder if",
    "Rivers have shaped the",
    "Looking at it now,",
    "I think most people",
    "Stars in the night sky",
    "To be fair, I",
    "I've spent a while",
    "The truth is, I",
    "I wasn't expecting this,",
    "Just the other day,",
    "I still can't believe",
    "My take on this",
    "Personally, I think",
    "I mean, when you",
    "What surprised me was",
    "I've always felt that",
    "It turns out that",
    "The natural cycle of",
    "I genuinely believe that",
    "So the thing is,",
    "I stumbled across this",
    "You probably already know",
    "I tried to figure",
    "From what I can",
    "The vast expanse of space",
    "I realized the other",
    "Water is perhaps the most",
    "Weather systems often form",
    "What I find interesting",
    # --- Culture / arts / society ---
    "I never thought about",
    "The more I learn,",
    "Art has the power to",
    "Books have shaped the way",
    "Communities around the world",
    "Cultural traditions often reveal",
    "Dance has been a form",
    "Education systems vary widely",
    "Festivals and celebrations mark",
    "Food traditions tell the story",
    "Gardens have served as",
    "Great thinkers of the past",
    "Human ingenuity has always",
    "In the realm of music,",
    "Innovation rarely happens in",
    "Language is one of the most",
    "Libraries have long served as",
    "Literature from this period",
    "Maps were once considered",
    "Merchants and traders of old",
    "Music has an extraordinary",
    "Oral traditions preserved the",
    "Painters of the Renaissance",
    "People have always gathered",
    "Philosophy asks some of the",
    "Poetry has the unique ability",
    "Public spaces have always",
    "Storytelling is one of the",
    "The art of navigation",
    "The craft of glassmaking",
    "I learned a lot about",
    "The evolution of language",
    "The exchange of ideas between",
    "The history of medicine",
    "The invention of the printing",
    "The practice of meditation",
    "The role of education in",
    "The spread of knowledge through",
    "The tradition of craftsmanship",
    "The written word has",
    "Theater and performance have",
    "Trade routes once connected",
    "Travelers in the ancient world",
    "Villages along the coast",
    "Wisdom passed down through",
    "Writers of the eighteenth",
    # --- Narrative / story-like ---
    "A chance encounter at",
    "A curious pattern emerged",
    "A letter arrived one morning",
    "A long-forgotten manuscript",
    "A peculiar thing happened",
    "A small team of researchers",
    "A strange silence fell over",
    "An unexpected discovery in",
    "As dawn broke over the",
    "As the seasons changed,",
    "By the time anyone noticed,",
    "Early one autumn morning,",
    "Elsewhere in the region,",
    "For reasons no one fully",
    "Hidden among the shelves,",
    "In a quiet village near",
    "In the summer of that year,",
    "It all started with a simple",
    "Late one evening, a curious",
    "News of the finding spread",
    "No one expected that a",
    "Nothing could have prepared",
    "On a clear winter night,",
    "On the outskirts of town,",
    "Something remarkable happened",
    "The expedition set out from",
    "The first clue appeared in",
    "The journey was long and",
    "The morning sun revealed",
)

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
    """Return the available topic names in the TOPICS registry."""
    return tuple(TOPICS)


def select_seed(data: bytes, topic: str | None = None) -> str:
    """Pick a seed phrase deterministically from the SHA-256 hash of *data*.

    Args:
        data: The payload bytes to hash.
        topic: If given, choose from that topic's phrases. When ``None`` (the
            default), choose from the neutral ``general`` set — its 256 entries
            give a uniform ``2**16 / 256`` selection and read plausibly for any
            content. Themed openers are opt-in via an explicit topic. Unknown
            topic raises ``ValueError``.
    """
    if topic is None:
        phrases = TOPICS["general"]
    else:
        phrases = TOPICS.get(topic)
        if phrases is None:
            raise ValueError(f"Unknown topic {topic!r}. Valid topics: {', '.join(TOPICS)}")
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
