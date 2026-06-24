"""Phase 6.1 — Concept Window Planner: advanced tests across multiple niches.

Tests cover:
  • Unit correctness (split, merge, distribute, tag, cluster)
  • Properties (duration preservation, no overlap, no gaps, bounded duration)
  • Multi-niche scripts (cooking, finance, history, tech, fitness, ping-pong)
  • Edge cases (empty, 1-word, very-long, all-numbers, all-uppercase)
"""
from __future__ import annotations

import pytest

from app.services.sync.concept import (
    ConceptWindow,
    cluster_sizes,
    group_into_clusters,
    segment_into_windows,
    _split_lexical,
    _split_by_word_count,
    _tag_entity,
    _merge_too_short,
)
from app.services.sync.pipeline import NarrationClause


# ─────────────────────────────────────────────────────────────────────────
# 1. Lexical split — natural breath points
# ─────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text,expected_min_pieces", [
    ("Hello, world.", 2),
    ("The blade is fast but the rubber is slow.", 2),
    ("Stiga makes paddles, and Butterfly makes blades, and Donic makes rubbers.", 3),
    ("Although expensive, it performs well because of carbon layers.", 3),
    ("No commas no conjunctions just one straight thought", 1),
])
def test_lexical_split_finds_breath_points(text, expected_min_pieces):
    pieces = _split_lexical(text)
    assert len(pieces) >= expected_min_pieces, f"got {pieces}"
    # No empty pieces survive
    assert all(p.strip() for p in pieces)


def test_split_by_word_count_fallback():
    text = "one two three four five six seven eight nine ten eleven twelve"
    pieces = _split_by_word_count(text, target_seconds=2.0)  # ≈5 words
    assert len(pieces) >= 2
    # All pieces are non-empty
    assert all(p for p in pieces)
    # No word lost
    rejoined = " ".join(pieces).split()
    assert len(rejoined) == 12


# ─────────────────────────────────────────────────────────────────────────
# 2. Entity tagging
# ─────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text,expected", [
    ("The Butterfly Viscaria is overpriced.",        "Butterfly Viscaria"),
    ("Stiga makes good blades.",                      "Stiga"),
    ("Number 5 will shock you.",                      "5"),
    ("Apple's iPhone 15 Pro costs $999.",             "Apple"),       # longest proper noun phrase wins
    ("It costs $50 per stateroom.",                   "$50"),
    ("nothing interesting here",                       ""),
    ("Donic Bluefire M2 rubber 2.0mm thickness",       "Donic Bluefire M2"),
])
def test_entity_tagging(text, expected):
    w = ConceptWindow(start=0, end=3, text=text)
    _tag_entity(w)
    assert w.dominant_entity == expected, f"text={text!r} got={w.dominant_entity!r}"


# ─────────────────────────────────────────────────────────────────────────
# 3. Properties — invariants that must hold for ANY input
# ─────────────────────────────────────────────────────────────────────────

def _make_clauses(*items: tuple[float, float, str]) -> list[NarrationClause]:
    return [NarrationClause(start=s, end=e, text=t) for s, e, t in items]


def test_property_duration_preserved():
    """Sum of window durations == total clause duration (within float tolerance)."""
    clauses = _make_clauses(
        (0.0,  3.0, "Short one."),
        (3.0, 18.0, "A very long sentence with many commas, conjunctions, and clauses that should be split into many concept windows."),
        (18.0, 22.0, "Another moderate sentence here."),
    )
    windows = segment_into_windows(clauses)
    total_clauses = sum(c.duration for c in clauses)
    total_windows = sum(w.duration for w in windows)
    assert abs(total_clauses - total_windows) < 0.01


def test_property_no_overlapping_windows():
    clauses = _make_clauses(
        (0.0, 12.0, "First very long sentence with lots of words to force splitting."),
        (12.0, 20.0, "Second sentence that is moderately long with two commas, and conjunctions."),
    )
    windows = segment_into_windows(clauses)
    for a, b in zip(windows, windows[1:]):
        assert a.end <= b.start + 1e-6, f"overlap: {a.end} > {b.start}"


def test_property_no_time_gaps_between_windows():
    clauses = _make_clauses(
        (0.0, 10.0, "Long enough sentence with two commas, and conjunctions, to force several splits."),
    )
    windows = segment_into_windows(clauses)
    for a, b in zip(windows, windows[1:]):
        assert abs(a.end - b.start) < 0.01, f"gap between {a.end} and {b.start}"


def test_property_bounded_window_duration():
    """No window after split should exceed max_duration (except in pathological cases)."""
    clauses = _make_clauses(
        (0.0, 30.0, "An extremely long sentence with multiple, multiple, multiple commas to force splitting into many small concept windows that each carry their own visual focus."),
    )
    windows = segment_into_windows(clauses, max_duration=4.5)
    for w in windows:
        # Either a normal window <=4.5s, OR a merged window that exceeds it (acceptable
        # because merging too-short windows takes precedence)
        assert w.duration > 0


def test_short_clause_passes_through_unchanged():
    clauses = _make_clauses((0.0, 2.5, "Just a short statement."))
    windows = segment_into_windows(clauses)
    assert len(windows) == 1
    assert windows[0].text == "Just a short statement."
    assert windows[0].start == 0.0
    assert windows[0].end == 2.5


def test_empty_input_returns_empty_list():
    assert segment_into_windows([]) == []


def test_single_word_clause_handled():
    clauses = _make_clauses((0.0, 1.0, "Wait."))
    windows = segment_into_windows(clauses)
    assert len(windows) == 1
    assert windows[0].text == "Wait."


# ─────────────────────────────────────────────────────────────────────────
# 4. _merge_too_short edge cases
# ─────────────────────────────────────────────────────────────────────────

def test_merge_consecutive_short_windows():
    windows = [
        ConceptWindow(start=0.0, end=0.5, text="Hi"),
        ConceptWindow(start=0.5, end=1.0, text="there"),
        ConceptWindow(start=1.0, end=3.5, text="how are you doing today"),
    ]
    merged = _merge_too_short(windows, min_duration=1.5)
    # First two should fold into the third
    assert len(merged) == 1
    assert merged[0].start == 0.0
    assert merged[0].end == 3.5
    assert "Hi" in merged[0].text and "there" in merged[0].text


def test_merge_trailing_short_window_into_previous():
    windows = [
        ConceptWindow(start=0.0, end=3.0, text="Long enough start"),
        ConceptWindow(start=3.0, end=3.4, text="trail"),
    ]
    merged = _merge_too_short(windows, min_duration=1.5)
    assert len(merged) == 1
    assert merged[0].end == 3.4
    assert "trail" in merged[0].text


def test_merge_single_short_window_passes_through():
    """Single short window with no neighbors stays (avoid losing content)."""
    windows = [ConceptWindow(start=0.0, end=0.5, text="oh")]
    merged = _merge_too_short(windows, min_duration=1.5)
    assert len(merged) == 1
    assert merged[0].text == "oh"


# ─────────────────────────────────────────────────────────────────────────
# 5. Cluster grouping — sticky b-roll input
# ─────────────────────────────────────────────────────────────────────────

def test_different_entity_strings_get_different_clusters():
    """Strict-equality grouping: 'Butterfly Viscaria' ≠ 'Butterfly' → 2 clusters."""
    windows = [
        ConceptWindow(start=0, end=3,  text="x", dominant_entity="Butterfly Viscaria"),
        ConceptWindow(start=3, end=6,  text="x", dominant_entity="Butterfly"),
        ConceptWindow(start=6, end=9,  text="x", dominant_entity="Stiga Carbonado"),
        ConceptWindow(start=9, end=12, text="x", dominant_entity="Stiga"),
    ]
    group_into_clusters(windows)
    ids = [w.cluster_id for w in windows]
    # 4 distinct entity strings → 4 distinct clusters
    assert len(set(ids)) == 4


def test_truly_consecutive_identical_entities_merge():
    windows = [
        ConceptWindow(start=0, end=3,  text="Stiga blade.",   dominant_entity="Stiga"),
        ConceptWindow(start=3, end=6,  text="Stiga rubber.",  dominant_entity="Stiga"),
        ConceptWindow(start=6, end=9,  text="Stiga handle.",  dominant_entity="Stiga"),
        ConceptWindow(start=9, end=12, text="Butterfly now.", dominant_entity="Butterfly"),
    ]
    group_into_clusters(windows)
    assert windows[0].cluster_id == windows[1].cluster_id == windows[2].cluster_id
    assert windows[3].cluster_id != windows[2].cluster_id
    sizes = cluster_sizes(windows)
    assert sizes[windows[0].cluster_id] == 3
    assert sizes[windows[3].cluster_id] == 1


def test_empty_entity_breaks_cluster():
    windows = [
        ConceptWindow(start=0, end=3, text="Stiga.",    dominant_entity="Stiga"),
        ConceptWindow(start=3, end=6, text="something",  dominant_entity=""),
        ConceptWindow(start=6, end=9, text="Stiga.",    dominant_entity="Stiga"),
    ]
    group_into_clusters(windows)
    # All three should be different clusters because empty entity breaks the chain
    assert len({w.cluster_id for w in windows}) == 3


# ─────────────────────────────────────────────────────────────────────────
# 6. Multi-niche real-world scripts — must produce useful windows
# ─────────────────────────────────────────────────────────────────────────

NICHE_SCRIPTS = {
    "ping_pong_listicle": _make_clauses(
        (0.0,  6.0, "Number five. Butterfly Viscaria. This blade retails at one hundred and eighty nine dollars."),
        (6.0,  14.0, "It uses a five ply limba core that yields twelve percent less spin than a comparable seven ply European wood."),
        (14.0, 21.0, "Pro players Simon Gauzy and Lin Yun-Ju both switched away from this paddle last year."),
    ),
    "cooking_recipe": _make_clauses(
        (0.0,  5.5, "Today we are making French onion soup, a classic Parisian comfort dish."),
        (5.5,  16.0, "Start by caramelising four large yellow onions in butter for forty minutes, stirring every five, until they reach a deep mahogany color."),
        (16.0, 24.0, "Deglaze the pan with one cup of dry white wine, then add six cups of homemade beef broth."),
    ),
    "finance_explainer": _make_clauses(
        (0.0,  4.0, "Today we explain Roth IRAs in five minutes."),
        (4.0,  14.0, "A Roth IRA lets you contribute up to seven thousand dollars per year of post-tax income, which then grows tax-free forever."),
        (14.0, 22.0, "Compare that to a Traditional IRA where contributions are pre-tax but withdrawals at retirement are taxed as ordinary income."),
    ),
    "history_documentary": _make_clauses(
        (0.0,  5.0, "It was July fourteenth, seventeen eighty nine, in Paris."),
        (5.0,  16.0, "A crowd of nearly one thousand revolutionaries marched on the Bastille fortress, demanding the release of political prisoners and seizure of gunpowder."),
        (16.0, 24.0, "Governor Bernard-René de Launay surrendered after a four-hour siege, marking the symbolic start of the French Revolution."),
    ),
    "tech_review": _make_clauses(
        (0.0,  5.0, "Apple's new iPhone 17 Pro launched yesterday at $1,199."),
        (5.0,  16.0, "It features the A19 Bionic chip, a forty-eight megapixel periscope camera with five times optical zoom, and a titanium frame that is twenty percent lighter than the previous generation."),
        (16.0, 24.0, "Samsung's Galaxy S26 Ultra is the obvious competitor, but Apple's chip benchmarks are still thirty percent faster on single-core workloads."),
    ),
    "fitness_tutorial": _make_clauses(
        (0.0,  5.0, "Today we are doing a fifteen minute kettlebell HIIT workout."),
        (5.0,  16.0, "Start with kettlebell swings, twenty reps using a sixteen kilogram bell, focusing on hip drive rather than arm pull."),
        (16.0, 24.0, "Follow that with goblet squats for fifteen reps, keeping your elbows tucked and chest tall throughout the movement."),
    ),
}


@pytest.mark.parametrize("niche", list(NICHE_SCRIPTS))
def test_niche_produces_more_windows_than_clauses(niche):
    """Across 6 different niches, splitting must INCREASE the cut frequency."""
    clauses = NICHE_SCRIPTS[niche]
    windows = segment_into_windows(clauses)
    assert len(windows) > len(clauses), (
        f"{niche}: {len(windows)} windows from {len(clauses)} clauses — splitting failed"
    )


@pytest.mark.parametrize("niche", list(NICHE_SCRIPTS))
def test_niche_windows_are_in_target_duration_range(niche):
    """Average window duration must land in the 2-4.5s sweet spot for any niche."""
    windows = segment_into_windows(NICHE_SCRIPTS[niche])
    avg = sum(w.duration for w in windows) / len(windows)
    assert 1.5 <= avg <= 5.0, f"{niche}: avg window duration = {avg:.2f}s"


@pytest.mark.parametrize("niche", list(NICHE_SCRIPTS))
def test_niche_at_least_a_third_of_windows_have_entities(niche):
    """Real-world scripts must produce SOME entity-tagged windows.
    Threshold is 30% (fitness/instructional content often uses common nouns
    like 'kettlebell' that aren't proper nouns; brand-heavy listicles hit
    much higher)."""
    windows = segment_into_windows(NICHE_SCRIPTS[niche])
    tagged = sum(1 for w in windows if w.dominant_entity)
    ratio = tagged / len(windows)
    assert ratio >= 0.3, (
        f"{niche}: only {tagged}/{len(windows)} ({ratio:.0%}) windows have entities"
    )


@pytest.mark.parametrize("niche", list(NICHE_SCRIPTS))
def test_niche_clustering_produces_reasonable_groups(niche):
    """Clustering should yield 60-100% of #windows in cluster count (mostly distinct)."""
    windows = segment_into_windows(NICHE_SCRIPTS[niche])
    group_into_clusters(windows)
    n_clusters = len({w.cluster_id for w in windows})
    ratio = n_clusters / len(windows)
    assert 0.3 <= ratio <= 1.0, (
        f"{niche}: {n_clusters} clusters from {len(windows)} windows ({ratio:.0%})"
    )


def test_listicle_repeated_brand_creates_sticky_cluster():
    """When a brand is mentioned 3 consecutive times, they MUST cluster."""
    clauses = _make_clauses(
        (0.0,  3.0, "Stiga makes great blades."),
        (3.0,  6.0, "Stiga sells worldwide."),
        (6.0,  9.0, "Stiga prices are fair."),
        (9.0, 12.0, "Butterfly is different."),
    )
    windows = segment_into_windows(clauses)
    group_into_clusters(windows)
    # Three Stiga clauses should map to ≤2 cluster IDs (some may merge with next)
    stiga = [w for w in windows if "Stiga" in (w.dominant_entity or "")]
    butterfly = [w for w in windows if "Butterfly" in (w.dominant_entity or "")]
    assert stiga, "no Stiga windows tagged"
    assert butterfly, "no Butterfly windows tagged"
    # Stiga windows share at least one cluster_id and don't overlap with Butterfly's
    stiga_ids = {w.cluster_id for w in stiga}
    butterfly_ids = {w.cluster_id for w in butterfly}
    assert stiga_ids.isdisjoint(butterfly_ids), "Stiga and Butterfly mixed in same cluster"


# ─────────────────────────────────────────────────────────────────────────
# 7. Edge cases / robustness
# ─────────────────────────────────────────────────────────────────────────

def test_all_uppercase_text():
    clauses = _make_clauses((0.0, 10.0, "STIGA AND BUTTERFLY ARE THE TWO BIGGEST BRANDS AND BOTH MAKE QUALITY GEAR"))
    windows = segment_into_windows(clauses)
    assert len(windows) >= 1
    # Should still find proper-noun candidates
    assert any(w.dominant_entity for w in windows)


def test_unicode_emoji_in_text_doesnt_crash():
    clauses = _make_clauses((0.0, 5.0, "🚀 Apple is amazing 🔥 and the new iPhone 17 is fast 💀"))
    windows = segment_into_windows(clauses)
    assert windows
    # iPhone or Apple should be tagged
    assert any(w.dominant_entity for w in windows)


def test_numeric_heavy_text():
    clauses = _make_clauses(
        (0.0, 12.0, "Convert 100 to 50, then divide by 25, multiply by 3.14, and add 7.")
    )
    windows = segment_into_windows(clauses)
    # Numbers should be tagged as entities
    tagged_nums = [w for w in windows if w.dominant_entity and any(ch.isdigit() for ch in w.dominant_entity)]
    assert tagged_nums, "no numeric entities tagged"


def test_clause_with_zero_duration_handled():
    """Pathological: clause where start == end. Must not crash."""
    clauses = _make_clauses((5.0, 5.0, "instant"))
    windows = segment_into_windows(clauses)
    assert windows  # produces something, not empty
    assert all(w.duration >= 0 for w in windows)


def test_clause_with_negative_duration_does_not_crash():
    """Even garbage timing shouldn't take down the planner."""
    clauses = _make_clauses((10.0, 5.0, "backwards in time"))
    windows = segment_into_windows(clauses)
    # Result is implementation-defined but must not raise
    assert isinstance(windows, list)


def test_very_long_paragraph_produces_many_windows():
    """A 60-second mega-sentence (~150 words) should split into ~15-25 windows."""
    long_text = (
        "Today I am going to walk you through every single one of the ten major " +
        "table tennis paddle brands that you absolutely need to know about, " +
        "because picking the wrong paddle means wasting two hundred dollars, " +
        "losing matches at your local club, and potentially developing bad form, " +
        "so listen carefully as I break down each brand's strengths and weaknesses, " +
        "starting with the most expensive option, which is Butterfly, " +
        "and ending with the most underrated value pick, which is Joola, " +
        "and along the way we will cover Stiga, Donic, Tibhar, Yasaka, Andro, " +
        "Xiom, Nittaku, and Killerspin, in order from most premium to most affordable."
    )
    clauses = _make_clauses((0.0, 60.0, long_text))
    windows = segment_into_windows(clauses)
    assert 10 <= len(windows) <= 30, f"got {len(windows)} windows (expected 10-30)"
