"""Phase 6.3 — Sticky b-roll pool allocator: advanced multi-niche tests.

Covers:
  • ClusterPool contract (size, is_empty, primary-then-secondary consumption)
  • Builder integration with stubbed base/brand searches
  • Brand classifier heuristic (positive + negative cases)
  • Primary brand asset wins over generic when entity is a brand
  • Non-brand entities skip brand_search
  • Anti-repeat: same cluster never returns the same candidate twice
  • Pool exhaustion → fallback search fires
  • All exceptions in search funcs are swallowed (no crash)
  • Multi-niche: 6 niches × 3 clusters each = stress allocation
"""
from __future__ import annotations
from dataclasses import dataclass
from unittest import mock

import pytest

from app.services.sync.concept import (
    ConceptWindow, segment_into_windows, group_into_clusters,
)
from app.services.sync.pipeline import NarrationClause
from app.services.sync.ranker import Candidate
from app.services.sync.sticky import (
    ClusterPool,
    StickyAllocator,
    _default_brand_classifier,
    _representative_entity,
    build_sticky_pools,
)


# ─────────────────────────────────────────────────────────────────────────
# 1. ClusterPool contract
# ─────────────────────────────────────────────────────────────────────────

def _cand(sid: str, source: str = "pexels", title: str = "") -> Candidate:
    return Candidate(source=source, source_id=sid,
                     url=f"http://x/{sid}", title=title or f"clip {sid}")


def test_cluster_pool_size_counts_primary_plus_secondaries():
    pool = ClusterPool(cluster_id=1, primary=_cand("p"),
                       secondaries=[_cand("s1"), _cand("s2")])
    assert pool.size == 3
    assert not pool.is_empty


def test_cluster_pool_empty_is_empty():
    assert ClusterPool(cluster_id=1).is_empty
    assert ClusterPool(cluster_id=1).size == 0


# ─────────────────────────────────────────────────────────────────────────
# 2. _default_brand_classifier
# ─────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("entity,expected", [
    ("Butterfly", True),
    ("Stiga Carbonado", True),
    ("Le Creuset", True),
    ("Apple", True),
    # rejections
    ("$50", False),
    ("12%", False),
    ("5", False),
    ("2024", False),
    ("", False),
    ("the apple", False),                       # lowercase first
    ("Butterfly Viscaria Pro Tournament Edition", False),  # too many words
    ("Apple Watch 9", False),                   # contains digit
])
def test_brand_classifier(entity, expected):
    assert _default_brand_classifier(entity) is expected, f"failed: {entity!r}"


# ─────────────────────────────────────────────────────────────────────────
# 3. _representative_entity picks most-frequent in cluster
# ─────────────────────────────────────────────────────────────────────────

def test_representative_picks_most_frequent_entity():
    group = [
        ConceptWindow(start=0, end=2, text="x", dominant_entity="Stiga"),
        ConceptWindow(start=2, end=4, text="x", dominant_entity="Stiga"),
        ConceptWindow(start=4, end=6, text="x", dominant_entity="Carbonado"),
    ]
    assert _representative_entity(group) == "Stiga"


def test_representative_breaks_ties_by_length():
    group = [
        ConceptWindow(start=0, end=2, text="x", dominant_entity="Stiga"),
        ConceptWindow(start=2, end=4, text="x", dominant_entity="Stiga Carbonado"),
    ]
    # Both occur once; longer wins as more specific
    assert _representative_entity(group) == "Stiga Carbonado"


def test_representative_empty_returns_empty():
    group = [ConceptWindow(start=0, end=2, text="x", dominant_entity="")]
    assert _representative_entity(group) == ""


# ─────────────────────────────────────────────────────────────────────────
# 4. Builder + brand priority
# ─────────────────────────────────────────────────────────────────────────

def test_brand_entity_uses_brand_search_for_primary():
    base_calls, brand_calls = [], []

    def base(q, n):
        base_calls.append(q)
        return [_cand(f"base-{q}-{i}") for i in range(n)]

    def brand(q, n):
        brand_calls.append(q)
        return [_cand(f"brand-{q}-0", source="wikimedia_brand")]

    windows = [ConceptWindow(start=0, end=3, text="x",
                              dominant_entity="Butterfly",
                              cluster_id=1)]
    alloc = build_sticky_pools(windows, base_search=base, brand_search=brand)
    pool = alloc.pools[1]
    assert pool.primary is not None
    assert pool.primary.source == "wikimedia_brand"  # brand asset wins
    assert "Butterfly" in brand_calls
    assert "Butterfly" in base_calls                 # secondaries still populated


def test_non_brand_entity_skips_brand_search():
    base_calls, brand_calls = [], []

    def base(q, n):
        base_calls.append(q)
        return [_cand(f"base-{q}-{i}") for i in range(n)]

    def brand(q, n):
        brand_calls.append(q)
        return [_cand("never")]

    # "$50" is not a brand
    windows = [ConceptWindow(start=0, end=3, text="x",
                              dominant_entity="$50", cluster_id=7)]
    build_sticky_pools(windows, base_search=base, brand_search=brand)
    assert brand_calls == []  # never called for non-brand
    assert base_calls         # but generic was called


def test_empty_entity_creates_no_pool():
    def base(q, n): return [_cand("x")]
    def brand(q, n): return []
    windows = [ConceptWindow(start=0, end=3, text="x",
                              dominant_entity="", cluster_id=1)]
    alloc = build_sticky_pools(windows, base_search=base, brand_search=brand)
    assert alloc.pools == {}


# ─────────────────────────────────────────────────────────────────────────
# 5. Allocator: anti-repeat + primary-then-secondary order
# ─────────────────────────────────────────────────────────────────────────

def test_next_for_returns_primary_then_secondaries_in_order():
    pool = ClusterPool(cluster_id=1, primary=_cand("P"),
                       secondaries=[_cand("S1"), _cand("S2")])
    alloc = StickyAllocator(pools={1: pool})
    w = ConceptWindow(start=0, end=2, text="x", cluster_id=1)
    assert alloc.next_for(w).source_id == "P"
    assert alloc.next_for(w).source_id == "S1"
    assert alloc.next_for(w).source_id == "S2"


def test_next_for_skips_already_consumed_secondaries():
    pool = ClusterPool(cluster_id=1, primary=None,
                       secondaries=[_cand("A"), _cand("B"), _cand("C")],
                       consumed_ids={"B"})
    alloc = StickyAllocator(pools={1: pool})
    w = ConceptWindow(start=0, end=2, text="x", cluster_id=1)
    assert alloc.next_for(w).source_id == "A"
    # B is skipped
    assert alloc.next_for(w).source_id == "C"


def test_next_for_uses_fallback_when_pool_exhausted():
    fallback_calls = []
    def fb(q, n):
        fallback_calls.append(q)
        return [_cand(f"FB-{q}")]
    alloc = StickyAllocator(pools={}, fallback_search=fb)
    w = ConceptWindow(start=0, end=2, text="x", dominant_entity="Apple", cluster_id=99)
    c = alloc.next_for(w)
    assert c is not None
    assert c.source_id == "FB-Apple"
    assert fallback_calls == ["Apple"]


def test_next_for_returns_none_when_no_pool_no_fallback():
    alloc = StickyAllocator(pools={})
    w = ConceptWindow(start=0, end=2, text="x", cluster_id=99)
    assert alloc.next_for(w) is None


def test_next_for_swallows_fallback_exceptions():
    def fb_bad(q, n):
        raise ConnectionError("network down")
    alloc = StickyAllocator(pools={}, fallback_search=fb_bad)
    w = ConceptWindow(start=0, end=2, text="x", dominant_entity="x", cluster_id=1)
    assert alloc.next_for(w) is None


def test_next_for_uses_text_when_no_dominant_entity():
    captured_q = []
    def fb(q, n):
        captured_q.append(q)
        return [_cand("FB")]
    alloc = StickyAllocator(pools={}, fallback_search=fb)
    w = ConceptWindow(start=0, end=2,
                       text="some narration sentence content",
                       dominant_entity="", cluster_id=1)
    alloc.next_for(w)
    # Query falls back to the window text
    assert "narration" in captured_q[0]


# ─────────────────────────────────────────────────────────────────────────
# 6. Builder: search exceptions don't crash
# ─────────────────────────────────────────────────────────────────────────

def test_builder_swallows_brand_search_exception():
    def base(q, n): return [_cand(f"B{q}")]
    def brand_bad(q, n): raise RuntimeError("API down")
    windows = [ConceptWindow(start=0, end=2, text="x",
                              dominant_entity="Apple", cluster_id=1)]
    alloc = build_sticky_pools(windows, base_search=base, brand_search=brand_bad)
    assert 1 in alloc.pools
    # base provided the primary even though brand failed
    assert alloc.pools[1].primary is not None


def test_builder_swallows_base_search_exception():
    def base_bad(q, n): raise TimeoutError("timeout")
    def brand(q, n): return [_cand("B", source="wikimedia_brand")]
    windows = [ConceptWindow(start=0, end=2, text="x",
                              dominant_entity="Apple", cluster_id=1)]
    alloc = build_sticky_pools(windows, base_search=base_bad, brand_search=brand)
    # Brand alone populates the pool
    assert alloc.pools[1].primary is not None
    assert alloc.pools[1].primary.source == "wikimedia_brand"


def test_builder_with_no_searches_returns_empty_alloc():
    windows = [ConceptWindow(start=0, end=2, text="x",
                              dominant_entity="Apple", cluster_id=1)]
    alloc = build_sticky_pools(windows)
    assert alloc.pools == {}


# ─────────────────────────────────────────────────────────────────────────
# 7. End-to-end multi-niche: anti-repeat across consecutive same-cluster windows
# ─────────────────────────────────────────────────────────────────────────

NICHE_LISTICLE_SCRIPTS = {
    "ping_pong": [
        "Number five. Butterfly Viscaria. Butterfly's blade retails at $189.",
        "Number four. Stiga Carbonado. Stiga sells worldwide.",
        "Number three. Donic Bluefire. Donic rubber is competitive.",
    ],
    "tech": [
        "Number five. Apple iPhone 17 Pro. Apple's chip is fast.",
        "Number four. Samsung Galaxy S26. Samsung competes hard.",
        "Number three. Google Pixel 10. Google focuses on AI.",
    ],
    "cars": [
        "Number five. Toyota Camry. Toyota dominates reliability.",
        "Number four. Honda Civic. Honda is affordable.",
        "Number three. Ford F-150. Ford rules trucks.",
    ],
    "kitchen": [
        "Number five. Le Creuset Dutch oven. Le Creuset is French.",
        "Number four. Lodge cast iron. Lodge is American.",
        "Number three. KitchenAid mixer. KitchenAid is the standard.",
    ],
    "watches": [
        "Number five. Rolex Submariner. Rolex holds value.",
        "Number four. Omega Speedmaster. Omega has heritage.",
        "Number three. Seiko SKX. Seiko is reliable.",
    ],
    "guitars": [
        "Number five. Fender Stratocaster. Fender is iconic.",
        "Number four. Gibson Les Paul. Gibson is classic.",
        "Number three. Martin D-28. Martin is the dreadnought king.",
    ],
}


@pytest.mark.parametrize("niche", list(NICHE_LISTICLE_SCRIPTS))
def test_niche_listicle_each_item_gets_its_own_cluster(niche):
    """For 6 different niches, 3 listicle items must end up in 3 distinct
    clusters with 3 different anchor clips — no cross-contamination."""
    texts = NICHE_LISTICLE_SCRIPTS[niche]
    # Build clauses (12s each)
    clauses = [
        NarrationClause(start=i * 12.0, end=(i + 1) * 12.0, text=t)
        for i, t in enumerate(texts)
    ]
    windows = segment_into_windows(clauses)
    group_into_clusters(windows)

    # Stub: brand-aware → returns a unique anchor PER QUERY (so each brand
    # gets its own anchor); base → returns 4 generic secondaries per query
    def brand(q, n):
        return [_cand(f"anchor-{q}", source="wikimedia_brand", title=f"{q} logo")]

    def base(q, n):
        return [_cand(f"sec-{q}-{i}", source="pexels", title=f"{q} variation {i}")
                for i in range(n)]

    alloc = build_sticky_pools(windows, base_search=base, brand_search=brand)
    # Each window gets a clip
    assigned: list[Candidate] = []
    for w in windows:
        c = alloc.next_for(w)
        if c is not None:
            assigned.append(c)

    # 1) Every window got a clip
    assert len(assigned) == len(windows), \
        f"{niche}: {len(assigned)} / {len(windows)} windows assigned"

    # 2) Anchor clips are distinct (one per brand, one per item)
    anchors = [c for c in assigned if c.source == "wikimedia_brand"]
    assert len({c.source_id for c in anchors}) == len(anchors), \
        f"{niche}: duplicate anchors"

    # 3) No clip is used twice GLOBALLY (anti-repeat across the whole listicle)
    all_ids = [c.source_id for c in assigned]
    assert len(set(all_ids)) == len(all_ids), \
        f"{niche}: clip reused — got {len(all_ids)} assignments, {len(set(all_ids))} unique"


def test_long_listicle_pool_exhaustion_falls_back():
    """A cluster with more windows than its pool size must use the fallback
    for the overflow without crashing."""
    # 6 windows all in cluster 1 — pool will only have ~4 clips
    windows = [
        ConceptWindow(start=i * 2.0, end=(i + 1) * 2.0, text=f"point {i}",
                       dominant_entity="Apple", cluster_id=1)
        for i in range(6)
    ]
    def base(q, n):
        return [_cand(f"A{i}") for i in range(min(3, n))]  # only 3 generic
    def brand(q, n):
        return [_cand("ANCHOR", source="wikimedia_brand")]

    alloc = build_sticky_pools(windows, base_search=base, brand_search=brand)
    consumed_main = 0
    fallback_hits = 0

    def fb(q, n):
        nonlocal fallback_hits
        fallback_hits += 1
        return [_cand(f"FB-{fallback_hits}")]
    alloc.fallback_search = fb

    seen_ids: set[str] = set()
    for w in windows:
        c = alloc.next_for(w)
        assert c is not None, f"window {w.start}s unassigned"
        seen_ids.add(c.source_id)
        if c.source_id.startswith("FB-"):
            consumed_main += 0
        else:
            consumed_main += 1

    # All 6 windows got distinct clips
    assert len(seen_ids) == 6
    # Fallback must have triggered for the overflow
    assert fallback_hits >= 1, "fallback never triggered despite pool exhaustion"


def test_two_adjacent_clusters_dont_share_clips():
    """Cluster A's primary must not show up in cluster B's allocation."""
    windows = [
        ConceptWindow(start=0,  end=2, text="x", dominant_entity="Apple",  cluster_id=1),
        ConceptWindow(start=2,  end=4, text="x", dominant_entity="Apple",  cluster_id=1),
        ConceptWindow(start=4,  end=6, text="x", dominant_entity="Stiga",  cluster_id=2),
        ConceptWindow(start=6,  end=8, text="x", dominant_entity="Stiga",  cluster_id=2),
    ]
    def brand(q, n):
        return [_cand(f"ANCHOR-{q}", source="wikimedia_brand")]
    def base(q, n):
        return [_cand(f"SEC-{q}-{i}", source="pexels") for i in range(n)]

    alloc = build_sticky_pools(windows, base_search=base, brand_search=brand)
    assigned: list[Candidate] = []
    for w in windows:
        c = alloc.next_for(w)
        assert c is not None
        assigned.append(c)

    # First 2 windows are Apple's pool, last 2 are Stiga's
    apple_ids = {c.source_id for c in assigned[:2]}
    stiga_ids = {c.source_id for c in assigned[2:]}
    assert apple_ids.isdisjoint(stiga_ids), \
        f"clusters bled into each other: apple={apple_ids} stiga={stiga_ids}"
    # Each cluster's anchor is named after its own brand
    assert any("Apple" in c.source_id for c in assigned[:2])
    assert any("Stiga" in c.source_id for c in assigned[2:])
