"""Sticky b-roll allocation — keep the same visual identity across consecutive
concept windows that share a dominant entity.

WHY: a listicle item like "Number 5. Stiga Carbonado." may take ~3 minutes
to narrate, split into ~30 concept windows. Without coordination, each
window gets its own clip → visual chaos. With sticky allocation:
  • One PRIMARY clip is reserved at cluster-start (preferably a brand asset)
  • Subsequent windows in the same cluster pull from a small SECONDARY
    pool — thematic variations on the same entity, never the primary again
  • Anti-repeat is local to the cluster (not global) so the next item's
    cluster is free to reuse the same visual language

API contract:
  build_sticky_pools(windows, *, base_search, brand_search) -> StickyAllocator
  alloc.next_for(window) -> Candidate | None
"""
from __future__ import annotations
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable, Iterable

from .concept import ConceptWindow
from .ranker import Candidate


# Type aliases mirror retriever.py's SourceFn
SearchFn = Callable[[str, int], list[Candidate]]
"""(query, max_results) -> list[Candidate]. Hits *one* source at a time."""

BrandSearchFn = SearchFn
"""Brand-aware search variant (e.g. wikimedia_brand)."""


@dataclass
class ClusterPool:
    """Per-cluster pool: 1 primary + N secondaries, drained as windows consume."""
    cluster_id: int
    entity: str = ""
    primary: Candidate | None = None
    secondaries: list[Candidate] = field(default_factory=list)
    consumed_ids: set[str] = field(default_factory=set)   # local anti-repeat

    @property
    def size(self) -> int:
        return (1 if self.primary else 0) + len(self.secondaries)

    @property
    def is_empty(self) -> bool:
        return self.primary is None and not self.secondaries


@dataclass
class StickyAllocator:
    """Holds a ClusterPool per cluster_id + the fallback search functions
    used when a pool exhausts."""
    pools: dict[int, ClusterPool] = field(default_factory=dict)
    fallback_search: SearchFn | None = None
    fallback_query_template: str = "{entity}"

    def next_for(self, window: ConceptWindow) -> Candidate | None:
        """Return the next clip for `window`. Tries (in order):
        1. cluster primary (if untouched)
        2. cluster secondaries (FIFO)
        3. fallback search using the window's text
        """
        pool = self.pools.get(window.cluster_id)
        if pool is not None and not pool.is_empty:
            if pool.primary is not None and pool.primary.source_id not in pool.consumed_ids:
                c = pool.primary
                pool.consumed_ids.add(c.source_id)
                pool.primary = None     # primary is consumed on first use
                return c
            # Drain secondaries in order
            while pool.secondaries:
                c = pool.secondaries.pop(0)
                if c.source_id in pool.consumed_ids:
                    continue
                pool.consumed_ids.add(c.source_id)
                return c
        # Fallback: ad-hoc search on the window's free text
        if self.fallback_search:
            try:
                results = self.fallback_search(
                    self.fallback_query_template.format(
                        entity=window.dominant_entity or window.text[:60]
                    ),
                    3,
                ) or []
                for c in results:
                    return c
            except Exception:
                return None
        return None


# ─────────────────────────────────────────────────────────────────────────
# Pool builder
# ─────────────────────────────────────────────────────────────────────────

def build_sticky_pools(
    windows: Iterable[ConceptWindow],
    *,
    base_search: SearchFn | None = None,
    brand_search: BrandSearchFn | None = None,
    brand_classifier: Callable[[str], bool] | None = None,
    candidates_per_cluster: int = 4,
) -> StickyAllocator:
    """Pre-compute the per-cluster pools.

    Algorithm:
      1. Bucket windows by cluster_id.
      2. For each cluster:
         a. Decide if the dominant entity is a brand
            (default heuristic: capitalized, contains no digits, short).
         b. If brand AND brand_search provided → call brand_search(entity).
            That yields the PRIMARY (most relevant brand asset).
         c. Then call base_search(entity) for the SECONDARY pool
            (thematic variations).
         d. Cluster keeps the first as primary, the rest as secondaries.

    Failure modes (network down, empty results) collapse to fallback at
    allocation time.
    """
    alloc = StickyAllocator(fallback_search=base_search)
    by_cluster: dict[int, list[ConceptWindow]] = OrderedDict()
    for w in windows:
        by_cluster.setdefault(w.cluster_id, []).append(w)

    is_brand = brand_classifier or _default_brand_classifier

    for cid, group in by_cluster.items():
        entity = _representative_entity(group)
        if not entity:
            continue
        pool = ClusterPool(cluster_id=cid, entity=entity)

        # 1) Brand primary (if applicable)
        if brand_search and is_brand(entity):
            try:
                brand_hits = brand_search(entity, 2) or []
                if brand_hits:
                    pool.primary = brand_hits[0]
                    # Brand-aware sometimes returns more than 1 — keep extras as secondaries
                    for extra in brand_hits[1:]:
                        if extra.source_id not in {pool.primary.source_id}:
                            pool.secondaries.append(extra)
            except Exception:
                pass

        # 2) Generic thematic pool
        if base_search:
            try:
                generic = base_search(entity, candidates_per_cluster) or []
            except Exception:
                generic = []
            for c in generic:
                if pool.primary is None:
                    pool.primary = c
                    continue
                if c.source_id == pool.primary.source_id:
                    continue
                pool.secondaries.append(c)

        if not pool.is_empty:
            alloc.pools[cid] = pool

    return alloc


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────

def _representative_entity(group: list[ConceptWindow]) -> str:
    """Pick the most common non-empty `dominant_entity` across windows
    in a cluster. Falls back to the first window's entity."""
    seen: dict[str, int] = {}
    for w in group:
        e = (w.dominant_entity or "").strip()
        if e:
            seen[e] = seen.get(e, 0) + 1
    if not seen:
        return ""
    # Most frequent, then longest (proxy for specificity)
    return max(seen.items(), key=lambda kv: (kv[1], len(kv[0])))[0]


def _default_brand_classifier(entity: str) -> bool:
    """Heuristic: 1–3 words, no digits, starts capitalized.
    Catches: 'Butterfly', 'Stiga Carbonado', 'Le Creuset'.
    Rejects:  '$50', '12%', '5', '2024'."""
    if not entity:
        return False
    if any(ch.isdigit() for ch in entity):
        return False
    parts = entity.split()
    if not (1 <= len(parts) <= 3):
        return False
    if not entity[0].isupper():
        return False
    return True
