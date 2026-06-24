"""Concept-window planner — the upgrade from sentence-level to 2-4s cuts.

A long sentence ("The Butterfly Viscaria blade retails at $189 and uses
a 5-ply limba core that yields 12% less spin than a comparable 7-ply...")
gets one clip in the legacy pipeline. Viderush-level requires it to
become ~5 windows, each with its own clip:

  [0-2s]  "The Butterfly Viscaria blade"  → close-up paddle / Butterfly logo
  [2-4s]  "retails at $189"               → price tag / dollar bills
  [4-6s]  "uses a 5-ply limba core"        → blade cross-section
  [6-9s]  "yields 12% less spin"           → spin test slow-mo
  [9-12s] "than a comparable 7-ply"        → comparison shot

This module:
  1. Takes the TTS sentence boundaries (which we already have)
  2. Splits each into 2-4s windows on natural breath points / clause boundaries
  3. Identifies the dominant entity per window (proper noun / number / verb-phrase)
  4. Groups consecutive windows that share an entity (for sticky b-roll downstream)

Backends:
  • Lexical (no deps) — regex on commas, conjunctions, prepositions
  • LLM (optional) — Gemini chain produces semantic windows + dominant entities

The lexical backend is the default — fast, deterministic, no network.
The LLM backend kicks in when the user opts in for max quality.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from .pipeline import NarrationClause


# Approximate speaking rate of Edge-TTS at default rate (-5%)
_WORDS_PER_SECOND = 2.6


@dataclass
class ConceptWindow:
    """A 2-4s slice of narration with its dominant visual concept."""
    start: float
    end: float
    text: str
    dominant_entity: str = ""       # main noun / brand / proper noun
    related_entities: list[str] = field(default_factory=list)
    mood: str = "neutral"           # neutral / dramatic / analytical / urgent
    cluster_id: int = -1            # filled by group_into_clusters()

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


# ─────────────────────────────────────────────────────────────────────────
# Lexical (no-deps) segmenter
# ─────────────────────────────────────────────────────────────────────────

_BREATH_RX = re.compile(
    r"[,;:]\s+|"
    r"\s+(?:and|but|or|yet|so|because|since|while|although|"
    r"that|which|when|where|whereas)\s+",
    re.I,
)
_SENTENCE_END_RX = re.compile(r"[.!?]\s*$")
_PROPER_NOUN_RX = re.compile(r"\b([A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+)*)\b")
_NUMBER_RX = re.compile(r"(\$\d+(?:[.,]\d+)?|\d+(?:[.,]\d+)?%?)")

# Words that look capitalized at sentence start but aren't proper nouns.
# These get stripped from candidate entities so we surface the real noun.
_FUNCTION_WORDS = frozenset({
    "the","a","an","this","that","these","those",
    "it","he","she","we","they","you","i","my","your","his","her","our","their",
    "today","yesterday","tomorrow","now","then","here","there",
    "number","numbers","item","step","tip","fact","point","rule",
    "and","but","or","so","yet","because","since","while","although",
    "first","second","third","fourth","fifth","sixth","seventh","eighth","ninth","tenth",
    "another","other","such","both","each","every","any","some","most","many","few",
    "however","therefore","moreover","furthermore","also","still","just",
    "well","ok","right","look","listen","wait","stop","go",
})


def segment_into_windows(
    clauses: Sequence[NarrationClause],
    *,
    min_duration: float = 1.5,
    target_duration: float = 3.0,
    max_duration: float = 4.5,
) -> list[ConceptWindow]:
    """Split sentence-level clauses into 1.5-4.5s concept windows.

    Algorithm:
      1. For each clause:
         a) If it's already short enough (≤max_duration), keep it.
         b) Else: split at breath points (commas, conjunctions).
         c) Each split piece gets a proportional time slice based on word count.
         d) Pieces shorter than min_duration get merged with the next piece.
      2. Tag each window with its dominant entity (proper noun > number > first noun).
    """
    out: list[ConceptWindow] = []
    for c in clauses:
        if c.duration <= max_duration:
            out.append(_window_from_clause(c))
            continue
        # Split this clause into sub-windows
        pieces = _split_lexical(c.text)
        if len(pieces) == 1:
            # No natural breath points — fall back to word-count slicing
            pieces = _split_by_word_count(c.text, target_duration)
        sub_windows = _distribute_time(pieces, c.start, c.end)
        sub_windows = _merge_too_short(sub_windows, min_duration)
        out.extend(sub_windows)
    return out


def _window_from_clause(c: NarrationClause) -> ConceptWindow:
    w = ConceptWindow(start=c.start, end=c.end, text=c.text)
    _tag_entity(w)
    return w


def _split_lexical(text: str) -> list[str]:
    """Split on natural breath points; pieces include their separator's terminator."""
    pieces = _BREATH_RX.split(text)
    return [p.strip() for p in pieces if p and p.strip()]


def _split_by_word_count(text: str, target_seconds: float) -> list[str]:
    """When no breath points exist, slice every N words."""
    words = text.split()
    n_per_piece = max(3, int(target_seconds * _WORDS_PER_SECOND))
    pieces = []
    for i in range(0, len(words), n_per_piece):
        pieces.append(" ".join(words[i : i + n_per_piece]))
    return pieces or [text]


def _distribute_time(pieces: list[str], start: float, end: float) -> list[ConceptWindow]:
    """Allocate time to each piece proportionally to its character length."""
    total_chars = sum(len(p) for p in pieces) or 1
    total_time = max(0.0, end - start)
    windows: list[ConceptWindow] = []
    cursor = start
    for i, p in enumerate(pieces):
        # Last piece gets the leftover to avoid float drift
        if i == len(pieces) - 1:
            piece_end = end
        else:
            piece_end = cursor + total_time * (len(p) / total_chars)
        w = ConceptWindow(start=cursor, end=piece_end, text=p)
        _tag_entity(w)
        windows.append(w)
        cursor = piece_end
    return windows


def _merge_too_short(windows: list[ConceptWindow],
                      min_duration: float) -> list[ConceptWindow]:
    """Merge any window <min_duration into the following one (or previous if it's last)."""
    if not windows:
        return windows
    out: list[ConceptWindow] = []
    pending: ConceptWindow | None = None
    for w in windows:
        if pending is not None:
            w = ConceptWindow(
                start=pending.start, end=w.end,
                text=f"{pending.text} {w.text}".strip(),
            )
            _tag_entity(w)
            pending = None
        if w.duration < min_duration:
            pending = w
            continue
        out.append(w)
    if pending is not None:
        if out:
            last = out[-1]
            merged = ConceptWindow(
                start=last.start, end=pending.end,
                text=f"{last.text} {pending.text}".strip(),
            )
            _tag_entity(merged)
            out[-1] = merged
        else:
            out.append(pending)
    return out


def _strip_function_word_prefix(candidate: str) -> str:
    """Remove leading determiners/pronouns/etc. (e.g. 'The Butterfly' → 'Butterfly')."""
    tokens = candidate.split()
    while tokens and tokens[0].lower() in _FUNCTION_WORDS:
        tokens.pop(0)
    return " ".join(tokens)


def _tag_entity(w: ConceptWindow) -> None:
    """Pick the most salient noun in the window as the dominant entity.
    Priority: longest cleaned proper-noun phrase > number > nothing."""
    proper_raw = _PROPER_NOUN_RX.findall(w.text)
    # Strip leading function words from each candidate
    proper = [c for c in (_strip_function_word_prefix(p) for p in proper_raw) if c]
    if proper:
        # Sort: longest non-empty cleaned candidate wins
        proper.sort(key=len, reverse=True)
        w.dominant_entity = proper[0]
        w.related_entities = [p for p in proper[1:5] if p and p != proper[0]]
        return
    nums = _NUMBER_RX.findall(w.text)
    if nums:
        w.dominant_entity = nums[0]


# ─────────────────────────────────────────────────────────────────────────
# Sticky cluster grouping
# ─────────────────────────────────────────────────────────────────────────

def group_into_clusters(windows: list[ConceptWindow]) -> list[ConceptWindow]:
    """Mark consecutive windows that share a dominant entity with the same
    `cluster_id`. The retriever can then reserve ONE primary clip for the
    whole cluster and fill secondary windows with thematic variations.

    Returns the SAME list (mutated) for convenience.
    """
    if not windows:
        return windows
    cluster_id = 0
    prev_entity = ""
    for w in windows:
        e = (w.dominant_entity or "").lower()
        if not e or e != prev_entity:
            cluster_id += 1
        w.cluster_id = cluster_id
        prev_entity = e
    return windows


def cluster_sizes(windows: list[ConceptWindow]) -> dict[int, int]:
    """Helper: returns {cluster_id: count_of_windows}."""
    sizes: dict[int, int] = {}
    for w in windows:
        sizes[w.cluster_id] = sizes.get(w.cluster_id, 0) + 1
    return sizes
