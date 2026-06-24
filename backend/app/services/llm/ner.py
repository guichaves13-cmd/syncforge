"""Named-entity recognition — structured extraction of brands, products,
people, places and dates from narration text.

Two backends:
  • LLM (Gemini/Groq chain in JSON mode) — semantic, niche-agnostic, ~50ms
  • Lexical regex (no deps) — fast offline fallback when LLM unavailable

Output schema: EntityCollection with 5 typed lists.

Use case in SyncForge: for a concept window tagged with `Butterfly`, NER
classifies it as a brand → retriever priorities Wikimedia/Openverse for
the actual logo/product shot instead of a generic ping-pong clip.
"""
from __future__ import annotations
import json
import re
from dataclasses import dataclass, field
from typing import Iterable

from .chain import LLMKeys, call_chain


@dataclass
class EntityCollection:
    """Typed entities extracted from a piece of narration."""
    brands: list[str] = field(default_factory=list)       # "Butterfly", "Apple"
    products: list[str] = field(default_factory=list)     # "iPhone 17 Pro", "Viscaria"
    people: list[str] = field(default_factory=list)       # "Ma Long", "Steve Jobs"
    places: list[str] = field(default_factory=list)       # "Tokyo", "Silicon Valley"
    dates: list[str] = field(default_factory=list)        # "2024", "July 1789"
    numbers: list[str] = field(default_factory=list)      # "$189", "12%"

    @property
    def has_any(self) -> bool:
        return bool(self.brands or self.products or self.people
                     or self.places or self.dates or self.numbers)

    def all_entities(self) -> list[str]:
        return [*self.brands, *self.products, *self.people,
                *self.places, *self.dates, *self.numbers]

    def primary_visual_anchor(self) -> str:
        """Pick the entity most likely to dominate the visual.
        Priority: product > brand > people > places."""
        for bucket in (self.products, self.brands, self.people, self.places):
            if bucket:
                return bucket[0]
        return ""


# ─────────────────────────────────────────────────────────────────────────
# LLM-based NER
# ─────────────────────────────────────────────────────────────────────────

_NER_PROMPT = """Extract the named entities from this short text. Be conservative — only entities EXPLICITLY mentioned, never inferred.

TEXT: \"\"\"{text}\"\"\"

Respond ONLY in JSON (no markdown):
{{
  "brands":   ["<company / manufacturer names>"],
  "products": ["<product or model names>"],
  "people":   ["<full names>"],
  "places":   ["<cities, countries, regions>"],
  "dates":    ["<years, decades, specific dates>"],
  "numbers":  ["<dollar amounts, percentages, key metrics>"]
}}

Rules:
- Skip pronouns (it/he/she/they/we).
- Skip generic nouns (paddle, blade) unless capitalized as brand-name.
- A product is something like 'iPhone 17 Pro', 'Viscaria', 'Model 3', 'Carbonado 2000'.
- Brand is the parent company.
- Empty list if nothing matches. Never fabricate.
"""


def extract_entities(text: str, keys: LLMKeys) -> EntityCollection:
    """LLM-first; falls back to lexical regex if all providers fail."""
    text = (text or "").strip()
    if not text:
        return EntityCollection()
    raw = call_chain(_NER_PROMPT.format(text=text.replace('"', "'")),
                      keys, max_tokens=400, temperature=0.0, json_mode=True)
    if raw:
        try:
            data = _parse_json_lenient(raw)
            return EntityCollection(
                brands=_clean_list(data.get("brands")),
                products=_clean_list(data.get("products")),
                people=_clean_list(data.get("people")),
                places=_clean_list(data.get("places")),
                dates=_clean_list(data.get("dates")),
                numbers=_clean_list(data.get("numbers")),
            )
        except Exception:
            pass
    return extract_entities_lexical(text)


# ─────────────────────────────────────────────────────────────────────────
# Lexical fallback — heuristics, no network
# ─────────────────────────────────────────────────────────────────────────

# Common-word filter shared with concept.py philosophy
_STOP = frozenset({
    "the","a","an","this","that","these","those","it","he","she","we","they",
    "you","i","my","your","his","her","our","their","today","yesterday","now",
    "then","number","item","step","first","second","third","fourth","fifth",
    "and","but","or","so","yet","because","since","while","although",
    "another","other","such","both","each","every","any","some","most","many","few",
    # Auxiliaries / common verbs that should never end up as brand candidates
    "is","are","was","were","be","been","being","am","do","does","did",
    "have","has","had","will","would","can","could","should","may","might","must",
    # Common sentence-leading directives
    "start","stop","go","wait","look","listen","try","keep","take","make",
    "on","in","at","by","for","to","of","with","from","into","over","under",
    # Months (often dates) handled separately below
})

_PROPER = re.compile(r"\b([A-Z][a-zA-Z0-9'-]+(?:\s+[A-Z][a-zA-Z0-9'-]+)*)\b")
_DOLLAR = re.compile(r"\$\d+(?:[.,]\d+)?(?:[KkMmBb])?")
_PERCENT = re.compile(r"\d+(?:[.,]\d+)?%")
_YEAR = re.compile(r"\b\d{3,4}\b")          # broad: 1789, 2024, 870 etc
_INTEGER = re.compile(r"\b\d+(?:[.,]\d+)?\b")  # plain numbers (20, 3.14)
_POSSESSIVE = re.compile(r"'s\b")           # strip "Apple's" → "Apple"
_PERSON_TITLE = re.compile(
    r"\b(?:Mr|Mrs|Ms|Dr|Prof|President|Senator|Governor|Director|Chief)\.?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)"
)


def extract_entities_lexical(text: str) -> EntityCollection:
    """Best-effort lexical extraction. Used when LLM is unavailable."""
    ec = EntityCollection()

    # Proper-noun candidates (filter stopwords + strip possessive "'s")
    candidates: list[str] = []
    for raw in _PROPER.findall(text):
        cleaned = _POSSESSIVE.sub("", raw)            # Apple's → Apple
        cleaned = _strip_stops(cleaned)
        if cleaned and cleaned not in candidates:
            candidates.append(cleaned)

    # People (heuristic: name with titles, OR 2+ word capitalised that aren't a place)
    people = []
    for m in _PERSON_TITLE.findall(text):
        if m and m not in people:
            people.append(m)
    ec.people = people

    # Numbers — dollar, percent, plain integers (in priority order, deduped)
    seen_n: set[str] = set()
    for rx in (_DOLLAR, _PERCENT, _INTEGER):
        for n in rx.findall(text):
            if n not in seen_n:
                seen_n.add(n)
                ec.numbers.append(n)

    # Dates — extract years (3-4 digits), plus partition integers that look year-like
    ec.dates = list(dict.fromkeys(_YEAR.findall(text)))
    # Move dates out of numbers so they aren't double-counted
    ec.numbers = [n for n in ec.numbers if n not in ec.dates]

    # Brands vs products: heuristic — multi-word candidates often contain a model
    # (e.g. "Stiga Carbonado"). Treat first word as brand, full string as product.
    for cand in candidates:
        if cand in ec.people:
            continue
        words = cand.split()
        if len(words) == 1:
            if cand not in ec.brands:
                ec.brands.append(cand)
        else:
            head = words[0]
            if head not in ec.brands:
                ec.brands.append(head)
            if cand not in ec.products:
                ec.products.append(cand)

    # De-dupe products that ARE the brand
    ec.products = [p for p in ec.products if p not in ec.brands]
    return ec


def _strip_stops(text: str) -> str:
    tokens = text.split()
    while tokens and tokens[0].lower() in _STOP:
        tokens.pop(0)
    return " ".join(tokens)


def _clean_list(items) -> list[str]:
    if not isinstance(items, list):
        return []
    out: list[str] = []
    for x in items:
        s = str(x or "").strip()
        if s and len(s) <= 80 and s not in out:
            out.append(s)
    return out[:10]


def _parse_json_lenient(text: str) -> dict:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).rsplit("```", 1)[0].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            return json.loads(m.group(0))
        raise
