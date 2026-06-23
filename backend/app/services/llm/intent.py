"""Intent extractor — per-clause structured JSON via LLM chain.

Output schema is the Intent dataclass from services/sync/pipeline.py.
"""
from __future__ import annotations
import json
import re

from ..sync.pipeline import Intent, NarrationClause
from .chain import LLMKeys, call_chain


_PROMPT = """You are a video b-roll director. Given a SINGLE NARRATION CLAUSE from a video script, produce a structured JSON describing the IDEAL VISUAL.

CLAUSE: \"\"\"{clause}\"\"\"
VIDEO TOPIC: {theme}

Respond ONLY in JSON (no prose, no markdown fence):
{{
  "main_entity": "<the central thing being shown — concrete noun phrase>",
  "action": "<what the entity is doing — verb phrase, short>",
  "location": "<setting, e.g. 'indoor tournament hall' / 'kitchen counter' / 'mountain trail'>",
  "era": "<'modern' / '1990s' / 'ancient' / etc — match the clause>",
  "mood": "<one word: cinematic / analytical / urgent / warm / dramatic>",
  "objects": ["<concrete object 1>", "<concrete object 2>"],
  "key_visuals": ["<close-up of X>", "<wide shot of Y>", "<3 total>"],
  "queries": [
    "<8 search queries in ENGLISH, each 4-8 words, covering different angles>",
    "<every query MUST include the TOPIC and the MAIN ENTITY>",
    "<no isolated words like 'two' or 'three'; no acronyms unfamiliar to stock libraries>"
  ]
}}

Rules:
- ALL queries must be in ENGLISH (stock libraries are 90%+ English-indexed).
- ALL queries must include enough TOPIC context that no false-positive niche matches.
- If clause mentions a real brand/product, include it.
- main_entity must be concrete (avoid 'concept', 'idea', 'feeling').
"""


def extract_intent(clause: NarrationClause, theme: str, keys: LLMKeys) -> Intent:
    prompt = _PROMPT.format(clause=clause.text.replace('"', "'"), theme=theme or "general")
    raw = call_chain(prompt, keys, max_tokens=600, temperature=0.4, json_mode=True)
    if not raw:
        return _fallback_intent(clause, theme)
    try:
        data = _parse_json_lenient(raw)
    except Exception:
        return _fallback_intent(clause, theme)
    return Intent(
        main_entity=str(data.get("main_entity", "") or "")[:120],
        action=str(data.get("action", "") or "")[:120],
        location=str(data.get("location", "") or "")[:120],
        era=str(data.get("era", "modern") or "modern"),
        mood=str(data.get("mood", "neutral") or "neutral"),
        objects=[str(x) for x in (data.get("objects") or [])][:6],
        key_visuals=[str(x) for x in (data.get("key_visuals") or [])][:6],
        queries=[str(x).strip() for x in (data.get("queries") or []) if str(x).strip()][:8],
    )


def _fallback_intent(clause: NarrationClause, theme: str) -> Intent:
    """When all LLMs fail — produce a defensible intent from clause keywords."""
    # crude noun extraction
    words = [w for w in re.findall(r"\b[A-Za-z]{4,}\b", clause.text)
             if w.lower() not in _STOP][:5]
    base = " ".join(words[:3]) or clause.text[:40]
    topic = theme or "general"
    return Intent(
        main_entity=base,
        action="",
        location="",
        era="modern",
        mood="neutral",
        objects=words,
        key_visuals=[f"close-up {base}", f"wide shot {base}"],
        queries=[
            f"{base} {topic}",
            f"{topic} {base} close up",
            f"{topic} {base} cinematic",
            f"{base} real footage",
        ],
    )


_STOP = {
    "this","that","with","from","into","over","under","when","what","then","than",
    "also","only","just","like","such","were","have","been","they","them","their",
    "those","these","there","because","could","would","should","might","must","into",
}


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
