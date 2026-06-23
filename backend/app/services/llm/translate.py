"""Translate search queries to English (Pexels/Pixabay/YouTube are 90%+ EN-indexed).

Uses the same LLM chain. Caches per-query → translation in memory.
"""
from __future__ import annotations
import re

from .chain import LLMKeys, call_chain


# Heuristic English detector — if 95%+ of words are pure ASCII alpha and
# common stopwords appear, we skip translation. Saves API calls.
_EN_STOPWORDS = {
    "the","of","and","to","in","a","is","that","for","with","on","at",
    "by","an","this","or","be","from","as","it",
}

_CACHE: dict[str, str] = {}


def is_already_english(text: str) -> bool:
    text = text.strip()
    if not text:
        return True
    # Any non-ASCII char (accents é/ñ, CJK 北, etc.) → definitely not English
    if not text.isascii():
        return False
    words = re.findall(r"[A-Za-z]+", text.lower())
    if not words:
        return False  # symbols-only / digits-only → translate to be safe
    has_stopword = any(w in _EN_STOPWORDS for w in words)
    # Short queries (1-3 words) of pure ASCII letters: assume English
    if len(words) <= 3:
        return True
    return has_stopword


def translate_to_english(text: str, keys: LLMKeys) -> str:
    text = (text or "").strip()
    if not text:
        return text
    if is_already_english(text):
        return text
    if text in _CACHE:
        return _CACHE[text]
    prompt = (
        "Translate the following short search query to ENGLISH. "
        "Return ONLY the translated query, no quotes, no explanation, no prefix.\n"
        f"Query: {text}\n"
        "English:"
    )
    out = call_chain(prompt, keys, max_tokens=80, temperature=0.0) or text
    out = out.strip().strip('"').strip("'").splitlines()[0][:120]
    _CACHE[text] = out
    return out
