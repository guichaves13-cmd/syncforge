"""Unit tests for intent fallback and ASS karaoke builder."""
from pathlib import Path

from app.services.llm.intent import _fallback_intent, _parse_json_lenient
from app.services.llm.chain import LLMKeys
from app.services.sync.pipeline import NarrationClause
from app.services.subtitles.karaoke import build_ass, _fmt


# ─── intent fallback ───────────────────────────────────────────────────

def test_fallback_intent_extracts_keywords():
    clause = NarrationClause(start=0, end=5, text="The Butterfly Viscaria blade is overpriced.")
    intent = _fallback_intent(clause, "table tennis")
    assert intent.main_entity  # non-empty
    assert intent.queries  # has fallback queries
    assert all("table tennis" in q.lower() or "butterfly" in q.lower()
               for q in intent.queries[:2])


def test_fallback_intent_handles_short_clause():
    clause = NarrationClause(start=0, end=2, text="Wait.")
    intent = _fallback_intent(clause, "general")
    assert intent.queries  # never empty


def test_parse_json_lenient_strips_fence():
    text = "```json\n{\"a\": 1}\n```"
    assert _parse_json_lenient(text) == {"a": 1}


def test_parse_json_lenient_finds_embedded_object():
    text = "Sure, here you go: {\"a\": 2} — let me know!"
    assert _parse_json_lenient(text) == {"a": 2}


def test_llm_keys_from_env_is_safe_when_unset(monkeypatch):
    for v in ("GROQ_API_KEY","GEMINI_API_KEY","CEREBRAS_API_KEY",
              "OPENROUTER_API_KEY","DEEPSEEK_API_KEY"):
        monkeypatch.delenv(v, raising=False)
    k = LLMKeys.from_env()
    assert k.groq == k.gemini == k.cerebras == k.openrouter == k.deepseek == ""


# ─── karaoke ASS builder ──────────────────────────────────────────────

def test_fmt_seconds_to_ass():
    assert _fmt(0) == "0:00:00.00"
    assert _fmt(65.5) == "0:01:05.50"
    assert _fmt(3661.25) == "1:01:01.25"
    assert _fmt(-1) == "0:00:00.00"


def test_build_ass_creates_valid_file(tmp_path):
    sents = [
        {"start": 0.0, "end": 1.5, "text": "Hello world."},
        {"start": 1.5, "end": 3.0, "text": "Second line, with comma."},
        {"start": 3.0, "end": 4.0, "text": ""},  # empty: must be skipped
    ]
    out = tmp_path / "test.ass"
    build_ass(sents, out)
    content = out.read_text(encoding="utf-8")
    assert "[Script Info]" in content
    assert "[V4+ Styles]" in content
    assert "[Events]" in content
    assert "Hello world." in content
    assert "Second line" in content
    # empty sentence skipped → exactly 2 Dialogue lines
    assert content.count("Dialogue:") == 2
