"""Phase 6.4 — Temporal Vision Verifier: advanced tests.

Covers:
  • Config defaults
  • _reject contract (always returns canonical dict, zeros vision score)
  • _parse_json_lenient (raw, fenced, embedded, malformed)
  • _prepare_clip ffmpeg trim path
  • verify() flow with mocked _call_gemini_video:
      - approve when both scores high & no flags
      - reject when relevance low
      - reject when action_match low (even if relevance OK)
      - reject on anachronism flag
      - reject on off_topic flag
      - rejection on missing local_path
      - rejection on ffmpeg trim failure
      - rejection on Gemini network exception
      - cleanup of trimmed temp file
  • verify() pollutes Candidate.vision and Candidate.vision_reason correctly
  • Approval boundaries (exactly threshold passes / fails)
  • Long action clauses across niches: cooking, sports, history, science
"""
from __future__ import annotations
import shutil
import subprocess
from pathlib import Path
from unittest import mock

import pytest

from app.services.sync.ranker import Candidate
from app.services.sync.temporal_verifier import (
    TemporalVerifyConfig,
    TemporalVisionVerifier,
    _parse_json_lenient,
)


_HAS_FFMPEG = shutil.which("ffmpeg") is not None


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────

def _make_clip(path: Path, duration: float = 3.0, size: str = "320x240") -> Path:
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi",
         "-i", f"testsrc2=size={size}:rate=24",
         "-t", f"{duration}", "-c:v", "libx264", "-preset", "ultrafast",
         "-pix_fmt", "yuv420p", str(path)],
        capture_output=True, timeout=30,
    )
    return path


def _verifier(api_key: str = "fake-key", **overrides) -> TemporalVisionVerifier:
    cfg = TemporalVerifyConfig(gemini_api_key=api_key, **overrides)
    return TemporalVisionVerifier(cfg)


def _candidate(path: str = "") -> Candidate:
    return Candidate(source="t", source_id="x",
                      url="http://x/y.mp4", local_path=path)


# ─────────────────────────────────────────────────────────────────────────
# 1. Config defaults
# ─────────────────────────────────────────────────────────────────────────

def test_config_defaults_are_sane():
    c = TemporalVerifyConfig()
    assert c.max_clip_duration == 20.0
    assert c.max_clip_bytes == 20_000_000
    assert c.min_relevance == 70
    assert c.min_action_match == 60
    assert c.gemini_model == "gemini-2.5-pro"


# ─────────────────────────────────────────────────────────────────────────
# 2. _parse_json_lenient
# ─────────────────────────────────────────────────────────────────────────

def test_parse_json_lenient_plain():
    assert _parse_json_lenient('{"a": 1}') == {"a": 1}


def test_parse_json_lenient_fenced():
    assert _parse_json_lenient('```json\n{"a": 2}\n```') == {"a": 2}


def test_parse_json_lenient_finds_embedded():
    assert _parse_json_lenient('intro {"a": 3} outro') == {"a": 3}


def test_parse_json_lenient_raises_on_pure_garbage():
    with pytest.raises(Exception):
        _parse_json_lenient("hello world no json here")


# ─────────────────────────────────────────────────────────────────────────
# 3. _reject canonical shape
# ─────────────────────────────────────────────────────────────────────────

def test_reject_returns_canonical_dict_and_zeros_candidate():
    v = _verifier()
    c = _candidate()
    out = v._reject(c, "anything went wrong")
    assert out["approved"] is False
    assert out["relevance_score"] == 0
    assert out["action_match_score"] == 0
    assert out["anachronism"] is False
    assert out["off_topic"] is False
    assert out["quality_issues"] == []
    assert c.vision == 0
    assert "anything went wrong" in c.vision_reason


def test_reject_truncates_long_reason():
    v = _verifier()
    c = _candidate()
    long_reason = "x" * 500
    v._reject(c, long_reason)
    assert len(c.vision_reason) <= 200


# ─────────────────────────────────────────────────────────────────────────
# 4. verify() — missing path / no local file
# ─────────────────────────────────────────────────────────────────────────

def test_verify_no_local_path_rejects(tmp_path):
    v = _verifier()
    c = _candidate(path="")
    out = v.verify(c, "any clause")
    assert out["approved"] is False
    assert "no local path" in c.vision_reason


def test_verify_nonexistent_path_rejects(tmp_path):
    v = _verifier()
    c = _candidate(path=str(tmp_path / "does_not_exist.mp4"))
    out = v.verify(c, "any clause")
    assert out["approved"] is False


# ─────────────────────────────────────────────────────────────────────────
# 5. verify() flow with mocked Gemini
# ─────────────────────────────────────────────────────────────────────────

def _approve_payload(relevance=85, action=75):
    return {
        "relevance_score": relevance,
        "action_match_score": action,
        "description": "Player serves a ball mid-rally",
        "detected_action": "serving ball",
        "anachronism": False,
        "off_topic": False,
        "quality_issues": [],
        "rationale": "tight action match",
    }


@pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg required")
def test_verify_approves_high_scores_no_flags(tmp_path, monkeypatch):
    clip = _make_clip(tmp_path / "clip.mp4", duration=2.0)
    v = _verifier()
    monkeypatch.setattr(v, "_call_gemini_video",
                         lambda prompt, mp4: _approve_payload(85, 75))
    c = _candidate(path=str(clip))
    out = v.verify(c, "the player serves the ball", required_action="serving")
    assert out["approved"] is True
    assert c.vision == 85
    assert "[T] rel=85 act=75" in c.vision_reason


@pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg required")
def test_verify_rejects_low_relevance(tmp_path, monkeypatch):
    clip = _make_clip(tmp_path / "clip.mp4", duration=2.0)
    v = _verifier()
    monkeypatch.setattr(v, "_call_gemini_video",
                         lambda prompt, mp4: _approve_payload(40, 80))
    c = _candidate(path=str(clip))
    out = v.verify(c, "any narration")
    assert out["approved"] is False


@pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg required")
def test_verify_rejects_low_action_match_even_with_high_relevance(tmp_path, monkeypatch):
    """The TEMPORAL upgrade: a clip that looks relevant but doesn't show the
    described action must still be rejected."""
    clip = _make_clip(tmp_path / "clip.mp4", duration=2.0)
    v = _verifier()
    monkeypatch.setattr(v, "_call_gemini_video",
                         lambda prompt, mp4: _approve_payload(85, 30))  # paddle present, no rally
    c = _candidate(path=str(clip))
    out = v.verify(c, "the player is mid-rally")
    assert out["approved"] is False


@pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg required")
def test_verify_rejects_on_anachronism(tmp_path, monkeypatch):
    clip = _make_clip(tmp_path / "clip.mp4", duration=2.0)
    v = _verifier()
    monkeypatch.setattr(v, "_call_gemini_video",
                         lambda p, m: {**_approve_payload(85, 80), "anachronism": True})
    c = _candidate(path=str(clip))
    out = v.verify(c, "1789 French Revolution")
    assert out["approved"] is False


@pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg required")
def test_verify_rejects_on_off_topic_flag(tmp_path, monkeypatch):
    clip = _make_clip(tmp_path / "clip.mp4", duration=2.0)
    v = _verifier()
    monkeypatch.setattr(v, "_call_gemini_video",
                         lambda p, m: {**_approve_payload(85, 80), "off_topic": True})
    c = _candidate(path=str(clip))
    out = v.verify(c, "any narration")
    assert out["approved"] is False


@pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg required")
def test_verify_handles_gemini_exception(tmp_path, monkeypatch):
    clip = _make_clip(tmp_path / "clip.mp4", duration=2.0)
    v = _verifier()
    def boom(*a, **kw):
        raise RuntimeError("Gemini quota exhausted")
    monkeypatch.setattr(v, "_call_gemini_video", boom)
    c = _candidate(path=str(clip))
    out = v.verify(c, "any narration")
    assert out["approved"] is False
    assert "vision error" in c.vision_reason
    assert "quota exhausted" in c.vision_reason


# ─────────────────────────────────────────────────────────────────────────
# 6. Approval gate boundary conditions
# ─────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg required")
@pytest.mark.parametrize("rel,act,expected", [
    (70, 60, True),    # exactly at threshold → approve
    (69, 60, False),   # 1 below relevance → reject
    (70, 59, False),   # 1 below action → reject
    (100, 100, True),  # max
    (0, 0, False),     # min
])
def test_approval_boundaries(rel, act, expected, tmp_path, monkeypatch):
    clip = _make_clip(tmp_path / "clip.mp4", duration=2.0)
    v = _verifier()
    monkeypatch.setattr(v, "_call_gemini_video",
                         lambda p, m: _approve_payload(rel, act))
    c = _candidate(path=str(clip))
    out = v.verify(c, "x")
    assert out["approved"] is expected


# ─────────────────────────────────────────────────────────────────────────
# 7. ffmpeg trim path
# ─────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg required")
def test_prepare_clip_passes_through_when_already_small(tmp_path):
    clip = _make_clip(tmp_path / "small.mp4", duration=3.0)
    v = _verifier()
    out = v._prepare_clip(str(clip))
    # Already small → returned as-is, no temp file
    assert out == str(clip)


@pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg required")
def test_prepare_clip_trims_when_too_long(tmp_path):
    """Force max_clip_duration=1s so the trim path engages."""
    clip = _make_clip(tmp_path / "long.mp4", duration=5.0)
    v = _verifier(max_clip_duration=1.0)
    out = v._prepare_clip(str(clip))
    # Different file path = trimmed copy
    assert out != str(clip)
    assert Path(out).exists()
    # Verify duration ≤ 1.0s
    dur = v._probe_duration(out)
    assert dur <= 1.5, f"trimmed duration was {dur:.2f}s"
    Path(out).unlink(missing_ok=True)


def test_prepare_clip_raises_when_ffmpeg_fails(tmp_path, monkeypatch):
    """Simulate ffmpeg producing no output → must raise cleanly."""
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a real mp4")
    v = _verifier(max_clip_duration=1.0)
    # Patch _probe_duration to return 100 (force the trim path) and let real
    # ffmpeg fail on the garbage file
    monkeypatch.setattr(v, "_probe_duration", lambda p: 100.0)
    with pytest.raises(Exception):
        v._prepare_clip(str(bad))


# ─────────────────────────────────────────────────────────────────────────
# 8. Verify() cleans up trimmed temp file
# ─────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg required")
def test_verify_deletes_trimmed_temp_file(tmp_path, monkeypatch):
    clip = _make_clip(tmp_path / "src.mp4", duration=5.0)
    v = _verifier(max_clip_duration=1.0)
    tracked_path: dict = {}
    def fake_call(prompt, mp4):
        tracked_path["mp4"] = mp4
        return _approve_payload(80, 70)
    monkeypatch.setattr(v, "_call_gemini_video", fake_call)
    c = _candidate(path=str(clip))
    out = v.verify(c, "x")
    assert out["approved"] is True
    # The trimmed file was different from source AND has been deleted
    assert tracked_path["mp4"] != str(clip)
    assert not Path(tracked_path["mp4"]).exists()


# ─────────────────────────────────────────────────────────────────────────
# 9. Multi-niche action validation
# ─────────────────────────────────────────────────────────────────────────

ACTION_CASES = {
    "cooking_action": ("the chef caramelises onions in butter", "caramelising"),
    "sports_rally":   ("the player serves the ball mid-rally", "serving"),
    "history_march":  ("the crowd marched toward the Bastille fortress", "marching"),
    "science_lab":    ("the lab technician pipettes the sample", "pipetting"),
    "fitness_lift":   ("the athlete deadlifts 200 kilograms", "lifting"),
    "tech_assembly":  ("the robot welds the chassis at 1000 degrees", "welding"),
}


@pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg required")
@pytest.mark.parametrize("niche,case", list(ACTION_CASES.items()))
def test_verify_uses_action_in_prompt(niche, case, tmp_path, monkeypatch):
    clause, action = case
    clip = _make_clip(tmp_path / f"{niche}.mp4", duration=2.0)
    v = _verifier()

    captured_prompt = {}
    def capture(prompt, mp4):
        captured_prompt["text"] = prompt
        return _approve_payload(80, 70)
    monkeypatch.setattr(v, "_call_gemini_video", capture)

    c = _candidate(path=str(clip))
    v.verify(c, clause, required_action=action)
    # Both the clause and the action wound up in the prompt
    assert clause in captured_prompt["text"], f"{niche}: clause missing from prompt"
    assert action in captured_prompt["text"], f"{niche}: action missing from prompt"


@pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg required")
def test_verify_default_action_when_unspecified(tmp_path, monkeypatch):
    clip = _make_clip(tmp_path / "clip.mp4", duration=2.0)
    v = _verifier()
    captured_prompt = {}
    monkeypatch.setattr(v, "_call_gemini_video",
                         lambda p, m: (captured_prompt.update(text=p) or _approve_payload(80, 70)))
    v.verify(_candidate(path=str(clip)), "x")
    assert "any plausible action" in captured_prompt["text"]


# ─────────────────────────────────────────────────────────────────────────
# 10. Pollutes candidate state correctly
# ─────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg required")
def test_verify_writes_vision_score_and_reason(tmp_path, monkeypatch):
    clip = _make_clip(tmp_path / "c.mp4", duration=2.0)
    v = _verifier()
    monkeypatch.setattr(v, "_call_gemini_video",
                         lambda p, m: _approve_payload(92, 85))
    c = _candidate(path=str(clip))
    v.verify(c, "x")
    assert c.vision == 92
    assert "rel=92" in c.vision_reason
    assert "act=85" in c.vision_reason


def test_module_imports_cleanly():
    """Final sanity: importing temporal_verifier shouldn't fail even
    without google-genai actually being callable (it's lazy-imported)."""
    import importlib
    importlib.import_module("app.services.sync.temporal_verifier")
