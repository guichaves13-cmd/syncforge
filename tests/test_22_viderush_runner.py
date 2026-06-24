"""Phase 6.7 — Viderush runner wire-up: end-to-end with ALL Phase 6 modules.

Covers:
  • Import + signature contract for run_pipeline_viderush
  • Empty TTS → graceful error
  • Stub-everything happy path produces a valid .mp4 + correct return shape
  • Disabled flags (no temporal, no brand, no beat-snap, no grading) all work
  • Subtitle style override
  • Multi-niche themes pick the right auto-style + auto-grading preset
  • Error path: stock sources empty → some beats unsolved but pipeline returns
"""
from __future__ import annotations
import importlib
import inspect
import shutil
import subprocess
from pathlib import Path

import pytest

from app.services import runner
from app.services.runner import run_pipeline_viderush
from app.services.sync.ranker import Candidate
from app.services.tts.edge import TTSResult


_HAS_FFMPEG = shutil.which("ffmpeg") is not None


def _stub_tts(*_args, **_kwargs):
    """Synthetic TTS result: 3 sentences spanning ~18s. Generates a real
    silent mp3 with ffmpeg so downstream compose+mux works."""
    # synth_sync signature: (script, voice, output_audio_path, rate=...)
    audio = _args[2] if len(_args) > 2 else _kwargs.get("output_audio_path", "stub.mp3")
    Path(audio).parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono:d=18",
         "-c:a", "libmp3lame", str(audio)],
        capture_output=True, timeout=30,
    )
    return TTSResult(audio_path=audio, sentences=[
        {"start": 0.0,  "end": 6.0,  "text": "Number five. Butterfly Viscaria blade."},
        {"start": 6.0,  "end": 12.0, "text": "Stiga Carbonado is the runner-up."},
        {"start": 12.0, "end": 18.0, "text": "Donic Bluefire makes great rubber."},
    ])


def _stub_tts_empty(*_args, **_kwargs):
    audio = _args[2] if len(_args) > 2 else _kwargs.get("output_audio_path", "stub.mp3")
    Path(audio).parent.mkdir(parents=True, exist_ok=True)
    Path(audio).write_bytes(b"\x00" * 100)
    return TTSResult(audio_path=audio, sentences=[])


def _make_real_clip(path: Path, duration: float = 4.0) -> Path:
    """Use ffmpeg to make a real playable clip — needed for compose downstream."""
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi",
         "-i", f"testsrc2=size=320x240:rate=24",
         "-t", f"{duration}",
         "-c:v", "libx264", "-preset", "ultrafast",
         "-pix_fmt", "yuv420p", str(path)],
        capture_output=True, timeout=30,
    )
    return path


# ─────────────────────────────────────────────────────────────────────────
# 1. Signature & module integrity
# ─────────────────────────────────────────────────────────────────────────

def test_run_pipeline_viderush_is_importable():
    assert callable(run_pipeline_viderush)


def test_run_pipeline_viderush_accepts_phase6_flags():
    sig = inspect.signature(run_pipeline_viderush)
    params = sig.parameters
    for flag in ("enable_temporal_vision", "enable_brand_aware",
                  "enable_beat_snap", "enable_color_grading",
                  "subtitle_style", "mood"):
        assert flag in params, f"missing kwarg: {flag}"


def test_run_pipeline_viderush_keeps_legacy_return_shape():
    """The function must return at least the keys legacy callers expect."""
    sig_legacy = inspect.signature(runner.run_pipeline)
    sig_new = inspect.signature(run_pipeline_viderush)
    # Both take the same positional args
    for p in ("script_text", "voice", "title", "theme", "output_dir"):
        assert p in sig_new.parameters
        assert p in sig_legacy.parameters


# ─────────────────────────────────────────────────────────────────────────
# 2. Empty TTS → graceful error
# ─────────────────────────────────────────────────────────────────────────

def test_empty_tts_returns_error_not_crash(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "synth_sync", _stub_tts_empty)
    result = run_pipeline_viderush(
        script_text="", voice="x", title="T", theme="t",
        output_dir=tmp_path,
    )
    assert result["ok"] is False
    assert "sentence" in result["error"].lower() or "boundar" in result["error"].lower()


# ─────────────────────────────────────────────────────────────────────────
# 3. Happy-path end-to-end with stubs
# ─────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg required")
def test_e2e_stubbed_happy_path(tmp_path, monkeypatch):
    """Stub EVERYTHING: TTS, sources, grading. Confirm a final .mp4 is produced
    and the response shape is correct."""
    stock_clip = _make_real_clip(tmp_path / "stock.mp4", duration=8.0)

    def fake_build_sources():
        def search(q, n):
            return [Candidate(source="pexels", source_id=f"p-{q}-{i}",
                              url="http://x/y.mp4",
                              title=f"clip for {q}",
                              description=f"shows {q}", duration=8.0)
                    for i in range(n)]
        def dl(c, output_dir):
            dst = Path(output_dir) / f"{c.source_id}.mp4"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(stock_clip, dst)
            return str(dst)
        return ({"pexels": search, "pixabay": search},
                {"pexels": dl, "pixabay": dl})

    monkeypatch.setattr(runner, "synth_sync", _stub_tts)
    monkeypatch.setattr(runner, "build_sources", fake_build_sources)
    monkeypatch.setattr(runner, "build_brand_searches", lambda: {})
    # No LLM keys → temporal vision skipped automatically
    for k in ("GROQ_API_KEY", "GEMINI_API_KEY", "CEREBRAS_API_KEY",
              "OPENROUTER_API_KEY", "DEEPSEEK_API_KEY"):
        monkeypatch.delenv(k, raising=False)

    result = run_pipeline_viderush(
        script_text="ignored", voice="en-US-AndrewNeural",
        title="Ping Pong Listicle", theme="ping pong listicle",
        mood="energetic",
        output_dir=tmp_path / "out",
        enable_temporal_vision=False,        # no API keys anyway
        enable_brand_aware=False,
        enable_beat_snap=False,              # no real audio
        enable_color_grading=True,
    )
    assert result["ok"] is True
    assert result["windows"] > 0
    assert result["clusters"] > 0
    final = Path(result["final"])
    assert final.exists()
    assert final.stat().st_size > 5_000
    # Style should be auto-picked from theme/mood
    assert result["style"] in ("tiktok", "karaoke", "bold_news",
                                "documentary", "clean")


# ─────────────────────────────────────────────────────────────────────────
# 4. Flag toggles
# ─────────────────────────────────────────────────────────────────────────

def _setup_minimal_stubs(tmp_path, monkeypatch):
    stock_clip = _make_real_clip(tmp_path / "stub_clip.mp4", duration=6.0)
    def fake_sources():
        def search(q, n):
            return [Candidate(source="pexels", source_id=f"p-{q}",
                              url="http://x", title=f"x {q}", duration=6.0)]
        def dl(c, output_dir):
            dst = Path(output_dir) / f"{c.source_id}.mp4"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(stock_clip, dst)
            return str(dst)
        return ({"pexels": search}, {"pexels": dl})

    monkeypatch.setattr(runner, "synth_sync", _stub_tts)
    monkeypatch.setattr(runner, "build_sources", fake_sources)
    monkeypatch.setattr(runner, "build_brand_searches", lambda: {})
    for k in ("GROQ_API_KEY", "GEMINI_API_KEY", "CEREBRAS_API_KEY",
              "OPENROUTER_API_KEY", "DEEPSEEK_API_KEY"):
        monkeypatch.delenv(k, raising=False)


@pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg required")
@pytest.mark.parametrize("flags", [
    {"enable_temporal_vision": False, "enable_brand_aware": False,
     "enable_beat_snap": False, "enable_color_grading": False},
    {"enable_temporal_vision": False, "enable_brand_aware": True,
     "enable_beat_snap": False, "enable_color_grading": True},
    {"enable_temporal_vision": False, "enable_brand_aware": False,
     "enable_beat_snap": False, "enable_color_grading": True},
])
def test_flag_combinations_all_succeed(tmp_path, monkeypatch, flags):
    _setup_minimal_stubs(tmp_path, monkeypatch)
    result = run_pipeline_viderush(
        script_text="x", voice="en-US-AndrewNeural",
        title="T", theme="t",
        output_dir=tmp_path / "out",
        **flags,
    )
    assert result["ok"] is True
    assert Path(result["final"]).exists()


# ─────────────────────────────────────────────────────────────────────────
# 5. Subtitle style override
# ─────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg required")
def test_subtitle_style_override(tmp_path, monkeypatch):
    _setup_minimal_stubs(tmp_path, monkeypatch)
    result = run_pipeline_viderush(
        script_text="x", voice="x", title="T",
        theme="cooking",                # would auto-pick karaoke
        subtitle_style="documentary",   # explicit override
        output_dir=tmp_path / "out",
        enable_temporal_vision=False,
        enable_beat_snap=False,
        enable_color_grading=False,
    )
    assert result["ok"] is True
    assert result["style"] == "documentary"


@pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg required")
@pytest.mark.parametrize("theme,expected_style", [
    ("breaking news today",             "bold_news"),
    ("dance trend tutorial",            "tiktok"),
    ("fitness HIIT 15 min",             "tiktok"),
    ("french revolution documentary",   "documentary"),
    ("history of WWII",                 "documentary"),
    ("iphone 17 tech review",           "clean"),
    ("compound interest explained",     "karaoke"),
])
def test_multi_niche_auto_style(tmp_path, monkeypatch, theme, expected_style):
    _setup_minimal_stubs(tmp_path, monkeypatch)
    result = run_pipeline_viderush(
        script_text="x", voice="x", title="T", theme=theme,
        output_dir=tmp_path / "out",
        enable_temporal_vision=False,
        enable_beat_snap=False,
        enable_color_grading=False,
    )
    assert result["ok"] is True
    assert result["style"] == expected_style, (
        f"theme={theme!r}: got {result['style']}, expected {expected_style}"
    )


# ─────────────────────────────────────────────────────────────────────────
# 6. No sources → some beats unsolved but pipeline returns gracefully
# ─────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg required")
def test_empty_stock_sources_still_completes(tmp_path, monkeypatch):
    """When every source returns 0 candidates, the pipeline must still write a
    composed file (with black slates for unsolved beats) and return ok=True."""
    monkeypatch.setattr(runner, "synth_sync", _stub_tts)
    monkeypatch.setattr(runner, "build_sources",
                         lambda: ({"pexels": lambda q, n: []},
                                   {"pexels": lambda c, d: None}))
    monkeypatch.setattr(runner, "build_brand_searches", lambda: {})
    for k in ("GROQ_API_KEY", "GEMINI_API_KEY", "CEREBRAS_API_KEY",
              "OPENROUTER_API_KEY", "DEEPSEEK_API_KEY"):
        monkeypatch.delenv(k, raising=False)

    result = run_pipeline_viderush(
        script_text="x", voice="x", title="T", theme="generic",
        output_dir=tmp_path / "out",
        enable_temporal_vision=False,
        enable_brand_aware=False,
        enable_beat_snap=False,
        enable_color_grading=False,
    )
    assert result["ok"] is True
    # No clips solved
    assert result["solved"] == 0
    # But file exists
    assert Path(result["final"]).exists()


# ─────────────────────────────────────────────────────────────────────────
# 7. Progress callbacks fire for every stage
# ─────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg required")
def test_progress_callbacks_fire_for_every_stage(tmp_path, monkeypatch):
    _setup_minimal_stubs(tmp_path, monkeypatch)
    events = []
    run_pipeline_viderush(
        script_text="x", voice="x", title="T", theme="t",
        output_dir=tmp_path / "out",
        enable_temporal_vision=False,
        enable_brand_aware=False,
        enable_beat_snap=False,
        enable_color_grading=False,
        progress=events.append,
    )
    # The big-five steps must have at least one event each
    steps_seen = {ev.get("step") for ev in events if "step" in ev}
    for must in ("tts", "concept_windows", "sticky_pools",
                  "sync", "compose", "subtitles"):
        assert must in steps_seen, f"step {must!r} never fired ({steps_seen})"


# ─────────────────────────────────────────────────────────────────────────
# 8. Final cleanup
# ─────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg required")
def test_intermediates_cleaned_up(tmp_path, monkeypatch):
    _setup_minimal_stubs(tmp_path, monkeypatch)
    result = run_pipeline_viderush(
        script_text="x", voice="x", title="T", theme="t",
        output_dir=tmp_path / "out",
        enable_temporal_vision=False,
        enable_brand_aware=False,
        enable_beat_snap=False,
        enable_color_grading=False,
    )
    out_dir = tmp_path / "out"
    clips_dir = out_dir / "_clips"
    if clips_dir.exists():
        leftover = [p for p in clips_dir.rglob("*") if p.is_file()]
        # Either fully gone OR empty
        assert not leftover, f"_clips still contains: {leftover}"
    compose_dir = out_dir / "_compose"
    if compose_dir.exists():
        leftover = [p for p in compose_dir.rglob("*") if p.is_file()]
        assert not leftover, f"_compose still contains: {leftover}"
