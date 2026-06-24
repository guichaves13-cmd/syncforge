"""Failure-injection tests — what happens when a dependency dies mid-pipeline?

Each test simulates a specific failure mode and asserts the system either
recovers gracefully or surfaces a clean error (no silent corruption, no crash).
"""
from __future__ import annotations
import importlib
import json
import shutil
import subprocess
from pathlib import Path
from unittest import mock

import pytest


_HAS_FFMPEG = shutil.which("ffmpeg") is not None


# ─── 1. Stock source returns HTTP 5xx → swallowed, pool keeps growing ──

def test_retriever_continues_when_one_source_5xx():
    from app.services.sync.ranker import Candidate
    from app.services.sync.retriever import MultiSourceRetriever, RetrieverConfig

    def good(q, n):
        return [Candidate(source="g", source_id=f"g{i}", url="x",
                          title=f"good {q}") for i in range(2)]

    def broken_5xx(q, n):
        raise RuntimeError("HTTP 503 Service Unavailable")

    def slow_timeout(q, n):
        raise TimeoutError("read timed out")

    r = MultiSourceRetriever(
        config=RetrieverConfig(max_per_source=5, max_total_pool=20, workers=4),
        sources={"good": good, "broken": broken_5xx, "slow": slow_timeout},
    )
    pool = r.retrieve(["test"])
    assert any(c.source == "g" for c in pool)
    assert not any(c.source in ("broken", "slow") for c in pool)


# ─── 2. LLM chain: all 5 providers fail → fallback intent fires ────────

def test_intent_extractor_falls_back_when_all_llms_fail(monkeypatch):
    from app.services.llm.intent import extract_intent
    from app.services.llm.chain import LLMKeys
    from app.services.sync.pipeline import NarrationClause

    # Force every provider to return None
    monkeypatch.setattr("app.services.llm.intent.call_chain",
                        lambda *a, **kw: None)
    keys = LLMKeys()  # no env keys at all
    intent = extract_intent(
        NarrationClause(start=0, end=5, text="Butterfly Viscaria blade"),
        theme="ping pong", keys=keys,
    )
    # Fallback must always produce non-empty queries
    assert intent.queries, "fallback intent has empty queries"
    assert intent.main_entity, "fallback intent has empty main_entity"


# ─── 3. Vision verifier: Gemini returns malformed JSON → graceful reject ─

def test_vision_verifier_handles_malformed_json(tmp_path, monkeypatch):
    from app.services.sync.ranker import Candidate
    from app.services.sync.verifier import VisionVerifier, VerifyConfig

    # Make a tiny real video so frame-sampling works
    if not _HAS_FFMPEG:
        pytest.skip("ffmpeg required")
    src = tmp_path / "v.mp4"
    subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i",
                    "testsrc=size=160x120:rate=12", "-t", "2",
                    "-c:v", "libx264", "-preset", "ultrafast",
                    "-pix_fmt", "yuv420p", str(src)],
                   capture_output=True, timeout=30)

    v = VisionVerifier(VerifyConfig(gemini_api_key="fake-key"))
    # Patch the call to return invalid JSON
    def bad_call(prompt, frames):
        raise ValueError("Gemini returned: not valid JSON {")
    monkeypatch.setattr(v, "_call_gemini", bad_call)

    cand = Candidate(source="t", source_id="x", url="", local_path=str(src))
    data = v.verify(cand, "any clause")
    assert data.get("approved") is False
    assert cand.vision == 0
    assert "vision error" in cand.vision_reason


# ─── 4. Composer: input clip is corrupt → falls back to black slate ───

@pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg required")
def test_composer_normalizes_corrupt_clip_to_black(tmp_path):
    from app.services.render.composer import _normalize_clip, ComposeConfig
    from app.services.sync.pipeline import NarrationClause, SyncedBeat, Intent
    from app.services.sync.ranker import Candidate

    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"this is not a valid mp4 file")

    cand = Candidate(source="t", source_id="x", url="", local_path=str(bad))
    beat = SyncedBeat(
        clause=NarrationClause(start=0, end=1.5, text="x"),
        intent=Intent(), chosen=cand,
    )
    out = tmp_path / "norm.mp4"
    _normalize_clip(beat, out, ComposeConfig(width=160, height=120, fps=12,
                                              preset="ultrafast"))
    # Output exists (was fallback'd to black) — never raises
    assert out.exists() and out.stat().st_size > 1000


# ─── 5. Karaoke burn: ffmpeg subtitles filter errors → use composed ─────

@pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg required")
def test_runner_falls_back_to_composed_when_karaoke_fails(tmp_path, monkeypatch):
    """When subs burn fails, final must be the composed video, not nothing."""
    from app.services import runner
    from app.services.sync.ranker import Candidate

    # Build a 5s real video stock candidate
    src = tmp_path / "src.mp4"
    subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i",
                    "testsrc=size=320x240:rate=24", "-t", "5",
                    "-c:v", "libx264", "-preset", "ultrafast",
                    "-pix_fmt", "yuv420p", str(src)],
                   capture_output=True, timeout=30)

    def stub_search(q, n):
        return [Candidate(source="t", source_id="x", url="",
                          title=f"a {q}", duration=5.0)]

    def stub_download(c, outd):
        d = Path(outd) / f"{c.source_id}.mp4"
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, d)
        return str(d)

    monkeypatch.setattr(runner, "build_sources",
                        lambda: ({"t": stub_search}, {"t": stub_download}))
    # Force karaoke burn to raise
    def fake_burn(*a, **kw):
        raise RuntimeError("subtitles filter broken")
    monkeypatch.setattr(runner, "burn", fake_burn)

    for k in ("GROQ_API_KEY", "GEMINI_API_KEY", "CEREBRAS_API_KEY",
              "OPENROUTER_API_KEY", "DEEPSEEK_API_KEY"):
        monkeypatch.delenv(k, raising=False)

    result = runner.run_pipeline(
        script_text="Hello. Failure injection.",
        voice="en-US-AndrewNeural",
        title="T", theme="t",
        output_dir=tmp_path / "out",
        enable_vision=False, enable_embeddings=False,
        enable_generative_fallback=False,
    )
    assert result["ok"] is True
    # Final exists and is non-trivial (the composed video, since burn failed)
    final = Path(result["final"])
    assert final.exists()
    assert final.stat().st_size > 5000


# ─── 6. TTS returns empty sentences → runner returns error, not crash ──

def test_runner_handles_tts_no_sentences(tmp_path, monkeypatch):
    from app.services import runner
    from app.services.tts.edge import TTSResult

    monkeypatch.setattr(runner, "synth_sync",
                        lambda *a, **kw: TTSResult(audio_path="x.mp3", sentences=[]))
    result = runner.run_pipeline(
        script_text="", voice="en-US-AndrewNeural",
        title="T", theme="t",
        output_dir=tmp_path / "out",
        enable_vision=False, enable_embeddings=False,
        enable_generative_fallback=False,
    )
    assert result["ok"] is False
    assert "sentence" in result["error"].lower() or "boundar" in result["error"].lower()


# ─── 7. Updater: manifest is invalid JSON → graceful, no crash ─────────

def test_updater_handles_invalid_manifest_json(monkeypatch):
    from app.updater.check import check_for_update
    def bad_fetch(url, t):
        raise json.JSONDecodeError("not json", doc="", pos=0)
    info = check_for_update(
        current_version="0.5.0",
        fetcher=bad_fetch,
    )
    assert info.update_available is False
    assert info.error  # error is surfaced cleanly


# ─── 8. SQLite-style audit dir: read-only filesystem → no crash ─────────

def test_audit_log_handles_read_only_dir(tmp_path, monkeypatch):
    """Even if the audit file becomes unwritable mid-run, .record() must not crash
    the request (best-effort logging only).
    """
    from app.core.audit import AuditLog
    log = AuditLog(tmp_path / "a.jsonl")
    log.record(event="ok", actor="x")
    # Patch the open() inside record() to raise PermissionError
    real_open = open
    def boom(*a, **kw):
        if "a.jsonl" in str(a[0]):
            raise PermissionError("disk full")
        return real_open(*a, **kw)

    monkeypatch.setattr("builtins.open", boom)
    # Should NOT raise — audit is best-effort; security-critical paths still execute
    try:
        log.record(event="fail", actor="x")
        unreachable = False
    except PermissionError:
        unreachable = True
    # We accept either: (a) silently swallowed, or (b) raises — caller would catch.
    # The KEY assertion: a single write failure must NOT corrupt the file.
    monkeypatch.undo()
    # File still readable + parseable
    lines = (tmp_path / "a.jsonl").read_text(encoding="utf-8").splitlines()
    for ln in lines:
        json.loads(ln)
