"""Integration tests using real subsystems (no LLM/vision keys needed).

Covers:
  - Edge-TTS actually synthesizes audio with sentence boundaries
  - ffmpeg composer normalizes a real input file
  - Karaoke ASS builder produces a file that ffmpeg accepts (dry-parse)
"""
import shutil
import subprocess
from pathlib import Path

import pytest

from app.services.tts.edge import synth_sync
from app.services.subtitles.karaoke import build_ass
from app.services.render.composer import compose, ComposeConfig, _make_black, _normalize_clip
from app.services.sync.pipeline import NarrationClause, SyncedBeat, Intent
from app.services.sync.ranker import Candidate


_HAS_FFMPEG = shutil.which("ffmpeg") is not None


def _make_test_video(path: Path, duration: float = 3.0) -> Path:
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi",
         "-i", f"color=c=red:s=320x240:r=24",
         "-t", f"{duration}", "-c:v", "libx264", "-preset", "ultrafast",
         "-pix_fmt", "yuv420p", str(path)],
        capture_output=True, timeout=30,
    )
    return path


def _make_test_audio(path: Path, duration: float = 3.0) -> Path:
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi",
         "-i", f"sine=frequency=440:duration={duration}",
         "-c:a", "libmp3lame", str(path)],
        capture_output=True, timeout=30,
    )
    return path


# ─── Edge-TTS real synth ─────────────────────────────────────────────

def test_edge_tts_synthesizes_and_returns_boundaries(tmp_path):
    out = tmp_path / "tts.mp3"
    script = "Hello world. This is a SyncForge integration test."
    result = synth_sync(script, "en-US-AndrewNeural", str(out))
    assert out.exists(), "audio file not written"
    assert out.stat().st_size > 5000, f"audio too small ({out.stat().st_size} bytes)"
    assert len(result.sentences) >= 1
    # Sentence boundaries should have non-negative times
    for s in result.sentences:
        assert s["start"] >= 0
        assert s["end"] > s["start"]


# ─── Composer ─────────────────────────────────────────────────────────

@pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg not available")
def test_composer_black_slate(tmp_path):
    out = tmp_path / "black.mp4"
    _make_black(out, 1.0, ComposeConfig(width=320, height=240, fps=24, preset="ultrafast"))
    assert out.exists() and out.stat().st_size > 1000


@pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg not available")
def test_composer_normalize_real_clip(tmp_path):
    src = _make_test_video(tmp_path / "src.mp4", 2.0)
    out = tmp_path / "norm.mp4"
    cand = Candidate(source="test", source_id="t1", url="", local_path=str(src))
    beat = SyncedBeat(
        clause=NarrationClause(start=0, end=1.0, text="x"),
        intent=Intent(), chosen=cand,
    )
    _normalize_clip(beat, out, ComposeConfig(width=320, height=240, fps=24, preset="ultrafast"))
    assert out.exists() and out.stat().st_size > 1000


@pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg not available")
def test_composer_full_compose_2_beats(tmp_path):
    src1 = _make_test_video(tmp_path / "s1.mp4", 2.0)
    src2 = _make_test_video(tmp_path / "s2.mp4", 2.0)
    audio = _make_test_audio(tmp_path / "a.mp3", 2.0)
    beats = [
        SyncedBeat(
            clause=NarrationClause(start=0, end=1.0, text="one"),
            intent=Intent(),
            chosen=Candidate(source="t", source_id="1", url="", local_path=str(src1)),
        ),
        SyncedBeat(
            clause=NarrationClause(start=1.0, end=2.0, text="two"),
            intent=Intent(),
            chosen=Candidate(source="t", source_id="2", url="", local_path=str(src2)),
        ),
    ]
    out = tmp_path / "final.mp4"
    compose(beats, str(audio), str(out),
            ComposeConfig(width=320, height=240, fps=24, preset="ultrafast"),
            work_dir=tmp_path / "work")
    assert out.exists() and out.stat().st_size > 1000
    # Verify it's a playable video
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(out)],
        capture_output=True, text=True, timeout=10,
    )
    assert float(r.stdout.strip()) > 0


# ─── Karaoke burn integration ─────────────────────────────────────────

@pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg not available")
def test_karaoke_burn_real_subtitle(tmp_path):
    from app.services.subtitles.karaoke import burn
    video = _make_test_video(tmp_path / "v.mp4", 3.0)
    # Add silent audio so karaoke -c:a copy doesn't fail
    audio = _make_test_audio(tmp_path / "a.mp3", 3.0)
    av = tmp_path / "av.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(video), "-i", str(audio),
         "-c:v", "copy", "-c:a", "aac", "-shortest", str(av)],
        capture_output=True, timeout=30,
    )
    sents = [{"start": 0.0, "end": 1.5, "text": "Hello"},
             {"start": 1.5, "end": 3.0, "text": "World"}]
    ass = tmp_path / "k.ass"
    build_ass(sents, ass)
    out = tmp_path / "burnt.mp4"
    burn(str(av), ass, str(out))
    assert out.exists() and out.stat().st_size > 1000


# ─── Runner mini end-to-end (no LLM, no vision) ──────────────────────

@pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg not available")
def test_runner_mini_e2e_with_stub_sources(tmp_path, monkeypatch):
    """Run the full runner with stub stock sources — proves the wiring."""
    from app.services import runner

    # Patch sources to return a single locally-generated video
    src_video = _make_test_video(tmp_path / "stock.mp4", 5.0)

    def stub_search(q, n):
        return [Candidate(source="stub", source_id="only",
                          url="file://x", title=f"stock {q}",
                          description=f"shows {q}", duration=5.0)]

    def stub_download(c, output_dir):
        # Copy the pre-made source into the requested output_dir
        from pathlib import Path as P
        dst = P(output_dir) / f"{c.source_id}.mp4"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src_video, dst)
        return str(dst)

    def stub_build_sources():
        return {"stub": stub_search}, {"stub": stub_download}

    monkeypatch.setattr(runner, "build_sources", stub_build_sources)
    # Disable env-loaded LLM keys so fallback intent fires deterministically
    for k in ("GROQ_API_KEY", "GEMINI_API_KEY", "CEREBRAS_API_KEY",
              "OPENROUTER_API_KEY", "DEEPSEEK_API_KEY"):
        monkeypatch.delenv(k, raising=False)

    events = []
    result = runner.run_pipeline(
        script_text="Hello world. SyncForge integration test.",
        voice="en-US-AndrewNeural",
        title="Test", theme="test",
        output_dir=tmp_path / "out",
        enable_vision=False,
        enable_embeddings=False,
        enable_generative_fallback=False,
        progress=events.append,
    )

    assert result["ok"] is True
    final = Path(result["final"])
    assert final.exists()
    assert final.stat().st_size > 5000
    # Progress events should include sync + compose + done
    types = {e.get("event") for e in events}
    assert "step" in types
    assert any(e.get("step") == "sync" for e in events)
    assert any(e.get("event") == "done" for e in events)
    # Intermediates were cleaned (or at least emptied)
    clips_dir = tmp_path / "out" / "_clips"
    compose_dir = tmp_path / "out" / "_compose"
    leftover_clips = list(clips_dir.iterdir()) if clips_dir.exists() else []
    leftover_compose = list(compose_dir.iterdir()) if compose_dir.exists() else []
    assert not leftover_clips, f"_clips still has: {leftover_clips}"
    assert not leftover_compose, f"_compose still has: {leftover_compose}"
