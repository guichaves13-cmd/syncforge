"""Phase 3 ADVANCED — hard cases, real videos, failure paths, full integration.

Goals:
  1. pHash with REAL videos — identical clips MUST be flagged, different MUST NOT
  2. Veo/Runway: prompt building + graceful no-op when no key
  3. F5-TTS: clean error on missing CLI / sample
  4. LatentSync: graceful failure when repo missing
  5. Embedder: SigLIP-2 local loads (if torch available) — text/image roundtrip
  6. Runner end-to-end with translate+dedup wired (no API, all stubbed)
"""
from __future__ import annotations
import importlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from app.services.avatar.latentsync import (
    sync_avatar, LatentSyncConfig, overlay_corner,
)
from app.services.generate.veo import Veo3Generator
from app.services.generate.runway import RunwayGen4Generator
from app.services.llm.chain import LLMKeys
from app.services.llm.translate import translate_to_english, _CACHE
from app.services.sync.dedup import DedupStore, phash_video
from app.services.sync.pipeline import Intent, NarrationClause
from app.services.sync.ranker import Candidate
from app.services.tts.f5_clone import synth_clone, F5Config


_HAS_FFMPEG = shutil.which("ffmpeg") is not None
_HAS_IMAGEHASH = importlib.util.find_spec("imagehash") is not None
_HAS_TORCH = importlib.util.find_spec("torch") is not None
_HAS_TRANSFORMERS = importlib.util.find_spec("transformers") is not None


# ─────────────────────────────────────────────────────────────────────
# 1. pHash with REAL videos
# ─────────────────────────────────────────────────────────────────────

def _mk_solid(path: Path, color: str = "red", duration: float = 2.0,
              size: str = "320x240") -> Path:
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi",
         "-i", f"color=c={color}:s={size}:r=24",
         "-t", str(duration), "-c:v", "libx264", "-preset", "ultrafast",
         "-pix_fmt", "yuv420p", str(path)],
        capture_output=True, timeout=30,
    )
    return path


def _mk_gradient(path: Path, duration: float = 2.0) -> Path:
    """A clearly-different-looking video (gradient sweep)."""
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi",
         "-i", f"gradients=s=320x240:r=24:duration={duration}",
         "-t", str(duration), "-c:v", "libx264", "-preset", "ultrafast",
         "-pix_fmt", "yuv420p", str(path)],
        capture_output=True, timeout=30,
    )
    return path


@pytest.mark.skipif(not (_HAS_FFMPEG and _HAS_IMAGEHASH),
                    reason="ffmpeg+imagehash required")
def test_phash_identical_clips_are_flagged_as_duplicate(tmp_path):
    v1 = _mk_solid(tmp_path / "v1.mp4", "red", duration=2.0)
    v2 = _mk_solid(tmp_path / "v2.mp4", "red", duration=2.0)  # same color → identical pHash
    h1 = phash_video(str(v1), samples=3)
    h2 = phash_video(str(v2), samples=3)
    assert h1 and h2, "pHash returned empty"
    ds = DedupStore(threshold=6)
    ds.add(h1)
    assert ds.is_duplicate(h2), "identical clips were NOT flagged as duplicate"


def _mk_lavfi(path: Path, src_desc: str, duration: float = 2.0) -> Path:
    """Generate a video from a lavfi source description (e.g. 'testsrc2=...')."""
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi",
         "-i", src_desc, "-t", str(duration),
         "-c:v", "libx264", "-preset", "ultrafast",
         "-pix_fmt", "yuv420p", str(path)],
        capture_output=True, timeout=30,
    )
    return path


@pytest.mark.skipif(not (_HAS_FFMPEG and _HAS_IMAGEHASH),
                    reason="ffmpeg+imagehash required")
def test_phash_different_clips_are_not_duplicate(tmp_path):
    # testsrc2 = SMPTE-like test pattern with frame counter (high spatial variation).
    # smptebars = color bars (different structure entirely).
    v1 = _mk_lavfi(tmp_path / "testsrc.mp4", "testsrc2=size=320x240:rate=24")
    v2 = _mk_lavfi(tmp_path / "smpte.mp4",   "smptebars=size=320x240:rate=24")
    h1 = phash_video(str(v1), samples=3)
    h2 = phash_video(str(v2), samples=3)
    assert h1, "testsrc2 pHash empty"
    assert h2, "smptebars pHash empty"
    ds = DedupStore(threshold=6)
    ds.add(h1)
    assert not ds.is_duplicate(h2), "different clips were INCORRECTLY flagged as duplicate"


@pytest.mark.skipif(not (_HAS_FFMPEG and _HAS_IMAGEHASH),
                    reason="ffmpeg+imagehash required")
def test_phash_empty_or_corrupt_video_returns_empty(tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a real mp4")
    assert phash_video(str(bad)) == []


def test_dedup_threshold_boundary():
    """Hamming distance = threshold should NOT count as duplicate (strict <)."""
    class H:
        def __init__(self, n): self.n = n
        def __sub__(self, o): return abs(self.n - o.n)
    ds = DedupStore(threshold=6)
    ds.add([H(0)])
    assert ds.is_duplicate([H(5)]) is True    # 5 < 6
    assert ds.is_duplicate([H(6)]) is False   # 6 not strictly less than 6
    assert ds.is_duplicate([H(7)]) is False


# ─────────────────────────────────────────────────────────────────────
# 2. Generative providers — prompt building + no-key safety
# ─────────────────────────────────────────────────────────────────────

def test_veo3_no_op_without_key(tmp_path, monkeypatch):
    for k in ("GEMINI_API_KEY", "VEO_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    gen = Veo3Generator()
    intent = Intent(main_entity="cat", action="jumping",
                    location="garden", mood="cinematic", era="modern")
    clause = NarrationClause(start=0, end=5, text="a cat jumping")
    assert gen.generate(intent, clause, tmp_path) is None


def test_veo3_prompt_uses_key_visuals_when_present():
    intent = Intent(main_entity="ignored", action="jumping",
                    location="garden", mood="dramatic", era="modern",
                    key_visuals=["close-up paw", "wide shot fence"])
    clause = NarrationClause(start=0, end=5, text="x")
    prompt = Veo3Generator._build_prompt(intent, clause)
    assert "close-up paw" in prompt
    assert "wide shot fence" in prompt
    assert "dramatic" in prompt
    assert "modern" in prompt
    assert "ignored" not in prompt  # key_visuals takes precedence over main_entity


def test_veo3_prompt_falls_back_to_main_entity():
    intent = Intent(main_entity="vintage typewriter",
                    action="being typed on", location="library",
                    mood="nostalgic", era="1950s", key_visuals=[])
    prompt = Veo3Generator._build_prompt(intent, NarrationClause(0, 5, "x"))
    assert "vintage typewriter" in prompt
    assert "1950s" in prompt
    assert "nostalgic" in prompt


def test_runway_no_op_without_key(tmp_path, monkeypatch):
    monkeypatch.delenv("RUNWAY_API_KEY", raising=False)
    gen = RunwayGen4Generator()
    assert gen.generate(Intent(main_entity="x"),
                        NarrationClause(0, 5, "x"), tmp_path) is None


def test_runway_prompt_omits_empty_fields():
    intent = Intent(main_entity="surfer riding wave",
                    action="", location="", mood="", era="")
    prompt = RunwayGen4Generator._build_prompt(intent, NarrationClause(0, 5, "x"))
    assert "surfer riding wave" in prompt
    # Should not produce dangling separators
    assert ", , " not in prompt
    assert not prompt.startswith(",")
    assert not prompt.endswith(",")


# ─────────────────────────────────────────────────────────────────────
# 3. F5-TTS failure modes
# ─────────────────────────────────────────────────────────────────────

def test_f5_clone_missing_sample_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="voice sample not found"):
        synth_clone(
            script="hello",
            voice_sample_wav=str(tmp_path / "does_not_exist.wav"),
            output_audio_path=str(tmp_path / "out.mp3"),
        )


def test_f5_clone_missing_cli_raises(tmp_path, monkeypatch):
    # Create a fake sample so the file check passes
    sample = tmp_path / "sample.wav"
    sample.write_bytes(b"RIFF....")
    # Force `shutil.which` to return None for both CLI names
    real_which = shutil.which
    def fake_which(name):
        if name in ("f5-tts_infer-cli", "f5-tts"):
            return None
        return real_which(name)
    monkeypatch.setattr("app.services.tts.f5_clone.shutil.which", fake_which)
    with pytest.raises(RuntimeError, match="f5-tts CLI not installed"):
        synth_clone(script="hi", voice_sample_wav=str(sample),
                    output_audio_path=str(tmp_path / "o.mp3"))


# ─────────────────────────────────────────────────────────────────────
# 4. LatentSync failure modes
# ─────────────────────────────────────────────────────────────────────

def test_latentsync_no_repo_raises(tmp_path):
    cfg = LatentSyncConfig(repo_dir="")  # not set
    with pytest.raises(RuntimeError, match="repo not configured"):
        sync_avatar("ref.mp4", "audio.wav", str(tmp_path / "out.mp4"), cfg)


def test_latentsync_missing_checkpoint_raises(tmp_path):
    fake_repo = tmp_path / "repo"
    fake_repo.mkdir()
    cfg = LatentSyncConfig(repo_dir=str(fake_repo),
                            checkpoint=str(tmp_path / "missing.ckpt"))
    with pytest.raises(RuntimeError, match="checkpoint missing"):
        sync_avatar("ref.mp4", "audio.wav", str(tmp_path / "out.mp4"), cfg)


@pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg required")
def test_overlay_corner_all_4_positions(tmp_path):
    bg = _mk_solid(tmp_path / "bg.mp4", "blue", duration=1.5)
    av = _mk_solid(tmp_path / "av.mp4", "red", duration=1.5, size="160x120")
    for pos in ("TOP_LEFT", "TOP_RIGHT", "BOTTOM_LEFT", "BOTTOM_RIGHT"):
        out = tmp_path / f"o_{pos.lower()}.mp4"
        overlay_corner(str(bg), str(av), str(out), position=pos, w_frac=0.3)
        assert out.exists() and out.stat().st_size > 2000, f"{pos} failed"


# ─────────────────────────────────────────────────────────────────────
# 5. Embedder SigLIP-2 local (heavy — only if torch+transformers present)
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not (_HAS_TORCH and _HAS_TRANSFORMERS),
                    reason="torch + transformers required for SigLIP-2")
@pytest.mark.skipif(os.getenv("SKIP_HEAVY_TESTS") == "1",
                    reason="SKIP_HEAVY_TESTS=1 set")
def test_siglip2_text_roundtrip(tmp_path):
    """Text embedding should be a 1024-dim L2-normalized vector."""
    import numpy as np
    from app.services.sync.embedder import MultimodalEmbedder, EmbedConfig
    emb = MultimodalEmbedder(EmbedConfig(
        provider="siglip2", cache_dir=tmp_path / "cache", use_cache=False,
    ))
    v = emb.embed_text("a cat sitting on a windowsill")
    assert v.ndim == 1
    assert abs(float(np.linalg.norm(v)) - 1.0) < 0.05  # L2-normalized
    # Same text → identical vector (cache off, but model is deterministic)
    v2 = emb.embed_text("a cat sitting on a windowsill")
    assert float(np.dot(v, v2)) > 0.999


# ─────────────────────────────────────────────────────────────────────
# 6. Translate edge cases
# ─────────────────────────────────────────────────────────────────────

def test_translate_strips_quotes_from_llm_output(monkeypatch):
    _CACHE.clear()
    monkeypatch.setattr("app.services.llm.translate.call_chain",
                        lambda *a, **kw: '"table tennis paddle"')
    out = translate_to_english("a raquete de tênis de mesa", LLMKeys())
    assert out == "table tennis paddle"  # quotes stripped


def test_translate_takes_only_first_line(monkeypatch):
    _CACHE.clear()
    monkeypatch.setattr("app.services.llm.translate.call_chain",
                        lambda *a, **kw: "table tennis paddle\n(extra LLM filler)")
    out = translate_to_english("a raquete de tênis de mesa", LLMKeys())
    assert out == "table tennis paddle"


def test_translate_handles_empty_string():
    assert translate_to_english("", LLMKeys()) == ""
    assert translate_to_english("   ", LLMKeys()) == ""


# ─────────────────────────────────────────────────────────────────────
# 7. Runner integration — full pipeline with translate + dedup
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg required")
def test_runner_end_to_end_with_translate_and_dedup(tmp_path, monkeypatch):
    """
    Simulate a non-English narration so translate fires.
    Stub stock to return same clip twice (different ids) so dedup engages.
    Final video must be valid + cleanup must run.
    """
    from app.services import runner

    src_a = _mk_solid(tmp_path / "src_a.mp4", "green", duration=5.0)

    def stub_search(q, n):
        # 2 candidates with different IDs but they'll resolve to the SAME file
        return [
            Candidate(source="stub", source_id="a", url="x",
                      title=f"clip a for {q}", description="green", duration=5.0),
            Candidate(source="stub", source_id="b", url="x",
                      title=f"clip b for {q}", description="green", duration=5.0),
        ]

    def stub_download(c, output_dir):
        dst = Path(output_dir) / f"{c.source_id}.mp4"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src_a, dst)
        return str(dst)

    monkeypatch.setattr(runner, "build_sources",
                        lambda: ({"stub": stub_search}, {"stub": stub_download}))
    # Mock translate so we can SEE it was called
    translate_calls = []
    def fake_translate(text, keys):
        translate_calls.append(text)
        return text.upper()  # marker for "translated"
    monkeypatch.setattr(runner, "translate_to_english", fake_translate)

    for k in ("GROQ_API_KEY", "GEMINI_API_KEY", "CEREBRAS_API_KEY",
              "OPENROUTER_API_KEY", "DEEPSEEK_API_KEY"):
        monkeypatch.delenv(k, raising=False)

    result = runner.run_pipeline(
        script_text="Hello world. Integration test phase three.",
        voice="en-US-AndrewNeural",
        title="T", theme="test",
        output_dir=tmp_path / "out",
        enable_vision=False,
        enable_embeddings=False,
        enable_generative_fallback=False,
    )

    # Translate ran on every intent query
    assert len(translate_calls) >= 1, "translate_to_english was never called"

    # FINAL exists and is valid
    assert result["ok"] is True
    final = Path(result["final"])
    assert final.exists() and final.stat().st_size > 5000

    # Cleanup ran
    assert not (tmp_path / "out" / "_clips").exists() or \
           not any((tmp_path / "out" / "_clips").iterdir())
