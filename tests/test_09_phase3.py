"""Phase 3 — quality bumps: translate, dedup, generative factory, avatar overlay."""
import importlib
import subprocess
from pathlib import Path

import pytest

from app.services.llm.translate import (
    is_already_english, translate_to_english, _CACHE,
)
from app.services.llm.chain import LLMKeys
from app.services.sync.dedup import DedupStore


# ─── module-level imports work ──────────────────────────────────────────

@pytest.mark.parametrize("mod", [
    "app.services.sync.dedup",
    "app.services.llm.translate",
    "app.services.generate.veo",
    "app.services.generate.runway",
    "app.services.generate.factory",
    "app.services.tts.f5_clone",
    "app.services.avatar.latentsync",
])
def test_phase3_modules_import(mod):
    importlib.import_module(mod)


# ─── translate ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("text,expected", [
    ("the quick brown fox jumps over the lazy dog", True),
    ("Hello world", True),
    ("dog", True),                                     # short, ASCII → assumed EN
    ("table tennis paddle", True),
    ("a casa é grande e bonita", False),               # accents → not EN
    ("北京欢迎你", False),                              # non-ASCII
    ("estaba caminando por la calle ayer noche", False),  # Spanish, has stopwords
])
def test_is_already_english(text, expected):
    assert is_already_english(text) is expected


def test_translate_passthrough_when_english(monkeypatch):
    _CACHE.clear()
    monkeypatch.setattr("app.services.llm.translate.call_chain",
                        lambda *a, **kw: pytest.fail("should not call LLM"))
    assert translate_to_english("table tennis paddle", LLMKeys()) == "table tennis paddle"


def test_translate_caches_result(monkeypatch):
    _CACHE.clear()
    calls = []
    def fake_call(prompt, keys, **kw):
        calls.append(prompt)
        return "table tennis paddle"
    monkeypatch.setattr("app.services.llm.translate.call_chain", fake_call)
    out1 = translate_to_english("a raquete de tênis de mesa", LLMKeys())
    out2 = translate_to_english("a raquete de tênis de mesa", LLMKeys())
    assert out1 == out2 == "table tennis paddle"
    assert len(calls) == 1  # cache hit on 2nd call


def test_translate_falls_back_on_llm_failure(monkeypatch):
    _CACHE.clear()
    monkeypatch.setattr("app.services.llm.translate.call_chain",
                        lambda *a, **kw: None)
    # Non-English with no LLM → returns original
    assert translate_to_english("ñoño raríssimo", LLMKeys()) == "ñoño raríssimo"


# ─── dedup ─────────────────────────────────────────────────────────────

class _FakeHash:
    """Mimics imagehash.ImageHash: subtraction returns Hamming distance."""
    def __init__(self, n: int): self.n = n
    def __sub__(self, other): return abs(self.n - other.n)


def test_dedup_first_clip_is_not_duplicate():
    ds = DedupStore()
    assert ds.is_duplicate([_FakeHash(0)]) is False


def test_dedup_catches_near_duplicate():
    ds = DedupStore(threshold=6)
    ds.add([_FakeHash(100)])
    assert ds.is_duplicate([_FakeHash(102)]) is True   # distance 2 < 6
    assert ds.is_duplicate([_FakeHash(200)]) is False  # distance 100 > 6


def test_dedup_accumulates():
    ds = DedupStore()
    ds.add([_FakeHash(10), _FakeHash(20)])
    ds.add([_FakeHash(30)])
    assert len(ds.seen) == 3


# ─── generative factory ────────────────────────────────────────────────

def test_generative_factory_returns_none_without_keys(monkeypatch):
    for k in ("GEMINI_API_KEY", "VEO_API_KEY", "RUNWAY_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    from app.services.generate.factory import build_generative_fn
    assert build_generative_fn() is None


def test_generative_factory_picks_veo_when_gemini_set(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-123")
    monkeypatch.delenv("RUNWAY_API_KEY", raising=False)
    from app.services.generate.factory import build_generative_fn
    fn = build_generative_fn()
    assert fn is not None
    assert callable(fn)


# ─── avatar overlay (ffmpeg integration) ───────────────────────────────

import shutil
_HAS_FFMPEG = shutil.which("ffmpeg") is not None


@pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg unavailable")
def test_avatar_overlay_corner(tmp_path):
    from app.services.avatar.latentsync import overlay_corner
    bg = tmp_path / "bg.mp4"
    av = tmp_path / "av.mp4"
    subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i",
                    "color=c=blue:s=320x240:r=24", "-t", "2",
                    "-c:v", "libx264", "-preset", "ultrafast",
                    "-pix_fmt", "yuv420p", str(bg)],
                   capture_output=True, timeout=30)
    subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i",
                    "color=c=red:s=160x120:r=24", "-t", "2",
                    "-c:v", "libx264", "-preset", "ultrafast",
                    "-pix_fmt", "yuv420p", str(av)],
                   capture_output=True, timeout=30)
    out = tmp_path / "overlay.mp4"
    overlay_corner(str(bg), str(av), str(out), position="TOP_LEFT", w_frac=0.3)
    assert out.exists() and out.stat().st_size > 2000
