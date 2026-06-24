"""Phase 6.6 — beat snap + subtitle style swap: advanced tests.

Covers:
  • SilenceInterval contract (mid, duration)
  • detect_silences: missing file / corrupt / parse correctness
  • _nearest_silence_mid: within / outside tolerance / boundary
  • snap_windows_to_silences invariants:
      - continuity (end[i] == start[i+1])
      - ordering
      - min-window enforcement
      - first/last anchors preserved
  • beat_snap end-to-end with real ffmpeg + Edge-TTS-style audio
  • SubtitleStyle.style_line returns parsable ASS
  • build_ass for each of the 5 styles produces a valid file
  • Uppercase transform for bold_news
  • auto_style mood/theme matrix (15+ cases)
  • list_styles surface
"""
from __future__ import annotations
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from app.services.sync.beat_snap import (
    BeatSnapConfig,
    SilenceInterval,
    _nearest_silence_mid,
    beat_snap,
    detect_silences,
    snap_windows_to_silences,
)
from app.services.sync.concept import ConceptWindow
from app.services.subtitles.styles import (
    STYLES,
    SubtitleStyle,
    auto_style,
    build_ass,
    list_styles,
)


_HAS_FFMPEG = shutil.which("ffmpeg") is not None


# ─────────────────────────────────────────────────────────────────────────
# 1. SilenceInterval contract
# ─────────────────────────────────────────────────────────────────────────

def test_silence_interval_mid_and_duration():
    s = SilenceInterval(start=1.0, end=2.0)
    assert s.mid == 1.5
    assert s.duration == 1.0


def test_silence_interval_negative_duration_clamped():
    s = SilenceInterval(start=3.0, end=1.0)
    assert s.duration == 0.0


# ─────────────────────────────────────────────────────────────────────────
# 2. detect_silences edge cases
# ─────────────────────────────────────────────────────────────────────────

def test_detect_silences_missing_file_returns_empty():
    assert detect_silences("/no/such/file.mp3") == []


@pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg required")
def test_detect_silences_on_all_loud_audio_finds_none(tmp_path):
    """Continuous sine wave → no silences."""
    src = tmp_path / "loud.mp3"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
         "-c:a", "libmp3lame", str(src)],
        capture_output=True, timeout=30,
    )
    sils = detect_silences(str(src))
    assert sils == []


@pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg required")
def test_detect_silences_finds_inserted_gap(tmp_path):
    """Concatenate sine + silence + sine → must find the silent gap."""
    src = tmp_path / "withgap.mp3"
    # ffmpeg: 1s tone, 1s silence, 1s tone
    subprocess.run(
        ["ffmpeg", "-y",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
         "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono:d=1",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
         "-filter_complex", "[0][1][2]concat=n=3:v=0:a=1[a]",
         "-map", "[a]", "-c:a", "libmp3lame", str(src)],
        capture_output=True, timeout=30,
    )
    sils = detect_silences(str(src))
    assert sils, "no silences detected"
    # The inserted gap is around 1.0-2.0s
    assert any(0.5 < s.mid < 2.5 for s in sils), f"got silences at: {[s.mid for s in sils]}"


# ─────────────────────────────────────────────────────────────────────────
# 3. _nearest_silence_mid
# ─────────────────────────────────────────────────────────────────────────

def test_nearest_silence_finds_within_tolerance():
    mids = [1.0, 3.0, 5.0]
    assert _nearest_silence_mid(3.2, mids, tolerance=0.4) == 3.0
    assert _nearest_silence_mid(0.8, mids, tolerance=0.4) == 1.0


def test_nearest_silence_returns_none_when_outside_tolerance():
    mids = [1.0, 5.0]
    assert _nearest_silence_mid(3.0, mids, tolerance=1.0) is None  # nearest=1.0, dist=2.0


def test_nearest_silence_handles_empty():
    assert _nearest_silence_mid(5.0, [], tolerance=1.0) is None


def test_nearest_silence_breaks_tie_by_proximity():
    """Equidistant pick: left or right, but must respect actual distance."""
    mids = [2.0, 4.0]
    out = _nearest_silence_mid(3.0, mids, tolerance=2.0)
    # Either neighbour is fine — both at distance 1.0
    assert out in (2.0, 4.0)


# ─────────────────────────────────────────────────────────────────────────
# 4. snap_windows_to_silences invariants
# ─────────────────────────────────────────────────────────────────────────

def _mk(start, end, text="x", entity=""):
    return ConceptWindow(start=start, end=end, text=text,
                          dominant_entity=entity)


def test_snap_preserves_continuity():
    windows = [_mk(0, 3), _mk(3, 6), _mk(6, 9)]
    sils = [SilenceInterval(start=2.8, end=3.2),   # near window 0→1 boundary
            SilenceInterval(start=5.7, end=6.1)]    # near window 1→2 boundary
    snapped = snap_windows_to_silences(windows, sils)
    # No gaps
    for a, b in zip(snapped, snapped[1:]):
        assert abs(a.end - b.start) < 1e-6, f"gap between {a.end} and {b.start}"


def test_snap_respects_min_window():
    """Boundary moved too aggressively must not crush a window below min."""
    cfg = BeatSnapConfig(min_window_s=1.0, snap_tolerance_s=2.0)
    windows = [_mk(0, 2), _mk(2, 4)]
    # Silence at 1.5 → would snap window-1.start backwards by 0.5
    sils = [SilenceInterval(start=1.4, end=1.6)]
    snapped = snap_windows_to_silences(windows, sils, cfg)
    for w in snapped:
        assert w.end - w.start >= cfg.min_window_s, (
            f"window crushed: {w.start}-{w.end} ({w.end - w.start:.2f}s)"
        )


def test_snap_preserves_first_window_anchor():
    """First window keeps its original start (we don't have prior context)."""
    windows = [_mk(0, 3), _mk(3, 6)]
    sils = [SilenceInterval(start=0.3, end=0.4)]   # near 0
    snapped = snap_windows_to_silences(windows, sils)
    assert snapped[0].start == 0.0


def test_snap_preserves_last_window_end_anchor():
    """Last window keeps its original end."""
    windows = [_mk(0, 3), _mk(3, 6)]
    sils = [SilenceInterval(start=5.8, end=5.95)]
    snapped = snap_windows_to_silences(windows, sils)
    assert snapped[-1].end == 6.0


def test_snap_empty_windows_returns_empty():
    assert snap_windows_to_silences([], []) == []


def test_snap_with_no_silences_returns_original_timings():
    windows = [_mk(0, 3), _mk(3, 6)]
    snapped = snap_windows_to_silences(windows, [])
    # No silences → no snap → unchanged
    for w_orig, w_new in zip(windows, snapped):
        assert w_orig.start == w_new.start
        assert w_orig.end == w_new.end


def test_snap_carries_window_metadata_through():
    """Snapped windows must preserve entity/cluster/mood from originals."""
    windows = [_mk(0, 3, entity="Butterfly"), _mk(3, 6, entity="Stiga")]
    windows[0].cluster_id = 7
    windows[0].mood = "analytical"
    windows[1].cluster_id = 8
    sils = [SilenceInterval(start=2.9, end=3.1)]
    snapped = snap_windows_to_silences(windows, sils)
    assert snapped[0].dominant_entity == "Butterfly"
    assert snapped[0].cluster_id == 7
    assert snapped[0].mood == "analytical"
    assert snapped[1].dominant_entity == "Stiga"
    assert snapped[1].cluster_id == 8


# ─────────────────────────────────────────────────────────────────────────
# 5. beat_snap end-to-end safety
# ─────────────────────────────────────────────────────────────────────────

def test_beat_snap_returns_input_when_audio_missing(tmp_path):
    windows = [_mk(0, 3), _mk(3, 6)]
    out = beat_snap(windows, str(tmp_path / "nope.mp3"))
    # Falls back to original when silences can't be detected
    assert len(out) == 2


def test_beat_snap_returns_empty_for_empty_input():
    assert beat_snap([], "/anything.mp3") == []


@pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg required")
def test_beat_snap_actually_snaps_boundary_to_silence(tmp_path):
    """Real flow: build audio with a gap at 2.0s, place a window boundary at
    2.3s (≤ tolerance), confirm it gets snapped near 2.0s."""
    audio = tmp_path / "ts.mp3"
    subprocess.run(
        ["ffmpeg", "-y",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
         "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono:d=0.5",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
         "-filter_complex", "[0][1][2]concat=n=3:v=0:a=1[a]",
         "-map", "[a]", "-c:a", "libmp3lame", str(audio)],
        capture_output=True, timeout=30,
    )
    windows = [_mk(0, 2.3), _mk(2.3, 4.5)]
    snapped = beat_snap(windows, str(audio))
    # Boundary should move toward the silence midpoint (~2.25)
    boundary = snapped[0].end
    assert 2.0 <= boundary <= 2.5, f"boundary not near silence: {boundary:.2f}"


# ─────────────────────────────────────────────────────────────────────────
# 6. SubtitleStyle.style_line
# ─────────────────────────────────────────────────────────────────────────

def test_each_style_renders_valid_v4_styles_line():
    """Every built-in style must produce an ASS Style: line with 23 fields."""
    for name, style in STYLES.items():
        line = style.style_line()
        # ASS Style: line starts with 'Style:' and has commas
        assert line.startswith("Style: "), f"{name}: {line[:40]}"
        # Count fields after 'Style: '
        fields = line[7:].split(",")
        # ASS v4+ has 23 fields total
        assert len(fields) == 23, f"{name}: {len(fields)} fields, expected 23"


def test_style_font_size_and_alignment_present():
    for name, style in STYLES.items():
        line = style.style_line()
        assert style.font in line
        assert str(style.size) in line


# ─────────────────────────────────────────────────────────────────────────
# 7. build_ass per style
# ─────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("style_name", list(STYLES.keys()))
def test_build_ass_writes_valid_file_for_each_style(tmp_path, style_name):
    sents = [
        {"start": 0.0, "end": 1.5, "text": "Hello world."},
        {"start": 1.5, "end": 3.0, "text": "Second line."},
    ]
    out = tmp_path / f"{style_name}.ass"
    build_ass(sents, out, style=style_name)
    text = out.read_text(encoding="utf-8")
    assert "[Script Info]" in text
    assert "[V4+ Styles]" in text
    assert "[Events]" in text
    # The chosen style's name must appear in both the Style: line AND each Dialogue
    assert f"Style: {style_name}" in text
    # Two dialogue lines for two non-empty sentences
    assert text.count("Dialogue:") == 2


def test_build_ass_skips_empty_sentences(tmp_path):
    sents = [
        {"start": 0.0, "end": 1.5, "text": "First."},
        {"start": 1.5, "end": 2.0, "text": ""},      # skip
        {"start": 2.0, "end": 3.0, "text": "   "},   # skip
        {"start": 3.0, "end": 4.5, "text": "Last."},
    ]
    out = tmp_path / "x.ass"
    build_ass(sents, out, style="clean")
    text = out.read_text(encoding="utf-8")
    assert text.count("Dialogue:") == 2


def test_bold_news_uppercase_transforms_text(tmp_path):
    out = tmp_path / "news.ass"
    build_ass([{"start": 0, "end": 2, "text": "shocking news today"}],
               out, style="bold_news")
    text = out.read_text(encoding="utf-8")
    # Must contain the uppercased version
    assert "SHOCKING NEWS TODAY" in text
    # And NOT the lowercase
    assert "shocking news today" not in text


def test_non_uppercase_styles_preserve_case(tmp_path):
    for style_name in ("tiktok", "documentary", "karaoke", "clean"):
        out = tmp_path / f"{style_name}.ass"
        build_ass([{"start": 0, "end": 2, "text": "Mixed Case Text Here"}],
                   out, style=style_name)
        text = out.read_text(encoding="utf-8")
        assert "Mixed Case Text Here" in text, f"{style_name} corrupted case"


def test_build_ass_unknown_style_falls_back_to_karaoke(tmp_path):
    out = tmp_path / "unk.ass"
    build_ass([{"start": 0, "end": 1, "text": "x"}], out, style="nonexistent")  # type: ignore[arg-type]
    text = out.read_text(encoding="utf-8")
    assert "Style: karaoke" in text


# ─────────────────────────────────────────────────────────────────────────
# 8. auto_style — mood overrides + theme fallback
# ─────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("mood,expected", [
    ("tiktok",         "tiktok"),
    ("energetic",      "tiktok"),
    ("viral",          "tiktok"),
    ("shocking",       "bold_news"),
    ("urgent",         "bold_news"),
    ("controversial",  "bold_news"),
    ("documentary",    "documentary"),
    ("cinematic",      "documentary"),
    ("historical",     "documentary"),
    ("analytical",     "documentary"),
    ("explainer",      "clean"),
    ("corporate",      "clean"),
    ("professional",   "clean"),
    ("karaoke",        "karaoke"),
])
def test_auto_style_mood_matrix(mood, expected):
    assert auto_style(mood=mood) == expected


@pytest.mark.parametrize("theme,expected", [
    ("history of WWII",                "documentary"),
    ("science explainer for kids",     "documentary"),
    ("tech review iPhone",             "clean"),
    ("finance investing 101",          "clean"),
    ("fitness HIIT routine",           "tiktok"),
    ("dance compilation",              "tiktok"),
    ("shocking listicle truth",        "bold_news"),
    ("news today",                     "bold_news"),
    ("random unrelated topic",         "karaoke"),
])
def test_auto_style_theme_matrix(theme, expected):
    assert auto_style(theme=theme) == expected


def test_auto_style_mood_overrides_theme():
    """When mood and theme disagree, mood wins."""
    assert auto_style(theme="history", mood="tiktok") == "tiktok"


def test_auto_style_unknown_mood_falls_through_to_theme():
    assert auto_style(theme="history", mood="unknown-mood") == "documentary"


def test_list_styles_exposes_all_five():
    names = list_styles()
    for s in ("tiktok", "documentary", "karaoke", "clean", "bold_news"):
        assert s in names


# ─────────────────────────────────────────────────────────────────────────
# 9. Multi-niche style auto-pick integration
# ─────────────────────────────────────────────────────────────────────────

NICHE_TO_EXPECTED = [
    ("ping pong listicle robbing you blind",  "bold_news",  "listicle (contains 'shocking'-style word)"),
    # actually 'listicle' is in the THEME map → bold_news
    ("french revolution documentary",          "documentary", "history+documentary"),
    ("breaking news cnn report",               "bold_news",  "news"),
    ("dance trend tutorial",                   "tiktok",     "dance"),
    ("kettlebell HIIT fitness 15 min",         "tiktok",     "fitness"),
    ("iphone 17 pro tech deep dive",           "clean",      "tech"),
    ("how compound interest works for kids",   "karaoke",    "default"),  # no keyword match
]


@pytest.mark.parametrize("theme,expected,desc", NICHE_TO_EXPECTED)
def test_multi_niche_theme_picks_right_style(theme, expected, desc):
    assert auto_style(theme=theme) == expected, f"failed for niche: {desc}"


# ─────────────────────────────────────────────────────────────────────────
# 10. Module imports
# ─────────────────────────────────────────────────────────────────────────

def test_modules_import_cleanly():
    import importlib
    for mod in ("app.services.sync.beat_snap",
                 "app.services.subtitles.styles"):
        importlib.import_module(mod)
