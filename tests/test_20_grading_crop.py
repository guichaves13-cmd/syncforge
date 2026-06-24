"""Phase 6.5 — color grading + smart crop: advanced tests.

Covers:
  • build_grading_filter for all 7 presets (+ off → empty)
  • auto_preset: theme mapping, mood override
  • grade_clip with real ffmpeg + corrupt file
  • CropPlan invariants (aspect match, bounds, passthrough)
  • plan_crop across 15 aspect conversions
  • plan_crop with saliency hints (left/right/top/bottom subjects)
  • detect_saliency: off / fast / cv2 fallback
  • _gradient_centroid on synthetic images (single bright corner → near corner)
  • smart_crop_plan integration
  • Multi-niche aspect conversion stress (16:9 → 9:16 vertical for TikTok,
    16:9 → 1:1 for Instagram, etc.)
"""
from __future__ import annotations
import shutil
import subprocess
from pathlib import Path

import pytest

from app.services.render.grading import (
    GradeParams,
    auto_preset,
    build_grading_filter,
    grade_clip,
    list_presets,
    _PRESETS,
)
from app.services.render.smart_crop import (
    CropPlan,
    _gradient_centroid,
    detect_saliency,
    plan_crop,
    smart_crop_plan,
)


_HAS_FFMPEG = shutil.which("ffmpeg") is not None


# ─────────────────────────────────────────────────────────────────────────
# 1. build_grading_filter
# ─────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("preset", ["neutral", "cinematic", "documentary",
                                      "vibrant", "warm", "cool"])
def test_build_grading_filter_returns_eq_for_all_real_presets(preset):
    out = build_grading_filter(preset)
    assert out != ""
    assert out.startswith("eq=")
    # Has all four eq parameters
    assert "saturation=" in out
    assert "contrast=" in out
    assert "brightness=" in out
    assert "gamma=" in out


def test_build_grading_filter_off_returns_empty():
    assert build_grading_filter("off") == ""


def test_build_grading_filter_warm_includes_curves():
    out = build_grading_filter("warm")
    assert "curves=preset=lighter" in out


def test_build_grading_filter_unknown_falls_back_to_neutral():
    out = build_grading_filter("does-not-exist")  # type: ignore[arg-type]
    assert out.startswith("eq=")
    # Neutral params verifiable in the string
    assert "saturation=1.020" in out


def test_list_presets_has_all_seven():
    presets = list_presets()
    for p in ("off", "neutral", "cinematic", "documentary",
              "vibrant", "warm", "cool"):
        assert p in presets


# ─────────────────────────────────────────────────────────────────────────
# 2. auto_preset
# ─────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("theme,expected", [
    ("history of rome",                       "cinematic"),
    ("documentary about whales",              "documentary"),
    ("tech review iPhone",                    "cool"),
    ("science explainer",                     "cool"),
    ("cooking French onion soup",              "warm"),
    ("travel Bali",                            "vibrant"),
    ("fitness HIIT",                           "vibrant"),
    ("finance investing 101",                  "neutral"),
    ("anything totally unrelated",             "neutral"),
    ("",                                       "neutral"),
])
def test_auto_preset_theme_mapping(theme, expected):
    assert auto_preset(theme=theme) == expected


def test_auto_preset_mood_overrides_theme():
    """An explicit mood like 'cinematic' must override even a theme that
    would map elsewhere."""
    assert auto_preset(theme="cooking", mood="cinematic") == "cinematic"


def test_auto_preset_unknown_mood_falls_through_to_theme():
    assert auto_preset(theme="cooking", mood="surprise-me") == "warm"


# ─────────────────────────────────────────────────────────────────────────
# 3. grade_clip with real ffmpeg
# ─────────────────────────────────────────────────────────────────────────

def _make_clip(path: Path, duration: float = 2.0, size: str = "320x240") -> Path:
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi",
         "-i", f"testsrc2=size={size}:rate=24",
         "-t", f"{duration}",
         "-c:v", "libx264", "-preset", "ultrafast",
         "-pix_fmt", "yuv420p", str(path)],
        capture_output=True, timeout=30,
    )
    return path


@pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg required")
@pytest.mark.parametrize("preset", ["neutral", "cinematic", "cool", "warm"])
def test_grade_clip_produces_valid_output(tmp_path, preset):
    src = _make_clip(tmp_path / "src.mp4", duration=2.0)
    dst = tmp_path / f"{preset}.mp4"
    grade_clip(str(src), str(dst), preset=preset)
    assert dst.exists() and dst.stat().st_size > 10_000


@pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg required")
def test_grade_clip_off_still_re_encodes(tmp_path):
    """Preset 'off' returns empty filter — re-encode happens without -vf."""
    src = _make_clip(tmp_path / "src.mp4", duration=2.0)
    dst = tmp_path / "off.mp4"
    grade_clip(str(src), str(dst), preset="off")
    assert dst.exists() and dst.stat().st_size > 5_000


def test_grade_clip_raises_on_corrupt_input(tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"definitely not an mp4")
    with pytest.raises(RuntimeError, match="produced no output"):
        grade_clip(str(bad), str(tmp_path / "out.mp4"), preset="neutral")


# ─────────────────────────────────────────────────────────────────────────
# 4. CropPlan contract
# ─────────────────────────────────────────────────────────────────────────

def test_cropplan_to_ffmpeg_filter():
    plan = CropPlan(target_w=720, target_h=1280, src_w=1920, src_h=1080,
                     x_offset=600, y_offset=0)
    assert plan.to_ffmpeg_filter() == "crop=720:1280:600:0"


def test_cropplan_passthrough_detection():
    same = CropPlan(target_w=1920, target_h=1080, src_w=1920, src_h=1080,
                     x_offset=0, y_offset=0)
    assert same.is_passthrough
    diff = CropPlan(target_w=720, target_h=1280, src_w=1920, src_h=1080,
                     x_offset=600, y_offset=0)
    assert not diff.is_passthrough


# ─────────────────────────────────────────────────────────────────────────
# 5. plan_crop — aspect math
# ─────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("src_w,src_h,tgt_w,tgt_h", [
    # 16:9 → 9:16  (TikTok)
    (1920, 1080,  720, 1280),
    # 16:9 → 1:1   (Instagram)
    (1920, 1080, 1080, 1080),
    # 4:3  → 16:9
    (1280,  960, 1920, 1080),
    # Vertical 9:16 → landscape 16:9
    (1080, 1920, 1920, 1080),
    # Already matching aspect
    (1920, 1080, 1280,  720),
    # Square → portrait
    (1080, 1080,  720, 1280),
])
def test_plan_crop_output_matches_target_aspect(src_w, src_h, tgt_w, tgt_h):
    plan = plan_crop(src_w, src_h, tgt_w, tgt_h)
    # The PRODUCED crop region must match the target's aspect ratio
    src_aspect = src_w / src_h
    tgt_aspect = tgt_w / tgt_h
    if abs(src_aspect - tgt_aspect) < 1e-3:
        assert plan.is_passthrough
        return
    crop_aspect = plan.target_w / plan.target_h
    assert abs(crop_aspect - tgt_aspect) < 0.02, (
        f"{src_w}x{src_h}→{tgt_w}x{tgt_h}: crop aspect {crop_aspect:.3f} "
        f"vs target {tgt_aspect:.3f}"
    )


def test_plan_crop_clips_to_bounds_when_saliency_at_edge():
    """Saliency at x=0.99 (right edge) should NOT produce x_offset > src_w - target_w."""
    plan = plan_crop(1920, 1080, 720, 1280, saliency_x=0.99)
    assert plan.x_offset + plan.target_w <= 1920
    assert plan.x_offset >= 0


def test_plan_crop_clips_to_bounds_when_saliency_at_zero():
    plan = plan_crop(1920, 1080, 720, 1280, saliency_x=0.01)
    assert plan.x_offset == 0


def test_plan_crop_centers_when_no_saliency_hint():
    plan = plan_crop(1920, 1080, 720, 1280)
    # Crop region = 607x1080 (rounded); centered means x_offset ~= (1920-607)/2 = 656
    expected_x = (1920 - plan.target_w) // 2
    assert abs(plan.x_offset - expected_x) <= 1


def test_plan_crop_vertical_saliency_for_tall_source():
    """When cropping a 9:16 source to 16:9, the vertical position must respect saliency_y."""
    plan = plan_crop(1080, 1920, 1920, 1080, saliency_y=0.2)  # subject near top
    # Crop region is shorter than source → y_offset should be lower (toward top)
    assert plan.y_offset < (1920 - plan.target_h) // 2


def test_plan_crop_raises_on_zero_dimensions():
    with pytest.raises(ValueError, match="non-positive"):
        plan_crop(0, 1080, 720, 1280)
    with pytest.raises(ValueError, match="non-positive"):
        plan_crop(1920, 1080, 0, 1280)


def test_plan_crop_clamps_saliency_outside_unit_range():
    """Saliency >1 or <0 must be clamped, not crash."""
    plan_lo = plan_crop(1920, 1080, 720, 1280, saliency_x=-5.0)
    assert plan_lo.x_offset == 0
    plan_hi = plan_crop(1920, 1080, 720, 1280, saliency_x=10.0)
    assert plan_hi.x_offset + plan_hi.target_w <= 1920


# ─────────────────────────────────────────────────────────────────────────
# 6. _gradient_centroid on synthetic images
# ─────────────────────────────────────────────────────────────────────────

def _make_test_image(tmp_path, mode="right"):
    """Create a 128x128 grayscale JPG with a bright spot."""
    from PIL import Image
    img = Image.new("L", (128, 128), 0)
    px = img.load()
    if mode == "right":
        # Bright vertical bar on the right
        for x in range(100, 120):
            for y in range(20, 100):
                px[x, y] = 255
    elif mode == "top":
        for y in range(10, 30):
            for x in range(20, 100):
                px[x, y] = 255
    elif mode == "center":
        for x in range(54, 74):
            for y in range(54, 74):
                px[x, y] = 255
    out = tmp_path / f"sal_{mode}.jpg"
    img.save(out)
    return str(out)


def test_gradient_centroid_finds_right_subject(tmp_path):
    path = _make_test_image(tmp_path, mode="right")
    cx, cy = _gradient_centroid(path)
    assert cx > 0.6, f"expected right (>0.6), got {cx:.3f}"


def test_gradient_centroid_finds_top_subject(tmp_path):
    path = _make_test_image(tmp_path, mode="top")
    cx, cy = _gradient_centroid(path)
    assert cy < 0.4, f"expected top (<0.4), got {cy:.3f}"


def test_gradient_centroid_centered_subject(tmp_path):
    path = _make_test_image(tmp_path, mode="center")
    cx, cy = _gradient_centroid(path)
    assert 0.35 < cx < 0.65 and 0.35 < cy < 0.65


def test_gradient_centroid_blank_image_returns_center(tmp_path):
    from PIL import Image
    Image.new("L", (128, 128), 128).save(tmp_path / "flat.jpg")
    cx, cy = _gradient_centroid(str(tmp_path / "flat.jpg"))
    assert (cx, cy) == (0.5, 0.5)


def test_gradient_centroid_tiny_image_returns_center(tmp_path):
    from PIL import Image
    Image.new("L", (2, 2), 128).save(tmp_path / "tiny.jpg")
    cx, cy = _gradient_centroid(str(tmp_path / "tiny.jpg"))
    assert (cx, cy) == (0.5, 0.5)


# ─────────────────────────────────────────────────────────────────────────
# 7. detect_saliency end-to-end
# ─────────────────────────────────────────────────────────────────────────

def test_detect_saliency_off_backend_returns_center():
    sx, sy = detect_saliency("/path/does/not/matter.mp4", backend="off")
    assert (sx, sy) == (0.5, 0.5)


def test_detect_saliency_nonexistent_file_returns_center():
    sx, sy = detect_saliency("/path/does/not/exist.mp4", backend="fast")
    assert (sx, sy) == (0.5, 0.5)


@pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg required")
def test_detect_saliency_fast_on_real_clip(tmp_path):
    """testsrc2 has lots of motion + structure → saliency should be sensible."""
    clip = _make_clip(tmp_path / "src.mp4", duration=2.0)
    sx, sy = detect_saliency(str(clip), backend="fast")
    assert 0.0 <= sx <= 1.0
    assert 0.0 <= sy <= 1.0


# ─────────────────────────────────────────────────────────────────────────
# 8. smart_crop_plan integration
# ─────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg required")
def test_smart_crop_plan_returns_valid_plan(tmp_path):
    clip = _make_clip(tmp_path / "src.mp4", duration=2.0, size="640x480")
    plan = smart_crop_plan(str(clip), 640, 480, 360, 640, backend="fast")
    # Must be in bounds
    assert plan.x_offset >= 0
    assert plan.y_offset >= 0
    assert plan.x_offset + plan.target_w <= 640
    assert plan.y_offset + plan.target_h <= 480
    # Output aspect close to target
    assert abs((plan.target_w / plan.target_h) - (360 / 640)) < 0.05


def test_smart_crop_plan_falls_back_when_video_missing(tmp_path):
    """No real video → saliency detection returns center → plan_crop centers."""
    plan = smart_crop_plan(str(tmp_path / "nope.mp4"),
                            1920, 1080, 720, 1280, backend="fast")
    expected_x = (1920 - plan.target_w) // 2
    assert abs(plan.x_offset - expected_x) <= 1


# ─────────────────────────────────────────────────────────────────────────
# 9. Multi-niche aspect conversion stress
# ─────────────────────────────────────────────────────────────────────────

PRACTICAL_CONVERSIONS = [
    # (description, src_w, src_h, tgt_w, tgt_h)
    ("YouTube 16:9 → TikTok 9:16",        1920, 1080,  720, 1280),
    ("YouTube 16:9 → Instagram square",   1920, 1080, 1080, 1080),
    ("YouTube 16:9 → Reels 9:16 portrait", 1920, 1080, 1080, 1920),
    ("Phone 9:16 → YouTube 16:9",         1080, 1920, 1920, 1080),
    ("4K 16:9 → 1080p 16:9 (no crop)",    3840, 2160, 1920, 1080),
    ("Cinema 21:9 → 16:9",                2560, 1080, 1920, 1080),
    ("Square Instagram → Story 9:16",     1080, 1080,  720, 1280),
    ("Pinterest 2:3 → Instagram 1:1",      800, 1200, 1080, 1080),
]


@pytest.mark.parametrize("desc,sw,sh,tw,th", PRACTICAL_CONVERSIONS)
def test_practical_conversion_produces_valid_crop(desc, sw, sh, tw, th):
    """All 8 real-world conversions must produce in-bounds, aspect-correct crops."""
    plan = plan_crop(sw, sh, tw, th)
    # In-bounds
    assert 0 <= plan.x_offset <= sw - plan.target_w, desc
    assert 0 <= plan.y_offset <= sh - plan.target_h, desc
    # Aspect within 2%
    if not plan.is_passthrough:
        crop_a = plan.target_w / plan.target_h
        tgt_a = tw / th
        assert abs(crop_a - tgt_a) < 0.02, f"{desc}: {crop_a:.3f} vs {tgt_a:.3f}"


@pytest.mark.parametrize("desc,sw,sh,tw,th", PRACTICAL_CONVERSIONS)
def test_practical_conversion_with_left_saliency(desc, sw, sh, tw, th):
    """Subject on the left (saliency_x=0.15) should keep x_offset low."""
    plan = plan_crop(sw, sh, tw, th, saliency_x=0.15, saliency_y=0.5)
    if plan.is_passthrough:
        return
    # When cropping width: x_offset should be < center-x
    if plan.target_h == sh:  # cropping width
        center_x = (sw - plan.target_w) // 2
        assert plan.x_offset <= center_x, (
            f"{desc}: left-saliency didn't shift left "
            f"(x_offset={plan.x_offset}, center={center_x})"
        )


# ─────────────────────────────────────────────────────────────────────────
# 10. Final regression sanity
# ─────────────────────────────────────────────────────────────────────────

def test_modules_import_cleanly():
    import importlib
    for mod in ("app.services.render.grading",
                 "app.services.render.smart_crop"):
        importlib.import_module(mod)
