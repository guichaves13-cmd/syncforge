"""Smart aspect-aware crop — keeps the salient subject in frame when
converting between aspect ratios (e.g. 16:9 stock → 9:16 TikTok).

Two backends:
  1. **Fast (default)** — Saliency-by-luminance: detect the column/row where
     visual energy peaks (gradient-magnitude on a downscaled luma plane) and
     center the crop on that point. No ML deps. ~50ms per clip.
  2. **Accurate (optional)** — OpenCV's `saliency.StaticSaliencyFineGrained`
     if cv2 is installed. ~200ms per clip but catches faces / focus subjects
     better than luminance peaks.

Both backends return a `CropPlan` describing the offsets for ffmpeg's
`crop=W:H:x:y` filter. The composer applies the plan; no destructive
re-encode is performed by this module itself.
"""
from __future__ import annotations
import shutil
import subprocess
import tempfile
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Backend = Literal["fast", "cv2", "off"]


@dataclass(frozen=True)
class CropPlan:
    """Output of saliency analysis — what ffmpeg should crop."""
    target_w: int
    target_h: int
    src_w: int
    src_h: int
    x_offset: int
    y_offset: int

    @property
    def is_passthrough(self) -> bool:
        """No actual cropping needed (already at target aspect)."""
        return (self.x_offset == 0 and self.y_offset == 0
                and self.target_w == self.src_w
                and self.target_h == self.src_h)

    def to_ffmpeg_filter(self) -> str:
        """Render to a `crop=W:H:x:y` filter fragment."""
        return f"crop={self.target_w}:{self.target_h}:{self.x_offset}:{self.y_offset}"


# ─────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────

def plan_crop(
    src_w: int, src_h: int,
    target_w: int, target_h: int,
    *,
    saliency_x: float | None = None,
    saliency_y: float | None = None,
) -> CropPlan:
    """Compute crop offsets that:
      - Match target aspect ratio
      - Center the crop on (saliency_x, saliency_y) when provided (0..1 floats)
      - Default to image center when no saliency hint is given
    """
    if src_w <= 0 or src_h <= 0 or target_w <= 0 or target_h <= 0:
        raise ValueError(f"non-positive dimensions: {src_w}x{src_h} → {target_w}x{target_h}")

    src_aspect = src_w / src_h
    tgt_aspect = target_w / target_h

    if abs(src_aspect - tgt_aspect) < 1e-3:
        # Already same aspect — pure scale, no crop
        return CropPlan(target_w=src_w, target_h=src_h,
                         src_w=src_w, src_h=src_h, x_offset=0, y_offset=0)

    # Crop in the dimension that's "too wide"
    if src_aspect > tgt_aspect:
        # Source wider → crop width
        crop_w = max(1, int(src_h * tgt_aspect))
        crop_h = src_h
        sx = saliency_x if saliency_x is not None else 0.5
        sx = max(0.0, min(1.0, sx))
        # Center the crop on the saliency point but clip to bounds
        x = int(sx * src_w - crop_w / 2)
        x = max(0, min(src_w - crop_w, x))
        return CropPlan(target_w=crop_w, target_h=crop_h,
                         src_w=src_w, src_h=src_h, x_offset=x, y_offset=0)
    else:
        # Source taller → crop height
        crop_w = src_w
        crop_h = max(1, int(src_w / tgt_aspect))
        sy = saliency_y if saliency_y is not None else 0.5
        sy = max(0.0, min(1.0, sy))
        y = int(sy * src_h - crop_h / 2)
        y = max(0, min(src_h - crop_h, y))
        return CropPlan(target_w=crop_w, target_h=crop_h,
                         src_w=src_w, src_h=src_h, x_offset=0, y_offset=y)


# ─────────────────────────────────────────────────────────────────────────
# Saliency detection
# ─────────────────────────────────────────────────────────────────────────

def detect_saliency(video_path: str, *, backend: Backend = "fast",
                     ffmpeg_bin: str = "ffmpeg") -> tuple[float, float]:
    """Return the (x, y) saliency centroid as floats in [0, 1].
    Returns (0.5, 0.5) on failure (center fallback)."""
    if backend == "off":
        return 0.5, 0.5
    try:
        if backend == "cv2":
            return _detect_cv2(video_path, ffmpeg_bin)
        return _detect_fast(video_path, ffmpeg_bin)
    except Exception:
        return 0.5, 0.5


def _detect_fast(video_path: str, ffmpeg_bin: str) -> tuple[float, float]:
    """Luminance-gradient saliency from a single midframe extracted by ffmpeg."""
    # Extract midframe into a tiny grayscale image
    fd, jpg = tempfile.mkstemp(prefix="syncforge_sal_", suffix=".jpg")
    os.close(fd)
    try:
        # -ss before -i is fast seek; we sample at 1s in (assumes ≥1s clip)
        subprocess.run(
            [ffmpeg_bin, "-y", "-ss", "1.0", "-i", video_path,
             "-vframes", "1", "-vf", "scale=128:-1,format=gray",
             jpg],
            capture_output=True, timeout=15,
        )
        if not Path(jpg).exists() or Path(jpg).stat().st_size < 200:
            return 0.5, 0.5
        return _gradient_centroid(jpg)
    finally:
        try: Path(jpg).unlink(missing_ok=True)
        except: pass


def _gradient_centroid(jpg_path: str) -> tuple[float, float]:
    """Compute saliency as the weighted centroid of |dx| + |dy| over the image.
    Pure Python + PIL (no numpy required, but used for speed if present)."""
    try:
        from PIL import Image
    except ImportError:
        return 0.5, 0.5
    img = Image.open(jpg_path).convert("L")
    w, h = img.size
    if w < 4 or h < 4:
        return 0.5, 0.5
    px = img.load()
    sx = sy = sw = 0.0
    # Compute |dx|+|dy| on a coarse grid (every 2px) for speed
    step = 2
    for y in range(1, h - 1, step):
        for x in range(1, w - 1, step):
            dx = abs(px[x + 1, y] - px[x - 1, y])
            dy = abs(px[x, y + 1] - px[x, y - 1])
            w_ = dx + dy
            if w_ <= 4:
                continue
            sx += x * w_
            sy += y * w_
            sw += w_
    if sw < 1:
        return 0.5, 0.5
    return (sx / sw / w, sy / sw / h)


def _detect_cv2(video_path: str, ffmpeg_bin: str) -> tuple[float, float]:
    """OpenCV StaticSaliencyFineGrained — better with faces / objects."""
    try:
        import cv2          # type: ignore
        import numpy as np  # type: ignore
    except ImportError:
        return _detect_fast(video_path, ffmpeg_bin)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0.5, 0.5
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    target_frame = int(fps * 1.0)
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return 0.5, 0.5
    sal = cv2.saliency.StaticSaliencyFineGrained_create()
    ok, sal_map = sal.computeSaliency(frame)
    if not ok:
        return 0.5, 0.5
    h, w = sal_map.shape
    sal_map = sal_map / max(1e-6, float(np.max(sal_map)))
    # Centroid weighted by saliency
    ys, xs = np.indices(sal_map.shape)
    total = float(np.sum(sal_map))
    if total < 1e-6:
        return 0.5, 0.5
    cx = float(np.sum(xs * sal_map)) / total / w
    cy = float(np.sum(ys * sal_map)) / total / h
    return cx, cy


# ─────────────────────────────────────────────────────────────────────────
# Combined: detect + plan in one call
# ─────────────────────────────────────────────────────────────────────────

def smart_crop_plan(
    video_path: str,
    src_w: int, src_h: int,
    target_w: int, target_h: int,
    *,
    backend: Backend = "fast",
) -> CropPlan:
    """Convenience: detect saliency on the video and produce a CropPlan."""
    sx, sy = detect_saliency(video_path, backend=backend)
    return plan_crop(src_w, src_h, target_w, target_h,
                      saliency_x=sx, saliency_y=sy)
