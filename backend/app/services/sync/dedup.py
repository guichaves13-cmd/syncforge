"""Perceptual hash anti-duplicate for video clips and photos.

Two clips with similar pHash are considered duplicates even if URLs differ.
Useful for catching:
  - Same Pexels clip uploaded twice under different IDs
  - YouTube re-uploads / mirror channels
  - Photos that are crops of the same source

We sample 3 frames per video (start, mid, end), compute pHash of each, then
compare against a global set. If the minimum Hamming distance to anything in
the set is below threshold (default 6), the clip is rejected.

Backed by `imagehash` (pip install imagehash). 64-bit hashes.
"""
from __future__ import annotations
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


@dataclass
class DedupStore:
    """In-process set of perceptual hashes seen so far."""
    seen: list = field(default_factory=list)  # list of imagehash.ImageHash
    threshold: int = 6  # Hamming distance below = duplicate

    def is_duplicate(self, new_hashes: Iterable) -> bool:
        for nh in new_hashes:
            for existing in self.seen:
                if (nh - existing) < self.threshold:
                    return True
        return False

    def add(self, hashes: Iterable) -> None:
        for h in hashes:
            self.seen.append(h)


def phash_video(video_path: str, samples: int = 3) -> list:
    """Sample `samples` frames uniformly from the video; return their perceptual hashes."""
    try:
        import imagehash
        from PIL import Image
    except ImportError:
        return []
    duration = _probe_duration(video_path)
    if duration <= 0:
        return []
    timestamps = [duration * (i + 1) / (samples + 1) for i in range(samples)]
    hashes = []
    with tempfile.TemporaryDirectory() as td:
        for i, ts in enumerate(timestamps):
            jpg = Path(td) / f"f{i}.jpg"
            subprocess.run(
                ["ffmpeg", "-y", "-ss", str(ts), "-i", video_path,
                 "-vframes", "1", "-q:v", "5", "-vf", "scale=256:-2",
                 str(jpg)],
                capture_output=True, timeout=15,
            )
            if jpg.exists() and jpg.stat().st_size > 500:
                try:
                    hashes.append(imagehash.phash(Image.open(jpg)))
                except Exception:
                    continue
    return hashes


def phash_image(image_path: str):
    try:
        import imagehash
        from PIL import Image
        return imagehash.phash(Image.open(image_path))
    except Exception:
        return None


def _probe_duration(path: str) -> float:
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", path],
            capture_output=True, text=True, timeout=10,
        )
        return float((r.stdout or "0").strip() or 0)
    except Exception:
        return 0.0
