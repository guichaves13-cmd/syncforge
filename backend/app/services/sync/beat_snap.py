"""Beat snap — snap concept-window boundaries to natural TTS rhythm points.

PROBLEM: legacy pipeline cuts at sentence boundaries from Edge-TTS, which
sometimes lands mid-syllable when the breath was uneven. Viderush-level
requires cuts to happen on the SILENCE between words / clauses.

This module:
  1. Runs ffmpeg's `silencedetect` filter on the TTS audio to find pause
     intervals (default: ≥0.18s of silence at -32dB).
  2. For each concept window boundary, finds the NEAREST silence-midpoint
     within tolerance (default ±0.4s) and snaps to it.
  3. Re-stitches windows to avoid overlap/gaps after snapping.

No external deps — pure ffmpeg + Python.
"""
from __future__ import annotations
import re
import shutil
import subprocess
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .concept import ConceptWindow


@dataclass(frozen=True)
class SilenceInterval:
    """A detected silent gap in the audio."""
    start: float
    end: float

    @property
    def mid(self) -> float:
        return (self.start + self.end) / 2.0

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class BeatSnapConfig:
    silence_db: int = -32              # dB threshold (Edge-TTS sits ~−18dB)
    min_silence_s: float = 0.18        # natural breath ≥180ms
    snap_tolerance_s: float = 0.40     # max distance to nudge a boundary
    min_window_s: float = 1.0          # never crush a window below this
    ffmpeg_bin: str = "ffmpeg"
    timeout_s: int = 60


# ─────────────────────────────────────────────────────────────────────────
# Silence detection
# ─────────────────────────────────────────────────────────────────────────

_SILENCE_START_RX = re.compile(r"silence_start:\s*([\d.]+)")
_SILENCE_END_RX = re.compile(r"silence_end:\s*([\d.]+)")


def detect_silences(audio_path: str,
                     cfg: BeatSnapConfig | None = None) -> list[SilenceInterval]:
    """Run ffmpeg silencedetect; parse the stderr log; return sorted intervals."""
    cfg = cfg or BeatSnapConfig()
    if not Path(audio_path).exists():
        return []
    cmd = [
        cfg.ffmpeg_bin, "-hide_banner", "-nostats",
        "-i", audio_path,
        "-af", f"silencedetect=noise={cfg.silence_db}dB:d={cfg.min_silence_s}",
        "-f", "null", "-",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=cfg.timeout_s)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return []
    # silencedetect logs to STDERR
    raw = (proc.stderr or b"").decode("utf-8", errors="replace")
    starts = [float(m) for m in _SILENCE_START_RX.findall(raw)]
    ends = [float(m) for m in _SILENCE_END_RX.findall(raw)]
    # Pair them by index; if counts mismatch, truncate to shorter (defensive)
    n = min(len(starts), len(ends))
    pairs = [(starts[i], ends[i]) for i in range(n) if ends[i] > starts[i]]
    return [SilenceInterval(start=s, end=e) for s, e in pairs]


# ─────────────────────────────────────────────────────────────────────────
# Snap logic
# ─────────────────────────────────────────────────────────────────────────

def _nearest_silence_mid(t: float,
                          mids: list[float],
                          tolerance: float) -> float | None:
    """Return the nearest silence midpoint within `tolerance` of `t`, or None."""
    if not mids:
        return None
    i = bisect_left(mids, t)
    # Check both neighbours
    candidates = []
    if i < len(mids):
        candidates.append(mids[i])
    if i > 0:
        candidates.append(mids[i - 1])
    best = min(candidates, key=lambda m: abs(m - t))
    return best if abs(best - t) <= tolerance else None


def snap_windows_to_silences(
    windows: list[ConceptWindow],
    silences: list[SilenceInterval],
    cfg: BeatSnapConfig | None = None,
) -> list[ConceptWindow]:
    """Adjust the start/end of each window to the nearest silence midpoint
    (within tolerance), while preserving:
      • Total ordering (start_i <= start_{i+1})
      • Continuity (end_i == start_{i+1})
      • Min window length (no window crushed below cfg.min_window_s)
    """
    cfg = cfg or BeatSnapConfig()
    if not windows:
        return []
    mids = sorted(s.mid for s in silences)

    out: list[ConceptWindow] = []
    prev_end: float | None = None
    for i, w in enumerate(windows):
        # Snap start (skip for the very first window — anchor to 0 / original)
        new_start = w.start
        if i > 0:
            snap = _nearest_silence_mid(w.start, mids, cfg.snap_tolerance_s)
            if snap is not None:
                new_start = snap
            # Maintain continuity with previous window
            if prev_end is not None:
                new_start = prev_end

        # Snap end (skip for the last window — anchor to original end)
        new_end = w.end
        if i < len(windows) - 1:
            snap = _nearest_silence_mid(w.end, mids, cfg.snap_tolerance_s)
            if snap is not None:
                new_end = snap

        # Enforce minimum length
        if new_end - new_start < cfg.min_window_s:
            new_end = new_start + cfg.min_window_s

        snapped = ConceptWindow(
            start=new_start, end=new_end,
            text=w.text,
            dominant_entity=w.dominant_entity,
            related_entities=list(w.related_entities),
            mood=w.mood,
            cluster_id=w.cluster_id,
        )
        out.append(snapped)
        prev_end = new_end

    return out


# ─────────────────────────────────────────────────────────────────────────
# One-shot helper: silence-detect + snap
# ─────────────────────────────────────────────────────────────────────────

def beat_snap(
    windows: list[ConceptWindow],
    audio_path: str,
    cfg: BeatSnapConfig | None = None,
) -> list[ConceptWindow]:
    """End-to-end: analyse the audio and return windows with snapped boundaries.
    Falls back to the input unchanged on any failure (best-effort upgrade)."""
    if not windows:
        return []
    try:
        silences = detect_silences(audio_path, cfg)
        if not silences:
            return windows
        return snap_windows_to_silences(windows, silences, cfg)
    except Exception:
        return windows
