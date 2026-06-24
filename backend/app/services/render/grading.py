"""Color grading — apply a consistent "look" across stock clips from
heterogeneous sources (warm Pexels handheld vs. clinical Wikimedia studio
shots vs. desaturated YouTube downloads).

Two ways to use it:
  1. Per-clip ffmpeg filter chain that runs during compose.
  2. Standalone CLI helper: `python -m app.services.render.grading <in> <out> <preset>`.

Presets are LIGHT-touch (no destructive grading) — designed to harmonise,
not stylize. They tweak saturation, contrast, exposure, and a subtle
shadow tint.

Pipeline integration: the composer calls `build_grading_filter(preset)` and
prepends the returned string to its existing scale/pad filtergraph.
"""
from __future__ import annotations
import subprocess
from dataclasses import dataclass
from typing import Literal


Preset = Literal["neutral", "cinematic", "documentary", "vibrant", "warm", "cool", "off"]


@dataclass(frozen=True)
class GradeParams:
    """Parameters fed to ffmpeg's `eq` + `curves` filters."""
    saturation: float = 1.0
    contrast: float = 1.0
    brightness: float = 0.0     # additive, range -1..1
    gamma: float = 1.0
    # Optional shadow/highlight tint via curves preset
    curves_preset: str = ""     # "" | "vintage" | "negative" | "lighter" | "darker"


_PRESETS: dict[Preset, GradeParams] = {
    "off":         GradeParams(),
    "neutral":     GradeParams(saturation=1.02, contrast=1.04, brightness=0.0, gamma=1.0),
    "cinematic":   GradeParams(saturation=0.92, contrast=1.12, brightness=-0.02, gamma=0.95),
    "documentary": GradeParams(saturation=0.95, contrast=1.05, brightness=0.0, gamma=1.0),
    "vibrant":     GradeParams(saturation=1.20, contrast=1.08, brightness=0.02, gamma=1.0),
    "warm":        GradeParams(saturation=1.08, contrast=1.04, brightness=0.01, gamma=1.0,
                                curves_preset="lighter"),
    "cool":        GradeParams(saturation=1.00, contrast=1.06, brightness=-0.01, gamma=1.0),
}


def build_grading_filter(preset: Preset = "neutral") -> str:
    """Return an ffmpeg `-vf` filter fragment that applies the preset.

    Returns "" when preset is "off" so callers can no-op cleanly.
    """
    p = _PRESETS.get(preset, _PRESETS["neutral"])
    if p == _PRESETS["off"]:
        return ""
    eq = (f"eq=saturation={p.saturation:.3f}:contrast={p.contrast:.3f}"
          f":brightness={p.brightness:.3f}:gamma={p.gamma:.3f}")
    if p.curves_preset:
        return f"{eq},curves=preset={p.curves_preset}"
    return eq


def list_presets() -> list[str]:
    """Names exposed to UI/CLI."""
    return list(_PRESETS.keys())


def grade_clip(src: str, dst: str, preset: Preset = "neutral",
                ffmpeg_bin: str = "ffmpeg", timeout: int = 300) -> str:
    """Standalone: re-encode `src` with the grading preset, write to `dst`.
    Returns `dst` on success; raises if ffmpeg fails or output is empty."""
    flt = build_grading_filter(preset)
    cmd = [ffmpeg_bin, "-y", "-i", src]
    if flt:
        cmd += ["-vf", flt]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
            "-c:a", "copy", "-movflags", "+faststart", dst]
    subprocess.run(cmd, capture_output=True, timeout=timeout)
    from pathlib import Path
    if not Path(dst).exists() or Path(dst).stat().st_size < 5_000:
        raise RuntimeError(f"grade_clip produced no output for preset={preset}")
    return dst


# ── Auto-preset selection from theme/mood ─────────────────────────────

_THEME_TO_PRESET: dict[str, Preset] = {
    "history": "cinematic",
    "documentary": "documentary",
    "tech": "cool",
    "science": "cool",
    "cooking": "warm",
    "travel": "vibrant",
    "fitness": "vibrant",
    "finance": "neutral",
    "default": "neutral",
}


def auto_preset(theme: str = "", mood: str = "") -> Preset:
    """Pick a grading preset from the theme/mood heuristically.

    Free-text mood overrides theme when it matches a preset name.
    """
    mood = (mood or "").lower().strip()
    if mood in _PRESETS:
        return mood  # type: ignore[return-value]
    theme = (theme or "").lower().strip()
    for keyword, preset in _THEME_TO_PRESET.items():
        if keyword and keyword in theme:
            return preset
    return _THEME_TO_PRESET["default"]
