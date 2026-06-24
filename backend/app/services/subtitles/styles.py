"""Subtitle style templates — multiple ASS looks selected by content mood.

A documentary about WWII shouldn't use the same yellow-bold-jiggle subtitles
as a TikTok dance compilation. This module exposes 5 carefully tuned styles
and an auto-picker.

Styles:
  • tiktok       — big, bold, white with black outline + drop shadow
  • documentary  — serif, smaller, soft drop shadow, slight transparency
  • karaoke      — primary white, secondary highlight on the active word
  • clean        — minimal sans, no shadow, centered (corporate/explainer)
  • bold_news    — wide sans, all caps look via uppercase transform, sharp
                   yellow accent (urgent news / shocking listicles)

API surface mirrors the existing `subtitles.karaoke` module so the composer
can swap styles without code changes upstream.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


StyleName = Literal["tiktok", "documentary", "karaoke", "clean", "bold_news"]


@dataclass(frozen=True)
class SubtitleStyle:
    """Encapsulates an ASS [V4+ Styles] line + content-shaping hints."""
    name: str
    font: str
    size: int
    primary_colour: str        # ASS &Hxxxxxx (BGR, AA optional)
    outline_colour: str
    back_colour: str
    bold: int                  # 0 / 1 / -1 (-1 = bold)
    italic: int                # 0 / 1
    border_style: int          # 1=outline+shadow, 3=opaque box
    outline: int               # px
    shadow: int                # px
    alignment: int             # ASS code (2=bottom-center, 5=top-center, 8=top-center anchored)
    margin_v: int              # vertical margin from anchor
    uppercase: bool = False    # transform sentence text BEFORE building dialogue lines

    def style_line(self) -> str:
        """Returns the [V4+ Styles] Style: line (sans trailing newline)."""
        return (
            f"Style: {self.name},{self.font},{self.size},"
            f"{self.primary_colour},{self.primary_colour},"
            f"{self.outline_colour},{self.back_colour},"
            f"{self.bold},{self.italic},0,0,100,100,0,0,"
            f"{self.border_style},{self.outline},{self.shadow},"
            f"{self.alignment},80,80,{self.margin_v},1"
        )


# ─────────────────────────────────────────────────────────────────────────
# Built-in styles
# ─────────────────────────────────────────────────────────────────────────

STYLES: dict[StyleName, SubtitleStyle] = {
    "tiktok": SubtitleStyle(
        name="tiktok",
        font="Inter",
        size=84,
        primary_colour="&H00FFFFFF",
        outline_colour="&H00000000",
        back_colour="&H80000000",
        bold=-1, italic=0,
        border_style=1, outline=6, shadow=3,
        alignment=2, margin_v=130, uppercase=False,
    ),
    "documentary": SubtitleStyle(
        name="documentary",
        font="Georgia",
        size=48,
        primary_colour="&H00F0F0F0",
        outline_colour="&H00101010",
        back_colour="&HA0000000",
        bold=0, italic=0,
        border_style=1, outline=2, shadow=2,
        alignment=2, margin_v=70, uppercase=False,
    ),
    "karaoke": SubtitleStyle(
        name="karaoke",
        font="Inter",
        size=72,
        primary_colour="&H00FFFFFF",
        outline_colour="&H00000000",
        back_colour="&H80000000",
        bold=-1, italic=0,
        border_style=1, outline=4, shadow=2,
        alignment=2, margin_v=100, uppercase=False,
    ),
    "clean": SubtitleStyle(
        name="clean",
        font="Inter",
        size=52,
        primary_colour="&H00FFFFFF",
        outline_colour="&H00000000",
        back_colour="&H00000000",
        bold=0, italic=0,
        border_style=1, outline=2, shadow=0,
        alignment=2, margin_v=80, uppercase=False,
    ),
    "bold_news": SubtitleStyle(
        name="bold_news",
        font="Inter",
        size=78,
        primary_colour="&H0000FFFF",   # yellow
        outline_colour="&H00000000",
        back_colour="&H80000000",
        bold=-1, italic=0,
        border_style=1, outline=5, shadow=3,
        alignment=2, margin_v=120, uppercase=True,
    ),
}


_HEADER_TEMPLATE = """[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 2

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
{style_line}

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


# ─────────────────────────────────────────────────────────────────────────
# Builder
# ─────────────────────────────────────────────────────────────────────────

def build_ass(sentences: list[dict], path: Path,
               style: StyleName = "karaoke") -> Path:
    """Same contract as `subtitles.karaoke.build_ass`, but parameterised
    on style name. Writes an .ass file with one Dialogue line per non-empty
    sentence."""
    s = STYLES.get(style, STYLES["karaoke"])
    lines = [_HEADER_TEMPLATE.format(style_line=s.style_line())]

    for sent in sentences:
        text = (sent.get("text") or "").replace("\n", " ").strip()
        if not text:
            continue
        if s.uppercase:
            text = text.upper()
        # Escape commas (the ASS format dialect)
        text = text.replace(",", "\\,")
        start = _fmt(sent.get("start", 0.0))
        end = _fmt(sent.get("end", 0.0))
        lines.append(
            f"Dialogue: 0,{start},{end},{s.name},,0,0,0,,{text}"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _fmt(t: float) -> str:
    """seconds → H:MM:SS.cs"""
    if t < 0:
        t = 0
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t % 60
    return f"{h}:{m:02d}:{s:05.2f}"


# ─────────────────────────────────────────────────────────────────────────
# Auto-picker
# ─────────────────────────────────────────────────────────────────────────

_MOOD_TO_STYLE: dict[str, StyleName] = {
    # Direct hits (mood == style)
    "tiktok": "tiktok",
    "documentary": "documentary",
    "karaoke": "karaoke",
    "clean": "clean",
    "bold_news": "bold_news",
    # Semantic aliases
    "energetic": "tiktok",
    "viral": "tiktok",
    "exciting": "tiktok",
    "shocking": "bold_news",
    "urgent": "bold_news",
    "controversial": "bold_news",
    "explainer": "clean",
    "corporate": "clean",
    "professional": "clean",
    "neutral": "clean",
    "analytical": "documentary",
    "cinematic": "documentary",
    "historical": "documentary",
    "narrative": "documentary",
}


_THEME_TO_STYLE: dict[str, StyleName] = {
    # Energy-heavy genres first (most specific intent)
    "shocking":    "bold_news",
    "news":        "bold_news",
    "listicle":    "bold_news",
    "dance":       "tiktok",
    "viral":       "tiktok",
    "fitness":     "tiktok",
    # Long-form documentary / cinematic narratives
    "history":     "documentary",
    "documentary": "documentary",
    "science":     "documentary",
    # Calm-explainer genres (last because 'tutorial' is too generic)
    "tech":        "clean",
    "finance":     "clean",
    "tutorial":    "clean",
    "default":     "karaoke",
}


def list_styles() -> list[str]:
    """Public style names."""
    return list(STYLES.keys())


def auto_style(theme: str = "", mood: str = "") -> StyleName:
    """Pick the best style from mood (overrides) or theme (fallback)."""
    mood = (mood or "").lower().strip()
    if mood in _MOOD_TO_STYLE:
        return _MOOD_TO_STYLE[mood]
    theme = (theme or "").lower().strip()
    for keyword, style in _THEME_TO_STYLE.items():
        if keyword and keyword in theme:
            return style
    return _THEME_TO_STYLE["default"]
