"""Karaoke subtitles — build ASS from per-sentence timings, burn via ffmpeg."""
from __future__ import annotations
import subprocess
from pathlib import Path


_HEADER = """[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 2

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, OutlineColour, BackColour, Bold, Italic, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Inter,72,&H00FFFFFF,&H00000000,&H80000000,1,0,1,4,2,2,80,80,90,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def build_ass(sentences: list[dict], path: Path) -> Path:
    """sentences: [{start, end, text}, ...]"""
    lines = [_HEADER]
    for s in sentences:
        text = (s.get("text") or "").replace("\n", " ").replace(",", "\\,").strip()
        if not text:
            continue
        start = _fmt(s["start"])
        end = _fmt(s["end"])
        lines.append(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}")
    path.write_text("".join("\n".join(lines).splitlines(keepends=True)), encoding="utf-8")
    # safer: write as plain
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def burn(video_path: str, ass_path: Path, output_path: str,
         timeout: int = 2400) -> str:
    ass_norm = str(ass_path.resolve()).replace("\\", "/").replace(":", r"\:")
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vf", f"subtitles='{ass_norm}'",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-c:a", "copy", "-pix_fmt", "yuv420p", output_path,
    ]
    subprocess.run(cmd, capture_output=True, timeout=timeout)
    return output_path


def _fmt(t: float) -> str:
    """seconds → H:MM:SS.cs (ASS format)"""
    if t < 0: t = 0
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t % 60
    return f"{h}:{m:02d}:{s:05.2f}"
