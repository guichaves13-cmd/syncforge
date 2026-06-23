"""ffmpeg composer — concatenate SyncedBeats into a final mp4 with audio + optional BGM.

Each beat = trim clip to clause.duration, scale/pad to 1920x1080, apply Ken Burns
for photos, concat all, mux with TTS audio.
"""
from __future__ import annotations
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ..sync.pipeline import SyncedBeat


@dataclass
class ComposeConfig:
    width: int = 1920
    height: int = 1080
    fps: int = 30
    crf: int = 23
    preset: str = "veryfast"
    background_color: str = "0x0a0a0a"
    photo_zoom: float = 1.05    # Ken Burns end-zoom
    ffmpeg: str = "ffmpeg"
    ffprobe: str = "ffprobe"


def compose(beats: list[SyncedBeat], audio_path: str, output_path: str,
            cfg: ComposeConfig | None = None,
            work_dir: Path | None = None) -> str:
    cfg = cfg or ComposeConfig()
    work = work_dir or Path(output_path).parent / "_compose_work"
    work.mkdir(parents=True, exist_ok=True)

    # 1) Build per-beat normalized clip
    parts: list[Path] = []
    for i, b in enumerate(beats):
        if not b.is_solved:
            # leave a black slate sized to clause duration
            seg = _make_black(work / f"seg_{i:04d}.mp4", b.clause.duration, cfg)
        else:
            seg = _normalize_clip(b, work / f"seg_{i:04d}.mp4", cfg)
        parts.append(seg)

    # 2) concat
    concat_file = work / "concat.txt"
    concat_file.write_text("\n".join(f"file '{p.as_posix()}'" for p in parts),
                            encoding="utf-8")
    silent_concat = work / "video_concat.mp4"
    subprocess.run([
        cfg.ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(concat_file),
        "-c", "copy", str(silent_concat),
    ], capture_output=True, timeout=600)

    # 3) mux with audio (trim audio to video length)
    subprocess.run([
        cfg.ffmpeg, "-y", "-i", str(silent_concat), "-i", audio_path,
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-shortest", output_path,
    ], capture_output=True, timeout=600)

    return output_path


def _make_black(out: Path, duration: float, cfg: ComposeConfig) -> Path:
    duration = max(0.5, float(duration))
    subprocess.run([
        cfg.ffmpeg, "-y", "-f", "lavfi",
        "-i", f"color=c={cfg.background_color}:s={cfg.width}x{cfg.height}:r={cfg.fps}",
        "-t", f"{duration:.3f}", "-c:v", "libx264", "-preset", cfg.preset,
        "-crf", str(cfg.crf), "-pix_fmt", "yuv420p", str(out),
    ], capture_output=True, timeout=120)
    return out


def _normalize_clip(beat: SyncedBeat, out: Path, cfg: ComposeConfig) -> Path:
    src = beat.chosen.local_path
    duration = max(0.5, beat.clause.duration)
    is_image = src.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))

    if is_image:
        # Ken Burns: zoompan over the full duration
        zoom = cfg.photo_zoom
        vf = (
            f"scale={cfg.width * 2}:{cfg.height * 2}:force_original_aspect_ratio=cover,"
            f"crop={cfg.width * 2}:{cfg.height * 2},"
            f"zoompan=z='min(zoom+0.0008,{zoom})':d={int(duration * cfg.fps)}"
            f":s={cfg.width}x{cfg.height}:fps={cfg.fps},"
            f"setsar=1"
        )
        cmd = [
            cfg.ffmpeg, "-y", "-loop", "1", "-i", src,
            "-t", f"{duration:.3f}", "-vf", vf,
            "-c:v", "libx264", "-preset", cfg.preset, "-crf", str(cfg.crf),
            "-pix_fmt", "yuv420p", str(out),
        ]
    else:
        vf = (
            f"scale={cfg.width}:{cfg.height}:force_original_aspect_ratio=decrease,"
            f"pad={cfg.width}:{cfg.height}:(ow-iw)/2:(oh-ih)/2:color={cfg.background_color},"
            f"setsar=1,fps={cfg.fps}"
        )
        cmd = [
            cfg.ffmpeg, "-y", "-i", src, "-t", f"{duration:.3f}",
            "-vf", vf, "-an",
            "-c:v", "libx264", "-preset", cfg.preset, "-crf", str(cfg.crf),
            "-pix_fmt", "yuv420p", str(out),
        ]
    subprocess.run(cmd, capture_output=True, timeout=300)
    if not out.exists() or out.stat().st_size < 5000:
        return _make_black(out, duration, cfg)
    return out
