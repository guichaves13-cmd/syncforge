"""YouTube via yt-dlp — search + download with quality filter + duration band.

Defensive defaults to dodge music videos, news broadcasts, AI generated clips.
"""
from __future__ import annotations
import json
import re
import subprocess
import sys
import uuid
from pathlib import Path

from ..sync.ranker import Candidate


_BAD_TITLE = re.compile(
    r"\b(lyrics|official video|music video|nightcore|karaoke|asmr|"
    r"abc news|cbs news|cnn|inside edition|fox news|news now|"
    r"ai generated|ai art|stable diffusion|midjourney|sora)\b",
    re.I,
)


class YouTubeSource:
    name = "youtube"

    def __init__(self, min_duration: int = 60, max_duration: int = 1800,
                 min_height: int = 720):
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.min_height = min_height

    def search(self, query: str, max_results: int = 10) -> list[Candidate]:
        cmd = [
            sys.executable, "-m", "yt_dlp",
            f"ytsearch{max_results * 2}:{query}",
            "--dump-json", "--no-warnings", "--skip-download",
            "--match-filter",
            f"duration > {self.min_duration} & duration < {self.max_duration}",
        ]
        out: list[Candidate] = []
        try:
            r = subprocess.run(cmd, capture_output=True, text=True,
                               timeout=60, encoding="utf-8", errors="ignore")
            for line in (r.stdout or "").splitlines():
                line = line.strip()
                if not line.startswith("{"):
                    continue
                try:
                    j = json.loads(line)
                except Exception:
                    continue
                title = j.get("title") or ""
                if _BAD_TITLE.search(title):
                    continue
                yid = j.get("id") or j.get("display_id")
                if not yid:
                    continue
                out.append(Candidate(
                    source=self.name,
                    source_id=yid,
                    url=j.get("webpage_url") or f"https://youtu.be/{yid}",
                    title=title,
                    description=(j.get("description") or "")[:500],
                    tags=j.get("tags") or [],
                    duration=float(j.get("duration") or 0),
                ))
                if len(out) >= max_results:
                    break
        except Exception:
            pass
        return out

    def download(self, c: Candidate, output_dir: Path,
                 clip_seconds: int = 30) -> str | None:
        """Download up to clip_seconds from start (or middle) of the video."""
        dest = output_dir / f"yt_{c.source_id}_{uuid.uuid4().hex[:6]}.mp4"
        # Pick a mid-video window so we skip intros
        start = max(5, int((c.duration or 60) * 0.2))
        cmd = [
            sys.executable, "-m", "yt_dlp",
            "-f", f"bv*[height>={self.min_height}][ext=mp4]+ba/b[ext=mp4]/b",
            "--merge-output-format", "mp4",
            "--download-sections", f"*{start}-{start + clip_seconds}",
            "--force-keyframes-at-cuts",
            "-o", str(dest),
            c.url,
            "--no-warnings", "--no-progress", "--no-playlist",
        ]
        try:
            subprocess.run(cmd, capture_output=True, timeout=180)
            return str(dest) if dest.exists() and dest.stat().st_size > 100_000 else None
        except Exception:
            return None
