"""Pixabay videos adapter."""
from __future__ import annotations
import os
import uuid
from pathlib import Path

import requests

from ..sync.ranker import Candidate


class PixabaySource:
    name = "pixabay"
    BASE = "https://pixabay.com/api/videos/"

    def __init__(self, api_key: str = ""):
        self.api_key = api_key or os.getenv("PIXABAY_API_KEY", "")

    def search(self, query: str, max_results: int = 10) -> list[Candidate]:
        if not self.api_key:
            return []
        out: list[Candidate] = []
        try:
            r = requests.get(
                self.BASE,
                params={"key": self.api_key, "q": query[:90],
                        "per_page": max(3, min(max_results, 200)),
                        "safesearch": "true", "video_type": "film"},
                timeout=10,
            )
            r.raise_for_status()
            for h in r.json().get("hits", []):
                vids = h.get("videos", {})
                best = vids.get("large") or vids.get("medium") or vids.get("small")
                if not best or not best.get("url"):
                    continue
                out.append(Candidate(
                    source=self.name,
                    source_id=str(h["id"]),
                    url=best["url"],
                    title=h.get("tags", ""),
                    description=h.get("pageURL", ""),
                    tags=str(h.get("tags", "")).split(", "),
                    duration=float(h.get("duration", 0)),
                ))
        except Exception:
            pass
        return out

    def download(self, c: Candidate, output_dir: Path) -> str | None:
        dest = output_dir / f"{c.source}_{c.source_id}_{uuid.uuid4().hex[:6]}.mp4"
        try:
            with requests.get(c.url, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 16):
                        f.write(chunk)
            return str(dest) if dest.stat().st_size > 10_000 else None
        except Exception:
            return None
