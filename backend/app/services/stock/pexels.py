"""Pexels videos + photos adapter."""
from __future__ import annotations
import os
import uuid
from pathlib import Path

import requests

from ..sync.ranker import Candidate


class PexelsSource:
    name = "pexels"
    BASE = "https://api.pexels.com"

    def __init__(self, api_key: str = ""):
        self.api_key = api_key or os.getenv("PEXELS_API_KEY", "")

    def search(self, query: str, max_results: int = 10) -> list[Candidate]:
        if not self.api_key:
            return []
        out: list[Candidate] = []
        try:
            r = requests.get(
                f"{self.BASE}/videos/search",
                headers={"Authorization": self.api_key},
                params={"query": query, "per_page": max_results, "size": "medium"},
                timeout=10,
            )
            r.raise_for_status()
            for v in r.json().get("videos", []):
                # pick best HD-friendly file
                files = sorted(
                    [f for f in v.get("video_files", []) if f.get("width", 0) >= 1280],
                    key=lambda f: f.get("width", 0),
                )
                if not files:
                    continue
                best = files[0]
                out.append(Candidate(
                    source=self.name,
                    source_id=str(v["id"]),
                    url=best["link"],
                    title=v.get("user", {}).get("name", ""),
                    description=v.get("url", ""),
                    tags=[query],
                    duration=float(v.get("duration", 0)),
                ))
        except Exception:
            pass
        return out

    def search_photos(self, query: str, max_results: int = 10) -> list[Candidate]:
        if not self.api_key:
            return []
        out: list[Candidate] = []
        try:
            r = requests.get(
                f"{self.BASE}/v1/search",
                headers={"Authorization": self.api_key},
                params={"query": query, "per_page": max_results, "orientation": "landscape"},
                timeout=10,
            )
            r.raise_for_status()
            for p in r.json().get("photos", []):
                out.append(Candidate(
                    source="pexels_photo",
                    source_id=str(p["id"]),
                    url=p["src"]["large2x"],
                    title=p.get("alt", "") or p.get("photographer", ""),
                    description=p.get("url", ""),
                    tags=[query],
                ))
        except Exception:
            pass
        return out

    def download(self, c: Candidate, output_dir: Path) -> str | None:
        ext = ".mp4" if c.source == "pexels" else ".jpg"
        dest = output_dir / f"{c.source}_{c.source_id}_{uuid.uuid4().hex[:6]}{ext}"
        try:
            with requests.get(c.url, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 16):
                        f.write(chunk)
            return str(dest) if dest.stat().st_size > 10_000 else None
        except Exception:
            return None
