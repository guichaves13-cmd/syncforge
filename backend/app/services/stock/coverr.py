"""Coverr (free cinemagraphs / b-roll) adapter."""
from __future__ import annotations
import os
import uuid
from pathlib import Path

import requests

from ..sync.ranker import Candidate


class CoverrSource:
    name = "coverr"
    BASE = "https://api.coverr.co/videos"

    def __init__(self, api_key: str = ""):
        self.api_key = api_key or os.getenv("COVERR_API_KEY", "")

    def search(self, query: str, max_results: int = 10) -> list[Candidate]:
        if not self.api_key:
            return []
        out: list[Candidate] = []
        try:
            r = requests.get(
                self.BASE,
                params={"query": query, "page_size": max_results},
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=10,
            )
            r.raise_for_status()
            for v in r.json().get("hits", []):
                urls = v.get("urls", {})
                best = urls.get("mp4_download") or urls.get("mp4")
                if not best:
                    continue
                out.append(Candidate(
                    source=self.name,
                    source_id=str(v.get("id", "")),
                    url=best,
                    title=v.get("title", ""),
                    description=v.get("description", ""),
                    tags=v.get("tags") or [],
                    duration=float(v.get("max_duration", 0)),
                ))
        except Exception:
            pass
        return out

    def download(self, c: Candidate, output_dir: Path) -> str | None:
        dest = output_dir / f"coverr_{c.source_id}_{uuid.uuid4().hex[:6]}.mp4"
        try:
            with requests.get(c.url, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 16):
                        f.write(chunk)
            return str(dest) if dest.stat().st_size > 10_000 else None
        except Exception:
            return None
