"""Wikimedia Commons — CC0/CC-BY video + images via official MediaWiki API."""
from __future__ import annotations
import uuid
from pathlib import Path

import requests

from ..sync.ranker import Candidate


class WikimediaSource:
    name = "wikimedia"
    API = "https://commons.wikimedia.org/w/api.php"

    def search(self, query: str, max_results: int = 8) -> list[Candidate]:
        out: list[Candidate] = []
        try:
            r = requests.get(self.API, params={
                "action": "query",
                "format": "json",
                "generator": "search",
                "gsrsearch": f"{query} filetype:video",
                "gsrlimit": max_results,
                "prop": "imageinfo",
                "iiprop": "url|size|mime|extmetadata",
            }, timeout=10)
            r.raise_for_status()
            pages = (r.json().get("query", {}) or {}).get("pages", {})
            for pid, p in pages.items():
                info = (p.get("imageinfo") or [{}])[0]
                url = info.get("url")
                if not url or not str(info.get("mime", "")).startswith("video"):
                    continue
                meta = info.get("extmetadata", {}) or {}
                out.append(Candidate(
                    source=self.name,
                    source_id=str(pid),
                    url=url,
                    title=p.get("title", ""),
                    description=(meta.get("ImageDescription", {}) or {}).get("value", "")[:300],
                    tags=[query],
                ))
        except Exception:
            pass
        return out

    def download(self, c: Candidate, output_dir: Path) -> str | None:
        ext = ".webm" if c.url.endswith(".webm") else ".mp4"
        dest = output_dir / f"wiki_{c.source_id}_{uuid.uuid4().hex[:6]}{ext}"
        try:
            with requests.get(
                c.url, stream=True, timeout=60,
                headers={"User-Agent": "SyncForge/0.1 (educational)"},
            ) as r:
                r.raise_for_status()
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 16):
                        f.write(chunk)
            return str(dest) if dest.stat().st_size > 10_000 else None
        except Exception:
            return None
