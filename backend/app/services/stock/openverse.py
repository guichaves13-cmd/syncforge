"""Openverse — meta-search of 800M+ CC-licensed images.

Great for **brand-aware retrieval**: a query like 'Butterfly Viscaria paddle'
often returns the actual product photo from Wikimedia, while generic Pexels
just returns abstract ping-pong scenes.

Public API, no key required for read access. Rate limit ~5 req/sec.
Returns ONLY images (CC0 + CC-BY); we filter out anything restrictive.

Docs: https://api.openverse.org/v1/
"""
from __future__ import annotations
import uuid
from pathlib import Path

import requests

from ..sync.ranker import Candidate


_ALLOWED_LICENSES = {"cc0", "pdm", "by"}    # public domain + attribution-only


class OpenverseSource:
    name = "openverse"
    BASE = "https://api.openverse.org/v1/images/"

    def __init__(self, user_agent: str = "SyncForge/0.5 (https://github.com/guichaves13-cmd/syncforge)"):
        self.user_agent = user_agent

    def search(self, query: str, max_results: int = 8) -> list[Candidate]:
        out: list[Candidate] = []
        try:
            r = requests.get(self.BASE, params={
                "q": query[:120],
                "license": ",".join(_ALLOWED_LICENSES),
                "page_size": max(1, min(max_results, 20)),
                "filter_dead": "true",
            }, headers={"User-Agent": self.user_agent}, timeout=10)
            r.raise_for_status()
            for item in r.json().get("results", []):
                url = item.get("url")
                if not url:
                    continue
                title = item.get("title", "") or ""
                license = item.get("license", "")
                source = item.get("source", "")
                tags = [t.get("name", "") for t in (item.get("tags") or [])
                         if isinstance(t, dict)]
                out.append(Candidate(
                    source=f"openverse_photo",
                    source_id=str(item.get("id", "")),
                    url=url,
                    title=title[:120],
                    description=f"[{license} via {source}] {item.get('foreign_landing_url','')}"[:300],
                    tags=tags[:6] or [query],
                ))
        except Exception:
            pass
        return out

    def download(self, c: Candidate, output_dir: Path) -> str | None:
        ext = ".jpg"
        for suffix in (".png", ".webp", ".jpeg"):
            if c.url.lower().endswith(suffix):
                ext = suffix
                break
        dest = output_dir / f"openverse_{c.source_id}_{uuid.uuid4().hex[:6]}{ext}"
        try:
            with requests.get(
                c.url, stream=True, timeout=30,
                headers={"User-Agent": self.user_agent},
            ) as r:
                r.raise_for_status()
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 15):
                        f.write(chunk)
            return str(dest) if dest.stat().st_size > 2_000 else None
        except Exception:
            return None
