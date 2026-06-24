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

    def search_brand_assets(self, brand: str, max_results: int = 6) -> list[Candidate]:
        """Brand-aware: prefers `<brand> logo` and `<brand> product` queries.

        These usually return PNG logos or studio product shots, not generic
        crowd photos. Falls back to plain brand-only query if results are thin.
        """
        out: list[Candidate] = []
        seen: set[str] = set()
        for q in (f"{brand} logo", f"{brand} product", f"{brand} brand"):
            for c in self._search_files(q, max_results, file_types=("image",)):
                if c.source_id in seen:
                    continue
                seen.add(c.source_id)
                # Re-tag as brand asset so downstream rankers can boost it
                c.source = "wikimedia_brand"
                c.tags = list(set([*c.tags, brand, "logo", "product"]))
                out.append(c)
                if len(out) >= max_results:
                    return out
        return out

    def _search_files(self, query: str, max_results: int,
                       file_types: tuple[str, ...] = ("video", "image")) -> list[Candidate]:
        """Internal: generic file search (image OR video) with mime filter."""
        out: list[Candidate] = []
        try:
            r = requests.get(self.API, params={
                "action": "query", "format": "json",
                "generator": "search",
                "gsrsearch": f"{query} " + " OR ".join(f"filetype:{t}" for t in file_types),
                "gsrlimit": max_results,
                "prop": "imageinfo",
                "iiprop": "url|size|mime",
            }, timeout=10)
            r.raise_for_status()
            for pid, p in (r.json().get("query", {}) or {}).get("pages", {}).items():
                info = (p.get("imageinfo") or [{}])[0]
                url = info.get("url")
                mime = str(info.get("mime", ""))
                if not url:
                    continue
                if not any(mime.startswith(t) for t in file_types):
                    continue
                out.append(Candidate(
                    source=self.name, source_id=str(pid),
                    url=url, title=p.get("title", ""),
                    description=f"[{mime}] {url}"[:300],
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
