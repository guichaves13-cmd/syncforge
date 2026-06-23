"""Mixkit free videos — public HTML scrape (no official API).

Best-effort, low-priority source.
"""
from __future__ import annotations
import re
import uuid
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from ..sync.ranker import Candidate


class MixkitSource:
    name = "mixkit"
    BASE = "https://mixkit.co"

    def search(self, query: str, max_results: int = 8) -> list[Candidate]:
        out: list[Candidate] = []
        try:
            r = requests.get(
                f"{self.BASE}/free-stock-video/search/",
                params={"q": query},
                timeout=10,
            )
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")
            for card in soup.select("a[href*='/free-stock-video/']")[: max_results]:
                href = card.get("href", "")
                if not href.startswith("/"):
                    continue
                # detail page
                try:
                    dr = requests.get(f"{self.BASE}{href}", timeout=10)
                    m = re.search(r'(https://[^"\']+\.mp4)', dr.text)
                    if not m:
                        continue
                    title_m = re.search(r"<title>([^<]+)</title>", dr.text)
                    out.append(Candidate(
                        source=self.name,
                        source_id=href.rstrip("/").rsplit("/", 1)[-1],
                        url=m.group(1),
                        title=(title_m.group(1) if title_m else "").replace(" | Mixkit", ""),
                        description=href,
                        tags=[query],
                    ))
                    if len(out) >= max_results:
                        break
                except Exception:
                    continue
        except Exception:
            pass
        return out

    def download(self, c: Candidate, output_dir: Path) -> str | None:
        dest = output_dir / f"mixkit_{c.source_id}_{uuid.uuid4().hex[:6]}.mp4"
        try:
            with requests.get(c.url, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 16):
                        f.write(chunk)
            return str(dest) if dest.stat().st_size > 10_000 else None
        except Exception:
            return None
