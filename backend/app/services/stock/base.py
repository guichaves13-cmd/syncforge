"""Stock source contract — every adapter exposes search() and download()."""
from __future__ import annotations
from pathlib import Path
from typing import Protocol

from ..sync.ranker import Candidate


class StockSource(Protocol):
    name: str

    def search(self, query: str, max_results: int = 10) -> list[Candidate]: ...
    def download(self, candidate: Candidate, output_dir: Path) -> str | None: ...
