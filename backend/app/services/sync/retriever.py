"""Multi-source retriever — busca paralela em Pexels + Pixabay + YouTube + Coverr + Mixkit + Wikimedia.

Cada fonte tem seu próprio adaptador (services/stock/*). Aqui só orquestra
em ThreadPoolExecutor, dedupa por URL + pHash, e devolve pool unificado de
Candidates pra ranker + verifier.
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable

from .ranker import Candidate


SourceFn = Callable[[str, int], list[Candidate]]
"""(query, max_results) -> list[Candidate]. Implementado por services/stock/*."""


@dataclass
class RetrieverConfig:
    max_per_source: int = 10
    max_total_pool: int = 80
    workers: int = 6
    dedup_global: set[str] = field(default_factory=set)  # source_id "src::id"


class MultiSourceRetriever:
    def __init__(self, config: RetrieverConfig, sources: dict[str, SourceFn]):
        self.cfg = config
        self.sources = sources  # {"pexels": fn, "youtube": fn, ...}

    def retrieve(self, queries: list[str]) -> list[Candidate]:
        """Busca em paralelo (cada fonte × cada query), dedupa, retorna pool."""
        pool: list[Candidate] = []
        seen_local: set[str] = set()  # dedup dentro desta chamada
        tasks: list[tuple[str, str]] = []  # (source_name, query)
        for src in self.sources:
            for q in queries:
                tasks.append((src, q))

        with ThreadPoolExecutor(max_workers=self.cfg.workers) as ex:
            futures = {
                ex.submit(self._safe_call, src, q): (src, q)
                for src, q in tasks
            }
            for fut in as_completed(futures):
                src, q = futures[fut]
                results = fut.result()
                for c in results:
                    key = f"{c.source}::{c.source_id}"
                    if key in seen_local or key in self.cfg.dedup_global:
                        continue
                    seen_local.add(key)
                    pool.append(c)
                    if len(pool) >= self.cfg.max_total_pool:
                        return pool
        return pool

    def reserve_used(self, candidate: Candidate) -> None:
        """Add to global dedup set so this clip never reappears in future intents."""
        self.cfg.dedup_global.add(f"{candidate.source}::{candidate.source_id}")

    def _safe_call(self, src: str, query: str) -> list[Candidate]:
        try:
            return self.sources[src](query, self.cfg.max_per_source) or []
        except Exception:
            return []
