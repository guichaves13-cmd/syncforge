"""End-to-end SyncEngine — junta intent extraction + retrieval + rank + verify
+ generative fallback em um pipeline coerente.

Uso:
    engine = SyncEngine(...)
    plan = engine.plan_for_clauses(clauses, theme="ping pong")
    # plan = [SyncedBeat(start, end, narration, chosen_clip, ...)]
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

from .embedder import MultimodalEmbedder
from .ranker import Candidate, MultiSignalRanker
from .retriever import MultiSourceRetriever
from .verifier import VisionVerifier


@dataclass
class NarrationClause:
    """One sentence/clause of narration with its precise word-level timing."""
    start: float
    end: float
    text: str

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class Intent:
    main_entity: str = ""
    action: str = ""
    location: str = ""
    era: str = "modern"
    mood: str = "neutral"
    objects: list[str] = field(default_factory=list)
    key_visuals: list[str] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)


@dataclass
class SyncedBeat:
    clause: NarrationClause
    intent: Intent
    chosen: Optional[Candidate] = None
    rejected: list[Candidate] = field(default_factory=list)
    generated: bool = False     # True if Veo/Runway fallback fired
    error: str = ""

    @property
    def is_solved(self) -> bool:
        return self.chosen is not None and bool(self.chosen.local_path)


IntentExtractFn = Callable[[NarrationClause, str], Intent]
"""(clause, theme) -> Intent. Implemented by services/llm/*."""

DownloadFn = Callable[[Candidate, Path], Optional[str]]
"""(candidate, output_dir) -> local_path or None. Implemented by services/stock/*."""

GenerativeFn = Callable[[Intent, NarrationClause, Path], Optional[Candidate]]
"""(intent, clause, output_dir) -> Candidate or None. Veo/Runway fallback."""


@dataclass
class SyncEngineConfig:
    theme: str = ""
    top_k_verify: int = 10               # Quantos vão pro Vision após rank
    min_accept_score: int = 70           # Vision min relevance
    enable_embeddings: bool = True
    enable_vision: bool = True
    enable_generative_fallback: bool = False
    workers_per_clause: int = 1          # 1 = sequential (preserva ordem logs)


class SyncEngine:
    def __init__(
        self,
        config: SyncEngineConfig,
        retriever: MultiSourceRetriever,
        ranker: MultiSignalRanker,
        verifier: Optional[VisionVerifier] = None,
        intent_fn: Optional[IntentExtractFn] = None,
        download_fn: Optional[DownloadFn] = None,
        generative_fn: Optional[GenerativeFn] = None,
        progress: Optional[Callable[[dict], None]] = None,
        download_dir: Path = Path("storage/clips"),
    ):
        self.cfg = config
        self.retriever = retriever
        self.ranker = ranker
        self.verifier = verifier
        self.intent_fn = intent_fn
        self.download_fn = download_fn
        self.generative_fn = generative_fn
        self.progress = progress or (lambda _e: None)
        self.download_dir = download_dir
        self.download_dir.mkdir(parents=True, exist_ok=True)

    # ──────────────────────────────────────────────────────────────────
    # Main entry
    # ──────────────────────────────────────────────────────────────────

    def plan_for_clauses(self, clauses: Sequence[NarrationClause]) -> list[SyncedBeat]:
        beats: list[SyncedBeat] = []
        total = len(clauses)
        for i, clause in enumerate(clauses, 1):
            t0 = time.time()
            self.progress({"event": "clause_start", "i": i, "total": total,
                          "text": clause.text[:80]})
            beat = self._process_one(clause)
            beats.append(beat)
            self.progress({"event": "clause_done", "i": i, "total": total,
                          "solved": beat.is_solved,
                          "source": (beat.chosen.source if beat.chosen else None),
                          "score": (beat.chosen.vision if beat.chosen else 0),
                          "elapsed_s": round(time.time() - t0, 2)})
        return beats

    # ──────────────────────────────────────────────────────────────────
    # Per-clause
    # ──────────────────────────────────────────────────────────────────

    def _process_one(self, clause: NarrationClause) -> SyncedBeat:
        # 1. Intent extraction
        intent = self.intent_fn(clause, self.cfg.theme) if self.intent_fn else Intent(
            main_entity=clause.text[:50],
            queries=[clause.text[:80]],
        )
        beat = SyncedBeat(clause=clause, intent=intent)

        # 2. Retrieval — pool of candidates
        queries = intent.queries or [clause.text[:80]]
        pool = self.retriever.retrieve(queries)
        if not pool:
            beat.error = "empty pool"
            return self._try_generative(beat) if self.cfg.enable_generative_fallback else beat

        # 3. Download top-K candidates (need local paths for embedding+vision)
        #    Heuristic: rank by BM25 first to pick which to download.
        bm_ranked = self.ranker.rank(clause.text, pool, use_embeddings=False)
        download_set = bm_ranked[: self.cfg.top_k_verify]
        if self.download_fn:
            for c in download_set:
                try:
                    p = self.download_fn(c, self.download_dir)
                    if p:
                        c.local_path = p
                except Exception:
                    continue

        # 4. Embedding rank (only those that actually downloaded)
        downloaded = [c for c in download_set if c.local_path]
        if not downloaded:
            beat.error = "no candidate downloaded"
            return self._try_generative(beat) if self.cfg.enable_generative_fallback else beat

        ranked = self.ranker.rank(
            clause.text, downloaded,
            use_embeddings=self.cfg.enable_embeddings,
        )

        # 5. Vision verify top-K
        if self.cfg.enable_vision and self.verifier:
            for c in ranked[: self.cfg.top_k_verify]:
                data = self.verifier.verify(
                    c, clause.text,
                    topic=self.cfg.theme or intent.main_entity,
                    era=intent.era, mood=intent.mood,
                )
                if data.get("approved") and c.vision >= self.cfg.min_accept_score:
                    beat.chosen = c
                    self.retriever.reserve_used(c)
                    return beat
                beat.rejected.append(c)

        # 6. If vision disabled, just pick the top ranked
        elif ranked:
            beat.chosen = ranked[0]
            self.retriever.reserve_used(ranked[0])
            return beat

        # 7. Generative fallback
        if self.cfg.enable_generative_fallback:
            return self._try_generative(beat)
        beat.error = "all candidates rejected"
        return beat

    def _try_generative(self, beat: SyncedBeat) -> SyncedBeat:
        if not self.generative_fn:
            return beat
        try:
            c = self.generative_fn(beat.intent, beat.clause, self.download_dir)
            if c and c.local_path:
                beat.chosen = c
                beat.generated = True
        except Exception as e:
            beat.error = f"generative fail: {e!s}"
        return beat
