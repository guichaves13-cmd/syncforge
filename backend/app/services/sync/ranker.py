"""Multi-signal ranking via Reciprocal Rank Fusion.

Combina 3 sinais por candidato:
  1) BM25  — match léxico título+descrição vs narração
  2) Embedding cosine — narração ↔ média de frames
  3) Vision — score LLM-as-judge (preenchido depois pelo Verifier)

RRF (Cormack et al. 2009):
  score(c) = Σ_signal  1/(k + rank_signal(c))   com k=60
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .embedder import MultimodalEmbedder


# ─────────────────────────────────────────────────────────────────────────────
# BM25 (sem dependência externa, suficiente p/ poucas dúzias de candidatos)
# ─────────────────────────────────────────────────────────────────────────────

import math
import re

_TOKEN = re.compile(r"\w+")


def _tokenize(s: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(s or "")]


def bm25_scores(query: str, docs: Sequence[str], k1: float = 1.5, b: float = 0.75) -> list[float]:
    q_tokens = _tokenize(query)
    doc_tokens = [_tokenize(d) for d in docs]
    if not doc_tokens:
        return []
    avgdl = sum(len(d) for d in doc_tokens) / len(doc_tokens) or 1.0
    df: dict[str, int] = {}
    for d in doc_tokens:
        for t in set(d):
            df[t] = df.get(t, 0) + 1
    N = len(doc_tokens)
    scores: list[float] = []
    for d in doc_tokens:
        s = 0.0
        for t in q_tokens:
            if t not in df:
                continue
            idf = math.log(1 + (N - df[t] + 0.5) / (df[t] + 0.5))
            tf = d.count(t)
            denom = tf + k1 * (1 - b + b * len(d) / avgdl)
            s += idf * (tf * (k1 + 1)) / (denom or 1.0)
        scores.append(s)
    return scores


# ─────────────────────────────────────────────────────────────────────────────
# Candidate + ranker
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Candidate:
    source: str                # "pexels" | "pixabay" | "youtube" | ...
    source_id: str
    url: str
    title: str = ""
    description: str = ""
    tags: list[str] = field(default_factory=list)
    local_path: str = ""       # filled after download
    duration: float = 0.0

    # Filled by ranker / verifier
    bm25: float = 0.0
    embed_cos: float = 0.0
    vision: float = 0.0
    vision_reason: str = ""
    final_score: float = 0.0
    rank: int = 0

    def doc_text(self) -> str:
        return " ".join([self.title or "", self.description or "", " ".join(self.tags)])


def rrf(ranks: Sequence[int], k: int = 60) -> float:
    return sum(1.0 / (k + r) for r in ranks if r > 0)


class MultiSignalRanker:
    def __init__(self, embedder: MultimodalEmbedder | None = None):
        self.embedder = embedder

    def rank(self, query_text: str, candidates: list[Candidate],
             use_embeddings: bool = True, k: int = 60) -> list[Candidate]:
        if not candidates:
            return []

        # ── 1. BM25 lexical match ──────────────────────────────────────
        docs = [c.doc_text() for c in candidates]
        bm = bm25_scores(query_text, docs)
        for c, s in zip(candidates, bm):
            c.bm25 = s

        # ── 2. Embedding cosine (requires local_path) ──────────────────
        if use_embeddings and self.embedder is not None:
            try:
                q_vec = self.embedder.embed_text(query_text)
                for c in candidates:
                    if not c.local_path:
                        c.embed_cos = 0.0
                        continue
                    try:
                        v = self.embedder.embed_video(c.local_path)
                        c.embed_cos = self.embedder.similarity(q_vec, v)
                    except Exception:
                        c.embed_cos = 0.0
            except Exception:
                pass

        # ── 3. RRF fusion ──────────────────────────────────────────────
        rank_bm   = _rank_by(candidates, "bm25")
        rank_emb  = _rank_by(candidates, "embed_cos")
        # Vision rank only contributes after Verifier runs
        rank_vis  = _rank_by(candidates, "vision") if any(c.vision for c in candidates) else None
        for c in candidates:
            ranks = [rank_bm[c.source_id], rank_emb[c.source_id]]
            if rank_vis is not None:
                ranks.append(rank_vis[c.source_id])
            c.final_score = rrf(ranks, k=k)

        # ── 4. Sort by final_score desc ────────────────────────────────
        ranked = sorted(candidates, key=lambda c: c.final_score, reverse=True)
        for i, c in enumerate(ranked, 1):
            c.rank = i
        return ranked


def _rank_by(candidates: list[Candidate], attr: str) -> dict[str, int]:
    """Returns {source_id: rank (1=best)}; ties get same rank."""
    sorted_c = sorted(candidates, key=lambda c: getattr(c, attr, 0.0), reverse=True)
    out: dict[str, int] = {}
    prev = None
    cur_rank = 0
    for i, c in enumerate(sorted_c, 1):
        v = getattr(c, attr, 0.0)
        if v != prev:
            cur_rank = i
            prev = v
        out[c.source_id] = cur_rank
    return out
