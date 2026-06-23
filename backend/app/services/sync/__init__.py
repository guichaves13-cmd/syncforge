"""SyncEngine — narration-to-visual semantic synchronization.

Modules:
  embedder.py  — Gemini Embedding 2 / SigLIP-2 multimodal vectorization
  retriever.py — multi-source parallel search (Pexels+Pixabay+YT+...)
  ranker.py    — BM25 + embedding cosine + Vision via RRF
  verifier.py  — Gemini 2.5 Pro Vision LLM-as-judge
  pipeline.py  — orchestrates the above into SyncedBeat plans
"""
from .embedder import MultimodalEmbedder, EmbedConfig
from .ranker import Candidate, MultiSignalRanker
from .retriever import MultiSourceRetriever, RetrieverConfig
from .verifier import VisionVerifier, VerifyConfig
from .pipeline import (
    SyncEngine, SyncEngineConfig, SyncedBeat,
    NarrationClause, Intent,
)

__all__ = [
    "MultimodalEmbedder", "EmbedConfig",
    "Candidate", "MultiSignalRanker",
    "MultiSourceRetriever", "RetrieverConfig",
    "VisionVerifier", "VerifyConfig",
    "SyncEngine", "SyncEngineConfig", "SyncedBeat",
    "NarrationClause", "Intent",
]
