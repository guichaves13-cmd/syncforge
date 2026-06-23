"""End-to-end pipeline test with everything mocked — proves the wiring."""
from pathlib import Path

from app.services.sync.pipeline import (
    Intent, NarrationClause, SyncEngine, SyncEngineConfig,
)
from app.services.sync.ranker import Candidate, MultiSignalRanker
from app.services.sync.retriever import MultiSourceRetriever, RetrieverConfig


def _stub_source(prefix):
    def fn(q, n):
        return [
            Candidate(source=prefix, source_id=f"{prefix}_{q}_{i}",
                      url=f"http://{prefix}/{i}",
                      title=f"{prefix} relevant {q}",
                      description=f"This clip shows {q} in detail",
                      duration=8.0)
            for i in range(n)
        ]
    return fn


def _stub_intent(clause, theme):
    return Intent(
        main_entity=clause.text[:30],
        action="showing",
        location="indoor",
        era="modern",
        mood="neutral",
        objects=[clause.text[:10]],
        key_visuals=[clause.text[:20]],
        queries=[clause.text[:40], f"{theme} {clause.text[:20]}"],
    )


def _stub_download_makes_file(tmp_root):
    """Pretend we downloaded by writing a 200KB dummy file."""
    def fn(c, output_dir):
        p = Path(output_dir) / f"{c.source_id}.mp4"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"\0" * 200_000)
        return str(p)
    return fn


def test_pipeline_solves_all_clauses_without_vision(tmp_path):
    clauses = [
        NarrationClause(start=0, end=5, text="Butterfly Viscaria blade overview."),
        NarrationClause(start=5, end=10, text="Stiga Carbonado lab results."),
        NarrationClause(start=10, end=15, text="Donek German manufacturing."),
    ]
    retriever = MultiSourceRetriever(
        config=RetrieverConfig(max_per_source=3, max_total_pool=20, workers=2),
        sources={"pexels": _stub_source("pexels"),
                 "youtube": _stub_source("youtube")},
    )
    ranker = MultiSignalRanker(embedder=None)
    engine = SyncEngine(
        config=SyncEngineConfig(theme="ping pong",
                                enable_embeddings=False,
                                enable_vision=False),
        retriever=retriever,
        ranker=ranker,
        intent_fn=_stub_intent,
        download_fn=_stub_download_makes_file(tmp_path),
        download_dir=tmp_path / "clips",
    )
    beats = engine.plan_for_clauses(clauses)
    assert len(beats) == 3
    assert all(b.is_solved for b in beats), f"unsolved: {[b for b in beats if not b.is_solved]}"
    # Anti-repeat: each clip used only once
    chosen_ids = [b.chosen.source_id for b in beats]
    assert len(set(chosen_ids)) == 3


def test_pipeline_progress_callback_fires(tmp_path):
    events = []
    clauses = [NarrationClause(start=0, end=5, text="test clause one"),
               NarrationClause(start=5, end=10, text="test clause two")]
    retriever = MultiSourceRetriever(
        config=RetrieverConfig(workers=1),
        sources={"x": _stub_source("x")},
    )
    engine = SyncEngine(
        config=SyncEngineConfig(theme="x",
                                enable_embeddings=False,
                                enable_vision=False),
        retriever=retriever,
        ranker=MultiSignalRanker(),
        intent_fn=_stub_intent,
        download_fn=_stub_download_makes_file(tmp_path),
        progress=events.append,
        download_dir=tmp_path / "clips",
    )
    engine.plan_for_clauses(clauses)
    types = {e["event"] for e in events}
    assert "clause_start" in types
    assert "clause_done" in types
    assert sum(1 for e in events if e["event"] == "clause_start") == 2


def test_pipeline_handles_empty_pool(tmp_path):
    """If no source returns anything, beat must mark error but not crash."""
    def empty(q, n):
        return []
    retriever = MultiSourceRetriever(
        config=RetrieverConfig(),
        sources={"none": empty},
    )
    engine = SyncEngine(
        config=SyncEngineConfig(theme="x",
                                enable_embeddings=False,
                                enable_vision=False,
                                enable_generative_fallback=False),
        retriever=retriever,
        ranker=MultiSignalRanker(),
        intent_fn=_stub_intent,
        download_fn=lambda c, d: None,
        download_dir=tmp_path / "clips",
    )
    beats = engine.plan_for_clauses(
        [NarrationClause(start=0, end=5, text="anything")]
    )
    assert len(beats) == 1
    assert not beats[0].is_solved
    assert beats[0].error  # error reason populated
