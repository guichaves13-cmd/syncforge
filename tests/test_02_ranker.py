"""Unit tests for BM25, RRF, and the multi-signal ranker."""
import pytest
from app.services.sync.ranker import (
    Candidate, MultiSignalRanker, bm25_scores, rrf, _rank_by,
)


def make_cand(sid, title="", desc="", duration=10.0):
    return Candidate(source="pexels", source_id=sid, url=f"u/{sid}",
                     title=title, description=desc, duration=duration)


# ─── BM25 ────────────────────────────────────────────────────────────

def test_bm25_returns_one_score_per_doc():
    s = bm25_scores("hello world", ["hello world", "foo bar", "hello"])
    assert len(s) == 3
    assert all(isinstance(x, float) for x in s)


def test_bm25_ranks_exact_match_first():
    docs = ["the cat sat on the mat",
            "a dog ran across the field",
            "the cat purrs"]
    s = bm25_scores("cat", docs)
    # docs[0] and docs[2] mention cat; docs[1] does not
    assert s[0] > s[1]
    assert s[2] > s[1]


def test_bm25_empty_inputs():
    assert bm25_scores("anything", []) == []
    s = bm25_scores("", ["a", "b"])
    assert s == [0.0, 0.0]


# ─── RRF ─────────────────────────────────────────────────────────────

def test_rrf_lower_ranks_score_higher():
    # rank 1 should beat rank 10
    high = rrf([1, 1, 1])
    low = rrf([10, 10, 10])
    assert high > low


def test_rrf_zero_ranks_ignored():
    # rank 0 means "no signal" — must not contribute
    assert rrf([0, 0, 5]) == rrf([5])


# ─── _rank_by ────────────────────────────────────────────────────────

def test_rank_by_orders_descending():
    cs = [make_cand("a"), make_cand("b"), make_cand("c")]
    cs[0].bm25 = 1.0; cs[1].bm25 = 5.0; cs[2].bm25 = 3.0
    ranks = _rank_by(cs, "bm25")
    assert ranks["b"] == 1
    assert ranks["c"] == 2
    assert ranks["a"] == 3


def test_rank_by_ties_get_same_rank():
    cs = [make_cand("a"), make_cand("b")]
    cs[0].bm25 = cs[1].bm25 = 4.0
    ranks = _rank_by(cs, "bm25")
    assert ranks["a"] == ranks["b"]


# ─── MultiSignalRanker ───────────────────────────────────────────────

def test_ranker_no_embeddings_uses_bm25_only():
    r = MultiSignalRanker(embedder=None)
    cs = [
        make_cand("hit",  title="ping pong paddle rubber"),
        make_cand("miss", title="kitchen knife sharpener"),
    ]
    out = r.rank("ping pong paddle", cs, use_embeddings=False)
    assert out[0].source_id == "hit"
    assert out[0].rank == 1
    assert out[1].rank == 2


def test_ranker_assigns_final_score():
    r = MultiSignalRanker(embedder=None)
    cs = [make_cand("a", title="dog"), make_cand("b", title="dog cat")]
    out = r.rank("cat", cs)
    assert all(c.final_score > 0 for c in out)


def test_ranker_empty_pool():
    assert MultiSignalRanker().rank("anything", []) == []
