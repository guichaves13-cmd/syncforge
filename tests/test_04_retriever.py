"""Test the parallel retriever orchestration with stub source callables."""
from app.services.sync.ranker import Candidate
from app.services.sync.retriever import MultiSourceRetriever, RetrieverConfig


def fake_source(prefix, items_per_query=3):
    def fn(query, n):
        return [
            Candidate(source=prefix, source_id=f"{prefix}_{query.replace(' ','_')}_{i}",
                      url=f"http://{prefix}/{i}", title=f"{prefix} {query} {i}")
            for i in range(min(items_per_query, n))
        ]
    return fn


def test_retriever_dedups_within_call():
    src = fake_source("pex")
    r = MultiSourceRetriever(
        config=RetrieverConfig(max_per_source=5, max_total_pool=100),
        sources={"pex": src},
    )
    pool = r.retrieve(["dog", "dog"])  # same query twice → dedup
    ids = {c.source_id for c in pool}
    assert len(ids) == len(pool)


def test_retriever_respects_max_total_pool():
    r = MultiSourceRetriever(
        config=RetrieverConfig(max_per_source=10, max_total_pool=5),
        sources={"a": fake_source("a", 10), "b": fake_source("b", 10)},
    )
    pool = r.retrieve(["x", "y", "z"])
    assert len(pool) <= 5


def test_retriever_swallows_source_exceptions():
    def bad(q, n):
        raise RuntimeError("api down")
    r = MultiSourceRetriever(
        config=RetrieverConfig(),
        sources={"good": fake_source("g"), "bad": bad},
    )
    pool = r.retrieve(["test"])
    # Should still get results from 'good' even though 'bad' raised
    assert any(c.source == "g" for c in pool)


def test_retriever_dedup_global_persists():
    src = fake_source("p")
    r = MultiSourceRetriever(
        config=RetrieverConfig(),
        sources={"p": src},
    )
    pool1 = r.retrieve(["a"])
    assert pool1
    # Reserve one
    r.reserve_used(pool1[0])
    pool2 = r.retrieve(["a"])
    used_id = pool1[0].source_id
    assert all(c.source_id != used_id for c in pool2)


def test_retriever_multi_source_aggregates():
    r = MultiSourceRetriever(
        config=RetrieverConfig(),
        sources={
            "pexels": fake_source("pexels", 2),
            "pixabay": fake_source("pixabay", 2),
            "youtube": fake_source("youtube", 2),
        },
    )
    pool = r.retrieve(["query"])
    sources = {c.source for c in pool}
    assert sources == {"pexels", "pixabay", "youtube"}
