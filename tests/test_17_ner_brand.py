"""Phase 6.2 — NER + brand asset retrieval: advanced tests.

Covers:
  • EntityCollection contract (primary_visual_anchor, has_any)
  • Lexical fallback NER on 6 niches (ping-pong, tech, history, finance,
    fitness, cooking) — each must extract sensible brands/products/people
  • LLM-based NER with mocked chain (JSON parsing, field cleaning, dedup)
  • Openverse adapter (search field mapping, license filter, no-key safe)
  • Wikimedia brand_assets (search query construction, re-tagging)
  • Factory wiring (openverse + wikimedia_brand both have downloaders)
  • Integration: full pipeline of clause → NER → brand-query → candidate

All tests run offline (mocks); real-API tests are marked `smoke`.
"""
from __future__ import annotations
import importlib
from pathlib import Path
from unittest import mock

import pytest

from app.services.llm.chain import LLMKeys
from app.services.llm.ner import (
    EntityCollection,
    extract_entities,
    extract_entities_lexical,
    _strip_stops,
)


# ─────────────────────────────────────────────────────────────────────────
# 1. EntityCollection contract
# ─────────────────────────────────────────────────────────────────────────

def test_entity_collection_empty():
    ec = EntityCollection()
    assert ec.has_any is False
    assert ec.all_entities() == []
    assert ec.primary_visual_anchor() == ""


def test_entity_collection_priority_product_over_brand():
    ec = EntityCollection(brands=["Apple"], products=["iPhone 17 Pro"])
    assert ec.primary_visual_anchor() == "iPhone 17 Pro"


def test_entity_collection_priority_brand_when_no_product():
    ec = EntityCollection(brands=["Butterfly"])
    assert ec.primary_visual_anchor() == "Butterfly"


def test_entity_collection_falls_back_to_people_then_places():
    ec = EntityCollection(people=["Ma Long"])
    assert ec.primary_visual_anchor() == "Ma Long"
    ec = EntityCollection(places=["Tokyo"])
    assert ec.primary_visual_anchor() == "Tokyo"


def test_entity_collection_all_entities_concatenates_in_order():
    ec = EntityCollection(brands=["Apple"], products=["iPhone"],
                            people=["Tim Cook"], places=["Cupertino"],
                            dates=["2024"], numbers=["$999"])
    out = ec.all_entities()
    assert out == ["Apple", "iPhone", "Tim Cook", "Cupertino", "2024", "$999"]


# ─────────────────────────────────────────────────────────────────────────
# 2. Lexical NER edge cases
# ─────────────────────────────────────────────────────────────────────────

def test_strip_stops_removes_function_words():
    assert _strip_stops("The Butterfly Viscaria") == "Butterfly Viscaria"
    assert _strip_stops("It is great") == "great"          # "It"+"is" stripped, "great" stays
    assert _strip_stops("Stiga") == "Stiga"
    assert _strip_stops("These Stiga blades") == "Stiga blades"
    assert _strip_stops("On July fourteenth") == "July fourteenth"  # "On" → stop


def test_lexical_ner_skips_pronouns_at_sentence_start():
    ec = extract_entities_lexical("It costs $50 per stateroom.")
    # Brands shouldn't contain "It"
    assert "It" not in ec.brands
    assert "$50" in ec.numbers


def test_lexical_ner_extracts_dollar_and_percent():
    ec = extract_entities_lexical("Apple sold $999 phones, growing 12% year over year in 2024.")
    assert "$999" in ec.numbers
    assert "12%" in ec.numbers
    assert "2024" in ec.dates


def test_lexical_ner_distinguishes_brand_from_product():
    ec = extract_entities_lexical("Stiga Carbonado 2000 is a 5-ply blade by Stiga.")
    assert "Stiga" in ec.brands
    # Multi-word capitalised should land in products
    assert any("Carbonado" in p for p in ec.products), f"products: {ec.products}"


def test_lexical_ner_handles_people_with_titles():
    ec = extract_entities_lexical("Dr. Yelena Petrov discovered a new compound.")
    assert "Yelena Petrov" in ec.people


def test_lexical_ner_empty_input():
    ec = extract_entities_lexical("")
    assert not ec.has_any
    ec2 = extract_entities_lexical("   ")
    assert not ec2.has_any


def test_lexical_ner_handles_unicode():
    ec = extract_entities_lexical("Apple's iPhone 17 launched in São Paulo.")
    # Should not crash and should pick up Apple
    assert "Apple" in ec.brands or any("Apple" in p for p in ec.products)


# ─────────────────────────────────────────────────────────────────────────
# 3. LLM-based NER (mocked)
# ─────────────────────────────────────────────────────────────────────────

def _mock_chain_return(payload: dict, monkeypatch):
    import json
    monkeypatch.setattr("app.services.llm.ner.call_chain",
                        lambda *a, **kw: json.dumps(payload))


def test_llm_ner_returns_structured_fields(monkeypatch):
    _mock_chain_return({
        "brands":   ["Butterfly", "Stiga"],
        "products": ["Viscaria", "Carbonado 2000"],
        "people":   ["Ma Long"],
        "places":   ["Tokyo"],
        "dates":    ["2024"],
        "numbers":  ["$189", "12%"],
    }, monkeypatch)
    ec = extract_entities("blah", LLMKeys())
    assert ec.brands == ["Butterfly", "Stiga"]
    assert "Viscaria" in ec.products
    assert ec.people == ["Ma Long"]
    assert ec.dates == ["2024"]
    assert "$189" in ec.numbers


def test_llm_ner_handles_markdown_fence(monkeypatch):
    import json
    monkeypatch.setattr(
        "app.services.llm.ner.call_chain",
        lambda *a, **kw: "```json\n" + json.dumps({"brands": ["Apple"]}) + "\n```",
    )
    ec = extract_entities("Apple sold phones.", LLMKeys())
    assert ec.brands == ["Apple"]


def test_llm_ner_falls_back_to_lexical_when_llm_fails(monkeypatch):
    monkeypatch.setattr("app.services.llm.ner.call_chain",
                        lambda *a, **kw: None)
    ec = extract_entities("Butterfly makes the Viscaria blade for $189.", LLMKeys())
    # Lexical fallback should still find something
    assert ec.has_any


def test_llm_ner_clean_list_caps_at_10_items(monkeypatch):
    _mock_chain_return({"brands": [f"B{i}" for i in range(50)]}, monkeypatch)
    ec = extract_entities("anything", LLMKeys())
    assert len(ec.brands) == 10


def test_llm_ner_clean_list_filters_empty_and_long(monkeypatch):
    _mock_chain_return({"brands": ["Real", "", "  ", "x" * 200, "Other"]}, monkeypatch)
    ec = extract_entities("x", LLMKeys())
    assert ec.brands == ["Real", "Other"]


def test_llm_ner_clean_list_dedupes(monkeypatch):
    _mock_chain_return({"brands": ["Apple", "Apple", "apple", "Apple"]}, monkeypatch)
    ec = extract_entities("x", LLMKeys())
    assert ec.brands.count("Apple") == 1


def test_llm_ner_handles_non_dict_response_gracefully(monkeypatch):
    monkeypatch.setattr("app.services.llm.ner.call_chain",
                        lambda *a, **kw: "[1,2,3]")  # not a dict
    ec = extract_entities("anything", LLMKeys())
    # Should fall back to lexical or return empty cleanly
    assert isinstance(ec, EntityCollection)


# ─────────────────────────────────────────────────────────────────────────
# 4. Multi-niche lexical NER (real scripts)
# ─────────────────────────────────────────────────────────────────────────

NICHE_NER_CASES = {
    "ping_pong": (
        "The Butterfly Viscaria blade retails at $189. Ma Long uses it in Tokyo tournaments.",
        {"brands": ["Butterfly"], "must_contain_in_all": "Viscaria"},
    ),
    "tech_review": (
        "Apple's iPhone 17 Pro launched for $1199. Samsung's Galaxy S26 Ultra is the obvious competitor.",
        {"brands": ["Apple", "Samsung"], "must_contain_in_all": "$1199"},
    ),
    "history": (
        "On July 14, 1789, the Bastille fortress in Paris was stormed by revolutionaries.",
        {"places_or_dates_present": True, "must_contain_in_all": "Paris"},
    ),
    "finance": (
        "A Roth IRA from Vanguard lets you contribute $7000 per year of post-tax income.",
        {"brands": ["Vanguard"], "must_contain_in_all": "$7000"},
    ),
    "fitness": (
        "Start with kettlebell swings, 20 reps, using a 16 kilogram bell.",
        {"numbers_present": True},
    ),
    "cooking": (
        "Caramelize four large yellow onions in butter for 40 minutes in a Le Creuset Dutch oven.",
        {"brands_likely": ["Le", "Creuset", "Le Creuset"]},
    ),
}


@pytest.mark.parametrize("niche,case", list(NICHE_NER_CASES.items()))
def test_niche_lexical_ner(niche, case):
    text, expects = case
    ec = extract_entities_lexical(text)
    if "brands" in expects:
        for b in expects["brands"]:
            assert b in ec.brands, f"{niche}: expected brand {b!r}, got {ec.brands}"
    if "brands_likely" in expects:
        all_text = " ".join(ec.brands + ec.products)
        assert any(b in all_text for b in expects["brands_likely"]), \
            f"{niche}: none of {expects['brands_likely']} in {ec.brands + ec.products}"
    if "must_contain_in_all" in expects:
        assert any(expects["must_contain_in_all"] in e for e in ec.all_entities()), \
            f"{niche}: {expects['must_contain_in_all']!r} missing from {ec.all_entities()}"
    if expects.get("numbers_present"):
        # Either dollar amounts or percentages should land
        assert ec.numbers or any(ch.isdigit() for e in ec.all_entities() for ch in e), \
            f"{niche}: no numbers extracted"
    if expects.get("places_or_dates_present"):
        assert ec.places or ec.dates, f"{niche}: expected places or dates"


# ─────────────────────────────────────────────────────────────────────────
# 5. Openverse adapter (offline, mocked)
# ─────────────────────────────────────────────────────────────────────────

def test_openverse_search_returns_empty_on_network_failure(monkeypatch):
    from app.services.stock.openverse import OpenverseSource
    def boom(*a, **kw):
        raise ConnectionError("offline")
    monkeypatch.setattr("app.services.stock.openverse.requests.get", boom)
    out = OpenverseSource().search("Butterfly Viscaria")
    assert out == []


def test_openverse_search_parses_results(monkeypatch):
    from app.services.stock.openverse import OpenverseSource
    fake_response = mock.MagicMock()
    fake_response.json.return_value = {
        "results": [
            {"id": "abc", "url": "https://x/img.jpg",
             "title": "Butterfly Viscaria paddle", "license": "cc0",
             "source": "wikimedia",
             "foreign_landing_url": "https://commons.wikimedia.org/x",
             "tags": [{"name": "paddle"}, {"name": "sports"}]},
            {"id": "xyz", "url": "https://x/img2.png",
             "title": "ping pong table", "license": "by",
             "source": "flickr",
             "foreign_landing_url": "https://flickr.com/y",
             "tags": []},
        ]
    }
    fake_response.raise_for_status = lambda: None
    monkeypatch.setattr("app.services.stock.openverse.requests.get",
                         lambda *a, **kw: fake_response)
    out = OpenverseSource().search("ping pong", max_results=5)
    assert len(out) == 2
    assert out[0].source == "openverse_photo"
    assert out[0].url.startswith("https://x/")
    assert out[0].title.startswith("Butterfly")
    # Tags propagate
    assert any("paddle" in t for t in out[0].tags)


def test_openverse_skips_results_without_url(monkeypatch):
    from app.services.stock.openverse import OpenverseSource
    fake_response = mock.MagicMock()
    fake_response.json.return_value = {
        "results": [
            {"id": "no-url", "title": "missing"},                 # no url → skip
            {"id": "ok",     "url": "https://x/img.jpg", "title": "ok",
             "license": "cc0", "source": "wikimedia", "tags": []},
        ]
    }
    fake_response.raise_for_status = lambda: None
    monkeypatch.setattr("app.services.stock.openverse.requests.get",
                         lambda *a, **kw: fake_response)
    out = OpenverseSource().search("anything")
    assert len(out) == 1
    assert out[0].source_id == "ok"


def test_openverse_download_handles_network_failure(tmp_path, monkeypatch):
    from app.services.stock.openverse import OpenverseSource
    from app.services.sync.ranker import Candidate
    def boom(*a, **kw):
        raise TimeoutError("read timed out")
    monkeypatch.setattr("app.services.stock.openverse.requests.get", boom)
    c = Candidate(source="openverse_photo", source_id="x",
                   url="https://x/img.jpg", title="x")
    assert OpenverseSource().download(c, tmp_path) is None


# ─────────────────────────────────────────────────────────────────────────
# 6. Wikimedia brand-aware search
# ─────────────────────────────────────────────────────────────────────────

def test_wikimedia_brand_search_uses_logo_and_product_queries(monkeypatch):
    from app.services.stock.wikimedia import WikimediaSource
    seen_queries = []

    def fake_search_files(self, query, max_results, file_types=("video", "image")):
        seen_queries.append(query)
        return []

    monkeypatch.setattr(WikimediaSource, "_search_files", fake_search_files)
    WikimediaSource().search_brand_assets("Butterfly", max_results=4)
    # We must have tried the logo+product+brand triplet
    assert "Butterfly logo" in seen_queries
    assert "Butterfly product" in seen_queries
    assert "Butterfly brand" in seen_queries


def test_wikimedia_brand_search_dedupes_across_queries(monkeypatch):
    from app.services.stock.wikimedia import WikimediaSource
    from app.services.sync.ranker import Candidate

    def stub_search(self, query, max_results, file_types=("video", "image")):
        # Both queries return the same file_id
        return [Candidate(source="wikimedia", source_id="dup-1",
                          url="https://x/logo.png", title=f"file from {query}")]

    monkeypatch.setattr(WikimediaSource, "_search_files", stub_search)
    out = WikimediaSource().search_brand_assets("Apple", max_results=4)
    assert len(out) == 1, f"expected dedup, got {len(out)}"
    assert out[0].source == "wikimedia_brand"
    assert "logo" in out[0].tags
    assert "Apple" in out[0].tags


def test_wikimedia_brand_search_returns_empty_when_no_results(monkeypatch):
    from app.services.stock.wikimedia import WikimediaSource
    monkeypatch.setattr(WikimediaSource, "_search_files",
                         lambda self, q, n, file_types=("video", "image"): [])
    assert WikimediaSource().search_brand_assets("NonexistentBrand") == []


# ─────────────────────────────────────────────────────────────────────────
# 7. Factory wiring
# ─────────────────────────────────────────────────────────────────────────

def test_factory_registers_openverse():
    from app.services.stock.factory import build_sources
    searches, downloads = build_sources()
    assert "openverse" in searches
    assert "openverse" in downloads
    assert "openverse_photo" in downloads
    assert callable(searches["openverse"])
    assert callable(downloads["openverse"])


def test_factory_registers_wikimedia_brand_downloader():
    from app.services.stock.factory import build_sources
    _, downloads = build_sources()
    assert "wikimedia_brand" in downloads
    # Both should be the same bound-method (== equality, not identity)
    assert downloads["wikimedia_brand"] == downloads["wikimedia"]
    # And both call the same underlying function
    assert downloads["wikimedia_brand"].__func__ is downloads["wikimedia"].__func__


def test_factory_build_brand_searches_returns_brand_aware_only():
    from app.services.stock.factory import build_brand_searches
    brand = build_brand_searches()
    assert set(brand.keys()) == {"wikimedia_brand", "openverse_brand"}
    for fn in brand.values():
        assert callable(fn)


# ─────────────────────────────────────────────────────────────────────────
# 8. Integration: concept window → NER → brand candidate
# ─────────────────────────────────────────────────────────────────────────

def test_e2e_concept_window_to_brand_pipeline(monkeypatch):
    """A real-world flow: 1 sentence → concept windows → for each window
    with a Brand entity, brand-aware retrieval surfaces brand assets."""
    from app.services.sync.concept import segment_into_windows, group_into_clusters
    from app.services.sync.pipeline import NarrationClause
    from app.services.stock.factory import build_brand_searches
    from app.services.sync.ranker import Candidate

    # Stub the brand search so we don't hit the network
    def fake_wm(brand, n=6):
        return [Candidate(source="wikimedia_brand", source_id=f"wm-{brand}",
                          url=f"https://x/{brand}.png", title=f"{brand} logo",
                          tags=[brand, "logo"])]
    def fake_ov(query, n=8):
        return [Candidate(source="openverse_photo", source_id=f"ov-{query}",
                          url=f"https://x/{query}.jpg", title=query)]

    brand_searches = build_brand_searches()
    monkeypatch.setitem(brand_searches, "wikimedia_brand", fake_wm)
    monkeypatch.setitem(brand_searches, "openverse_brand", fake_ov)

    clauses = [NarrationClause(start=0, end=18,
        text="Butterfly Viscaria blade retails at $189, while Stiga Carbonado costs $159.")]
    windows = segment_into_windows(clauses)
    group_into_clusters(windows)

    brand_assets: list[Candidate] = []
    for w in windows:
        if not w.dominant_entity:
            continue
        # extract brand head from dominant_entity
        brand_head = w.dominant_entity.split()[0]
        for name, fn in brand_searches.items():
            for c in fn(brand_head, 2):
                brand_assets.append(c)

    # At least Butterfly and/or Stiga assets should have been retrieved
    brands_found = {c.title.split()[0] for c in brand_assets}
    assert "Butterfly" in brands_found or "Stiga" in brands_found, \
        f"got brand assets for: {brands_found}"
