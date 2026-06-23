"""Smoke: every module must import without error."""
import importlib
import pytest

MODULES = [
    "app.main",
    "app.services.sync",
    "app.services.sync.embedder",
    "app.services.sync.ranker",
    "app.services.sync.retriever",
    "app.services.sync.verifier",
    "app.services.sync.pipeline",
    "app.services.llm.chain",
    "app.services.llm.intent",
    "app.services.stock.base",
    "app.services.stock.pexels",
    "app.services.stock.pixabay",
    "app.services.stock.youtube",
    "app.services.stock.coverr",
    "app.services.stock.mixkit",
    "app.services.stock.wikimedia",
    "app.services.stock.factory",
    "app.services.tts.edge",
    "app.services.subtitles.karaoke",
    "app.services.render.composer",
    "app.services.runner",
]


@pytest.mark.parametrize("mod", MODULES)
def test_imports(mod):
    importlib.import_module(mod)
