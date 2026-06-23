"""Real-API smoke tests — hit free endpoints to confirm adapters actually work.

These are SKIPPED by default. Run with:  pytest -m smoke
"""
import os
import pytest
import requests

from app.services.stock.wikimedia import WikimediaSource
from app.services.stock.pixabay import PixabaySource
from app.services.stock.pexels import PexelsSource


pytestmark = pytest.mark.smoke


def _online():
    try:
        requests.head("https://www.google.com", timeout=3)
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _online(), reason="offline")
def test_wikimedia_real_search():
    """Wikimedia needs no API key — must return at least 1 result."""
    src = WikimediaSource()
    out = src.search("table tennis", max_results=3)
    # Wikimedia has video files for almost any common topic
    assert isinstance(out, list)
    if out:
        c = out[0]
        assert c.url.startswith("http")
        assert c.source_id


@pytest.mark.skipif(not os.getenv("PIXABAY_API_KEY"), reason="no Pixabay key")
def test_pixabay_real_search():
    src = PixabaySource()
    out = src.search("table tennis", max_results=3)
    assert isinstance(out, list)
    if out:
        assert out[0].url.startswith("http")


@pytest.mark.skipif(not os.getenv("PEXELS_API_KEY"), reason="no Pexels key")
def test_pexels_real_search():
    src = PexelsSource()
    out = src.search("table tennis", max_results=3)
    assert isinstance(out, list)
    if out:
        assert out[0].url.startswith("http")
        assert out[0].duration >= 0
