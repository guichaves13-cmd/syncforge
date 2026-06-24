"""Build the {source_name: search_fn} and {source_name: download_fn} maps
from environment / config."""
from __future__ import annotations
import os
from pathlib import Path
from typing import Callable

from ..sync.ranker import Candidate
from .pexels import PexelsSource
from .pixabay import PixabaySource
from .youtube import YouTubeSource
from .coverr import CoverrSource
from .mixkit import MixkitSource
from .wikimedia import WikimediaSource
from .openverse import OpenverseSource


def build_sources() -> tuple[
    dict[str, Callable[[str, int], list[Candidate]]],
    dict[str, Callable[[Candidate, Path], str | None]],
]:
    pex = PexelsSource()
    pix = PixabaySource()
    yt = YouTubeSource()
    cov = CoverrSource()
    mix = MixkitSource()
    wik = WikimediaSource()
    ov = OpenverseSource()

    searches: dict[str, Callable[[str, int], list[Candidate]]] = {
        "pexels": pex.search,
        "pixabay": pix.search,
        "youtube": yt.search,
        "wikimedia": wik.search,
        "openverse": ov.search,
    }
    # Optional sources only if env present / scraper desired
    if os.getenv("COVERR_API_KEY"):
        searches["coverr"] = cov.search
    if os.getenv("SYNCFORGE_USE_MIXKIT", "1") == "1":
        searches["mixkit"] = mix.search
    # Photos as last-resort same-namespace adapter
    searches["pexels_photo"] = pex.search_photos

    downloads: dict[str, Callable[[Candidate, Path], str | None]] = {
        "pexels": pex.download,
        "pexels_photo": pex.download,
        "pixabay": pix.download,
        "youtube": yt.download,
        "wikimedia": wik.download,
        "wikimedia_brand": wik.download,        # same file pipeline, just re-tagged
        "openverse": ov.download,
        "openverse_photo": ov.download,
        "coverr": cov.download,
        "mixkit": mix.download,
    }
    return searches, downloads


def build_brand_searches() -> dict[str, Callable[[str, int], list[Candidate]]]:
    """Brand-specific searches — used by the retriever when a concept window
    has a `dominant_entity` classified as a Brand or Product."""
    return {
        "wikimedia_brand": WikimediaSource().search_brand_assets,
        "openverse_brand": OpenverseSource().search,    # generic search works well for brands
    }


def download_by_source(c: Candidate, output_dir: Path,
                       downloads: dict[str, Callable[[Candidate, Path], str | None]]) -> str | None:
    fn = downloads.get(c.source)
    return fn(c, output_dir) if fn else None
