"""Pick a generative provider based on env."""
from __future__ import annotations
import os
from pathlib import Path
from typing import Callable, Optional

from ..sync.pipeline import Intent, NarrationClause
from ..sync.ranker import Candidate
from .veo import Veo3Generator
from .runway import RunwayGen4Generator


def build_generative_fn() -> Optional[Callable[[Intent, NarrationClause, Path], Optional[Candidate]]]:
    """Returns a `generative_fn` compatible with SyncEngine, or None if unavailable."""
    if os.getenv("VEO_API_KEY") or os.getenv("GEMINI_API_KEY"):
        gen = Veo3Generator()
        return lambda intent, clause, out: gen.generate(intent, clause, out)
    if os.getenv("RUNWAY_API_KEY"):
        gen = RunwayGen4Generator()
        return lambda intent, clause, out: gen.generate(intent, clause, out)
    return None
