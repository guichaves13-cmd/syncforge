"""Google Veo 3 generative fallback — generate a clip when retrieval+rank+vision
all fail.

Veo 3 API: text → video, 6-8s clips, 720p/1080p. Cost ~$0.30 per clip.

Usage:
    veo = Veo3Generator(api_key=..., duration=6)
    candidate = veo.generate(intent, clause, output_dir)
"""
from __future__ import annotations
import os
import time
import uuid
from pathlib import Path
from typing import Optional

import requests

from ..sync.pipeline import Intent, NarrationClause
from ..sync.ranker import Candidate


class Veo3Generator:
    name = "veo3"

    def __init__(self, api_key: str = "", duration: int = 6,
                 model: str = "veo-3.0-generate-001",
                 poll_interval: int = 5, poll_max: int = 120):
        self.api_key = api_key or os.getenv("VEO_API_KEY") or os.getenv("GEMINI_API_KEY", "")
        self.duration = duration
        self.model = model
        self.poll_interval = poll_interval
        self.poll_max = poll_max

    def generate(self, intent: Intent, clause: NarrationClause,
                 output_dir: Path) -> Optional[Candidate]:
        if not self.api_key:
            return None
        prompt = self._build_prompt(intent, clause)
        try:
            from google import genai
            from google.genai import types
            client = genai.Client(api_key=self.api_key)
            op = client.models.generate_videos(
                model=self.model,
                prompt=prompt,
                config=types.GenerateVideosConfig(
                    aspect_ratio="16:9",
                    duration_seconds=self.duration,
                    number_of_videos=1,
                ),
            )
            # Poll for completion
            waited = 0
            while not op.done and waited < self.poll_max:
                time.sleep(self.poll_interval)
                op = client.operations.get(op)
                waited += self.poll_interval
            if not op.done or not op.response or not op.response.generated_videos:
                return None
            video = op.response.generated_videos[0].video
            dest = output_dir / f"veo3_{uuid.uuid4().hex[:8]}.mp4"
            dest.parent.mkdir(parents=True, exist_ok=True)
            client.files.download(file=video)
            video.save(str(dest))
            return Candidate(
                source=self.name,
                source_id=dest.stem,
                url=f"generated://{dest.name}",
                title=f"[Veo 3] {prompt[:80]}",
                description=prompt,
                local_path=str(dest),
                duration=float(self.duration),
                vision=85,  # generated clips inherently match the prompt
                vision_reason="generated from intent",
            )
        except Exception:
            return None

    @staticmethod
    def _build_prompt(intent: Intent, clause: NarrationClause) -> str:
        bits = []
        if intent.key_visuals:
            bits.append(", ".join(intent.key_visuals[:2]))
        elif intent.main_entity:
            bits.append(intent.main_entity)
        if intent.action:
            bits.append(intent.action)
        if intent.location:
            bits.append(f"in {intent.location}")
        mood = intent.mood or "cinematic"
        era = intent.era or "modern"
        prompt = ", ".join(filter(None, bits))
        return f"{prompt}, {mood}, {era}, realistic, no text overlays".strip(", ")
